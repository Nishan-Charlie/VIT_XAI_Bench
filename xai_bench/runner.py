"""Benchmark runner: sweeps (model x method x image) and evaluates metrics.

Design points that matter scientifically:

* **Deterministic per-item seeding.** Several methods (SmoothGrad, VarGrad,
  GradientSHAP, RISE, LIME) and the robustness metric sample internally. The RNG
  is therefore re-seeded from ``hash(master_seed, model, method, image_idx)``
  before every evaluation, so a cell's value does not depend on how many items
  ran before it. Resuming a crashed run reproduces the same numbers as a clean
  run — which a single global RNG stream cannot guarantee.
* **Failures are recorded, not dropped.** A method or metric that raises writes
  an explicit ``status="error"`` row. Silently skipping would let a method be
  scored on the subset of images it happened to survive.
* **Attribution cost is a first-class metric.** ``time_ms`` and peak GPU memory
  are aggregated into the summary alongside the quality metrics.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
from typing import Any

import torch
from tqdm import tqdm

from xai_bench.config import RunConfig
from xai_bench.registry import DATASETS, METHODS, METRICS, MODELS
from xai_bench.utils import provenance, set_seed

#: Cost columns the runner always produces, aggregated like any other metric.
COST_COLUMNS = ("time_ms", "peak_gpu_mb")

#: How often (in rows) to flush the incremental results file.
_FLUSH_EVERY = 25


def derive_seed(master_seed: int, *parts: Any) -> int:
    """A stable 32-bit seed from a master seed and the item's identity.

    Uses blake2b rather than :func:`hash` because Python's string hashing is
    randomised per process, which would break reproducibility across runs.
    """
    payload = "|".join([str(master_seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


class BenchmarkRunner:
    """Runs a :class:`RunConfig` sweep and writes results incrementally."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.device = torch.device(config.resolved_device())
        self.master_seed = config.seeds[0] if config.seeds else 0

        if not self.config.run_name:
            self.config.run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, self.config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.provenance = provenance.collect(
            seed=self.master_seed,
            extra={
                "run_name": self.config.run_name,
                "dataset": self.config.dataset,
                "device": str(self.device),
                "deterministic": self.config.deterministic,
            },
        )

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)
        with open(os.path.join(self.run_dir, "provenance.json"), "w", encoding="utf-8") as fh:
            json.dump(self.provenance, fh, indent=2)

        # Seed once up front so dataset construction and model init are reproducible.
        set_seed(self.master_seed, deterministic=self.config.deterministic)
        print(f"Provenance: {provenance.summary_line(self.provenance)}")
        if self.provenance.get("git_dirty"):
            print("  WARNING: working tree is dirty; results are not fully traceable "
                  "to a commit.")

    # ────────────────────────────────── output ──────────────────────────────

    def _metric_columns(self, df) -> list[str]:
        cols = [m for m in self.config.metrics if m in df.columns]
        cols += [c for c in COST_COLUMNS if c in df.columns]
        return cols

    def _write_csv(self, results: list[dict]) -> None:
        """Write per-image results plus a (model, method) aggregate summary."""
        if not results:
            return
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available; skipping CSV export.")
            return

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.run_dir, "results.csv"), index=False)

        ok = df[df.get("status", "ok") == "ok"] if "status" in df.columns else df
        metric_cols = self._metric_columns(ok)
        if metric_cols and not ok.empty:
            agg = ok.groupby(["model", "method"])[metric_cols].agg(["mean", "std", "count"])
            agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
            agg = agg.reset_index()
            # Record how many evaluations errored, so a high score on a small
            # surviving subset is visible rather than hidden.
            if "status" in df.columns:
                errs = (
                    df[df["status"] != "ok"]
                    .groupby(["model", "method"])
                    .size()
                    .rename("n_errors")
                    .reset_index()
                )
                agg = agg.merge(errs, on=["model", "method"], how="left")
                agg["n_errors"] = agg["n_errors"].fillna(0).astype(int)
            agg.to_csv(os.path.join(self.run_dir, "summary.csv"), index=False)

    def _flush(self, results: list[dict]) -> None:
        with open(os.path.join(self.run_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    # ─────────────────────────────────── run ────────────────────────────────

    def run(self) -> list[dict]:
        dataset_fn = DATASETS.get(self.config.dataset)
        dataset = dataset_fn(**self.config.dataset_kwargs)

        num_images = self.config.num_images or len(dataset)
        num_images = min(num_images, len(dataset))

        results: list[dict] = []
        completed = set()
        results_file = os.path.join(self.run_dir, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, encoding="utf-8") as fh:
                    results = json.load(fh)
                completed = {(r["model"], r["method"], r["image_idx"]) for r in results}
                print(f"Resuming: loaded {len(results)} existing rows from {self.run_dir}")
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                print(f"Could not resume from {results_file}: {exc}")
                results, completed = [], set()

        since_flush = 0
        for model_name in self.config.models:
            print(f"Loading model: {model_name}")
            model_wrapper = MODELS.get(model_name)()
            model_wrapper.model.to(self.device)
            model_wrapper.model.eval()

            for method_name in self.config.methods:
                supports_attention = getattr(
                    model_wrapper, "supports_attention", model_wrapper.is_vit
                )
                if not supports_attention and method_name in (
                    "attention_rollout",
                    "attention_gradient",
                ):
                    if self.config.limit_methods_to_supported:
                        print(f"  Skipping {method_name}: {model_name} has no global "
                              "CLS-token attention")
                        continue

                print(f"  Setting up method: {method_name}")
                try:
                    method = METHODS.get(method_name)(
                        model_wrapper=model_wrapper, model=model_wrapper.model
                    )
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    print(f"    FAILED to initialize {method_name}: {exc}")
                    results.append(
                        {
                            "model": model_name,
                            "method": method_name,
                            "image_idx": -1,
                            "status": "init_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                for i in tqdm(range(num_images), desc=f"    {method_name}"):
                    if (model_name, method_name, i) in completed:
                        continue
                    row = self._evaluate_one(
                        dataset, i, model_wrapper, model_name, method, method_name
                    )
                    results.append(row)
                    since_flush += 1
                    if since_flush >= _FLUSH_EVERY:
                        self._flush(results)
                        since_flush = 0

            print(f"Finished model: {model_name}. Writing CSV...")
            self._flush(results)
            self._write_csv(results)

            del model_wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._flush(results)
        self._write_csv(results)

        n_err = sum(1 for r in results if r.get("status") != "ok")
        print(f"Run complete: {len(results)} rows ({n_err} errored) -> {self.run_dir}")
        return results

    def _evaluate_one(
        self,
        dataset,
        idx: int,
        model_wrapper,
        model_name: str,
        method,
        method_name: str,
    ) -> dict[str, Any]:
        """Evaluate one (model, method, image) cell under a derived seed."""
        row: dict[str, Any] = {
            "model": model_name,
            "method": method_name,
            "image_idx": idx,
            "seed": self.master_seed,
            "status": "ok",
        }

        # Re-seed per item so the value is independent of iteration order.
        item_seed = derive_seed(self.master_seed, model_name, method_name, idx)
        set_seed(item_seed, deterministic=self.config.deterministic)

        try:
            img, target, metadata = dataset[idx]
            img = img.to(self.device)
            batch_img = img.unsqueeze(0) if img.dim() == 3 else img
            row["image_name"] = metadata.get("img_name", str(idx))
            row["target"] = target

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            attr = method(batch_img, target=target)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row["time_ms"] = (time.perf_counter() - t0) * 1000.0
            if torch.cuda.is_available():
                row["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

            if isinstance(attr, tuple):
                attr = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in attr)
            elif isinstance(attr, torch.Tensor):
                attr = attr.detach()
        except Exception as exc:  # noqa: BLE001 - recorded explicitly
            row["status"] = "attribution_error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row

        def explain_func(model, inputs, targets, **kwargs):
            """Adapter so Quantus can re-explain perturbed inputs."""
            device = next(model.parameters()).device
            if isinstance(inputs, torch.Tensor):
                inputs_t = inputs.to(device=device, dtype=torch.float32)
            else:
                inputs_t = torch.as_tensor(inputs, device=device, dtype=torch.float32)
            target_t = targets[0].item() if hasattr(targets, "__len__") else targets
            return method(inputs_t, target=int(target_t)).detach().cpu().numpy()

        for metric_name in self.config.metrics:
            # Metrics that perturb (max-sensitivity, faithfulness) sample too;
            # give each its own reproducible stream.
            set_seed(derive_seed(item_seed, metric_name), deterministic=self.config.deterministic)
            try:
                val = METRICS.get(metric_name)(
                    attr=attr[0],
                    metadata=metadata,
                    model=model_wrapper.model,
                    image=img,
                    target=target,
                    explain_func=explain_func,
                    metric_kwargs=dict(self.config.metric_kwargs.get(metric_name, {})),
                )
            except Exception as exc:  # noqa: BLE001
                row[metric_name] = None
                row.setdefault("metric_errors", {})[metric_name] = (
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            try:
                val = float(val)
                if math.isnan(val) or math.isinf(val):
                    val = None
            except (TypeError, ValueError):
                val = None
            row[metric_name] = val

        return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the XAI attribution benchmark")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    parser.add_argument("--num-images", type=int, default=None, help="override image count")
    parser.add_argument("--device", type=str, default=None, help="auto | cuda | cpu")
    parser.add_argument("--run-name", type=str, default=None, help="override run name")
    parser.add_argument("--output-dir", type=str, default=None, help="override output dir")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="force deterministic cuDNN kernels (slower, bit-reproducible)",
    )
    args = parser.parse_args()

    config = RunConfig.from_yaml(args.config)
    config.config_path = args.config
    if args.seed is not None:
        config.seeds = [args.seed]
    if args.num_images is not None:
        config.num_images = args.num_images
    if args.device is not None:
        config.device = args.device
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.deterministic:
        config.deterministic = True

    BenchmarkRunner(config).run()


if __name__ == "__main__":
    main()
