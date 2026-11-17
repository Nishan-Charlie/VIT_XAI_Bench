"""Benchmark runner: sweeps (model x method x image) and evaluates metrics.

Design points that matter scientifically:

* **Deterministic per-evaluation seeding.** Several methods (SmoothGrad, VarGrad,
  GradientSHAP, RISE, LIME) and the perturbation metrics (Faithfulness
  Correlation, Max-Sensitivity) sample internally. Python, NumPy and Torch are
  reseeded before every evaluation from ``crc32(seed|model|method|image)``, and
  the attribution and its metrics then draw from that one stream, so a row does
  not depend on which rows ran before it. Resuming a crashed run, reordering
  ``methods`` or dropping a model leaves every remaining row unchanged.
* **Failures are recorded, not dropped.** An attribution that raises is a row
  with an ``error`` field, counted in ``summary.csv`` (``n_failed``) and listed in
  ``failures.json``; a resumed run retries it. A metric that raises leaves the
  other metrics of the row intact and its reason in ``metric_errors``.
* **Every image is scored.** A constant attribution map carries no information
  and every metric is undefined on it. It is flagged (``degenerate_map``) and
  scored by :mod:`xai_bench.scoring` in the summary: no credit on the bounded
  metrics, unbounded on Max-Sensitivity, whose cell statistic is the median.
* **Attribution cost is a first-class metric.** ``time_ms`` (GPU-synchronised)
  and peak GPU memory are aggregated into the summary.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import time
import zlib
from typing import Any

import torch
from tqdm import tqdm

# The registries are filled when they are first read through the package
# (xai_bench.__getattr__ imports models, methods, datasets and metrics);
# importing them from xai_bench.registry directly would leave them empty.
from xai_bench import DATASETS, METHODS, METRICS, MODELS, scoring
from xai_bench.config import RunConfig
from xai_bench.utils import provenance, set_seed

#: Cost columns the runner always produces.
COST_COLUMNS = ("time_ms", "peak_gpu_mb")


def derive_seed(seed: int, *parts: Any) -> int:
    """A stable 32-bit seed from a seed and an item's identity.

    crc32 rather than :func:`hash`, because Python's string hashing is salted per
    process, which would break reproducibility across runs and machines.
    """
    return zlib.crc32("|".join([str(seed), *(str(p) for p in parts)]).encode("utf-8"))


def evaluation_seed(seed: int, model_name: str, method_name: str, image_idx: int) -> int:
    """The RNG seed of one (seed, model, method, image) evaluation."""
    return derive_seed(seed, model_name, method_name, image_idx)


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work, so a timed window covers the whole computation."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


class BenchmarkRunner:
    """Runs a :class:`RunConfig` sweep and writes results incrementally."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.device = torch.device(config.resolved_device())

        if not self.config.run_name:
            self.config.run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.output_dir, self.config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        seed = self.config.seeds[0] if self.config.seeds else 0
        self.provenance = provenance.collect(
            seed=seed,
            extra={
                "run_name": self.config.run_name,
                "seeds": list(self.config.seeds),
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
        set_seed(seed, deterministic=self.config.deterministic)
        print(f"Provenance: {provenance.summary_line(self.provenance)}")
        if self.provenance.get("git_dirty"):
            print("  WARNING: working tree is dirty; results are not fully traceable "
                  "to a commit.")

    # ────────────────────────────────── output ──────────────────────────────

    def _write_csv(self, results: list[dict]) -> None:
        """Write per-image results and the (model, method) summary.

        summary.csv, per (model, method):
          n_evaluations       rows attempted (images x seeds)
          n_failed            rows whose attribution raised (kept in results.json
                              with an "error" field; retried when the run resumes)
          n_constant_maps     rows whose attribution map is constant
          <metric>_mean/median/std/count
                              over every successful row, constant maps and
                              undefined values scored as in xai_bench.scoring (no
                              image is dropped); a cell reports
                              scoring.cell_statistic(metric) (the median for MS,
                              whose failures score +inf, the mean otherwise)
          <metric>_mean_defined / <metric>_undefined
                              the raw mean over the rows where the metric is
                              defined, and the number of rows where it is not
          time_ms_*, peak_gpu_mb_*   attribution cost, GPU-synchronised
        results.json keeps the raw values (None where undefined).
        """
        if not results:
            return
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available; skipping CSV export.")
            return

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.run_dir, "results.csv"), index=False)

        keys = ["model", "method"]
        if "error" not in df.columns:
            df["error"] = None
        failed = df["error"].notna()
        ok = df[~failed]
        summary = df.groupby(keys).size().rename("n_evaluations").to_frame()
        summary["n_failed"] = failed.groupby([df[k] for k in keys]).sum().astype(int)

        if "degenerate_map" in ok.columns and len(ok):
            summary["n_constant_maps"] = ok["degenerate_map"].eq(True) \
                .groupby([ok[k] for k in keys]).sum().astype(int)
        metric_cols = [m for m in self.config.metrics if m in df.columns]
        if metric_cols and len(ok):
            raw = ok[keys + metric_cols].copy()
            for m in metric_cols:
                raw[m] = pd.to_numeric(raw[m], errors="coerce") \
                    .replace([math.inf, -math.inf], math.nan)
            scored = scoring.score(ok, metric_cols, group=keys)
            agg = scored.groupby(keys)[metric_cols].agg(["mean", "median", "std", "count"])
            agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
            graw = raw.groupby(keys)
            for m in metric_cols:
                agg[f"{m}_mean_defined"] = graw[m].mean()
                agg[f"{m}_undefined"] = graw[m].apply(lambda s: int(s.isna().sum()))
            summary = summary.join(agg, how="left")
        for cost in COST_COLUMNS:
            if cost in ok.columns and len(ok):
                t = pd.to_numeric(ok[cost], errors="coerce").groupby([ok[k] for k in keys])
                summary = summary.join(
                    t.agg(["mean", "median", "std"]).add_prefix(f"{cost}_"), how="left")
        summary.reset_index().to_csv(os.path.join(self.run_dir, "summary.csv"), index=False)
        print(f"  CSV updated: {len(df)} rows ({int(failed.sum())} failed) -> "
              f"{os.path.join(self.run_dir, 'results.csv')}")

    def _write_failures(self, results: list[dict], init_failures: dict[str, str]) -> None:
        """failures.json: methods that could not be set up, and every failed evaluation."""
        evals = [{k: r.get(k) for k in ("model", "method", "seed", "image_idx", "error")}
                 for r in results if r.get("error")]
        with open(os.path.join(self.run_dir, "failures.json"), "w", encoding="utf-8") as fh:
            json.dump({"method_init": init_failures, "evaluations": evals}, fh, indent=2)

    def _save(self, results: list[dict]) -> None:
        with open(os.path.join(self.run_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    def _resume(self) -> tuple[list[dict], set]:
        """Rows of a previous attempt, without its failures, which are retried."""
        results_file = os.path.join(self.run_dir, "results.json")
        if not os.path.exists(results_file):
            return [], set()
        try:
            with open(results_file, encoding="utf-8") as fh:
                results = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not resume from {results_file}: {exc}")
            return [], set()
        retry = [r for r in results if r.get("error")]
        if retry:
            results = [r for r in results if not r.get("error")]
            print(f"Retrying {len(retry)} evaluations that failed in the previous attempt.")
        done = {(r["model"], r["method"], r["image_idx"], r.get("seed", self.config.seeds[0]))
                for r in results}
        print(f"Resuming: loaded {len(results)} existing rows from {self.run_dir}")
        return results, done

    # ─────────────────────────────────── run ────────────────────────────────

    def run(self) -> list[dict]:
        dataset = DATASETS.get(self.config.dataset)(**self.config.dataset_kwargs)
        num_images = min(self.config.num_images or len(dataset), len(dataset))

        results, done = self._resume()
        init_failures: dict[str, str] = {}
        for model_name in self.config.models:
            print(f"Loading model: {model_name}")
            model_wrapper = MODELS.get(model_name)()
            model_wrapper.model.to(self.device)
            model_wrapper.model.eval()

            for method_name in self.config.methods:
                supports_attention = getattr(
                    model_wrapper, "supports_attention", model_wrapper.is_vit)
                if not supports_attention and method_name in (
                        "attention_rollout", "attention_gradient"):
                    if self.config.limit_methods_to_supported:
                        print(f"  Skipping {method_name}: {model_name} has no global "
                              "CLS-token attention")
                        continue

                print(f"  Setting up method: {method_name}")
                try:
                    method = METHODS.get(method_name)(
                        model_wrapper=model_wrapper, model=model_wrapper.model)
                except Exception as exc:  # noqa: BLE001 - recorded in failures.json
                    init_failures[f"{model_name}/{method_name}"] = _error_text(exc)
                    print(f"    FAILED to initialize {method_name}: {_error_text(exc)} "
                          "(recorded in failures.json)")
                    continue

                evals = [(seed, i) for seed in self.config.seeds for i in range(num_images)]
                for seed, i in tqdm(evals, desc=f"    {method_name}"):
                    if (model_name, method_name, i, seed) in done:
                        continue
                    results.append(self._evaluate_one(
                        dataset, i, seed, model_wrapper, model_name, method, method_name))
                    self._save(results)

            print(f"Finished model: {model_name}. Writing CSV...")
            self._write_csv(results)
            self._write_failures(results, init_failures)

            del model_wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save(results)
        self._write_csv(results)
        self._write_failures(results, init_failures)

        n_failed = sum(1 for r in results if r.get("error"))
        print(f"Run complete: {len(results)} rows -> {self.run_dir}")
        if n_failed or init_failures:
            print(f"WARNING: {n_failed} failed evaluations and {len(init_failures)} methods "
                  "that could not be set up; see failures.json. Rerun to retry the failed "
                  "evaluations.")
        return results

    def _evaluate_one(self, dataset, idx: int, seed: int, model_wrapper, model_name: str,
                      method, method_name: str) -> dict[str, Any]:
        """Evaluate one (seed, model, method, image) under its own seed."""
        # Covers everything stochastic in this evaluation: the attribution and
        # every metric after it, which continue the same stream.
        set_seed(evaluation_seed(seed, model_name, method_name, idx),
                 deterministic=self.config.deterministic)
        img, target, metadata = dataset[idx]
        img = img.to(self.device)
        row: dict[str, Any] = {
            "model": model_name,
            "method": method_name,
            "image_idx": idx,
            "image_name": metadata.get("img_name", str(idx)),
            "seed": seed,
            "target": target,
        }

        # Attribution. A failure is a row with an "error" field (metrics None), so
        # it is counted in summary.csv and failures.json instead of disappearing.
        try:
            batch_img = img.unsqueeze(0) if img.dim() == 3 else img
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            _sync(self.device)
            t0 = time.perf_counter()
            attr = method(batch_img, target=target)
            if isinstance(attr, tuple):
                attr = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in attr)
            elif isinstance(attr, torch.Tensor):
                attr = attr.detach()
            _sync(self.device)
            row["time_ms"] = (time.perf_counter() - t0) * 1000.0
            if self.device.type == "cuda":
                row["peak_gpu_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        except Exception as exc:  # noqa: BLE001 - recorded explicitly
            row["error"] = _error_text(exc)
            row["time_ms"] = None
            for metric_name in self.config.metrics:
                row[metric_name] = None
            print(f"    FAILED image {idx} (seed {seed}) with {method_name}: {row['error']}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return row

        # Quantus calls this with keyword arguments; the method is bound explicitly
        # so the closure cannot see a later loop value.
        def explain_func(model, inputs, targets, _method=method, **kwargs):
            device = next(model.parameters()).device
            inputs_t = torch.as_tensor(inputs, device=device, dtype=torch.float32)
            target_t = targets[0].item() if hasattr(targets, "__len__") else targets
            # detach: CAM and attention methods return grad-requiring tensors
            return _method(inputs_t, target=int(target_t)).detach().cpu().numpy()

        # A constant map carries no information and every metric is undefined on
        # it; it is flagged here and scored in summary.csv (xai_bench.scoring).
        row["degenerate_map"] = scoring.is_constant_map(attr[0])

        # Metrics, each on its own: a failing metric does not drop the others.
        metric_errors = {}
        for metric_name in self.config.metrics:
            if row["degenerate_map"]:
                row[metric_name] = None
                continue
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
            except Exception as exc:  # noqa: BLE001 - recorded in metric_errors
                metric_errors[metric_name] = _error_text(exc)
                val = None
            try:
                val = float(val)
                if not math.isfinite(val):
                    val = None   # undefined: counted and scored in summary.csv
            except (TypeError, ValueError):
                val = None
            row[metric_name] = val
        if metric_errors:
            row["metric_errors"] = metric_errors
            print(f"    Metric failure on image {idx} with {method_name}: {metric_errors}")
        return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the XAI attribution benchmark")
    parser.add_argument("--config", type=str, required=True, help="path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the seed list")
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
