"""Fast end-to-end check of the benchmark pipeline.

Exercises every stage — dataset -> model -> attribution -> metrics -> result
records -> validation -> JSON export — on synthetic data with a randomly
initialised backbone. No dataset download, no pretrained weights, no GPU, so it
runs anywhere in well under a minute and is safe for CI.

It verifies that the machinery is *wired correctly*, not that the science is
right: the scores it produces are meaningless by construction.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --keep   # keep the temp output for inspection
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xai_bench.registry import DATASETS, METHODS, METRICS, MODELS  # noqa: E402
from xai_bench.results_schema import (  # noqa: E402
    MetricValue,
    Provenance,
    ResultRecord,
    assert_valid,
)
from xai_bench.runner import derive_seed  # noqa: E402
from xai_bench.utils import provenance, set_seed  # noqa: E402

IMAGE_SIZE = 64          # small enough to keep CPU attribution fast
N_IMAGES = 3
N_CLASSES = 10
SEED = 42

_CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class SyntheticDataset:
    """(image, target, metadata) triples with a known bright square as 'object'.

    The square doubles as the localisation ground truth, so the bounding-box
    metrics have something meaningful to consume.
    """

    def __init__(self, n: int = N_IMAGES, size: int = IMAGE_SIZE):
        self.size = size
        gen = torch.Generator().manual_seed(SEED)
        self.items = []
        for i in range(n):
            img = torch.randn(3, size, size, generator=gen) * 0.2
            lo, hi = size // 4, 3 * size // 4
            img[:, lo:hi, lo:hi] += 2.0          # the "object"
            bbox = [lo / size, lo / size, hi / size, hi / size]   # normalised xyxy
            self.items.append(
                (img, i % N_CLASSES, {"img_name": f"synthetic_{i}", "bboxes": [bbox]})
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def build_model():
    """A randomly initialised small CNN wrapped like any benchmark backbone."""
    import timm

    from xai_bench.models.timm_models import ModelWrapper

    model = timm.create_model("resnet18", pretrained=False, num_classes=N_CLASSES)
    model.eval()
    return ModelWrapper(model, "resnet50", cam_format="nchw", supports_attention=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the temp output dir")
    args = ap.parse_args()

    print("XAI-Bench smoke test")
    print("=" * 60)
    set_seed(SEED)

    info = provenance.collect(seed=SEED)
    print(f"Environment: {provenance.summary_line(info)}")
    print()

    out_dir = Path(tempfile.mkdtemp(prefix="xai_bench_smoke_"))
    try:
        # 1 ─ registries populated
        print("1. Registries")
        import xai_bench.datasets  # noqa: F401
        import xai_bench.methods  # noqa: F401
        import xai_bench.metrics  # noqa: F401
        import xai_bench.models  # noqa: F401

        check("models registered", len(MODELS.names()) > 0, f"{len(MODELS.names())} models")
        check("methods registered", len(METHODS.names()) > 0, f"{len(METHODS.names())} methods")
        check("metrics registered", len(METRICS.names()) > 0, f"{len(METRICS.names())} metrics")
        check("datasets registered", len(DATASETS.names()) > 0, f"{len(DATASETS.names())} datasets")
        check("separate-paper method absent", "hilrp" not in METHODS)

        # 2 ─ dataset
        print("\n2. Dataset")
        dataset = SyntheticDataset()
        img, target, meta = dataset[0]
        check("dataset yields images", tuple(img.shape) == (3, IMAGE_SIZE, IMAGE_SIZE),
              str(tuple(img.shape)))
        check("metadata carries bboxes", bool(meta.get("bboxes")))

        # 3 ─ model
        print("\n3. Model")
        wrapper = build_model()
        with torch.inference_mode():
            logits = wrapper.model(img.unsqueeze(0))
        check("forward pass", tuple(logits.shape) == (1, N_CLASSES), str(tuple(logits.shape)))
        check("model in eval mode", not wrapper.model.training)

        # 4 ─ attribution
        print("\n4. Attribution")
        attributions = {}
        for method_name in ("saliency", "grad_cam", "input_x_gradient"):
            set_seed(derive_seed(SEED, "resnet50", method_name, 0))
            fn = METHODS.get(method_name)(model_wrapper=wrapper, model=wrapper.model)
            attr = fn(img.unsqueeze(0), target=target)
            attr = attr[0] if isinstance(attr, tuple) else attr
            attr = attr.detach()
            spatial = attr[0] if attr.dim() >= 3 else attr
            while spatial.dim() > 2:
                spatial = spatial.sum(0)
            attributions[method_name] = spatial
            finite = bool(torch.isfinite(spatial).all())
            right_size = tuple(spatial.shape) == (IMAGE_SIZE, IMAGE_SIZE)
            check(f"{method_name}: finite values", finite)
            check(f"{method_name}: input resolution", right_size, str(tuple(spatial.shape)))

        # 5 ─ determinism of a stochastic method
        print("\n5. Determinism")
        def run_smoothgrad():
            set_seed(derive_seed(SEED, "resnet50", "smoothgrad", 0))
            fn = METHODS.get("smoothgrad")(model_wrapper=wrapper, model=wrapper.model)
            out = fn(img.unsqueeze(0), target=target)
            out = out[0] if isinstance(out, tuple) else out
            return out.detach().clone()

        check("stochastic method reproduces under the same derived seed",
              torch.allclose(run_smoothgrad(), run_smoothgrad(), atol=1e-6))

        # 6 ─ metrics
        # Grad-CAM's final ReLU can zero the whole map on a randomly initialised
        # backbone (all channel weights negative). That is expected without
        # trained weights, so drive the metrics with a map that is non-degenerate
        # by construction.
        print("\n6. Metrics")
        attr = attributions["saliency"].abs()
        check("attribution is non-degenerate for metric input", float(attr.max()) > 0)
        scores = {}
        for metric_name in ("pointing_game", "sparseness"):
            set_seed(derive_seed(SEED, "metric", metric_name, 0))
            try:
                val = METRICS.get(metric_name)(
                    attr=attr, metadata=meta, model=wrapper.model,
                    image=img, target=target, metric_kwargs={},
                )
                val = float(val)
                scores[metric_name] = val
                check(f"{metric_name} computed", np.isfinite(val), f"{val:.4f}")
            except Exception as exc:  # noqa: BLE001
                check(f"{metric_name} computed", False, f"{type(exc).__name__}: {exc}")

        # 7 ─ records + validation
        print("\n7. Result records and validation")
        rec = ResultRecord(
            model="resnet50", method="grad_cam",
            metrics={k: MetricValue(mean=v, std=0.0, count=N_IMAGES)
                     for k, v in scores.items()},
            provenance=Provenance(source_run="smoke_test", seed=SEED,
                                  git_commit=info.get("git_commit"),
                                  num_images=N_IMAGES),
        )
        try:
            assert_valid([rec])
            check("records pass schema validation", True)
        except Exception as exc:  # noqa: BLE001
            check("records pass schema validation", False, str(exc))

        bad = ResultRecord(model="resnet50", method="grad_cam",
                           metrics={"pointing_game": MetricValue(mean=float("nan"))})
        try:
            assert_valid([bad])
            check("validation rejects NaN results", False, "NaN was accepted")
        except Exception:  # noqa: BLE001
            check("validation rejects NaN results", True)

        # 8 ─ persistence
        print("\n8. Result persistence")
        payload = {"schema_version": "1.0.0", "records": [rec.to_dict()]}
        out_file = out_dir / "records.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reloaded = json.loads(out_file.read_text(encoding="utf-8"))
        check("results written and re-read",
              reloaded["records"][0]["model"] == "resnet50", str(out_file))
        check("provenance recorded on disk",
              reloaded["records"][0]["provenance"]["seed"] == SEED)

        # 9 ─ published results present and loadable
        print("\n9. Published benchmark results")
        records_path = REPO_ROOT / "results" / "processed" / "records.json"
        if records_path.exists():
            data = json.loads(records_path.read_text(encoding="utf-8"))
            recs = data.get("records", [])
            check("results/processed/records.json loads", len(recs) > 0, f"{len(recs)} records")
            check("no separate-paper records present",
                  not any("hilrp" in json.dumps(r).lower() for r in recs))
        else:
            check("results/processed/records.json loads", False,
                  "missing - run scripts/build_results.py")
    finally:
        if args.keep:
            print(f"\nTemp output kept at {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)

    failed = [name for name, ok, _ in _CHECKS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(_CHECKS) - len(failed)}/{len(_CHECKS)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
