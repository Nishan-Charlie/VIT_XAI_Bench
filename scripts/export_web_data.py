"""Export the validated benchmark results as JSON for the website.

The website reads only what this script writes, so a rebuilt benchmark shows up
on the site by re-running::

    python scripts/build_results.py
    python scripts/export_web_data.py

Nothing is hardcoded in the frontend. Dimensions with no measurement are
exported with ``"status": "not_available"`` and a reason, so the UI can say so
explicitly instead of showing a blank or an invented number.

Outputs (website/public/data/):
    benchmark.json   models, methods, metrics, records, rankings, coverage
    meta.json        build provenance and dataset-level counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai_bench import taxonomy  # noqa: E402
from xai_bench.results_schema import SCHEMA_VERSION  # noqa: E402
from xai_bench.utils import provenance  # noqa: E402

DEFAULT_RECORDS = REPO_ROOT / "results" / "processed" / "records.json"
DEFAULT_MANIFEST = REPO_ROOT / "results" / "processed" / "manifest.json"
DEFAULT_OUT = REPO_ROOT / "website" / "public" / "data"

TRANSFORMER_FAMILIES = ("isotropic_vit", "hierarchical", "hybrid", "linear_attention")


def _val(rec: dict, metric: str) -> float | None:
    mv = (rec.get("metrics") or {}).get(metric)
    return mv.get("mean") if mv else None


def build_rankings(records: list[dict], metric: str) -> dict:
    """Per-scope method rankings for the ranking explorer.

    Scopes are ``all``, each architecture family, and the CNN/transformer split
    used by the transferability view.
    """
    def rank_for(models: list[str]) -> list[dict]:
        rows = []
        for method in taxonomy.METHODS:
            vals = [
                _val(r, metric) for r in records
                if r["method"] == method and r["model"] in models
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            rows.append({
                "method": method,
                "value": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "n_models": len(vals),
            })
        higher = taxonomy.METRICS[metric].higher_is_better
        rows.sort(key=lambda r: r["value"], reverse=higher)
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        return rows

    all_models = list(taxonomy.MODELS)
    scopes = {"all": rank_for(all_models)}
    for family in taxonomy.ARCH_FAMILIES:
        models = [m for m, s in taxonomy.MODELS.items() if s.family == family]
        rows = rank_for(models)
        if rows:
            scopes[family] = rows
    scopes["cnn_only"] = rank_for(
        [m for m, s in taxonomy.MODELS.items() if s.family == "cnn"]
    )
    scopes["transformer_only"] = rank_for(
        [m for m, s in taxonomy.MODELS.items() if s.family in TRANSFORMER_FAMILIES]
    )
    return scopes


def build_transfer(records: list[dict], metric: str) -> list[dict]:
    """CNN rank vs transformer rank per method — the paper's central question."""
    rankings = build_rankings(records, metric)
    cnn = {r["method"]: r for r in rankings.get("cnn_only", [])}
    tfm = {r["method"]: r for r in rankings.get("transformer_only", [])}
    out = []
    for method in sorted(set(cnn) & set(tfm)):
        out.append({
            "method": method,
            "cnn_rank": cnn[method]["rank"],
            "transformer_rank": tfm[method]["rank"],
            "cnn_value": cnn[method]["value"],
            "transformer_value": tfm[method]["value"],
            "rank_delta": tfm[method]["rank"] - cnn[method]["rank"],
        })
    out.sort(key=lambda r: r["cnn_rank"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.records.exists():
        print(f"ERROR: {args.records} not found. Run scripts/build_results.py first.")
        return 1

    payload = json.loads(args.records.read_text(encoding="utf-8"))
    records = payload["records"]
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest.exists() else {}
    )

    measured_metrics = sorted({m for r in records for m in (r.get("metrics") or {})})

    dimensions = []
    for dim in taxonomy.DIMENSIONS:
        keys = [k for k in taxonomy.METRICS_BY_DIMENSION[dim] if k in measured_metrics]
        if keys:
            dimensions.append({
                "key": dim,
                "label": dim.replace("_", " ").title(),
                "metrics": keys,
                "status": "measured",
            })
        else:
            dimensions.append({
                "key": dim,
                "label": dim.replace("_", " ").title(),
                "metrics": [],
                "status": "not_available",
                "reason": (
                    "The archived benchmark runs did not record runtime or GPU "
                    "memory. The current runner does (time_ms, peak_gpu_mb), so "
                    "re-running the benchmark will populate this dimension."
                    if dim == "computational_cost"
                    else "No metric for this dimension is present in the results."
                ),
            })

    data = {
        "schema_version": SCHEMA_VERSION,
        "models": [
            {
                "key": s.key, "label": s.label, "family": s.family,
                "family_label": taxonomy.ARCH_FAMILY_LABELS[s.family],
                "attention": s.attention,
                "in_results": any(r["model"] == s.key for r in records),
            }
            for s in taxonomy.MODELS.values()
        ],
        "methods": [
            {
                "key": s.key, "label": s.label, "family": s.family,
                "family_label": taxonomy.METHOD_FAMILY_LABELS[s.family],
                "stochastic": s.stochastic, "vit_only": s.vit_only,
                "in_results": any(r["method"] == s.key for r in records),
            }
            for s in taxonomy.METHODS.values()
        ],
        "metrics": [
            {
                "key": s.key, "label": s.label, "dimension": s.dimension,
                "higher_is_better": s.higher_is_better,
                "range": list(s.value_range) if s.value_range else None,
                "description": s.description,
                "measured": s.key in measured_metrics,
            }
            for s in taxonomy.METRICS.values()
        ],
        "arch_families": [
            {"key": f, "label": taxonomy.ARCH_FAMILY_LABELS[f]}
            for f in taxonomy.ARCH_FAMILIES
        ],
        "method_families": [
            {"key": f, "label": taxonomy.METHOD_FAMILY_LABELS[f]}
            for f in taxonomy.METHOD_FAMILIES
        ],
        "dimensions": dimensions,
        "records": records,
        "rankings": {m: build_rankings(records, m) for m in measured_metrics},
        "transfer": {m: build_transfer(records, m) for m in measured_metrics},
        "coverage": manifest.get("metric_coverage", {}),
        "missing_cells": manifest.get("missing_cells", []),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    prov = provenance.collect()
    meta = {
        "generated_utc": prov["timestamp_utc"],
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "schema_version": SCHEMA_VERSION,
        "n_records": len(records),
        "n_models_in_results": len({r["model"] for r in records}),
        "n_methods_in_results": len({r["method"] for r in records}),
        "measured_metrics": measured_metrics,
        "unmeasured_dimensions": [
            d["key"] for d in dimensions if d["status"] == "not_available"
        ],
        "provenance_gaps": manifest.get("provenance_gaps", {}),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = (args.out / "benchmark.json").stat().st_size / 1024
    print(f"wrote {args.out/'benchmark.json'} ({size:.0f} KB)")
    print(f"wrote {args.out/'meta.json'}")
    print(f"  {len(records)} records | {meta['n_models_in_results']} models | "
          f"{meta['n_methods_in_results']} methods")
    if meta["unmeasured_dimensions"]:
        print(f"  dimensions without data: {', '.join(meta['unmeasured_dimensions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
