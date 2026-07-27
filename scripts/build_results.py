"""Consolidate raw benchmark run outputs into the canonical results tree.

Reads the aggregated per-run summaries under ``results/raw/`` and emits:

    results/processed/records.json   canonical, validated (model, method) records
    results/processed/manifest.json  coverage + provenance-gap report

Rows that do not belong to this benchmark (for example runs of a different
paper's method) are dropped here, and the drop is reported rather than silent.

Usage::

    python scripts/build_results.py
    python scripts/build_results.py --raw results/raw/benchmark_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from xai_bench import taxonomy
from xai_bench.results_schema import (
    SCHEMA_VERSION,
    MetricValue,
    Provenance,
    ResultRecord,
    validate_collection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "results" / "raw" / "benchmark_runs.csv"
OUT_DIR = REPO_ROOT / "results" / "processed"

#: Runs belonging to a different paper. Their rows are excluded from this
#: benchmark; see docs/scientific_integrity.md.
EXCLUDED_RUN_PREFIXES = ("hilrp",)


def _float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return None if math.isnan(val) else val


def _is_excluded(source_run: str) -> bool:
    stem = source_run.split("/")[0].lower()
    return any(stem.startswith(p) for p in EXCLUDED_RUN_PREFIXES)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def build_records(rows: list[dict]) -> tuple[list[ResultRecord], dict]:
    """Turn raw summary rows into validated records plus a build report."""
    report = {
        "rows_read": len(rows),
        "dropped_excluded_run": 0,
        "dropped_no_identity": 0,
        "dropped_unknown_model": Counter(),
        "dropped_unknown_method": Counter(),
        "dropped_no_metrics": 0,
    }
    records: dict[tuple, ResultRecord] = {}

    for row in rows:
        source_run = (row.get("source_run") or "").strip()
        if _is_excluded(source_run):
            report["dropped_excluded_run"] += 1
            continue

        model = (row.get("model") or "").strip()
        method = (row.get("method") or "").strip()
        if not model or not method:
            # per-image diagnostic rows carry no (model, method) identity
            report["dropped_no_identity"] += 1
            continue
        if model not in taxonomy.MODELS:
            report["dropped_unknown_model"][model] += 1
            continue
        if method not in taxonomy.METHODS:
            report["dropped_unknown_method"][method] += 1
            continue

        metrics: dict[str, MetricValue] = {}
        for metric_key in taxonomy.METRICS:
            mean = _float(row.get(f"{metric_key}_mean"))
            if mean is None:
                continue
            std = _float(row.get(f"{metric_key}_std"))
            count_raw = _float(row.get(f"{metric_key}_count"))
            metrics[metric_key] = MetricValue(
                mean=mean,
                std=std,
                count=int(count_raw) if count_raw is not None else None,
            )

        if not metrics:
            report["dropped_no_metrics"] += 1
            continue

        counts = [m.count for m in metrics.values() if m.count is not None]
        rec = ResultRecord(
            model=model,
            method=method,
            metrics=metrics,
            provenance=Provenance(
                source_run=source_run or None,
                num_images=max(counts) if counts else None,
            ),
        )
        key = (model, method)
        if key in records:
            # Later runs supersede earlier ones for the same cell; keep the one
            # with the larger sample count so a re-run at higher n wins.
            prev = records[key]
            prev_n = prev.provenance.num_images or 0
            if (rec.provenance.num_images or 0) <= prev_n:
                continue
        records[key] = rec

    report["dropped_unknown_model"] = dict(report["dropped_unknown_model"])
    report["dropped_unknown_method"] = dict(report["dropped_unknown_method"])
    return list(records.values()), report


def build_manifest(records: list[ResultRecord], report: dict, problems: list[str]) -> dict:
    """Coverage matrix and honest accounting of what is and is not measured."""
    covered = {(r.model, r.method) for r in records}
    metric_coverage = {
        key: sum(1 for r in records if key in r.metrics) for key in taxonomy.METRICS
    }
    dimensions = {}
    for dim, metric_keys in taxonomy.METRICS_BY_DIMENSION.items():
        measured = sum(metric_coverage.get(k, 0) for k in metric_keys)
        dimensions[dim] = {
            "metrics": metric_keys,
            "records_with_data": measured,
            "status": "measured" if measured else "not_available",
        }

    missing_cells = []
    for model in taxonomy.MODELS:
        for method in taxonomy.METHODS:
            if (model, method) in covered:
                continue
            spec = taxonomy.METHODS[method]
            reason = (
                "not_applicable"
                if spec.vit_only and not taxonomy.MODELS[model].family == "isotropic_vit"
                else "not_run"
            )
            missing_cells.append({"model": model, "method": method, "reason": reason})

    prov_gaps = Counter()
    for rec in records:
        for field_name in rec.provenance.missing_fields():
            prov_gaps[field_name] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(records),
        "n_models": len({r.model for r in records}),
        "n_methods": len({r.method for r in records}),
        "build_report": report,
        "metric_coverage": metric_coverage,
        "dimensions": dimensions,
        "missing_cells": missing_cells,
        "provenance_gaps": dict(prov_gaps),
        "validation_problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw summary CSV")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"raw results not found: {args.raw}")

    rows = load_rows(args.raw)
    records, report = build_records(rows)
    records.sort(key=lambda r: (r.model, r.method))

    # Range violations are reported, not fatal: an unbounded metric such as
    # max-sensitivity legitimately exceeds naive bounds.
    problems = validate_collection(records, strict_range=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [r.to_dict() for r in records],
    }
    (args.out / "records.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    manifest = build_manifest(records, report, problems)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"rows read              : {report['rows_read']}")
    print(f"dropped (other paper)  : {report['dropped_excluded_run']}")
    print(f"dropped (no identity)  : {report['dropped_no_identity']}")
    print(f"dropped (unknown model): {report['dropped_unknown_model']}")
    print(f"dropped (unknown meth.): {report['dropped_unknown_method']}")
    print(f"records written        : {len(records)}")
    print(f"  models {manifest['n_models']}  methods {manifest['n_methods']}")
    print("metric coverage:")
    for key, n in manifest["metric_coverage"].items():
        print(f"  {key:28s} {n:3d} records")
    for dim, info in manifest["dimensions"].items():
        if info["status"] == "not_available":
            print(f"  ** dimension '{dim}' has NO data **")
    if problems:
        print(f"\n{len(problems)} validation problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
    print(f"\nwrote {args.out/'records.json'} and {args.out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
