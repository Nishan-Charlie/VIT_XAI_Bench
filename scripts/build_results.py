"""Consolidate benchmark run outputs into the canonical results tree.

The records are the five-metric grid of the paper: the seeded re-run in which
every image is scored (``GRID_RUNS``; configs/revision_grid_1000*.yaml), plus
the dense-mask localisation (EBPG) of the revision analysis (``EBPG_SUMMARY``).
Emits:

    results/processed/records.json   canonical, validated (model, method) records
    results/processed/manifest.json  coverage + provenance-gap report

Each cell reports its metric's cell statistic over all evaluated images
(xai_bench/scoring.py): the median for Max-Sensitivity, the mean otherwise.

Rows that do not belong to this benchmark (for example runs of a different
paper's method) are dropped here, and the drop is reported rather than silent.

Usage::

    python scripts/build_results.py
    # the superseded July 2026 summaries (unseeded, images with an undefined
    # metric left out, mean Max-Sensitivity):
    python scripts/build_results.py --raw results/raw/benchmark_runs.csv --no-ebpg \
        --out results/archive/july2026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai_bench import taxonomy  # noqa: E402
from xai_bench.results_schema import (  # noqa: E402
    SCHEMA_VERSION,
    MetricValue,
    Provenance,
    ResultRecord,
    validate_collection,
)
from xai_bench.scoring import cell_statistic  # noqa: E402

RESULTS = REPO_ROOT / "results"
#: Run directories of the seeded five-metric grid, each with the runner's
#: summary.csv and config.json. Every cell comes from exactly one of them: all
#: 1,000 images of every backbone and method (the earlier seeded runs on 100/50 and
#: 500 images, results/revision_grid_seeded* and revision_grid_500, are superseded and
#: only feed prefill_500.py and prefill_1000.py).
GRID_RUNS = ("revision_grid_1000",)
#: EBPG per cell over the 1,000 images (all 13 methods),
#: written by scripts/revision_neucom/analyze_revision.py.
EBPG_SUMMARY = RESULTS / "revision_neucom" / "analysis" / "summary.json"
#: Methods defined only on the isotropic ViT, whose class token they read.
ATTENTION_NATIVE = ("attention_rollout", "attnlrp")
DEFAULT_COST = REPO_ROOT / "results" / "raw" / "computational_cost.csv"
OUT_DIR = REPO_ROOT / "results" / "processed"

#: Runs belonging to a different paper. Their rows are excluded from this
#: benchmark; see docs/scientific_integrity.md.
EXCLUDED_RUN_PREFIXES = ("hilrp",)


def _float(raw: str | float | None) -> float | None:
    """A finite number, +/-inf, or None for a missing or NaN value."""
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return None if math.isnan(val) else val


def _finite(raw: str | float | None) -> float | None:
    val = _float(raw)
    return val if val is not None and math.isfinite(val) else None


def _int(raw: str | float | None) -> int | None:
    val = _finite(raw)
    return int(val) if val is not None else None


def metric_value(row: dict, metric: str) -> MetricValue | None:
    """The scored cell of one metric from a runner summary row, or None.

    The runner scores every image (xai_bench/scoring.py) and writes the mean,
    median, standard deviation, count and number of undefined images of the
    scored values. An unbounded image makes the mean infinite, stored as None;
    a cell whose median is infinite is stored as unbounded. Summaries without a
    median (the July 2026 runs) give a mean-only cell.
    """
    mean = _float(row.get(f"{metric}_mean"))
    if mean is None:
        return None
    median = _float(row.get(f"{metric}_median"))
    statistic = cell_statistic(metric) if median is not None else "mean"
    return MetricValue(
        mean=_finite(mean),
        std=_finite(row.get(f"{metric}_std")),
        count=_int(row.get(f"{metric}_count")),
        median=_finite(median),
        statistic=statistic,
        unbounded=statistic == "median" and math.isinf(median),
        n_undefined=_int(row.get(f"{metric}_undefined")),
    )


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
            mv = metric_value(row, metric_key)
            if mv is not None:
                metrics[metric_key] = mv

        if not metrics:
            report["dropped_no_metrics"] += 1
            continue

        counts = [m.count for m in metrics.values() if m.count is not None]
        run_cfg = row.get("_config") or {}
        n_eval = _int(row.get("n_evaluations"))
        rec = ResultRecord(
            model=model,
            method=method,
            metrics=metrics,
            provenance=Provenance(
                source_run=source_run or None,
                config_file=row.get("_config_file"),
                seed=(run_cfg.get("seeds") or [None])[0],
                dataset=run_cfg.get("dataset"),
                num_images=n_eval if n_eval is not None else (max(counts) if counts else None),
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


def load_grid_rows(results: Path = RESULTS, runs: tuple[str, ...] = GRID_RUNS) -> list[dict]:
    """Summary rows of the seeded grid, tagged with their run and configuration."""
    rows = []
    for run in runs:
        summary = results / run / "summary.csv"
        if not summary.exists():
            raise SystemExit(f"grid run not found: {summary}")
        cfg_path = results / run / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        config_file = f"configs/{run}.yaml"
        for row in load_rows(summary):
            row["source_run"] = f"{run}/summary.csv"
            row["_config"] = cfg
            row["_config_file"] = config_file if (REPO_ROOT / config_file).exists() else None
            rows.append(row)
    return rows


def add_ebpg(records: list[ResultRecord], path: Path = EBPG_SUMMARY) -> int:
    """Attach the dense-mask EBPG of the revision analysis to the grid records.

    EBPG is scored on the 1,000 images (100 for IG and the perturbation methods);
    a constant map gets no credit. Attention-native methods are defined only on
    the isotropic ViT. Returns the number of cells attached.
    """
    if not path.exists():
        raise SystemExit(f"EBPG summary not found: {path} (or pass --no-ebpg)")
    summary = json.loads(path.read_text(encoding="utf-8"))
    ebpg = summary["loc_gt_abs"]["ebpg_abs"]
    counts = summary["loc_n"]
    source = path.relative_to(REPO_ROOT).as_posix()
    by_cell = {(r.model, r.method): r for r in records}
    attached = 0
    for method, per_model in ebpg.items():
        for model, raw in per_model.items():
            rec = by_cell.get((model, method))
            value = _finite(raw)
            if rec is None or value is None:
                continue
            if method in ATTENTION_NATIVE and taxonomy.model_family(model) != "isotropic_vit":
                continue
            rec.metrics["ebpg"] = MetricValue(
                mean=value, count=_int(counts.get(method, {}).get(model)), source=source)
            attached += 1
    return attached



def load_cost(path: Path) -> list[dict]:
    """Method-level attribution timings.

    Kept separate from ``records.json`` because this is one figure per method,
    not one per (model, method) cell. Folding it into the cell records would
    mean repeating the same number across every backbone as though it had been
    measured there, which it was not.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))

    out = []
    for row in rows:
        method = (row.get("method") or "").strip()
        if method not in taxonomy.METHODS:
            print(f"  warning: unknown method in cost data: {method!r}")
            continue
        value = _float(row.get("time_ms"))
        if value is None:
            continue
        out.append({
            "method": method,
            "method_label": taxonomy.METHODS[method].label,
            "method_family": taxonomy.method_family(method),
            "time_ms": value,
            "restricted_backbones": (row.get("restricted_backbones") or "").strip().lower()
            == "true",
        })
    out.sort(key=lambda r: r["time_ms"])
    return out


def build_manifest(
    records: list[ResultRecord],
    report: dict,
    problems: list[str],
    cost: list[dict] | None = None,
) -> dict:
    """Coverage matrix and honest accounting of what is and is not measured."""
    cost = cost or []
    covered = {(r.model, r.method) for r in records}
    metric_coverage = {
        key: sum(1 for r in records if key in r.metrics) for key in taxonomy.METRICS
    }
    # Attribution timings live outside the per-cell records (one figure per
    # method), so count them from the cost table rather than from the records.
    if cost:
        metric_coverage["attribution_time_ms"] = len(cost)

    dimensions = {}
    for dim, metric_keys in taxonomy.METRICS_BY_DIMENSION.items():
        measured = sum(metric_coverage.get(k, 0) for k in metric_keys)
        dimensions[dim] = {
            "metrics": metric_keys,
            "records_with_data": measured,
            "status": "measured" if measured else "not_available",
            # Cost is reported per method, not per (model, method) cell.
            "granularity": "method" if dim == "computational_cost" else "cell",
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
    ap.add_argument("--raw", type=Path, default=None,
                    help="build from a flat summary CSV instead of the seeded grid runs")
    ap.add_argument("--no-ebpg", action="store_true",
                    help="do not attach the dense-mask EBPG of the revision analysis")
    ap.add_argument("--cost", type=Path, default=DEFAULT_COST,
                    help="method-level attribution timings CSV")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args()

    if args.raw is not None:
        if not args.raw.exists():
            raise SystemExit(f"raw results not found: {args.raw}")
        rows = load_rows(args.raw)
    else:
        rows = load_grid_rows()
    records, report = build_records(rows)
    if not args.no_ebpg:
        report["ebpg_cells"] = add_ebpg(records)
    records.sort(key=lambda r: (r.model, r.method))

    # Range violations are reported, not fatal: an unbounded metric such as
    # max-sensitivity legitimately exceeds naive bounds.
    problems = validate_collection(records, strict_range=True)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "records": [r.to_dict() for r in records],
    }
    # allow_nan=False: an infinite or NaN value must never reach the website,
    # whose JSON parser rejects it; an unbounded cell is stored with a flag.
    (args.out / "records.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    cost = load_cost(args.cost)
    (args.out / "method_cost.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "methods": cost}, indent=2),
        encoding="utf-8",
    )

    manifest = build_manifest(records, report, problems, cost)
    manifest["cost_methods"] = len(cost)
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
        elif info.get("granularity") == "method":
            print(f"  note: '{dim}' is reported per method, not per cell")
    if problems:
        print(f"\n{len(problems)} validation problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
    print(f"\nwrote {args.out/'records.json'} and {args.out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
