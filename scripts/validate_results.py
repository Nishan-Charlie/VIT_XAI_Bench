"""Validate the published benchmark results against the canonical schema.

Runs standalone — no torch, no GPU — so CI can gate every change on result
integrity. Exits non-zero when a record is malformed, duplicated, out of range,
or carries a name that is not in the taxonomy.

Usage::

    python scripts/validate_results.py
    python scripts/validate_results.py --records results/processed/records.json
    python scripts/validate_results.py --warn-only
"""

from __future__ import annotations

import argparse
import json
import sys
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

DEFAULT_RECORDS = REPO_ROOT / "results" / "processed" / "records.json"


def load_records(path: Path) -> tuple[list[ResultRecord], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for raw in payload.get("records", []):
        metrics = {
            key: MetricValue(
                mean=val.get("mean"), std=val.get("std"), count=val.get("count")
            )
            for key, val in (raw.get("metrics") or {}).items()
        }
        prov_raw = raw.get("provenance") or {}
        known = set(Provenance.__dataclass_fields__)
        prov = Provenance(**{k: v for k, v in prov_raw.items() if k in known})
        records.append(
            ResultRecord(
                model=raw.get("model", ""),
                method=raw.get("method", ""),
                metrics=metrics,
                provenance=prov,
            )
        )
    return records, payload.get("schema_version", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument(
        "--warn-only",
        action="store_true",
        help="report problems but exit 0 (useful while results are in flux)",
    )
    args = ap.parse_args()

    if not args.records.exists():
        print(f"ERROR: no results at {args.records}")
        print("Run: python scripts/build_results.py")
        return 1

    records, version = load_records(args.records)
    print(f"Validating {args.records.relative_to(REPO_ROOT)}")
    print(f"  schema version : {version} (expected {SCHEMA_VERSION})")
    print(f"  records        : {len(records)}")
    print(f"  models         : {len({r.model for r in records})}")
    print(f"  methods        : {len({r.method for r in records})}")

    if version != SCHEMA_VERSION:
        print(f"  WARNING: schema version mismatch ({version} != {SCHEMA_VERSION})")

    if not records:
        print("ERROR: results file contains no records")
        return 1

    problems = validate_collection(records, strict_range=True)

    # Integrity guard: no record may reference a run from a different paper.
    for rec in records:
        source = (rec.provenance.source_run or "").lower()
        if "hilrp" in source or "hilrp" in rec.method.lower():
            problems.append(
                f"{rec.model}/{rec.method}: record traces to an excluded run "
                f"('{rec.provenance.source_run}') belonging to a different paper"
            )

    # Coverage report — informational, never fatal.
    print("\nDimension coverage:")
    for dim, metric_keys in taxonomy.METRICS_BY_DIMENSION.items():
        n = sum(1 for r in records if any(k in r.metrics for k in metric_keys))
        flag = "" if n else "   <- NO DATA"
        print(f"  {dim:20s} {n:4d} records{flag}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        if not args.warn_only:
            return 1
        print("\n(--warn-only: exiting 0)")
        return 0

    print("\nAll records valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
