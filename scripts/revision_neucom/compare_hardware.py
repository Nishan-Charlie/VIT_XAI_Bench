"""Agreement between two runs of the same seeded evaluations on different machines.

Every (seed, model, method, image) evaluation is seeded on its own, so two
machines running the same rows differ only through floating-point arithmetic
(GPU architecture, kernels, thread counts). This script matches the rows of a
reference run and a new run on (model, method, image_idx, seed) and reports, per
metric and method, how many values are identical, how far the others move, and
how many binary Pointing Game hits flip.

usage: python compare_hardware.py REF_RESULTS[,REF_RESULTS...] NEW_RESULTS[,NEW_RESULTS...]
       (results.csv or results.json files of xai_bench.runner runs)
"""
import json
import sys

import numpy as np
import pandas as pd

METRICS = ["pointing_game", "faithfulness_correlation", "faithfulness_estimate", "max_sensitivity", "sparseness"]
KEY = ["model", "method", "image_idx", "seed"]


def load(paths):
    frames = []
    for p in paths.split(","):
        if p.endswith(".json"):
            with open(p, encoding="utf-8") as fh:
                frames.append(pd.DataFrame(json.load(fh)))
        else:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if "error" in df:
        df = df[df["error"].isna() | (df["error"] == "")]
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.drop_duplicates(KEY)


def main(ref_paths, new_paths):
    ref, new = load(ref_paths), load(new_paths)
    j = ref.merge(new, on=KEY, suffixes=("_ref", "_new"))
    print(f"{len(j)} matched evaluations ({j.model.nunique()} backbones, {j.method.nunique()} methods, "
          f"{j.image_idx.nunique()} images)")
    if "degenerate_map_ref" in j:
        d = (j.degenerate_map_ref.astype(str) != j.degenerate_map_new.astype(str)).sum()
        print(f"constant-map flag differs on {d} evaluations")
    out = []
    for m in METRICS:
        a, b = j[f"{m}_ref"].to_numpy(float), j[f"{m}_new"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        diff = np.abs(a[both] - b[both])
        rel = diff / np.maximum(np.abs(a[both]), 1e-12)
        row = dict(metric=m, n=int(both.sum()), identical=float(np.mean(diff == 0)) if both.any() else np.nan,
                   median_abs=float(np.median(diff)) if both.any() else np.nan,
                   p99_abs=float(np.quantile(diff, 0.99)) if both.any() else np.nan,
                   max_abs=float(diff.max()) if both.any() else np.nan,
                   median_rel=float(np.median(rel)) if both.any() else np.nan, defined_mismatch=nan_mismatch)
        if m == "pointing_game":
            row["flips"] = int((diff > 0.5).sum())
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nper method, share of identical values over the five metrics:")
    per = {}
    for me, g in j.groupby("method"):
        vals = []
        for m in METRICS:
            a, b = g[f"{m}_ref"].to_numpy(float), g[f"{m}_new"].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            vals += list(a[both] == b[both])
        per[me] = (np.mean(vals) if vals else np.nan, len(g))
    for me, (v, n) in sorted(per.items()):
        print(f"  {me:24s} {100 * v:5.1f}% identical  ({n} evaluations)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
