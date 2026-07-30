"""Generate the paper figures from the canonical benchmark results.

Every figure is derived from ``results/processed/records.json`` — nothing is
hardcoded, so re-running the benchmark and rebuilding the records regenerates
the figures with the new numbers.

Figures whose underlying measurement does not exist in the results are **not**
drawn and are reported as skipped, with the reason. They are never approximated
from a proxy metric.

Usage::

    python scripts/generate_figures.py
    python scripts/generate_figures.py --out figures/paper --formats pdf png svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from xai_bench import taxonomy  # noqa: E402

# ── design tokens (validated categorical palette; see dataviz/references) ────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a87"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

# Sequential: one hue, light -> dark. Diverging: two hues + neutral gray middle.
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#7fb0e8", "#2a78d6", "#16407a"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_or_bl", ["#c2521f", "#eb6834", "#dedede", "#5b9ae0", "#1b4f96"])

FAMILY_COLOR = {
    "cnn": SERIES[0],
    "isotropic_vit": SERIES[1],
    "hierarchical": SERIES[2],
    "hybrid": SERIES[3],
    "linear_attention": SERIES[4],
}

FAMILY_METHOD_COLOR = {
    "gradient": SERIES[0],
    "cam": SERIES[2],
    "perturbation": SERIES[1],
    "attention_lrp": "#16407a",
}

TRANSFORMER_FAMILIES = ("isotropic_vit", "hierarchical", "hybrid", "linear_attention")

_SKIPPED: list[tuple] = []


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def skip(name: str, reason: str) -> None:
    _SKIPPED.append((name, reason))
    print(f"  SKIP  {name}\n        reason: {reason}")


def save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight",
                    dpi=300 if fmt == "png" else None)
    plt.close(fig)
    print(f"  OK    {name}  ->  {', '.join(formats)}")


# ─────────────────────────────── data access ────────────────────────────────

class Results:
    """Indexed view over the canonical records."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.by_cell: dict[tuple, dict] = {(r["model"], r["method"]): r for r in records}
        self.models = [m for m in taxonomy.MODELS if any(r["model"] == m for r in records)]
        self.methods = [m for m in taxonomy.METHODS if any(r["method"] == m for r in records)]

    def value(self, model: str, method: str, metric: str) -> float | None:
        rec = self.by_cell.get((model, method))
        if not rec:
            return None
        mv = rec.get("metrics", {}).get(metric)
        return mv.get("mean") if mv else None

    def matrix(self, metric: str) -> np.ndarray:
        """[method x model] matrix of means, NaN where not measured."""
        return np.array([
            [self.value(mo, me, metric) if self.value(mo, me, metric) is not None else np.nan
             for mo in self.models]
            for me in self.methods
        ], dtype=float)

    def has_metric(self, metric: str) -> bool:
        return any(metric in r.get("metrics", {}) for r in self.records)

    def family_models(self, family: str) -> list[str]:
        return [m for m in self.models if taxonomy.MODELS[m].family == family]


def load(records_path: Path) -> Results:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    return Results(payload["records"])


# ──────────────────────────────── figures ───────────────────────────────────

def fig_heatmaps(res: Results, out: Path, formats: list[str]) -> None:
    """Fig 1 — Architecture x Attribution heatmap, one panel per metric."""
    metrics = [m for m in ("pointing_game", "faithfulness_correlation",
                           "max_sensitivity", "sparseness") if res.has_metric(m)]
    if not metrics:
        return skip("architecture_method_heatmap", "no metrics present in results")

    for metric in metrics:
        spec = taxonomy.METRICS[metric]
        mat = res.matrix(metric)
        fig, ax = plt.subplots(figsize=(1.05 * len(res.models) + 2.6,
                                        0.36 * len(res.methods) + 1.9))
        finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            plt.close(fig)
            skip(f"heatmap_{metric}", "metric present but all values missing")
            continue

        im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto",
                       vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))

        ax.set_xticks(range(len(res.models)))
        ax.set_xticklabels([taxonomy.MODELS[m].label for m in res.models],
                           rotation=38, ha="right")
        ax.set_yticks(range(len(res.methods)))
        ax.set_yticklabels([taxonomy.METHODS[m].label for m in res.methods])

        # Cell values, with contrast-aware ink so the number is always legible.
        lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
        rng = (hi - lo) or 1.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "–", ha="center", va="center",
                            color=INK_MUTED, fontsize=7)
                    continue
                dark = (v - lo) / rng > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="#ffffff" if dark else INK_PRIMARY)

        # Separate architecture families with a surface-coloured gap.
        for j in range(1, len(res.models)):
            if taxonomy.MODELS[res.models[j]].family != taxonomy.MODELS[res.models[j - 1]].family:
                ax.axvline(j - 0.5, color=SURFACE, lw=2.5)

        ax.set_xticks(np.arange(-0.5, len(res.models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(res.methods), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        arrow = "higher is better" if spec.higher_is_better else "lower is better"
        ax.set_title(f"{spec.label}  ({arrow})", pad=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=0)
        fig.text(0.5, -0.02, "– = not measured", ha="center",
                 fontsize=7, color=INK_MUTED)
        save(fig, out, f"heatmap_{metric}", formats)


def fig_transfer(res: Results, out: Path, formats: list[str]) -> None:
    """Fig 3 — do CNN-established rankings transfer to transformers?"""
    metric = "pointing_game"
    if not res.has_metric(metric):
        return skip("cnn_to_transformer_transfer", f"{metric} not present in results")

    cnn_models = res.family_models("cnn")
    tf_models = [m for m in res.models if taxonomy.MODELS[m].family in TRANSFORMER_FAMILIES]
    if not cnn_models or not tf_models:
        return skip("cnn_to_transformer_transfer",
                    "results lack either CNN or transformer backbones")

    rows = []
    for me in res.methods:
        c = [res.value(m, me, metric) for m in cnn_models]
        t = [res.value(m, me, metric) for m in tf_models]
        c = [v for v in c if v is not None]
        t = [v for v in t if v is not None]
        if not c or not t:
            continue
        rows.append((me, float(np.mean(c)), float(np.mean(t))))
    if len(rows) < 3:
        return skip("cnn_to_transformer_transfer",
                    "fewer than 3 methods have results on both CNNs and transformers")

    rows.sort(key=lambda r: -r[1])
    names = [taxonomy.METHODS[r[0]].label for r in rows]
    cnn_rank = {r[0]: i for i, r in enumerate(sorted(rows, key=lambda r: -r[1]))}
    tf_rank = {r[0]: i for i, r in enumerate(sorted(rows, key=lambda r: -r[2]))}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 0.42 * len(rows) + 2.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: paired scores.
    y = np.arange(len(rows))
    ax0.barh(y - 0.19, [r[1] for r in rows], height=0.34, color=SERIES[0],
             label="CNNs", zorder=3)
    ax0.barh(y + 0.19, [r[2] for r in rows], height=0.34, color=SERIES[1],
             label="Transformers", zorder=3)
    ax0.set_yticks(y)
    ax0.set_yticklabels(names)
    ax0.invert_yaxis()
    ax0.set_xlabel(taxonomy.METRICS[metric].label)
    ax0.set_title("Localisation score by architecture class")
    ax0.xaxis.grid(True, zorder=0)
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    # Right: rank transfer (bump chart). Colour encodes direction of change.
    for me, _, _ in rows:
        r0, r1 = cnn_rank[me], tf_rank[me]
        delta = r1 - r0
        color = INK_MUTED if delta == 0 else (SERIES[1] if delta > 0 else SERIES[0])
        ax1.plot([0, 1], [r0, r1], color=color, lw=2.0, zorder=3,
                 solid_capstyle="round")
        ax1.scatter([0, 1], [r0, r1], s=34, color=color, zorder=4,
                    edgecolors=SURFACE, linewidths=2)
    for me, _, _ in rows:
        ax1.text(-0.045, cnn_rank[me], taxonomy.METHODS[me].label,
                 ha="right", va="center", fontsize=7.6, color=INK_SECONDARY)
        ax1.text(1.045, tf_rank[me], taxonomy.METHODS[me].label,
                 ha="left", va="center", fontsize=7.6, color=INK_SECONDARY)
    n_moved = sum(1 for me, _, _ in rows if cnn_rank[me] != tf_rank[me])
    ax1.set_xlim(-0.62, 1.62)
    ax1.set_ylim(len(rows) - 0.4, -0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Rank on CNNs", "Rank on Transformers"])
    ax1.set_yticks([])
    ax1.set_title(f"Rank transfer — {n_moved}/{len(rows)} methods change rank")
    for s in ax1.spines.values():
        s.set_visible(False)

    fig.suptitle("Does the CNN-established ranking transfer to transformers?",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, out, "cnn_to_transformer_transfer", formats)


def fig_family_performance(res: Results, out: Path, formats: list[str]) -> None:
    """Fig 6 — method performance grouped by architecture family."""
    metric = "pointing_game"
    if not res.has_metric(metric):
        return skip("architecture_family_performance", f"{metric} not present")

    families = [f for f in taxonomy.ARCH_FAMILIES if res.family_models(f)]
    methods = [me for me in res.methods
               if sum(1 for f in families
                      for mo in res.family_models(f)
                      if res.value(mo, me, metric) is not None) >= 2]
    if not methods:
        return skip("architecture_family_performance", "no method spans >=2 backbones")

    fig, ax = plt.subplots(figsize=(1.5 + 0.72 * len(methods), 4.2))
    width = 0.8 / len(families)
    x = np.arange(len(methods))
    for k, fam in enumerate(families):
        models = res.family_models(fam)
        vals, errs = [], []
        for me in methods:
            v = [res.value(mo, me, metric) for mo in models]
            v = [t for t in v if t is not None]
            vals.append(np.mean(v) if v else np.nan)
            errs.append(np.std(v) if len(v) > 1 else 0.0)
        ax.bar(x + k * width - 0.4 + width / 2, vals, width * 0.88,
               yerr=errs, color=FAMILY_COLOR[fam], zorder=3,
               label=taxonomy.ARCH_FAMILY_LABELS[fam],
               error_kw={"lw": 0.8, "ecolor": INK_MUTED, "capsize": 1.6})
    ax.set_xticks(x)
    ax.set_xticklabels([taxonomy.METHODS[m].label for m in methods],
                       rotation=32, ha="right")
    ax.set_ylabel(taxonomy.METRICS[metric].label)
    ax.set_title("Localisation by architecture family "
                 "(bars: mean over backbones; whiskers: spread)")
    ax.yaxis.grid(True, zorder=0)
    ax.legend(ncol=2, loc="upper right")
    save(fig, out, "architecture_family_performance", formats)


def fig_metric_correlation(res: Results, out: Path, formats: list[str]) -> None:
    """Fig 4 — how much do the evaluation dimensions agree?"""
    metrics = [m for m in taxonomy.METRICS if res.has_metric(m)]
    if len(metrics) < 3:
        return skip("metric_correlation", "fewer than 3 metrics present")

    cols = []
    for m in metrics:
        cols.append([res.value(mo, me, m)
                     for mo in res.models for me in res.methods])
    arr = np.array([[np.nan if v is None else v for v in c] for c in cols], dtype=float)

    n = len(metrics)
    corr = np.full((n, n), np.nan)
    pairs = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            mask = np.isfinite(arr[i]) & np.isfinite(arr[j])
            pairs[i, j] = int(mask.sum())
            if mask.sum() >= 4:
                a = arr[i][mask].argsort().argsort()      # Spearman via ranks
                b = arr[j][mask].argsort().argsort()
                if a.std() > 0 and b.std() > 0:
                    corr[i, j] = float(np.corrcoef(a, b)[0, 1])

    fig, ax = plt.subplots(figsize=(1.05 * n + 2.4, 1.05 * n + 1.9))
    im = ax.imshow(corr, cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    labels = [taxonomy.METRICS[m].label.replace(" ", "\n", 1) for m in metrics]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7.2)
    for i in range(n):
        for j in range(n):
            if np.isfinite(corr[i, j]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.4,
                        color="#ffffff" if abs(corr[i, j]) > 0.6 else INK_PRIMARY)
            else:
                ax.text(j, i, "–", ha="center", va="center",
                        color=INK_MUTED, fontsize=7)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    ax.set_title("Spearman correlation between evaluation dimensions\n"
                 f"(over {int(np.nanmax(pairs))} model x method cells)")
    save(fig, out, "metric_correlation", formats)



def fig_cost(out: Path, formats: list[str],
             cost_path: Path | None = None) -> None:
    """Fig 9 - attribution cost per method, log scale."""
    cost_path = cost_path or (REPO_ROOT / "results" / "processed" / "method_cost.json")
    if not cost_path.exists():
        return skip("computational_cost", f"{cost_path} not found")
    rows = json.loads(cost_path.read_text(encoding="utf-8")).get("methods", [])
    if not rows:
        return skip("computational_cost", "no attribution timings recorded")

    rows = sorted(rows, key=lambda r: r["time_ms"])
    labels = [r["method_label"] + (" †" if r.get("restricted_backbones") else "")
              for r in rows]
    values = [r["time_ms"] for r in rows]
    colors = [FAMILY_METHOD_COLOR.get(r["method_family"], INK_MUTED) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.7))
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    # Three orders of magnitude: a linear axis collapses every gradient method.
    ax.set_xscale("log")
    ax.set_xlabel("Time per explanation (ms, log scale)")
    ax.xaxis.grid(True, which="both", zorder=0)
    ax.set_axisbelow(True)

    for yi, v in zip(y, values, strict=True):
        text = f"{v/1000:.2f} s" if v >= 1000 else f"{v:.1f} ms"
        ax.text(v * 1.12, yi, text, va="center", fontsize=7, color=INK_SECONDARY)
    ax.set_xlim(right=max(values) * 3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_METHOD_COLOR[f])
        for f in taxonomy.METHOD_FAMILIES if any(r["method_family"] == f for r in rows)
    ]
    names = [taxonomy.METHOD_FAMILY_LABELS[f] for f in taxonomy.METHOD_FAMILIES
             if any(r["method_family"] == f for r in rows)]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=7.5)

    ratio = values[-1] / values[0]
    ax.set_title(f"Cost of one explanation — {rows[-1]['method_label']} is "
                 f"~{ratio:.0f}x {rows[0]['method_label']}")
    fig.text(0.5, -0.12, "† measured only on backbones where the method applies",
             ha="center", fontsize=7, color=INK_MUTED)
    save(fig, out, "computational_cost", formats)


def fig_coverage(res: Results, out: Path, formats: list[str]) -> None:
    """An honesty figure: exactly which cells the benchmark measured."""
    metrics = list(taxonomy.METRICS)
    grid = np.zeros((len(res.methods), len(res.models)))
    for i, me in enumerate(res.methods):
        for j, mo in enumerate(res.models):
            rec = res.by_cell.get((mo, me))
            grid[i, j] = len(rec.get("metrics", {})) if rec else 0

    fig, ax = plt.subplots(figsize=(1.02 * len(res.models) + 2.4,
                                    0.34 * len(res.methods) + 1.9))
    im = ax.imshow(grid, cmap=SEQ_CMAP, vmin=0, vmax=len(metrics), aspect="auto")
    ax.set_xticks(range(len(res.models)))
    ax.set_xticklabels([taxonomy.MODELS[m].label for m in res.models],
                       rotation=38, ha="right")
    ax.set_yticks(range(len(res.methods)))
    ax.set_yticklabels([taxonomy.METHODS[m].label for m in res.methods])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = int(grid[i, j])
            ax.text(j, i, str(v) if v else "·", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > len(metrics) * 0.55 else INK_PRIMARY)
    ax.set_xticks(np.arange(-0.5, len(res.models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(res.methods), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    ax.set_title(f"Benchmark coverage — metrics recorded per cell (max {len(metrics)})")
    save(fig, out, "benchmark_coverage", formats)


# ─────────────────────────────────── main ───────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path,
                    default=REPO_ROOT / "results" / "processed" / "records.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "paper")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    if not args.records.exists():
        print(f"ERROR: {args.records} not found. Run scripts/build_results.py first.")
        return 1

    _style()
    res = load(args.records)
    print(f"Loaded {len(res.records)} records "
          f"({len(res.models)} models x {len(res.methods)} methods)\n")

    fig_heatmaps(res, args.out, args.formats)
    fig_transfer(res, args.out, args.formats)
    fig_family_performance(res, args.out, args.formats)
    fig_metric_correlation(res, args.out, args.formats)
    fig_cost(args.out, args.formats)
    fig_coverage(res, args.out, args.formats)

    # Figures the paper describes but the current results cannot support.
    if not res.has_metric("pointing_game_dense"):
        skip("bbox_vs_dense_mask",
             "dense-mask localisation (EBPG / dense pointing game) is not in "
             "results/processed/records.json. Produce it with "
             "scripts/bench/ebpg_eval.py, then rebuild the records.")
    skip("robustness_curves",
         "results store only aggregated max-sensitivity, not per-perturbation-"
         "magnitude curves. A sweep over perturbation strength is required.")
    skip("faithfulness_curves",
         "results store only aggregated correlations, not deletion/insertion "
         "curves. Per-step pixel-flipping traces are required.")
    skip("explanation_gallery",
         "cannot be derived from aggregated records — it needs the attribution "
         "maps themselves. Static versions already exist at "
         "figures/bench/{qualitative_grid_cat,bench_qualitative,"
         "cam_upsampling_artifact}.png; regenerate them with "
         "scripts/bench/qualitative_grid_cat.py on a GPU.")

    print(f"\n{len(_SKIPPED)} figure(s) skipped for lack of source data.")
    print(f"Figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
