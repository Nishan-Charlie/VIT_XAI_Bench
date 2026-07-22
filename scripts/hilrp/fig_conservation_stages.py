"""Figure: HiLRP relevance conservation across Swin-B resolution stages.

Reads results/hilrp_gate2real/conservation.csv (per-image relevance sum / logit at
each capture) and plots mean +/- std per stage against the exact-conservation line.
Directly visualizes the paper's central quantitative claim: conservation is exact at
the token level and deviates, monotonically and by a measured amount, only at the
convolutional pixel stem.

  python scripts/hilrp/fig_conservation_stages.py    (CWD = repo root)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
})
GREEN, VERM, GREY = "#009E73", "#D55E00", "#8A8A8A"

cons = pd.read_csv("results/hilrp_gate2real/conservation.csv")
stages = ["tokens_56", "stage1_in_28", "stage2_in_14", "stage3_in_7", "final_7", "pixels"]
labels = ["tokens\n$56^2$", "stage 1\n$28^2$", "stage 2\n$14^2$", "stage 3\n$7^2$",
          "head\n$7^2$", "pixels\n$224^2$"]
mean, std = cons[stages].mean().values, cons[stages].std().values
x = np.arange(len(stages))

fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.axhline(1.0, color=GREY, ls="--", lw=1.2, zorder=1, label="exact conservation")
ax.fill_between(x, mean - std, mean + std, color=GREEN, alpha=0.15, zorder=1)
ax.errorbar(x, mean, yerr=std, marker="o", ms=7, lw=2, color=GREEN, capsize=3,
            zorder=3, label=r"HiLRP  $\sum_i R_i / f_c(x)$")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel(r"relevance sum / logit"); ax.set_ylim(0.55, 1.5)
ax.set_title(f"Conservation across Swin-B resolution stages ($n={len(cons)}$)", fontsize=11)
ax.annotate("bounded inflation\nat deep stages", (3, mean[3]), (2.0, 1.40),
            fontsize=8.5, color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
ax.annotate("$\\gamma$-rule deviation\nat the conv. stem", (5, mean[-1]), (4.0, 0.66),
            fontsize=8.5, color=VERM, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
ax.legend(frameon=False, loc="lower left", fontsize=9)
fig.tight_layout()
os.makedirs("figures/hilrp", exist_ok=True)
fig.savefig("figures/hilrp/conservation_stages.pdf", bbox_inches="tight")
fig.savefig("figures/hilrp/conservation_stages.png", dpi=300, bbox_inches="tight")
print("wrote figures/hilrp/conservation_stages.png")
print("stage means:", dict(zip(stages, np.round(mean, 3))))
