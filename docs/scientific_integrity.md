# Scientific integrity record

This document records decisions that affect what the benchmark reports, so that
a reader can tell what was measured, what was excluded, and why.

---

## 1. Exclusion of HiLRP results

**HiLRP is the novel contribution of a separate paper and is not part of this
benchmark.** It was previously present in this repository as an implementation,
a set of configs, a set of figures, and rows inside the aggregated results file.
All of it has been removed.

### What was removed

| Kind | Path |
|---|---|
| Method implementation | `xai_bench/methods/hilrp/` (17 modules + tests) |
| Registry entry | `xai_bench/methods/hilrp_method.py` (registered `"hilrp"`) |
| Configs | `configs/hilrp_grid*.yaml` (4 files) |
| Figures | `figures/hilrp/`, `figures/hilrp_diagrams.drawio.xml` |
| Comparison scripts | `cheetah_comparison.py`, `qualitative_comparison.py`, `paper_comparison_figures.py`, `heatmap_gallery.py`, `scaled_eval.py`, `diagnose_vit_mobilevit.py` |
| Tests of the HiLRP backend | `tests/test_mobilevit*.py`, `tests/test_detached_mean.py` |
| Result rows | 400 rows sourced from `hilrp_gate3/*` runs |

Everything is recoverable from git history; nothing was destroyed.

Four further scripts were deleted as **unrunnable dead code** discovered when CI
lint flagged their import block: `scripts/sweep_gamma.py`,
`scripts/sweep_gamma_fast.py`, `scripts/test_cplrp.py` and
`scripts/test_cplrp_deit.py`. All four did
`from scripts.scaled_eval import load_cache, pointing` — a module path that
**never existed** (`scripts/` is not a package, and the only `scaled_eval.py`
lived at `scripts/bench/`). They could not execute before this audit either, and
their sole helper source was the HiLRP win-condition script removed above.

### What was deliberately **kept**

The benchmark legitimately evaluates published relevance-propagation baselines,
and those must not be removed along with HiLRP:

* **AttnLRP** (Achtibat et al., 2024) — registered as `attnlrp`, has results in
  the benchmark, and is one of the evaluated methods.
* **CP-LRP** — the detached-softmax variant, also from the literature (not part
  of the published grid).

Both were implemented inside the `hilrp/` package directory (`vit_lxt.py`) even
though neither is HiLRP. That code was extracted into
`xai_bench/methods/vit_lrp_backend.py`, stripped of HiLRP framing and of the
label-free SSL-similarity branch (which belongs to the other paper), and
`attnlrp_method.py` now imports from there.

**Deleting the `hilrp/` package without this extraction would have silently
broken the `attnlrp` benchmark method.**

### A labelling defect found during the audit

`scripts/plot_grouped_bar.py`, `scripts/plot_max_sensitivity.py`,
`scripts/plot_pareto_scatter.py` and `scripts/run_imagenets_xai.py` contained:

```python
'hilrp': 'AttnLRP',
```

This renders rows of the method `hilrp` under the display name of the published
baseline `attnlrp`. Any figure built from a results file containing `hilrp` rows
would therefore present a separate paper's novel-method results as if they were
the AttnLRP baseline.

* The mapping has been removed from all four scripts.
* The three tracked figures produced by those scripts have been moved to
  `figures/_unverified/` with an explanation. Their source CSV
  (`results/clean_summary.csv`) is absent from the repository, so it is **not
  possible to determine retrospectively** whether these particular renders were
  affected. They are quarantined rather than deleted or trusted.

### Enforcement

Three independent guards:

1. **`scripts/check_excluded_methods.py`** scans the files the repository
   actually publishes (`git ls-files`) for `hilrp` / `hi-lrp` / `hi_lrp`, and
   exits non-zero on any hit. A short allow-list covers the files that name the
   method *in order to document its removal* — this record, the
   quarantine notice, and the exclusion logic itself.
2. **`scripts/validate_results.py`** rejects any record whose provenance traces
   to an excluded run.
3. **`tests/test_excluded_methods.py`** asserts the guard catches every spelling
   while *not* matching the published `attnlrp` / `CP-LRP` baselines, and that
   the AttnLRP backend survived the removal.

CI runs all three in the `results-integrity` job.

---

## 2. Corrected defects that can change reported numbers

The July 2026 audit and the November 2026 revision found defects that affect
values, not just ergonomics. All are fixed in the current code, and **the
published result set was produced again with it**: the five-metric grid in
`results/revision_grid_seeded*/` and the localisation, robustness and CAM
analyses in `results/revision_neucom/`. The July summaries, which carry the
effects of §2.1–2.3 and §2.5, stay in `results/raw/benchmark_runs.csv` for the
record only; no figure, table or web page is built from them.

### 2.1 The random seed was never applied (highest severity)

`xai_bench/utils/seed.py` defined `set_seed()` and `RunConfig` carried a `seeds`
field, but **no code path ever called it**. Five evaluated methods sample
internally (SmoothGrad, VarGrad, GradientSHAP, RISE, LIME), as do the
robustness metric (Max-Sensitivity draws perturbations) and Faithfulness
Correlation (random subset selection).

Consequence: those rows were **not reproducible between runs**, and the reported
value depended on the interpreter's arbitrary starting RNG state.

Fix: the runner reseeds Python, NumPy and PyTorch before every evaluation from
`crc32(seed|model|method|image_idx)`, and the attribution and its metrics then
draw from that one stream. Because the seed derives from the evaluation's
identity rather than from a running stream, a resumed run reproduces a clean run
exactly, and reordering methods or dropping a model changes no other row.
`tests/test_seeding.py` pins this, including stability across `PYTHONHASHSEED`
(Python's `hash()` is randomised per process).

### 2.2 Failed evaluations were dropped silently

The old runner wrapped each image in `try/except Exception` and, on failure,
printed a line and moved on without recording anything. Quantus metric failures
similarly returned `NaN` with only a warning. Aggregation then took the mean
over surviving images.

Consequence: a method that failed on the hard 80% of images would be scored on
the easy 20%, with nothing in the output indicating it.

Fix: a failed attribution is a row with an `error` field; `summary.csv` counts
it (`n_failed`), `failures.json` lists it, and a resumed run retries it. A metric
that raises leaves the other metrics of its row intact and records its reason in
`metric_errors`. The revision scripts append their failures to
`results/revision_neucom/failures.csv`.

### 2.3 Images with an undefined metric were left out

A method can return a constant map (Grad-CAM++ near the pole of its weights, an
all-zero LIME surrogate). Every metric is undefined on it, so the image used to
drop out of the cell's mean, and a method was averaged over the images on which
it happened to work. Quantus's bounding-box Pointing Game went the other way: it
counts a hit when any maximal pixel lies in the box, so a constant map, all of
whose pixels tie for the maximum, scored a hit.

Fix (`xai_bench/scoring.py`): every image is scored. A constant map gets no
credit (0 on PG, EBPG, FC, FE and SP). For Max-Sensitivity it counts as a
robustness failure, scored +∞; because MS is unbounded and heavy-tailed, its
cells are medians over all images, and a cell is ∞ when the method fails on at
least half of them. The runner flags constant maps (`degenerate_map`) and
`summary.csv` reports the scored statistics next to the number of undefined
images, so the raw counts stay visible.

### 2.4 Computational cost was measured and then discarded

The runner recorded `time_ms` per attribution, but the summary aggregated only
`config.metrics`, so the cost column never reached `summary.csv`, and the timed
window did not wait for queued GPU work.

Fix: `time_ms` (GPU-synchronised) and `peak_gpu_mb` are aggregated as
first-class columns. The published cost table comes from a separate synchronised
re-timing on the eight backbones (`scripts/revision_neucom/time_methods.py`,
`results/raw/computational_cost.csv`).

### 2.5 Grad-CAM on ViT-B/16 read a layer without gradient

The head of a class-token ViT reads only token 0 of `forward_features`, so the
patch tokens of that output receive no gradient and Grad-CAM on them is an
all-zero map. Between 1 July and 11 November 2026 the registry Grad-CAM read
`forward_features` on every backbone, including the class-token ViTs.

Fix: for class-token ViTs the CAM methods target the input normalisation of the
last block (`blocks[-1].norm1`, the standard pytorch-grad-cam choice), and any
CAM whose target layer receives no gradient at any position now raises instead
of returning a blank map. The published gallery had been rendered with the
faulty code: its ViT-B/16 Grad-CAM and Grad-CAM++ cells were blank. It was
replaced.

### 2.6 ViT-B/16 received the wrong input statistics

ViT-B/16 (mean = std = 0.5) and MobileViT-v2 (mean 0, std 1) were trained with
non-ImageNet statistics but received ImageNet-normalised images; ViT-B/16 drops
from 88.1% to 84.8% top-1 on the 1,000 images under them. Fix: an exact affine
map inside the model (`InputStatsAdapter`) converts the input for every call
path, including the Captum and Quantus perturbations.

### 2.7 AttnLRP was shown where only a fallback runs

AttnLRP's propagation rules are defined for the isotropic ViT. On the other
backbones the code falls back to generic LXT rules, which are not AttnLRP. The
old gallery drew those fallback maps under the AttnLRP heading on six backbones.
They are now shown as not applicable, and AttnLRP is evaluated on ViT-B/16 only.

### 2.8 The APF metric mutated shared configuration

`attention_preserving_faithfulness()` did:

```python
kwargs['metric_kwargs']['perturb_func'] = apf_perturb_func
```

`metric_kwargs` was passed through from the config object, so this wrote into
the live configuration dict. Fix: the dict is copied before modification. APF is
not part of the published grid.

### 2.9 The command-line runner started with empty registries

`python -m xai_bench.runner` imported the registries from
`xai_bench.registry`, which is filled only when the registries are first read
through the `xai_bench` package, so every run stopped at "unknown dataset". The
runner now imports them through the package.

---

## 3. Resolved: one Grad-CAM layer for both localisation metrics

The July audit recorded that Grad-CAM came from two different layers depending
on the script: the registry read `forward_features` (the bounding-box Pointing
Game) while `scripts/bench/ebpg_eval.py` re-implemented Grad-CAM and hooked
`blocks[-1].norm1` on ViTs (the dense-mask EBPG). Part of the gap between the two
metrics could therefore have been a layer effect.

Both published localisation metrics now come from the registry Grad-CAM, which
reads `blocks[-1].norm1` on class-token ViTs (§2.5): the Pointing Game of the
seeded grid and the EBPG of `scripts/revision_neucom/loc.py`, which scores the
bounding-box PG and EBPG from the same attribution. `ebpg_eval.py` is kept only
as the older, standalone script; no published number comes from it.

---

## 4. Scope of the published results

| Claim in the paper | In `results/processed/records.json` |
|---|---|
| 13 attribution methods | 13: the 11 methods defined on every backbone, plus Attention Rollout and AttnLRP on ViT-B/16 |
| 8 representative backbones | 8 |
| 5 evaluation dimensions | 5: faithfulness, localisation, robustness and complexity per cell; computational cost per method |
| 1,000 evaluation images | EBPG on the 1,000 (100 for IG and the perturbation methods); the five-metric grid on the leading 100 (ResNet-50, ViT-B/16) or 50 images |

The website and figures report the counts found in the data.
