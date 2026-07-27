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

### What was deliberately **kept**

The benchmark legitimately evaluates published relevance-propagation baselines,
and those must not be removed along with HiLRP:

* **AttnLRP** (Achtibat et al., 2024) — registered as `attnlrp`, has results in
  the benchmark, and is one of the evaluated methods.
* **CP-LRP** — the detached-softmax variant, also from the literature.

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

CI fails if the string `hilrp` reappears anywhere in the repository
(`.github/workflows/ci.yml`, job `results-integrity`), and
`scripts/validate_results.py` rejects any record whose provenance traces to an
excluded run.

---

## 2. Corrected defects that can change reported numbers

The audit found defects that affect values, not just ergonomics. They are fixed
in the current code, but **the published result set in `results/` was produced
by the previous code** and therefore still carries their effects. Re-running the
benchmark is required for the numbers to reflect the fixes.

### 2.1 The random seed was never applied (highest severity)

`xai_bench/utils/seed.py` defined `set_seed()` and `RunConfig` carried a `seeds`
field, but **no code path ever called it**. Six evaluated methods sample
internally — SmoothGrad, VarGrad, GradientSHAP, RISE, LIME — as does the
robustness metric (Max-Sensitivity draws perturbations) and the faithfulness
metrics (random subset selection).

Consequence: those rows were **not reproducible between runs**, and the reported
value depended on the interpreter's arbitrary starting RNG state.

Fix: the runner now seeds per evaluation item from
`blake2b(master_seed, model, method, image_idx)`. Because the seed derives from
the item's identity rather than from a running stream, a resumed run reproduces
a clean run exactly. `tests/test_seeding.py` pins this, including stability
across `PYTHONHASHSEED` (Python's `hash()` is randomised per process and would
have silently broken cross-machine reproducibility).

### 2.2 Failed evaluations were dropped silently

The old runner wrapped each image in `try/except Exception` and, on failure,
printed a line and moved on without recording anything. Quantus metric failures
similarly returned `NaN` with only a warning. Aggregation then took the mean
over surviving images.

Consequence: a method that failed on the hard 80% of images would be scored on
the easy 20%, with nothing in the output indicating it.

Fix: failures are written as explicit rows with `status` and `error`, the
summary carries an `n_errors` column per `(model, method)`, and per-metric
failures are recorded in `metric_errors`.

### 2.3 Computational cost was measured and then discarded

The runner recorded `time_ms` per attribution, but the summary aggregated only
`config.metrics`, so the cost column never reached `summary.csv`. This is why
"computational cost" — one of the paper's five dimensions — has **no data** in
the published result set.

Fix: `time_ms` and `peak_gpu_mb` are now aggregated as first-class columns.
Re-running the benchmark will populate the dimension. Until then, the figure
generator and the website both report it as `not_available` rather than
substituting a proxy.

### 2.4 The APF metric mutated shared configuration

`attention_preserving_faithfulness()` did:

```python
kwargs['metric_kwargs']['perturb_func'] = apf_perturb_func
```

`metric_kwargs` was passed through from the config object, so this wrote into
the live configuration dict. Fix: the dict is copied before modification.

---

## 3. Known inconsistency, not corrected

**Grad-CAM is computed from two different layers depending on which script runs
it, and the two paths feed the paper's bounding-box vs dense-mask comparison.**

* `xai_bench/methods/cam_methods.py` (used for all bbox/pointing-game results)
  takes `model.forward_features(x)` for every backbone, including isotropic
  ViTs.
* `scripts/bench/ebpg_eval.py` (dense-mask EBPG) re-implements Grad-CAM locally
  and, for ViTs, hooks `model.blocks[-1].norm1` instead — documented in that
  file as deliberate, since the ViT head pools only the class token.

The choice is defensible in isolation, but it means the bbox number and the
dense-mask number for ViT Grad-CAM come from **different layers**. The paper
attributes the gap between them to metric saturation. Part of it may instead be
a layer effect.

This has **not** been silently changed, because doing so would alter a reported
result. Recommended resolution: run both metrics through the single registry
Grad-CAM, or run both through the `blocks[-1].norm1` variant, and confirm the
dense-mask conclusion survives. See "Recommended next steps" in the audit report.

---

## 4. Scope discrepancies between the paper text and the result data

Stated for transparency; neither has been "fixed", because the result files are
the evidence and the paper text is not in this repository.

| Claim in the abstract | Present in `results/processed/records.json` |
|---|---|
| 13 attribution methods | **14** distinct methods |
| 8 representative backbones | **10** distinct backbones |
| 5 evaluation dimensions | **4** with data; computational cost has none |

The website and figures report the counts found in the data.
