# Revision analyses (NEUCOM-D-26-16590)

The analyses added when the paper was revised. Each script computes one kind of
per-image result on one backbone and appends it to a CSV in
`results/revision_neucom/`; `analyze_revision.py` turns those CSVs into the
tables and statistics, and `fig_revision.py` into the figures. The scripts change
to the repository root themselves, so they can be started from anywhere.

Every attribution is computed by the benchmark's own registry methods, and every
(seed, backbone, method, image) is seeded exactly as in `xai_bench/runner.py`.
Scoring follows `xai_bench/scoring.py`: a constant map gets no credit, and an
undefined Max-Sensitivity counts as unbounded.

| Script | What it measures | Output |
| --- | --- | --- |
| `acc.py [EXTRA_MODEL...]` | Top-1/top-5 accuracy and confidence of the eight backbones (and any extra ones) on the 1,000 images | `accuracy_*.csv` |
| `loc.py MODEL N METHODS` | Bounding-box PG, dense-mask EBPG and dense PG, under the ground-truth and the predicted class, for the absolute, positive and signed map | `loc_<model>_<methods>_n<N>.csv` |
| `ffe.py MODEL N METHODS` | FC and FE of the signed methods under the three map protocols | `ffe_*.csv` |
| `ms.py MODEL N SEEDS K METHODS` | Max-Sensitivity over several seeds and K perturbations, plus the bounded Grad-CAM++ ablation (`grad_cam_plus_plus_stab`) | `ms_*.csv` |
| `gcpp.py MODEL N` | Distance of the Grad-CAM++ weights from their pole, per image | `gcpp_*.csv` |
| `time_methods.py MODEL N METHODS` | GPU-synchronised time per explanation | `timing_*.csv` |
| `cost_table.py` | Mean time per method over the backbones where it applies | `analysis/cost_table.csv` |
| `analyze_revision.py` | All tables and statistics of the revision | `analysis/summary.json`, `analysis/tab_*.tex` |
| `fig_revision.py` | The Grad-CAM++ mechanism and capacity figures | `figures/revision/*.pdf` |
| `grid_tables.py` | The five grid tables and their Friedman / Nemenyi and ranking-transfer statistics | `analysis/grid_tables*.json` |

For example:

```bash
python scripts/revision_neucom/loc.py resnet50 1000 saliency,grad_cam
python scripts/revision_neucom/ms.py vit_base_patch16_224 500 0,1,2 10 grad_cam,grad_cam_plus_plus
python scripts/revision_neucom/analyze_revision.py
python scripts/revision_neucom/fig_revision.py
```

`extend_1000.sh` and `queue_lean.sh` are the queues that produced the first
results; `queue_500.py` (with `prefill_500.py`) extended them and the grid to
500 images, and the localisation of all 13 methods to 1,000 images.
`image_stability.py` measures how much the rankings depend on the images drawn, and
`fig_method_family.py` draws the method taxonomy figure. AttnLRP patches timm classes, so it must run alone, in its own process.
Each run resumes where it stopped, and a failed image is appended to
`results/revision_neucom/failures.csv` rather than skipped silently.
