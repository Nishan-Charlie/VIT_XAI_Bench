# Reproducibility guide

Exact commands, expected output, and honest statements about what this
repository can and cannot reproduce on its own.

---

## 1. Environment

| Requirement | Value |
|---|---|
| Python | 3.10 – 3.12 (3.11 recommended; CI tests all three) |
| PyTorch | ≥ 2.1 |
| CUDA | 12.1 / 12.4 for GPU runs; CPU works for everything except the full sweep |
| GPU VRAM | ≥ 8 GB for `*_base` backbones at batch size 1 |
| System RAM | ≥ 16 GB |
| Disk | ~5 GB for timm weights + ~2 GB for cached data |

Verified locally on: PyTorch 2.6.0+cu124, CUDA 12.4, NVIDIA RTX 4070 Laptop.

```bash
git clone https://github.com/Nishan-Charlie/VIT_XAI_Bench
cd VIT_XAI_Bench

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Relevance-propagation methods (attnlrp) need extra packages:
pip install -e ".[lrp]"
```

`quantus` is a **core** dependency, not an extra — every module in
`xai_bench.metrics` imports it at module level.

### Lightweight analysis without the ML stack

Result analysis and validation deliberately do not require torch:

```python
from xai_bench import taxonomy, results_schema   # no torch import triggered
```

This is what CI's `results-integrity` job uses.

---

## 2. Verify the install

```bash
python scripts/smoke_test.py
```

Runs the whole pipeline — dataset → model → attribution → metrics → records →
validation → JSON — on synthetic data with a randomly initialised ResNet-18. No
downloads, no GPU, ~30 s.

Expected output ends with:

```
25/25 checks passed
Smoke test PASSED
```

Then the test suite:

```bash
pytest tests -v
```

Expected: **101 passed**.

---

## 3. Data

The benchmark evaluates on **ImageNet-S** — ImageNet validation images paired
with pixel-level semantic segmentation masks. Neither the images nor the masks
are redistributed here; `data/` is gitignored.

| | |
|---|---|
| Source | [`braceletboy/imagenet-s`](https://huggingface.co/datasets/braceletboy/imagenet-s) (Hugging Face Hub) |
| Split | `validation` |
| Original project | [LUSSeg/ImageNet-S](https://github.com/LUSSeg/ImageNet-S) — cite `gao2022luss` |

Both dataset entry points resolve to the same source:

```python
from datasets import load_dataset

ds = load_dataset("braceletboy/imagenet-s", split="validation", streaming=True)
```

* `imagenets` — streams directly (`xai_bench/datasets/imagenet_s.py`)
* `imagenets_cached` — reads a local `.pt` cache; **this is what the configs use**

Build the caches once:

```bash
pip install -e ".[data]"            # `datasets`, needed only for streaming
python scripts/cache_imagenets.py   # -> data/ImageNetS/cache_validation_*.pt
python scripts/cache_voc.py         # -> data/VOC/cache_voc_1000.pt  (VOC-2007 boxes)
```

`cache_imagenets.py` retries because HF streaming drops connections on long
pulls. Caching is not just a speed optimisation: it pins every run to a **fixed
sample set**, so two runs compare the same images rather than whatever the
stream happened to yield.

> If the caches are absent the benchmark configs fail at dataset construction.
> The smoke test and the unit tests do not need them.

Model weights are downloaded by `timm` on first use from the Hugging Face Hub;
none are committed.

---

## 4. Run the benchmark

Configuration is YAML; every field can be overridden on the command line.

```bash
# Smallest useful run: one config, few images
python -m xai_bench.runner --config configs/smoke_5metrics.yaml --seed 42

# The published five-metric grid (seed 0)
python -m xai_bench.runner --config configs/revision_grid_500.yaml          # eight backbones, 500 images
python -m xai_bench.runner --config configs/revision_grid_500_attnlrp.yaml  # AttnLRP on ViT-B/16
```

The full grid is long; `scripts/revision_neucom/queue_500.py` splits it into
shards (backbone x method group) that several GPUs can work through, and
`scripts/revision_neucom/prefill_500.py` seeds each shard with the rows of the
earlier 100/50-image runs (`results/revision_grid_seeded*`), whose evaluations
are identical, so only the missing images are computed. See
`scripts/revision_neucom/README.md`.

Available overrides: `--seed`, `--num-images`, `--device`, `--run-name`,
`--output-dir`, `--deterministic`.

The localisation (EBPG), multi-seed robustness, Grad-CAM++ and timing analyses
are separate scripts in `scripts/revision_neucom/`; see its README.

### Runs that must be isolated

`attnlrp` applies **class-level patches** to timm modules. Once applied they
change the gradients of every other gradient-based method in the same process,
so AttnLRP has its own config and must run in its own process.

### What a run writes

```
results/<run_name>/
├── config.json       the resolved configuration
├── provenance.json   git commit + dirty flag, python/torch/CUDA versions, GPU, seeds
├── results.json      one row per (model, method, image, seed), failures included
├── results.csv       the same, flat
├── failures.json     methods that could not be set up, and every failed evaluation
└── summary.csv       per (model, method): n_evaluations, n_failed, n_constant_maps,
                      scored <metric>_mean/median/std/count, <metric>_undefined,
                      time_ms_*, peak_gpu_mb_*
```

A failed evaluation is retried when the same run is started again; every other
row is kept.

### Expected runtime

| Run | Hardware | Time |
|---|---|---|
| `smoke_5metrics.yaml` | any | minutes |
| `revision_grid_500*.yaml`, ViT-B/16, Swin-B, PVT-v2, MaxViT-S | RTX A6000 GPUs | about 140 GPU-hours |
| `revision_grid_500*.yaml`, ResNet-50, MobileViT-v2, EfficientViT-B1/B2 | one RTX 4070 Laptop GPU (8 GB) | about three days |

On an 8 GB laptop GPU, run one worker at a time: two CUDA processes on the same
laptop GPU produced driver faults. Single-image FC and FE values differ between
GPU models by about 0.01-0.02 (median), from floating-point differences, and
average out over the images of a cell.

Occlusion, LIME, RISE and Max-Sensitivity dominate, because each re-explains or
re-evaluates the image many times. Per-attribution times are in every
`summary.csv`; the per-method table is `results/raw/computational_cost.csv`.

---

## 5. Determinism

Seeding is derived per evaluation:

```
seed = crc32(seed | model | method | image_idx)
```

Python, NumPy and torch (CPU and CUDA) are reseeded from it before the
attribution, and the metrics continue the same stream. A value therefore does
not depend on which rows ran before it, and a resumed run reproduces a clean run.

`--deterministic` additionally sets `cudnn.deterministic=True` and disables
`cudnn.benchmark`. The tradeoff:

| Mode | Reproducibility | Speed |
|---|---|---|
| default | Same seed → same sampling. Small float drift possible from non-deterministic cuDNN kernel selection. | Faster |
| `--deterministic` | Bit-identical on the same hardware + library versions. | Typically 10–30% slower |

Bit-identical results across **different** devices are not achievable and are
not claimed. Run on CPU instead of GPU, the same evaluation differs by small
floating-point amounts that peaked maps amplify: a Saliency peak can move by a
pixel and flip a Pointing Game hit, and Faithfulness Correlation can change in
the first decimal. Smooth maps such as Grad-CAM agree to about two decimals.

---

## 6. From results to figures and website

```bash
python scripts/build_results.py       # raw run summaries -> canonical records
                                      #   + results/raw/computational_cost.csv
python scripts/validate_results.py    # schema, ranges, duplicates, integrity
python scripts/generate_figures.py    # -> figures/paper/*.{pdf,png}
python scripts/export_web_data.py     # -> website/public/data/*.json
python scripts/export_web_assets.py   # -> website/public/figures/*.png (downscaled)
```

Or all of it:

```bash
make reproduce
```

Serve the site (fetch does not work from `file://`):

```bash
python -m http.server 8000 --directory website
# http://localhost:8000
```

### Expected output of `build_results.py`

```
rows read              : 90
dropped (other paper)  : 0
dropped (no identity)  : 0
dropped (unknown model): {}
dropped (unknown meth.): {}
records written        : 90
  models 8  methods 13
metric coverage:
  faithfulness_correlation      90 records
  faithfulness_estimate         90 records
  apf                            0 records
  pointing_game                 90 records
  ebpg                          90 records
  max_sensitivity               90 records
  sparseness                    90 records
  attribution_time_ms           13 records
  note: 'computational_cost' is reported per method, not per cell

wrote results/processed/records.json and results/processed/manifest.json
```

### Expected output of `validate_results.py`

```
Validating results/processed/records.json
  schema version : 1.1.0 (expected 1.1.0)
  records        : 90
  models         : 8
  methods        : 13

Dimension coverage:
  faithfulness           90 records
  localization           90 records
  robustness             90 records
  complexity             90 records
  computational_cost      0 records   <- NO DATA

All records valid.
```

The `NO DATA` line is expected: cost is measured per method, not per
(model, method) cell, and lives in `results/processed/method_cost.json`.

---

## 7. What this repository cannot reproduce on its own

Stated plainly rather than papered over.

| Item | Status | Why |
|---|---|---|
| The published records | **Re-derivable** from the committed runs (`results/revision_grid_500/`, `results/revision_neucom/analysis/summary.json`). | — |
| Per-image rows | **Committed** for the seeded grid (`results.csv`) and for every revision analysis. | — |
| Re-running the benchmark end to end | **Blocked without user-supplied data.** | ImageNet-S images are not redistributable. |
| Robustness / faithfulness *curves* | **Not available.** | Only aggregated scalars are stored; curves need a per-perturbation sweep. |
| Explanation gallery images | **Available as static figures** (`figures/bench/qualitative_grid_cat.png`, `bench_qualitative.png`, `cam_upsampling_artifact.png`), shipped to the website by `scripts/export_web_assets.py`. | Regenerating them needs pretrained weights; they cannot be derived from aggregated records. |
| Provenance of the seeded grid | **Partial.** | Its runs predate the merged runner and wrote `config.json` but no `provenance.json`; the records carry run, config, seed and dataset, and leave the rest empty. New runs record everything. |
| The July 2026 result set | **Kept for the record** in `results/raw/benchmark_runs.csv`; rebuild with `python scripts/build_results.py --raw results/raw/benchmark_runs.csv --no-ebpg --out <dir>`. | Unseeded, undefined images left out, mean MS; superseded. |
