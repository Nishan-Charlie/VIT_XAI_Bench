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

Expected: **80 passed**.

---

## 3. Data

The benchmark evaluates on **ImageNet-S** (ImageNet validation images with
pixel-level segmentation masks) and uses VOC-2007 boxes for some localisation
runs. Neither is redistributed here.

```bash
# ImageNet-S masks + the ImageNet validation images they index
#   https://github.com/LUSSeg/ImageNet-S
# Then build the tensor caches the configs expect:
python scripts/cache_imagenets.py      # -> data/ImageNetS/cache_validation_*.pt
python scripts/cache_voc.py            # -> data/VOC/cache_voc_1000.pt
```

> **You must supply the source images yourself.** `data/` is gitignored. If the
> caches are absent, the benchmark configs will fail at dataset construction —
> the smoke test does not need them.

Model weights are downloaded by `timm` on first use from the HuggingFace hub;
none are committed.

---

## 4. Run the benchmark

Configuration is YAML; every field can be overridden on the command line.

```bash
# Smallest useful run — one config, few images
python -m xai_bench.runner --config configs/smoke_5metrics.yaml --seed 42

# Medium: all methods x 4 backbones x 100 images x 5 metrics
python -m xai_bench.runner --config configs/full_benchmark_100.yaml --seed 42

# The efficient/hierarchical/hybrid suite
python -m xai_bench.runner --config configs/vit_suite_10.yaml --seed 42
```

Available overrides: `--seed`, `--num-images`, `--device`, `--run-name`,
`--output-dir`, `--deterministic`.

### Runs that must be isolated

`attnlrp` applies **class-level monkey patches** to timm's
`vision_transformer` module. Once applied they change the gradients of every
other gradient-based method in the same process. Run them from their own
configs, in their own processes:

```bash
python -m xai_bench.runner --config configs/attnlrp_grid_vit.yaml    # flat ViTs
python -m xai_bench.runner --config configs/attnlrp_grid_hier.yaml   # hierarchical
```

Do not combine these two in one process either: the flat-ViT path patches the
timm norm subclasses, which would silently change the hierarchical baseline.

### What a run writes

```
results/<run_name>/
├── config.json       the resolved configuration
├── provenance.json   git commit + dirty flag, python/torch/CUDA versions, GPU, seed
├── results.json      one row per (model, method, image), including failures
├── results.csv       the same, flat
└── summary.csv       per (model, method): mean/std/count per metric + n_errors
```

### Expected runtime

Measured shapes, not promises — these depend heavily on GPU and on which
methods are enabled. RISE, LIME and Max-Sensitivity dominate the cost because
they re-explain the image many times.

| Run | Scale | Hardware |
|---|---|---|
| `smoke_5metrics.yaml` | minutes | CPU or GPU |
| `full_benchmark_100.yaml` | hours | single modern GPU |
| Full suite, n=1000 | ~a day per backbone group | single modern GPU |

> The archived runs did not record wall-clock time (see
> `docs/scientific_integrity.md` §2.3), so precise figures are **not available**
> and are not invented here. Runs made with the current code record `time_ms`
> and `peak_gpu_mb` per attribution.

---

## 5. Determinism

Seeding is derived per item:

```
item_seed = blake2b(master_seed | model | method | image_idx)
```

so a value does not depend on how many items ran before it, and a resumed run
reproduces a clean run. Python, NumPy, torch and CUDA RNGs are all seeded.

`--deterministic` additionally sets `cudnn.deterministic=True` and disables
`cudnn.benchmark`. The tradeoff:

| Mode | Reproducibility | Speed |
|---|---|---|
| default | Same seed → same sampling. Small float drift possible from non-deterministic cuDNN kernel selection. | Faster |
| `--deterministic` | Bit-identical on the same hardware + library versions. | Typically 10–30% slower |

Bit-identical results across **different** GPUs or CUDA versions are not
achievable and are not claimed.

---

## 6. From results to figures and website

```bash
python scripts/build_results.py       # raw run summaries -> canonical records
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
rows read              : 127
dropped (other paper)  : 0
records written        : 117
  models 10  methods 14
  ** dimension 'computational_cost' has NO data **
```

### Expected output of `validate_results.py`

```
  records        : 117
Dimension coverage:
  faithfulness          117 records
  localization          117 records
  robustness            117 records
  complexity            117 records
  computational_cost      0 records   <- NO DATA
All records valid.
```

---

## 7. What this repository cannot reproduce on its own

Stated plainly rather than papered over.

| Item | Status | Why |
|---|---|---|
| The 117 published records | **Re-derivable from `results/raw/benchmark_runs.csv`**, which is committed. | — |
| Re-running the benchmark end to end | **Blocked without user-supplied data.** | ImageNet-S / VOC images are not redistributable. |
| Per-image rows behind the archived summaries | **Not available.** | Only aggregated summaries survived; `results/<run>/results.json` files were never committed. |
| Computational-cost dimension | **No data.** | The old runner discarded `time_ms` before aggregation. Fixed going forward. |
| Dense-mask (EBPG) localisation | **No data in `results/`.** | `scripts/bench/ebpg_eval.py` writes `results/ebpg_results.json`, which was never committed. Needs a GPU re-run. |
| Robustness / faithfulness *curves* | **Not available.** | Only aggregated scalars were stored; curves need a per-perturbation sweep. |
| Explanation gallery images | **Available as static figures** (`figures/bench/qualitative_grid_cat.png`, `bench_qualitative.png`, `cam_upsampling_artifact.png`), shipped to the website by `scripts/export_web_assets.py`. | Regenerating them requires a GPU and pretrained weights; they are not derivable from the aggregated records. |
| Provenance for archived runs | **Partial.** | Git commit, seed, dataset and timestamps were not recorded before this audit. `manifest.json` lists the gaps; new runs record everything. |
