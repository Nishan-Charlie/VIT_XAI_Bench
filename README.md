<div align="center">

---

## TL;DR

Attribution-method rankings established on CNNs **do not transfer** to
transformer backbones. In the published result set, scored by bounding-box
pointing game, **10 of 11 methods that have results on both occupy a different
rank** on transformers than on CNNs.

```bash
git clone https://github.com/Nishan-Charlie/VIT_XAI_Bench && cd VIT_XAI_Bench
pip install -e ".[dev]"
python scripts/smoke_test.py     # ~30 s, CPU, no downloads
make reproduce                   # records -> validation -> figures -> website data
```

Every number in the figures and on the website is generated from
`results/processed/records.json`. Nothing is hardcoded.

---

## Contents

- [Research question](#research-question)
- [Main findings](#main-findings)
- [Benchmark scope](#benchmark-scope)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Configuration](#configuration)
- [Result format](#result-format)
- [Interactive website](#interactive-website)
- [Hardware and runtime](#hardware-and-runtime)
- [Known gaps](#known-gaps)
- [Scientific integrity](#scientific-integrity)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Research question

> **Are attribution methods that perform well on CNNs equally reliable on modern
> transformer architectures?**

Benchmarks routinely establish a ranking of attribution methods on a
convolutional backbone and then apply the winner elsewhere. That inference holds
only if the ranking is a property of the _method_ rather than of the
_architecture it was measured on_. This benchmark fixes the dataset, the metrics
and the protocol, varies the backbone across five architecture families, and
reads off whether the order survives.

---

## Main findings

Each is traceable to `results/processed/records.json` and reproducible with
`scripts/generate_figures.py`.

| Finding                                                                                                                  | Where to verify                                      |
| ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Rankings are architecture-dependent; most methods change rank between CNNs and transformers.                             | `figures/paper/cnn_to_transformer_transfer.pdf`      |
| CAM methods score highest under bounding-box localisation on CNNs and most ViTs, and drop on linear-attention backbones. | `figures/paper/heatmap_pointing_game.pdf`            |
| Faithfulness correlation barely separates methods — values cluster near zero across the whole matrix.                    | `figures/paper/heatmap_faithfulness_correlation.pdf` |
| The evaluation dimensions disagree with each other, so a single-metric ranking is not defensible.                        | `figures/paper/metric_correlation.pdf`               |

> **Two claims in the abstract are not currently backed by data in this
> repository**: the dense-mask (metric-saturation) result and everything about
> computational cost. See [Known gaps](#known-gaps).

---

## Benchmark scope

### Architectures (10 in the result set)

| Family           | Backbones                        | Attention                          |
| ---------------- | -------------------------------- | ---------------------------------- |
| CNN              | ResNet-50, ConvNeXt-B            | none (convolutional)               |
| Isotropic ViT    | ViT-B/16, DeiT-B/16              | global self-attention              |
| Hierarchical     | Swin-B, PVTv2-B2                 | shifted-window / spatial-reduction |
| Hybrid           | MaxViT-S, MobileViTv2-1.0        | multi-axis / separable             |
| Linear attention | EfficientViT-B1, EfficientViT-B2 | cascaded linear attention          |

### Attribution methods (14 in the result set)

| Family          | Methods                                                                             |
| --------------- | ----------------------------------------------------------------------------------- |
| Gradient        | Saliency, Input × Gradient, Integrated Gradients, SmoothGrad, VarGrad, GradientSHAP |
| CAM             | Grad-CAM, Grad-CAM++                                                                |
| Perturbation    | Occlusion, RISE, LIME                                                               |
| Attention / LRP | Attention Rollout, Attention × Gradient, AttnLRP                                    |

### Evaluation dimensions

| Dimension          | Metric                                               | Data present           |
| ------------------ | ---------------------------------------------------- | ---------------------- |
| Faithfulness       | Faithfulness Correlation, Faithfulness Estimate, APF | ✅ 117 cells           |
| Localisation       | Pointing Game (bbox)                                 | ✅ 117 cells           |
| Robustness         | Max-Sensitivity                                      | ✅ 117 cells           |
| Complexity         | Sparseness (Gini)                                    | ✅ 117 cells           |
| Computational cost | runtime, GPU memory                                  | ❌**none** — see below |

Canonical names live in [`xai_bench/taxonomy.py`](xai_bench/taxonomy.py); adding
a backbone or method means editing that one file.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"       # core + test/lint
pip install -e ".[lrp]"       # optional: AttnLRP / CP-LRP (needs lxt, zennit)
```

Python 3.10–3.12. `quantus` is a core dependency, not an extra.

---

## Quick start

### Verify the install (no data, no GPU, no downloads)

```bash
python scripts/smoke_test.py
```

Exercises dataset → model → attribution → metrics → records → validation → JSON
on synthetic data with a randomly initialised backbone. Ends with
`25/25 checks passed`.

### Minimal example

```python
import torch
from xai_bench import MODELS, METHODS, METRICS
from xai_bench.utils import set_seed

set_seed(42)

wrapper = MODELS.get("resnet50")()          # timm backbone + benchmark metadata
wrapper.model.eval()

explain = METHODS.get("grad_cam")(model_wrapper=wrapper, model=wrapper.model)
image = torch.randn(1, 3, 224, 224)
attribution = explain(image, target=207)     # [1, 224, 224]

score = METRICS.get("sparseness")(
    attr=attribution[0], metadata={}, model=wrapper.model,
    image=image[0], target=207, metric_kwargs={},
)
print(f"sparseness: {float(score):.4f}")
```

### Run a benchmark sweep

```bash
python -m xai_bench.runner --config configs/full_benchmark_100.yaml --seed 42
```

---

## Reproducing the benchmark

Full detail — including data acquisition and expected runtimes — is in
**[docs/reproducibility.md](docs/reproducibility.md)**.

```bash
make reproduce      # build records -> validate -> figures -> website data
```

Stage by stage:

| Stage                        | Command                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| 1. Verify install            | `python scripts/smoke_test.py`                              |
| 2. Prepare data              | `python scripts/cache_imagenets.py` (you supply the images) |
| 3. Run attribution + metrics | `python -m xai_bench.runner --config <cfg>`                 |
| 4. Consolidate               | `python scripts/build_results.py`                           |
| 5. Validate                  | `python scripts/validate_results.py`                        |
| 6. Figures                   | `python scripts/generate_figures.py`                        |
| 7. Website data              | `python scripts/export_web_data.py`                         |

`make help` lists every target.

> **Methods that must run in isolation.** `attnlrp` monkey-patches timm at class
> level; once patched it changes every other gradient-based method in the same
> process. Use `configs/attnlrp_grid_vit.yaml` / `configs/attnlrp_grid_hier.yaml`
> in separate processes.

---

## Configuration

Experiments are YAML; the CLI overrides any field.

```yaml
models: ["vit_base_patch16_224", "resnet50"]
methods: ["grad_cam", "integrated_gradients", "attention_rollout"]
metrics:
  ["faithfulness_correlation", "pointing_game", "max_sensitivity", "sparseness"]

metric_kwargs:
  faithfulness_correlation:
    { nr_runs: 20, subset_size: 224, return_aggregate: false }
  max_sensitivity: { nr_samples: 3, lower_bound: 0.2 }

dataset: "imagenets_cached"
dataset_kwargs:
  { cache_path: "data/ImageNetS/cache_validation_100.pt", num_samples: 100 }
num_images: 100
seeds: [42]
device: "auto"
output_dir: "results"
limit_methods_to_supported: true # skip attention methods on backbones with no CLS attention
```

```bash
python -m xai_bench.runner --config configs/full_benchmark_100.yaml \
    --seed 7 --num-images 25 --device cpu --deterministic
```

---

## Result format

Every run writes provenance next to its results:

```
results/<run_name>/
├── config.json      resolved configuration
├── provenance.json  git commit + dirty flag, python/torch/CUDA, GPU, seed
├── results.json     one row per (model, method, image) — including failures
├── results.csv
└── summary.csv      per (model, method): mean/std/count + n_errors
```

`scripts/build_results.py` consolidates run summaries into the canonical record
set. One record is one `(model, method)` cell:

```json
{
  "model": "vit_base_patch16_224",
  "model_family": "isotropic_vit",
  "method": "grad_cam",
  "method_family": "cam",
  "metrics": {
    "pointing_game": { "mean": 0.927, "std": 0.261, "count": 96 }
  },
  "provenance": {
    "source_run": "full_benchmark_100/summary.csv",
    "num_images": 96
  }
}
```

`scripts/validate_results.py` fails loudly on NaN, infinity, out-of-range values,
non-positive counts, unknown model/method/metric names, and duplicate cells.
It runs in CI and needs no torch.

---

## Interactive website

An academic project page rendered entirely from the exported JSON —
architecture explorer, filterable result matrix, ranking explorer,
transferability view, and an explicit panel stating which dimensions have no
data.

```bash
python scripts/export_web_data.py
python -m http.server 8000 --directory website   # http://localhost:8000
```

The page fetches `website/public/data/benchmark.json` at load time, so
re-exporting refreshes everything without touching frontend code. It must be
served over HTTP — `fetch` is blocked on `file://`.

---

## Hardware and runtime

| Requirement | Value                                                         |
| ----------- | ------------------------------------------------------------- |
| Python      | 3.10–3.12                                                     |
| PyTorch     | ≥ 2.1 (CUDA 12.1/12.4 for GPU)                                |
| GPU VRAM    | ≥ 8 GB for`*_base` backbones                                  |
| RAM         | ≥ 16 GB                                                       |
| Disk        | ~5 GB weights + ~2 GB cached data                             |
| CPU-only    | fine for smoke test and tests; impractical for the full sweep |

RISE, LIME and Max-Sensitivity dominate cost — each re-explains an image many
times. Precise per-method timings are **not stated** because the archived runs
never recorded them; the current runner does.

---

## Known gaps

Stated rather than hidden. Full accounting in
[docs/reproducibility.md §7](docs/reproducibility.md).

| Gap                              | Status                                                                                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Computational cost               | No data. Old runner discarded`time_ms` before aggregation; fixed, needs a re-run.                                                                                  |
| Dense-mask (EBPG) localisation   | No data in`results/`. `scripts/bench/ebpg_eval.py` needs a GPU re-run.                                                                                             |
| Robustness / faithfulness curves | Only aggregated scalars were stored.                                                                                                                               |
| Explanation gallery              | Needs re-running attribution with pretrained weights.                                                                                                              |
| Grad-CAM layer inconsistency     | The bbox and dense-mask paths use**different ViT layers**. Documented, not silently changed — see [docs/scientific_integrity.md §3](docs/scientific_integrity.md). |

---

## Scientific integrity

**HiLRP is the novel method of a separate paper and is not part of this
benchmark.** Its implementation, configs, figures, and 400 result rows have been
removed; CI fails if any reference returns. The published AttnLRP and CP-LRP
baselines were extracted and kept, because they are legitimately evaluated here.

A labelling defect that rendered `hilrp` rows under the display name `AttnLRP`
was found and fixed; three figures whose provenance could not be verified are
quarantined in [`figures/_unverified/`](figures/_unverified/).

Read **[docs/scientific_integrity.md](docs/scientific_integrity.md)** before
citing any number.

---

## Troubleshooting

| Symptom                                           | Cause and fix                                                                                   |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: xai_bench`                  | Install the package:`pip install -e .` (or set `PYTHONPATH=.`).                                 |
| `ModuleNotFoundError: quantus`                    | `pip install -e .` — quantus is a core dependency.                                              |
| `ModuleNotFoundError: lxt` / `zennit`             | Only needed for`attnlrp`: `pip install -e ".[lrp]"`.                                            |
| Website shows "Could not load the benchmark data" | Serve over HTTP;`fetch` is blocked on `file://`.                                                |
| Dataset cache not found                           | `data/` is gitignored — supply the images and run `scripts/cache_imagenets.py`.                 |
| Gradient methods give odd values after an LRP run | `attnlrp` patches timm at class level. Run it in a separate process.                            |
| Attention methods missing for a backbone          | Expected: only isotropic ViTs have CLS-token attention.`limit_methods_to_supported` skips them. |
| CUDA out of memory                                | Lower`num_images`, use `--device cpu`, or drop RISE/LIME from `methods`.                        |

---

## Contributing

Adding a method or backbone:

1. Register it (`@METHODS.register("name")` / `@MODELS.register("name")`).
2. Add its `MethodSpec` / `ModelSpec` to `xai_bench/taxonomy.py`.
3. `pytest tests` — the taxonomy tests assert registry and taxonomy agree.
4. Add a config under `configs/`, run it, then `make reproduce`.

CI runs ruff, the test suite on three Python versions, the smoke test, and the
results-integrity check. None of it requires a GPU.

---

## Citation

```bibtex
@article{nishankar_does_explainability_transfer,
  title   = {Does Explainability Transfer? A Controlled Benchmark of Attribution
             Methods on Vision Transformers and CNNs},
  author  = {Nishankar, Sathiyamohan and Pathirana, Nethmi and Sanjeewani, Pubudu
             and Perera, Asanka and Thuseethan, Selvarajah},
  year    = {2026}
}
```

Built on [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus),
[timm](https://github.com/huggingface/pytorch-image-models),
[Captum](https://captum.ai/), and [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers).

## License

MIT — see [LICENSE](LICENSE).
