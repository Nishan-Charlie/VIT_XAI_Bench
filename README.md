# Does Explainability Transfer?

### A Controlled Benchmark of Attribution Methods on Vision Transformers and CNNs

Sathiyamohan Nishankar · Nethmi Pathirana · Pubudu Sanjeewani · Asanka Perera · Selvarajah Thuseethan

[![CI](https://github.com/Nishan-Charlie/VIT_XAI_Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Nishan-Charlie/VIT_XAI_Bench/actions/workflows/ci.yml)
[![Pages](https://github.com/Nishan-Charlie/VIT_XAI_Bench/actions/workflows/pages.yml/badge.svg)](https://github.com/Nishan-Charlie/VIT_XAI_Bench/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Project page](https://nishan-charlie.github.io/VIT_XAI_Bench/)** ·
[Reproducibility guide](docs/reproducibility.md)

Attribution-method rankings measured on CNNs **do not transfer** to transformer
backbones. Of the 11 methods that run on every backbone, **all 11 change rank**
between ResNet-50 and the transformers under the Pointing Game, and **8 of 11**
under the dense-mask Energy-Based Pointing Game (EBPG).

![Grad-CAM across six benchmark backbones on three images](figures/bench/bench_qualitative.png)

_Grad-CAM on three images across six backbones. It localises the object on the
CNN, hierarchical and hybrid models, and fails on the linear-attention
EfficientViT variants._

## Main findings

- **The Pointing Game saturates.** A random peak already scores about 0.6. Under
  EBPG no method exceeds 0.60.
- **Grad-CAM fails on linear attention.** Its EBPG on EfficientViT-B2 is 0.25,
  below the 0.31 of a random map.
- **Grad-CAM++ can be unstable.** Its closed-form weights have a pole, and near
  it the map collapses or blows up. On MaxViT-S the map is zero on 71% of the
  1,000 images.
- **Architecture family matters more than size.** Over 21 backbones, the family
  explains 85% of the variance of Grad-CAM EBPG and 87% of its ranked median
  Max-Sensitivity. Parameter count alone explains less than 1% of either.
- **Faithfulness Correlation barely separates methods.** Its values cluster near
  zero across the whole grid.
- **Cost varies widely.** Occlusion takes about 365× as long as Saliency per
  explanation.

## What is evaluated

| | |
| --- | --- |
| **CNN** | ResNet-50 |
| **Isotropic ViT** | ViT-B/16 |
| **Hierarchical** | Swin-B, PVTv2-B2 |
| **Hybrid / multi-axis** | MaxViT-S, MobileViTv2-1.0 |
| **Linear attention** | EfficientViT-B1, EfficientViT-B2 |
| **Gradient methods** | Saliency, Input × Gradient, Integrated Gradients, SmoothGrad, VarGrad, GradientSHAP |
| **CAM methods** | Grad-CAM, Grad-CAM++ |
| **Perturbation methods** | Occlusion, RISE, LIME |
| **Attention / LRP methods** | Attention Rollout, AttnLRP (ViT-B/16 only) |
| **Metrics** | Faithfulness Correlation, Faithfulness Estimate, Pointing Game, EBPG, Max-Sensitivity, Sparseness, attribution time |
| **Dataset** | 1,000 validation images of [ImageNet-S](https://github.com/LUSSeg/ImageNet-S) |

The five-metric grid uses the first 100 images for ResNet-50 and ViT-B/16 and
the first 50 for the other backbones. EBPG uses all 1,000 images, except for
Integrated Gradients and the perturbation methods (first 100).

## Quick start

Requires Python 3.10–3.12.

```bash
git clone https://github.com/Nishan-Charlie/VIT_XAI_Bench && cd VIT_XAI_Bench
pip install -e ".[dev]"
python scripts/smoke_test.py   # about 30 s, CPU, no downloads
make reproduce                 # rebuilds the results, figures and website data
```

To re-run the grid, prepare the data and start a run:

```bash
pip install -e ".[data]"
python scripts/cache_imagenets.py
python -m xai_bench.runner --config configs/revision_grid_seeded.yaml
```

The grid took about 16 hours on one RTX 4070 Laptop GPU. The
[reproducibility guide](docs/reproducibility.md) covers the remaining configs,
the output format, expected runtimes and known limits.

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

Please also cite ImageNet-S:

```bibtex
@article{gao2022luss,
  title={Large-scale Unsupervised Semantic Segmentation},
  author={Gao, Shanghua and Li, Zhong-Yu and Yang, Ming-Hsuan and Cheng, Ming-Ming
          and Han, Junwei and Torr, Philip},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2022}
}
```

Built on [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus),
[timm](https://github.com/huggingface/pytorch-image-models),
[Captum](https://captum.ai/) and [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers).

## License

MIT; see [LICENSE](LICENSE).
