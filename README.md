<div align="center">
  <h1>Does Explainability Transfer? A Controlled Benchmark of Attribution Methods on Vision
Transformers and CNNs</h1>
  <p><strong>Most evidence on the effectiveness of explainable artificial intelligence (XAI) attribution methods has been established on con-
volutional neural networks (CNNs), with limited investigation into whether these conclusions generalise to the diverse Vision
Transformer (ViT) architectures that now dominate computer vision. This paper presents a controlled benchmark that evaluates
attribution quality across five dimensions: faithfulness, localisation, robustness, complexity, and computational cost. A standard-
ised framework assesses 13 attribution methods from four algorithmic families on eight representative backbones spanning CNNs,
isotropic ViTs, hierarchical transformers, hybrid architectures, and linear-attention transformers. The results show that attribution
performance is strongly architecture-dependent and that rankings established on CNNs do not reliably transfer to transformer-based
models. CAM-based methods achieve the highest scores under the conventional bounding-box localisation metric on CNNs and
most ViTs but perform poorly on linear-attention architectures. Pixel-level dense-mask evaluation further reveals that these gains
largely reflect metric saturation rather than accurate localisation. CAM-based methods also exhibit limited robustness on global-
attention transformers, whereas attention rollout provides consistently stable explanations with poor localisation. Furthermore,
faithfulness correlation offers limited discrimination between attribution methods, highlighting the limitations of single-metric
evaluation. These findings challenge prevailing conclusions on attribution performance and demonstrate the need for architecture-
aware, multi-dimensional evaluation. The open-source code for the evaluation framework and benchmark results is available at</strong></p>

<br />

## 📖 Overview

While Explainable AI (XAI) has been heavily benchmarked on small Convolutional Neural Networks (CNNs), the same rigor is often missing for Vision Transformer (ViT) foundation models.

**XAI-Bench** provides a controlled, statistically rigorous framework that measures attribution **fidelity** across three orthogonal axes:

1. 🏗️ **Architecture:** CNN vs. ViT
2. 🎓 **Pretraining Objective:** Supervised vs. Self-supervised (DINOv2) vs. Masked (MAE/BEiT) vs. Contrastive (CLIP/EVA)
3. 📈 **Model Scale:** Small to Giant models

We leverage both **exact/causal ground truth** (synthetic & intervention-based, e.g., FunnyBirds) and **proxy faithfulness metrics** (e.g., Quantus) to evaluate existing explanation methods fairly and transparently.

---

## 📊 Results

Here is a snapshot of the results from our benchmark.

### Faithfulness Correlation Mean

| model                | attention_rollout |     grad_cam | integrated_gradients |
| :------------------- | ----------------: | -----------: | -------------------: |
| convnext_base        |               nan |   0.00502714 |          -0.00218806 |
| resnet50             |               nan |    0.0158362 |           0.00510865 |
| vit_base_patch16_224 |        -0.0232956 | -0.000751883 |            0.0119544 |

### Pointing Game Mean

| model                | attention_rollout | grad_cam | integrated_gradients |
| :------------------- | ----------------: | -------: | -------------------: |
| convnext_base        |               nan |        1 |                 0.76 |
| resnet50             |               nan |     0.98 |                 0.92 |
| vit_base_patch16_224 |              0.64 | 0.927083 |                 0.64 |

---

## ✨ Visuals

Here are some sample results from the benchmark:

<p align="center">
  <img src="figures/bench/arch_method_comparison.png" width="800">
  <em>Figure 1: Comparison of attribution methods across different architectures.</em>
</p>

<p align="center">
  <img src="figures/bench/bench_pareto.png" width="800">
  <em>Figure 2: Pareto frontier of attribution methods, trading off fidelity and complexity.</em>
</p>

<p align="center">
  <img src="figures/bench/qualitative_gallery.png" width="800">
  <em>Figure 3: Qualitative comparison of different XAI methods.</em>
</p>

---

## 🎯 Key Research Questions Evaluated

- **RQ1 — Transfer:** Do attribution-method fidelity rankings established on CNNs hold on ViT foundation models?
- **RQ2 — Pretraining:** Holding architecture and scale fixed, does the pretraining objective change attribution faithfulness?
- **RQ3 — Scaling:** How does fidelity scale with model size? Is there an "explainability scaling law"?
- **RQ4 — Attention vs. Gradients:** Do attention-native methods beat gradient/perturbation methods on ViTs, and do _register tokens_ fix attention-based attribution?
- **RQ5 — Metric Agreement:** How much do faithfulness metrics agree with each other and with controllable ground truth?

---

## 🔬 Benchmark Scope

### Supported Models

- **Self-Supervised & Contrastive:** DINOv2 (with/without register tokens), CLIP-ViT, EVA02-CLIP
- **Masked Autoencoders:** MAE, BEiT3
- **Supervised Baselines:** Supervised ViTs, ResNet, ConvNeXt

### Evaluated XAI Methods

- **Gradient-based:** Saliency, Integrated Gradients, SmoothGrad
- **CAM-based:** Grad-CAM, Grad-CAM++
- **Attention-native:** Rollout, Attention Flow, Chefer/LRP, Hi-LRP
- **Perturbation-based:** Occlusion, RISE, LIME, KernelSHAP

### Datasets & Metrics

- **Datasets:** ImageNet-S, ImageNet-1k, FunnyBirds, MS COCO
- **Metrics:** Faithfulness, Localization, Robustness, and Complexity (powered by [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)).

---

## 🚀 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/XAI-Bench.git
   cd XAI-Bench
   ```

2. **Install the base package:**

   ```bash
   pip install -e .
   ```

3. _(Optional)_ **Install extra dependencies** (for clustering, advanced CAMs, and Quantus metrics):

   ```bash
   pip install -e ".[extra]"
   ```

---

## 💻 Quickstart

### Running a Benchmark

The easiest way to run a benchmark is to use the `xai-bench` command-line tool with a configuration file.

1. **Choose a configuration file** from the `configs/` directory. For example, `configs/smoke_test_quantus.yaml`.
2. **Run the benchmark:**

   ```bash
   xai-bench --config configs/smoke_test_quantus.yaml
   ```

This will run the benchmark with the specified models, methods, and metrics, and save the results in the `results/` directory.

### Running a single explanation

To run a single explanation on an image, you can use the `run_imagenets_xai.py` script, which has been moved to the `scripts/` directory.

```bash
python scripts/run_imagenets_xai.py
```

---

## 📂 Repository Structure

The repository is structured as follows:

```
├── configs/            # Configs for benchmarking sweeps
├── data/               # Data loading scripts and dataset structures
├── figures/            # Figures for the paper and README
├── images/             # Sample images for testing
├── scripts/            # Helper scripts for running experiments, plotting, etc.
├── tests/              # Unit tests
└── xai_bench/          # Core Python framework for XAI methods & models
```

---

## 📝 License & Citation

This project is open-source under the MIT License. If you use XAI-Bench in your research, please cite our paper:
_(Citation details coming soon)_
