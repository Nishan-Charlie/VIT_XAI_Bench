"""Canonical names for architectures, attribution methods and metrics.

Every result record, figure and website payload resolves its labels through this
module, so a model or method is named and grouped identically everywhere. Adding
a backbone or method to the benchmark means adding it here once.
"""

from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────── architectures ──────────────────────────────

#: Architecture families spanned by the benchmark, in presentation order.
ARCH_FAMILIES: list[str] = [
    "cnn",
    "isotropic_vit",
    "hierarchical",
    "hybrid",
    "linear_attention",
]

ARCH_FAMILY_LABELS: dict[str, str] = {
    "cnn": "CNN",
    "isotropic_vit": "Isotropic ViT",
    "hierarchical": "Hierarchical Transformer",
    "hybrid": "Hybrid (Conv-Transformer)",
    "linear_attention": "Linear-Attention Transformer",
}


@dataclass(frozen=True)
class ModelSpec:
    """A backbone in the benchmark suite."""

    key: str            # registry key, also the value stored in result records
    label: str          # human-readable name used in figures and the website
    family: str         # one of ARCH_FAMILIES
    attention: str      # the attention mechanism, for the architecture explorer


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec("resnet50", "ResNet-50", "cnn", "none (convolutional)"),
        ModelSpec("convnext_base", "ConvNeXt-B", "cnn", "none (convolutional)"),
        ModelSpec("vit_base_patch16_224", "ViT-B/16", "isotropic_vit", "global self-attention"),
        ModelSpec("deit_base_patch16_224", "DeiT-B/16", "isotropic_vit", "global self-attention"),
        ModelSpec("swin_base_patch4_window7_224", "Swin-B", "hierarchical", "shifted-window attention"),
        ModelSpec("pvt_v2_b2", "PVTv2-B2", "hierarchical", "spatial-reduction attention"),
        ModelSpec("maxvit_small_tf_224", "MaxViT-S", "hybrid", "multi-axis (block + grid)"),
        ModelSpec("mobilevitv2_100", "MobileViTv2-1.0", "hybrid", "separable self-attention"),
        ModelSpec("efficientvit_b1", "EfficientViT-B1", "linear_attention", "cascaded linear attention"),
        ModelSpec("efficientvit_b2", "EfficientViT-B2", "linear_attention", "cascaded linear attention"),
    ]
}

# ──────────────────────────── attribution methods ───────────────────────────

#: Algorithmic families of the evaluated attribution methods.
METHOD_FAMILIES: list[str] = ["gradient", "cam", "perturbation", "attention_lrp"]

METHOD_FAMILY_LABELS: dict[str, str] = {
    "gradient": "Gradient-based",
    "cam": "CAM-based",
    "perturbation": "Perturbation-based",
    "attention_lrp": "Attention / LRP-based",
}


@dataclass(frozen=True)
class MethodSpec:
    """An attribution method in the benchmark."""

    key: str
    label: str
    family: str
    stochastic: bool    # whether the method samples, i.e. is seed-sensitive
    vit_only: bool      # requires a global CLS-token attention map


METHODS: dict[str, MethodSpec] = {
    m.key: m
    for m in [
        MethodSpec("saliency", "Saliency", "gradient", False, False),
        MethodSpec("input_x_gradient", "Input × Gradient", "gradient", False, False),
        MethodSpec("integrated_gradients", "Integrated Gradients", "gradient", False, False),
        MethodSpec("smoothgrad", "SmoothGrad", "gradient", True, False),
        MethodSpec("vargrad", "VarGrad", "gradient", True, False),
        MethodSpec("gradient_shap", "GradientSHAP", "gradient", True, False),
        MethodSpec("grad_cam", "Grad-CAM", "cam", False, False),
        MethodSpec("grad_cam_plus_plus", "Grad-CAM++", "cam", False, False),
        MethodSpec("occlusion", "Occlusion", "perturbation", False, False),
        MethodSpec("rise", "RISE", "perturbation", True, False),
        MethodSpec("lime", "LIME", "perturbation", True, False),
        MethodSpec("attention_rollout", "Attention Rollout", "attention_lrp", False, True),
        MethodSpec("attention_gradient", "Attention × Gradient", "attention_lrp", False, True),
        MethodSpec("attnlrp", "AttnLRP", "attention_lrp", False, False),
        MethodSpec("lrp", "LRP (epsilon)", "attention_lrp", False, False),
    ]
}

# ─────────────────────────────────── metrics ────────────────────────────────

#: The five evaluation dimensions of the benchmark.
DIMENSIONS: list[str] = [
    "faithfulness",
    "localization",
    "robustness",
    "complexity",
    "computational_cost",
]


@dataclass(frozen=True)
class MetricSpec:
    """A scalar metric, its dimension, and how to read its scale."""

    key: str
    label: str
    dimension: str
    higher_is_better: bool
    #: Theoretical range, or None where the metric is unbounded.
    value_range: tuple | None
    description: str


METRICS: dict[str, MetricSpec] = {
    m.key: m
    for m in [
        MetricSpec(
            "faithfulness_correlation", "Faithfulness Correlation", "faithfulness",
            True, (-1.0, 1.0),
            "Correlation between attribution mass in a random subset and the drop in "
            "target logit when that subset is perturbed (Bhatt et al., 2020).",
        ),
        MetricSpec(
            "faithfulness_estimate", "Faithfulness Estimate", "faithfulness",
            True, (-1.0, 1.0),
            "Correlation between attribution and prediction change under incremental "
            "feature removal (Alvarez-Melis & Jaakkola, 2018).",
        ),
        MetricSpec(
            "apf", "Attention-Preserving Faithfulness", "faithfulness",
            True, (-1.0, 1.0),
            "Faithfulness correlation with mean-imputation instead of a constant "
            "baseline, so perturbed inputs keep sequence statistics.",
        ),
        MetricSpec(
            "pointing_game", "Pointing Game (bbox)", "localization",
            True, (0.0, 1.0),
            "Fraction of images whose attribution peak falls inside the object "
            "bounding box (Zhang et al., 2018). Coarse: a bounding box includes "
            "background.",
        ),
        MetricSpec(
            "max_sensitivity", "Max-Sensitivity", "robustness",
            False, (0.0, None),
            "Largest change in the explanation over small input perturbations "
            "(Yeh et al., 2019). Lower is more stable.",
        ),
        MetricSpec(
            "sparseness", "Sparseness (Gini)", "complexity",
            True, (0.0, 1.0),
            "Gini coefficient of the attribution map (Chalasani et al., 2020). "
            "Higher means fewer features carry the explanation.",
        ),
    ]
}

#: Metrics grouped by evaluation dimension.
METRICS_BY_DIMENSION: dict[str, list[str]] = {
    dim: [k for k, m in METRICS.items() if m.dimension == dim] for dim in DIMENSIONS
}


def model_family(model_key: str) -> str:
    """Architecture family for a model key, or ``"unknown"`` if unregistered."""
    spec = MODELS.get(model_key)
    return spec.family if spec else "unknown"


def method_family(method_key: str) -> str:
    """Algorithmic family for a method key, or ``"unknown"`` if unregistered."""
    spec = METHODS.get(method_key)
    return spec.family if spec else "unknown"
