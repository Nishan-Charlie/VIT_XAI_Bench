"""LRP backend for flat (isotropic) timm Vision Transformers.

This module implements two *published* relevance-propagation recipes for plain
ViT/DeiT/BEiT backbones, both used as baselines in the benchmark:

``attn_mode="attnlrp"``
    AttnLRP (Achtibat et al., 2024). In a Gradient x Input formulation the
    plain softmax vector-Jacobian product *is* the softmax Taylor rule
    (Proposition 3.1), and ``divide_gradient`` implements the uniform bilinear
    rule (Eq. 7) on both ``q @ k^T`` and ``attn @ v``.

``attn_mode="cp"``
    CP-LRP: the softmax attention matrix is detached and treated as constant
    weights, so relevance flows through the value path only.

Both run on the LXT efficient backend with zennit Gamma composites on Conv2d /
Linear layers. Forward numerics are identical to stock timm; only the backward
(relevance) semantics change.

.. warning::
   The patches applied here are **class-level** monkey patches on timm's
   ``vision_transformer`` module. Once applied they alter the gradients of every
   other gradient-based method in the same process. Run LRP methods from a
   dedicated config/process; never mix them with gradient baselines.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from lxt.efficient.core import monkey_patch
from lxt.efficient.patches import layer_norm_forward, non_linear_forward, patch_method
from lxt.efficient.rules import divide_gradient
from lxt.efficient.zennit_patches import monkey_patch_zennit
from timm.layers.norm import LayerNorm as TimmLayerNorm
from timm.models import vision_transformer as _vit_mod
from timm.models.vision_transformer import Attention
from zennit.composites import LayerMapComposite
from zennit.rules import Gamma

__all__ = ["attribute_vit", "ensure_patched"]

# Selects the attention rule used by the patched forward. Module-level because
# the patch is installed on the timm class itself, so the forward has no other
# channel to receive per-call options.
ATTN_MODE = "cp"

_DEFAULT_GAMMA = 0.25
_ZERO_DIVISION_EPS = 1e-12


def cp_vit_attention_forward(self, x, attn_mask=None, is_causal=False):
    """timm ``Attention.forward`` under LRP rules; see :data:`ATTN_MODE`.

    Forward numerics are identical to timm — the rules only affect the backward
    pass, via ``detach`` (CP-LRP) or ``divide_gradient`` (AttnLRP).
    """
    assert attn_mask is None and not is_causal, "classification path only"
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if ATTN_MODE == "attnlrp":
        attn = divide_gradient(attn, 2)       # uniform bilinear rule on q@k^T
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    if ATTN_MODE == "cp":
        x = attn.detach() @ v                 # CP-LRP: attention as constant weights
    else:
        x = divide_gradient(attn @ v, 2)      # uniform rule on attn@v; softmax vjp = Taylor

    x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ``TimmLayerNorm`` is a *subclass* of nn.LayerNorm; patching only nn.LayerNorm
# silently leaves timm's subclass on the standard (non-identity) rule.
VIT_PATCH_MAP = {
    nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    nn.LayerNorm: partial(patch_method, layer_norm_forward),
    TimmLayerNorm: partial(patch_method, layer_norm_forward),
    Attention: partial(patch_method, cp_vit_attention_forward),
}

_PATCHED = False


def ensure_patched(verbose: bool = False) -> None:
    """Install the class-level LRP patches exactly once per process."""
    global _PATCHED
    if not _PATCHED:
        monkey_patch(_vit_mod, patch_map=VIT_PATCH_MAP, verbose=verbose)
        monkey_patch_zennit(verbose=verbose)
        _PATCHED = True


def make_composite(gamma: float = _DEFAULT_GAMMA, conv_gamma: float | None = None):
    """Gamma composite over the linear layers (Conv2d for the patch embed)."""
    conv_gamma = gamma if conv_gamma is None else conv_gamma
    return LayerMapComposite([
        (nn.Conv2d, Gamma(conv_gamma)),
        (nn.Linear, Gamma(gamma)),
    ])


def _run_backward(
    model: nn.Module,
    x: torch.Tensor,
    scalar_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    gamma: float,
    conv_gamma: float | None,
):
    """Forward under the composite, backward from ``scalar_fn``'s scalar output.

    Returns ``(scalar_value, pixel_map)`` where ``pixel_map`` is the Gradient x
    Input relevance summed over channels, shape ``[H, W]``.
    """
    ensure_patched()
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    composite = make_composite(gamma, conv_gamma)
    with composite.context(model) as mod:
        scalar = scalar_fn(mod, x)
        mod.zero_grad(set_to_none=True)
        scalar.backward()
    pixel_map = (x * x.grad)[0].sum(0).detach().cpu()
    return scalar.item(), pixel_map


def attribute_vit(
    model: nn.Module,
    x: torch.Tensor,
    target: int | None = None,
    gamma: float = _DEFAULT_GAMMA,
    conv_gamma: float | None = None,
    attn_mode: str = "cp",
) -> dict[str, Any]:
    """Class-logit LRP attribution for a plain timm ViT.

    Args:
        model: an unwrapped timm ViT (no :class:`InputStatsAdapter`).
        x: input batch of shape ``[1, 3, H, W]``, already in the model's input space.
        target: class index to explain; ``None`` uses the argmax prediction.
        gamma: zennit Gamma parameter for Linear layers.
        conv_gamma: Gamma for the patch-embed Conv2d; defaults to ``gamma``.
        attn_mode: ``"attnlrp"`` or ``"cp"`` — see the module docstring.

    Returns:
        Dict with ``target``, ``logit``, ``pixel_map`` ``[H, W]`` and
        ``relevance_ratio`` (sum of relevance divided by the explained logit; a
        conservation diagnostic, ~1.0 when relevance is conserved).
    """
    if attn_mode not in ("attnlrp", "cp"):
        raise ValueError(f"attn_mode must be 'attnlrp' or 'cp', got {attn_mode!r}")

    global ATTN_MODE
    ATTN_MODE = attn_mode
    holder: dict[str, int] = {}

    def scalar_fn(mod: nn.Module, xin: torch.Tensor) -> torch.Tensor:
        logits = mod(xin)
        t = int(logits[0].argmax()) if target is None else int(target)
        holder["target"] = t
        return logits[0, t]

    logit, pixel_map = _run_backward(model, x, scalar_fn, gamma, conv_gamma)
    denom = logit if abs(logit) > _ZERO_DIVISION_EPS else 1.0
    return {
        "target": holder["target"],
        "logit": logit,
        "pixel_map": pixel_map,
        "relevance_ratio": pixel_map.sum().item() / denom,
    }
