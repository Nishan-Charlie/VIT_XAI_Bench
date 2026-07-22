# Prompt: HiLRP Methodology / Graphical-Abstract Diagram

Use this prompt with a diagram/vector generator (or as a TikZ / draw.io spec). Every
element below is grounded in the actual implementation; keep the names and rules
verbatim. The figure is the paper's methodology overview / graphical abstract for
*"One Trustable Explanation for Any Vision Transformer: Conservation-Valid
Attribution via Attention Primitives."*

## What the figure must say at a glance
Two messages, side by side:
1. **TRUSTABLE** - the attribution is an *exact decomposition of the prediction*:
   pixel relevance conserves and sums to the class logit (verified to machine
   precision).
2. **UNIVERSAL** - it works on *any* ViT because every ViT is a composition of
   **four operation primitives**, each with **one** conservation-preserving
   relevance rule. A new architecture is a new *arrangement* of the same four.

## Canvas & style
- Landscape, ~2-column (double-column) width, single coherent left-to-right flow.
- Flat/modern, thin rounded rectangles, one restrained accent color per stage,
  white background, dark sans-serif labels. Must survive grayscale printing
  (use distinct shapes/line-weights, not just hue).
- Two horizontal "guarantee" ribbons spanning the relevant stages (see below).
- Small monospace annotations for the exact code operations (optional layer).

## Main pipeline (5 stages, left -> right)

**A. INPUT - "Any Vision Transformer."**
A short stack of backbone chips showing the diversity the method must cover:
isotropic ViT/DeiT, Swin (shifted windows), PVT (spatial reduction),
EfficientViT (linear cross-covariance attention), MobileViT (conv hybrid),
MaxViT (multi-axis), CrossViT (cross-attention), CLIP (image-text). Include one
input image (e.g., the cheetah). Sub-label: *"present or future; any mix of
stems, attention variants, patch-merging, normalizations."*

**B. DECOMPOSE.**
A single box: *"Decompose the network into 4 operation primitives."* Arrow to C.

**C. THE FOUR PRIMITIVES (visual centerpiece).**
Four colored cards (a 2x2 grid or a 4-row rail). Each card has: number, name,
`covers:` (the operations), `rule:` (the conserving relevance rule). Use exactly:

1. **Linear map** - covers: Q/K/V and output projections, MLP, patch-embed conv,
   spatial-reduction conv, poolings, kernel maps. *rule:* epsilon/gamma-LRP
   (`lrp_linear`; zennit `Gamma(0.25)` on `nn.Linear` and `nn.Conv2d`).
2. **Bilinear mixing** - covers: the attention matmuls Q.K^T and attn.V; linear-
   attention products. *rule:* uniform bilinear split (`bilinear_lrp` /
   `divide_gradient`) - one-half of the relevance to each operand; conserves.
3. **Normalization / gating** - covers: softmax, LayerNorm, linear-attention
   denominator, sigmoid / squeeze-excite channel gate, cosine norm. *rule:*
   **detach the normalizing denominator / detach the gate** -> locally linear ->
   pass-through identity (`lrp_softmax_identity`, `lrp_layernorm_identity`,
   `gate_lrp`). CP-LRP: softmax is detached so relevance flows through the value
   path.
4. **Reindexing / sparsity** - covers: heads, windows, shifts (Swin), dilation,
   groups, patch merging, spatial reduction, gather/scatter. *rule:* **exact
   permutation** (Lemma 2) - a selection/permutation is a 0/1 doubly-stochastic
   map, so it conserves exactly.

**D. ONE BACKWARD PASS.**
Box: *"Single backward pass under the composite."* Key points to render:
- Forward numerics are **unchanged**; only the *gradient path* is redefined by the
  per-primitive rules (class-level forward patches on `GELU`, `LayerNorm`
  including the timm `LayerNorm` subclass, and the `Attention` class).
- The efficient LXT Gradient x Input backend => relevance = input x input-gradient.
- Per-family attention flag: **CP-LRP by default** (AttnLRP-through mode is used
  only for the lineage comparison).

**E. OUTPUT - "Conservation-valid attribution."**
The HiLRP pixel heatmap overlaid on the input (cheetah), localizing on
discriminative parts. Optionally show a small stage-wise relevance bar (patch
embed -> stages -> pixels) all summing to the logit.

## Guarantee ribbons
- **UNIVERSAL** ribbon spanning A -> C: *"Any new architecture is a new arrangement
  of the same four primitives - covered by construction, not by a new derivation."*
- **TRUSTABLE** ribbon spanning C -> E: *"Relevance conserves: sum of pixel
  relevance = class logit (machine precision, ~1e-6 relative error)."* Render the
  equation `sum_i R_i = f_c(x)`.

## Exact code annotations (small monospace, optional overlay)
- Primitive 1: `zennit Gamma(0.25) on nn.Linear, nn.Conv2d`
- Primitive 3: `x = attn.detach() @ v   # CP-LRP`
- Stage D: `pixel_map = (x * x.grad).sum(channel);  pixel_map.sum() / logit ~= 1`
- Label-free offshoot from D (optional, thin branch): *"SSL scalar:
  `s = cos(cls(x), cls(x_ref).detach())` - attribute self-supervised similarity,
  no labels, no classifier head."*

## Do / Don't
- DO make the 4-primitive panel the focal point; keep one clean left-to-right story.
- DO keep the primitive names and the `covers:` / `rule:` structure verbatim.
- DO show that forward is unchanged and only the backward/gradient path is ruled.
- DON'T invent metrics, layer counts, or numbers beyond those listed here.
- DON'T draw transformer internals beyond the four primitive cards.

## Source of truth (files this diagram summarizes)
`xai_bench/methods/hilrp/attention_primitives.py` (the four primitives + rules),
`rules.py` (`lrp_linear`, `bilinear_lrp`, `lrp_softmax_identity`,
`lrp_layernorm_identity`, `gate_lrp`), `vit_lxt.py` (patching, `Gamma` composite,
single-backward driver, CP vs AttnLRP flag), `hilrp_method.py` (per-family
dispatch).
