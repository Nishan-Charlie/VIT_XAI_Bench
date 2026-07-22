"""Regenerate the RQ2 label-free figure (view-invariance scalar under different
pretraining objectives) on clean single-object images (koala, panda, owl).
Figure only; the quantitative RQ2 table stays the n=30 result. -> figures/hilrp/rq2_maps.png
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size); heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h); mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _pu.find_pruneable_heads_and_indices = _fp

import numpy as np
from PIL import Image
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/hilrp"
os.makedirs(OUT, exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODELS = {
    "supervised": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "DINO":       "vit_base_patch16_224.dino",
    "DINOv2":     "vit_base_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_base_patch14_reg4_dinov2.lvd142m",
    "MAE":        "vit_base_patch16_224.mae",
}
IMAGES = ["images/koala.jpg", "images/panda.jpg", "images/owl.jpg"]


def load(path):
    pil = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (x - MEAN) / STD, np.array(pil) / 255.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from xai_bench.methods.hilrp.vit_lxt import attribute_ssl_similarity, ensure_patched
    ensure_patched()
    models = {t: timm.create_model(n, pretrained=True, num_classes=0, img_size=224).eval().to(device)
              for t, n in MODELS.items()}
    tags = list(MODELS)
    inputs = [load(p) for p in IMAGES]

    demo = {t: [] for t in tags}
    for x, _ in inputs:
        x = x.to(device); x_ref = torch.flip(x, dims=(3,))
        for t in tags:
            res = attribute_ssl_similarity(models[t], x, x_ref)
            demo[t].append(res["pixel_map"].numpy())
        print(f"done one image", flush=True)

    n = len(inputs)
    fig, ax = plt.subplots(n, 1 + len(tags), figsize=(2.1 * (1 + len(tags)), 2.1 * n))
    for i in range(n):
        ax[i, 0].imshow(inputs[i][1]); ax[i, 0].axis("off")
        if i == 0: ax[i, 0].set_title("input", fontsize=8)
        for j, t in enumerate(tags):
            m = demo[t][i]; v = np.abs(m).max() + 1e-12
            ax[i, 1 + j].imshow(m, cmap="bwr", vmin=-v, vmax=v); ax[i, 1 + j].axis("off")
            if i == 0: ax[i, 1 + j].set_title(t, fontsize=8)
    fig.suptitle(f"RQ2: view-invariance evidence under {len(tags)} pretraining objectives "
                 "(frozen ViT-B)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rq2_maps.png"), dpi=150, bbox_inches="tight")
    print(f"figure -> {OUT}/rq2_maps.png")


if __name__ == "__main__":
    main()
