"""Multi-modal HiLRP on CLIP (ViT-B/16), text-conditioned attribution on clean
single-object images (koala, panda, owl). Qualitative demonstration: a caption
matching the image concentrates positive relevance on the object; an unrelated
caption does not. Regenerates figures/hilrp/clip_multimodal.png.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# transformers-compat shim for lxt.efficient (find_pruneable_heads_and_indices removed upstream)
import torch
import transformers.pytorch_utils as _pu
if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def _fp(heads, n_heads, head_size, already):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already
        for h in heads:
            h = h - sum(1 for x in already if x < h)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _pu.find_pruneable_heads_and_indices = _fp

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip

from xai_bench.methods.hilrp.clip_lxt import attribute_clip_image_text, ensure_patched
from xai_bench.methods.hilrp.viz import normalize_for_display

OUT = "figures/hilrp"
os.makedirs(OUT, exist_ok=True)

# (image file, matching caption, unrelated caption)
ITEMS = [
    ("images/koala.jpg", "a photo of a koala", "a photo of a car"),
    ("images/panda.jpg", "a photo of a giant panda", "a photo of a car"),
    ("images/owl.jpg",   "a photo of an owl", "a photo of a car"),
]


def main():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    ensure_patched()

    rows = []
    for path, cap_ok, cap_no in ITEMS:
        pil = Image.open(path).convert("RGB")
        x = preprocess(pil).unsqueeze(0)                      # CLIP-normalized [1,3,224,224]
        disp = np.array(pil.resize((224, 224))) / 255.0       # for display
        with torch.no_grad():
            te = model.encode_text(tok([cap_ok, cap_no]))
            te = te / te.norm(dim=-1, keepdim=True)
        r_ok = attribute_clip_image_text(model, x, te[0].detach())
        r_no = attribute_clip_image_text(model, x, te[1].detach())
        rows.append((disp, cap_ok, r_ok, cap_no, r_no))
        print(f"{path}: sim(ok)={r_ok['similarity']:.2f} sim(no)={r_no['similarity']:.2f}", flush=True)

    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(6.6, 2.2 * n))
    for r, (disp, cap_ok, r_ok, cap_no, r_no) in enumerate(rows):
        ax[r, 0].imshow(disp); ax[r, 0].axis("off")
        if r == 0:
            ax[r, 0].set_title("input", fontsize=9)
        for c, (res, cap) in enumerate([(r_ok, cap_ok), (r_no, cap_no)], start=1):
            m, v = normalize_for_display(res["pixel_map"], percentile=99, smooth_sigma=1.0)
            ax[r, c].imshow(m, cmap="bwr", vmin=-v, vmax=v); ax[r, c].axis("off")
            ax[r, c].set_title(f'"{cap}"\nsim={res["similarity"]:.2f}', fontsize=8)
    fig.suptitle("Multi-modal HiLRP: text-conditioned attribution on CLIP", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "clip_multimodal.png"), dpi=150, bbox_inches="tight")
    print(f"figure -> {OUT}/clip_multimodal.png")


if __name__ == "__main__":
    main()
