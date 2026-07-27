import warnings

import timm
import torch

from scripts.scaled_eval import load_cache, pointing
from xai_bench.methods.vit_lrp_backend import attribute_vit

warnings.filterwarnings('ignore')

model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
data, _ = load_cache()
n = 100

cfg = model.pretrained_cfg
m = torch.tensor(cfg['mean']).view(3, 1, 1)
s = torch.tensor(cfg['std']).view(3, 1, 1)
IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

gammas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
best_gamma = -1
best_score = -1

print("Sweeping gamma for ViT-B using CP-LRP...")
for g in gammas:
    hits = 0
    for i in range(n):
        img = data[i]['image'].unsqueeze(0)
        img = ((img * IS + IM).clamp(0, 1) - m) / s
        res = attribute_vit(model, img, gamma=g, attn_mode='cp')
        hits += pointing(res['pixel_map'].numpy(), data[i]['metadata']['bboxes'])

    score = hits / n
    print(f"Gamma={g:.2f} -> Score: {score:.3f}")
    if score > best_score:
        best_score = score
        best_gamma = g

print(f"\nBest Gamma: {best_gamma:.2f} with Score: {best_score:.3f}")
