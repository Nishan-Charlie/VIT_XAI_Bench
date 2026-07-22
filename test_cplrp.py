import timm
import torch
import numpy as np
from xai_bench.methods.hilrp.vit_lxt import attribute_vit
from scripts.scaled_eval import load_cache, pointing
import warnings
warnings.filterwarnings('ignore')

model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
data, _ = load_cache()
hits = 0
n = 100
for i in range(n):
    img = data[i]['image'].unsqueeze(0)
    cfg = model.pretrained_cfg
    m = torch.tensor(cfg['mean']).view(3, 1, 1)
    s = torch.tensor(cfg['std']).view(3, 1, 1)
    IM = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IS = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = ((img * IS + IM).clamp(0, 1) - m) / s
    res = attribute_vit(model, img, gamma=0.25, attn_mode='cp')
    hits += pointing(res['pixel_map'].numpy(), data[i]['metadata']['bboxes'])

print(f'ViT-B with CP-LRP (n={n}): {hits/n:.3f}')
