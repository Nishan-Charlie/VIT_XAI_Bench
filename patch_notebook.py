import nbformat as nbf
import json, os

filename = 'visualize_xai_methods.ipynb'
nb = nbf.read(filename, as_version=4)

# The new comprehensive cell code provided by the user, with fixes applied:
# 1. methods list uses 'attnlrp' (lowercase, matching the registry key)
# 2. No 'hilrp' remapping -- attnlrp is now directly registered
# 3. Skip attnlrp on non-vit architectures with N/A

new_cell_source = '''import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from torchvision import transforms
from xai_bench.registry import MODELS, METHODS

# --- Setup ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['figure.dpi'] = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_DISPLAY_NAMES = {
    'vit_base_patch16_224':            'ViT-B/16',
    'deit_base_patch16_224':           'DeiT-B/16',
    'resnet50':                        'ResNet-50',
    'swin_base_patch4_window7_224':    'Swin-B',
    'maxvit_small_tf_224':             'MaxViT-S',
    'mobilevitv2_100':                 'MobileViT-v2',
    'efficientvit_b1':                 'EfficientViT-B1',
    'efficientvit_b2':                 'EfficientViT-B2',
}

METHOD_DISPLAY_NAMES = {
    'saliency':            'Saliency',
    'integrated_gradients':'Integrated Grads',
    'smoothgrad':          'SmoothGrad',
    'grad_cam':            'Grad-CAM',
    'grad_cam_plus_plus':  'Grad-CAM++',
    'attention_rollout':   'Attn Rollout',
    'attnlrp':             'AttnLRP',          # LXT AttnLRP baseline (Achtibat 2024)
}

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor

img_path = 'images/cat.jpg'
img, tensor = load_image(img_path)

models_to_test = [
    'vit_base_patch16_224',
    'deit_base_patch16_224',
    'resnet50',
    'swin_base_patch4_window7_224',
    'maxvit_small_tf_224',
    'mobilevitv2_100',
    'efficientvit_b1',
    'efficientvit_b2',
]

# Methods list -- 'attnlrp' now refers directly to the AttnLRP (LXT) baseline,
# NOT HiLRP. AttnLRP uses ATTN_MODE='attnlrp' (softmax Taylor + divide_gradient).
methods_to_test = [
    'saliency',
    'integrated_gradients',
    'smoothgrad',
    'grad_cam',
    'grad_cam_plus_plus',
    'attention_rollout',
    'attnlrp',
]

# AttnLRP is only valid for flat ViT-family models.
FLAT_VIT_PREFIXES = ('vit', 'deit', 'beit')

# Attention Rollout is also only valid for flat ViT-family models.
FLAT_VIT_ATTENTION_PREFIXES = ('vit', 'deit', 'beit')

num_models = len(models_to_test)
num_cols   = len(methods_to_test) + 1   # +1 for original image

fig, axes = plt.subplots(nrows=num_models, ncols=num_cols,
                         figsize=(5 * num_cols, 5 * num_models))

if num_models == 1:
    axes = np.expand_dims(axes, axis=0)

def draw_placeholder(axis, text, method_title):
    axis.imshow(np.ones((224, 224, 3)))
    axis.text(112, 112, text, ha='center', va='center',
              fontsize=32, color='gray', fontweight='bold')
    axis.set_title(f"{method_title}\\n({text})")
    axis.axis('off')

for i, model_key in enumerate(models_to_test):
    display_model = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    print(f"\\n=== Evaluating {display_model} ===")

    wrapper = MODELS.get(model_key)()
    model   = wrapper.model
    model.to(device).eval()

    with torch.no_grad():
        output       = model(tensor)
        target_class = output.argmax(dim=1).item()

    # Column 0: original image
    ax_orig = axes[i, 0]
    ax_orig.imshow(img.resize((224, 224)))
    ax_orig.set_title(f'{display_model}\\nOriginal')
    ax_orig.axis('off')

    for j, method_name in enumerate(methods_to_test):
        ax = axes[i, j + 1]
        ax.axis('off')
        display_method = METHOD_DISPLAY_NAMES.get(method_name,
                             method_name.replace('_', ' ').title())

        # --- Attention Rollout: flat ViT only ---
        if method_name == 'attention_rollout':
            if not any(model_key.startswith(p) for p in FLAT_VIT_ATTENTION_PREFIXES):
                draw_placeholder(ax, 'N/A', display_method)
                continue

        # --- AttnLRP (LXT baseline): flat ViT only ---
        # attnlrp is registered in xai_bench/methods/attnlrp_method.py
        # It uses ATTN_MODE='attnlrp' (softmax Taylor + divide_gradient),
        # which is the Achtibat 2024 baseline -- NOT HiLRP (CP-LRP).
        if method_name == 'attnlrp':
            if not any(model_key.startswith(p) for p in FLAT_VIT_PREFIXES):
                draw_placeholder(ax, 'N/A', display_method)
                continue

        try:
            print(f"  Running {display_method}...", end=' ', flush=True)
            method  = METHODS.get(method_name)(model_wrapper=wrapper, model=model)
            heatmap = method(tensor, target=target_class)
            if isinstance(heatmap, tuple):
                heatmap = heatmap[0]
            hmap = heatmap.squeeze().cpu().detach().numpy()

            h_min, h_max = hmap.min(), hmap.max()
            if h_max > h_min:
                hmap = (hmap - h_min) / (h_max - h_min)
            else:
                hmap = np.zeros_like(hmap)

            ax.imshow(img.resize((224, 224)))
            ax.imshow(hmap, cmap='jet', alpha=0.5)
            ax.set_title(display_method)
            print('OK')
        except Exception as e:
            draw_placeholder(ax, 'Failed', display_method)
            print(f'FAILED: {e}')

    del model, wrapper
    torch.cuda.empty_cache()

plt.tight_layout()
os.makedirs('results/plots_publication', exist_ok=True)
out_path = 'results/plots_publication/qualitative_grid.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\\nSaved to {out_path}")
plt.show()'''

# Replace or append the main qualitative-grid cell
# Find and remove any existing cell containing 'registry_name = .hilrp.' so we
# don't duplicate, then append the corrected version.
cells_to_keep = []
for cell in nb.cells:
    src = ''.join(cell.get('source', []))
    # Drop old qualitative-grid cells we wrote previously
    if "registry_name = 'hilrp'" in src or ("attnLRP" in src and "METHOD_DISPLAY_NAMES" in src):
        print(f"Removing old cell (id={cell.get('id','?')})")
        continue
    cells_to_keep.append(cell)

nb.cells = cells_to_keep

# Prepend a markdown header + the corrected code cell
nb.cells.append(nbf.v4.new_markdown_cell(
    '## Qualitative Heatmap Grid\n\n'
    'Rows: each model. Columns: original image + one column per XAI method.\n\n'
    '**AttnLRP** here is the **LXT baseline** (Achtibat 2024) using ATTN_MODE="attnlrp" '
    '(softmax Taylor rule + divide_gradient bilinear rule). '
    'It is **not** HiLRP (which uses CP-LRP). '
    'AttnLRP and Attention Rollout are marked N/A for non-flat-ViT architectures.'
))
nb.cells.append(nbf.v4.new_code_cell(new_cell_source))

with open(filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook updated successfully.')
