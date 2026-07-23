import nbformat as nbf

filename = 'visualize_xai_methods.ipynb'
nb = nbf.read(filename, as_version=4)

ALL_MODELS = [
    'vit_base_patch16_224',
    'deit_base_patch16_224',
    'resnet50',
    'swin_base_patch4_window7_224',
    'maxvit_small_tf_224',
    'mobilevitv2_100',
    'efficientvit_b1',
    'efficientvit_b2',
]

ALL_METHODS = [
    'saliency',
    'integrated_gradients',
    'smoothgrad',
    'grad_cam',
    'grad_cam_plus_plus',
    'attention_rollout',
    'attnlrp',
]

new_config_src = (
    "# === Model and Method Configuration ===\n"
    "# All models registered in the xai_bench MODELS registry.\n"
    "# AttnLRP and Attention Rollout are only valid for flat ViT-family models\n"
    "# (vit_*, deit_*, beit_*); all other rows show N/A for those columns.\n"
    "models_to_test = [\n"
    "    'vit_base_patch16_224',\n"
    "    'deit_base_patch16_224',\n"
    "    'resnet50',\n"
    "    'swin_base_patch4_window7_224',\n"
    "    'maxvit_small_tf_224',\n"
    "    'mobilevitv2_100',\n"
    "    'efficientvit_b1',\n"
    "    'efficientvit_b2',\n"
    "]\n"
    "\n"
    "methods_to_test = [\n"
    "    'saliency',\n"
    "    'integrated_gradients',\n"
    "    'smoothgrad',\n"
    "    'grad_cam',\n"
    "    'grad_cam_plus_plus',\n"
    "    'attention_rollout',\n"
    "    'attnlrp',          # LXT AttnLRP baseline (Achtibat 2024), NOT HiLRP\n"
    "]\n"
    "\n"
    "print(f'Models: {len(models_to_test)}, Methods: {len(methods_to_test)}')"
)

replaced = 0
for i, cell in enumerate(nb.cells):
    src = ''.join(cell.get('source', []))
    # Target: old 3-model config cell OR any config cell with old 3-model list
    if cell['cell_type'] == 'code' and 'models_to_test' in src and 'methods_to_test' in src and 'MODEL_DISPLAY_NAMES' not in src:
        print(f"Replacing config cell {i} (id={cell.get('id','?')})")
        print(f"  Old content preview: {src[:120]!r}")
        cell['source'] = new_config_src
        cell['outputs'] = []
        replaced += 1

if replaced == 0:
    print("No old config cell found -- inserting new one before the grid cell.")
    # Find the grid cell and insert before it
    grid_idx = None
    for i, cell in enumerate(nb.cells):
        src = ''.join(cell.get('source', []))
        if 'MODEL_DISPLAY_NAMES' in src and 'FLAT_VIT_PREFIXES' in src:
            grid_idx = i
            break
    if grid_idx is not None:
        nb.cells.insert(grid_idx, nbf.v4.new_code_cell(new_config_src))
        print(f"Inserted config cell before grid cell at index {grid_idx}")

with open(filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Done.')
