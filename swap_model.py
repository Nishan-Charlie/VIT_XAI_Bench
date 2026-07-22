import nbformat as nbf

filename = 'visualize_xai_methods.ipynb'
nb = nbf.read(filename, as_version=4)

for i, cell in enumerate(nb.cells):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'models_to_test' in src and 'MODEL_DISPLAY_NAMES' not in src:
        new_src = src.replace(
            "    'efficientvit_b2',",
            "    'pvt_v2_b2',"
        )
        if new_src != src:
            cell['source'] = new_src
            cell['outputs'] = []
            print(f"Updated cell {i}: replaced efficientvit_b2 with pvt_v2_b2")
        else:
            print(f"Cell {i}: efficientvit_b2 not found, no change")

# Also update the main grid cell's MODEL_DISPLAY_NAMES if present
for i, cell in enumerate(nb.cells):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'MODEL_DISPLAY_NAMES' in src and 'efficientvit_b2' in src:
        new_src = src.replace(
            "    'efficientvit_b2':                 'EfficientViT-B2',",
            "    'pvt_v2_b2':                       'PVT-v2-B2',"
        )
        cell['source'] = new_src
        cell['outputs'] = []
        print(f"Updated MODEL_DISPLAY_NAMES in cell {i}")

with open(filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Done.')
