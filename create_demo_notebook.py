import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Visualizing XAI Methods across Vision Architectures\nThis notebook demonstrates how to load various architectures (CNN, Isotropic ViT, Hierarchical ViT) and apply different XAI methods to visualize their explanations."),
    nbf.v4.new_code_cell(
"""import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
from PIL import Image
from torchvision import transforms

# Suppress lxt patching warnings when running multiple methods
warnings.filterwarnings('ignore')

# Import from the local xai_bench package
from xai_bench.registry import MODELS, METHODS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"""),
    
    nbf.v4.new_markdown_cell("## 1. Helper Functions for Loading Images and Visualization"),
    nbf.v4.new_code_cell(
"""def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor

def plot_heatmap(img, heatmap, title):
    # Normalize heatmap to [0, 1] for visualization
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    plt.figure(figsize=(4, 4))
    plt.imshow(img.resize((224, 224)))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()"""),

    nbf.v4.new_markdown_cell("## 2. Models and Methods Configuration\nWe will evaluate the methods across a comprehensive suite of vision architectures. To prevent GPU memory exhaustion (OOM), we dynamically load and clear each model one at a time."),
    nbf.v4.new_code_cell(
"""models_to_test = [
    'resnet50',
    'vit_base_patch16_224',
    'deit_base_patch16_224',
    'swin_base_patch4_window7_224',
    'maxvit_small_tf_224',
    'mobilevitv2_100',
    'efficientvit_b1',
    'efficientvit_b2'
]

methods_to_test = [
    'saliency',
    'integrated_gradients',
    'smoothgrad',
    'grad_cam',
    'grad_cam_plus_plus',
    'attention_rollout',  # Only for ViT
    'attnLRP'
]"""),

    nbf.v4.new_markdown_cell("## 3. Run Methods and Visualize\nWe load a sample image, iterate through the models sequentially, and build a collage grid."),
    nbf.v4.new_code_cell(
"""# Provide the path to a sample image.
img_path = 'images/cat.jpg'
img, tensor = load_image(img_path)

# Create a grid for the collage
num_models = len(models_to_test)
num_cols = len(methods_to_test) + 1  # +1 for the original image
fig, axes = plt.subplots(nrows=num_models, ncols=num_cols, figsize=(4 * num_cols, 4 * num_models))

for i, model_key in enumerate(models_to_test):
    print(f"Evaluating {model_key}...")
    
    # Dynamically load model to save memory
    wrapper = MODELS.get(model_key)()
    model = wrapper.model
    model.to(device).eval()
    
    # Forward pass to get the predicted class
    with torch.no_grad():
        output = model(tensor)
        target_class = output.argmax(dim=1).item()
        
    # Plot original image in the first column
    ax_orig = axes[i, 0]
    ax_orig.imshow(img.resize((224, 224)))
    ax_orig.set_title(f'{model_key}\\nOriginal')
    ax_orig.axis('off')
    
    for j, method_name in enumerate(methods_to_test):
        ax = axes[i, j + 1]
        ax.axis('off')
        
        # Attention Rollout only works on standard ViTs
        if method_name == 'attention_rollout' and not ('vit' in model_key or 'deit' in model_key or 'beit' in model_key):
            ax.set_title(f"{method_name}\\n(N/A)")
            continue
            
        try:
            # Instantiate method for the current model
            registry_name = 'hilrp' if method_name == 'attnLRP' else method_name
            
            # Skip AttnLRP on CNNs gracefully
            if registry_name == 'hilrp' and ('resnet' in model_key or 'efficientvit' in model_key or 'mobilevit' in model_key or 'maxvit' in model_key):
                # Wait, HiLRP works on efficientvit, mobilevit, maxvit! It only fails on pure CNNs (ResNet) without self-attention.
                if 'resnet' in model_key:
                    ax.set_title(f"{method_name}\\n(N/A)")
                    continue
                
            method = METHODS.get(registry_name)(model_wrapper=wrapper, model=model)
            
            # Compute heatmap
            heatmap = method(tensor, target=target_class)
            if isinstance(heatmap, tuple):
                heatmap = heatmap[0]
            heatmap_np = heatmap.squeeze().cpu().detach().numpy()
            
            # Normalize heatmap
            heatmap_np = heatmap_np - heatmap_np.min()
            if heatmap_np.max() > 0:
                heatmap_np = heatmap_np / heatmap_np.max()
            
            # Plot on the specific axis
            ax.imshow(img.resize((224, 224)))
            ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
            ax.set_title(f"{method_name}")
            
        except Exception as e:
            ax.set_title(f"{method_name}\\n(Failed)")
            print(f"Failed to run {method_name} on {model_key}: {e}")
            
    # Aggressive memory cleanup before loading the next model
    del model
    del wrapper
    torch.cuda.empty_cache()

plt.tight_layout()
plt.show()
""")
]

with open("visualize_xai_methods.ipynb", "w", encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook 'visualize_xai_methods.ipynb' created successfully!")
