import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("Imports complete, starting script...", flush=True)

from xai_bench.registry import MODELS, METHODS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor

def main():
    img_path = 'images/cat.jpg'
    print(f"Loading image {img_path}...", flush=True)
    img, tensor = load_image(img_path)
    
    models_to_test = {
        'vit_base_patch16_224': 'ViT-B/16',
        'deit_base_patch16_224': 'DeiT-B/16',
        'resnet50': 'ResNet-50'
    }
    
    methods_to_test = [
        'saliency',
        'grad_cam',
        'grad_cam_plus_plus',
        'attention_rollout',
        'hilrp'
    ]
    
    method_labels = {
        'saliency': 'Saliency',
        'grad_cam': 'Grad-CAM',
        'grad_cam_plus_plus': 'Grad-CAM++',
        'attention_rollout': 'Attn Rollout',
        'hilrp': 'AttnLRP'
    }
    
    num_models = len(models_to_test)
    num_cols = len(methods_to_test) + 1
    
    fig, axes = plt.subplots(nrows=num_models, ncols=num_cols, figsize=(3 * num_cols, 3 * num_models))
    
    for i, (model_key, model_name) in enumerate(models_to_test.items()):
        print(f"Generating for {model_name}...", flush=True)
        wrapper = MODELS.get(model_key)()
        model = wrapper.model
        model.to(device).eval()
        
        with torch.no_grad():
            output = model(tensor)
            target_class = output.argmax(dim=1).item()
            
        ax_orig = axes[i, 0]
        ax_orig.imshow(img.resize((224, 224)))
        if i == 0:
            ax_orig.set_title('Original', fontweight="bold")
        ax_orig.set_ylabel(model_name, fontweight="bold", fontsize=14)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        
        for j, method_name in enumerate(methods_to_test):
            ax = axes[i, j + 1]
            ax.axis('off')
            
            if i == 0:
                ax.set_title(method_labels[method_name], fontweight="bold")
                
            if method_name == 'attention_rollout' and 'ResNet' in model_name:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=16, color='gray')
                continue
            if method_name == 'hilrp' and 'ResNet' in model_name:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=16, color='gray')
                continue
                
            print(f"  -> Running {method_name}...", flush=True)
            try:
                method = METHODS.get(method_name)(model_wrapper=wrapper, model=model)
                heatmap = method(tensor, target=target_class)
                if isinstance(heatmap, tuple):
                    heatmap = heatmap[0]
                heatmap_np = heatmap.squeeze().cpu().detach().numpy()
                
                heatmap_np = heatmap_np - heatmap_np.min()
                if heatmap_np.max() > 0:
                    heatmap_np = heatmap_np / heatmap_np.max()
                    
                ax.imshow(img.resize((224, 224)))
                ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
            except Exception as e:
                print(f"Error {method_name} on {model_name}: {e}", flush=True)
                ax.text(0.5, 0.5, "Error", ha='center', va='center', fontsize=16, color='red')
                
        del model
        del wrapper
        torch.cuda.empty_cache()
        
    plt.tight_layout()
    
    output_dir = "results/plots_publication"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "qualitative_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Qualitative grid saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
