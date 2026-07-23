import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import networkx as nx
import community as community_louvain # python-louvain
import warnings
warnings.filterwarnings('ignore')

from xai_bench.registry import MODELS, METHODS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    img, tensor = load_image(img_path)
    
    wrapper = MODELS.get('vit_base_patch16_224')()
    model = wrapper.model
    model.to(device).eval()
    
    with torch.no_grad():
        output = model(tensor)
        target_class = output.argmax(dim=1).item()
        
    print("Computing Grad-CAM...")
    gradcam_method = METHODS.get('grad_cam')(model_wrapper=wrapper, model=model)
    gradcam_map = gradcam_method(tensor, target=target_class)
    if isinstance(gradcam_map, tuple): gradcam_map = gradcam_map[0]
    gradcam_np = gradcam_map.squeeze().cpu().detach().numpy()
    
    print("Computing Input x Gradient...")
    ig_method = METHODS.get('input_x_gradient')(model_wrapper=wrapper, model=model)
    ig_map = ig_method(tensor, target=target_class)
    if isinstance(ig_map, tuple): ig_map = ig_map[0]
    ig_np = ig_map.squeeze().cpu().detach().numpy()
    
    # Normalize to [0,1]
    def normalize(hm):
        hm = hm - hm.min()
        if hm.max() > 0:
            hm = hm / hm.max()
        return hm
        
    gradcam_np = normalize(gradcam_np)
    ig_np = normalize(ig_np)
    
    # Element-wise product
    combined = gradcam_np * ig_np
    combined = normalize(combined)
    
    # Resize to 224x224 to match original image space
    from skimage.transform import resize
    combined_224 = resize(combined, (224, 224), anti_aliasing=True)
    
    print("Building 4-connected graph from thresholded region...")
    threshold = np.percentile(combined_224, 85) # Top 15%
    mask = combined_224 > threshold
    
    # Create full grid graph
    G = nx.grid_2d_graph(224, 224)
    
    # Keep only nodes that pass the threshold
    nodes_to_remove = [(i, j) for i in range(224) for j in range(224) if not mask[i, j]]
    G.remove_nodes_from(nodes_to_remove)
    
    # Add node weights (mass)
    for node in G.nodes():
        G.nodes[node]['mass'] = combined_224[node[0], node[1]]
        
    print("Running Louvain community detection...")
    # Louvain
    partition = community_louvain.best_partition(G)
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show original image in background
    ax.imshow(img.resize((224, 224)))
    
    # Extract node coords, colors (community), and sizes (mass)
    xs = [node[1] for node in G.nodes()] # columns (x)
    ys = [node[0] for node in G.nodes()] # rows (y)
    colors = [partition[node] for node in G.nodes()]
    sizes = [G.nodes[node]['mass'] * 50 for node in G.nodes()] # scaled for visibility
    
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap='tab20', alpha=0.8, edgecolors='none')
    
    ax.set_title("Pixel-Graph Clustering (Grad-CAM $\\times$ Input*Grad, Louvain)", fontweight="bold", fontsize=14)
    ax.axis('off')
    
    out_dir = "results/ImageNetS"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_vit_graph.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Graph clustering saved to {out_path}")

if __name__ == "__main__":
    main()
