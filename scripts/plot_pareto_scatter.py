import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def format_name(name):
    name_map = {
        'grad_cam': 'Grad-CAM',
        'grad_cam_plus_plus': 'Grad-CAM++',
        'integrated_gradients': 'Integrated Gradients',
        'attention_rollout': 'Attention Rollout',
        'smoothgrad': 'SmoothGrad',
        'saliency': 'Saliency',
        'hilrp': 'AttnLRP',
        'lime': 'LIME',
        'input_x_gradient': 'Input x Gradient',
        'gradient_shap': 'GradientSHAP'
    }
    return name_map.get(name, name.replace('_', ' ').title())

def main():
    data_path = "results/full_benchmark_100/results.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Filter only ViT-B/16
    df = df[df['model'] == 'vit_base_patch16_224']
    
    # Calculate means
    means = df.groupby('method')[['time_ms', 'pointing_game']].mean().reset_index()
    means['method_formatted'] = means['method'].apply(format_name)
    
    # Sort for Pareto frontier calculation (Minimize cost, Maximize pointing_game)
    # We want low time_ms and high pointing_game
    pts = means[['time_ms', 'pointing_game']].values
    
    # Find Pareto optimal points
    pareto_mask = np.ones(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts):
        # A point is NOT pareto optimal if there is another point j such that:
        # time_j <= time_i AND pointing_j >= pointing_i AND (time_j < time_i OR pointing_j > pointing_i)
        for j, q in enumerate(pts):
            if i != j:
                if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                    pareto_mask[i] = False
                    break
                    
    pareto_points = means[pareto_mask].sort_values(by='time_ms')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot all points
    ax.scatter(means['time_ms'], means['pointing_game'], color='blue', s=100, alpha=0.7, edgecolors='black', label='Methods')
    
    # Outline Pareto Frontier
    ax.plot(pareto_points['time_ms'], pareto_points['pointing_game'], color='red', linestyle='--', linewidth=2, label='Pareto Frontier')
    
    # Annotate points
    for idx, row in means.iterrows():
        ax.annotate(row['method_formatted'], (row['time_ms'], row['pointing_game']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=11, alpha=0.9)
                    
    ax.set_xscale('log')
    ax.set_title("Cost vs Fidelity for ViT-B/16", fontweight="bold", fontsize=16)
    ax.set_xlabel("Mean Cost per Explanation (log ms)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Pointing Game Score", fontweight="bold", fontsize=14)
    
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    output_dir = "results/plots_publication"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "scatter_pareto_vit.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Pareto scatter chart saved to {out_path}")

if __name__ == "__main__":
    main()
