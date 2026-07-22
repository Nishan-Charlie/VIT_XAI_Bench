import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set research-based plot aesthetics
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

def format_model(name):
    model_map = {
        'vit_base_patch16_224': 'ViT-B/16',
        'deit_base_patch16_224': 'DeiT-B/16',
        'resnet50': 'ResNet-50'
    }
    return model_map.get(name, name)

def main():
    df = pd.read_csv("results/clean_summary.csv")
    
    # Filter only the models we want
    target_models = ['vit_base_patch16_224', 'deit_base_patch16_224', 'resnet50']
    df = df[df['model'].isin(target_models)].copy()
    
    df['method_formatted'] = df['method'].apply(format_name)
    df['model_formatted'] = df['model'].apply(format_model)
    
    # Pivot the data so methods are rows and models are columns for plotting
    pivot_df = df.pivot_table(index='method_formatted', columns='model_formatted', values='pointing_game', aggfunc='mean')
    
    # Drop methods that have NaN for all 3 models (should not happen, but safe)
    pivot_df = pivot_df.dropna(how='all')
    
    # We want a specific column order for the legend
    col_order = ['ViT-B/16', 'DeiT-B/16', 'ResNet-50']
    available_cols = [c for c in col_order if c in pivot_df.columns]
    pivot_df = pivot_df[available_cols]
    
    if 'ResNet-50' in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by='ResNet-50', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    pivot_df.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.8, alpha=0.9)
    
    ax.set_title("Pointing-Game Accuracy per Method", fontweight="bold", fontsize=16, pad=15)
    ax.set_ylabel("Pointing Game Score", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Highlight Grad-CAM++ collapse with an annotation
    try:
        method_idx = list(pivot_df.index).index('Grad-CAM++')
        vit_val = pivot_df.loc['Grad-CAM++', 'ViT-B/16']
        x_pos = method_idx - 0.27
        ax.annotate('Collapse on ViT', 
                    xy=(x_pos, vit_val), 
                    xytext=(x_pos-1.5, vit_val+0.3),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                    fontsize=12, fontweight='bold', color='red')
    except:
        pass

    plt.legend(title='Architecture', title_fontsize='13', fontsize='12', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_dir = "results/plots_publication"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grouped_pointing_game.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Grouped bar chart saved to {out_path}")

if __name__ == "__main__":
    main()
