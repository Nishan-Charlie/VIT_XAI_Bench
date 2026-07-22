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

def format_model(name):
    model_map = {
        'vit_base_patch16_224': 'ViT-B/16',
        'deit_base_patch16_224': 'DeiT-B/16',
        'resnet50': 'ResNet-50'
    }
    return model_map.get(name, name)

def main():
    data_path = "results/full_benchmark_100/results.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    target_models = ['vit_base_patch16_224', 'deit_base_patch16_224', 'resnet50']
    df = df[df['model'].isin(target_models)].copy()
    
    df['method_formatted'] = df['method'].apply(format_name)
    df['model_formatted'] = df['model'].apply(format_model)
    
    # Calculate means
    means = df.groupby(['model_formatted', 'method_formatted'])['max_sensitivity'].mean().unstack()
    
    # Drop methods with NaN for all
    means = means.dropna(how='all')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Reorder columns to ensure ViT-B/16, DeiT-B/16, ResNet-50 if possible
    col_order = ['ViT-B/16', 'DeiT-B/16', 'ResNet-50']
    available_cols = [c for c in col_order if c in means.index]
    means = means.loc[available_cols]
    
    # Transpose so methods are on X axis
    pivot_df = means.T
    
    # Sort methods by average sensitivity
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values(by='avg', ascending=True)
    pivot_df = pivot_df.drop(columns=['avg'])
    
    pivot_df.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.8, alpha=0.9)
    
    ax.set_yscale('log')
    ax.set_title("Max-Sensitivity per Method (Robustness)", fontweight="bold", fontsize=16, pad=15)
    ax.set_ylabel("Max-Sensitivity (log scale, Lower=More Robust)", fontweight="bold", fontsize=14)
    ax.set_xlabel("")
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Highlight Attention Rollout
    try:
        method_idx = list(pivot_df.index).index('Attention Rollout')
        vit_val = pivot_df.loc['Attention Rollout', 'ViT-B/16']
        x_pos = method_idx - 0.27
        ax.annotate('Highly Robust', 
                    xy=(x_pos, vit_val), 
                    xytext=(x_pos-1.5, vit_val*5),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                    fontsize=12, fontweight='bold', color='green')
    except:
        pass
        
    # Highlight Grad-CAM++ on DeiT
    try:
        method_idx = list(pivot_df.index).index('Grad-CAM++')
        deit_val = pivot_df.loc['Grad-CAM++', 'DeiT-B/16']
        x_pos = method_idx
        ax.annotate('Extreme Outlier', 
                    xy=(x_pos, deit_val), 
                    xytext=(x_pos-1.5, deit_val*0.2),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                    fontsize=12, fontweight='bold', color='red')
    except:
        pass

    plt.legend(title='Architecture', title_fontsize='13', fontsize='12', loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_dir = "results/plots_publication"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "grouped_max_sensitivity.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Max-sensitivity chart saved to {out_path}")

if __name__ == "__main__":
    main()
