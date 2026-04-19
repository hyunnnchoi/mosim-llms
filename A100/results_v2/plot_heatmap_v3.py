import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
})

df = pd.read_csv('/mnt/user-data/uploads/results_summary.csv')

model_order = ['vgg16', 'whisper', 'gpt2', 'bert', 'resnet50', 'densenet100_k12',
               'inception3', 'googlenet', 'resnet110', 'densenet40_k12', 'resnet44']
model_labels = ['VGG-16', 'Whisper', 'GPT-2', 'BERT', 'ResNet-50', 'DenseNet-100',
                'Inception-v3', 'GoogLeNet', 'ResNet-110', 'DenseNet-40', 'ResNet-44']

n = len(model_order)
heatmap_data = np.full((n, n), np.nan)

for i, pm in enumerate(model_order):
    for j, pt in enumerate(model_order):
        match = df[(df['model'] == pm) & (df['partner'] == pt) & (df['mode'] == 'pair')]
        if len(match) > 0:
            heatmap_data[i, j] = match.iloc[0]['slowdown_ratio']

heatmap_display = np.clip(heatmap_data, 1.0, None)

# Larger canvas -> when scaled to 3.45in column, text shrinks
fig, ax = plt.subplots(figsize=(6.5, 6.0))

norm = Normalize(vmin=1.0, vmax=2.2)
im = ax.imshow(heatmap_display, cmap='YlOrRd', norm=norm, aspect='equal')

for i in range(n):
    for j in range(n):
        val = heatmap_display[i, j]
        if not np.isnan(val):
            color = 'white' if val > 1.6 else 'black'
            fontweight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=9, fontweight=fontweight)

for i in range(n):
    rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                          linewidth=1.5, edgecolor='black', facecolor='none', zorder=5)
    ax.add_patch(rect)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(model_labels, rotation=45, ha='right')
ax.set_yticklabels(model_labels)

ax.set_xlabel('Co-located Job (4 GPUs)')
ax.set_ylabel('Primary Job (4 GPUs)')

cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label('Slowdown Ratio', fontsize=10)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(pad=0.4)
plt.savefig('/home/claude/interference_heatmap_small.pdf', bbox_inches='tight', pad_inches=0.03)
plt.savefig('/home/claude/interference_heatmap_small.png', bbox_inches='tight', pad_inches=0.03)
print("Done!")
