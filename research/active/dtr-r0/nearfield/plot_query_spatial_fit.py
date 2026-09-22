"""Render sealed tiny-training results; no cutoff or example optimization."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from query_occupancy_data import read
from run_query_spatial_fit import DEST


def plot(root=DEST):
    root = Path(root)
    result = read(root/'result.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    colors = ['#64748b', '#0e7490']
    for i, arm in enumerate(('global', 'spatial')):
        metrics = result['arms'][arm]
        values = [metrics['query']['precision'], metrics['query']['recall'], metrics['mask_iou'],
                  metrics['same_image_query_order']['all']['win_rate']]
        bars = axes[0].bar(np.arange(4)+(i-.5)*.35, np.array(values)*100, .35, color=colors[i], label=arm)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=9)
        history = read(root/f'{arm}-history.json')
        axes[1].plot([h['updates'] for h in history], [h['loss'] for h in history], color=colors[i], label=arm)
    axes[0].scatter(np.arange(4), [95, 95, 50, 95], marker='_', s=300, color='#b45309', label='Fixed fit gate')
    axes[0].set(xticks=np.arange(4), xticklabels=['Precision', 'Recall', 'Mask IoU', 'Query pair order'],
                ylim=(0, 108), ylabel='Percent', title='Final epoch 100 | fixed cutoffs 0.5')
    axes[1].set(xlabel='Optimizer updates', ylabel='Shared fitting loss', title='All training epochs, no checkpoint selection')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('84 consumed training images | 4 groups | training fit, not generalization')
    fig.savefig(root/'spatial-fit.png', dpi=170)
    fig.savefig(root/'spatial-fit.svg')
    plt.close(fig)

    cohort = read(root/'cohort.json')['metadata']
    target = np.load(root/'training-targets.npz')['mask']
    predictions = {a: np.load(root/f'{a}-predictions.npz')['mask'] for a in ('global', 'spatial')}
    families = sorted({m['type_id'] for m in cohort})
    fig, axes = plt.subplots(4, 3, figsize=(10, 9), layout='constrained')
    for r, family in enumerate(families):
        index = next(i for i, m in enumerate(cohort) if m['type_id'] == family
                     and m['layout_relation'] == 'INSIDE' and m['frame_in_clip'] == 5)
        q = 4 if cohort[index]['layer'] == 'HEAD' else 1
        for c, data in enumerate((target[index, q], predictions['global'][index, q], predictions['spatial'][index, q])):
            im = axes[r, c].imshow(data, cmap='viridis', vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(family+'\ncentre query', fontsize=9)
            if r == 0:
                axes[r, c].set_title(('Native area target', 'Global probability', 'Spatial probability')[c])
    fig.colorbar(im, ax=axes, shrink=.6, label='Fraction / probability')
    fig.suptitle('Fixed examples: INSIDE, frame 5, each family | training images')
    fig.savefig(root/'spatial-fit-masks.png', dpi=170)
    plt.close(fig)
    print(root/'spatial-fit.png')


if __name__ == '__main__':
    plot()
