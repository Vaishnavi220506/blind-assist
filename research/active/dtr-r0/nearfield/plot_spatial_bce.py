"""Render the already-sealed train/dev evidence; never select another method."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    rows = read(root / 'development-thresholds.json')
    op = read(root / 'operating-point.json')
    train = [json.loads(line) for line in (root / 'fit/training.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].plot([r['step'] for r in train], [r['train_batch_bce'] for r in train], color='#236878')
    axes[0].set(title='One fixed BCE fit', xlabel='Optimizer updates', ylabel='Logged batch BCE')
    for ax, name in zip(axes[1:], ('Core', 'Boundary')):
        data = [r['summary'][name + '_current'] for r in rows]
        fp = [r['B_FP'] for r in data]
        recall = [100*r['B_TP']/r['positives'] for r in data]
        ax.scatter(fp, recall, s=8, color='#8098b0', label='B: dev threshold diagnostics')
        retained = [i for i, r in enumerate(rows) if r['checks']['retain_Core_current']
                    and r['checks']['retain_Core_hold']]
        ax.scatter(np.array(fp)[retained], np.array(recall)[retained], s=9,
                   color='#d98c35', label='B: retains all Core baseline TP')
        base = data[0]
        ax.scatter([base['A_FP']], [100*base['A_TP']/base['positives']], marker='*',
                   color='#215f46', s=160, label='A: frozen baseline', zorder=5)
        if op['selected']:
            picked = op['selected']['summary'][name + '_current']
            ax.scatter([picked['B_FP']], [100*picked['B_TP']/picked['positives']],
                       marker='D', color='#7e2758', s=55, label='Admitted dev threshold')
        ax.set(title=f'Development {name}: current frame', xlabel='False-positive frames',
               ylabel='Positive-frame recall (%)', ylim=(-3, 103))
        ax.grid(alpha=.2)
    axes[2].legend(fontsize=7, loc='lower right')
    fig.suptitle(f"Spatial RGB-ToF BCE | {op['status']}\n"
                 '24 train / 8 dev / 8 test procedural groups; curves are dev diagnostics, not test results', fontsize=11)
    fig.savefig(root / 'development-evidence.png', dpi=170)
    fig.savefig(root / 'development-evidence.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
