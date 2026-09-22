"""Posthoc paired failure/gain examples, never selection or tuning inputs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ba_nfo_depthpro_echo as run
from ba_nfo_frozen_transfer import evaluation_domains
from render_ba_nfo_depthpro import overlay


def main():
    rows = run.read(run.OUT/'manifest.json')
    frames = run.read(run.OUT/'frame-results.json')
    ranked = sorted((i for i, f in enumerate(frames) if f['metrics']['native']['far_small']['tp']+
                     f['metrics']['native']['far_small']['fn'] > 0),
                    key=lambda i: (frames[i]['metrics']['layered']['far_small']['tp']-
                                   frames[i]['metrics']['global']['far_small']['tp'], i))
    chosen = ranked[:2]+ranked[-2:]
    fig, axes = plt.subplots(4, 6, figsize=(16, 9))
    selection = []
    for ri, index in enumerate(chosen):
        row, frame = rows[index], frames[index]
        with np.load(run.SOURCE/row['prepared']) as data:
            a = {k: data[k].copy() for k in ('rgb', 'depth', 'boxes', 'values')}
        truth, domains = evaluation_domains(a); known = domains['full']
        with np.load(run.OLD/'predictions/native'/f'{row["id"]}.npz') as data:
            native = data['depth'] < 2.
        with np.load(run.OUT/'predictions'/f'{row["id"]}.npz') as data:
            global_, layer = data['global_depth'] < 2., data['layered_depth'] < 2.
            support = data['support'][data['labels']] > 0
        y, x = np.where(domains['far_small'])
        y0, y1, x0, x1 = max(0, y.min()-8), min(192, y.max()+9), max(0, x.min()-8), min(256, x.max()+9)
        crop = np.s_[y0:y1, x0:x1]
        images = [a['rgb'], overlay(a['rgb'], truth, truth, known),
                  overlay(a['rgb'], native, truth, known), overlay(a['rgb'], global_, truth, known),
                  overlay(a['rgb'], layer, truth, known), np.where(support[..., None], [20, 160, 230], [35, 35, 35])]
        for ax, im, title in zip(axes[ri], images, ('RGB', 'Reference <2m', 'Frozen Depth Pro', '+ Global scale', '+ Layer scale', 'ToF-assigned pixels')):
            ax.imshow(im[crop]); ax.set_title(title, fontsize=9); ax.axis('off')
        delta = frame['metrics']['layered']['far_small']['tp']-frame['metrics']['global']['far_small']['tp']
        axes[ri, 0].set_title(f'{row["id"]}\nLayer minus global TP: {delta:+d}', fontsize=7)
        selection.append(dict(id=row['id'], layer_minus_global_far_small_tp=delta,
                              crop_y0_y1_x0_x1=[int(v) for v in (y0, y1, x0, x1)]))
    fig.suptitle('Retrospective extremes: two largest layer losses and two largest gains vs global\n'
                 'Green: correct near; orange: missed near; magenta: false near; grey: UNKNOWN. '
                 'Blue support means algorithm-assigned, not verified correct.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2)
    fig.savefig(run.OUT/'paired-extremes.png', dpi=150); plt.close(fig)
    run.write(run.OUT/'visual-selection.json', dict(rule='Posthoc extremes by layered-minus-global far-small TP; all500 scores authoritative; crop bounds are evaluator-only far-small extent plus8pixels',
                                                   frames=selection, renderer_sha256=run.sha(__file__)))
    print('ECHO_RENDERED', run.OUT/'paired-extremes.png')


if __name__ == '__main__':
    main()
