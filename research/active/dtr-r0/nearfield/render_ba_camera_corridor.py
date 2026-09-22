"""All fixed clips/times; visualization never selects a better operating point."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2
import numpy as np

import ba_camera_corridor as c


def main():
    result = c.read(c.OUT/'results.json'); frames = c.read(c.OUT/'frame-results.json')
    sources = {r['id']: r for r in c.read(c.OUT/'evaluator-source.json')}
    observation = {r['id']: r for r in c.read(c.OUT/'observations.json')}
    clips = list(dict.fromkeys(r['clip_id'] for r in frames))
    arms = ('raw_tof','nfo','depthpro_global')
    colors = dict(tp='#299f7a',fp='#e64759',fn='#f29d38',tn='#e9edf2',unknown='#9298a4',boundary='#c09a30')
    from matplotlib.colors import to_rgb
    fig, axes = plt.subplots(8,1,figsize=(12,12),sharex=True)
    for ax, clip in zip(axes, clips):
        rows = sorted((r for r in frames if r['clip_id'] == clip), key=lambda r:r['frame_in_clip'])
        canvas = np.ones((4,12,3))
        for i,row in enumerate(rows):
            canvas[0,i] = to_rgb(colors['unknown'] if row['truth'] is None else colors['boundary'] if row['boundary'] else colors['tp'] if row['truth'] else colors['tn'])
            for j,arm in enumerate(arms,1):
                p = row['predictions'][arm]
                if row['truth'] is None:key='unknown'
                elif p['alert']:key='tp' if row['truth'] else 'fp'
                elif row['truth']:key='fn'
                else:key='unknown' if p['unknown'] else 'tn'
                canvas[j,i]=to_rgb(colors[key])
        ax.imshow(canvas,aspect='auto',interpolation='nearest',extent=(-.1,2.3,3.5,-.5))
        for row in rows:
            for j,arm in enumerate(arms,1):
                if row['predictions'][arm]['ambiguous']:
                    ax.plot(row['time_s'],j,'k.',markersize=3)
        ax.set_yticks(range(4),['Reference','Raw ToF','NFO','DP + global'],fontsize=7)
        ax.set_title(rows[0]['source_clip_id'],loc='left',fontsize=9)
        ax.set_xticks(np.arange(0,2.21,.2)); ax.grid(axis='x',alpha=.25)
    axes[-1].set_xlabel('Nominal sampled time (s), not measured real-time latency')
    fig.suptitle('All8 controlled clips / fixed96 samples\nBlack dot: geometrically ambiguous alert (still counted as TP or FP)',fontsize=12)
    fig.legend(handles=[Patch(color=colors[k],label=v) for k,v in
        [('tp','True hit / positive reference'),('fp','False alert'),('fn','Miss'),('tn','No alert on negative'),('unknown','UNKNOWN / abstention'),('boundary','Reference boundary')]],
        loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95)); fig.savefig(c.OUT/'all-clips-timeline.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(8,3,figsize=(12,15))
    for ri,clip in enumerate(clips):
        selected=[next(r for r in frames if r['clip_id']==clip and r['frame_in_clip']==i) for i in (0,6,11)]
        for ax,row in zip(axes[ri],selected):
            obs=observation[row['id']]
            rgb=cv2.cvtColor(cv2.imread(str(c.OUT/obs['rgb_path'])),cv2.COLOR_BGR2RGB)
            ax.imshow(rgb); ax.axis('off')
            ax.set_title(f'{row["source_clip_id"]}  t={row["time_s"]:.1f}s  truth={row["truth"]}'+(' boundary' if row['boundary'] else ''),fontsize=8)
    fig.suptitle('Predeclared source views: frame0,6,11 of every clip; no outcome selection',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(c.OUT/'source-views.png',dpi=110);plt.close(fig)
    c.write(c.OUT/'visual-receipt.json',dict(frames=96,clips=8,source_frames=[0,6,11],
        timeline='Every prediction, including UNKNOWN and boundary',renderer_sha256=c.sha(__file__),
        results_sha256=c.sha(c.OUT/'results.json')))
    print('CORRIDOR_VISUALS_COMPLETE')


if __name__ == '__main__':
    main()
