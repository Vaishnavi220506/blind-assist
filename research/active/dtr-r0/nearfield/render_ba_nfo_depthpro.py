"""Review images from sealed predictions; selection is explicitly retrospective."""
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ba_nfo_depthpro import OUT, BASE, SOURCE, ROOT, write, sha


def overlay(rgb, prediction, truth, known):
    color=np.zeros_like(rgb)
    color[prediction&truth&known]=[20,220,140]
    color[prediction&~truth&known]=[255,45,180]
    color[~prediction&truth&known]=[255,70,40]
    color[~known]=[150,150,150]
    mask=(prediction|truth|~known)
    out=rgb.copy(); out[mask]=(.35*rgb[mask]+.65*color[mask]).astype(np.uint8)
    return out


def main():
    frames=json.loads((OUT/'frame-results.json').read_text())
    rows=json.loads((OUT/'manifest.json').read_text())
    eligible=[(i,r) for i,r in enumerate(frames) if r['metrics']['nfo']['far_small']['iou'] is not None]
    ranked=sorted(eligible,key=lambda ir:ir[1]['metrics']['low']['far_small']['iou']-ir[1]['metrics']['nfo']['far_small']['iou'])
    chosen=ranked[:2]+ranked[-2:]
    with np.load(BASE/'predictions.npz') as a:
        nfo=np.unpackbits(a['masks'][:,0],axis=1,bitorder='little').reshape(500,192,256).astype(bool)
    fig,axes=plt.subplots(4,5,figsize=(16,13))
    selection=[]
    for ri,(index,result) in enumerate(chosen):
        row=rows[index]
        with np.load(SOURCE/row['prepared']) as a:rgb=a['rgb']; depth=a['depth']
        known=np.isfinite(depth)&(depth>0); truth=known&(depth<2)
        preds={'NFO':nfo[index]}
        for arm in ['low','native']:
            with np.load(OUT/'predictions'/arm/f'{row["id"]}.npz') as a:preds[arm]=a['depth']<2
        images=[rgb,overlay(rgb,truth,truth,known)]+[overlay(rgb,p,truth,known) for p in preds.values()]
        for ax,im,title in zip(axes[ri],images,['RGB','Reference <2m','NFO .081','Depth Pro low','Depth Pro native']):
            ax.imshow(im); ax.set_title(title); ax.axis('off')
        delta=result['metrics']['low']['far_small']['iou']-result['metrics']['nfo']['far_small']['iou']
        axes[ri,0].set_title(f'{row["id"]}\nfar-small IoU delta {delta:+.3f}',fontsize=8)
        selection.append(dict(id=row['id'],delta=delta))
    fig.suptitle('Retrospective extremes: 2 largest losses + 2 largest gains in LOW far-small IoU\nGreen: true near; magenta: false near; orange: missed near; grey: unknown. All500 metrics remain authoritative.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95),h_pad=2.5);fig.savefig(OUT/'paired-extremes.png',dpi=140);plt.close(fig)
    g5=ROOT/'artifacts.local/nearfield/distinct-views-20260907-v1'
    record=json.loads((g5/'predictions/result.json').read_text())['rows'][5]
    rgb=cv2.cvtColor(cv2.imread(record['rgb_path']),cv2.COLOR_BGR2RGB)
    arrays=[np.load(record['native_path']),np.load(record['predicted_path'])]
    for arm in ['low','native']:
        with np.load(OUT/'diagnostics'/arm/'g5-0005.npz') as a:arrays.append(a['depth'])
    fig,axes=plt.subplots(1,5,figsize=(20,4))
    axes[0].imshow(rgb);axes[0].set_title('Old suspended-bar RGB')
    for ax,depth,label in zip(axes[1:],arrays,['Reference optical z','Cached DA V2','Depth Pro low','Depth Pro native']):
        im=ax.imshow(depth,cmap='turbo',vmin=0,vmax=6);ax.set_title(label)
    for ax in axes:ax.axis('off')
    fig.colorbar(im,ax=axes[1:],label='Optical-axis depth (m); color clipped at6m',shrink=.65)
    fig.savefig(OUT/'suspended-bar.png',dpi=140,bbox_inches='tight');plt.close(fig)
    write(OUT/'visual-selection.json',dict(rule='Posthoc two minimum and two maximum low-minus-NFO far-small IoU; diagnostic only',
        frames=selection,bar='Fixed historical G5 bar_near index5',renderer_sha256=sha(__file__)))
    print('RENDERED paired-extremes.png suspended-bar.png')


if __name__=='__main__':
    main()
