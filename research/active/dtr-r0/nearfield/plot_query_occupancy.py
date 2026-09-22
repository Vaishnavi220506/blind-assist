"""Deterministic figures from the complete sealed held comparison."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from query_occupancy_data import read, stage_path, CENTRE, EDGES


def main(root):
    root=Path(root)
    out=stage_path(root,'evaluated')
    report=read(out/'metrics.json')
    seal=read(stage_path(root,'predictions')/'prediction-seal.json')
    assert seal['held_labels_opened'] is False
    colors={'classifier':'#64748b','occupancy':'#0e7490'}
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for arm in colors:
        curve=report['descriptive_curves'][arm]
        ax[0].plot([p['FPR']*100 for p in curve],[p['recall']*100 for p in curve],label=arm,color=colors[arm])
        p=report['metrics']['arms'][arm]['frames']['all_known']
        ax[0].scatter([100*p['false_alert_rate_known_negative']],[100*p['recall']],color=colors[arm],s=50)
    for arm,marker,color in [('A_current','*','#b45309'),('A_hold','D','#9333ea')]:
        p=report['metrics']['arms'][arm]['frames']['all_known']
        ax[0].scatter([100*p['false_alert_rate_known_negative']],[100*p['recall']],
                      label=arm+' reference',marker=marker,color=color,s=85)
    ax[0].set(xlabel='False-positive frames / known negative frames (%)',ylabel='Recall (%)',
              title='Held layouts: complete descriptive curves',xlim=(0,100),ylim=(0,105))
    ax[0].legend(loc='lower right')
    families=list(report['strata']['type_id'])
    x=np.arange(len(families))
    for k,arm in enumerate(colors):
        values=[report['strata']['type_id'][f]['arms'][arm]['frames']['all_known']['recall']*100 for f in families]
        bars=ax[1].bar(x+(k-.5)*.36,values,width=.36,label=arm,color=colors[arm])
        ax[1].bar_label(bars,fmt='%.1f',fontsize=8)
    ax[1].set(xticks=x,xticklabels=[f.replace('_','\n') for f in families],ylabel='Recall (%)',ylim=(0,112),
              title='Fixed dev-selected cutoffs, all four families')
    for a in ax:
        a.grid(axis='y',alpha=.2)
    fig.suptitle('Single-frame query occupancy | controlled simulation Development')
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)

    # Predeclared display rule: first evaluation INSIDE group per family, frame5.
    identities=read(stage_path(root,'prepared')/'observations/identities.json')
    rgb=np.load(stage_path(root,'prepared')/'observations/rgb.npy',mmap_mode='r')
    labels=dict(np.load(stage_path(root,'prepared')/'labels/evaluation.npz'))
    occ=dict(np.load(stage_path(root,'predictions')/'occupancy.npz'))
    loc={int(i):j for j,i in enumerate(labels['indices'])}
    chosen=[next(r for r in identities if r['split']=='evaluation' and r['type_id']==f
                  and r['layout_relation']=='INSIDE' and r['frame_in_clip']==5) for f in families]
    fig,axes=plt.subplots(4,3,figsize=(11,10),layout='constrained')
    for row,(meta,panels) in enumerate(zip(chosen,axes)):
        j=loc[meta['index']]
        q=1 if meta['layer']=='BODY' else 4
        panels[0].imshow(rgb[meta['index']].transpose(1,2,0)); panels[0].set_title(meta['type_id'],fontsize=10)
        panels[1].imshow(labels['mask'][j,q],vmin=0,vmax=1,cmap='Blues')
        panels[1].contour(occ['mask'][j,q],levels=[.5],colors=['#e11d48'],linewidths=.7)
        panels[1].set_title('Visible occupancy (blue), prediction 0.5 (red)',fontsize=9)
        panels[2].bar(np.arange(7),occ['distribution'][j,q],color='#0e7490')
        panels[2].set(xticks=np.arange(7),xticklabels=['.75','1.25','1.75','2.25','2.75','3','none'],ylim=(0,1))
        panels[2].tick_params(axis='x',labelsize=8)
        truth=labels['classes'][j,q]
        panels[2].set_title(f'First-hit bin probability | truth bin {truth}',fontsize=9)
        panels[2].set_xlabel('Bin upper edge (m); none is a model class',fontsize=8)
        for p in panels[:2]:
            p.axis('off')
    fig.suptitle('Fixed representative held frames; predictions do not certify free space',fontsize=12)
    fig.savefig(out/'examples.png',dpi=160)
    plt.close(fig)
    print(out/'comparison.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
