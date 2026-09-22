"""Render descriptive train/dev diagnostic values without selecting a cutoff."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from query_occupancy_data import read
from diagnose_query_occupancy import DEST


def plot(root=DEST):
    root=Path(root);report=read(root/'result.json');assert report['status']=='PASS'
    data=report['splits'];fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors=['#64748b','#0e7490'];x=np.arange(2)
    for j,mode in enumerate(('classifier','occupancy')):
        values=[data[s][mode]['fixed_cutoff']['recall']*100 for s in ('train','dev')]
        bars=axes[0].bar(x+(j-.5)*.32,values,.32,color=colors[j],label=mode)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=9)
    axes[0].set(xticks=x,xticklabels=['train','dev'],ylabel='Frame recall (%)',ylim=(0,100),title='Original fixed alert cutoffs')
    axes[0].legend()
    keys=['lateral','height']
    for j,s in enumerate(('train','dev')):
        values=[data[s]['occupancy']['same_image_query_order'][k]['win_rate']*100 for k in keys]
        bars=axes[1].bar(x+(j-.5)*.32,values,.32,color=colors[j],label=s)
        axes[1].bar_label(bars,fmt='%.1f',fontsize=9)
    axes[1].axhline(50,color='#b45309',linestyle='--',linewidth=1)
    axes[1].set(xticks=x,xticklabels=['Left / centre / right','BODY / HEAD'],ylabel='Positive query ranks higher (%)',ylim=(0,100),title='Same image, differing query truth')
    axes[1].legend()
    for j,key in enumerate(('ap','prevalence')):
        values=[data[s]['occupancy']['visible_positive_ranking'][key]['mean']*100 for s in ('train','dev')]
        bars=axes[2].bar(x+(j-.5)*.32,values,.32,color=colors[j],label='Foreground AP' if key=='ap' else 'Constant-score AP')
        axes[2].bar_label(bars,fmt='%.2f',fontsize=9)
    axes[2].set(xticks=x,xticklabels=['train','dev'],ylabel='Macro average precision (%)',ylim=(0,100),title='Subthreshold heatmap ranking')
    axes[2].legend()
    for ax in axes: ax.grid(axis='y',alpha=.18)
    fig.suptitle('Frozen selected checkpoints | consumed simulation train/dev | no refit')
    fig.savefig(root/'diagnostic.png',dpi=170);fig.savefig(root/'diagnostic.svg');plt.close(fig)
    print(root/'diagnostic.png')


if __name__=='__main__':
    plot()
