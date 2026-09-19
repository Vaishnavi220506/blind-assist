"""Every fixed endpoint and every selected segment, without outcome selection."""
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/ba-contour-parallax-20260919'


def main():
    predictions=json.loads((OUT/'predictions.json').read_text())
    observations=json.loads((OUT/'observations.json').read_text())
    assert (OUT/'prediction-seal.json').exists()
    fig,axes=plt.subplots(4,3,figsize=(16,12))
    for ax,row,observation in zip(axes.ravel(),predictions['rows'],observations):
        assert row['id']==observation['id']
        image=cv2.cvtColor(cv2.imread(observation['frames'][-1]['path']),cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        for line in row['matcher']['lines']:
            points=np.array(line['endpoints']);accepted=line['accepted']
            ax.plot(points[:,0],points[:,1],color='#08ff90' if accepted else '#ffb842',
                    linewidth=1.1 if accepted else .45,alpha=.95 if accepted else .38)
        count=sum(line['accepted'] for line in row['matcher']['lines'])
        ax.set_title(f'{row["id"]}: {count}/{row["matcher"]["candidate_count"]} accepted; {row["timing"]["algorithm_ms"]:.0f} ms',fontsize=10)
        ax.axis('off')
    fig.suptitle('All 12 fixed endpoints | Green = accepted distance, orange = UNKNOWN\nAcceptance is not a claim of correct native depth or a corridor alert',fontsize=13)
    fig.legend(handles=[Line2D([0],[0],color='#08ff90',label='Accepted interval'),
        Line2D([0],[0],color='#ffb842',label='Rejected / UNKNOWN')],loc='lower center',ncol=2)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'all-endpoint-contours.png',dpi=120);plt.close(fig)
    receipt={'selection':'All12 endpoints and all selected contours; no outcome or crop selection',
        'predictions_sha256':hashlib.sha256((OUT/'predictions.json').read_bytes()).hexdigest(),
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'visual-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('CONTOUR_VISUAL_COMPLETE')


if __name__=='__main__':
    main()
