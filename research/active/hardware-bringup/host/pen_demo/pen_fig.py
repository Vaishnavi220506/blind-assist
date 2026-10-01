import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pen_load import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
d=np.load('E:/linnan/linnan/artifacts.local/hardware-bringup/registration-20260928/pen_z.npy',allow_pickle=True).item()
main=tof(P/'tof/frames.jsonl'); bg=np.median(np.concatenate([tof(C/'01-background/tof/frames.jsonl')['H'],tof(C/'04-restored-background/tof/frames.jsonl')['H']]),0)
t,z=d['t'],d['z']; near=z[:,:,1]
persistent=28-8  # placeholder
det=near>6
# exclude the unexplained persistent zone: find zone lit >50% of frames in 0-150 s
early=(t<150); pz=np.flatnonzero(det[early].mean(0)>0.5); print('persistent zones (lit >50% in 0-150 s):', pz)
detc=det.copy(); detc[:,pz]=False
empty=((t>=200)&(t<210))|((t>=241)&(t<260))
obj=~empty & (t>2) & ~(((t>=195)&(t<200))|((t>=210)&(t<213))|((t>=238)&(t<241))|((t>=260)&(t<263)))
fw=(d['D']>0)&(d['D']<550)&np.isin(d['S'],(5,6,9))
print('empty frames %d: frames with >=1 near zone (excl. persistent) %.3f ; firmware near %.3f'%(empty.sum(), detc[empty].any(1).mean(), fw[empty].any(1).mean()))
print('object-period frames %d: frames with >=1 near zone %.3f ; firmware near (lenient) %.3f'%(obj.sum(), detc[obj].any(1).mean(), fw[obj].any(1).mean()))
# demo figure: three moments
cam=camera(); ct=np.array([c['host_received_monotonic_ns'] for c in cam],np.int64)
ii=np.arange(8); r,c=np.meshgrid(ii,ii,indexing='ij'); r=7-r; c=7-c; r,c=c,r; rr,cc=r.ravel(),c.ravel()
s,u0,v0=36.,136.,56.
fig,ax=plt.subplots(2,3,figsize=(15,8.5))
for j,sec in enumerate((105,215,275)):
    k=int(np.argmin(abs(t-sec))); ci=int(np.argmin(abs(ct-d['host'][k])))
    im=Image.open(P/'camera'/cam[ci]['filename']).convert('RGB'); dr=ImageDraw.Draw(im,'RGBA')
    zz=near[k].copy(); zz[pz]=0
    for zi in range(64):
        box=[u0+cc[zi]*s, v0+rr[zi]*s, u0+cc[zi]*s+s, v0+rr[zi]*s+s]
        if zz[zi]>6: dr.rectangle(box, fill=(255,60,30,110))
        dr.rectangle(box, outline=(80,220,255,90))
    zbest=int(np.argmax(zz))
    x0,y0=u0+cc[zbest]*s, v0+rr[zbest]*s; dr.rectangle([x0,y0,x0+s,y0+s], outline=(255,255,0,255), width=3)
    ax[0,j].imshow(im); ax[0,j].axis('off')
    fwd=main['D'][k,zbest]; fws=main['S'][k,zbest]
    ax[0,j].set_title(f't={sec}s  red: histogram near-object zones (0.3-0.6 m)\nyellow zone firmware: {fwd:.0f} mm (status {fws})', fontsize=9)
    bins=np.arange(16)*0.3+0.15
    ax[1,j].plot(bins[:8], bg[zbest,:8], 'o-', color='#e69a2e', label='background (cabinet only)')
    ax[1,j].plot(bins[:8], main['H'][k,zbest,:8], 'o-', color='#2e86de', label='current frame')
    ax[1,j].axvspan(0.3,0.6,color='r',alpha=0.08); ax[1,j].set_xlabel('distance (m), 0.3 m bins'); ax[1,j].set_ylabel('CNH value')
    ax[1,j].set_title(f'yellow zone histogram: near-bin z = {zz[zbest]:.0f}', fontsize=9); ax[1,j].legend(fontsize=8)
plt.tight_layout(); out=r'E:/linnan/linnan/artifacts.local/hardware-bringup/registration-20260928/pen_histogram_vs_firmware.png'; plt.savefig(out, dpi=110); print(out)
