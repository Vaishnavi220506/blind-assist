"""Rough camera-ToF registration from a moving foreground object (bring-up aid, not calibration).

Pairs each 8x8 CNH frame with the nearest camera frame (host receive time), marks ToF zones whose
histogram changes >30% vs a static background and camera pixels that change, then grid-searches
scale, offset and the 8 grid orientations for the best zone-vs-foreground correlation.
"""
import json, glob, numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
R = Path('E:/linnan/linnan/artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e')
def read(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
def tof(seg):
    out=[]
    for x in read(R/seg/'tof/frames.jsonl'):
        s=x['sensor']
        if len(s.get('hist_raw',[]))!=64: continue
        H=np.array(s['hist_raw'],float)/2.0**np.array(s['hist_scaler'],float)
        out.append((x['host_received_monotonic_ns'], H))
    return out
def cam(seg):
    c=read(R/seg/'camera/frames.jsonl')
    return np.array([x['host_received_monotonic_ns'] for x in c]), [R/seg/'camera'/x['filename'] for x in c]
def gray(p, W=160):
    im=Image.open(p).convert('L').resize((W, W*3//4)).filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(im, float)
# background: segment 01
bgH=np.median(np.array([h for _,h in tof('01-background-attempt1')]),0)
ct, cp = cam('01-background-attempt1')
bgI=np.median(np.array([gray(p) for p in cp[::20]]),0)
rows=[]
for seg in ('02-object-attempt2','03-movement-attempt3'):
    T=tof(seg); ct,cp=cam(seg)
    for t,H in T[::2]:
        ci=int(np.argmin(abs(ct-t)))
        if abs(ct[ci]-t)>150e6: continue
        dz=np.abs(H-bgH).sum(1)/np.maximum(np.abs(bgH).sum(1),1e-9)      # per-zone relative CNH change
        fg=np.abs(gray(cp[ci])-bgI)
        rows.append((dz, fg))
DZ=np.array([r[0] for r in rows]); FG=np.array([r[1] for r in rows])
print('pairs', len(rows), 'image', FG.shape[1:], 'zone change median/p90', np.median(DZ), np.percentile(DZ,90))
A=(DZ>0.3).astype(float)                      # ToF active zones
M=(FG>25).astype(float)                       # camera foreground
Hh,Ww=M.shape[1:]
ii=np.arange(8)
def orient(k):
    r,c=np.meshgrid(ii,ii,indexing='ij')
    if k&1: r=7-r
    if k&2: c=7-c
    if k&4: r,c=c,r
    return r.ravel(), c.ravel()       # image row/col index of each ToF zone z
cum=np.pad(M.cumsum(1).cumsum(2),((0,0),(1,0),(1,0)))
best=[]
for k in range(8):
    rr,cc=orient(k)
    for s in np.arange(6,22,1.0):             # pixels per zone in the 160x120 image
        for v0 in np.arange(-10,Hh-8*s+10,2):
            for u0 in np.arange(-10,Ww-8*s+10,2):
                y0=np.clip((v0+rr*s).astype(int),0,Hh); y1=np.clip((v0+(rr+1)*s).astype(int),0,Hh)
                x0=np.clip((u0+cc*s).astype(int),0,Ww); x1=np.clip((u0+(cc+1)*s).astype(int),0,Ww)
                area=np.maximum((y1-y0)*(x1-x0),1)
                if (area<s*s*0.5).sum()>8: continue
                frac=(cum[:,y1,x1]-cum[:,y0,x1]-cum[:,y1,x0]+cum[:,y0,x0])/area
                a=A.ravel(); f=frac.ravel()
                corr=np.corrcoef(a,f)[0,1]
                best.append((corr,k,s,u0,v0))
best.sort(reverse=True)
for b in best[:8]: print('corr %.3f orient %d scale %.0f px/zone (160px) u0 %.0f v0 %.0f'%b)
np.save('C:/Users/26442/AppData/Local/Temp/claude/E--linnan-linnan/113c7040-5e73-4bb3-b2d5-100826c3841d/scratchpad/reg_best.npy', np.array(best[:50]))
