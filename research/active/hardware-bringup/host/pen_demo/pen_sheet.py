import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pen_load import *
from PIL import Image, ImageDraw, ImageFont
d=np.load('E:/linnan/linnan/artifacts.local/hardware-bringup/registration-20260928/pen_z.npy',allow_pickle=True).item()
cam=camera(); ct=np.array([c['host_received_monotonic_ns'] for c in cam],np.int64)
ii=np.arange(8); r,c=np.meshgrid(ii,ii,indexing='ij'); r=7-r; c=7-c; r,c=c,r; rr,cc=r.ravel(),c.ravel()   # orient 7
s,u0,v0=36.,136.,56.
tiles=[]
for sec in range(5,300,10):
    k=int(np.argmin(abs(d['t']-sec))); ci=int(np.argmin(abs(ct-d['host'][k])))
    im=Image.open(P/'camera'/cam[ci]['filename']).convert('RGB'); dr=ImageDraw.Draw(im,'RGBA')
    zz=d['z'][k,:,1]
    for zi in range(64):
        box=[u0+cc[zi]*s, v0+rr[zi]*s, u0+cc[zi]*s+s, v0+rr[zi]*s+s]
        if zz[zi]>6: dr.rectangle(box, fill=(255,60,30,110))
        dr.rectangle(box, outline=(80,220,255,90))
    dr.rectangle([0,0,200,26], fill=(0,0,0,170)); dr.text((6,6), f't={sec}s near zones={int((zz>6).sum())} max z={zz.max():.0f}', fill=(255,255,255))
    tiles.append(im.resize((320,240)))
W=6; Hn=(len(tiles)+W-1)//W; out=Image.new('RGB',(320*W,240*Hn))
for i,t in enumerate(tiles): out.paste(t,((i%W)*320,(i//W)*240))
out.save('E:/linnan/linnan/artifacts.local/hardware-bringup/registration-20260928/pen_sheet.jpg', quality=80)
print('ok', len(tiles))
