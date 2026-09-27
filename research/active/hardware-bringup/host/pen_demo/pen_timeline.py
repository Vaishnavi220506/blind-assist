import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pen_load import *
from PIL import Image, ImageDraw
main=tof(P/'tof/frames.jsonl'); bg1=tof(C/'01-background/tof/frames.jsonl'); bg4=tof(C/'04-restored-background/tof/frames.jsonl')
Hbg=np.median(np.concatenate([bg1['H'],bg4['H']]),0)
H,A=main['H'],main['A']
v=0.07*np.clip(H,0,None)+0.10*A[:,:,None]+0.02
z=(H-Hbg)/np.sqrt(v)
t=(main['t']-main['t'][0])/1e9
near=z[:,:,1]                       # 0.3-0.6 m
cnt=(near>6).sum(1); mx=near.max(1)
fwnear_strict=((main['D']>0)&(main['D']<550)&(main['S']==5)).sum(1)
fwnear_len=((main['D']>0)&(main['D']<550)&np.isin(main['S'],(5,6,9))).sum(1)
print('frames',len(t),'duration %.0f s'%t[-1])
for s in range(0,int(t[-1])+1,10):
    m=(t>=s)&(t<s+10)
    print('%3d-%3d s  zones z_bin1>6: median %4.1f max %2d | max z %6.1f | firmware near zones strict %4.1f lenient %4.1f'%(s,s+10,np.median(cnt[m]),cnt[m].max(),np.median(mx[m]),np.median(fwnear_strict[m]),np.median(fwnear_len[m])))
np.save('E:/linnan/linnan/artifacts.local/hardware-bringup/registration-20260928/pen_z.npy', dict(t=t,z=z,D=main['D'],S=main['S'],host=main['t']), allow_pickle=True)
# contact sheet: one camera frame every 15 s
cam=camera(); key='host_received_monotonic_ns' if 'host_received_monotonic_ns' in cam[0] else None
print('camera keys', list(cam[0].keys()))
