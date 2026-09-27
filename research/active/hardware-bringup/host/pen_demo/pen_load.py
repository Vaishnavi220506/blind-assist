import json, numpy as np
from pathlib import Path
P=Path('E:/linnan/linnan/artifacts.local/hardware-bringup/previews/preview-b46be2db/data')
C=Path('E:/linnan/linnan/artifacts.local/hardware-bringup/captures/live-small-1790532816778')
def tof(path):
    fr=[]
    for l in open(path, encoding='utf-8', errors='replace'):
        try: x=json.loads(l)
        except Exception: continue
        s=x.get('sensor',{})
        if s.get('type')!='cnh_frame' or len(s.get('hist_raw',[]))!=64: continue
        fr.append(x)
    fr.sort(key=lambda x:x['sensor']['seq'])
    H=np.array([np.array(x['sensor']['hist_raw'],float)/2.0**np.array(x['sensor']['hist_scaler'],float) for x in fr])
    A=np.array([np.array(x['sensor']['ambient_raw'],float)/2.0**np.array(x['sensor']['ambient_scaler'],float) for x in fr])
    D=np.array([x['sensor']['distance_mm'] for x in fr],float); S=np.array([x['sensor']['target_status'] for x in fr])
    t=np.array([x['host_received_monotonic_ns'] for x in fr],np.int64)
    return dict(H=H,A=A,D=D,S=S,t=t)
def camera():
    c=[json.loads(l) for l in open(P/'camera/frames.jsonl',encoding='utf-8')]
    return c
