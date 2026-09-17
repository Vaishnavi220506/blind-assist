"""Run one resident bundle on any public JSONL capture, without evaluator files."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from threadpoolctl import threadpool_limits
from single_positive_inference import SinglePositiveSystem


def main(bundle,raw,rgb_root,output):
    if output.exists():raise FileExistsError('Do not overwrite predictions: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    system=SinglePositiveSystem(bundle)
    episode=None;yaw=0.;frames=0
    with raw.open(encoding='utf-8-sig') as source,output.open('x',encoding='utf-8') as dest:
        for line in source:
            if not line.strip():continue
            row=json.loads(line)
            if row['episode_id']!=episode:yaw=0.
            if row['imu_valid']:yaw+=row['delta_yaw']
            episode=row['episode_id']
            image=cv2.imread(str(rgb_root/row['rgb_path']))
            if image is None:raise ValueError('RGB missing: '+row['id'])
            result=system.predict(row,image,yaw)
            result.update(id=row['id'],episode_id=episode,time_s=row['time_s'])
            dest.write(json.dumps(result,allow_nan=False)+'\n');frames+=1
    print(json.dumps(dict(status='PASS',frames=frames,output=str(output),models_loaded_once=True,evaluator_read=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(4):main(a.bundle,a.raw,a.rgb_root,a.output)
