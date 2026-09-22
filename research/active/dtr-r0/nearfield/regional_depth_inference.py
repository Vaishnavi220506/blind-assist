"""Observation-only local-depth inference, independent of source/evaluator files."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from regional_depth_data import build_observation
from regional_depth_model import Head, RECIPE, alert_score, depths


def load(checkpoint, device='cpu'):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['recipe']!=RECIPE:
        raise ValueError('Checkpoint does not match frozen recipe')
    head=Head(); head.load_state_dict(payload['state_dict'])
    return head.to(device).eval()


@torch.inference_mode()
def predict(head, rgb, ranges, boxes):
    x=torch.from_numpy(build_observation(rgb,ranges,boxes)[None]).to(next(head.parameters()).device)
    logits=head.distribution_logits(x); score,local=alert_score(logits,x)
    expected=(logits.softmax(1)*depths(logits.device)[None,:,None,None]).sum(1)
    return dict(score=float(score[0].cpu()),corridor_probability=local[0].cpu().numpy(),
                expected_depth=expected[0].cpu().numpy())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Raw supplement score; combine with A and frozen hold separately.')
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--rgb',required=True)
    parser.add_argument('--ranges',required=True,help='Public 64-vector .npy')
    parser.add_argument('--boxes',required=True,help='Public integer [64,4] .npy')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with Image.open(args.rgb) as image:
        result=predict(load(args.checkpoint,args.device),np.asarray(image.convert('RGB')),
                       np.load(args.ranges,allow_pickle=False),np.load(args.boxes,allow_pickle=False))
    with output.open('xb') as stream:
        np.savez_compressed(stream,**result)
    print(json.dumps(dict(score=result['score'],meaning='Model prediction, not observed depth or free-space certificate')))
