"""Uniform resident public-input A + one positive head; no evaluator dependency."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch
from public_positive import PositiveHead,spatial_features
from public_return_tokens import encode_tokens

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mz143_corridor_features import extract


class SinglePositiveSystem:
    def __init__(self,bundle):
        bundle=Path(bundle)
        config=json.loads((bundle/'config.json').read_text(encoding='utf-8'))
        for name,h in config['model_hashes'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=h:
                raise ValueError('Model identity mismatch: '+name)
        self.a=pickle.loads((bundle/'A.pkl').read_bytes())
        self.control=pickle.loads((bundle/'A-retrained.pkl').read_bytes())
        self.head=PositiveHead()
        self.head.load_state_dict(torch.load(bundle/'positive.pt',map_location='cpu',weights_only=True)['state_dict'])
        self.head.eval().requires_grad_(False)
        self.a_tau=config['A_threshold']
        self.positive_tau=config['positive_threshold']
        self.control_tau=config['control_threshold']

    @torch.inference_mode()
    def predict(self,row,rgb,yaw):
        start=time.perf_counter()
        f=extract(row,rgb,yaw)
        base=np.r_[f['sensor'],f['geometry']]
        a_score=float(self.a.predict_proba(base[None])[0,1])
        a_alert=a_score>=self.a_tau
        a_ms=(time.perf_counter()-start)*1000
        head_start=time.perf_counter()
        token=encode_tokens(row,yaw)
        valid=token['valid'][:128]
        x=spatial_features(token['tokens'])[None]
        logits=self.head(torch.from_numpy(x),torch.from_numpy(valid[None]))[0].numpy()
        scores=np.where(valid,logits,-30.)
        slot=int(scores.argmax()) if valid.any() else None
        score=float(scores.max())
        positive=bool(valid.any() and score>=self.positive_tau)
        head_ms=(time.perf_counter()-head_start)*1000
        control_start=time.perf_counter()
        control_score=float(self.control.predict_proba(base[None])[0,1])
        control_ms=(time.perf_counter()-control_start)*1000
        return dict(A_score=a_score,A=bool(a_alert),positive_score=score,
            positive=positive,alert=bool(a_alert or positive),max_return_slot=slot,
            usable_tof_returns=int(valid.sum()),state='RETURN_PRESENT' if valid.any() else 'UNKNOWN',
            control_score=control_score,control=bool(control_score>=self.control_tau),
            A_algorithm_ms=a_ms,head_increment_ms=head_ms,control_head_ms=control_ms)
