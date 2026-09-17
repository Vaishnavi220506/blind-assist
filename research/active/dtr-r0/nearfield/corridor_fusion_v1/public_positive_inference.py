"""Public-only frozen artifact inference; no evaluator or dataset import."""
from pathlib import Path
import math
import numpy as np
import torch
from public_return_tokens import encode_tokens
from public_positive import PositiveHead, spatial_features


class PublicPositivePredictor:
    """A single caller-selected research checkpoint, never a scene-ID router.

    Caller retains A and the causal public IMU yaw integration. The threshold
    must accompany the checkpoint; six-fold evaluation is not one deployed head.
    """
    def __init__(self, checkpoint, threshold, device='cpu'):
        if not math.isfinite(threshold):
            raise ValueError('Finite artifact threshold required')
        self.threshold = float(threshold)
        self.device = device
        self.model = PositiveHead().to(device)
        self.model.load_state_dict(torch.load(Path(checkpoint), map_location=device,
                                              weights_only=True)['state_dict'])
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, row, yaw, a_alert):
        public = encode_tokens(row, yaw)
        valid = public['valid'][:128]
        features = spatial_features(public['tokens'])[None]
        logits = self.model(torch.from_numpy(features).to(self.device),
                            torch.from_numpy(valid[None]).to(self.device))[0].cpu().numpy()
        score = float(np.where(valid, logits, -30.).max())
        branch = bool(valid.any() and score >= self.threshold)
        return dict(alert=bool(a_alert or branch), positive_evidence=branch,
                    evidence_logit=score, usable_tof_returns=int(valid.sum()),
                    source_state='RETURN_PRESENT' if valid.any() else 'UNKNOWN')
