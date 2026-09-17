"""Standalone frozen A*: public RGB/ToF/Radar features and one HGB only."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mz143_corridor_features import extract


class AStarSystem:
    """`alert` is the sole A* decision for storage, UI and reminder consumers."""

    def __init__(self, bundle):
        bundle = Path(bundle)
        self.config = json.loads((bundle/'config.json').read_text(encoding='utf-8'))
        if self.config['schema'] != 'BLINDASSIST_STANDALONE_ASTAR_V1':
            raise ValueError('Expected the standalone A* bundle')
        model_path = bundle/self.config['model_file']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.config['model_sha256']:
            raise ValueError('A* checkpoint identity mismatch')
        self.model = pickle.loads(model_path.read_bytes())
        self.threshold = float(self.config['threshold'])
        if not np.isfinite(self.threshold):
            raise ValueError('Finite frozen threshold required')
        if self.model.n_features_in_ != 2485:
            raise ValueError('Expected the frozen 2485-column feature interface')

    def predict(self, row, rgb, yaw):
        started = time.perf_counter()
        features = extract(row, rgb, yaw)
        vector = np.r_[features['sensor'], features['geometry']]
        score = float(self.model.predict_proba(vector[None])[0, 1])
        alert = bool(score >= self.threshold)
        return dict(model_id=self.config['model_id'], score=score,
            threshold=self.threshold, alert=alert,
            algorithm_ms=(time.perf_counter()-started)*1000)
