import unittest
import numpy as np
import torch
from prepare_public_positive import WORK,read
from single_positive_inference import SinglePositiveSystem
from public_positive import spatial_features
from public_return_tokens import encode_tokens


class SingleVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.home=WORK/'corridor-public-single-20260917'
        cls.system=SinglePositiveSystem(cls.home/'bundle')
        cls.row=__import__('json').loads((WORK/'mz170-mean-confirmation-20260916/source/returned-v1/capture-v1/raw.jsonl').read_text().splitlines()[0])

    def test_unique_model_and_finite_thresholds(self):
        config=read(self.home/'bundle/config.json')
        self.assertEqual(sum(p.numel() for p in self.system.head.parameters()),1537)
        self.assertTrue(config['no_scene_router'])
        self.assertTrue(all(np.isfinite(config[k]) for k in ['A_threshold','positive_threshold','control_threshold']))

    def test_inference_uses_frozen_full_fit(self):
        raw=encode_tokens(self.row,0.)
        x=spatial_features(raw['tokens'])[None];v=raw['valid'][None,:128]
        trained=torch.load(self.home/'head/model.pt',weights_only=True,map_location='cpu')['state_dict']
        for key,value in self.system.head.state_dict().items():
            self.assertTrue(torch.equal(value,trained[key]))
        with torch.inference_mode():
            z=self.system.head(torch.from_numpy(x),torch.from_numpy(v))
        self.assertTrue(torch.isfinite(z).all())


if __name__=='__main__':unittest.main()
