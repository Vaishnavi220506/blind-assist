import unittest
import cv2
import numpy as np
from mz148_background_residual import CausalResidual,align_pair


def texture():
    rng=np.random.default_rng(148016)
    gray=cv2.GaussianBlur(rng.random((180,320)).astype(np.float32),(0,0),2.)
    gray=(gray-gray.min())/(gray.max()-gray.min())
    return gray.astype(np.float32)


class MotionTests(unittest.TestCase):
    def test_background_translation_recovery(self):
        reference=texture()
        current=cv2.warpAffine(reference,np.float32([[1,0,3],[0,1,2]]),(320,180),borderMode=cv2.BORDER_REFLECT)
        warped,valid,a=align_pair(current,reference)
        self.assertTrue(a['valid'])
        valid[:8]=False;valid[-8:]=False;valid[:,:8]=False;valid[:,-8:]=False
        after=float(np.abs(current[valid]-warped[valid]).mean())
        before=float(np.abs(current[valid]-reference[valid]).mean())
        self.assertLess(after,before*.1)

    def test_prefix_causality_episode_gap_reset_and_metadata_invariance(self):
        base=texture();images=[cv2.cvtColor((np.roll(base,k,axis=1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR) for k in range(5)]
        rows=[dict(episode_id='a',time_s=k*.25) for k in range(5)]
        model=CausalResidual();all_values=[model.update(r,i)['compensated'] for r,i in zip(rows,images)]
        for n in (1,3,5):
            model=CausalResidual()
            for k in range(n):
                value=model.update(dict(rows[k],truth=True,native_bounds=[42],family='secret'),images[k])
                np.testing.assert_array_equal(value['compensated'],all_values[k])
        for row in [dict(episode_id='b',time_s=1.25),dict(episode_id='b',time_s=2.)]:
            value=model.update(row,images[0]);self.assertFalse(any(a['reference_available'] for a in value['audit']))


if __name__=='__main__':unittest.main()
