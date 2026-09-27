import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cnh_h3_recording_diagnostics import load_recording, temporal_diagnostics, contiguous_runs


class DiagnosticsTest(unittest.TestCase):
    def test_parser_counts_all_lines_and_acquisition_order(self):
        f = dict(type='cnh_frame',hist_raw=np.ones((64,16),int).tolist(),hist_scaler=np.zeros((64,16),int).tolist(),
                 ambient_raw=[1]*64,ambient_scaler=[0]*64,distance_mm=[635]*64,target_status=[5]*64,nb_target=[1]*64)
        lines = [b'broken start', b'']
        for seq in (10,11,13,13,2):
            lines.append(json.dumps(dict(f,seq=seq,ms=seq*200)).encode())
            if seq==11: lines.append(b'{"type":"cnh_fra')
        lines += [b'{"type":"hello"}', b'broken end']
        root=Path(__file__).resolve().parents[4]/'artifacts.local'/'work'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root,prefix='h3-diagnostics-test-') as t:
            path=Path(t)/'raw.bin'; path.write_bytes(b'\n'.join(lines)+b'\n')
            d=load_recording(path)
        q=d['quality']
        self.assertEqual(q['total_lines'],10)
        self.assertEqual(q['bad_lines'],4)
        self.assertEqual(q['internal_bad_lines'],1)
        self.assertEqual(q['boundary_bad_lines'],3)
        self.assertEqual(q['missing_seq'],1)
        self.assertEqual(q['duplicate_seq'],1)
        self.assertEqual(q['resets_or_reorders'],1)
        self.assertEqual(d['seq'].tolist(),[10,11,13,13,2])

    def test_gap_blocks_not_joined(self):
        d=dict(seq=np.array([1,2,3,5,6,7,8,9]),ms=np.arange(8)*200,H=np.ones((8,64,16)))
        self.assertEqual([len(r) for r in contiguous_runs(d)[0]],[3,5])
        out=temporal_diagnostics(d)['groups']['target_bins1_4']['per_channel']
        self.assertEqual(out['four_frame']['blocks'],1)
        self.assertEqual(out['lag']['1']['pairs'],6)
        self.assertEqual(out['lag']['1']['distribution']['n'],0)

    def test_iid_ar_and_covariance_are_detected(self):
        rng=np.random.default_rng(284)
        h=rng.normal(size=(4096,64,16))
        # AR(1) target channels with substantial time correlation.
        for i in range(1,len(h)):
            h[i,:,1:5] += .7*h[i-1,:,1:5]
        # Anticorrelated adjacent bins: summing cancels most variance.
        h[:,:,9]=-h[:,:,8]+.1*rng.normal(size=(4096,64))
        d=dict(H=h,seq=np.arange(len(h)),ms=np.arange(len(h))*200)
        g=temporal_diagnostics(d)['groups']
        self.assertLess(abs(g['crosstalk_bin0']['per_channel']['lag']['1']['distribution']['mean']),.03)
        self.assertTrue(.9 < g['crosstalk_bin0']['per_channel']['four_frame']['distribution']['mean'] < 1.1)
        self.assertTrue(.65 < g['target_bins1_4']['per_channel']['lag']['1']['distribution']['mean'] < .75)
        self.assertGreater(g['target_bins1_4']['per_channel']['four_frame']['distribution']['mean'],2)
        self.assertLess(g['far_bins8_13']['window_variance_over_sum_bin_variances']['mean'],.75)

if __name__=='__main__': unittest.main()
