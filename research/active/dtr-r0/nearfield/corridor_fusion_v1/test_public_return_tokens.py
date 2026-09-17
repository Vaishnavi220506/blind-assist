"""Authenticated full TRAIN192 token parity plus public input invariants."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from public_return_tokens import encode_tokens

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'artifacts.local/work'
OLD=WORK/'mz161-dense-task-20260916/run-v1'
CAP=WORK/'mz136-corridor-pair-20260914/source/returned-v1/capture-v1'
OUT=WORK/'corridor-public-positive-20260917/preparation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class PublicReturnTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        done=read(OLD/'completion.json'); assert done['status']=='PASS'
        assert sha(OLD/'prediction-seal.json')==done['prediction_seal_sha256']
        seal=read(OLD/'prediction-seal.json')
        assert sha(OLD/'input-seal.json')==seal['input_seal_sha256']
        inp=read(OLD/'input-seal.json')
        assert sha(OLD/'public-tokens.npz')==inp['tokens_sha256']
        assert sha(OLD/'freeze.json')==inp['freeze_sha256']
        frozen=read(OLD/'freeze.json')
        receipt=read(CAP/'receipt.json')
        assert receipt['status']=='PASS' and sha(CAP/'raw.jsonl')==receipt['hashes']['raw.jsonl']
        for p,h in frozen['inputs'].items():
            if Path(p).name in ('raw.jsonl','spec.json','receipt.json'):assert sha(Path(p))==h
        assert sha(CAP/'spec.json')==receipt['spec_sha256']
        ids=[r['id'] for r in read(OLD/'input-audit.json')]
        trainids={f['id'] for f in read(CAP/'spec.json')['frames'] if f['split']=='train'}
        assert len(ids)==len(set(ids))==192 and set(ids)==trainids
        rows={r['id']:r for r in map(json.loads,(CAP/'raw.jsonl').read_text().splitlines()) if r['id'] in trainids}
        cls.rows=[rows[i] for i in ids];cls.cache=np.load(OLD/'public-tokens.npz')

    def test_all_192_bitwise_parity(self):
        outside=merged=0
        for i,row in enumerate(self.rows):
            out=encode_tokens(row,float(self.cache['yaws'][i]))
            np.testing.assert_array_equal(out['tokens'],self.cache['tokens'][i])
            np.testing.assert_array_equal(out['valid'],self.cache['valid'][i])
            outside+=out['audit']['outside_or_partial_rgb_tof_tokens']
            merged+=int(out['tokens'][:128,1].sum())
        OUT.mkdir(parents=True,exist_ok=True)
        record=dict(status='PASS',frames=192,tokens_shape=[192,132,21],token_dtype='float32',valid_dtype='bool',
            tokens_bitwise_equal=True,valid_bitwise_equal=True,full_cached_yaws=True,
            outside_or_partial_rgb_tokens=outside,merged_tokens=merged,rgb_reads=0,evaluator_reads=0,
            source_sha256=sha(Path(__file__).with_name('public_return_tokens.py')),
            cache_sha256=sha(OLD/'public-tokens.npz'),raw_sha256=sha(CAP/'raw.jsonl'))
        (OUT/'token-parity-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

    def test_metadata_and_no_rgb_path_invariance(self):
        row=copy.deepcopy(self.rows[0]);baseline=encode_tokens(row,0.)
        row.pop('rgb_path',None)
        row.update(id='poison',family='poison',scene_group='poison',native_bounds=object(),zonal_tof_native=object())
        altered=encode_tokens(row,0.)
        np.testing.assert_array_equal(baseline['tokens'],altered['tokens'])
        np.testing.assert_array_equal(baseline['valid'],altered['valid'])

    def test_missing_packets_disable_all(self):
        row=copy.deepcopy(self.rows[0]);row['tof_packet_received']=row['radar_packet_received']=False
        out=encode_tokens(row,0.)
        self.assertFalse(out['valid'].any());self.assertFalse(out['tokens'][:,0].any())

    def test_absent_tof_slots_keep_fixed_shape(self):
        row=copy.deepcopy(self.rows[0])
        for z in row['tof_zones']:z['targets']=[]
        out=encode_tokens(row,0.)
        self.assertEqual(out['tokens'].shape,(132,21));self.assertFalse(out['valid'][:128].any())


if __name__=='__main__':unittest.main(verbosity=2)
