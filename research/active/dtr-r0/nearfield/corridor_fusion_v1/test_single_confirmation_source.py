import json,hashlib,unittest
from pathlib import Path
from collections import defaultdict,Counter
from single_confirmation_source import source,check_source
ROOT=Path(__file__).resolve().parents[5]
def signatures(spec):
    episodes=defaultdict(list)
    for f in spec['frames']:episodes[f['episode']].append({k:f[k] for k in ['camera','body_origin_m','objects','time_s']})
    return {hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for v in episodes.values()}
def admit_geometry(spec):
    states=[]
    for f in spec['frames']:
        def hits(o,width):
            lo=[o['center_m'][a]-o['size_m'][a]/2-f['body_origin_m'][a] for a in range(3)]
            hi=[o['center_m'][a]+o['size_m'][a]/2-f['body_origin_m'][a] for a in range(3)]
            return all(hi[a]>=v for a,v in enumerate([.2,-width,.4])) and all(lo[a]<=v for a,v in enumerate([3.6,width,2.05]))
        states.append('positive' if any(hits(o,.25) for o in f['objects']) else 'boundary' if any(hits(o,.35) for o in f['objects']) else 'negative')
    assert Counter(states)==dict(positive=108,negative=108,boundary=72)
    sig=signatures(spec);assert len(sig)==48
    roots=['mz136-corridor-pair-20260914','mz146-fresh-corridor-confirmation-20260916','mz158-crossview-agreement-20260916','mz170-mean-confirmation-20260916','corridor-depth-confirmation-recovery-20260917']
    for r in roots:
        prior=json.loads((ROOT/'artifacts.local/work'/r/'source/returned-v1/capture-v1/spec.json').read_text())
        assert not sig&signatures(prior),r
    return dict(clear_frames=216,boundary_frames=72,complete_trajectory_signature_overlap=0,prior_sources_checked=roots)
class SourceTest(unittest.TestCase):
    def test_complete_geometry_and_unique(self):
        s=source();check_source(s);admit_geometry(s)
    def test_deterministic(self):self.assertEqual(source(),source())
if __name__=='__main__':unittest.main(verbosity=2)
