import unittest
from tolerance_eval import geometry,classify,temporal

def frame(objects,origin=(0,0,0)):return dict(body_origin_m=origin,native_bounds=objects)
def obj(y,width=.02,x=2.,z=1.,name='object'):
    return dict(name=name,center_m=[x,y,z],extent_m=[.1,width/2,.1])

class ToleranceContract(unittest.TestCase):
    def test_thin_central_rod_is_core_despite_small_overlap(self):
        g=geometry(frame([obj(0)]));self.assertAlmostEqual(g['lateral_margin_m'],.31)
        for t in [.03,.05,.1]:self.assertEqual(classify(g,t),'positive')
    def test_outer_contact_is_boundary_and_core_contact_positive(self):
        # Binary-exact values avoid conflating native float precision with convention.
        t=.05
        for y,expected in [(.2,'positive'),(.30,'boundary'),(.4,'negative')]:
            self.assertEqual(classify(geometry(frame([obj(y,.02)])),t),expected)
        self.assertEqual(classify(dict(lateral_margin_m=t),t),'positive')
        self.assertEqual(classify(dict(lateral_margin_m=-t),t),'boundary')
    def test_all_objects_and_original_xz_range(self):
        self.assertEqual(classify(geometry(frame([obj(.7),obj(0,name='other') ])),.05),'positive')
        for o in [obj(0,x=5),obj(0,z=3)]:self.assertEqual(classify(geometry(frame([o])),.05),'negative')
        self.assertFalse(geometry(frame([]))['strict'])
    def test_mirror_translation_and_monotone_coverage(self):
        a=geometry(frame([obj(.275)]));b=geometry(frame([obj(-.275)]));self.assertEqual(a['lateral_margin_m'],b['lateral_margin_m'])
        o=obj(.275);o['center_m']=[3.,1.275,2.]
        self.assertAlmostEqual(a['lateral_margin_m'],geometry(frame([o],(1,1,1)))['lateral_margin_m'])
        self.assertEqual([classify(a,t) for t in [.01,.03,.05]],['positive','positive','boundary'])
    def test_temporal_keeps_episodes_and_negative_burden(self):
        r=[dict(episode_id='e',time_s=i*.25) for i in range(6)]
        s=['negative','boundary','positive','positive','negative','negative'];p=[1,1,0,1,1,0]
        v=temporal(r,s,p);self.assertEqual(v['core_events_detected'],1)
        self.assertEqual(v['events'][0]['first_in_core_delay_s'],.25)
        self.assertEqual(v['events'][0]['first_boundary_or_core_alert_offset_s'],-.25)
        self.assertEqual(v['clear_negative_alert_frames'],2)
        self.assertEqual(v['release'][0]['first_off_delay_s'],.25)

if __name__=='__main__':unittest.main()
