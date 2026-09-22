import json
import unittest
from cnh_replay import nearest_camera, saved_scores, safe_json


class ReplayTests(unittest.TestCase):
    def test_camera_pairing_has_bound_and_earlier_tie(self):
        cameras=[{'host_ns':100},{'host_ns':200}]
        self.assertEqual(nearest_camera(cameras,[100,200],150,60)['host_ns'],100)
        self.assertEqual(nearest_camera(cameras,[100,200],151,60)['host_ns'],200)
        self.assertIsNone(nearest_camera(cameras,[100,200],1000,60))
        self.assertIsNone(nearest_camera([],[],100))

    def test_saved_values_and_absence_survive_without_scoring(self):
        zones=[{'zone':z} for z in range(16)]
        value={'seq':7,'score':-2,'late':False,'scalar_known':False}
        zones[2]['phases']={'mixture':{'frames':[value]}}
        scores=saved_scores({'zones':zones})
        self.assertIs(scores[(2,'mixture',7)],value)
        self.assertIsNone(scores.get((2,'foreground',7)))
        zones[2]['phases']['mixture']['frames'].append(value)
        with self.assertRaises(ValueError):
            saved_scores({'zones':zones})

    def test_embedded_data_cannot_close_script(self):
        value={'text':'</script><script>alert(1)</script>\u2028中文'}
        encoded=safe_json(value)
        self.assertNotIn('<',encoded)
        self.assertEqual(json.loads(encoded),value)
        with self.assertRaises(ValueError):
            safe_json({'bad':float('nan')})


if __name__=='__main__':
    unittest.main()
