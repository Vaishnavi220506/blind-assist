import unittest
from unittest.mock import patch

import active_view as av
import witness_guided_path as guide


class GuideTests(unittest.TestCase):
    def test_path_budget(self):
        for action in range(4):
            path=guide.poses(action)
            self.assertEqual(len(path),13)
            self.assertEqual(path[0],(0.,0.))
            self.assertEqual(path[-1],av.ACTIONS[action])
            self.assertAlmostEqual(sum(sum(abs(a-b) for a,b in zip(p,q)) for p,q in zip(path,path[1:])),.12)

    def test_missing_pair_is_not_a_claim_of_no_object(self):
        self.assertEqual(guide.choose((1,)*8,{'witnesses':{'IN':None,'OUT':None}})['action'],0)

    def test_choices_use_only_hypothetical_witnesses(self):
        pair={'IN':{'boxes':[{'x':0.,'z':1.,'width':.1,'thickness':.04}],'wall_z':4.2},
              'OUT':{'boxes':[{'x':.6,'z':1.,'width':.1,'thickness':.04}],'wall_z':4.2}}
        # Controlled public forecasts: early split only on negative-X path.
        def forecast(scene,pose):
            return (int(scene.boxes[0].x>.3 and pose[0]<-.02),)*8
        with patch.object(av,'observe',side_effect=forecast),patch.object(av,'intersects_query',side_effect=AssertionError('truth')):
            result=guide.choose((0,)*8,{'witnesses':pair})
        self.assertEqual(result['action'],1)
        with patch.object(av,'observe',return_value=(1,)*8):
            with self.assertRaises(ValueError):
                guide.choose((0,)*8,{'witnesses':pair})


if __name__=='__main__':
    unittest.main()
