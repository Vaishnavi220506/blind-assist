import unittest
import numpy as np
from specialist_gate import policy,features


class GateContract(unittest.TestCase):
    def test_only_veto_and_failed_specialist_keeps_a(self):
        result,invoked,veto=policy([0,1,1,1],[1,1,0,1],.5,[0,0,0,np.nan],.7)
        np.testing.assert_array_equal(result,[0,0,1,1])
        np.testing.assert_array_equal(invoked,[0,1,0,1])
        np.testing.assert_array_equal(veto,[0,1,0,0])

    def test_metadata_and_expensive_information_not_read(self):
        row=dict(tof_zones=[],tof_packet_received=False,radar_packet_received=True,imu_valid=True)
        a=features(row,np.array([2,3]),.6,[0,1])
        row.update(family='rod',truth=True,scene_id='secret',C_score=.99,depth=[99])
        np.testing.assert_array_equal(a,features(row,np.array([2,3]),.6,[0,1]))

if __name__=='__main__':unittest.main()
