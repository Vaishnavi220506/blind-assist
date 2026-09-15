import unittest
import numpy as np
from mz143_corridor_features import SLOT_NAMES
from mz152_contained_return import anchors,predict


class ContainedReturnTests(unittest.TestCase):
    def test_full_envelope_and_status_required_without_rgb(self):
        names=['global.imu_valid']+[f'zone{z:02d}.slot{s}.{f}'
            for z in range(64) for s in range(2) for f in SLOT_NAMES]
        x=np.zeros((5,len(names)));x[:,0]=1.
        for field,value in dict(usable=1.,support_x_lo=1.,support_x_hi=2.,
            support_y_lo=-.2,support_y_hi=.2,support_z_lo=.5,support_z_hi=1.8).items():
            x[:,names.index('zone28.slot0.'+field)]=value
        x[1,names.index('zone28.slot0.support_y_hi')]=.31
        x[2,names.index('zone28.slot0.merged')]=1.
        x[3,0]=0.;x[4,names.index('zone28.slot0.usable')]=0.
        np.testing.assert_array_equal(anchors(x,names)[0],[True,False,False,False,False])

    def test_anchor_is_additive_without_clearing_learned_alert(self):
        rows=[dict(episode_id='a',time_s=0.),dict(episode_id='b',time_s=0.)]
        np.testing.assert_array_equal(predict(rows,[0.,.9],[True,False],.2,.8),[True,True])


if __name__=='__main__':unittest.main()
