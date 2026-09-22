import copy
import unittest

import build_replay as builder


def fixture(clip,i,a=False,b=False,n=False,previous_a=False,previous_and=False):
    ident=clip+str(i);current=a or (b and n)
    flags=dict(A_current=a,B_control_current=b,N_current=n,A_hold=a or previous_a)
    unknown={k:True for k in flags}
    row=dict(id=ident,index=ord(clip)*10+i,clip_id=clip,frame_in_clip=i,time_s=round(i*.2,8),
        flags=flags,current_unknown=unknown,truth=False)
    prediction=copy.deepcopy(row)
    combination=dict(id=ident,index=row['index'],flags=dict(A_current=a,AND_current=current,A_hold=a or previous_a,AND_hold=current or previous_and),
        current_unknown={k:True for k in ('A_current','AND_current','A_hold','AND_hold')})
    return row,prediction,combination


class ReplayTests(unittest.TestCase):
    def test_interleaved_clips_reset_and_nonrecursive_hold(self):
        fixtures=[fixture('x',0,a=True),fixture('y',0),fixture('x',1,previous_a=True,previous_and=True),fixture('y',1),fixture('x',2)]
        result=builder.joined_rows(*[[f[i] for f in fixtures] for i in range(3)])
        self.assertEqual([r['display']['aHold'] for r in result],[True,True,False,False,False])
        self.assertTrue(result[0]['display']['unknown'])

    def test_exact_prediction_and_rgb_identity_rejected(self):
        row,p,c=fixture('x',0)
        p['index']+=1
        with self.assertRaises(ValueError):builder.joined_rows([row],[p],[c])
        entry=dict(sample_index=row['index'],clip_id='wrong',frame_in_clip=0,time_s=0.,id='x_00')
        with self.assertRaises(ValueError):builder.match_image(row,{row['index']:entry})

    def test_boolean_consensus_not_either_branch(self):
        fixtures=[fixture('x',0,b=True),fixture('x',1,n=True),fixture('x',2,b=True,n=True),fixture('x',3,previous_and=True)]
        result=builder.joined_rows(*[[f[i] for f in fixtures] for i in range(3)])
        self.assertEqual([r['display']['andCurrent'] for r in result],[False,False,True,False])
        self.assertEqual([r['display']['andHold'] for r in result],[False,False,True,True])

    def test_event_delay_separate_from_early_false_alert(self):
        frames=[dict(time=round(i*.2,8),truth=i in (2,3,4),unknown=True,aHold=i in (0,3,5,6)) for i in range(7)]
        m=builder.clip_metrics(frames,'aHold')
        self.assertEqual((m['TP'],m['FP'],m['FN']),(1,3,2))
        self.assertEqual((m['firstAlertTime'],m['eventEntry'],m['firstEventAlertTime'],m['delay']),(0.,.4,.6,.2))
        self.assertEqual((m['fpSegments'],m['fpSeconds']),(2,.6))
        self.assertEqual(m['fpIntervals'][1]['endExclusive'],1.4)
        self.assertEqual(m['unknownFrames'],7)


if __name__=='__main__':unittest.main()
