"""Check paired intervention isolation and causal alert readout before freeze."""
import copy
import unittest
from unittest.mock import patch
from appearance_probe import ROOT, read, make_spec, held, predict


class AppearanceTests(unittest.TestCase):
    def test_only_named_appearance_changes_on_complete_training_clips(self):
        source=read(ROOT/'artifacts.local/work/ba-data-coverage-20260921/spec.json')
        spec=make_spec(source)
        self.assertEqual(len(spec['cases']),1152)
        self.assertEqual(len({r['clip_id'] for r in spec['cases']}),48)
        refs={}
        for row in spec['cases']:
            self.assertEqual(row['split'],'train')
            key=row['source_id']
            value=copy.deepcopy(row)
            for k in ('name','clip_id','condition'):value.pop(k)
            if row['condition']=='reference':refs[key]=value
            expected=copy.deepcopy(refs[key])
            if row['condition'] in ('target','background'):
                index=next(i for i,o in enumerate(value['objects']) if o['name']==row['condition'])
                self.assertNotEqual(value['objects'][index]['material'],expected['objects'][index]['material'])
                value['objects'][index]['material']=expected['objects'][index]['material']
            self.assertEqual(value,expected)

    def test_hold_is_nonrecursive_and_does_not_cross_clips(self):
        ids=[dict(clip_id='a' if i<4 else 'b',frame_in_clip=i if i<4 else 0) for i in range(5)]
        self.assertEqual(held([True,False,False,True,False],ids),[True,True,False,True,False])

    def test_truth_labels_cannot_change_selection(self):
        source=read(ROOT/'artifacts.local/work/ba-data-coverage-20260921/spec.json')
        changed=copy.deepcopy(source)
        for g in changed['groups']:g['truth']=False;g['model_score']=999
        self.assertEqual(make_spec(source),make_spec(changed))

    def test_failed_pair_admission_stops_before_seal_or_model_loading(self):
        with patch('appearance_probe.verify'), patch('appearance_probe.read', return_value={'status':'NOT_EVALUABLE'}), patch('appearance_probe.check_seal') as seal:
            with self.assertRaisesRegex(ValueError, 'inference is not admitted'):
                predict(ROOT/'unused-fixture')
            seal.assert_not_called()


if __name__=='__main__':unittest.main()
