import unittest
from collections import Counter
from core_transfer_spec import specification, bounds, classify


class CoreSourceTest(unittest.TestCase):
    def test_full_extent_not_center(self):
        self.assertEqual(classify([.35,0,1],[.45,.1,1.1])['relation'],'OUTSIDE')
        self.assertEqual(classify([.2,0,1],[.6,.1,1.1])['relation'],'INSIDE')
        self.assertEqual(classify([.1,0,1],[.7,.1,1.1])['relation'],'INSIDE')
        self.assertEqual(classify([.3,0,1],[.7,.1,1.1])['relation'],'BOUNDARY')
        self.assertTrue(classify([.3,0,1],[.7,.1,1.1])['truth'])
        self.assertFalse(classify([.31,0,1],[.7,.1,1.1])['truth'])

    def test_budget_independence_and_entries(self):
        spec=specification()
        self.assertEqual(len(spec['cases']),432)
        self.assertEqual(len({c['arrangement_id'] for c in spec['clips']}),36)
        self.assertEqual(set(Counter(c['clip_id'] for c in spec['cases']).values()),{12})
        signatures=[]
        for clip in spec['clips']:
            rows=[c for c in spec['cases'] if c['clip_id']==clip['clip_id']]
            signatures.append(str((rows[0]['objects'],rows[0]['camera'])))
            self.assertEqual(classify(*bounds(rows[-1]))['relation'],clip['layout_relation'])
            self.assertEqual(classify(*bounds(rows[0]))['relation'],'OUTSIDE')
            self.assertFalse(classify(*bounds(rows[0]))['truth'])
            self.assertTrue(all(c['objects']==rows[0]['objects'] for c in rows))
            if clip['layout_relation']=='OUTSIDE':
                self.assertFalse(any(classify(*bounds(c))['truth'] for c in rows))
        self.assertEqual(len(set(signatures)),36)


if __name__=='__main__':unittest.main()
