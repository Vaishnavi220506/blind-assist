import unittest

import torch

from mz174_return_graph import ReturnGraph,build_graph


def inputs(count=12):
    t = torch.zeros(1,132,21)
    v = torch.zeros(1,132,dtype=torch.bool)
    v[:,:count] = True
    t[:,:count,0] = 1
    t[:,:count,16] = 1
    t[:,:count,2] = torch.linspace(.3,.9,count)
    t[:,:count,4] = torch.linspace(.1,1.,count)
    # Distinct nonsymmetric centers avoid nearest-distance ties.
    t[:,:count,5:8] = torch.rand(1,count,3)
    t[:,:count,8:14] = torch.randn(1,count,6)
    return t,v


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(174016)

    def test_metric_neighbor_order_and_actual_edge_descriptors(self):
        t,v = inputs(3)
        v[0,128] = True
        t[0,128,0] = t[0,128,17] = 1
        t[0,0,5:8] = torch.tensor([1/4,0.,1/3])
        t[0,1,5:8] = torch.tensor([1.8/4,0.,1/3])
        t[0,2,5:8] = torch.tensor([1/4,.6/2,1/3])
        t[0,128,5:8] = torch.tensor([1/4,0.,0.])
        t[0,0,2] = .5; t[0,2,2] = .8
        graph = build_graph(t,v)
        # Normalized-coordinate distance would incorrectly prefer node1.
        self.assertEqual(graph['neighbor_indices'][0,0,:3].tolist(),[2,1,128])
        torch.testing.assert_close(graph['relative_descriptors'][0,0,0],torch.tensor([0.,.6,0.,1.2,1.]))
        self.assertEqual(float(graph['relative_descriptors'][0,0,2,-1]),0.)
        self.assertEqual(int(graph['neighbor_count'][0,0]),3)
        self.assertTrue(torch.all(graph['neighbor_indices'][~graph['neighbor_mask']]==-1))

    def test_natural_node_permutation_equivariance(self):
        t,v = inputs(17)
        t[0,3,16:18] = torch.tensor([0.,1.])
        model = ReturnGraph('natural').eval()
        permutation = torch.randperm(132)
        before = model(t,v)
        after = model(t[:,permutation],v[:,permutation])
        torch.testing.assert_close(after['logits'],before['logits'][:,permutation],rtol=1e-6,atol=1e-6)
        mask = after['audit']['neighbor_mask']
        translated = permutation[after['audit']['neighbor_indices'].clamp_min(0)].masked_fill(~mask,-1)
        self.assertTrue(torch.equal(translated,before['audit']['neighbor_indices'][:,permutation]))

    def test_control_changes_only_sender_content_and_keeps_raw_receiver(self):
        t,v = inputs()
        old_t,old_v = t.clone(),v.clone()
        natural,shuffled = ReturnGraph('natural'),ReturnGraph('shuffled')
        shuffled.load_state_dict(natural.state_dict())
        a,b = natural(t,v),shuffled(t,v)
        for key in ('neighbor_indices','neighbor_mask','relative_descriptors','neighbor_count','receiver_features','raw_receiver_logit'):
            self.assertTrue(torch.equal(a['audit'][key],b['audit'][key]),key)
        mask = a['audit']['neighbor_mask']
        self.assertTrue(torch.all(a['audit']['content_indices'][mask] != b['audit']['content_indices'][mask]))
        perm = b['audit']['sender_permutation'][0,v[0]]
        self.assertTrue(torch.equal(perm.sort().values,torch.arange(int(v.sum()))))
        self.assertGreater(float((a['logits']-b['logits']).detach().abs().max()),1e-7)
        self.assertTrue(torch.equal(t,old_t) and torch.equal(v,old_v))
        # Only the explicit raw receiver skip remains: changing senders cannot
        # change this node's output, but changing its own raw channel must.
        for model in (natural,shuffled):
            with torch.no_grad():
                for parameter in model.parameters(): parameter.zero_()
                model.raw_skip.weight[0,4] = 2.
        torch.testing.assert_close(natural(t,v)['logits'][v],2*t[:,:,4][v])
        self.assertTrue(torch.equal(natural(t,v)['logits'],shuffled(t,v)['logits']))

    def test_invalid_senders_and_empty_neighborhood_are_finite_and_inert(self):
        t,v = inputs(3)
        poisoned = t.clone(); poisoned[~v] = float('nan')
        for arm in ('natural','shuffled'):
            model = ReturnGraph(arm)
            expected,actual = model(t,v),model(poisoned,v)
            self.assertTrue(torch.equal(expected['logits'],actual['logits']))
            self.assertTrue(torch.all(actual['logits'][~v]==-30.))
            all_missing = model(torch.full_like(t,float('inf')),torch.zeros_like(v))
            self.assertTrue(torch.all(all_missing['logits']==-30.))
            self.assertTrue(torch.all(all_missing['audit']['aggregated_message']==0))
            one = v.clone(); one[:,1:] = False
            single = model(t,one)
            self.assertTrue(torch.all(single['audit']['aggregated_message']==0))
            self.assertTrue(torch.isfinite(single['logits']).all())
            mixed_valid = torch.cat((v,one,torch.zeros_like(v)))
            mixed = model(t.expand(3,-1,-1),mixed_valid)
            expected_batch = torch.cat((expected['logits'],single['logits'],all_missing['logits']))
            torch.testing.assert_close(mixed['logits'],expected_batch,rtol=1e-6,atol=1e-6)

    def test_neighbor_content_has_gradient_without_raw_receiver_change(self):
        t,v = inputs(2)
        t = t.requires_grad_()
        model = ReturnGraph()
        # Explicit positive path sender channel4 -> message -> output prevents
        # a random dead ReLU from making this dependency test inconclusive.
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
            model.stem[0].weight[0,4] = 1.
            model.message[0].weight[0,32] = 1.
            model.message[2].weight[0,0] = 1.
            model.output[0].weight[0,32] = 1.
            model.output[2].weight[0,0] = 1.
        out = model(t,v)
        out['logits'][0,0].backward()
        self.assertGreater(float(t.grad[0,1,4]),.99)
        self.assertEqual(float(t.grad[0,0,4]),0.)
        self.assertEqual(float(t.grad[0,2:].abs().sum()),0.)
        self.assertGreater(float(model.message[0].weight.grad.abs().sum()),0.)


if __name__ == '__main__':
    unittest.main()
