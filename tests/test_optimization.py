import unittest

import torch

from fedbrain.model import FedBrainModel
from fedbrain.selection import ValidationCheckpoint,scaled_alignment_loss


class SelectionTests(unittest.TestCase):
    def test_restores_all_expert_and_gate_parameters_without_aliasing(self):
        models={s:FedBrainModel(4,8) for s in ('A','B')}
        expected={s:{k:v.detach().clone() for k,v in m.state_dict().items()} for s,m in models.items()}
        selector=ValidationCheckpoint()
        self.assertTrue(selector.consider(2,.7,models))
        with torch.no_grad():
            for model in models.values():
                for p in model.parameters():
                    p.add_(10)
        self.assertFalse(selector.consider(3,.7,models))
        self.assertFalse(selector.consider(4,.6,models))
        selector.restore(models)
        self.assertEqual(selector.round,2)
        for site,model in models.items():
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,expected[site][key])

    def test_paper_sum_objective_and_gradient_equivalence(self):
        logits=torch.tensor([-.3,.5,1.],requires_grad=True)
        labels=torch.tensor([0.,1.,1.])
        ce=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
        alignment=logits.square().sum()
        actual,weighted=scaled_alignment_loss(ce,alignment,3,1.,'batch')
        expected=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='sum')+alignment)/3
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(ga,ge)
        torch.testing.assert_close(weighted,alignment/3)

    def test_reference_refresh_uses_current_shared_encoder(self):
        import numpy as np
        from fedbrain.runner import _reference_moments
        models={s:FedBrainModel(4,8) for s in ('A','B')}
        x=np.ones((4,4),dtype=np.float32)
        y=np.array([0,0,1,1],dtype=np.float32)
        data={s:(x,y) for s in models}
        before=_reference_moments(models,data,'cpu')
        shared=models['A'].shared_state_dict()
        shared['encoder.0.weight'].fill_(.5)
        shared['encoder.0.bias'].fill_(1.)
        for model in models.values():
            model.load_shared_state_dict(shared)
        after=_reference_moments(models,data,'cpu')
        for site in models:
            torch.testing.assert_close(after[site]['marginal'],torch.full((8,),3.))
            self.assertFalse(torch.equal(before[site]['marginal'],after[site]['marginal']))

    def test_nonfinite_validation_rejected(self):
        with self.assertRaises(ValueError):
            ValidationCheckpoint().consider(1,float('nan'),{})


if __name__=='__main__':
    unittest.main()
