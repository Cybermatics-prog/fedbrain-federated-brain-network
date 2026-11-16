import unittest

import numpy as np
import torch

from fedbrain.metrics import binary_metrics, evaluator_self_check
from fedbrain.privacy import (PackedPaillierAggregator, PaillierKey,
                              average_shared_states, perturb_shared_state)


class MetricsTests(unittest.TestCase):
    def test_self_check(self):
        result = evaluator_self_check()
        self.assertEqual(result['gold_accuracy'], 1.0)
        self.assertTrue(.47 <= result['shuffled_accuracy'] <= .53)

    def test_confusion_order(self):
        m = binary_metrics([0,0,1,1], [.1,.6,.2,.9])
        self.assertEqual(m['confusion_matrix'], [[1,1],[1,1]])
        self.assertEqual(m['accuracy'], .5)

    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):
            binary_metrics([0,1], [float('nan'), .4])


class PrivacyTests(unittest.TestCase):
    def test_actual_additive_encryption(self):
        key = PaillierKey.generate(bits=256)
        a, b = key.encrypt(19), key.encrypt(23)
        self.assertEqual(key.decrypt(key.add(a,b)), 42)
        self.assertNotEqual(key.encrypt(19), key.encrypt(19))

    def test_packed_signed_mean(self):
        a = np.array([-2.14,0.,3.45,.001,1.99,-9.], dtype=np.float32)
        b = np.array([1.12,-.2,5.1,.012,-1.1,8.], dtype=np.float32)
        engine = PackedPaillierAggregator(bits=256)
        actual = engine.aggregate([a,b])
        np.testing.assert_allclose(actual, (a+b)/2, atol=1e-6)
        self.assertGreater(engine.last_wire_bytes, 0)

    def test_no_private_aggregation(self):
        states = [{'encoder.weight':torch.tensor([1.,2.])}, {'encoder.weight':torch.tensor([3.,4.])}]
        result = average_shared_states(states)
        torch.testing.assert_close(result['encoder.weight'], torch.tensor([2.,3.]))

    def test_parameter_perturbation_reproducible(self):
        state = {'weight':torch.ones(4)}
        a = perturb_shared_state(state,.01,generator=np.random.default_rng(42))
        b = perturb_shared_state(state,.01,generator=np.random.default_rng(42))
        torch.testing.assert_close(a['weight'], b['weight'])
        self.assertFalse(torch.equal(a['weight'], state['weight']))


if __name__ == '__main__':
    unittest.main()
