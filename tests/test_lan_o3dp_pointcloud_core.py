import unittest

import torch
import torch.nn as nn

from robomimic.models.obs_core import LanO3DPPointCloudCore


class TestLanO3DPPointCloudCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = LanO3DPPointCloudCore(input_shape=(3, 512))

    def test_exact_architecture_and_shortcut_source(self):
        self.assertEqual(
            [type(layer) for layer in self.encoder.main],
            [
                nn.Linear,
                nn.LayerNorm,
                nn.ReLU,
                nn.Linear,
                nn.LayerNorm,
                nn.ReLU,
                nn.Linear,
                nn.LayerNorm,
                nn.ReLU,
            ],
        )
        self.assertEqual(self.encoder.shortcut.in_features, 3)
        self.assertEqual(self.encoder.shortcut.out_features, 64)
        self.assertIsInstance(self.encoder.projection[0], nn.Linear)
        self.assertIsInstance(self.encoder.projection[1], nn.LayerNorm)
        self.assertNotIsInstance(self.encoder.projection[-1], nn.ReLU)

        captured = {}

        def capture_shortcut(_module, args):
            captured["input"] = args[0].detach().clone()

        handle = self.encoder.shortcut.register_forward_pre_hook(capture_shortcut)
        points = torch.randn(2, 3, 512)
        self.encoder(points)
        handle.remove()
        torch.testing.assert_close(captured["input"], points.transpose(-1, -2))

    def test_shape_permutation_invariance_and_finite_gradient(self):
        points = torch.randn(4, 3, 512, requires_grad=True)
        encoded = self.encoder(points)
        self.assertEqual(tuple(encoded.shape), (4, 64))
        permutation = torch.randperm(512)
        torch.testing.assert_close(
            encoded.detach(),
            self.encoder(points.detach()[:, :, permutation]),
            rtol=0,
            atol=1e-6,
        )
        encoded.square().mean().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
