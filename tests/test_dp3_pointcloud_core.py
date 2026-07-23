import unittest

import torch

from robomimic.models.obs_core import DP3PointCloudCore


class TestDP3PointCloudCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = DP3PointCloudCore(input_shape=(3, 512))

    def test_shape_and_finite_gradient(self):
        points = torch.randn(4, 3, 512, requires_grad=True)
        encoded = self.encoder(points)
        self.assertEqual(tuple(encoded.shape), (4, 64))
        encoded.square().mean().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), 0.0)

    def test_permutation_invariance(self):
        points = torch.randn(2, 3, 512)
        permutation = torch.randperm(512)
        torch.testing.assert_close(
            self.encoder(points),
            self.encoder(points[:, :, permutation]),
            rtol=0,
            atol=1e-6,
        )

    def test_repeat_padding_does_not_change_max_pool(self):
        unique = torch.randn(1, 3, 256)
        repeated = unique.repeat(1, 1, 2)
        short_encoder = DP3PointCloudCore(input_shape=(3, 256))
        short_encoder.load_state_dict(self.encoder.state_dict())
        torch.testing.assert_close(
            short_encoder(unique),
            self.encoder(repeated),
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
