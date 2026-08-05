import unittest

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.algo.sdp_utils import (
    apply_desired_set_operator,
    desired_set_membership,
    desired_set_radii,
    sample_desired_action_chunks,
)


class ZeroNoisePredictor(torch.nn.Module):
    def forward(self, sample, timestep, global_cond):
        del timestep, global_cond
        return torch.zeros_like(sample)


class TestSDPUtils(unittest.TestCase):
    def setUp(self):
        self.positive = torch.zeros(2, 4, 3)
        self.negative = torch.zeros_like(self.positive)
        self.negative[..., 0] = 2.0

    def test_radius_and_membership(self):
        radii = desired_set_radii(
            self.positive, self.negative, radius_ratio=0.25
        )
        torch.testing.assert_close(radii, torch.full((2, 4, 1), 0.5))

        candidate = self.positive.clone()
        candidate[..., 1] = 0.49
        self.assertTrue(
            bool(
                torch.all(
                    desired_set_membership(
                        candidate,
                        self.positive,
                        self.negative,
                        radius_ratio=0.25,
                    )
                )
            )
        )
        candidate[..., 1] = 0.51
        self.assertFalse(
            bool(
                torch.any(
                    desired_set_membership(
                        candidate,
                        self.positive,
                        self.negative,
                        radius_ratio=0.25,
                    )
                )
            )
        )

    def test_operator_replaces_whole_single_step_action(self):
        candidate = self.positive.clone()
        candidate[0, 0] = torch.tensor([0.1, 0.2, 0.3])
        candidate[0, 1] = torch.tensor([0.6, 0.0, 0.0])
        projected, membership = apply_desired_set_operator(
            candidate,
            self.positive,
            self.negative,
            radius_ratio=0.25,
        )
        self.assertTrue(bool(membership[0, 0]))
        self.assertFalse(bool(membership[0, 1]))
        torch.testing.assert_close(projected[0, 0], candidate[0, 0])
        torch.testing.assert_close(projected[0, 1], self.positive[0, 1])

    def test_zero_radius_reduces_to_positive_target(self):
        candidate = torch.randn_like(self.positive)
        projected, membership = apply_desired_set_operator(
            candidate,
            self.positive,
            self.negative,
            radius_ratio=0.0,
        )
        self.assertFalse(bool(torch.any(membership)))
        torch.testing.assert_close(projected, self.positive)

    def test_shared_history_zero_radius(self):
        negative = self.negative.clone()
        negative[:, 0] = self.positive[:, 0]
        candidate = self.positive.clone()
        candidate[:, 0, 2] = 0.01
        candidate[:, 1, 2] = 0.01
        projected, membership = apply_desired_set_operator(
            candidate,
            self.positive,
            negative,
            radius_ratio=0.25,
        )
        self.assertFalse(bool(torch.any(membership[:, 0])))
        self.assertTrue(bool(torch.all(membership[:, 1])))
        torch.testing.assert_close(projected[:, 0], self.positive[:, 0])
        torch.testing.assert_close(projected[:, 1], candidate[:, 1])

    def test_target_sampler_is_deterministic_in_set_and_detached(self):
        scheduler = DDPMScheduler(
            num_train_timesteps=20,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        condition = torch.randn(2, 5, requires_grad=True)
        positive = self.positive.clone().requires_grad_(True)
        negative = self.negative.clone()

        generator_a = torch.Generator().manual_seed(123)
        targets_a, stats_a = sample_desired_action_chunks(
            noise_pred_net=ZeroNoisePredictor(),
            scheduler=scheduler,
            observation_condition=condition,
            positive_actions=positive,
            negative_actions=negative,
            radius_ratio=0.25,
            num_samples=3,
            start_timestep=4,
            generator=generator_a,
        )
        generator_b = torch.Generator().manual_seed(123)
        targets_b, stats_b = sample_desired_action_chunks(
            noise_pred_net=ZeroNoisePredictor(),
            scheduler=scheduler,
            observation_condition=condition,
            positive_actions=positive,
            negative_actions=negative,
            radius_ratio=0.25,
            num_samples=3,
            start_timestep=4,
            generator=generator_b,
        )

        self.assertEqual(tuple(targets_a.shape), (2, 3, 4, 3))
        self.assertFalse(targets_a.requires_grad)
        torch.testing.assert_close(targets_a, targets_b)
        for key in stats_a:
            torch.testing.assert_close(stats_a[key], stats_b[key])
        repeated_positive = self.positive[:, None].expand_as(targets_a)
        repeated_negative = self.negative[:, None].expand_as(targets_a)
        self.assertTrue(
            bool(
                torch.all(
                    desired_set_membership(
                        targets_a,
                        repeated_positive,
                        repeated_negative,
                        radius_ratio=0.25,
                    )
                )
            )
        )
        self.assertIsNone(condition.grad)
        self.assertIsNone(positive.grad)

    def test_grouped_loss_matches_clean_bc_loss(self):
        prediction = torch.randn(5, 4, 3)
        target = torch.randn_like(prediction)
        sample_indices = torch.arange(5)
        grouped_loss = DiffusionPolicyUNet._grouped_diffusion_loss(
            prediction=prediction,
            target=target,
            original_sample_indices=sample_indices,
            original_batch_size=5,
        )
        torch.testing.assert_close(
            grouped_loss,
            F.mse_loss(prediction, target),
        )

    def test_training_target_follows_scheduler_prediction_type(self):
        actions = torch.randn(3, 4, 2)
        noise = torch.randn_like(actions)
        timesteps = torch.tensor([0, 4, 9])

        for scheduler_class in (DDPMScheduler, DDIMScheduler):
            for prediction_type, expected in (
                ("epsilon", noise),
                ("sample", actions),
            ):
                with self.subTest(
                    scheduler=scheduler_class.__name__,
                    prediction_type=prediction_type,
                ):
                    scheduler = scheduler_class(
                        num_train_timesteps=10,
                        prediction_type=prediction_type,
                    )
                    actual = DiffusionPolicyUNet._diffusion_training_target(
                        actions=actions,
                        noise=noise,
                        timesteps=timesteps,
                        noise_scheduler=scheduler,
                    )
                    torch.testing.assert_close(actual, expected)

    def test_velocity_training_target_uses_scheduler_definition(self):
        actions = torch.randn(3, 4, 2)
        noise = torch.randn_like(actions)
        timesteps = torch.tensor([0, 4, 9])
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
            prediction_type="v_prediction",
        )

        actual = DiffusionPolicyUNet._diffusion_training_target(
            actions=actions,
            noise=noise,
            timesteps=timesteps,
            noise_scheduler=scheduler,
        )
        expected = scheduler.get_velocity(actions, noise, timesteps)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
