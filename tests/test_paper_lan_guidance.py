from collections import OrderedDict

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from robomimic.algo import algo_factory, algo_name_to_factory_func
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.paper_lan_guidance_utils import (
    PaperLanGuidanceContext,
    apply_output_tangent_guidance,
    closest_obstacle_point,
    paper_guidance_gradient,
    paper_obstacle_cost,
    predicted_clean_action,
)


def scheduler(prediction_type="epsilon", clip_sample=False):
    return DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=prediction_type,
        clip_sample=clip_sample,
    )


def test_registration_is_opt_in():
    config = config_factory("paper_guided_lan_o3dp")
    assert config.algo_name == "paper_guided_lan_o3dp"
    assert callable(algo_name_to_factory_func("paper_guided_lan_o3dp"))


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_predicted_clean_matches_scheduler(prediction_type):
    noise_scheduler = scheduler(prediction_type=prediction_type, clip_sample=True)
    sample = torch.randn(2, 4, 7)
    output = torch.randn_like(sample)
    timestep = torch.tensor(6)
    expected = noise_scheduler.step(output, timestep, sample).pred_original_sample
    actual = predicted_clean_action(sample, output, timestep, noise_scheduler)
    torch.testing.assert_close(actual, expected)


def test_closest_obstacle_point_is_fixed_from_current_eef():
    points = torch.tensor(
        [[0.05, 0.0, 2.0], [0.01, 0.02, -2.0], [0.5, 0.5, 0.0]]
    )
    point = closest_obstacle_point(
        [0.0, 0.0, 0.0],
        points,
        xy_only=True,
    )
    torch.testing.assert_close(point, points[1])


def test_thresholded_cost_negative_gradient_points_outward():
    action = torch.tensor([[[0.02, 0.0, 0.0, 0, 0, 0, 0]]], requires_grad=True)
    cost, distances, penetration = paper_obstacle_cost(
        action,
        obstacle_point=[0.0, 0.0, 0.0],
        safety_distance=0.05,
        xy_only=True,
    )
    gradient = torch.autograd.grad(cost, action)[0]
    assert distances.item() == pytest.approx(0.02, abs=1e-6)
    assert penetration.item() == pytest.approx(0.03, abs=1e-6)
    assert gradient[0, 0, 0].item() < 0
    assert torch.all(gradient[..., 1:] == 0)
    assert (action - 0.1 * gradient)[0, 0, 0].item() > action[0, 0, 0].item()


def test_huber_hinge_gradient_ramps_then_matches_hinge():
    shallow = torch.tensor([[[0.045, 0.0, 0.0]]], requires_grad=True)
    shallow_cost, _, shallow_penetration = paper_obstacle_cost(
        shallow,
        obstacle_point=[0.0, 0.0, 0.0],
        safety_distance=0.05,
        smoothing=0.01,
        cost_type="huber_hinge",
    )
    shallow_gradient = torch.autograd.grad(shallow_cost, shallow)[0]
    assert shallow_penetration.item() == pytest.approx(0.005, abs=1e-6)
    assert shallow_cost.item() == pytest.approx(0.00125, abs=1e-6)
    assert shallow_gradient[0, 0, 0].item() == pytest.approx(-0.5, abs=1e-5)

    deep = torch.tensor([[[0.02, 0.0, 0.0]]], requires_grad=True)
    deep_cost, _, _ = paper_obstacle_cost(
        deep,
        obstacle_point=[0.0, 0.0, 0.0],
        safety_distance=0.05,
        smoothing=0.01,
        cost_type="huber_hinge",
    )
    deep_gradient = torch.autograd.grad(deep_cost, deep)[0]
    assert deep_cost.item() == pytest.approx(0.025, abs=1e-6)
    assert deep_gradient[0, 0, 0].item() == pytest.approx(-1.0, abs=1e-5)


def test_huber_hinge_has_zero_gradient_at_and_outside_boundary():
    action = torch.tensor(
        [[[0.05, 0.0, 0.0], [0.06, 0.0, 0.0]]], requires_grad=True
    )
    cost, _, penetration = paper_obstacle_cost(
        action,
        obstacle_point=[0.0, 0.0, 0.0],
        safety_distance=0.05,
        smoothing=0.01,
        cost_type="huber_hinge",
    )
    gradient = torch.autograd.grad(cost, action)[0]
    assert torch.all(penetration == 0)
    assert torch.all(gradient == 0)


def test_output_tangent_guidance_zero_ratio_is_exact_noop():
    action = torch.tensor(
        [[[-0.03, -0.15, 0.40, 1.0], [-0.03, 0.10, 0.50, -1.0]]]
    )
    context = PaperLanGuidanceContext(
        current_eef_pos=np.asarray([-0.03, -0.23, 0.4], dtype=np.float32),
        obstacle_points=np.asarray([[0.0, 0.0, 0.4]], dtype=np.float32),
        action_scale=np.ones(4),
        action_offset=np.zeros(4),
        guidance_scale=45.0,
        safety_distance=1.0,
        position_indices=(0, 1, 2),
        tangent_ratio=0.0,
    )
    corrected, side = apply_output_tangent_guidance(action, context)
    assert torch.equal(corrected, action)
    assert side == 0


def test_output_tangent_guidance_directly_moves_forward_aligned_xy_only():
    action = torch.tensor(
        [[[-0.03, -0.15, 0.40, 1.0], [-0.03, 0.10, 0.50, -1.0]]]
    )
    context = PaperLanGuidanceContext(
        current_eef_pos=np.asarray([-0.03, -0.23, 0.4], dtype=np.float32),
        obstacle_points=np.asarray([[0.0, 0.0, 0.4]], dtype=np.float32),
        action_scale=np.ones(4),
        action_offset=np.zeros(4),
        guidance_scale=45.0,
        safety_distance=1.0,
        position_indices=(0, 1, 2),
        tangent_ratio=0.5,
        forward_direction_xy=(0.0, 1.0),
    )
    corrected, side = apply_output_tangent_guidance(action, context)

    displacement = corrected - action
    forward = action[0, -1, :2] - torch.tensor(context.current_eef_pos[:2])
    assert side == -1
    assert torch.dot(displacement[0, 0, :2], forward).item() > 0
    # World -X is upward in the rotated start-left / goal-right view.
    assert displacement[0, 0, 0].item() < 0
    assert displacement[0, 0, 1].item() > 0
    assert torch.all(displacement[..., 2:] == 0)


def test_output_tangent_guidance_can_follow_normal_guidance_trigger():
    action = torch.tensor(
        [[[-0.20, -0.40, 0.40], [-0.20, -0.30, 0.40]]]
    )
    context = PaperLanGuidanceContext(
        current_eef_pos=np.asarray([-0.02, -0.07, 0.4], dtype=np.float32),
        obstacle_points=np.asarray([[0.0, 0.0, 0.4]], dtype=np.float32),
        action_scale=np.ones(3),
        action_offset=np.zeros(3),
        guidance_scale=45.0,
        safety_distance=0.08,
        tangent_ratio=0.5,
        forward_direction_xy=(0.0, 1.0),
    )
    unchanged, inactive_side = apply_output_tangent_guidance(action, context)
    corrected, active_side = apply_output_tangent_guidance(
        action, context, force_active=True
    )

    assert torch.equal(unchanged, action)
    assert inactive_side == 0
    assert active_side != 0
    displacement = torch.linalg.vector_norm(
        corrected[..., :2] - action[..., :2], dim=-1
    )
    torch.testing.assert_close(displacement, torch.full_like(displacement, 0.04))


def test_output_tangent_guidance_smoothly_ramps_near_threshold():
    action = torch.tensor(
        [[[-0.20, -0.40, 0.40], [-0.20, -0.30, 0.40]]]
    )

    def displacement_at(distance):
        context = PaperLanGuidanceContext(
            current_eef_pos=np.asarray([0.0, -distance, 0.4], dtype=np.float32),
            obstacle_points=np.asarray([[0.0, 0.0, 0.4]], dtype=np.float32),
            action_scale=np.ones(3),
            action_offset=np.zeros(3),
            guidance_scale=45.0,
            safety_distance=0.08,
            obstacle_cost_type="huber_hinge",
            obstacle_cost_smoothing=0.01,
            tangent_ratio=0.1,
            forward_direction_xy=(0.0, 1.0),
        )
        corrected, _ = apply_output_tangent_guidance(
            action, context, force_active=True
        )
        return torch.linalg.vector_norm(
            corrected[0, 0, :2] - action[0, 0, :2]
        ).item()

    assert displacement_at(0.091) == pytest.approx(0.0, abs=1e-7)
    assert displacement_at(0.080) == pytest.approx(0.004, abs=1e-6)
    assert displacement_at(0.069) == pytest.approx(0.008, abs=1e-6)


def test_tangent_ratio_does_not_enter_denoiser_gradient():
    noise_scheduler = scheduler(prediction_type="sample", clip_sample=False)
    noisy = torch.tensor(
        [[[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]], requires_grad=True
    )
    common = dict(
        current_eef_pos=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        obstacle_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        action_scale=np.ones(3),
        action_offset=np.zeros(3),
        guidance_scale=1.0,
        safety_distance=1.0,
    )
    baseline, _ = paper_guidance_gradient(
        noisy_action=noisy,
        model_output=noisy,
        timestep=torch.tensor(5),
        scheduler=noise_scheduler,
        context=PaperLanGuidanceContext(**common),
    )
    with_tangent, diagnostics = paper_guidance_gradient(
        noisy_action=noisy,
        model_output=noisy,
        timestep=torch.tensor(5),
        scheduler=noise_scheduler,
        context=PaperLanGuidanceContext(**common, tangent_ratio=0.5),
    )
    assert torch.equal(baseline, with_tangent)
    assert diagnostics.tangent_side == 0
    assert diagnostics.tangent_ratio == pytest.approx(0.5)


def test_guidance_gradient_includes_denoiser_jacobian():
    noise_scheduler = scheduler(prediction_type="epsilon", clip_sample=False)
    timestep = torch.tensor(5)
    noisy = torch.full((1, 1, 7), 0.02, requires_grad=True)
    # A denoiser with a nonzero d epsilon / d A_k. Detaching model_output would
    # produce a measurably different gradient.
    model_output = 0.5 * noisy
    context = PaperLanGuidanceContext(
        current_eef_pos=np.zeros(3),
        obstacle_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        action_scale=np.ones(7),
        action_offset=np.zeros(7),
        guidance_scale=0.1,
        safety_distance=1.0,
    )
    update, diagnostics = paper_guidance_gradient(
        noisy_action=noisy,
        model_output=model_output,
        timestep=timestep,
        scheduler=noise_scheduler,
        context=context,
    )

    detached_noisy = noisy.detach().requires_grad_(True)
    detached_clean = predicted_clean_action(
        detached_noisy,
        model_output.detach(),
        timestep,
        noise_scheduler,
    )
    detached_cost, _, _ = paper_obstacle_cost(
        detached_clean,
        [0.0, 0.0, 0.0],
        safety_distance=1.0,
    )
    detached_update = 0.1 * torch.autograd.grad(detached_cost, detached_noisy)[0]
    assert not torch.allclose(update, detached_update)
    assert diagnostics.active_waypoint_count == 1
    assert np.isfinite(diagnostics.noisy_action_gradient_norm)


def test_guidance_gradient_skips_backward_outside_safety_distance(monkeypatch):
    noise_scheduler = scheduler(prediction_type="sample", clip_sample=False)
    noisy = torch.zeros((1, 2, 7), requires_grad=True)
    model_output = noisy + 1.0
    context = PaperLanGuidanceContext(
        current_eef_pos=np.zeros(3),
        obstacle_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        action_scale=np.ones(7),
        action_offset=np.zeros(7),
        guidance_scale=0.1,
        safety_distance=0.03,
    )

    def unexpected_backward(*args, **kwargs):
        raise AssertionError("inactive guidance should not invoke autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", unexpected_backward)
    update, diagnostics = paper_guidance_gradient(
        noisy_action=noisy,
        model_output=model_output,
        timestep=torch.tensor(5),
        scheduler=noise_scheduler,
        context=context,
    )

    assert torch.count_nonzero(update) == 0
    assert diagnostics.active_waypoint_count == 0
    assert diagnostics.noisy_action_gradient_norm == 0.0


def test_context_validation():
    with pytest.raises(ValueError, match="safety_distance"):
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(3),
            obstacle_points=np.zeros((1, 3)),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1.0,
            safety_distance=0.0,
        )
    with pytest.raises(ValueError, match="positive obstacle_cost_smoothing"):
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(3),
            obstacle_points=np.zeros((1, 3)),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1.0,
            safety_distance=0.1,
            obstacle_cost_type="huber_hinge",
        )
    with pytest.raises(ValueError, match="tangent_ratio"):
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(3),
            obstacle_points=np.zeros((1, 3)),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1.0,
            safety_distance=0.1,
            tangent_ratio=-0.1,
        )
    with pytest.raises(ValueError, match="tangent_side"):
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(3),
            obstacle_points=np.zeros((1, 3)),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1.0,
            safety_distance=0.1,
            tangent_side=2,
        )


def small_policy(algo_name):
    config = config_factory(algo_name)
    with config.unlocked():
        config.train.cuda = False
        config.train.action_keys = ["abs_eef_pose_action"]
        config.algo.unet.down_dims = [32, 64]
        config.algo.unet.diffusion_step_embed_dim = 32
        config.algo.ddpm.num_train_timesteps = 4
        config.algo.ddpm.num_inference_timesteps = 4
        config.algo.ddpm.prediction_type = "epsilon"
        config.algo.optim_params.policy.num_train_batches = 1
        config.algo.optim_params.policy.num_epochs = 1
        config.observation.modalities.obs.low_dim = ["agent_pos"]
        config.observation.modalities.obs.scan = ["task_pointcloud"]
        config.observation.modalities.obs.rgb = []
        config.observation.modalities.obs.depth = []
        config.observation.encoder.scan.core_class = "LanO3DPPointCloudCore"
        config.observation.encoder.scan.core_kwargs = {}
    ObsUtils.initialize_obs_utils_with_config(config)
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=OrderedDict(
            [("agent_pos", (6,)), ("task_pointcloud", (3, 16))]
        ),
        ac_dim=7,
        device=torch.device("cpu"),
    )
    model.set_eval()
    return model


def test_guided_algorithm_disabled_parity_and_enabled_diagnostics():
    source = small_policy("lan_o3dp")
    guided = small_policy("paper_guided_lan_o3dp")
    guided.deserialize(source.serialize())
    observation = {
        "agent_pos": torch.randn(1, 2, 6),
        "task_pointcloud": torch.randn(1, 2, 3, 16),
    }

    torch.manual_seed(7)
    with torch.no_grad():
        expected = source._get_action_trajectory(observation)
    torch.manual_seed(7)
    with torch.no_grad():
        actual = guided._get_action_trajectory(observation)
    assert torch.equal(actual, expected)

    guided.set_guidance_context(
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(3),
            obstacle_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=0.01,
            safety_distance=10.0,
        )
    )
    torch.manual_seed(7)
    with torch.no_grad():
        guided_action = guided._get_action_trajectory(observation)
    assert torch.isfinite(guided_action).all()
    assert len(guided.get_guidance_diagnostics()) == 4
    assert all(parameter.grad is None for parameter in guided.nets.parameters())
