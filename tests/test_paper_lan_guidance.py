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
