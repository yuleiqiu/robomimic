"""Unit checks for clean-image delta-EEF guided denoising utilities."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from robomimic.algo.algo import REGISTERED_ALGO_FACTORY_FUNCS
from robomimic.config.base_config import get_all_registered_configs
import robomimic.algo  # noqa: F401 - triggers algorithm registration
import robomimic.config  # noqa: F401 - triggers config registration
import robomimic.utils.guided_denoising_utils as GuidedUtils


def make_scheduler():
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(10)
    return scheduler


def make_context(**overrides):
    values = dict(
        current_eef_pos=torch.zeros(3),
        obstacle_centers=torch.tensor([[0.05, 0.0, 0.0]]),
        obstacle_radii=torch.tensor([0.06]),
        action_scale=torch.ones(7),
        action_offset=torch.zeros(7),
        guidance_scale=0.001,
        clearance_margin=0.0,
    )
    values.update(overrides)
    return GuidedUtils.GuidedDenoisingContext(**values)


def test_registered_guided_variant():
    assert "guided_diffusion_policy" in REGISTERED_ALGO_FACTORY_FUNCS
    assert "guided_lan_o3dp" in REGISTERED_ALGO_FACTORY_FUNCS
    assert "guided_diffusion_policy" in get_all_registered_configs()
    assert "guided_lan_o3dp" in get_all_registered_configs()


def test_previous_alpha_uses_actual_ddim_and_ddpm_inference_sequences():
    reference = torch.zeros(())
    ddim = make_scheduler()
    previous, alpha = GuidedUtils.previous_alpha_cumprod(
        ddim, ddim.timesteps[0], reference
    )
    assert previous == int(ddim.timesteps[1])
    assert torch.allclose(alpha, ddim.alphas_cumprod[previous])

    ddpm = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    ddpm.set_timesteps(100)
    previous, alpha = GuidedUtils.previous_alpha_cumprod(
        ddpm, ddpm.timesteps[0], reference
    )
    assert previous == 98
    assert torch.allclose(alpha, ddpm.alphas_cumprod[98])

    previous, alpha = GuidedUtils.previous_alpha_cumprod(
        ddpm, ddpm.timesteps[-1], reference
    )
    assert previous == -1
    assert torch.allclose(alpha, ddpm.one)


def test_ddpm_100_guidance_is_finite_and_changes_reverse_sample():
    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(100)
    predicted_clean = torch.zeros((1, 16, 7))
    reverse_sample = torch.randn((1, 16, 7), generator=torch.Generator().manual_seed(17))
    guided, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        predicted_clean_action=predicted_clean,
        reverse_sample=reverse_sample,
        timestep=scheduler.timesteps[0],
        noise_scheduler=scheduler,
        context=make_context(max_waypoint_displacement_m=0.01),
        observation_horizon=2,
        action_horizon=8,
    )
    assert torch.all(torch.isfinite(guided))
    assert torch.any(guided != reverse_sample)
    assert diagnostics.previous_timestep == 98
    assert diagnostics.max_waypoint_displacement_m <= 0.010001


def test_delta_eef_reconstruction_matches_hand_computation():
    current = torch.tensor([10.0, 20.0, 30.0])
    deltas = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 3.0]]]
    )
    expected = torch.tensor(
        [[[11.0, 20.0, 30.0], [11.0, 22.0, 30.0], [10.0, 22.0, 33.0]]]
    )
    actual = GuidedUtils.reconstruct_delta_eef_positions(current, deltas)
    assert torch.allclose(actual, expected)


def test_point_unnormalization_and_displacement_conversion_are_distinct():
    normalized = torch.tensor([[[1.0, -1.0, 0.5]]])
    scale = torch.tensor([2.0, 4.0, 8.0])
    offset = torch.tensor([10.0, 20.0, 30.0])
    raw = GuidedUtils.unnormalize_action_points(normalized, scale, offset)
    assert torch.allclose(raw, torch.tensor([[[12.0, 16.0, 34.0]]]))

    physical_displacement = torch.tensor([[[2.0, 4.0, 8.0]]])
    normalized_displacement = GuidedUtils.physical_displacement_to_normalized(
        physical_displacement, scale
    )
    assert torch.allclose(normalized_displacement, torch.ones_like(physical_displacement))


def test_penetration_cost_is_unsquared_and_has_finite_gradient_at_zero_distance():
    waypoint = torch.zeros((1, 1, 3), requires_grad=True)
    cost, penetrations, clearances = GuidedUtils.lan_xy_penetration_cost(
        waypoint,
        obstacle_centers=torch.zeros((1, 3)),
        obstacle_radii=torch.tensor([0.1]),
        clearance_margin=0.02,
    )
    cost.backward()
    assert cost.item() > 0.0
    assert penetrations.item() > 0.0
    assert clearances.item() < 0.0
    assert torch.all(torch.isfinite(waypoint.grad))


def test_safe_waypoint_has_zero_cost_and_zero_gradient():
    waypoint = torch.zeros((1, 1, 3), requires_grad=True)
    cost, _, _ = GuidedUtils.lan_xy_penetration_cost(
        waypoint,
        obstacle_centers=torch.tensor([[1.0, 1.0, 0.0]]),
        obstacle_radii=torch.tensor([0.1]),
        clearance_margin=0.02,
    )
    cost.backward()
    assert torch.isclose(cost, torch.tensor(0.0))
    assert torch.equal(waypoint.grad, torch.zeros_like(waypoint))


def test_vector_field_gradient_is_radius_normalized_and_decays_to_boundary():
    gradient_norms = []
    for radius in (0.04, 0.10):
        waypoint = torch.tensor([[[radius * 0.5, 0.0, 0.0]]], requires_grad=True)
        cost, depth, clearance = GuidedUtils.decaying_vector_field_cost(
            waypoint,
            obstacle_centers=torch.zeros((1, 3)),
            obstacle_radii=torch.tensor([radius]),
            clearance_margin=0.0,
        )
        cost.backward()
        gradient_norms.append(torch.linalg.vector_norm(waypoint.grad))
        assert torch.allclose(depth, torch.tensor([[[0.5]]]), atol=1e-5)
        assert torch.allclose(clearance, torch.tensor([[[-radius * 0.5]]]), atol=1e-5)
    assert torch.allclose(gradient_norms[0], torch.tensor(0.5), atol=1e-4)
    assert torch.allclose(gradient_norms[0], gradient_norms[1], atol=1e-4)

    safe = torch.tensor([[[0.11, 0.0, 0.0]]], requires_grad=True)
    cost, depth, _ = GuidedUtils.decaying_vector_field_cost(
        safe,
        obstacle_centers=torch.zeros((1, 3)),
        obstacle_radii=torch.tensor([0.10]),
        clearance_margin=0.0,
    )
    cost.backward()
    assert torch.isclose(cost, torch.tensor(0.0))
    assert torch.equal(depth, torch.zeros_like(depth))
    assert torch.equal(safe.grad, torch.zeros_like(safe))


def test_vector_field_rejects_unknown_cost_type():
    try:
        make_context(cost_type="not_a_cost")
    except ValueError as exc:
        assert "Unsupported guidance cost type" in str(exc)
    else:
        raise AssertionError("Unknown guidance cost type should fail")


def test_pushed_waypoints_difference_back_to_exact_delta_update():
    current = torch.zeros(3)
    waypoints = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    displacement = torch.zeros_like(waypoints)
    displacement[:, 1, 1] = 0.5

    delta_update = GuidedUtils.waypoint_displacement_to_delta_update(
        current, waypoints, displacement
    )
    original_deltas = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    reconstructed = GuidedUtils.reconstruct_delta_eef_positions(
        current, original_deltas + delta_update
    )
    assert torch.allclose(reconstructed, waypoints + displacement)
    assert torch.allclose(reconstructed[:, 2], waypoints[:, 2])


def test_guidance_changes_only_executed_xy_slice():
    scheduler = make_scheduler()
    predicted_clean = torch.zeros((1, 16, 7))
    reverse_sample = torch.randn((1, 16, 7), generator=torch.Generator().manual_seed(7))

    guided, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        predicted_clean_action=predicted_clean,
        reverse_sample=reverse_sample,
        timestep=scheduler.timesteps[0],
        noise_scheduler=scheduler,
        context=make_context(),
        observation_horizon=2,
        action_horizon=8,
    )

    difference = guided - reverse_sample
    assert torch.any(difference[:, 1:9, :2] != 0)
    assert torch.equal(difference[:, :1], torch.zeros_like(difference[:, :1]))
    assert torch.equal(difference[:, 9:], torch.zeros_like(difference[:, 9:]))
    assert torch.equal(difference[:, 1:9, 2:], torch.zeros_like(difference[:, 1:9, 2:]))
    assert diagnostics.active_penetration_count == 8
    assert diagnostics.cost > diagnostics.resulting_clean_action_cost
    before = torch.tensor(diagnostics.before_waypoints_m)
    after = torch.tensor(diagnostics.after_waypoints_m)
    displacement = torch.tensor(diagnostics.waypoint_displacements_m)
    assert torch.allclose(after - before, displacement)
    assert torch.any(displacement[..., :2] != 0)
    assert torch.equal(displacement[..., 2], torch.zeros_like(displacement[..., 2]))


def test_disabled_empty_safe_and_zero_scale_preserve_reverse_sample_exactly():
    scheduler = make_scheduler()
    predicted_clean = torch.zeros((1, 16, 7))
    reverse_sample = torch.randn((1, 16, 7), generator=torch.Generator().manual_seed(11))
    common = dict(
        predicted_clean_action=predicted_clean,
        reverse_sample=reverse_sample,
        timestep=scheduler.timesteps[0],
        noise_scheduler=scheduler,
        observation_horizon=2,
        action_horizon=8,
    )

    disabled, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        context=make_context(enabled=False), **common
    )
    assert disabled is reverse_sample
    assert diagnostics is None

    empty, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        context=make_context(
            obstacle_centers=torch.zeros((0, 3)),
            obstacle_radii=torch.zeros((0,)),
        ),
        **common,
    )
    assert empty is reverse_sample
    assert diagnostics.active_penetration_count == 0

    safe, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        context=make_context(obstacle_centers=torch.tensor([[1.0, 1.0, 0.0]])),
        **common,
    )
    assert safe is reverse_sample
    assert diagnostics.active_penetration_count == 0

    zero_scale, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        context=make_context(guidance_scale=0.0), **common
    )
    assert zero_scale is reverse_sample
    assert diagnostics.cost > 0.0
    assert diagnostics.normalized_applied_update_norm == 0.0


def test_guidance_timestep_gate_and_waypoint_displacement_cap():
    scheduler = make_scheduler()
    predicted_clean = torch.zeros((1, 16, 7))
    reverse_sample = torch.randn((1, 16, 7), generator=torch.Generator().manual_seed(13))
    common = dict(
        predicted_clean_action=predicted_clean,
        reverse_sample=reverse_sample,
        noise_scheduler=scheduler,
        observation_horizon=2,
        action_horizon=8,
    )
    context = make_context(
        cost_type="vector_field",
        guidance_scale=1.0,
        max_guidance_timestep=20,
        max_waypoint_displacement_m=0.03,
    )

    skipped, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        timestep=scheduler.timesteps[0],
        context=context,
        **common,
    )
    assert skipped is reverse_sample
    assert diagnostics is None

    applied, diagnostics = GuidedUtils.apply_guidance_to_reverse_sample(
        timestep=scheduler.timesteps[-1],
        context=context,
        **common,
    )
    assert torch.any(applied != reverse_sample)
    assert diagnostics.max_waypoint_displacement_m <= 0.030001


if __name__ == "__main__":
    test_registered_guided_variant()
    test_previous_alpha_uses_actual_ddim_and_ddpm_inference_sequences()
    test_ddpm_100_guidance_is_finite_and_changes_reverse_sample()
    test_delta_eef_reconstruction_matches_hand_computation()
    test_point_unnormalization_and_displacement_conversion_are_distinct()
    test_penetration_cost_is_unsquared_and_has_finite_gradient_at_zero_distance()
    test_safe_waypoint_has_zero_cost_and_zero_gradient()
    test_vector_field_gradient_is_radius_normalized_and_decays_to_boundary()
    test_vector_field_rejects_unknown_cost_type()
    test_pushed_waypoints_difference_back_to_exact_delta_update()
    test_guidance_changes_only_executed_xy_slice()
    test_disabled_empty_safe_and_zero_scale_preserve_reverse_sample_exactly()
    test_guidance_timestep_gate_and_waypoint_displacement_cap()
    print("guided denoising utility checks passed")
