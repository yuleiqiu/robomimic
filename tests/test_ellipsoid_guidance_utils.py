import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robomimic.utils.ellipsoid_guidance_utils import (
    EllipsoidGuidanceContext,
    directional_ellipsoid_clearance,
    ellipsoid_guidance_gradient,
    ellipsoid_separation_cost,
    fit_minimum_volume_enclosing_ellipsoid,
    reconstruct_delta_eef_poses,
    rotation_vector_to_matrix,
    transform_local_ellipsoids,
)


def test_mvee_recovers_rotated_axis_extrema_and_encloses_inputs():
    angle = 0.37
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.asarray([0.4, -0.2, 1.1])
    axes = np.asarray([0.25, 0.12, 0.06])
    local_extrema = np.concatenate([np.diag(axes), -np.diag(axes)], axis=0)
    points = center + local_extrema @ rotation.T

    fitted = fit_minimum_volume_enclosing_ellipsoid(points)
    np.testing.assert_allclose(fitted.center, center, atol=1e-8)
    np.testing.assert_allclose(fitted.semi_axes, axes, rtol=2e-3, atol=1e-8)
    expected_quadratic = rotation @ np.diag(1.0 / np.square(axes)) @ rotation.T
    np.testing.assert_allclose(
        fitted.quadratic,
        expected_quadratic,
        rtol=3e-3,
        atol=1e-6,
    )
    assert fitted.converged
    assert fitted.maximum_input_radius <= 1.0 + 1e-10


def test_mvee_padding_is_metric_and_coplanar_points_are_rejected():
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    base = fit_minimum_volume_enclosing_ellipsoid(points)
    padded = fit_minimum_volume_enclosing_ellipsoid(points, padding=0.03)
    np.testing.assert_allclose(padded.semi_axes - base.semi_axes, 0.03)
    with pytest.raises(ValueError, match="span all"):
        fit_minimum_volume_enclosing_ellipsoid(points[:4])


def test_rotation_vector_and_delta_pose_reconstruction_use_world_left_updates():
    zero = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    identity = rotation_vector_to_matrix(zero)
    torch.testing.assert_close(
        identity,
        torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
    )
    identity.sum().backward()
    assert torch.isfinite(zero.grad).all()

    delta_positions = torch.tensor(
        [[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]],
        dtype=torch.float64,
    )
    delta_rotations = torch.tensor(
        [[[0.0, 0.0, math.pi / 2.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    positions, rotations = reconstruct_delta_eef_poses(
        [0.0, 0.0, 0.0],
        torch.eye(3, dtype=torch.float64),
        delta_positions,
        delta_rotations,
    )
    torch.testing.assert_close(
        positions,
        torch.tensor([[[0.1, 0.0, 0.0], [0.1, 0.2, 0.0]]], dtype=torch.float64),
    )
    expected_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rotations[0, 0], expected_rotation, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(rotations[0, 1], expected_rotation, atol=1e-7, rtol=1e-7)


def test_local_actor_transform_tracks_eef_rotation_and_offset():
    eef_positions = torch.tensor([[[1.0, 2.0, 3.0]]])
    eef_rotations = rotation_vector_to_matrix(
        torch.tensor([[[0.0, 0.0, math.pi / 2.0]]])
    )
    centers, rotations = transform_local_ellipsoids(
        eef_positions,
        eef_rotations,
        local_centers=[[0.1, 0.0, 0.0]],
        local_rotations=np.asarray([np.eye(3)]),
    )
    torch.testing.assert_close(centers, torch.tensor([[[[1.0, 2.1, 3.0]]]]))
    torch.testing.assert_close(rotations[:, :, 0], eef_rotations)


def test_directional_clearance_matches_spheres_and_uses_orientation():
    actor_center = torch.tensor([[[[0.35, 0.0, 0.0]]]], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    sphere_clearance = directional_ellipsoid_clearance(
        actor_center,
        identity,
        torch.tensor([0.10, 0.10, 0.10], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        identity,
        torch.tensor([0.20, 0.20, 0.20], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sphere_clearance,
        torch.tensor([[[0.05]]], dtype=torch.float64),
    )

    elongated = torch.tensor([0.20, 0.05, 0.05], dtype=torch.float64)
    aligned = directional_ellipsoid_clearance(
        actor_center,
        identity,
        elongated,
        torch.zeros(3, dtype=torch.float64),
        identity,
        elongated,
    )
    quarter_turn = rotation_vector_to_matrix(
        torch.tensor([0.0, 0.0, math.pi / 2.0], dtype=torch.float64)
    )
    transverse = directional_ellipsoid_clearance(
        actor_center,
        quarter_turn,
        elongated,
        torch.zeros(3, dtype=torch.float64),
        quarter_turn,
        elongated,
    )
    assert float(aligned.item()) < 0.0
    assert float(transverse.item()) > 0.0


def test_separation_cost_negative_gradient_pushes_actor_outward():
    center = torch.tensor([[[[0.15, 0.0, 0.0]]]], requires_grad=True)
    identity = torch.eye(3)
    axes = torch.tensor([0.10, 0.10, 0.10])
    cost, penetrations, clearances = ellipsoid_separation_cost(
        center,
        identity,
        axes,
        torch.zeros(3),
        identity,
        axes,
        cost_type="huber_hinge",
        smoothing=0.01,
    )
    cost.backward()
    assert float(clearances.item()) == pytest.approx(-0.05, abs=1e-6)
    assert float(penetrations.item()) == pytest.approx(0.05, abs=1e-6)
    assert float(center.grad[..., 0].item()) < 0.0
    assert float(center.grad[..., 1:].abs().max().item()) == 0.0


def test_guidance_gradient_changes_only_executed_pose_coordinates():
    noisy_action = torch.zeros(1, 16, 7, requires_grad=True)
    context = EllipsoidGuidanceContext(
        current_eef_position=np.asarray([0.15, 0.0, 0.0]),
        current_eef_rotation=np.eye(3),
        actor_local_centers=np.zeros((1, 3)),
        actor_local_rotations=np.asarray([np.eye(3)]),
        actor_semi_axes=np.asarray([[0.10, 0.10, 0.10]]),
        obstacle_center=np.zeros(3),
        obstacle_rotation=np.eye(3),
        obstacle_semi_axes=np.asarray([0.10, 0.10, 0.10]),
        action_scale=np.ones(7),
        action_offset=np.zeros(7),
        guidance_scale=0.01,
    )
    scheduler = SimpleNamespace(
        config=SimpleNamespace(prediction_type="sample", clip_sample=False),
        alphas_cumprod=torch.ones(4),
    )
    update, diagnostics = ellipsoid_guidance_gradient(
        noisy_action=noisy_action,
        model_output=noisy_action,
        timestep=3,
        scheduler=scheduler,
        context=context,
        observation_horizon=2,
        action_horizon=8,
    )
    assert diagnostics.active_actor_waypoint_count == 8
    assert torch.count_nonzero(update[:, 1:9, 0]) > 0
    assert torch.count_nonzero(update[:, :1]) == 0
    assert torch.count_nonzero(update[:, 9:]) == 0
    assert torch.count_nonzero(update[..., 3:]) == 0
