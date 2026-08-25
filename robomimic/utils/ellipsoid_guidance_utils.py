"""Minimum-volume ellipsoids and differentiable swept-body guidance costs."""

import json
import itertools
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch

from robomimic.utils.guided_denoising_utils import (
    flatten_action_normalization_stats,
    unnormalize_action_points,
)
from robomimic.utils.paper_lan_guidance_utils import predicted_clean_action


@dataclass(frozen=True)
class MinimumVolumeEllipsoid:
    """Ellipsoid ``(x-c)^T A (x-c) <= 1`` in world coordinates."""

    center: np.ndarray
    rotation: np.ndarray
    semi_axes: np.ndarray
    quadratic: np.ndarray
    iterations: int
    converged: bool
    maximum_input_radius: float

    def to_dict(self):
        return {
            "center": self.center.tolist(),
            "rotation": self.rotation.tolist(),
            "semi_axes": self.semi_axes.tolist(),
            "quadratic": self.quadratic.tolist(),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "maximum_input_radius": float(self.maximum_input_radius),
        }


def robosuite_geom_surface_points(raw_env, geom_ids, *, cylinder_segments=32):
    """Sample collision geometry in world coordinates from compiled MuJoCo data.

    Mesh vertices and cylinder rims are sampled directly. Other primitive
    types use their compiled local AABB corners, yielding a conservative actor
    envelope rather than silently under-approximating collision geometry.
    """

    cylinder_segments = int(cylinder_segments)
    if cylinder_segments < 8:
        raise ValueError("cylinder_segments must be at least 8")
    geom_ids = tuple(sorted(set(int(geom_id) for geom_id in geom_ids)))
    if not geom_ids:
        raise ValueError("geom_ids must be non-empty")
    model = raw_env.sim.model
    data = raw_env.sim.data
    points = []
    for geom_id in geom_ids:
        if geom_id < 0 or geom_id >= model.ngeom:
            raise ValueError("geom id {} is outside the model".format(geom_id))
        geom_type = int(model.geom_type[geom_id])
        if geom_type == 7 and int(model.geom_dataid[geom_id]) >= 0:
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            end = start + int(model.mesh_vertnum[mesh_id])
            local = np.asarray(model.mesh_vert[start:end], dtype=np.float64)
        elif geom_type == 5:
            center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
            radius, half_height = np.asarray(
                model.geom_size[geom_id, :2],
                dtype=np.float64,
            )
            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                cylinder_segments,
                endpoint=False,
            )
            local = np.asarray(
                [
                    center + [radius * np.cos(angle), radius * np.sin(angle), z]
                    for z in (-half_height, half_height)
                    for angle in angles
                ]
            )
        else:
            center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
            half_size = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64)
            local = np.asarray(
                [
                    center + half_size * np.asarray(signs)
                    for signs in itertools.product((-1.0, 1.0), repeat=3)
                ]
            )
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        points.append(position + local @ rotation.T)
    return np.unique(np.concatenate(points, axis=0), axis=0)


def fit_robosuite_geom_ellipsoid(
    raw_env,
    geom_ids,
    *,
    reference_position=None,
    reference_rotation=None,
    **mvee_kwargs,
):
    """Fit collision geoms in world or an explicitly supplied local frame."""

    points = robosuite_geom_surface_points(raw_env, geom_ids)
    if (reference_position is None) != (reference_rotation is None):
        raise ValueError("reference position and rotation must be supplied together")
    if reference_position is not None:
        reference_position = np.asarray(reference_position, dtype=np.float64)
        reference_rotation = np.asarray(reference_rotation, dtype=np.float64)
        if reference_position.shape != (3,) or reference_rotation.shape != (3, 3):
            raise ValueError("reference pose must have shapes [3] and [3, 3]")
        points = (points - reference_position) @ reference_rotation
    return fit_minimum_volume_enclosing_ellipsoid(points, **mvee_kwargs)


def fit_minimum_volume_enclosing_ellipsoid(
    points,
    *,
    tolerance=1e-4,
    max_iterations=20000,
    padding=0.0,
    rank_epsilon=1e-10,
):
    """Fit a deterministic full-dimensional MVEE with Khachiyan updates.

    The final axes are inflated by the measured numerical constraint violation,
    so every supplied point is enclosed even when the optimizer terminates at a
    finite tolerance. ``padding`` then expands every semi-axis in metres.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape [N, D] with D >= 2")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    tolerance = float(tolerance)
    padding = float(padding)
    rank_epsilon = float(rank_epsilon)
    max_iterations = int(max_iterations)
    if tolerance <= 0 or rank_epsilon <= 0:
        raise ValueError("tolerance and rank_epsilon must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    points = np.unique(points, axis=0)
    count, dimensions = points.shape
    if count < dimensions + 1:
        raise ValueError("MVEE requires at least D + 1 unique points")
    centered = points - points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_threshold = rank_epsilon * max(float(singular_values[0]), 1.0)
    if int(np.count_nonzero(singular_values > rank_threshold)) < dimensions:
        raise ValueError("MVEE points must span all D dimensions")

    homogeneous = np.vstack([points.T, np.ones(count, dtype=np.float64)])
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        moment = (homogeneous * weights) @ homogeneous.T
        inverse_times_points = np.linalg.solve(moment, homogeneous)
        leverage = np.sum(homogeneous * inverse_times_points, axis=0)
        selected = int(np.argmax(leverage))
        maximum = float(leverage[selected])
        step = (maximum - dimensions - 1.0) / (
            (dimensions + 1.0) * (maximum - 1.0)
        )
        if step <= tolerance:
            converged = True
            break
        step = float(np.clip(step, 0.0, 1.0))
        weights *= 1.0 - step
        weights[selected] += step
    if not converged:
        raise RuntimeError(
            "MVEE did not converge in {} iterations at tolerance {}".format(
                max_iterations,
                tolerance,
            )
        )

    center = weights @ points
    offsets = points - center
    covariance = offsets.T @ (weights[:, None] * offsets)
    quadratic = np.linalg.inv(covariance) / dimensions
    quadratic = 0.5 * (quadratic + quadratic.T)

    normalized_squared = np.einsum(
        "ni,ij,nj->n",
        offsets,
        quadratic,
        offsets,
    )
    numerical_inflation = np.sqrt(max(1.0, float(normalized_squared.max())))
    eigenvalues, rotation = np.linalg.eigh(quadratic)
    if np.any(eigenvalues <= 0):
        raise RuntimeError("MVEE quadratic matrix is not positive definite")
    order = np.argsort(1.0 / np.sqrt(eigenvalues))[::-1]
    rotation = rotation[:, order]
    semi_axes = (1.0 / np.sqrt(eigenvalues[order])) * numerical_inflation
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    semi_axes += padding
    quadratic = rotation @ np.diag(1.0 / np.square(semi_axes)) @ rotation.T

    final_squared = np.einsum(
        "ni,ij,nj->n",
        offsets,
        quadratic,
        offsets,
    )
    return MinimumVolumeEllipsoid(
        center=center,
        rotation=rotation,
        semi_axes=semi_axes,
        quadratic=quadratic,
        iterations=iterations,
        converged=converged,
        maximum_input_radius=float(np.sqrt(max(0.0, final_squared.max()))),
    )


def rotation_vector_to_matrix(rotation_vector):
    """Differentiable SO(3) exponential for rotation vectors ``[..., 3]``."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must have final dimension 3")
    x, y, z = rotation_vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=-1,
    ).reshape(rotation_vector.shape[:-1] + (3, 3))
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1)
    sine_over_angle = torch.sinc(angle / torch.pi)
    one_minus_cosine_over_angle_squared = 0.5 * torch.sinc(
        angle / (2.0 * torch.pi)
    ).square()
    identity = torch.eye(
        3,
        device=rotation_vector.device,
        dtype=rotation_vector.dtype,
    ).expand(rotation_vector.shape[:-1] + (3, 3))
    return (
        identity
        + sine_over_angle[..., None, None] * skew
        + one_minus_cosine_over_angle_squared[..., None, None]
        * torch.matmul(skew, skew)
    )


def reconstruct_delta_eef_poses(
    current_eef_position,
    current_eef_rotation,
    delta_positions,
    delta_rotation_vectors,
):
    """Reconstruct world EEF centers and rotations from executable deltas."""

    if delta_positions.ndim != 3 or delta_positions.shape[-1] != 3:
        raise ValueError("delta_positions must have shape [B, T, 3]")
    if delta_rotation_vectors.shape != delta_positions.shape:
        raise ValueError("delta rotation vectors must match delta positions")
    batch_size = delta_positions.shape[0]
    position = torch.as_tensor(
        current_eef_position,
        device=delta_positions.device,
        dtype=delta_positions.dtype,
    )
    if position.ndim == 1:
        position = position.unsqueeze(0)
    if position.shape == (1, 3) and batch_size > 1:
        position = position.expand(batch_size, -1)
    if position.shape != (batch_size, 3):
        raise ValueError("current EEF position must have shape [3] or [B, 3]")

    rotation = torch.as_tensor(
        current_eef_rotation,
        device=delta_positions.device,
        dtype=delta_positions.dtype,
    )
    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(0)
    if rotation.shape == (1, 3, 3) and batch_size > 1:
        rotation = rotation.expand(batch_size, -1, -1)
    if rotation.shape != (batch_size, 3, 3):
        raise ValueError("current EEF rotation must have shape [3, 3] or [B, 3, 3]")

    positions = position.unsqueeze(1) + torch.cumsum(delta_positions, dim=1)
    delta_rotations = rotation_vector_to_matrix(delta_rotation_vectors)
    rotations = []
    running = rotation
    for step in range(delta_rotations.shape[1]):
        running = torch.matmul(delta_rotations[:, step], running)
        rotations.append(running)
    return positions, torch.stack(rotations, dim=1)


def transform_local_ellipsoids(
    eef_positions,
    eef_rotations,
    local_centers,
    local_rotations,
):
    """Transform ``K`` EEF-local actor ellipsoids over a ``[B,T]`` trajectory."""

    local_centers = torch.as_tensor(
        local_centers,
        device=eef_positions.device,
        dtype=eef_positions.dtype,
    )
    local_rotations = torch.as_tensor(
        local_rotations,
        device=eef_positions.device,
        dtype=eef_positions.dtype,
    )
    if local_centers.ndim != 2 or local_centers.shape[-1] != 3:
        raise ValueError("local_centers must have shape [K, 3]")
    if local_rotations.shape != (local_centers.shape[0], 3, 3):
        raise ValueError("local_rotations must have shape [K, 3, 3]")
    world_centers = eef_positions.unsqueeze(2) + torch.einsum(
        "btij,kj->btki",
        eef_rotations,
        local_centers,
    )
    world_rotations = torch.einsum(
        "btij,kjl->btkil",
        eef_rotations,
        local_rotations,
    )
    return world_centers, world_rotations


def ellipsoid_support_radius(direction, rotation, semi_axes):
    """Support radius of an oriented ellipsoid along world ``direction``."""

    rotation = torch.as_tensor(
        rotation,
        device=direction.device,
        dtype=direction.dtype,
    )
    semi_axes = torch.as_tensor(
        semi_axes,
        device=direction.device,
        dtype=direction.dtype,
    )
    if direction.shape[-1] != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("directions and rotations must be 3-D")
    if semi_axes.shape[-1] != 3 or torch.any(semi_axes <= 0):
        raise ValueError("semi_axes must be positive with final dimension 3")
    local_direction = torch.matmul(
        rotation.transpose(-1, -2),
        direction.unsqueeze(-1),
    ).squeeze(-1)
    return torch.sqrt(torch.sum((semi_axes * local_direction).square(), dim=-1))


def directional_ellipsoid_clearance(
    actor_centers,
    actor_rotations,
    actor_semi_axes,
    obstacle_center,
    obstacle_rotation,
    obstacle_semi_axes,
    *,
    clearance_margin=0.0,
    norm_epsilon=1e-8,
):
    """Conservative center-line support-plane clearance in metres."""

    if norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be positive")
    obstacle_center = torch.as_tensor(
        obstacle_center,
        device=actor_centers.device,
        dtype=actor_centers.dtype,
    )
    delta = actor_centers - obstacle_center
    distances = torch.linalg.vector_norm(delta, dim=-1)
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1.0
    direction = torch.where(
        (distances > norm_epsilon).unsqueeze(-1),
        delta / torch.clamp(distances.unsqueeze(-1), min=norm_epsilon),
        fallback,
    )
    actor_radius = ellipsoid_support_radius(
        direction,
        actor_rotations,
        actor_semi_axes,
    )
    obstacle_radius = ellipsoid_support_radius(
        direction,
        obstacle_rotation,
        obstacle_semi_axes,
    )
    return (
        distances
        - actor_radius
        - obstacle_radius
        - float(clearance_margin)
    )


def ellipsoid_separation_cost(
    actor_centers,
    actor_rotations,
    actor_semi_axes,
    obstacle_center,
    obstacle_rotation,
    obstacle_semi_axes,
    *,
    clearance_margin=0.0,
    cost_type="huber_hinge",
    smoothing=0.01,
    norm_epsilon=1e-8,
):
    """Hinge or Huberized-hinge penalty for actor-obstacle overlap."""

    clearances = directional_ellipsoid_clearance(
        actor_centers,
        actor_rotations,
        actor_semi_axes,
        obstacle_center,
        obstacle_rotation,
        obstacle_semi_axes,
        clearance_margin=clearance_margin,
        norm_epsilon=norm_epsilon,
    )
    penetrations = torch.relu(-clearances)
    if cost_type == "hinge":
        costs = penetrations
    elif cost_type == "huber_hinge":
        smoothing = float(smoothing)
        if smoothing <= 0:
            raise ValueError("huber_hinge requires positive smoothing")
        costs = torch.where(
            penetrations < smoothing,
            penetrations.square() / (2.0 * smoothing),
            penetrations - 0.5 * smoothing,
        )
    else:
        raise ValueError("cost_type must be 'hinge' or 'huber_hinge'")
    return costs.sum(), penetrations, clearances


@dataclass(frozen=True)
class EllipsoidGuidanceContext:
    """Runtime geometry fixed during one reverse-diffusion action chunk."""

    current_eef_position: Any
    current_eef_rotation: Any
    actor_local_centers: Any
    actor_local_rotations: Any
    actor_semi_axes: Any
    obstacle_center: Any
    obstacle_rotation: Any
    obstacle_semi_axes: Any
    action_scale: Any
    action_offset: Any
    guidance_scale: float
    clearance_margin: float = 0.0
    cost_type: str = "huber_hinge"
    cost_smoothing: float = 0.01
    enabled: bool = True
    norm_epsilon: float = 1e-8
    position_indices: Any = (0, 1, 2)
    rotation_indices: Any = (3, 4, 5)
    action_key: str = "delta_eef_pose_action"
    actor_labels: Any = None

    def __post_init__(self):
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.clearance_margin < 0:
            raise ValueError("clearance_margin must be non-negative")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        if self.cost_type not in ("hinge", "huber_hinge"):
            raise ValueError("cost_type must be 'hinge' or 'huber_hinge'")
        if self.cost_type == "huber_hinge" and self.cost_smoothing <= 0:
            raise ValueError("huber_hinge requires positive cost_smoothing")
        position_indices = tuple(int(index) for index in self.position_indices)
        rotation_indices = tuple(int(index) for index in self.rotation_indices)
        if len(position_indices) != 3 or len(set(position_indices)) != 3:
            raise ValueError("position_indices must contain three unique indices")
        if len(rotation_indices) != 3 or len(set(rotation_indices)) != 3:
            raise ValueError("rotation_indices must contain three unique indices")
        if min(position_indices + rotation_indices) < 0:
            raise ValueError("action indices must be non-negative")
        object.__setattr__(self, "position_indices", position_indices)
        object.__setattr__(self, "rotation_indices", rotation_indices)

        centers = np.asarray(self.actor_local_centers)
        rotations = np.asarray(self.actor_local_rotations)
        axes = np.asarray(self.actor_semi_axes)
        if centers.ndim != 2 or centers.shape[-1] != 3 or centers.shape[0] == 0:
            raise ValueError("actor_local_centers must have non-empty shape [K, 3]")
        if rotations.shape != (centers.shape[0], 3, 3):
            raise ValueError("actor_local_rotations must have shape [K, 3, 3]")
        if axes.shape != (centers.shape[0], 3) or np.any(axes <= 0):
            raise ValueError("actor_semi_axes must have positive shape [K, 3]")
        labels = self.actor_labels
        if labels is None:
            labels = tuple("actor_{}".format(index) for index in range(len(centers)))
        else:
            labels = tuple(str(label) for label in labels)
            if len(labels) != len(centers) or any(not label for label in labels):
                raise ValueError("actor_labels must name every actor ellipsoid")
        object.__setattr__(self, "actor_labels", labels)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")


@dataclass(frozen=True)
class EllipsoidGuidanceDiagnostics:
    timestep: int
    active_actor_waypoint_count: int
    cost: float
    minimum_clearance_m: float
    actor_minimum_clearances_m: Any
    noisy_action_gradient_norm: float
    applied_update_norm: float

    def to_dict(self):
        return asdict(self)


def ellipsoid_guidance_context_from_rollout_policy(
    rollout_policy,
    *,
    current_eef_position,
    current_eef_rotation,
    actor_local_centers,
    actor_local_rotations,
    actor_semi_axes,
    obstacle_center,
    obstacle_rotation,
    obstacle_semi_axes,
    guidance_scale,
    clearance_margin=0.0,
    cost_type="huber_hinge",
    cost_smoothing=0.01,
    enabled=True,
    actor_labels=None,
):
    """Build ellipsoid guidance with exact checkpoint normalization stats."""

    action_keys = list(rollout_policy.policy.global_config.train.action_keys)
    action_key = "delta_eef_pose_action"
    if action_keys != [action_key]:
        raise ValueError(
            "Ellipsoid guidance requires only {}, got {}".format(
                action_key,
                action_keys,
            )
        )
    scale, offset = flatten_action_normalization_stats(
        rollout_policy.action_normalization_stats,
        action_keys,
    )
    return EllipsoidGuidanceContext(
        current_eef_position=current_eef_position,
        current_eef_rotation=current_eef_rotation,
        actor_local_centers=actor_local_centers,
        actor_local_rotations=actor_local_rotations,
        actor_semi_axes=actor_semi_axes,
        obstacle_center=obstacle_center,
        obstacle_rotation=obstacle_rotation,
        obstacle_semi_axes=obstacle_semi_axes,
        action_scale=scale,
        action_offset=offset,
        guidance_scale=guidance_scale,
        clearance_margin=clearance_margin,
        cost_type=cost_type,
        cost_smoothing=cost_smoothing,
        enabled=enabled,
        actor_labels=actor_labels,
    )


def ellipsoid_guidance_gradient(
    *,
    noisy_action,
    model_output,
    timestep,
    scheduler,
    context,
    observation_horizon,
    action_horizon,
):
    """Differentiate executed predicted-clean swept ellipsoids to ``A_k``."""

    if not noisy_action.requires_grad:
        raise ValueError("noisy_action must require gradients")
    clean = predicted_clean_action(
        noisy_action=noisy_action,
        model_output=model_output,
        timestep=timestep,
        scheduler=scheduler,
    )
    raw_clean = unnormalize_action_points(
        normalized_action=clean,
        action_scale=context.action_scale,
        action_offset=context.action_offset,
    )
    start = int(observation_horizon) - 1
    end = start + int(action_horizon)
    if start < 0 or end > raw_clean.shape[1]:
        raise ValueError("Executed action slice is outside the prediction horizon")
    all_indices = context.position_indices + context.rotation_indices
    if max(all_indices) >= raw_clean.shape[-1]:
        raise ValueError("Ellipsoid guidance action index is outside the action vector")
    executable = raw_clean[:, start:end]
    delta_positions = executable[..., list(context.position_indices)]
    delta_rotations = executable[..., list(context.rotation_indices)]
    eef_positions, eef_rotations = reconstruct_delta_eef_poses(
        context.current_eef_position,
        context.current_eef_rotation,
        delta_positions,
        delta_rotations,
    )
    actor_centers, actor_rotations = transform_local_ellipsoids(
        eef_positions,
        eef_rotations,
        context.actor_local_centers,
        context.actor_local_rotations,
    )
    actor_axes = torch.as_tensor(
        context.actor_semi_axes,
        device=noisy_action.device,
        dtype=noisy_action.dtype,
    ).reshape(1, 1, -1, 3)
    cost, penetrations, clearances = ellipsoid_separation_cost(
        actor_centers,
        actor_rotations,
        actor_axes,
        context.obstacle_center,
        context.obstacle_rotation,
        context.obstacle_semi_axes,
        clearance_margin=context.clearance_margin,
        cost_type=context.cost_type,
        smoothing=context.cost_smoothing,
        norm_epsilon=context.norm_epsilon,
    )
    active_count = int(torch.count_nonzero(penetrations > 0).item())
    if active_count:
        gradient = torch.autograd.grad(
            cost,
            noisy_action,
            retain_graph=False,
            create_graph=False,
        )[0]
    else:
        gradient = torch.zeros_like(noisy_action)
    update = float(context.guidance_scale) * gradient
    actor_minimum = torch.amin(clearances, dim=(0, 1)).detach().cpu().tolist()
    diagnostics = EllipsoidGuidanceDiagnostics(
        timestep=int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
        active_actor_waypoint_count=active_count,
        cost=float(cost.detach().item()),
        minimum_clearance_m=float(torch.min(clearances).detach().item()),
        actor_minimum_clearances_m={
            label: float(value)
            for label, value in zip(context.actor_labels, actor_minimum)
        },
        noisy_action_gradient_norm=float(
            torch.linalg.vector_norm(gradient).detach().item()
        ),
        applied_update_norm=float(torch.linalg.vector_norm(update).detach().item()),
    )
    return update.detach(), diagnostics


def ellipsoid_guided_policy_from_checkpoint(
    device=None,
    ckpt_path=None,
    ckpt_dict=None,
    verbose=False,
):
    """Load a delta-EEF Diffusion Policy into ellipsoid guidance in memory."""

    from robomimic.utils import file_utils as FileUtils

    source = FileUtils.maybe_dict_from_checkpoint(
        ckpt_path=ckpt_path,
        ckpt_dict=ckpt_dict,
    )
    source_algo = source.get("algo_name")
    supported = {"diffusion_policy", "ellipsoid_guided_diffusion_policy"}
    if source_algo not in supported:
        raise ValueError(
            "Expected a Diffusion Policy checkpoint, got '{}'".format(source_algo)
        )
    adapted = dict(source)
    config_dict = json.loads(source["config"])
    target_algo = "ellipsoid_guided_diffusion_policy"
    config_dict["algo_name"] = target_algo
    adapted["algo_name"] = target_algo
    adapted["config"] = json.dumps(config_dict)
    for stats_key in ("obs_normalization_stats", "action_normalization_stats"):
        if stats_key in source and source[stats_key] is not None:
            adapted[stats_key] = {
                key: {name: list(value) for name, value in values.items()}
                for key, values in source[stats_key].items()
            }
    return FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_dict=adapted,
        verbose=verbose,
    )
