"""Paper-equation LAN-O3DP obstacle guidance utilities.

Unlike the historical project guidance path, this module differentiates the
clean-action estimate through the denoiser all the way back to ``A_k``.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from robomimic.utils.guided_denoising_utils import (
    flatten_action_normalization_stats,
    physical_displacement_to_normalized,
    unnormalize_action_points,
)


@dataclass(frozen=True)
class PaperLanGuidanceContext:
    """Runtime inputs fixed for one reverse-diffusion action chunk."""

    current_eef_pos: Any
    obstacle_points: Any
    action_scale: Any
    action_offset: Any
    guidance_scale: float
    safety_distance: float
    enabled: bool = True
    xy_only: bool = True
    norm_epsilon: float = 1e-6
    position_indices: Any = (0, 1, 2)
    action_key: str = "abs_eef_pose_action"
    obstacle_cost_type: str = "hinge"
    obstacle_cost_smoothing: float = 0.0
    tangent_ratio: float = 0.0
    tangent_side: int = 0
    forward_direction_xy: Any = None

    def __post_init__(self):
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.safety_distance <= 0:
            raise ValueError("safety_distance must be positive")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
        if self.obstacle_cost_type not in ("hinge", "huber_hinge"):
            raise ValueError(
                "obstacle_cost_type must be 'hinge' or 'huber_hinge'"
            )
        if self.obstacle_cost_smoothing < 0:
            raise ValueError("obstacle_cost_smoothing must be non-negative")
        if self.tangent_ratio < 0:
            raise ValueError("tangent_ratio must be non-negative")
        if int(self.tangent_side) not in (-1, 0, 1):
            raise ValueError("tangent_side must be -1, 0, or 1")
        object.__setattr__(self, "tangent_side", int(self.tangent_side))
        if self.forward_direction_xy is not None:
            forward = torch.as_tensor(self.forward_direction_xy).reshape(-1)
            if forward.numel() != 2 or float(torch.linalg.vector_norm(forward)) <= 0:
                raise ValueError("forward_direction_xy must be a nonzero XY vector")
        if (
            self.obstacle_cost_type == "huber_hinge"
            and self.obstacle_cost_smoothing <= 0
        ):
            raise ValueError(
                "huber_hinge requires positive obstacle_cost_smoothing"
            )
        indices = tuple(int(index) for index in self.position_indices)
        if len(indices) not in (2, 3) or len(set(indices)) != len(indices):
            raise ValueError("position_indices must contain two or three unique indices")
        if min(indices) < 0:
            raise ValueError("position_indices must be non-negative")
        object.__setattr__(self, "position_indices", indices)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")


@dataclass(frozen=True)
class PaperLanGuidanceDiagnostics:
    timestep: int
    closest_obstacle_point: Any
    active_waypoint_count: int
    cost: float
    minimum_distance_m: float
    noisy_action_gradient_norm: float
    applied_update_norm: float
    tangent_ratio: float
    tangent_side: int

    def to_dict(self):
        return asdict(self)


def paper_lan_guidance_context_from_rollout_policy(
    rollout_policy,
    *,
    current_eef_pos,
    obstacle_points,
    guidance_scale,
    safety_distance,
    enabled=True,
    xy_only=True,
    norm_epsilon=1e-6,
    obstacle_cost_type="hinge",
    obstacle_cost_smoothing=0.0,
    tangent_ratio=0.0,
    forward_direction_xy=None,
):
    """Build paper guidance with the checkpoint's exact action statistics."""

    action_keys = list(rollout_policy.policy.global_config.train.action_keys)
    if action_keys != ["abs_eef_pose_action"]:
        raise ValueError(
            "Paper LAN guidance requires the absolute EEF action key, got {}".format(
                action_keys
            )
        )
    scale, offset = flatten_action_normalization_stats(
        rollout_policy.action_normalization_stats,
        action_keys,
    )
    return PaperLanGuidanceContext(
        current_eef_pos=current_eef_pos,
        obstacle_points=obstacle_points,
        action_scale=scale,
        action_offset=offset,
        guidance_scale=guidance_scale,
        safety_distance=safety_distance,
        enabled=enabled,
        xy_only=xy_only,
        norm_epsilon=norm_epsilon,
        obstacle_cost_type=obstacle_cost_type,
        obstacle_cost_smoothing=obstacle_cost_smoothing,
        tangent_ratio=tangent_ratio,
        forward_direction_xy=forward_direction_xy,
    )


def point_trajectory_guidance_context_from_rollout_policy(
    rollout_policy,
    *,
    current_position,
    obstacle_points,
    guidance_scale,
    safety_distance,
    action_key,
    position_indices,
    enabled=True,
    norm_epsilon=1e-6,
    obstacle_cost_type="hinge",
    obstacle_cost_smoothing=0.0,
    tangent_ratio=0.0,
    forward_direction_xy=None,
):
    """Build generic point guidance in flattened policy-action coordinates."""

    action_keys = list(rollout_policy.policy.global_config.train.action_keys)
    if action_key not in action_keys:
        raise ValueError(
            "Guidance action key '{}' is absent from {}".format(action_key, action_keys)
        )
    scale, offset = flatten_action_normalization_stats(
        rollout_policy.action_normalization_stats,
        action_keys,
    )
    flat_offset = 0
    for key in action_keys:
        width = int(
            torch.as_tensor(
                rollout_policy.action_normalization_stats[key]["scale"]
            ).numel()
        )
        if key == action_key:
            break
        flat_offset += width
    local_indices = tuple(int(index) for index in position_indices)
    return PaperLanGuidanceContext(
        current_eef_pos=current_position,
        obstacle_points=obstacle_points,
        action_scale=scale,
        action_offset=offset,
        guidance_scale=guidance_scale,
        safety_distance=safety_distance,
        enabled=enabled,
        xy_only=False,
        norm_epsilon=norm_epsilon,
        position_indices=tuple(flat_offset + index for index in local_indices),
        action_key=action_key,
        obstacle_cost_type=obstacle_cost_type,
        obstacle_cost_smoothing=obstacle_cost_smoothing,
        tangent_ratio=tangent_ratio,
        forward_direction_xy=forward_direction_xy,
    )


def predicted_clean_action(noisy_action, model_output, timestep, scheduler):
    """Compute the scheduler's ``A_0|k`` without breaking its autograd graph."""

    timestep_index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    alpha_bar = scheduler.alphas_cumprod[timestep_index].to(
        device=noisy_action.device,
        dtype=noisy_action.dtype,
    )
    beta_bar = 1.0 - alpha_bar
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        clean = (
            noisy_action - beta_bar.sqrt() * model_output
        ) / alpha_bar.sqrt()
    elif prediction_type == "sample":
        clean = model_output
    elif prediction_type == "v_prediction":
        clean = alpha_bar.sqrt() * noisy_action - beta_bar.sqrt() * model_output
    else:
        raise ValueError(
            "Unsupported scheduler prediction type: {}".format(prediction_type)
        )
    if bool(getattr(scheduler.config, "clip_sample", False)):
        clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        clean = clean.clamp(-clip_range, clip_range)
    return clean


def closest_obstacle_point(
    current_eef_pos,
    obstacle_points,
    *,
    xy_only=True,
    dimensions=None,
):
    """Select ``C_ob`` once from the visible obstacle cloud, as in Algorithm 1."""

    points = torch.as_tensor(obstacle_points)
    current = torch.as_tensor(
        current_eef_pos,
        device=points.device,
        dtype=points.dtype,
    ).reshape(-1)
    if points.ndim != 2 or points.shape[-1] not in (2, 3) or points.shape[0] == 0:
        raise ValueError("obstacle_points must have non-empty shape [N, 2 or 3]")
    if dimensions is None:
        dimensions = 2 if xy_only else points.shape[-1]
    dimensions = int(dimensions)
    if dimensions not in (2, 3) or points.shape[-1] < dimensions:
        raise ValueError("Requested point dimensions are unavailable")
    if current.numel() < dimensions:
        raise ValueError("current position has insufficient dimensions")
    distances = torch.linalg.vector_norm(
        points[:, :dimensions] - current[:dimensions],
        dim=-1,
    )
    return points[torch.argmin(distances)]


def paper_obstacle_cost(
    raw_action,
    obstacle_point,
    *,
    safety_distance,
    xy_only=True,
    norm_epsilon=1e-6,
    position_indices=None,
    cost_type="hinge",
    smoothing=0.0,
):
    """Thresholded waypoint cost whose negative gradient pushes outward.

    ``huber_hinge`` preserves the unit-gradient deep-penetration regime of the
    paper hinge while ramping its gradient continuously from zero over the
    first ``smoothing`` metres inside the safety boundary.
    """

    if position_indices is None:
        position_indices = (0, 1, 2)
    indices = tuple(int(index) for index in position_indices)
    if max(indices) >= raw_action.shape[-1]:
        raise ValueError("Position index lies outside the action vector")
    dimensions = min(2, len(indices)) if xy_only else len(indices)
    point = torch.as_tensor(
        obstacle_point,
        device=raw_action.device,
        dtype=raw_action.dtype,
    ).reshape(-1)
    if point.numel() < dimensions:
        raise ValueError("obstacle_point has insufficient dimensions")
    selected = raw_action[..., list(indices[:dimensions])]
    delta = selected - point[:dimensions]
    distances = torch.sqrt(torch.sum(delta.square(), dim=-1) + norm_epsilon ** 2)
    penetrations = torch.relu(float(safety_distance) - distances)
    if cost_type == "hinge":
        costs = penetrations
    elif cost_type == "huber_hinge":
        smoothing = float(smoothing)
        if smoothing <= 0:
            raise ValueError("huber_hinge requires positive smoothing")
        quadratic = penetrations.square() / (2.0 * smoothing)
        linear = penetrations - 0.5 * smoothing
        costs = torch.where(penetrations < smoothing, quadratic, linear)
    else:
        raise ValueError("cost_type must be 'hinge' or 'huber_hinge'")
    return costs.sum(), distances, penetrations


def apply_output_tangent_guidance(normalized_action, context, *, force_active=False):
    """Add a tangential displacement directly to a sampled action trajectory.

    This post-processing path intentionally bypasses the denoiser Jacobian.
    The two tangent orientations are compared directly with the sampled
    trajectory's forward direction, and the forward-aligned side is applied in
    physical XY metres. Other action coordinates are preserved exactly.
    """

    if context.tangent_ratio <= 0 or not context.enabled:
        return normalized_action, 0
    indices = tuple(int(index) for index in context.position_indices)
    if len(indices) < 2 or max(indices[:2]) >= normalized_action.shape[-1]:
        raise ValueError("Tangential guidance requires two valid XY indices")

    raw_action = unnormalize_action_points(
        normalized_action=normalized_action,
        action_scale=context.action_scale,
        action_offset=context.action_offset,
    )
    selected = raw_action[..., list(indices[:2])]
    point = closest_obstacle_point(
        current_eef_pos=context.current_eef_pos,
        obstacle_points=torch.as_tensor(
            context.obstacle_points,
            device=normalized_action.device,
            dtype=normalized_action.dtype,
        ),
        xy_only=True,
        dimensions=2,
    )
    current = torch.as_tensor(
        context.current_eef_pos,
        device=normalized_action.device,
        dtype=normalized_action.dtype,
    ).reshape(-1)
    if current.numel() < 2:
        raise ValueError("Current position must contain XY")

    delta = selected - point[:2]
    distances = torch.sqrt(
        torch.sum(delta.square(), dim=-1) + context.norm_epsilon ** 2
    )
    penetrations = torch.relu(float(context.safety_distance) - distances)
    output_active = penetrations > 0
    if not bool(torch.any(output_active)) and not force_active:
        return normalized_action, 0

    current_delta = current[:2] - point[:2]
    current_distance = torch.sqrt(
        torch.sum(current_delta.square()) + context.norm_epsilon ** 2
    )
    current_outward = current_delta / current_distance
    reference_tangent = torch.stack(
        (-current_outward[1], current_outward[0]), dim=-1
    ).reshape(1, 2)
    if context.forward_direction_xy is None:
        forward = selected[..., -1, :] - current[:2]
    else:
        forward = torch.as_tensor(
            context.forward_direction_xy,
            device=normalized_action.device,
            dtype=normalized_action.dtype,
        ).reshape(1, 2)
        forward = forward.expand(selected.shape[0], -1)
    alignment = torch.sum(forward * reference_tangent, dim=-1)
    if context.tangent_side == 0:
        side = torch.where(
            torch.abs(alignment) > context.norm_epsilon,
            torch.sign(alignment),
            torch.ones_like(alignment),
        )
    else:
        side = torch.full_like(alignment, float(context.tangent_side))
    tangent = reference_tangent * side.unsqueeze(-1)

    smoothing = float(context.obstacle_cost_smoothing)
    if smoothing > 0:
        # Cubic smoothstep over [Q - smoothing, Q + smoothing]. At the
        # threshold the direct offset is half strength, and both endpoints
        # have zero slope, avoiding the old binary 0-to-max jump.
        phase = torch.clamp(
            (
                float(context.safety_distance)
                + smoothing
                - current_distance
            )
            / (2.0 * smoothing),
            min=0.0,
            max=1.0,
        )
        activation = phase.square() * (3.0 - 2.0 * phase)
    else:
        activation = (current_distance < float(context.safety_distance)).to(
            normalized_action.dtype
        )
    if float(activation.detach().item()) <= 0:
        return normalized_action, 0

    physical_displacement = torch.zeros_like(raw_action)
    physical_displacement[..., list(indices[:2])] = (
        float(context.tangent_ratio)
        * float(context.safety_distance)
        * activation
        * tangent.unsqueeze(-2)
    )
    normalized_displacement = physical_displacement_to_normalized(
        physical_displacement,
        context.action_scale,
    )
    selected_side = int(side.detach().reshape(-1)[0].item())
    return normalized_action + normalized_displacement, selected_side


def paper_guidance_gradient(
    *,
    noisy_action,
    model_output,
    timestep,
    scheduler,
    context,
):
    """Return ``rho * grad_{A_k} D(A_0|k, C_ob)`` and diagnostics."""

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
    obstacle_point = closest_obstacle_point(
        current_eef_pos=context.current_eef_pos,
        obstacle_points=torch.as_tensor(
            context.obstacle_points,
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        ),
        xy_only=context.xy_only,
        dimensions=(
            min(2, len(context.position_indices))
            if context.xy_only
            else len(context.position_indices)
        ),
    )
    cost, distances, penetrations = paper_obstacle_cost(
        raw_action=raw_clean,
        obstacle_point=obstacle_point,
        safety_distance=context.safety_distance,
        xy_only=context.xy_only,
        norm_epsilon=context.norm_epsilon,
        position_indices=context.position_indices,
        cost_type=context.obstacle_cost_type,
        smoothing=context.obstacle_cost_smoothing,
    )
    active_waypoint_count = int(torch.count_nonzero(penetrations > 0).item())
    if active_waypoint_count:
        gradient = torch.autograd.grad(
            cost,
            noisy_action,
            retain_graph=False,
            create_graph=False,
        )[0]
    else:
        # ReLU is locally constant when every waypoint is outside Q*. Avoid a
        # mathematically redundant backward pass through the full denoiser.
        gradient = torch.zeros_like(noisy_action)
    update = float(context.guidance_scale) * gradient
    timestep_index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    diagnostics = PaperLanGuidanceDiagnostics(
        timestep=timestep_index,
        closest_obstacle_point=obstacle_point.detach().cpu().tolist(),
        active_waypoint_count=active_waypoint_count,
        cost=float(cost.detach().item()),
        minimum_distance_m=float(torch.min(distances).detach().item()),
        noisy_action_gradient_norm=float(torch.linalg.vector_norm(gradient).detach().item()),
        applied_update_norm=float(torch.linalg.vector_norm(update).detach().item()),
        tangent_ratio=float(context.tangent_ratio),
        tangent_side=0,
    )
    return update.detach(), diagnostics


def paper_guided_policy_from_checkpoint(
    device=None,
    ckpt_path=None,
    ckpt_dict=None,
    verbose=False,
):
    """Load LAN weights into the exact paper-gradient inference class in memory."""

    from robomimic.utils import file_utils as FileUtils

    source = FileUtils.maybe_dict_from_checkpoint(
        ckpt_path=ckpt_path,
        ckpt_dict=ckpt_dict,
    )
    source_algo = source.get("algo_name")
    supported = {"lan_o3dp", "paper_guided_lan_o3dp"}
    if source_algo not in supported:
        raise ValueError(
            "Expected a LAN-O3DP checkpoint, got '{}'".format(source_algo)
        )
    adapted = dict(source)
    config_dict = json.loads(source["config"])
    config_dict["algo_name"] = "paper_guided_lan_o3dp"
    adapted["algo_name"] = "paper_guided_lan_o3dp"
    adapted["config"] = json.dumps(config_dict)
    for stats_key in ("obs_normalization_stats", "action_normalization_stats"):
        if stats_key in source:
            adapted[stats_key] = {
                key: {name: list(value) for name, value in values.items()}
                for key, values in source[stats_key].items()
            }
    return FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_dict=adapted,
        verbose=verbose,
    )


def point_guided_policy_from_checkpoint(
    device=None,
    ckpt_path=None,
    ckpt_dict=None,
    verbose=False,
):
    """Load a standard low-dimensional policy into generic point guidance."""

    from robomimic.utils import file_utils as FileUtils

    source = FileUtils.maybe_dict_from_checkpoint(
        ckpt_path=ckpt_path,
        ckpt_dict=ckpt_dict,
    )
    source_algo = source.get("algo_name")
    supported = {"diffusion_policy", "point_guided_diffusion_policy"}
    if source_algo not in supported:
        raise ValueError(
            "Expected a Diffusion Policy checkpoint, got '{}'".format(source_algo)
        )
    adapted = dict(source)
    config_dict = json.loads(source["config"])
    config_dict["algo_name"] = "point_guided_diffusion_policy"
    adapted["algo_name"] = "point_guided_diffusion_policy"
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


PointTrajectoryGuidanceContext = PaperLanGuidanceContext
PointTrajectoryGuidanceDiagnostics = PaperLanGuidanceDiagnostics
