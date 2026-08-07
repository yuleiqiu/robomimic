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

    def __post_init__(self):
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.safety_distance <= 0:
            raise ValueError("safety_distance must be positive")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")
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
):
    """Thresholded waypoint cost whose negative gradient pushes outward."""

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
    return penetrations.sum(), distances, penetrations


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
