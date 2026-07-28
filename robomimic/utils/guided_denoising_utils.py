"""Utilities for deployment-time guidance of delta-EEF diffusion policies."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass(frozen=True)
class GuidedDenoisingContext:
    """Runtime-only inputs for one guided action-chunk sample."""

    current_eef_pos: Any
    obstacle_centers: Any
    obstacle_radii: Any
    action_scale: Any
    action_offset: Any
    guidance_scale: float
    clearance_margin: float = 0.02
    enabled: bool = True
    normalize_waypoint_gradient: bool = False
    max_waypoint_displacement_m: Optional[float] = None
    norm_epsilon: float = 1e-6
    cost_type: str = "penetration"
    max_guidance_timestep: Optional[int] = None

    def __post_init__(self):
        if self.cost_type not in ("penetration", "vector_field"):
            raise ValueError("Unsupported guidance cost type '{}'".format(self.cost_type))
        if self.max_guidance_timestep is not None and self.max_guidance_timestep < 0:
            raise ValueError("max_guidance_timestep must be non-negative when provided")


@dataclass(frozen=True)
class GuidanceStepDiagnostics:
    timestep: int
    previous_timestep: int
    scheduler_factor: float
    active_penetration_count: int
    cost: float
    min_clearance_m: float
    raw_physical_gradient_norm: float
    physical_waypoint_displacement_norm: float
    max_waypoint_displacement_m: float
    normalized_applied_update_norm: float
    resulting_clean_action_cost: float
    before_waypoints_m: Any = None
    after_waypoints_m: Any = None
    waypoint_displacements_m: Any = None

    def to_dict(self):
        return asdict(self)


def _as_reference_tensor(value, reference):
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def flatten_action_normalization_stats(
    action_normalization_stats,
    action_keys: Sequence[str],
):
    """Flatten checkpoint action statistics in policy-vector order."""

    if action_normalization_stats is None:
        raise ValueError("Guided denoising requires checkpoint action-normalization statistics")

    scales = []
    offsets = []
    for key in action_keys:
        if key not in action_normalization_stats:
            raise KeyError("Action normalization statistics are missing key '{}'".format(key))
        stats = action_normalization_stats[key]
        scales.append(torch.as_tensor(stats["scale"]).reshape(-1))
        offsets.append(torch.as_tensor(stats["offset"]).reshape(-1))
    return torch.cat(scales), torch.cat(offsets)


def guidance_context_from_rollout_policy(
    rollout_policy,
    *,
    current_eef_pos,
    obstacle_centers,
    obstacle_radii,
    guidance_scale,
    clearance_margin=0.02,
    enabled=True,
    normalize_waypoint_gradient=False,
    max_waypoint_displacement_m=None,
    norm_epsilon=1e-6,
    cost_type="penetration",
    max_guidance_timestep=None,
):
    """Build a context using the exact statistics stored by ``RolloutPolicy``."""

    action_keys = rollout_policy.policy.global_config.train.action_keys
    scale, offset = flatten_action_normalization_stats(
        rollout_policy.action_normalization_stats,
        action_keys,
    )
    return GuidedDenoisingContext(
        current_eef_pos=current_eef_pos,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        action_scale=scale,
        action_offset=offset,
        guidance_scale=guidance_scale,
        clearance_margin=clearance_margin,
        enabled=enabled,
        normalize_waypoint_gradient=normalize_waypoint_gradient,
        max_waypoint_displacement_m=max_waypoint_displacement_m,
        norm_epsilon=norm_epsilon,
        cost_type=cost_type,
        max_guidance_timestep=max_guidance_timestep,
    )


def unnormalize_action_points(normalized_action, action_scale, action_offset):
    """Convert normalized action points to raw action coordinates."""

    scale = _as_reference_tensor(action_scale, normalized_action).reshape(1, 1, -1)
    offset = _as_reference_tensor(action_offset, normalized_action).reshape(1, 1, -1)
    if scale.shape[-1] != normalized_action.shape[-1]:
        raise ValueError(
            "Action-normalization dimension {} does not match action dimension {}".format(
                scale.shape[-1], normalized_action.shape[-1]
            )
        )
    if torch.any(scale == 0):
        raise ValueError("Action-normalization scale must be non-zero")
    return normalized_action * scale + offset


def physical_displacement_to_normalized(displacement, action_scale):
    """Convert a physical displacement vector without applying affine offset."""

    scale = _as_reference_tensor(action_scale, displacement)
    while scale.ndim < displacement.ndim:
        scale = scale.unsqueeze(0)
    if torch.any(scale == 0):
        raise ValueError("Action-normalization scale must be non-zero")
    return displacement / scale


def reconstruct_delta_eef_positions(current_eef_pos, delta_positions):
    """Reconstruct absolute EEF positions from executable world-frame deltas."""

    current = _as_reference_tensor(current_eef_pos, delta_positions)
    if current.ndim == 1:
        current = current.unsqueeze(0)
    if current.shape[-1] != 3 or delta_positions.shape[-1] != 3:
        raise ValueError("EEF positions and position deltas must have dimension 3")
    if current.shape[0] == 1 and delta_positions.shape[0] > 1:
        current = current.expand(delta_positions.shape[0], -1)
    if current.shape[0] != delta_positions.shape[0]:
        raise ValueError("Current EEF position batch dimension does not match action batch")
    return current.unsqueeze(1) + torch.cumsum(delta_positions, dim=1)


def lan_xy_penetration_cost(
    waypoints,
    obstacle_centers,
    obstacle_radii,
    *,
    clearance_margin=0.02,
    norm_epsilon=1e-6,
):
    """Return the LAN-equivalent unsquared XY penetration cost and geometry."""

    centers = _as_reference_tensor(obstacle_centers, waypoints)
    radii = _as_reference_tensor(obstacle_radii, waypoints).reshape(-1)
    if centers.numel() == 0:
        centers = centers.reshape(0, 2)
    elif centers.ndim != 2 or centers.shape[-1] not in (2, 3):
        raise ValueError("Obstacle centers must have shape [N, 2] or [N, 3]")
    if centers.shape[0] != radii.shape[0]:
        raise ValueError("Obstacle center and radius counts do not match")
    if norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be positive")

    if centers.shape[0] == 0:
        empty = waypoints.new_zeros((waypoints.shape[0], waypoints.shape[1], 0))
        return waypoints.sum() * 0.0, empty, empty

    xy_delta = waypoints[..., :2].unsqueeze(2) - centers[:, :2].reshape(1, 1, -1, 2)
    distances = torch.sqrt(torch.sum(xy_delta.square(), dim=-1) + norm_epsilon ** 2)
    effective_radii = radii + float(clearance_margin)
    clearances = distances - effective_radii.reshape(1, 1, -1)
    penetrations = torch.relu(-clearances)
    return penetrations.sum(), penetrations, clearances


def decaying_vector_field_cost(
    waypoints,
    obstacle_centers,
    obstacle_radii,
    *,
    clearance_margin=0.02,
    norm_epsilon=1e-6,
):
    """Radius-normalized decaying potential field.

    Cost per waypoint-obstacle pair is ``R / 2 * (1 - d/R)^2`` when
    ``d < R``, else zero. Away from the exact obstacle center, this gives a
    radius-independent gradient magnitude ``1 - d/R``. The epsilon-safe norm
    keeps the exact-center gradient finite (and therefore zero by symmetry).
    """

    centers = _as_reference_tensor(obstacle_centers, waypoints)
    radii = _as_reference_tensor(obstacle_radii, waypoints).reshape(-1)
    if centers.numel() == 0:
        centers = centers.reshape(0, 2)
    elif centers.ndim != 2 or centers.shape[-1] not in (2, 3):
        raise ValueError("Obstacle centers must have shape [N, 2] or [N, 3]")
    if centers.shape[0] != radii.shape[0]:
        raise ValueError("Obstacle center and radius counts do not match")
    if norm_epsilon <= 0:
        raise ValueError("norm_epsilon must be positive")

    if centers.shape[0] == 0:
        empty = waypoints.new_zeros((waypoints.shape[0], waypoints.shape[1], 0))
        return waypoints.sum() * 0.0, empty, empty

    xy_delta = waypoints[..., :2].unsqueeze(2) - centers[:, :2].reshape(1, 1, -1, 2)
    distances = torch.sqrt(torch.sum(xy_delta.square(), dim=-1) + norm_epsilon ** 2)
    effective_radii = (radii + float(clearance_margin)).reshape(1, 1, -1)
    if torch.any(effective_radii <= 0):
        raise ValueError("Effective obstacle radii must be positive")
    clearances = distances - effective_radii
    normalized_depth = torch.relu(1.0 - distances / effective_radii)
    cost_per_pair = 0.5 * effective_radii * normalized_depth.square()
    return cost_per_pair.sum(), normalized_depth, clearances


def waypoint_displacement_to_delta_update(current_eef_pos, waypoints, waypoint_displacement):
    """Difference pushed absolute waypoints back into executable delta actions."""

    if waypoints.shape != waypoint_displacement.shape:
        raise ValueError("Waypoint and displacement shapes must match")
    current = _as_reference_tensor(current_eef_pos, waypoints)
    if current.ndim == 1:
        current = current.unsqueeze(0)
    if current.shape[0] == 1 and waypoints.shape[0] > 1:
        current = current.expand(waypoints.shape[0], -1)

    pushed = waypoints + waypoint_displacement
    pushed_with_start = torch.cat((current.unsqueeze(1), pushed), dim=1)
    pushed_deltas = pushed_with_start[:, 1:] - pushed_with_start[:, :-1]

    original_with_start = torch.cat((current.unsqueeze(1), waypoints), dim=1)
    original_deltas = original_with_start[:, 1:] - original_with_start[:, :-1]
    return pushed_deltas - original_deltas


def previous_alpha_cumprod(noise_scheduler, timestep, reference):
    """Get the alpha product at the next scheduler timestep.

    This follows the scheduler's configured inference sequence, so it supports
    both the legacy DDIM-10 policy and LAN-O3DP's DDPM-100 policy.
    """

    if noise_scheduler.num_inference_steps is None:
        raise ValueError("Noise scheduler timesteps have not been initialized")
    timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    inference_timesteps = noise_scheduler.timesteps
    matches = torch.nonzero(
        inference_timesteps == timestep_int, as_tuple=False
    ).reshape(-1)
    if matches.numel() != 1:
        raise ValueError(
            "Timestep {} does not occur exactly once in the scheduler sequence".format(
                timestep_int
            )
        )
    index = int(matches.item())
    previous_timestep = (
        int(inference_timesteps[index + 1].item())
        if index + 1 < len(inference_timesteps)
        else -1
    )
    if previous_timestep >= 0:
        alpha = noise_scheduler.alphas_cumprod[previous_timestep]
    else:
        alpha = getattr(
            noise_scheduler,
            "final_alpha_cumprod",
            getattr(noise_scheduler, "one", 1.0),
        )
    return previous_timestep, _as_reference_tensor(alpha, reference)


def _clip_waypoint_displacement(displacement, max_norm, epsilon):
    if max_norm is None:
        return displacement
    if max_norm <= 0:
        raise ValueError("max_waypoint_displacement_m must be positive when provided")
    norms = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    factors = torch.clamp(float(max_norm) / torch.clamp(norms, min=epsilon), max=1.0)
    return displacement * factors


def apply_guidance_to_reverse_sample(
    *,
    predicted_clean_action,
    reverse_sample,
    timestep,
    noise_scheduler,
    context,
    observation_horizon,
    action_horizon,
):
    """Apply one LAN-style delta-EEF update to a scheduler reverse sample."""

    if context is None or not context.enabled:
        return reverse_sample, None
    if predicted_clean_action.shape != reverse_sample.shape:
        raise ValueError("Predicted clean action and reverse sample shapes must match")

    timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    if (
        context.max_guidance_timestep is not None
        and timestep_int > context.max_guidance_timestep
    ):
        return reverse_sample, None

    start = int(observation_horizon) - 1
    end = start + int(action_horizon)
    if start < 0 or end > predicted_clean_action.shape[1]:
        raise ValueError("Executed action slice is outside the prediction horizon")

    action_scale = _as_reference_tensor(context.action_scale, predicted_clean_action).reshape(-1)
    action_offset = _as_reference_tensor(context.action_offset, predicted_clean_action).reshape(-1)
    if predicted_clean_action.shape[-1] < 3:
        raise ValueError("Delta-EEF guidance requires at least three action dimensions")

    clean_slice = predicted_clean_action[:, start:end].detach()
    raw_slice = unnormalize_action_points(clean_slice, action_scale, action_offset)
    raw_delta_positions = raw_slice[..., :3]
    waypoints = reconstruct_delta_eef_positions(context.current_eef_pos, raw_delta_positions)

    previous_timestep, alpha_previous = previous_alpha_cumprod(
        noise_scheduler, timestep, predicted_clean_action
    )
    scheduler_factor_tensor = float(context.guidance_scale) / torch.sqrt(alpha_previous)

    centers = _as_reference_tensor(context.obstacle_centers, waypoints)
    radii = _as_reference_tensor(context.obstacle_radii, waypoints).reshape(-1)
    if (centers.numel() == 0) != (radii.numel() == 0):
        raise ValueError("Obstacle center and radius counts do not match")
    if centers.numel() == 0:
        waypoint_list = waypoints.detach().cpu().tolist()
        zero_displacement_list = torch.zeros_like(waypoints).cpu().tolist()
        diagnostics = GuidanceStepDiagnostics(
            timestep=int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            previous_timestep=previous_timestep,
            scheduler_factor=float(scheduler_factor_tensor.item()),
            active_penetration_count=0,
            cost=0.0,
            min_clearance_m=float("inf"),
            raw_physical_gradient_norm=0.0,
            physical_waypoint_displacement_norm=0.0,
            max_waypoint_displacement_m=0.0,
            normalized_applied_update_norm=0.0,
            resulting_clean_action_cost=0.0,
            before_waypoints_m=waypoint_list,
            after_waypoints_m=waypoint_list,
            waypoint_displacements_m=zero_displacement_list,
        )
        return reverse_sample, diagnostics

    with torch.enable_grad():
        guidance_waypoints = waypoints.detach().requires_grad_(True)
        if context.cost_type == "vector_field":
            cost_fn = decaying_vector_field_cost
        elif context.cost_type == "penetration":
            cost_fn = lan_xy_penetration_cost
        else:
            raise ValueError("Unsupported guidance cost type '{}'".format(context.cost_type))
        cost, penetrations, clearances = cost_fn(
            guidance_waypoints,
            centers,
            radii,
            clearance_margin=context.clearance_margin,
            norm_epsilon=context.norm_epsilon,
        )
        waypoint_gradient = torch.autograd.grad(cost, guidance_waypoints)[0]

    raw_gradient = waypoint_gradient.detach()
    update_gradient = raw_gradient
    if context.normalize_waypoint_gradient:
        gradient_norm = torch.linalg.vector_norm(update_gradient, dim=-1, keepdim=True)
        update_gradient = torch.where(
            gradient_norm > context.norm_epsilon,
            update_gradient / torch.clamp(gradient_norm, min=context.norm_epsilon),
            torch.zeros_like(update_gradient),
        )

    waypoint_displacement = -scheduler_factor_tensor * update_gradient
    waypoint_displacement = _clip_waypoint_displacement(
        waypoint_displacement,
        context.max_waypoint_displacement_m,
        context.norm_epsilon,
    )
    delta_update = waypoint_displacement_to_delta_update(
        context.current_eef_pos,
        waypoints,
        waypoint_displacement,
    )

    normalized_xy_update = physical_displacement_to_normalized(
        delta_update[..., :2],
        action_scale[:2],
    )
    full_update = torch.zeros_like(reverse_sample)
    full_update[:, start:end, :2] = normalized_xy_update

    with torch.no_grad():
        resulting_cost, _, _ = cost_fn(
            waypoints + waypoint_displacement,
            centers,
            radii,
            clearance_margin=context.clearance_margin,
            norm_epsilon=context.norm_epsilon,
        )
        displacement_norms = torch.linalg.vector_norm(waypoint_displacement, dim=-1)
        diagnostics = GuidanceStepDiagnostics(
            timestep=int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            previous_timestep=previous_timestep,
            scheduler_factor=float(scheduler_factor_tensor.item()),
            active_penetration_count=int(torch.count_nonzero(penetrations > 0).item()),
            cost=float(cost.detach().item()),
            min_clearance_m=float(torch.min(clearances).detach().item()),
            raw_physical_gradient_norm=float(torch.linalg.vector_norm(raw_gradient).item()),
            physical_waypoint_displacement_norm=float(
                torch.linalg.vector_norm(waypoint_displacement).item()
            ),
            max_waypoint_displacement_m=float(torch.max(displacement_norms).item()),
            normalized_applied_update_norm=float(torch.linalg.vector_norm(full_update).item()),
            resulting_clean_action_cost=float(resulting_cost.item()),
            before_waypoints_m=waypoints.detach().cpu().tolist(),
            after_waypoints_m=(waypoints + waypoint_displacement).detach().cpu().tolist(),
            waypoint_displacements_m=waypoint_displacement.detach().cpu().tolist(),
        )

    if context.guidance_scale == 0 or not torch.any(full_update != 0):
        return reverse_sample, diagnostics
    return reverse_sample + full_update, diagnostics


def guided_policy_from_checkpoint(device=None, ckpt_path=None, ckpt_dict=None, verbose=False):
    """Load diffusion-policy weights into the registered guided DP variant.

    The checkpoint object is adapted in memory. The checkpoint file and its model
    state dictionary are not modified or duplicated.
    """

    from robomimic.utils import file_utils as FileUtils

    source = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)
    source_algo = source.get("algo_name")
    target_algos = {
        "diffusion_policy": "guided_diffusion_policy",
        "guided_diffusion_policy": "guided_diffusion_policy",
        "lan_o3dp": "guided_lan_o3dp",
        "guided_lan_o3dp": "guided_lan_o3dp",
    }
    if source_algo not in target_algos:
        raise ValueError(
            "Expected a diffusion_policy or lan_o3dp checkpoint, got '{}'".format(
                source_algo
            )
        )

    adapted = dict(source)
    config_dict = json.loads(source["config"])
    target_algo = target_algos[source_algo]
    config_dict["algo_name"] = target_algo
    adapted["algo_name"] = target_algo
    adapted["config"] = json.dumps(config_dict)

    # policy_from_checkpoint converts these small nested lists in place.
    for stats_key in ("obs_normalization_stats", "action_normalization_stats"):
        if stats_key in source:
            stats = source[stats_key]
            adapted[stats_key] = {
                key: {name: list(value) for name, value in values.items()}
                for key, values in stats.items()
            }

    return FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_dict=adapted,
        verbose=verbose,
    )
