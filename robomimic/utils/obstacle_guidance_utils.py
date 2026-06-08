"""
Utilities for obstacle-aware inference-time guidance for Diffusion Policy.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from robomimic.envs.wrappers import EnvWrapper


def unwrap_env(env):
    """
    Return the innermost robomimic environment wrapper target.
    """
    while isinstance(env, EnvWrapper):
        env = env.env
    return env


def get_raw_env(env):
    """
    Return the underlying robosuite environment when available.
    """
    env = unwrap_env(env)
    return getattr(env, "env", env)


def _object_name(obj):
    return getattr(obj, "name", None)


def _object_body_name(obj):
    for attr in ("root_body", "body_name", "root_body_name"):
        value = getattr(obj, attr, None)
        if value is not None:
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                value = value[0]
            return value
    name = _object_name(obj)
    if name is not None:
        return "{}_main".format(name)
    return None


def _sim_body_pos(sim, body_name):
    try:
        body_id = sim.model.body_name2id(body_name)
    except Exception:
        return None
    return np.array(sim.data.body_xpos[body_id], dtype=np.float32)


def _sim_geom_center(sim, geom_names):
    geom_pos = []
    for geom_name in geom_names:
        try:
            geom_id = sim.model.geom_name2id(geom_name)
        except Exception:
            continue
        geom_pos.append(np.array(sim.data.geom_xpos[geom_id], dtype=np.float32))
    if len(geom_pos) == 0:
        return None
    return np.mean(np.stack(geom_pos, axis=0), axis=0)


def get_oracle_obstacle_circles(
    env,
    target_object_name=None,
    obstacle_names=None,
    obstacle_radius=0.06,
):
    """
    Return obstacle centers and radii from the simulator.

    Args:
        env: robomimic environment or wrapper.
        target_object_name (str or None): object name to exclude.
        obstacle_names (list or None): if provided, only use these object names.
        obstacle_radius (float): fixed radius for all returned obstacles.

    Returns:
        centers_xy (np.ndarray): shape [N, 2], float32.
        radii (np.ndarray): shape [N], float32.
        names (list): object names in the same order.
    """
    raw_env = get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    target_lower = target_object_name.lower() if target_object_name is not None else None
    obstacle_name_set = None
    if obstacle_names is not None and len(obstacle_names) > 0:
        obstacle_name_set = set(name.lower() for name in obstacle_names)

    centers = []
    names = []
    for obj in getattr(raw_env, "objects", []):
        name = _object_name(obj)
        if name is None:
            continue
        name_lower = name.lower()
        if target_lower is not None and name_lower == target_lower:
            continue
        if obstacle_name_set is not None and name_lower not in obstacle_name_set:
            continue

        pos = None
        body_name = _object_body_name(obj)
        if body_name is not None:
            pos = _sim_body_pos(sim, body_name)
        if pos is None:
            geom_names = []
            geom_names.extend(getattr(obj, "visual_geoms", []))
            geom_names.extend(getattr(obj, "contact_geoms", []))
            pos = _sim_geom_center(sim, geom_names)
        if pos is None:
            continue

        centers.append(pos[:2])
        names.append(name)

    if len(centers) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    centers_xy = np.stack(centers, axis=0).astype(np.float32)
    radii = np.full((len(centers),), obstacle_radius, dtype=np.float32)
    return centers_xy, radii, names


def _as_batched_tensor(x, device=None, dtype=torch.float32):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        x = x.to(device=device, dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def action_chunk_to_eef_xy_traj(
    action_chunk,
    current_eef_pos,
    horizon=None,
    delta_pos_scale=1.0,
):
    """
    Convert a normalized action chunk into an approximate eef xy trajectory.

    Args:
        action_chunk (torch.Tensor): shape [B, H, A].
        current_eef_pos (torch.Tensor or np.ndarray): shape [B, 3] or [3].
        horizon (int or None): number of action steps to use.
        delta_pos_scale (float or sequence): scale from action units to meters.

    Returns:
        traj_xy (torch.Tensor): shape [B, horizon, 2].
    """
    if horizon is None:
        horizon = action_chunk.shape[1]
    horizon = min(horizon, action_chunk.shape[1])

    current_eef_pos = _as_batched_tensor(
        current_eef_pos,
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )
    if current_eef_pos.shape[0] == 1 and action_chunk.shape[0] > 1:
        current_eef_pos = current_eef_pos.expand(action_chunk.shape[0], -1)

    scale = torch.as_tensor(delta_pos_scale, dtype=action_chunk.dtype, device=action_chunk.device)
    if scale.ndim == 0:
        scale = scale.repeat(3)
    scale = scale.reshape(1, 1, 3)

    delta_pos = action_chunk[:, :horizon, :3] * scale
    traj = current_eef_pos[:, None, :3] + torch.cumsum(delta_pos, dim=1)
    return traj[..., :2]


def obstacle_xy_cost(
    action_chunk,
    current_eef_pos,
    obstacle_centers_xy,
    obstacle_radii,
    horizon=8,
    delta_pos_scale=1.0,
    return_stats=False,
):
    """
    Differentiable xy obstacle penetration cost.

    Args:
        action_chunk (torch.Tensor): shape [B, H, A].
        current_eef_pos (torch.Tensor or np.ndarray): shape [B, 3] or [3].
        obstacle_centers_xy (torch.Tensor or np.ndarray): shape [N, 2].
        obstacle_radii (torch.Tensor or np.ndarray): shape [N].
        horizon (int): number of action steps to use.
        delta_pos_scale (float or sequence): scale from action units to meters.
        return_stats (bool): if True, return (cost, stats).

    Returns:
        cost or (cost, stats).
    """
    centers = _as_batched_tensor(
        obstacle_centers_xy,
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )
    if obstacle_radii is None:
        radii = None
    else:
        radii = torch.as_tensor(obstacle_radii, dtype=action_chunk.dtype, device=action_chunk.device)

    if centers is None or radii is None or centers.numel() == 0 or radii.numel() == 0:
        cost = action_chunk.sum() * 0.0
        stats = OrderedDict(cost=cost.detach(), min_distance=None, num_obstacles=0)
        return (cost, stats) if return_stats else cost

    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("obstacle_centers_xy must have shape [N, 2], got {}".format(tuple(centers.shape)))
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("obstacle_radii must have shape [N], got {}".format(tuple(radii.shape)))

    traj_xy = action_chunk_to_eef_xy_traj(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        horizon=horizon,
        delta_pos_scale=delta_pos_scale,
    )
    dist = torch.linalg.norm(traj_xy[:, :, None, :] - centers[None, None, :, :], dim=-1)
    penetration = F.relu(radii.reshape(1, 1, -1) - dist)
    per_batch_cost = torch.sum(penetration ** 2, dim=(1, 2))
    cost = torch.sum(per_batch_cost)

    stats = OrderedDict(
        cost=cost.detach(),
        per_batch_cost=per_batch_cost.detach(),
        min_distance=torch.amin(dist, dim=(1, 2)).detach(),
        num_obstacles=centers.shape[0],
    )
    return (cost, stats) if return_stats else cost


def estimate_clean_action_from_scheduler(
    scheduler,
    sample,
    timestep,
    model_output,
    step_output=None,
):
    """
    Estimate x0 from scheduler conventions.
    """
    if step_output is not None and hasattr(step_output, "pred_original_sample"):
        pred = step_output.pred_original_sample
        if pred is not None:
            return pred

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "sample":
        return model_output
    if prediction_type != "epsilon":
        raise NotImplementedError("Unsupported scheduler prediction_type '{}'".format(prediction_type))

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t = alpha_prod_t.to(device=sample.device, dtype=sample.dtype)
    beta_prod_t = 1 - alpha_prod_t
    return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()


def guidance_scale_for_step(guidance_scale, schedule, step_index, num_steps):
    """
    Return rho_t for a denoising step.
    """
    if guidance_scale <= 0:
        return 0.0
    if schedule == "constant":
        return guidance_scale
    if schedule == "late":
        if num_steps <= 1:
            progress = 1.0
        else:
            progress = float(step_index) / float(num_steps - 1)
        return guidance_scale * progress
    raise ValueError("Unsupported obstacle guidance schedule '{}'".format(schedule))


def normalized_negative_cost_grad_update(update_sample, cost, scale, grad_source=None, eps=1e-6):
    """
    Apply a per-batch normalized gradient step that decreases cost.
    """
    if scale == 0:
        return update_sample.detach(), None
    if grad_source is None:
        grad_source = update_sample
    grad = torch.autograd.grad(cost, grad_source, retain_graph=False, create_graph=False)[0]
    grad_flat = grad.reshape(grad.shape[0], -1)
    grad_norm = torch.linalg.norm(grad_flat, dim=1).clamp_min(eps)
    grad = grad / grad_norm.reshape(-1, *([1] * (grad.ndim - 1)))
    return (update_sample - scale * grad).detach(), grad_norm.detach()
