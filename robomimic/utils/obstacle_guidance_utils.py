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


def get_robomimic_env(env):
    """
    Return the innermost robomimic environment wrapper before the raw simulator env.
    """
    return unwrap_env(env)


def controller_delta_pos_mapping_from_config(controller_config):
    """
    Derive action-command to meter-scale xyz delta mapping from a robosuite controller config.
    """
    if "body_parts" in controller_config:
        controller_config = controller_config["body_parts"].get("right", None)
    if controller_config is None:
        raise ValueError("Could not find right body part controller config")

    input_min = np.array(controller_config["input_min"], dtype=np.float32)
    input_max = np.array(controller_config["input_max"], dtype=np.float32)
    output_min = np.array(controller_config["output_min"], dtype=np.float32)[:3]
    output_max = np.array(controller_config["output_max"], dtype=np.float32)[:3]

    if input_min.ndim == 0:
        input_min = np.full((3,), float(input_min), dtype=np.float32)
    else:
        input_min = input_min.reshape(-1)[:3]
    if input_max.ndim == 0:
        input_max = np.full((3,), float(input_max), dtype=np.float32)
    else:
        input_max = input_max.reshape(-1)[:3]

    if output_min.shape[0] != 3 or output_max.shape[0] != 3:
        raise ValueError("Controller output_min/output_max must include xyz dimensions")

    scale = (output_max - output_min) / (input_max - input_min)
    offset = output_min - input_min * scale
    return scale.astype(np.float32), offset.astype(np.float32)


def get_controller_delta_pos_mapping(env):
    """
    Read controller config from a robomimic robosuite env and derive xyz delta mapping.
    """
    robomimic_env = get_robomimic_env(env)
    init_kwargs = getattr(robomimic_env, "_init_kwargs", None)
    if init_kwargs is None or "controller_configs" not in init_kwargs:
        raise ValueError("Could not read controller_configs from environment")
    return controller_delta_pos_mapping_from_config(init_kwargs["controller_configs"])


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


def _object_geom_names(obj):
    geom_names = []
    for attr in ("visual_geoms", "contact_geoms"):
        for geom_name in getattr(obj, attr, []):
            if geom_name not in geom_names:
                geom_names.append(geom_name)
    return geom_names


def _sim_geom_id(sim, geom_name):
    try:
        return sim.model.geom_name2id(geom_name)
    except Exception:
        return None


def _geom_xy_radius_and_z_extent(sim, geom_id):
    """
    Estimate an axis-aligned XY enclosing radius and positive Z extent for a geom.
    MuJoCo geom sizes are half-sizes for boxes, radius / half-height for cylinders,
    and radius for spheres.
    """
    size = np.array(sim.model.geom_size[geom_id], dtype=np.float32)
    geom_type = int(sim.model.geom_type[geom_id])

    # MuJoCo enum values: sphere=2, capsule=3, ellipsoid=4, cylinder=5, box=6.
    if geom_type == 2:
        xy_radius = size[0]
        z_extent = size[0]
    elif geom_type == 3:
        xy_radius = size[0]
        z_extent = size[0] + size[1]
    elif geom_type == 4:
        xy_radius = np.linalg.norm(size[:2])
        z_extent = size[2]
    elif geom_type == 5:
        xy_radius = size[0]
        z_extent = size[1]
    elif geom_type == 6:
        xy_radius = np.linalg.norm(size[:2])
        z_extent = size[2]
    else:
        xy_radius = np.linalg.norm(size[:2])
        if xy_radius == 0.0:
            xy_radius = np.max(size)
        z_extent = size[2] if size.shape[0] > 2 and size[2] > 0.0 else np.max(size)

    return float(xy_radius), float(z_extent)


def _object_center_from_sim(sim, obj):
    pos = None
    body_name = _object_body_name(obj)
    if body_name is not None:
        pos = _sim_body_pos(sim, body_name)
    if pos is None:
        pos = _sim_geom_center(sim, _object_geom_names(obj))
    return pos


def _object_geometry_from_sim(sim, obj, xy_clearance=0.0):
    center = _object_center_from_sim(sim, obj)
    if center is None:
        raise ValueError("Could not read center for obstacle object '{}'".format(_object_name(obj)))

    geom_names = _object_geom_names(obj)
    if len(geom_names) == 0:
        raise ValueError("Could not find geoms for obstacle object '{}'".format(_object_name(obj)))

    physical_radius = None
    top_z = None
    for geom_name in geom_names:
        geom_id = _sim_geom_id(sim, geom_name)
        if geom_id is None:
            continue
        geom_pos = np.array(sim.data.geom_xpos[geom_id], dtype=np.float32)
        geom_radius, geom_z_extent = _geom_xy_radius_and_z_extent(sim, geom_id)
        enclosing_radius = float(np.linalg.norm(geom_pos[:2] - center[:2]) + geom_radius)
        geom_top_z = float(geom_pos[2] + geom_z_extent)
        physical_radius = enclosing_radius if physical_radius is None else max(physical_radius, enclosing_radius)
        top_z = geom_top_z if top_z is None else max(top_z, geom_top_z)

    if physical_radius is None or top_z is None:
        raise ValueError("Could not read geom geometry for obstacle object '{}'".format(_object_name(obj)))

    return center.astype(np.float32), float(physical_radius), float(physical_radius + xy_clearance), float(top_z)


def _iter_obstacle_objects(raw_env, target_object_name=None, obstacle_names=None):
    target_lower = target_object_name.lower() if target_object_name is not None else None
    obstacle_name_set = None
    if obstacle_names is not None and len(obstacle_names) > 0:
        obstacle_name_set = set(name.lower() for name in obstacle_names)

    for obj in getattr(raw_env, "objects", []):
        name = _object_name(obj)
        if name is None:
            continue
        name_lower = name.lower()
        if target_lower is not None and name_lower == target_lower:
            continue
        if obstacle_name_set is not None and name_lower not in obstacle_name_set:
            continue
        yield obj, name


def get_oracle_obstacle_geometry(
    env,
    target_object_name=None,
    obstacle_names=None,
    xy_clearance=0.02,
):
    """
    Return simulator-derived obstacle geometry for guidance.

    Returns:
        centers_xyz (np.ndarray): shape [N, 3].
        physical_radii (np.ndarray): shape [N].
        safety_radii (np.ndarray): shape [N], physical radius + xy clearance.
        top_z (np.ndarray): shape [N].
        names (list): object names in the same order.
    """
    raw_env = get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        raise ValueError("Obstacle guidance requires simulator access")

    centers = []
    physical_radii = []
    safety_radii = []
    top_z = []
    names = []
    for obj, name in _iter_obstacle_objects(
        raw_env=raw_env,
        target_object_name=target_object_name,
        obstacle_names=obstacle_names,
    ):
        center, physical_radius, safety_radius, object_top_z = _object_geometry_from_sim(
            sim=sim,
            obj=obj,
            xy_clearance=xy_clearance,
        )
        centers.append(center)
        physical_radii.append(physical_radius)
        safety_radii.append(safety_radius)
        top_z.append(object_top_z)
        names.append(name)

    if len(centers) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )

    return (
        np.stack(centers, axis=0).astype(np.float32),
        np.array(physical_radii, dtype=np.float32),
        np.array(safety_radii, dtype=np.float32),
        np.array(top_z, dtype=np.float32),
        names,
    )


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

    centers = []
    names = []
    for obj, name in _iter_obstacle_objects(
        raw_env=raw_env,
        target_object_name=target_object_name,
        obstacle_names=obstacle_names,
    ):
        pos = _object_center_from_sim(sim, obj)
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


def _as_traj_scale_or_offset(x, action_chunk, default, dim=3):
    if x is None:
        x = default
    x = torch.as_tensor(x, dtype=action_chunk.dtype, device=action_chunk.device)
    if x.ndim == 0:
        x = x.repeat(dim)
    elif x.numel() == 1:
        x = x.reshape(1).repeat(dim)
    if x.numel() != dim:
        x = x.reshape(-1)[:dim]
    return x.reshape(1, 1, dim)


def unnormalize_action_chunk(action_chunk, action_scale=None, action_offset=None):
    """
    Convert a normalized action chunk back to raw action coordinates.
    """
    if action_scale is None and action_offset is None:
        return action_chunk

    scale = torch.as_tensor(
        1.0 if action_scale is None else action_scale,
        dtype=action_chunk.dtype,
        device=action_chunk.device,
    )
    offset = torch.as_tensor(
        0.0 if action_offset is None else action_offset,
        dtype=action_chunk.dtype,
        device=action_chunk.device,
    )
    if scale.ndim == 0:
        scale = scale.repeat(action_chunk.shape[-1])
    if offset.ndim == 0:
        offset = offset.repeat(action_chunk.shape[-1])
    scale = scale.reshape(1, 1, -1)
    offset = offset.reshape(1, 1, -1)
    return action_chunk * scale + offset


def action_chunk_to_eef_xyz_traj(
    action_chunk,
    current_eef_pos,
    horizon=None,
    delta_pos_scale=1.0,
    delta_pos_offset=0.0,
):
    """
    Convert an action chunk into an approximate commanded eef xyz trajectory.
    The first 3 action dimensions are mapped to meter-scale position deltas.
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

    scale = _as_traj_scale_or_offset(delta_pos_scale, action_chunk, default=1.0, dim=3)
    offset = _as_traj_scale_or_offset(delta_pos_offset, action_chunk, default=0.0, dim=3)
    delta_pos = action_chunk[:, :horizon, :3] * scale + offset
    return current_eef_pos[:, None, :3] + torch.cumsum(delta_pos, dim=1)


def action_chunk_to_eef_xy_traj(
    action_chunk,
    current_eef_pos,
    horizon=None,
    delta_pos_scale=1.0,
    delta_pos_offset=0.0,
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

    return action_chunk_to_eef_xyz_traj(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        horizon=horizon,
        delta_pos_scale=delta_pos_scale,
        delta_pos_offset=delta_pos_offset,
    )[..., :2]


def obstacle_xy_cost(
    action_chunk,
    current_eef_pos,
    obstacle_centers_xy,
    obstacle_radii,
    horizon=8,
    delta_pos_scale=1.0,
    delta_pos_offset=0.0,
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
        stats = OrderedDict(cost=cost.detach(), min_distance=None, min_xy_distance=None, num_obstacles=0)
        return (cost, stats) if return_stats else cost

    if centers.ndim != 2 or centers.shape[-1] not in (2, 3):
        raise ValueError("obstacle_centers_xy must have shape [N, 2] or [N, 3], got {}".format(tuple(centers.shape)))
    centers_xy = centers[..., :2]
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("obstacle_radii must have shape [N], got {}".format(tuple(radii.shape)))

    traj_xy = action_chunk_to_eef_xy_traj(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        horizon=horizon,
        delta_pos_scale=delta_pos_scale,
        delta_pos_offset=delta_pos_offset,
    )
    dist = torch.linalg.norm(traj_xy[:, :, None, :] - centers_xy[None, None, :, :], dim=-1)
    penetration = F.relu(radii.reshape(1, 1, -1) - dist)
    per_batch_cost = torch.sum(penetration ** 2, dim=(1, 2))
    cost = torch.sum(per_batch_cost)
    min_xy_distance = torch.amin(dist, dim=(1, 2)).detach()

    stats = OrderedDict(
        cost=cost.detach(),
        per_batch_cost=per_batch_cost.detach(),
        min_distance=min_xy_distance,
        min_xy_distance=min_xy_distance,
        num_obstacles=centers.shape[0],
    )
    return (cost, stats) if return_stats else cost


def obstacle_xyz_cylinder_cost(
    action_chunk,
    current_eef_pos,
    obstacle_centers_xyz,
    obstacle_radii,
    obstacle_top_z,
    z_clearance=0.03,
    horizon=8,
    delta_pos_scale=1.0,
    delta_pos_offset=0.0,
    return_stats=False,
):
    """
    Differentiable cylindrical obstacle cost in xyz.
    """
    centers = _as_batched_tensor(
        obstacle_centers_xyz,
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )
    radii = None if obstacle_radii is None else torch.as_tensor(
        obstacle_radii,
        dtype=action_chunk.dtype,
        device=action_chunk.device,
    )
    top_z = None if obstacle_top_z is None else torch.as_tensor(
        obstacle_top_z,
        dtype=action_chunk.dtype,
        device=action_chunk.device,
    )

    if centers is None or radii is None or top_z is None or centers.numel() == 0 or radii.numel() == 0:
        cost = action_chunk.sum() * 0.0
        stats = OrderedDict(
            cost=cost.detach(),
            min_distance=None,
            min_xy_distance=None,
            min_z_clearance=None,
            num_obstacles=0,
        )
        return (cost, stats) if return_stats else cost

    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("obstacle_centers_xyz must have shape [N, 3], got {}".format(tuple(centers.shape)))
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("obstacle_radii must have shape [N], got {}".format(tuple(radii.shape)))
    if top_z.ndim != 1 or top_z.shape[0] != centers.shape[0]:
        raise ValueError("obstacle_top_z must have shape [N], got {}".format(tuple(top_z.shape)))

    traj = action_chunk_to_eef_xyz_traj(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        horizon=horizon,
        delta_pos_scale=delta_pos_scale,
        delta_pos_offset=delta_pos_offset,
    )
    dist_xy = torch.linalg.norm(traj[:, :, None, :2] - centers[None, None, :, :2], dim=-1)
    xy_pen = F.relu(radii.reshape(1, 1, -1) - dist_xy)

    z_limit = top_z.reshape(1, 1, -1) + float(z_clearance)
    z_clear = traj[:, :, None, 2] - z_limit
    z_pen = F.relu(-z_clear)

    per_batch_cost = torch.sum((xy_pen ** 2) * (z_pen ** 2), dim=(1, 2))
    cost = torch.sum(per_batch_cost)
    min_xy_distance = torch.amin(dist_xy, dim=(1, 2)).detach()
    min_z_clearance = torch.amin(z_clear, dim=(1, 2)).detach()

    stats = OrderedDict(
        cost=cost.detach(),
        per_batch_cost=per_batch_cost.detach(),
        min_distance=min_xy_distance,
        min_xy_distance=min_xy_distance,
        min_z_clearance=min_z_clearance,
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
