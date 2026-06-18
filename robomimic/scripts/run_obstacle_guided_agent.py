"""
Evaluate a trained Diffusion Policy with obstacle guidance.

This script is intentionally separate from run_trained_agent.py so that the
standard checkpoint rollout path stays lightweight. It supports the existing
oracle-center guidance baseline and PC-1 point-cloud guidance from agentview
depth plus oracle distractor segmentation.
"""

import argparse
import json
import os
from copy import deepcopy

import imageio
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper


def get_current_eef_pos_from_obs(obs, obs_key="robot0_eef_pos"):
    if obs_key not in obs:
        raise KeyError("Observation key '{}' is required for obstacle guidance".format(obs_key))
    eef_pos = np.array(obs[obs_key], dtype=np.float32)
    if eef_pos.ndim == 2:
        eef_pos = eef_pos[-1]
    if eef_pos.shape[-1] != 3:
        raise ValueError("Expected '{}' to have final dimension 3, got {}".format(obs_key, eef_pos.shape))
    return eef_pos


def get_action_normalization_vector(policy):
    stats = getattr(policy, "action_normalization_stats", None)
    if stats is None:
        return None, None
    algo = getattr(policy, "policy", policy)
    action_keys = algo.global_config.train.action_keys
    scales = []
    offsets = []
    for key in action_keys:
        scales.append(np.array(stats[key]["scale"], dtype=np.float32).reshape(-1))
        offsets.append(np.array(stats[key]["offset"], dtype=np.float32).reshape(-1))
    return np.concatenate(scales, axis=0), np.concatenate(offsets, axis=0)


def env_from_checkpoint_for_guidance(ckpt_dict, env_name=None, render=False, render_offscreen=False, use_depth_obs=False):
    env_meta = deepcopy(ckpt_dict["env_metadata"])
    shape_meta = ckpt_dict["shape_metadata"]
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=ckpt_dict["algo_name"],
        ckpt_dict=ckpt_dict,
        verbose=False,
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=shape_meta.get("use_images", False) or use_depth_obs,
        use_depth_obs=use_depth_obs or shape_meta.get("use_depths", False),
    )
    return EnvUtils.wrap_env_from_config(env, config=config)


def sim_depth_planes(env):
    raw_env = ObstacleGuidanceUtils.get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return None, None
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return float(near), float(far)


def default_workspace_bounds(env):
    raw_env = ObstacleGuidanceUtils.get_raw_env(env)
    bin1_pos = np.array(getattr(raw_env, "bin1_pos", [0.1, -0.25, 0.8]), dtype=np.float32)
    table_size = np.array(getattr(raw_env, "table_full_size", [0.8, 0.8, 0.1]), dtype=np.float32)
    xy_margin = 0.12
    z_min = float(bin1_pos[2] - 0.04)
    z_max = float(bin1_pos[2] + 0.35)
    lower = np.array(
        [
            bin1_pos[0] - table_size[0] / 2.0 - xy_margin,
            bin1_pos[1] - table_size[1] / 2.0 - xy_margin,
            z_min,
        ],
        dtype=np.float32,
    )
    upper = np.array(
        [
            bin1_pos[0] + table_size[0] / 2.0 + xy_margin,
            bin1_pos[1] + table_size[1] / 2.0 + xy_margin,
            z_max,
        ],
        dtype=np.float32,
    )
    return np.stack((lower, upper), axis=0)


def normalize_depth_for_image(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros(depth.shape[:2], dtype=np.uint8)
    dmin = np.percentile(depth[finite], 1)
    dmax = np.percentile(depth[finite], 99)
    if dmax <= dmin:
        dmax = dmin + 1e-6
    depth_img = np.clip((depth[..., 0] - dmin) / (dmax - dmin), 0.0, 1.0)
    return (depth_img * 255.0).astype(np.uint8)


def draw_topdown(points_world, workspace_bounds, oracle_target=None, oracle_obstacles=None, image_size=512):
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    bounds = np.asarray(workspace_bounds, dtype=np.float32)
    xmin, ymin = bounds[0, :2]
    xmax, ymax = bounds[1, :2]

    def to_px(xy):
        xy = np.asarray(xy, dtype=np.float32)
        px = (xy[..., 0] - xmin) / max(xmax - xmin, 1e-6) * (image_size - 1)
        py = (ymax - xy[..., 1]) / max(ymax - ymin, 1e-6) * (image_size - 1)
        return np.stack((px, py), axis=-1).round().astype(np.int32)

    image[[0, -1], :, :] = 0
    image[:, [0, -1], :] = 0

    points = np.asarray(points_world, dtype=np.float32)
    if points.size > 0:
        px = to_px(points[:, :2])
        valid = (px[:, 0] >= 0) & (px[:, 0] < image_size) & (px[:, 1] >= 0) & (px[:, 1] < image_size)
        px = px[valid]
        image[px[:, 1], px[:, 0]] = np.array([30, 120, 220], dtype=np.uint8)

    origin = to_px([0.0, 0.0])
    if 0 <= origin[0] < image_size and 0 <= origin[1] < image_size:
        image[max(origin[1] - 4, 0):origin[1] + 5, max(origin[0] - 4, 0):origin[0] + 5] = [0, 0, 0]

    if oracle_obstacles is not None and len(oracle_obstacles) > 0:
        obs_px = to_px(np.asarray(oracle_obstacles)[..., :2])
        for p in obs_px:
            if 0 <= p[0] < image_size and 0 <= p[1] < image_size:
                image[max(p[1] - 5, 0):p[1] + 6, max(p[0] - 5, 0):p[0] + 6] = [220, 40, 40]

    if oracle_target is not None:
        target_px = to_px(np.asarray(oracle_target)[..., :2])
        if 0 <= target_px[0] < image_size and 0 <= target_px[1] < image_size:
            image[max(target_px[1] - 5, 0):target_px[1] + 6, max(target_px[0] - 5, 0):target_px[0] + 6] = [40, 180, 60]

    return image


def depth_colored_mask_image(depth, mask):
    """
    Visualize masked depth pixels in image coordinates. This is the most direct
    sanity check that the point cloud came from the intended segmentation mask.
    """
    image = np.full(depth.shape[:2] + (3,), 255, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image

    z = np.asarray(depth[..., 0], dtype=np.float32)[ys, xs]
    zmin, zmax = np.percentile(z, [1, 99])
    if zmax <= zmin:
        zmax = zmin + 1e-6
    t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    colors = np.stack(
        (
            40 + 215 * t,
            220 - 300 * np.abs(t - 0.5),
            255 - 220 * t,
        ),
        axis=-1,
    ).astype(np.uint8)
    for y, x, color in zip(ys, xs, colors):
        image[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = color
    return image


def reproject_world_points_image(rgb, points_world, intrinsics, camera_to_world):
    """
    Reproject world-frame points into the camera image for alignment checks.
    """
    image = rgb.copy()
    points = TensorUtils.to_numpy(points_world)
    if points.shape[0] == 0:
        return image

    K_exp = np.eye(4, dtype=np.float32)
    K_exp[:3, :3] = np.asarray(intrinsics, dtype=np.float32)
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float32))
    transform = K_exp @ world_to_camera
    points_h = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)
    pixels_h = (transform @ points_h.T).T
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-6)

    height, width = image.shape[:2]
    u = np.round(pixels[:, 0]).astype(np.int32)
    v = np.round(pixels[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for y, x in zip(v[valid], u[valid]):
        image[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = [0, 255, 255]
    return image


def dump_pointcloud_debug(
    env,
    obs,
    debug_dir,
    camera_name,
    depth,
    mask,
    points_camera,
    points_world,
    workspace_bounds,
    intrinsics,
    camera_to_world,
    target_object_name=None,
    obstacle_names=None,
    step_i=0,
):
    os.makedirs(debug_dir, exist_ok=True)
    prefix = "step_{:05d}".format(step_i)
    height, width = depth.shape[:2]
    rgb = env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
    imageio.imwrite(os.path.join(debug_dir, "{}_rgb.png".format(prefix)), rgb)
    imageio.imwrite(os.path.join(debug_dir, "{}_depth.png".format(prefix)), normalize_depth_for_image(depth))
    imageio.imwrite(os.path.join(debug_dir, "{}_distractor_mask.png".format(prefix)), (mask.astype(np.uint8) * 255))
    imageio.imwrite(
        os.path.join(debug_dir, "{}_pointcloud_imageplane.png".format(prefix)),
        depth_colored_mask_image(depth=depth, mask=mask),
    )
    imageio.imwrite(
        os.path.join(debug_dir, "{}_pointcloud_reprojection.png".format(prefix)),
        reproject_world_points_image(
            rgb=rgb,
            points_world=points_world,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        ),
    )
    np.save(os.path.join(debug_dir, "{}_pointcloud_camera.npy".format(prefix)), TensorUtils.to_numpy(points_camera))
    np.save(os.path.join(debug_dir, "{}_pointcloud_world.npy".format(prefix)), TensorUtils.to_numpy(points_world))

    oracle_target = None
    oracle_obstacles = None
    try:
        centers, _, _, _, names = ObstacleGuidanceUtils.get_oracle_obstacle_geometry(
            env=env,
            target_object_name=target_object_name,
            obstacle_names=obstacle_names,
            xy_clearance=0.0,
        )
        oracle_obstacles = centers
        if target_object_name is not None:
            raw_env = ObstacleGuidanceUtils.get_raw_env(env)
            for obj in getattr(raw_env, "objects", []):
                if getattr(obj, "name", "").lower() == target_object_name.lower():
                    oracle_target = ObstacleGuidanceUtils._object_center_from_sim(raw_env.sim, obj)
                    break
    except Exception:
        names = []

    topdown = draw_topdown(
        points_world=TensorUtils.to_numpy(points_world),
        workspace_bounds=workspace_bounds,
        oracle_target=oracle_target,
        oracle_obstacles=oracle_obstacles,
    )
    imageio.imwrite(os.path.join(debug_dir, "{}_pointcloud_topdown.png".format(prefix)), topdown)
    with open(os.path.join(debug_dir, "{}_summary.json".format(prefix)), "w") as f:
        json.dump({"debug_only_obstacle_names": names}, f, indent=4)


def build_pointcloud_context_fields(env, obs, args, step_i=0):
    depth_key = args.pc_depth_obs_key
    if depth_key is None:
        depth_key = "{}_depth".format(args.pc_camera_name)
    if depth_key not in obs:
        raise KeyError(
            "Pointcloud guidance requires depth obs '{}'. Make sure the guided env is created with depth enabled."
            .format(depth_key)
        )

    depth = np.asarray(obs[depth_key], dtype=np.float32)
    if depth.ndim == 4:
        depth = depth[-1]
    if depth.ndim == 2:
        depth = depth[..., None]
    height, width = depth.shape[:2]

    mask, obstacle_names = ObstacleGuidanceUtils.render_obstacle_mask(
        env=env,
        camera_name=args.pc_camera_name,
        height=height,
        width=width,
        target_object_name=args.target_object_name,
        obstacle_names=args.obstacle_names,
    )

    robomimic_env = ObstacleGuidanceUtils.get_robomimic_env(env)
    intrinsics = robomimic_env.get_camera_intrinsic_matrix(
        camera_name=args.pc_camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = robomimic_env.get_camera_extrinsic_matrix(camera_name=args.pc_camera_name)
    near, far = sim_depth_planes(env)
    workspace_bounds = default_workspace_bounds(env) if args.pc_workspace_crop else None

    points_world, pc_stats, points_camera = ObstacleGuidanceUtils.depth_mask_to_world_pointcloud(
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        near=near,
        far=far,
        workspace_bounds=workspace_bounds,
        voxel_size=args.pc_voxel_size,
        max_points=args.pc_max_points,
        return_camera_points=True,
    )

    if args.pc_debug_visualization and step_i % max(args.pc_debug_interval, 1) == 0:
        dump_pointcloud_debug(
            env=env,
            obs=obs,
            debug_dir=args.pc_debug_dir,
            camera_name=args.pc_camera_name,
            depth=depth,
            mask=mask,
            points_camera=points_camera,
            points_world=points_world,
            workspace_bounds=workspace_bounds if workspace_bounds is not None else default_workspace_bounds(env),
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            target_object_name=args.target_object_name,
            obstacle_names=args.obstacle_names,
            step_i=step_i,
        )

    return dict(
        obstacle_points_world=TensorUtils.to_numpy(points_world),
        pc_stats=pc_stats,
        pc_obstacle_names=obstacle_names,
    )


def set_obstacle_guidance_context(policy, env, obs, args, step_i=0):
    algo = getattr(policy, "policy", policy)
    if not hasattr(algo, "set_obstacle_guidance_context"):
        raise ValueError("Loaded policy does not support obstacle guidance context")

    current_eef_pos = get_current_eef_pos_from_obs(obs=obs, obs_key=args.eef_pos_obs_key)
    delta_pos_scale, delta_pos_offset = ObstacleGuidanceUtils.get_controller_delta_pos_mapping(env)
    action_scale, action_offset = get_action_normalization_vector(policy)

    context = dict(
        enabled=True,
        geometry_source=args.guidance_geometry_source,
        guidance_mode=(
            args.guidance_mode
            if args.guidance_geometry_source == "oracle_center"
            else "pointcloud_{}".format(args.pc_distance_mode)
        ),
        current_eef_pos=current_eef_pos,
        guidance_scale=args.guidance_scale,
        guidance_horizon=args.guidance_horizon,
        guidance_schedule=args.guidance_schedule,
        delta_pos_scale=delta_pos_scale,
        delta_pos_offset=delta_pos_offset,
        action_scale=action_scale,
        action_offset=action_offset,
        final_collision_refine=args.final_collision_refine,
        collision_refine_steps=args.collision_refine_steps,
        collision_refine_scale=args.collision_refine_scale,
        final_collision_cost_threshold=args.final_collision_cost_threshold,
    )

    info = dict(
        current_eef_pos=current_eef_pos,
        geometry_source=args.guidance_geometry_source,
        guidance_mode=args.guidance_mode,
        delta_pos_scale=delta_pos_scale,
    )

    if args.guidance_geometry_source == "oracle_center":
        centers_xyz, physical_radii, safety_radii, top_z, names = ObstacleGuidanceUtils.get_oracle_obstacle_geometry(
            env=env,
            target_object_name=args.target_object_name,
            obstacle_names=args.obstacle_names,
            xy_clearance=args.xy_clearance,
        )
        context.update(
            obstacle_centers_xyz=centers_xyz,
            obstacle_physical_radii=physical_radii,
            obstacle_radii=safety_radii,
            obstacle_top_z=top_z,
            xy_clearance=args.xy_clearance,
            z_clearance=args.z_clearance,
        )
        info.update(
            obstacle_names=names,
            obstacle_centers_xyz=centers_xyz,
            obstacle_physical_radii=physical_radii,
            obstacle_radii=safety_radii,
            obstacle_top_z=top_z,
        )
    elif args.guidance_geometry_source == "pointcloud":
        pc_fields = build_pointcloud_context_fields(env=env, obs=obs, args=args, step_i=step_i)
        context.update(
            obstacle_points_world=pc_fields["obstacle_points_world"],
            pc_safe_distance=args.pc_safe_distance,
            pc_distance_mode=args.pc_distance_mode,
        )
        info.update(pc_fields)
    else:
        raise ValueError("Unsupported guidance_geometry_source '{}'".format(args.guidance_geometry_source))

    algo.set_obstacle_guidance_context(context)
    return info


def rollout(policy, env, horizon, args, video_writer=None):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    video_count = 0
    guidance_costs = []
    guidance_min_distances = []
    point_counts = []
    guidance_chunk_count = getattr(getattr(policy, "policy", policy), "obstacle_guidance_sample_count", 0)
    log_printed = False

    for step_i in range(horizon):
        obstacle_info = set_obstacle_guidance_context(
            policy=policy,
            env=env,
            obs=obs,
            args=args,
            step_i=step_i,
        )
        if args.guidance_geometry_source == "pointcloud":
            point_counts.append(float(obstacle_info["pc_stats"]["point_count"]))

        if not log_printed:
            print("Obstacle guidance source: {}".format(obstacle_info["geometry_source"]))
            print("Obstacle guidance mode: {}".format(obstacle_info["guidance_mode"]))
            if args.guidance_geometry_source == "pointcloud":
                print("Pointcloud camera: {}".format(args.pc_camera_name))
                print("Pointcloud stats: {}".format(dict(obstacle_info["pc_stats"])))
                print("Pointcloud obstacle objects: {}".format(obstacle_info["pc_obstacle_names"]))
            else:
                print("Obstacle guidance objects: {}".format(obstacle_info["obstacle_names"]))
                print("Obstacle guidance centers xyz: {}".format(obstacle_info["obstacle_centers_xyz"].tolist()))
            log_printed = True

        act = policy(ob=obs)
        algo = getattr(policy, "policy", policy)
        new_guidance_chunk_count = getattr(algo, "obstacle_guidance_sample_count", guidance_chunk_count)
        if new_guidance_chunk_count != guidance_chunk_count:
            guidance_info = getattr(algo, "last_obstacle_guidance_info", None)
            if guidance_info is not None and guidance_info.get("applied", False):
                guidance_costs.append(guidance_info["cost"])
                min_dist = guidance_info.get("min_pointcloud_distance", guidance_info.get("min_distance", None))
                if min_dist is not None:
                    guidance_min_distances.append(float(np.min(min_dist)))
            guidance_chunk_count = new_guidance_chunk_count

        next_obs, reward, done, _ = env.step(act)
        total_reward += reward
        success = env.is_success()["task"]

        if args.render:
            env.render(mode="human", camera_name=args.camera_names[0])
        if video_writer is not None:
            if video_count % args.video_skip == 0:
                video_img = []
                for cam_name in args.camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_writer.append_data(np.concatenate(video_img, axis=1))
            video_count += 1

        obs = deepcopy(next_obs)
        if done or success:
            break

    stats = dict(
        Return=total_reward,
        Horizon=step_i + 1,
        Success_Rate=float(env.is_success()["task"]),
        Obstacle_Guidance_Cost=float(np.mean(guidance_costs)) if len(guidance_costs) > 0 else 0.0,
        Obstacle_Guidance_Min_Distance=(
            float(np.min(guidance_min_distances)) if len(guidance_min_distances) > 0 else 0.0
        ),
        Pointcloud_Point_Count=float(np.mean(point_counts)) if len(point_counts) > 0 else 0.0,
    )
    return stats


def run_obstacle_guided_agent(args):
    if args.render:
        assert len(args.camera_names) == 1

    write_video = args.video_path is not None
    needs_depth = args.guidance_geometry_source == "pointcloud"

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)
    if args.guidance_geometry_source == "pointcloud":
        raw_obs_keys = getattr(policy.policy.global_config, "all_obs_keys", [])
        if args.pc_depth_obs_key in raw_obs_keys:
            print("Warning: policy was trained with '{}'; PC-1 will still use it only for guidance.".format(args.pc_depth_obs_key))
        # The checkpoint was trained without depth keys, so ObsUtils would
        # otherwise classify agentview_depth as an unknown key and the
        # EnvRobosuite wrapper would drop it from observations.
        ObsUtils.OBS_KEYS_TO_MODALITIES[args.pc_depth_obs_key] = "depth"

    if args.horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        args.horizon = config.experiment.rollout.horizon

    env = env_from_checkpoint_for_guidance(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(write_video or needs_depth),
        use_depth_obs=needs_depth,
    )
    EnvUtils.set_env_specific_obs_processing(env=env)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = imageio.get_writer(args.video_path, fps=20) if write_video else None
    rollout_stats = []
    try:
        for rollout_i in range(args.n_rollouts):
            stats = rollout(
                policy=policy,
                env=env,
                horizon=args.horizon,
                args=args,
                video_writer=video_writer,
            )
            rollout_stats.append(stats)
            print(
                "Rollout {}/{}: success={:.0f}, return={:.3f}, horizon={}, guidance_cost={:.6g}".format(
                    rollout_i + 1,
                    args.n_rollouts,
                    stats["Success_Rate"],
                    stats["Return"],
                    int(stats["Horizon"]),
                    stats["Obstacle_Guidance_Cost"],
                ),
                flush=True,
            )
    finally:
        if video_writer is not None:
            video_writer.close()

    per_rollout_stats = rollout_stats
    rollout_stats_by_key = TensorUtils.list_of_flat_dict_to_dict_of_list(per_rollout_stats)
    avg_rollout_stats = {k: float(np.mean(rollout_stats_by_key[k])) for k in rollout_stats_by_key}
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats_by_key["Success_Rate"]))
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))
    if args.stats_path is not None:
        stats_dir = os.path.dirname(args.stats_path)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_path, "w") as f:
            json.dump(
                dict(
                    average=avg_rollout_stats,
                    rollouts=[{k: float(v) for k, v in stats.items()} for stats in per_rollout_stats],
                    args=vars(args),
                ),
                f,
                indent=4,
            )
        print("Wrote rollout stats to {}".format(args.stats_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=None, help="optional rollout horizon override")
    parser.add_argument("--env", type=str, default=None, help="optional environment name override")
    parser.add_argument("--render", action="store_true", help="render on-screen")
    parser.add_argument("--video_path", type=str, default=None, help="optional rollout video path")
    parser.add_argument("--video_skip", type=int, default=5, help="write every n-th frame to video")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"], help="video/render camera names")
    parser.add_argument("--seed", type=int, default=None, help="optional rollout seed")
    parser.add_argument("--stats_path", type=str, default=None, help="optional JSON path for rollout stats")

    parser.add_argument(
        "--guidance_geometry_source",
        type=str,
        choices=["oracle_center", "pointcloud"],
        default="pointcloud",
        help="obstacle geometry source for guidance",
    )
    parser.add_argument("--guidance_scale", type=float, default=0.03, help="guidance gradient step scale")
    parser.add_argument(
        "--guidance_mode",
        type=str,
        choices=["xy", "xyz_cylinder"],
        default="xyz_cylinder",
        help="oracle-center guidance mode; ignored for pointcloud",
    )
    parser.add_argument("--xy_clearance", type=float, default=0.02, help="oracle-center xy clearance in metres")
    parser.add_argument("--z_clearance", type=float, default=0.03, help="oracle-center z clearance in metres")
    parser.add_argument("--guidance_horizon", type=int, default=8, help="number of predicted action steps in cost")
    parser.add_argument(
        "--guidance_schedule",
        type=str,
        choices=["constant", "late"],
        default="late",
        help="guidance scale schedule over denoising steps",
    )
    parser.add_argument("--target_object_name", type=str, default="Can", help="target object excluded from obstacles")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None, help="optional explicit obstacle names")
    parser.add_argument("--eef_pos_obs_key", type=str, default="robot0_eef_pos", help="EEF position observation key")
    parser.add_argument("--final_collision_refine", action="store_true", help="post-hoc final action refinement")
    parser.add_argument("--collision_refine_steps", type=int, default=5, help="final refinement steps")
    parser.add_argument("--collision_refine_scale", type=float, default=0.02, help="final refinement gradient scale")
    parser.add_argument(
        "--final_collision_cost_threshold",
        type=float,
        default=1e-8,
        help="surrogate cost threshold for final refinement",
    )

    parser.add_argument("--pc_camera_name", type=str, default="agentview", help="single camera used for PC-1")
    parser.add_argument("--pc_depth_obs_key", type=str, default="agentview_depth", help="depth obs key used for PC-1")
    parser.add_argument("--pc_safe_distance", type=float, default=0.02, help="pointcloud safe distance in metres")
    parser.add_argument(
        "--pc_distance_mode",
        type=str,
        choices=["xy", "xyz"],
        default="xy",
        help="pointcloud nearest-distance mode",
    )
    parser.add_argument("--pc_voxel_size", type=float, default=0.005, help="pointcloud voxel size in metres")
    parser.add_argument("--pc_max_points", type=int, default=1024, help="max pointcloud points used by guidance")
    parser.add_argument("--pc_workspace_crop", action="store_true", default=True, help="crop points to tabletop workspace")
    parser.add_argument("--pc_no_workspace_crop", action="store_false", dest="pc_workspace_crop")
    parser.add_argument("--pc_debug_visualization", action="store_true", help="dump PC-1 debug images and arrays")
    parser.add_argument("--pc_debug_dir", type=str, default="outputs/pc1_debug", help="PC-1 debug output directory")
    parser.add_argument("--pc_debug_interval", type=int, default=25, help="debug dump interval in env steps")
    return parser.parse_args()


if __name__ == "__main__":
    run_obstacle_guided_agent(parse_args())
