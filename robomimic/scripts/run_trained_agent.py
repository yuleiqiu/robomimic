"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5

    # Evaluate a masked-image Diffusion Policy with obstacle guidance.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --obstacle_guidance --guidance_scale 0.03 \
        --guidance_mode xyz_cylinder --xy_clearance 0.02 --z_clearance 0.03 --guidance_horizon 8 \
        --guidance_schedule late --target_object_name Can
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def resize_nearest(image, height, width):
    """
    Resize an HWC image with nearest-neighbor sampling.
    """
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[y_idx][:, x_idx]


def make_target_mask_grid_video_frame(env, obs, camera_names, height=512, width=512):
    """
    Builds a 2-column video frame for each camera: original RGB render on the
    left, policy observation image on the right. Rows correspond to cameras.
    """
    rows = []
    for cam_name in camera_names:
        rgb = env.render(mode="rgb_array", height=height, width=width, camera_name=cam_name)
        obs_key = "{}_image".format(cam_name)
        if obs_key not in obs:
            raise KeyError("Observation key '{}' not found for target-mask video".format(obs_key))
        mask = obs[obs_key]
        if mask.ndim == 4:
            mask = mask[-1]
        if mask.shape[:2] != (height, width):
            mask = resize_nearest(mask, height=height, width=width)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[-1] == 1:
            mask = np.repeat(mask, 3, axis=-1)
        rows.append(np.concatenate([rgb, mask.astype(np.uint8)], axis=1))
    return np.concatenate(rows, axis=0)


def get_current_eef_pos_from_obs(obs, obs_key="robot0_eef_pos"):
    """
    Return current eef position from a rollout observation dict.
    """
    if obs_key not in obs:
        raise KeyError("Observation key '{}' is required for obstacle guidance".format(obs_key))
    eef_pos = np.array(obs[obs_key], dtype=np.float32)
    if eef_pos.ndim == 2:
        eef_pos = eef_pos[-1]
    if eef_pos.shape[-1] != 3:
        raise ValueError("Expected '{}' to have final dimension 3, got {}".format(obs_key, eef_pos.shape))
    return eef_pos


def get_action_normalization_vector(policy):
    """
    Return flat action unnormalization scale / offset from RolloutPolicy stats.
    """
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


def set_obstacle_guidance_context(policy, env, obs, guidance_config):
    """
    Query oracle obstacle state from the env and pass it into the underlying
    policy for optional inference-time guidance.
    """
    algo = getattr(policy, "policy", policy)
    if not hasattr(algo, "set_obstacle_guidance_context"):
        raise ValueError("Loaded policy does not support obstacle guidance context")

    if not guidance_config.get("enabled", False):
        algo.set_obstacle_guidance_context(None)
        return None

    current_eef_pos = get_current_eef_pos_from_obs(
        obs=obs,
        obs_key=guidance_config.get("eef_pos_obs_key", "robot0_eef_pos"),
    )
    centers_xyz, physical_radii, safety_radii, top_z, names = ObstacleGuidanceUtils.get_oracle_obstacle_geometry(
        env=env,
        target_object_name=guidance_config.get("target_object_name", None),
        obstacle_names=guidance_config.get("obstacle_names", None),
        xy_clearance=guidance_config.get("xy_clearance", 0.02),
    )
    delta_pos_scale, delta_pos_offset = ObstacleGuidanceUtils.get_controller_delta_pos_mapping(env)
    action_scale, action_offset = get_action_normalization_vector(policy)

    context = dict(
        enabled=True,
        guidance_mode=guidance_config.get("guidance_mode", "xyz_cylinder"),
        current_eef_pos=current_eef_pos,
        obstacle_centers_xyz=centers_xyz,
        obstacle_physical_radii=physical_radii,
        obstacle_radii=safety_radii,
        obstacle_top_z=top_z,
        xy_clearance=guidance_config.get("xy_clearance", 0.02),
        z_clearance=guidance_config.get("z_clearance", 0.03),
        guidance_scale=guidance_config.get("guidance_scale", 0.0),
        guidance_horizon=guidance_config.get("guidance_horizon", 8),
        guidance_schedule=guidance_config.get("guidance_schedule", "late"),
        delta_pos_scale=delta_pos_scale,
        delta_pos_offset=delta_pos_offset,
        action_scale=action_scale,
        action_offset=action_offset,
    )
    algo.set_obstacle_guidance_context(context)
    return dict(
        current_eef_pos=current_eef_pos,
        guidance_mode=context["guidance_mode"],
        obstacle_centers_xyz=centers_xyz,
        obstacle_physical_radii=physical_radii,
        obstacle_radii=safety_radii,
        obstacle_top_z=top_z,
        obstacle_names=names,
        xy_clearance=context["xy_clearance"],
        z_clearance=context["z_clearance"],
        delta_pos_scale=delta_pos_scale,
    )


def min_eef_obstacle_xy_distance(eef_pos, obstacle_centers):
    """
    Compute current xy distance from eef to nearest obstacle.
    """
    if obstacle_centers is None or len(obstacle_centers) == 0:
        return None
    centers_xy = np.array(obstacle_centers)[..., :2]
    dist = np.linalg.norm(centers_xy - np.array(eef_pos[:2])[None], axis=-1)
    return float(np.min(dist))


def min_eef_obstacle_z_clearance(eef_pos, obstacle_top_z, z_clearance):
    """
    Compute current z clearance from eef to nearest obstacle z limit.
    """
    if obstacle_top_z is None or len(obstacle_top_z) == 0:
        return None
    clearance = float(eef_pos[2]) - (np.array(obstacle_top_z, dtype=np.float32) + float(z_clearance))
    return float(np.min(clearance))


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
    video_target_mask_grid=False,
    obstacle_guidance_config=None,
):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    guidance_costs = []
    guidance_min_distances = []
    guidance_min_z_clearances = []
    actual_min_distances = []
    actual_min_z_clearances = []
    guidance_chunk_count = getattr(getattr(policy, "policy", policy), "obstacle_guidance_sample_count", 0)
    obstacle_log_printed = False

    try:
        for step_i in range(horizon):

            # get action from policy
            obstacle_info = None
            if obstacle_guidance_config is not None and obstacle_guidance_config.get("enabled", False):
                obstacle_info = set_obstacle_guidance_context(
                    policy=policy,
                    env=env,
                    obs=obs,
                    guidance_config=obstacle_guidance_config,
                )
                if not obstacle_log_printed:
                    print("Obstacle guidance mode: {}".format(obstacle_info["guidance_mode"]))
                    print("Obstacle guidance objects: {}".format(obstacle_info["obstacle_names"]))
                    print("Obstacle guidance centers xyz: {}".format(obstacle_info["obstacle_centers_xyz"].tolist()))
                    print("Obstacle guidance physical xy radii: {}".format(obstacle_info["obstacle_physical_radii"].tolist()))
                    print("Obstacle guidance safety radii: {}".format(obstacle_info["obstacle_radii"].tolist()))
                    print("Obstacle guidance top z: {}".format(obstacle_info["obstacle_top_z"].tolist()))
                    print("Obstacle guidance xy_clearance: {}".format(obstacle_info["xy_clearance"]))
                    print("Obstacle guidance z_clearance: {}".format(obstacle_info["z_clearance"]))
                    print("Obstacle guidance delta_pos_scale: {}".format(obstacle_info["delta_pos_scale"].tolist()))
                    obstacle_log_printed = True
                actual_min_dist = min_eef_obstacle_xy_distance(
                    eef_pos=obstacle_info["current_eef_pos"],
                    obstacle_centers=obstacle_info["obstacle_centers_xyz"],
                )
                if actual_min_dist is not None:
                    actual_min_distances.append(actual_min_dist)
                actual_min_z_clearance = min_eef_obstacle_z_clearance(
                    eef_pos=obstacle_info["current_eef_pos"],
                    obstacle_top_z=obstacle_info["obstacle_top_z"],
                    z_clearance=obstacle_info["z_clearance"],
                )
                if actual_min_z_clearance is not None:
                    actual_min_z_clearances.append(actual_min_z_clearance)
            act = policy(ob=obs)
            algo = getattr(policy, "policy", policy)
            new_guidance_chunk_count = getattr(algo, "obstacle_guidance_sample_count", guidance_chunk_count)
            if new_guidance_chunk_count != guidance_chunk_count:
                guidance_info = getattr(algo, "last_obstacle_guidance_info", None)
                if guidance_info is not None and guidance_info.get("applied", False):
                    guidance_costs.append(guidance_info["cost"])
                    min_xy_distance = guidance_info.get("min_xy_distance", guidance_info.get("min_distance", None))
                    if min_xy_distance is not None:
                        guidance_min_distances.append(float(np.min(min_xy_distance)))
                    min_z_clearance = guidance_info.get("min_z_clearance", None)
                    if min_z_clearance is not None:
                        guidance_min_z_clearances.append(float(np.min(min_z_clearance)))
                guidance_chunk_count = new_guidance_chunk_count

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    if video_target_mask_grid:
                        video_img = make_target_mask_grid_video_frame(
                            env=env,
                            obs=next_obs,
                            camera_names=camera_names,
                            height=512,
                            width=512,
                        )
                    else:
                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    if obstacle_guidance_config is not None and obstacle_guidance_config.get("enabled", False):
        stats["Obstacle_Guidance_Applied"] = float(len(guidance_costs) > 0)
        stats["Obstacle_Guidance_Cost"] = float(np.mean(guidance_costs)) if len(guidance_costs) > 0 else 0.0
        stats["Obstacle_Guidance_Min_Distance"] = (
            float(np.min(guidance_min_distances)) if len(guidance_min_distances) > 0 else 0.0
        )
        stats["Obstacle_Guidance_Min_XY_Distance"] = stats["Obstacle_Guidance_Min_Distance"]
        stats["Obstacle_Guidance_Min_Z_Clearance"] = (
            float(np.min(guidance_min_z_clearances)) if len(guidance_min_z_clearances) > 0 else 0.0
        )
        stats["Actual_Min_Eef_Obstacle_Distance"] = (
            float(np.min(actual_min_distances)) if len(actual_min_distances) > 0 else 0.0
        )
        stats["Actual_Min_Eef_Obstacle_Z_Clearance"] = (
            float(np.min(actual_min_z_clearances)) if len(actual_min_z_clearances) > 0 else 0.0
        )

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1
    if args.video_target_mask_grid and args.camera_names == ["agentview"]:
        args.camera_names = ["agentview", "robot0_eye_in_hand"]
    obstacle_guidance_config = dict(
        enabled=args.obstacle_guidance,
        guidance_mode=args.guidance_mode,
        guidance_scale=args.guidance_scale,
        xy_clearance=args.xy_clearance,
        z_clearance=args.z_clearance,
        guidance_horizon=args.guidance_horizon,
        guidance_schedule=args.guidance_schedule,
        target_object_name=args.target_object_name,
        obstacle_names=args.obstacle_names,
        eef_pos_obs_key=args.eef_pos_obs_key,
    )

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            video_target_mask_grid=args.video_target_mask_grid,
            obstacle_guidance_config=obstacle_guidance_config,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    parser.add_argument(
        "--video_target_mask_grid",
        action="store_true",
        help="render a 2-column video per camera: original RGB on the left and policy image observation on the right",
    )

    parser.add_argument(
        "--obstacle_guidance",
        action="store_true",
        help="enable inference-time obstacle guidance for diffusion policy sampling",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.03,
        help="obstacle guidance gradient step scale",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        choices=["xy", "xyz_cylinder"],
        default="xyz_cylinder",
        help="obstacle guidance cost mode",
    )
    parser.add_argument(
        "--xy_clearance",
        type=float,
        default=0.02,
        help="xy safety margin added to simulator-derived obstacle radius in meters",
    )
    parser.add_argument(
        "--z_clearance",
        type=float,
        default=0.03,
        help="minimum eef clearance above obstacle top z in meters for xyz_cylinder guidance",
    )
    parser.add_argument(
        "--guidance_horizon",
        type=int,
        default=8,
        help="number of predicted action steps used in obstacle cost",
    )
    parser.add_argument(
        "--guidance_schedule",
        type=str,
        choices=["constant", "late"],
        default="late",
        help="guidance scale schedule over denoising steps",
    )
    parser.add_argument(
        "--target_object_name",
        type=str,
        default="Can",
        help="object name to exclude from obstacle guidance",
    )
    parser.add_argument(
        "--obstacle_names",
        type=str,
        nargs="*",
        default=None,
        help="optional explicit obstacle object names; defaults to all non-target objects",
    )
    parser.add_argument(
        "--eef_pos_obs_key",
        type=str,
        default="robot0_eef_pos",
        help="observation key used for current end-effector position",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)
