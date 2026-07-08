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

"""
import argparse
import json
import os
import shlex
import sys
import h5py
import imageio
import numpy as np
from collections import OrderedDict
from copy import deepcopy

import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def make_json_serializable(x):
    """
    Convert numpy values to Python containers for json dumping.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: make_json_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_serializable(v) for v in x]
    return x


def is_scalar_number(x):
    return isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_))


def command_output_dir(args):
    for path in (args.stats_path, args.video_path, args.dataset_path):
        if path is not None:
            dirname = os.path.dirname(path)
            if dirname:
                return dirname
    return None


def write_command_file(args):
    output_dir = command_output_dir(args)
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    command = shlex.join([sys.executable] + sys.argv)
    if "MUJOCO_GL" in os.environ:
        command = "MUJOCO_GL={} {}".format(shlex.quote(os.environ["MUJOCO_GL"]), command)
    path = os.path.join(output_dir, "command.txt")
    with open(path, "w") as f:
        f.write(command)
        f.write("\n")
    print("Wrote rollout command to {}".format(path))


def unwrap_env(env):
    while isinstance(env, EnvWrapper):
        env = env.env
    return env


def get_raw_env(env):
    env = unwrap_env(env)
    return getattr(env, "env", env)


def sim_geom_id(sim, geom_name):
    try:
        return sim.model.geom_name2id(geom_name)
    except Exception:
        return None


def object_is_active_in_scene(sim, obj):
    for attr in ("root_body", "body_name", "root_body_name"):
        body_name = getattr(obj, attr, None)
        if isinstance(body_name, (list, tuple)):
            body_name = body_name[0] if len(body_name) > 0 else None
        if body_name is None:
            continue
        try:
            body_id = sim.model.body_name2id(body_name)
        except Exception:
            continue
        return bool(sim.model.body_pos[body_id][2] > -10.0)
    return True


def iter_non_target_objects(raw_env, target_object_name=None, obstacle_names=None):
    target_lower = target_object_name.lower() if target_object_name is not None else None
    obstacle_name_set = set(name.lower() for name in obstacle_names) if obstacle_names else None
    for obj in getattr(raw_env, "objects", []):
        name = getattr(obj, "name", None)
        if name is None:
            continue
        name_lower = name.lower()
        if target_lower is not None and name_lower == target_lower:
            continue
        if obstacle_name_set is not None and name_lower not in obstacle_name_set:
            continue
        yield obj, name


def get_obstacle_contact_geom_ids_by_name(env, target_object_name=None, obstacle_names=None):
    raw_env = get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        raise ValueError("Obstacle contact tracking requires simulator access")

    geom_ids_by_name = OrderedDict()
    for obj, name in iter_non_target_objects(raw_env, target_object_name, obstacle_names):
        if not object_is_active_in_scene(sim=sim, obj=obj):
            continue
        geom_ids = []
        for geom_name in getattr(obj, "contact_geoms", []):
            geom_id = sim_geom_id(sim, geom_name)
            if geom_id is not None:
                geom_ids.append(geom_id)
        if len(geom_ids) > 0:
            geom_ids_by_name[name] = sorted(set(geom_ids))
    return geom_ids_by_name


def get_robot_contact_geom_ids(env):
    raw_env = get_raw_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        raise ValueError("Robot contact tracking requires simulator access")

    geom_names = []
    for robot in getattr(raw_env, "robots", []):
        robot_model = getattr(robot, "robot_model", None)
        if robot_model is not None:
            geom_names.extend(getattr(robot_model, "contact_geoms", []))

        gripper = getattr(robot, "gripper", None)
        grippers = gripper.values() if isinstance(gripper, dict) else [gripper]
        for grip in grippers:
            if grip is None:
                continue
            geom_names.extend(getattr(grip, "contact_geoms", []))

    geom_ids = []
    for geom_name in geom_names:
        geom_id = sim_geom_id(sim, geom_name)
        if geom_id is not None:
            geom_ids.append(geom_id)
    return sorted(set(geom_ids))


def make_non_target_collision_tracker(env, target_object_name="Can", obstacle_names=None):
    raw_env = get_raw_env(env)
    robot_geom_ids = set(get_robot_contact_geom_ids(env))
    object_geom_ids_by_name = get_obstacle_contact_geom_ids_by_name(
        env=env,
        target_object_name=target_object_name,
        obstacle_names=obstacle_names,
    )
    object_name_by_geom_id = {}
    for obj_name, geom_ids in object_geom_ids_by_name.items():
        for geom_id in geom_ids:
            object_name_by_geom_id[geom_id] = obj_name
    return dict(
        raw_env=raw_env,
        robot_geom_ids=robot_geom_ids,
        object_name_by_geom_id=object_name_by_geom_id,
        object_counts={name: 0 for name in object_geom_ids_by_name},
    )


def update_non_target_collision_counts(collision_tracker):
    raw_env = collision_tracker["raw_env"]
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return False

    robot_geom_ids = collision_tracker["robot_geom_ids"]
    object_name_by_geom_id = collision_tracker["object_name_by_geom_id"]
    touched_objects = set()
    for contact_i in range(sim.data.ncon):
        contact = sim.data.contact[contact_i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in robot_geom_ids and geom2 in object_name_by_geom_id:
            touched_objects.add(object_name_by_geom_id[geom2])
        elif geom2 in robot_geom_ids and geom1 in object_name_by_geom_id:
            touched_objects.add(object_name_by_geom_id[geom1])

    for obj_name in touched_objects:
        collision_tracker["object_counts"][obj_name] += 1
    return len(touched_objects) > 0


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
    progress_prefix=None,
    progress_interval=100,
    target_object_name="Can",
    obstacle_names=None,
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
    non_target_collision_step_count = 0
    collision_tracker = make_non_target_collision_tracker(
        env=env,
        target_object_name=target_object_name,
        obstacle_names=obstacle_names,
    )
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]
            if update_non_target_collision_counts(collision_tracker):
                non_target_collision_step_count += 1

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

            if (
                progress_prefix is not None
                and progress_interval is not None
                and progress_interval > 0
                and ((step_i + 1) % progress_interval == 0)
            ):
                print("{} step {}/{}".format(progress_prefix, step_i + 1, horizon), flush=True)

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    rollout_horizon = step_i + 1
    non_target_collision_count = int(sum(collision_tracker["object_counts"].values()))
    stats = dict(
        Return=total_reward,
        Horizon=rollout_horizon,
        Success_Rate=float(success),
        Obstacle_Guidance_Cost=None,
        Obstacle_Guidance_Min_Distance=None,
        Obstacle_Guidance_Trigger_Count=None,
        Obstacle_Guidance_Trigger_Rate=None,
        Obstacle_Guidance_Positive_Cost_Count=None,
        Obstacle_Guidance_Positive_Cost_Rate=None,
        Non_Target_Collision_Count=float(non_target_collision_count),
        Non_Target_Collision_Step_Count=float(non_target_collision_step_count),
        Non_Target_Collision_Rate=float(non_target_collision_step_count / max(rollout_horizon, 1)),
        Non_Target_Collision_Any=float(non_target_collision_step_count > 0),
        Non_Target_Collision_Object_Counts={
            obj_name: int(count) for obj_name, count in collision_tracker["object_counts"].items()
        },
        Pointcloud_Total_Point_Count=None,
        Pointcloud_Point_Count=None,
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
    write_command_file(args)

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=args.verbose_load)

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
        verbose=args.verbose_load,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        dataset_dir = os.path.dirname(args.dataset_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(rollout_num_episodes):
        progress_prefix = "Rollout {}/{}".format(i + 1, rollout_num_episodes)
        print("{} start".format(progress_prefix), flush=True)
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
            progress_prefix=progress_prefix,
            progress_interval=args.progress_interval,
            target_object_name=args.target_object_name,
            obstacle_names=args.obstacle_names,
        )
        rollout_stats.append(stats)
        print(
            "{} done: horizon={}, success={}, return={:.4f}".format(
                progress_prefix,
                stats["Horizon"],
                int(stats["Success_Rate"]),
                stats["Return"],
            ),
            flush=True,
        )

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

    per_rollout_stats = rollout_stats
    scalar_keys = [
        k for k, v in per_rollout_stats[0].items()
        if is_scalar_number(v)
    ]
    rollout_stats_by_key = {
        k: [float(stats[k]) for stats in per_rollout_stats]
        for k in scalar_keys
    }
    avg_rollout_stats = {k: float(np.mean(rollout_stats_by_key[k])) for k in rollout_stats_by_key}
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats_by_key["Success_Rate"]))
    if "Non_Target_Collision_Any" in rollout_stats_by_key:
        avg_rollout_stats["Num_Non_Target_Collision_Rollouts"] = float(
            np.sum(rollout_stats_by_key["Non_Target_Collision_Any"])
        )
    avg_rollout_stats.update(
        Obstacle_Guidance_Cost=None,
        Obstacle_Guidance_Min_Distance=None,
        Obstacle_Guidance_Trigger_Count=None,
        Obstacle_Guidance_Trigger_Rate=None,
        Obstacle_Guidance_Positive_Cost_Count=None,
        Obstacle_Guidance_Positive_Cost_Rate=None,
        Pointcloud_Total_Point_Count=None,
        Pointcloud_Point_Count=None,
    )

    total_collision_counts = {}
    for stats in per_rollout_stats:
        for obj_name, count in stats.get("Non_Target_Collision_Object_Counts", {}).items():
            total_collision_counts[obj_name] = total_collision_counts.get(obj_name, 0) + int(count)
    avg_collision_counts = {
        obj_name: float(count / max(len(per_rollout_stats), 1))
        for obj_name, count in total_collision_counts.items()
    }
    print("Average Rollout Stats")
    print(json.dumps(make_json_serializable(avg_rollout_stats), indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    stats_path = args.stats_path
    if stats_path is None and write_video:
        stats_path = os.path.splitext(args.video_path)[0] + "_stats.json"
    if stats_path is not None:
        stats_dir = os.path.dirname(stats_path)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        stats_payload = dict(
            average=avg_rollout_stats,
            totals=dict(
                Non_Target_Collision_Object_Counts=total_collision_counts,
                Obstacle_Guidance_Trigger_Count=None,
                Obstacle_Guidance_Positive_Cost_Count=None,
            ),
            per_rollout_average=dict(
                Non_Target_Collision_Object_Counts=avg_collision_counts,
            ),
            rollouts=[make_json_serializable(stats) for stats in per_rollout_stats],
            args=vars(args),
        )
        with open(stats_path, "w") as f:
            json.dump(make_json_serializable(stats_payload), f, indent=4)
        print("Wrote rollout stats to {}".format(stats_path))


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
        "--stats_path",
        type=str,
        default=None,
        help="(optional) write rollout stats to this JSON path",
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

    parser.add_argument(
        "--target_object_name",
        type=str,
        default="Can",
        help="target object excluded from non-target collision counts",
    )

    parser.add_argument(
        "--obstacle_names",
        type=str,
        nargs="*",
        default=None,
        help="optional explicit non-target object names for collision counts",
    )

    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="print rollout progress every n environment steps; set <= 0 to disable step progress",
    )

    parser.add_argument(
        "--verbose_load",
        action="store_true",
        help="print full checkpoint policy and environment details while loading",
    )

    args = parser.parse_args()
    run_trained_agent(args)
