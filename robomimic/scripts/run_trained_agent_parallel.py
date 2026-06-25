"""
Multi-process parallel rollout script for robomimic.

Distributes rollouts across N worker processes, each with its own
environment + policy instance. Supports video and dataset output.

Usage:
    MUJOCO_GL=egl uv run python run_trained_agent_parallel.py \\
        --agent /path/to/model.pth --n_rollouts 50 --n_workers 8 \\
        --seed 42 --stats_path /tmp/stats.json
"""
import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import multiprocessing as mp

import h5py
import imageio
import numpy as np

import torch

from run_trained_agent import (
    rollout,
    is_scalar_number,
    make_json_serializable,
    write_command_file,
)

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils


def _resolve_log_dir(args):
    """Pick an output-adjacent directory for per-worker log files."""
    for candidate in (args.stats_path, args.video_path, args.dataset_path):
        if candidate is not None:
            d = os.path.dirname(os.path.abspath(candidate))
            if d:
                return os.path.join(d, "worker_logs")
    return os.path.join(tempfile.gettempdir(), "parallel_worker_logs")


def _worker_fn(rank, args, ckpt_path, rollout_start, rollout_end, result_queue, log_dir):
    """Worker process: creates env + policy, runs assigned rollouts, sends results back."""
    # redirect stdout to a per-worker log file
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker_{}.log".format(rank))
    sys.stdout = open(log_path, "w", buffering=1)

    # seed RNGs for independent streams
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2 ** 31 - 1)
    worker_seed = base_seed + rank
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

    # load policy (safe with spawn start method)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=args.verbose_load,
    )

    # read horizon from config if not overridden
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=(args.video_path is not None),
        verbose=args.verbose_load,
    )

    # per-worker temp video writer
    video_writer = None
    temp_video_path = None
    if args.video_path is not None:
        temp_video_path = os.path.join(
            tempfile.gettempdir(), "parallel_rollout_video_w{}.mp4".format(rank),
        )
        video_writer = imageio.get_writer(temp_video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    return_obs = (write_dataset and args.dataset_obs)

    for i in range(rollout_start, rollout_end):
        progress_prefix = "Rollout {}/{} [worker {}]".format(i + 1, args.n_rollouts, rank)
        print("{} start".format(progress_prefix), flush=True)
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=return_obs,
            camera_names=args.camera_names,
            video_target_mask_grid=args.video_target_mask_grid,
            progress_prefix=progress_prefix,
            progress_interval=args.progress_interval,
            target_object_name=args.target_object_name,
            obstacle_names=args.obstacle_names,
        )
        result_queue.put((i, stats, traj if write_dataset else None))
        print(
            "{} done: horizon={}, success={}, return={:.4f}".format(
                progress_prefix,
                stats["Horizon"],
                int(stats["Success_Rate"]),
                stats["Return"],
            ),
            flush=True,
        )

    if video_writer is not None:
        video_writer.close()

    try:
        env.close()
    except AttributeError:
        pass


def aggregate_rollout_stats(per_rollout_stats, args, stats_path_override=None):
    """Aggregate per-rollout stats, print averages, and optionally write to stats_path."""
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

    stats_path = stats_path_override or args.stats_path
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


def concat_videos(temp_dir, output_path, n_workers):
    """Concatenate per-worker temp videos into a single output file using ffmpeg."""
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for rank in range(n_workers):
            vpath = os.path.join(temp_dir, "parallel_rollout_video_w{}.mp4".format(rank))
            if os.path.exists(vpath):
                f.write("file '{}'\n".format(vpath))
    if os.path.getsize(concat_file) == 0:
        print("Warning: no worker videos found, skipping concat")
        return
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]
    print("Concatenating worker videos ...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    print("Wrote concatenated video to {}".format(output_path))


def run_trained_agent_parallel(args):
    """Parallel rollout entry point."""
    n_workers = args.n_workers
    mp.set_start_method("spawn", force=True)

    write_command_file(args)

    log_dir = _resolve_log_dir(args)

    # distribute rollouts among workers
    num_r = args.n_rollouts
    ckpt_path = args.agent
    assignments = []
    start = 0
    for rank in range(n_workers):
        count = num_r // n_workers + (1 if rank < num_r % n_workers else 0)
        if count > 0:
            assignments.append((rank, start, start + count))
            start += count

    result_queue = mp.Queue()
    workers = []
    for rank, r_start, r_end in assignments:
        p = mp.Process(
            target=_worker_fn,
            args=(rank, args, ckpt_path, r_start, r_end, result_queue, log_dir),
        )
        p.start()
        workers.append(p)

    # collect results with progress bar
    rollout_stats = [None] * num_r
    rollout_trajs = [None] * num_r
    for i in range(num_r):
        idx, stats, traj = result_queue.get()
        rollout_stats[idx] = stats
        rollout_trajs[idx] = traj
        print("\rProgress: {}/{}".format(i + 1, num_r), end="", flush=True)
    print()

    for p in workers:
        p.join()

    # write dataset hdf5 (if requested)
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        dataset_dir = os.path.dirname(args.dataset_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
        total_samples = 0
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        for i, traj in enumerate(rollout_trajs):
            if traj is None:
                print("Warning: rollout {} has no trajectory data, skipping".format(i))
                continue
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]
        data_grp.attrs["total"] = total_samples
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))

    # concat worker videos (if requested)
    if args.video_path is not None:
        concat_videos(tempfile.gettempdir(), args.video_path, n_workers)

    # aggregate and report stats
    effective_stats_path = args.stats_path
    if effective_stats_path is None and args.video_path is not None:
        effective_stats_path = os.path.splitext(args.video_path)[0] + "_stats.json"
    aggregate_rollout_stats(rollout_stats, args, stats_path_override=effective_stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel rollout evaluation for robomimic policies",
    )

    parser.add_argument("--agent", type=str, required=True,
                        help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27,
                        help="number of rollouts")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="number of parallel worker processes")
    parser.add_argument("--horizon", type=int, default=None,
                        help="override maximum horizon from checkpoint")
    parser.add_argument("--env", type=str, default=None,
                        help="override environment name from checkpoint")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed for rollouts (each worker gets seed + rank)")

    # video
    parser.add_argument("--video_path", type=str, default=None,
                        help="output video path (worker videos concatenated)")
    parser.add_argument("--video_skip", type=int, default=5,
                        help="render frames to video every n steps")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"],
                        help="camera name(s) for video rendering")
    parser.add_argument("--video_target_mask_grid", action="store_true",
                        help="render split-frame video: RGB left, policy obs right")

    # dataset
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="if provided, write rollout trajectories to hdf5")
    parser.add_argument("--dataset_obs", action="store_true",
                        help="include observations in dataset hdf5")

    # stats
    parser.add_argument("--stats_path", type=str, default=None,
                        help="write rollout stats to this JSON path")

    # collision tracking
    parser.add_argument("--target_object_name", type=str, default="Can",
                        help="target object excluded from non-target collision counts")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None,
                        help="explicit non-target object names for collision counts")

    # misc
    parser.add_argument("--progress_interval", type=int, default=100,
                        help="print step progress every n steps")
    parser.add_argument("--verbose_load", action="store_true",
                        help="print checkpoint/policy details on load")

    args = parser.parse_args()
    run_trained_agent_parallel(args)
