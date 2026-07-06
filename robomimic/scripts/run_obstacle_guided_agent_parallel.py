"""
Parallel rollout evaluation for obstacle-guided Diffusion Policy.

This mirrors run_trained_agent_parallel.py, but uses the guided rollout path
from run_obstacle_guided_agent.py, including optional forward-model trajectory
guidance and optional per-worker video writing.

Example:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python \\
        third_party/robomimic/robomimic/scripts/run_obstacle_guided_agent_parallel.py \\
        --agent outputs/robomimic/checkpoints/diffusion_policy_can_yq_masked_image/model_epoch_140_image_v15_can_mask_success_1.0.pth \\
        --env PickPlaceBreadCan --n_rollouts 10 --n_workers 4 \\
        --trajectory_backend forward_model \\
        --forward_model_path outputs/forward_model/osc_eef_forward_image_v15/model.pth \\
        --stats_path /tmp/guided_parallel_stats.json
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import shlex
import subprocess
import sys
import tempfile
from copy import deepcopy

import imageio
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.osc_forward_model_utils as OSCForwardModelUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.guided_diffusion_policy import wrap_as_guided

from run_obstacle_guided_agent import (
    is_scalar_number,
    make_json_serializable,
    rollout,
    serializable_args,
)


def command_output_dir(args):
    for path in (args.stats_path, args.video_path):
        if path is not None:
            dirname = os.path.dirname(os.path.abspath(path))
            if dirname:
                return dirname
    return None


def write_command_file(args):
    output_dir = command_output_dir(args)
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    command = shlex.join([sys.executable] + sys.argv)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        command = "CUDA_VISIBLE_DEVICES={} {}".format(shlex.quote(os.environ["CUDA_VISIBLE_DEVICES"]), command)
    if "MUJOCO_GL" in os.environ:
        command = "MUJOCO_GL={} {}".format(shlex.quote(os.environ["MUJOCO_GL"]), command)
    path = os.path.join(output_dir, "command.txt")
    with open(path, "w") as f:
        f.write(command + "\n")
    print("Wrote rollout command to {}".format(path))


def resolve_log_dir(args):
    output_dir = command_output_dir(args)
    if output_dir is not None:
        return os.path.join(output_dir, "worker_logs")
    return os.path.join(tempfile.gettempdir(), "guided_parallel_worker_logs")


def resolve_video_temp_dir(args):
    output_dir = command_output_dir(args)
    if output_dir is not None:
        return os.path.join(output_dir, "worker_videos")
    return os.path.join(tempfile.gettempdir(), "guided_parallel_worker_videos")


def load_guided_policy_and_env(args, ckpt_path, device):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=args.verbose_load,
    )
    wrap_as_guided(policy)

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        rollout_horizon = config.experiment.rollout.horizon

    if args.guidance_geometry_source == "pointcloud":
        if args.pc_depth_obs_key not in ObsUtils.OBS_KEYS_TO_MODALITIES:
            ObsUtils.OBS_KEYS_TO_MODALITIES[args.pc_depth_obs_key] = "depth"

    render_offscreen = args.video_path is not None or args.guidance_geometry_source == "pointcloud"
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=render_offscreen,
        verbose=args.verbose_load,
    )
    return policy, env, rollout_horizon


def worker_fn(rank, args, ckpt_path, rollout_start, rollout_end, result_queue, log_dir, video_temp_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "worker_{}.log".format(rank))
    sys.stdout = open(log_path, "w", buffering=1)
    sys.stderr = sys.stdout

    base_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    worker_seed = base_seed + rank
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, env, rollout_horizon = load_guided_policy_and_env(args=args, ckpt_path=ckpt_path, device=device)

    args = deepcopy(args)
    args.forward_model = None
    if args.trajectory_backend == "forward_model":
        if args.forward_model_path is None:
            raise ValueError("--trajectory_backend forward_model requires --forward_model_path")
        args.forward_model = OSCForwardModelUtils.load_osc_forward_model(args.forward_model_path, device=device)
        print(
            "Loaded OSC forward model from {} (horizon={})".format(
                args.forward_model_path,
                args.forward_model.horizon,
            ),
            flush=True,
        )

    video_writer = None
    temp_video_path = None
    if args.video_path is not None:
        os.makedirs(video_temp_dir, exist_ok=True)
        temp_video_path = os.path.join(video_temp_dir, "worker_{}.mp4".format(rank))
        video_writer = imageio.get_writer(temp_video_path, fps=20)

    try:
        for i in range(rollout_start, rollout_end):
            progress_prefix = "Rollout {}/{} [worker {}]".format(i + 1, args.n_rollouts, rank)
            print("{} start".format(progress_prefix), flush=True)
            stats = rollout(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                args=args,
                video_writer=video_writer,
            )
            result_queue.put((i, stats))
            print(
                "{} done: horizon={}, success={}, return={:.4f}".format(
                    progress_prefix,
                    stats["Horizon"],
                    int(stats["Success_Rate"]),
                    stats["Return"],
                ),
                flush=True,
            )
    finally:
        if video_writer is not None:
            video_writer.close()
        try:
            env.close()
        except AttributeError:
            pass


def aggregate_rollout_stats(per_rollout_stats, args, stats_path_override=None):
    scalar_keys = [k for k, v in per_rollout_stats[0].items() if is_scalar_number(v)]
    rollout_stats_by_key = {
        k: [float(stats[k]) for stats in per_rollout_stats]
        for k in scalar_keys
    }
    avg_rollout_stats = {k: float(np.mean(vals)) for k, vals in rollout_stats_by_key.items()}
    avg_rollout_stats["Num_Success"] = float(np.sum(rollout_stats_by_key["Success_Rate"]))
    if "Non_Target_Collision_Any" in rollout_stats_by_key:
        avg_rollout_stats["Num_Non_Target_Collision_Rollouts"] = float(
            np.sum(rollout_stats_by_key["Non_Target_Collision_Any"])
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
        payload = dict(
            average=avg_rollout_stats,
            totals=dict(
                Non_Target_Collision_Object_Counts=total_collision_counts,
                Obstacle_Guidance_Trigger_Count=(
                    float(np.sum(rollout_stats_by_key["Obstacle_Guidance_Trigger_Count"]))
                    if "Obstacle_Guidance_Trigger_Count" in rollout_stats_by_key
                    else 0.0
                ),
                Obstacle_Guidance_Positive_Cost_Count=(
                    float(np.sum(rollout_stats_by_key["Obstacle_Guidance_Positive_Cost_Count"]))
                    if "Obstacle_Guidance_Positive_Cost_Count" in rollout_stats_by_key
                    else 0.0
                ),
            ),
            per_rollout_average=dict(
                Non_Target_Collision_Object_Counts=avg_collision_counts,
            ),
            rollouts=[make_json_serializable(stats) for stats in per_rollout_stats],
            args=serializable_args(args),
        )
        with open(stats_path, "w") as f:
            json.dump(make_json_serializable(payload), f, indent=4)
        print("Wrote rollout stats to {}".format(stats_path))


def concat_videos(video_temp_dir, output_path):
    worker_videos = sorted(
        os.path.join(video_temp_dir, name)
        for name in os.listdir(video_temp_dir)
        if name.endswith(".mp4")
    ) if os.path.isdir(video_temp_dir) else []
    if len(worker_videos) == 0:
        print("Warning: no worker videos found, skipping concat")
        return

    concat_file = os.path.join(video_temp_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        for vpath in worker_videos:
            f.write("file '{}'\n".format(os.path.abspath(vpath)))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c",
        "copy",
        output_path,
    ]
    print("Concatenating worker videos ...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    print("Wrote concatenated video to {}".format(output_path))


def run_obstacle_guided_agent_parallel(args):
    if args.render:
        raise ValueError("Parallel guided rollout does not support on-screen render")
    if args.video_target_mask_grid and args.camera_names == ["agentview"]:
        args.camera_names = ["agentview", "robot0_eye_in_hand"]
    if args.trajectory_backend == "forward_model" and args.forward_model_path is None:
        raise ValueError("--trajectory_backend forward_model requires --forward_model_path")

    mp.set_start_method("spawn", force=True)
    write_command_file(args)

    log_dir = resolve_log_dir(args)
    video_temp_dir = resolve_video_temp_dir(args)

    num_rollouts = int(args.n_rollouts)
    n_workers = min(int(args.n_workers), num_rollouts)
    assignments = []
    start = 0
    for rank in range(n_workers):
        count = num_rollouts // n_workers + (1 if rank < num_rollouts % n_workers else 0)
        if count > 0:
            assignments.append((rank, start, start + count))
            start += count

    result_queue = mp.Queue()
    workers = []
    for rank, r_start, r_end in assignments:
        p = mp.Process(
            target=worker_fn,
            args=(rank, args, args.agent, r_start, r_end, result_queue, log_dir, video_temp_dir),
        )
        p.start()
        workers.append(p)

    rollout_stats = [None] * num_rollouts
    for i in range(num_rollouts):
        idx, stats = result_queue.get()
        rollout_stats[idx] = stats
        print("\rProgress: {}/{}".format(i + 1, num_rollouts), end="", flush=True)
    print()

    failed = []
    for p in workers:
        p.join()
        if p.exitcode != 0:
            failed.append(p.exitcode)
    if failed:
        raise RuntimeError("One or more workers failed with exit codes {}".format(failed))

    if args.video_path is not None:
        concat_videos(video_temp_dir=video_temp_dir, output_path=args.video_path)

    effective_stats_path = args.stats_path
    if effective_stats_path is None and args.video_path is not None:
        effective_stats_path = os.path.splitext(args.video_path)[0] + "_stats.json"
    aggregate_rollout_stats(rollout_stats, args, stats_path_override=effective_stats_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel obstacle-guided rollout evaluation")
    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27, help="number of rollouts")
    parser.add_argument("--n_workers", type=int, default=4, help="number of parallel worker processes")
    parser.add_argument("--horizon", type=int, default=None, help="optional rollout horizon override")
    parser.add_argument("--env", type=str, default=None, help="optional environment name override")
    parser.add_argument("--render", action="store_true", help="unsupported in parallel mode")
    parser.add_argument("--video_path", type=str, default=None, help="optional rollout video path")
    parser.add_argument("--video_skip", type=int, default=5, help="write every n-th frame to video")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["agentview"], help="video camera names")
    parser.add_argument("--video_target_mask_grid", action="store_true", help="render RGB / target-mask grid video")
    parser.add_argument("--seed", type=int, default=None, help="base rollout seed")
    parser.add_argument("--stats_path", type=str, default=None, help="optional JSON path for rollout stats")

    parser.add_argument(
        "--guidance_geometry_source",
        type=str,
        choices=["oracle_center", "pointcloud"],
        default="pointcloud",
        help="obstacle geometry source for guidance",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["none", "gradient", "ranking"],
        default="gradient",
        help="inference-time obstacle intervention: disabled, gradient guidance, or action-chunk ranking",
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
        "--ranking_num_candidates",
        type=int,
        default=8,
        help="number of independently sampled action chunks for --selection_mode ranking",
    )
    parser.add_argument(
        "--ranking_safe_cost_threshold",
        type=float,
        default=1e-8,
        help="cost threshold used to count ranking candidates as geometry-safe",
    )
    parser.add_argument(
        "--ranking_cost_tie_tolerance",
        type=float,
        default=1e-10,
        help="cost tolerance for ranking tie-breaks by maximum clearance",
    )
    parser.add_argument(
        "--ranking_only_if_first_unsafe",
        action="store_true",
        help="leave the first sampled chunk unchanged when its ranking cost is already safe",
    )
    parser.add_argument(
        "--guidance_position_only",
        action="store_true",
        help="apply guidance gradients only to xyz action dimensions; leave rotation and gripper unchanged",
    )
    parser.add_argument(
        "--trajectory_backend",
        type=str,
        choices=["cumsum", "forward_model"],
        default="cumsum",
        help="action-to-EEF trajectory backend used by guidance cost",
    )
    parser.add_argument("--forward_model_path", type=str, default=None, help="OSC forward model checkpoint path")
    parser.add_argument(
        "--forward_model_state_obs_keys",
        type=str,
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        help="low-dimensional obs keys concatenated as forward model state",
    )
    parser.add_argument(
        "--guidance_schedule",
        type=str,
        choices=["constant", "late"],
        default="late",
        help="guidance scale schedule over denoising steps",
    )
    parser.add_argument("--guidance_start_step_pct", type=float, default=0.7)
    parser.add_argument("--target_object_name", type=str, default="Can")
    parser.add_argument("--obstacle_names", type=str, nargs="*", default=None)
    parser.add_argument("--eef_pos_obs_key", type=str, default="robot0_eef_pos")
    parser.add_argument("--final_collision_refine", action="store_true")
    parser.add_argument("--collision_refine_steps", type=int, default=5)
    parser.add_argument("--collision_refine_scale", type=float, default=0.02)
    parser.add_argument("--final_collision_cost_threshold", type=float, default=1e-8)

    parser.add_argument("--pc_camera_name", type=str, default="agentview")
    parser.add_argument("--pc_depth_obs_key", type=str, default="agentview_depth")
    parser.add_argument("--pc_safe_distance", type=float, default=0.02)
    parser.add_argument("--pc_distance_mode", type=str, choices=["xy", "xyz"], default="xy")
    parser.add_argument("--pc_voxel_size", type=float, default=0.005)
    parser.add_argument("--pc_max_points", type=int, default=1024)
    parser.add_argument("--pc_workspace_crop", action="store_true", default=True)
    parser.add_argument("--pc_no_workspace_crop", action="store_false", dest="pc_workspace_crop")
    parser.add_argument("--pc_debug_visualization", action="store_true")
    parser.add_argument("--pc_debug_dir", type=str, default="outputs/pc1_debug")
    parser.add_argument("--pc_debug_interval", type=int, default=25)
    parser.add_argument("--no-pc-cache", action="store_true")
    parser.add_argument("--verbose_load", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_obstacle_guided_agent_parallel(parse_args())
