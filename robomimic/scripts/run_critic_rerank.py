"""
Rollout script with critic-based action chunk reranking.

Loads a base Diffusion Policy and an Action Value Critic, then at each
control step samples K candidate action chunks from the base policy,
scores them with the critic, and executes the highest-scoring chunk.

Example usage:

    python run_critic_rerank.py \
        --policy_ckpt /path/to/diffusion_policy.pth \
        --critic_ckpt /path/to/critic_best.pth \
        --n_rollouts 50 --horizon 400 \
        --num_candidates 8 \
        --dataset_path /path/to/output.hdf5 --dataset_obs
"""

import argparse
import json
import h5py
import imageio
import numpy as np
from collections import OrderedDict, deque
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.train_action_value_critic import load_critic_checkpoint


def rollout_with_critic(
    policy,
    critic,
    env,
    horizon,
    num_candidates=8,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
):
    """
    Run a rollout with critic-based action chunk reranking.

    Args:
        policy (RolloutPolicy): base policy
        critic (ActionValueCritic): critic model for scoring action chunks
        env (EnvBase): environment
        horizon (int): maximum rollout horizon
        num_candidates (int): number of candidate action chunks to sample (K)
        render (bool): whether to render on-screen
        video_writer: optional video writer
        video_skip (int): render every N frames
        return_obs (bool): whether to return observations
        camera_names (list): camera names for rendering

    Returns:
        stats (dict): rollout statistics
        traj (dict): rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    algo = policy.policy
    To = algo.algo_config.horizon.observation_horizon
    Ta = algo.algo_config.horizon.action_horizon

    try:
        for step_i in range(horizon):
            if len(algo.action_queue) == 0:
                obs_dict = policy._prepare_observation(obs, batched_ob=False)

                with torch.no_grad():
                    candidates = []
                    for _ in range(num_candidates):
                        action_seq = algo._get_action_trajectory(obs_dict=obs_dict)
                        candidates.append(action_seq[0])

                    candidates = torch.stack(candidates, dim=0)

                    critic_obs = {}
                    for k in critic.obs_shapes:
                        v = obs_dict[k]
                        if v.shape[0] == 1:
                            v = v.squeeze(0)
                        critic_obs[k] = v.unsqueeze(0).expand(num_candidates, *[-1] * v.ndim).contiguous()

                    scores = critic(critic_obs, candidates).squeeze(-1)

                    best_idx = torch.argmax(scores).item()
                    best_action_seq = candidates[best_idx]

                algo.action_queue.extend(best_action_seq)

            action = policy(ob=obs)

            next_obs, r, done, _ = env.step(action)
            total_reward += r
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            traj["actions"].append(action)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_critic_rerank(args):
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("Using device: {}".format(device))

    print("\nLoading base policy...")
    policy, policy_ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.policy_ckpt, device=device, verbose=True
    )

    print("\nLoading critic...")
    critic, critic_ckpt = load_critic_checkpoint(args.critic_ckpt, device=device)
    critic.eval()
    print("Loaded critic from epoch {}".format(critic_ckpt.get("epoch", "unknown")))

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=policy_ckpt_dict)
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        rollout_horizon = config.experiment.rollout.horizon

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=policy_ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=(args.video_path is not None),
        verbose=True,
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in range(args.n_rollouts):
        print("\nRollout {}/{}".format(i + 1, args.n_rollouts))
        stats, traj = rollout_with_critic(
            policy=policy,
            critic=critic,
            env=env,
            horizon=rollout_horizon,
            num_candidates=args.num_candidates,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)
        print("  Return: {:.2f}, Horizon: {}, Success: {}".format(
            stats["Return"], stats["Horizon"], stats["Success_Rate"]
        ))

        if write_dataset:
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

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Success_Rate"] = np.mean(rollout_stats["Success_Rate"])

    print("\n" + "=" * 50)
    print("Critic Reranking Rollout Results (K={})".format(args.num_candidates))
    print("=" * 50)
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print("\nWrote dataset trajectories to {}".format(args.dataset_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="path to JSON config file")
    parser.add_argument("--policy_ckpt", type=str, default=None,
                        help="path to base policy checkpoint")
    parser.add_argument("--critic_ckpt", type=str, default=None,
                        help="path to critic checkpoint")

    parser.add_argument("--n_rollouts", type=int, default=50,
                        help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=None,
                        help="override rollout horizon")
    parser.add_argument("--num_candidates", type=int, default=8,
                        help="number of candidate action chunks to sample (K)")

    parser.add_argument("--env", type=str, default=None,
                        help="override environment name")
    parser.add_argument("--render", action="store_true",
                        help="on-screen rendering")
    parser.add_argument("--video_path", type=str, default=None,
                        help="render to video file")
    parser.add_argument("--video_skip", type=int, default=5,
                        help="render every N frames")
    parser.add_argument("--camera_names", type=str, nargs="+",
                        default=["agentview"],
                        help="camera names for rendering")

    parser.add_argument("--dataset_path", type=str, default=None,
                        help="save rollouts to HDF5")
    parser.add_argument("--dataset_obs", action="store_true",
                        help="include observations in HDF5")

    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        rollout_config = config.get("rollout", {})

        if args.policy_ckpt is None and "policy_ckpt" in rollout_config:
            args.policy_ckpt = rollout_config["policy_ckpt"]
        if args.critic_ckpt is None and "critic_ckpt" in rollout_config:
            args.critic_ckpt = rollout_config["critic_ckpt"]

        if args.n_rollouts == 50 and "n_rollouts" in rollout_config:
            args.n_rollouts = rollout_config["n_rollouts"]
        if args.horizon is None and "horizon" in rollout_config:
            args.horizon = rollout_config["horizon"]
        if args.num_candidates == 8 and "num_candidates" in rollout_config:
            args.num_candidates = rollout_config["num_candidates"]
        if args.dataset_path is None and "dataset_path" in rollout_config:
            args.dataset_path = rollout_config["dataset_path"]
        if args.seed is None and "seed" in rollout_config:
            args.seed = rollout_config["seed"]

        if "dataset_obs" in rollout_config:
            args.dataset_obs = rollout_config["dataset_obs"]

    if args.policy_ckpt is None:
        parser.error("--policy_ckpt or config.rollout.policy_ckpt is required")
    if args.critic_ckpt is None:
        parser.error("--critic_ckpt or config.rollout.critic_ckpt is required")

    run_critic_rerank(args)


if __name__ == "__main__":
    main()
