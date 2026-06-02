"""
Diagnostic script for analyzing critic action-dependence.

1. Samples K action chunks from base policy for each observation
2. Scores them with the critic and reports mean/std of scores
3. Action-shuffle diagnostic: shuffles actions across batch and checks
   if critic scores change (if not, critic may not be using action info)

Example usage:

    python diagnose_critic.py \
        --policy_ckpt /path/to/diffusion_policy.pth \
        --critic_ckpt /path/to/critic_best.pth \
        --dataset /path/to/rollouts.hdf5 \
        --num_candidates 8 --num_samples 32
"""

import argparse
import json
import numpy as np
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.scripts.train_action_value_critic import load_critic_checkpoint


def load_obs_from_dataset(dataset_path, obs_keys, num_samples, seed=None):
    """Load observations from a rollout dataset."""
    import h5py

    if seed is not None:
        np.random.seed(seed)

    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())

    obs_list = []
    for _ in range(num_samples):
        demo_key = np.random.choice(demos)
        ep_grp = f["data/{}".format(demo_key)]
        num_steps = ep_grp.attrs["num_samples"]
        t = np.random.randint(0, num_steps)

        obs = {}
        for k in obs_keys:
            obs[k] = ep_grp["obs/{}".format(k)][t]
        obs_list.append(obs)

    f.close()

    batched_obs = {}
    for k in obs_keys:
        batched_obs[k] = np.stack([o[k] for o in obs_list], axis=0)

    return batched_obs


def diagnose(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print("Using device: {}".format(device))

    print("\nLoading base policy...")
    policy, policy_ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.policy_ckpt, device=device, verbose=False
    )

    print("Loading critic...")
    critic, critic_ckpt = load_critic_checkpoint(args.critic_ckpt, device=device)
    critic.eval()

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=policy_ckpt_dict)
    shape_meta = FileUtils._select_checkpoint_metadata(
        policy_ckpt_dict["shape_metadata"], "shape_metadata"
    )
    obs_keys = list(shape_meta["all_obs_keys"])

    print("\nLoading observations from dataset...")
    batched_obs = load_obs_from_dataset(
        dataset_path=args.dataset,
        obs_keys=obs_keys,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    obs_tensor = {}
    for k in obs_keys:
        v = torch.from_numpy(batched_obs[k]).float().to(device)
        if ObsUtils.key_is_obs_modality(k, "rgb") or ObsUtils.key_is_obs_modality(k, "depth"):
            v = ObsUtils.process_obs(obs=v, obs_key=k)
        obs_tensor[k] = v

    algo = policy.policy
    algo.set_eval()
    To = algo.algo_config.horizon.observation_horizon

    def make_policy_obs(single_obs):
        policy_obs = {}
        for k, v in single_obs.items():
            if k not in algo.obs_shapes:
                continue
            if v.ndim == len(algo.obs_shapes[k]):
                v = v.unsqueeze(0).unsqueeze(0).expand(-1, To, *[-1] * v.ndim).contiguous()
            elif v.ndim == 1 + len(algo.obs_shapes[k]):
                v = v.unsqueeze(0)
            policy_obs[k] = v
        return policy_obs

    def make_critic_obs(single_obs, num_expand):
        critic_obs = {}
        for k in critic.obs_shapes:
            v = single_obs[k]
            if v.ndim == len(critic.obs_shapes[k]):
                v = v.unsqueeze(0).expand(To, *[-1] * v.ndim).contiguous()
            critic_obs[k] = v.unsqueeze(0).expand(num_expand, *[-1] * v.ndim).contiguous()
        return critic_obs

    print("\n" + "=" * 60)
    print("Action-Dependence Diagnostic")
    print("=" * 60)
    print("Number of observations: {}".format(args.num_samples))
    print("Number of candidates per observation: {}".format(args.num_candidates))

    with torch.no_grad():
        all_scores = []
        for i in range(args.num_samples):
            single_obs = {k: v[i] for k, v in obs_tensor.items()}
            policy_obs = make_policy_obs(single_obs)

            candidates = []
            for _ in range(args.num_candidates):
                action_seq = algo._get_action_trajectory(obs_dict=policy_obs)
                candidates.append(action_seq[0])

            candidates = torch.stack(candidates, dim=0)
            critic_obs = make_critic_obs(single_obs, args.num_candidates)

            scores = critic(critic_obs, candidates).squeeze(-1)
            all_scores.append(scores.cpu().numpy())

        all_scores = np.array(all_scores)

        print("\n--- Score Statistics ---")
        print("Mean score per observation: {:.4f} (std: {:.4f})".format(
            all_scores.mean(), all_scores.std()
        ))
        print("Mean within-obs std: {:.4f}".format(all_scores.std(axis=1).mean()))
        print("Min score: {:.4f}, Max score: {:.4f}".format(
            all_scores.min(), all_scores.max()
        ))

        print("\n--- Action Shuffle Diagnostic ---")
        print("Testing if critic scores change when actions are shuffled...")

        original_scores = []
        shuffled_scores = []

        batch_size = min(16, args.num_samples)
        for i in range(batch_size):
            single_obs = {k: v[i] for k, v in obs_tensor.items()}
            policy_obs = make_policy_obs(single_obs)

            candidates = []
            for _ in range(args.num_candidates):
                action_seq = algo._get_action_trajectory(obs_dict=policy_obs)
                candidates.append(action_seq[0])
            candidates = torch.stack(candidates, dim=0)

            critic_obs = make_critic_obs(single_obs, args.num_candidates)

            orig_scores = critic(critic_obs, candidates).squeeze(-1)
            original_scores.append(orig_scores.cpu().numpy())

            perm = torch.randperm(args.num_candidates)
            shuffled_candidates = candidates[perm]
            shuf_scores = critic(critic_obs, shuffled_candidates).squeeze(-1)
            shuffled_scores.append(shuf_scores.cpu().numpy())

        original_scores = np.array(original_scores)
        shuffled_scores = np.array(shuffled_scores)

        score_diff = np.abs(original_scores - shuffled_scores).mean()
        print("Mean absolute score difference after shuffle: {:.6f}".format(score_diff))

        if score_diff < 1e-5:
            print("\n⚠️  WARNING: Critic scores barely change after action shuffle!")
            print("   This suggests the critic may not be using action information.")
        else:
            print("\n✓ Critic scores change after action shuffle - action information is being used.")

    print("\n" + "=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_ckpt", type=str, required=True,
                        help="path to base policy checkpoint")
    parser.add_argument("--critic_ckpt", type=str, required=True,
                        help="path to critic checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to rollout HDF5 for sampling observations")

    parser.add_argument("--num_candidates", type=int, default=8,
                        help="number of candidate action chunks (K)")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="number of observations to analyze")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
