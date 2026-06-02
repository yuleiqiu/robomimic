"""
Dataset for training the Action Value Critic.

Loads rollout trajectories and produces (obs_t, action_chunk_t, success) samples.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import robomimic.utils.obs_utils as ObsUtils


class ActionValueCriticDataset(Dataset):
    """
    Dataset that loads rollout trajectories and produces training samples
    for the action value critic.

    Each sample is:
        obs_dict: observation at timestep t
        action_chunk: actions[t : t + H], padded with last action if needed
        success: final episode success label (0 or 1)
    """

    def __init__(
        self,
        dataset_paths,
        obs_keys,
        action_horizon,
        observation_horizon=2,
        filter_key=None,
    ):
        """
        Args:
            dataset_paths (list): list of paths to rollout HDF5 files
            obs_keys (list): observation keys to load
            action_horizon (int): length of action chunk H
            observation_horizon (int): number of observation frames To (default 2)
            filter_key (str): optional mask key to filter demos
        """
        super(ActionValueCriticDataset, self).__init__()

        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        self.dataset_paths = dataset_paths
        self.obs_keys = list(obs_keys)
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.filter_key = filter_key

        self.all_data = []
        self.index_map = []

        self.num_success = 0
        self.num_failure = 0

        for file_idx, path in enumerate(self.dataset_paths):
            data = self._load_file(path, file_idx)
            self.all_data.append(data)

        self._build_index_map()
        self._print_stats()

    def _load_file(self, path, file_idx):
        """Load a single HDF5 file into memory."""
        f = h5py.File(path, "r")

        if self.filter_key is not None and "mask/{}".format(self.filter_key) in f:
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(self.filter_key)][:])]
        else:
            demos = list(f["data"].keys())

        demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))

        file_data = {
            "path": path,
            "demos": [],
        }

        for demo_key in demos:
            ep_grp = f["data/{}".format(demo_key)]

            if "success" not in ep_grp.attrs:
                raise ValueError(
                    "Demo {} in {} is missing 'success' attr. "
                    "Run annotate_dataset_success.py first.".format(demo_key, path)
                )

            success = int(ep_grp.attrs["success"])
            num_samples = int(ep_grp.attrs["num_samples"])

            actions = ep_grp["actions"][()]

            obs = {}
            for k in self.obs_keys:
                obs[k] = ep_grp["obs/{}".format(k)][()]

            file_data["demos"].append({
                "demo_key": demo_key,
                "success": success,
                "num_samples": num_samples,
                "actions": actions,
                "obs": obs,
            })

            if success == 1:
                self.num_success += 1
            else:
                self.num_failure += 1

        f.close()
        return file_data

    def _build_index_map(self):
        """Build mapping from global index to (file_idx, demo_idx, timestep)."""
        self.index_map = []
        for file_idx, file_data in enumerate(self.all_data):
            for demo_idx, demo_data in enumerate(file_data["demos"]):
                num_samples = demo_data["num_samples"]
                for t in range(num_samples):
                    self.index_map.append((file_idx, demo_idx, t))

    def _print_stats(self):
        """Print dataset statistics."""
        total = self.num_success + self.num_failure
        print("=" * 50)
        print("ActionValueCriticDataset")
        print("=" * 50)
        print("Files: {}".format(self.dataset_paths))
        print("Total samples: {}".format(len(self.index_map)))
        print("Success demos: {}".format(self.num_success))
        print("Failure demos: {}".format(self.num_failure))
        if total > 0:
            print("Success ratio: {:.2%}".format(self.num_success / total))
        print("Action horizon: {}".format(self.action_horizon))
        print("Obs keys: {}".format(self.obs_keys))
        print("=" * 50)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, demo_idx, t = self.index_map[idx]

        demo_data = self.all_data[file_idx]["demos"][demo_idx]

        To = self.observation_horizon
        obs_dict = {}
        for k in self.obs_keys:
            frames = []
            for i in range(To):
                src_t = max(0, t - To + 1 + i)
                obs_val = demo_data["obs"][k][src_t]
                if ObsUtils.key_is_obs_modality(k, "rgb") or ObsUtils.key_is_obs_modality(k, "depth"):
                    obs_val = ObsUtils.process_obs(obs=obs_val, obs_key=k)
                frames.append(obs_val)
            obs_stack = np.stack(frames, axis=0)
            obs_dict[k] = torch.from_numpy(obs_stack).float()

        actions = demo_data["actions"]
        num_samples = demo_data["num_samples"]
        action_dim = actions.shape[1]

        action_chunk = np.zeros((self.action_horizon, action_dim), dtype=np.float32)
        for h in range(self.action_horizon):
            src_t = min(t + h, num_samples - 1)
            action_chunk[h] = actions[src_t]

        action_chunk = torch.from_numpy(action_chunk)

        success = torch.tensor(demo_data["success"], dtype=torch.float32)

        return {
            "obs": obs_dict,
            "action_chunk": action_chunk,
            "success": success,
        }
