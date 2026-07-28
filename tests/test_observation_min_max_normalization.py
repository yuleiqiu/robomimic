import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from robomimic.algo.algo import RolloutPolicy
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils


class TestObservationMinMaxNormalization(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "synthetic.hdf5"
        with h5py.File(self.path, "w") as output:
            data = output.create_group("data")
            for demo_name, shift in (("demo_1", 0.0), ("demo_2", 100.0)):
                demo = data.create_group(demo_name)
                demo.attrs["num_samples"] = 2
                obs = demo.create_group("obs")
                pointcloud = np.asarray(
                    [
                        [[0, 10, -2], [2, 14, 0]],
                        [[4, 18, 2], [6, 22, 4]],
                    ],
                    dtype=np.float32,
                )
                obs.create_dataset("task_pointcloud", data=pointcloud + shift)
                obs.create_dataset(
                    "robot_state",
                    data=np.asarray([[0, 10], [2, 14]], dtype=np.float32) + shift,
                )
                demo.create_dataset("actions", data=np.zeros((2, 1), dtype=np.float32))
            mask = output.create_group("mask")
            mask.create_dataset("train", data=np.asarray(["demo_1"], dtype="S"))
            mask.create_dataset("valid", data=np.asarray(["demo_2"], dtype="S"))

        ObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "low_dim": ["robot_state"],
                    "rgb": [],
                    "depth": [],
                    "scan": ["task_pointcloud"],
                }
            }
        )
        self.dataset = SequenceDataset(
            hdf5_path=str(self.path),
            obs_keys=["task_pointcloud", "robot_state"],
            action_keys=["actions"],
            dataset_keys=["actions"],
            action_config={"actions": {"normalization": None}},
            observation_config={
                "task_pointcloud": {
                    "normalization": "min_max",
                    "last_n_dims": 1,
                },
                "robot_state": {
                    "normalization": "min_max",
                    "last_n_dims": 1,
                },
            },
            hdf5_cache_mode=None,
            hdf5_normalize_obs=True,
            filter_by_attribute="train",
        )

    def tearDown(self):
        self.dataset.close_and_delete_hdf5_handle()
        self.tempdir.cleanup()

    def test_pointcloud_reduces_time_and_points_but_retains_xyz(self):
        stats = self.dataset.get_obs_normalization_stats()
        self.assertEqual(stats["task_pointcloud"]["offset"].shape, (1, 3, 1))
        self.assertEqual(stats["task_pointcloud"]["scale"].shape, (1, 3, 1))

        raw = np.asarray(
            [
                [[0, 10, -2], [2, 14, 0]],
                [[4, 18, 2], [6, 22, 4]],
            ],
            dtype=np.float32,
        )
        processed = ObsUtils.process_obs(raw, obs_key="task_pointcloud")
        normalized = ObsUtils.normalize_dict(
            {"task_pointcloud": processed.copy()},
            {"task_pointcloud": stats["task_pointcloud"]},
        )["task_pointcloud"]
        np.testing.assert_allclose(normalized.min(axis=(0, 2)), -0.999999, atol=1e-6)
        np.testing.assert_allclose(normalized.max(axis=(0, 2)), 0.999999, atol=1e-6)

    def test_validation_and_rollout_reuse_training_stats(self):
        stats = self.dataset.get_obs_normalization_stats()
        valid_raw = {
            "task_pointcloud": np.full((512, 3), 100.0, dtype=np.float32),
            "robot_state": np.asarray([100.0, 100.0], dtype=np.float32),
        }
        config = SimpleNamespace(
            all_obs_keys=["task_pointcloud", "robot_state"]
        )
        dummy_policy = SimpleNamespace(
            device=torch.device("cpu"),
            global_config=config,
        )
        rollout = RolloutPolicy(
            dummy_policy,
            obs_normalization_stats=stats,
        )
        prepared = rollout._prepare_observation(valid_raw)
        self.assertEqual(tuple(prepared["task_pointcloud"].shape), (1, 3, 512))
        self.assertGreater(float(prepared["task_pointcloud"].min()), 1.0)
        self.assertGreater(float(prepared["robot_state"].min()), 1.0)


if __name__ == "__main__":
    unittest.main()
