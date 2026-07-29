import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data._utils.collate import default_collate

from robomimic.utils.dataset import (
    MetaDataset,
    PairedCorrectionDataset,
    SequenceDataset,
)
from robomimic.utils.train_utils import (
    assert_action_normalization_stats_match,
    set_reference_action_normalization,
)
import robomimic.utils.obs_utils as ObsUtils


class TestPairedCorrectionDataset(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.clean_path = root / "clean.hdf5"
        self.pair_path = root / "pairs.hdf5"
        self.action_config = {
            "delta_eef_pose_action": {"normalization": "min_max"}
        }
        self._write_clean_file()
        self._write_pair_file()
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "low_dim": ["robot_state"],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                }
            }
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_clean_file(self):
        with h5py.File(self.clean_path, "w") as output:
            data = output.create_group("data")
            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = 20
            obs = demo.create_group("obs")
            obs.create_dataset(
                "robot_state",
                data=np.arange(60, dtype=np.float32).reshape(20, 3),
            )
            action = np.linspace(-2.0, 2.0, 140, dtype=np.float32).reshape(
                20, 7
            )
            demo.create_dataset("delta_eef_pose_action", data=action)
            demo.create_dataset("rewards", data=np.zeros(20, dtype=np.float32))
            demo.create_dataset("dones", data=np.zeros(20, dtype=np.float32))
            mask = output.create_group("mask")
            mask.create_dataset("train", data=np.asarray(["demo_0"], dtype="S"))
            mask.create_dataset("valid", data=np.asarray(["demo_0"], dtype="S"))

    def _write_pair_file(self):
        with h5py.File(self.pair_path, "w") as output:
            output.attrs["schema_version"] = "paired-correction-pairs-v1"
            output.attrs["observation_horizon"] = 2
            output.attrs["prediction_horizon"] = 16
            output.attrs["action_horizon"] = 8
            pairs = output.create_group("pairs")
            obs = pairs.create_group("obs")
            obs.create_dataset(
                "robot_state",
                data=np.arange(18, dtype=np.float32).reshape(3, 2, 3),
            )
            positive = np.linspace(
                -1.0, 1.0, 3 * 16 * 7, dtype=np.float32
            ).reshape(3, 16, 7)
            pairs.create_dataset("positive_actions", data=positive)
            pairs.create_dataset("negative_actions", data=-positive)
            pairs.create_dataset(
                "episode_id",
                data=np.asarray(["demo_0", "demo_0", "demo_1"], dtype="S"),
            )
            mask = output.create_group("mask")
            mask.create_dataset(
                "train_episode_ids",
                data=np.asarray(["demo_0"], dtype="S"),
            )
            mask.create_dataset(
                "valid_episode_ids",
                data=np.asarray(["demo_1"], dtype="S"),
            )

    def _pair_dataset(
        self,
        split="train",
        cache_in_memory=False,
        return_negative_actions=False,
    ):
        return PairedCorrectionDataset(
            hdf5_path=str(self.pair_path),
            obs_keys=["robot_state"],
            action_keys=["delta_eef_pose_action"],
            dataset_keys=["delta_eef_pose_action", "rewards", "dones"],
            action_config=self.action_config,
            frame_stack=2,
            seq_length=16,
            filter_by_attribute=split,
            cache_in_memory=cache_in_memory,
            return_negative_actions=return_negative_actions,
        )

    def test_split_and_pre_truncation_padding(self):
        dataset = self._pair_dataset()
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(sample["obs"]["robot_state"].shape, (17, 3))
        self.assertEqual(sample["actions"].shape, (17, 7))
        self.assertEqual(sample["rewards"].shape, (17,))
        np.testing.assert_array_equal(
            sample["obs"]["robot_state"][:2],
            np.arange(6, dtype=np.float32).reshape(2, 3),
        )
        np.testing.assert_array_equal(
            sample["obs"]["robot_state"][2:],
            np.repeat(sample["obs"]["robot_state"][1:2], 15, axis=0),
        )

        # These are the exact tensors consumed after Diffusion Policy truncates
        # the standard SequenceDataset pre-collation representation.
        consumed_obs = sample["obs"]["robot_state"][:2]
        consumed_actions = sample["delta_eef_pose_action"][:16]
        with h5py.File(self.pair_path, "r") as pair_file:
            np.testing.assert_array_equal(
                consumed_obs, pair_file["pairs/obs/robot_state"][0]
            )
            np.testing.assert_array_equal(
                consumed_actions, pair_file["pairs/positive_actions"][0]
            )

    def test_clean_and_pair_samples_collate(self):
        clean = SequenceDataset(
            hdf5_path=str(self.clean_path),
            obs_keys=["robot_state"],
            action_keys=["delta_eef_pose_action"],
            dataset_keys=["delta_eef_pose_action", "rewards", "dones"],
            action_config=self.action_config,
            frame_stack=2,
            seq_length=16,
            hdf5_cache_mode=None,
            load_next_obs=False,
            filter_by_attribute="train",
        )
        pair = self._pair_dataset()
        mixed = MetaDataset(
            datasets=[clean, pair],
            ds_weights=[0.5, 0.5],
            normalize_weights_by_ds_size=True,
        )
        batch = default_collate([mixed[0], mixed[len(clean)]])
        self.assertEqual(tuple(batch["actions"].shape), (2, 17, 7))
        self.assertEqual(
            tuple(batch["obs"]["robot_state"].shape), (2, 17, 3)
        )
        self.assertLessEqual(float(batch["actions"].abs().max()), 1.0)

    def test_raw_memory_cache_matches_lazy_reads(self):
        lazy = self._pair_dataset()
        cached = self._pair_dataset(cache_in_memory=True)
        lazy.set_action_normalization_stats(
            cached.get_action_normalization_stats()
        )
        for key, lazy_value in lazy[1].items():
            cached_value = cached[1][key]
            if isinstance(lazy_value, dict):
                for obs_key in lazy_value:
                    np.testing.assert_array_equal(
                        lazy_value[obs_key], cached_value[obs_key]
                    )
            else:
                np.testing.assert_array_equal(lazy_value, cached_value)

    def test_action_normalization_reference_assertion(self):
        dataset = self._pair_dataset()
        stats = dataset.get_action_normalization_stats()
        self.assertEqual(
            assert_action_normalization_stats_match(stats, stats), 0.0
        )
        changed = {
            key: {
                stat: value.copy()
                for stat, value in action_stats.items()
            }
            for key, action_stats in stats.items()
        }
        changed["delta_eef_pose_action"]["offset"][0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assert_action_normalization_stats_match(stats, changed)

    def test_reference_stats_override_mixed_refit(self):
        clean = SequenceDataset(
            hdf5_path=str(self.clean_path),
            obs_keys=["robot_state"],
            action_keys=["delta_eef_pose_action"],
            dataset_keys=["delta_eef_pose_action", "rewards", "dones"],
            action_config=self.action_config,
            frame_stack=2,
            seq_length=16,
            hdf5_cache_mode=None,
            load_next_obs=False,
            filter_by_attribute="train",
        )
        reference = clean.get_action_normalization_stats()
        pair = self._pair_dataset()
        mixed = MetaDataset(
            datasets=[clean, pair],
            ds_weights=[0.5, 0.5],
            normalize_weights_by_ds_size=True,
        )
        summary = set_reference_action_normalization(
            train_dataset=mixed,
            valid_dataset=None,
            reference_stats=reference,
        )
        self.assertEqual(summary["sequence_maximum_difference"], 0.0)
        self.assertLessEqual(
            summary["correction_maximum_absolute_normalized"], 1.0
        )
        self.assertEqual(
            assert_action_normalization_stats_match(
                mixed.get_action_normalization_stats(), reference
            ),
            0.0,
        )

    def test_set_supervision_collates_negative_actions_and_mask(self):
        clean = SequenceDataset(
            hdf5_path=str(self.clean_path),
            obs_keys=["robot_state"],
            action_keys=["delta_eef_pose_action"],
            dataset_keys=["delta_eef_pose_action", "rewards", "dones"],
            action_config=self.action_config,
            frame_stack=2,
            seq_length=16,
            hdf5_cache_mode=None,
            load_next_obs=False,
            filter_by_attribute="train",
        )
        pair = self._pair_dataset(return_negative_actions=True)
        mixed = MetaDataset(
            datasets=[clean, pair],
            ds_weights=[0.5, 0.5],
            normalize_weights_by_ds_size=True,
        )
        batch = default_collate([mixed[0], mixed[len(clean)]])
        self.assertEqual(
            tuple(batch["negative_actions"].shape), (2, 17, 7)
        )
        np.testing.assert_array_equal(
            batch["negative_actions"][0].numpy(),
            batch["actions"][0].numpy(),
        )
        np.testing.assert_array_equal(
            batch["is_paired_correction"].numpy(),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )
        with h5py.File(self.pair_path, "r") as pair_file:
            raw_negative = pair_file["pairs/negative_actions"][0]
        stats = pair.get_action_normalization_stats()
        scale = stats["delta_eef_pose_action"]["scale"]
        offset = stats["delta_eef_pose_action"]["offset"]
        expected = (raw_negative - offset) / scale
        np.testing.assert_allclose(
            batch["negative_actions"][1, :16].numpy(),
            expected,
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
