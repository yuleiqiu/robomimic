import unittest

import numpy as np
import torch

from robomimic.algo.algo import RolloutPolicy
import robomimic.utils.obs_utils as ObsUtils


class _Policy:
    device = torch.device("cpu")


class TestRolloutScanProcessing(unittest.TestCase):
    def test_scan_matches_dataset_processing(self):
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "low_dim": ["state"],
                    "rgb": [],
                    "depth": [],
                    "scan": ["task_pointcloud"],
                }
            }
        )
        wrapper = RolloutPolicy(_Policy())
        points = np.arange(15, dtype=np.float32).reshape(5, 3)
        prepared = wrapper._prepare_observation(
            {"state": np.ones(2), "task_pointcloud": points}
        )
        self.assertEqual(tuple(prepared["task_pointcloud"].shape), (1, 3, 5))
        np.testing.assert_array_equal(
            prepared["task_pointcloud"].numpy()[0],
            points.T,
        )


if __name__ == "__main__":
    unittest.main()
