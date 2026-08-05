import unittest

import numpy as np
from robosuite.utils import transform_utils as T

from robomimic.envs.env_robosuite import (
    eef_pose_observation_from_raw,
    normalize_eef_pose_observation_config,
)


class TestEEFPoseObservation(unittest.TestCase):
    def test_default_config_is_opt_in(self):
        self.assertFalse(normalize_eef_pose_observation_config(None)["enabled"])
        config = normalize_eef_pose_observation_config(True)
        self.assertTrue(config["enabled"])
        self.assertEqual(config["obs_key"], "agent_pos")
        self.assertEqual(config["robot_prefix"], "robot0")

    def test_world_position_and_site_orientation_become_six_d_state(self):
        axis_angle = np.array([0.1, -0.2, 0.3])
        quaternion = T.axisangle2quat(axis_angle)
        obs = {
            "robot0_eef_pos": np.array([0.4, -0.1, 0.9]),
            "robot0_eef_quat_site": quaternion,
        }
        actual = eef_pose_observation_from_raw(
            obs,
            normalize_eef_pose_observation_config(True),
        )
        self.assertEqual(actual.shape, (6,))
        np.testing.assert_allclose(actual[:3], obs["robot0_eef_pos"])
        np.testing.assert_allclose(actual[3:], axis_angle, atol=1e-7)

    def test_missing_site_quaternion_fails_loudly(self):
        with self.assertRaises(KeyError):
            eef_pose_observation_from_raw(
                {"robot0_eef_pos": np.zeros(3)},
                normalize_eef_pose_observation_config(True),
            )


if __name__ == "__main__":
    unittest.main()
