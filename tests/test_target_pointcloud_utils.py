import unittest

import numpy as np

from robomimic.utils.target_pointcloud_utils import (
    deterministic_farthest_point_sample,
    normalize_target_pointcloud_config,
    unproject_depth_pixels_to_world,
)


class TestTargetPointCloudUtils(unittest.TestCase):
    def test_unprojection_identity_camera(self):
        depth = np.full((3, 4), 2.0)
        pixels = np.array([[1, 2], [0, 0]])
        intrinsic = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        expected = np.array([[0.0, 0.0, 2.0], [-2.0, -1.0, 2.0]], dtype=np.float32)
        actual = unproject_depth_pixels_to_world(depth, pixels, intrinsic, np.eye(4))
        np.testing.assert_allclose(actual, expected)

    def test_unprojection_world_translation(self):
        depth = np.ones((1, 1))
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [1.0, 2.0, 3.0]
        actual = unproject_depth_pixels_to_world(
            depth, np.array([[0, 0]]), np.eye(3), camera_pose
        )
        np.testing.assert_allclose(actual, [[1.0, 2.0, 4.0]])

    def test_fps_is_deterministic_and_selects_unique_points(self):
        points = np.stack(
            np.meshgrid(np.arange(6), np.arange(5), np.arange(2)), axis=-1
        ).reshape(-1, 3)
        first = deterministic_farthest_point_sample(points, 16)
        second = deterministic_farthest_point_sample(points, 16)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first, axis=0)), 16)

    def test_repeat_padding_cycles_valid_points(self):
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        sampled = deterministic_farthest_point_sample(points, 5)
        np.testing.assert_array_equal(sampled, points[[0, 1, 0, 1, 0]])

    def test_zero_padding_matches_official_lan_preprocessing(self):
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        sampled = deterministic_farthest_point_sample(
            points,
            5,
            padding_mode="zero",
        )
        self.assertEqual(sampled.shape, (5, 3))
        self.assertEqual(np.count_nonzero(np.all(sampled == 0, axis=1)), 3)
        for point in points:
            self.assertTrue(np.any(np.all(sampled == point, axis=1)))

    def test_config_validation(self):
        config = normalize_target_pointcloud_config({"num_points": 12})
        self.assertEqual(config["num_points"], 12)
        self.assertEqual(config["obs_key"], "task_pointcloud")
        self.assertEqual(config["padding_mode"], "repeat")
        with self.assertRaises(ValueError):
            normalize_target_pointcloud_config({"height": 0})
        with self.assertRaises(ValueError):
            normalize_target_pointcloud_config({"padding_mode": "invalid"})


if __name__ == "__main__":
    unittest.main()
