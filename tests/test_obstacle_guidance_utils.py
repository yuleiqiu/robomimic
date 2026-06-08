"""
Minimal checks for obstacle guidance geometry utilities.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils


def test_obstacle_xy_cost():
    action_chunk = torch.zeros((1, 8, 7), dtype=torch.float32, requires_grad=True)
    current_eef_pos = torch.zeros((1, 3), dtype=torch.float32)

    cost = ObstacleGuidanceUtils.obstacle_xy_cost(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        obstacle_centers_xy=torch.zeros((0, 2), dtype=torch.float32),
        obstacle_radii=torch.zeros((0,), dtype=torch.float32),
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert torch.isclose(cost, torch.tensor(0.0))

    cost = ObstacleGuidanceUtils.obstacle_xy_cost(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        obstacle_centers_xy=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        obstacle_radii=torch.tensor([0.06], dtype=torch.float32),
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert torch.isclose(cost, torch.tensor(0.0))

    cost = ObstacleGuidanceUtils.obstacle_xy_cost(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        obstacle_centers_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        obstacle_radii=torch.tensor([0.06], dtype=torch.float32),
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert cost.item() > 0.0

    cost.backward()
    assert action_chunk.grad is not None


if __name__ == "__main__":
    test_obstacle_xy_cost()
    print("obstacle guidance utility checks passed")
