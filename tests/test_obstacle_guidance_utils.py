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


def test_obstacle_xyz_cylinder_cost():
    obstacle_centers_xyz = torch.tensor([[0.0, 0.0, 0.05]], dtype=torch.float32)
    obstacle_radii = torch.tensor([0.06], dtype=torch.float32)
    obstacle_top_z = torch.tensor([0.10], dtype=torch.float32)
    z_clearance = 0.03

    action_chunk = torch.zeros((1, 8, 7), dtype=torch.float32, requires_grad=True)
    cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
        action_chunk=action_chunk,
        current_eef_pos=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        obstacle_centers_xyz=obstacle_centers_xyz,
        obstacle_radii=obstacle_radii,
        obstacle_top_z=obstacle_top_z,
        z_clearance=z_clearance,
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert torch.isclose(cost, torch.tensor(0.0))

    cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
        action_chunk=action_chunk,
        current_eef_pos=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32),
        obstacle_centers_xyz=obstacle_centers_xyz,
        obstacle_radii=obstacle_radii,
        obstacle_top_z=obstacle_top_z,
        z_clearance=z_clearance,
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert torch.isclose(cost, torch.tensor(0.0))

    cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
        action_chunk=action_chunk,
        current_eef_pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        obstacle_centers_xyz=obstacle_centers_xyz,
        obstacle_radii=obstacle_radii,
        obstacle_top_z=obstacle_top_z,
        z_clearance=z_clearance,
        horizon=8,
        delta_pos_scale=0.05,
    )
    assert cost.item() > 0.0

    cost.backward()
    assert action_chunk.grad is not None
    assert torch.all(torch.isfinite(action_chunk.grad))


def test_controller_delta_pos_mapping():
    controller_config = {
        "body_parts": {
            "right": {
                "input_min": -1,
                "input_max": 1,
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            },
        },
    }
    scale, offset = ObstacleGuidanceUtils.controller_delta_pos_mapping_from_config(controller_config)
    assert torch.allclose(torch.tensor(scale), torch.tensor([0.05, 0.05, 0.05]))
    assert torch.allclose(torch.tensor(offset), torch.zeros(3))

    action_chunk = torch.zeros((1, 1, 7), dtype=torch.float32)
    action_chunk[0, 0, :3] = torch.tensor([1.0, -1.0, 0.5])
    traj = ObstacleGuidanceUtils.action_chunk_to_eef_xyz_traj(
        action_chunk=action_chunk,
        current_eef_pos=torch.zeros((1, 3), dtype=torch.float32),
        horizon=1,
        delta_pos_scale=scale,
        delta_pos_offset=offset,
    )
    assert torch.allclose(traj[0, 0], torch.tensor([0.05, -0.05, 0.025]))


def test_post_hoc_refinement_reduces_cost():
    action_chunk = torch.zeros((1, 4, 7), dtype=torch.float32)
    current_eef_pos = torch.zeros((1, 3), dtype=torch.float32)
    obstacle_centers_xyz = torch.tensor([[0.0, 0.0, 0.05]], dtype=torch.float32)
    obstacle_radii = torch.tensor([0.06], dtype=torch.float32)
    obstacle_top_z = torch.tensor([0.10], dtype=torch.float32)

    initial_cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
        action_chunk=action_chunk,
        current_eef_pos=current_eef_pos,
        obstacle_centers_xyz=obstacle_centers_xyz,
        obstacle_radii=obstacle_radii,
        obstacle_top_z=obstacle_top_z,
        z_clearance=0.03,
        horizon=4,
        delta_pos_scale=0.05,
    )
    refined = action_chunk
    for _ in range(5):
        refined_in = refined.detach().requires_grad_(True)
        cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
            action_chunk=refined_in,
            current_eef_pos=current_eef_pos,
            obstacle_centers_xyz=obstacle_centers_xyz,
            obstacle_radii=obstacle_radii,
            obstacle_top_z=obstacle_top_z,
            z_clearance=0.03,
            horizon=4,
            delta_pos_scale=0.05,
        )
        refined, _ = ObstacleGuidanceUtils.normalized_negative_cost_grad_update(
            update_sample=refined_in,
            cost=cost,
            scale=0.1,
            grad_source=refined_in,
        )
        refined = torch.clamp(refined, -1.0, 1.0)

    final_cost = ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
        action_chunk=refined,
        current_eef_pos=current_eef_pos,
        obstacle_centers_xyz=obstacle_centers_xyz,
        obstacle_radii=obstacle_radii,
        obstacle_top_z=obstacle_top_z,
        z_clearance=0.03,
        horizon=4,
        delta_pos_scale=0.05,
    )
    assert final_cost.item() < initial_cost.item()


if __name__ == "__main__":
    test_obstacle_xy_cost()
    test_obstacle_xyz_cylinder_cost()
    test_controller_delta_pos_mapping()
    test_post_hoc_refinement_reduces_cost()
    print("obstacle guidance utility checks passed")
