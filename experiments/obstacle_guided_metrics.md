# Obstacle-Guided Evaluation Metrics

This note defines the extra metrics emitted by
`robomimic/scripts/run_obstacle_guided_agent.py` for obstacle-guided rollouts.
The goal is to make safety and guidance diagnostics explicit, without
overloading task success metrics.

## Task Metrics

- `Return`: rollout return from the environment.
- `Horizon`: number of environment steps executed before success, done, or the
  horizon limit.
- `Success_Rate`: `1.0` for a successful rollout and `0.0` otherwise.
- `Num_Success`: aggregate count of successful rollouts.

These are the primary task-performance metrics and should be compared against
the baseline rollouts from `run_trained_agent.py`.

## Pointcloud Diagnostics

- `Pointcloud_Total_Point_Count`: mean number of obstacle pointcloud points
  constructed per environment step after mask, depth validity, workspace crop,
  voxel downsampling, and max-point filtering.
- `Pointcloud_Point_Count`: backward-compatible alias for
  `Pointcloud_Total_Point_Count`.

This is a diagnostic sanity check, not a safety metric. It answers whether the
pointcloud pipeline sees obstacle points. It does not indicate how many points
contribute nonzero guidance cost, and it is expected to be similar across
no-guidance, PC-0, and PC-1 when they observe similar trajectories.

`Visual*` target-bin marker objects in PickPlace are intentionally excluded.
They are display/layout markers, not real distractor objects. `PickPlaceCan`
has no active non-target distractors, so a zero pointcloud count is expected.

## Guidance Activity Metrics

- `Obstacle_Guidance_Trigger_Count`: number of action-trajectory samples where
  obstacle guidance was applied during the rollout.
- `Obstacle_Guidance_Trigger_Rate`: trigger count divided by rollout horizon.
- `Obstacle_Guidance_Cost`: mean guidance cost over applied samples.
- `Obstacle_Guidance_Min_Distance`: minimum predicted distance to obstacle
  geometry over applied samples.
- `Obstacle_Guidance_Positive_Cost_Count`: number of applied samples with
  strictly positive guidance cost.
- `Obstacle_Guidance_Positive_Cost_Rate`: positive-cost count divided by
  trigger count.

Trigger count is measured at diffusion action-chunk sampling frequency, not at
every environment step. With an action horizon of 8, a 60-step rollout typically
has about 8 guidance triggers.

## Non-Target Collision Metrics

Collision metrics count contacts between robot / gripper collision geoms and
active non-target object contact geoms. Target object contacts, table/bin
contacts, and `Visual*` marker contacts are excluded.

For each environment step, each non-target object can be counted at most once,
even if multiple MuJoCo geom contacts occur in that step.

- `Non_Target_Collision_Object_Counts`: per-rollout dictionary from object name
  to number of environment steps in contact with that object.
- `Non_Target_Collision_Count`: sum of per-object counts for the rollout.
- `Non_Target_Collision_Step_Count`: number of rollout steps where at least one
  non-target object was contacted.
- `Non_Target_Collision_Rate`: collision step count divided by rollout horizon.
- `Non_Target_Collision_Any`: `1.0` if the rollout had any non-target collision,
  else `0.0`.

The stats JSON also includes aggregate helpers:

- `totals.Non_Target_Collision_Object_Counts`: object-level counts summed over
  all rollouts.
- `per_rollout_average.Non_Target_Collision_Object_Counts`: object-level counts
  averaged over rollouts.
- `Num_Non_Target_Collision_Rollouts`: number of rollouts with any non-target
  collision.

These metrics are intended to measure safety side effects of obstacle guidance
and should be reported alongside task success.
