# PC Obstacle Guidance Experiment

## Code Context

- Branch: `exp/pc-obstacle-guidance`
- Rollout logging commit: `5c2d535 Add rollout progress and video stats logging`
- Guidance commit: `6cb8d3e Add pointcloud obstacle-guided rollout script`
- The rollout logging commit is intentionally independent so it can be cherry-picked without the guidance code.

## Checkpoint

```bash
robomimic/runs/trained_models/diffusion_policy_can_yq_masked_image/20260506153143/models/model_epoch_140_image_v15_can_mask_success_1.0.pth
```

## Completed Baseline Rollouts

Baseline rollouts used `robomimic/scripts/run_trained_agent.py` with:

- seeds: `600`, `601`, `602`
- horizon: `400`
- rollouts per run: `50`
- cameras: `agentview robot0_eye_in_hand`
- video mode: `--video_target_mask_grid`

Local outputs are intentionally not tracked by git:

```bash
outputs/rollouts_mask_grid_image_v15_can/
```

That directory contains per-environment videos, logs, stats JSON files, and a local `REPRODUCE.md` with exact commands.

| Env | Seed | Success Rate | Num Success | Avg Horizon |
|---|---:|---:|---:|---:|
| PickPlaceCan | 600 | 0.94 | 47/50 | 296.76 |
| PickPlaceCan | 601 | 0.96 | 48/50 | 296.42 |
| PickPlaceCan | 602 | 0.96 | 48/50 | 292.70 |
| PickPlaceBreadCan | 600 | 0.96 | 48/50 | 289.30 |
| PickPlaceBreadCan | 601 | 0.92 | 46/50 | 303.46 |
| PickPlaceBreadCan | 602 | 0.98 | 49/50 | 298.86 |
| PickPlaceBreadCerealCan | 600 | 0.88 | 44/50 | 298.64 |
| PickPlaceBreadCerealCan | 601 | 0.76 | 38/50 | 324.92 |
| PickPlaceBreadCerealCan | 602 | 0.80 | 40/50 | 312.36 |
| PickPlaceBreadCerealMilkCan | 600 | 0.74 | 37/50 | 325.98 |
| PickPlaceBreadCerealMilkCan | 601 | 0.80 | 40/50 | 331.48 |
| PickPlaceBreadCerealMilkCan | 602 | 0.78 | 39/50 | 316.86 |

## Guidance Entry Point

Use this script for obstacle / pointcloud guidance experiments:

```bash
robomimic/scripts/run_obstacle_guided_agent.py
```

The current guidance work adds pointcloud obstacle geometry support in addition to oracle-center guidance. The intended next comparison is to rerun the same 4 environments x 3 seeds setup with guidance enabled, then compare success rate, average horizon, and guidance/collision diagnostics against the baseline above.

Metric definitions for obstacle-guided rollouts are documented in
`experiments/obstacle_guided_metrics.md`.
