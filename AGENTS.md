# robomimic Context

> Forked robomimic used by the parent project for clean-image Diffusion Policy
> training and rollout on executable EEF-pose OSC action targets.

> **Status (2026-08-07)**: the paper-aligned D2 guidance stage is archived with
> zero collision-to-clear transitions. The active Empty2D mechanism test uses
> standard low-dimensional Diffusion Policy plus a generic opt-in 2-D/3-D
> predicted-clean point-trajectory guidance variant. Its data and batch-256
> preflight gates pass; formal training is unrun.

## 1. Architecture Overview

```
robomimic/
├── algo/           — Algorithm implementations
├── config/         — Config system
├── models/         — Neural nets
├── envs/           — Environment wrappers
├── scripts/        — Entry points: train.py, run_trained_agent.py, parallel rollout
├── utils/          — Data loading, file I/O, tensor ops
├── exps/           — Experiment configs, including EEF-pose OSC configs
└── macros.py       — Global DEBUG/W&B defaults
```

### Algo Registry

Global `REGISTERED_ALGO_FACTORY_FUNCS` maps algo name to factory function. Each
algo module registers via `@register_algo_factory_func("name")`.
`algo_factory()` in `algo/algo.py` looks up the factory and instantiates the
algo class.

### Key Classes

| Class | Location | Role |
|-------|----------|------|
| `Algo` | `algo/algo.py` | Root base: owns `self.nets`, creates optimizer, handles checkpoint serialize / deserialize |
| `PolicyAlgo` | `algo/algo.py` | Adds abstract `get_action()` |
| `DiffusionPolicyUNet` | `algo/diffusion_policy.py` | Core DDPM / DDIM diffusion policy |
| `PairedCorrectionDataset` | `utils/dataset.py` | Fixed-window positive / negative correction pairs |
| `GuidedDiffusionPolicyUNet` | `algo/guided_diffusion_policy.py` | Registered opt-in DDIM guided variant |
| `LanO3DPUNet` | `algo/lan_o3dp.py` | LAN-aligned point-cloud diffusion policy |
| `GuidedLanO3DPUNet` | `algo/lan_o3dp.py` | Opt-in guided LAN variant for DDPM checkpoints |
| `PointGuidedDiffusionPolicyUNet` | `algo/point_guided_diffusion_policy.py` | Generic paper-gradient guidance for arbitrary 2-D/3-D absolute position action indices |
| `DP3PointCloudCore` | `models/obs_core.py` | 3→32→64→64 per-point MLP, residual, max-pool, 64-D output |
| `RolloutPolicy` | `algo/algo.py` | Rollout wrapper: obs norm, action unnorm, policy call |

## 2. Diffusion Policy Pipeline

`DiffusionPolicyUNet` creates:

- `ObservationGroupEncoder`
- `ConditionalUnet1D`
- DDPM or DDIM scheduler
- optional EMA model

Training:

1. Encode obs and flatten observation history into the global condition.
2. Sample noise and diffusion timestep.
3. Forward diffuse the action sequence.
4. Predict noise with the UNet.
5. Optimize MSE on predicted noise.

Inference:

1. If the internal action queue is empty, sample a full prediction horizon.
2. Slice the action horizon.
3. Queue actions and execute them left-to-right.
4. `RolloutPolicy` unnormalizes actions before `env.step`.

The base diffusion policy exposes an identity reverse-step hook. The guided
variant overrides only that hook; the base policy and `RolloutPolicy` remain
the unguided baseline. `utils/guided_denoising_utils.py` contains delta-EEF
trajectory reconstruction, the XY LAN penetration cost, waypoint pushing and
differencing, normalization-vector conversion, per-step diagnostics, and the
in-memory loader that maps an existing `diffusion_policy` checkpoint to the
registered guided variant without rewriting the checkpoint.

## 3. Project EEF-Pose OSC Configs

| Config | Action Key | Purpose |
|--------|------------|---------|
| `robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json` | `delta_eef_pose_action` | Preferred clean-image policy target |
| `robomimic/exps/absolute_eef_osc/diffusion_policy_can_image.json` | `abs_eef_pose_action` | Absolute EEF comparison baseline |
| `robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_pointcloud_ddpm100.json` | `delta_eef_pose_action` | Completed seed-500 point-cloud DDPM100 baseline |
| `robomimic/exps/delta_eef_pose_osc/lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json` | `delta_eef_pose_action` | Completed LAN-aligned 40-demo baseline |
| `robomimic/exps/empty2d_guidance/diffusion_policy_sample_seed42.json` | `actions` | Active low-dimensional absolute-XY Empty2D source |

The clean-image delta and absolute configs use DDIM with
`num_train_timesteps=100`, `num_inference_timesteps=10`, and min-max action
normalization.

The point-cloud configs use `task_pointcloud` through the existing `scan`
modality, DDPM epsilon prediction with 100 train / inference steps, horizons
2 / 16 / 8, and min-max normalization. They are historical experiment configs,
not the starting point for targeted fine-tuning.

## 4. Key Scripts

| Script | Purpose |
|--------|---------|
| `robomimic/scripts/train.py` | Training entry: config -> data -> model -> train loop -> checkpoint |
| `robomimic/scripts/run_trained_agent.py` | Roll out a checkpoint in one process |
| `robomimic/scripts/run_trained_agent_parallel.py` | Multi-process checkpoint rollout |
| `robomimic/scripts/create_target_mask_image_dataset.py` | Preprocess PickPlace datasets with target mask images |

## 5. Project-Specific Notes

- `envs/env_robosuite.py` includes controller refresh after `reset_to`, needed
  for reliable OSC absolute / desired-goal delta replay.
- `envs/env_robosuite.py` also supports an opt-in `target_pointcloud`
  observation provider and forces offscreen rendering for it even when RGB is
  not a policy observation. Offline generation and runtime rollout both call
  `utils/target_pointcloud_utils.py`.
- `utils/torch_utils.py` honors `ROBOMIMIC_GPU_ID` so separate training runs can
  bind PyTorch devices without using `CUDA_VISIBLE_DEVICES`.
- `scripts/train.py` honors `ROBOMIMIC_TORCH_THREADS` for local throughput
  tuning.
- `train.data` accepts multiple HDF5 dataset entries. `MetaDataset` provides
  weighted sampling; with `normalize_weights_by_ds_size=true`, dataset-level
  weights control expected source proportions instead of raw sequence counts.
- Paired-correction entries can opt into normalized `negative_actions` and an
  `is_paired_correction` label. Clean components receive identity negatives so
  standard collation remains valid.
- `algo.sdp` is opt-in and backward-compatible with old locked checkpoint
  configs. It generates detached constrained DDPM targets from the online
  UNet and averages `N` target losses within each original sample before the
  mixed batch mean.
- `experiment.ckpt_path` initializes model and EMA weights for a new training
  run with a fresh optimizer. `--resume` instead restores optimizer,
  scheduler, epoch, and variable state from the latest checkpoint.
- Action normalization is computed by the dataset before batches are created
  and saved in checkpoints. Targeted fine-tuning must verify that correction
  actions preserve the epoch-260 min-max coordinate system.
- `experiment.save.latest_every_n_epochs` controls how often the resumable
  `last.pth` / backup pair is rewritten; the final epoch is always saved.
- The old forward-model implementation is intentionally absent. Use
  `outputs/eef_pose_osc_policy/README.md` in the parent repo for the current
  conclusion and `docs/forward_model_guidance_next_steps.md` only as an
  archived result document.
- Guided deployment context is runtime-only. Build it from the existing
  `RolloutPolicy.action_normalization_stats`, then call
  `policy.policy.set_guidance_context(...)` after `start_episode` and before a
  new action chunk is sampled.
- `experiment.logging.wandb_required=true` is fail-fast: it requires online
  initialization, aborts on initialization or logging failure, and writes
  `logs/wandb_run.json` with run ID, URL, full config, and checkpoint mapping.
- `point_guided_policy_from_checkpoint` adapts an ordinary
  `diffusion_policy` checkpoint in memory. Its context names one action key and
  two or three local position indices; the obstacle point is fixed once per
  sampled chunk. The legacy paper-LAN 3-D API and behavior remain compatible.
- The RGB guided variant uses DDIM. The guided LAN variant also accepts DDPM
  scheduler outputs with a predicted clean sample. Both evaluate the executed
  `[:, 1:9]` clean-action slice and directly update only that slice's XY
  delta-position coordinates.

## 6. Guided-Denoising Verification

Run the relevant unit checks from the parent repository:

```bash
uv run pytest -q \
  third_party/robomimic/tests/test_guided_denoising_utils.py \
  third_party/robomimic/tests/test_lan_o3dp_pointcloud_core.py \
  third_party/robomimic/tests/test_lan_o3dp_registration.py \
  third_party/robomimic/tests/test_observation_min_max_normalization.py \
  third_party/robomimic/tests/test_rollout_checkpoint_selection.py
```

The Empty2D addition is covered by
`tests/test_point_trajectory_guidance.py`, `tests/test_wandb_required.py`, and
the unchanged `tests/test_paper_lan_guidance.py` suite in the parent command.

The tests cover reconstruction, point versus displacement normalization,
finite cost gradients, pushed-waypoint differencing, recorded before / after
waypoint displacement vectors, executed-slice / action-dimension preservation,
zero-cost parity, and algorithm / config registration.
Checkpoint smoke verification should additionally confirm that the epoch-260
delta-EEF checkpoint loads through `guided_policy_from_checkpoint`, that
disabled and zero-scale samples exactly match the base policy under the same
observation and random seed, and that a nonzero scale produces ten finite DDIM
diagnostic records.
