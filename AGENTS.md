# robomimic Context

> Forked robomimic — adds guided diffusion policy with obstacle avoidance + parallel rollout.

> **Status (2026-07-06)**: The obstacle-guidance code (`GuidedDiffusionPolicyUNet` + `utils/obstacle_guidance_utils.py`) is **retained as a reference implementation but the gradient-guided OSC-action deployment path was rejected**. Forward-model and action-ranking code is also retained as a completed diagnostic branch. The parent repo has reopened Route B after confirming that built-in `OSC_POSE` can replay full-pose absolute EEF actions of the form `[next_eef_pos, quat2axisangle(next_eef_quat_site), gripper]`; see `../docs/route_b_validation/report.md`.

## 1. Architecture Overview

```
robomimic/
├── algo/         — Algorithm implementations (algo.py base class + per-algo files)
├── config/       — Config system (base_config.py + per-algo config subclasses)
├── models/       — Neural nets (obs encoders, ConditionalUnet1D, transformers)
├── envs/         — Environment wrappers (env_robosuite.py, wrappers.py)
├── scripts/      — Entry points (train.py, run_trained_agent.py, run_obstacle_guided_agent.py)
├── utils/        — Data loading, file I/O, guidance utils, tensor ops
├── exps/templates/ — Auto-generated JSON config templates per algo
└── macros.py     — Global DEBUG/W&B flags
```

### Algo Registry

Global `REGISTERED_ALGO_FACTORY_FUNCS` dict maps algo name → factory function. Each algo module registers via `@register_algo_factory_func("name")` decorator. `algo_factory()` at `algo/algo.py:54` looks up the factory, calls it with config, instantiates the algo class.

### Class Hierarchy

| Class | Location | Role |
|-------|----------|------|
| `Algo` | `algo/algo.py:89` | Root base: owns `self.nets`, creates optimizer, handles checkpoint serialize/deserialize |
| `PolicyAlgo(Algo)` | `algo/algo.py:372` | Adds abstract `get_action()` |
| `DiffusionPolicyUNet(PolicyAlgo)` | `algo/diffusion_policy.py:52` | Core DDPM diffusion policy |
| `GuidedDiffusionPolicyUNet(DiffusionPolicyUNet)` | `algo/guided_diffusion_policy.py:42` | Guided variant with obstacle cost gradients |
| `RolloutPolicy` | `algo/algo.py:496` | **Wrapper** (not subclass) for rollout interaction: handles obs norm, action unnorm, queuing |

### Training Pipeline

`scripts/train.py` → loads config + data → `algo_factory()` → `TrainUtils.run_epoch()` (calls `model.train_on_batch()` per batch) → periodic eval rollouts → saves checkpoint (state_dict + config + env_meta + normalization stats as `.pth`).

## 2. Config System

- **`BaseConfig(Config)`** (`config/base_config.py:48`): Nested dict with attribute access. Constructor calls 5 setup methods: `experiment_config()`, `train_config()`, `algo_config()` (abstract), `observation_config()`, `meta_config()`. Then `lock_keys()`.
- **`DiffusionPolicyConfig(BaseConfig)`** (`config/diffusion_policy_config.py:7`): Our config. Sets `seq_length=16`, `horizon=[obs=2, action=8, pred=16]`, DDPM with 100 steps (squaredcos_cap_v2 beta schedule, `prediction_type='epsilon'`), UNet with `down_dims=[256,512,1024]`, AdamW LR=1e-4.
- **`config_factory(algo_name)`** at `config/base_config.py:24`: Creates config from registered algo name.
- **`exps/templates/diffusion_policy.json`**: Auto-generated default config (184 lines).
- **Checkpoint loading**: `FileUtils.policy_from_checkpoint()` at `utils/file_utils.py:373` — loads `.pth`, reinstantiates config + model, deserializes weights.

## 3. Diffusion Policy Pipeline

### `DiffusionPolicyUNet` (`algo/diffusion_policy.py`)

**Network**: `_create_networks()` (L53) creates `ObservationGroupEncoder` → `ConditionalUnet1D` (from `models/diffusion_policy_nets.py:110`, 1D conv UNet with FiLM conditioning). Replaces all BatchNorm with GroupNorm for EMA compatibility.

**Training** (`train_on_batch`, L156):
1. Encode obs → flatten to `[B, T*D]` global cond
2. Sample noise ε, timestep t
3. Forward diffuse: `noisy = add_noise(action, ε, t)`
4. Predict: `ε_pred = unet(noisy, t, cond=obs)`
5. Loss: `MSE(ε_pred, ε)`
6. Backprop + EMA step

**Inference** (`get_action`, L273): Uses action queue — if empty, runs `_get_action_trajectory()` (L304) which does full reverse diffusion over `prediction_horizon=16` steps, then fills queue with first `action_horizon=8` actions.

### `GuidedDiffusionPolicyUNet` (`algo/guided_diffusion_policy.py`)

Extends `DiffusionPolicyUNet`. Key additions:

- **`_guided_scheduler_step()`** (L155): For each denoising step, detaches `naction`, enables grad, predicts noise → estimates `x0_hat` (clean action) via `estimate_clean_action_from_scheduler()` → computes obstacle cost on `x0_hat` → applies `sample -= rho_t * norm(grad)`.
- **`_obstacle_guidance_cost()`** (L95): Dispatches to `obstacle_xy_cost`, `obstacle_xyz_cylinder_cost`, or `obstacle_pointcloud_cost` based on config.
- **`_refine_obstacle_guidance_action()`** (L238): Post-denoising iterative gradient refinement (optional).
- **`_get_action_trajectory()`** (L310): Overrides base — interleaves standard scheduler steps with `_guided_scheduler_step()` when `step_index >= guidance_start_step`.

### `wrap_as_guided(policy)` (L13)

In-place monkey-patch factory: copies `GuidedDiffusionPolicyUNet`'s guidance methods + attributes onto a loaded `DiffusionPolicyUNet` instance. Enables adding guidance at rollout time without retraining.

## 4. Guidance Utilities (`utils/obstacle_guidance_utils.py`)

### Core Mapping (action → EEF trajectory)

- **`action_chunk_to_eef_xyz_traj()`** (L668): `traj = eef_pos + cumsum(action[:,:,:3] * scale + offset)`. This remains an unreliable trajectory proxy for OSC-action guidance: the 3-4 cm RMSE vs actual OSC PD-controller dynamics is the controller's tracking behavior at 20 Hz, not a calibration issue. This does **not** reject Route B in its corrected form; full-pose absolute EEF actions can be executed directly through built-in `OSC_POSE` when the action includes both position and site orientation target.

### Cost Functions

| Function | Type | Description |
|----------|------|-------------|
| `obstacle_xy_cost` (L729) | 2D | Circle penetration: `Σ ReLU(r - dist)²` |
| `obstacle_xyz_cylinder_cost` (L798) | 3D | Cylinder penetration (XY × Z) |
| `obstacle_pointcloud_cost` (L877) | 3D | Point-to-pointcloud via `torch.cdist` |

All return `(cost, stats_dict)`, fully differentiable.

### Geometry Sources

- **`get_oracle_obstacle_geometry()`** (L242): Reads MuJoCo geom positions/sizes → returns `(centers_xyz, physical_radii, safety_radii, top_z, names)`.
- **`depth_mask_to_world_pointcloud()`** (L443): Renders obstacle segmentation mask → backprojects depth pixels to world point cloud.

### Guidance Math

- **`estimate_clean_action_from_scheduler()`** (L958): `x0 = (sample - sqrt(1-ᾱ_t)·ε_pred) / sqrt(ᾱ_t)`.
- **`guidance_scale_for_step()`** (L985): Returns rho_t per step (constant or late-ramp schedule).
- **`normalized_negative_cost_grad_update()`** (L1008): `sample -= scale * grad/‖grad‖`.

## 5. Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Training entry: config → data → model → train loop → save checkpoint |
| `scripts/run_trained_agent.py` | Baseline rollout: loads checkpoint → `RolloutPolicy` → N episodes |
| `scripts/run_trained_agent_parallel.py` | Multi-process version of above |
| `scripts/run_obstacle_guided_agent.py` | Guided rollout: loads checkpoint → `wrap_as_guided()` → per-step guidance context with obstacle cost injection |
| `scripts/create_target_mask_image_dataset.py` | Preprocesses PickPlace datasets with target mask images |

## 6. Quick Index (non-core modules, consult on demand)

| Module | File | What's there |
|--------|------|-------------|
| Obs encoder | `models/obs_nets.py` | `ObservationGroupEncoder` (L426) — multi-modal, time-distributed |
| UNet | `models/diffusion_policy_nets.py` | `ConditionalUnet1D` (L110) — FiLM-conditioned 1D conv UNet |
| Base nets | `models/base_nets.py` | `MLP`, `RNN_Base`, `ResNet18Conv`, `SpatialSoftmax` |
| Data | `utils/dataset.py` | `SequenceDataset` — hdf5 sequence sampling |
| File I/O | `utils/file_utils.py` | `policy_from_checkpoint()` (L373), `env_from_checkpoint()`, `config_from_checkpoint()` |
| Train loop | `utils/train_utils.py` | `run_epoch()` (L637), `save_model()` (L588), learning rate scheduling |
| Env wrappers | `envs/env_robosuite.py` | `EnvRobosuite(EnvBase)` (L46) — robosuite env wrapper, camera matrices, action processing |
| Env wrappers | `envs/wrappers.py` | `EnvWrapper` (L12) — delegation chain for obs randomizers, frame stacking |

## 7. Key Constants

| Constant | Location | Value |
|----------|----------|-------|
| `prediction_horizon` | `config/diffusion_policy_config.py:49` | 16 |
| `action_horizon` | `config/diffusion_policy_config.py:48` | 8 |
| `observation_horizon` | `config/diffusion_policy_config.py:47` | 2 |
| DDPM train/inference steps | `config/diffusion_policy_config.py:65-66` | 100/100 |
| UNet `down_dims` | `config/diffusion_policy_config.py:54` | `[256, 512, 1024]` |
| Default `guidance_scale` | `scripts/run_obstacle_guided_agent.py:876` | 0.03 |
| Default `guidance_start_step_pct` | `scripts/run_obstacle_guided_agent.py:896` | 0.7 |
| Default `safe_distance` (pointcloud) | `scripts/run_obstacle_guided_agent.py:914` | 0.02 |
| Default `z_clearance` | `scripts/run_obstacle_guided_agent.py:883` | 0.03 |
| Default `pc_voxel_size` | `scripts/run_obstacle_guided_agent.py:922` | 0.005 |
