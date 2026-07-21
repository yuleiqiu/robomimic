# robomimic Context

> Forked robomimic used by the parent project for clean-image Diffusion Policy
> training and rollout on executable EEF-pose OSC action targets.

> **Status (2026-07-15)**: A registered `guided_diffusion_policy` inference
> variant implements the first delta-EEF guided-denoising contract. The old
> obstacle-guidance, OSC forward-model, and action-ranking implementations
> remain removed. The active guidance path reconstructs trajectories directly
> from `delta_eef_pose_action`.

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
| `GuidedDiffusionPolicyUNet` | `algo/guided_diffusion_policy.py` | Registered opt-in DDIM guided variant |
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

## 3. Active EEF-Pose OSC Configs

| Config | Action Key | Purpose |
|--------|------------|---------|
| `robomimic/exps/delta_eef_pose_osc/diffusion_policy_can_image.json` | `delta_eef_pose_action` | Preferred clean-image policy target |
| `robomimic/exps/absolute_eef_osc/diffusion_policy_can_image.json` | `abs_eef_pose_action` | Absolute EEF comparison baseline |

Both use clean image observations, DDIM with `num_train_timesteps=100` and
`num_inference_timesteps=10`, and min-max action normalization.

## 4. Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Training entry: config -> data -> model -> train loop -> checkpoint |
| `scripts/run_trained_agent.py` | Roll out a checkpoint in one process |
| `scripts/run_trained_agent_parallel.py` | Multi-process checkpoint rollout |
| `scripts/create_target_mask_image_dataset.py` | Preprocess PickPlace datasets with target mask images |

## 5. Project-Specific Notes

- `envs/env_robosuite.py` includes controller refresh after `reset_to`, needed
  for reliable OSC absolute / desired-goal delta replay.
- `utils/torch_utils.py` honors `ROBOMIMIC_GPU_ID` so separate training runs can
  bind PyTorch devices without using `CUDA_VISIBLE_DEVICES`.
- `scripts/train.py` honors `ROBOMIMIC_TORCH_THREADS` for local throughput
  tuning.
- The old forward-model implementation is intentionally absent. Use
  `outputs/eef_pose_osc_policy/README.md` in the parent repo for the current
  conclusion and `docs/forward_model_guidance_next_steps.md` only as an
  archived result document.
- Guided deployment context is runtime-only. Build it from the existing
  `RolloutPolicy.action_normalization_stats`, then call
  `policy.policy.set_guidance_context(...)` after `start_episode` and before a
  new action chunk is sampled.
- Guidance is DDIM-only, evaluates the executed `[:, 1:9]` clean-action slice,
  and directly updates only that slice's XY delta-position coordinates.

## 6. Guided-Denoising Verification

Run the independent unit checks from the robomimic repo:

```bash
uv run python tests/test_guided_denoising_utils.py
```

The tests cover reconstruction, point versus displacement normalization,
finite cost gradients, pushed-waypoint differencing, recorded before / after
waypoint displacement vectors, executed-slice / action-dimension preservation,
zero-cost parity, and algorithm / config registration.
Checkpoint smoke verification should additionally confirm that the epoch-260
delta-EEF checkpoint loads through `guided_policy_from_checkpoint`, that
disabled and zero-scale samples exactly match the base policy under the same
observation and random seed, and that a nonzero scale produces ten finite DDIM
diagnostic records.
