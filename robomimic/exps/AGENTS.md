# Experiment Config Directory

Project experiment configs live under `robomimic/exps/`. Generated default
templates live under `templates/` and should not be edited manually.

## Current Layout

```text
exps/
├── AGENTS.md
├── templates/                    # generated algorithm defaults
├── baseline/                     # original robosuite OSC action labels
├── delta_eef/                    # older position/action-interface experiments
├── absolute_eef_osc/             # executable absolute full-pose EEF baseline
└── delta_eef_pose_osc/           # executable delta full-pose EEF policies
    ├── diffusion_policy_can_image.json
    ├── diffusion_policy_can_pointcloud_ddpm100.json
    └── lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json
```

## Active Starting Config

The parent project's reliable source policy uses:

```text
delta_eef_pose_osc/diffusion_policy_can_image.json
```

Its action key is `delta_eef_pose_action`, with a 7-D executable world-frame
delta EEF full-pose action, min-max normalization, DDIM, and `2 / 16 / 8`
observation / prediction / execution horizons.

Targeted fine-tuning configs should be derived from this file and should:

- initialize from the epoch-260 checkpoint with `experiment.ckpt_path`;
- keep the observation and action contracts unchanged;
- list clean and correction HDF5 files separately under `train.data`;
- use explicit dataset weights;
- keep action normalization identical to the source checkpoint;
- use a distinct experiment name and output directory.

## Historical Configs

| Directory / config | Status |
|---|---|
| `baseline/` | original OSC-action experiments; not the active interface |
| `delta_eef/` | superseded position-only / adapter-era experiments |
| `absolute_eef_osc/` | retained comparison baseline |
| `diffusion_policy_can_pointcloud_ddpm100.json` | completed seed-500 point-cloud baseline; missed clean gate |
| `lan_o3dp_can_delta_pose_eps_residual_40demo_seed42.json` | completed 40-demo LAN-aligned baseline; branch closed |

Do not reuse a historical config merely because its training command still
runs. Check the parent repository's `docs/RESEARCH_LOG.md` and active plan
before launching an experiment.

## Naming

Use:

```text
<algorithm>_<environment>_<important-variant>.json
```

Keep data-source identifiers such as annotator names in `train.data.path`, not
in the config filename. Use a unique `experiment.name` for every result family.

## Adding a Config

1. Copy the closest active config.
2. Change only the intended experimental axis.
3. Verify `train.data`, observation modalities, action keys, normalization,
   horizons, scheduler, seed, checkpoint initialization, and output name.
4. Run a debug train and checkpoint reload before a production run.
5. Record the exact config, repository commit, checkpoint hash, and dataset
   hashes in the parent-project experiment manifest.
