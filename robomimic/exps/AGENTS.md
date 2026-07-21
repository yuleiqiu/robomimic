# Config Directory Convention

## Directory Structure

```
exps/
├── AGENTS.md            ← this file
├── templates/           ← auto-generated algorithm templates (do not edit)
│   ├── diffusion_policy.json
│   ├── bc.json
│   └── ...
├── baseline/            ← original OSC action label experiments
│   ├── diffusion_policy_can_image.json
│   └── diffusion_policy_can_masked_image.json
├── delta_eef/           ← real EEF delta label experiments
│   └── diffusion_policy_can_image.json
└── <future_folder>/     ← e.g., absolute_eef, noise_augment, etc.
```

## Naming Convention

### Directories

Each directory represents an **experiment variant** — a single axis of the experiment matrix.

Current axes:
- `baseline` — original OSC delta action labels
- `delta_eef` — real EEF delta labels (eliminates action→trajectory mapping error)

When adding a new variant, create a new top-level folder under `exps/`.

### Files

Format: `{algo}_{env}_{modality}.json`

| Segment | Meaning | Example |
|---------|---------|---------|
| `algo` | Algorithm name | `diffusion_policy` |
| `env` | Environment shorthand | `can` = PickPlaceCan |
| `modality` | Observation modality | `image`, `masked_image` |

Examples:
```
baseline/diffusion_policy_can_image.json
baseline/diffusion_policy_can_masked_image.json
delta_eef/diffusion_policy_can_image.json
```

Do NOT include data-source identifier (e.g., `yq`) in filenames — that belongs in `train.data.path`.

## Key Mapping

Each experiment variant corresponds to a specific `action_keys` value:

| Folder | `action_keys` | `dataset_keys` | Data file |
|--------|---------------|----------------|-----------|
| `baseline` | `["actions"]` | `["actions", "rewards", ...]` | `image_v15.hdf5` / `image_v15_mask.hdf5` |
| `delta_eef` | `["delta_eef_action"]` | `["delta_eef_action", "rewards", ...]` | `image_v15_delta_eef.hdf5` |

## How to Add a New Experiment

1. Create a new folder: `exps/<variant_name>/`
2. Copy an existing config from the closest variant
3. Modify:
   - `experiment.name` — unique experiment name
   - `experiment.logging.wandb_proj_name` — W&B project
   - `train.data[0].path` — data file
   - `train.dataset_keys` / `train.action_keys` / `train.action_config` — if label type changed
4. Leave `algo`, `observation`, `meta` unchanged unless deliberately modifying the model or inputs
