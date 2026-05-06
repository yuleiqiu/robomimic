"""
Create a copy of a robomimic image dataset whose RGB observations are replaced
by binary target-object mask images rendered from robosuite segmentation.

Example:
    python robomimic/scripts/create_target_mask_image_dataset.py \
        --dataset datasets/can/yq/image_v15.hdf5 \
        --output datasets/can/yq/image_v15_can_mask.hdf5 \
        --target_object Can \
        --obs_keys agentview_image robot0_eye_in_hand_image
"""

import argparse
import json
import os
import shutil

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.target_mask_utils as TargetMaskUtils


def default_output_path(dataset_path, target_object):
    root, ext = os.path.splitext(dataset_path)
    return "{}_{}_mask{}".format(root, target_object.lower(), ext)


def get_camera_specs(hdf5_file, obs_keys):
    demos = sorted(list(hdf5_file["data"].keys()))
    if len(demos) == 0:
        raise ValueError("Dataset has no demos")

    first_demo = hdf5_file["data/{}".format(demos[0])]
    camera_names = []
    height = None
    width = None
    for obs_key in obs_keys:
        if "obs/{}".format(obs_key) not in first_demo:
            raise ValueError("Observation key '{}' not found in dataset".format(obs_key))
        shape = first_demo["obs/{}".format(obs_key)].shape
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError("Expected '{}' to have shape (T, H, W, 3), got {}".format(obs_key, shape))
        key_height, key_width = shape[1], shape[2]
        if height is None:
            height = key_height
            width = key_width
        elif height != key_height or width != key_width:
            raise ValueError("All cameras must use the same H/W for env creation")
        camera_names.append(TargetMaskUtils.image_obs_key_to_camera_name(obs_key))
    return camera_names, height, width


def make_processing_env(dataset_path, obs_keys):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    with h5py.File(dataset_path, "r") as f:
        camera_names, camera_height, camera_width = get_camera_specs(f, obs_keys)

    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
    )


def render_masks_for_current_state(
    env,
    demo_grp,
    obs_keys,
    target_object,
    target_value,
    background_value,
    target_geom_ids,
):
    masks = {}
    raw_env = env.env
    for obs_key in obs_keys:
        _, height, width, _ = demo_grp["obs/{}".format(obs_key)].shape
        camera_name = TargetMaskUtils.image_obs_key_to_camera_name(obs_key)
        masks[obs_key] = TargetMaskUtils.render_target_mask_image(
            raw_env=raw_env,
            camera_name=camera_name,
            height=height,
            width=width,
            target_object=target_object,
            target_value=target_value,
            background_value=background_value,
            target_geom_ids=target_geom_ids,
        )
    return masks


def process_demo(env, demo_grp, obs_keys, target_object, target_value, background_value, write_next_obs=True):
    states = demo_grp["states"][()]
    actions = demo_grp["actions"][()]
    traj_len = states.shape[0]

    initial_state = {"states": states[0]}
    if "model_file" in demo_grp.attrs:
        initial_state["model"] = demo_grp.attrs["model_file"]
    if "ep_meta" in demo_grp.attrs:
        initial_state["ep_meta"] = demo_grp.attrs["ep_meta"]

    env.reset_to(initial_state)
    target_geom_ids = TargetMaskUtils.get_target_geom_ids(raw_env=env.env, target_object=target_object)

    obs_masks = {}
    for obs_key in obs_keys:
        obs_dataset = demo_grp["obs/{}".format(obs_key)]
        obs_masks[obs_key] = np.empty(obs_dataset.shape, dtype=obs_dataset.dtype)

    for t in tqdm(range(traj_len), desc="frames", leave=False):
        if t > 0:
            env.reset_to({"states": states[t]})
        cur_masks = render_masks_for_current_state(
            env=env,
            demo_grp=demo_grp,
            obs_keys=obs_keys,
            target_object=target_object,
            target_value=target_value,
            background_value=background_value,
            target_geom_ids=target_geom_ids,
        )
        for obs_key in obs_keys:
            obs_masks[obs_key][t] = cur_masks[obs_key]

    for obs_key in obs_keys:
        demo_grp["obs/{}".format(obs_key)][:] = obs_masks[obs_key]

    if not write_next_obs or "next_obs" not in demo_grp:
        return

    next_obs_masks = {
        obs_key: np.empty(
            demo_grp["next_obs/{}".format(obs_key)].shape,
            dtype=demo_grp["next_obs/{}".format(obs_key)].dtype,
        )
        for obs_key in obs_keys
        if "next_obs/{}".format(obs_key) in demo_grp
    }
    for obs_key in next_obs_masks:
        if traj_len > 1:
            next_obs_masks[obs_key][:-1] = obs_masks[obs_key][1:]

    env.reset_to({"states": states[-1]})
    env.step(actions[-1])
    final_masks = render_masks_for_current_state(
        env=env,
        demo_grp=demo_grp,
        obs_keys=list(next_obs_masks.keys()),
        target_object=target_object,
        target_value=target_value,
        background_value=background_value,
        target_geom_ids=target_geom_ids,
    )
    for obs_key in next_obs_masks:
        next_obs_masks[obs_key][-1] = final_masks[obs_key]
        demo_grp["next_obs/{}".format(obs_key)][:] = next_obs_masks[obs_key]


def create_target_mask_dataset(args):
    dataset_path = os.path.expanduser(args.dataset)
    output_path = os.path.expanduser(args.output or default_output_path(dataset_path, args.target_object))

    if os.path.abspath(dataset_path) == os.path.abspath(output_path):
        raise ValueError("Output path must be different from input dataset path")
    if os.path.exists(output_path):
        if not args.overwrite:
            raise FileExistsError("Output path already exists: {}. Use --overwrite to replace it.".format(output_path))
        os.remove(output_path)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(dataset_path, output_path)

    env = make_processing_env(dataset_path=dataset_path, obs_keys=args.obs_keys)

    mask_metadata = {
        "enabled": True,
        "target_object": args.target_object,
        "obs_keys": args.obs_keys,
        "target_value": args.target_value,
        "background_value": args.background_value,
    }

    with h5py.File(output_path, "r+") as f:
        f["data"].attrs["target_mask_image"] = json.dumps(mask_metadata, indent=4)
        demos = sorted(list(f["data"].keys()))
        if args.demo_limit is not None:
            demos = demos[: args.demo_limit]

        for demo_key in tqdm(demos, desc="demos"):
            process_demo(
                env=env,
                demo_grp=f["data/{}".format(demo_key)],
                obs_keys=args.obs_keys,
                target_object=args.target_object,
                target_value=args.target_value,
                background_value=args.background_value,
                write_next_obs=(not args.skip_next_obs),
            )

    print("Wrote target-mask dataset to {}".format(output_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to source robomimic hdf5 dataset")
    parser.add_argument("--output", type=str, default=None, help="Path to output hdf5 dataset")
    parser.add_argument("--target_object", type=str, default="Can", help="Target object name in robosuite")
    parser.add_argument(
        "--obs_keys",
        type=str,
        nargs="+",
        default=["agentview_image", "robot0_eye_in_hand_image"],
        help="RGB observation keys to replace with target masks",
    )
    parser.add_argument("--target_value", type=int, default=255, help="Pixel value for target-object pixels")
    parser.add_argument("--background_value", type=int, default=0, help="Pixel value for non-target pixels")
    parser.add_argument("--skip_next_obs", action="store_true", help="Only replace obs images, not next_obs images")
    parser.add_argument("--demo_limit", type=int, default=None, help="Optional debug limit on number of demos to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output path if it already exists")
    return parser.parse_args()


if __name__ == "__main__":
    create_target_mask_dataset(parse_args())
