"""
Utilities for converting robosuite RGB observations into binary target-object
mask images using MuJoCo segmentation renders.
"""

import numpy as np


def image_obs_key_to_camera_name(obs_key):
    """
    Converts a robomimic image observation key to a robosuite camera name.
    """
    suffix = "_image"
    if not obs_key.endswith(suffix):
        raise ValueError("Expected image observation key ending with '{}', got '{}'".format(suffix, obs_key))
    return obs_key[: -len(suffix)]


def _matching_target_object(raw_env, target_object):
    target_object = target_object.lower()
    for obj in getattr(raw_env, "objects", []):
        if getattr(obj, "name", "").lower() == target_object:
            return obj
    return None


def _geom_name_to_id(sim, geom_name):
    try:
        return sim.model.geom_name2id(geom_name)
    except Exception:
        return None


def get_target_geom_ids(raw_env, target_object):
    """
    Returns MuJoCo geom ids for the active target object, excluding visual goal
    markers such as ``VisualCan_*``.
    """
    sim = raw_env.sim
    geom_names = []

    target_obj = _matching_target_object(raw_env, target_object)
    if target_obj is not None:
        geom_names.extend(getattr(target_obj, "visual_geoms", []))
        geom_names.extend(getattr(target_obj, "contact_geoms", []))

    # Fallback for restored XMLs or custom envs where object bookkeeping differs.
    target_prefix = "{}_".format(target_object)
    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name is None:
            continue
        if geom_name.startswith(target_prefix) and not geom_name.startswith("Visual"):
            geom_names.append(geom_name)

    geom_ids = []
    for geom_name in geom_names:
        geom_id = _geom_name_to_id(sim, geom_name)
        if geom_id is not None:
            geom_ids.append(geom_id)

    geom_ids = sorted(set(geom_ids))
    if len(geom_ids) == 0:
        raise ValueError("Could not find geoms for target object '{}'".format(target_object))
    return geom_ids


def render_target_mask_image(
    raw_env,
    camera_name,
    height,
    width,
    target_object="Can",
    target_value=255,
    background_value=0,
    target_geom_ids=None,
    flip=True,
):
    """
    Renders a binary HWC uint8 image where target-object pixels have
    ``target_value`` and every other pixel has ``background_value``.
    """
    seg = raw_env.sim.render(
        camera_name=camera_name,
        width=width,
        height=height,
        depth=False,
        segmentation=True,
    )
    geom_ids = seg[:, :, 1]
    if flip:
        geom_ids = geom_ids[::-1]

    if target_geom_ids is None:
        target_geom_ids = get_target_geom_ids(raw_env=raw_env, target_object=target_object)
    mask = np.isin(geom_ids, target_geom_ids)

    image = np.full((height, width, 3), background_value, dtype=np.uint8)
    image[mask] = target_value
    return image


def apply_target_mask_images_to_obs(raw_env, obs, mask_config):
    """
    Replaces configured RGB observations in ``obs`` with target-object mask
    images rendered from the current simulator state.
    """
    if not mask_config or not mask_config.get("enabled", False):
        return obs

    obs_keys = mask_config.get("obs_keys", mask_config.get("image_obs_keys", []))
    target_object = mask_config.get("target_object", "Can")
    target_value = mask_config.get("target_value", 255)
    background_value = mask_config.get("background_value", 0)

    for obs_key in obs_keys:
        if obs_key not in obs:
            continue
        height, width = obs[obs_key].shape[:2]
        camera_name = image_obs_key_to_camera_name(obs_key)
        obs[obs_key] = render_target_mask_image(
            raw_env=raw_env,
            camera_name=camera_name,
            height=height,
            width=width,
            target_object=target_object,
            target_value=target_value,
            background_value=background_value,
        )
    return obs
