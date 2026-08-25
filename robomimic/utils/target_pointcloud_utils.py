"""
Deterministic single- or multi-view point clouds for robosuite environments.

The extractor intentionally uses oracle MuJoCo geom segmentation. It merges the
active target object with its translucent ``Visual<target>`` goal marker and
excludes every other geom. Depth pixels are unprojected into world coordinates
before deterministic farthest-point sampling.
"""

from dataclasses import dataclass

import numpy as np
from robosuite.utils import camera_utils as CameraUtils


POINTCLOUD_OBS_KEY = "task_pointcloud"
DEFAULT_POINTCLOUD_CONFIG = {
    "enabled": True,
    "obs_key": POINTCLOUD_OBS_KEY,
    # ``camera_name`` is the backward-compatible single-view spelling.
    # Supplying ``camera_names`` renders every listed view, concatenates their
    # world-frame points, and applies one deterministic FPS to the fused set.
    "camera_name": "agentview",
    "camera_names": None,
    "height": 256,
    "width": 256,
    "target_object": "Can",
    "include_visual_goal": True,
    # Optional explicit task-goal object names. When omitted, the historical
    # Visual<target> convention is preserved.
    "goal_objects": None,
    "num_points": 512,
    "padding_mode": "repeat",
    # Reject transparent-goal segmentation pixels whose depth belongs to an
    # occluding/background geom. This is comfortably larger than a Can.
    "max_geom_distance": 0.15,
}


@dataclass(frozen=True)
class TargetPointCloudRender:
    points: np.ndarray
    valid_points: np.ndarray
    target_geom_ids: tuple
    camera_names: tuple
    valid_point_counts: tuple


def normalize_target_pointcloud_config(config):
    """Return a validated config with stable defaults."""
    if config is True:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("target_pointcloud must be a bool or dict")
    normalized = dict(DEFAULT_POINTCLOUD_CONFIG)
    normalized.update(config)
    if not normalized["enabled"]:
        return normalized
    for key in ("height", "width", "num_points"):
        normalized[key] = int(normalized[key])
        if normalized[key] <= 0:
            raise ValueError("{} must be positive".format(key))
    for key in ("obs_key", "camera_name", "target_object"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise ValueError("{} must be a non-empty string".format(key))
    camera_names = normalized.get("camera_names")
    if camera_names is None:
        camera_names = [normalized["camera_name"]]
    elif isinstance(camera_names, str):
        camera_names = [camera_names]
    if not isinstance(camera_names, (list, tuple)) or not camera_names or not all(
        isinstance(name, str) and name for name in camera_names
    ):
        raise ValueError("camera_names must be null, a string, or non-empty strings")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("camera_names must not contain duplicates")
    normalized["camera_names"] = list(camera_names)
    normalized["camera_name"] = camera_names[0]
    normalized["include_visual_goal"] = bool(normalized["include_visual_goal"])
    goal_objects = normalized.get("goal_objects")
    if goal_objects is not None:
        if isinstance(goal_objects, str):
            goal_objects = [goal_objects]
        if not isinstance(goal_objects, (list, tuple)) or not all(
            isinstance(name, str) and name for name in goal_objects
        ):
            raise ValueError("goal_objects must be null, a string, or non-empty strings")
        goal_objects = list(goal_objects)
    normalized["goal_objects"] = goal_objects
    if normalized["padding_mode"] not in ("repeat", "zero"):
        raise ValueError("padding_mode must be 'repeat' or 'zero'")
    max_geom_distance = normalized.get("max_geom_distance")
    if max_geom_distance is not None:
        max_geom_distance = float(max_geom_distance)
        if max_geom_distance <= 0:
            raise ValueError("max_geom_distance must be positive or null")
    normalized["max_geom_distance"] = max_geom_distance
    return normalized


def get_target_and_goal_geom_ids(
    raw_env,
    target_object="Can",
    include_visual_goal=True,
    goal_objects=None,
):
    """
    Select exactly ``<target>_*`` and optionally ``Visual<target>_*`` geoms.

    Prefix matching is deliberate: it includes both contact and visual geoms
    for the physical object, while avoiding similarly named distractors.
    """
    prefixes = ["{}_".format(target_object)]
    if include_visual_goal:
        prefixes.append("Visual{}_".format(target_object))
    if goal_objects is not None:
        if isinstance(goal_objects, str):
            goal_objects = [goal_objects]
        prefixes.extend("{}_".format(name) for name in goal_objects)

    geom_ids = []
    for geom_id in range(raw_env.sim.model.ngeom):
        geom_name = raw_env.sim.model.geom_id2name(geom_id)
        if geom_name is not None and any(geom_name.startswith(prefix) for prefix in prefixes):
            geom_ids.append(geom_id)
    if not geom_ids:
        raise ValueError(
            "Could not find target geoms for {!r} (include_visual_goal={})".format(
                target_object, include_visual_goal
            )
        )
    return tuple(sorted(set(geom_ids)))


def unproject_depth_pixels_to_world(depth_m, pixels_rc, intrinsic, camera_pose):
    """Unproject top-left-origin image pixels with metric depth to world XYZ."""
    depth_m = np.asarray(depth_m)
    pixels_rc = np.asarray(pixels_rc)
    intrinsic = np.asarray(intrinsic)
    camera_pose = np.asarray(camera_pose)
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    if pixels_rc.ndim != 2 or pixels_rc.shape[1] != 2:
        raise ValueError("pixels_rc must have shape (N, 2)")
    if intrinsic.shape != (3, 3) or camera_pose.shape != (4, 4):
        raise ValueError("intrinsic and camera_pose must have shapes (3,3) and (4,4)")
    if len(pixels_rc) == 0:
        return np.empty((0, 3), dtype=np.float32)

    rows = pixels_rc[:, 0]
    cols = pixels_rc[:, 1]
    z = depth_m[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    world_points = camera_points @ camera_pose.T
    return world_points[:, :3].astype(np.float32)


def deterministic_farthest_point_sample(
    points,
    num_points,
    padding_mode="repeat",
):
    """
    Deterministically FPS to ``num_points`` and apply the requested fallback
    padding if the visible cloud is undersized.

    The first point is the input point farthest from the centroid. Stable
    ``argmax`` tie-breaking and row-major pixel enumeration make extraction
    reproducible for a fixed simulator state.
    """
    points = np.asarray(points, dtype=np.float32)
    num_points = int(num_points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud")

    if padding_mode not in ("repeat", "zero"):
        raise ValueError("padding_mode must be 'repeat' or 'zero'")
    if len(points) < num_points:
        if padding_mode == "zero":
            padding = np.zeros(
                (num_points - len(points), 3),
                dtype=points.dtype,
            )
            points = np.concatenate((points, padding), axis=0)
        else:
            repeats = (num_points + len(points) - 1) // len(points)
            return np.tile(points, (repeats, 1))[:num_points].copy()
    if len(points) == num_points:
        return points.copy()

    centroid = points.astype(np.float64).mean(axis=0)
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    min_sq_dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for index in range(1, num_points):
        selected[index] = int(np.argmax(min_sq_dist))
        candidate_sq_dist = np.sum((points - points[selected[index]]) ** 2, axis=1)
        np.minimum(min_sq_dist, candidate_sq_dist, out=min_sq_dist)
    return points[selected].copy()


def fuse_and_sample_pointcloud_views(
    pointclouds,
    num_points,
    padding_mode="repeat",
):
    """Fuse non-empty world-frame views, then deterministically sample once."""

    validated = []
    for index, points in enumerate(pointclouds):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "pointclouds[{}] must have shape (N, 3)".format(index)
            )
        if len(points):
            validated.append(points)
    if not validated:
        raise ValueError("Cannot fuse only empty point-cloud views")
    fused = np.concatenate(validated, axis=0)
    return deterministic_farthest_point_sample(
        fused,
        num_points,
        padding_mode=padding_mode,
    )


def render_target_pointcloud(raw_env, config=None, target_geom_ids=None, return_details=False):
    """Render, fuse, and sample a fixed-size world-frame point cloud."""
    config = normalize_target_pointcloud_config(config or {})
    if not config["enabled"]:
        raise ValueError("Cannot render a disabled target_pointcloud provider")
    if target_geom_ids is None:
        target_geom_ids = get_target_and_goal_geom_ids(
            raw_env=raw_env,
            target_object=config["target_object"],
            include_visual_goal=config["include_visual_goal"],
            goal_objects=config["goal_objects"],
        )

    valid_by_camera = []
    valid_point_counts = []
    target_geom_ids_array = np.asarray(target_geom_ids)
    for camera_name in config["camera_names"]:
        segmentation, normalized_depth = raw_env.sim.render(
            camera_name=camera_name,
            height=config["height"],
            width=config["width"],
            depth=True,
            segmentation=True,
        )
        # MuJoCo renders bottom-up; robomimic observations use a top-left origin.
        segmentation = segmentation[::-1]
        normalized_depth = normalized_depth[::-1]
        geom_ids = segmentation[..., 1]
        mask = np.isin(geom_ids, target_geom_ids_array)
        pixels_rc = np.argwhere(mask)
        if len(pixels_rc) == 0:
            valid_by_camera.append(np.empty((0, 3), dtype=np.float32))
            valid_point_counts.append(0)
            continue

        depth_m = CameraUtils.get_real_depth_map(raw_env.sim, normalized_depth)
        intrinsic = CameraUtils.get_camera_intrinsic_matrix(
            raw_env.sim,
            camera_name=camera_name,
            camera_height=config["height"],
            camera_width=config["width"],
        )
        camera_pose = CameraUtils.get_camera_extrinsic_matrix(
            raw_env.sim, camera_name=camera_name
        )
        valid_points = unproject_depth_pixels_to_world(
            depth_m=depth_m,
            pixels_rc=pixels_rc,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
        )
        finite = np.isfinite(valid_points).all(axis=1)
        if config["max_geom_distance"] is not None:
            pixel_geom_ids = geom_ids[mask]
            geom_centers = np.asarray(
                [raw_env.sim.data.geom_xpos[int(geom_id)] for geom_id in pixel_geom_ids]
            )
            finite &= (
                np.linalg.norm(
                    valid_points.astype(np.float64) - geom_centers,
                    axis=1,
                )
                <= config["max_geom_distance"]
            )
        valid_points = valid_points[finite]
        valid_by_camera.append(valid_points)
        valid_point_counts.append(len(valid_points))

    if not any(valid_point_counts):
        raise RuntimeError(
            "No target pixels rendered for geoms {} from cameras {}".format(
                target_geom_ids,
                config["camera_names"],
            )
        )
    valid_points = np.concatenate(
        [points for points in valid_by_camera if len(points)],
        axis=0,
    )
    points = fuse_and_sample_pointcloud_views(
        valid_by_camera,
        config["num_points"],
        padding_mode=config["padding_mode"],
    )
    if return_details:
        return TargetPointCloudRender(
            points=points,
            valid_points=valid_points,
            target_geom_ids=tuple(target_geom_ids),
            camera_names=tuple(config["camera_names"]),
            valid_point_counts=tuple(valid_point_counts),
        )
    return points
