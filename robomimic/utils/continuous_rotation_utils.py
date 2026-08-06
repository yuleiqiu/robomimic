"""Continuous quaternion-to-rotation-vector conversion utilities.

Quaternion signs are physically equivalent, but ``quat2axisangle`` maps the
two signs to different rotation-vector branches.  These helpers lift a
quaternion sequence onto one sign-continuous branch before conversion.  This
keeps absolute rotation vectors continuous when a trajectory crosses the
principal angle boundary at pi.
"""

import numpy as np
from robosuite.utils import transform_utils as T


ROTATION_VECTOR_MODES = ("principal", "continuous")


def normalize_quaternion_xyzw(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(
            "Expected an xyzw quaternion with shape (4,), got {}".format(
                quaternion.shape
            )
        )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must have a finite positive norm")
    return quaternion / norm


def align_quaternion_sign(quaternion, reference_quaternion):
    """Return the physically equivalent sign closest to ``reference``."""

    quaternion = normalize_quaternion_xyzw(quaternion)
    reference = normalize_quaternion_xyzw(reference_quaternion)
    if float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    return quaternion


def continuous_quaternion_sequence_to_rotation_vectors(
    quaternions,
    reference_quaternion,
):
    """Convert an xyzw quaternion sequence to one continuous rotvec branch."""

    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            "Expected quaternion sequence shape (T, 4), got {}".format(
                quaternions.shape
            )
        )
    previous = normalize_quaternion_xyzw(reference_quaternion)
    rotation_vectors = np.empty((len(quaternions), 3), dtype=np.float64)
    for index, quaternion in enumerate(quaternions):
        aligned = align_quaternion_sign(quaternion, previous)
        rotation_vectors[index] = T.quat2axisangle(aligned.copy())
        previous = aligned
    return rotation_vectors


class ContinuousRotationVectorState:
    """Stateful online converter using the same episode reference as offline."""

    def __init__(self, reference_quaternion):
        self.reference_quaternion = normalize_quaternion_xyzw(
            reference_quaternion
        )
        self.previous_quaternion = None

    def reset(self):
        self.previous_quaternion = None

    def convert(self, quaternion):
        reference = (
            self.reference_quaternion
            if self.previous_quaternion is None
            else self.previous_quaternion
        )
        aligned = align_quaternion_sign(quaternion, reference)
        self.previous_quaternion = aligned
        return T.quat2axisangle(aligned.copy()).astype(np.float32)
