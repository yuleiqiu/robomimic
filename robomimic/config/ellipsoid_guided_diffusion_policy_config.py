"""Config registration for swept-ellipsoid Diffusion Policy guidance."""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class EllipsoidGuidedDiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "ellipsoid_guided_diffusion_policy"
