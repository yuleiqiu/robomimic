"""Config registration for generic point-trajectory guidance."""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class PointGuidedDiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "point_guided_diffusion_policy"
