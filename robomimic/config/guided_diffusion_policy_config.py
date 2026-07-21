"""Config registration for the guided Diffusion Policy inference variant."""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class GuidedDiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "guided_diffusion_policy"
