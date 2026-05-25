"""
Config for outcome-weighted Diffusion Policy.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DiffusionPolicyWeightedConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_weighted"

    def train_config(self):
        """
        Extend Diffusion Policy training config with outcome-weighted loss settings.
        """
        super(DiffusionPolicyWeightedConfig, self).train_config()

        self.train.weighted_loss.enabled = True
        self.train.weighted_loss.human_demo_weight = 1.0
        self.train.weighted_loss.success_rollout_weight = 1.0
        self.train.weighted_loss.failed_rollout_weight = 0.1
        self.train.weighted_loss.prefailure_weight = 0.3
        self.train.weighted_loss.postfailure_weight = 0.0
        self.train.weighted_loss.failure_window = 10
        self.train.weighted_loss.eps = 1e-8
