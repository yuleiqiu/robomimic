"""
Config for Diffusion Policy with agentview reference-image gating.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DiffusionPolicyRefGatingConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_ref_gating"

    def train_config(self):
        super(DiffusionPolicyRefGatingConfig, self).train_config()
        self.train.output_dir = "../{}_trained_models".format(self.algo_name)

    def algo_config(self):
        super(DiffusionPolicyRefGatingConfig, self).algo_config()

        self.algo.reference_gating.enabled = True
        self.algo.reference_gating.reference_obs_key = "agentview_image"
        self.algo.reference_gating.reference_bank_path = None
        self.algo.reference_gating.max_references = 16
        self.algo.reference_gating.similarity_reduce = "max"
        self.algo.reference_gating.reference_reduce = "mean"
        self.algo.reference_gating.heatmap_activation = "sigmoid"
        self.algo.reference_gating.heatmap_scale = 10.0
        self.algo.reference_gating.debug_store_heatmap = False
        self.algo.reference_gating.do_not_lock_keys()
