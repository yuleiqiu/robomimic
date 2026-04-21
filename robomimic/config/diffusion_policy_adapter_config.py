"""
Config for adapter-based Diffusion Policy fine-tuning.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class DiffusionPolicyAdapterConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_adapter"

    def train_config(self):
        super(DiffusionPolicyAdapterConfig, self).train_config()

        # keep the original DP dataset path semantics for the single-domain data
        self.train.multi_data = None

        # optional overrides for the few-shot multi-domain loader
        self.train.multi_batch_size = None
        self.train.multi_num_data_workers = None
        self.train.multi_hdf5_filter_key = None
        self.train.multi_hdf5_validation_filter_key = None

    def algo_config(self):
        super(DiffusionPolicyAdapterConfig, self).algo_config()

        # adapter optimizer mirrors the default diffusion-policy optimizer
        self.algo.optim_params.adapter.optimizer_type = "adamw"
        self.algo.optim_params.adapter.learning_rate.initial = 1e-4
        self.algo.optim_params.adapter.learning_rate.decay_factor = 0.1
        self.algo.optim_params.adapter.learning_rate.step_every_batch = True
        self.algo.optim_params.adapter.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.adapter.learning_rate.num_cycles = 0.5
        self.algo.optim_params.adapter.learning_rate.warmup_steps = 500
        self.algo.optim_params.adapter.learning_rate.epoch_schedule = []
        self.algo.optim_params.adapter.learning_rate.do_not_lock_keys()
        self.algo.optim_params.adapter.regularization.L2 = 1e-6

        self.algo.adapter.hidden_dim = None
        self.algo.adapter.lambda_ret = 1.0
        self.algo.adapter.lambda_sparse = 1e-2
        self.algo.adapter.alpha = 1e-3

        # initialize the frozen base policy from a pretrained single-domain DP
        self.algo.adapter.base_ckpt_path = None
        self.algo.adapter.use_base_ema = True
