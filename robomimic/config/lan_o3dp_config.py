"""Configuration defaults for the LAN-O3DP policy reproduction."""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class LanO3DPConfig(DiffusionPolicyConfig):
    """LAN-aligned point-cloud Diffusion Policy defaults.

    The project intentionally keeps epsilon prediction and the executable
    delta-EEF action interface so the resulting checkpoint remains suitable
    for deployment-time guided denoising.
    """

    ALGO_NAME = "lan_o3dp"

    def algo_config(self):
        super(LanO3DPConfig, self).algo_config()

        self.algo.optim_params.policy.optimizer_kwargs.betas = [0.95, 0.999]
        self.algo.optim_params.policy.optimizer_kwargs.eps = 1e-8

        self.algo.unet.diffusion_step_embed_dim = 128
        self.algo.unet.down_dims = [512, 1024, 2048]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8
        self.algo.unet.cond_predict_scale = True

        self.algo.ddpm.enabled = True
        self.algo.ddpm.num_train_timesteps = 100
        self.algo.ddpm.num_inference_timesteps = 100
        self.algo.ddpm.prediction_type = "epsilon"
        self.algo.ddim.enabled = False
