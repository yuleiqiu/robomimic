"""Registered deployment-time guided variant of Diffusion Policy."""

from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.utils.guided_denoising_utils import apply_guidance_to_reverse_sample


@register_algo_factory_func("guided_diffusion_policy")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return GuidedDiffusionPolicyUNet, {}
    if algo_config.transformer.enabled:
        raise NotImplementedError()
    raise RuntimeError()


@register_algo_factory_func("guided_lan_o3dp")
def lan_algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return GuidedLanO3DPUNet, {}
    if algo_config.transformer.enabled:
        raise NotImplementedError()
    raise RuntimeError()


class GuidedDiffusionPolicyUNet(DiffusionPolicyUNet):
    """Diffusion Policy with an opt-in guided reverse-step hook."""

    def _create_networks(self):
        super(GuidedDiffusionPolicyUNet, self)._create_networks()
        self.guidance_context = None
        self.guidance_diagnostics = []

    def reset(self):
        super(GuidedDiffusionPolicyUNet, self).reset()
        self.guidance_context = None
        self.guidance_diagnostics = []

    def set_guidance_context(self, context):
        """Set deployment context for the next sampled action chunk."""

        self.guidance_context = context

    def clear_guidance_context(self):
        self.guidance_context = None

    def get_guidance_diagnostics(self):
        return [diagnostic.to_dict() for diagnostic in self.guidance_diagnostics]

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        self.guidance_diagnostics = []
        return super(GuidedDiffusionPolicyUNet, self)._get_action_trajectory(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
        )

    def _process_reverse_step(
        self,
        step_output,
        timestep,
        observation_horizon,
        action_horizon,
    ):
        context = self.guidance_context
        if context is None or not context.enabled:
            return step_output.prev_sample
        if not hasattr(step_output, "pred_original_sample") or step_output.pred_original_sample is None:
            raise ValueError(
                "guided_diffusion_policy requires the scheduler to return pred_original_sample"
            )

        guided_sample, diagnostics = apply_guidance_to_reverse_sample(
            predicted_clean_action=step_output.pred_original_sample,
            reverse_sample=step_output.prev_sample,
            timestep=timestep,
            noise_scheduler=self.noise_scheduler,
            context=context,
            observation_horizon=observation_horizon,
            action_horizon=action_horizon,
        )
        if diagnostics is not None:
            self.guidance_diagnostics.append(diagnostics)
        return guided_sample


class GuidedLanO3DPUNet(GuidedDiffusionPolicyUNet):
    """Guided LAN-O3DP variant that preserves its linear encoder fusion."""

    def _obs_encoder_feature_activation(self):
        return None
