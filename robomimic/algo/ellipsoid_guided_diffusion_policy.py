"""Predicted-clean swept-ellipsoid guidance for delta-EEF Diffusion Policy."""

import torch

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet
from robomimic.utils.ellipsoid_guidance_utils import ellipsoid_guidance_gradient


@register_algo_factory_func("ellipsoid_guided_diffusion_policy")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return EllipsoidGuidedDiffusionPolicyUNet, {}
    if algo_config.transformer.enabled:
        raise NotImplementedError()
    raise RuntimeError()


class EllipsoidGuidedDiffusionPolicyUNet(DiffusionPolicyUNet):
    """Guide predicted executable delta-EEF poses using rigid ellipsoids."""

    def _create_networks(self):
        super()._create_networks()
        self.guidance_context = None
        self.guidance_diagnostics = []
        self.last_predicted_action_chunk = None

    def reset(self):
        super().reset()
        self.guidance_context = None
        self.guidance_diagnostics = []
        self.last_predicted_action_chunk = None

    def set_guidance_context(self, context):
        self.guidance_context = context

    def clear_guidance_context(self):
        self.guidance_context = None

    def get_guidance_diagnostics(self):
        return [item.to_dict() for item in self.guidance_diagnostics]

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        observation_horizon = self.algo_config.horizon.observation_horizon
        action_horizon = self.algo_config.horizon.action_horizon
        prediction_horizon = self.algo_config.horizon.prediction_horizon
        if self.algo_config.ddpm.enabled:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("DDPM or DDIM must be enabled")

        nets = self.ema.averaged_model if self.ema is not None else self.nets
        inputs = {"obs": obs_dict, "goal": goal_dict}
        for key in self.obs_shapes:
            if inputs["obs"][key].ndim - 1 == len(self.obs_shapes[key]):
                inputs["obs"][key] = inputs["obs"][key].unsqueeze(1)
            assert inputs["obs"][key].ndim - 2 == len(self.obs_shapes[key])
        with torch.no_grad():
            features = TensorUtils.time_distributed(
                inputs,
                nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )
            observation_condition = features.flatten(start_dim=1)
            noisy_action = torch.randn(
                (features.shape[0], prediction_horizon, self.ac_dim),
                device=self.device,
            )

        self.noise_scheduler.set_timesteps(num_inference_timesteps)
        self.guidance_diagnostics = []
        context = self.guidance_context
        for timestep in self.noise_scheduler.timesteps:
            enabled = (
                context is not None
                and context.enabled
                and context.guidance_scale > 0
            )
            if enabled:
                with torch.enable_grad():
                    current = noisy_action.detach().requires_grad_(True)
                    model_output = nets["policy"]["noise_pred_net"](
                        sample=current,
                        timestep=timestep,
                        global_cond=observation_condition,
                    )
                    update, diagnostics = ellipsoid_guidance_gradient(
                        noisy_action=current,
                        model_output=model_output,
                        timestep=timestep,
                        scheduler=self.noise_scheduler,
                        context=context,
                        observation_horizon=observation_horizon,
                        action_horizon=action_horizon,
                    )
                with torch.no_grad():
                    reverse = self.noise_scheduler.step(
                        model_output=model_output.detach(),
                        timestep=timestep,
                        sample=current.detach(),
                    ).prev_sample
                    noisy_action = reverse - update
                self.guidance_diagnostics.append(diagnostics)
            else:
                with torch.no_grad():
                    model_output = nets["policy"]["noise_pred_net"](
                        sample=noisy_action,
                        timestep=timestep,
                        global_cond=observation_condition,
                    )
                    noisy_action = self.noise_scheduler.step(
                        model_output=model_output,
                        timestep=timestep,
                        sample=noisy_action,
                    ).prev_sample

        start = observation_horizon - 1
        action = noisy_action[:, start : start + action_horizon]
        self.last_predicted_action_chunk = action.detach().cpu()
        return action
