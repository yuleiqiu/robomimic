"""LAN-O3DP inference using the paper's full clean-estimate gradient."""

from dataclasses import replace

import torch

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.lan_o3dp import LanO3DPUNet
from robomimic.utils.paper_lan_guidance_utils import (
    closest_obstacle_point,
    paper_guidance_gradient,
)


@register_algo_factory_func("paper_guided_lan_o3dp")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return PaperGuidedLanO3DPUNet, {}
    if algo_config.transformer.enabled:
        raise NotImplementedError()
    raise RuntimeError()


class PaperGuidedLanO3DPUNet(LanO3DPUNet):
    """Apply ``A_{k-1} -= rho * grad_{A_k} D(A_{0|k}, C_ob)``."""

    def _create_networks(self):
        super(PaperGuidedLanO3DPUNet, self)._create_networks()
        self.guidance_context = None
        self.guidance_diagnostics = []

    def reset(self):
        super(PaperGuidedLanO3DPUNet, self).reset()
        self.guidance_context = None
        self.guidance_diagnostics = []

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
        if context is not None and context.enabled and context.guidance_scale > 0:
            # Algorithm 1 chooses C_ob once, before the reverse process. Keep
            # only that point so no denoising iterate can change the choice.
            fixed_point = closest_obstacle_point(
                context.current_eef_pos,
                torch.as_tensor(
                    context.obstacle_points,
                    device=self.device,
                    dtype=noisy_action.dtype,
                ),
                xy_only=context.xy_only,
            )
            context = replace(
                context,
                obstacle_points=fixed_point.detach().reshape(1, 3),
            )
        for timestep in self.noise_scheduler.timesteps:
            guidance_enabled = (
                context is not None
                and context.enabled
                and context.guidance_scale > 0
            )
            if guidance_enabled:
                with torch.enable_grad():
                    current = noisy_action.detach().requires_grad_(True)
                    model_output = nets["policy"]["noise_pred_net"](
                        sample=current,
                        timestep=timestep,
                        global_cond=observation_condition,
                    )
                    update, diagnostics = paper_guidance_gradient(
                        noisy_action=current,
                        model_output=model_output,
                        timestep=timestep,
                        scheduler=self.noise_scheduler,
                        context=context,
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
        return noisy_action[:, start : start + action_horizon]
