"""
Adapter-based Diffusion Policy fine-tuning with a frozen base policy.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import PolicyAlgo, register_algo_factory_func
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet, replace_bn_with_gn


class CondAdapter(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(CondAdapter, self).__init__()
        self.delta = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.gate = nn.Linear(cond_dim, 1)

    def forward(self, cond):
        delta_h = self.delta(cond)
        gate = torch.sigmoid(self.gate(cond))
        cond_tilde = cond + gate * delta_h
        return cond_tilde, gate, delta_h


@register_algo_factory_func("diffusion_policy_adapter")
def algo_config_to_class(algo_config):
    return DiffusionPolicyAdapter, {}


class DiffusionPolicyAdapter(DiffusionPolicyUNet):
    def _create_networks(self):
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)

        obs_dim = obs_encoder.output_shape()[0]
        cond_dim = obs_dim * self.algo_config.horizon.observation_horizon
        hidden_dim = self.algo_config.adapter.hidden_dim or cond_dim

        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=cond_dim,
        )
        adapter = CondAdapter(cond_dim=cond_dim, hidden_dim=hidden_dim)

        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net,
            }),
            "adapter": adapter,
        })
        nets = nets.float().to(self.device)

        for param in nets["policy"].parameters():
            param.requires_grad = False

        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type,
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type,
            )
        else:
            raise RuntimeError()

        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

    def _encode_obs(self, obs_dict, goal_dict=None, nets=None):
        if nets is None:
            nets = self.nets
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict,
        }
        for key in self.obs_shapes:
            assert inputs["obs"][key].ndim - 2 == len(self.obs_shapes[key])
        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert obs_features.ndim == 3
        return obs_features.flatten(start_dim=1)

    def _forward_adapted_from_cond(self, obs_cond, noisy_actions, timesteps, nets=None):
        if nets is None:
            nets = self.nets
        cond_tilde, gate, delta_h = nets["adapter"](obs_cond)
        noise_pred = nets["policy"]["noise_pred_net"](
            noisy_actions,
            timesteps,
            global_cond=cond_tilde,
        )
        return noise_pred, cond_tilde, gate, delta_h

    def _forward_base_from_cond(self, obs_cond, noisy_actions, timesteps, nets=None):
        if nets is None:
            nets = self.nets
        with torch.no_grad():
            noise_pred = nets["policy"]["noise_pred_net"](
                noisy_actions,
                timesteps,
                global_cond=obs_cond.detach(),
            )
        return noise_pred

    def _sample_noisy_actions(self, actions):
        batch_size = actions.shape[0]
        noise = torch.randn(actions.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        return noisy_actions, noise, timesteps

    def _compute_domain_terms(self, batch, teacher_batch=False):
        actions = batch["actions"]
        obs_cond = self._encode_obs(batch["obs"], batch["goal_obs"])
        noisy_actions, noise, timesteps = self._sample_noisy_actions(actions)
        noise_pred, cond_tilde, gate, delta_h = self._forward_adapted_from_cond(
            obs_cond=obs_cond,
            noisy_actions=noisy_actions,
            timesteps=timesteps,
        )

        outputs = {
            "obs_cond": obs_cond,
            "cond_tilde": cond_tilde,
            "gate": gate,
            "delta_h": delta_h,
            "noise_pred": noise_pred,
            "noise": noise,
            "timesteps": timesteps,
            "noisy_actions": noisy_actions,
        }
        if teacher_batch:
            outputs["noise_base"] = self._forward_base_from_cond(
                obs_cond=obs_cond,
                noisy_actions=noisy_actions,
                timesteps=timesteps,
            )
        return outputs

    def train_on_batch(self, batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)

            multi_outputs = self._compute_domain_terms(batch["multi"], teacher_batch=False)
            single_outputs = self._compute_domain_terms(batch["single"], teacher_batch=True)

            loss_multi = F.mse_loss(multi_outputs["noise_pred"], multi_outputs["noise"])
            loss_ret = F.mse_loss(single_outputs["noise_pred"], single_outputs["noise_base"])

            alpha = self.algo_config.adapter.alpha
            loss_sparse = (
                multi_outputs["gate"].mean()
                + single_outputs["gate"].mean()
                + alpha * (
                    multi_outputs["delta_h"].pow(2).mean()
                    + single_outputs["delta_h"].pow(2).mean()
                )
            )

            loss = (
                loss_multi
                + self.algo_config.adapter.lambda_ret * loss_ret
                + self.algo_config.adapter.lambda_sparse * loss_sparse
            )

            losses = {
                "total_loss": loss,
                "multi_loss": loss_multi,
                "ret_loss": loss_ret,
                "sparse_loss": loss_sparse,
                "gate_multi": multi_outputs["gate"].mean(),
                "gate_single": single_outputs["gate"].mean(),
                "delta_multi_l2": multi_outputs["delta_h"].pow(2).mean(),
                "delta_single_l2": single_outputs["delta_h"].pow(2).mean(),
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets["adapter"],
                    optim=self.optimizers["adapter"],
                    loss=loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info["adapter_grad_norms"] = grad_norms

        return info

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Loss_Multi"] = info["losses"]["multi_loss"].item()
        log["Loss_Ret"] = info["losses"]["ret_loss"].item()
        log["Loss_Sparse"] = info["losses"]["sparse_loss"].item()
        log["Gate_Multi"] = info["losses"]["gate_multi"].item()
        log["Gate_Single"] = info["losses"]["gate_single"].item()
        log["Delta_Multi_L2"] = info["losses"]["delta_multi_l2"].item()
        log["Delta_Single_L2"] = info["losses"]["delta_single_l2"].item()
        if "adapter_grad_norms" in info:
            log["Adapter_Grad_Norms"] = info["adapter_grad_norms"]
        return log

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        to = self.algo_config.horizon.observation_horizon
        ta = self.algo_config.horizon.action_horizon
        tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        inputs = {
            "obs": obs_dict,
            "goal": goal_dict,
        }
        for key in self.obs_shapes:
            if inputs["obs"][key].ndim - 1 == len(self.obs_shapes[key]):
                inputs["obs"][key] = inputs["obs"][key].unsqueeze(1)
            assert inputs["obs"][key].ndim - 2 == len(self.obs_shapes[key])

        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert obs_features.ndim == 3
        batch_size = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)
        obs_cond, _, _ = nets["adapter"](obs_cond)

        noisy_action = torch.randn((batch_size, tp, action_dim), device=self.device)
        naction = noisy_action

        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for timestep in self.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=timestep,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=naction,
            ).prev_sample

        start = to - 1
        end = start + ta
        return naction[:, start:end]

    @staticmethod
    def _strip_prefix(state_dict, prefix):
        prefix_with_sep = prefix + "."
        stripped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(prefix_with_sep):
                stripped[key[len(prefix_with_sep):]] = value
        return stripped

    @staticmethod
    def _looks_like_base_policy_state(model_dict):
        nets_state = model_dict.get("nets", {})
        if not isinstance(nets_state, dict):
            return False
        has_policy = any(key.startswith("policy.") for key in nets_state)
        has_adapter = any(key.startswith("adapter.") for key in nets_state)
        return has_policy and (not has_adapter)

    def load_base_policy_from_model_dict(self, model_dict):
        nets_state = model_dict["nets"]
        ema_state = model_dict.get("ema", None)
        source_state = nets_state
        if self.algo_config.adapter.use_base_ema and ema_state is not None:
            source_state = ema_state

        policy_state = self._strip_prefix(source_state, "policy")
        self.nets["policy"].load_state_dict(policy_state, strict=True)

        if self.ema is not None:
            self.ema.averaged_model["policy"].load_state_dict(policy_state, strict=True)

        for param in self.nets["policy"].parameters():
            param.requires_grad = False

    def load_base_policy_from_checkpoint(self, ckpt_dict):
        model_dict = ckpt_dict["model"] if "model" in ckpt_dict else ckpt_dict
        self.load_base_policy_from_model_dict(model_dict)

    def deserialize(self, model_dict, load_optimizers=False):
        try:
            self.nets.load_state_dict(model_dict["nets"])
        except RuntimeError:
            if self._looks_like_base_policy_state(model_dict):
                if load_optimizers:
                    raise RuntimeError("cannot load optimizer state from a base diffusion_policy checkpoint")
                self.load_base_policy_from_model_dict(model_dict)
                return
            raise

        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None and self.ema is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for key in model_dict["optimizers"]:
                if key in self.optimizers:
                    self.optimizers[key].load_state_dict(model_dict["optimizers"][key])
            for key in model_dict["lr_schedulers"]:
                if key in self.lr_schedulers and model_dict["lr_schedulers"][key] is not None:
                    self.lr_schedulers[key].load_state_dict(model_dict["lr_schedulers"][key])
