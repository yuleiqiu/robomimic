"""
Outcome-weighted Diffusion Policy.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if key in config:
        return config[key]
    return default


@register_algo_factory_func("diffusion_policy_weighted")
def algo_config_to_class(algo_config):
    """
    Maps weighted Diffusion Policy config to the implementation class.
    """
    if algo_config.unet.enabled:
        return DiffusionPolicyWeightedUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyWeightedUNet(DiffusionPolicyUNet):
    """
    Diffusion Policy with optional per-sequence outcome-weighted denoising loss.
    """

    @property
    def weighted_loss_config(self):
        train_config = self.global_config.train
        return train_config["weighted_loss"] if "weighted_loss" in train_config else None

    @property
    def weighted_loss_enabled(self):
        return bool(_config_get(self.weighted_loss_config, "enabled", False))

    def process_batch_for_training(self, batch):
        input_batch = super(DiffusionPolicyWeightedUNet, self).process_batch_for_training(batch)
        if "sample_weight" in batch:
            input_batch["sample_weight"] = TensorUtils.to_device(
                TensorUtils.to_float(batch["sample_weight"]),
                self.device,
            )
        return input_batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed(
                inputs,
                self.nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )
            assert obs_features.ndim == 3
            obs_cond = obs_features.flatten(start_dim=1)

            noise = torch.randn(actions.shape, device=self.device)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=self.device,
            ).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions,
                timesteps,
                global_cond=obs_cond,
            )

            loss_per_sample = F.mse_loss(noise_pred, noise, reduction="none")
            loss_per_sample = loss_per_sample.reshape(B, -1).mean(dim=1)
            unweighted_loss = loss_per_sample.mean()

            sample_weight = batch.get("sample_weight", None)
            if self.weighted_loss_enabled:
                if sample_weight is None:
                    raise ValueError(
                        "train.weighted_loss.enabled is true, but batch is missing 'sample_weight'."
                    )
                sample_weight = sample_weight.reshape(B).to(device=self.device, dtype=loss_per_sample.dtype)
                eps = float(_config_get(self.weighted_loss_config, "eps", 1e-8))
                loss = torch.sum(sample_weight * loss_per_sample) / (torch.sum(sample_weight) + eps)
            else:
                loss = unweighted_loss

            losses = OrderedDict()
            losses["l2_loss"] = loss
            losses["unweighted_l2_loss"] = unweighted_loss
            if self.weighted_loss_enabled:
                losses["weighted_l2_loss"] = loss
                losses["mean_sample_weight"] = sample_weight.mean()
                losses["sum_sample_weight"] = sample_weight.sum()
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                if self.ema is not None:
                    self.ema.step(self.nets)

                info.update({"policy_grad_norms": policy_grad_norms})

        return info

    def log_info(self, info):
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Unweighted_L2_Loss"] = info["losses"]["unweighted_l2_loss"].item()
        if "weighted_l2_loss" in info["losses"]:
            log["Weighted_L2_Loss"] = info["losses"]["weighted_l2_loss"].item()
            log["Mean_Sample_Weight"] = info["losses"]["mean_sample_weight"].item()
            log["Sum_Sample_Weight"] = info["losses"]["sum_sample_weight"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
