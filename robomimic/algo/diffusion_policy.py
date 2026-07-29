"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
import time
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.sdp_utils import sample_desired_action_chunks


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _obs_encoder_feature_activation(self):
        """Activation applied by robomimic after each observation core."""

        return nn.ReLU

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=self._obs_encoder_feature_activation(),
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=list(self.algo_config.unet.down_dims),
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
            cond_predict_scale=self.algo_config.unet.get("cond_predict_scale", True),
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.sdp_config = self.algo_config.get("sdp", None)
        self.sdp_enabled = bool(
            self.sdp_config is not None
            and self.sdp_config.get("enabled", False)
        )
        self.sdp_target_scheduler = None
        if self.sdp_enabled:
            if self.sdp_config.start_timestep >= (
                noise_scheduler.config.num_train_timesteps
            ):
                raise ValueError(
                    "SDP start_timestep must be smaller than the training "
                    "diffusion horizon"
                )
            self.sdp_target_scheduler = DDPMScheduler(
                num_train_timesteps=(
                    noise_scheduler.config.num_train_timesteps
                ),
                beta_schedule=noise_scheduler.config.beta_schedule,
                clip_sample=noise_scheduler.config.clip_sample,
                prediction_type=noise_scheduler.config.prediction_type,
            )
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        if "negative_actions" in batch:
            input_batch["negative_actions"] = batch[
                "negative_actions"
            ][:, :Tp, :]
        if "is_paired_correction" in batch:
            input_batch["is_paired_correction"] = batch[
                "is_paired_correction"
            ]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            action_tensors = [input_batch["actions"]]
            if self.sdp_enabled:
                if "negative_actions" not in input_batch:
                    raise ValueError(
                        "SDP is enabled but negative_actions are absent"
                    )
                action_tensors.append(input_batch["negative_actions"])
            for action_tensor in action_tensors:
                in_range = (-1 <= action_tensor) & (action_tensor <= 1)
                all_in_range = torch.all(in_range).item()
                if not all_in_range:
                    raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _prepare_sdp_training_targets(self, batch, obs_cond):
        """Replace correction labels with policy-relative desired-set samples."""

        if "negative_actions" not in batch:
            raise ValueError(
                "SDP training requires paired negative actions in the batch"
            )
        if "is_paired_correction" not in batch:
            raise ValueError(
                "SDP training requires is_paired_correction labels"
            )

        actions = batch["actions"]
        correction_mask = batch["is_paired_correction"].reshape(-1) > 0.5
        correction_indices = torch.nonzero(
            correction_mask, as_tuple=False
        ).flatten()
        clean_indices = torch.nonzero(
            ~correction_mask, as_tuple=False
        ).flatten()
        original_batch_size = actions.shape[0]

        if correction_indices.numel() == 0:
            return (
                actions,
                obs_cond,
                torch.arange(
                    original_batch_size, device=self.device, dtype=torch.long
                ),
                {
                    "correction_pair_count": torch.zeros(
                        (), device=self.device
                    ),
                    "desired_target_count": torch.zeros(
                        (), device=self.device
                    ),
                    "replacement_rate": torch.zeros(
                        (), device=self.device
                    ),
                    "final_preprojection_retention_rate": torch.zeros(
                        (), device=self.device
                    ),
                    "mean_set_radius": torch.zeros(
                        (), device=self.device
                    ),
                    "maximum_set_radius": torch.zeros(
                        (), device=self.device
                    ),
                    "mean_target_distance": torch.zeros(
                        (), device=self.device
                    ),
                    "maximum_target_distance": torch.zeros(
                        (), device=self.device
                    ),
                    "nonpositive_timestep_rate": torch.zeros(
                        (), device=self.device
                    ),
                    "target_generation_seconds": torch.zeros(
                        (), device=self.device
                    ),
                },
            )

        start_time = time.perf_counter()
        desired_targets, sampler_statistics = sample_desired_action_chunks(
            noise_pred_net=self.nets["policy"]["noise_pred_net"],
            scheduler=self.sdp_target_scheduler,
            observation_condition=obs_cond[correction_indices].detach(),
            positive_actions=actions[correction_indices].detach(),
            negative_actions=batch["negative_actions"][
                correction_indices
            ].detach(),
            radius_ratio=self.sdp_config.radius_ratio,
            num_samples=self.sdp_config.num_samples,
            start_timestep=self.sdp_config.start_timestep,
            initialization=self.sdp_config.initialization,
            tolerance=self.sdp_config.constraint_tolerance,
        )
        target_generation_seconds = time.perf_counter() - start_time
        num_desired_samples = desired_targets.shape[1]

        training_actions = torch.cat(
            [
                actions[clean_indices],
                desired_targets.flatten(0, 1),
            ],
            dim=0,
        )
        training_condition = torch.cat(
            [
                obs_cond[clean_indices],
                obs_cond[correction_indices].repeat_interleave(
                    num_desired_samples, dim=0
                ),
            ],
            dim=0,
        )
        original_sample_indices = torch.cat(
            [
                clean_indices,
                correction_indices.repeat_interleave(
                    num_desired_samples
                ),
            ],
            dim=0,
        )
        statistics = {
            **sampler_statistics,
            "correction_pair_count": correction_indices.numel()
            * torch.ones((), device=self.device),
            "desired_target_count": desired_targets.shape[0]
            * desired_targets.shape[1]
            * torch.ones((), device=self.device),
            "target_generation_seconds": torch.tensor(
                target_generation_seconds,
                device=self.device,
            ),
        }
        return (
            training_actions,
            training_condition,
            original_sample_indices,
            statistics,
        )

    @staticmethod
    def _grouped_diffusion_loss(
        noise_prediction,
        noise,
        original_sample_indices,
        original_batch_size,
    ):
        """
        Average desired targets within each original sample before the batch.

        This prevents ``N`` SDP targets from multiplying the correction
        exposure relative to one clean BC target.
        """

        per_target_loss = F.mse_loss(
            noise_prediction,
            noise,
            reduction="none",
        ).flatten(start_dim=1).mean(dim=1)
        per_sample_loss = torch.zeros(
            original_batch_size,
            device=per_target_loss.device,
            dtype=per_target_loss.dtype,
        )
        per_sample_count = torch.zeros_like(per_sample_loss)
        per_sample_loss.scatter_add_(
            0, original_sample_indices, per_target_loss
        )
        per_sample_count.scatter_add_(
            0,
            original_sample_indices,
            torch.ones_like(per_target_loss),
        )
        if torch.any(per_sample_count == 0):
            raise RuntimeError("An original training sample has no target")
        return torch.mean(per_sample_loss / per_sample_count)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)

            original_batch_size = actions.shape[0]
            original_sample_indices = torch.arange(
                original_batch_size,
                device=self.device,
                dtype=torch.long,
            )
            sdp_statistics = None
            if self.sdp_enabled:
                (
                    actions,
                    obs_cond,
                    original_sample_indices,
                    sdp_statistics,
                ) = self._prepare_sdp_training_targets(batch, obs_cond)
            B = actions.shape[0]
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            if self.sdp_enabled:
                loss = self._grouped_diffusion_loss(
                    noise_prediction=noise_pred,
                    noise=noise,
                    original_sample_indices=original_sample_indices,
                    original_batch_size=original_batch_size,
                )
            else:
                loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)
            if sdp_statistics is not None:
                info["sdp"] = TensorUtils.detach(sdp_statistics)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "sdp" in info:
            for key, value in info["sdp"].items():
                log["SDP/{}".format(key)] = value.item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            step_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            )
            naction = self._process_reverse_step(
                step_output=step_output,
                timestep=k,
                observation_horizon=To,
                action_horizon=Ta,
            )

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def _process_reverse_step(
        self,
        step_output,
        timestep,
        observation_horizon,
        action_horizon,
    ):
        """Hook for inference variants that modify a scheduler reverse step."""

        return step_output.prev_sample

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
