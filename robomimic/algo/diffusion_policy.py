"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
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
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


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
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
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
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        self.obstacle_guidance_context = None
        self.last_obstacle_guidance_info = None
        self.obstacle_guidance_sample_count = 0
    
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
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
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
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

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
        self.last_obstacle_guidance_info = None
        self.obstacle_guidance_sample_count = 0

    def set_obstacle_guidance_context(self, context=None):
        """
        Set optional inference-time obstacle guidance context for the next action
        trajectory sample. Passing None or an ``enabled=False`` context disables
        guidance without changing the baseline sampler path.
        """
        self.obstacle_guidance_context = context

    def _obstacle_guidance_enabled(self):
        context = self.obstacle_guidance_context
        if context is None:
            return False
        if not context.get("enabled", False):
            return False
        centers = context.get("obstacle_centers_xyz", context.get("obstacle_centers_xy", None))
        radii = context.get("obstacle_radii", None)
        if centers is None or radii is None:
            return False
        if len(centers) == 0 or len(radii) == 0:
            return False
        return context.get("guidance_scale", 0.0) > 0.0

    def _obstacle_guidance_cost(self, action_chunk, horizon=None, return_stats=True):
        """
        Compute obstacle guidance cost for an action chunk in policy action coordinates.
        """
        context = self.obstacle_guidance_context
        if horizon is None:
            horizon = context.get("guidance_horizon", self.algo_config.horizon.action_horizon)
        guidance_mode = context.get("guidance_mode", "xyz_cylinder")
        action_for_cost = ObstacleGuidanceUtils.unnormalize_action_chunk(
            action_chunk=action_chunk,
            action_scale=context.get("action_scale", None),
            action_offset=context.get("action_offset", None),
        )
        if guidance_mode == "xy":
            return ObstacleGuidanceUtils.obstacle_xy_cost(
                action_chunk=action_for_cost,
                current_eef_pos=context["current_eef_pos"],
                obstacle_centers_xy=context["obstacle_centers_xyz"],
                obstacle_radii=context["obstacle_radii"],
                horizon=horizon,
                delta_pos_scale=context.get("delta_pos_scale", 1.0),
                delta_pos_offset=context.get("delta_pos_offset", 0.0),
                return_stats=return_stats,
            )
        if guidance_mode == "xyz_cylinder":
            return ObstacleGuidanceUtils.obstacle_xyz_cylinder_cost(
                action_chunk=action_for_cost,
                current_eef_pos=context["current_eef_pos"],
                obstacle_centers_xyz=context["obstacle_centers_xyz"],
                obstacle_radii=context["obstacle_radii"],
                obstacle_top_z=context["obstacle_top_z"],
                z_clearance=context.get("z_clearance", 0.03),
                horizon=horizon,
                delta_pos_scale=context.get("delta_pos_scale", 1.0),
                delta_pos_offset=context.get("delta_pos_offset", 0.0),
                return_stats=return_stats,
            )
        raise ValueError("Unsupported obstacle guidance mode '{}'".format(guidance_mode))

    def _update_last_obstacle_guidance_info(self, updates):
        if self.last_obstacle_guidance_info is None:
            self.last_obstacle_guidance_info = dict(applied=True)
        self.last_obstacle_guidance_info.update(updates)

    def _guided_scheduler_step(
        self,
        nets,
        naction,
        timestep,
        obs_cond,
        step_index,
        num_steps,
    ):
        """
        Run one reverse diffusion step with optional obstacle cost guidance.
        """
        context = self.obstacle_guidance_context
        naction_in = naction.detach().requires_grad_(True)

        noise_pred = nets["policy"]["noise_pred_net"](
            sample=naction_in,
            timestep=timestep,
            global_cond=obs_cond,
        )
        step_output = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=naction_in,
        )
        x0_hat = ObstacleGuidanceUtils.estimate_clean_action_from_scheduler(
            scheduler=self.noise_scheduler,
            sample=naction_in,
            timestep=timestep,
            model_output=noise_pred,
            step_output=step_output,
        )

        guidance_mode = context.get("guidance_mode", "xyz_cylinder")
        cost, cost_stats = self._obstacle_guidance_cost(
            action_chunk=x0_hat,
            horizon=context.get("guidance_horizon", self.algo_config.horizon.action_horizon),
            return_stats=True,
        )
        rho_t = ObstacleGuidanceUtils.guidance_scale_for_step(
            guidance_scale=context.get("guidance_scale", 0.0),
            schedule=context.get("guidance_schedule", "late"),
            step_index=step_index,
            num_steps=num_steps,
        )
        guided_sample, grad_norm = ObstacleGuidanceUtils.normalized_negative_cost_grad_update(
            update_sample=step_output.prev_sample,
            cost=cost,
            scale=rho_t,
            grad_source=naction_in,
        )

        min_xy_distance = cost_stats["min_xy_distance"]
        min_z_clearance = cost_stats.get("min_z_clearance", None)
        self.last_obstacle_guidance_info = dict(
            applied=True,
            guidance_mode=guidance_mode,
            rho_t=float(rho_t),
            cost=float(cost.detach().cpu().item()),
            min_distance=None if min_xy_distance is None else TensorUtils.to_numpy(min_xy_distance),
            min_xy_distance=None if min_xy_distance is None else TensorUtils.to_numpy(min_xy_distance),
            min_z_clearance=None if min_z_clearance is None else TensorUtils.to_numpy(min_z_clearance),
            grad_norm=None if grad_norm is None else TensorUtils.to_numpy(grad_norm),
            num_obstacles=int(cost_stats["num_obstacles"]),
            obstacle_top_z=context.get("obstacle_top_z", None),
            z_clearance=context.get("z_clearance", None),
            delta_pos_scale=context.get("delta_pos_scale", None),
        )
        return guided_sample

    def _refine_obstacle_guidance_action(self, action):
        """
        Optionally repair the final clean action chunk with post-hoc obstacle
        cost gradient steps before the action queue is populated.
        """
        context = self.obstacle_guidance_context
        if context is None or not context.get("final_collision_refine", False):
            return action

        threshold = context.get("final_collision_cost_threshold", 1e-8)
        num_steps = int(context.get("collision_refine_steps", 5))
        scale = context.get("collision_refine_scale", 0.02)
        refined = action.detach()
        initial_cost = None
        final_cost = None
        final_stats = None
        last_grad_norm = None
        steps_taken = 0

        for step in range(max(num_steps, 0) + 1):
            refined_in = refined.detach().requires_grad_(step < num_steps)
            cost, cost_stats = self._obstacle_guidance_cost(
                action_chunk=refined_in,
                horizon=refined_in.shape[1],
                return_stats=True,
            )
            cost_value = float(cost.detach().cpu().item())
            if initial_cost is None:
                initial_cost = cost_value
            final_cost = cost_value
            final_stats = cost_stats
            if cost_value <= threshold or step == num_steps:
                refined = refined_in.detach()
                break

            refined, last_grad_norm = ObstacleGuidanceUtils.normalized_negative_cost_grad_update(
                update_sample=refined_in,
                cost=cost,
                scale=scale,
                grad_source=refined_in,
            )
            refined = torch.clamp(refined, -1.0, 1.0)
            steps_taken += 1

        min_xy_distance = final_stats["min_xy_distance"] if final_stats is not None else None
        min_z_clearance = None if final_stats is None else final_stats.get("min_z_clearance", None)
        self._update_last_obstacle_guidance_info(dict(
            applied=True,
            final_collision_refine=True,
            final_collision_cost_before=initial_cost,
            final_collision_cost_after=final_cost,
            final_collision_free=bool(final_cost is not None and final_cost <= threshold),
            final_collision_threshold=float(threshold),
            collision_refine_steps=steps_taken,
            collision_refine_grad_norm=None if last_grad_norm is None else TensorUtils.to_numpy(last_grad_norm),
            final_min_xy_distance=None if min_xy_distance is None else TensorUtils.to_numpy(min_xy_distance),
            final_min_z_clearance=None if min_z_clearance is None else TensorUtils.to_numpy(min_z_clearance),
        ))
        return refined.detach()
    
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

        guidance_enabled = self._obstacle_guidance_enabled()
        context = self.obstacle_guidance_context
        final_refine_enabled = (
            context is not None
            and context.get("enabled", False)
            and context.get("final_collision_refine", False)
        )
        self.last_obstacle_guidance_info = dict(applied=False)
        self.obstacle_guidance_sample_count += 1
        for step_index, k in enumerate(self.noise_scheduler.timesteps):
            if guidance_enabled:
                naction = self._guided_scheduler_step(
                    nets=nets,
                    naction=naction,
                    timestep=k,
                    obs_cond=obs_cond,
                    step_index=step_index,
                    num_steps=len(self.noise_scheduler.timesteps),
                )
            else:
                # predict noise
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        if guidance_enabled or final_refine_enabled:
            action = self._refine_obstacle_guidance_action(action)
        return action

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
