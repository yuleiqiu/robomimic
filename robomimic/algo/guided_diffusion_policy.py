"""
Guided Diffusion Policy — extends DiffusionPolicyUNet with inference-time
obstacle cost guidance during the denoising process.
"""
import torch

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obstacle_guidance_utils as ObstacleGuidanceUtils

from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


def wrap_as_guided(policy):
    """
    Promote a RolloutPolicy's inner DiffusionPolicyUNet to a
    GuidedDiffusionPolicyUNet by assigning guidance methods and
    attributes in-place. This allows loading a standard checkpoint
    and adding guidance at rollout time.
    """
    inner = getattr(policy, "policy", policy)
    attrs = {
        "obstacle_guidance_context": None,
        "last_obstacle_guidance_info": None,
        "obstacle_guidance_sample_count": 0,
    }
    methods = [
        "set_obstacle_guidance_context",
        "_obstacle_guidance_enabled",
        "_obstacle_guidance_cost",
        "_update_last_obstacle_guidance_info",
        "_guided_scheduler_step",
        "_refine_obstacle_guidance_action",
        "_sample_unguided_action_predictions",
        "_rank_action_predictions",
        "_get_action_trajectory",
    ]
    for k, v in attrs.items():
        setattr(inner, k, v)
    for name in methods:
        setattr(inner, name, getattr(GuidedDiffusionPolicyUNet, name).__get__(inner, type(inner)))
    return policy


class GuidedDiffusionPolicyUNet(DiffusionPolicyUNet):
    """
    Diffusion Policy with optional inference-time obstacle cost guidance.

    Usage: wrap a trained checkpoint at rollout time:

        from robomimic.algo.guided_diffusion_policy import GuidedDiffusionPolicyUNet
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path, device)
        wrapped = GuidedDiffusionPolicyUNet.from_checkpoint(policy, ckpt_dict)
        wrapped.set_obstacle_guidance_context(context)
    """

    def _create_networks(self):
        super()._create_networks()
        self.obstacle_guidance_context = None
        self.last_obstacle_guidance_info = None
        self.obstacle_guidance_sample_count = 0

    def reset(self):
        super().reset()
        self.last_obstacle_guidance_info = None
        self.obstacle_guidance_sample_count = 0

    # ------------------------------------------------------------------
    # Guidance context management
    # ------------------------------------------------------------------

    def set_obstacle_guidance_context(self, context=None):
        self.obstacle_guidance_context = context

    def _obstacle_guidance_enabled(self):
        context = self.obstacle_guidance_context
        if context is None:
            return False
        if not context.get("enabled", False):
            return False
        if context.get("selection_mode", "gradient") == "none":
            return False
        if context.get("geometry_source", "oracle_center") == "pointcloud":
            points = context.get("obstacle_points_world", None)
            if points is None or len(points) == 0:
                return False
            if context.get("selection_mode", "gradient") == "ranking":
                return int(context.get("ranking_num_candidates", 1)) > 0
            return context.get("guidance_scale", 0.0) > 0.0
        centers = context.get("obstacle_centers_xyz", context.get("obstacle_centers_xy", None))
        radii = context.get("obstacle_radii", None)
        if centers is None or radii is None:
            return False
        if len(centers) == 0 or len(radii) == 0:
            return False
        if context.get("selection_mode", "gradient") == "ranking":
            return int(context.get("ranking_num_candidates", 1)) > 0
        return context.get("guidance_scale", 0.0) > 0.0

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _obstacle_guidance_cost(self, action_chunk, horizon=None, return_stats=True):
        context = self.obstacle_guidance_context
        if horizon is None:
            horizon = context.get("guidance_horizon", self.algo_config.horizon.action_horizon)
        guidance_mode = context.get("guidance_mode", "xyz_cylinder")
        geometry_source = context.get("geometry_source", "oracle_center")
        action_for_cost = ObstacleGuidanceUtils.unnormalize_action_chunk(
            action_chunk=action_chunk,
            action_scale=context.get("action_scale", None),
            action_offset=context.get("action_offset", None),
        )
        if geometry_source == "pointcloud":
            return ObstacleGuidanceUtils.obstacle_pointcloud_cost(
                action_chunk=action_for_cost,
                current_eef_pos=context["current_eef_pos"],
                obstacle_points_world=context["obstacle_points_world"],
                safe_distance=context.get("pc_safe_distance", context.get("safe_distance", 0.02)),
                distance_mode=context.get("pc_distance_mode", context.get("distance_mode", "xy")),
                horizon=horizon,
                delta_pos_scale=context.get("delta_pos_scale", 1.0),
                delta_pos_offset=context.get("delta_pos_offset", 0.0),
                trajectory_model=context.get("trajectory_model", None),
                trajectory_model_state=context.get("trajectory_model_state", None),
                return_stats=return_stats,
            )
        if geometry_source != "oracle_center":
            raise ValueError("Unsupported obstacle geometry_source '{}'".format(geometry_source))
        if guidance_mode == "xy":
            return ObstacleGuidanceUtils.obstacle_xy_cost(
                action_chunk=action_for_cost,
                current_eef_pos=context["current_eef_pos"],
                obstacle_centers_xy=context["obstacle_centers_xyz"],
                obstacle_radii=context["obstacle_radii"],
                horizon=horizon,
                delta_pos_scale=context.get("delta_pos_scale", 1.0),
                delta_pos_offset=context.get("delta_pos_offset", 0.0),
                trajectory_model=context.get("trajectory_model", None),
                trajectory_model_state=context.get("trajectory_model_state", None),
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
                trajectory_model=context.get("trajectory_model", None),
                trajectory_model_state=context.get("trajectory_model_state", None),
                return_stats=return_stats,
            )
        raise ValueError("Unsupported obstacle guidance mode '{}'".format(guidance_mode))

    def _update_last_obstacle_guidance_info(self, updates):
        if self.last_obstacle_guidance_info is None:
            self.last_obstacle_guidance_info = dict(applied=True)
        self.last_obstacle_guidance_info.update(updates)

    # ------------------------------------------------------------------
    # Guided denoising step
    # ------------------------------------------------------------------

    def _guided_scheduler_step(
        self,
        nets,
        naction,
        timestep,
        obs_cond,
        step_index,
        num_steps,
        guidance_start_step=0,
    ):
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
            guidance_start_step=guidance_start_step,
        )
        guidance_grad_mask = context.get("guidance_grad_mask", None)
        guided_sample, grad_norm = ObstacleGuidanceUtils.normalized_negative_cost_grad_update(
            update_sample=step_output.prev_sample,
            cost=cost,
            scale=rho_t,
            grad_source=naction_in,
            grad_mask=guidance_grad_mask,
        )

        min_distance = cost_stats.get("min_distance", None)
        min_xy_distance = cost_stats.get("min_xy_distance", None)
        display_distance = min_xy_distance if min_xy_distance is not None else min_distance
        min_z_clearance = cost_stats.get("min_z_clearance", None)
        min_pointcloud_distance = cost_stats.get("min_pointcloud_distance", None)
        self.last_obstacle_guidance_info = dict(
            applied=True,
            guidance_mode=guidance_mode,
            geometry_source=context.get("geometry_source", "oracle_center"),
            rho_t=float(rho_t),
            cost=float(cost.detach().cpu().item()),
            min_distance=None if display_distance is None else TensorUtils.to_numpy(display_distance),
            min_xy_distance=None if min_xy_distance is None else TensorUtils.to_numpy(min_xy_distance),
            min_pointcloud_distance=(
                None if min_pointcloud_distance is None
                else TensorUtils.to_numpy(min_pointcloud_distance)
            ),
            min_z_clearance=None if min_z_clearance is None
            else TensorUtils.to_numpy(min_z_clearance),
            grad_norm=None if grad_norm is None else TensorUtils.to_numpy(grad_norm),
            guidance_grad_mask=None if guidance_grad_mask is None else TensorUtils.to_numpy(guidance_grad_mask),
            num_obstacles=int(cost_stats["num_obstacles"]),
            num_points=int(cost_stats.get("num_points", 0)),
            obstacle_top_z=context.get("obstacle_top_z", None),
            z_clearance=context.get("z_clearance", None),
            delta_pos_scale=context.get("delta_pos_scale", None),
        )
        return guided_sample

    # ------------------------------------------------------------------
    # Post-hoc collision refinement
    # ------------------------------------------------------------------

    def _refine_obstacle_guidance_action(self, action):
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
                grad_mask=context.get("guidance_grad_mask", None),
            )
            refined = torch.clamp(refined, -1.0, 1.0)
            steps_taken += 1

        min_xy_distance = None if final_stats is None else final_stats.get("min_xy_distance", None)
        min_distance = None if final_stats is None else final_stats.get("min_distance", None)
        display_distance = min_xy_distance if min_xy_distance is not None else min_distance
        min_pointcloud_distance = None if final_stats is None else final_stats.get("min_pointcloud_distance", None)
        min_z_clearance = None if final_stats is None else final_stats.get("min_z_clearance", None)
        self._update_last_obstacle_guidance_info(dict(
            applied=True,
            final_collision_refine=True,
            final_collision_cost_before=initial_cost,
            final_collision_cost_after=final_cost,
            final_collision_free=bool(final_cost is not None and final_cost <= threshold),
            final_collision_threshold=float(threshold),
            collision_refine_steps=steps_taken,
            collision_refine_grad_norm=None if last_grad_norm is None
            else TensorUtils.to_numpy(last_grad_norm),
            final_min_distance=None if display_distance is None
            else TensorUtils.to_numpy(display_distance),
            final_min_xy_distance=None if min_xy_distance is None
            else TensorUtils.to_numpy(min_xy_distance),
            final_min_pointcloud_distance=(
                None if min_pointcloud_distance is None
                else TensorUtils.to_numpy(min_pointcloud_distance)
            ),
            final_min_z_clearance=None if min_z_clearance is None
            else TensorUtils.to_numpy(min_z_clearance),
        ))
        return refined.detach()

    # ------------------------------------------------------------------
    # Override inference with guided denoising loop
    # ------------------------------------------------------------------

    def _sample_unguided_action_predictions(
        self,
        nets,
        obs_cond,
        prediction_horizon,
        action_dim,
        num_samples,
    ):
        if obs_cond.shape[0] != 1:
            raise ValueError("Action-chunk ranking currently expects rollout batch size 1")
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        obs_cond = obs_cond.expand(num_samples, -1)
        naction = torch.randn((num_samples, prediction_horizon, action_dim), device=self.device)

        self.noise_scheduler.set_timesteps(num_inference_timesteps)
        for k in self.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample
        return naction

    def _rank_action_predictions(self, action_predictions):
        context = self.obstacle_guidance_context
        cost, cost_stats = self._obstacle_guidance_cost(
            action_chunk=action_predictions,
            horizon=context.get("guidance_horizon", self.algo_config.horizon.action_horizon),
            return_stats=True,
        )
        per_cost = cost_stats.get("per_batch_cost", None)
        if per_cost is None:
            per_cost = torch.zeros(
                (action_predictions.shape[0],),
                dtype=action_predictions.dtype,
                device=action_predictions.device,
            )

        min_xy_distance = cost_stats.get("min_xy_distance", None)
        min_distance = cost_stats.get("min_distance", None)
        min_pointcloud_distance = cost_stats.get("min_pointcloud_distance", None)
        display_distance = min_pointcloud_distance
        if display_distance is None:
            display_distance = min_xy_distance if min_xy_distance is not None else min_distance

        min_cost = torch.amin(per_cost)
        candidate_mask = per_cost <= (
            min_cost + float(context.get("ranking_cost_tie_tolerance", 1e-10))
        )
        if display_distance is not None and bool(torch.any(candidate_mask)):
            masked_distance = torch.where(
                candidate_mask,
                display_distance,
                torch.full_like(display_distance, -float("inf")),
            )
            best_index = int(torch.argmax(masked_distance).detach().cpu().item())
        else:
            best_index = int(torch.argmin(per_cost).detach().cpu().item())

        first_cost = float(per_cost[0].detach().cpu().item())
        best_cost = float(per_cost[best_index].detach().cpu().item())
        safe_threshold = float(context.get("ranking_safe_cost_threshold", 1e-8))
        safe_count = int(torch.sum(per_cost <= safe_threshold).detach().cpu().item())
        first_is_safe = first_cost <= safe_threshold
        ranking_skipped = bool(context.get("ranking_only_if_first_unsafe", False) and first_is_safe)
        if ranking_skipped:
            best_index = 0
            best_cost = first_cost

        best_distance = None
        first_distance = None
        if display_distance is not None:
            best_distance = float(display_distance[best_index].detach().cpu().item())
            first_distance = float(display_distance[0].detach().cpu().item())

        self.last_obstacle_guidance_info = dict(
            applied=True,
            selection_mode="ranking",
            guidance_mode=context.get("guidance_mode", "xyz_cylinder"),
            geometry_source=context.get("geometry_source", "oracle_center"),
            trajectory_backend=context.get("trajectory_backend", "cumsum"),
            candidate_count=int(action_predictions.shape[0]),
            best_index=best_index,
            ranking_skipped=ranking_skipped,
            ranking_first_is_safe=first_is_safe,
            cost=best_cost,
            ranking_first_cost=first_cost,
            ranking_best_cost=best_cost,
            ranking_cost_improvement=first_cost - best_cost,
            ranking_safe_count=safe_count,
            ranking_safe_rate=float(safe_count / max(int(action_predictions.shape[0]), 1)),
            ranking_first_distance=first_distance,
            ranking_best_distance=best_distance,
            ranking_distance_improvement=(
                None if best_distance is None or first_distance is None
                else best_distance - first_distance
            ),
            ranking_all_costs=TensorUtils.to_numpy(per_cost.detach()),
            ranking_all_distances=(
                None if display_distance is None else TensorUtils.to_numpy(display_distance.detach())
            ),
            min_distance=None if best_distance is None else best_distance,
            min_xy_distance=None if min_xy_distance is None else TensorUtils.to_numpy(min_xy_distance[best_index:best_index + 1]),
            min_pointcloud_distance=(
                None if min_pointcloud_distance is None
                else TensorUtils.to_numpy(min_pointcloud_distance[best_index:best_index + 1])
            ),
            min_z_clearance=(
                None if cost_stats.get("min_z_clearance", None) is None
                else TensorUtils.to_numpy(cost_stats["min_z_clearance"][best_index:best_index + 1])
            ),
            num_obstacles=int(cost_stats["num_obstacles"]),
            num_points=int(cost_stats.get("num_points", 0)),
            obstacle_top_z=context.get("obstacle_top_z", None),
            z_clearance=context.get("z_clearance", None),
            delta_pos_scale=context.get("delta_pos_scale", None),
        )
        return best_index

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

        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3
        B = obs_features.shape[0]

        obs_cond = obs_features.flatten(start_dim=1)

        noisy_action = torch.randn((B, Tp, action_dim), device=self.device)
        naction = noisy_action

        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        guidance_enabled = self._obstacle_guidance_enabled()
        context = self.obstacle_guidance_context
        selection_mode = context.get("selection_mode", "gradient") if context else "gradient"
        if guidance_enabled and selection_mode == "ranking":
            self.last_obstacle_guidance_info = dict(applied=False)
            self.obstacle_guidance_sample_count += 1
            with torch.no_grad():
                action_predictions = self._sample_unguided_action_predictions(
                    nets=nets,
                    obs_cond=obs_cond,
                    prediction_horizon=Tp,
                    action_dim=action_dim,
                    num_samples=int(context.get("ranking_num_candidates", 1)),
                )
                best_index = self._rank_action_predictions(action_predictions)
            start = To - 1
            end = start + Ta
            action = action_predictions[best_index:best_index + 1, start:end].detach()
            if self.last_obstacle_guidance_info is not None:
                first_action = action_predictions[0:1, start:end].detach()
                selected_action_for_exec = ObstacleGuidanceUtils.unnormalize_action_chunk(
                    action_chunk=action,
                    action_scale=context.get("action_scale", None),
                    action_offset=context.get("action_offset", None),
                )
                first_action_for_exec = ObstacleGuidanceUtils.unnormalize_action_chunk(
                    action_chunk=first_action,
                    action_scale=context.get("action_scale", None),
                    action_offset=context.get("action_offset", None),
                )
                self.last_obstacle_guidance_info.update(
                    ranking_selected_action_chunk=TensorUtils.to_numpy(selected_action_for_exec[0]),
                    ranking_first_action_chunk=TensorUtils.to_numpy(first_action_for_exec[0]),
                )
            return action

        final_refine_enabled = (
            context is not None
            and context.get("enabled", False)
            and context.get("final_collision_refine", False)
            and selection_mode == "gradient"
        )
        guidance_start_pct = context.get("guidance_start_step_pct", 0.0) if context else 0.0
        guidance_start_step = int(len(self.noise_scheduler.timesteps) * guidance_start_pct)
        self.last_obstacle_guidance_info = dict(applied=False)
        self.obstacle_guidance_sample_count += 1
        for step_index, k in enumerate(self.noise_scheduler.timesteps):
            if guidance_enabled and step_index >= guidance_start_step:
                naction = self._guided_scheduler_step(
                    nets=nets,
                    naction=naction,
                    timestep=k,
                    obs_cond=obs_cond,
                    step_index=step_index,
                    num_steps=len(self.noise_scheduler.timesteps),
                    guidance_start_step=guidance_start_step,
                )
            else:
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        start = To - 1
        end = start + Ta
        action = naction[:, start:end]
        if guidance_enabled or final_refine_enabled:
            action = self._refine_obstacle_guidance_action(action)
        return action
