"""Pure tensor utilities for Set-Supervised Diffusion Policy targets."""

import torch


def desired_set_radii(positive_actions, negative_actions, radius_ratio):
    """Return per-timestep Euclidean radii with shape ``[..., T, 1]``."""

    if positive_actions.shape != negative_actions.shape:
        raise ValueError(
            "Positive and negative action shapes differ: {} vs {}".format(
                tuple(positive_actions.shape),
                tuple(negative_actions.shape),
            )
        )
    if radius_ratio < 0.0 or radius_ratio > 1.0:
        raise ValueError("radius_ratio must be in [0, 1]")
    return (
        float(radius_ratio)
        * torch.linalg.vector_norm(
            positive_actions - negative_actions,
            dim=-1,
            keepdim=True,
        )
    )


def desired_set_membership(
    candidate_actions,
    positive_actions,
    negative_actions,
    radius_ratio,
    tolerance=1e-6,
):
    """Test membership in each per-timestep desired-action ball."""

    if candidate_actions.shape != positive_actions.shape:
        raise ValueError(
            "Candidate and positive action shapes differ: {} vs {}".format(
                tuple(candidate_actions.shape),
                tuple(positive_actions.shape),
            )
        )
    radii = desired_set_radii(
        positive_actions,
        negative_actions,
        radius_ratio,
    )
    distances = torch.linalg.vector_norm(
        candidate_actions - positive_actions,
        dim=-1,
        keepdim=True,
    )
    return distances <= radii + float(tolerance)


def apply_desired_set_operator(
    candidate_actions,
    positive_actions,
    negative_actions,
    radius_ratio,
    tolerance=1e-6,
):
    """
    Apply SDP Eq. (14).

    A complete single-step action outside its desired ball is replaced by the
    paired positive action. In-set actions are retained unchanged.
    """

    membership = desired_set_membership(
        candidate_actions=candidate_actions,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        radius_ratio=radius_ratio,
        tolerance=tolerance,
    )
    projected = torch.where(
        membership,
        candidate_actions,
        positive_actions,
    )
    return projected, membership


@torch.no_grad()
def sample_desired_action_chunks(
    noise_pred_net,
    scheduler,
    observation_condition,
    positive_actions,
    negative_actions,
    radius_ratio,
    num_samples,
    start_timestep,
    initialization="gaussian",
    tolerance=1e-6,
    generator=None,
):
    """
    Sample policy-relative targets with truncated constrained DDPM denoising.

    Returns:
        targets: ``[B, N, T, D]`` detached desired chunks.
        statistics: scalar tensor diagnostics.
    """

    if positive_actions.ndim != 3:
        raise ValueError(
            "Expected positive actions [B, T, D], got {}".format(
                tuple(positive_actions.shape)
            )
        )
    if positive_actions.shape != negative_actions.shape:
        raise ValueError("Positive and negative action shapes must match")
    if observation_condition.shape[0] != positive_actions.shape[0]:
        raise ValueError("Observation and action batch sizes must match")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if start_timestep < 0:
        raise ValueError("start_timestep must be non-negative")
    if start_timestep >= scheduler.config.num_train_timesteps:
        raise ValueError(
            "start_timestep {} exceeds scheduler horizon {}".format(
                start_timestep,
                scheduler.config.num_train_timesteps,
            )
        )
    if initialization not in ("gaussian", "positive"):
        raise ValueError(
            "initialization must be 'gaussian' or 'positive'"
        )

    batch_size, horizon, action_dim = positive_actions.shape
    repeated_positive = positive_actions.repeat_interleave(
        num_samples, dim=0
    )
    repeated_negative = negative_actions.repeat_interleave(
        num_samples, dim=0
    )
    repeated_condition = observation_condition.repeat_interleave(
        num_samples, dim=0
    )

    if initialization == "gaussian":
        samples = torch.randn(
            (batch_size * num_samples, horizon, action_dim),
            device=positive_actions.device,
            dtype=positive_actions.dtype,
            generator=generator,
        )
    else:
        samples = repeated_positive.clone()

    operator_replacement_count = torch.zeros(
        (), device=positive_actions.device, dtype=torch.long
    )
    operator_element_count = torch.zeros(
        (), device=positive_actions.device, dtype=torch.long
    )
    for timestep in range(start_timestep, -1, -1):
        timestep_batch = torch.full(
            (samples.shape[0],),
            timestep,
            device=samples.device,
            dtype=torch.long,
        )
        noise_prediction = noise_pred_net(
            samples,
            timestep_batch,
            global_cond=repeated_condition,
        )
        samples = scheduler.step(
            model_output=noise_prediction,
            timestep=timestep,
            sample=samples,
            generator=generator,
        ).prev_sample
        samples, membership = apply_desired_set_operator(
            candidate_actions=samples,
            positive_actions=repeated_positive,
            negative_actions=repeated_negative,
            radius_ratio=radius_ratio,
            tolerance=tolerance,
        )
        operator_replacement_count += torch.count_nonzero(~membership)
        operator_element_count += membership.numel()

    final_membership = desired_set_membership(
        candidate_actions=samples,
        positive_actions=repeated_positive,
        negative_actions=repeated_negative,
        radius_ratio=radius_ratio,
        tolerance=tolerance,
    )
    if not torch.all(final_membership):
        raise RuntimeError("Desired target sampler returned out-of-set actions")

    target_distance = torch.linalg.vector_norm(
        samples - repeated_positive,
        dim=-1,
    )
    set_radii = desired_set_radii(
        repeated_positive,
        repeated_negative,
        radius_ratio,
    ).squeeze(-1)
    targets = samples.reshape(
        batch_size,
        num_samples,
        horizon,
        action_dim,
    ).detach()
    statistics = {
        "replacement_rate": (
            operator_replacement_count.float()
            / operator_element_count.clamp_min(1).float()
        ),
        "final_preprojection_retention_rate": membership.float().mean(),
        "mean_set_radius": set_radii.mean(),
        "maximum_set_radius": set_radii.max(),
        "mean_target_distance": target_distance.mean(),
        "maximum_target_distance": target_distance.max(),
        "nonpositive_timestep_rate": torch.mean(
            (target_distance > tolerance).float()
        ),
    }
    return targets, statistics
