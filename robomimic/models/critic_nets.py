"""
Action Value Critic network for Monte-Carlo action chunk reranking.

C_phi(o_t, A_t) ≈ P(S = 1 | o_t, A_t)

where S is the final episode success label of the rollout.
"""

import torch
import torch.nn as nn

from collections import OrderedDict

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


class ActionValueCritic(nn.Module):
    """
    Monte-Carlo action-value critic that estimates the probability of
    episode success given an observation and an action chunk.

    Architecture:
        z_obs = obs_encoder(obs_t)                    # [B, D_obs]
        z_act = action_mlp(flatten(action_chunk_t))   # [B, D_act]
        logit = critic_head(concat(z_obs, z_act))     # [B, 1]
    """

    def __init__(
        self,
        obs_shapes,
        action_dim,
        action_horizon,
        obs_encoder,
        observation_horizon=2,
        hidden_dim=256,
        freeze_obs_encoder=True,
    ):
        """
        Args:
            obs_shapes (OrderedDict): maps obs key to shape
            action_dim (int): dimension of action space
            action_horizon (int): length of action chunk H
            obs_encoder (ObservationGroupEncoder): observation encoder (from base policy)
            observation_horizon (int): number of observation frames To (default 2)
            hidden_dim (int): hidden dimension for MLPs
            freeze_obs_encoder (bool): if True, freeze obs_encoder parameters
        """
        super(ActionValueCritic, self).__init__()

        self.obs_shapes = obs_shapes
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.obs_encoder = obs_encoder

        if freeze_obs_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False

        obs_feat_dim = obs_encoder.output_shape()[0] * observation_horizon

        action_input_dim = action_horizon * action_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        critic_input_dim = obs_feat_dim + hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_dict, action_chunk):
        """
        Forward pass.

        Args:
            obs_dict (dict): observation dictionary, each value has shape [B, To, ...]
                where To is observation_horizon (frame-stacked observations)
            action_chunk (torch.Tensor): action chunk [B, H, action_dim]

        Returns:
            logit (torch.Tensor): scalar logit [B, 1]
        """
        inputs = {"obs": obs_dict}
        for k in self.obs_shapes:
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        
        obs_features = TensorUtils.time_distributed(
            inputs, self.obs_encoder, inputs_as_kwargs=True
        )
        obs_features = obs_features.flatten(start_dim=1)

        B = action_chunk.shape[0]
        action_flat = action_chunk.reshape(B, -1)
        action_features = self.action_mlp(action_flat)

        combined = torch.cat([obs_features, action_features], dim=-1)
        logit = self.critic_head(combined)

        return logit


def create_obs_encoder_from_checkpoint(ckpt_dict, device=None):
    """
    Create and load observation encoder from a policy checkpoint.

    Args:
        ckpt_dict (dict): loaded checkpoint dictionary
        device (torch.device): device to put encoder on

    Returns:
        obs_encoder (ObservationGroupEncoder): loaded observation encoder
        obs_shapes (OrderedDict): observation shapes
    """
    from robomimic.utils.file_utils import config_from_checkpoint, _select_checkpoint_metadata

    algo_name = ckpt_dict["algo_name"]
    config, _ = config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict)

    ObsUtils.initialize_obs_utils_with_config(config)

    shape_meta = _select_checkpoint_metadata(ckpt_dict["shape_metadata"], "shape_metadata")
    obs_shapes = OrderedDict()
    for k in shape_meta["all_obs_keys"]:
        obs_shapes[k] = shape_meta["all_shapes"][k]

    observation_group_shapes = OrderedDict()
    observation_group_shapes["obs"] = obs_shapes

    encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(config.observation.encoder)

    obs_encoder = ObsNets.ObservationGroupEncoder(
        observation_group_shapes=observation_group_shapes,
        encoder_kwargs=encoder_kwargs,
    )

    from robomimic.algo.diffusion_policy import replace_bn_with_gn
    obs_encoder = replace_bn_with_gn(obs_encoder)

    model_state = ckpt_dict["model"]
    if model_state.get("ema", None) is not None:
        encoder_state_dict = model_state["ema"]
    else:
        encoder_state_dict = model_state["nets"]

    obs_encoder_state = {}
    prefix = "policy.obs_encoder."
    for k, v in encoder_state_dict.items():
        if k.startswith(prefix):
            obs_encoder_state[k[len(prefix):]] = v

    obs_encoder.load_state_dict(obs_encoder_state)

    if device is not None:
        obs_encoder = obs_encoder.to(device)

    return obs_encoder, obs_shapes
