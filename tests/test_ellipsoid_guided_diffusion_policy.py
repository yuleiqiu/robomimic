from collections import OrderedDict

import numpy as np
import torch

from robomimic.algo import algo_factory, algo_name_to_factory_func
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.ellipsoid_guidance_utils import EllipsoidGuidanceContext


def small_policy(name):
    config = config_factory(name)
    with config.unlocked():
        config.train.cuda = False
        config.train.action_keys = ["delta_eef_pose_action"]
        config.algo.unet.down_dims = [32, 64]
        config.algo.unet.diffusion_step_embed_dim = 32
        config.algo.ddpm.num_train_timesteps = 4
        config.algo.ddpm.num_inference_timesteps = 4
        config.algo.ddpm.prediction_type = "sample"
        config.algo.optim_params.policy.num_train_batches = 1
        config.algo.optim_params.policy.num_epochs = 1
        config.observation.modalities.obs.low_dim = ["agent_pos"]
        config.observation.modalities.obs.scan = []
    ObsUtils.initialize_obs_utils_with_config(config)
    model = algo_factory(
        name,
        config,
        obs_key_shapes=OrderedDict([("agent_pos", (6,))]),
        ac_dim=7,
        device=torch.device("cpu"),
    )
    model.set_eval()
    return model


def test_ellipsoid_guidance_registration_and_disabled_parity():
    assert callable(algo_name_to_factory_func("ellipsoid_guided_diffusion_policy"))
    source = small_policy("diffusion_policy")
    guided = small_policy("ellipsoid_guided_diffusion_policy")
    guided.deserialize(source.serialize())
    observation = {"agent_pos": torch.randn(1, 2, 6)}
    torch.manual_seed(825)
    with torch.no_grad():
        expected = source._get_action_trajectory(observation)
    torch.manual_seed(825)
    with torch.no_grad():
        actual = guided._get_action_trajectory(observation)
    assert torch.equal(expected, actual)


def test_enabled_ellipsoid_guidance_is_finite_and_records_each_reverse_step():
    guided = small_policy("ellipsoid_guided_diffusion_policy")
    guided.set_guidance_context(
        EllipsoidGuidanceContext(
            current_eef_position=np.zeros(3),
            current_eef_rotation=np.eye(3),
            actor_local_centers=np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.0, -0.08]],
                dtype=np.float32,
            ),
            actor_local_rotations=np.asarray([np.eye(3), np.eye(3)]),
            actor_semi_axes=np.asarray(
                [[0.05, 0.05, 0.10], [0.04, 0.04, 0.07]],
                dtype=np.float32,
            ),
            obstacle_center=np.zeros(3),
            obstacle_rotation=np.eye(3),
            obstacle_semi_axes=np.asarray([2.0, 2.0, 2.0]),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1e-4,
            actor_labels=("gripper", "target"),
        )
    )
    observation = {"agent_pos": torch.randn(1, 2, 6)}
    torch.manual_seed(826)
    with torch.no_grad():
        action = guided._get_action_trajectory(observation)
    assert torch.isfinite(action).all()
    diagnostics = guided.get_guidance_diagnostics()
    assert len(diagnostics) == 4
    assert all(item["active_actor_waypoint_count"] > 0 for item in diagnostics)
    assert all(set(item["actor_minimum_clearances_m"]) == {"gripper", "target"} for item in diagnostics)
    assert guided.last_predicted_action_chunk.shape == (1, 8, 7)
