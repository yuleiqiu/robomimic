from collections import OrderedDict

import numpy as np
import torch

from robomimic.algo import algo_factory, algo_name_to_factory_func
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.paper_lan_guidance_utils import (
    PaperLanGuidanceContext,
    paper_guidance_gradient,
)


def small_policy(name):
    config = config_factory(name)
    with config.unlocked():
        config.train.cuda = False
        config.train.action_keys = ["actions"]
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
        obs_key_shapes=OrderedDict([("agent_pos", (2,))]),
        ac_dim=2,
        device=torch.device("cpu"),
    )
    model.set_eval()
    return model


def test_generic_guidance_registration_and_disabled_parity():
    assert callable(algo_name_to_factory_func("point_guided_diffusion_policy"))
    source = small_policy("diffusion_policy")
    guided = small_policy("point_guided_diffusion_policy")
    guided.deserialize(source.serialize())
    observation = {"agent_pos": torch.randn(1, 2, 2)}
    torch.manual_seed(8)
    with torch.no_grad():
        expected = source._get_action_trajectory(observation)
    torch.manual_seed(8)
    with torch.no_grad():
        actual = guided._get_action_trajectory(observation)
    assert torch.equal(expected, actual)


def test_generic_sample_prediction_guidance_uses_selected_2d_indices():
    guided = small_policy("point_guided_diffusion_policy")
    guided.set_guidance_context(
        PaperLanGuidanceContext(
            current_eef_pos=np.zeros(2),
            obstacle_points=np.asarray([[0.0, 0.0]], dtype=np.float32),
            action_scale=np.ones(2),
            action_offset=np.zeros(2),
            guidance_scale=0.01,
            safety_distance=10.0,
            xy_only=False,
            position_indices=(0, 1),
            action_key="actions",
        )
    )
    observation = {"agent_pos": torch.randn(1, 2, 2)}
    torch.manual_seed(9)
    with torch.no_grad():
        action = guided._get_action_trajectory(observation)
    assert torch.isfinite(action).all()
    assert len(guided.get_guidance_diagnostics()) == 4
    assert guided.last_predicted_action_chunk.shape == (1, 8, 2)
