from collections import OrderedDict

import numpy as np
import torch

from robomimic.algo import algo_factory, algo_name_to_factory_func
from robomimic.algo.ellipsoid_guided_lan_o3dp import EllipsoidGuidedLanO3DPUNet
from robomimic.algo.lan_o3dp import LanO3DPUNet
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.ellipsoid_guidance_utils import (
    EllipsoidGuidanceContext,
    ellipsoid_guided_policy_from_checkpoint,
    reconstruct_eef_poses,
)


def small_lan_config(algo_name):
    config = config_factory(algo_name)
    with config.unlocked():
        config.train.cuda = False
        config.algo.unet.down_dims = [32, 64]
        config.algo.unet.diffusion_step_embed_dim = 32
        config.algo.optim_params.policy.num_train_batches = 1
        config.algo.optim_params.policy.num_epochs = 1
        config.observation.modalities.obs.low_dim = ["agent_pos"]
        config.observation.modalities.obs.scan = ["task_pointcloud"]
        config.observation.modalities.obs.rgb = []
        config.observation.modalities.obs.depth = []
        config.observation.encoder.scan.core_class = "LanO3DPPointCloudCore"
        config.observation.encoder.scan.core_kwargs = {}
    return config


def make_lan_model(config):
    ObsUtils.initialize_obs_utils_with_config(config)
    return algo_factory(
        config.algo_name,
        config,
        obs_key_shapes=OrderedDict(
            [
                ("agent_pos", (6,)),
                ("task_pointcloud", (3, 12)),
            ]
        ),
        ac_dim=7,
        device=torch.device("cpu"),
    )


def test_absolute_reconstruction_ignores_current_pose():
    positions = torch.tensor([[[0.2, -0.3, 0.9], [0.25, 0.1, 0.8]]])
    rotations = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2.0]]],
        dtype=torch.float32,
    )
    actual_positions, actual_rotations = reconstruct_eef_poses(
        [999.0, 999.0, 999.0],
        torch.eye(3),
        positions,
        rotations,
        position_mode="absolute",
    )
    torch.testing.assert_close(actual_positions, positions)
    torch.testing.assert_close(
        actual_rotations[0, 0],
        torch.eye(3),
    )
    torch.testing.assert_close(
        actual_rotations[0, 1],
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_lan_ellipsoid_variant_registration_and_disabled_parity():
    assert callable(algo_name_to_factory_func("ellipsoid_guided_lan_o3dp"))
    source = make_lan_model(small_lan_config("lan_o3dp"))
    guided = make_lan_model(small_lan_config("ellipsoid_guided_lan_o3dp"))
    guided.deserialize(source.serialize())
    assert isinstance(guided, EllipsoidGuidedLanO3DPUNet)
    source.set_eval()
    guided.set_eval()

    observation = {
        "agent_pos": torch.randn(1, 2, 6),
        "task_pointcloud": torch.randn(1, 2, 3, 12),
    }
    torch.manual_seed(7)
    with torch.no_grad():
        expected = source._get_action_trajectory(observation)
    torch.manual_seed(7)
    with torch.no_grad():
        actual = guided._get_action_trajectory(observation)
    assert torch.equal(expected, actual)


def test_lan_absolute_context_gradient_is_finite_and_preserves_gripper():
    guided = make_lan_model(small_lan_config("ellipsoid_guided_lan_o3dp"))
    guided.set_eval()
    guided.set_guidance_context(
        EllipsoidGuidanceContext(
            current_eef_position=np.zeros(3),
            current_eef_rotation=np.eye(3),
            actor_local_centers=np.zeros((1, 3)),
            actor_local_rotations=np.asarray([np.eye(3)]),
            actor_semi_axes=np.asarray([[0.08, 0.08, 0.08]]),
            obstacle_center=np.zeros(3),
            obstacle_rotation=np.eye(3),
            obstacle_semi_axes=np.asarray([0.08, 0.08, 0.08]),
            action_scale=np.ones(7),
            action_offset=np.zeros(7),
            guidance_scale=1e-4,
            position_mode="absolute",
        )
    )
    observation = {
        "agent_pos": torch.randn(1, 2, 6),
        "task_pointcloud": torch.randn(1, 2, 3, 12),
    }
    torch.manual_seed(8)
    with torch.no_grad():
        action = guided._get_action_trajectory(observation)
    assert torch.isfinite(action).all()
    assert len(guided.get_guidance_diagnostics()) > 0


def test_lan_checkpoint_loader_returns_ellipsoid_lan_class():
    source_config = small_lan_config("lan_o3dp")
    model = make_lan_model(source_config)
    ckpt = {
        "algo_name": "lan_o3dp",
        "config": source_config.dump(),
        "env_metadata": {},
        "shape_metadata": {
            "all_shapes": OrderedDict(
                [
                    ("agent_pos", (6,)),
                    ("task_pointcloud", (3, 12)),
                ]
            ),
            "ac_dim": 7,
        },
        "model": model.serialize(),
    }
    policy, _ = ellipsoid_guided_policy_from_checkpoint(
        ckpt_dict=ckpt,
        device=torch.device("cpu"),
    )
    assert isinstance(policy.policy, EllipsoidGuidedLanO3DPUNet)
