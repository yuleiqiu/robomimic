import unittest
from collections import OrderedDict

import torch

from robomimic.algo import algo_factory
from robomimic.algo.guided_diffusion_policy import GuidedLanO3DPUNet
from robomimic.algo.lan_o3dp import LanO3DPUNet
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.guided_denoising_utils as GuidedUtils
import robomimic.utils.obs_utils as ObsUtils


def small_config(algo_name="lan_o3dp", down_dims=(32, 64), embed_dim=32):
    config = config_factory(algo_name)
    with config.unlocked():
        config.train.cuda = False
        config.algo.unet.down_dims = list(down_dims)
        config.algo.unet.diffusion_step_embed_dim = embed_dim
        config.algo.optim_params.policy.num_train_batches = 2
        config.algo.optim_params.policy.num_epochs = 2
        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        config.observation.modalities.obs.scan = ["task_pointcloud"]
        config.observation.modalities.obs.rgb = []
        config.observation.modalities.obs.depth = []
        config.observation.encoder.scan.core_class = "LanO3DPPointCloudCore"
        config.observation.encoder.scan.core_kwargs = {}
    return config


def make_model(config):
    ObsUtils.initialize_obs_utils_with_config(config)
    return algo_factory(
        config.algo_name,
        config,
        obs_key_shapes=OrderedDict(
            [
                ("robot0_eef_pos", (3,)),
                ("robot0_eef_quat", (4,)),
                ("robot0_gripper_qpos", (2,)),
                ("task_pointcloud", (3, 32)),
            ]
        ),
        ac_dim=7,
        device=torch.device("cpu"),
    )


class TestLanO3DPRegistration(unittest.TestCase):
    def test_config_and_algorithm_factory(self):
        config = small_config()
        self.assertEqual(config.algo_name, "lan_o3dp")
        self.assertEqual(list(config.algo.unet.down_dims), [32, 64])
        model = make_model(config)
        self.assertIsInstance(model, LanO3DPUNet)
        obs_encoder = model.nets["policy"]["obs_encoder"].nets["obs"]
        self.assertIsNone(obs_encoder.activation)
        self.assertEqual(
            model.optimizers["policy"].defaults["betas"], (0.95, 0.999)
        )
        self.assertEqual(model.optimizers["policy"].defaults["eps"], 1e-8)

    def test_checkpoint_reload_returns_lan_class(self):
        config = small_config()
        model = make_model(config)
        ckpt = {
            "algo_name": "lan_o3dp",
            "config": config.dump(),
            "shape_metadata": {
                "all_shapes": OrderedDict(
                    [
                        ("robot0_eef_pos", (3,)),
                        ("robot0_eef_quat", (4,)),
                        ("robot0_gripper_qpos", (2,)),
                        ("task_pointcloud", (3, 32)),
                    ]
                ),
                "ac_dim": 7,
            },
            "model": model.serialize(),
        }
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt, device=torch.device("cpu")
        )
        self.assertIsInstance(policy.policy, LanO3DPUNet)

        guided_policy, _ = GuidedUtils.guided_policy_from_checkpoint(
            ckpt_dict=ckpt, device=torch.device("cpu")
        )
        self.assertIsInstance(guided_policy.policy, GuidedLanO3DPUNet)
        guided_obs_encoder = guided_policy.policy.nets["policy"]["obs_encoder"].nets["obs"]
        self.assertIsNone(guided_obs_encoder.activation)

        obs = {
            "robot0_eef_pos": torch.randn(1, 2, 3),
            "robot0_eef_quat": torch.randn(1, 2, 4),
            "robot0_gripper_qpos": torch.randn(1, 2, 2),
            "task_pointcloud": torch.randn(1, 2, 3, 32),
        }
        model.set_eval()
        guided_policy.policy.set_eval()
        torch.manual_seed(123)
        with torch.no_grad():
            original_action = model._get_action_trajectory(obs)
        torch.manual_seed(123)
        with torch.no_grad():
            guided_disabled_action = guided_policy.policy._get_action_trajectory(obs)
        self.assertTrue(torch.equal(original_action, guided_disabled_action))

    def test_unet_config_changes_actual_structure_and_parameter_count(self):
        first = make_model(small_config(down_dims=(32, 64), embed_dim=32))
        second = make_model(small_config(down_dims=(64, 128), embed_dim=64))
        first_unet = first.nets["policy"]["noise_pred_net"]
        second_unet = second.nets["policy"]["noise_pred_net"]
        self.assertEqual(first_unet.down_modules[0][0].blocks[0].block[0].out_channels, 32)
        self.assertEqual(second_unet.down_modules[0][0].blocks[0].block[0].out_channels, 64)
        first_count = sum(parameter.numel() for parameter in first_unet.parameters())
        second_count = sum(parameter.numel() for parameter in second_unet.parameters())
        self.assertNotEqual(first_count, second_count)

        legacy = config_factory("diffusion_policy")
        self.assertEqual(list(legacy.algo.unet.down_dims), [256, 512, 1024])
        self.assertEqual(legacy.algo.unet.diffusion_step_embed_dim, 256)
        self.assertEqual(dict(legacy.algo.optim_params.policy.optimizer_kwargs), {})


if __name__ == "__main__":
    unittest.main()
