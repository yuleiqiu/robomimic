"""
Config for Action Value Critic training.
"""

from robomimic.config.base_config import BaseConfig


class CriticConfig(BaseConfig):
    ALGO_NAME = "action_value_critic"

    def experiment_config(self):
        super(CriticConfig, self).experiment_config()
        self.experiment.name = "action_value_critic"
        self.experiment.validate = True
        self.experiment.save.enabled = True
        self.experiment.save.every_n_epochs = 10
        self.experiment.rollout.enabled = False

    def train_config(self):
        super(CriticConfig, self).train_config()

        self.train.rollout_dataset_path = None

        self.train.base_policy_ckpt = None

        self.train.action_horizon = 8

        self.train.obs_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]

        self.train.num_epochs = 100
        self.train.batch_size = 64
        self.train.num_data_workers = 2

        self.train.learning_rate = 1e-4
        self.train.weight_decay = 1e-6

        self.train.hidden_dim = 256

        self.train.train_val_split = 0.9

        self.train.balance_success_failure = False

        self.train.freeze_obs_encoder = True

        self.train.cuda = True

        self.train.seed = 1

        self.train.filter_key = None

    def algo_config(self):
        pass
