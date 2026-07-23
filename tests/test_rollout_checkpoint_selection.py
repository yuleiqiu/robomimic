import unittest

from robomimic.utils.train_utils import should_save_from_rollout_logs


class TestRolloutCheckpointSelection(unittest.TestCase):
    def select(self, success, loss, epoch, best_success, best_loss, best_epoch):
        return should_save_from_rollout_logs(
            all_rollout_logs={"env": {"Return": 0.0, "Success_Rate": success}},
            best_return={"env": 0.0},
            best_success_rate={"env": best_success},
            epoch_ckpt_name="model_epoch_{}".format(epoch),
            save_on_best_rollout_return=False,
            save_on_best_rollout_success_rate=True,
            rollout_success_tiebreak_validation=True,
            current_validation_loss=loss,
            current_epoch=epoch,
            best_success_validation_loss={"env": best_loss},
            best_success_epoch={"env": best_epoch},
        )

    def test_higher_success_wins(self):
        result = self.select(0.9, 2.0, 20, 0.8, 1.0, 10)
        self.assertTrue(result["should_save_ckpt"])
        self.assertEqual(result["best_success_rate"]["env"], 0.9)
        self.assertEqual(result["best_success_validation_loss"]["env"], 2.0)

    def test_equal_success_lower_validation_wins(self):
        result = self.select(0.9, 0.8, 20, 0.9, 1.0, 10)
        self.assertTrue(result["should_save_ckpt"])
        self.assertEqual(result["best_success_epoch"]["env"], 20)

    def test_equal_success_and_loss_keeps_earlier_epoch(self):
        result = self.select(0.9, 1.0, 20, 0.9, 1.0, 10)
        self.assertFalse(result["should_save_ckpt"])
        self.assertEqual(result["best_success_epoch"]["env"], 10)


if __name__ == "__main__":
    unittest.main()
