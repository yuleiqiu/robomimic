import json
import sys
from types import SimpleNamespace

import pytest

from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger
import robomimic.macros as Macros


def required_config():
    config = config_factory("diffusion_policy")
    with config.unlocked():
        config.experiment.name = "required-test"
        config.experiment.logging.log_wandb = True
        config.experiment.logging.wandb_required = True
        config.experiment.logging.wandb_proj_name = "empty2d_guidance"
    return config


def test_required_wandb_fails_fast(monkeypatch, tmp_path):
    fake = SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("auth")))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(Macros, "WANDB_ENTITY", "entity")
    with pytest.raises(RuntimeError, match="Required online W&B initialization failed"):
        DataLogger(tmp_path, required_config(), log_tb=False, log_wandb=True)


def test_required_wandb_writes_run_checkpoint_mapping(monkeypatch, tmp_path):
    run = SimpleNamespace(
        id="run-id",
        url="https://wandb.example/run-id",
        settings=SimpleNamespace(mode="online"),
    )
    fake = SimpleNamespace(
        init=lambda **kwargs: run,
        config=SimpleNamespace(update=lambda values, **kwargs: None),
        run=run,
        log=lambda *args, **kwargs: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(Macros, "WANDB_ENTITY", "entity")
    logger = DataLogger(tmp_path, required_config(), log_tb=False, log_wandb=True)
    logger.record_checkpoint(tmp_path / "model_epoch_100.pth", epoch=100, kind="scheduled")
    manifest = json.loads((tmp_path / "wandb_run.json").read_text())
    assert manifest["run_id"] == "run-id"
    assert manifest["mode"] == "online"
    assert manifest["checkpoints"][0]["epoch"] == 100
