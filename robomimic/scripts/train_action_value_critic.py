"""
Training script for the Action Value Critic.

Trains C_phi(o_t, A_t) ≈ P(S = 1 | o_t, A_t) using rollout trajectories
with success/failure labels.

Example usage:

    python train_action_value_critic.py \
        --config /path/to/critic_config.json \
        --dataset /path/to/rollouts.hdf5 \
        --policy_ckpt /path/to/diffusion_policy.pth
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config.critic_config import CriticConfig
from robomimic.models.critic_nets import ActionValueCritic, create_obs_encoder_from_checkpoint
from robomimic.utils.critic_dataset import ActionValueCriticDataset
from robomimic.utils.log_utils import DataLogger, PrintLogger


def compute_metrics(preds, labels):
    """Compute accuracy and optionally ROC-AUC."""
    pred_labels = (preds > 0.5).float()
    accuracy = (pred_labels == labels).float().mean().item()

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acc = (pred_labels[pos_mask] == labels[pos_mask]).float().mean().item() if pos_mask.sum() > 0 else 0.0
    neg_acc = (pred_labels[neg_mask] == labels[neg_mask]).float().mean().item() if neg_mask.sum() > 0 else 0.0

    roc_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            roc_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    except ImportError:
        pass

    return {
        "accuracy": accuracy,
        "pos_accuracy": pos_acc,
        "neg_accuracy": neg_acc,
        "roc_auc": roc_auc,
    }


def train_epoch(model, data_loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in data_loader:
        obs_dict = {k: v.to(device) for k, v in batch["obs"].items()}
        action_chunk = batch["action_chunk"].to(device)
        labels = batch["success"].to(device)

        optimizer.zero_grad()
        logits = model(obs_dict, action_chunk).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach())
        all_labels.append(labels.detach())
        num_batches += 1

    avg_loss = total_loss / num_batches
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def validate(model, data_loader, criterion, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in data_loader:
        obs_dict = {k: v.to(device) for k, v in batch["obs"].items()}
        action_chunk = batch["action_chunk"].to(device)
        labels = batch["success"].to(device)

        logits = model(obs_dict, action_chunk).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits))
        all_labels.append(labels)
        num_batches += 1

    avg_loss = total_loss / num_batches
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = avg_loss

    return metrics


def save_critic_checkpoint(model, config, obs_shapes, action_dim, ckpt_path, epoch, best_val_loss, policy_ckpt_path=None):
    """Save critic checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "obs_shapes": dict(obs_shapes),
        "action_dim": action_dim,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "policy_ckpt_path": policy_ckpt_path,
    }
    torch.save(checkpoint, ckpt_path)
    print("Saved checkpoint to {}".format(ckpt_path))


def load_critic_checkpoint(ckpt_path, device=None, policy_ckpt_path=None):
    """Load critic checkpoint.
    
    Args:
        ckpt_path: path to critic checkpoint
        device: torch device
        policy_ckpt_path: path to base policy checkpoint (needed to recreate obs_encoder).
            If not provided, will try to use the path stored in the critic checkpoint.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    obs_shapes = OrderedDict(checkpoint["obs_shapes"])
    action_dim = checkpoint["action_dim"]
    action_horizon = checkpoint["config"]["train"]["action_horizon"]
    hidden_dim = checkpoint["config"]["train"]["hidden_dim"]
    freeze_obs_encoder = checkpoint["config"]["train"].get("freeze_obs_encoder", True)

    if policy_ckpt_path is None:
        policy_ckpt_path = checkpoint.get("policy_ckpt_path", None)
    
    if policy_ckpt_path is None:
        raise ValueError(
            "policy_ckpt_path is required to load critic checkpoint. "
            "Either pass it as argument or save it in the critic checkpoint."
        )

    policy_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=policy_ckpt_path)
    obs_encoder, _ = create_obs_encoder_from_checkpoint(policy_ckpt_dict, device=device)

    policy_config, _ = FileUtils.config_from_checkpoint(
        algo_name=policy_ckpt_dict["algo_name"], ckpt_dict=policy_ckpt_dict
    )
    observation_horizon = policy_config.algo.horizon.observation_horizon

    model = ActionValueCritic(
        obs_shapes=obs_shapes,
        action_dim=action_dim,
        action_horizon=action_horizon,
        obs_encoder=obs_encoder,
        observation_horizon=observation_horizon,
        hidden_dim=hidden_dim,
        freeze_obs_encoder=freeze_obs_encoder,
    )

    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)

    return model, checkpoint


def log_metrics(data_logger, prefix, metrics, epoch):
    """Log a metrics dict with native robomimic logger naming."""
    for k, v in metrics.items():
        if v is not None:
            data_logger.record("{}/{}".format(prefix, k), v, epoch)


def train(args, config):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=args.cuda)
    print("Using device: {}".format(device))

    os.makedirs(args.output_dir, exist_ok=True)
    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(args.output_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    print("\nLoading base policy checkpoint...")
    policy_ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.policy_ckpt)
    obs_encoder, obs_shapes = create_obs_encoder_from_checkpoint(policy_ckpt_dict, device=device)
    print("Loaded obs encoder with output shape: {}".format(obs_encoder.output_shape()))

    policy_config, _ = FileUtils.config_from_checkpoint(
        algo_name=policy_ckpt_dict["algo_name"], ckpt_dict=policy_ckpt_dict
    )
    observation_horizon = policy_config.algo.horizon.observation_horizon
    print("Observation horizon from policy: {}".format(observation_horizon))

    shape_meta = FileUtils._select_checkpoint_metadata(
        policy_ckpt_dict["shape_metadata"], "shape_metadata"
    )
    action_dim = shape_meta["ac_dim"]

    policy_obs_keys = list(shape_meta["all_obs_keys"])
    if args.obs_keys is None:
        obs_keys = policy_obs_keys
        print("Using all obs keys from policy checkpoint: {}".format(obs_keys))
    else:
        obs_keys = args.obs_keys

    print("\nLoading rollout dataset...")
    dataset_paths = args.dataset if isinstance(args.dataset, list) else [args.dataset]

    train_filter = args.train_filter_key or "train"
    val_filter = args.val_filter_key or "valid"

    print("Using trajectory-level split: train_filter='{}', val_filter='{}'".format(
        train_filter, val_filter
    ))

    train_dataset = ActionValueCriticDataset(
        dataset_paths=dataset_paths,
        obs_keys=obs_keys,
        action_horizon=args.action_horizon,
        observation_horizon=observation_horizon,
        filter_key=train_filter,
    )
    val_dataset = ActionValueCriticDataset(
        dataset_paths=dataset_paths,
        obs_keys=obs_keys,
        action_horizon=args.action_horizon,
        observation_horizon=observation_horizon,
        filter_key=val_filter,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_data_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_data_workers,
    )

    print("\nCreating critic model...")
    model = ActionValueCritic(
        obs_shapes=obs_shapes,
        action_dim=action_dim,
        action_horizon=args.action_horizon,
        obs_encoder=obs_encoder,
        observation_horizon=observation_horizon,
        hidden_dim=args.hidden_dim,
        freeze_obs_encoder=args.freeze_obs_encoder,
    ).to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config_dict = {
        "algo_name": config.algo_name,
        "experiment": config.experiment.to_dict(),
        "train": {
            "rollout_dataset_paths": dataset_paths,
            "base_policy_ckpt": args.policy_ckpt,
            "action_horizon": args.action_horizon,
            "obs_keys": obs_keys,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "train_filter_key": train_filter,
            "val_filter_key": val_filter,
            "freeze_obs_encoder": args.freeze_obs_encoder,
            "num_data_workers": args.num_data_workers,
            "seed": args.seed,
            "cuda": args.cuda,
        },
        "obs_shapes": dict(obs_shapes),
        "action_dim": action_dim,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    data_logger = DataLogger(
        args.output_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )

    print("\nStarting training...")
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        print("Epoch {}/{} - Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.2%}".format(
            epoch, args.num_epochs,
            train_metrics["loss"], val_metrics["loss"], val_metrics["accuracy"]
        ))

        if val_metrics["roc_auc"] is not None:
            print("  Val ROC-AUC: {:.4f}, Pos Acc: {:.2%}, Neg Acc: {:.2%}".format(
                val_metrics["roc_auc"], val_metrics["pos_accuracy"], val_metrics["neg_accuracy"]
            ))

        log_metrics(data_logger, "Train", train_metrics, epoch)
        log_metrics(data_logger, "Valid", val_metrics, epoch)
        data_logger.record("Timing_Stats/Epoch", time.time() - start_time, epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_critic_checkpoint(
                model=model,
                config=config_dict,
                obs_shapes=obs_shapes,
                action_dim=action_dim,
                ckpt_path=os.path.join(args.output_dir, "critic_best.pth"),
                epoch=epoch,
                best_val_loss=best_val_loss,
                policy_ckpt_path=args.policy_ckpt,
            )

        if epoch % args.save_every_n_epochs == 0:
            save_critic_checkpoint(
                model=model,
                config=config_dict,
                obs_shapes=obs_shapes,
                action_dim=action_dim,
                ckpt_path=os.path.join(args.output_dir, "critic_epoch_{}.pth".format(epoch)),
                epoch=epoch,
                best_val_loss=best_val_loss,
                policy_ckpt_path=args.policy_ckpt,
            )

    save_critic_checkpoint(
        model=model,
        config=config_dict,
        obs_shapes=obs_shapes,
        action_dim=action_dim,
        ckpt_path=os.path.join(args.output_dir, "critic_last.pth"),
        epoch=args.num_epochs,
        best_val_loss=best_val_loss,
        policy_ckpt_path=args.policy_ckpt,
    )

    elapsed = time.time() - start_time
    print("\nTraining completed in {:.1f}s".format(elapsed))
    print("Best validation loss: {:.4f}".format(best_val_loss))
    data_logger.close()


def build_config(args, ext_cfg=None):
    config = CriticConfig()
    with config.unlocked():
        if ext_cfg is not None:
            config.update(ext_cfg)

        config.experiment.logging.log_wandb = args.wandb
        config.experiment.logging.wandb_proj_name = args.wandb_project
        if args.wandb_name is not None:
            config.experiment.name = args.wandb_name
        config.experiment.output_dir = args.output_dir

        config.train.rollout_dataset_paths = args.dataset
        config.train.base_policy_ckpt = args.policy_ckpt
        config.train.action_horizon = args.action_horizon
        config.train.obs_keys = args.obs_keys
        config.train.batch_size = args.batch_size
        config.train.num_epochs = args.num_epochs
        config.train.learning_rate = args.learning_rate
        config.train.weight_decay = args.weight_decay
        config.train.hidden_dim = args.hidden_dim
        config.train.train_filter_key = args.train_filter_key
        config.train.val_filter_key = args.val_filter_key
        config.train.freeze_obs_encoder = args.freeze_obs_encoder
        config.train.num_data_workers = args.num_data_workers
        config.train.seed = args.seed
        config.train.cuda = args.cuda

    config.lock()
    return config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="path to JSON config file")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="path(s) to rollout HDF5 file(s)")
    parser.add_argument("--policy_ckpt", type=str, default=None,
                        help="path to base policy checkpoint")
    parser.add_argument("--output_dir", type=str, default="./critic_train_output",
                        help="output directory for checkpoints and logs")

    parser.add_argument("--obs_keys", type=str, nargs="+", default=None,
                        help="observation keys to use (default: all keys from policy checkpoint)")
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="action chunk length H")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--freeze_obs_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_obs_encoder", dest="freeze_obs_encoder", action="store_false")
    parser.add_argument("--num_data_workers", type=int, default=2)
    parser.add_argument("--save_every_n_epochs", type=int, default=10)
    parser.add_argument("--train_filter_key", type=str, default=None,
                        help="mask key for training demos (default: 'train')")
    parser.add_argument("--val_filter_key", type=str, default=None,
                        help="mask key for validation demos (default: 'valid')")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="action_value_critic",
                        help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="wandb run name (default: experiment name from config)")

    args = parser.parse_args()

    ext_cfg = None
    if args.config is not None:
        with open(args.config, 'r') as f:
            ext_cfg = json.load(f)
        train_config = ext_cfg.get("train", {})
        experiment_config = ext_cfg.get("experiment", {})

        if args.dataset is None and "rollout_dataset_paths" in train_config:
            args.dataset = train_config["rollout_dataset_paths"]
        if args.policy_ckpt is None and "base_policy_ckpt" in train_config:
            args.policy_ckpt = train_config["base_policy_ckpt"]
        if args.output_dir == "./critic_train_output" and "output_dir" in experiment_config:
            args.output_dir = experiment_config["output_dir"]

        for key in ["action_horizon", "batch_size", "num_epochs", "learning_rate",
                    "weight_decay", "hidden_dim", "num_data_workers", "seed",
                    "train_filter_key", "val_filter_key", "obs_keys"]:
            if key in train_config:
                setattr(args, key, train_config[key])

        if "freeze_obs_encoder" in train_config:
            args.freeze_obs_encoder = train_config["freeze_obs_encoder"]
        if "cuda" in train_config:
            args.cuda = train_config["cuda"]
        if "save" in experiment_config and "every_n_epochs" in experiment_config["save"]:
            args.save_every_n_epochs = experiment_config["save"]["every_n_epochs"]

        logging_config = experiment_config.get("logging", {})
        if "log_wandb" in logging_config:
            args.wandb = logging_config["log_wandb"]
        if "wandb_proj_name" in logging_config:
            args.wandb_project = logging_config["wandb_proj_name"]

    if args.dataset is None:
        parser.error("--dataset or config.train.rollout_dataset_paths is required")
    if args.policy_ckpt is None:
        parser.error("--policy_ckpt or config.train.base_policy_ckpt is required")

    config = build_config(args, ext_cfg=ext_cfg)
    train(args, config)


if __name__ == "__main__":
    main()
