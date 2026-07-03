"""Utilities for OSC action -> EEF trajectory forward models."""

from pathlib import Path

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OSCForwardModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        horizon,
        state_embed_dim=128,
        action_embed_dim=256,
        hidden_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.state_net = nn.Sequential(
            MLPBlock(state_dim, 64, dropout),
            MLPBlock(64, state_embed_dim, dropout),
        )
        self.action_net = nn.Sequential(
            MLPBlock(action_dim, action_embed_dim, dropout),
            MLPBlock(action_embed_dim, action_embed_dim, dropout),
        )
        fusion_dim = state_embed_dim + action_embed_dim
        self.fusion_net = nn.Sequential(
            MLPBlock(fusion_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, hidden_dim, dropout),
            MLPBlock(hidden_dim, 256, dropout),
            nn.Linear(256, self.horizon * 3),
        )

    def forward(self, state, action):
        state_emb = self.state_net(state)
        action_emb = self.action_net(action)
        out = self.fusion_net(torch.cat([state_emb, action_emb], dim=-1))
        return out.view(out.shape[0], self.horizon, 3)


def _as_tensor_stats(stats, device):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in stats.items()
    }


class LoadedOSCForwardModel:
    def __init__(self, model, stats, config, metrics, device):
        self.model = model
        self.stats = stats
        self.config = config
        self.metrics = metrics
        self.device = device
        self.horizon = int(config["resolved_horizon"])

    def predict_delta_traj(self, state, action_chunk):
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action_chunk [B,H,A], got {tuple(action_chunk.shape)}")
        if action_chunk.shape[1] < self.horizon:
            raise ValueError(f"Expected at least horizon {self.horizon}, got {action_chunk.shape[1]}")

        state = state.to(device=self.device, dtype=torch.float32)
        action_flat = action_chunk[:, : self.horizon].to(device=self.device, dtype=torch.float32).reshape(action_chunk.shape[0], -1)
        state_norm = (state - self.stats["state_mean"]) / self.stats["state_std"]
        action_norm = (action_flat - self.stats["action_mean"]) / self.stats["action_std"]
        pred_norm = self.model(state_norm, action_norm)
        return pred_norm * self.stats["target_std"] + self.stats["target_mean"]

    def predict_abs_traj(self, state, action_chunk):
        delta_traj = self.predict_delta_traj(state=state, action_chunk=action_chunk)
        current_eef = state[:, :3].to(device=self.device, dtype=torch.float32)
        return current_eef[:, None, :] + delta_traj


def load_osc_forward_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    metrics = ckpt["metrics"]
    model_config = metrics["model_config"]
    model = OSCForwardModel(
        state_dim=int(model_config["state_dim"]),
        action_dim=int(model_config["action_dim"]),
        horizon=int(config["resolved_horizon"]),
        state_embed_dim=int(model_config["state_embed_dim"]),
        action_embed_dim=int(model_config["action_embed_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = _as_tensor_stats(ckpt["stats"], device=device)
    return LoadedOSCForwardModel(model=model, stats=stats, config=config, metrics=metrics, device=device)
