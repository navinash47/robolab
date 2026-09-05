"""MLP policy/value backbone — one-file plug-in."""

from __future__ import annotations

import torch
from torch import nn

from robolab.core.arch import Architecture, register_arch


@register_arch("mlp")
class MLPPolicyNet(Architecture):
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__(obs_dim, act_dim, cfg)
        hidden = list(self.cfg.get("hidden_sizes", [64, 64]))
        act_name = str(self.cfg.get("activation", "tanh")).lower()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(act_name, nn.Tanh)

        def build(out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = obs_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(act_cls())
                prev = h
            # Ensure latent dims match last hidden
            if prev != out_dim:
                layers.append(nn.Linear(prev, out_dim))
                layers.append(act_cls())
            return nn.Sequential(*layers)

        latent = hidden[-1] if hidden else 64
        self._latent_pi = int(self.cfg.get("latent_dim_pi", latent))
        self._latent_vf = int(self.cfg.get("latent_dim_vf", latent))
        self.pi_net = build(self._latent_pi)
        self.vf_net = build(self._latent_vf)

    @property
    def latent_dim_pi(self) -> int:
        return self._latent_pi

    @property
    def latent_dim_vf(self) -> int:
        return self._latent_vf

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi_net(obs), self.vf_net(obs)
