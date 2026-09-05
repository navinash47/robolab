"""SB3 ActorCriticPolicy that pulls its backbone from the architecture registry."""

from __future__ import annotations

from typing import Any, Callable

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from robolab.core.arch import Architecture, get_arch


class ArchExtractor(nn.Module):
    """Wraps a RoboLab Architecture as SB3's mlp_extractor."""

    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch
        self.latent_dim_pi = arch.latent_dim_pi
        self.latent_dim_vf = arch.latent_dim_vf

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.arch(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.arch(features)[0]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.arch(features)[1]


class RoboLabActorCriticPolicy(ActorCriticPolicy):
    """Custom policy: Architecture registry → pi/vf latents → SB3 action/value heads."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args: Any,
        arch_name: str = "mlp",
        arch_cfg: dict | None = None,
        **kwargs: Any,
    ):
        self._arch_name = arch_name
        self._arch_cfg = arch_cfg or {}
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        obs_dim = int(self.features_dim)
        act_dim = int(spaces.flatdim(self.action_space))
        arch_cls = get_arch(self._arch_name)
        arch = arch_cls(obs_dim=obs_dim, act_dim=act_dim, cfg=dict(self._arch_cfg))
        self.mlp_extractor = ArchExtractor(arch)
