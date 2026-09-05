"""Architecture ABC + registry decorator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

_ARCH_REGISTRY: dict[str, type[Architecture]] = {}


def register_arch(name: str):
    def decorator(cls: type[Architecture]):
        cls.name = name
        _ARCH_REGISTRY[name] = cls
        return cls

    return decorator


def get_arch(name: str) -> type[Architecture]:
    if name not in _ARCH_REGISTRY:
        import robolab.archs  # noqa: F401

    if name not in _ARCH_REGISTRY:
        raise KeyError(f"Unknown arch '{name}'. Registered: {sorted(_ARCH_REGISTRY)}")
    return _ARCH_REGISTRY[name]


def list_archs() -> list[str]:
    import robolab.archs  # noqa: F401

    return sorted(_ARCH_REGISTRY)


class Architecture(nn.Module, ABC):
    """Policy/value feature backbone. One file per arch under robolab/archs/."""

    name: str

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg or {}

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pi_features, vf_features)."""

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def latent_dim_pi(self) -> int:
        return int(self.cfg.get("latent_dim_pi", self.cfg.get("hidden_sizes", [64, 64])[-1]))

    @property
    def latent_dim_vf(self) -> int:
        return int(self.cfg.get("latent_dim_vf", self.cfg.get("hidden_sizes", [64, 64])[-1]))
