"""SimAdapter ABC + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym

from robolab.core.run import DomainParams
from robolab.core.task import TaskSpec

_SIM_REGISTRY: dict[str, type[SimAdapter]] = {}


def register_sim(name: str):
    def decorator(cls: type[SimAdapter]):
        cls.name = name
        _SIM_REGISTRY[name] = cls
        return cls

    return decorator


def get_sim(name: str) -> SimAdapter:
    if name not in _SIM_REGISTRY:
        raise KeyError(f"Unknown sim '{name}'. Registered: {sorted(_SIM_REGISTRY)}")
    return _SIM_REGISTRY[name]()


def list_sims() -> list[str]:
    return sorted(_SIM_REGISTRY)


class SimAdapter(ABC):
    name: str

    @abstractmethod
    def load_robot(self, urdf_path: Path, **kw: Any) -> Any:
        ...

    @abstractmethod
    def make_env(
        self,
        task: TaskSpec,
        robot: Any,
        domain: DomainParams,
        render: bool = False,
    ) -> gym.Env:
        ...

    @abstractmethod
    def perturb(self, env: gym.Env, params: DomainParams) -> None:
        ...

    @abstractmethod
    def render_frame(self, env: gym.Env) -> Any:
        ...

    @abstractmethod
    def capabilities(self) -> set[str]:
        ...
