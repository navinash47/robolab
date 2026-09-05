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


def list_sim_details() -> dict[str, dict[str, list[str]]]:
    """name → {capabilities: [...]} for UI capability flags."""
    out: dict[str, dict[str, list[str]]] = {}
    for name, cls in sorted(_SIM_REGISTRY.items()):
        try:
            caps = sorted(cls().capabilities())
        except Exception:
            caps = ["error"]
        out[name] = {"capabilities": caps}
    return out


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
