"""TaskSpec contract + task registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_TASK_REGISTRY: dict[str, Callable[[], TaskSpec]] = {}


def register_task(name: str):
    def decorator(factory: Callable[[], TaskSpec]):
        _TASK_REGISTRY[name] = factory
        return factory

    return decorator


def get_task(name: str) -> TaskSpec:
    if name not in _TASK_REGISTRY:
        # Import side-effect registration
        import robolab.tasks  # noqa: F401

    if name not in _TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Registered: {sorted(_TASK_REGISTRY)}")
    return _TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    import robolab.tasks  # noqa: F401

    return sorted(_TASK_REGISTRY)


class ObsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[str]
    shape: tuple[int, ...]
    low: float = 0.0
    high: float = 10.0
    dtype: str = "float32"
    notes: str = ""


class ActSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, ...]
    low: float = -1.0
    high: float = 1.0
    dtype: str = "float32"
    continuous: bool = True
    notes: str = ""


# Callables live on the concrete task modules; TaskSpec holds references.
RewardFn = Callable[[dict[str, Any]], float]
TermFn = Callable[[dict[str, Any]], bool]
SuccessFn = Callable[[dict[str, Any]], bool]


class TaskSpec(BaseModel):
    """Observation / action / reward / termination contract.

    Reward/termination/success callables are excluded from serialization
    (they live in Python task modules, never in YAML).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    observation: ObsSpec
    action: ActSpec
    max_steps: int = 500
    reward: Any = Field(exclude=True, default=None)
    termination: Any = Field(exclude=True, default=None)
    success: Any = Field(exclude=True, default=None)
    meta: dict[str, Any] = Field(default_factory=dict)
