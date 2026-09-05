"""Evaluator ABC + registry (stubs for Phase 1; used heavily from Phase 6+)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

_EVAL_REGISTRY: dict[str, type[Evaluator]] = {}


def register_evaluator(name: str):
    def decorator(cls: type[Evaluator]):
        cls.name = name
        _EVAL_REGISTRY[name] = cls
        return cls

    return decorator


def get_evaluator(name: str) -> type[Evaluator]:
    if name not in _EVAL_REGISTRY:
        raise KeyError(f"Unknown evaluator '{name}'. Registered: {sorted(_EVAL_REGISTRY)}")
    return _EVAL_REGISTRY[name]


class EvalContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    extras: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    name: str
    value: float
    ci: tuple[float, float] | None = None
    per_seed: list[float] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class Evaluator(ABC):
    name: str
    requires: set[str] = set()

    @abstractmethod
    def run(self, ctx: EvalContext) -> EvalResult:
        ...
