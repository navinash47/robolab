"""Core interfaces — tiny and stable. Ask before changing."""

from robolab.core.arch import Architecture, get_arch, list_archs, register_arch
from robolab.core.evaluator import EvalContext, EvalResult, Evaluator, register_evaluator
from robolab.core.run import DomainParams, RunConfig, RunRecord, RunStatus, TrainerCfg
from robolab.core.sim import SimAdapter, get_sim, list_sims, register_sim
from robolab.core.task import ActSpec, ObsSpec, TaskSpec, get_task, list_tasks, register_task

__all__ = [
    "ActSpec",
    "Architecture",
    "DomainParams",
    "EvalContext",
    "EvalResult",
    "Evaluator",
    "ObsSpec",
    "RunConfig",
    "RunRecord",
    "RunStatus",
    "SimAdapter",
    "TaskSpec",
    "TrainerCfg",
    "get_arch",
    "get_sim",
    "get_task",
    "list_archs",
    "list_sims",
    "list_tasks",
    "register_arch",
    "register_evaluator",
    "register_sim",
    "register_task",
]
