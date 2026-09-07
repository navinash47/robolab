"""API route modules."""

from robolab_api.routes.architectures import router as architectures_router
from robolab_api.routes.compare import router as compare_router
from robolab_api.routes.costs import router as costs_router
from robolab_api.routes.failures import router as failures_router
from robolab_api.routes.runs import router as runs_router

__all__ = [
    "architectures_router",
    "runs_router",
    "compare_router",
    "costs_router",
    "failures_router",
]
