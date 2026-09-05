"""Task package — import registers tasks."""

from robolab.tasks import docking as docking  # noqa: F401
from robolab.tasks import figure8_tracking as figure8_tracking  # noqa: F401
from robolab.tasks import go_to_goal as go_to_goal  # noqa: F401
from robolab.tasks import maze as maze  # noqa: F401
from robolab.tasks import wall_follow as wall_follow  # noqa: F401

__all__ = [
    "docking",
    "figure8_tracking",
    "go_to_goal",
    "maze",
    "wall_follow",
]
