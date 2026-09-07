"""Architecture package — import registers archs."""

from robolab.archs import avinash_wall as avinash_wall  # noqa: F401
from robolab.archs import fan as fan  # noqa: F401
from robolab.archs import gpkan as gpkan  # noqa: F401
from robolab.archs import kaf as kaf  # noqa: F401
from robolab.archs import kan as kan  # noqa: F401
from robolab.archs import mlp as mlp  # noqa: F401

__all__ = ["mlp", "kan", "kaf", "gpkan", "fan", "avinash_wall"]
