"""Cross-platform availability check for the native pathfinding extension."""
from __future__ import annotations

from . import bootstrap  # noqa: F401 - puts src/ on sys.path


def require_pathfinder_backend() -> None:
    """Fail before loading a DEM when the extension is missing or incompatible."""
    try:
        # Python selects .pyd on Windows and .so on macOS/Linux.
        from missile.planning import missile_backend  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "C++ pathfinding engine is unavailable for this Python interpreter. "
            "Build src/missile/planning/cpp with CMake using the same Python "
            "environment that runs frontend/run.py. See frontend/README.md."
        ) from exc
