"""Default worker budget for PyQED calculations.

Worker pools are independent of OpenMP and BLAS thread counts. This module
does not change numerical-library thread settings or resize existing pools.
"""

import json
import operator
import os
from pathlib import Path


_workers = None


def _positive_workers(value):
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError("workers must be a positive integer") from None
    # operator.index accepts numpy integers without silently truncating floats.
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError("workers must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError("workers must be a positive integer") from None
    if value < 1:
        raise ValueError("workers must be a positive integer")
    return value


def _config_path():
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "pyqed" / "parallel.json"


def num_workers(n=None):
    """Get or set the default number of calculation workers.

    ``num_workers(4)`` sets the process default and returns 4. With no argument,
    precedence is the process setting, ``PYQED_NUM_WORKERS``, the ``workers``
    value in ``~/.config/pyqed/parallel.json`` (or under ``XDG_CONFIG_HOME``),
    then 1. Explicit calculation arguments override this default.

    The setter also exports ``PYQED_NUM_WORKERS`` so newly spawned processes
    inherit the setting. Existing solvers and worker pools retain their resolved
    budget. BLAS/OpenMP threads are configured separately.
    """
    global _workers
    if n is not None:
        _workers = _positive_workers(n)
        os.environ["PYQED_NUM_WORKERS"] = str(_workers)
    if _workers is not None:
        return _workers
    value = os.environ.get("PYQED_NUM_WORKERS")
    if value is None:
        path = _config_path()
        try:
            value = json.loads(path.read_text()).get("workers", 1)
        except FileNotFoundError:
            value = 1
    return _positive_workers(value)


def resolve_workers(workers=None):
    """Resolve an explicit worker count or the current default."""
    return num_workers() if workers is None else _positive_workers(workers)
