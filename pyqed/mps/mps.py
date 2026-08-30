"""Public compatibility façade for finite MPS and DMRG functionality.

Implementations are split by responsibility:

* :mod:`._mps_state` owns MPS/MPO state operations.
* :mod:`._abelian_local_engine` owns block-sparse local actions.
* :mod:`._moving_environment` owns environment caching and compiled backends.
* :mod:`._dmrg_sweep` owns finite-system sweep orchestration.

The façade retains historical imports and synchronizes monkeypatched solver
hooks before entering the sweep implementation.
"""

from ._mps_common import *
from ._mps_state import *
from ._abelian_local_engine import *
from ._moving_environment import *
from . import _dmrg_sweep as _sweep
from ._dmrg_sweep import *


_optimize_two_sites_impl = _sweep.optimize_two_sites
_two_site_dmrg_impl = _sweep.two_site_dmrg


def _sync_sweep_globals():
    for name, value in globals().items():
        if (
            name.startswith("__")
            or name in {"optimize_two_sites", "two_site_dmrg"}
            or name not in _sweep.__dict__
        ):
            continue
        setattr(_sweep, name, value)


def _copy_optimize_metadata(source):
    for name, value in vars(source).items():
        setattr(_optimize_two_sites_wrapper, name, value)
        selected = globals().get("optimize_two_sites")
        if callable(selected):
            setattr(selected, name, value)


def optimize_two_sites(*args, **kwargs):
    """Compatibility wrapper around the two-site local optimizer."""
    _sync_sweep_globals()
    _sweep.optimize_two_sites = _optimize_two_sites_impl
    try:
        return _optimize_two_sites_impl(*args, **kwargs)
    finally:
        _copy_optimize_metadata(_optimize_two_sites_impl)


_optimize_two_sites_wrapper = optimize_two_sites


def two_site_dmrg(*args, **kwargs):
    """Compatibility wrapper around the finite-system DMRG sweep."""
    _sync_sweep_globals()
    selected_optimizer = globals()["optimize_two_sites"]
    if selected_optimizer is _optimize_two_sites_wrapper:
        selected_optimizer = _optimize_two_sites_impl
    _sweep.optimize_two_sites = selected_optimizer
    try:
        return _two_site_dmrg_impl(*args, **kwargs)
    finally:
        _copy_optimize_metadata(_optimize_two_sites_impl)
        if selected_optimizer is not _optimize_two_sites_impl:
            for name, value in vars(_optimize_two_sites_wrapper).items():
                setattr(selected_optimizer, name, value)
        _sweep.optimize_two_sites = _optimize_two_sites_impl
