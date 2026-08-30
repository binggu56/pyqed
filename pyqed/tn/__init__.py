"""Canonical dense tensor-network data structures and Hamiltonian builders."""

from .mpo import MPO
from .hamiltonian import Hamiltonian, LocalHamiltonian, LocalTerm, OperatorString
from .tree import TTN
from .effective_operator import PackedBlockEffectiveOperator, resolve_workers


def __getattr__(name):
    if name == "MPS":
        from pyqed.mps.mps import MPS

        return MPS
    raise AttributeError(name)


__all__ = [
    "Hamiltonian",
    "LocalTerm",
    "MPO",
    "MPS",
    "OperatorString",
    "PackedBlockEffectiveOperator",
    "resolve_workers",
    "TTN",
]
