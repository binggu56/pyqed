"""Top-level package for :mod:`pyqed`.

Keep this initializer lightweight.  Subpackages such as ``pyqed.pbc.gw`` should
not pay the import cost of plotting, qchem, MPS, or optional accelerator stacks
just because Python imports the top-level package first.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np

from ._version import __version__
from .units import *  # noqa: F401,F403


def dagger(a):
    return a.conjugate().transpose()


def dag(a):
    return a.conjugate().transpose()


def is_positive_def(a):
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False


_LAZY_ATTRS = {
    "view": ("pyqed.visualization", "view"),
    "MoleculeView": ("pyqed.visualization", "MoleculeView"),
    "ScalarField3D": ("pyqed.visualization", "ScalarField3D"),
    "SceneView": ("pyqed.visualization", "SceneView"),
    "VolumeView": ("pyqed.visualization", "VolumeView"),
    "Mol": ("pyqed.mol", "Mol"),
    "Result": ("pyqed.mol", "Result"),
    "Cavity": ("pyqed.cavity", "Cavity"),
    "Composite": ("pyqed.cavity", "Composite"),
    "SineDVR": ("pyqed.dvr.dvr_1d", "SineDVR"),
    "Molecule": ("pyqed.qchem", "Molecule"),
    "GW": ("pyqed.gw", "GW"),
    "BSE": ("pyqed.gw", "BSE"),
    "TDA": ("pyqed.gw", "TDA"),
    "commutator": ("pyqed.phys", "commutator"),
    "anticommutator": ("pyqed.phys", "anticommutator"),
    "comm": ("pyqed.phys", "comm"),
    "anticomm": ("pyqed.phys", "anticomm"),
    "discretize": ("pyqed.phys", "discretize"),
    "sort": ("pyqed.phys", "sort"),
    "polar2cartesian": ("pyqed.phys", "polar2cartesian"),
    "rk4": ("pyqed.phys", "rk4"),
    "tensor": ("pyqed.phys", "tensor"),
    "obs": ("pyqed.phys", "obs"),
    "obs_dm": ("pyqed.phys", "obs_dm"),
    "overlap": ("pyqed.phys", "overlap"),
    "interval": ("pyqed.phys", "interval"),
    "meshgrid": ("pyqed.phys", "meshgrid"),
    "norm2": ("pyqed.phys", "norm2"),
    "sinc": ("pyqed.phys", "sinc"),
    "gwp": ("pyqed.phys", "gwp"),
    "ket2dm": ("pyqed.phys", "ket2dm"),
    "expect": ("pyqed.phys", "expect"),
    "destroy": ("pyqed.phys", "destroy"),
    "transform": ("pyqed.phys", "transform"),
    "basis_transform": ("pyqed.phys", "basis_transform"),
    "cartesian_product": ("pyqed.phys", "cartesian_product"),
    "eigh": ("pyqed.phys", "eigh"),
    "expm": ("pyqed.phys", "expm"),
    "coth": ("pyqed.phys", "coth"),
    "basis": ("pyqed.phys", "basis"),
    "boson": ("pyqed.phys", "boson"),
    "driven_dynamics": ("pyqed.phys", "driven_dynamics"),
    "pauli": ("pyqed.phys", "pauli"),
    "sigmax": ("pyqed.phys", "sigmax"),
    "sigmay": ("pyqed.phys", "sigmay"),
    "sigmaz": ("pyqed.phys", "sigmaz"),
    "isherm": ("pyqed.phys", "isherm"),
    "isunitary": ("pyqed.phys", "isunitary"),
    "householder": ("pyqed.phys", "householder"),
    "dominant_eig": ("pyqed.jax_eigs", "dominant_eig"),
    "dominant_eigval": ("pyqed.jax_eigs", "dominant_eigval"),
    "isdiag": ("pyqed.phys", "isdiag"),
    "hadamard": ("pyqed.qip", "hadamard"),
    "SpinHalfFermionOperators": (
        "pyqed.qchem.jordan_wigner.spinful",
        "SpinHalfFermionOperators",
    ),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
