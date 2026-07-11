"""Quantum-chemistry NARG drivers."""

from __future__ import annotations

from .letta import LETTA


def _load_backends():
    from .bare import NARG as bare_narg
    from .abelian import (
        NARG as abelian_narg,
        energy_groups as abelian_energy_groups_fn,
        hierarchical_kernel as abelian_hierarchical_kernel_fn,
        kernel as abelian_kernel_fn,
        supersite_kernel as abelian_supersite_kernel_fn,
    )
    from .su2 import NARG as su2_narg

    globals()["BareNARG"] = bare_narg
    globals()["AbelianNARG"] = abelian_narg
    globals()["SU2NARG"] = su2_narg
    globals()["abelian_energy_groups"] = abelian_energy_groups_fn
    globals()["abelian_kernel"] = abelian_kernel_fn
    globals()["abelian_hierarchical_kernel"] = abelian_hierarchical_kernel_fn
    globals()["abelian_supersite_kernel"] = abelian_supersite_kernel_fn
    return (
        bare_narg,
        abelian_narg,
        su2_narg,
        abelian_kernel_fn,
        abelian_supersite_kernel_fn,
        abelian_energy_groups_fn,
        abelian_hierarchical_kernel_fn,
    )


class NARG:
    """Bare qchem NARG by default, with optional backend dispatch.

    ``symmetry=None`` selects the bare backend.  ``symmetry="abelian"`` selects
    the U(1)xU(1) backend, and ``symmetry="su2"`` selects the SU(2) backend.
    """

    def __new__(cls, mf, *args, symmetry=None, **kwargs):
        if cls is not NARG:
            return super().__new__(cls)
        bare_narg, abelian_narg, su2_narg, _, _, _, _ = _load_backends()
        if symmetry is None:
            return bare_narg(mf, *args, **kwargs)
        key = str(symmetry).lower().replace("-", "").replace("_", "")
        if key in {"none", "bare", "nosymmetry", "nosym"}:
            return bare_narg(mf, *args, **kwargs)
        if key in {"abelian", "u1", "u1xu1", "u1u1"}:
            return abelian_narg(mf, *args, **kwargs)
        if key in {"su2", "nonabelian"}:
            return su2_narg(mf, *args, **kwargs)
        raise ValueError(
            f"Unknown NARG symmetry {symmetry!r}; expected None, 'abelian', or 'su2'."
        )


def kernel(*args, **kwargs):
    """Run the default Abelian qchem NARG kernel."""
    _, _, _, abelian_kernel_fn, _, _, _ = _load_backends()
    return abelian_kernel_fn(*args, **kwargs)


def hierarchical_kernel(*args, **kwargs):
    """Run the balanced-tree Abelian qchem NARG prototype."""
    _, _, _, _, _, _, abelian_hierarchical_kernel_fn = _load_backends()
    return abelian_hierarchical_kernel_fn(*args, **kwargs)


def supersite_kernel(*args, **kwargs):
    """Run Abelian qchem NARG with explicit composite local supersites."""
    _, _, _, _, abelian_supersite_kernel_fn, _, _ = _load_backends()
    return abelian_supersite_kernel_fn(*args, **kwargs)


def energy_groups(*args, **kwargs):
    """Build energy-ordered groups for Abelian qchem supersite growth."""
    _, _, _, _, _, abelian_energy_groups_fn, _ = _load_backends()
    return abelian_energy_groups_fn(*args, **kwargs)


def __getattr__(name):
    if name in {
        "BareNARG",
        "AbelianNARG",
        "SU2NARG",
        "abelian_energy_groups",
        "abelian_kernel",
        "abelian_hierarchical_kernel",
        "abelian_supersite_kernel",
    }:
        _load_backends()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NARG",
    "LETTA",
    "BareNARG",
    "AbelianNARG",
    "SU2NARG",
    "abelian_energy_groups",
    "abelian_hierarchical_kernel",
    "abelian_kernel",
    "abelian_supersite_kernel",
    "energy_groups",
    "hierarchical_kernel",
    "kernel",
    "supersite_kernel",
]
