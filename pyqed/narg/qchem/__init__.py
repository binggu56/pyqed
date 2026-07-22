"""Quantum-chemistry NARG drivers."""

from __future__ import annotations

import numpy as np

from pyqed.narg.hamiltonian import (
    IntegralHamiltonian,
    MPOHamiltonian,
    normalize_symmetry,
    normalize_orbital_blocks,
)

from .letta import LETTA
from .orbopt import NARGOpt, NARGSCF


def _load_bare_backend():
    from .bare import NARG as bare_narg

    globals()["BareNARG"] = bare_narg
    return bare_narg


def _load_abelian_backend():
    from .abelian import (
        NARG as abelian_narg,
        energy_groups as abelian_energy_groups_fn,
        hierarchical_kernel as abelian_hierarchical_kernel_fn,
        kernel as abelian_kernel_fn,
        supersite_kernel as abelian_supersite_kernel_fn,
    )

    globals()["AbelianNARG"] = abelian_narg
    globals()["abelian_energy_groups"] = abelian_energy_groups_fn
    globals()["abelian_kernel"] = abelian_kernel_fn
    globals()["abelian_hierarchical_kernel"] = abelian_hierarchical_kernel_fn
    globals()["abelian_supersite_kernel"] = abelian_supersite_kernel_fn
    return (
        abelian_narg,
        abelian_kernel_fn,
        abelian_supersite_kernel_fn,
        abelian_energy_groups_fn,
        abelian_hierarchical_kernel_fn,
    )


def _load_su2_backend():
    from .su2 import NARG as su2_narg

    globals()["SU2NARG"] = su2_narg
    return su2_narg


def _load_backends():
    bare_narg = _load_bare_backend()
    (
        abelian_narg,
        abelian_kernel_fn,
        abelian_supersite_kernel_fn,
        abelian_energy_groups_fn,
        abelian_hierarchical_kernel_fn,
    ) = _load_abelian_backend()
    su2_narg = _load_su2_backend()

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


def _reference_nelectron(mf):
    mol = getattr(mf, "mol", None)
    for obj in (mf, mol):
        if obj is None:
            continue
        for name in ("nelec", "nelectron"):
            value = getattr(obj, name, None)
            if value is not None:
                return int(np.sum(np.asarray(value, dtype=int).reshape(-1)))
    return None


def _nelecas_count(nelecas):
    if nelecas is None:
        return None
    return int(np.sum(np.asarray(nelecas, dtype=int).reshape(-1)))


def _infer_ncore(mf, nelecas, ncore):
    if ncore is not None:
        return int(ncore)
    active_electrons = _nelecas_count(nelecas)
    total_electrons = _reference_nelectron(mf)
    if active_electrons is None or total_electrons is None:
        return 0
    core_electrons = total_electrons - active_electrons
    if core_electrons < 0 or core_electrons % 2:
        raise ValueError(
            "Cannot infer ncore: total electrons minus active electrons must be a "
            "non-negative even number."
        )
    return core_electrons // 2


def _as_spatial_mo_coeff(mf, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("NARG active workflow needs mo_coeff; run SCF or pass mo_coeff=...")
    if isinstance(mo_coeff, (tuple, list)):
        raise NotImplementedError("High-level NARG active workflow currently expects restricted spatial orbitals.")
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array.")
    return mo_coeff


def _active_indices_from_options(mf, kwargs, mo_coeff):
    active = kwargs.pop("active", kwargs.pop("active_orbitals", None))
    ncore = kwargs.get("ncore", None)
    ncas = kwargs.get("ncas", None)
    if active is None:
        if ncas is not None:
            start = 0 if ncore is None else int(ncore)
            active = range(start, start + int(ncas))
        else:
            active = range(mo_coeff.shape[1])
    active = tuple(int(i) for i in active)
    if not active:
        raise ValueError("active must contain at least one orbital.")
    if len(set(active)) != len(active):
        raise ValueError("active cannot contain duplicate orbital indices.")
    if min(active) < 0 or max(active) >= mo_coeff.shape[1]:
        raise ValueError("active contains orbital indices outside mo_coeff.")
    return active


def _reorder_mo_for_active(mo_coeff, active, ncore):
    nmo = mo_coeff.shape[1]
    active = tuple(int(i) for i in active)
    rest = [i for i in range(nmo) if i not in set(active)]
    ncore = int(ncore)
    if ncore < 0:
        raise ValueError("ncore must be non-negative.")
    if ncore > len(rest):
        raise ValueError("ncore leaves no room for the requested active orbitals.")
    order = tuple(rest[:ncore] + list(active) + rest[ncore:])
    return np.array(mo_coeff[:, order], copy=True), order


def _localize_active_orbitals(mf, mo_coeff, active, localization, localize_kwargs):
    if not hasattr(mf, "localize_orbitals"):
        raise NotImplementedError(
            "orbitals='localized' requires mf.localize_orbitals(); pass a localized "
            "mo_coeff explicitly for this reference."
        )
    localize_kwargs = {} if localize_kwargs is None else dict(localize_kwargs)
    localized_block = mf.localize_orbitals(
        method=localization,
        mo_coeff=mo_coeff[:, active],
        **localize_kwargs,
    )
    localized_block = np.asarray(localized_block)
    if localized_block.shape != (mo_coeff.shape[0], len(active)):
        raise ValueError(
            f"localized active block has shape {localized_block.shape}, "
            f"expected {(mo_coeff.shape[0], len(active))}."
        )
    out = np.array(mo_coeff, copy=True)
    out[:, active] = localized_block
    return out


def _map_clusters_to_active(clusters, active):
    if clusters is None:
        return None
    if isinstance(clusters, str):
        key = clusters.lower().replace("-", "_")
        if key in {"none", "off", "false"}:
            return None
        raise ValueError("clusters must be 'auto', None, or an explicit orbital-block list.")
    if clusters is False:
        return None
    active = tuple(int(i) for i in active)
    active_lookup = {idx: pos for pos, idx in enumerate(active)}
    normalized = tuple(tuple(int(i) for i in block) for block in clusters)
    flat = tuple(i for block in normalized for i in block)
    ncas = len(active)
    if sorted(flat) == list(range(ncas)):
        return normalize_orbital_blocks(normalized, norb=ncas)
    if all(i in active_lookup for i in flat):
        mapped = tuple(tuple(active_lookup[i] for i in block) for block in normalized)
        return normalize_orbital_blocks(mapped, norb=ncas)
    raise ValueError(
        "clusters must use active-space indices 0..ncas-1 or original MO labels "
        "contained in active."
    )


def _prepare_active_cluster_workflow(mf, kwargs, symmetry):
    """Translate high-level active/cluster keywords into backend options."""
    kwargs = dict(kwargs)
    if "blocks" in kwargs:
        raise TypeError("NARG(..., blocks=...) was removed; use symmetry=... instead.")

    workflow_keys = {
        "active",
        "active_orbitals",
        "orbitals",
        "clusters",
        "cluster_method",
        "cluster_max_size",
        "max_cluster_size",
        "cluster_weights",
        "cluster_dm",
        "localization",
        "localize_kwargs",
    }
    if not any(key in kwargs for key in workflow_keys):
        return kwargs, symmetry, None

    orbitals = str(kwargs.pop("orbitals", "canonical")).lower().replace("-", "_")
    clusters = kwargs.pop("clusters", None)
    cluster_method = kwargs.pop("cluster_method", "spectral")
    cluster_max_size = kwargs.pop("cluster_max_size", kwargs.pop("max_cluster_size", 4))
    cluster_weights = kwargs.pop("cluster_weights", "integral+rdm")
    cluster_dm = kwargs.pop("cluster_dm", None)
    localization = kwargs.pop("localization", "pm")
    localize_kwargs = kwargs.pop("localize_kwargs", None)

    mo_coeff = _as_spatial_mo_coeff(mf, kwargs.get("mo_coeff", None))
    active = _active_indices_from_options(mf, kwargs, mo_coeff)
    ncore = _infer_ncore(mf, kwargs.get("nelecas", None), kwargs.get("ncore", None))
    kwargs["ncore"] = ncore
    kwargs["ncas"] = len(active)

    cluster_info = None
    if isinstance(clusters, str) and clusters.lower().replace("-", "_") == "auto":
        from pyqed.qchem.orbital_clustering import cluster_mf_orbitals

        clusters, workflow_mo, cluster_info = cluster_mf_orbitals(
            mf,
            method=cluster_method,
            max_size=cluster_max_size,
            weights=cluster_weights,
            orbitals=orbitals,
            localization=localization,
            mo_coeff=mo_coeff,
            dm=cluster_dm,
            active=active,
            localize_kwargs=localize_kwargs,
            return_orbitals=True,
            return_info=True,
        )
    else:
        if orbitals in {"localized", "localised", "local"}:
            workflow_mo = _localize_active_orbitals(
                mf,
                mo_coeff,
                active,
                localization,
                localize_kwargs,
            )
        elif orbitals in {"canonical", "mo", "scf", "original"}:
            workflow_mo = mo_coeff
        else:
            raise ValueError("orbitals must be 'canonical' or 'localized'.")

    kwargs["mo_coeff"], mo_order = _reorder_mo_for_active(workflow_mo, active, ncore)
    orbital_blocks = _map_clusters_to_active(clusters, active)
    if orbital_blocks is not None:
        kwargs["orbital_blocks"] = orbital_blocks
        if symmetry is None:
            symmetry = "number"

    workflow = {
        "active": active,
        "orbital_space": active,
        "ncore": ncore,
        "ncas": len(active),
        "mo_order": mo_order,
        "orbitals": orbitals,
        "localization": localization if orbitals in {"localized", "localised", "local"} else None,
        "clusters": orbital_blocks,
        "cluster_info": cluster_info,
    }
    return kwargs, symmetry, workflow


def _attach_workflow(solver, workflow):
    if workflow is not None:
        setattr(solver, "workflow", workflow)
        setattr(solver, "cluster_info", workflow.get("cluster_info"))
    return solver


def _backend_from_symmetry(symmetry):
    """Return the concrete qchem backend key for a public symmetry label."""
    if symmetry is None:
        return None
    key = normalize_symmetry(symmetry)
    if key == "none":
        return None
    if key == "number":
        return "abelian"
    if key == "spin":
        return "su2"
    if key == "momentum":
        raise NotImplementedError(
            "symmetry='momentum' is reserved for total-K sectors and is not implemented yet."
        )
    raise ValueError(
        f"Unknown NARG symmetry {symmetry!r}; expected 'none', 'number', or 'spin'."
    )


class NARG:
    """Bare qchem NARG by default, with optional backend dispatch.

    ``symmetry="none"`` selects the bare backend, ``symmetry="number"`` selects
    the U(1)xU(1) backend, and ``symmetry="spin"`` selects the SU(2) backend.
    Backend aliases such as ``"abelian"`` and ``"su2"`` are also accepted.

    High-level active-space workflows may use chemistry-facing names instead
    of backend plumbing::

        solver = NARG(
            mf,
            symmetry="number",
            active=[4, 5, 6, 7],
            nelecas=(2, 2),
            orbitals="localized",
            clusters="auto",
            cluster_max_size=2,
            D=128,
        )
        solver.run()

    Here ``symmetry`` chooses the quantum-number symmetry, while ``clusters``
    chooses orbital supersites.  ``active`` is the complete orbital space passed
    to NARG; if omitted, all spatial MOs are active.  The dispatcher derives
    ``mo_coeff``, ``ncore``, ``ncas``, and backend ``orbital_blocks`` internally.
    """

    def __new__(cls, mf, *args, symmetry=None, **kwargs):
        if cls is not NARG:
            return super().__new__(cls)
        if "blocks" in kwargs:
            raise TypeError("NARG(..., blocks=...) was removed; use symmetry=... instead.")
        workflow = None
        if isinstance(mf, MPOHamiltonian):
            raise NotImplementedError(
                "NARG(MPOHamiltonian) is reserved for the generic MPO-NARG backend, "
                "which is not implemented yet. Use form='integrals' for now."
            )
        if isinstance(mf, IntegralHamiltonian):
            if args:
                raise TypeError("NARG(IntegralHamiltonian, ...) does not accept positional backend arguments.")
            hamiltonian = mf
            if hamiltonian.mf is None:
                raise ValueError("IntegralHamiltonian needs an mf facade for qchem NARG.")
            mf = hamiltonian.mf
            kwargs.setdefault("mol", hamiltonian.mol)
            kwargs.setdefault("h1e", hamiltonian.h1e)
            kwargs.setdefault("eri", hamiltonian.eri)
            if hamiltonian.orbital_blocks is not None:
                kwargs.setdefault("orbital_blocks", hamiltonian.orbital_blocks)
            if symmetry is None:
                symmetry = hamiltonian.symmetry
            backend = _backend_from_symmetry(symmetry)
            if backend == "su2" and isinstance(hamiltonian.target, dict):
                spin = hamiltonian.target.get("spin")
                if spin is not None:
                    kwargs.setdefault("target_j2", abs(int(spin)))
        else:
            kwargs, symmetry, workflow = _prepare_active_cluster_workflow(mf, kwargs, symmetry)
            if symmetry is None and kwargs.get("orbital_blocks", None) is not None:
                symmetry = "number"
        backend = _backend_from_symmetry(symmetry)
        if backend is None:
            bare_narg = _load_bare_backend()
            return _attach_workflow(bare_narg(mf, *args, **kwargs), workflow)
        if backend == "abelian":
            abelian_narg, *_ = _load_abelian_backend()
            return _attach_workflow(abelian_narg(mf, *args, **kwargs), workflow)
        if backend == "su2":
            su2_narg = _load_su2_backend()
            return _attach_workflow(su2_narg(mf, *args, **kwargs), workflow)
        raise ValueError(
            f"Unknown NARG symmetry {symmetry!r}; expected 'none', 'number', or 'spin'."
        )


def kernel(*args, **kwargs):
    """Run the default Abelian qchem NARG kernel."""
    _, abelian_kernel_fn, _, _, _ = _load_abelian_backend()
    return abelian_kernel_fn(*args, **kwargs)


def hierarchical_kernel(*args, **kwargs):
    """Run the balanced-tree Abelian qchem NARG prototype."""
    *_, abelian_hierarchical_kernel_fn = _load_abelian_backend()
    return abelian_hierarchical_kernel_fn(*args, **kwargs)


def supersite_kernel(*args, **kwargs):
    """Run Abelian qchem NARG with explicit composite local supersites."""
    _, _, abelian_supersite_kernel_fn, _, _ = _load_abelian_backend()
    return abelian_supersite_kernel_fn(*args, **kwargs)


def energy_groups(*args, **kwargs):
    """Build energy-ordered groups for Abelian qchem supersite growth."""
    _, _, _, abelian_energy_groups_fn, _ = _load_abelian_backend()
    return abelian_energy_groups_fn(*args, **kwargs)


def __getattr__(name):
    if name == "BareNARG":
        _load_bare_backend()
        return globals()[name]
    if name == "SU2NARG":
        _load_su2_backend()
        return globals()[name]
    if name in {
        "AbelianNARG",
        "abelian_energy_groups",
        "abelian_kernel",
        "abelian_hierarchical_kernel",
        "abelian_supersite_kernel",
    }:
        _load_abelian_backend()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NARG",
    "LETTA",
    "NARGOpt",
    "NARGSCF",
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
