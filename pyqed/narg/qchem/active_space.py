"""Active-space preparation shared by qchem NARG backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.qchem.mcscf.casci import (
    _normalize_active_electrons,
    h1e_for_cas,
    transform_spatial_eri_to_mo,
)


CAS_OPTION_DEFAULTS = {
    "ncas": None,
    "nelecas": None,
    "ncore": None,
    "mo_coeff": None,
    "spin": None,
    "use_cholesky": None,
}


@dataclass
class ActiveSpace:
    ncas: int
    nelecas: int | tuple[int, int]
    nelecas_spin: tuple[int, int]
    ncore: int
    spin: int
    energy_core: float
    mo_core: np.ndarray
    mo_cas: np.ndarray
    base_mol: object


class ActiveSpaceMolecule:
    """Molecule facade carrying active electron counts and frozen-core energy."""

    def __init__(self, base_mol, active_space: ActiveSpace):
        self.base_mol = base_mol
        self.active_space = active_space
        self.nelec = active_space.nelecas_spin
        self.spin = active_space.spin
        self.ncas = active_space.ncas
        self.nelecas = active_space.nelecas
        self.ncore = active_space.ncore

    def energy_nuc(self):
        return self.active_space.energy_core

    def __getattr__(self, name):
        return getattr(self.base_mol, name)


def pop_active_space_options(options):
    return {key: options.pop(key, None) for key in CAS_OPTION_DEFAULTS}


def _reference_nelectron(mf, mol):
    nelec = getattr(mf, "nelec", None)
    if nelec is None and mol is not None:
        nelec = getattr(mol, "nelec", None)
    if nelec is None and mol is not None:
        nelec = getattr(mol, "nelectron", None)
    if nelec is None:
        return None
    return int(np.sum(np.asarray(nelec, dtype=int).reshape(-1)))


def _as_mo_coeff(mf, mo_coeff):
    if mo_coeff is None:
        mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("CAS-NARG needs mo_coeff; pass mo_coeff=... or run an MO mean-field first.")
    if isinstance(mo_coeff, (tuple, list)):
        raise NotImplementedError("CAS-NARG currently supports restricted spatial orbitals only.")
    return np.asarray(mo_coeff)


def _active_spin(nelecas, spin, mol):
    if spin is not None:
        return int(round(spin))
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]) - int(nelecas[1])
    return int(getattr(mol, "spin", 0)) if mol is not None else 0


def _slice_core_active(mo_coeff, ncore, ncas):
    ncore = int(ncore)
    ncas = int(ncas)
    return mo_coeff[:, :ncore], mo_coeff[:, ncore:ncore + ncas]


def prepare_active_space(
    mf,
    mol,
    *,
    h1e=None,
    eri=None,
    ncas=None,
    nelecas=None,
    ncore=None,
    mo_coeff=None,
    spin=None,
    use_cholesky=None,
):
    """Return NARG integrals and a mol-like object for optional CAS calculations."""

    cas_requested = any(
        value is not None for value in (ncas, nelecas, ncore, mo_coeff)
    )
    if not cas_requested:
        if spin is not None:
            raise ValueError("Use target_spin for spin filtering; spin is a CAS active-space option.")
        if h1e is None:
            h1e = mf.get_hcore_mo()
        if eri is None:
            eri = mf.get_eri_mo()
        return h1e, eri, mol, None

    if ncas is None:
        raise ValueError("CAS-NARG requires ncas.")
    ncas = int(ncas)
    if ncas <= 0:
        raise ValueError("ncas must be positive.")

    base_mol = mol if mol is not None else getattr(mf, "mol", None)
    mo_coeff = _as_mo_coeff(mf, mo_coeff)
    nmo = int(mo_coeff.shape[1])
    if ncas > nmo:
        raise ValueError(f"ncas={ncas} exceeds the number of orbitals {nmo}.")

    total_electrons = _reference_nelectron(mf, base_mol)
    if nelecas is None:
        if ncore is None or total_electrons is None:
            raise ValueError("CAS-NARG requires nelecas unless ncore and total electron count are known.")
        nelecas = int(total_electrons - 2 * int(ncore))

    active_spin = _active_spin(nelecas, spin, base_mol)
    nelecas_spin = tuple(int(x) for x in _normalize_active_electrons(nelecas, active_spin))
    nelecas_total = int(sum(nelecas_spin))
    if ncore is None:
        if total_electrons is None:
            raise ValueError("Cannot infer ncore without a total electron count.")
        ncore_electrons = total_electrons - nelecas_total
        if ncore_electrons < 0 or ncore_electrons % 2:
            raise ValueError(
                "Frozen-core CAS-NARG requires total electrons minus active electrons "
                "to be a non-negative even number."
            )
        ncore = ncore_electrons // 2
    ncore = int(ncore)
    if ncore < 0:
        raise ValueError("ncore must be non-negative.")
    if total_electrons is not None and 2 * ncore + nelecas_total != total_electrons:
        raise ValueError(
            "Frozen-core CAS-NARG requires 2*ncore + nelecas to match the reference "
            f"electron count ({2 * ncore + nelecas_total} != {total_electrons})."
        )
    if ncore + ncas > nmo:
        raise ValueError(f"ncore+ncas={ncore + ncas} exceeds the number of orbitals {nmo}.")

    if (h1e is None) ^ (eri is None):
        raise ValueError("Pass both h1e and eri for an explicit CAS Hamiltonian, or neither.")

    h1_cas, energy_core = h1e_for_cas(mf, ncas=ncas, ncore=ncore, mo_coeff=mo_coeff)
    mo_core, mo_cas = _slice_core_active(mo_coeff, ncore, ncas)
    if h1_cas.ndim != 2:
        raise NotImplementedError("CAS-NARG currently supports restricted spatial h1e only.")

    if h1e is None:
        h1e = h1_cas
        eri = transform_spatial_eri_to_mo(
            mf,
            mo_cas,
            use_cholesky=bool(use_cholesky) if use_cholesky is not None else False,
        )
    else:
        h1e = np.asarray(h1e)
        eri = np.asarray(eri)
        if h1e.shape != (ncas, ncas):
            raise ValueError(f"Explicit CAS h1e has shape {h1e.shape}, expected {(ncas, ncas)}.")
        if eri.shape != (ncas, ncas, ncas, ncas):
            raise ValueError(
                f"Explicit CAS eri has shape {eri.shape}, expected {(ncas, ncas, ncas, ncas)}."
            )

    active_space = ActiveSpace(
        ncas=ncas,
        nelecas=nelecas,
        nelecas_spin=nelecas_spin,
        ncore=ncore,
        spin=int(nelecas_spin[0] - nelecas_spin[1]),
        energy_core=float(np.real(energy_core)),
        mo_core=mo_core,
        mo_cas=mo_cas,
        base_mol=base_mol,
    )
    return h1e, eri, ActiveSpaceMolecule(base_mol, active_space), active_space
