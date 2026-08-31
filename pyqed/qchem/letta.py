"""Quantum-chemistry LETTA driver."""

from __future__ import annotations

import numpy as np


def _electron_count(mol, explicit=None):
    if explicit is not None:
        return int(explicit)
    nelec = getattr(mol, "nelec", None)
    if nelec is None:
        nelec = getattr(mol, "nelectron", None)
    if nelec is None:
        raise ValueError("LETTA could not infer the electron count; pass nelec=...")
    return int(np.sum(np.asarray(nelec, dtype=int).reshape(-1)))


class LETTA:
    """Build a native symmetry-adapted LETTA state from quantum chemistry.

    The canonical interface accepts a completed restricted mean-field object::

        state = LETTA(mf, symmetry="su2", D=32)

    Active-space calculations additionally accept ``ncas``, ``nelecas``,
    ``ncore``, and ``mo_coeff``.
    """

    def __new__(
        cls,
        mf,
        *args,
        symmetry="su2",
        mol=None,
        h1e=None,
        eri=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        mo_coeff=None,
        nelec=None,
        spin=None,
        use_cholesky=None,
        **kwargs,
    ):
        if cls is not LETTA:
            return super().__new__(cls)
        if args:
            raise TypeError("LETTA(mf, ...) accepts chemistry options by keyword.")
        key = str(symmetry).lower().replace("-", "").replace("_", "")
        if key not in {"su2", "spin"}:
            raise NotImplementedError("LETTA(mf, ...) currently supports symmetry='su2'.")

        from pyqed.letta import SU2LETTA
        from pyqed.narg.qchem.active_space import prepare_active_space

        base_mol = mol if mol is not None else getattr(mf, "mol", None)
        cas_requested = any(
            value is not None for value in (ncas, nelecas, ncore, mo_coeff)
        )
        h1e, eri, prepared_mol, active_space = prepare_active_space(
            mf,
            base_mol,
            h1e=h1e,
            eri=eri,
            ncas=ncas,
            nelecas=nelecas,
            ncore=ncore,
            mo_coeff=mo_coeff,
            spin=spin if cas_requested else None,
            use_cholesky=use_cholesky,
        )
        target_nelec = _electron_count(prepared_mol, nelec)
        if active_space is None:
            target_spin = int(
                getattr(prepared_mol, "spin", 0) if spin is None else spin
            )
            ecore = float(prepared_mol.energy_nuc()) if prepared_mol is not None else 0.0
        else:
            target_spin = int(active_space.spin if spin is None else spin)
            ecore = float(active_space.energy_core)

        state = SU2LETTA.from_integrals(
            h1e,
            eri=eri,
            nelec=target_nelec,
            spin=target_spin,
            ecore=ecore,
            **kwargs,
        )
        state.mf = mf
        state.mol = prepared_mol
        state.active_space = active_space
        state.ncas = int(np.asarray(h1e).shape[0])
        state.nelecas = target_nelec
        state.ncore = 0 if active_space is None else int(active_space.ncore)
        return state

    @classmethod
    def from_integrals(cls, h1e, eri=None, *, symmetry="su2", **kwargs):
        """Build from already prepared spatial-orbital integrals."""
        key = str(symmetry).lower().replace("-", "").replace("_", "")
        if key not in {"su2", "spin"}:
            raise NotImplementedError("LETTA.from_integrals currently supports SU(2).")
        from pyqed.letta import SU2LETTA

        return SU2LETTA.from_integrals(h1e, eri=eri, **kwargs)


__all__ = ["LETTA"]
