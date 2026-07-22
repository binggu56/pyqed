"""Time-dependent CIS propagation in a fixed orbital basis."""

from __future__ import annotations

import numpy as np

from pyqed.qchem.ci.fci import CI_H, SlaterCondon
from pyqed.qchem.mcscf.casci import _slice_active_orbitals
from pyqed.qchem.mcscf.direct_ci import CASCI as DirectCASCI
from pyqed.qchem.tdcasci import TDCASCI


def _spin_occupations_from_mf(mf):
    mo_occ = np.asarray(mf.mo_occ)
    if mo_occ.ndim == 1:
        occ = mo_occ > 0
        occ_a = occ.astype(np.int8)
        occ_b = occ.astype(np.int8)
    elif mo_occ.ndim == 2 and mo_occ.shape[0] == 2:
        occ_a = (mo_occ[0] > 0).astype(np.int8)
        occ_b = (mo_occ[1] > 0).astype(np.int8)
    else:
        raise ValueError(f"Unsupported mo_occ shape for TD-CIS: {mo_occ.shape}.")
    if occ_a.shape != occ_b.shape:
        raise ValueError("Alpha and beta occupation arrays must have the same length.")
    return occ_a, occ_b


def cis_determinant_basis(mf):
    """Return determinant occupations for HF plus all spin-orbital singles."""
    occ_a, occ_b = _spin_occupations_from_mf(mf)
    ref = np.stack((occ_a, occ_b)).astype(np.int8, copy=True)
    determinants = [ref]
    seen = {ref.tobytes()}
    for spin, occ in enumerate((occ_a, occ_b)):
        occupied = np.flatnonzero(occ > 0)
        virtual = np.flatnonzero(occ == 0)
        for i in occupied:
            for a in virtual:
                det = ref.copy()
                det[spin, i] = 0
                det[spin, a] = 1
                key = det.tobytes()
                if key not in seen:
                    seen.add(key)
                    determinants.append(det)
    return np.asarray(determinants, dtype=np.int8)


class TDCIS(TDCASCI):
    """
    Time-dependent CIS propagation in the HF + singles determinant space.

    ``TDCIS`` reuses the fixed-orbital ``TDCASCI`` propagation machinery after
    restricting the determinant basis to the reference determinant and all
    single spin-orbital excitations.
    """

    def __init__(
        self,
        mf,
        nstates=None,
        interaction_mo=None,
        field=None,
        h1_mo=None,
        use_cholesky=None,
        verbose=0,
    ):
        if getattr(mf, "mo_coeff", None) is None or getattr(mf, "mo_occ", None) is None:
            raise ValueError("Run HF before starting TD-CIS.")

        occ_a, occ_b = _spin_occupations_from_mf(mf)
        nmo = int(occ_a.size)
        na = int(np.count_nonzero(occ_a))
        nb = int(np.count_nonzero(occ_b))
        binary = cis_determinant_basis(mf)
        if nstates is None:
            nstates = min(int(binary.shape[0]), 10)
        nstates = int(nstates)
        if nstates < 1:
            raise ValueError("nstates must be positive.")
        nstates = min(nstates, int(binary.shape[0]))

        solver = DirectCASCI(
            mf,
            ncas=nmo,
            nelecas=(na, nb),
            ms2=na - nb,
            verbose=verbose,
        )
        solver.binary = binary
        mo_coeff = mf.mo_coeff
        if isinstance(mo_coeff, (tuple, list)) and len(mo_coeff) == 2:
            spin_mo_coeff = mo_coeff
        else:
            spin_mo_coeff = (mo_coeff, mo_coeff)
        solver.mo_coeff = spin_mo_coeff
        solver.mo_core, solver.mo_cas = _slice_active_orbitals(
            solver.mo_coeff,
            solver.ncore,
            solver.ncas,
        )
        h1e, h2e = solver.get_SO_matrix(use_cholesky=use_cholesky)
        h2e[0, 0] -= h2e[0, 0].swapaxes(1, 3)
        h2e[1, 1] -= h2e[1, 1].swapaxes(1, 3)
        sc1, sc2 = SlaterCondon(binary)
        h_cis = CI_H(binary, h1e, h2e, sc1, sc2)
        e_active, vecs = np.linalg.eigh(0.5 * (h_cis + h_cis.conj().T))
        order = np.argsort(e_active.real)[:nstates]
        e_active = e_active[order].real
        vecs = vecs[:, order]
        solver.solver_backend = "tdcis_dense_subspace"
        solver.hcore = h1e
        solver.eri_so = h2e
        solver.h2e_cas = None
        solver.SC1 = sc1
        solver.SC2 = sc2
        solver.H = h_cis
        solver.e_tot = e_active + solver.e_core
        solver.ci = [vecs[:, i] for i in range(vecs.shape[1])]
        solver.nstates = int(vecs.shape[1])
        solver.ci_sigma = lambda c, h=h_cis: h @ np.asarray(c)
        solver.ci_diagonal = lambda h=h_cis: np.diag(h)
        self.solver = solver
        self.cis_binary = binary
        self.nstates = nstates
        super().__init__(
            solver,
            interaction_mo=interaction_mo,
            field=field,
            h1_mo=h1_mo,
        )


__all__ = ["TDCIS", "cis_determinant_basis"]
