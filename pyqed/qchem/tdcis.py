"""Time-dependent CIS propagation in a fixed orbital basis.

The implementation constructs the full CIS determinant-space Hamiltonian and
uses the dense TD-CASCI propagator.  It is intended for modest orbital spaces,
where the quadratic storage and cubic dense-linear-algebra costs are practical.
"""

from __future__ import annotations

import numpy as np

from pyqed.qchem.ci.fci import CI_H, SlaterCondon
from pyqed.qchem.mcscf.casci import _slice_active_orbitals
from pyqed.qchem.mcscf.direct_ci import CASCI as DirectCASCI
from pyqed.qchem.tdcasci import TDCASCI


def _spin_occupations_from_mf(mf):
    mo_occ = np.asarray(mf.mo_occ)
    if mo_occ.ndim == 1:
        singly_occupied = np.isclose(mo_occ, 1.0)
        if np.any(singly_occupied):
            raise ValueError(
                "ROHF-style one-dimensional mo_occ arrays with singly occupied "
                "orbitals are not supported by TD-CIS; use an RHF reference or "
                "spin-resolved UHF occupations."
            )
        closed_shell = np.isclose(mo_occ, 0.0) | np.isclose(mo_occ, 2.0)
        if not np.all(closed_shell):
            raise ValueError(
                "One-dimensional mo_occ arrays for TD-CIS must contain only "
                "closed-shell occupations 0 and 2."
            )
        occ = np.isclose(mo_occ, 2.0)
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
    """Return determinant occupations for HF plus all spin-orbital singles.

    Closed-shell RHF occupations (a one-dimensional 0/2 array) and
    spin-resolved UHF occupations (a two-row 0/1 array) are supported.
    ROHF-style one-dimensional arrays containing singly occupied orbitals are
    rejected because this implementation does not define their spin assignment.
    """
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

    The reference must provide either closed-shell RHF occupations or
    spin-resolved UHF occupations.  Open-shell ROHF ``mo_occ`` arrays are not
    supported; convert the reference to UHF before constructing ``TDCIS``.
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
