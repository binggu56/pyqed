"""Exact diagonalization solver for small qchem active spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS as DenseMPS
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.dmrg.dmrg import _build_spatial_active_hamiltonian_matrix


@dataclass
class EDResult:
    e_tot: float | np.ndarray
    e_active: float | np.ndarray
    state_s2: np.ndarray | None
    states: list[DenseMPS]
    history: list[dict]
    converged: bool = True
    backend: str = "ed"


def _charge_basis_indices(nsites, nelec):
    """Return full spatial-basis indices with total particle number ``nelec``."""
    charges = np.array([0, 1, 1, 2], dtype=int)
    indices = []
    labels = []

    def visit(site, remaining, index, occ):
        if site == nsites:
            if remaining == 0:
                indices.append(index)
                labels.append(tuple(occ))
            return
        power = 4 ** (nsites - site - 1)
        for state, charge in enumerate(charges):
            if charge > remaining:
                continue
            occ.append(state)
            visit(site + 1, remaining - int(charge), index + state * power, occ)
            occ.pop()

    visit(0, int(nelec), 0, [])
    return np.asarray(indices, dtype=int), labels


def _spin_square_in_charge_basis(labels):
    """Build S^2 in the fixed-charge spatial occupation basis."""
    dim = len(labels)
    pos = {label: i for i, label in enumerate(labels)}
    sz = np.zeros(dim, dtype=float)
    sp = np.zeros((dim, dim), dtype=float)
    sm = np.zeros((dim, dim), dtype=float)

    for col, label in enumerate(labels):
        nup = sum(1 for state in label if state in (1, 3))
        ndn = sum(1 for state in label if state in (2, 3))
        sz[col] = 0.5 * (nup - ndn)
        for site, state in enumerate(label):
            if state == 2:
                flipped = list(label)
                flipped[site] = 1
                sp[pos[tuple(flipped)], col] += 1.0
            elif state == 1:
                flipped = list(label)
                flipped[site] = 2
                sm[pos[tuple(flipped)], col] += 1.0

    return np.diag(sz * sz) + 0.5 * (sp @ sm + sm @ sp)


def _dense_mps_from_spatial_vector(vector, nsites):
    tensor = np.asarray(vector, dtype=complex).reshape((4,) * int(nsites))
    factors = decompose(tensor, rank=tensor.size)
    return DenseMPS(factors, labels=["lv", "p", "rv"]).normalize()


def _spin_adapted_dense_roots_from_matrix(
    h_dense,
    *,
    ncas,
    nelecas,
    spin,
    nstates,
    max_dense_dim=4096,
    spin_tol=1.0e-7,
):
    """Return exact target-spin roots from a dense spatial active Hamiltonian."""
    nsites = int(ncas)
    full_dim = 4 ** nsites
    if full_dim > int(max_dense_dim):
        raise NotImplementedError(
            "Exact spin-adapted ED is limited to full spatial dimension "
            f"<= {max_dense_dim}; got {full_dim}."
        )

    charge_indices, labels = _charge_basis_indices(nsites, nelecas)
    if charge_indices.size == 0:
        raise ValueError(
            f"No spatial determinants with nelec={nelecas} for ncas={ncas}."
        )

    h_dense = np.asarray(h_dense, dtype=complex)
    if h_dense.shape != (full_dim, full_dim):
        raise ValueError(
            f"Dense Hamiltonian shape {h_dense.shape} does not match spatial dimension {full_dim}."
        )
    h_charge = h_dense[np.ix_(charge_indices, charge_indices)]
    h_charge = 0.5 * (h_charge + h_charge.conj().T)
    s2_charge = _spin_square_in_charge_basis(labels)

    evals, evecs = np.linalg.eigh(h_charge)
    target_s = 0.5 * abs(float(spin))
    target_s2 = target_s * (target_s + 1.0)
    candidates = []
    start = 0
    while start < evals.size:
        stop = start + 1
        while stop < evals.size and abs(float(evals[stop] - evals[start])) <= 1.0e-8:
            stop += 1
        sub = evecs[:, start:stop]
        s2_sub = sub.conj().T @ s2_charge @ sub
        s2_vals, s2_vecs = np.linalg.eigh(0.5 * (s2_sub + s2_sub.conj().T))
        for col, s2_val in enumerate(np.real(s2_vals)):
            vec_charge = sub @ s2_vecs[:, col]
            energy = float(np.real(np.vdot(vec_charge, h_charge @ vec_charge)))
            if abs(s2_val - target_s2) <= spin_tol:
                full_vec = np.zeros(full_dim, dtype=complex)
                full_vec[charge_indices] = vec_charge
                norm = np.linalg.norm(full_vec)
                if norm > 0.0:
                    full_vec /= norm
                candidates.append((energy, float(s2_val), full_vec))
        start = stop

    candidates.sort(key=lambda item: item[0])
    if len(candidates) < nstates:
        raise RuntimeError(
            f"Found only {len(candidates)} roots with target <S^2>={target_s2:.8g}; "
            f"requested {nstates}."
        )

    energies = [item[0] for item in candidates[:nstates]]
    s2_values = [item[1] for item in candidates[:nstates]]
    states = [_dense_mps_from_spatial_vector(item[2], nsites) for item in candidates[:nstates]]
    return energies, s2_values, states


def _spin_adapted_dense_roots(qcdmrg, nstates, *, max_dense_dim=4096, spin_tol=1.0e-7):
    """Return exact target-spin active-space roots for a qchem solver with active integrals."""
    h_dense, _spatial_ops = _build_spatial_active_hamiltonian_matrix(
        qcdmrg.h1e,
        qcdmrg.h2e,
        spin_purification=False,
    )
    return _spin_adapted_dense_roots_from_matrix(
        h_dense,
        ncas=qcdmrg.ncas,
        nelecas=qcdmrg.nelecas,
        spin=qcdmrg.spin,
        nstates=nstates,
        max_dense_dim=max_dense_dim,
        spin_tol=spin_tol,
    )


class ED:
    """Exact diagonalization for small qchem active spaces."""

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        *,
        symmetry="su2",
        spin=None,
        max_dense_dim=4096,
        spin_tol=1.0e-7,
        verbose=0,
        **dmrg_kwargs,
    ):
        self.mf = mf
        self.ncas = int(ncas)
        self.nelecas = int(nelecas)
        self.symmetry = symmetry
        self.spin = mf.mol.spin if spin is None else spin
        self.max_dense_dim = int(max_dense_dim)
        self.spin_tol = float(spin_tol)
        self.verbose = int(verbose)
        self.dmrg_kwargs = dict(dmrg_kwargs)
        self.e_tot = None
        self.e_active = None
        self.state_s2 = None
        self.states = None
        self.history = None
        self.converged = False
        self.qcdmrg = None

    def run(self, nstates=1, weights=None, mo_coeff=None):
        nstates = int(nstates)
        if nstates < 1:
            raise ValueError("nstates must be positive.")
        if weights is None:
            weights = np.ones(nstates, dtype=float) / nstates
        else:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if weights.size != nstates:
                raise ValueError("weights must match nstates.")
            weights = weights / np.sum(weights)

        qcdmrg = DMRG(
            self.mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            D=self.dmrg_kwargs.pop("D", 1),
            site="spatial",
            symmetry=self.symmetry,
            spin=self.spin,
            verbose=self.verbose,
            **self.dmrg_kwargs,
        )
        if mo_coeff is None:
            qcdmrg.mo_coeff = self.mf.mo_coeff
        else:
            qcdmrg.mo_coeff = mo_coeff
        qcdmrg.mo_core = qcdmrg.mo_coeff[:, :qcdmrg.ncore]
        qcdmrg.mo_cas = qcdmrg.mo_coeff[:, qcdmrg.ncore:qcdmrg.ncore + qcdmrg.ncas]
        h1e, eri, pair_factors = qcdmrg._get_active_hamiltonian_inputs()
        if eri is None and pair_factors is not None:
            flat_pair_factors = pair_factors.reshape(pair_factors.shape[0], -1)
            eri_aa = (flat_pair_factors.conj().T @ flat_pair_factors).reshape(
                qcdmrg.ncas,
                qcdmrg.ncas,
                qcdmrg.ncas,
                qcdmrg.ncas,
            )
            eri = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))
        qcdmrg.h1e = h1e
        qcdmrg.h2e = eri
        qcdmrg.h2e_factors = pair_factors
        energies, s2_values, states = _spin_adapted_dense_roots(
            qcdmrg,
            nstates,
            max_dense_dim=self.max_dense_dim,
            spin_tol=self.spin_tol,
        )
        e_active = np.asarray(energies, dtype=float)
        e_tot = e_active + float(qcdmrg.e_core)
        if nstates == 1:
            e_active = float(e_active[0])
            e_tot = float(e_tot[0])

        self.qcdmrg = qcdmrg
        self.e_active = e_active
        self.e_tot = e_tot
        self.state_s2 = np.asarray(s2_values, dtype=float)
        self.states = states
        self.history = [
            {
                "solver": "ed",
                "backend": "spin_adapted_dense_ed",
                "state_energies": [float(x) for x in energies],
                "state_average_energy": float(np.dot(weights, energies)),
                "state_average_weights": [float(x) for x in weights],
                "state_s2": [float(x) for x in s2_values],
                "converged": True,
            }
        ]
        self.converged = True
        return self

    @property
    def result(self):
        return EDResult(
            e_tot=self.e_tot,
            e_active=self.e_active,
            state_s2=self.state_s2,
            states=self.states or [],
            history=self.history or [],
            converged=self.converged,
        )
