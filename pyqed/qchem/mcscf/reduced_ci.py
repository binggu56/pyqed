#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced CI subspace utilities for coupled second-order CASSCF.

The analytic coupled AH solver needs a small orthonormal CI expansion space
containing the optimized CASCI roots plus preconditioned Q-space correction
vectors.  This module provides that bookkeeping independently of the orbital
optimizer.  When available, it uses the CASCI object's matrix-free direct-CI
sigma action; otherwise it falls back to the existing dense Slater-Condon
builder for small/debug cases.
"""

from dataclasses import dataclass

import numpy as np

from pyqed.qchem.ci.fci import CI_H
from pyqed.qchem.mcscf.casci import make_tdm1, make_tdm2
from pyqed.qchem.mcscf.orbopt import (
    embed_rdm2,
    generalized_fock,
    orbital_gradient,
    pack_nonredundant,
)


def _as_vector_matrix(vectors, ndet=None):
    arr = np.asarray(vectors, dtype=np.result_type(vectors, float))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("CI vectors must be a vector or a 2D array.")
    if ndet is not None and arr.shape[0] != ndet:
        if arr.shape[1] == ndet:
            arr = arr.T
        else:
            raise ValueError(
                "CI vector dimension {} is incompatible with ndet={}.".format(
                    arr.shape,
                    ndet,
                )
            )
    return arr


def orthonormalize_ci_vectors(vectors, ndet=None, tol=1.0e-10):
    """
    Return an orthonormal column basis spanning the supplied CI vectors.
    """
    arr = _as_vector_matrix(vectors, ndet=ndet)
    if arr.shape[1] == 0:
        return np.zeros((arr.shape[0], 0), dtype=arr.dtype)

    q, r = np.linalg.qr(arr)
    keep = np.abs(np.diag(r)) > tol
    if not np.any(keep):
        return np.zeros((arr.shape[0], 0), dtype=arr.dtype)
    return q[:, keep]


def _spin_block_eri_from_spatial(spatial_eri):
    spatial_eri = np.asarray(spatial_eri)
    h2e = np.stack(
        (
            np.stack((spatial_eri.copy(), spatial_eri.copy())),
            np.stack((spatial_eri.copy(), spatial_eri.copy())),
        )
    )
    h2e[0, 0] -= h2e[0, 0].swapaxes(1, 3)
    h2e[1, 1] -= h2e[1, 1].swapaxes(1, 3)
    return h2e


def ci_hamiltonian_matrix(mc):
    """
    Build the active-space CI Hamiltonian matrix for an initialized CASCI object.

    The returned matrix excludes ``mc.e_core``.  Add ``mc.e_core`` to projected
    eigenvalues to compare with public CASCI total energies.
    """
    if getattr(mc, "binary", None) is None:
        raise ValueError("CASCI determinant basis is not initialized.")
    if getattr(mc, "hcore", None) is None:
        raise ValueError("CASCI one-electron active Hamiltonian is not available.")

    if hasattr(mc, "ensure_slater_condon_cache"):
        sc1, sc2 = mc.ensure_slater_condon_cache()
    else:
        sc1, sc2 = mc.SC1, mc.SC2
    if sc1 is None or sc2 is None:
        raise ValueError("CASCI Slater-Condon cache is not initialized.")

    h1e = np.asarray(mc.hcore)
    h2e = getattr(mc, "eri_so", None)
    if h2e is None:
        spatial_eri = getattr(mc, "h2e_cas", None)
        if spatial_eri is None:
            raise NotImplementedError(
                "Reduced CI dense Hamiltonian construction needs eri_so or h2e_cas."
            )
        h2e = _spin_block_eri_from_spatial(spatial_eri)
    return CI_H(mc.binary, h1e, np.asarray(h2e), sc1, sc2)


def ci_sigma(mc, vector):
    """
    Apply the active-space CI Hamiltonian to ``vector``.
    """
    if hasattr(mc, "ci_sigma"):
        return mc.ci_sigma(vector)
    hci = ci_hamiltonian_matrix(mc)
    return hci @ np.asarray(vector)


def ci_diagonal(mc):
    """
    Return the active-space CI Hamiltonian diagonal excluding core energy.
    """
    if hasattr(mc, "ci_diagonal"):
        return mc.ci_diagonal()
    return np.diag(ci_hamiltonian_matrix(mc))


def ci_rotation_pairs(nvec, nstates=1):
    """
    Return nonredundant CI rotations from optimized states to external vectors.

    Rotations among optimized states are intentionally excluded.  In the common
    state-specific case this returns ``[(p, 0) for p >= 1]``.
    """
    nvec = int(nvec)
    nstates = int(nstates)
    if nstates < 1 or nstates > nvec:
        raise ValueError("nstates must satisfy 1 <= nstates <= nvec.")
    return [(p, m) for m in range(nstates) for p in range(nstates, nvec)]


def ci_rotation_gradient(hamiltonian, nstates=1, weights=None):
    """
    CI-rotation gradient in the reduced subspace.

    The convention follows the coupled-AH parameter ``S[p,m]`` rotating
    optimized state ``m`` into external reduced-space vector ``p``.
    """
    hamiltonian = np.asarray(hamiltonian)
    nvec = hamiltonian.shape[0]
    if weights is None:
        weights = np.ones(nstates, dtype=float) / float(nstates)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != int(nstates):
        raise ValueError("weights must have length nstates.")

    grad = []
    pairs = ci_rotation_pairs(nvec, nstates=nstates)
    for p, m in pairs:
        grad.append(2.0 * weights[m] * hamiltonian[p, m])
    return np.asarray(grad, dtype=np.result_type(hamiltonian, float)), pairs


def ci_rotation_hessian(hamiltonian, nstates=1, weights=None):
    """
    Reduced CI-rotation Hessian block for state-to-external rotations.

    This is the equal-weight external-space form used by the paper when the
    optimized states diagonalize the reduced Hamiltonian:
    ``H_cc[(p,m),(q,n)] = 2 W_m delta_mn (H[p,q] - E_m delta_pq)``.
    """
    hamiltonian = np.asarray(hamiltonian)
    nvec = hamiltonian.shape[0]
    if weights is None:
        weights = np.ones(nstates, dtype=float) / float(nstates)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != int(nstates):
        raise ValueError("weights must have length nstates.")

    pairs = ci_rotation_pairs(nvec, nstates=nstates)
    hess = np.zeros((len(pairs), len(pairs)), dtype=np.result_type(hamiltonian, float))
    state_energies = np.diag(hamiltonian)[:nstates]
    for row, (p, m) in enumerate(pairs):
        for col, (q, n) in enumerate(pairs):
            if m != n:
                continue
            val = hamiltonian[p, q]
            if p == q:
                val -= state_energies[m]
            hess[row, col] = 2.0 * weights[m] * val
    return 0.5 * (hess + hess.conj().T), pairs


def _transition_rdms_with_core(mc, cibra, ciket, nmo=None):
    """
    Build transition 1/2-RDMs including inactive-core response blocks.
    """
    if hasattr(mc, "ensure_slater_condon_cache"):
        sc1, sc2 = mc.ensure_slater_condon_cache()
    else:
        sc1, sc2 = mc.SC1, mc.SC2
    if sc1 is None or sc2 is None:
        raise ValueError("CASCI Slater-Condon cache is not initialized.")

    ncore = int(mc.ncore)
    ncas = int(mc.ncas)
    nocc = ncore + ncas
    if nmo is None:
        nmo = int(getattr(mc.mf, "nmo", nocc))

    tdm1_act = make_tdm1(cibra, ciket, mc.binary, sc1)
    tdm2_act = make_tdm2(cibra, ciket, mc.binary, sc1, sc2)

    tdm1 = np.zeros((nmo, nmo), dtype=np.result_type(tdm1_act, tdm2_act, float))
    tdm1[ncore:nocc, ncore:nocc] = tdm1_act

    tdm2_occ = np.zeros((nocc, nocc, nocc, nocc), dtype=tdm1.dtype)
    for i in range(ncore):
        tdm2_occ[i, i, ncore:nocc, ncore:nocc] = 2.0 * tdm1_act
        tdm2_occ[ncore:nocc, ncore:nocc, i, i] = 2.0 * tdm1_act
        tdm2_occ[i, ncore:nocc, i, ncore:nocc] = -tdm1_act
        tdm2_occ[ncore:nocc, i, ncore:nocc, i] = -tdm1_act
    tdm2_occ[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = tdm2_act
    return tdm1, embed_rdm2(tdm2_occ, nmo)


def orbital_ci_coupling(
    mc,
    subspace,
    h1_mo,
    eri_mo,
    nstates=1,
    weights=None,
    nmo=None,
):
    """
    Build the orbital-CI coupling block using transition RDMs.

    Columns correspond to ``ci_rotation_pairs(subspace.nvec, nstates)`` and rows
    are packed nonredundant orbital rotations.
    """
    if weights is None:
        weights = np.ones(nstates, dtype=float) / float(nstates)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != int(nstates):
        raise ValueError("weights must have length nstates.")
    if nmo is None:
        nmo = np.asarray(h1_mo).shape[0]

    pairs = ci_rotation_pairs(subspace.nvec, nstates=nstates)
    if not pairs:
        return np.zeros((0, 0), dtype=float), pairs

    cols = []
    for p, m in pairs:
        c_p = subspace.basis[:, p]
        c_m = subspace.basis[:, m]
        tdm1_pm, tdm2_pm = _transition_rdms_with_core(mc, c_p, c_m, nmo=nmo)
        tdm1_mp, tdm2_mp = _transition_rdms_with_core(mc, c_m, c_p, nmo=nmo)
        dm1_delta = weights[m] * (tdm1_pm + tdm1_mp)
        dm2_delta = weights[m] * (tdm2_pm + tdm2_mp)
        fock_delta = generalized_fock(h1_mo, eri_mo, dm1_delta, dm2_delta)
        grad_delta = orbital_gradient(fock_delta)
        cols.append(pack_nonredundant(grad_delta, mc.ncore, mc.ncas, nmo))

    return np.column_stack(cols), pairs


@dataclass
class ReducedCISubspace:
    """
    Orthonormal CI expansion space and projected Hamiltonian.
    """

    basis: np.ndarray
    hamiltonian: np.ndarray
    e_core: float = 0.0

    @classmethod
    def from_casci(cls, mc, root_ids=None, extra_vectors=None, tol=1.0e-10):
        """
        Build a reduced subspace from CASCI roots and optional extra vectors.
        """
        if getattr(mc, "ci", None) is None:
            raise ValueError("Run CASCI before building a reduced CI subspace.")
        ndet = len(mc.ci[0])
        if root_ids is None:
            root_ids = range(len(mc.ci))

        columns = [np.asarray(mc.ci[i], dtype=np.result_type(mc.ci[i], float)) for i in root_ids]
        if extra_vectors is not None:
            extra = _as_vector_matrix(extra_vectors, ndet=ndet)
            columns.extend(extra[:, i] for i in range(extra.shape[1]))
        if not columns:
            raise ValueError("At least one CI vector is required.")

        basis = orthonormalize_ci_vectors(np.column_stack(columns), ndet=ndet, tol=tol)
        if basis.shape[1] == 0:
            raise ValueError("All CI subspace vectors were linearly dependent.")
        return cls.from_basis(mc, basis)

    @classmethod
    def from_basis(cls, mc, basis):
        basis = orthonormalize_ci_vectors(basis, ndet=len(mc.ci[0]))
        sigma = np.column_stack([ci_sigma(mc, basis[:, i]) for i in range(basis.shape[1])])
        hred = basis.conj().T @ sigma
        hred = 0.5 * (hred + hred.conj().T)
        return cls(
            basis=basis,
            hamiltonian=hred,
            e_core=float(getattr(mc, "e_core", 0.0)),
        )

    @property
    def ndet(self):
        return int(self.basis.shape[0])

    @property
    def nvec(self):
        return int(self.basis.shape[1])

    def diagonalize(self, nroots=None, include_core=True):
        """
        Diagonalize the projected Hamiltonian and return energies and CI vectors.
        """
        evals, evecs = np.linalg.eigh(self.hamiltonian)
        if nroots is not None:
            evals = evals[:nroots]
            evecs = evecs[:, :nroots]
        if include_core:
            evals = evals + self.e_core
        return evals, self.basis @ evecs

    def residuals(self, mc, ci_vectors, energies, include_core=True):
        """
        Return CI residuals ``H c - E c`` in determinant space.
        """
        ci_mat = _as_vector_matrix(ci_vectors, ndet=self.ndet)
        energies = np.asarray(energies)
        if energies.ndim == 0:
            energies = energies.reshape(1)
        if include_core:
            energies = energies - self.e_core
        if energies.shape[0] != ci_mat.shape[1]:
            raise ValueError("Number of energies must match number of CI vectors.")
        out = []
        for i in range(ci_mat.shape[1]):
            out.append(ci_sigma(mc, ci_mat[:, i]) - energies[i] * ci_mat[:, i])
        return np.column_stack(out)

    def rayleigh_energies(self, mc, ci_vectors, include_core=True):
        """
        Return Rayleigh quotient energies for determinant-space CI vectors.
        """
        ci_mat = _as_vector_matrix(ci_vectors, ndet=self.ndet)
        out = []
        for i in range(ci_mat.shape[1]):
            sigma = ci_sigma(mc, ci_mat[:, i])
            out.append(np.vdot(ci_mat[:, i], sigma).real)
        out = np.asarray(out, dtype=float)
        if include_core:
            out = out + self.e_core
        return out

    def rotated_state_vectors(self, step, pairs, nstates=1, tol=1.0e-12):
        """
        Apply first-order CI rotations and return updated optimized states.

        ``pairs`` must follow the ``(external_vector, optimized_state)``
        convention returned by :func:`ci_rotation_pairs`.
        """
        nstates = int(nstates)
        if nstates < 1 or nstates > self.nvec:
            raise ValueError("nstates must satisfy 1 <= nstates <= nvec.")
        step = np.asarray(step, dtype=np.result_type(step, self.basis, float))
        if step.shape[0] != len(pairs):
            raise ValueError("CI step length must match number of rotation pairs.")

        states = np.array(self.basis[:, :nstates], copy=True)
        for amp, (p, m) in zip(step, pairs):
            if abs(amp) <= tol:
                continue
            states[:, m] += amp * self.basis[:, p]
        return orthonormalize_ci_vectors(states, ndet=self.ndet, tol=tol)

    def expand_with_residuals(
        self,
        mc,
        ci_vectors,
        energies,
        max_vectors=None,
        precondition=True,
        tol=1.0e-10,
    ):
        """
        Append preconditioned Q-space residual vectors to the reduced CI basis.
        """
        residuals = self.residuals(mc, ci_vectors, energies, include_core=True)
        ci_mat = _as_vector_matrix(ci_vectors, ndet=self.ndet)
        energies = np.asarray(energies, dtype=float)
        if energies.ndim == 0:
            energies = energies.reshape(1)
        if max_vectors is None:
            max_vectors = residuals.shape[1]
        max_vectors = max(0, int(max_vectors))
        if max_vectors == 0:
            return self, 0

        diag = ci_diagonal(mc) if precondition else None
        new_cols = []
        for i in range(residuals.shape[1]):
            vec = residuals[:, i].copy()
            if precondition:
                theta = energies[i] - self.e_core
                denom = theta - diag
                safe = np.where(
                    np.abs(denom) > 1.0e-10,
                    denom,
                    np.where(denom >= 0.0, 1.0e-10, -1.0e-10),
                )
                vec = vec / safe
            vec -= self.basis @ (self.basis.conj().T @ vec)
            vec -= ci_mat @ (ci_mat.conj().T @ vec)
            for prev in new_cols:
                vec -= prev * np.vdot(prev, vec)
            norm = np.linalg.norm(vec)
            if norm <= tol:
                continue
            new_cols.append(vec / norm)
            if len(new_cols) >= max_vectors:
                break

        if not new_cols:
            return self, 0
        expanded_basis = np.column_stack((self.basis, np.column_stack(new_cols)))
        return type(self).from_basis(mc, expanded_basis), len(new_cols)

    def rotation_gradient(self, nstates=1, weights=None):
        """Return the reduced CI-rotation gradient and rotation pairs."""
        return ci_rotation_gradient(self.hamiltonian, nstates=nstates, weights=weights)

    def rotation_hessian(self, nstates=1, weights=None):
        """Return the reduced CI-rotation Hessian and rotation pairs."""
        return ci_rotation_hessian(self.hamiltonian, nstates=nstates, weights=weights)

    def orbital_coupling(self, mc, h1_mo, eri_mo, nstates=1, weights=None, nmo=None):
        """Return the orbital-CI coupling block and rotation pairs."""
        return orbital_ci_coupling(
            mc,
            self,
            h1_mo,
            eri_mo,
            nstates=nstates,
            weights=weights,
            nmo=nmo,
        )
