#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time TDHF propagation in an AO basis.
"""

import numpy as np
from scipy.linalg import eigh, expm

from .hf.rhf import energy_elec, get_veff


def _axis_to_index(axis):
    if isinstance(axis, str):
        key = axis.lower()
        if key == 'x':
            return 0
        if key == 'y':
            return 1
        if key == 'z':
            return 2
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    return int(axis)


def gaussian_pulse(amplitude, center, width, frequency=0.0, phase=0.0,
                   polarization=(1.0, 0.0, 0.0)):
    """
    Build a Gaussian-envelope electric field callable.
    """
    polarization = np.asarray(polarization, dtype=float)
    norm = np.linalg.norm(polarization)
    if norm == 0.0:
        raise ValueError("polarization must be non-zero.")
    polarization = polarization / norm

    def field(time):
        env = amplitude * np.exp(-((time - center) ** 2) / (2.0 * width ** 2))
        carrier = np.cos(frequency * (time - center) + phase)
        return env * carrier * polarization

    return field


class RTTDHF:
    """
    Real-time TDHF propagation starting from a converged RHF reference.

    Notes
    -----
    The propagation is carried out in a Löwdin-orthogonalized AO basis using a
    midpoint unitary step.  When an external field is supplied, the AO
    interaction operator is taken to be the position operator by default:

        F(t) = F_HF[D(t)] - E(t) . r
    """

    def __init__(self, mf, interaction_ao=None, field=None, s_thresh=1e-12):
        if mf.mo_coeff is None or mf.dm is None:
            raise ValueError("Run RHF before starting real-time TDHF.")

        self._scf = mf
        self.mol = mf.mol
        self.field = field
        self.interaction_ao = interaction_ao
        self.s_thresh = s_thresh

        self.overlap = np.asarray(mf.get_ovlp(), dtype=float)
        self.hcore = np.asarray(mf.get_hcore(), dtype=float)
        self.e_nuc = mf.energy_nuc()

        self._x = None
        self._xinv = None

        self.times = None
        self.dms = None
        self.energies = None
        self.dipoles = None
        self.fields = None
        self.dm = np.asarray(mf.dm, dtype=complex)

    def _build_orthogonalizer(self):
        if self._x is not None and self._xinv is not None:
            return self._x, self._xinv

        evals, evecs = eigh(self.overlap)
        if np.any(evals < self.s_thresh):
            raise ValueError("Overlap matrix is singular or ill-conditioned.")

        s_inv_sqrt = np.diag(evals ** -0.5)
        s_sqrt = np.diag(evals ** 0.5)
        self._x = evecs @ s_inv_sqrt @ evecs.conj().T
        self._xinv = evecs @ s_sqrt @ evecs.conj().T
        return self._x, self._xinv

    def ao_to_orth(self, dm):
        """
        Transform an AO density matrix to the orthogonalized AO basis.
        """
        _, xinv = self._build_orthogonalizer()
        return xinv @ dm @ xinv.conj().T

    def orth_to_ao(self, dm_orth):
        """
        Transform an orthogonal-basis density matrix back to the AO basis.
        """
        x, _ = self._build_orthogonalizer()
        return x @ dm_orth @ x.conj().T

    def operator_to_orth(self, op):
        """
        Transform an AO operator to the orthogonalized AO basis.
        """
        x, _ = self._build_orthogonalizer()
        return x.conj().T @ op @ x

    def get_interaction_ao(self):
        """
        AO representation of the external-field interaction operator.
        """
        if self.interaction_ao is None:
            op = np.asarray(self.mol.moment_integral(), dtype=float)
            if op.ndim != 3:
                raise ValueError("moment_integral() must return a rank-3 array.")
            if op.shape[0] == 3:
                self.interaction_ao = op
            elif op.shape[-1] == 3:
                self.interaction_ao = np.moveaxis(op, -1, 0)
            else:
                raise ValueError(
                    "interaction_ao must have shape (3, nao, nao) or (nao, nao, 3)."
                )
        return self.interaction_ao

    def field_vector(self, time, field=None):
        """
        Evaluate the external field at a given time.
        """
        source = self.field if field is None else field
        if source is None:
            return np.zeros(3)

        value = source(time) if callable(source) else source
        vec = np.asarray(value, dtype=float)

        if vec.ndim == 0:
            out = np.zeros(3)
            out[0] = float(vec)
            return out

        vec = vec.reshape(-1)
        if vec.size != 3:
            raise ValueError("field must evaluate to a scalar or a length-3 vector.")
        return vec

    def get_fock(self, dm, time=0.0, field=None):
        """
        Instantaneous AO Fock matrix, optionally including an external field.
        """
        fock = self.hcore + get_veff(self.mol, dm)
        field_vec = self.field_vector(time, field=field)
        if np.any(field_vec):
            interaction = self.get_interaction_ao()
            fock = fock - np.einsum('x,xij->ij', field_vec, interaction, optimize=True)
        return 0.5 * (fock + fock.conj().T)

    def energy(self, dm=None):
        """
        Field-free Hartree-Fock energy for the current density matrix.
        """
        if dm is None:
            dm = self.dm
        vhf = get_veff(self.mol, dm)
        return energy_elec(dm, self.hcore, vhf) + self.e_nuc

    def electron_count(self, dm=None):
        """
        Number of electrons, Tr[S D].
        """
        if dm is None:
            dm = self.dm
        return np.einsum('ij,ji->', self.overlap, dm).real

    def dipole_moment(self, dm=None):
        """
        Expectation value of the AO position operator.
        """
        if dm is None:
            dm = self.dm
        return np.einsum('xij,ji->x', self.get_interaction_ao(), dm, optimize=True).real

    def kick(self, dm=None, strength=1e-4, axis='x', interaction_ao=None):
        """
        Apply a delta-kick to the density matrix in the orthogonalized basis.
        """
        if dm is None:
            dm = self.dm
        op = self.get_interaction_ao() if interaction_ao is None else np.asarray(interaction_ao)
        idx = _axis_to_index(axis)

        p = self.ao_to_orth(dm)
        u = expm(-1j * strength * self.operator_to_orth(op[idx]))
        dm_new = self.orth_to_ao(u @ p @ u.conj().T)
        return 0.5 * (dm_new + dm_new.conj().T)

    def step(self, dm, time, dt, field=None):
        """
        Propagate the AO density matrix by one midpoint time step.
        """
        p = self.ao_to_orth(dm)

        fock_0 = self.operator_to_orth(self.get_fock(dm, time=time, field=field))
        u_half = expm(-0.5j * dt * fock_0)
        p_half = u_half @ p @ u_half.conj().T
        dm_half = self.orth_to_ao(p_half)

        fock_half = self.operator_to_orth(
            self.get_fock(dm_half, time=time + 0.5 * dt, field=field)
        )
        u = expm(-1j * dt * fock_half)
        p_new = u @ p @ u.conj().T
        dm_new = self.orth_to_ao(p_new)
        return 0.5 * (dm_new + dm_new.conj().T)

    def run(self, dt, nsteps, dm0=None, field=None, t0=0.0, store_dm=True,
            kick=None):
        """
        Propagate the density matrix for ``nsteps`` steps of size ``dt``.

        Parameters
        ----------
        dt : float
            Time step in atomic units.
        nsteps : int
            Number of propagation steps.
        dm0 : ndarray, optional
            Initial AO density matrix.  If omitted, use the converged RHF
            density.
        field : callable or array-like, optional
            External field specification.  A callable should return either a
            scalar or a length-3 vector.
        t0 : float, optional
            Initial time.
        store_dm : bool, optional
            Whether to store the full density matrix trajectory.
        kick : dict, optional
            Optional delta-kick parameters forwarded to ``kick()`` before the
            first propagation step.
        """
        dm = np.asarray(self._scf.dm if dm0 is None else dm0, dtype=complex)
        if kick is not None:
            dm = self.kick(dm=dm, **kick)

        times = t0 + dt * np.arange(nsteps + 1, dtype=float)
        dms = None
        if store_dm:
            dms = np.zeros((nsteps + 1,) + dm.shape, dtype=complex)
        energies = np.zeros(nsteps + 1, dtype=float)
        dipoles = np.zeros((nsteps + 1, 3), dtype=float)
        fields = np.zeros((nsteps + 1, 3), dtype=float)

        for istep, time in enumerate(times):
            if store_dm:
                dms[istep] = dm
            energies[istep] = self.energy(dm)
            dipoles[istep] = self.dipole_moment(dm)
            fields[istep] = self.field_vector(time, field=field)

            if istep < nsteps:
                dm = self.step(dm, time, dt, field=field)

        self.times = times
        self.dms = dms
        self.energies = energies
        self.dipoles = dipoles
        self.fields = fields
        self.dm = dm
        return self


RealTimeTDHF = RTTDHF
