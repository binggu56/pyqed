#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time TDDFT propagation in an AO basis.
"""

import numpy as np

from .rttdhf import RTTDHF, gaussian_pulse


class RTTDDFT(RTTDHF):
    """
    Real-time TDDFT propagation starting from a converged RKS reference.

    The propagation is carried out in a Löwdin-orthogonalized AO basis using
    the same midpoint unitary step as ``RTTDHF``, but with the instantaneous
    Kohn-Sham Fock matrix.
    """

    def __init__(self, mf, interaction_ao=None, field=None, s_thresh=1e-12):
        if mf.mo_coeff is None or mf.dm is None:
            raise ValueError("Run RKS before starting real-time TDDFT.")
        super().__init__(mf, interaction_ao=interaction_ao, field=field, s_thresh=s_thresh)

    def get_fock(self, dm, time=0.0, field=None):
        """
        Instantaneous AO Kohn-Sham Fock matrix, optionally including a field.
        """
        fock = np.asarray(self._scf.get_fock(dm), dtype=complex)
        field_vec = self.field_vector(time, field=field)
        if np.any(field_vec):
            interaction = self.get_interaction_ao()
            fock = fock - np.einsum('x,xij->ij', field_vec, interaction, optimize=True)
        return 0.5 * (fock + fock.conj().T)

    def energy(self, dm=None):
        """
        Field-free Kohn-Sham total energy for the current density matrix.
        """
        if dm is None:
            dm = self.dm
        return self._scf.energy_elec(dm) + self.e_nuc


RealTimeTDDFT = RTTDDFT
