#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-CASCI spin-orbit state interaction helpers.
"""

from dataclasses import dataclass

import numpy as np

from pyqed.qchem.soc import get_soc_1e_spin_orbital, get_soc_somf_spin_orbital


@dataclass
class SOCStateInteractionResult:
    states: list
    energies: np.ndarray
    h_scalar: np.ndarray
    h_soc: np.ndarray
    h_total: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def _normalize_states(states):
    if not states:
        raise ValueError("states must contain at least one CASCI state.")

    normalized = []
    for entry in states:
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError("state entries must be CASCI objects or (casci, state_id) pairs.")
            casci, state_id = entry
        else:
            casci, state_id = entry, 0
        normalized.append((casci, int(state_id)))
    return normalized


def _validate_compatible_states(states):
    ref_casci, _ = states[0]
    ref_mo = np.asarray(ref_casci.mo_cas)
    ref_ncas = ref_casci.ncas
    if ref_mo.size == 0:
        raise ValueError("CASCI active orbitals are not available. Run CASCI first.")

    for casci, _ in states[1:]:
        mo = np.asarray(casci.mo_cas)
        if casci.ncas != ref_ncas:
            raise ValueError("All CASCI states must use the same ncas for SOC state interaction.")
        if mo.shape != ref_mo.shape or not np.allclose(mo, ref_mo):
            raise ValueError(
                "All CASCI states must share the same active orbitals for SOC state interaction."
            )
    return ref_casci


def soc_state_interaction(states, hso=None, one_center=True,
                          with_prefactor=True, light_speed=None,
                          order='grouped', soc_model='1e', dm=None):
    """
    Build and diagonalize a post-CASCI SOC state-interaction Hamiltonian.

    Parameters
    ----------
    states : sequence
        Sequence of CASCI objects or ``(casci, state_id)`` pairs.
    hso : ndarray, optional
        Active-space SOC operator in a spin-orbital basis.  If
        omitted, it is built from the active orbitals of the first CASCI state.
    one_center : bool
        Use the one-center approximation when building ``hso`` internally.
    with_prefactor : bool
        Include the Breit-Pauli prefactor in the internally built ``hso``.
    light_speed : float, optional
        Speed of light in atomic units for the prefactor.
    order : {'grouped', 'interleaved'}
        Spin-orbital ordering used by ``hso`` and the CASCI transition
        densities.
    """
    states = _normalize_states(states)
    ref_casci = _validate_compatible_states(states)

    if hso is None:
        model = soc_model.lower()
        if model == '1e':
            hso = get_soc_1e_spin_orbital(
                ref_casci.mf,
                representation='mo',
                mo_coeff=ref_casci.mo_cas,
                one_center=one_center,
                with_prefactor=with_prefactor,
                light_speed=light_speed,
                order=order,
            )
        elif model == 'somf':
            hso = get_soc_somf_spin_orbital(
                ref_casci.mf,
                representation='mo',
                mo_coeff=ref_casci.mo_cas,
                states=states,
                dm=dm,
                one_center=one_center,
                with_prefactor=with_prefactor,
                light_speed=light_speed,
                order=order,
            )
        else:
            raise ValueError("soc_model must be '1e' or 'somf'.")
    else:
        hso = np.asarray(hso)

    expected_shape = (2 * ref_casci.ncas, 2 * ref_casci.ncas)
    if hso.shape != expected_shape:
        raise ValueError(
            f"hso has shape {hso.shape}, expected active spin-orbital shape {expected_shape}."
        )

    nstates = len(states)
    energies = np.asarray([casci.e_tot[state_id] for casci, state_id in states], dtype=float)
    h_scalar = np.diag(energies.astype(complex))
    h_soc = np.zeros((nstates, nstates), dtype=complex)

    for i, (casci_i, state_i) in enumerate(states):
        for j in range(i, nstates):
            casci_j, state_j = states[j]
            val = casci_i.soc_matrix_element(
                state_i,
                ket_id=state_j,
                other=casci_j,
                hso=hso,
                order=order,
            )
            if i == j:
                h_soc[i, i] = 0.5 * (val + val.conjugate())
            else:
                h_soc[i, j] = val
                h_soc[j, i] = val.conjugate()

    h_total = h_scalar + h_soc
    eigenvalues, eigenvectors = np.linalg.eigh(h_total)
    return SOCStateInteractionResult(
        states=states,
        energies=energies,
        h_scalar=h_scalar,
        h_soc=h_soc,
        h_total=h_total,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )
