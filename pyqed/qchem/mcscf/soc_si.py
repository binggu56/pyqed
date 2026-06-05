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


@dataclass
class SingletTripletSOCResult:
    singlet: object
    triplets: dict
    singlet_root: int
    triplet_root: int
    components: dict
    norm: float
    hso: np.ndarray


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


def st_soc(
    mf,
    ncas,
    nelecas,
    ncore=None,
    singlet_root=0,
    triplet_root=0,
    mo_coeff=None,
    method='direct_ci',
    hso=None,
    one_center=True,
    with_prefactor=True,
    light_speed=None,
    order='grouped',
    model='somf',
    dm=None,
    spin_root_cushion=None,
    spin_selection_tol=None,
    verbose=0,
    run_kwargs=None,
):
    """
    Compute the three singlet-triplet SOC components.

    The public labels are triplet ``M_S = -1, 0, +1``. Internally these
    correspond to CASCI determinant sectors ``ms2 = -2, 0, +2``. The
    ``M_S = 0`` triplet is selected from the ``ms2=0`` determinant block by
    ``multiplicity=3`` and ``<S^2>`` filtering.
    """
    from pyqed.qchem.mcscf.casci import CASCI

    singlet_root = int(singlet_root)
    triplet_root = int(triplet_root)
    run_kwargs = {} if run_kwargs is None else dict(run_kwargs)

    def _run_casci(ms2, multiplicity, root):
        mc = CASCI(
            mf,
            ncas=ncas,
            nelecas=nelecas,
            ncore=ncore,
            ms2=ms2,
            multiplicity=multiplicity,
            verbose=verbose,
        )
        mc.run(
            nstates=root + 1,
            mo_coeff=mo_coeff,
            method=method,
            spin_root_cushion=spin_root_cushion,
            spin_selection_tol=spin_selection_tol,
            **run_kwargs,
        )
        return mc

    singlet = _run_casci(ms2=0, multiplicity=1, root=singlet_root)
    triplets = {
        ms: _run_casci(ms2=2 * ms, multiplicity=3, root=triplet_root)
        for ms in (-1, 0, 1)
    }

    states = [(singlet, singlet_root)] + [
        (triplets[ms], triplet_root) for ms in (-1, 0, 1)
    ]
    _validate_compatible_states(states)

    if hso is None:
        model_name = model.lower()
        if model_name == '1e':
            hso = get_soc_1e_spin_orbital(
                singlet.mf,
                representation='mo',
                mo_coeff=singlet.mo_cas,
                one_center=one_center,
                with_prefactor=with_prefactor,
                light_speed=light_speed,
                order=order,
            )
        elif model_name == 'somf':
            hso = get_soc_somf_spin_orbital(
                singlet.mf,
                representation='mo',
                mo_coeff=singlet.mo_cas,
                states=states if dm is None else None,
                dm=dm,
                one_center=one_center,
                with_prefactor=with_prefactor,
                light_speed=light_speed,
                order=order,
            )
        else:
            raise ValueError("model must be '1e' or 'somf'.")
    else:
        hso = np.asarray(hso)

    expected_shape = (2 * singlet.ncas, 2 * singlet.ncas)
    if hso.shape != expected_shape:
        raise ValueError(
            f"hso has shape {hso.shape}, expected active spin-orbital shape {expected_shape}."
        )

    components = {
        ms: singlet.soc_matrix_element(
            singlet_root,
            ket_id=triplet_root,
            other=triplets[ms],
            hso=hso,
            order=order,
        )
        for ms in (-1, 0, 1)
    }
    norm = float(np.sqrt(sum(abs(value) ** 2 for value in components.values())))

    return SingletTripletSOCResult(
        singlet=singlet,
        triplets=triplets,
        singlet_root=singlet_root,
        triplet_root=triplet_root,
        components=components,
        norm=norm,
        hso=hso,
    )
