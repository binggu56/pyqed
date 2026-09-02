#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin-orbit coupling helpers for scalar-reference wavefunctions.

The functions in this module are designed for the common "RHF + perturbative
SOC" workflow:

1. Run a scalar RHF calculation to obtain orthonormal MOs.
2. Build Breit-Pauli SOC integrals in the AO or MO basis.
3. Expand the spatial operator to a spin-orbital matrix for use in
   post-HF/state-interaction models.

Notes
-----
This module currently provides the one-electron nuclear spin-orbit operator and
the standard spin-orbit mean-field (SOMF) reduction of the two-electron
Breit-Pauli term.  By default the one-electron part uses the one-center
approximation, which is often the cheapest useful first model.
"""

import numpy as np
from opt_einsum import contract

from pyqed.units import fine_structure

LIGHT_SPEED = 1.0 / fine_structure


def soc_1e_prefactor(light_speed=None):
    """
    Breit-Pauli SOC prefactor in atomic units.
    """
    if light_speed is None:
        light_speed = LIGHT_SPEED
    return 1.0 / (4.0 * light_speed ** 2)


def _normalize_state_list(states):
    if states is None:
        return None

    normalized = []
    for entry in states:
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError("state entries must be CASCI objects or (casci, state_id) pairs.")
            casci, state_id = entry
        else:
            casci, state_id = entry, 0
        normalized.append((casci, int(state_id)))
    if not normalized:
        raise ValueError("states must contain at least one CASCI state.")
    return normalized


def _state_dm1_ao(casci, state_id):
    dm_mo = np.asarray(casci.make_rdm1(state_id, with_core=True, with_vir=True))
    mo_coeff = np.asarray(casci.mf.mo_coeff)
    return contract('pi,ij,qj->pq', mo_coeff, dm_mo, mo_coeff.conj())


def _resolve_soc_reference_density(mf, dm=None, states=None):
    """
    Return a spin-traced AO density matrix for SOMF contractions.
    """
    if dm is not None and states is not None:
        raise ValueError("Pass either dm or states when building a SOMF operator, not both.")

    normalized_states = _normalize_state_list(states)
    if normalized_states is not None:
        dm_ao = None
        for casci, state_id in normalized_states:
            state_dm = _state_dm1_ao(casci, state_id)
            if dm_ao is None:
                dm_ao = np.zeros_like(state_dm, dtype=complex)
            elif state_dm.shape != dm_ao.shape:
                raise ValueError("All SOMF reference states must share the same AO basis.")
            dm_ao += state_dm
        dm_ao /= len(normalized_states)
    elif dm is not None:
        dm_ao = np.asarray(dm)
    else:
        if not hasattr(mf, 'make_rdm1'):
            raise ValueError("mf must provide make_rdm1() when dm and states are omitted.")
        dm_ao = np.asarray(mf.make_rdm1())

    if dm_ao.ndim == 3:
        if dm_ao.shape[0] != 2:
            raise ValueError("Spin-resolved densities must have shape (2, nao, nao).")
        dm_ao = dm_ao[0] + dm_ao[1]
    if dm_ao.ndim != 2 or dm_ao.shape[0] != dm_ao.shape[1]:
        raise ValueError("The SOMF reference density must be a square AO density matrix.")
    return 0.5 * (dm_ao + dm_ao.conj().T)


def _get_pyscf_mol(mol):
    if hasattr(mol, 'intor') and hasattr(mol, 'nao_nr'):
        return mol
    if hasattr(mol, 'topyscf'):
        return mol.topyscf()
    raise ValueError("A PySCF Mole object or a pyqed Molecule with topyscf() is required.")


def get_pvxp_ao(mol, one_center=True):
    """
    Raw three-component ``p V x p`` operator in the AO basis.

    Parameters
    ----------
    mol : pyqed.qchem.Molecule-like
        Must provide ``topyscf()``.
    one_center : bool
        If ``True``, build the standard one-center approximation by keeping
        only atom-local shell blocks for each nucleus.  If ``False``, build the
        full one-electron nuclear SOC operator by summing over all nuclei.

    Returns
    -------
    ndarray
        Array of shape ``(3, nao, nao)`` with components ``(x, y, z)``.
    """
    pmol = _get_pyscf_mol(mol)
    pmol.build()
    mat = np.zeros((3, pmol.nao_nr(), pmol.nao_nr()), dtype=float)
    atom_charges = np.asarray(pmol.atom_charges(), dtype=float)
    ao_slices = pmol.aoslice_by_atom()
    for ia in range(pmol.natm):
        with pmol.with_rinv_as_nucleus(ia):
            block = -atom_charges[ia] * pmol.intor('int1e_prinvxp', comp=3)
        if one_center:
            p0, p1 = ao_slices[ia, 2:4]
            mat[:, p0:p1, p0:p1] = block[:, p0:p1, p0:p1]
        else:
            mat += block
    if hasattr(mol, 'pyscf_ao_permutation'):
        permutation = mol.pyscf_ao_permutation(pmol)
        mat = mat[:, permutation][:, :, permutation]
    return mat


def get_soc_1e_ao(mol, one_center=True, with_prefactor=True, light_speed=None):
    """
    One-electron Breit-Pauli SOC vector operator in the AO basis.
    """
    mat = get_pvxp_ao(mol, one_center=one_center)
    if with_prefactor:
        mat = soc_1e_prefactor(light_speed=light_speed) * mat
    return mat


def get_soc_2e_somf_ao(mf, dm=None, states=None, with_prefactor=True, light_speed=None):
    """
    Two-electron Breit-Pauli SOC reduced to a SOMF AO operator.

    Parameters
    ----------
    mf : RHF-like object
        Mean-field reference used to define the AO basis.
    dm : ndarray, optional
        Spin-traced AO density matrix for the SOMF contraction.
    states : sequence, optional
        CASCI states to average into the SOMF reference density. Entries can be
        CASCI objects or ``(casci, state_id)`` pairs.
    """
    pmol = _get_pyscf_mol(mf.mol)
    dm_ao = _resolve_soc_reference_density(mf, dm=dm, states=states)
    nao = pmol.nao_nr()
    if dm_ao.shape != (nao, nao):
        raise ValueError(
            f"SOMF density has shape {dm_ao.shape}, expected AO shape {(nao, nao)}."
        )

    permutation = None
    dm_pyscf = dm_ao
    if hasattr(mf.mol, 'pyscf_ao_permutation'):
        permutation = mf.mol.pyscf_ao_permutation(pmol)
        inverse = np.argsort(permutation)
        dm_pyscf = dm_ao[np.ix_(inverse, inverse)]

    g = pmol.intor('int2e_p1vxp1', comp=3)
    term1 = contract('xpqrs,rs->xpq', g, dm_pyscf)
    term2 = contract('xprsq,rs->xpq', g, dm_pyscf)
    term3 = contract('xsqpr,rs->xpq', g, dm_pyscf)
    mat = term1 - 1.5 * term2 - 1.5 * term3
    if permutation is not None:
        mat = mat[:, permutation][:, :, permutation]
    if with_prefactor:
        mat = soc_1e_prefactor(light_speed=light_speed) * mat
    return mat


def get_soc_1e_mo(mf, mo_coeff=None, one_center=True, with_prefactor=True,
                  light_speed=None):
    """
    One-electron Breit-Pauli SOC vector operator in the MO basis.

    Parameters
    ----------
    mf : RHF-like object
        Must provide ``mol`` and converged ``mo_coeff``.
    mo_coeff : ndarray, optional
        MO coefficients.  If ``None``, use ``mf.mo_coeff``.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    if mo_coeff is None:
        raise ValueError("MO coefficients are required. Run RHF first.")

    hso_ao = get_soc_1e_ao(
        mf.mol,
        one_center=one_center,
        with_prefactor=with_prefactor,
        light_speed=light_speed,
    )
    return contract('xpq,pi,qj->xij', hso_ao, mo_coeff.conj(), mo_coeff)


def get_soc_2e_somf_mo(mf, mo_coeff=None, dm=None, states=None,
                       with_prefactor=True, light_speed=None):
    """
    Two-electron SOMF SOC vector operator in the MO basis.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    if mo_coeff is None:
        raise ValueError("MO coefficients are required. Run RHF first.")

    hso_ao = get_soc_2e_somf_ao(
        mf,
        dm=dm,
        states=states,
        with_prefactor=with_prefactor,
        light_speed=light_speed,
    )
    return contract('xpq,pi,qj->xij', hso_ao, mo_coeff.conj(), mo_coeff)


def get_soc_somf_ao(mf, dm=None, states=None, include_1e=True, one_center=True,
                    with_prefactor=True, light_speed=None):
    """
    Full SOMF AO operator: one-electron Breit-Pauli term plus the SOMF
    reduction of the two-electron SOC operator.
    """
    hso = get_soc_2e_somf_ao(
        mf,
        dm=dm,
        states=states,
        with_prefactor=with_prefactor,
        light_speed=light_speed,
    )
    if include_1e:
        hso = hso + get_soc_1e_ao(
            mf.mol,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    return hso


def get_soc_somf_mo(mf, mo_coeff=None, dm=None, states=None, include_1e=True,
                    one_center=True, with_prefactor=True, light_speed=None):
    """
    Full SOMF SOC vector operator in the MO basis.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    if mo_coeff is None:
        raise ValueError("MO coefficients are required. Run RHF first.")

    hso_ao = get_soc_somf_ao(
        mf,
        dm=dm,
        states=states,
        include_1e=include_1e,
        one_center=one_center,
        with_prefactor=with_prefactor,
        light_speed=light_speed,
    )
    return contract('xpq,pi,qj->xij', hso_ao, mo_coeff.conj(), mo_coeff)


def reorder_spin_orbital_matrix(mat, source='interleaved', target='grouped'):
    """
    Reorder a spin-orbital matrix between interleaved and grouped layouts.

    Parameters
    ----------
    mat : ndarray
        Square matrix with shape ``(2*n, 2*n)``.
    source : {'interleaved', 'grouped'}
        Current spin-orbital ordering of ``mat``.
    target : {'interleaved', 'grouped'}
        Requested spin-orbital ordering.
    """
    source = source.lower()
    target = target.lower()
    if source == target:
        return mat
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] % 2 != 0:
        raise ValueError("mat must be a square (2*n, 2*n) spin-orbital matrix.")
    if source not in {'interleaved', 'grouped'} or target not in {'interleaved', 'grouped'}:
        raise ValueError("source and target must be 'interleaved' or 'grouped'.")

    norb = mat.shape[0] // 2
    interleaved_from_grouped = np.empty(2 * norb, dtype=int)
    interleaved_from_grouped[0::2] = np.arange(norb)
    interleaved_from_grouped[1::2] = norb + np.arange(norb)

    if source == 'grouped' and target == 'interleaved':
        perm = interleaved_from_grouped
    else:
        perm = np.argsort(interleaved_from_grouped)
    return mat[np.ix_(perm, perm)]


def spatial_soc_to_spin_orbital(hso_xyz, order='interleaved'):
    """
    Expand a 3-component spatial SOC operator to a 2-spinor matrix.

    Parameters
    ----------
    hso_xyz : ndarray
        Array of shape ``(3, n, n)``.
    order : {'interleaved', 'grouped'}
        Spin-orbital ordering of the returned matrix.

    Returns
    -------
    ndarray
        Complex Hermitian matrix of shape ``(2*n, 2*n)`` in the spin-orbital
        basis.
    """
    pauli = 1j * np.asarray([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ], dtype=complex)
    n = hso_xyz.shape[-1]
    mat = np.einsum('sxy,spq->xpyq', pauli, hso_xyz).reshape(2 * n, 2 * n)
    return reorder_spin_orbital_matrix(mat, source='grouped', target=order)


def get_soc_1e_spin_orbital(mf, representation='mo', mo_coeff=None,
                            one_center=True, with_prefactor=True,
                            light_speed=None, order='interleaved'):
    """
    One-electron SOC Hamiltonian in a spin-orbital basis.

    Parameters
    ----------
    mf : RHF-like object
        Mean-field reference.
    representation : {'ao', 'mo'}
        Spatial basis in which the spin-orbital Hamiltonian is built.
    """
    rep = representation.lower()
    if rep == 'ao':
        hso_xyz = get_soc_1e_ao(
            mf.mol,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    elif rep == 'mo':
        hso_xyz = get_soc_1e_mo(
            mf,
            mo_coeff=mo_coeff,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    else:
        raise ValueError("representation must be 'ao' or 'mo'.")

    return spatial_soc_to_spin_orbital(hso_xyz, order=order)


def get_soc_2e_somf_spin_orbital(mf, representation='mo', mo_coeff=None, dm=None,
                                 states=None, with_prefactor=True,
                                 light_speed=None, order='interleaved'):
    """
    Two-electron SOMF SOC Hamiltonian in a spin-orbital basis.
    """
    rep = representation.lower()
    if rep == 'ao':
        hso_xyz = get_soc_2e_somf_ao(
            mf,
            dm=dm,
            states=states,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    elif rep == 'mo':
        hso_xyz = get_soc_2e_somf_mo(
            mf,
            mo_coeff=mo_coeff,
            dm=dm,
            states=states,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    else:
        raise ValueError("representation must be 'ao' or 'mo'.")

    return spatial_soc_to_spin_orbital(hso_xyz, order=order)


def get_soc_somf_spin_orbital(mf, representation='mo', mo_coeff=None, dm=None,
                              states=None, include_1e=True, one_center=True,
                              with_prefactor=True, light_speed=None,
                              order='interleaved'):
    """
    Full SOMF SOC Hamiltonian in a spin-orbital basis.
    """
    rep = representation.lower()
    if rep == 'ao':
        hso_xyz = get_soc_somf_ao(
            mf,
            dm=dm,
            states=states,
            include_1e=include_1e,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    elif rep == 'mo':
        hso_xyz = get_soc_somf_mo(
            mf,
            mo_coeff=mo_coeff,
            dm=dm,
            states=states,
            include_1e=include_1e,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )
    else:
        raise ValueError("representation must be 'ao' or 'mo'.")

    return spatial_soc_to_spin_orbital(hso_xyz, order=order)
