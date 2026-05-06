#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:48:18 2026

Quantum Chemitry DMRG with U(1) particle number Symmetry Support

@author: Shuoyi Hu (hushuoyi@westlake.edu.cn)


"""


import numpy as np
import scipy.constants as const
import hashlib

from scipy.sparse.linalg import eigsh

import logging
import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson

from pyqed import au2ev, au2angstrom

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB
from pyqed.qchem.mcscf.casci import h1e_for_cas


_GLOBAL_HAMILTONIAN_MPO_CACHE = {}
_GLOBAL_HAMILTONIAN_MPO_CACHE_MAXSIZE = 8


def _store_global_hamiltonian_mpo_cache(cache_key, *, factors, info, hamiltonian=None):
    """Store one process-local Hamiltonian MPO cache entry with FIFO eviction."""

    if cache_key in _GLOBAL_HAMILTONIAN_MPO_CACHE:
        _GLOBAL_HAMILTONIAN_MPO_CACHE.pop(cache_key)
    elif len(_GLOBAL_HAMILTONIAN_MPO_CACHE) >= _GLOBAL_HAMILTONIAN_MPO_CACHE_MAXSIZE:
        _GLOBAL_HAMILTONIAN_MPO_CACHE.pop(next(iter(_GLOBAL_HAMILTONIAN_MPO_CACHE)))
    _GLOBAL_HAMILTONIAN_MPO_CACHE[cache_key] = {
        "factors": factors,
        "info": dict(info),
        "hamiltonian": hamiltonian,
    }

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators

# from numba import vectorize, float64, jit
import time
from opt_einsum import contract

from collections import namedtuple
from scipy.sparse import identity, kron, csr_matrix, diags

# from pyqed import Molecule
from pyqed.qchem.mcscf.casci import (
    CASCI,
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
    transform_eri_factors_to_mo_pair,
)
from pyqed.mps import DMRG as TensorDMRG, MPS, dense_to_symmetric_mpo
from pyqed.mps.mps import MPO as TensorMPO
from pyqed.mps.decompose import compress
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.qchem.dmrg.spatial_terms import (
    BasisSpatialFermion,
    accumulate_spatial_jw_term as _accumulate_spatial_jw_term,
    merge_term_maps as _merge_spatial_term_maps,
    spatial_one_body_term_map as _spatial_one_body_term_map,
    spatial_two_body_term_map as _spatial_two_body_term_map,
)
try:
    import pyqed.mps.symmetry as sym_module
    from pyqed.mps.symmetry import BlockTensor, tensordot, QN, SymmetryManager as BaseSymmetryManager
    SYMMETRY_AVAILABLE = True
except ImportError:
    SYMMETRY_AVAILABLE = False
    BlockTensor = None
    BaseSymmetryManager = object
from collections import defaultdict

logger = logging.getLogger(__name__)


def _combine_operator_terms(terms, tol=1e-14):
    """Combine duplicate symbolic operator terms before MPO construction."""
    combined = {}
    order = []
    for term in terms:
        qn_key = tuple(tuple(int(x) for x in np.asarray(qn).reshape(-1)) for qn in term.qn_list)
        key = (term.symbol, tuple(term.dofs), qn_key)
        if key not in combined:
            combined[key] = complex(term.factor)
            order.append((key, term))
        else:
            combined[key] += complex(term.factor)

    merged = []
    for key, template in order:
        factor = combined[key]
        if abs(factor) <= tol:
            continue
        qn = [np.array(q, dtype=int) for q in key[2]]
        merged.append(Op(template.symbol, list(template.dofs), factor=factor, qn=qn))
    return merged


def _array_digest(arr):
    """Return a stable content digest for a NumPy array."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _active_hamiltonian_cache_key(h1e, eri, *, spin_purification=False, shift=None):
    """Build a cache key for the active-space Hamiltonian MPO."""
    shift_key = None if shift is None else float(shift)
    return (
        _array_digest(h1e),
        _array_digest(eri),
        bool(spin_purification),
        shift_key,
    )


def _normalize_site(site):
    """Normalize public site-basis aliases used by the qchem DMRG wrapper."""
    basis = str(site).lower().replace("-", "_")
    if basis in {"spin", "spin_orbital", "spinorbital", "so"}:
        return "spin_orbital"
    if basis in {"spatial", "spatial_orbital", "spatialorbital"}:
        return "spatial"
    raise ValueError(
        "site must be one of 'spin_orbital' or 'spatial' "
        f"(got {site!r})."
    )


def _kron_all(operators):
    """Kronecker product for a small list of dense local operators."""
    out = np.asarray(operators[0], dtype=complex)
    for op in operators[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def _build_spatial_fermion_operators(ncas):
    """
    Build dense global fermion operators for one d=4 site per spatial orbital.

    Local basis follows ``SpinHalfFermionOperators``:
    ``|0>``, ``|up>``, ``|down>``, ``|up down>``.  A site-level JW string
    accounts for all fermions on earlier spatial orbitals, while the local
    down-spin operator already carries the intra-site up-spin sign.
    """
    local = {name: np.asarray(op, dtype=complex) for name, op in SpinHalfFermionOperators().items()}
    eye = np.eye(4, dtype=complex)
    jw = local["JW"]

    ann_up = []
    ann_down = []
    cre_up = []
    cre_down = []
    n_up = []
    n_down = []

    for p in range(ncas):
        prefix = [jw if i < p else eye for i in range(ncas)]

        ops = list(prefix)
        ops[p] = local["Cu"]
        cu = _kron_all(ops)
        ann_up.append(cu)
        cre_up.append(cu.conj().T)

        ops = list(prefix)
        ops[p] = local["Cd"]
        cd = _kron_all(ops)
        ann_down.append(cd)
        cre_down.append(cd.conj().T)

        ops = [eye] * ncas
        ops[p] = local["Nu"]
        n_up.append(_kron_all(ops))

        ops = [eye] * ncas
        ops[p] = local["Nd"]
        n_down.append(_kron_all(ops))

    return {
        "ann": [ann_up, ann_down],
        "cre": [cre_up, cre_down],
        "n_up": n_up,
        "n_down": n_down,
    }


def _build_spatial_s2_matrix(spatial_ops):
    """Dense total-spin S^2 matrix in the spatial-orbital d=4 basis."""
    ncas = len(spatial_ops["n_up"])
    dim = spatial_ops["n_up"][0].shape[0]
    sz = np.zeros((dim, dim), dtype=complex)
    sp = np.zeros_like(sz)
    sm = np.zeros_like(sz)
    for p in range(ncas):
        sz += 0.5 * (spatial_ops["n_up"][p] - spatial_ops["n_down"][p])
        sp += spatial_ops["cre"][0][p] @ spatial_ops["ann"][1][p]
        sm += spatial_ops["cre"][1][p] @ spatial_ops["ann"][0][p]
    return sz @ sz + 0.5 * (sp @ sm + sm @ sp)


def _build_spatial_active_hamiltonian_matrix(h1e, eri, *, spin_purification=False, shift=None, cutoff=1e-10):
    """Build a small dense active-space Hamiltonian in the spatial-site basis."""
    h_spatial = np.asarray(h1e[0], dtype=complex)
    eri_spatial = 0.5 * np.asarray(eri[0, 0], dtype=complex)
    ncas = h_spatial.shape[0]
    spatial_ops = _build_spatial_fermion_operators(ncas)
    dim = 4 ** ncas
    hmat = np.zeros((dim, dim), dtype=complex)

    for p, q in np.argwhere(np.abs(h_spatial) > cutoff):
        val = h_spatial[p, q]
        hmat += val * (
            spatial_ops["cre"][0][p] @ spatial_ops["ann"][0][q]
            + spatial_ops["cre"][1][p] @ spatial_ops["ann"][1][q]
        )

    for p, q, r, s in np.argwhere(np.abs(eri_spatial) > cutoff):
        val = eri_spatial[p, q, r, s]
        if p != r and s != q:
            hmat += val * (
                spatial_ops["cre"][0][p]
                @ spatial_ops["cre"][0][r]
                @ spatial_ops["ann"][0][s]
                @ spatial_ops["ann"][0][q]
            )
            hmat += val * (
                spatial_ops["cre"][1][p]
                @ spatial_ops["cre"][1][r]
                @ spatial_ops["ann"][1][s]
                @ spatial_ops["ann"][1][q]
            )
        hmat += val * (
            spatial_ops["cre"][0][p]
            @ spatial_ops["cre"][1][r]
            @ spatial_ops["ann"][1][s]
            @ spatial_ops["ann"][0][q]
        )
        hmat += val * (
            spatial_ops["cre"][1][p]
            @ spatial_ops["cre"][0][r]
            @ spatial_ops["ann"][0][s]
            @ spatial_ops["ann"][1][q]
        )

    if spin_purification:
        hmat += float(shift) * _build_spatial_s2_matrix(spatial_ops)

    hmat = 0.5 * (hmat + hmat.conj().T)
    return hmat, spatial_ops


def _spatial_hf_guess(nelecas, ncas, *, spin=0, noise=1e-3):
    """Product-state HF guess for one d=4 site per spatial orbital."""
    nelec = int(nelecas)
    n_double = nelec // 2
    has_single = nelec % 2
    single_state = 1 if spin >= 0 else 2
    mps_guess = []
    for p in range(ncas):
        vec = np.zeros((1, 4, 1), dtype=complex)
        if p < n_double:
            occ = 3
        elif p == n_double and has_single:
            occ = single_state
        else:
            occ = 0
        vec[0, occ, 0] = 1.0
        if noise:
            vec += (np.random.rand(1, 4, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess


def _mps_to_dense_vector(state):
    """Contract a small dense MPS into a full state vector."""
    psi = np.array([1.0 + 0.0j])
    for site in range(state.L):
        tensor = state._get_std_B(site)
        psi = np.tensordot(psi, tensor, axes=([-1], [0]))
    return np.squeeze(psi, axis=-1).reshape(-1)


def _build_spatial_s2_term_map(ncas, *, scale=1.0, cutoff=1e-10):
    """Build symbolic spatial-site terms for total S^2."""
    term_map = {}

    # Sz^2 = 1/4 sum_pq (n_up_p - n_down_p)(n_up_q - n_down_q)
    for p in range(ncas):
        for q in range(ncas):
            _accumulate_symbolic_term(term_map, "nu nu", [p, q], 0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "nu nd", [p, q], -0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "nd nu", [p, q], -0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "nd nd", [p, q], 0.25 * scale, tol=cutoff)

    # 1/2 (S+ S- + S- S+)
    for p in range(ncas):
        for q in range(ncas):
            _accumulate_spatial_jw_term(
                term_map,
                ["cdu", "cd", "cdd", "cu"],
                [p, p, q, q],
                0.5 * scale,
                tol=cutoff,
            )
            _accumulate_spatial_jw_term(
                term_map,
                ["cdd", "cu", "cdu", "cd"],
                [p, p, q, q],
                0.5 * scale,
                tol=cutoff,
            )

    return term_map


def _build_spatial_hamiltonian_tensor_mpo(h1e, eri, *, spin_purification=False, shift=None, cutoff=1e-10):
    """Build the spatial-orbital Hamiltonian directly as a d=4 symbolic MPO."""
    h_spatial = np.asarray(h1e[0])
    eri_spatial = np.asarray(eri[0, 0])
    ncas = h_spatial.shape[0]
    term_map = _merge_spatial_term_maps(
        _spatial_one_body_term_map(h_spatial, cutoff=cutoff),
        _spatial_two_body_term_map(eri_spatial, cutoff=cutoff),
        cutoff=cutoff,
    )

    spin_term_count = 0
    if spin_purification:
        spin_terms = _build_spatial_s2_term_map(ncas, scale=float(shift), cutoff=cutoff)
        spin_term_count = len(spin_terms)
        for (symbol, dofs), factor in spin_terms.items():
            _accumulate_symbolic_term(term_map, symbol, list(dofs), factor, tol=cutoff)

    basis_sites = [BasisSpatialFermion(i) for i in range(ncas)]
    tensor_mpo, term_count = _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        term_map,
        cutoff=cutoff,
    )
    return tensor_mpo, int(term_count), int(spin_term_count)


def _build_spin_orbital_dense_hamiltonian_tensor_mpo(h1e, eri, ncas, *, spin_purification=False, shift=None, cutoff=1e-10):
    """Build the dense-integral spin-orbital Hamiltonian MPO used by qchem DMRG."""
    ham_term_map = {}
    for p, q in np.argwhere(np.abs(h1e[0]) > cutoff):
        val = h1e[0][p, q]
        symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p, 2 * q], val)
        _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
        symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p + 1, 2 * q + 1], val)
        _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

    eri_spatial = 0.5 * eri[0, 0]
    for p, q, r, s in np.argwhere(np.abs(eri_spatial) > cutoff):
        val = eri_spatial[p, q, r, s]

        if p != r and s != q:
            symbol, dofs, factor = get_jw_term_spec(
                [r"a^\dagger", r"a^\dagger", "a", "a"],
                [2 * p, 2 * r, 2 * s, 2 * q],
                val,
            )
            _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
            symbol, dofs, factor = get_jw_term_spec(
                [r"a^\dagger", r"a^\dagger", "a", "a"],
                [2 * p + 1, 2 * r + 1, 2 * s + 1, 2 * q + 1],
                val,
            )
            _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

        symbol, dofs, factor = get_jw_term_spec(
            [r"a^\dagger", r"a^\dagger", "a", "a"],
            [2 * p, 2 * r + 1, 2 * s + 1, 2 * q],
            val,
        )
        _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
        symbol, dofs, factor = get_jw_term_spec(
            [r"a^\dagger", r"a^\dagger", "a", "a"],
            [2 * p + 1, 2 * r, 2 * s, 2 * q + 1],
            val,
        )
        _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

    basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
    tensor_mpo, term_count = _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        ham_term_map,
        cutoff=cutoff,
    )
    spin_term_count = 0
    if spin_purification:
        spin_term_map = _build_spin_purification_term_map(ncas, shift, cutoff=cutoff)
        spin_mpo, spin_term_count = _build_tensor_mpo_from_symbolic_terms(
            basis_sites,
            spin_term_map,
            cutoff=cutoff,
        )
        tensor_mpo = tensor_mpo + spin_mpo

    return tensor_mpo, int(term_count), int(spin_term_count)


def _group_spin_orbital_mpo_pairs(tensor_mpo):
    """
    Fuse adjacent spin-orbital MPO sites ``(alpha, beta)`` into spatial d=4 sites.

    The spin-orbital product basis is reordered from
    ``|00>, |01>, |10>, |11>`` to the spatial convention
    ``|empty>, |up>, |down>, |double>``.
    """
    factors = tensor_mpo.factors if isinstance(tensor_mpo, TensorMPO) else tensor_mpo
    if len(factors) % 2:
        raise ValueError("Spin-orbital MPO must have an even number of sites to group into spatial sites.")

    product_for_spatial = [0, 2, 1, 3]
    grouped = []
    for i in range(0, len(factors), 2):
        up = np.asarray(factors[i])
        down = np.asarray(factors[i + 1])
        if up.shape[2:] != (2, 2) or down.shape[2:] != (2, 2):
            raise ValueError("Expected spin-orbital MPO physical dimensions (2, 2).")
        pair = np.tensordot(up, down, axes=([1], [0]))
        pair = pair.transpose(0, 3, 1, 4, 2, 5).reshape(up.shape[0], down.shape[1], 4, 4)
        pair = pair[:, :, product_for_spatial, :][:, :, :, product_for_spatial]
        grouped.append(pair)
    return TensorMPO(grouped, homogenous=False)


def _build_grouped_spatial_s2_tensor_mpo(ncas, *, cutoff=1e-10):
    """Build total S^2 as a grouped spin-orbital MPO on spatial d=4 sites."""
    basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
    spin_mpo, _ = _build_tensor_mpo_from_symbolic_terms(
        basis_sites,
        _build_s2_term_map(ncas, scale=1.0, cutoff=cutoff),
        cutoff=cutoff,
    )
    return _group_spin_orbital_mpo_pairs(spin_mpo)


def _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=1e-14):
    """Accumulate a symbolic operator term without instantiating `Op` eagerly."""
    if abs(factor) <= tol:
        return
    key = (symbol, tuple(dofs))
    term_map[key] = term_map.get(key, 0.0) + complex(factor)
    if abs(term_map[key]) <= tol:
        term_map.pop(key, None)


def _materialize_symbolic_terms(term_map, tol=1e-14):
    """Convert accumulated symbolic terms into `Op` objects once."""
    terms = []
    for (symbol, dofs), factor in term_map.items():
        if abs(factor) <= tol:
            continue
        terms.append(Op(symbol, list(dofs), factor=factor))
    return terms


def _build_tensor_mpo_from_symbolic_terms(basis_sites, term_map, *, cutoff=1e-14):
    """Build a dense MPO from symbolic terms and wrap it in the high-level MPO class."""
    terms = _materialize_symbolic_terms(term_map, tol=cutoff)
    model = Model(basis=basis_sites, ham_terms=terms)
    mpo = Mpo(model, algo="qr")
    factors = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
    return TensorMPO(factors, homogenous=False), len(terms)


def _build_one_body_tensor_mpo(basis_sites, spatial_matrix, *, cutoff=1e-14):
    """Build the spin-summed one-body MPO O = sum_pqσ M_pq a†_{pσ} a_{qσ}."""
    ncas = spatial_matrix.shape[0]
    term_map = {}
    for p, q in np.argwhere(np.abs(spatial_matrix) > cutoff):
        val = spatial_matrix[p, q]
        symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p, 2 * q], val)
        _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)
        symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2 * p + 1, 2 * q + 1], val)
        _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)
    return _build_tensor_mpo_from_symbolic_terms(basis_sites, term_map, cutoff=cutoff)


def _compress_tensor_mpo(tensor_mpo, chi_max=None):
    """Compress a dense MPO by reshaping it to an MPS-like chain and SVD-compressing bonds."""
    if chi_max is None:
        return tensor_mpo
    if max(tensor_mpo.bond_orders()) <= chi_max:
        return tensor_mpo

    mps_factors = []
    phys_dims = []
    for W in tensor_mpo.factors:
        phys_dims.append((W.shape[2], W.shape[3]))
        W_ready = W.reshape(W.shape[0], W.shape[1], W.shape[2] * W.shape[3]).transpose(0, 2, 1)
        mps_factors.append(W_ready)

    compressed_factors = compress(mps_factors, chi_max)
    final_factors = []
    for B, (d_up, d_down) in zip(compressed_factors, phys_dims):
        B_transposed = B.transpose(0, 2, 1)
        final_factors.append(B_transposed.reshape(B_transposed.shape[0], B_transposed.shape[1], d_up, d_down))
    return TensorMPO(final_factors, homogenous=False)


def _maybe_compress_tensor_mpo(tensor_mpo, *, chi_max=None, trigger_bond=None):
    """Compress only when the MPO bond dimension exceeds a trigger."""
    if chi_max is None:
        return tensor_mpo
    max_bond = max(tensor_mpo.bond_orders())
    if trigger_bond is None:
        trigger_bond = chi_max
    if max_bond <= trigger_bond:
        return tensor_mpo
    return _compress_tensor_mpo(tensor_mpo, chi_max=chi_max)


def _build_spin_purification_term_map(ncas, shift, *, cutoff=1e-10):
    """Build symbolic terms for the first-order spin-purification penalty."""
    return _build_s2_term_map(ncas, scale=shift, cutoff=cutoff)


def _build_s2_term_map(ncas, *, scale=1.0, cutoff=1e-10):
    """
    Build symbolic spin-orbital terms for the total-spin operator ``S^2``.

    The active-space sites are interleaved spin orbitals:
    ``2*p -> alpha`` and ``2*p+1 -> beta`` for spatial orbital ``p``.

    In this basis,

    ``S^2 = S_z^2 + 1/2 (S_+ S_- + S_- S_+)``

    expands to the same on-site, density-density, and spin-flip exchange terms
    used by the direct-CI spin diagnostics. Keeping that algebra in one helper
    makes the DMRG spin-penalty MPO and the diagnostic ``<S^2>`` operator share
    exactly the same definition.
    """
    term_map = {}

    # On-site terms
    for p in range(ncas):
        _accumulate_symbolic_term(term_map, "n", [2 * p], 0.75 * scale, tol=cutoff)
        _accumulate_symbolic_term(term_map, "n", [2 * p + 1], 0.75 * scale, tol=cutoff)
        _accumulate_symbolic_term(term_map, "n n", [2 * p, 2 * p + 1], -1.5 * scale, tol=cutoff)

    # Cross-site terms
    for p in range(ncas):
        for q in range(ncas):
            if p == q:
                continue
            _accumulate_symbolic_term(term_map, "n n", [2 * p, 2 * q], 0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "n n", [2 * p + 1, 2 * q + 1], 0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "n n", [2 * p, 2 * q + 1], -0.25 * scale, tol=cutoff)
            _accumulate_symbolic_term(term_map, "n n", [2 * p + 1, 2 * q], -0.25 * scale, tol=cutoff)
            symbol, dofs, factor = get_jw_term_spec(
                [r"a^\dagger", "a", r"a^\dagger", "a"],
                [2 * p, 2 * p + 1, 2 * q + 1, 2 * q],
                scale,
            )
            _accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=cutoff)

    return term_map


def _build_low_rank_hamiltonian_tensor_mpo(
    basis_sites,
    h1_spatial,
    pair_factors,
    *,
    cutoff=1e-10,
    chi_max=None,
    trigger_bond=None,
    batch_size=4,
):
    """
    Build the electronic Hamiltonian MPO from Cholesky/low-rank pair factors
    without reconstructing the full active-space pqrs tensor.

    Using (pq|rs) = sum_P L_P[pq]^* L_P[rs], the spin-free two-electron term is

        1/2 sum_P [A_P B_P - C_P]

    where
        A_P = sum_pq L_P[pq]^* E_pq
        B_P = sum_rs L_P[rs]   E_rs
        C_P = sum_ps (L_P^* L_P)_{ps} E_ps

    and E_pq = sum_sigma a†_{pσ} a_{qσ}.
    The one-body correction sum_P C_P is folded into the base one-body MPO.
    """
    correction = np.einsum("Ppq,Pqs->ps", pair_factors.conj(), pair_factors, optimize=True)
    total_mpo, base_terms = _build_one_body_tensor_mpo(
        basis_sites,
        h1_spatial - 0.5 * correction,
        cutoff=cutoff,
    )
    total_mpo = _maybe_compress_tensor_mpo(
        total_mpo,
        chi_max=chi_max,
        trigger_bond=trigger_bond,
    )

    max_bond = max(total_mpo.bond_orders())
    factor_term_counts = []
    batch_mpo = None
    batch_terms = 0

    def _flush_batch(total_mpo, batch_mpo):
        if batch_mpo is None:
            return total_mpo, None
        total_mpo = total_mpo + batch_mpo
        total_mpo = _maybe_compress_tensor_mpo(
            total_mpo,
            chi_max=chi_max,
            trigger_bond=trigger_bond,
        )
        return total_mpo, None

    for pair_factor in pair_factors:
        left_mpo, left_terms = _build_one_body_tensor_mpo(
            basis_sites,
            pair_factor.conj(),
            cutoff=cutoff,
        )
        right_mpo, right_terms = _build_one_body_tensor_mpo(
            basis_sites,
            pair_factor,
            cutoff=cutoff,
        )
        product_mpo = left_mpo.matmul(right_mpo)
        product_mpo = _maybe_compress_tensor_mpo(
            product_mpo,
            chi_max=chi_max,
            trigger_bond=trigger_bond,
        )
        product_mpo = product_mpo * 0.5

        if batch_mpo is None:
            batch_mpo = product_mpo
        else:
            batch_mpo = batch_mpo + product_mpo

        batch_terms += 1
        batch_mpo = _maybe_compress_tensor_mpo(
            batch_mpo,
            chi_max=chi_max,
            trigger_bond=trigger_bond,
        )

        if batch_terms >= batch_size:
            total_mpo, batch_mpo = _flush_batch(total_mpo, batch_mpo)
            batch_terms = 0

        max_bond = max(max_bond, max(total_mpo.bond_orders()), max(batch_mpo.bond_orders()) if batch_mpo is not None else 0)
        factor_term_counts.append((left_terms, right_terms))

    total_mpo, batch_mpo = _flush_batch(total_mpo, batch_mpo)

    info = {
        "representation": "low_rank_mpo",
        "aux_rank": int(pair_factors.shape[0]),
        "base_one_body_terms": int(base_terms),
        "factor_one_body_terms_max": int(max((max(x) for x in factor_term_counts), default=0)),
        "mpo_max_bond": int(max_bond),
        "batch_size": int(batch_size),
    }
    return total_mpo, info

#  Fermionic Logic patch adding JW chain
def get_jw_term_robust(op_str_list, indices, factor):
    """
    Constructs a fermionic term with explicit Jordan-Wigner strings (sigma_z)
    and correct sign handling (parity).
    """
    # 1. Canonical Sort: Sort operators by site index
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
                swaps += 1

    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    # 2. Insert sigma_z filling (Jordan-Wigner String)
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        # Fill gap between previous site and current site with Z
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        # 3. Handle Creation/Annihilation Phase
        # If we are applying 'a' and there are an odd number of operators to the right, flip sign
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)


def get_jw_term_spec(op_str_list, indices, factor):
    """Return the symbolic specification for a JW term without building `Op`."""
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n - i - 1):
            if chain[j][0] > chain[j + 1][0]:
                chain[j], chain[j + 1] = chain[j + 1], chain[j]
                swaps += 1

    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        if k > 0:
            prev_site = sorted_indices[k - 1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    final_factor = factor * ((-1) ** swaps) * extra_sign
    return final_op_string, final_indices, final_factor


class SymmetryManager(BaseSymmetryManager):
    """QC-DMRG compatibility alias for the shared symmetry manager."""
    pass


def _normalize_dmrg_symmetry(symmetry=None, *, symmetry_list=None):
    """
    Normalize the public DMRG symmetry selector to symmetry-manager labels.

    ``symmetry`` is the preferred public API.  It intentionally selects both
    symmetry quantum numbers and the implementation path: ``"su2"`` routes to
    the non-Abelian backend, while ``"sz"`` uses the Abelian charge/Sz backend.

    :param symmetry: Public selector such as ``None``, ``"charge"``,
        ``"sz"``, ``"u1"``, ``"su2"``, or an explicit sequence of labels.
    :param symmetry_list: Backward-compatible explicit symmetry-label list.
    :returns: ``None`` for dense DMRG or a normalized list of symmetry labels.
    """
    spec = symmetry if symmetry is not None else symmetry_list
    if spec is None or spec is False:
        return None
    if spec is True:
        return ["charge", "sz"]

    aliases = {
        "none": None,
        "off": None,
        "false": None,
        "dense": None,
        "charge": ["charge"],
        "number": ["charge"],
        "particle": ["charge"],
        "n": ["charge"],
        "sz": ["charge", "sz"],
        "s_z": ["charge", "sz"],
        "u1": ["charge", "sz"],
        "abelian": ["charge", "sz"],
        "su2": ["charge", "su2"],
        "s2": ["charge", "su2"],
        "spin": ["charge", "su2"],
        "nonabelian": ["charge", "su2"],
        "non-abelian": ["charge", "su2"],
    }
    if isinstance(spec, str):
        key = spec.strip().lower()
        key = key.replace(" ", "").replace("+", "_")
        out = aliases.get(key)
        if key in aliases:
            return None if out is None else list(out)
        if key in {"charge_sz", "charge,sz"}:
            return ["charge", "sz"]
        if key in {"charge_su2", "charge,su2"}:
            return ["charge", "su2"]
        raise ValueError(
            "Unknown DMRG symmetry {!r}. Use None, 'charge', 'sz', 'u1', or 'su2'.".format(spec)
        )

    labels = []
    for item in spec:
        key = str(item).strip().lower()
        mapped = aliases.get(key, [key])
        if mapped is None:
            continue
        labels.extend(mapped)

    normalized = []
    for label in labels:
        if label == "s2":
            label = "su2"
        if label not in {"charge", "sz", "su2"}:
            raise ValueError(
                "Unknown DMRG symmetry label {!r}. Use 'charge', 'sz', or 'su2'.".format(label)
            )
        if label not in normalized:
            normalized.append(label)
    if "su2" in normalized and "charge" not in normalized:
        normalized.insert(0, "charge")
    if "sz" in normalized and "charge" not in normalized:
        normalized.insert(0, "charge")
    if "su2" in normalized and "sz" in normalized:
        raise ValueError("DMRG symmetry cannot combine 'su2' and 'sz'.")
    return normalized or None


# Configuration generators helpers for initial guess
# non-normalized in those configs is fine. it is handeled in build_mps_from_configs.
def gen_hf_config(nelec, nsites):
    """Returns HF occupation list [1, 1, ..., 0, 0]"""
    return [1]*nelec + [0]*(nsites - nelec)

def gen_cid_configs(nelec, nsites, mixing=0.1):
    """Returns list of (config, amp) for HF + Doubles"""
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)] # HF gets weight 1.0
    # Simple Double: 2 on HOMO -> 2 on LUMO
    if nelec >= 2 and (nsites - nelec) >= 2:
        dbl = list(hf)
        dbl[nelec-1] = 0; dbl[nelec-2] = 0
        dbl[nelec]   = 1; dbl[nelec+1] = 1
        configs.append((tuple(dbl), mixing))
    return configs

def gen_random_cisd_configs(nelec, nsites, n_states=10, mixing=0.1):
    """Returns HF + Random Singles/Doubles that strictly conserve Sz."""
    # Assuming gen_hf_config returns a list like [1, 1, 1, 1, 0, 0, ...]
    hf = gen_hf_config(nelec, nsites)
    configs = [(tuple(hf), 1.0)]

    # Segregate occupied and virtual indices by Spin (Alpha=Even, Beta=Odd)
    occ_alpha = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 0]
    occ_beta  = [i for i, x in enumerate(hf) if x == 1 and i % 2 == 1]
    vir_alpha = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 0]
    vir_beta  = [i for i, x in enumerate(hf) if x == 0 and i % 2 == 1]

    for _ in range(n_states):
        new_cfg = list(hf)

        # Determine physically valid excitations based on available electrons/holes
        exc_types = []
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1: exc_types.append('S_alpha')
        if len(occ_beta) >= 1 and len(vir_beta) >= 1: exc_types.append('S_beta')
        if len(occ_alpha) >= 2 and len(vir_alpha) >= 2: exc_types.append('D_aa')
        if len(occ_beta) >= 2 and len(vir_beta) >= 2: exc_types.append('D_bb')
        if len(occ_alpha) >= 1 and len(vir_alpha) >= 1 and len(occ_beta) >= 1 and len(vir_beta) >= 1: exc_types.append('D_ab')

        if not exc_types:
            break # Active space too small for further excitations

        choice = np.random.choice(exc_types)

        if choice == 'S_alpha':
            i = np.random.choice(occ_alpha); a = np.random.choice(vir_alpha)
            new_cfg[i] = 0; new_cfg[a] = 1

        elif choice == 'S_beta':
            i = np.random.choice(occ_beta); a = np.random.choice(vir_beta)
            new_cfg[i] = 0; new_cfg[a] = 1

        elif choice == 'D_aa':
            i, j = np.random.choice(occ_alpha, 2, replace=False)
            a, b = np.random.choice(vir_alpha, 2, replace=False)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1

        elif choice == 'D_bb':
            i, j = np.random.choice(occ_beta, 2, replace=False)
            a, b = np.random.choice(vir_beta, 2, replace=False)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1

        elif choice == 'D_ab':
            # The most important correlation for singlet states
            i = np.random.choice(occ_alpha); a = np.random.choice(vir_alpha)
            j = np.random.choice(occ_beta);  b = np.random.choice(vir_beta)
            new_cfg[i] = 0; new_cfg[j] = 0; new_cfg[a] = 1; new_cfg[b] = 1

        configs.append((tuple(new_cfg), mixing))

    return configs

def build_mps_from_configs(configs_with_amps, sym_mgr, nsites, noise_scale=1e-5):
    """
    Constructs an entangled U(1) symmetric MPS from a list of determinant configurations.

    Args:
        configs_with_amps: List of tuples (occupation_list, amplitude).
        sym_mgr: SymmetryManager instance.
        nsites: Total number of sites.
        noise_scale: Magnitude of random noise to inject for symmetry breaking.

    Returns:
        List[BlockTensor]: The resulting MPS in (Left, Right, Phys) convention.
    """
    # Pre-calculate QN Trajectories for all configurations
    # traj[k] is a list of bond QNs [BoundL, Q1, Q2, ..., BoundR] for config k
    trajectories = []
    vac_qn = sym_mgr.get_vac_qn()

    for cfg, _ in configs_with_amps:
        curr_q = vac_qn
        traj = [curr_q]
        for site_i, occ in enumerate(cfg):
            state_str = 'occ' if occ > 0 else 'emp'
            phys_q = sym_mgr.get_phys_qn(site_i, state_str)
            curr_q = sym_mgr.combine(curr_q, phys_q)
            traj.append(curr_q)
        trajectories.append(traj)
    mps = []
    # 2. Build Tensors Site by Site
    for i in range(nsites):
        # Grouping Logic
        # Map QN -> List of configuration indices passing through this sector
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)

        for k, _ in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            left_groups[qL].append(k)
            right_groups[qR].append(k)
        # Fill Block Data
        data = {}
        for k, (cfg, amp) in enumerate(configs_with_amps):
            qL = trajectories[k][i]
            qR = trajectories[k][i+1]
            state_str = 'occ' if cfg[i] > 0 else 'emp'
            qP = sym_mgr.get_phys_qn(i, state_str)
            key = (qL, qR, qP)
            # Determine Matrix Coordinates (Fan-Out / Fan-In boundaries)
            # At i=0 (Left Boundary), all configs share row 0.
            # At i=L-1 (Right Boundary), all configs share col 0.
            row = 0 if i == 0 else left_groups[qL].index(k)
            col = 0 if i == nsites - 1 else right_groups[qR].index(k)

            # Initialize Block if missing
            if key not in data:
                dL = 1 if i == 0 else len(left_groups[qL])
                dR = 1 if i == nsites - 1 else len(right_groups[qR])
                # Phys dimension is always 1 per sector for spin-orbitals
                data[key] = np.zeros((dL, dR, 1), dtype=complex)

            # value = Amplitude (only applied at first site) + Noise
            val = amp if i == 0 else 1.0
            noise = (np.random.rand() - 0.5) * noise_scale
            data[key][row, col, 0] += val + noise

        # get basis data
        # Construct flat lists of QNs for the BlockTensor axes
        # Left Bond QNs
        if i == 0:
            final_qns_L = [trajectories[0][0]] # [Vacuum]
        else:
            # Repeat QN 'n' times if 'n' configs pass through it
            final_qns_L = [q for q in sorted(left_groups.keys()) for _ in left_groups[q]]
        # Right Bond QNs
        if i == nsites - 1:
            final_qns_R = [trajectories[0][-1]] # [Target]
        else:
            final_qns_R = [q for q in sorted(right_groups.keys()) for _ in right_groups[q]]
        # Physical QNs (Generic from Manager)
        final_qns_P = [sym_mgr.get_phys_qn(i, 'emp'), sym_mgr.get_phys_qn(i, 'occ')]
        # Create BlockTensor
        bt = BlockTensor(data, [final_qns_L, final_qns_R, final_qns_P], [-1, 1, 1])
        # Normalize
        nrm = bt.norm()
        if nrm > 1e-12:
            bt = bt * (1.0 / nrm)
        mps.append(bt)
    return mps


def _spin_config_to_spatial_config(cfg):
    """Convert interleaved spin-orbital occupations to spatial d=4 states."""
    spatial = []
    for p in range(len(cfg) // 2):
        up = int(cfg[2 * p])
        down = int(cfg[2 * p + 1])
        if up and down:
            spatial.append(3)
        elif up:
            spatial.append(1)
        elif down:
            spatial.append(2)
        else:
            spatial.append(0)
    return tuple(spatial)


def _spatial_state_label(state):
    return ("empty", "up", "down", "double")[int(state)]


def build_spatial_mps_from_configs(configs_with_amps, sym_mgr, nsites, noise_scale=1e-5):
    """Construct a spatial-site U(1) BlockTensor MPS from d=4 local configurations."""
    trajectories = []
    vac_qn = sym_mgr.get_vac_qn()
    phys_qns = [sym_mgr.get_phys_qn(0, label) for label in ("empty", "up", "down", "double")]
    phys_offsets = {}
    for state, qn in enumerate(phys_qns):
        phys_offsets.setdefault(qn, {})
        phys_offsets[qn][state] = len(phys_offsets[qn])

    for cfg, _ in configs_with_amps:
        curr_q = vac_qn
        traj = [curr_q]
        for state in cfg:
            phys_q = sym_mgr.get_phys_qn(0, _spatial_state_label(state))
            curr_q = sym_mgr.combine(curr_q, phys_q)
            traj.append(curr_q)
        trajectories.append(traj)

    mps = []
    for i in range(nsites):
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)
        for k, _ in enumerate(configs_with_amps):
            left_groups[trajectories[k][i]].append(k)
            right_groups[trajectories[k][i + 1]].append(k)

        data = {}
        for k, (cfg, amp) in enumerate(configs_with_amps):
            state = int(cfg[i])
            qL = trajectories[k][i]
            qR = trajectories[k][i + 1]
            qP = sym_mgr.get_phys_qn(0, _spatial_state_label(state))
            key = (qL, qR, qP)
            row = 0 if i == 0 else left_groups[qL].index(k)
            col = 0 if i == nsites - 1 else right_groups[qR].index(k)
            phys_pos = phys_offsets[qP][state]

            if key not in data:
                dL = 1 if i == 0 else len(left_groups[qL])
                dR = 1 if i == nsites - 1 else len(right_groups[qR])
                dP = len(phys_offsets[qP])
                data[key] = np.zeros((dL, dR, dP), dtype=complex)

            val = amp if i == 0 else 1.0
            noise = (np.random.rand() - 0.5) * noise_scale
            data[key][row, col, phys_pos] += val + noise

        final_qns_L = [trajectories[0][0]] if i == 0 else [q for q in sorted(left_groups.keys()) for _ in left_groups[q]]
        final_qns_R = [trajectories[0][-1]] if i == nsites - 1 else [q for q in sorted(right_groups.keys()) for _ in right_groups[q]]
        bt = BlockTensor(data, [final_qns_L, final_qns_R, phys_qns], [-1, 1, 1])
        nrm = bt.norm()
        if nrm > 1e-12:
            bt = bt * (1.0 / nrm)
        mps.append(bt)

    return mps


def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates an MPS guess based on filling the first N_elec spin-orbitals.
    used in dense branch,
    Corrected Shape: (Left, Phys, Right) -> (1, d, 1)
    """
    d = 2
    mps_guess = []
    filled_count = 0
    for i in range(n_spin):
        # (Left=1, Phys=d, Right=1)
        vec = np.zeros((1, d, 1))
        if filled_count < n_elec:
            vec[0, 1, 0] = 1.0 # Occupied
            filled_count += 1
        else:
            vec[0, 0, 0] = 1.0 # Empty
        # noise
        rand_noise = (np.random.rand(1, d, 1) - 0.5) * noise
        vec += rand_noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess


def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

# def infinite_system_algorithm(L, m):

#     initial_block = Block(length=1, basis_size=4, operator_dict={
#         "H": H1,
#         "Cu": ops['Cu'],
#         "Cd": ops['Cd'],
#         "Nu": ops['Nu'],
#         "Nd": ops['Nd']
#     })

#     block = initial_block
#     # Repeatedly enlarge the system by performing a single DMRG step, using a
#     # reflection of the current block as the environment.
#     while 2 * block.length < L:
#         print("L =", block.length * 2 + 2)
#         block, energy = single_dmrg_step(block, block, m=m)
#         print("E/L =", energy / (block.length * 2))



class DMRG(CASCI):
    """
    ab initio DRMG quantum chemistry calculation
    """
    def __init__(self, mf, ncas, nelecas, D, init_guess='hf', m_warmup=None,\
                 spin=None, tol=1e-6, low_rank_mpo=False, low_rank_mpo_bond=None,
                 low_rank_mpo_batch_size=4, verbose=0, site='spin_orbital',
                 site_basis=None, orbital_layout=None, spatial_reduced_mpo=None,
                 symmetry=None, spatial_site_basis="canonical"):
        """
        DMRG sweeping algorithm directly using DVR set (without SCF calculations)

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        D : TYPE, optional
            maximum bond dimension. The default is None.
        tol: float
            tolerance for energy convergence

        Returns
        -------
        None.

        """
        # assert(isinstance(mf, RHF1D))

        self.mf = mf
        self.verbose = int(verbose)
        normalized_symmetry = _normalize_dmrg_symmetry(symmetry)

        if site_basis is not None:
            site = site_basis
        if orbital_layout is not None:
            site = orbital_layout
        if normalized_symmetry is not None and "su2" in normalized_symmetry:
            site = "spatial"
        self.site = _normalize_site(site)
        self.site_basis = self.site
        self.orbital_layout = self.site
        spatial_site_basis = str(spatial_site_basis).lower()
        if spatial_site_basis in {"reduced", "fully-reduced", "fully_reduced_su2"}:
            spatial_site_basis = "fully_reduced"
        if spatial_site_basis not in {"canonical", "fully_reduced"}:
            raise ValueError("spatial_site_basis must be 'canonical' or 'fully_reduced'.")
        self.spatial_site_basis = spatial_site_basis

        self.d = 3 if self.site == "spatial" and self.spatial_site_basis == "fully_reduced" else 4 if self.site == "spatial" else 2

        self.nsites = self.L = ncas

        # assert(mf.eri.shape == (self.L, self.L))

        self.spin_purification = False


        self.D = self.m = D

        self.tol = tol # tolerance for energy convergence
        self.rigid_shift = 0

        if m_warmup is None:
            m_warmup = D
        self.m_warmup = m_warmup


        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        self.nelec = mf.nelec

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        assert(ncore >= 0)


        self.ncore = ncore

        if ncas > 20:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = None
        self.mo_cas = None

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin
        self.shift = None
        self.ss = None

        self.mf = mf
        # self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None
        self.H_raw = None
        self._hamiltonian_mpo_cache_key = None
        self._symmetric_mpo_cache = {}
        self._s2_mpo_cache = {}
        self._spatial_operator_cache = None
        self._active_hamiltonian = None
        self._active_integral_build_info = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space

        self.spin_purification = False

        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None
        self.h2e_factors = None

        self.init_guess = init_guess
        self.low_rank_mpo = bool(low_rank_mpo)
        self.low_rank_mpo_bond = low_rank_mpo_bond
        self.low_rank_mpo_batch_size = int(low_rank_mpo_batch_size)
        if spatial_reduced_mpo is None:
            spatial_reduced_mpo = normalized_symmetry is not None and "su2" in normalized_symmetry
        self.spatial_reduced_mpo = bool(spatial_reduced_mpo)
        self.symmetry = normalized_symmetry
        self.saved_symmetry_list = self.symmetry

    def _log(self, message, level=1):
        if self.verbose >= level:
            print(message)

    def export_initial_guess(self, state=0, dense=False):
        """Return a reusable copy of a converged DMRG state."""
        if not hasattr(self, 'dmrg') or self.dmrg is None or self.dmrg.states is None:
            raise ValueError("No converged DMRG state available. Run DMRG first.")
        guess = self.dmrg.states[state].copy()
        if dense and hasattr(guess.factors[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            guess = symmetric_to_dense(guess)
        return guess.copy()

    def reuse_guess_from(self, other, state=0, dense=False):
        """Adopt a converged MPS from another DMRG object as the next guess."""
        self.init_guess = other.export_initial_guess(state=state, dense=dense)
        return self


    def get_initial_guess_symmetric(self, method='cid'):
        """
        New Robust Initial Guess Dispatcher.
        """
        method = method.lower()
        if self.site == "spatial":
            nspin = 2 * self.ncas
            if method == 'hf':
                configs = [(_spin_config_to_spatial_config(gen_hf_config(self.nelecas, nspin)), 1.0)]
            elif method == 'cid':
                configs = [
                    (_spin_config_to_spatial_config(cfg), amp)
                    for cfg, amp in gen_cid_configs(self.nelecas, nspin, mixing=0.5)
                ]
            elif method == 'cisd' or method == 'random':
                configs = [
                    (_spin_config_to_spatial_config(cfg), amp)
                    for cfg, amp in gen_random_cisd_configs(self.nelecas, nspin, n_states=20)
                ]
            else:
                self._log(f"  [Warning] Method {method} not found. Defaulting to HF.")
                configs = [(_spin_config_to_spatial_config(gen_hf_config(self.nelecas, nspin)), 1.0)]
            return build_spatial_mps_from_configs(configs, self.sym_mgr, self.ncas)

        nsites = 2 * self.ncas

        # Ensure Manager exists (created in run())
        if not hasattr(self, 'sym_mgr'):
            self.sym_mgr = SymmetryManager(['charge', 'sz']) # Default fallback

        self._log(f"  [InitGuess] Generating guess: '{method}' with {self.sym_mgr.sym_types}")

        # 1. Generate Configurations (Physics)
        if method == 'hf':
            hf_cfg = gen_hf_config(self.nelecas, nsites)
            configs = [(hf_cfg, 1.0)]

        elif method == 'cid':
            configs = gen_cid_configs(self.nelecas, nsites, mixing=0.5)

        elif method == 'cisd' or method == 'random':
            configs = gen_random_cisd_configs(self.nelecas, nsites, n_states=20)

        else:
            # Fallback to HF
            self._log(f"  [Warning] Method {method} not found. Defaulting to HF.")
            hf_cfg = gen_hf_config(self.nelecas, nsites)
            configs = [(hf_cfg, 1.0)]

        # 2. Build Tensor (Math)
        mps = build_mps_from_configs(configs, self.sym_mgr, nsites)
        return mps

    def get_initial_guess_dense(self, noise=1e-3):
        if self.site == "spatial":
            return _spatial_hf_guess(self.nelecas, self.ncas, spin=self.spin, noise=noise)
        return get_noisy_hf_guess(self.nelecas, 2*self.ncas, noise=noise)

    def _resolve_initial_guess(self, use_symmetry):
        guess = self.init_guess
        if isinstance(guess, MPS):
            self._log("  Reusing MPS initial guess.")
            return guess.copy()

        if isinstance(guess, str) and guess.lower() == 'previous':
            if hasattr(self, 'dmrg') and self.dmrg is not None and self.dmrg.ground_state is not None:
                self._log("  Reusing previous DMRG state as initial guess.")
                return self.dmrg.ground_state.copy()
            self._log("  [Warning] previous initial guess requested, but no prior DMRG state exists. Falling back to CID.")
            guess = 'cid'

        if use_symmetry:
            if not isinstance(guess, str):
                raise TypeError(f"Unsupported symmetric initial guess type: {type(guess)}")
            self._log(f"  Generating Initial Guess ({guess})...")
            return self.get_initial_guess_symmetric(method=guess.lower())

        if isinstance(guess, str):
            self._log(f"  Generating Initial Guess ({guess})...")
            if guess.lower() == 'hf':
                return self.get_initial_guess_dense(noise=1e-3)
            if guess.lower() in {'cid', 'cisd', 'random', 'previous'}:
                return self.get_initial_guess_dense(noise=1e-3)
            raise ValueError(f"Unsupported dense initial guess string: {guess}")

        raise TypeError(f"Unsupported initial guess type: {type(guess)}")

    def fix_nelec(self, shift):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{N} - N)^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    # def fix_spin(self, shift, spin=0, ss = 0):
    #     """
    #     fix the number of electrons by energy penalty

    #     .. math::

    #         \mathcal{H} = H + \lambda (\hat{S}^2 - S(S+1))^2

    #     Parameters
    #     ----------
    #     shift : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # self.h1e += ...
    #     # self.eri += ...
    #     return self

    def fix_spin(self, s=None, ss=0, shift=0.2):
        """
        Bias the DMRG optimization toward spin-pure states with a linear ``S^2`` penalty.

        .. math::

            H' = H + \mu \hat{S}^2

        Parameters
        ----------
        s : TYPE, optional
            DESCRIPTION. The default is None.
        ss : TYPE, optional
            DESCRIPTION. The default is 0.
        shift : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        None.
        """
        if s is None:
            s = (np.sqrt(4*ss+1)-1)/2
            if not np.isclose(2*s, round(2*s)):
                raise Warning("s = {} inconsistent spin value".format(s))
        else:
            if ss is None:
                ss = s * (s+1)
            else:
                raise ValueError('s and ss cannot be specified simultaneously.')
        ss = float(ss)
        shift = float(shift)

        ms = abs(float(self.spin)) / 2.0
        min_ss = ms * (ms + 1.0)
        if not np.isclose(ss, min_ss):
            warnings.warn(
                "The current DMRG spin-purification path adds a linear +shift*S^2 penalty. "
                "With fixed Sz symmetry this biases toward the lowest-S state in the selected "
                "Sz sector, but it is not an exact projector onto an arbitrary target S(S+1).",
                RuntimeWarning,
            )

        # First-order spin penalty J. Phys. Chem. A 2022, 126, 12, 2050-2060:
        # H' = H + J \hat{S}^2
        #
        # The target ``ss`` is still stored for reporting/diagnostics, but the
        # linear penalty itself only depends on S^2.  Any constant target shift
        # would not change the optimized wavefunction.
        self.ss = ss
        self.shift = shift
        self.spin_purification = True

        return self

    def get_SO_matrix(
        self,
        spin_flip=False,
        H1=None,
        H2=None,
    ):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip

        Returns
        -------
        H1: list of [h1e_a, h1e_b]
        H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
        """
        # from pyscf import ao2mo

        mf = self.mf
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky=None)

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = h1e_for_cas(mf, ncas=self.ncas, ncore=self.ncore, mo_coeff=self.mo_coeff)

        self.e_core = energy_core


        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        # H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        # nmo = Ca.shape[1] # n

        if use_cholesky:
            eri_factors = _get_mf_cholesky_factors(mf)
            pair_factors = transform_eri_factors_to_mo_pair(eri_factors, Ca)
            flat_pair_factors = pair_factors.reshape(pair_factors.shape[0], -1)
            eri_aa = (flat_pair_factors.conj().T @ flat_pair_factors).reshape(self.ncas, self.ncas, self.ncas, self.ncas)
            build_mode = 'cholesky'
            aux_rank = int(pair_factors.shape[0])
        else:
            eri = mf.eri  # (pq||rs) 1^* 1 2^* 2
            if eri is None:
                raise ValueError(
                    "DMRG dense active-integral build requires mf.eri. "
                    "Enable cholesky on the mean-field reference when running "
                    "on factor-only RHF references."
                )
            eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)
            build_mode = 'dense'
            aux_rank = None

        # physicts notation <pq|rs>
        # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

        # eri_aa -= eri_aa.swapaxes(1,3)
        eri_bb = eri_aa.copy()
        eri_ab = eri_aa.copy()
        eri_ba = eri_aa.copy()




        # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
        #                         compact=False)).reshape((n,n,n,n), order="C")
        # eri_aa -= eri_aa.swapaxes(1,3)

        # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # eri_bb -= eri_bb.swapaxes(1,3)

        # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
        # compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))
        self._active_integral_build_info = {
            'mode': build_mode,
            'aux_rank': aux_rank,
            'ncas': self.ncas,
        }

        # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
                         # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
        H1 = [H, H]

        if spin_flip:
            raise NotImplementedError('Spin-flip matrix elements not implemented yet')
        #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
        #     return H1, H2, H2_SF
        # else:
        #     return H1, H2
        return H1, H2

    def _get_active_hamiltonian_inputs(self):
        """
        Return the active-space one-body Hamiltonian together with either the
        dense active ERI tensor or Cholesky pair factors.
        """
        mf = self.mf
        use_cholesky = _resolve_use_cholesky_integrals(self.mf, use_cholesky=None)

        H, energy_core = h1e_for_cas(
            mf,
            ncas=self.ncas,
            ncore=self.ncore,
            mo_coeff=self.mo_coeff,
        )
        self.e_core = energy_core

        if use_cholesky:
            pair_factors = transform_eri_factors_to_mo_pair(
                _get_mf_cholesky_factors(mf),
                self.mo_cas,
            )
            self._active_integral_build_info = {
                "mode": "cholesky",
                "aux_rank": int(pair_factors.shape[0]),
                "ncas": self.ncas,
            }
            return [H, H], None, pair_factors

        eri = mf.eri
        if eri is None:
            raise ValueError(
                "DMRG dense active-integral build requires mf.eri. "
                "Enable cholesky on the mean-field reference when running "
                "on factor-only RHF references."
            )
        eri_aa = contract("ip, jq, ijkl, kr, ls -> pqrs", self.mo_cas.conj(), self.mo_cas, eri, self.mo_cas.conj(), self.mo_cas)
        H2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))
        self._active_integral_build_info = {
            "mode": "dense",
            "aux_rank": None,
            "ncas": self.ncas,
        }
        return [H, H], H2, None

    def build(self, mo_coeff=None):

        # 1. Extract Integrals & dims
        # mol = mf.mol
        # mf = self.mf
        # if self.ncore == 0:
        #     h1 = mf.get_hcore_mo()
        #     eri = mf.get_eri_mo(notation='chem') # (pq|rs)
        # else:
        #     h1e, eri = self.get_SO_matrix()

        # self.nstates = nstates

        # if method == 'ci':

        ncore = self.ncore
        ncas = self.ncas

        # define the core and active space orbitals
        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff # use HF MOs
        else:
            self.mo_coeff = mo_coeff

        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]


        # effective H for CAS
        h1e, eri, pair_factors = self._get_active_hamiltonian_inputs()
        use_low_rank_mpo = bool(self.low_rank_mpo)
        if self.site == "spatial":
            use_low_rank_mpo = False
        if eri is None and pair_factors is not None and not use_low_rank_mpo:
            flat_pair_factors = pair_factors.reshape(pair_factors.shape[0], -1)
            eri_aa = (flat_pair_factors.conj().T @ flat_pair_factors).reshape(self.ncas, self.ncas, self.ncas, self.ncas)
            eri = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))
        self.h1e = h1e
        self.h2e = eri
        self.h2e_factors = pair_factors
        cache_key = _active_hamiltonian_cache_key(
            h1e,
            eri if pair_factors is None else pair_factors,
            spin_purification=self.spin_purification,
            shift=self.shift,
        ) + (
            self.site,
            use_low_rank_mpo,
            self.low_rank_mpo_bond,
            self.low_rank_mpo_batch_size,
            self.spatial_reduced_mpo,
            self.spatial_site_basis,
        )
        if cache_key == self._hamiltonian_mpo_cache_key and self.H is not None and self.H_raw is not None:
            self._log("  Reusing Hamiltonian MPO cache.")
            return self



        # h2e[0,0] -= h2e[0,0].swapaxes(1,3)
        # h2e[1,1] -= h2e[1,1].swapaxes(1,3)


        n_spatial = self.ncas
        nso = 2 * n_spatial
        self._log(f"  System: {n_spatial} spatial orbitals, {nso} spin-orbitals")

        if self.site == "spatial":
            if self.spatial_reduced_mpo:
                if self.spin_purification:
                    raise NotImplementedError(
                        "spatial_reduced_mpo does not support spin-purification penalties."
                    )
                cached = _GLOBAL_HAMILTONIAN_MPO_CACHE.get(cache_key)
                if cached is not None:
                    self._log("  Reusing global spatial reduced Hamiltonian MPO cache.")
                    self._spatial_operator_cache = None
                    self._active_hamiltonian = cached.get("hamiltonian")
                    self.H_raw = cached["factors"]
                    self.H = cached["factors"]
                    self._hamiltonian_mpo_cache_key = cache_key
                    self._symmetric_mpo_cache = {}
                    self._active_integral_build_info.update(dict(cached["info"]))
                    return self
                self._log("  Building spatial-orbital Hamiltonian MPO with reduced SU(2) channels...")
                from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo

                reduced_hamiltonian = build_spatial_reduced_hamiltonian_mpo(
                    h1e,
                    eri,
                    cutoff=1e-10,
                    fully_reduced=self.spatial_site_basis == "fully_reduced",
                    n_elec=self.nelecas,
                    spin=self.spin,
                    ecore=self.e_core,
                )
                self._spatial_operator_cache = None
                self._active_hamiltonian = reduced_hamiltonian
                self.H_raw = reduced_hamiltonian.factors
                self.H = reduced_hamiltonian.factors
                self._hamiltonian_mpo_cache_key = cache_key
                self._symmetric_mpo_cache = {}
                self._active_integral_build_info.update(reduced_hamiltonian.info)
                _store_global_hamiltonian_mpo_cache(
                    cache_key,
                    factors=reduced_hamiltonian.factors,
                    info=reduced_hamiltonian.info,
                    hamiltonian=reduced_hamiltonian,
                )
                return self

            self._log("  Building spatial-orbital Hamiltonian MPO by grouping spin-orbital pairs...")
            self._active_hamiltonian = None
            spin_tensor_mpo, spin_term_count, spin_penalty_term_count = _build_spin_orbital_dense_hamiltonian_tensor_mpo(
                h1e,
                eri,
                n_spatial,
                spin_purification=self.spin_purification,
                shift=self.shift,
                cutoff=1e-10,
            )
            tensor_mpo = _group_spin_orbital_mpo_pairs(spin_tensor_mpo)
            self._spatial_operator_cache = None
            self.H_raw = tensor_mpo.factors
            self.H = tensor_mpo.factors
            self._hamiltonian_mpo_cache_key = cache_key
            self._symmetric_mpo_cache = {}
            self._active_integral_build_info.update(
                {
                    "representation": "spatial_grouped_spin_mpo",
                    "symbolic_terms": int(spin_term_count),
                    "mpo_max_bond": int(max(tensor_mpo.bond_orders())),
                    "site": "spatial",
                }
            )
            if self.spin_purification:
                self._active_integral_build_info["spin_penalty_terms"] = int(spin_penalty_term_count)
            return self

        # 2. Build Hamiltonian (Using Robust JW Builder)
        self._log("  Building Hamiltonian MPO...")
        cutoff = 1e-10
        basis_sites = [BasisSimpleElectron(i) for i in range(nso)]
        if pair_factors is not None and use_low_rank_mpo:
            low_rank_chi_max = self.low_rank_mpo_bond
            if low_rank_chi_max is None:
                low_rank_chi_max = max(4 * int(self.D), 64)
            low_rank_trigger_bond = max(2 * int(low_rank_chi_max), int(low_rank_chi_max))
            tensor_mpo, low_rank_info = _build_low_rank_hamiltonian_tensor_mpo(
                basis_sites,
                np.asarray(h1e[0]),
                np.asarray(pair_factors),
                cutoff=cutoff,
                chi_max=low_rank_chi_max,
                trigger_bond=low_rank_trigger_bond,
                batch_size=self.low_rank_mpo_batch_size,
            )
            self._active_integral_build_info.update(low_rank_info)
            self._active_integral_build_info["compression_bond"] = int(low_rank_chi_max)
            self._active_integral_build_info["compression_trigger_bond"] = int(low_rank_trigger_bond)
        else:
            ham_term_map = {}
            # --- One-Body Terms: h_pq a+_p a_q ---
            for p, q in np.argwhere(np.abs(h1e[0]) > cutoff):
                val = h1e[0][p, q]
                symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2*p, 2*q], val)
                _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
                symbol, dofs, factor = get_jw_term_spec([r"a^\dagger", "a"], [2*p+1, 2*q+1], val)
                _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

            # --- Two-Body Terms: 0.5 * (pq|rs) a+_p a+_r a_s a_q ---
            eri_spatial = 0.5 * eri[0, 0]
            for p, q, r, s in np.argwhere(np.abs(eri_spatial) > cutoff):
                val = eri_spatial[p, q, r, s]

                if p != r and s != q:
                    symbol, dofs, factor = get_jw_term_spec(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2*p, 2*r, 2*s, 2*q], val
                    )
                    _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
                    symbol, dofs, factor = get_jw_term_spec(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2*p+1, 2*r+1, 2*s+1, 2*q+1], val
                    )
                    _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

                symbol, dofs, factor = get_jw_term_spec(
                    [r"a^\dagger", r"a^\dagger", "a", "a"],
                    [2*p, 2*r+1, 2*s+1, 2*q], val
                )
                _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)
                symbol, dofs, factor = get_jw_term_spec(
                    [r"a^\dagger", r"a^\dagger", "a", "a"],
                    [2*p+1, 2*r, 2*s, 2*q+1], val
                )
                _accumulate_symbolic_term(ham_term_map, symbol, dofs, factor, tol=cutoff)

            tensor_mpo, dense_term_count = _build_tensor_mpo_from_symbolic_terms(
                basis_sites,
                ham_term_map,
                cutoff=cutoff,
            )
            representation = "dense_term_mpo" if pair_factors is None else "cholesky_dense_active_mpo"
            self._active_integral_build_info.update(
                {
                    "representation": representation,
                    "symbolic_terms": int(dense_term_count),
                    "mpo_max_bond": int(max(tensor_mpo.bond_orders())),
                }
            )

        if self.spin_purification:
            spin_term_map = _build_spin_purification_term_map(ncas, self.shift, cutoff=cutoff)
            spin_mpo, spin_term_count = _build_tensor_mpo_from_symbolic_terms(
                basis_sites,
                spin_term_map,
                cutoff=cutoff,
            )
            tensor_mpo = tensor_mpo + spin_mpo
            self._active_integral_build_info["spin_penalty_terms"] = int(spin_term_count)

        self.H_raw = tensor_mpo.factors
        self.H = tensor_mpo.factors
        self._hamiltonian_mpo_cache_key = cache_key
        self._symmetric_mpo_cache = {}

        return self

    def calc_spin_square(self):
        """
        Builds the S^2 MPO and evaluates its expectation value.

        Returns
        -------
        _type_
            _description_
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            return 0.0

        if self.site == "spatial":
            import pyqed.mps.mps as mps_lib

            s2_cache_key = ("spatial_grouped", int(self.ncas))
            s2_mpo = self._s2_mpo_cache.get(s2_cache_key)
            if s2_mpo is None:
                s2_mpo = _build_grouped_spatial_s2_tensor_mpo(self.ncas)
                self._s2_mpo_cache[s2_cache_key] = s2_mpo
            states_to_eval = self.dmrg.states if getattr(self.dmrg, "states", None) is not None else [self.dmrg.ground_state]
            s2_vals = []
            for state in states_to_eval:
                state_for_eval = state
                if hasattr(state.Bs[0], 'qns'):
                    from pyqed.mps.mps import symmetric_to_dense
                    state_for_eval = symmetric_to_dense(state)
                s2 = mps_lib.expect_mps(state_for_eval.Bs, s2_mpo.factors, state_for_eval.Bs)
                norm = state_for_eval.norm()
                s2_vals.append(float(np.real(s2 / norm)))
            return np.array(s2_vals) if self.nstates > 1 else s2_vals[0]

        import pyqed.mps.mps as mps_lib

        ncas = self.ncas
        s2_term_map = _build_s2_term_map(ncas, scale=1.0)

        s2_cache_key = int(ncas)
        mpo_dense = self._s2_mpo_cache.get(s2_cache_key)
        if mpo_dense is None:
            basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
            s2_terms = _materialize_symbolic_terms(s2_term_map)
            model = Model(basis=basis_sites, ham_terms=s2_terms)
            mpo = Mpo(model, algo="qr")
            mpo_dense = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
            self._s2_mpo_cache[s2_cache_key] = mpo_dense

        states_to_eval = self.dmrg.states
        if (hasattr(self.dmrg, 'states') and self.dmrg.states is not None):
            states_to_eval = self.dmrg.states
        else:
            states_to_eval = [self.dmrg.ground_state]
        s2_vals = []

        for state in states_to_eval:
            if hasattr(state.Bs[0], 'qns'):
                dense_state = mps_lib.symmetric_to_dense(state)
                psi_for_eval = dense_state.Bs
            else:
                psi_for_eval = state.Bs

            s2 = mps_lib.expect_mps(psi_for_eval, mpo_dense, psi_for_eval)
            s2_vals.append(float(np.real(s2)))

        return np.array(s2_vals) if self.nstates > 1 else s2_vals[0]

    def overlap(self, other, bra_state_ids=None, ket_state_ids=None, s=None):
        from pyqed.qchem.dmrg.overlap import overlap as dmrg_overlap

        return dmrg_overlap(
            self,
            other,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
        )

    def overlap_unitary(
        self,
        other,
        bra_state_ids=None,
        ket_state_ids=None,
        orbital_transform=None,
        s=None,
        use_polar=False,
        unitary_tol=1e-8,
        chi_max=None,
        mpo_bond_dim=None,
        order=8,
        scale=2,
    ):
        from pyqed.qchem.dmrg.overlap import unitary_overlap

        return unitary_overlap(
            self,
            other,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            orbital_transform=orbital_transform,
            s=s,
            use_polar=use_polar,
            unitary_tol=unitary_tol,
            chi_max=chi_max,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
        )

    def overlap_biorthogonal(
        self,
        other,
        bra_state_ids=None,
        ket_state_ids=None,
        s=None,
        chi_max=None,
        mpo_bond_dim=None,
        order=4,
        scale=1,
        identity_tol=1e-10,
        phase_align_tol=1e-14,
        backend="structured",
    ):
        from pyqed.qchem.dmrg.overlap import biorthogonal_overlap

        return biorthogonal_overlap(
            self,
            other,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
            chi_max=chi_max,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
            identity_tol=identity_tol,
            phase_align_tol=phase_align_tol,
            backend=backend,
        )

    def overlap_auto(
        self,
        other,
        bra_state_ids=None,
        ket_state_ids=None,
        s=None,
        unitary_tol=1e-8,
        chi_max=None,
        mpo_bond_dim=None,
        order=8,
        scale=2,
        return_info=False,
    ):
        from pyqed.qchem.dmrg.overlap import automatic_overlap

        return automatic_overlap(
            self,
            other,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
            unitary_tol=unitary_tol,
            chi_max=chi_max,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
            return_info=return_info,
        )

    def overlap_biorthogonal_diagnostics(
        self,
        other,
        bra_state_ids=None,
        ket_state_ids=None,
        s=None,
        chi_max=None,
        mpo_bond_dim=None,
        order=4,
        scale=1,
        identity_tol=1e-10,
        phase_align_tol=1e-14,
    ):
        from pyqed.qchem.dmrg.overlap import biorthogonal_overlap_diagnostics

        return biorthogonal_overlap_diagnostics(
            self,
            other,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
            chi_max=chi_max,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
            identity_tol=identity_tol,
            phase_align_tol=phase_align_tol,
        )

    @staticmethod
    def _normalize_dmrg_schedule(D, nsweeps, D_schedule=None, nsweeps_schedule=None):
        if D_schedule is None:
            d_list = [int(D)]
        else:
            d_list = [int(x) for x in D_schedule]
            if not d_list:
                raise ValueError("D_schedule must contain at least one stage.")

        if nsweeps_schedule is None:
            total_sweeps = int(nsweeps)
            if len(d_list) == 1:
                sweep_list = [total_sweeps]
            else:
                if total_sweeps < len(d_list):
                    raise ValueError(
                        "Total nsweeps is smaller than the number of D_schedule stages. "
                        "Increase nsweeps or provide nsweeps_schedule explicitly."
                    )
                base = total_sweeps // len(d_list)
                rem = total_sweeps % len(d_list)
                sweep_list = [base] * len(d_list)
                for i in range(rem):
                    sweep_list[-(i + 1)] += 1
        elif np.isscalar(nsweeps_schedule):
            sweep_list = [int(nsweeps_schedule)] * len(d_list)
        else:
            sweep_list = [int(x) for x in nsweeps_schedule]
            if len(sweep_list) != len(d_list):
                raise ValueError("nsweeps_schedule must match D_schedule length.")

        return list(zip(d_list, sweep_list))

    def run(
        self,
        nstates=1,
        weights=None,
        symmetry_list=None,
        symmetry=None,
        nsweeps=50,
        D_schedule=None,
        nsweeps_schedule=None,
        initial_guess=None,
        mo_coeff=None,
        compute_s2=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        symmetry : str, sequence, bool, or None
            Preferred public symmetry selector.  Use ``"su2"`` for the
            non-Abelian spatial-orbital backend, ``"sz"``/``"u1"`` for the
            Abelian charge/Sz backend, ``"charge"`` for particle-number-only
            Abelian symmetry, and ``None``/``False`` for dense DMRG.
        symmetry_list : list of strings or bool
            Backward-compatible explicit labels such as ``['charge', 'sz']``.
        """
        self.nstates = nstates
        if weights is None:
            self.weights = np.ones(nstates) / nstates
        else:
            self.weights = np.array(weights)
        if symmetry is not None:
            normalized_symmetry = _normalize_dmrg_symmetry(symmetry)
            if symmetry_list is not None:
                legacy_symmetry = _normalize_dmrg_symmetry(symmetry_list=symmetry_list)
                if legacy_symmetry != normalized_symmetry:
                    raise ValueError(
                        "Received conflicting DMRG symmetry and symmetry_list arguments."
                    )
            symmetry_list = normalized_symmetry
            self.symmetry = normalized_symmetry
            self.saved_symmetry_list = normalized_symmetry
        elif symmetry_list is not None:
            symmetry_list = _normalize_dmrg_symmetry(symmetry_list=symmetry_list)
            self.symmetry = symmetry_list
            self.saved_symmetry_list = symmetry_list
        else:
            symmetry_list = getattr(self, 'saved_symmetry_list', None)
            symmetry_list = _normalize_dmrg_symmetry(symmetry_list=symmetry_list)
            self.symmetry = symmetry_list
        if initial_guess is not None:
            self.init_guess = initial_guess
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if self.H_raw is None:
            self.build()
        # Initialize Symmetry
        self.sym_mgr = SymmetryManager(symmetry_list)
        if self.sym_mgr.enabled:
            if getattr(self.sym_mgr, "has_nonabelian", False):
                if self.site != "spatial":
                    raise NotImplementedError("Non-Abelian qchem DMRG currently requires site='spatial'.")
                from pyqed.qchem.dmrg.backends.nonabelian import run_spatial_qchem_dmrg

                t0 = time.time()
                self._log(f"  [Symmetry] Enabled: {self.sym_mgr.sym_types}")
                dmrg = run_spatial_qchem_dmrg(
                    self,
                    nsweeps=nsweeps,
                    max_bond=self.D,
                    initial_guess=initial_guess,
                    conv_tol=kwargs.pop("conv_tol", self.tol),
                    nstates=nstates,
                    weights=self.weights,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.dmrg = dmrg
                e_dmrg_total = np.asarray(dmrg.e_tot, dtype=float) + self.e_core
                if nstates == 1:
                    e_dmrg_total = float(e_dmrg_total)
                self.e_tot = e_dmrg_total
                if compute_s2:
                    s = abs(float(self.spin)) / 2.0
                    self.s2 = s * (s + 1.0)
                if self.verbose >= 1:
                    print(f"  RHF Energy:         {self.mf.e_tot:.8f} Ha")
                    if nstates == 1:
                        print(f"  E(DMRG) =           {e_dmrg_total:.8f} Ha")
                        print(f"  Correlation Energy = {e_dmrg_total - self.mf.e_tot:.8f} Ha")
                    else:
                        for root, energy in enumerate(np.asarray(e_dmrg_total).reshape(-1)):
                            print(f"  Root {root} E(DMRG) = {energy:.8f} Ha")
                    if compute_s2:
                        print(f"  <S^2> =             {self.s2:.6f}")
                    print(f"  Time:               {time.time()-t0:.2f} s")
                return dmrg
            if self.site == "spatial" and "charge" not in self.sym_mgr.sym_types:
                raise NotImplementedError(
                    "Spatial-site Abelian DMRG currently requires charge symmetry. "
                    "Use symmetry_list=['charge'] or ['charge', 'sz']."
                )
            self._log(f"  [Symmetry] Enabled: {self.sym_mgr.sym_types}")
            site_qn_maps = []
            if self.site == "spatial":
                site_qn_maps = [
                    {
                        0: self.sym_mgr.get_phys_qn(i, 'empty', site_model='spatial'),
                        1: self.sym_mgr.get_phys_qn(i, 'up', site_model='spatial'),
                        2: self.sym_mgr.get_phys_qn(i, 'down', site_model='spatial'),
                        3: self.sym_mgr.get_phys_qn(i, 'double', site_model='spatial'),
                    }
                    for i in range(self.ncas)
                ]
            else:
                for i in range(self.ncas):
                    map_up = {
                        0: self.sym_mgr.get_phys_qn(2*i, 'emp'),
                        1: self.sym_mgr.get_phys_qn(2*i, 'occ')
                    }
                    site_qn_maps.append(map_up)
                    map_dn = {
                        0: self.sym_mgr.get_phys_qn(2*i+1, 'emp'),
                        1: self.sym_mgr.get_phys_qn(2*i+1, 'occ')
                    }
                    site_qn_maps.append(map_dn)
            # get MPO in symmetric form with QN index
            sym_cache_key = tuple(self.sym_mgr.sym_types)
            final_H = self._symmetric_mpo_cache.get(sym_cache_key)
            if final_H is None:
                self._log("  Converting MPO to BlockTensors...")
                final_H = dense_to_symmetric_mpo(self.H, site_qn_maps)
                self._symmetric_mpo_cache[sym_cache_key] = final_H
                self._log(f"  MPO Converted. Sites: {len(final_H)}")
            else:
                self._log("  Reusing symmetric MPO cache.")
            # Calculate Target QN
            target_qn = self.sym_mgr.get_target_qn(self.nelecas, self.spin)
            self._log(f"  Target QN set to: {target_qn}")
            use_symmetry = True
        else: # dense branch without U(1) symmetry
            final_H = self.H
            target_qn = None
            use_symmetry = False
            self.sym_mgr = None
        mps0 = self._resolve_initial_guess(use_symmetry=use_symmetry)
        schedule = self._normalize_dmrg_schedule(self.D, nsweeps, D_schedule=D_schedule, nsweeps_schedule=nsweeps_schedule)
        t0 = time.time()
        current_guess = mps0
        for stage_idx, (stage_D, stage_sweeps) in enumerate(schedule, start=1):
            if len(schedule) == 1:
                self._log(f"  Starting Sweeps (D={stage_D})...")
            else:
                self._log(f"  Starting Sweeps Stage {stage_idx}/{len(schedule)} (D={stage_D}, nsweeps={stage_sweeps})...")
            dmrg = TensorDMRG(
                final_H,
                D=stage_D,
                nsweeps=stage_sweeps,
                init_guess=current_guess,
                symmetry=use_symmetry,
                target_qn=target_qn,
                sym_mgr=self.sym_mgr,
                not_conv_err=False,
                nstates=self.nstates,
                weights=self.weights,
                verbose=self.verbose,
            )
            dmrg.run()
            current_guess = dmrg.ground_state.copy()
        self.dmrg = dmrg
        # Report
        e_dmrg_total = dmrg.e_tot + self.e_core
        if self.spin_purification:
            compute_s2 = True
        s2_val = self.calc_spin_square() if compute_s2 else None
        if self.spin_purification:
            e_dmrg_total -= self.shift * s2_val
        self.e_tot = e_dmrg_total
        if self.verbose >= 1:
            print(f"  RHF Energy:         {self.mf.e_tot:.8f} Ha")
            if self.nstates == 1:
                print(f"  E(DMRG) =           {e_dmrg_total:.8f} Ha")
                print(f"  Correlation Energy = {e_dmrg_total - self.mf.e_tot:.8f} Ha")
                if s2_val is not None:
                    print(f"  <S^2> =             {s2_val:.6f}")
                    if self.ss is not None:
                        print(f"  Target <S^2> =      {self.ss:.6f}")
            else:
                for i in range(self.nstates):
                    print(f"  Root {i} E(DMRG) = {e_dmrg_total[i]:.8f} Ha")
                    if s2_val is not None:
                        print(f"  Root {i} E(DMRG) = {e_dmrg_total[i]:.8f} Ha, <S^2> = {s2_val[i]:.6f}")
                        if self.ss is not None:
                            print(f"  Root {i} target <S^2> = {self.ss:.6f}")
            print(f"  Time:               {time.time()-t0:.2f} s")
        if use_symmetry and self.verbose >= 1:
            self.check_abelian_symmetry()
        return dmrg

    def dump(self):
        pass

    def check_abelian_symmetry(self):
        """
        Post-run analysis: Checks conservation of all active symmetries
        (Charge, Sz, etc.) by calculating expectation values via 1-RDMs.
        """
        if self.dmrg.ground_state is None:
            raise RuntimeError("  [Error] No ground state found. Run DMRG first.")

        print("\n" + "="*60)
        print("  Symmetry Conservation Check")
        print("="*60)
        # Calculate local site RDMs, returns a dict {site_idx: rho_dense (d,d)}
        try:
            rdms = self.dmrg.make_local_site_rdm()
        except Exception as e:
            print(f"  [Error] Failed to calculate RDM: {e}")
            return

        # initialize storage for quantum number
        total_N_calc = 0.0
        total_Sz_calc = 0.0

        print(f"{'Orb':<5} {'Spin':<6} {'Occ':<10} {'Sz_local':<10} {'Status'}")
        print("-" * 60)
        # Iterate over Spatial Orbitals (each splits into 2 Spin-Orbitals)
        # now still assuming d=2 (Spin-Orbital) mapping: 2*i = Up, 2*i+1 = Down
        for i in range(self.ncas):
            # Spin Up Site
            idx_up = 2 * i
            rho_up = rdms[idx_up]
            n_up = rho_up[1, 1].real
            # Spin Down Site
            idx_dn = 2 * i + 1
            rho_dn = rdms[idx_dn]
            n_dn = rho_dn[1, 1].real

            # Charge = N_up + N_dn
            n_local = n_up + n_dn
            # Spin = 1/2 * (N_up - N_dn)
            sz_local = 0.5 * (n_up - n_dn)

            total_N_calc += n_local
            total_Sz_calc += sz_local
            # get print nice looking
            def status(n):
                if n > 0.98: return "Full"
                if n < 0.02: return "."
                return "~" # Entangled
            print(f"{i:<5} {'Up':<6} {n_up:<10.5f} {0.5*n_up:<10.5f} {status(n_up)}")
            print(f"{i:<5} {'Down':<6} {n_dn:<10.5f} {-0.5*n_dn:<10.5f} {status(n_dn)}")
        print("-" * 60)

        # compare with Targets
        target_qns = self.sym_mgr.get_target_qn(self.nelecas, self.spin)
        print(f"\n  Global Conservation Summary:")
        # iterate over the active symmetries in the manager
        for idx, sym_type in enumerate(self.sym_mgr.sym_types):
            target_val = target_qns[idx]
            if sym_type in ['charge', 'n', 'particle']:
                measured = total_N_calc
                diff = abs(measured - target_val)
                label = "Charge (N)"
            elif sym_type in ['sz', 'spin', 's_z']:
                measured = total_Sz_calc * 2.0
                diff = abs(measured - target_val)
                label = "Spin (2Sz)"
            else:
                measured = 0.0
                diff = 0.0
                label = f"Unknown ({sym_type})"
            print(f"    {label:<12} : Target={target_val:<8.4f} | Measured={measured:<8.4f} | Diff={diff:.2e} ")

    def _get_state_for_rdm(self, state_id):
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a state.")
        if hasattr(self.dmrg, 'states') and isinstance(self.dmrg.states, list):
            return self.dmrg.states[state_id]
        return self.dmrg.ground_state

    def _get_spatial_ops_for_rdm(self):
        if self._spatial_operator_cache is None:
            self._spatial_operator_cache = _build_spatial_fermion_operators(self.ncas)
        return self._spatial_operator_cache

    def _make_spatial_site_rdm1(self, state_id=0, spatial=False, with_core=False):
        """1-RDM for the d=4 spatial-site backend."""
        state = self._get_state_for_rdm(state_id)
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            state = symmetric_to_dense(state)
        psi = _mps_to_dense_vector(state)
        norm = np.vdot(psi, psi)
        ops = self._get_spatial_ops_for_rdm()
        holes = {
            (sigma, p): ops["ann"][sigma][p] @ psi
            for sigma in range(2)
            for p in range(self.ncas)
        }

        p_raw = np.zeros((2 * self.ncas, 2 * self.ncas), dtype=complex)
        for sigma in range(2):
            for p in range(self.ncas):
                for q in range(self.ncas):
                    p_raw[2 * p + sigma, 2 * q + sigma] = np.vdot(holes[(sigma, p)], holes[(sigma, q)]) / norm

        if spatial or with_core:
            p_spatial = np.zeros((self.ncas, self.ncas), dtype=float)
            for p in range(self.ncas):
                for q in range(self.ncas):
                    val = p_raw[2 * p, 2 * q] + p_raw[2 * p + 1, 2 * q + 1]
                    p_spatial[q, p] = float(np.real(val))
            p_out = p_spatial
        else:
            p_out = p_raw

        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            dmat = np.zeros((norb, norb), dtype=float)
            if ncore > 0:
                np.fill_diagonal(dmat[:ncore, :ncore], 2.0)
            dmat[ncore:norb, ncore:norb] = p_out
            return dmat

        return p_out

    def _make_spatial_site_rdm2(self, state_id=0, spatial=False, with_core=False, idx_pairs=None):
        """2-RDM for the d=4 spatial-site backend."""
        state = self._get_state_for_rdm(state_id)
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            state = symmetric_to_dense(state)
        psi = _mps_to_dense_vector(state)
        norm = np.vdot(psi, psi)
        ops = self._get_spatial_ops_for_rdm()

        double_holes = {}
        for sigma in range(2):
            for p in range(self.ncas):
                first_hole = ops["ann"][sigma][p] @ psi
                for tau in range(2):
                    for r in range(self.ncas):
                        double_holes[(sigma, p, tau, r)] = ops["ann"][tau][r] @ first_hole

        if spatial or with_core:
            g_out = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas), dtype=float)
            for p in range(self.ncas):
                for q in range(self.ncas):
                    for r in range(self.ncas):
                        for s in range(self.ncas):
                            val = 0.0j
                            for sigma in range(2):
                                for tau in range(2):
                                    left = double_holes[(sigma, p, tau, r)]
                                    right = double_holes[(sigma, q, tau, s)]
                                    val += np.vdot(left, right) / norm
                            g_out[p, q, r, s] = float(np.real(val))
        else:
            nspin = 2 * self.ncas
            g_out = np.zeros((nspin, nspin, nspin, nspin), dtype=complex)
            for p in range(self.ncas):
                for q in range(self.ncas):
                    for r in range(self.ncas):
                        for s in range(self.ncas):
                            for sigma in range(2):
                                for tau in range(2):
                                    left = double_holes[(sigma, p, tau, r)]
                                    right = double_holes[(sigma, q, tau, s)]
                                    g_out[
                                        2 * p + sigma,
                                        2 * r + tau,
                                        2 * s + tau,
                                        2 * q + sigma,
                                    ] = np.vdot(left, right) / norm

        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            d2 = np.zeros((norb, norb, norb, norb), dtype=float)
            if ncore > 0:
                eye = np.eye(ncore)
                d2[:ncore, :ncore, :ncore, :ncore] = (
                    4 * np.einsum('ij,kl->ijkl', eye, eye)
                    - 2 * np.einsum('ps,rq->pqrs', eye, eye)
                )
                dm1 = self._make_spatial_site_rdm1(state_id, spatial=True, with_core=False)
                for i in range(ncore):
                    d2[i, i, ncore:norb, ncore:norb] = 2 * dm1
                    d2[ncore:norb, ncore:norb, i, i] = 2 * dm1
                    d2[i, ncore:norb, i, ncore:norb] = -dm1
                    d2[ncore:norb, i, ncore:norb, i] = -dm1
            d2[ncore:norb, ncore:norb, ncore:norb, ncore:norb] = g_out
            return d2

        return g_out

    def make_rdm1(self, state_id=0, spatial=False, with_core=False):
        """
        Calculates the 1-RDM.
        If spatial=True, spin-traces to the spatial MO basis.
        If with_core=True, re-embeds the frozen core electrons on the diagonal.
        \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>, same as CASCI make_rdm1
        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default False
        with_core : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if self.site == "spatial":
            return self._make_spatial_site_rdm1(state_id, spatial=spatial, with_core=with_core)

        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a state.")
        if hasattr(self.dmrg, 'states') and isinstance(self.dmrg.states, list):
            state = self.dmrg.states[state_id]
        else:
            state = self.dmrg.ground_state

        # Get Spin-Orbital RDM
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_state = symmetric_to_dense(state)
            dense_state.dim = 2
            P_raw = dense_state.make_rdm1()
        else:
            P_raw = state.make_rdm1()

        # Convert to Spatial MO basis if requested (or if with_core is True)
        if spatial or with_core:
            ncas = self.ncas
            P_spatial = np.zeros((ncas, ncas), dtype=float)
            for p in range(ncas):
                for q in range(ncas):
                    val = P_raw[2*p, 2*q] + P_raw[2*p+1, 2*q+1]
                    P_spatial[q,p] = float(np.real(val))
            P_out = P_spatial
        else:
            P_out = P_raw

        # Embed Frozen Core for CASSCF optimizations
        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            D = np.zeros((norb, norb), dtype=float)
            if ncore > 0:
                np.fill_diagonal(D[:ncore, :ncore], 2.0)
            D[ncore:norb, ncore:norb] = P_out
            return D

        return P_out

    def make_rdm2(self, state_id=0, spatial=False, with_core=False, idx_pairs=None):
        """
        Calculates the 2-RDM.
        If spatial=True, spin-traces to the spatial MO basis.

        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default False
        with_core : bool, optional
            _description_, by default False
        idx_pairs : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if self.site == "spatial":
            return self._make_spatial_site_rdm2(
                state_id,
                spatial=spatial,
                with_core=with_core,
                idx_pairs=idx_pairs,
            )

        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a state.")
        if hasattr(self.dmrg, 'states') and isinstance(self.dmrg.states, list):
            state = self.dmrg.states[state_id]
        else:
            state = self.dmrg.ground_state

        # Get Spin-Orbital RDM
        if hasattr(state.Bs[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            dense_state = symmetric_to_dense(state)
            dense_state.dim = 2
            G_raw = dense_state.make_rdm2()
        else:
            G_raw = state.make_rdm2()

        # Convert to Spatial MO basis if requested
        if spatial or with_core:
            ncas = self.ncas
            D_spatial = np.zeros((ncas, ncas, ncas, ncas), dtype=float)
            for p in range(ncas):
                for q in range(ncas):
                    for r in range(ncas):
                        for s in range(ncas):
                            # p^dag r^dag sq Spatial Convention: dm2[p,q,r,s] = sum_{sig, tau} <p_sig^dag r_tau^dag s_tau q_sig>
                            val = G_raw[2*p,   2*r,   2*s,   2*q] + \
                                  G_raw[2*p,   2*r+1, 2*s+1, 2*q] + \
                                  G_raw[2*p+1, 2*r,   2*s,   2*q+1] + \
                                  G_raw[2*p+1, 2*r+1, 2*s+1, 2*q+1]
                            D_spatial[p, q, r, s] = float(np.real(val))
            G_out = D_spatial
        else:
            G_out = G_raw

        # Embed Frozen Core
        if with_core:
            ncore = self.ncore
            norb = ncore + self.ncas
            D2 = np.zeros((norb, norb, norb, norb), dtype=float)
            if ncore > 0:
                I = np.eye(ncore)
                D2[:ncore, :ncore, :ncore, :ncore] = 4 * np.einsum('ij,kl->ijkl', I, I) - 2 * np.einsum('ps,rq->pqrs', I, I)

                dm1 = self.make_rdm1(state_id, spatial=True, with_core=False)
                for i in range(ncore):
                    D2[i, i, ncore:norb, ncore:norb] = 2 * dm1
                    D2[ncore:norb, ncore:norb, i, i] = 2 * dm1
                    D2[i, ncore:norb, i, ncore:norb] = -dm1
                    D2[ncore:norb, i, ncore:norb, i] = -dm1

            D2[ncore:norb, ncore:norb, ncore:norb, ncore:norb] = G_out
            return D2

        return G_out

    def make_rdm12(self, state_id=0, spatial=True, with_core=False):
        """
        standard rdm calculator used for SCF

        Parameters
        ----------
        state_id : int, optional
            _description_, by default 0
        spatial : bool, optional
            _description_, by default True
        with_core : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        return self.make_rdm1(state_id, spatial, with_core), self.make_rdm2(state_id, spatial, with_core)

    def make_local_site_rdm(self, idx=None):
        """
        Calculate the local reduced density matrices for individual, isolated spin-orbitals.

        This method traces out the rest of the chain to isolate the internal
        quantum state of specific sites.

        Parameters
        ----------
        idx : int or list of int, optional
            The specific site index (or indices) to evaluate. If None, evaluates
            the local density matrices for all sites in the active space.
            By default None.

        Returns
        -------
        dict
            A dictionary mapping the requested site indices (int) to their corresponding
            local density matrices (numpy.ndarray). For spin-orbitals with a physical
            dimension `d`, the returned matrix shape is `(d, d)`.

        Raises
        ------
        ValueError
            If the DMRG solver has not been run and no ground state is available.
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
        return self.dmrg.make_local_site_rdm(idx=idx)

    def make_diagonal_rdm2(self, idx_pairs=None):
        """
        Calculate the diagonal blocks of the 2-site reduced density matrix.

        Extracts the two-site quantum state :math:`\rho_{ij}` needed to compute
        density-density correlations (e.g., :math:`\langle n_i n_j \rangle`) without
        evaluating the full :math:`\mathcal{O}(L^4)` global 2-RDM tensor.

        Parameters
        ----------
        idx_pairs : list of tuple of int, optional
            A list of site index pairs `(i, j)` to calculate the 2-site RDM for.
            If None, computes RDMs for all possible unique pairs in the active space.
            By default None.

        Returns
        -------
        dict
            A dictionary mapping each requested `(i, j)` tuple to its corresponding
            dense reduced density matrix (numpy.ndarray). If the physical dimension
            of a single site is `d`, the returned matrix shape is `(d*d, d*d)`.

        Raises
        ------
        ValueError
            If the DMRG solver has not been run and no ground state is available.
        """
        if not hasattr(self, 'dmrg') or self.dmrg.ground_state is None:
            raise ValueError("Run DMRG first to generate a ground state.")
        return self.dmrg.make_diagonal_rdm2(idx_pairs=idx_pairs)


class DMRGSCF(DMRG):
    """
    optimize the orbitals
    """
    pass


if __name__=='__main__':

    from pyqed.qchem.mcscf.direct_ci import CASCI

    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


    from pyqed.qchem.mol import atomic_chain

    natom = 6
    z = np.linspace(-6, 6, natom)
    mol = atomic_chain(natom, z)

    # mol.basis = 'aug-ccpvdz'
    mol.basis = 'ccpvdz'
    mol.build(driver='pyscf')

    mf = mol.RHF().run()


    dmrg = DMRG(mf, ncas=10, nelecas=6, D=40) #here we could assign number of electron wanted to be not equal to the number of electron in the HF state.
    dmrg.build().run(symmetry_list=['charge','sz'], initial_guess='cid')


QCDMRG = DMRG

    # mc = CASCI(mf, ncas=8, nelecas=4)
    # mc.run()

    # conn refers to the connection operator, that is, the operator on the edge of
    # the block, on the interior of the chain.  We need to be able to represent S^z
    # and S^+ on that site in the current basis in order to grow the chain.
    # initial_block = Block(length=1, basis_size=model_d, operator_dict={
    #     "H": H1,
    #     "Cu": ops['Cu'],
    #     "Cd": ops['Cd'],
    #     "Nu": ops['Nu'],
    #     "Nd": ops['Nd']
    # })

    #infinite_system_algorithm(L=100, m=20)
    # finite_system_algorithm(L=nsites, m_warmup=10, m=10)
