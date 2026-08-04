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
import operator

from scipy.sparse.linalg import eigsh

import logging
import warnings
from copy import deepcopy

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson

from pyqed import au2ev, au2angstrom

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB
from pyqed.qchem.mcscf.casci import h1e_for_cas


_GLOBAL_HAMILTONIAN_MPO_CACHE = {}
_GLOBAL_HAMILTONIAN_MPO_CACHE_MAXSIZE = 8
_GLOBAL_SYMMETRIC_MPO_CACHE = {}
_GLOBAL_SYMMETRIC_MPO_CACHE_MAXSIZE = 16


def _store_global_hamiltonian_mpo_cache(
    cache_key,
    *,
    factors,
    info,
    hamiltonian=None,
    **extra,
):
    """Store one process-local Hamiltonian MPO cache entry with FIFO eviction."""

    if cache_key in _GLOBAL_HAMILTONIAN_MPO_CACHE:
        _GLOBAL_HAMILTONIAN_MPO_CACHE.pop(cache_key)
    elif len(_GLOBAL_HAMILTONIAN_MPO_CACHE) >= _GLOBAL_HAMILTONIAN_MPO_CACHE_MAXSIZE:
        _GLOBAL_HAMILTONIAN_MPO_CACHE.pop(next(iter(_GLOBAL_HAMILTONIAN_MPO_CACHE)))
    entry = {
        "factors": factors,
        "info": deepcopy(dict(info)),
        "hamiltonian": hamiltonian,
    }
    entry.update(extra)
    _GLOBAL_HAMILTONIAN_MPO_CACHE[cache_key] = entry


def _store_global_symmetric_mpo_cache(
    cache_key,
    *,
    hamiltonian,
    complementary_mpos,
):
    """Store converted Abelian MPO tensors for reuse across DMRG instances."""

    if cache_key is None:
        return
    if cache_key in _GLOBAL_SYMMETRIC_MPO_CACHE:
        _GLOBAL_SYMMETRIC_MPO_CACHE.pop(cache_key)
    elif len(_GLOBAL_SYMMETRIC_MPO_CACHE) >= _GLOBAL_SYMMETRIC_MPO_CACHE_MAXSIZE:
        _GLOBAL_SYMMETRIC_MPO_CACHE.pop(next(iter(_GLOBAL_SYMMETRIC_MPO_CACHE)))
    _GLOBAL_SYMMETRIC_MPO_CACHE[cache_key] = {
        "hamiltonian": hamiltonian,
        "complementary_mpos": complementary_mpos,
    }


def _qn_cache_signature(qn):
    if hasattr(qn, "labels") and hasattr(qn, "components"):
        return (
            type(qn).__name__,
            tuple(str(label) for label in qn.labels),
            tuple(_qn_cache_signature(component) for component in qn.components),
        )
    if isinstance(qn, (tuple, list)):
        return (type(qn).__name__, tuple(_qn_cache_signature(item) for item in qn))
    if isinstance(qn, (np.integer, int)):
        return ("int", int(qn))
    if isinstance(qn, (np.floating, float)):
        return ("float", float(qn))
    if isinstance(qn, str):
        return ("str", qn)
    return (type(qn).__name__, repr(qn))


def _site_qn_maps_cache_signature(site_qn_maps):
    return tuple(
        tuple(
            (int(local_index), _qn_cache_signature(qn))
            for local_index, qn in sorted(site_map.items())
        )
        for site_map in tuple(site_qn_maps or ())
    )


def _global_symmetric_mpo_cache_key(
    active_hamiltonian_key,
    *,
    sym_types,
    native_site_storage,
    site_qn_maps,
):
    if active_hamiltonian_key is None:
        return None
    return (
        "qchem_global_symmetric_mpo",
        active_hamiltonian_key,
        tuple(str(sym) for sym in sym_types),
        bool(native_site_storage),
        _site_qn_maps_cache_signature(site_qn_maps),
    )

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
    transform_eri_factors_to_mo_pair,
)
from pyqed.mps import (
    DMRG as TensorDMRG,
    MPS,
    dense_to_symmetric_mpo,
    resolve_abelian_matvec_options,
)
from pyqed.mps.mps import MPO as TensorMPO
from pyqed.mps.abelian_storage import make_abelian_site_tensor
from pyqed.mps.symmetry import SymmetryManager as BaseSymmetryManager
from pyqed.mps.decompose import compress
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.qchem.dmrg.spatial_terms import (
    BasisSpatialFermion,
    accumulate_spatial_jw_term as _accumulate_spatial_jw_term,
    merge_term_maps as _merge_spatial_term_maps,
    spatial_complementary_family_term_maps as _spatial_family_term_maps,
    spatial_one_body_term_map as _spatial_one_body_term_map,
    spatial_two_generator_family_term_map as _spatial_two_generator_family_term_map,
    spatial_two_body_spinfree_term_map as _spatial_two_body_spinfree_term_map,
    spatial_two_body_term_map as _spatial_two_body_term_map,
)
from pyqed.qchem.dmrg.spatial_mpo import (
    SpatialCarrierMPO,
    build_spatial_block2_carrier_mpo,
)
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


def _normalize_integral_backend(integral_backend):
    backend = str(integral_backend or "auto").lower().replace("-", "_")
    aliases = {
        "cd": "cholesky",
        "chol": "cholesky",
        "factor": "cholesky",
        "factors": "cholesky",
        "factorized": "cholesky",
        "density_fitting": "ri",
        "df": "ri",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "dense", "ri", "cholesky"}:
        raise ValueError(
            "integral_backend must be 'auto', 'dense', 'ri', or 'cholesky' "
            f"(got {integral_backend!r})."
        )
    return backend


def _normalize_spatial_abelian_symbolic_algo(algo):
    key = str(algo or "Hopcroft-Karp").strip().lower().replace("_", "-")
    aliases = {
        "qr": "qr",
        "hopcroft-karp": "Hopcroft-Karp",
        "hopcroft": "Hopcroft-Karp",
        "hk": "Hopcroft-Karp",
        "hungarian": "Hungarian",
    }
    if key not in aliases:
        raise ValueError(
            "spatial_abelian_symbolic_algo must be one of "
            "'qr', 'Hopcroft-Karp', or 'Hungarian'."
        )
    return aliases[key]


def _normalize_spatial_family_environment_backend(backend):
    key = str(backend or "block2_table").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "none": "none",
        "false": "none",
        "0": "none",
        "block2": "block2",
        "block2_like": "block2",
        "renormalized": "block2",
        "renormalized_generators": "block2",
        "block2_table": "block2_table",
        "family_table": "block2_table",
        "operator_table": "block2_table",
        "renormalized_table": "block2_table",
        "renormalized_operator_table": "block2_table",
        "generator_table": "generator_table",
        "native_table": "generator_table",
        "block2_generator_table": "generator_table",
        "renormalized_generator_table": "generator_table",
        "block2_adaptive": "block2_adaptive",
        "adaptive_block2": "block2_adaptive",
        "block2_native": "block2_native",
        "native_block2": "block2_native",
        "native_generators": "block2_native",
        "autompo": "block2",
        "mpo": "block2",
        "generic_mpo": "block2",
        "direct": "direct_terms",
        "direct_terms": "direct_terms",
        "term": "direct_terms",
        "terms": "direct_terms",
        "generator": "generator_terms",
        "generators": "generator_terms",
        "generator_terms": "generator_terms",
        "raw_generators": "generator_terms",
    }
    if key not in aliases:
        raise ValueError(
            "spatial_family_environment_backend must be 'block2', "
            "'block2_table', 'generator_table', 'block2_native', 'none', 'autompo', "
            "'direct_terms', or 'generator_terms'."
        )
    return aliases[key]


def _normalize_spatial_native_p_grouping(grouping):
    key = str(grouping or "first_site_order").strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "all": "none",
        "single": "none",
        "unsplit": "none",
        "first": "first_site_order",
        "first_site": "first_site_order",
        "first_site_order": "first_site_order",
        "balanced": "first_site_order",
        "first_two": "first_two_site_order",
        "first_two_sites": "first_two_site_order",
        "first_two_site_order": "first_two_site_order",
        "site_order": "site_order",
        "full_site_order": "site_order",
        "order": "site_order",
    }
    if key not in aliases:
        raise ValueError(
            "spatial_native_p_grouping must be one of 'none', "
            "'first_site_order', 'first_two_site_order', or 'site_order'."
        )
    return aliases[key]


def _mf_has_factorized_eris(mf):
    return (
        getattr(mf, "eri_factors", None) is not None
        or getattr(getattr(mf, "mol", None), "eri_factors", None) is not None
    )


def _mf_has_dense_eris(mf):
    mol = getattr(mf, "mol", None)
    return (
        getattr(mf, "eri", None) is not None
        or getattr(mf, "eri_s4", None) is not None
        or getattr(mf, "eri_s8", None) is not None
        or getattr(mol, "eri", None) is not None
        or getattr(mol, "eri_s4", None) is not None
        or getattr(mol, "eri_s8", None) is not None
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


def _nonabelian_mps_to_dense_vector(state):
    """Contract a small non-Abelian spatial MPS into a full dense vector."""

    from pyqed.mps.nonabelian.environment import _site_to_dense

    sites = list(getattr(state, "sites", state))
    psi = np.array([1.0 + 0.0j])
    for site in sites:
        tensor = _site_to_dense(site)
        psi = np.tensordot(psi, tensor, axes=([-1], [0]))
    return np.squeeze(psi, axis=-1).reshape(-1)


def _spatial_rdm_dense_mps(state, site_qn_maps=None):
    """Return a standard dense d=4 MPS for spatial-site RDM contractions."""

    if hasattr(state, "sites"):
        return None
    if hasattr(state.Bs[0], "qns"):
        from pyqed.mps.mps import symmetric_to_dense

        state = symmetric_to_dense(state, site_qn_maps=site_qn_maps)
    else:
        state = state.to_order(["lv", "p", "rv"])
    if state.Bs[0].shape[1] != 4:
        raise NotImplementedError(
            f"Spatial-site RDM contractions require d=4 sites, got d={state.Bs[0].shape[1]}."
        )
    return state


def _apply_spatial_annihilation_mps(state, sigma, site):
    """Apply one spatial-site annihilation operator, including inter-site JW strings."""

    local = SpinHalfFermionOperators()
    jw = np.asarray(local["JW"], dtype=complex)
    op = np.asarray(local["Cu" if sigma == 0 else "Cd"], dtype=complex)
    eye = np.eye(4, dtype=complex)

    new_Bs = []
    for i in range(state.L):
        B = state._get_std_B(i)
        local_op = jw if i < site else op if i == site else eye
        new_B = np.tensordot(local_op, B, axes=(1, 1)).transpose(1, 0, 2)
        new_Bs.append(new_B)
    return MPS(new_Bs, labels=["lv", "p", "rv"], bc=state.bc)


def _two_hole_gram_block(two_hole_states, norm, *, zero_tol=1.0e-14):
    """Build one Hermitian Gram block for a fixed two-spin annihilation channel."""

    nstates = len(two_hole_states)
    gram = np.zeros((nstates, nstates), dtype=complex)
    active = []
    for i, state_i in enumerate(two_hole_states):
        diag = state_i._mps_dot(state_i, state_i)
        if abs(diag) <= zero_tol:
            continue
        gram[i, i] = diag / norm
        active.append(i)

    for pos, i in enumerate(active):
        state_i = two_hole_states[i]
        for j in active[pos + 1:]:
            val = state_i._mps_dot(state_i, two_hole_states[j]) / norm
            gram[i, j] = val
            gram[j, i] = val.conjugate()
    return gram


def _spatial_fermion_string_expectation_mps(state, op_specs, norm):
    """Contract a spatial-site fermion string directly against a dense MPS."""

    from pyqed.mps.mps import expect_mps

    local = SpinHalfFermionOperators()
    eye = np.eye(4, dtype=complex)
    jw = np.asarray(local["JW"], dtype=complex)
    local_by_spec = {
        ("ann", 0): np.asarray(local["Cu"], dtype=complex),
        ("ann", 1): np.asarray(local["Cd"], dtype=complex),
        ("cre", 0): np.asarray(local["Cdu"], dtype=complex),
        ("cre", 1): np.asarray(local["Cdd"], dtype=complex),
    }

    mpo = []
    for i in range(state.L):
        op_i = eye
        for kind, sigma, site in op_specs:
            if i < site:
                factor = jw
            elif i == site:
                factor = local_by_spec[(kind, sigma)]
            else:
                factor = eye
            op_i = op_i @ factor
        mpo.append(op_i.reshape(1, 1, 4, 4))
    return expect_mps(state.Bs, mpo, state.Bs) / norm


class _SpatialNPDMContractions:
    """Environment-cached contractions for spatial-site NPDM elements."""

    def __init__(self, state):
        self.state = state
        self.L = state.L
        self.Bs = [state._get_std_B(i) for i in range(self.L)]
        local = SpinHalfFermionOperators()
        self.local_ops = {
            "I": np.eye(4, dtype=complex),
            "JW": np.asarray(local["JW"], dtype=complex),
            ("ann", 0): np.asarray(local["Cu"], dtype=complex),
            ("ann", 1): np.asarray(local["Cd"], dtype=complex),
            ("cre", 0): np.asarray(local["Cdu"], dtype=complex),
            ("cre", 1): np.asarray(local["Cdd"], dtype=complex),
        }
        self._site_op_cache = {}
        self._transfer_matrix_cache = {}
        self.left_env, self.right_env = self._build_identity_envs()
        self.norm = state._mps_dot(state, state)
        self._use_transfer_matrices = all(
            max(B.shape[0] * B.shape[0], B.shape[2] * B.shape[2]) <= 512
            for B in self.Bs
        )

    @staticmethod
    def _transfer(E, B, op):
        return np.einsum("ab,asr,st,btu->ru", E, B.conj(), op, B, optimize=True)

    def _build_identity_envs(self):
        left_env = [np.array([[1.0 + 0.0j]])]
        for i in range(self.L - 1):
            left_env.append(self._transfer(left_env[-1], self.Bs[i], self.local_ops["I"]))

        right_env = [None] * self.L
        right_env[-1] = np.array([[1.0 + 0.0j]])
        for i in range(self.L - 1, 0, -1):
            B = self.Bs[i]
            R = right_env[i]
            right_env[i - 1] = np.einsum(
                "ru,asr,st,btu->ab",
                R,
                B.conj(),
                self.local_ops["I"],
                B,
                optimize=True,
            )
        return left_env, right_env

    def _site_op_key(self, op_specs, site):
        key = []
        for kind, sigma, op_site in op_specs:
            if site < op_site:
                key.append("JW")
            elif site == op_site:
                key.append((kind, sigma))
        return tuple(key)

    def _site_op(self, key):
        if not key:
            return self.local_ops["I"]
        op = self._site_op_cache.get(key)
        if op is None:
            op = self.local_ops["I"]
            for part in key:
                op = op @ self.local_ops[part]
            self._site_op_cache[key] = op
        return op

    def _site_transfer_matrix(self, site, key):
        cache_key = (site, key)
        mat = self._transfer_matrix_cache.get(cache_key)
        if mat is None:
            B = self.Bs[site]
            op = self._site_op(key)
            mat = np.einsum("asr,st,btu->ruab", B.conj(), op, B, optimize=True)
            mat = mat.reshape(B.shape[2] * B.shape[2], B.shape[0] * B.shape[0])
            self._transfer_matrix_cache[cache_key] = mat
        return mat

    def _close_with_right(self, E, site):
        return np.sum(E * self.right_env[site])

    def expect_string(self, op_specs):
        sites = [site for _, _, site in op_specs]
        first = min(sites)
        last = max(sites)
        if self._use_transfer_matrices:
            vec = self.left_env[first].reshape(-1)
            for site in range(first, last + 1):
                key = self._site_op_key(op_specs, site)
                vec = self._site_transfer_matrix(site, key) @ vec
            return np.dot(vec, self.right_env[last].reshape(-1)) / self.norm

        E = self.left_env[first]
        for site in range(first, last + 1):
            key = self._site_op_key(op_specs, site)
            E = self._transfer(E, self.Bs[site], self._site_op(key))
        return self._close_with_right(E, last) / self.norm


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


def _build_spatial_hamiltonian_tensor_mpo(
    h1e,
    eri,
    *,
    spin_purification=False,
    shift=None,
    cutoff=1e-10,
    symbolic_algo="qr",
):
    """Build the spatial-orbital Hamiltonian directly as a d=4 symbolic MPO."""
    h_spatial = np.asarray(h1e[0])
    eri_spatial = np.asarray(eri[0, 0])
    ncas = h_spatial.shape[0]
    term_map = _merge_spatial_term_maps(
        _spatial_one_body_term_map(h_spatial, cutoff=cutoff),
        _spatial_two_body_spinfree_term_map(eri_spatial, cutoff=cutoff),
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
        algo=symbolic_algo,
    )
    return tensor_mpo, int(term_count), int(spin_term_count)


def _dense_cores_from_nonabelian_mpo(mpo):
    return [
        np.asarray(core.as_dense() if hasattr(core, "as_dense") else core, dtype=complex)
        for core in mpo
    ]


def _build_spatial_native_generator_family_mpos(
    complementary,
    n_sites,
    *,
    cutoff=1e-10,
    p_grouping="first_site_order",
):
    """Build R/P family MPOs from spin-free generator families."""
    from pyqed.mps.nonabelian import AutoMPO, physical_leg_from_spatial_orbital
    from pyqed.mps.nonabelian.models import (
        add_spatial_one_body_terms,
        add_spatial_two_generator_product_terms,
    )

    leg = physical_leg_from_spatial_orbital()
    site_legs = [leg] * int(n_sites)
    family_tensor_mpos = {}
    family_mpo_info = {}

    r_family = complementary.get("R")
    r_entries = dict(getattr(r_family, "entries", {}) or {})
    if r_entries:
        r_matrix = np.zeros((int(n_sites), int(n_sites)), dtype=complex)
        for (p, q), coeff in r_entries.items():
            if abs(coeff) > cutoff:
                r_matrix[int(p), int(q)] += complex(coeff)
        if np.any(np.abs(r_matrix) > cutoff):
            builder = AutoMPO(site_legs)
            add_spatial_one_body_terms(
                builder,
                r_matrix,
                cutoff=cutoff,
                family="R",
            )
            mpo = builder.build()
            if mpo:
                cores = _dense_cores_from_nonabelian_mpo(mpo)
                family_tensor_mpos["R"] = cores
                bond_orders = [int(core.shape[1]) for core in cores]
                family_mpo_info["R"] = {
                    "source": "native_spinfree_generator_autompo",
                    "generator_terms": int(len(r_entries)),
                    "mpo_max_bond": int(max(bond_orders)),
                    "bond_orders": tuple(bond_orders),
                }

    p_family = complementary.get("P")
    p_entries = dict(getattr(p_family, "entries", {}) or {})
    if p_entries:
        p_grouping = _normalize_spatial_native_p_grouping(p_grouping)
        p_four = {}
        p_repeated = {}
        for key, coeff in p_entries.items():
            bucket = p_four if len({int(index) for index in key}) == 4 else p_repeated
            bucket[tuple(int(index) for index in key)] = complex(coeff)

        def _four_distinct_site_order(key):
            original = (
                (int(key[0]), 0),
                (int(key[1]), 1),
                (int(key[2]), 2),
                (int(key[3]), 3),
            )
            return tuple(
                original_index
                for _site, original_index in sorted(original, key=lambda item: item[0])
            )

        def _site_order_group(order):
            if p_grouping == "none":
                return ()
            if p_grouping == "first_site_order":
                return (int(order[0]),)
            if p_grouping == "first_two_site_order":
                return (int(order[0]), int(order[1]))
            return tuple(int(index) for index in order)

        def _site_order_group_label(group):
            return "all" if not group else "".join(str(index) for index in group)

        p_summary = {
            "source": "native_spinfree_two_generator_split",
            "generator_terms": int(len(p_entries)),
            "split_family_names": (),
        }
        if p_four:
            p_four_by_order_group = {}
            for key, coeff in p_four.items():
                order = _four_distinct_site_order(key)
                group = _site_order_group(order)
                p_four_by_order_group.setdefault(group, {})[key] = coeff
            p_summary["four_distinct_site_order_grouping"] = p_grouping
            p_summary["four_distinct_site_order_groups"] = int(
                len(p_four_by_order_group)
            )
            for middle_irrep in (0, 2):
                for group, entries in sorted(
                    p_four_by_order_group.items(),
                    key=lambda item: repr(item[0]),
                ):
                    group_label = _site_order_group_label(group)
                    name = f"P:we{middle_irrep}:g{group_label}"
                    builder = AutoMPO(site_legs)
                    p_info = add_spatial_two_generator_product_terms(
                        builder,
                        entries,
                        cutoff=cutoff,
                        family=name,
                        reduced_we=True,
                        we_middle_irreps=(middle_irrep,),
                        return_info=True,
                    )
                    mpo = builder.build()
                    if not mpo:
                        continue
                    cores = _dense_cores_from_nonabelian_mpo(mpo)
                    family_tensor_mpos[name] = cores
                    bond_orders = [int(core.shape[1]) for core in cores]
                    family_mpo_info[name] = {
                        "source": "native_spinfree_two_generator_we_site_order",
                        "middle_irrep": int(middle_irrep),
                        "site_order_group": tuple(int(index) for index in group),
                        "generator_terms": int(len(entries)),
                        "symbolic_product_terms": int(
                            p_info.get("symbolic_product_terms", 0)
                        ),
                        "we_product_terms": int(p_info.get("we_product_terms", 0)),
                        "total_product_terms": int(p_info.get("total_product_terms", 0)),
                        "raw_spin_component_terms": int(
                            p_info.get("raw_spin_component_terms", 0)
                        ),
                        "four_distinct_generator_terms": int(
                            p_info.get("four_distinct_generator_terms", 0)
                        ),
                        "repeated_generator_terms": int(
                            p_info.get("repeated_generator_terms", 0)
                        ),
                        "unique_index_histogram": dict(
                            p_info.get("unique_index_histogram", {})
                        ),
                        "mpo_max_bond": int(max(bond_orders)),
                        "bond_orders": tuple(bond_orders),
                    }
                    p_summary["split_family_names"] = (
                        tuple(p_summary["split_family_names"]) + (name,)
                    )

        if p_repeated:
            name = "P:repeated"
            repeated_terms = _spatial_two_generator_family_term_map(
                p_repeated,
                cutoff=cutoff,
            )
            repeated_mpo, repeated_count = _build_tensor_mpo_from_symbolic_terms(
                [BasisSpatialFermion(i) for i in range(int(n_sites))],
                repeated_terms,
                cutoff=cutoff,
                algo="Hopcroft-Karp",
            )
            family_tensor_mpos[name] = repeated_mpo.factors
            bond_orders = [int(x) for x in repeated_mpo.bond_orders()]
            unique_histogram = {}
            for key in p_repeated:
                unique = len({int(index) for index in key})
                unique_histogram[str(unique)] = (
                    int(unique_histogram.get(str(unique), 0)) + 1
                )
            family_mpo_info[name] = {
                "source": "native_repeated_generator_local_collapse",
                "generator_terms": int(len(p_repeated)),
                "symbolic_terms": int(repeated_count),
                "four_distinct_generator_terms": 0,
                "repeated_generator_terms": int(len(p_repeated)),
                "unique_index_histogram": dict(sorted(unique_histogram.items())),
                "mpo_max_bond": int(max(bond_orders)),
                "bond_orders": tuple(bond_orders),
            }
            p_summary["split_family_names"] = (
                tuple(p_summary["split_family_names"]) + (name,)
            )
        if p_summary["split_family_names"]:
            p_summary["split_family_names"] = tuple(p_summary["split_family_names"])
            p_summary["mpo_max_bond"] = int(
                max(
                    family_mpo_info[name]["mpo_max_bond"]
                    for name in p_summary["split_family_names"]
                )
            )
            family_mpo_info["P"] = p_summary

    return family_tensor_mpos, family_mpo_info


def _compare_spatial_family_term_map(reference, family, *, cutoff=1e-10):
    """Return compact diagnostics comparing canonical and family term maps."""
    keys = set(reference) | set(family)
    max_abs = 0.0
    l2 = 0.0
    mismatches = 0
    for key in keys:
        diff = complex(family.get(key, 0.0)) - complex(reference.get(key, 0.0))
        adiff = abs(diff)
        if adiff > cutoff:
            mismatches += 1
        max_abs = max(max_abs, float(adiff))
        l2 += float(adiff) ** 2
    return {
        "enabled": True,
        "reference_terms": int(len(reference)),
        "family_terms": int(len(family)),
        "mismatched_terms": int(mismatches),
        "max_abs_diff": float(max_abs),
        "l2_diff": float(np.sqrt(l2)),
        "tol": float(cutoff),
        "ok": bool(mismatches == 0),
    }


def _spatial_family_generator_entry_counts(complementary):
    """Return raw R/P generator-entry counts without building symbolic JW maps."""
    if complementary is None:
        return {}
    counts = {}
    for name in ("R", "P"):
        family = complementary.get(name) if hasattr(complementary, "get") else None
        entries = getattr(family, "entries", None) if family is not None else None
        counts[name] = int(len(entries or {}))
    return counts


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


def _build_tensor_mpo_from_symbolic_terms(basis_sites, term_map, *, cutoff=1e-14, algo="qr"):
    """Build a dense MPO from symbolic terms and wrap it in the high-level MPO class."""
    terms = _materialize_symbolic_terms(term_map, tol=cutoff)
    model = Model(basis=basis_sites, ham_terms=terms)
    mpo = Mpo(model, algo=algo)
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
        "pg": ["charge", "sz", "pg"],
        "pointgroup": ["charge", "sz", "pg"],
        "point_group": ["charge", "sz", "pg"],
        "abelianpg": ["charge", "sz", "pg"],
        "abelian_pg": ["charge", "sz", "pg"],
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
        if key in {"charge_sz_pg", "charge,sz,pg", "sz_pg"}:
            return ["charge", "sz", "pg"]
        raise ValueError(
            "Unknown DMRG symmetry {!r}. Use None, 'charge', 'sz', 'u1', 'su2', or 'pg'.".format(spec)
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
        if label in {"point_group", "abelianpg", "abelian_pg", "irrep", "orb_sym"}:
            label = "pg"
        if label not in {"charge", "sz", "su2", "pg"}:
            raise ValueError(
                "Unknown DMRG symmetry label {!r}. Use 'charge', 'sz', 'su2', or 'pg'.".format(label)
            )
        if label not in normalized:
            normalized.append(label)
    if "su2" in normalized and "charge" not in normalized:
        normalized.insert(0, "charge")
    if "sz" in normalized and "charge" not in normalized:
        normalized.insert(0, "charge")
    if "pg" in normalized and "charge" not in normalized:
        normalized.insert(0, "charge")
    if "pg" in normalized and "sz" not in normalized and "su2" not in normalized:
        normalized.insert(1, "sz")
    if "su2" in normalized and "sz" in normalized:
        raise ValueError("DMRG symmetry cannot combine 'su2' and 'sz'.")
    if "su2" in normalized and "pg" in normalized:
        raise NotImplementedError("DMRG SU(2)+AbelianPG is not wired yet; use Abelian symmetry='pg'.")
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

def build_mps_from_configs(
    configs_with_amps,
    sym_mgr,
    nsites,
    noise_scale=1e-5,
    *,
    native_site_storage=False,
):
    """
    Constructs an entangled U(1) symmetric MPS from a list of determinant configurations.

    Args:
        configs_with_amps: List of tuples (occupation_list, amplitude).
        sym_mgr: SymmetryManager instance.
        nsites: Total number of sites.
        noise_scale: Magnitude of random noise to inject for symmetry breaking.

    Returns:
        list: The resulting symmetric MPS tensors in (Left, Right, Phys) convention.
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
        # Construct flat lists of QNs for the symmetric site-tensor axes.
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
        bt = make_abelian_site_tensor(
            data,
            [final_qns_L, final_qns_R, final_qns_P],
            [-1, 1, 1],
            native_site_storage=native_site_storage,
        )
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


def build_spatial_mps_from_configs(
    configs_with_amps,
    sym_mgr,
    nsites,
    noise_scale=1e-5,
    *,
    native_site_storage=False,
):
    """Construct a spatial-site U(1) MPS from d=4 local configurations."""
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
        bt = make_abelian_site_tensor(
            data,
            [final_qns_L, final_qns_R, phys_qns],
            [-1, 1, 1],
            native_site_storage=native_site_storage,
        )
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
    def __init__(self, mf, ncas, nelecas, D=None, init_guess='hf', m_warmup=None,\
                 spin=None, tol=1e-6, low_rank_mpo=False, low_rank_mpo_bond=None,
                 low_rank_mpo_batch_size=4, verbose=0, site='spin_orbital',\
                 orbital_layout=None, spatial_reduced_mpo=None,
                 symmetry=None, spatial_site_basis="canonical",
                 integral_backend="auto", spatial_abelian_mpo="auto",
                 spatial_abelian_symbolic_algo="Hopcroft-Karp",
                 spatial_family_environment_backend="generator_table",
                 spatial_native_p_grouping="first_site_order",
                 spatial_block2_table_p_split_metric="auto",
                 spatial_block2_table_p_split_groups="auto",
                 spatial_block2_table_native_p=False,
                 spatial_complementary_payload_tensor_matvec=True,
                 spatial_precontracted_family_environment=True,
                 spatial_boundary_table_max_dim=32,
                 spatial_exact_component_compression_policy="auto",
                 spatial_exact_component_compression_validate=True,
                 spatial_exact_component_compression_validation_vectors=1,
                 spatial_exact_component_compression_min_reduction=1,
                 spatial_exact_component_compression_max_group_size=64,
                 spatial_enable_cpp_boundary_r=False,
                 spatial_validate_cpp_boundary_r=True,
                 spatial_enable_cpp_boundary_p=True,
                 spatial_validate_cpp_boundary_p=False,
                 spatial_cpp_boundary_p_validation_policy="off",
                 spatial_direct_operator_batch_min_entries=2,
                 dmrg_performance="block2-like",
                 abelian_matvec_options=None,
                 debug_complementary_action_check=False,
                 debug_complementary_action_check_tol=1.0e-10,
                 debug_complementary_action_check_limit=32,
                 debug_spatial_family_hamiltonian_check=False,
                 orb_sym=None):
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


        self.D = self.m = None if D is None else int(D)

        self.tol = tol # tolerance for energy convergence
        self.rigid_shift = 0

        if m_warmup is None:
            m_warmup = self.D
        self.m_warmup = m_warmup


        self.ncas = ncas # number of MOs in active space
        self.nelecas = int(sum(nelecas)) if isinstance(nelecas, (tuple, list)) else int(nelecas)

        self.nelec = mf.nelec
        nelec_total = int(sum(mf.nelec)) if isinstance(mf.nelec, (tuple, list)) else int(mf.nelec)

        ncore = nelec_total//2 - self.nelecas//2 # core orbs
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
        self.energy = None
        self.energies = None
        self.state_average_energy = None
        self.ground_state = None
        self.states = []
        self.history = []
        self.dmrg = None
        self.converged = False
        self.success = False
        self.message = "not run"
        self.ncompleted = 0
        self.ncompleted_half_sweeps = 0
        self.max_sweeps = 0
        self.ci = None # CI coefficients
        self.H = None
        self.H_raw = None
        self._hamiltonian_mpo_cache_key = None
        self._symmetric_mpo_cache = {}
        self._s2_mpo_cache = {}
        self._spatial_operator_cache = None
        self.spatial_rdm2_algorithm = "npdm"
        self._active_hamiltonian = None
        self.complementary_operators = None
        self.complementary_operator_mpos = None
        self.complementary_operator_term_maps = None
        self.complementary_operator_generator_entries = None
        self._active_integral_build_info = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space

        self.spin_purification = False

        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None
        self.h2e_factors = None

        self.init_guess = init_guess
        self.integral_backend = _normalize_integral_backend(integral_backend)
        self.orb_sym = None if orb_sym is None else tuple(int(x) for x in orb_sym)
        self.low_rank_mpo = bool(low_rank_mpo)
        self.low_rank_mpo_bond = low_rank_mpo_bond
        self.low_rank_mpo_batch_size = int(low_rank_mpo_batch_size)
        p_split_metric = str(spatial_block2_table_p_split_metric or "auto")
        p_split_metric = p_split_metric.lower().replace("-", "_")
        metric_aliases = {
            "rightmost": "max_site",
            "max": "max_site",
            "leftmost": "min_site",
            "min": "min_site",
            "first": "first_site",
            "first_dof": "first_site",
            "support": "span",
            "support_span": "span",
            "extent": "span",
        }
        p_split_metric = metric_aliases.get(p_split_metric, p_split_metric)
        if p_split_metric not in {
            "auto",
            "first_site",
            "min_site",
            "max_site",
            "center",
            "span",
        }:
            p_split_metric = "auto"
        self.spatial_block2_table_p_split_metric = p_split_metric
        if str(spatial_block2_table_p_split_groups).lower() == "auto":
            p_split_groups = "auto"
        else:
            try:
                p_split_groups = max(1, int(spatial_block2_table_p_split_groups))
            except Exception:
                p_split_groups = "auto"
        self.spatial_block2_table_p_split_groups = p_split_groups
        self.spatial_block2_table_native_p = bool(spatial_block2_table_native_p)
        self.spatial_complementary_payload_tensor_matvec = bool(
            spatial_complementary_payload_tensor_matvec
        )
        self.spatial_precontracted_family_environment = bool(
            spatial_precontracted_family_environment
        )
        self.spatial_boundary_table_max_dim = int(spatial_boundary_table_max_dim)
        self.spatial_exact_component_compression_policy = str(
            spatial_exact_component_compression_policy
        ).lower().replace("-", "_")
        self.spatial_exact_component_compression_validate = bool(
            spatial_exact_component_compression_validate
        )
        self.spatial_exact_component_compression_validation_vectors = int(
            spatial_exact_component_compression_validation_vectors
        )
        self.spatial_exact_component_compression_min_reduction = int(
            spatial_exact_component_compression_min_reduction
        )
        self.spatial_exact_component_compression_max_group_size = int(
            spatial_exact_component_compression_max_group_size
        )
        self.spatial_enable_cpp_boundary_r = bool(
            spatial_enable_cpp_boundary_r
        )
        self.spatial_validate_cpp_boundary_r = bool(
            spatial_validate_cpp_boundary_r
        )
        self.spatial_enable_cpp_boundary_p = bool(
            spatial_enable_cpp_boundary_p
        )
        self.spatial_validate_cpp_boundary_p = bool(
            spatial_validate_cpp_boundary_p
        )
        policy = str(spatial_cpp_boundary_p_validation_policy or "off")
        policy = policy.lower().replace("-", "_")
        if policy in {"true", "yes", "on"}:
            policy = "first_pass"
        if policy in {"false", "no", "off", "none", "disabled"}:
            policy = "off"
        if policy not in {"off", "first_pass", "always"}:
            policy = "off"
        self.spatial_cpp_boundary_p_validation_policy = policy
        self.spatial_direct_operator_batch_min_entries = max(
            2,
            int(spatial_direct_operator_batch_min_entries),
        )
        self.dmrg_performance = str(dmrg_performance or "auto")
        self.abelian_matvec_options = (
            None if abelian_matvec_options is None else dict(abelian_matvec_options)
        )
        self.debug_complementary_action_check = bool(debug_complementary_action_check)
        self.debug_complementary_action_check_tol = float(
            debug_complementary_action_check_tol
        )
        self.debug_complementary_action_check_limit = int(
            debug_complementary_action_check_limit
        )
        self.debug_spatial_family_hamiltonian_check = bool(
            debug_spatial_family_hamiltonian_check
        )
        self.spatial_family_environment_backend = (
            _normalize_spatial_family_environment_backend(
                spatial_family_environment_backend
            )
        )
        spatial_abelian_mpo = str(spatial_abelian_mpo).lower().replace("-", "_")
        if spatial_abelian_mpo in {"block2", "block2_spatial", "native_spatial"}:
            spatial_abelian_mpo = "spatial"
        if spatial_abelian_mpo in {"auto", "default"}:
            performance_key = str(self.dmrg_performance or "auto").lower()
            performance_key = performance_key.replace("_", "-")
            block2_like_performance = performance_key in {
                "auto",
                "block2",
                "block2-like",
                "block2-style",
                "packed-block2-style",
                "cpp",
                "c++",
                "block2-cpp",
                "packed-cpp-fast",
                "packed-compiled-fast",
            }
            can_use_block2_carrier = bool(
                self.site == "spatial"
                and normalized_symmetry is not None
                and "charge" in normalized_symmetry
                and "su2" not in normalized_symmetry
                and self.spatial_family_environment_backend
                in {"block2_table", "generator_table"}
                and block2_like_performance
            )
            spatial_abelian_mpo = "spatial" if can_use_block2_carrier else "grouped"
        if spatial_abelian_mpo not in {"spatial", "direct", "grouped"}:
            raise ValueError(
                "spatial_abelian_mpo must be 'auto', 'spatial', 'direct', or 'grouped'."
            )
        self.spatial_abelian_mpo = spatial_abelian_mpo
        self.spatial_abelian_symbolic_algo = _normalize_spatial_abelian_symbolic_algo(
            spatial_abelian_symbolic_algo
        )
        self.spatial_native_p_grouping = _normalize_spatial_native_p_grouping(
            spatial_native_p_grouping
        )
        if spatial_reduced_mpo is None:
            spatial_reduced_mpo = normalized_symmetry is not None and "su2" in normalized_symmetry
        self.spatial_reduced_mpo = bool(spatial_reduced_mpo)
        self.symmetry = normalized_symmetry
        self.saved_symmetry_list = self.symmetry

    def _log(self, message, level=1):
        if self.verbose >= level:
            print(message)

    def _invalidate_hamiltonian_mpo(self):
        """Drop representation-dependent Hamiltonian state before rebuilding."""
        self.H = None
        self.H_raw = None
        self._hamiltonian_mpo_cache_key = None
        self._symmetric_mpo_cache = {}
        self._s2_mpo_cache = {}
        self._spatial_operator_cache = None
        self._active_hamiltonian = None
        self.complementary_operators = None
        self.complementary_operator_mpos = None
        self.complementary_operator_term_maps = None
        self.complementary_operator_generator_entries = None

    def _require_su2_cpp_integral_reference(self):
        mol = getattr(self.mf, "mol", None)
        driver = str(getattr(mol, "_build_driver", "") or "").lower()
        info = getattr(mol, "_builtin_build_info", None)
        info = info if isinstance(info, dict) else {}
        eri_backend = str(info.get("eri_backend", "") or "").lower()
        dense_builder = str(info.get("dense_builder", "") or "").lower()
        factor_builder = str(info.get("factor_builder", "") or "").lower()
        ri_info = info.get("ri")
        ri_info = ri_info if isinstance(ri_info, dict) else {}
        ri_builder = " ".join(
            str(ri_info.get(key, "") or "").lower()
            for key in (
                "tensor_backend",
                "three_center_builder",
                "metric_builder",
            )
        )
        compiled_builder = bool(
            eri_backend == "cpp"
            or "cpp" in dense_builder
            or "cpp" in factor_builder
            or "cpp" in ri_builder
            or "native" in ri_builder
        )
        if driver != "builtin" or not compiled_builder:
            raise RuntimeError(
                "SU(2) DMRG requires Molecule.build(driver='builtin', ...) "
                "with PyQED's compiled C++ integral builder; GBasis, PySCF, "
                "and uncompiled integral references are not supported."
            )

    def export_ground_state(self, state=0, dense=False):
        """Return a reusable copy of a converged DMRG state."""
        if not hasattr(self, 'dmrg') or self.dmrg is None or self.dmrg.states is None:
            raise ValueError("No converged DMRG state available. Run DMRG first.")
        guess = self.dmrg.states[state].copy()
        if dense and hasattr(guess.factors[0], 'qns'):
            from pyqed.mps.mps import symmetric_to_dense
            guess = symmetric_to_dense(guess, site_qn_maps=self._dense_site_qn_maps())
        return guess.copy()

    def _dense_site_qn_maps(self):
        """Physical-state maps needed to recover dense local basis ordering."""
        dmrg = getattr(self, "dmrg", None)
        if dmrg is not None:
            site_qn_maps = getattr(dmrg, "site_qn_maps", None)
            if site_qn_maps is not None:
                return site_qn_maps
        return getattr(self, "_site_qn_maps", None)

    def reuse_guess_from(self, other, state=0, dense=False):
        """Adopt a converged MPS from another DMRG object as the next guess."""
        self.init_guess = other.export_ground_state(state=state, dense=dense)
        return self


    def get_initial_guess_symmetric(self, method='cid', *, native_site_storage=False):
        """
        New Robust Initial Guess Dispatcher.
        """
        method = method.lower()
        if self.site == "spatial":
            nspin = 2 * self.ncas
            if method == 'hf':
                configs = [
                    (_spin_config_to_spatial_config(cfg), amp)
                    for cfg, amp in gen_cid_configs(self.nelecas, nspin, mixing=1.0e-5)
                ]
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
            return build_spatial_mps_from_configs(
                configs,
                self.sym_mgr,
                self.ncas,
                native_site_storage=native_site_storage,
            )

        nsites = 2 * self.ncas

        # Ensure Manager exists (created in run())
        if not hasattr(self, 'sym_mgr'):
            self.sym_mgr = SymmetryManager(['charge', 'sz'], orb_sym=getattr(self, "orb_sym", None)) # Default fallback

        self._log(f"  [InitGuess] Generating guess: '{method}' with {self.sym_mgr.sym_types}")

        # 1. Generate Configurations (Physics)
        if method == 'hf':
            configs = gen_cid_configs(self.nelecas, nsites, mixing=1.0e-5)

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
        mps = build_mps_from_configs(
            configs,
            self.sym_mgr,
            nsites,
            native_site_storage=native_site_storage,
        )
        return mps

    def get_initial_guess_dense(self, noise=1e-3):
        if self.site == "spatial":
            return _spatial_hf_guess(self.nelecas, self.ncas, spin=self.spin, noise=noise)
        return get_noisy_hf_guess(self.nelecas, 2*self.ncas, noise=noise)

    def _resolve_initial_guess(self, use_symmetry, *, native_site_storage=False):
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
            return self.get_initial_guess_symmetric(
                method=guess.lower(),
                native_site_storage=native_site_storage,
            )

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
        backend = self.integral_backend
        use_cholesky = backend in {"ri", "cholesky"} or (
            backend == "auto" and not _mf_has_dense_eris(mf) and _mf_has_factorized_eris(mf)
        )

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
            factor_source = getattr(
                getattr(mf, "mol", None),
                "builtin_resolved_eri_representation",
                getattr(getattr(mf, "mol", None), "native_resolved_eri_representation", None),
            )
            if factor_source is None:
                factor_source = "cholesky" if getattr(mf, "cholesky_jk", False) else "ri"
            if str(factor_source).lower() in {"dense+ri"}:
                factor_source = "ri"
            elif str(factor_source).lower() in {"dense", "dense+factors", "factors"}:
                factor_source = "cholesky"
            build_mode = str(factor_source).lower()
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
            'factorized_integrals': bool(use_cholesky),
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
        dense active ERI tensor or RI/Cholesky pair factors.
        """
        mf = self.mf
        active_integrals = getattr(mf, "active_space_integrals", None)
        if active_integrals is not None:
            h1e, eri, pair_factors, energy_core, info = active_integrals(
                mo_coeff=self.mo_coeff,
                mo_core=self.mo_core,
                mo_cas=self.mo_cas,
                ncore=self.ncore,
                ncas=self.ncas,
                nelecas=self.nelecas,
            )
            self.e_core = float(energy_core)
            self._active_integral_build_info = dict(info)
            return h1e, eri, pair_factors

        backend = self.integral_backend
        use_factors = backend in {"ri", "cholesky"} or (
            backend == "auto" and not _mf_has_dense_eris(mf) and _mf_has_factorized_eris(mf)
        )

        H, energy_core = h1e_for_cas(
            mf,
            ncas=self.ncas,
            ncore=self.ncore,
            mo_coeff=self.mo_coeff,
        )
        self.e_core = energy_core

        if use_factors:
            pair_factors = transform_eri_factors_to_mo_pair(
                _get_mf_cholesky_factors(mf),
                self.mo_cas,
            )
            factor_source = getattr(
                getattr(mf, "mol", None),
                "builtin_resolved_eri_representation",
                getattr(getattr(mf, "mol", None), "native_resolved_eri_representation", None),
            )
            if factor_source is None:
                factor_source = "cholesky" if getattr(mf, "cholesky_jk", False) else "ri"
            if str(factor_source).lower() in {"dense+ri"}:
                factor_source = "ri"
            elif str(factor_source).lower() in {"dense", "dense+factors", "factors"}:
                factor_source = "cholesky"
            self._active_integral_build_info = {
                "mode": str(factor_source).lower(),
                "factorized_integrals": True,
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
            "factorized_integrals": False,
            "aux_rank": None,
            "ncas": self.ncas,
        }
        return [H, H], H2, None

    def _can_use_spatial_block2_carrier(self):
        sym = _normalize_dmrg_symmetry(getattr(self, "symmetry", None))
        return bool(sym is not None and "charge" in sym and "su2" not in sym)

    def _build_spatial_complementary_family_data(self, h1e, eri, *, cutoff=1e-10):
        """Build the shared Abelian spatial complementary R/P family package."""

        from pyqed.qchem.dmrg.backends.reduced import (
            ComplementaryOperatorFamily,
            SpatialComplementaryOperatorFamilies,
            build_spatial_complementary_operator_families,
        )

        h1_spatial = np.asarray(h1e[0])
        eri_spatial = np.asarray(eri[0, 0])
        timings = {}
        native_fallback_reason = None
        t0 = time.perf_counter()
        complementary = None
        native_entries = None
        native_n_sites = None
        native_terms = None
        native_carrier = None
        native_family_mpos = None
        native_family_mpo_info = None
        native_family_mpo_owner = None
        native_family_descriptor_key = None
        native_family_descriptor_names = ()
        native_family_descriptor_info = None
        self._native_spatial_family_descriptor_key = None
        self._native_spatial_family_descriptor_names = ()

        def _record_native_family_mpo_payload(native_family_mpo_payload):
            nonlocal native_family_mpos, native_family_mpo_info
            native_family_mpos = {
                str(name): list(factors)
                for name, factors in dict(
                    native_family_mpo_payload.get("family_mpos", {})
                ).items()
            }
            native_family_mpo_info = {
                str(name): dict(info)
                for name, info in dict(
                    native_family_mpo_payload.get("family_mpo_info", {})
                ).items()
            }
            timings["family_mpo_backend_actual"] = str(
                native_family_mpo_payload.get(
                    "backend_actual",
                    "cpp_spatial_string_sum_family_mpos",
                )
            )
            timings["family_mpo_cpp_seconds"] = float(
                native_family_mpo_payload.get("seconds", 0.0)
            )
            timings["family_mpo_cpp_counts"] = dict(
                native_family_mpo_payload.get("counts", {})
            )
        needs_symbolic_family_terms = bool(
            self.debug_spatial_family_hamiltonian_check
            or self.spatial_family_environment_backend
            in {"block2", "block2_table", "block2_adaptive", "block2_native", "direct_terms"}
        )
        needs_native_carrier = bool(
            self.spatial_abelian_mpo == "spatial"
            and not self.spin_purification
            and self._can_use_spatial_block2_carrier()
        )
        needs_native_family_mpos = bool(
            needs_native_carrier
            and self.spatial_family_environment_backend == "block2_table"
        )
        try:
            from pyqed.mps import cpp_davidson as _cpp_davidson

            native_setup_builder = getattr(
                _cpp_davidson,
                "build_spatial_qchem_block2_setup",
                None,
            )
            if native_setup_builder is not None:
                owner_cls = getattr(_cpp_davidson, "MovingEnvironment", None)
                build_family_mpos_in_setup = bool(needs_native_family_mpos)
                if needs_native_family_mpos and owner_cls is not None:
                    build_family_mpos_in_setup = False
                native_setup = native_setup_builder(
                    h1_spatial,
                    eri,
                    cutoff,
                    True,
                    needs_symbolic_family_terms,
                    needs_native_carrier,
                    4,
                    build_family_mpos_in_setup,
                    self.spatial_block2_table_p_split_groups,
                    self.spatial_block2_table_p_split_metric,
                )
                native_result = native_setup["family_entries"]
                native_n_sites = int(
                    native_setup.get(
                        "n_sites",
                        native_result.get("n_sites", self.ncas),
                    )
                )
                native_terms = native_setup.get("family_term_maps")
                native_carrier = native_setup.get("carrier")
                native_family_mpo_payload = native_setup.get("family_mpos")
                if (
                    native_family_mpo_payload is None
                    and needs_native_family_mpos
                    and native_terms is not None
                    and owner_cls is not None
                ):
                    try:
                        native_family_mpo_owner = getattr(
                            self,
                            "_native_spatial_family_mpo_owner",
                            None,
                        )
                        owner_builder = getattr(
                            native_family_mpo_owner,
                            "build_spatial_qchem_family_mpos",
                            None,
                        )
                        if owner_builder is None:
                            native_family_mpo_owner = owner_cls()
                            self._native_spatial_family_mpo_owner = (
                                native_family_mpo_owner
                            )
                            owner_builder = (
                                native_family_mpo_owner
                                .build_spatial_qchem_family_mpos
                            )
                        t_owner_mpo = time.perf_counter()
                        descriptor_installer = getattr(
                            native_family_mpo_owner,
                            "install_spatial_qchem_family_descriptor",
                            None,
                        )
                        descriptor_builder = getattr(
                            native_family_mpo_owner,
                            "build_spatial_qchem_family_mpos_from_descriptor",
                            None,
                        )
                        if (
                            descriptor_installer is not None
                            and descriptor_builder is not None
                        ):
                            descriptor_layout = (
                                self.spatial_block2_table_p_split_groups,
                                self.spatial_block2_table_p_split_metric,
                            )
                            descriptor_key = (
                                "qchem-family-descriptor:"
                                f"{id(self)}:{native_n_sites}:"
                                f"{hash(descriptor_layout)}"
                            )
                            t_descriptor = time.perf_counter()
                            native_family_descriptor_info = descriptor_installer(
                                descriptor_key,
                                native_terms,
                                native_n_sites,
                                cutoff,
                                self.spatial_block2_table_p_split_groups,
                                self.spatial_block2_table_p_split_metric,
                            )
                            native_family_descriptor_key = descriptor_key
                            native_family_descriptor_names = tuple(
                                str(name)
                                for name in native_family_descriptor_info.get(
                                    "family_names",
                                    (),
                                )
                            )
                            self._native_spatial_family_descriptor_key = (
                                native_family_descriptor_key
                            )
                            self._native_spatial_family_descriptor_names = (
                                native_family_descriptor_names
                            )
                            timings["family_descriptor_backend_actual"] = str(
                                native_family_descriptor_info.get(
                                    "backend_actual",
                                    "cpp_spatial_qchem_family_descriptor",
                                )
                            )
                            timings["family_descriptor_install_s"] = float(
                                time.perf_counter() - t_descriptor
                            )
                            timings["family_descriptor_families"] = int(
                                len(native_family_descriptor_names)
                            )
                            native_family_mpo_payload = descriptor_builder(
                                descriptor_key
                            )
                            timings["family_mpo_owner_builder"] = (
                                "cpp_descriptor"
                            )
                        else:
                            native_family_mpo_payload = owner_builder(
                                native_terms,
                                native_n_sites,
                                cutoff,
                                self.spatial_block2_table_p_split_groups,
                                self.spatial_block2_table_p_split_metric,
                            )
                            timings["family_mpo_owner_builder"] = (
                                "cpp_term_maps"
                            )
                        owner_key = (
                            native_family_mpo_owner
                            .spatial_route_plan_cache_owner_key()
                        )
                        owner_stats = dict(
                            native_family_mpo_owner
                            .spatial_route_plan_cache_stats()
                        )
                        timings["family_mpo_owner_backend_actual"] = (
                            "cpp_moving_environment"
                        )
                        timings["family_mpo_owner_build_s"] = float(
                            time.perf_counter() - t_owner_mpo
                        )
                        timings["family_mpo_route_cache_owner"] = str(owner_key)
                        timings["family_mpo_route_cache_records"] = int(
                            owner_stats.get("records", 0)
                        )
                    except Exception as exc:
                        timings["family_mpo_owner_fallback_reason"] = repr(exc)
                        standalone_family_mpo_builder = getattr(
                            _cpp_davidson,
                            "build_spatial_qchem_family_mpos",
                            None,
                        )
                        if standalone_family_mpo_builder is not None:
                            try:
                                native_family_mpo_payload = (
                                    standalone_family_mpo_builder(
                                        native_terms,
                                        native_n_sites,
                                        cutoff,
                                        self.spatial_block2_table_p_split_groups,
                                        self.spatial_block2_table_p_split_metric,
                                    )
                                )
                                timings["family_mpo_owner_backend_actual"] = (
                                    "standalone_fallback"
                                )
                            except Exception as fallback_exc:
                                timings["family_mpo_cpp_fallback_reason"] = repr(
                                    fallback_exc
                                )
                if native_family_mpo_payload is not None:
                    _record_native_family_mpo_payload(native_family_mpo_payload)
                elif native_setup.get("family_mpo_error") is not None:
                    timings["family_mpo_cpp_fallback_reason"] = str(
                        native_setup.get("family_mpo_error")
                    )
                timings["qchem_block2_setup_backend_actual"] = str(
                    native_setup.get(
                        "backend_actual",
                        "cpp_qchem_spatial_block2_setup",
                    )
                )
                timings["qchem_block2_setup_s"] = float(
                    native_setup.get("seconds", 0.0)
                )
            else:
                native_builder = getattr(
                    _cpp_davidson,
                    "build_spatial_qchem_family_entries",
                    None,
                )
                if native_builder is None:
                    native_fallback_reason = "cpp_qchem_family_builder_unavailable"
                    native_result = None
                else:
                    native_result = native_builder(h1_spatial, eri, cutoff, True)
            if native_result is not None:
                native_entries = native_result["entries"]
                native_n_sites = int(native_result["n_sites"])
                ranks = {"S": 1, "R": 2, "A": 2, "P": 4, "B": 2, "Q": 3}
                descriptions = {
                    "S": "single-orbital spinor source channels",
                    "R": "effective one-body complementary coefficients",
                    "A": "pair/scalar-generator structural channels",
                    "P": "two-generator ERI complementary coefficients",
                    "B": "particle-hole scalar-generator structural channels",
                    "Q": "delta-contracted one-body correction complementary coefficients",
                }
                families = {
                    name: ComplementaryOperatorFamily(
                        name=name,
                        rank=ranks[name],
                        entries=dict(native_entries[name]),
                        description=descriptions[name],
                    )
                    for name in ("S", "R", "A", "P", "B", "Q")
                }
                complementary = SpatialComplementaryOperatorFamilies(
                    families=families,
                    n_sites=int(native_n_sites),
                    cutoff=float(cutoff),
                    include_half=True,
                    prefer_complementary_payload_tensor_matvec=(
                        self.spatial_complementary_payload_tensor_matvec
                    ),
                    prefer_precontracted_family_environment=(
                        self.spatial_precontracted_family_environment
                    ),
                    boundary_table_max_dim=self.spatial_boundary_table_max_dim,
                    debug_complementary_action_check=(
                        self.debug_complementary_action_check
                    ),
                    debug_complementary_action_check_tol=(
                        self.debug_complementary_action_check_tol
                    ),
                    debug_complementary_action_check_limit=(
                        self.debug_complementary_action_check_limit
                    ),
                    exact_component_compression_policy=(
                        self.spatial_exact_component_compression_policy
                    ),
                    exact_component_compression_validate=(
                        self.spatial_exact_component_compression_validate
                    ),
                    exact_component_compression_validation_vectors=(
                        self.spatial_exact_component_compression_validation_vectors
                    ),
                    exact_component_compression_min_reduction=(
                        self.spatial_exact_component_compression_min_reduction
                    ),
                    exact_component_compression_max_group_size=(
                        self.spatial_exact_component_compression_max_group_size
                    ),
                    enable_cpp_boundary_r=self.spatial_enable_cpp_boundary_r,
                    validate_cpp_boundary_r=(
                        self.spatial_validate_cpp_boundary_r
                    ),
                    enable_cpp_boundary_p=self.spatial_enable_cpp_boundary_p,
                    validate_cpp_boundary_p=self.spatial_validate_cpp_boundary_p,
                    cpp_boundary_p_validation_policy=(
                        self.spatial_cpp_boundary_p_validation_policy
                    ),
                    direct_operator_batch_min_entries=(
                        self.spatial_direct_operator_batch_min_entries
                    ),
                )
                timings["native_qchem_family_compile_s"] = float(
                    native_result.get("seconds", 0.0)
                )
                timings["qchem_family_backend_actual"] = "cpp"
                timings["qchem_family_counts"] = dict(native_result["counts"])
        except Exception as exc:
            native_fallback_reason = repr(exc)
            complementary = None

        if complementary is None:
            complementary = build_spatial_complementary_operator_families(
                h1_spatial,
                eri,
                cutoff=cutoff,
                include_half=True,
                prefer_complementary_payload_tensor_matvec=(
                    self.spatial_complementary_payload_tensor_matvec
                ),
                prefer_precontracted_family_environment=(
                    self.spatial_precontracted_family_environment
                ),
                boundary_table_max_dim=self.spatial_boundary_table_max_dim,
                exact_component_compression_policy=(
                    self.spatial_exact_component_compression_policy
                ),
                exact_component_compression_validate=(
                    self.spatial_exact_component_compression_validate
                ),
                exact_component_compression_validation_vectors=(
                    self.spatial_exact_component_compression_validation_vectors
                ),
                exact_component_compression_min_reduction=(
                    self.spatial_exact_component_compression_min_reduction
                ),
                exact_component_compression_max_group_size=(
                    self.spatial_exact_component_compression_max_group_size
                ),
                enable_cpp_boundary_r=self.spatial_enable_cpp_boundary_r,
                validate_cpp_boundary_r=self.spatial_validate_cpp_boundary_r,
                enable_cpp_boundary_p=self.spatial_enable_cpp_boundary_p,
                validate_cpp_boundary_p=self.spatial_validate_cpp_boundary_p,
                cpp_boundary_p_validation_policy=(
                    self.spatial_cpp_boundary_p_validation_policy
                ),
                direct_operator_batch_min_entries=(
                    self.spatial_direct_operator_batch_min_entries
                ),
                debug_complementary_action_check=self.debug_complementary_action_check,
                debug_complementary_action_check_tol=(
                    self.debug_complementary_action_check_tol
                ),
                debug_complementary_action_check_limit=(
                    self.debug_complementary_action_check_limit
                ),
            )
            timings["qchem_family_backend_actual"] = "python"
            if native_fallback_reason is not None:
                timings["qchem_family_fallback_reason"] = native_fallback_reason
        timings["complementary_family_build_s"] = float(time.perf_counter() - t0)
        generator_entry_counts = _spatial_family_generator_entry_counts(complementary)
        if needs_symbolic_family_terms:
            t0 = time.perf_counter()
            family_term_maps = None
            if native_terms is not None:
                dynamic_ops = dict(native_terms.get("local_ops", {}) or {})
                if dynamic_ops:
                    from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

                    spatial_local_ops().update(
                        {
                            str(name): np.asarray(matrix, dtype=complex)
                            for name, matrix in dynamic_ops.items()
                        }
                    )
                family_term_maps = {
                    str(name): dict(term_map)
                    for name, term_map in dict(
                        native_terms.get("term_maps", {})
                    ).items()
                }
                timings["family_term_map_build_s"] = float(time.perf_counter() - t0)
                timings["family_term_map_backend_actual"] = str(
                    native_terms.get(
                        "backend_actual",
                        "cpp_spatial_jw_family_term_maps",
                    )
                )
                timings["family_term_map_cpp_seconds"] = float(
                    native_terms.get("seconds", 0.0)
                )
                timings["family_term_map_cpp_counts"] = dict(
                    native_terms.get("counts", {})
                )
                timings["family_term_map_cpp_dynamic_local_ops"] = int(
                    len(dynamic_ops)
                )
            if family_term_maps is None and native_entries is not None:
                try:
                    from pyqed.mps import cpp_davidson as _cpp_davidson

                    native_term_builder = getattr(
                        _cpp_davidson,
                        "build_spatial_qchem_family_term_maps",
                        None,
                    )
                    if native_term_builder is not None:
                        native_terms = native_term_builder(
                            native_entries,
                            int(native_n_sites if native_n_sites is not None else self.ncas),
                            cutoff,
                        )
                        dynamic_ops = dict(native_terms.get("local_ops", {}) or {})
                        if dynamic_ops:
                            from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops

                            spatial_local_ops().update(
                                {
                                    str(name): np.asarray(matrix, dtype=complex)
                                    for name, matrix in dynamic_ops.items()
                                }
                            )
                        family_term_maps = {
                            str(name): dict(term_map)
                            for name, term_map in dict(
                                native_terms.get("term_maps", {})
                            ).items()
                        }
                        timings["family_term_map_build_s"] = float(
                            time.perf_counter() - t0
                        )
                        timings["family_term_map_backend_actual"] = str(
                            native_terms.get(
                                "backend_actual",
                                "cpp_spatial_jw_family_term_maps",
                            )
                        )
                        timings["family_term_map_cpp_seconds"] = float(
                            native_terms.get("seconds", 0.0)
                        )
                        timings["family_term_map_cpp_counts"] = dict(
                            native_terms.get("counts", {})
                        )
                        timings["family_term_map_cpp_dynamic_local_ops"] = int(
                            len(dynamic_ops)
                        )
                except Exception as exc:
                    timings["family_term_map_cpp_fallback_reason"] = repr(exc)
                    family_term_maps = None
            if family_term_maps is None:
                family_term_maps = _spatial_family_term_maps(
                    complementary,
                    cutoff=cutoff,
                )
                timings["family_term_map_build_s"] = float(time.perf_counter() - t0)
                timings["family_term_map_backend_actual"] = "symbolic_spatial_jw"
        else:
            family_term_maps = {}
            timings["family_term_map_build_s"] = 0.0
            timings["family_term_map_backend_actual"] = "skipped"
            timings["family_term_map_skip_reason"] = (
                f"{self.spatial_family_environment_backend}_uses_generator_entries"
            )
        timings["family_generator_entry_counts"] = dict(generator_entry_counts)
        if self.debug_spatial_family_hamiltonian_check:
            t0 = time.perf_counter()
            reference_terms = _merge_spatial_term_maps(
                _spatial_one_body_term_map(h1_spatial, cutoff=cutoff),
                _spatial_two_body_spinfree_term_map(eri_spatial, cutoff=cutoff),
                cutoff=cutoff,
            )
            family_hamiltonian_terms = _merge_spatial_term_maps(
                *family_term_maps.values(),
                cutoff=cutoff,
            )
            hamiltonian_check = _compare_spatial_family_term_map(
                reference_terms,
                family_hamiltonian_terms,
                cutoff=cutoff,
            )
            timings["family_hamiltonian_check_s"] = float(
                time.perf_counter() - t0
            )
        else:
            hamiltonian_check = {
                "enabled": False,
                "reason": "set debug_spatial_family_hamiltonian_check=True",
            }
        return {
            "complementary": complementary,
            "term_maps": family_term_maps,
            "carrier": native_carrier,
            "native_family_mpos": native_family_mpos,
            "native_family_mpo_info": native_family_mpo_info,
            "native_family_mpo_owner": native_family_mpo_owner,
            "native_family_descriptor_key": native_family_descriptor_key,
            "native_family_descriptor_names": native_family_descriptor_names,
            "native_family_descriptor_info": native_family_descriptor_info,
            "hamiltonian_check": hamiltonian_check,
            "timings": timings,
        }

    def _build_spatial_family_environment_mpos(
        self,
        complementary,
        family_term_maps,
        *,
        cutoff=1e-10,
        native_family_mpos=None,
        native_family_mpo_info=None,
    ):
        """Build optional family MPO environments for block2-like backends."""

        backend = self.spatial_family_environment_backend
        family_tensor_mpos = {}
        family_mpo_info = {}
        if backend == "block2_table" and native_family_mpos:
            return (
                {
                    str(name): list(factors)
                    for name, factors in dict(native_family_mpos).items()
                },
                {
                    str(name): dict(info)
                    for name, info in dict(native_family_mpo_info or {}).items()
                },
            )
        if backend == "generator_table":
            return {}, {
                "R": {
                    "source": "native_generator_entries",
                    "generator_terms": int(
                        len(getattr(complementary.get("R"), "entries", {}) or {})
                    ),
                },
                "P": {
                    "source": "native_generator_entries",
                    "generator_terms": int(
                        len(getattr(complementary.get("P"), "entries", {}) or {})
                    ),
                },
            }
        if backend in {"block2_adaptive", "block2_native"}:
            try:
                (
                    family_tensor_mpos,
                    family_mpo_info,
                ) = _build_spatial_native_generator_family_mpos(
                    complementary,
                    self.ncas,
                    cutoff=cutoff,
                    p_grouping=self.spatial_native_p_grouping,
                )
            except Exception as exc:
                if backend == "block2_native":
                    raise RuntimeError(
                        "C++ spin-free generator family MPO build failed."
                    ) from exc
                warnings.warn(
                    "Falling back to symbolic R/P family MPOs after native "
                    f"spin-free generator build failed: {exc}",
                    RuntimeWarning,
                )
                family_tensor_mpos = {}
                family_mpo_info = {}

        if backend not in {"block2", "block2_adaptive", "block2_table"}:
            return family_tensor_mpos, family_mpo_info

        def _symbolic_table_term_groups(family_name, family_terms):
            name = str(family_name)
            terms = dict(family_terms or {})
            if backend != "block2_table" or name != "P" or len(terms) <= 1:
                return [(name, terms, {})]
            split_groups = getattr(
                self,
                "spatial_block2_table_p_split_groups",
                "auto",
            )
            if split_groups == "auto":
                n_groups = 2 if int(self.ncas) > 1 else 1
            else:
                n_groups = int(split_groups)
            n_groups = min(max(1, n_groups), max(1, int(self.ncas)))
            if n_groups <= 1:
                return [(name, terms, {})]

            metric = str(
                getattr(
                    self,
                    "spatial_block2_table_p_split_metric",
                    "auto",
                )
            )
            if metric == "auto":
                metric = "first_site"

            def _split_site(dofs):
                dofs = tuple(int(site) for site in dofs)
                if not dofs:
                    return 0
                if metric == "first_site":
                    return dofs[0]
                if metric == "min_site":
                    return min(dofs)
                if metric == "max_site":
                    return max(dofs)
                if metric == "center":
                    return int(round(float(sum(dofs)) / float(len(dofs))))
                if metric == "span":
                    return max(dofs) - min(dofs)
                return max(dofs)

            groups = {}
            for key, coeff in terms.items():
                try:
                    _symbol, dofs = key
                    split_site = _split_site(dofs)
                    group_key = min(
                        n_groups - 1,
                        max(0, split_site * n_groups // int(self.ncas)),
                    )
                except Exception:
                    group_key = 0
                groups.setdefault(group_key, {})[key] = coeff
            if len(groups) <= 1:
                return [(name, terms, {})]
            return [
                (
                    f"{name}:g{group_key}",
                    group_terms,
                    {
                        "family_base": name,
                        "split_key": int(group_key),
                        "split_source": f"symbolic_{metric}_window",
                        "split_metric": metric,
                        "split_groups_requested": int(n_groups),
                    },
                )
                for group_key, group_terms in sorted(groups.items())
                if group_terms
            ]

        family_basis_sites = [BasisSpatialFermion(i) for i in range(self.ncas)]
        for family_name, family_terms in family_term_maps.items():
            name = str(family_name)
            if (
                backend == "block2_table"
                and name == "P"
                and bool(getattr(self, "spatial_block2_table_native_p", False))
            ):
                p_family = complementary.get("P")
                family_mpo_info[name] = {
                    "source": "native_direct_generator_table",
                    "family_base": name,
                    "generator_terms": int(
                        len(getattr(p_family, "entries", {}) or {})
                    ),
                    "symbolic_mpo_replaced": True,
                }
                continue
            symbolic_groups = _symbolic_table_term_groups(name, family_terms)
            if len(symbolic_groups) > 1:
                split_names = []
                max_bond = 0
                total_terms = 0
                for group_name, group_terms, group_info in symbolic_groups:
                    family_mpo, family_term_count = _build_tensor_mpo_from_symbolic_terms(
                        family_basis_sites,
                        group_terms,
                        cutoff=cutoff,
                        algo=self.spatial_abelian_symbolic_algo,
                    )
                    bond_orders = tuple(int(x) for x in family_mpo.bond_orders())
                    max_bond = max(max_bond, int(max(bond_orders)))
                    total_terms += int(family_term_count)
                    family_tensor_mpos[group_name] = family_mpo.factors
                    family_mpo_info[group_name] = {
                        "source": "symbolic_spatial_term_map_split",
                        "symbolic_terms": int(family_term_count),
                        "mpo_max_bond": int(max(bond_orders)),
                        "bond_orders": bond_orders,
                        **group_info,
                    }
                    split_names.append(group_name)
                family_mpo_info[name] = {
                    "source": "symbolic_spatial_term_map_split_summary",
                    "family_base": name,
                    "split_family_names": tuple(split_names),
                    "split_groups": int(len(split_names)),
                    "split_metric": (
                        family_mpo_info[split_names[0]].get("split_metric")
                        if split_names
                        else None
                    ),
                    "split_groups_requested": int(
                        family_mpo_info[split_names[0]].get(
                            "split_groups_requested",
                            len(split_names),
                        )
                    )
                    if split_names
                    else int(len(split_names)),
                    "symbolic_terms": int(total_terms),
                    "mpo_max_bond": int(max_bond),
                }
                continue

            group_name, group_terms, _group_info = symbolic_groups[0]
            family_mpo, family_term_count = _build_tensor_mpo_from_symbolic_terms(
                family_basis_sites,
                group_terms,
                cutoff=cutoff,
                algo=self.spatial_abelian_symbolic_algo,
            )
            symbolic_info = {
                "source": "symbolic_spatial_term_map",
                "symbolic_terms": int(family_term_count),
                "mpo_max_bond": int(max(family_mpo.bond_orders())),
                "bond_orders": tuple(int(x) for x in family_mpo.bond_orders()),
            }
            native_info = family_mpo_info.get(name)
            native_bond = (
                None
                if native_info is None
                else int(native_info.get("mpo_max_bond", 0))
            )
            if (
                backend == "block2"
                or backend == "block2_table"
                or native_bond is None
                or symbolic_info["mpo_max_bond"] < native_bond
            ):
                family_tensor_mpos[group_name] = family_mpo.factors
                family_mpo_info[group_name] = {
                    "source": (
                        "symbolic_spatial_term_map"
                        if backend in {"block2", "block2_table"}
                        else "symbolic_spatial_term_map_selected"
                    ),
                    "symbolic_terms": symbolic_info["symbolic_terms"],
                    "mpo_max_bond": symbolic_info["mpo_max_bond"],
                }
                if native_info is not None:
                    family_mpo_info[group_name]["native_candidate"] = native_info
            else:
                family_mpo_info[name] = {
                    **native_info,
                    "symbolic_candidate": symbolic_info,
                }
        return family_tensor_mpos, family_mpo_info

    def _expose_spatial_family_environment(
        self,
        complementary,
        family_term_maps,
        family_tensor_mpos,
        *,
        expose_direct_terms,
    ):
        """Attach complementary family payloads consumed by the sweep code."""

        backend = self.spatial_family_environment_backend
        self.complementary_operators = complementary
        self.complementary_operator_mpos = family_tensor_mpos or None
        expose_symbolic_terms = expose_direct_terms and (
            backend == "direct_terms"
            or (backend == "block2_table" and not family_tensor_mpos)
        )
        self.complementary_operator_term_maps = (
            family_term_maps
            if expose_symbolic_terms
            else None
        )
        if backend in {"generator_terms", "generator_table"}:
            r_family = complementary.get("R")
            p_family = complementary.get("P")
            self.complementary_operator_generator_entries = {
                "R": dict(getattr(r_family, "entries", {})),
                "P": dict(getattr(p_family, "entries", {})),
            }
        elif backend == "block2_table" and bool(
            getattr(self, "spatial_block2_table_native_p", False)
        ):
            p_family = complementary.get("P")
            self.complementary_operator_generator_entries = {
                "P": dict(getattr(p_family, "entries", {})),
            }
        else:
            self.complementary_operator_generator_entries = None

    def _spatial_family_build_metadata(
        self,
        family_data,
        *,
        family_tensor_mpos=None,
        family_mpo_info=None,
        expose_direct_terms=False,
    ):
        """Return compact metadata for the Abelian spatial family build."""

        complementary = family_data["complementary"]
        family_term_maps = family_data["term_maps"]
        timings = dict(family_data.get("timings", {}))
        backend = self.spatial_family_environment_backend
        symbolic_term_counts = {
            str(name): int(len(term_map))
            for name, term_map in family_term_maps.items()
        }
        generator_entry_counts = timings.get("family_generator_entry_counts")
        if generator_entry_counts is None:
            generator_entry_counts = _spatial_family_generator_entry_counts(complementary)
        environment = {
            "backend": backend,
            "uses_family_mpos": bool(family_tensor_mpos),
            "uses_symbolic_term_environments": bool(
                expose_direct_terms and backend == "direct_terms"
            ),
            "uses_symbolic_terms_as_table_builder": bool(
                expose_direct_terms and backend == "block2_table"
                and not bool(family_tensor_mpos)
            ),
            "uses_raw_generator_entries": bool(
                backend in {"generator_terms", "generator_table"}
                or (
                    backend == "block2_table"
                    and bool(getattr(self, "spatial_block2_table_native_p", False))
                )
            ),
            "uses_renormalized_operator_tables": bool(
                backend in {"block2_table", "generator_table"}
            ),
            "long_term_path": bool(backend in {"block2_table", "generator_table"}),
            "uses_native_generator_tables": bool(backend == "generator_table"),
            "uses_native_p_generator_table": bool(
                backend == "block2_table"
                and bool(getattr(self, "spatial_block2_table_native_p", False))
            ),
        }
        metadata = {
            "complementary_operator_families": complementary.as_metadata(),
            "complementary_operator_family_names": complementary.names,
            "complementary_operator_total_terms": int(complementary.n_terms),
            "complementary_operator_builder": "spatial_spinfree_sparse_S/R/A/P/B/Q",
            "complementary_operator_family_term_counts": symbolic_term_counts,
            "complementary_operator_family_generator_entry_counts": dict(
                generator_entry_counts
            ),
            "complementary_operator_family_symbolic_term_maps": {
                "built": bool(symbolic_term_counts),
                "backend": timings.get("family_term_map_backend_actual"),
                "skip_reason": timings.get("family_term_map_skip_reason"),
            },
            "complementary_operator_family_hamiltonian_check": (
                family_data["hamiltonian_check"]
            ),
            "complementary_operator_family_environment": environment,
            "complementary_operator_family_build_timings": timings,
        }
        if family_mpo_info is not None:
            metadata["complementary_operator_family_mpos"] = family_mpo_info
        return metadata

    def build(self, mo_coeff=None):
        if self.symmetry is not None and "su2" in self.symmetry:
            self._require_su2_cpp_integral_reference()

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
        self.complementary_operators = None
        self.complementary_operator_mpos = None
        self.complementary_operator_term_maps = None
        self.complementary_operator_generator_entries = None
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
            self.spatial_abelian_mpo,
            self.spatial_abelian_symbolic_algo,
            self.spatial_family_environment_backend,
            self.spatial_native_p_grouping,
            self.spatial_block2_table_p_split_metric,
            self.spatial_block2_table_p_split_groups,
            self.spatial_block2_table_native_p,
            self.spatial_complementary_payload_tensor_matvec,
            self.spatial_precontracted_family_environment,
            self.spatial_boundary_table_max_dim,
            self.spatial_exact_component_compression_policy,
            self.spatial_exact_component_compression_validate,
            self.spatial_exact_component_compression_validation_vectors,
            self.spatial_exact_component_compression_min_reduction,
            self.spatial_exact_component_compression_max_group_size,
            self.spatial_enable_cpp_boundary_r,
            self.spatial_validate_cpp_boundary_r,
            self.spatial_enable_cpp_boundary_p,
            self.spatial_validate_cpp_boundary_p,
            self.spatial_cpp_boundary_p_validation_policy,
            self.spatial_direct_operator_batch_min_entries,
            self.debug_complementary_action_check,
            self.debug_complementary_action_check_tol,
            self.debug_complementary_action_check_limit,
            self.debug_spatial_family_hamiltonian_check,
        )
        if cache_key == self._hamiltonian_mpo_cache_key and self.H is not None and self.H_raw is not None:
            self._log("  Reusing Hamiltonian MPO cache.")
            return self

        owns_su2_solver_state = bool(
            self.site == "spatial" and self.spatial_reduced_mpo
        )
        cached = (
            None
            if owns_su2_solver_state
            else _GLOBAL_HAMILTONIAN_MPO_CACHE.get(cache_key)
        )
        if cached is not None:
            self._log("  Reusing global Hamiltonian MPO cache.")
            self._spatial_operator_cache = None
            self._active_hamiltonian = cached.get("hamiltonian")
            self.complementary_operators = cached.get("complementary_operators")
            self.complementary_operator_mpos = cached.get("complementary_operator_mpos")
            self.complementary_operator_term_maps = cached.get(
                "complementary_operator_term_maps"
            )
            self.complementary_operator_generator_entries = cached.get(
                "complementary_operator_generator_entries"
            )
            self.H_raw = cached["factors"]
            self.H = cached["factors"]
            self._hamiltonian_mpo_cache_key = cache_key
            self._symmetric_mpo_cache = {}
            self._active_integral_build_info.update(deepcopy(dict(cached["info"])))
            return self



        # h2e[0,0] -= h2e[0,0].swapaxes(1,3)
        # h2e[1,1] -= h2e[1,1].swapaxes(1,3)


        n_spatial = self.ncas
        nso = 2 * n_spatial
        self._log(f"  System: {n_spatial} spatial orbitals, {nso} spin-orbitals")
        build_timings = {}

        def _record_build_time(name, start):
            build_timings[str(name)] = float(time.perf_counter() - start)

        if self.site == "spatial":
            if self.spatial_reduced_mpo:
                if self.spin_purification:
                    raise NotImplementedError(
                        "spatial_reduced_mpo does not support spin-purification penalties."
                    )
                cached = (
                    None
                    if owns_su2_solver_state
                    else _GLOBAL_HAMILTONIAN_MPO_CACHE.get(cache_key)
                )
                if cached is not None:
                    self._log("  Reusing global spatial reduced Hamiltonian MPO cache.")
                    self._spatial_operator_cache = None
                    self._active_hamiltonian = cached.get("hamiltonian")
                    self.complementary_operators = getattr(
                        self._active_hamiltonian,
                        "complementary_operators",
                        None,
                    )
                    self.H_raw = cached["factors"]
                    self.H = cached["factors"]
                    self._hamiltonian_mpo_cache_key = cache_key
                    self._symmetric_mpo_cache = {}
                    self._active_integral_build_info.update(
                        deepcopy(dict(cached["info"]))
                    )
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
                self.complementary_operators = reduced_hamiltonian.complementary_operators
                self.H_raw = reduced_hamiltonian.factors
                self.H = reduced_hamiltonian.factors
                self._hamiltonian_mpo_cache_key = cache_key
                self._symmetric_mpo_cache = {}
                self._active_integral_build_info.update(reduced_hamiltonian.info)
                if not owns_su2_solver_state:
                    _store_global_hamiltonian_mpo_cache(
                        cache_key,
                        factors=reduced_hamiltonian.factors,
                        info=reduced_hamiltonian.info,
                        hamiltonian=reduced_hamiltonian,
                    )
                return self

            if self.spatial_abelian_mpo == "direct":
                self._log("  Building spatial-orbital Hamiltonian MPO directly in d=4 channels...")
                t_carrier = time.perf_counter()
                tensor_mpo, spatial_term_count, spin_penalty_term_count = _build_spatial_hamiltonian_tensor_mpo(
                    h1e,
                    eri,
                    spin_purification=self.spin_purification,
                    shift=self.shift,
                    cutoff=1e-10,
                    symbolic_algo=self.spatial_abelian_symbolic_algo,
                )
                _record_build_time("carrier_build_s", t_carrier)
                t_family_data = time.perf_counter()
                family_data = self._build_spatial_complementary_family_data(
                    h1e,
                    eri,
                    cutoff=1e-10,
                )
                _record_build_time("family_data_s", t_family_data)
                complementary = family_data["complementary"]
                t_family_mpo = time.perf_counter()
                family_tensor_mpos, family_mpo_info = (
                    self._build_spatial_family_environment_mpos(
                        complementary,
                        family_data["term_maps"],
                        cutoff=1e-10,
                        native_family_mpos=family_data.get("native_family_mpos"),
                        native_family_mpo_info=family_data.get(
                            "native_family_mpo_info"
                        ),
                    )
                )
                _record_build_time("family_mpo_build_s", t_family_mpo)
                self._expose_spatial_family_environment(
                    complementary,
                    family_data["term_maps"],
                    family_tensor_mpos,
                    expose_direct_terms=True,
                )
                self._spatial_operator_cache = None
                self._active_hamiltonian = None
                self.H_raw = tensor_mpo.factors
                self.H = tensor_mpo.factors
                self._hamiltonian_mpo_cache_key = cache_key
                self._symmetric_mpo_cache = {}
                self._active_integral_build_info.update(
                    {
                        "representation": "spatial_direct_symbolic_mpo",
                        "symbolic_terms": int(spatial_term_count),
                        "mpo_max_bond": int(max(tensor_mpo.bond_orders())),
                        "site": "spatial",
                        "spatial_abelian_mpo": "direct",
                        "spatial_abelian_symbolic_algo": self.spatial_abelian_symbolic_algo,
                        "spatial_family_environment_backend": (
                            self.spatial_family_environment_backend
                        ),
                        "spatial_native_p_grouping": self.spatial_native_p_grouping,
                        "pipeline": (
                            "qchem_integrals->spatial_d4_symbolic_terms"
                            f"->autompo_{self.spatial_abelian_symbolic_algo}"
                        ),
                        "build_timings": dict(build_timings),
                        "spatial_direct_term_representation": "spinfree_R/P_compatible",
                        **self._spatial_family_build_metadata(
                            family_data,
                            family_tensor_mpos=family_tensor_mpos,
                            family_mpo_info=family_mpo_info,
                            expose_direct_terms=True,
                        ),
                    }
                )
                if self.spin_purification:
                    self._active_integral_build_info["spin_penalty_terms"] = int(spin_penalty_term_count)
                return self

            t_family_data = time.perf_counter()
            family_data = self._build_spatial_complementary_family_data(
                h1e,
                eri,
                cutoff=1e-10,
            )
            _record_build_time("family_data_s", t_family_data)
            complementary = family_data["complementary"]
            t_family_mpo = time.perf_counter()
            family_tensor_mpos, family_mpo_info = (
                self._build_spatial_family_environment_mpos(
                    complementary,
                    family_data["term_maps"],
                    cutoff=1e-10,
                    native_family_mpos=family_data.get("native_family_mpos"),
                    native_family_mpo_info=family_data.get("native_family_mpo_info"),
                )
            )
            _record_build_time("family_mpo_build_s", t_family_mpo)
            if (
                self.spatial_abelian_mpo == "spatial"
                and (
                    family_tensor_mpos
                    or self.spatial_family_environment_backend == "generator_table"
                )
                and not self.spin_purification
                and self._can_use_spatial_block2_carrier()
            ):
                self._log("  Building spatial block2-table carrier MPO in d=4 channels...")
                t_carrier = time.perf_counter()
                native_carrier = family_data.get("carrier")
                if native_carrier is not None:
                    carrier = SpatialCarrierMPO(
                        factors=list(native_carrier["factors"]),
                        info=dict(native_carrier["info"]),
                    )
                    build_timings["carrier_build_backend_actual"] = (
                        "cpp_qchem_spatial_block2_setup"
                    )
                else:
                    carrier = build_spatial_block2_carrier_mpo(
                        n_spatial,
                        local_dim=4,
                    )
                    build_timings["carrier_build_backend_actual"] = str(
                        carrier.info.get("source", "python_spatial_identity_scaffold")
                    )
                _record_build_time("carrier_build_s", t_carrier)
                self._expose_spatial_family_environment(
                    complementary,
                    family_data["term_maps"],
                    family_tensor_mpos,
                    expose_direct_terms=False,
                )
                self._spatial_operator_cache = None
                self._active_hamiltonian = None
                self.H_raw = carrier.factors
                self.H = carrier.factors
                self._hamiltonian_mpo_cache_key = cache_key
                self._symmetric_mpo_cache = {}
                if self.spatial_family_environment_backend == "generator_table":
                    family_pipeline_stage = "native_generator_tables"
                elif family_tensor_mpos:
                    family_pipeline_stage = "block2_table_family_mpos"
                else:
                    family_pipeline_stage = "carrier_only"
                self._active_integral_build_info.update(
                    {
                        **carrier.info,
                        "spatial_abelian_mpo": "spatial",
                        "spatial_family_environment_backend": (
                            self.spatial_family_environment_backend
                        ),
                        "spatial_native_p_grouping": self.spatial_native_p_grouping,
                        "pipeline": (
                            "qchem_integrals->spatial_d4_carrier_scaffold"
                            f"->{family_pipeline_stage}"
                        ),
                        "build_timings": dict(build_timings),
                        **self._spatial_family_build_metadata(
                            family_data,
                            family_tensor_mpos=family_tensor_mpos,
                            family_mpo_info=family_mpo_info,
                            expose_direct_terms=False,
                        ),
                    }
                )
                _store_global_hamiltonian_mpo_cache(
                    cache_key,
                    factors=carrier.factors,
                    info=self._active_integral_build_info,
                    hamiltonian=None,
                    complementary_operators=self.complementary_operators,
                    complementary_operator_mpos=self.complementary_operator_mpos,
                    complementary_operator_term_maps=(
                        self.complementary_operator_term_maps
                    ),
                    complementary_operator_generator_entries=(
                        self.complementary_operator_generator_entries
                    ),
                )
                return self

            self._log("  Building spatial-orbital Hamiltonian MPO by grouping spin-orbital pairs...")
            self._active_hamiltonian = None
            t_carrier = time.perf_counter()
            spin_tensor_mpo, spin_term_count, spin_penalty_term_count = _build_spin_orbital_dense_hamiltonian_tensor_mpo(
                h1e,
                eri,
                n_spatial,
                spin_purification=self.spin_purification,
                shift=self.shift,
                cutoff=1e-10,
            )
            _record_build_time("spin_orbital_carrier_build_s", t_carrier)
            t_group = time.perf_counter()
            tensor_mpo = _group_spin_orbital_mpo_pairs(spin_tensor_mpo)
            _record_build_time("group_spin_orbital_carrier_s", t_group)
            build_timings["carrier_build_s"] = (
                build_timings["spin_orbital_carrier_build_s"]
                + build_timings["group_spin_orbital_carrier_s"]
            )
            self._expose_spatial_family_environment(
                complementary,
                family_data["term_maps"],
                family_tensor_mpos,
                expose_direct_terms=False,
            )
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
                    "spatial_abelian_mpo": "grouped",
                    "spatial_family_environment_backend": (
                        self.spatial_family_environment_backend
                    ),
                    "spatial_native_p_grouping": self.spatial_native_p_grouping,
                    "spatial_grouped_carrier_reason": (
                        "requested_grouped"
                        if self.spatial_abelian_mpo == "grouped"
                        else "dense_run_requires_full_carrier"
                        if not self._can_use_spatial_block2_carrier()
                        else "spatial_carrier_requires_family_mpos"
                        if not family_tensor_mpos
                        else "spin_purification_requires_full_carrier"
                    ),
                    "build_timings": dict(build_timings),
                    **self._spatial_family_build_metadata(
                        family_data,
                        family_tensor_mpos=family_tensor_mpos,
                        family_mpo_info=family_mpo_info,
                        expose_direct_terms=False,
                    ),
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
                if hasattr(state_for_eval, "sites"):
                    psi = _nonabelian_mps_to_dense_vector(state_for_eval)
                    norm = np.vdot(psi, psi)
                    s2_mat = _build_spatial_s2_matrix(self._get_spatial_ops_for_rdm())
                    s2_vals.append(float(np.real(np.vdot(psi, s2_mat @ psi) / norm)))
                    continue
                if hasattr(state_for_eval.Bs[0], 'qns'):
                    from pyqed.mps.mps import symmetric_to_dense
                    state_for_eval = symmetric_to_dense(
                        state,
                        site_qn_maps=self._dense_site_qn_maps(),
                    )
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
                dense_state = mps_lib.symmetric_to_dense(
                    state,
                    site_qn_maps=self._dense_site_qn_maps(),
                )
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

        if any(value < 1 for value in sweep_list):
            raise ValueError("Every DMRG schedule stage requires at least one complete sweep.")

        return list(zip(d_list, sweep_list))

    def run(
        self,
        nstates=1,
        weights=None,
        symmetry_list=None,
        symmetry=None,
        nsweeps=50,
        D=None,
        D_schedule=None,
        nsweeps_schedule=None,
        initial_guess=None,
        mo_coeff=None,
        compute_s2=False,
        *,
        su2_kernel_backend="auto",
        require_convergence=True,
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
        nstates : int
            Number of roots. Values larger than one run state-averaged DMRG.
        weights : array_like, optional
            Nonnegative state-average weights. They are normalized internally.
        su2_kernel_backend : {"auto", "cpp", "python"}
            SU(2) execution backend. ``"cpp"`` requires the compiled engine;
            ``"python"`` selects the slower reference implementation.
        nsweeps : int
            Maximum number of complete sweeps. One complete sweep is a
            left-to-right pass followed by a right-to-left pass. The default
            is 50 and the solver may stop earlier after a converged complete
            sweep.
        require_convergence : bool
            Raise ``RuntimeError`` if the solver reaches ``nsweeps`` without
            converging. Set this to ``False`` only for forced-sweep benchmarks
            or diagnostic runs.

        Returns
        -------
        SU2DMRG or TensorDMRG
            The backend solver. For SU(2) this is an ``SU2DMRG`` owner with
            ``energy``, ``energies``, ``state_average_energy``, ``states``,
            ``history``, status fields, and compact ``diagnostics``.
        """
        if isinstance(nstates, (bool, np.bool_)):
            raise TypeError("nstates must be a positive integer.")
        try:
            nstates = operator.index(nstates)
        except TypeError as exc:
            raise TypeError("nstates must be a positive integer.") from exc
        if nstates < 1:
            raise ValueError("nstates must be positive.")
        if isinstance(nsweeps, (bool, np.bool_)):
            raise TypeError("nsweeps must be a positive integer.")
        try:
            nsweeps = operator.index(nsweeps)
        except TypeError as exc:
            raise TypeError("nsweeps must be a positive integer.") from exc
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        self.max_sweeps = int(nsweeps)
        complete_sweep_limit = (
            f"{nsweeps} complete "
            f"{'sweep' if nsweeps == 1 else 'sweeps'}"
        )
        su2_kernel_backend = str(su2_kernel_backend).lower()
        if su2_kernel_backend not in {"auto", "cpp", "python"}:
            raise ValueError(
                "su2_kernel_backend must be 'auto', 'cpp', or 'python'."
            )
        if "fully_reduced_state_average" in kwargs:
            raise TypeError(
                "fully_reduced_state_average was removed. State-averaged "
                "SU(2) uses the fully reduced C++ representation automatically; "
                "use su2_kernel_backend='python' for the canonical reference path."
            )
        if D is not None:
            self.D = self.m = int(D)
            if self.m_warmup is None:
                self.m_warmup = self.D
        elif self.D is None:
            if D_schedule is None:
                raise ValueError("DMRG.run requires D=... or D_schedule=... when no constructor D was set.")
            self.D = self.m = max(int(x) for x in D_schedule)
            if self.m_warmup is None:
                self.m_warmup = self.D
        self.nstates = nstates
        if weights is None:
            self.weights = np.ones(nstates) / nstates
        else:
            self.weights = np.asarray(weights, dtype=float).reshape(-1)
            if self.weights.size != int(nstates):
                raise ValueError("weights must match nstates.")
            if not np.all(np.isfinite(self.weights)):
                raise ValueError("weights must be finite.")
            if np.any(self.weights < 0.0):
                raise ValueError("weights must be nonnegative.")
            weight_sum = float(np.sum(self.weights))
            if weight_sum <= 0.0:
                raise ValueError("weights must have a positive sum.")
            self.weights = self.weights / weight_sum
        previous_symmetry = tuple(self.symmetry or ())
        symmetry_was_explicit = symmetry is not None or symmetry_list is not None
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
        has_su2 = bool(symmetry_list and "su2" in symmetry_list)
        if not has_su2 and su2_kernel_backend != "auto":
            raise ValueError(
                "su2_kernel_backend is only valid when symmetry='su2'."
            )
        if has_su2:
            self._require_su2_cpp_integral_reference()
        if (
            has_su2
            and su2_kernel_backend != "python"
            and self.spatial_site_basis != "fully_reduced"
        ):
            # Production SU(2) uses one genuinely reduced site representation.
            # The canonical four-state representation retains explicit magnetic
            # components and is reserved for the Python reference backend.
            self.spatial_site_basis = "fully_reduced"
            self.spatial_reduced_mpo = True
            self.d = 3
            self._invalidate_hamiltonian_mpo()
        if symmetry_was_explicit:
            if has_su2:
                self.site = self.site_basis = self.orbital_layout = "spatial"
                self.spatial_reduced_mpo = True
            elif "su2" in previous_symmetry:
                self.spatial_reduced_mpo = False
            self.d = (
                3
                if self.site == "spatial" and self.spatial_site_basis == "fully_reduced"
                else 4 if self.site == "spatial" else 2
            )
            representation_changed = (
                tuple(symmetry_list or ()) != previous_symmetry
                or (has_su2 and self.site != "spatial")
            )
            if representation_changed:
                self._invalidate_hamiltonian_mpo()
        if initial_guess is not None:
            self.init_guess = initial_guess
        if mo_coeff is not None:
            self.build(mo_coeff=mo_coeff)
        if self.H_raw is None:
            self.build()
        sweep_tol = kwargs.pop("sweep_tol", kwargs.pop("conv_tol", self.tol))
        davidson_tol_explicit = "davidson_tol" in kwargs
        davidson_max_iter_explicit = "davidson_max_iter" in kwargs
        davidson_tol = kwargs.pop(
            "davidson_tol",
            1.0e-3 if has_su2 else 1.0e-5,
        )
        davidson_max_iter = kwargs.pop("davidson_max_iter", 30)
        noise = kwargs.pop("noise", 1.0e-4)
        noise_decay = kwargs.pop("noise_decay", 0.1)
        noise_cutoff = kwargs.pop("noise_cutoff", 1.0e-9)
        local_dense_max_dim = kwargs.pop("local_dense_max_dim", 0)
        final_expectation = kwargs.pop("final_expectation", None)
        dmrg_performance = kwargs.pop("dmrg_performance", self.dmrg_performance)
        abelian_matvec_options = kwargs.pop(
            "abelian_matvec_options",
            kwargs.pop("dmrg_matvec_options", self.abelian_matvec_options),
        )
        resolved_abelian_options = resolve_abelian_matvec_options(
            dmrg_performance or "auto",
            abelian_matvec_options,
        )
        abelian_matvec_options = dict(resolved_abelian_options)

        def _metadata_abelian_options(options):
            metadata = dict(options or {})
            if "moving_environment_cpp_state_owner_instance" in metadata:
                metadata["moving_environment_cpp_state_owner_instance"] = (
                    "cpp_moving_environment_build_owner"
                )
            return metadata

        native_symmetric_mpo_storage = bool(
            resolved_abelian_options.get("native_site_storage", False)
        )
        # Initialize Symmetry
        self.sym_mgr = SymmetryManager(symmetry_list, orb_sym=getattr(self, "orb_sym", None))
        if self.sym_mgr.enabled:
            if getattr(self.sym_mgr, "has_nonabelian", False):
                if self.site != "spatial":
                    raise NotImplementedError("Non-Abelian qchem DMRG currently requires site='spatial'.")
                from pyqed.qchem.dmrg.backends.nonabelian import run_spatial_qchem_dmrg

                local_solver_kwargs = dict(
                    kwargs.pop("local_solver_kwargs", {}) or {}
                )
                local_tol_explicit = "tol" in local_solver_kwargs
                local_itermax_explicit = "itermax" in local_solver_kwargs
                local_solver_kwargs.setdefault("tol", davidson_tol)
                local_solver_kwargs.setdefault("itermax", davidson_max_iter)
                if (
                    has_su2
                    and not davidson_tol_explicit
                    and not local_tol_explicit
                    and "local_solver_schedule" not in kwargs
                ):
                    tolerances = [1.0e-3] * 4 + [1.0e-5] * 4 + [1.0e-8]
                    if davidson_max_iter_explicit or local_itermax_explicit:
                        kwargs["local_solver_schedule"] = [
                            {"tol": tolerance}
                            for tolerance in tolerances
                            for _ in range(2)
                        ]
                    else:
                        iterations = [30] * 4 + [60] * 4 + [100]
                        kwargs["local_solver_schedule"] = [
                            {"tol": tolerance, "itermax": itermax}
                            for tolerance, itermax in zip(
                                tolerances,
                                iterations,
                            )
                            for _ in range(2)
                        ]
                t0 = time.time()
                self._log(f"  [Symmetry] Enabled: {self.sym_mgr.sym_types}")
                dmrg = run_spatial_qchem_dmrg(
                    self,
                    nsweeps=nsweeps,
                    max_bond=self.D,
                    initial_guess=initial_guess,
                    conv_tol=sweep_tol,
                    nstates=nstates,
                    weights=self.weights,
                    local_solver_kwargs=local_solver_kwargs,
                    su2_kernel_backend=su2_kernel_backend,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.dmrg = dmrg
                includes_core_energy = bool(
                    (self._active_integral_build_info or {}).get(
                        "includes_core_energy",
                        False,
                    )
                )
                raw_dmrg_energy = np.asarray(dmrg.e_tot, dtype=float)
                if includes_core_energy:
                    e_dmrg_total = raw_dmrg_energy
                    e_dmrg_active = raw_dmrg_energy - self.e_core
                else:
                    e_dmrg_total = raw_dmrg_energy + self.e_core
                    e_dmrg_active = raw_dmrg_energy
                if nstates == 1:
                    e_dmrg_total = float(e_dmrg_total)
                    e_dmrg_active = float(e_dmrg_active)
                self.e_tot = e_dmrg_total
                if compute_s2:
                    s = abs(float(self.spin)) / 2.0
                    self.s2 = s * (s + 1.0)
                    dmrg.s2 = self.s2
                dmrg.e_active = e_dmrg_active
                dmrg.e_core = self.e_core
                dmrg.e_tot = self.e_tot
                dmrg.energies = np.asarray(e_dmrg_total, dtype=float).reshape(-1)
                dmrg.energy = float(dmrg.energies[0])
                dmrg.state_average_energy = float(
                    np.dot(self.weights, dmrg.energies)
                )
                self.energy = dmrg.energy
                self.energies = dmrg.energies
                self.state_average_energy = dmrg.state_average_energy
                self.ground_state = dmrg.ground_state
                self.states = dmrg.states
                self.history = dmrg.history
                self.ncompleted = dmrg.ncompleted
                self.ncompleted_half_sweeps = dmrg.ncompleted_half_sweeps
                self.converged = bool(dmrg.converged)
                self.success = bool(dmrg.success)
                self.message = str(dmrg.message)
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
                if require_convergence and not dmrg.converged:
                    raise RuntimeError(
                        "SU(2) DMRG did not converge within "
                        f"{complete_sweep_limit}. Increase nsweeps or D, "
                        "loosen sweep_tol, or pass require_convergence=False "
                        "for a forced-sweep diagnostic run."
                    )
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
            sym_cache_key = (
                tuple(self.sym_mgr.sym_types),
                "native" if native_symmetric_mpo_storage else "legacy",
            )
            global_sym_cache_key = _global_symmetric_mpo_cache_key(
                getattr(self, "_hamiltonian_mpo_cache_key", None),
                sym_types=self.sym_mgr.sym_types,
                native_site_storage=native_symmetric_mpo_storage,
                site_qn_maps=site_qn_maps,
            )
            global_sym_cache = (
                None
                if global_sym_cache_key is None
                else _GLOBAL_SYMMETRIC_MPO_CACHE.get(global_sym_cache_key)
            )
            if global_sym_cache is not None:
                self._log("  Reusing global symmetric MPO cache.")
                final_H = global_sym_cache["hamiltonian"]
                final_complementary_mpos = global_sym_cache["complementary_mpos"]
                self._symmetric_mpo_cache[sym_cache_key] = final_H
                if final_complementary_mpos:
                    for family_name, family_final in final_complementary_mpos.items():
                        self._symmetric_mpo_cache[
                            (
                                sym_cache_key,
                                "complementary_family",
                                str(family_name),
                            )
                        ] = family_final
                timings = self._active_integral_build_info.setdefault(
                    "build_timings",
                    {},
                )
                timings["symmetric_mpo_global_cache_hits"] = int(
                    timings.get("symmetric_mpo_global_cache_hits", 0)
                ) + 1
            else:
                final_H = self._symmetric_mpo_cache.get(sym_cache_key)
                if final_H is None:
                    self._log(
                        "  Converting MPO to native Abelian tensors..."
                        if native_symmetric_mpo_storage
                        else "  Converting MPO to legacy symmetric tensors..."
                    )
                    t_convert = time.perf_counter()
                    final_H = dense_to_symmetric_mpo(
                        self.H,
                        site_qn_maps,
                        native_site_storage=native_symmetric_mpo_storage,
                    )
                    timings = self._active_integral_build_info.setdefault(
                        "build_timings",
                        {},
                    )
                    timings["symmetric_hamiltonian_convert_s"] = float(
                        time.perf_counter() - t_convert
                    )
                    self._symmetric_mpo_cache[sym_cache_key] = final_H
                    self._log(f"  MPO Converted. Sites: {len(final_H)}")
                else:
                    self._log("  Reusing symmetric MPO cache.")
                final_complementary_mpos = None
                if getattr(self, "complementary_operator_mpos", None):
                    final_complementary_mpos = {}
                    family_convert_total = 0.0
                    family_convert_by_name = {}
                    for family_name, family_mpo in self.complementary_operator_mpos.items():
                        family_cache_key = (
                            sym_cache_key,
                            "complementary_family",
                            str(family_name),
                        )
                        family_final = self._symmetric_mpo_cache.get(family_cache_key)
                        if family_final is None:
                            t_convert = time.perf_counter()
                            family_final = dense_to_symmetric_mpo(
                                family_mpo,
                                site_qn_maps,
                                native_site_storage=native_symmetric_mpo_storage,
                            )
                            elapsed = float(time.perf_counter() - t_convert)
                            family_convert_total += elapsed
                            family_convert_by_name[str(family_name)] = elapsed
                            self._symmetric_mpo_cache[family_cache_key] = family_final
                        final_complementary_mpos[str(family_name)] = family_final
                    if family_convert_by_name:
                        timings = self._active_integral_build_info.setdefault(
                            "build_timings",
                            {},
                        )
                        timings["symmetric_family_convert_s"] = float(
                            timings.get("symmetric_family_convert_s", 0.0)
                            + family_convert_total
                        )
                        timings["symmetric_family_convert_by_name_s"] = {
                            **dict(timings.get("symmetric_family_convert_by_name_s", {})),
                            **family_convert_by_name,
                        }
                _store_global_symmetric_mpo_cache(
                    global_sym_cache_key,
                    hamiltonian=final_H,
                    complementary_mpos=final_complementary_mpos,
                )
                if global_sym_cache_key is not None:
                    timings = self._active_integral_build_info.setdefault(
                        "build_timings",
                        {},
                    )
                    timings["symmetric_mpo_global_cache_stores"] = int(
                        timings.get("symmetric_mpo_global_cache_stores", 0)
                    ) + 1
            final_complementary_term_maps = getattr(
                self,
                "complementary_operator_term_maps",
                None,
            )
            final_complementary_generator_entries = getattr(
                self,
                "complementary_operator_generator_entries",
                None,
            )
            # Calculate Target QN
            target_qn = self.sym_mgr.get_target_qn(self.nelecas, self.spin)
            self._log(f"  Target QN set to: {target_qn}")
            use_symmetry = True
        else: # dense branch without U(1) symmetry
            final_H = self.H
            final_complementary_mpos = None
            final_complementary_term_maps = None
            final_complementary_generator_entries = None
            target_qn = None
            use_symmetry = False
            self.sym_mgr = None
            site_qn_maps = None
        self._site_qn_maps = site_qn_maps if use_symmetry else None
        active_info = self._active_integral_build_info or {}
        carrier_only_family_hamiltonian = (
            active_info.get("representation") == "spatial_block2_table_carrier_mpo"
        )
        if carrier_only_family_hamiltonian:
            abelian_matvec_options = dict(abelian_matvec_options or {})
            abelian_matvec_options.update(
                {
                    "packed_local_flat_matvec": False,
                    "packed_local_flat_projected_matvec": False,
                    "packed_local_flat_preconditioner": False,
                    "packed_local_family_flat_matvec": True,
                    "packed_local_family_flat_matvec_max_dim": 10**18,
                }
            )
            native_family_mpo_owner = getattr(
                self,
                "_native_spatial_family_mpo_owner",
                None,
            )
            reuse_native_family_mpo_owner = bool(
                native_family_mpo_owner is not None
                and abelian_matvec_options.get(
                    "moving_environment_cpp_state_owner",
                    False,
                )
            )
            if reuse_native_family_mpo_owner:
                abelian_matvec_options[
                    "moving_environment_cpp_state_owner_instance"
                ] = native_family_mpo_owner
                native_family_descriptor_key = getattr(
                    self,
                    "_native_spatial_family_descriptor_key",
                    None,
                )
                native_family_descriptor_names = tuple(
                    getattr(
                        self,
                        "_native_spatial_family_descriptor_names",
                        (),
                    )
                    or ()
                )
                if native_family_descriptor_key:
                    abelian_matvec_options[
                        "moving_environment_cpp_qchem_family_descriptor_key"
                    ] = native_family_descriptor_key
                    abelian_matvec_options[
                        "moving_environment_cpp_qchem_family_descriptor_names"
                    ] = native_family_descriptor_names
                if final_complementary_mpos:
                    try:
                        family_items = tuple(
                            sorted(
                                final_complementary_mpos.items(),
                                key=lambda item: str(item[0]),
                            )
                        )
                        family_names = tuple(str(name) for name, _ in family_items)
                        owned_family_mpo_key = (
                            "qchem-converted-family-mpos:"
                            f"{id(self)}:{hash(family_names)}"
                        )
                        t_owned = time.perf_counter()
                        native_family_mpo_owner.install_owned_family_mpos(
                            owned_family_mpo_key,
                            family_names,
                            tuple(factors for _, factors in family_items),
                        )
                        abelian_matvec_options[
                            "moving_environment_cpp_owned_family_mpo_key"
                        ] = owned_family_mpo_key
                        abelian_matvec_options[
                            "moving_environment_cpp_owned_family_mpo_names"
                        ] = family_names
                        timings = self._active_integral_build_info.setdefault(
                            "build_timings",
                            {},
                        )
                        timings[
                            "cpp_owned_converted_family_mpo_register_s"
                        ] = float(time.perf_counter() - t_owned)
                        timings[
                            "cpp_owned_converted_family_mpo_families"
                        ] = int(len(family_names))
                    except Exception as exc:
                        timings = self._active_integral_build_info.setdefault(
                            "build_timings",
                            {},
                        )
                        timings[
                            "cpp_owned_converted_family_mpo_register_error"
                        ] = repr(exc)
            final_expectation = False
            if self._active_integral_build_info is not None:
                self._active_integral_build_info[
                    "carrier_only_family_hamiltonian"
                ] = True
                self._active_integral_build_info[
                    "carrier_only_sweep_energy_final"
                ] = True
                self._active_integral_build_info[
                    "carrier_only_disabled_flat_local_shortcuts"
                ] = True
                self._active_integral_build_info[
                    "carrier_only_forced_family_flat_csr"
                ] = True
                self._active_integral_build_info[
                    "moving_environment_cpp_owner_reused_from_build"
                ] = reuse_native_family_mpo_owner
        native_initial_guess_storage = bool(
            use_symmetry
            and resolved_abelian_options.get("native_site_storage", False)
        )
        if self._active_integral_build_info is not None:
            self._active_integral_build_info["dmrg_performance"] = str(
                dmrg_performance or "auto"
            )
            self._active_integral_build_info["abelian_matvec_options"] = (
                None
                if abelian_matvec_options is None
                else _metadata_abelian_options(abelian_matvec_options)
            )
            self._active_integral_build_info[
                "native_initial_guess_storage"
            ] = bool(native_initial_guess_storage)
            self._active_integral_build_info[
                "native_symmetric_mpo_storage"
            ] = bool(use_symmetry and native_symmetric_mpo_storage)
        mps0 = self._resolve_initial_guess(
            use_symmetry=use_symmetry,
            native_site_storage=native_initial_guess_storage,
        )
        schedule = self._normalize_dmrg_schedule(self.D, nsweeps, D_schedule=D_schedule, nsweeps_schedule=nsweeps_schedule)
        t0 = time.time()
        current_guess = mps0
        all_stage_history = []
        completed_sweeps = 0
        completed_half_sweeps = 0
        for stage_idx, (stage_D, stage_sweeps) in enumerate(schedule, start=1):
            if len(schedule) == 1:
                self._log(f"  Starting Complete Sweeps (D={stage_D})...")
            else:
                self._log(
                    f"  Starting Complete Sweeps Stage {stage_idx}/{len(schedule)} "
                    f"(D={stage_D}, complete_sweeps={stage_sweeps})..."
                )
            dmrg = TensorDMRG(
                final_H,
                D=stage_D,
                nsweeps=2 * stage_sweeps,
                init_guess=current_guess,
                symmetry=use_symmetry,
                target_qn=target_qn,
                sym_mgr=self.sym_mgr,
                not_conv_err=False,
                nstates=self.nstates,
                weights=self.weights,
                verbose=self.verbose,
                sweep_tol=sweep_tol,
                davidson_tol=davidson_tol,
                davidson_max_iter=davidson_max_iter,
                noise=noise,
                noise_decay=noise_decay,
                noise_cutoff=noise_cutoff,
                local_dense_max_dim=local_dense_max_dim,
                complementary_operator_families=getattr(self, "complementary_operators", None),
                complementary_operator_mpos=final_complementary_mpos,
                complementary_operator_term_maps=final_complementary_term_maps,
                complementary_operator_generator_entries=final_complementary_generator_entries,
                site_qn_maps=site_qn_maps if use_symmetry else None,
                performance=dmrg_performance,
                abelian_matvec_options=abelian_matvec_options,
                final_expectation=(
                    True if final_expectation is None else bool(final_expectation)
                ),
                converge_on_full_sweeps=True,
            )
            if self._active_integral_build_info is not None:
                self._active_integral_build_info[
                    "resolved_abelian_matvec_options"
                ] = _metadata_abelian_options(
                    getattr(dmrg, "abelian_matvec_options", {}) or {}
                )
            dmrg.run()
            half_rows = [
                row
                for row in dmrg.sweep_history
                if row.get("direction") in {"lr", "rl"}
            ]
            for half_index, row in enumerate(half_rows):
                row["half_sweep"] = half_index + 1
                row["stage_sweep"] = half_index // 2 + 1
                row["sweep"] = completed_sweeps + row["stage_sweep"]
                row["sweep_complete"] = row.get("direction") == "rl"
                row["stage"] = stage_idx
            stage_completed_half_sweeps = len(half_rows)
            stage_completed_sweeps = stage_completed_half_sweeps // 2
            for row in dmrg.sweep_history:
                if row.get("direction") not in {"lr", "rl"}:
                    row["half_sweep"] = None
                    row["stage_sweep"] = stage_completed_sweeps
                    row["sweep"] = completed_sweeps + stage_completed_sweeps
                    row["sweep_complete"] = False
                    row["stage"] = stage_idx
            if not half_rows and dmrg.converged and dmrg.sweep_history:
                stage_completed_sweeps = 1
                dmrg.sweep_history[-1]["stage_sweep"] = 1
                dmrg.sweep_history[-1]["sweep"] = completed_sweeps + 1
                dmrg.sweep_history[-1]["sweep_complete"] = True
                dmrg.sweep_history[-1]["stage"] = stage_idx
            completed_sweeps += stage_completed_sweeps
            completed_half_sweeps += stage_completed_half_sweeps
            all_stage_history.extend(dmrg.sweep_history)
            dmrg.ncompleted = completed_sweeps
            dmrg.ncompleted_half_sweeps = completed_half_sweeps
            dmrg.max_sweeps = int(nsweeps)
            dmrg.success = bool(dmrg.converged)
            dmrg.message = (
                "converged"
                if dmrg.converged
                else "completed requested DMRG complete sweeps without convergence"
            )
            current_guess = dmrg.ground_state.copy()
        dmrg.sweep_history = all_stage_history
        self.dmrg = dmrg
        # Report
        e_dmrg_total = dmrg.e_tot + self.e_core
        if self.spin_purification:
            compute_s2 = True
        s2_val = self.calc_spin_square() if compute_s2 else None
        if self.spin_purification:
            e_dmrg_total -= self.shift * s2_val
        self.e_tot = e_dmrg_total
        self.energies = np.asarray(e_dmrg_total, dtype=float).reshape(-1)
        self.energy = float(self.energies[0])
        self.state_average_energy = float(np.dot(self.weights, self.energies))
        self.ground_state = dmrg.ground_state
        self.states = dmrg.states
        self.history = dmrg.sweep_history
        self.ncompleted = dmrg.ncompleted
        self.ncompleted_half_sweeps = dmrg.ncompleted_half_sweeps
        self.converged = bool(dmrg.converged)
        self.success = bool(dmrg.success)
        self.message = str(dmrg.message)
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
        if require_convergence and not dmrg.converged:
            raise RuntimeError(
                "DMRG did not converge within "
                f"{complete_sweep_limit}. Increase nsweeps or D, loosen "
                "sweep_tol, or pass require_convergence=False for a "
                "forced-sweep diagnostic run."
            )
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
        for i in range(self.ncas):
            if self.site == "spatial":
                rho = rdms[i]
                if rho.shape[0] < 4:
                    print("  [Warning] Spatial-site symmetry check requires canonical d=4 sites.")
                    return
                n_up = (rho[1, 1] + rho[3, 3]).real
                n_dn = (rho[2, 2] + rho[3, 3]).real
            else:
                idx_up = 2 * i
                rho_up = rdms[idx_up]
                n_up = rho_up[1, 1].real
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

    def _fully_reduced_spatial_expectation(self, state, *, h1=None, eri=None):
        """Evaluate a spin-free operator directly in the reduced SU(2) MPS."""
        from pyqed.mps.nonabelian.environment import contract_chain_expectation
        from pyqed.mps.nonabelian.models import (
            build_spatial_one_body_reduced_mpo,
            build_spatial_spinfree_eri_mpo,
        )
        from pyqed.qchem.dmrg.backends.nonabelian import (
            _identity_mpo_factors_for_sites_and_mpo,
        )

        sites = state.sites
        if h1 is not None:
            mpo = build_spatial_one_body_reduced_mpo(sites, h1)
        elif eri is not None:
            mpo = build_spatial_spinfree_eri_mpo(
                sites,
                eri,
                include_half=False,
            )
        else:
            raise ValueError("A reduced one- or two-body operator is required.")
        numerator = contract_chain_expectation(sites, mpo)
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, mpo)
        denominator = contract_chain_expectation(sites, identity)
        if abs(denominator) <= 1.0e-14:
            raise ValueError("Cannot build RDMs from a zero-norm MPS.")
        return float(np.real(numerator / denominator))

    def _make_fully_reduced_spatial_rdm1(self, state_id=0, spatial=False, with_core=False):
        state = self._get_state_for_rdm(state_id)
        gamma = np.zeros((self.ncas, self.ncas), dtype=float)
        for p in range(self.ncas):
            for q in range(self.ncas):
                h1 = np.zeros((self.ncas, self.ncas), dtype=float)
                h1[p, q] = 1.0
                gamma[q, p] = self._fully_reduced_spatial_expectation(state, h1=h1)
        if spatial or with_core:
            out = gamma
        else:
            out = np.zeros((2 * self.ncas, 2 * self.ncas), dtype=float)
            for p in range(self.ncas):
                for q in range(self.ncas):
                    out[2 * p, 2 * q] = 0.5 * gamma[p, q]
                    out[2 * p + 1, 2 * q + 1] = 0.5 * gamma[p, q]
        if not with_core:
            return out
        norb = self.ncore + self.ncas
        embedded = np.zeros((norb, norb), dtype=float)
        np.fill_diagonal(embedded[:self.ncore, :self.ncore], 2.0)
        embedded[self.ncore:norb, self.ncore:norb] = gamma
        return embedded

    def _make_fully_reduced_spatial_rdm2(self, state_id=0, spatial=False, with_core=False):
        state = self._get_state_for_rdm(state_id)
        gamma2 = np.zeros((self.ncas,) * 4, dtype=float)
        for p in range(self.ncas):
            for q in range(self.ncas):
                for r in range(self.ncas):
                    for s in range(self.ncas):
                        eri = np.zeros((self.ncas,) * 4, dtype=float)
                        eri[p, q, r, s] = 1.0
                        gamma2[p, q, r, s] = self._fully_reduced_spatial_expectation(
                            state,
                            eri=eri,
                        )
        if not (spatial or with_core):
            raise NotImplementedError(
                "The fully reduced SU(2) backend exposes the spin-traced spatial 2-RDM only."
            )
        if not with_core:
            return gamma2
        ncore = self.ncore
        norb = ncore + self.ncas
        embedded = np.zeros((norb,) * 4, dtype=float)
        if ncore > 0:
            eye = np.eye(ncore)
            embedded[:ncore, :ncore, :ncore, :ncore] = (
                4 * np.einsum('ij,kl->ijkl', eye, eye)
                - 2 * np.einsum('ps,rq->pqrs', eye, eye)
            )
            dm1 = self._make_fully_reduced_spatial_rdm1(
                state_id,
                spatial=True,
                with_core=False,
            )
            for i in range(ncore):
                embedded[i, i, ncore:norb, ncore:norb] = 2 * dm1
                embedded[ncore:norb, ncore:norb, i, i] = 2 * dm1
                embedded[i, ncore:norb, i, ncore:norb] = -dm1
                embedded[ncore:norb, i, ncore:norb, i] = -dm1
        embedded[ncore:norb, ncore:norb, ncore:norb, ncore:norb] = gamma2
        return embedded

    def _make_spatial_site_rdm1(self, state_id=0, spatial=False, with_core=False):
        """1-RDM for the d=4 spatial-site backend."""
        if self.spatial_site_basis == "fully_reduced":
            return self._make_fully_reduced_spatial_rdm1(
                state_id,
                spatial=spatial,
                with_core=with_core,
            )
        state = self._get_state_for_rdm(state_id)
        mps_state = _spatial_rdm_dense_mps(
            state,
            site_qn_maps=self._dense_site_qn_maps(),
        )
        if mps_state is not None:
            norm = mps_state._mps_dot(mps_state, mps_state)
            if abs(norm) < 1e-14:
                raise ValueError("Cannot build RDMs from a zero-norm MPS.")
            holes = {
                (sigma, p): _apply_spatial_annihilation_mps(mps_state, sigma, p)
                for sigma in range(2)
                for p in range(self.ncas)
            }

            p_raw = np.zeros((2 * self.ncas, 2 * self.ncas), dtype=complex)
            for sigma in range(2):
                for p in range(self.ncas):
                    for q in range(self.ncas):
                        p_raw[2 * p + sigma, 2 * q + sigma] = (
                            holes[(sigma, p)]._mps_dot(holes[(sigma, p)], holes[(sigma, q)])
                            / norm
                        )
        else:
            psi = _nonabelian_mps_to_dense_vector(state)
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
        if self.spatial_site_basis == "fully_reduced":
            if idx_pairs is not None:
                raise NotImplementedError(
                    "idx_pairs is not implemented for the fully reduced SU(2) 2-RDM."
                )
            return self._make_fully_reduced_spatial_rdm2(
                state_id,
                spatial=spatial,
                with_core=with_core,
            )
        state = self._get_state_for_rdm(state_id)
        mps_state = _spatial_rdm_dense_mps(
            state,
            site_qn_maps=self._dense_site_qn_maps(),
        )
        if mps_state is not None:
            norm = mps_state._mps_dot(mps_state, mps_state)
            if abs(norm) < 1e-14:
                raise ValueError("Cannot build RDMs from a zero-norm MPS.")

            rdm2_algorithm = getattr(self, "spatial_rdm2_algorithm", "gram")
            if rdm2_algorithm not in {"gram", "direct", "npdm"}:
                raise ValueError("spatial_rdm2_algorithm must be 'gram', 'direct', or 'npdm'.")

            double_holes = {}
            if rdm2_algorithm == "gram":
                single_holes = {
                    (sigma, p): _apply_spatial_annihilation_mps(mps_state, sigma, p)
                    for sigma in range(2)
                    for p in range(self.ncas)
                }
                for sigma in range(2):
                    for p in range(self.ncas):
                        first_hole = single_holes[(sigma, p)]
                        for tau in range(2):
                            for r in range(self.ncas):
                                double_holes[(sigma, p, tau, r)] = _apply_spatial_annihilation_mps(
                                    first_hole,
                                    tau,
                                    r,
                                )
            npdm = _SpatialNPDMContractions(mps_state) if rdm2_algorithm == "npdm" else None
            if npdm is not None and abs(npdm.norm) < 1e-14:
                raise ValueError("Cannot build RDMs from a zero-norm MPS.")

            if spatial or with_core:
                g_out = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas), dtype=float)
                if rdm2_algorithm in {"direct", "npdm"}:
                    pairs = [(p, r) for p in range(self.ncas) for r in range(self.ncas)]
                    for sigma in range(2):
                        for tau in range(2):
                            for a, (p, r) in enumerate(pairs):
                                for b in range(a, len(pairs)):
                                    q, s = pairs[b]
                                    op_specs = [
                                        ("cre", sigma, p),
                                        ("cre", tau, r),
                                        ("ann", tau, s),
                                        ("ann", sigma, q),
                                    ]
                                    if rdm2_algorithm == "npdm":
                                        val = npdm.expect_string(op_specs).real
                                    else:
                                        val = _spatial_fermion_string_expectation_mps(
                                            mps_state,
                                            op_specs,
                                            norm,
                                        ).real
                                    g_out[p, q, r, s] += val
                                    if b != a:
                                        g_out[q, p, s, r] += val
                else:
                    for sigma in range(2):
                        for tau in range(2):
                            states = [
                                double_holes[(sigma, p, tau, r)]
                                for p in range(self.ncas)
                                for r in range(self.ncas)
                            ]
                            gram = _two_hole_gram_block(states, norm)
                            block = gram.reshape(self.ncas, self.ncas, self.ncas, self.ncas)
                            g_out += block.transpose(0, 2, 1, 3).real
            else:
                nspin = 2 * self.ncas
                g_out = np.zeros((nspin, nspin, nspin, nspin), dtype=complex)
                if rdm2_algorithm in {"direct", "npdm"}:
                    pairs = [(p, r) for p in range(self.ncas) for r in range(self.ncas)]
                    for sigma in range(2):
                        for tau in range(2):
                            for a, (p, r) in enumerate(pairs):
                                for b in range(a, len(pairs)):
                                    q, s = pairs[b]
                                    op_specs = [
                                        ("cre", sigma, p),
                                        ("cre", tau, r),
                                        ("ann", tau, s),
                                        ("ann", sigma, q),
                                    ]
                                    if rdm2_algorithm == "npdm":
                                        val = npdm.expect_string(op_specs)
                                    else:
                                        val = _spatial_fermion_string_expectation_mps(
                                            mps_state,
                                            op_specs,
                                            norm,
                                        )
                                    g_out[
                                        2 * p + sigma,
                                        2 * r + tau,
                                        2 * s + tau,
                                        2 * q + sigma,
                                    ] = val
                                    if b != a:
                                        g_out[
                                            2 * q + sigma,
                                            2 * s + tau,
                                            2 * r + tau,
                                            2 * p + sigma,
                                        ] = val.conjugate()
                else:
                    for sigma in range(2):
                        for tau in range(2):
                            states = [
                                double_holes[(sigma, p, tau, r)]
                                for p in range(self.ncas)
                                for r in range(self.ncas)
                            ]
                            block = _two_hole_gram_block(states, norm).reshape(
                                self.ncas,
                                self.ncas,
                                self.ncas,
                                self.ncas,
                            )
                            for p in range(self.ncas):
                                for q in range(self.ncas):
                                    for r in range(self.ncas):
                                        for s in range(self.ncas):
                                            g_out[
                                                2 * p + sigma,
                                                2 * r + tau,
                                                2 * s + tau,
                                                2 * q + sigma,
                                            ] = block[p, r, q, s]
        else:
            psi = _nonabelian_mps_to_dense_vector(state)
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
            dense_state = symmetric_to_dense(
                state,
                site_qn_maps=self._dense_site_qn_maps(),
            )
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
            dense_state = symmetric_to_dense(
                state,
                site_qn_maps=self._dense_site_qn_maps(),
            )
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
