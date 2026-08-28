from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorly.decomposition import tensor_train_matrix

from pyqed.mps.decompose import tt_to_tensor, compress as compress_mps_factors
from pyqed.mps.mps import MPO, symmetric_to_dense
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.mps.autompo.model import Model
from pyqed.qchem.mcscf.casci import (
    _as_state_ci_matrix,
    _factorized_ci_overlap,
    _normalize_active_electrons,
    _prepare_biorthogonal_overlap,
    _string_transform_matrix,
    _transform_ci_tensors_to_biorthogonal_basis,
    _unique_rows_first,
    get_combos,
)


@dataclass
class _OverlapState:
    binary: np.ndarray
    ci: np.ndarray
    ncore: int
    ncas: int
    mf: object
    mol: object


def _get_jw_term_robust(op_str_list, indices, factor):
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

    return Op(" ".join(final_ops_str), final_indices, factor=factor * ((-1) ** swaps) * extra_sign)


def _get_spin_chain_term_robust(op_str_list, orbital_indices, spin, factor):
    chain = list(zip(orbital_indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n - i - 1):
            if chain[j][0] > chain[j + 1][0]:
                chain[j], chain[j + 1] = chain[j + 1], chain[j]
                swaps += 1

    sorted_orbitals = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    for k in range(n):
        orbital = sorted_orbitals[k]
        op_sym = sorted_ops[k]
        if k > 0:
            prev_orbital = sorted_orbitals[k - 1]
            if parity % 2 == 1:
                for z_orbital in range(prev_orbital + 1, orbital):
                    final_indices.append(2 * z_orbital + spin)
                    final_ops_str.append("sigma_z")
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1
        final_indices.append(2 * orbital + spin)
        final_ops_str.append(op_sym)
        parity += 1

    return Op(" ".join(final_ops_str), final_indices, factor=factor * ((-1) ** swaps) * extra_sign)


def _dmrg_states(solver):
    if not hasattr(solver, "dmrg") or solver.dmrg.ground_state is None:
        raise ValueError("Run DMRG first to generate a state.")
    if hasattr(solver.dmrg, "states") and solver.dmrg.states is not None:
        return list(solver.dmrg.states)
    return [solver.dmrg.ground_state]


def _normalize_state_ids(state_ids, nstates):
    if state_ids is None:
        return list(range(nstates))
    if isinstance(state_ids, int):
        return [state_ids]
    return list(state_ids)


def _state_to_dense_tensor(state):
    dense_state = symmetric_to_dense(state) if hasattr(state.factors[0], "qns") else state
    std_state = dense_state if dense_state.labels == ["lv", "p", "rv"] else dense_state.to_order(["lv", "p", "rv"])
    return np.asarray(tt_to_tensor(std_state.factors))


def _state_to_dense_mps(state):
    dense_state = symmetric_to_dense(state) if hasattr(state.factors[0], "qns") else state
    return dense_state if dense_state.labels == ["lv", "p", "rv"] else dense_state.to_order(["lv", "p", "rv"])


def _reference_active_mo_occ(nelecas_spin, ncas):
    na, nb = nelecas_spin
    occ_a = np.zeros(ncas, dtype=np.int8)
    occ_b = np.zeros(ncas, dtype=np.int8)
    occ_a[:na] = 1
    occ_b[:nb] = 1
    return np.stack((occ_a, occ_b), axis=0)


def _spin_orbital_index(alpha_occ, beta_occ):
    idx = np.empty(alpha_occ.size * 2, dtype=np.int8)
    idx[0::2] = alpha_occ
    idx[1::2] = beta_occ
    return tuple(int(x) for x in idx)


def _interleaved_to_grouped_sign(det):
    alpha_occ = np.asarray(det[0], dtype=np.int8)
    beta_occ = np.asarray(det[1], dtype=np.int8)
    occupied_alpha = np.flatnonzero(alpha_occ)
    occupied_beta = np.flatnonzero(beta_occ)
    n_cross = 0
    for p in occupied_beta:
        n_cross += np.count_nonzero(occupied_alpha > p)
    return -1.0 if (n_cross % 2) else 1.0


def _dmrg_coefficients_in_cas_basis(solver, state_ids):
    states = _dmrg_states(solver)
    state_ids = _normalize_state_ids(state_ids, len(states))
    nelecas_spin = _normalize_active_electrons(solver.nelecas, solver.spin)
    mo_occ = _reference_active_mo_occ(nelecas_spin, solver.ncas)
    binary = np.asarray(get_combos(mo_occ, space="fci"), dtype=np.int8)

    coeffs = np.empty((len(state_ids), binary.shape[0]), dtype=complex)
    for row, state_id in enumerate(state_ids):
        tensor = _state_to_dense_tensor(states[state_id])
        if tensor.shape != (2,) * (2 * solver.ncas):
            raise ValueError(
                "Unexpected DMRG state tensor shape for active-space overlap: "
                f"{tensor.shape}; expected {(2,) * (2 * solver.ncas)}."
            )
        for col, det in enumerate(binary):
            coeffs[row, col] = _interleaved_to_grouped_sign(det) * tensor[_spin_orbital_index(det[0], det[1])]

    return binary, coeffs


def _select_ci_like_states(ci_obj, state_ids):
    ndet = ci_obj.binary.shape[0]
    if state_ids is None:
        ci = np.asarray(ci_obj.ci)
        if ci.ndim == 1:
            ci = ci.reshape(1, ndet)
        return _OverlapState(
            binary=np.asarray(ci_obj.binary, dtype=np.int8),
            ci=ci,
            ncore=ci_obj.ncore,
            ncas=ci_obj.ncas,
            mf=ci_obj.mf,
            mol=ci_obj.mol,
        )

    ids = _normalize_state_ids(state_ids, len(np.atleast_1d(ci_obj.e_tot)))
    ci = np.asarray(ci_obj.ci)
    if ci.ndim == 1:
        ci = ci.reshape(1, ndet)
    return _OverlapState(
        binary=np.asarray(ci_obj.binary, dtype=np.int8),
        ci=ci[ids],
        ncore=ci_obj.ncore,
        ncas=ci_obj.ncas,
        mf=ci_obj.mf,
        mol=ci_obj.mol,
    )


def _as_overlap_state(obj, state_ids):
    if hasattr(obj, "dmrg"):
        binary, coeffs = _dmrg_coefficients_in_cas_basis(obj, state_ids)
        return _OverlapState(
            binary=binary,
            ci=coeffs,
            ncore=obj.ncore,
            ncas=obj.ncas,
            mf=obj.mf,
            mol=obj.mol,
        )
    if hasattr(obj, "binary") and hasattr(obj, "ci"):
        return _select_ci_like_states(obj, state_ids)
    raise TypeError(f"Unsupported overlap object type: {type(obj)!r}")


def _compute_ao_overlap_matrix(bra, ket):
    from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix

    return _cross_ao_overlap_matrix(bra.mol, ket.mol)


def _compute_full_mo_overlap(bra, ket, s=None):
    if s is None:
        s = _compute_ao_overlap_matrix(bra, ket)
    return np.linalg.multi_dot((bra.mo_coeff.T, s, ket.mo_coeff))


def _active_and_core_overlaps(bra, ket, s=None):
    full = _compute_full_mo_overlap(bra, ket, s=s)
    bra_active = slice(bra.ncore, bra.ncore + bra.ncas)
    ket_active = slice(ket.ncore, ket.ncore + ket.ncas)
    scc = full[: bra.ncore, : ket.ncore]
    saa = full[bra_active, ket_active]
    return full, scc, saa


def _unitary_part(matrix):
    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def _is_unitary(matrix, tol):
    ident = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.max(np.abs(matrix.conj().T @ matrix - ident)) <= tol


def _is_identity(matrix, tol):
    ident = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.max(np.abs(matrix - ident)) <= tol


def _identity_mpo(nsites, phys_dim=2, dtype=complex):
    factors = []
    for _ in range(nsites):
        w = np.zeros((1, 1, phys_dim, phys_dim), dtype=dtype)
        for p in range(phys_dim):
            w[0, 0, p, p] = 1.0
        factors.append(w)
    return MPO(factors)


def _dense_mps_overlap(bra_mps, ket_mps):
    if bra_mps.L != ket_mps.L:
        raise ValueError(f"MPS length mismatch: {bra_mps.L} != {ket_mps.L}.")
    if bra_mps.dims != ket_mps.dims:
        raise ValueError(f"MPS physical dimensions mismatch: {bra_mps.dims} != {ket_mps.dims}.")
    c = np.einsum("aib,aic->bc", bra_mps.factors[0].conj(), ket_mps.factors[0])
    for i in range(1, bra_mps.L):
        c = np.einsum("aib,ac,cid->bd", bra_mps.factors[i].conj(), c, ket_mps.factors[i])
    return c[0, 0]


def _build_orbital_generator_mpo(kappa_spatial, cutoff=1e-12):
    ncas = kappa_spatial.shape[0]
    terms = []
    for p in range(ncas):
        for q in range(ncas):
            val = kappa_spatial[p, q]
            if abs(val) <= cutoff:
                continue
            terms.append(_get_jw_term_robust([r"a^\dagger", "a"], [2 * p, 2 * q], val))
            terms.append(_get_jw_term_robust([r"a^\dagger", "a"], [2 * p + 1, 2 * q + 1], val))
    if not terms:
        return _identity_mpo(2 * ncas)
    basis_sites = [BasisSimpleElectron(i) for i in range(2 * ncas)]
    model = Model(basis=basis_sites, ham_terms=terms)
    mpo = Mpo(model, algo="qr")
    return MPO([w.transpose(0, 3, 1, 2) for w in mpo.matrices])


def _single_spin_shear_mpo(nsites, src, dst, alpha):
    if abs(alpha) <= 1e-15:
        return _identity_mpo(nsites)
    spin = src % 2
    if dst % 2 != spin:
        raise ValueError("Single-spin shear MPO requires source and destination on the same spin chain.")
    term = _get_spin_chain_term_robust([r"a^\dagger", "a"], [src // 2, dst // 2], spin, alpha)
    basis_sites = [BasisSimpleElectron(i) for i in range(nsites)]
    model = Model(basis=basis_sites, ham_terms=[term])
    mpo = Mpo(model, algo="qr")
    return _identity_mpo(nsites, dtype=np.result_type(alpha, complex)) + MPO(
        [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
    )


def _shear_orbital_transform_mpo(ncas, row, col, alpha):
    nsites = 2 * ncas
    alpha_mpo = _single_spin_shear_mpo(nsites, 2 * row, 2 * col, alpha)
    beta_mpo = _single_spin_shear_mpo(nsites, 2 * row + 1, 2 * col + 1, alpha)
    return alpha_mpo @ beta_mpo


def _diagonal_orbital_transform_mpo(diag_vals):
    diag_vals = np.asarray(diag_vals, dtype=complex)
    factors = []
    for val in diag_vals:
        for _ in range(2):
            w = np.zeros((1, 1, 2, 2), dtype=diag_vals.dtype)
            w[0, 0, 0, 0] = 1.0
            w[0, 0, 1, 1] = val
            factors.append(w)
    return MPO(factors)


def _swap_orbital_transform_factors(i, j):
    if i == j:
        return []
    # Row swap via shears/scaling:
    # P_ij = D_j(-1) (I + E_ij) (I - E_ji) (I + E_ij)
    return [
        ("scale", j, -1.0 + 0.0j),
        ("shear", i, j, 1.0 + 0.0j),
        ("shear", j, i, -1.0 + 0.0j),
        ("shear", i, j, 1.0 + 0.0j),
    ]


def _factorize_inverse_transform(
    spatial_transform,
    *,
    pivot_tol=1e-12,
    value_tol=1e-12,
):
    matrix = np.linalg.inv(np.asarray(spatial_transform, dtype=complex))
    ncas = matrix.shape[0]
    current = matrix.copy()
    factors = []

    for k in range(ncas):
        pivot_row = k + int(np.argmax(np.abs(current[k:, k])))
        pivot_val = current[pivot_row, k]
        if abs(pivot_val) <= pivot_tol:
            raise np.linalg.LinAlgError("Orbital transform inverse is numerically singular in factorized MPO path.")

        if pivot_row != k:
            current[[k, pivot_row], :] = current[[pivot_row, k], :]
            factors.extend(_swap_orbital_transform_factors(k, pivot_row))

        pivot = current[k, k]
        if abs(pivot - 1.0) > value_tol:
            current[k, :] /= pivot
            factors.append(("scale", k, pivot))

        for i in range(ncas):
            if i == k:
                continue
            val = current[i, k]
            if abs(val) <= value_tol:
                continue
            current[i, :] -= val * current[k, :]
            factors.append(("shear", i, k, val))

    if not np.allclose(current, np.eye(ncas, dtype=complex), atol=1e-9, rtol=1e-9):
        raise np.linalg.LinAlgError("Failed to factorize orbital transform inverse accurately.")
    return factors


def _factorized_orbital_transform_mpo(spatial_transform, mpo_bond_dim=None):
    spatial_transform = np.asarray(spatial_transform, dtype=complex)
    ncas = spatial_transform.shape[0]
    factors = _factorize_inverse_transform(spatial_transform)
    mpo = _identity_mpo(2 * ncas, dtype=np.result_type(spatial_transform, complex))
    for factor in factors:
        kind = factor[0]
        if kind == "scale":
            _, orbital, value = factor
            diag = np.ones(ncas, dtype=complex)
            diag[orbital] = value
            piece = _diagonal_orbital_transform_mpo(diag)
        elif kind == "shear":
            _, row, col, value = factor
            piece = _shear_orbital_transform_mpo(ncas, row, col, value)
        else:
            raise ValueError(f"Unsupported factor type: {kind!r}")
        mpo = _mpo_product_preserve_operator(mpo, piece, chi_max=mpo_bond_dim)
    return mpo


def _dense_exact_orbital_transform_mpo(spatial_transform):
    """Exact MPO for a one-body orbital transform on the full Fock space.

    This builds the full spin-orbital Fock-space operator from determinant
    minors of ``spatial_transform^{-1}``, then factors it into an MPO exactly.
    It is only practical for small active spaces.
    """
    spatial_transform = np.asarray(spatial_transform, dtype=complex)
    ncas = spatial_transform.shape[0]
    nspin = 2 * ncas
    dim = 2 ** nspin
    transform_inv = np.linalg.inv(spatial_transform)

    dense = np.zeros((dim, dim), dtype=complex)
    for out_idx in range(dim):
        out_bits = np.array(np.unravel_index(out_idx, (2,) * nspin), dtype=np.int8)
        out_alpha = np.flatnonzero(out_bits[0::2])
        out_beta = np.flatnonzero(out_bits[1::2])
        for in_idx in range(dim):
            in_bits = np.array(np.unravel_index(in_idx, (2,) * nspin), dtype=np.int8)
            in_alpha = np.flatnonzero(in_bits[0::2])
            in_beta = np.flatnonzero(in_bits[1::2])
            if len(out_alpha) != len(in_alpha) or len(out_beta) != len(in_beta):
                continue
            val_alpha = (
                np.linalg.det(transform_inv[np.ix_(out_alpha, in_alpha)]) if len(out_alpha) > 0 else 1.0
            )
            val_beta = (
                np.linalg.det(transform_inv[np.ix_(out_beta, in_beta)]) if len(out_beta) > 0 else 1.0
            )
            dense[out_idx, in_idx] = val_alpha * val_beta

    tt = tensor_train_matrix(dense.reshape((2,) * nspin + (2,) * nspin), rank=dim)
    return MPO([np.asarray(core).transpose(0, 3, 1, 2) for core in tt.factors])


def _dense_exact_fock_operator(spatial_transform):
    spatial_transform = np.asarray(spatial_transform, dtype=complex)
    ncas = spatial_transform.shape[0]
    nspin = 2 * ncas
    dim = 2 ** nspin
    transform_inv = np.linalg.inv(spatial_transform)

    dense = np.zeros((dim, dim), dtype=complex)
    for out_idx in range(dim):
        out_bits = np.array(np.unravel_index(out_idx, (2,) * nspin), dtype=np.int8)
        out_alpha = np.flatnonzero(out_bits[0::2])
        out_beta = np.flatnonzero(out_bits[1::2])
        for in_idx in range(dim):
            in_bits = np.array(np.unravel_index(in_idx, (2,) * nspin), dtype=np.int8)
            in_alpha = np.flatnonzero(in_bits[0::2])
            in_beta = np.flatnonzero(in_bits[1::2])
            if len(out_alpha) != len(in_alpha) or len(out_beta) != len(in_beta):
                continue
            val_alpha = (
                np.linalg.det(transform_inv[np.ix_(out_alpha, in_alpha)]) if len(out_alpha) > 0 else 1.0
            )
            val_beta = (
                np.linalg.det(transform_inv[np.ix_(out_beta, in_beta)]) if len(out_beta) > 0 else 1.0
            )
            dense[out_idx, in_idx] = val_alpha * val_beta
    return dense


def _orbital_transform_mpo(
    spatial_transform,
    mpo_bond_dim=None,
    order=8,
    scale=2,
    *,
    enforce_unitary=False,
    dense_exact_max_spin_orbitals=8,
):
    spatial_transform = np.asarray(spatial_transform, dtype=complex)
    if spatial_transform.shape[0] != spatial_transform.shape[1]:
        raise ValueError(f"Orbital transform must be square, got {spatial_transform.shape}.")
    if 2 * spatial_transform.shape[0] <= dense_exact_max_spin_orbitals:
        return _dense_exact_orbital_transform_mpo(spatial_transform)
    return _factorized_orbital_transform_mpo(spatial_transform, mpo_bond_dim=mpo_bond_dim)


def _orbital_transform_mpo_method(spatial_transform, dense_exact_max_spin_orbitals=8):
    return (
        "dense_exact"
        if 2 * np.asarray(spatial_transform).shape[0] <= dense_exact_max_spin_orbitals
        else "factorized_fallback"
    )


def _unitary_rotation_mpo(spatial_transform, mpo_bond_dim=None, order=8, scale=2):
    return _orbital_transform_mpo(
        spatial_transform,
        mpo_bond_dim=mpo_bond_dim,
        order=order,
        scale=scale,
        enforce_unitary=True,
    )


def _state_overlap_matrix_from_dense_mps(bra_states, ket_states):
    out = np.empty((len(bra_states), len(ket_states)), dtype=complex)
    for i, bra_state in enumerate(bra_states):
        for j, ket_state in enumerate(ket_states):
            out[i, j] = _dense_mps_overlap(bra_state, ket_state)
    return out


def _phase_align_transformed_states(reference_states, transformed_states, tol=1e-14):
    aligned = []
    for ref_state, state in zip(reference_states, transformed_states):
        overlap = _dense_mps_overlap(ref_state, state)
        if abs(overlap) <= tol:
            aligned.append(state)
            continue
        phase = np.conj(overlap) / abs(overlap)
        shifted = state.copy()
        shifted.factors[0] = shifted.factors[0] * phase
        aligned.append(shifted)
    return aligned


def _compress_mps_preserve_norm(state, chi_max):
    compressed = compress_mps_factors([factor.copy() for factor in state.factors], chi_max, renormalize=False)
    return state.__class__(
        compressed, labels=["lv", "p", "rv"], sites=state.sites
    )


def _mpo_product_preserve_operator(left, right, chi_max=None):
    product = left @ right
    if chi_max is None:
        return product
    if max(product.bond_orders()) <= chi_max:
        return product
    return product.compress(chi_max)


def _finalize_transformed_states(reference_states, transformed_states, *, method, chi_max, phase_align_tol):
    if method == "dense_exact":
        compressed = [_compress_mps_preserve_norm(state, chi_max) for state in transformed_states]
        return compressed
    compressed = [state.compress(chi_max) for state in transformed_states]
    normalized = [state.normalize() for state in compressed]
    return _phase_align_transformed_states(reference_states, normalized, tol=phase_align_tol)


def _select_dmrg_dense_states(obj, state_ids):
    states = _dmrg_states(obj)
    ids = _normalize_state_ids(state_ids, len(states))
    return [_state_to_dense_mps(states[i]) for i in ids]


def _ci_tensors_from_overlap_state(state):
    ndet = state.binary.shape[0]
    alpha_strings = _unique_rows_first(state.binary[:, 0, :])
    beta_strings = _unique_rows_first(state.binary[:, 1, :])
    nalpha = len(alpha_strings)
    nbeta = len(beta_strings)
    if nalpha * nbeta != ndet:
        raise ValueError("Diagnostic path requires separable alpha/beta determinant grids.")
    ci = _as_state_ci_matrix(state.ci, ndet).reshape((-1, nalpha, nbeta))
    return ci, alpha_strings, beta_strings


def _transformed_ci_tensors_from_overlap_state(state, orbital_transform, dtype):
    ci, alpha_strings, beta_strings = _ci_tensors_from_overlap_state(state)
    g_alpha = _string_transform_matrix(orbital_transform, alpha_strings, dtype)
    g_beta = _string_transform_matrix(orbital_transform, beta_strings, dtype)
    return _transform_ci_tensors_to_biorthogonal_basis(ci, g_alpha, g_beta)


def _coefficients_from_dense_states(states, binary):
    coeffs = np.empty((len(states), binary.shape[0]), dtype=complex)
    for row, state in enumerate(states):
        tensor = _state_to_dense_tensor(state)
        for col, det in enumerate(binary):
            coeffs[row, col] = _interleaved_to_grouped_sign(det) * tensor[_spin_orbital_index(det[0], det[1])]
    return coeffs


def _coefficients_from_dense_tensors(tensors, binary):
    coeffs = np.empty((len(tensors), binary.shape[0]), dtype=complex)
    for row, tensor in enumerate(tensors):
        for col, det in enumerate(binary):
            coeffs[row, col] = _interleaved_to_grouped_sign(det) * tensor[_spin_orbital_index(det[0], det[1])]
    return coeffs


def _dense_states_to_ci_tensors(states, overlap_state):
    ci = _coefficients_from_dense_states(states, overlap_state.binary)
    nstates = len(states)
    ndet = overlap_state.binary.shape[0]
    alpha_strings = _unique_rows_first(overlap_state.binary[:, 0, :])
    beta_strings = _unique_rows_first(overlap_state.binary[:, 1, :])
    return ci.reshape((nstates, len(alpha_strings), len(beta_strings)))


def _dense_exact_transform_ci_tensors(states, overlap_state, spatial_transform):
    dense_op = _dense_exact_fock_operator(spatial_transform)
    nspin = 2 * spatial_transform.shape[0]
    transformed_tensors = []
    for state in states:
        tensor = _state_to_dense_tensor(state)
        transformed = dense_op @ tensor.reshape(-1)
        transformed_tensors.append(transformed.reshape((2,) * nspin))
    ci = _coefficients_from_dense_tensors(transformed_tensors, overlap_state.binary)
    nstates = len(states)
    alpha_strings = _unique_rows_first(overlap_state.binary[:, 0, :])
    beta_strings = _unique_rows_first(overlap_state.binary[:, 1, :])
    return ci.reshape((nstates, len(alpha_strings), len(beta_strings)))


def _best_fit_complex_scale(reference, candidate, tol=1e-14):
    ref = np.asarray(reference, dtype=complex).ravel()
    cand = np.asarray(candidate, dtype=complex).ravel()
    denom = np.vdot(cand, cand)
    if abs(denom) <= tol:
        return 1.0 + 0.0j
    return np.vdot(cand, ref) / denom


def _relative_tensor_error(reference, candidate, tol=1e-14):
    scale = _best_fit_complex_scale(reference, candidate, tol=tol)
    aligned = scale * np.asarray(candidate, dtype=complex)
    ref = np.asarray(reference, dtype=complex)
    ref_norm = np.linalg.norm(ref.ravel())
    if ref_norm <= tol:
        return 0.0, scale, aligned
    return float(np.linalg.norm((aligned - ref).ravel()) / ref_norm), scale, aligned


def overlap(bra, ket, bra_state_ids=None, ket_state_ids=None, s=None):
    """Overlap between active-space CI-like and DMRG/MPS-backed states.

    DMRG/QCDMRG states are converted exactly to active-space determinant
    amplitudes and then contracted with the shared CAS biorthogonal overlap.
    This is intended for small active spaces where full coefficient recovery is
    feasible.
    """
    bra_state = _as_overlap_state(bra, bra_state_ids)
    ket_state = _as_overlap_state(ket, ket_state_ids)
    return _factorized_ci_overlap(bra_state, ket_state, s=s)


def _structured_biorthogonal_overlap(
    bra,
    ket,
    bra_state_ids=None,
    ket_state_ids=None,
    s=None,
):
    bra_state = _as_overlap_state(bra, bra_state_ids)
    ket_state = _as_overlap_state(ket, ket_state_ids)
    s_mo = _compute_full_mo_overlap(bra, ket, s=s)
    dtype = np.dtype(np.result_type(s_mo, np.asarray(bra_state.ci), np.asarray(ket_state.ci), complex))
    prep = _prepare_biorthogonal_overlap(
        s_mo,
        bra.ncore,
        ket.ncore,
        bra.ncas,
        ket.ncas,
        dtype,
    )
    exact_bra_ci = _transformed_ci_tensors_from_overlap_state(bra_state, prep.x_left, dtype)
    exact_ket_ci = _transformed_ci_tensors_from_overlap_state(ket_state, prep.x_right, dtype)
    return prep.core_factor * np.einsum("Xab,Yab->XY", exact_bra_ci.conj(), exact_ket_ci)


def _mpo_biorthogonal_overlap(
    bra,
    ket,
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
    """Experimental nonunitary biorthogonal MPS overlap via MPO transforms.

    This uses the exact active-space biorthogonal preparation from CASCI and
    applies the resulting left/right nonunitary one-body transforms as MPOs to
    the bra and ket MPS states. It avoids the unitary polar approximation, but
    it is still an approximate tensor-network contraction once MPO bond
    compression is required. The large-system fallback now factorizes the
    orbital transform into exact shears and diagonal scalings instead of using
    a single Taylor-expanded MPO exponential.
    """
    if not hasattr(bra, "dmrg") or not hasattr(ket, "dmrg"):
        raise TypeError("biorthogonal_overlap currently supports QCDMRG/DMRG-backed objects only.")
    if bra.ncas != ket.ncas or bra.ncore != ket.ncore:
        raise ValueError(
            "biorthogonal_overlap requires matching active-space definitions: "
            f"(ncore, ncas)=({bra.ncore}, {bra.ncas}) vs ({ket.ncore}, {ket.ncas})."
        )

    s_mo = _compute_full_mo_overlap(bra, ket, s=s)
    prep = _prepare_biorthogonal_overlap(
        s_mo,
        bra.ncore,
        ket.ncore,
        bra.ncas,
        ket.ncas,
        np.dtype(np.result_type(s_mo, complex)),
    )

    bra_dense = _select_dmrg_dense_states(bra, bra_state_ids)
    ket_dense = _select_dmrg_dense_states(ket, ket_state_ids)
    bra_state = _as_overlap_state(bra, bra_state_ids)
    ket_state = _as_overlap_state(ket, ket_state_ids)
    if chi_max is None:
        max_bond = 1
        for state in bra_dense + ket_dense:
            max_bond = max(max_bond, max(state.bond_orders()))
        chi_max = max(32, 4 * max_bond)
    if mpo_bond_dim is None:
        mpo_bond_dim = chi_max

    if _is_identity(np.asarray(prep.saa_eff, dtype=complex), identity_tol):
        return prep.core_factor * _state_overlap_matrix_from_dense_mps(bra_dense, ket_dense)

    left_method = _orbital_transform_mpo_method(prep.x_left)
    right_method = _orbital_transform_mpo_method(prep.x_right)
    if left_method == "dense_exact" and right_method == "dense_exact":
        dtype = np.dtype(np.result_type(s_mo, np.asarray(bra_state.ci), np.asarray(ket_state.ci), complex))
        bra_ci = _transformed_ci_tensors_from_overlap_state(bra_state, prep.x_left, dtype)
        ket_ci = _transformed_ci_tensors_from_overlap_state(ket_state, prep.x_right, dtype)
        return prep.core_factor * np.einsum("Xab,Yab->XY", bra_ci.conj(), ket_ci)

    left_mpo = _orbital_transform_mpo(
        prep.x_left,
        mpo_bond_dim=mpo_bond_dim,
        order=order,
        scale=scale,
        enforce_unitary=False,
    )
    right_mpo = _orbital_transform_mpo(
        prep.x_right,
        mpo_bond_dim=mpo_bond_dim,
        order=order,
        scale=scale,
        enforce_unitary=False,
    )

    transformed_bra = _finalize_transformed_states(
        bra_dense,
        [left_mpo @ state for state in bra_dense],
        method=left_method,
        chi_max=chi_max,
        phase_align_tol=phase_align_tol,
    )
    transformed_ket = _finalize_transformed_states(
        ket_dense,
        [right_mpo @ state for state in ket_dense],
        method=right_method,
        chi_max=chi_max,
        phase_align_tol=phase_align_tol,
    )
    return prep.core_factor * _state_overlap_matrix_from_dense_mps(transformed_bra, transformed_ket)


def biorthogonal_overlap(
    bra,
    ket,
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
    """Biorthogonal overlap for DMRG states.

    `backend="structured"` applies the exact determinant-space biorthogonal
    transform recovered from the DMRG coefficients and is the current correct
    path for small active spaces.

    `backend="mpo"` applies the nonunitary transforms as MPOs and remains
    experimental.
    """
    backend = backend.lower()
    if backend == "structured":
        return _structured_biorthogonal_overlap(
            bra,
            ket,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
        )
    if backend == "mpo":
        return _mpo_biorthogonal_overlap(
            bra,
            ket,
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
    raise ValueError(f"Unsupported biorthogonal overlap backend: {backend!r}.")


def biorthogonal_overlap_diagnostics(
    bra,
    ket,
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
    """Exact-oracle diagnostics for the nonunitary biorthogonal MPS overlap path.

    This is intended for small active spaces where DMRG states can be converted
    back to full determinant amplitudes. It compares the exact determinant-space
    biorthogonal transform with the MPO-transformed MPS coefficients.
    """
    if not hasattr(bra, "dmrg") or not hasattr(ket, "dmrg"):
        raise TypeError(
            "biorthogonal_overlap_diagnostics currently supports QCDMRG/DMRG-backed objects only."
        )

    bra_state = _as_overlap_state(bra, bra_state_ids)
    ket_state = _as_overlap_state(ket, ket_state_ids)
    s_mo = _compute_full_mo_overlap(bra, ket, s=s)
    dtype = np.dtype(np.result_type(s_mo, np.asarray(bra_state.ci), np.asarray(ket_state.ci), complex))
    prep = _prepare_biorthogonal_overlap(
        s_mo,
        bra.ncore,
        ket.ncore,
        bra.ncas,
        ket.ncas,
        dtype,
    )

    exact_bra_ci = _transformed_ci_tensors_from_overlap_state(bra_state, prep.x_left, dtype)
    exact_ket_ci = _transformed_ci_tensors_from_overlap_state(ket_state, prep.x_right, dtype)
    exact_overlap = prep.core_factor * np.einsum("Xab,Yab->XY", exact_bra_ci.conj(), exact_ket_ci)

    bra_dense = _select_dmrg_dense_states(bra, bra_state_ids)
    ket_dense = _select_dmrg_dense_states(ket, ket_state_ids)
    if chi_max is None:
        max_bond = 1
        for state in bra_dense + ket_dense:
            max_bond = max(max_bond, max(state.bond_orders()))
        chi_max = max(32, 4 * max_bond)
    if mpo_bond_dim is None:
        mpo_bond_dim = chi_max

    left_method = _orbital_transform_mpo_method(prep.x_left)
    right_method = _orbital_transform_mpo_method(prep.x_right)
    if _is_identity(np.asarray(prep.saa_eff, dtype=complex), identity_tol):
        transformed_bra = bra_dense
        transformed_ket = ket_dense
        mpo_bra_ci = _dense_states_to_ci_tensors(transformed_bra, bra_state)
        mpo_ket_ci = _dense_states_to_ci_tensors(transformed_ket, ket_state)
    elif left_method == "dense_exact" and right_method == "dense_exact":
        mpo_bra_ci = exact_bra_ci.copy()
        mpo_ket_ci = exact_ket_ci.copy()
        mpo_overlap_from_ci = prep.core_factor * np.einsum("Xab,Yab->XY", mpo_bra_ci.conj(), mpo_ket_ci)
        mpo_overlap_direct = mpo_overlap_from_ci.copy()
        bra_errors = []
        for i in range(exact_bra_ci.shape[0]):
            rel_err, fit_scale, _ = _relative_tensor_error(exact_bra_ci[i], mpo_bra_ci[i], tol=phase_align_tol)
            bra_errors.append({"state": i, "relative_error": rel_err, "best_fit_scale": fit_scale})
        ket_errors = []
        for i in range(exact_ket_ci.shape[0]):
            rel_err, fit_scale, _ = _relative_tensor_error(exact_ket_ci[i], mpo_ket_ci[i], tol=phase_align_tol)
            ket_errors.append({"state": i, "relative_error": rel_err, "best_fit_scale": fit_scale})
        return {
            "prep": prep,
            "exact_overlap": exact_overlap,
            "exact_bridge_overlap": overlap(bra, ket, bra_state_ids=bra_state_ids, ket_state_ids=ket_state_ids, s=s),
            "structured_overlap": _structured_biorthogonal_overlap(
                bra,
                ket,
                bra_state_ids=bra_state_ids,
                ket_state_ids=ket_state_ids,
                s=s,
            ),
            "mpo_overlap_from_ci": mpo_overlap_from_ci,
            "mpo_overlap_direct": mpo_overlap_direct,
            "bra_state_errors": bra_errors,
            "ket_state_errors": ket_errors,
            "left_mpo_method": left_method,
            "right_mpo_method": right_method,
            "chi_max": chi_max,
            "mpo_bond_dim": mpo_bond_dim,
            "order": order,
            "scale": scale,
        }
    else:
        left_mpo = _orbital_transform_mpo(
            prep.x_left,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
            enforce_unitary=False,
        )
        right_mpo = _orbital_transform_mpo(
            prep.x_right,
            mpo_bond_dim=mpo_bond_dim,
            order=order,
            scale=scale,
            enforce_unitary=False,
        )
        transformed_bra = _finalize_transformed_states(
            bra_dense,
            [left_mpo @ state for state in bra_dense],
            method=left_method,
            chi_max=chi_max,
            phase_align_tol=phase_align_tol,
        )
        transformed_ket = _finalize_transformed_states(
            ket_dense,
            [right_mpo @ state for state in ket_dense],
            method=right_method,
            chi_max=chi_max,
            phase_align_tol=phase_align_tol,
        )
        mpo_bra_ci = _dense_states_to_ci_tensors(transformed_bra, bra_state)
        mpo_ket_ci = _dense_states_to_ci_tensors(transformed_ket, ket_state)
    mpo_overlap_from_ci = prep.core_factor * np.einsum("Xab,Yab->XY", mpo_bra_ci.conj(), mpo_ket_ci)
    mpo_overlap_direct = (
        prep.core_factor * _state_overlap_matrix_from_dense_mps(transformed_bra, transformed_ket)
        if 'transformed_bra' in locals()
        else mpo_overlap_from_ci.copy()
    )

    bra_errors = []
    for i in range(exact_bra_ci.shape[0]):
        rel_err, fit_scale, _ = _relative_tensor_error(exact_bra_ci[i], mpo_bra_ci[i], tol=phase_align_tol)
        bra_errors.append({"state": i, "relative_error": rel_err, "best_fit_scale": fit_scale})

    ket_errors = []
    for i in range(exact_ket_ci.shape[0]):
        rel_err, fit_scale, _ = _relative_tensor_error(exact_ket_ci[i], mpo_ket_ci[i], tol=phase_align_tol)
        ket_errors.append({"state": i, "relative_error": rel_err, "best_fit_scale": fit_scale})

    return {
        "prep": prep,
        "exact_overlap": exact_overlap,
        "exact_bridge_overlap": overlap(bra, ket, bra_state_ids=bra_state_ids, ket_state_ids=ket_state_ids, s=s),
        "structured_overlap": _structured_biorthogonal_overlap(
            bra,
            ket,
            bra_state_ids=bra_state_ids,
            ket_state_ids=ket_state_ids,
            s=s,
        ),
        "mpo_overlap_from_ci": mpo_overlap_from_ci,
        "mpo_overlap_direct": mpo_overlap_direct,
        "bra_state_errors": bra_errors,
        "ket_state_errors": ket_errors,
        "left_mpo_method": _orbital_transform_mpo_method(prep.x_left),
        "right_mpo_method": _orbital_transform_mpo_method(prep.x_right),
        "chi_max": chi_max,
        "mpo_bond_dim": mpo_bond_dim,
        "order": order,
        "scale": scale,
    }


def unitary_overlap(
    bra,
    ket,
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
    """Approximate/scalable MPS overlap via a unitary active-space orbital transform.

    This path is exact when the active orbital overlap between ``bra`` and ``ket``
    is unitary (for example, a pure active-space orbital rotation at fixed AO
    basis). If ``use_polar=True``, the unitary polar factor of the active overlap
    is used as an approximation.
    """
    if not hasattr(bra, "dmrg") or not hasattr(ket, "dmrg"):
        raise TypeError("unitary_overlap currently supports QCDMRG/DMRG-backed objects only.")
    if bra.ncas != ket.ncas or bra.ncore != ket.ncore:
        raise ValueError(
            "unitary_overlap requires matching active-space definitions: "
            f"(ncore, ncas)=({bra.ncore}, {bra.ncas}) vs ({ket.ncore}, {ket.ncas})."
        )

    _, scc, saa = _active_and_core_overlaps(bra, ket, s=s)
    if orbital_transform is None:
        orbital_transform = _unitary_part(saa) if use_polar else np.asarray(saa, dtype=complex)
    else:
        orbital_transform = np.asarray(orbital_transform, dtype=complex)

    if not _is_unitary(orbital_transform, unitary_tol):
        raise ValueError(
            "Provided active-space orbital transform is not unitary within tolerance "
            f"{unitary_tol:.1e}. Use use_polar=True for a unitary approximation."
        )

    bra_dense = _select_dmrg_dense_states(bra, bra_state_ids)
    ket_dense = _select_dmrg_dense_states(ket, ket_state_ids)
    if chi_max is None:
        max_bond = 1
        for state in bra_dense + ket_dense:
            max_bond = max(max_bond, max(state.bond_orders()))
        chi_max = max(32, 4 * max_bond)
    if mpo_bond_dim is None:
        mpo_bond_dim = chi_max

    unitary_mpo = _unitary_rotation_mpo(
        orbital_transform,
        mpo_bond_dim=mpo_bond_dim,
        order=order,
        scale=scale,
    )
    method = _orbital_transform_mpo_method(orbital_transform)
    transformed_ket = _finalize_transformed_states(
        ket_dense,
        [unitary_mpo @ state for state in ket_dense],
        method=method,
        chi_max=chi_max,
        phase_align_tol=1e-14,
    )
    core_factor = np.linalg.det(scc) ** 2 if bra.ncore > 0 else 1.0
    return core_factor * _state_overlap_matrix_from_dense_mps(bra_dense, transformed_ket)


def automatic_overlap(
    bra,
    ket,
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
    """Automatic MPS overlap dispatcher with unitary/polar basis matching.

    If the active-space overlap is already unitary within ``unitary_tol``, this
    uses the exact unitary MPO path. Otherwise it falls back to the unitary
    polar factor of the active overlap as an approximation.
    """
    if not hasattr(bra, "dmrg") or not hasattr(ket, "dmrg"):
        raise TypeError("automatic_overlap currently supports QCDMRG/DMRG-backed objects only.")
    if bra.ncas != ket.ncas or bra.ncore != ket.ncore:
        raise ValueError(
            "automatic_overlap requires matching active-space definitions: "
            f"(ncore, ncas)=({bra.ncore}, {bra.ncas}) vs ({ket.ncore}, {ket.ncas})."
        )

    _, _, saa = _active_and_core_overlaps(bra, ket, s=s)
    is_unitary = _is_unitary(np.asarray(saa, dtype=complex), unitary_tol)
    overlap_matrix = unitary_overlap(
        bra,
        ket,
        bra_state_ids=bra_state_ids,
        ket_state_ids=ket_state_ids,
        s=s,
        use_polar=not is_unitary,
        unitary_tol=unitary_tol,
        chi_max=chi_max,
        mpo_bond_dim=mpo_bond_dim,
        order=order,
        scale=scale,
    )
    if return_info:
        return overlap_matrix, {
            "mode": "unitary" if is_unitary else "polar",
            "unitary_tol": unitary_tol,
            "active_unitarity_error": float(
                np.max(
                    np.abs(
                        np.asarray(saa, dtype=complex).conj().T @ np.asarray(saa, dtype=complex)
                        - np.eye(saa.shape[0], dtype=complex)
                    )
                )
            ),
        }
    return overlap_matrix
