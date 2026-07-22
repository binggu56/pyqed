"""Analytic terminal-response helpers for SU(2)-adapted qchem NARG."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from pyqed.narg.irrep_tensor import Irrep, IrrepTensor, OpIrrep
from pyqed.qchem.mcscf.orbopt import orbital_eri_response, orbital_h1_response

from .su2_rdm import build_su2_rdms
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    add_reduced_tensors,
    coupled_reduced_product,
    reduced_tensor_from_components,
    scale_reduced_tensor,
)


@dataclass(frozen=True)
class TerminalResponse:
    """First-order eigenvector response inside a retained NARG sector."""

    vector: np.ndarray
    energy: float
    first_order_energy: float
    root_index: int
    residual_norm: float
    min_gap: float


@dataclass(frozen=True)
class TruncationTangent:
    """First-order response of one SU2-NARG truncation map."""

    d_transform_blocks: dict[tuple, np.ndarray]
    d_hamiltonian: IrrepTensor
    root_energy_derivatives: dict[tuple, float]
    min_gap: float


@dataclass(frozen=True)
class TruncationBilinearTangent:
    """Mixed second response of one fixed-root SU2-NARG truncation map."""

    x: TruncationTangent
    y: TruncationTangent
    dxy_transform_blocks: dict[tuple, np.ndarray]
    dxy_hamiltonian: IrrepTensor
    root_energy_mixed_derivatives: dict[tuple, float]
    min_gap: float


@dataclass(frozen=True)
class RecursivePerturbation:
    """Recursive fixed-pattern SU2-NARG perturbation for one active Hamiltonian."""

    tensor: IrrepTensor
    block: np.ndarray
    min_gap: float
    block_count: int


@dataclass(frozen=True)
class RecursiveBilinearPerturbation:
    """Recursive fixed-pattern mixed perturbation for two active Hamiltonians."""

    tensor: IrrepTensor
    block: np.ndarray
    min_gap: float
    block_count: int


@dataclass(frozen=True)
class RecursiveActiveResponseBasis:
    """Recursive response blocks for independent symmetric active integrals."""

    h1_keys: tuple[tuple[int, int], ...]
    eri_keys: tuple[tuple[int, int, int, int], ...]
    blocks: np.ndarray
    min_gap: float
    block_count: int
    paths: tuple[RecursiveTangentPath, ...] | None = None
    h1_components: np.ndarray | None = None
    eri_components: np.ndarray | None = None
    build_seconds: float = 0.0
    worker_count: int = 1


@dataclass(frozen=True)
class RecursiveBilinearActiveAdjoint:
    """Adjoint mixed responses for a fixed second perturbation."""

    h1_keys: tuple[tuple[int, int], ...]
    eri_keys: tuple[tuple[int, int, int, int], ...]
    x_values: np.ndarray
    xy_values: np.ndarray
    min_gap: float
    block_count: int
    evaluation_count: int
    worker_count: int = 1


@dataclass(frozen=True)
class RecursiveTangentPath:
    """Cached first-order recursive path for one active perturbation."""

    sources: dict[int, object]
    grown_hamiltonians: dict[int, IrrepTensor]
    responses: dict[int, TruncationTangent]
    min_gap: float
    block_count: int
    pre_rotation_parts: dict[int, dict] | None = None


def active_integral_response(h1e, eri, kappa):
    """Return analytic active-space integral derivatives for ``C -> C exp(kappa)``."""
    return orbital_h1_response(h1e, kappa), orbital_eri_response(eri, kappa)


def _index_array(index, nmo: int) -> np.ndarray:
    if isinstance(index, slice):
        return np.arange(int(nmo), dtype=int)[index]
    arr = np.asarray(index, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _h1_pair_response_slice(h1_mo, p: int, q: int, rows, cols, value=1.0):
    rows = _index_array(rows, h1_mo.shape[0])
    cols = _index_array(cols, h1_mo.shape[0])
    out = np.zeros((rows.size, cols.size), dtype=np.result_type(h1_mo, value))
    value = float(value)
    if rows.size == 0 or cols.size == 0 or p == q or value == 0.0:
        return out

    q_cols = np.flatnonzero(cols == q)
    if q_cols.size:
        out[:, q_cols] += value * h1_mo[np.ix_(rows, [p])]
    p_cols = np.flatnonzero(cols == p)
    if p_cols.size:
        out[:, p_cols] -= value * h1_mo[np.ix_(rows, [q])]
    q_rows = np.flatnonzero(rows == q)
    if q_rows.size:
        out[q_rows, :] += value * h1_mo[np.ix_([p], cols)]
    p_rows = np.flatnonzero(rows == p)
    if p_rows.size:
        out[p_rows, :] -= value * h1_mo[np.ix_([q], cols)]
    return out


def _eri_pair_response_slice(eri_mo, p: int, q: int, i0, i1, i2, i3, value=1.0):
    nmo = eri_mo.shape[0]
    idx0 = _index_array(i0, nmo)
    idx1 = _index_array(i1, nmo)
    idx2 = _index_array(i2, nmo)
    idx3 = _index_array(i3, nmo)
    out = np.zeros(
        (idx0.size, idx1.size, idx2.size, idx3.size),
        dtype=np.result_type(eri_mo, value),
    )
    value = float(value)
    if p == q or value == 0.0:
        return out

    for pos, sign, source in (
        (np.flatnonzero(idx0 == q), 1.0, p),
        (np.flatnonzero(idx0 == p), -1.0, q),
    ):
        if pos.size:
            out[pos, :, :, :] += (
                value
                * sign
                * eri_mo[np.ix_([source], idx1, idx2, idx3)]
            )
    for pos, sign, source in (
        (np.flatnonzero(idx1 == q), 1.0, p),
        (np.flatnonzero(idx1 == p), -1.0, q),
    ):
        if pos.size:
            out[:, pos, :, :] += (
                value
                * sign
                * eri_mo[np.ix_(idx0, [source], idx2, idx3)]
            )
    for pos, sign, source in (
        (np.flatnonzero(idx2 == q), 1.0, p),
        (np.flatnonzero(idx2 == p), -1.0, q),
    ):
        if pos.size:
            out[:, :, pos, :] += (
                value
                * sign
                * eri_mo[np.ix_(idx0, idx1, [source], idx3)]
            )
    for pos, sign, source in (
        (np.flatnonzero(idx3 == q), 1.0, p),
        (np.flatnonzero(idx3 == p), -1.0, q),
    ):
        if pos.size:
            out[:, :, :, pos] += (
                value
                * sign
                * eri_mo[np.ix_(idx0, idx1, idx2, [source])]
            )
    return out


def cas_integral_response_from_pair(
    h1_mo,
    eri_mo,
    pair,
    *,
    ncore: int,
    ncas: int,
    value: float = 1.0,
):
    """Return effective-CAS integral derivatives for one anti-Hermitian pair.

    This is equivalent to ``cas_integral_response_from_full`` with a sparse
    generator satisfying ``kappa[p, q] = value`` and
    ``kappa[q, p] = -value``, but avoids forming the full four-index
    ``deri_full`` tensor.
    """
    ncore = int(ncore)
    ncas = int(ncas)
    h1_mo = np.asarray(h1_mo)
    eri_mo = np.asarray(eri_mo)
    nmo = h1_mo.shape[0]
    if h1_mo.shape != (nmo, nmo):
        raise ValueError("h1_mo must be square")
    if eri_mo.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("eri_mo is inconsistent with h1_mo")
    if ncore < 0 or ncas <= 0 or ncore + ncas > nmo:
        raise ValueError("invalid ncore/ncas for full-space integral response")
    p, q = (int(pair[0]), int(pair[1]))
    if p < 0 or p >= nmo or q < 0 or q >= nmo:
        raise ValueError("orbital pair contains an out-of-range MO index")

    active = np.arange(ncore, ncore + ncas, dtype=int)
    core = np.arange(ncore, dtype=int)
    dh1 = _h1_pair_response_slice(h1_mo, p, q, active, active, value=value)
    if ncore:
        d_aaii = np.diagonal(
            _eri_pair_response_slice(
                eri_mo,
                p,
                q,
                active,
                active,
                core,
                core,
                value=value,
            ),
            axis1=2,
            axis2=3,
        )
        d_aiia = np.diagonal(
            _eri_pair_response_slice(
                eri_mo,
                p,
                q,
                active,
                core,
                core,
                active,
                value=value,
            ),
            axis1=1,
            axis2=2,
        )
        dh1 += 2.0 * np.sum(d_aaii, axis=-1)
        dh1 -= np.sum(d_aiia, axis=-1)
    deri = _eri_pair_response_slice(
        eri_mo,
        p,
        q,
        active,
        active,
        active,
        active,
        value=value,
    )
    return dh1, deri


def cas_integral_response_from_pairs(h1_mo, eri_mo, pairs, coeffs, *, ncore: int, ncas: int):
    """Linear combination of effective-CAS responses for packed pair variables."""
    pairs = list(pairs)
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (len(pairs),):
        raise ValueError("coeffs must have one entry per pair")
    ncas = int(ncas)
    h1_mo = np.asarray(h1_mo)
    eri_mo = np.asarray(eri_mo)
    dh1 = np.zeros((ncas, ncas), dtype=np.result_type(h1_mo, coeffs))
    deri = np.zeros((ncas, ncas, ncas, ncas), dtype=np.result_type(eri_mo, coeffs))
    for pair, coeff in zip(pairs, coeffs):
        if coeff == 0.0:
            continue
        pair_dh1, pair_deri = cas_integral_response_from_pair(
            h1_mo,
            eri_mo,
            pair,
            ncore=ncore,
            ncas=ncas,
            value=float(coeff),
        )
        dh1 += pair_dh1
        deri += pair_deri
    return dh1, deri


def cas_integral_response_from_full(h1_mo, eri_mo, kappa, *, ncore: int, ncas: int):
    """Return effective-CAS integral derivatives for a full MO rotation.

    The scalar frozen-core energy derivative is intentionally omitted: it is an
    identity contribution to the active Hamiltonian and drops out of the
    projected terminal-response equation.
    """
    ncore = int(ncore)
    ncas = int(ncas)
    h1_mo = np.asarray(h1_mo)
    eri_mo = np.asarray(eri_mo)
    kappa = np.asarray(kappa)
    nmo = h1_mo.shape[0]
    if h1_mo.shape != (nmo, nmo) or kappa.shape != (nmo, nmo):
        raise ValueError("h1_mo and kappa must be square matrices of the same size")
    if eri_mo.shape != (nmo, nmo, nmo, nmo):
        raise ValueError("eri_mo is inconsistent with h1_mo")
    if ncore < 0 or ncas <= 0 or ncore + ncas > nmo:
        raise ValueError("invalid ncore/ncas for full-space integral response")

    dh1_full = orbital_h1_response(h1_mo, kappa)
    deri_full = orbital_eri_response(eri_mo, kappa)
    active = slice(ncore, ncore + ncas)
    dh1 = np.array(dh1_full[active, active], copy=True)
    for i in range(ncore):
        dh1 += 2.0 * deri_full[active, active, i, i]
        dh1 -= deri_full[active, i, i, active]
    deri = np.array(deri_full[active, active, active, active], copy=True)
    return dh1, deri


def _truncation_spectral_data(narg, truncated):
    cache = getattr(truncated, "_su2_response_spectral_cache", None)
    if cache is not None and cache.get("source_id") == id(narg):
        return cache["grouped"]

    grouped_roots = {}
    for root_index, root in enumerate(truncated.kept_roots):
        grouped_roots.setdefault(root.irrep, []).append((root_index, root))

    grouped = {}
    for irrep, roots in grouped_roots.items():
        h_block = np.asarray(narg.hamiltonian.block(irrep, irrep), dtype=complex)
        if h_block.size == 0:
            continue
        h_block = 0.5 * (h_block + h_block.conj().T)
        evals, evecs = np.linalg.eigh(h_block)
        kept_eig_indices = {int(root.local_index) for _, root in roots}
        aligned_vectors = []
        root_data = []
        for root_index, root in roots:
            eig_index = int(root.local_index)
            if eig_index < 0 or eig_index >= evals.size:
                raise IndexError("retained root local_index is outside its sector")
            u = evecs[:, eig_index].copy()
            overlap = np.vdot(u, root.vector)
            if abs(overlap) > 1.0e-14:
                u *= overlap / abs(overlap)
            aligned_vectors.append(u)
            root_data.append((root_index, root, eig_index, u))
        grouped[irrep] = {
            "evals": evals,
            "evecs": evecs,
            "kept_eig_indices": kept_eig_indices,
            "kept_vectors": np.column_stack(aligned_vectors) if aligned_vectors else np.zeros((h_block.shape[0], 0), dtype=complex),
            "roots": root_data,
        }

    truncated._su2_response_spectral_cache = {
        "source_id": id(narg),
        "grouped": grouped,
    }
    return grouped


def truncation_tangent(
    narg,
    truncated,
    perturbation,
    *,
    gap_tol: float = 1.0e-10,
    include_retained_mixing: bool = True,
):
    r"""Differentiate a fixed-root SU2 truncation map analytically.

    For each retained root ``a`` in a sector,

    $$
    |\partial u_a\rangle =
    \sum_{b\ne a}|u_b\rangle
    {\langle u_b|\partial H|u_a\rangle\over E_a-E_b}.
    $$

    The kept root set is held fixed; this is the smooth tangent inside one
    truncation pattern, not a derivative through root reordering or D changes.
    """
    d_transform_blocks = {}
    d_h_blocks = {}
    d_energies = {}
    min_gap = np.inf

    grouped = _truncation_spectral_data(narg, truncated)
    for irrep, data in grouped.items():
        dh_block = np.asarray(perturbation.block(irrep, irrep), dtype=complex)
        dh_block = 0.5 * (dh_block + dh_block.conj().T)
        evals = data["evals"]
        evecs = data["evecs"]
        kept_eig_indices = data["kept_eig_indices"]
        d_columns = []
        d_diag = []
        for root_index, root, eig_index, u in data["roots"]:
            coeffs = evecs.conj().T @ (dh_block @ u)
            du = np.zeros_like(u, dtype=complex)
            for other_index, value in enumerate(evals):
                if other_index == eig_index:
                    continue
                if not include_retained_mixing and other_index in kept_eig_indices:
                    continue
                gap = float(evals[eig_index] - value)
                min_gap = min(min_gap, abs(gap))
                if abs(gap) <= gap_tol:
                    if abs(coeffs[other_index]) > 10.0 * gap_tol:
                        raise np.linalg.LinAlgError(
                            "truncation tangent is singular in a near-degenerate sector"
                        )
                    continue
                du += evecs[:, other_index] * coeffs[other_index] / gap
            du -= u * np.vdot(u, du)
            d_energy = float(np.real(np.vdot(u, dh_block @ u)))
            d_columns.append(du)
            d_diag.append(d_energy)
            d_energies[(irrep.charge, int(root.local_index), int(root_index))] = d_energy
        d_transform_blocks[(irrep, irrep)] = np.column_stack(d_columns)
        if include_retained_mixing:
            d_h_blocks[(irrep, irrep)] = np.diag(d_diag)
        else:
            kept_vectors = data["kept_vectors"]
            d_h_blocks[(irrep, irrep)] = (
                kept_vectors.conj().T @ dh_block @ kept_vectors
            )

    d_hamiltonian = IrrepTensor(
        truncated.site,
        truncated.site,
        truncated.hamiltonian.op,
        d_h_blocks,
    )
    return TruncationTangent(
        d_transform_blocks=d_transform_blocks,
        d_hamiltonian=d_hamiltonian,
        root_energy_derivatives=d_energies,
        min_gap=float(min_gap),
    )


def truncation_tangent_adjoint(
    narg,
    truncated,
    *,
    transform_adjoint_blocks=None,
    hamiltonian_adjoint: IrrepTensor | None = None,
    gap_tol: float = 1.0e-10,
    include_retained_mixing: bool = True,
) -> IrrepTensor:
    r"""Adjoint of :func:`truncation_tangent` with respect to ``perturbation``.

    This returns the source-sector tensor ``A`` satisfying

    $$
    \operatorname{Re}\langle A, \partial H\rangle
      =
    \operatorname{Re}\langle \bar U, \partial U\rangle
      + \operatorname{Re}\langle \bar H_t, \partial H_t\rangle ,
    $$

    for the fixed kept-root pattern used by ``truncation_tangent``.
    """
    transform_adjoint_blocks = transform_adjoint_blocks or {}
    out_blocks = {}
    grouped = _truncation_spectral_data(narg, truncated)
    for irrep, data in grouped.items():
        h_block = np.asarray(narg.hamiltonian.block(irrep, irrep), dtype=complex)
        if h_block.size == 0:
            continue
        grad = np.zeros_like(h_block, dtype=complex)
        evals = data["evals"]
        evecs = data["evecs"]
        kept_eig_indices = data["kept_eig_indices"]

        if hamiltonian_adjoint is not None:
            adj_block = np.asarray(
                hamiltonian_adjoint.block(irrep, irrep),
                dtype=complex,
            )
            if adj_block.size:
                if include_retained_mixing:
                    for col, (_root_index, _root, _eig_index, u) in enumerate(
                        data["roots"]
                    ):
                        if col < adj_block.shape[0] and col < adj_block.shape[1]:
                            grad += float(np.real(adj_block[col, col])) * np.outer(
                                u,
                                np.conjugate(u),
                            )
                else:
                    kept = data["kept_vectors"]
                    grad += kept @ adj_block @ kept.conj().T

        transform_adj = transform_adjoint_blocks.get((irrep, irrep))
        if transform_adj is not None:
            transform_adj = np.asarray(transform_adj, dtype=complex)
            for col, (_root_index, _root, eig_index, u) in enumerate(data["roots"]):
                if col >= transform_adj.shape[1]:
                    continue
                adj_col = transform_adj[:, col]
                for other_index, value in enumerate(evals):
                    if other_index == eig_index:
                        continue
                    if (
                        not include_retained_mixing
                        and other_index in kept_eig_indices
                    ):
                        continue
                    gap = float(evals[eig_index] - value)
                    if abs(gap) <= gap_tol:
                        continue
                    other = evecs[:, other_index]
                    alpha = np.vdot(adj_col, other) / gap
                    grad += np.conjugate(alpha) * np.outer(other, np.conjugate(u))

        grad = 0.5 * (grad + grad.conj().T)
        if np.any(np.abs(grad) > 1.0e-14):
            out_blocks[(irrep, irrep)] = grad

    return IrrepTensor(
        narg.hamiltonian.bra,
        narg.hamiltonian.ket,
        narg.hamiltonian.op,
        out_blocks,
    )


def _zero_truncation_tangent(narg, truncated) -> TruncationTangent:
    d_transform_blocks = {}
    root_energy_derivatives = {}
    for irrep, data in _truncation_spectral_data(narg, truncated).items():
        roots = data["roots"]
        if roots:
            rows = data["evecs"].shape[0]
            d_transform_blocks[(irrep, irrep)] = np.zeros(
                (rows, len(roots)),
                dtype=complex,
            )
        for root_index, root, _eig_index, _u in roots:
            root_energy_derivatives[
                (irrep.charge, int(root.local_index), int(root_index))
            ] = 0.0
    return TruncationTangent(
        d_transform_blocks=d_transform_blocks,
        d_hamiltonian=_zero_irrep_tensor_like(truncated.hamiltonian),
        root_energy_derivatives=root_energy_derivatives,
        min_gap=np.inf,
    )


def truncation_bilinear_tangent(
    narg,
    truncated,
    perturbation_x,
    perturbation_y,
    perturbation_xy,
    *,
    gap_tol: float = 1.0e-10,
    include_retained_mixing: bool = True,
    tangent_x: TruncationTangent | None = None,
    tangent_y: TruncationTangent | None = None,
):
    r"""Differentiate a fixed-root SU2 truncation map to mixed second order.

    The input ``perturbation_xy`` is the explicit mixed Hamiltonian derivative
    $\partial_x\partial_y H$ in the untruncated sector.  For each retained root
    $a$, the off-root component is

    $$
    |u_{xy}\rangle_m =
    {\langle m|H_{xy}|a\rangle
      + \langle m|(H_x-E_x)|u_y\rangle
      + \langle m|(H_y-E_y)|u_x\rangle
    \over E_a-E_m}.
    $$
    """

    tx = (
        tangent_x
        if tangent_x is not None
        else truncation_tangent(
            narg,
            truncated,
            perturbation_x,
            gap_tol=gap_tol,
            include_retained_mixing=include_retained_mixing,
        )
    )
    ty = (
        tangent_y
        if tangent_y is not None
        else truncation_tangent(
            narg,
            truncated,
            perturbation_y,
            gap_tol=gap_tol,
            include_retained_mixing=include_retained_mixing,
        )
    )

    dxy_transform_blocks = {}
    dxy_energies = {}
    min_gap = min(float(tx.min_gap), float(ty.min_gap))
    grouped = _truncation_spectral_data(narg, truncated)
    for irrep, data in grouped.items():
        h_block = np.asarray(narg.hamiltonian.block(irrep, irrep), dtype=complex)
        h_block = 0.5 * (h_block + h_block.conj().T)
        hx_block = np.asarray(perturbation_x.block(irrep, irrep), dtype=complex)
        hy_block = np.asarray(perturbation_y.block(irrep, irrep), dtype=complex)
        hxy_block = np.asarray(perturbation_xy.block(irrep, irrep), dtype=complex)
        hx_block = 0.5 * (hx_block + hx_block.conj().T)
        hy_block = 0.5 * (hy_block + hy_block.conj().T)
        hxy_block = 0.5 * (hxy_block + hxy_block.conj().T)

        evals = data["evals"]
        evecs = data["evecs"]
        kept_eig_indices = data["kept_eig_indices"]
        dx_cols = tx.d_transform_blocks.get((irrep, irrep))
        dy_cols = ty.d_transform_blocks.get((irrep, irrep))
        if dx_cols is None or dy_cols is None:
            continue
        dxy_cols = []
        for col, (root_index, root, eig_index, u) in enumerate(data["roots"]):
            ux = dx_cols[:, col]
            uy = dy_cols[:, col]
            root_key = (irrep.charge, int(root.local_index), int(root_index))
            ex = float(tx.root_energy_derivatives[root_key])
            ey = float(ty.root_energy_derivatives[root_key])
            rhs = (
                hxy_block @ u
                + (hx_block - ex * np.eye(hx_block.shape[0])) @ uy
                + (hy_block - ey * np.eye(hy_block.shape[0])) @ ux
            )
            coeffs = evecs.conj().T @ rhs
            uxy = np.zeros_like(u, dtype=complex)
            for other_index, value in enumerate(evals):
                if other_index == eig_index:
                    continue
                if not include_retained_mixing and other_index in kept_eig_indices:
                    continue
                gap = float(evals[eig_index] - value)
                min_gap = min(min_gap, abs(gap))
                if abs(gap) <= gap_tol:
                    if abs(coeffs[other_index]) > 10.0 * gap_tol:
                        raise np.linalg.LinAlgError(
                            "mixed truncation tangent is singular in a near-degenerate sector"
                        )
                    continue
                uxy += evecs[:, other_index] * coeffs[other_index] / gap
            uxy -= u * np.vdot(uy, ux)

            e_xy_x = np.vdot(u, hxy_block @ u) + 2.0 * np.real(
                np.vdot(uy, hx_block @ u)
            )
            e_xy = float(np.real(e_xy_x))
            dxy_cols.append(uxy)
            dxy_energies[root_key] = e_xy
        dxy_transform_blocks[(irrep, irrep)] = np.column_stack(dxy_cols)

    if include_retained_mixing:
        d_h_blocks = {}
        for irrep, data in grouped.items():
            diag = []
            for root_index, root, _eig_index, _u in data["roots"]:
                root_key = (irrep.charge, int(root.local_index), int(root_index))
                diag.append(dxy_energies[root_key])
            d_h_blocks[(irrep, irrep)] = np.diag(diag)
        dxy_hamiltonian = IrrepTensor(
            truncated.site,
            truncated.site,
            truncated.hamiltonian.op,
            d_h_blocks,
        )
    else:
        provisional = TruncationBilinearTangent(
            x=tx,
            y=ty,
            dxy_transform_blocks=dxy_transform_blocks,
            dxy_hamiltonian=truncated.hamiltonian,
            root_energy_mixed_derivatives=dxy_energies,
            min_gap=float(min_gap),
        )
        dxy_hamiltonian = rotate_irrep_tensor_bilinear(
            truncated,
            narg.hamiltonian,
            perturbation_x,
            perturbation_y,
            perturbation_xy,
            provisional,
        )

    return TruncationBilinearTangent(
        x=tx,
        y=ty,
        dxy_transform_blocks=dxy_transform_blocks,
        dxy_hamiltonian=dxy_hamiltonian,
        root_energy_mixed_derivatives=dxy_energies,
        min_gap=float(min_gap),
    )


def truncation_bilinear_tangent_adjoint_x(
    narg,
    truncated,
    response: TruncationBilinearTangent,
    perturbation_y,
    *,
    transform_x_adjoint_blocks=None,
    transform_xy_adjoint_blocks=None,
    hamiltonian_x_adjoint: IrrepTensor | None = None,
    hamiltonian_xy_adjoint: IrrepTensor | None = None,
    gap_tol: float = 1.0e-10,
    include_retained_mixing: bool = True,
) -> tuple[IrrepTensor, IrrepTensor]:
    r"""Adjoint of :func:`truncation_bilinear_tangent` wrt ``H_x`` and ``H_{xy}``.

    The ``y`` direction is held fixed.  This covers the retained-mixing branch
    used by recursive SU2-NARG growth, where the projected mixed Hamiltonian is
    represented by root-energy derivatives.
    """
    if not include_retained_mixing:
        raise NotImplementedError(
            "bilinear truncation adjoint currently supports retained mixing"
        )

    transform_x_adjoint_blocks = transform_x_adjoint_blocks or {}
    transform_xy_adjoint_blocks = transform_xy_adjoint_blocks or {}
    direct_x_blocks = {}
    direct_xy_blocks = {}
    extra_x_transform_adjoint = {}
    grouped = _truncation_spectral_data(narg, truncated)

    def add_matrix(target, key, value):
        if np.any(np.abs(value) > 1.0e-14):
            if key in target:
                target[key] = target[key] + value
            else:
                target[key] = value

    for irrep, data in grouped.items():
        h_block = np.asarray(narg.hamiltonian.block(irrep, irrep), dtype=complex)
        if h_block.size == 0:
            continue
        h_block = 0.5 * (h_block + h_block.conj().T)
        hy_block = np.asarray(perturbation_y.block(irrep, irrep), dtype=complex)
        hy_block = 0.5 * (hy_block + hy_block.conj().T)
        evals = data["evals"]
        evecs = data["evecs"]
        kept_eig_indices = data["kept_eig_indices"]
        roots = data["roots"]
        xy_transform_adj = transform_xy_adjoint_blocks.get((irrep, irrep))
        if xy_transform_adj is not None:
            xy_transform_adj = np.asarray(xy_transform_adj, dtype=complex)
            hx_grad = np.zeros_like(h_block, dtype=complex)
            hxy_grad = np.zeros_like(h_block, dtype=complex)
            ux_adj = np.zeros_like(xy_transform_adj, dtype=complex)
            for col, (_root_index, root, eig_index, u) in enumerate(roots):
                if col >= xy_transform_adj.shape[1]:
                    continue
                adj_col = xy_transform_adj[:, col]
                rhs_adj = np.zeros_like(u, dtype=complex)
                for other_index, value in enumerate(evals):
                    if other_index == eig_index:
                        continue
                    if (
                        not include_retained_mixing
                        and other_index in kept_eig_indices
                    ):
                        continue
                    gap = float(evals[eig_index] - value)
                    if abs(gap) <= gap_tol:
                        continue
                    other = evecs[:, other_index]
                    rhs_adj += other * (np.vdot(other, adj_col) / gap)

                uy_cols = response.y.d_transform_blocks.get((irrep, irrep))
                if uy_cols is None or col >= uy_cols.shape[1]:
                    continue
                uy = uy_cols[:, col]
                root_key = (irrep.charge, int(root.local_index), int(_root_index))
                ey = float(response.y.root_energy_derivatives[root_key])

                hxy_grad += np.outer(rhs_adj, np.conjugate(u))
                hx_grad += np.outer(rhs_adj, np.conjugate(uy))
                hx_grad -= float(np.real(np.vdot(rhs_adj, uy))) * np.outer(
                    u,
                    np.conjugate(u),
                )
                ux_adj[:, col] += (hy_block - ey * np.eye(hy_block.shape[0])).conj().T @ rhs_adj
                ux_adj[:, col] -= uy * np.vdot(u, adj_col)

            if np.any(np.abs(hx_grad) > 1.0e-14):
                hx_grad = 0.5 * (hx_grad + hx_grad.conj().T)
                add_matrix(direct_x_blocks, (irrep, irrep), hx_grad)
            if np.any(np.abs(hxy_grad) > 1.0e-14):
                hxy_grad = 0.5 * (hxy_grad + hxy_grad.conj().T)
                add_matrix(direct_xy_blocks, (irrep, irrep), hxy_grad)
            if np.any(np.abs(ux_adj) > 1.0e-14):
                add_matrix(extra_x_transform_adjoint, (irrep, irrep), ux_adj)

        if hamiltonian_xy_adjoint is not None:
            adj_block = np.asarray(
                hamiltonian_xy_adjoint.block(irrep, irrep),
                dtype=complex,
            )
            if adj_block.size:
                hx_grad = np.zeros_like(h_block, dtype=complex)
                hxy_grad = np.zeros_like(h_block, dtype=complex)
                uy_cols = response.y.d_transform_blocks.get((irrep, irrep))
                for col, (_root_index, _root, _eig_index, u) in enumerate(roots):
                    if col >= adj_block.shape[0] or col >= adj_block.shape[1]:
                        continue
                    coeff = float(np.real(adj_block[col, col]))
                    if coeff == 0.0:
                        continue
                    hxy_grad += coeff * np.outer(u, np.conjugate(u))
                    if uy_cols is not None and col < uy_cols.shape[1]:
                        uy = uy_cols[:, col]
                        hx_grad += 2.0 * coeff * np.outer(uy, np.conjugate(u))
                if np.any(np.abs(hx_grad) > 1.0e-14):
                    hx_grad = 0.5 * (hx_grad + hx_grad.conj().T)
                    add_matrix(direct_x_blocks, (irrep, irrep), hx_grad)
                if np.any(np.abs(hxy_grad) > 1.0e-14):
                    hxy_grad = 0.5 * (hxy_grad + hxy_grad.conj().T)
                    add_matrix(direct_xy_blocks, (irrep, irrep), hxy_grad)

    combined_x_transform_adjoint = dict(transform_x_adjoint_blocks)
    for key, value in extra_x_transform_adjoint.items():
        if key in combined_x_transform_adjoint:
            combined_x_transform_adjoint[key] = (
                combined_x_transform_adjoint[key] + value
            )
        else:
            combined_x_transform_adjoint[key] = value

    tangent_x_adjoint = truncation_tangent_adjoint(
        narg,
        truncated,
        transform_adjoint_blocks=combined_x_transform_adjoint,
        hamiltonian_adjoint=hamiltonian_x_adjoint,
        gap_tol=gap_tol,
        include_retained_mixing=include_retained_mixing,
    )
    for key, value in tangent_x_adjoint.blocks.items():
        add_matrix(direct_x_blocks, key, value)

    perturbation_x_adjoint = IrrepTensor(
        narg.hamiltonian.bra,
        narg.hamiltonian.ket,
        narg.hamiltonian.op,
        direct_x_blocks,
    )
    perturbation_xy_adjoint = IrrepTensor(
        narg.hamiltonian.bra,
        narg.hamiltonian.ket,
        narg.hamiltonian.op,
        direct_xy_blocks,
    )
    return perturbation_x_adjoint, perturbation_xy_adjoint


def rotate_irrep_tensor_tangent(
    truncated,
    operator,
    operator_tangent,
    truncation_response: TruncationTangent,
    *,
    atol: float = 1.0e-12,
):
    r"""Differentiate ``U^\dagger O U`` for one truncated IrrepTensor."""
    blocks = {}
    keys = set(operator.blocks) | set(operator_tangent.blocks)
    for bra_irrep, ket_irrep in keys:
        if bra_irrep not in truncated.site.dims or ket_irrep not in truncated.site.dims:
            continue
        if (
            truncated.source.site.sector_dim(bra_irrep) == 0
            or truncated.source.site.sector_dim(ket_irrep) == 0
        ):
            continue
        if not truncated.site.symmetry.allows(
            bra_irrep.charge,
            operator.op.charge,
            ket_irrep.charge,
        ):
            continue
        u_bra = truncated.transform.block(bra_irrep, bra_irrep)
        u_ket = truncated.transform.block(ket_irrep, ket_irrep)
        du_bra = truncation_response.d_transform_blocks.get(
            (bra_irrep, bra_irrep),
            np.zeros_like(u_bra),
        )
        du_ket = truncation_response.d_transform_blocks.get(
            (ket_irrep, ket_irrep),
            np.zeros_like(u_ket),
        )
        old_block = operator.block(bra_irrep, ket_irrep)
        old_tangent = operator_tangent.block(bra_irrep, ket_irrep)
        block = (
            du_bra.conj().T @ old_block @ u_ket
            + u_bra.conj().T @ old_tangent @ u_ket
            + u_bra.conj().T @ old_block @ du_ket
        )
        if np.any(np.abs(block) > atol):
            blocks[(bra_irrep, ket_irrep)] = block
    return IrrepTensor(truncated.site, truncated.site, operator.op, blocks)


def rotate_irrep_tensor_tangent_adjoint(
    truncated,
    operator,
    rotated_adjoint,
    *,
    atol: float = 1.0e-12,
):
    r"""Adjoint of the tangent rotation ``\partial(U^\dagger O U)``.

    Returns ``(operator_tangent_adjoint, transform_adjoint_blocks)`` for the
    linear map

    $$
    \partial O_t
      = \partial U^\dagger O U
        + U^\dagger \partial O\, U
        + U^\dagger O\, \partial U .
    $$
    """
    operator_blocks = {}
    transform_adjoint_blocks = {}

    for (bra_irrep, ket_irrep), adj_block in rotated_adjoint.blocks.items():
        if bra_irrep not in truncated.site.dims or ket_irrep not in truncated.site.dims:
            continue
        if (
            truncated.source.site.sector_dim(bra_irrep) == 0
            or truncated.source.site.sector_dim(ket_irrep) == 0
        ):
            continue
        if not truncated.site.symmetry.allows(
            bra_irrep.charge,
            operator.op.charge,
            ket_irrep.charge,
        ):
            continue

        adj_block = np.asarray(adj_block, dtype=complex)
        if adj_block.size == 0:
            continue
        u_bra = truncated.transform.block(bra_irrep, bra_irrep)
        u_ket = truncated.transform.block(ket_irrep, ket_irrep)
        old_block = operator.block(bra_irrep, ket_irrep)

        op_adj = u_bra @ adj_block @ u_ket.conj().T
        if np.any(np.abs(op_adj) > atol):
            key = (bra_irrep, ket_irrep)
            if key in operator_blocks:
                operator_blocks[key] = operator_blocks[key] + op_adj
            else:
                operator_blocks[key] = op_adj

        bra_adj = old_block @ u_ket @ adj_block.conj().T
        if np.any(np.abs(bra_adj) > atol):
            key = (bra_irrep, bra_irrep)
            if key in transform_adjoint_blocks:
                transform_adjoint_blocks[key] = (
                    transform_adjoint_blocks[key] + bra_adj
                )
            else:
                transform_adjoint_blocks[key] = bra_adj

        ket_adj = old_block.conj().T @ u_bra @ adj_block
        if np.any(np.abs(ket_adj) > atol):
            key = (ket_irrep, ket_irrep)
            if key in transform_adjoint_blocks:
                transform_adjoint_blocks[key] = (
                    transform_adjoint_blocks[key] + ket_adj
                )
            else:
                transform_adjoint_blocks[key] = ket_adj

    operator_adjoint = IrrepTensor(
        operator.bra,
        operator.ket,
        operator.op,
        operator_blocks,
    )
    return operator_adjoint, transform_adjoint_blocks


def _bilinear_rotation_pairs(truncated, truncation_response, op):
    cache_key = (id(truncated), tuple(op.charge))
    cache = getattr(truncation_response, "_su2_bilinear_rotation_pair_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(
            truncation_response,
            "_su2_bilinear_rotation_pair_cache",
            cache,
        )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    source_site = truncated.source.site
    target_site = truncated.site
    transform = truncated.transform
    pair_data = {}
    for bra_irrep in target_site.dims:
        if source_site.sector_dim(bra_irrep) == 0:
            continue
        u_bra = transform.block(bra_irrep, bra_irrep)
        ux_bra = truncation_response.x.d_transform_blocks.get((bra_irrep, bra_irrep))
        uy_bra = truncation_response.y.d_transform_blocks.get((bra_irrep, bra_irrep))
        uxy_bra = truncation_response.dxy_transform_blocks.get((bra_irrep, bra_irrep))
        for ket_irrep in target_site.dims:
            if source_site.sector_dim(ket_irrep) == 0:
                continue
            if not target_site.symmetry.allows(
                bra_irrep.charge,
                op.charge,
                ket_irrep.charge,
            ):
                continue
            u_ket = transform.block(ket_irrep, ket_irrep)
            ux_ket = truncation_response.x.d_transform_blocks.get(
                (ket_irrep, ket_irrep)
            )
            uy_ket = truncation_response.y.d_transform_blocks.get(
                (ket_irrep, ket_irrep)
            )
            uxy_ket = truncation_response.dxy_transform_blocks.get(
                (ket_irrep, ket_irrep)
            )
            pair_data[(bra_irrep, ket_irrep)] = (
                u_bra,
                u_ket,
                ux_bra,
                ux_ket,
                uy_bra,
                uy_ket,
                uxy_bra,
                uxy_ket,
            )
    cache[cache_key] = pair_data
    return pair_data


def rotate_irrep_tensor_bilinear(
    truncated,
    operator,
    operator_x,
    operator_y,
    operator_xy,
    truncation_response: TruncationBilinearTangent,
    *,
    atol: float = 1.0e-12,
):
    r"""Mixed derivative of ``U^\dagger O U`` for one truncated tensor."""
    blocks = {}
    pair_data = _bilinear_rotation_pairs(truncated, truncation_response, operator.op)
    keys = (
        set(operator.blocks)
        | set(operator_x.blocks)
        | set(operator_y.blocks)
        | set(operator_xy.blocks)
    )
    for key in keys:
        data = pair_data.get(key)
        if data is None:
            continue
        u_bra, u_ket, ux_bra, ux_ket, uy_bra, uy_ket, uxy_bra, uxy_ket = data
        old = operator.blocks.get(key)
        ox = operator_x.blocks.get(key)
        oy = operator_y.blocks.get(key)
        oxy = operator_xy.blocks.get(key)
        block = None

        def add_term(left, middle, right):
            nonlocal block
            term = left.conj().T @ middle @ right
            block = term if block is None else block + term

        if old is not None:
            if uxy_bra is not None:
                add_term(uxy_bra, old, u_ket)
            if uy_bra is not None and ux_ket is not None:
                add_term(uy_bra, old, ux_ket)
            if ux_bra is not None and uy_ket is not None:
                add_term(ux_bra, old, uy_ket)
            if uxy_ket is not None:
                add_term(u_bra, old, uxy_ket)
        if ox is not None:
            if uy_bra is not None:
                add_term(uy_bra, ox, u_ket)
            if uy_ket is not None:
                add_term(u_bra, ox, uy_ket)
        if oy is not None:
            if ux_bra is not None:
                add_term(ux_bra, oy, u_ket)
            if ux_ket is not None:
                add_term(u_bra, oy, ux_ket)
        if oxy is not None:
            add_term(u_bra, oxy, u_ket)
        if block is None:
            continue
        if np.any(np.abs(block) > atol):
            blocks[key] = block
    return IrrepTensor(truncated.site, truncated.site, operator.op, blocks)


def rotate_irrep_tensor_bilinear_adjoint_x(
    truncated,
    operator,
    operator_y,
    rotated_adjoint,
    truncation_response: TruncationBilinearTangent,
    *,
    atol: float = 1.0e-12,
):
    r"""Adjoint of the mixed rotation with respect to the ``x`` direction.

    ``operator``, ``operator_y``, ``U`` and ``U_y`` are fixed.  The returned
    tuple contains adjoints for ``operator_x``, ``operator_xy``, ``U_x`` and
    ``U_{xy}`` in the map implemented by :func:`rotate_irrep_tensor_bilinear`.
    """
    operator_x_blocks = {}
    operator_xy_blocks = {}
    transform_x_adjoint_blocks = {}
    transform_xy_adjoint_blocks = {}

    def add_block(target, key, block):
        if np.any(np.abs(block) > atol):
            if key in target:
                target[key] = target[key] + block
            else:
                target[key] = block

    pair_data = _bilinear_rotation_pairs(truncated, truncation_response, operator.op)
    for (bra_irrep, ket_irrep), adj_block in rotated_adjoint.blocks.items():
        data = pair_data.get((bra_irrep, ket_irrep))
        if data is None:
            continue
        adj_block = np.asarray(adj_block, dtype=complex)
        if adj_block.size == 0:
            continue

        u_bra, u_ket, ux_bra, ux_ket, uy_bra, uy_ket, uxy_bra, uxy_ket = data
        old = operator.blocks.get((bra_irrep, ket_irrep))
        oy = operator_y.blocks.get((bra_irrep, ket_irrep))

        if old is not None:
            if uxy_bra is not None:
                add_block(
                    transform_xy_adjoint_blocks,
                    (bra_irrep, bra_irrep),
                    old @ u_ket @ adj_block.conj().T,
                )
            if uy_bra is not None and ux_ket is not None:
                add_block(
                    transform_x_adjoint_blocks,
                    (ket_irrep, ket_irrep),
                    old.conj().T @ uy_bra @ adj_block,
                )
            if ux_bra is not None and uy_ket is not None:
                add_block(
                    transform_x_adjoint_blocks,
                    (bra_irrep, bra_irrep),
                    old @ uy_ket @ adj_block.conj().T,
                )
            if uxy_ket is not None:
                add_block(
                    transform_xy_adjoint_blocks,
                    (ket_irrep, ket_irrep),
                    old.conj().T @ u_bra @ adj_block,
                )

        if uy_bra is not None:
            add_block(
                operator_x_blocks,
                (bra_irrep, ket_irrep),
                uy_bra @ adj_block @ u_ket.conj().T,
            )
        if uy_ket is not None:
            add_block(
                operator_x_blocks,
                (bra_irrep, ket_irrep),
                u_bra @ adj_block @ uy_ket.conj().T,
            )

        if oy is not None:
            if ux_bra is not None:
                add_block(
                    transform_x_adjoint_blocks,
                    (bra_irrep, bra_irrep),
                    oy @ u_ket @ adj_block.conj().T,
                )
            if ux_ket is not None:
                add_block(
                    transform_x_adjoint_blocks,
                    (ket_irrep, ket_irrep),
                    oy.conj().T @ u_bra @ adj_block,
                )

        add_block(
            operator_xy_blocks,
            (bra_irrep, ket_irrep),
            u_bra @ adj_block @ u_ket.conj().T,
        )

    operator_x_adjoint = IrrepTensor(
        operator.bra,
        operator.ket,
        operator.op,
        operator_x_blocks,
    )
    operator_xy_adjoint = IrrepTensor(
        operator.bra,
        operator.ket,
        operator.op,
        operator_xy_blocks,
    )
    return (
        operator_x_adjoint,
        operator_xy_adjoint,
        transform_x_adjoint_blocks,
        transform_xy_adjoint_blocks,
    )


def rotate_reduced_tensor_tangent(
    truncated,
    tensor: ReducedSU2Tensor,
    tensor_tangent: ReducedSU2Tensor,
    truncation_response: TruncationTangent,
    *,
    atol: float = 1.0e-12,
):
    """Differentiate projection of a reduced tensor through a truncation map."""
    return ReducedSU2Tensor(
        rotate_irrep_tensor_tangent(
            truncated,
            tensor.tensor,
            tensor_tangent.tensor,
            truncation_response,
            atol=atol,
        )
    )


def rotate_reduced_tensor_bilinear(
    truncated,
    tensor: ReducedSU2Tensor,
    tensor_x: ReducedSU2Tensor,
    tensor_y: ReducedSU2Tensor,
    tensor_xy: ReducedSU2Tensor,
    truncation_response: TruncationBilinearTangent,
    *,
    atol: float = 1.0e-12,
):
    """Mixed derivative of a reduced tensor through a truncation map."""
    return ReducedSU2Tensor(
        rotate_irrep_tensor_bilinear(
            truncated,
            tensor.tensor,
            tensor_x.tensor,
            tensor_y.tensor,
            tensor_xy.tensor,
            truncation_response,
            atol=atol,
        )
    )


def _zero_reduced_like(tensor: ReducedSU2Tensor) -> ReducedSU2Tensor:
    return ReducedSU2Tensor(IrrepTensor(tensor.site, tensor.site, tensor.op, {}))


def _zero_reduced(site, op) -> ReducedSU2Tensor:
    return ReducedSU2Tensor(IrrepTensor(site, site, op, {}))


def _zero_irrep_tensor_like(tensor: IrrepTensor) -> IrrepTensor:
    return IrrepTensor(tensor.bra, tensor.ket, tensor.op, {})


def _accumulate_bilinear_block_adjoint(
    entries,
    adjoint_matrix,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    *,
    prefactor: complex = 1.0,
) -> dict[tuple, np.ndarray]:
    """Adjoint of ``accumulate_bilinear_entries`` with respect to block tensor."""
    adjoint_matrix = np.asarray(adjoint_matrix, dtype=complex)
    out: dict[tuple, np.ndarray] = {}
    packed_groups = getattr(entries, "groups", None)
    if packed_groups is None:
        for (
            row,
            col,
            coeff,
            block_key,
            block_row,
            block_col,
            local_key,
            local_row,
            local_col,
        ) in entries:
            local_block = local_tensor.block(*local_key)
            if local_block.size == 0:
                continue
            block = out.get(block_key)
            if block is None:
                template = block_tensor.block(*block_key)
                block = np.zeros_like(template, dtype=complex)
                out[block_key] = block
            scale = np.conjugate(prefactor * coeff * local_block[local_row, local_col])
            block[block_row, block_col] += adjoint_matrix[row, col] * scale
        return out

    for group in packed_groups:
        local_block = local_tensor.block(*group.local_key)
        if local_block.size == 0:
            continue
        block = out.get(group.block_key)
        if block is None:
            template = block_tensor.block(*group.block_key)
            block = np.zeros_like(template, dtype=complex)
            out[group.block_key] = block
        scale = np.conjugate(
            prefactor
            * group.coeffs
            * local_block[group.local_rows, group.local_cols]
        )
        np.add.at(
            block,
            (group.block_rows, group.block_cols),
            adjoint_matrix[group.rows, group.cols] * scale,
        )
    return out


def reduced_scalar_product_block_adjoint(
    block,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    scalar_adjoint: IrrepTensor,
    *,
    prefactor: complex = 1.0,
) -> ReducedSU2Tensor:
    """Adjoint of ``reduced_scalar_product_irrep_tensor`` wrt block tensor."""
    from .su2_three_site import scalar_product_angular_terms

    _site, terms_by_irrep = scalar_product_angular_terms(
        block,
        block_tensor.op,
        local_tensor.op,
    )
    out_blocks: dict[tuple, np.ndarray] = {}
    for irrep, entries in terms_by_irrep.items():
        adj = scalar_adjoint.block(irrep, irrep)
        if adj.size == 0:
            continue
        contrib = _accumulate_bilinear_block_adjoint(
            entries,
            adj,
            block_tensor,
            local_tensor,
            prefactor=prefactor,
        )
        for key, value in contrib.items():
            if key in out_blocks:
                out_blocks[key] = out_blocks[key] + value
            else:
                out_blocks[key] = value
    out_blocks = {
        key: value
        for key, value in out_blocks.items()
        if np.any(np.abs(value) > 1.0e-14)
    }
    return ReducedSU2Tensor(
        IrrepTensor(block_tensor.site, block_tensor.site, block_tensor.op, out_blocks)
    )


def reduced_product_tensor_block_adjoint(
    block,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    product_adjoint: ReducedSU2Tensor,
    *,
    total_rank2: int,
) -> ReducedSU2Tensor:
    """Adjoint of ``reduced_product_tensor_irrep`` wrt the block tensor."""
    from .su2_three_site import product_tensor_angular_terms

    _site, _op, terms_by_pair = product_tensor_angular_terms(
        block,
        block_tensor.op,
        local_tensor.op,
        int(total_rank2),
    )
    out_blocks: dict[tuple, np.ndarray] = {}
    for pair_key, entries in terms_by_pair.items():
        adj = product_adjoint.block(*pair_key)
        if adj.size == 0:
            continue
        contrib = _accumulate_bilinear_block_adjoint(
            entries,
            adj,
            block_tensor,
            local_tensor,
        )
        for key, value in contrib.items():
            if key in out_blocks:
                out_blocks[key] = out_blocks[key] + value
            else:
                out_blocks[key] = value
    out_blocks = {
        key: value
        for key, value in out_blocks.items()
        if np.any(np.abs(value) > 1.0e-14)
    }
    return ReducedSU2Tensor(
        IrrepTensor(block_tensor.site, block_tensor.site, block_tensor.op, out_blocks)
    )


def coupled_reduced_product_adjoint(
    left: ReducedSU2Tensor,
    right: ReducedSU2Tensor,
    product_adjoint: ReducedSU2Tensor,
    *,
    rank2: int,
    scale: complex = 1.0,
    atol: float = 1.0e-12,
) -> tuple[ReducedSU2Tensor, ReducedSU2Tensor]:
    """Adjoint of ``coupled_reduced_product`` wrt both input tensors."""
    from .su2_reduced_tensor import _coupled_product_angular_terms, _site_charge_signature

    terms = _coupled_product_angular_terms(
        _site_charge_signature(left.site),
        tuple(int(x) for x in left.op.charge),
        tuple(int(x) for x in right.op.charge),
        int(rank2),
        float(atol),
    )
    left_blocks: dict[tuple, np.ndarray] = {}
    right_blocks: dict[tuple, np.ndarray] = {}
    for bra_charge, ket_charge, mid_terms in terms:
        bra = Irrep(bra_charge)
        ket = Irrep(ket_charge)
        adj = product_adjoint.block(bra, ket)
        if adj.size == 0:
            continue
        for mid_charge, weight in mid_terms:
            mid = Irrep(mid_charge)
            left_block = left.block(bra, mid)
            right_block = right.block(mid, ket)
            if left_block.size == 0 or right_block.size == 0:
                continue
            factor = np.conjugate(scale * weight)
            lkey = (bra, mid)
            lcontrib = factor * (adj @ right_block.conj().T)
            if lkey in left_blocks:
                left_blocks[lkey] = left_blocks[lkey] + lcontrib
            else:
                left_blocks[lkey] = lcontrib
            rkey = (mid, ket)
            rcontrib = factor * (left_block.conj().T @ adj)
            if rkey in right_blocks:
                right_blocks[rkey] = right_blocks[rkey] + rcontrib
            else:
                right_blocks[rkey] = rcontrib
    left_blocks = {
        key: value
        for key, value in left_blocks.items()
        if np.any(np.abs(value) > 1.0e-14)
    }
    right_blocks = {
        key: value
        for key, value in right_blocks.items()
        if np.any(np.abs(value) > 1.0e-14)
    }
    return (
        ReducedSU2Tensor(IrrepTensor(left.site, left.site, left.op, left_blocks)),
        ReducedSU2Tensor(IrrepTensor(right.site, right.site, right.op, right_blocks)),
    )


def direct_reduced_spinor_coupling_block_adjoint(
    block,
    block_spinor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    scalar_adjoint: IrrepTensor,
) -> ReducedSU2Tensor:
    r"""Adjoint wrt block spinor for ``S L + (S L)^\dagger`` couplings."""
    left = reduced_scalar_product_block_adjoint(
        block,
        block_spinor,
        local_tensor,
        scalar_adjoint,
        prefactor=np.sqrt(2.0),
    )
    right = reduced_scalar_product_block_adjoint(
        block,
        block_spinor,
        local_tensor,
        scalar_adjoint.adjoint(),
        prefactor=np.sqrt(2.0),
    )
    return add_reduced_tensors(left, right)


def direct_reduced_full_hamiltonian_tangent_adjoint(
    block,
    scalar_adjoint: IrrepTensor,
    *,
    site_index: int,
) -> tuple[IrrepTensor, dict[tuple, ReducedSU2Tensor]]:
    """Adjoint wrt source tangent data for ``direct_reduced_full_hamiltonian_tensor``.

    This covers the tangent-block use where future-site couplings are carried
    as ``Next*`` reduced operators on ``block``.
    """
    from .su2_three_site import (
        block_retained_scalar_tensor,
        local_reduced_operator,
        local_spin_density_tensor,
    )

    site_index = int(site_index)
    zero_h_blocks = {
        irrep: np.zeros(
            (
                block.truncated.site.sector_dim(irrep),
                block.truncated.site.sector_dim(irrep),
            ),
            dtype=complex,
        )
        for irrep in block.truncated.site.irreps
    }
    h_template = block_retained_scalar_tensor(block, zero_h_blocks)
    h_reduced_adjoint = reduced_scalar_product_block_adjoint(
        block,
        h_template,
        local_reduced_operator("I"),
        scalar_adjoint,
    )
    h_blocks = {}
    for (bra, ket), value in h_reduced_adjoint.blocks.items():
        if bra != ket:
            continue
        _, j2 = bra.charge
        h_blocks[(bra, ket)] = np.sqrt(j2 + 1.0) * value
    hamiltonian_adjoint = IrrepTensor(
        block.truncated.hamiltonian.bra,
        block.truncated.hamiltonian.ket,
        block.truncated.hamiltonian.op,
        h_blocks,
    )

    op_adjoint: dict[tuple, ReducedSU2Tensor] = {}

    def add_op(key, tensor):
        if not tensor.blocks:
            return
        if key in op_adjoint:
            op_adjoint[key] = add_reduced_tensors(op_adjoint[key], tensor)
        else:
            op_adjoint[key] = tensor

    key = ("NextDensity", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        add_op(
            key,
            reduced_scalar_product_block_adjoint(
                block,
                tensor,
                local_reduced_operator("Ntot"),
                scalar_adjoint,
            ),
        )

    key = ("NextExchangeDensity", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        add_op(
            key,
            scale_reduced_tensor(
                reduced_scalar_product_block_adjoint(
                    block,
                    scale_reduced_tensor(tensor, -0.5),
                    local_reduced_operator("Ntot"),
                    scalar_adjoint,
                ),
                -0.5,
            ),
        )

    key = ("NextExchangeSpinDensity", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        add_op(
            key,
            reduced_scalar_product_block_adjoint(
                block,
                tensor,
                local_spin_density_tensor(),
                scalar_adjoint,
                prefactor=np.sqrt(3.0),
            ),
        )

    key = ("NextPairAnnihilate", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        pair_adj = reduced_scalar_product_block_adjoint(
            block,
            tensor,
            local_reduced_operator("PairCreate"),
            scalar_adjoint,
        )
        pair_adj = add_reduced_tensors(
            pair_adj,
            reduced_scalar_product_block_adjoint(
                block,
                tensor,
                local_reduced_operator("PairCreate"),
                scalar_adjoint.adjoint(),
            ),
        )
        add_op(key, pair_adj)

    key = ("NextV1Spinor", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        add_op(
            key,
            direct_reduced_spinor_coupling_block_adjoint(
                block,
                tensor,
                local_reduced_operator("JWCtilde"),
                scalar_adjoint,
            ),
        )

    key = ("NextV3Cdag", site_index)
    tensor = block.reduced_operators.get(key)
    if tensor is not None:
        add_op(
            key,
            direct_reduced_spinor_coupling_block_adjoint(
                block,
                tensor,
                local_reduced_operator("JWDensityCtilde"),
                scalar_adjoint,
            ),
        )

    return hamiltonian_adjoint, op_adjoint


def _tensor_has_nonzero_blocks(tensor) -> bool:
    return bool(getattr(tensor, "blocks", None))


def _operator_dict_has_nonzero_blocks(operators) -> bool:
    return any(
        _tensor_has_nonzero_blocks(tensor)
        for tensor in (operators or {}).values()
    )


def _add_reduced_adjoint(target: dict, key, value) -> None:
    if value is None or not value.blocks:
        return
    if key in target:
        target[key] = add_reduced_tensors(target[key], value)
    else:
        target[key] = value


def _merge_reduced_adjoint(target: dict, source: dict) -> None:
    for key, value in (source or {}).items():
        _add_reduced_adjoint(target, key, value)


def _reduced_tensor_pairing(left: ReducedSU2Tensor, right: ReducedSU2Tensor) -> float:
    total = 0.0
    for key in set(left.blocks) | set(right.blocks):
        lblock = left.block(*key)
        rblock = right.block(*key)
        total += float(np.real(np.vdot(lblock, rblock)))
    return total


def _irrep_tensor_pairing(left: IrrepTensor, right: IrrepTensor) -> float:
    total = 0.0
    for key in set(left.blocks) | set(right.blocks):
        lblock = left.block(*key)
        rblock = right.block(*key)
        total += float(np.real(np.vdot(lblock, rblock)))
    return total


def _scale_adjoint(tensor: ReducedSU2Tensor, coeff) -> ReducedSU2Tensor:
    return scale_reduced_tensor(tensor, np.conjugate(coeff))


def _weighted_coefficients_have_support(
    dh1e,
    deri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool,
) -> bool:
    site_count = int(site_count)
    if site_count <= 0:
        return False
    if build_v1:
        for q in future_sites:
            q = int(q)
            if np.any(np.abs(dh1e[:site_count, q]) > 0.0):
                return True
    for q in future_sites:
        q = int(q)
        if np.any(np.abs(deri[:site_count, q, q, q]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, :site_count, q, q]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, q, q, :site_count]) > 0.0):
            return True
        if np.any(np.abs(deri[q, :site_count, q, :site_count]) > 0.0):
            return True
        if build_v1 and np.any(
            np.abs(deri[:site_count, q, :site_count, :site_count]) > 0.0
        ):
            return True
    return False


def _new_site_weighted_coefficients_have_support(
    deri,
    site_count: int,
    future_sites,
) -> bool:
    site_count = int(site_count)
    if site_count <= 0:
        return False
    new_site = site_count - 1
    for q in future_sites:
        q = int(q)
        if abs(deri[new_site, q, q, q]) > 0.0:
            return True
        if np.any(np.abs(deri[new_site, :site_count, q, q]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, new_site, q, q]) > 0.0):
            return True
        if np.any(np.abs(deri[new_site, q, q, :site_count]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, q, q, new_site]) > 0.0):
            return True
        if np.any(np.abs(deri[q, new_site, q, :site_count]) > 0.0):
            return True
        if np.any(np.abs(deri[q, :site_count, q, new_site]) > 0.0):
            return True
    return False


def _component_v1_coefficients_have_support(
    dh1e,
    deri,
    site_count: int,
    future_sites,
) -> bool:
    site_count = int(site_count)
    if site_count <= 0:
        return False
    for q in future_sites:
        q = int(q)
        if np.any(np.abs(dh1e[:site_count, q]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, q, :site_count, :site_count]) > 0.0):
            return True
    return False


def _grown_v1_coefficients_have_support(
    dh1e,
    deri,
    site_count: int,
    future_sites,
) -> bool:
    site_count = int(site_count)
    if site_count <= 0:
        return False
    new_site = site_count - 1
    for q in future_sites:
        q = int(q)
        if abs(dh1e[new_site, q]) > 0.0:
            return True
        if np.any(np.abs(deri[new_site, q, :site_count, :site_count]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, q, new_site, :site_count]) > 0.0):
            return True
        if np.any(np.abs(deri[:site_count, q, :site_count, new_site]) > 0.0):
            return True
    return False

def _add_optional_tensors(tensors):
    tensors = [tensor for tensor in tensors if tensor is not None and tensor.blocks]
    if not tensors:
        return None
    return add_reduced_tensors(*tensors)


def _weighted_tangent_terms(terms):
    pieces = []
    for tensor, dtensor, coeff, dcoeff in terms:
        if abs(dcoeff) > 0.0 and tensor.blocks:
            pieces.append(scale_reduced_tensor(tensor, dcoeff))
        if abs(coeff) > 0.0 and dtensor is not None and dtensor.blocks:
            pieces.append(scale_reduced_tensor(dtensor, coeff))
    return _add_optional_tensors(pieces)


def _weighted_bilinear_terms(terms):
    pieces = []
    for tensor, xtensor, ytensor, xytensor, coeff, xcoeff, ycoeff, xycoeff in terms:
        if abs(xycoeff) > 0.0 and tensor.blocks:
            pieces.append(scale_reduced_tensor(tensor, xycoeff))
        if abs(xcoeff) > 0.0 and ytensor is not None and ytensor.blocks:
            pieces.append(scale_reduced_tensor(ytensor, xcoeff))
        if abs(ycoeff) > 0.0 and xtensor is not None and xtensor.blocks:
            pieces.append(scale_reduced_tensor(xtensor, ycoeff))
        if abs(coeff) > 0.0 and xytensor is not None and xytensor.blocks:
            pieces.append(scale_reduced_tensor(xytensor, coeff))
    return _add_optional_tensors(pieces)


def _coupled_product_tangent(left, dleft, right, dright, rank2, *, scale=1.0):
    terms = []
    if dleft is not None and dleft.blocks and right.blocks:
        terms.append(coupled_reduced_product(dleft, right, rank2=rank2, scale=scale))
    if dright is not None and dright.blocks and left.blocks:
        terms.append(coupled_reduced_product(left, dright, rank2=rank2, scale=scale))
    return _add_optional_tensors(terms) or _zero_reduced(
        left.site,
        OpIrrep(
            (
                left.op.charge[0] + right.op.charge[0],
                int(rank2),
            )
        ),
    )


def _coupled_product_bilinear(
    left,
    xleft,
    yleft,
    xyleft,
    right,
    xright,
    yright,
    xyright,
    rank2,
    *,
    scale=1.0,
):
    terms = []
    if xyleft is not None and xyleft.blocks and right.blocks:
        terms.append(coupled_reduced_product(xyleft, right, rank2=rank2, scale=scale))
    if xleft is not None and xleft.blocks and yright is not None and yright.blocks:
        terms.append(coupled_reduced_product(xleft, yright, rank2=rank2, scale=scale))
    if yleft is not None and yleft.blocks and xright is not None and xright.blocks:
        terms.append(coupled_reduced_product(yleft, xright, rank2=rank2, scale=scale))
    if xyright is not None and xyright.blocks and left.blocks:
        terms.append(coupled_reduced_product(left, xyright, rank2=rank2, scale=scale))
    return _add_optional_tensors(terms) or _zero_reduced(
        left.site,
        OpIrrep(
            (
                left.op.charge[0] + right.op.charge[0],
                int(rank2),
            )
        ),
    )


def _coupled_product_bilinear_adjoint_x(
    left,
    xleft,
    yleft,
    xyleft,
    right,
    xright,
    yright,
    xyright,
    adjoint,
    rank2,
    *,
    scale=1.0,
):
    """Adjoint of ``_coupled_product_bilinear`` wrt the ``x`` and ``xy`` inputs."""
    left_x_adj = None
    right_x_adj = None
    left_xy_adj = None
    right_xy_adj = None

    def add_tensor(current, value):
        if value is None or not value.blocks:
            return current
        if current is None:
            return value
        return add_reduced_tensors(current, value)

    if xyleft is not None and xyleft.blocks and right.blocks:
        lhs_adj, _rhs_adj = coupled_reduced_product_adjoint(
            xyleft,
            right,
            adjoint,
            rank2=rank2,
            scale=scale,
        )
        left_xy_adj = add_tensor(left_xy_adj, lhs_adj)
    if xleft is not None and xleft.blocks and yright is not None and yright.blocks:
        lhs_adj, _rhs_adj = coupled_reduced_product_adjoint(
            xleft,
            yright,
            adjoint,
            rank2=rank2,
            scale=scale,
        )
        left_x_adj = add_tensor(left_x_adj, lhs_adj)
    if yleft is not None and yleft.blocks and xright is not None and xright.blocks:
        _lhs_adj, rhs_adj = coupled_reduced_product_adjoint(
            yleft,
            xright,
            adjoint,
            rank2=rank2,
            scale=scale,
        )
        right_x_adj = add_tensor(right_x_adj, rhs_adj)
    if xyright is not None and xyright.blocks and left.blocks:
        _lhs_adj, rhs_adj = coupled_reduced_product_adjoint(
            left,
            xyright,
            adjoint,
            rank2=rank2,
            scale=scale,
        )
        right_xy_adj = add_tensor(right_xy_adj, rhs_adj)

    return left_x_adj, right_x_adj, left_xy_adj, right_xy_adj


def _density_tangent(operators, doperators, cache, dcache, i, j):
    key = (int(i), int(j))
    if key in dcache:
        return dcache[key]
    op_key = ("Density", key[0], key[1])
    tensor = operators.get(op_key)
    direct = doperators.get(op_key)
    if tensor is not None and direct is not None:
        dcache[key] = direct
        cache[key] = tensor
        return direct
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=0,
            scale=np.sqrt(2.0),
        )
    dtensor = _coupled_product_tangent(
        operators[("Cdag", key[0])],
        doperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        doperators.get(("Ctilde", key[1])),
        0,
        scale=np.sqrt(2.0),
    )
    cache[key] = tensor
    dcache[key] = dtensor
    return dtensor


def _density_tangent_adjoint(operators, doperators, tangent_adjoint, i, j):
    key = (int(i), int(j))
    op_key = ("Density", key[0], key[1])
    tensor = operators.get(op_key)
    direct = doperators.get(op_key)
    out = {}
    if tensor is not None and direct is not None:
        _add_reduced_adjoint(out, op_key, tangent_adjoint)
        return out
    left = operators[("Cdag", key[0])]
    right = operators[("Ctilde", key[1])]
    left_adj, right_adj = coupled_reduced_product_adjoint(
        left,
        right,
        tangent_adjoint,
        rank2=0,
        scale=np.sqrt(2.0),
    )
    _add_reduced_adjoint(out, ("Cdag", key[0]), left_adj)
    _add_reduced_adjoint(out, ("Ctilde", key[1]), right_adj)
    return out


def _density_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    cache,
    xcache,
    ycache,
    xycache,
    i,
    j,
):
    key = (int(i), int(j))
    if key in xycache:
        return xycache[key]
    op_key = ("Density", key[0], key[1])
    tensor = operators.get(op_key)
    direct = xyoperators.get(op_key)
    if tensor is not None and direct is not None:
        xycache[key] = direct
        cache[key] = tensor
        return direct
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=0,
            scale=np.sqrt(2.0),
        )
    xtensor = _density_tangent(operators, xoperators, cache, xcache, key[0], key[1])
    ytensor = _density_tangent(operators, yoperators, cache, ycache, key[0], key[1])
    xytensor = _coupled_product_bilinear(
        operators[("Cdag", key[0])],
        xoperators.get(("Cdag", key[0])),
        yoperators.get(("Cdag", key[0])),
        xyoperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        0,
        scale=np.sqrt(2.0),
    )
    del xtensor, ytensor
    cache[key] = tensor
    xycache[key] = xytensor
    return xytensor


def _density_bilinear_adjoint_x(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    tangent_adjoint,
    i,
    j,
):
    key = (int(i), int(j))
    op_key = ("Density", key[0], key[1])
    tensor = operators.get(op_key)
    direct = xyoperators.get(op_key)
    out_x = {}
    out_xy = {}
    if tensor is not None and direct is not None:
        _add_reduced_adjoint(out_xy, op_key, tangent_adjoint)
        return out_x, out_xy
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=0,
            scale=np.sqrt(2.0),
        )
    left_x, right_x, left_xy, right_xy = _coupled_product_bilinear_adjoint_x(
        operators[("Cdag", key[0])],
        xoperators.get(("Cdag", key[0])),
        yoperators.get(("Cdag", key[0])),
        xyoperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        tangent_adjoint,
        0,
        scale=np.sqrt(2.0),
    )
    _add_reduced_adjoint(out_x, ("Cdag", key[0]), left_x)
    _add_reduced_adjoint(out_x, ("Ctilde", key[1]), right_x)
    _add_reduced_adjoint(out_xy, ("Cdag", key[0]), left_xy)
    _add_reduced_adjoint(out_xy, ("Ctilde", key[1]), right_xy)
    return out_x, out_xy


def _spin_density_tangent(operators, doperators, cache, dcache, i, j):
    key = (int(i), int(j))
    if key in dcache:
        return dcache[key]
    op_key = ("SpinDensity", key[0], key[1])
    tensor = operators.get(op_key)
    direct = doperators.get(op_key)
    if tensor is not None and direct is not None:
        dcache[key] = direct
        cache[key] = tensor
        return direct
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=2,
        )
    dtensor = _coupled_product_tangent(
        operators[("Cdag", key[0])],
        doperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        doperators.get(("Ctilde", key[1])),
        2,
    )
    cache[key] = tensor
    dcache[key] = dtensor
    return dtensor


def _spin_density_tangent_adjoint(operators, doperators, tangent_adjoint, i, j):
    key = (int(i), int(j))
    op_key = ("SpinDensity", key[0], key[1])
    tensor = operators.get(op_key)
    direct = doperators.get(op_key)
    out = {}
    if tensor is not None and direct is not None:
        _add_reduced_adjoint(out, op_key, tangent_adjoint)
        return out
    left = operators[("Cdag", key[0])]
    right = operators[("Ctilde", key[1])]
    left_adj, right_adj = coupled_reduced_product_adjoint(
        left,
        right,
        tangent_adjoint,
        rank2=2,
    )
    _add_reduced_adjoint(out, ("Cdag", key[0]), left_adj)
    _add_reduced_adjoint(out, ("Ctilde", key[1]), right_adj)
    return out


def _spin_density_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    cache,
    xcache,
    ycache,
    xycache,
    i,
    j,
):
    key = (int(i), int(j))
    if key in xycache:
        return xycache[key]
    op_key = ("SpinDensity", key[0], key[1])
    tensor = operators.get(op_key)
    direct = xyoperators.get(op_key)
    if tensor is not None and direct is not None:
        xycache[key] = direct
        cache[key] = tensor
        return direct
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=2,
        )
    _spin_density_tangent(operators, xoperators, cache, xcache, key[0], key[1])
    _spin_density_tangent(operators, yoperators, cache, ycache, key[0], key[1])
    xytensor = _coupled_product_bilinear(
        operators[("Cdag", key[0])],
        xoperators.get(("Cdag", key[0])),
        yoperators.get(("Cdag", key[0])),
        xyoperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        2,
    )
    cache[key] = tensor
    xycache[key] = xytensor
    return xytensor


def _spin_density_bilinear_adjoint_x(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    tangent_adjoint,
    i,
    j,
):
    key = (int(i), int(j))
    op_key = ("SpinDensity", key[0], key[1])
    tensor = operators.get(op_key)
    direct = xyoperators.get(op_key)
    out_x = {}
    out_xy = {}
    if tensor is not None and direct is not None:
        _add_reduced_adjoint(out_xy, op_key, tangent_adjoint)
        return out_x, out_xy
    if tensor is None:
        tensor = coupled_reduced_product(
            operators[("Cdag", key[0])],
            operators[("Ctilde", key[1])],
            rank2=2,
        )
    left_x, right_x, left_xy, right_xy = _coupled_product_bilinear_adjoint_x(
        operators[("Cdag", key[0])],
        xoperators.get(("Cdag", key[0])),
        yoperators.get(("Cdag", key[0])),
        xyoperators.get(("Cdag", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        tangent_adjoint,
        2,
    )
    _add_reduced_adjoint(out_x, ("Cdag", key[0]), left_x)
    _add_reduced_adjoint(out_x, ("Ctilde", key[1]), right_x)
    _add_reduced_adjoint(out_xy, ("Cdag", key[0]), left_xy)
    _add_reduced_adjoint(out_xy, ("Ctilde", key[1]), right_xy)
    return out_x, out_xy


def _pair_annihilate_tangent(operators, doperators, cache, dcache, i, j):
    key = (int(i), int(j))
    if key in dcache:
        return dcache[key]
    tensor = coupled_reduced_product(
        operators[("Ctilde", key[0])],
        operators[("Ctilde", key[1])],
        rank2=0,
        scale=-1.0 / np.sqrt(2.0),
    )
    dtensor = _coupled_product_tangent(
        operators[("Ctilde", key[0])],
        doperators.get(("Ctilde", key[0])),
        operators[("Ctilde", key[1])],
        doperators.get(("Ctilde", key[1])),
        0,
        scale=-1.0 / np.sqrt(2.0),
    )
    cache[key] = tensor
    dcache[key] = dtensor
    return dtensor


def _pair_annihilate_tangent_adjoint(operators, doperators, tangent_adjoint, i, j):
    key = (int(i), int(j))
    left = operators[("Ctilde", key[0])]
    right = operators[("Ctilde", key[1])]
    left_adj, right_adj = coupled_reduced_product_adjoint(
        left,
        right,
        tangent_adjoint,
        rank2=0,
        scale=-1.0 / np.sqrt(2.0),
    )
    out = {}
    _add_reduced_adjoint(out, ("Ctilde", key[0]), left_adj)
    _add_reduced_adjoint(out, ("Ctilde", key[1]), right_adj)
    return out


def _reduced_tensor_from_components_adjoint(
    multiplets,
    reduced_adjoint: ReducedSU2Tensor,
    q2_values,
):
    """Adjoint of ``reduced_tensor_from_components`` for selected components."""
    from .su2_core import cg
    from .su2_reduced_tensor import component_basis, group_multiplets, site_from_multiplets

    groups = group_multiplets(multiplets)
    site = site_from_multiplets(multiplets)
    basis_cache = {}

    def basis(irrep: Irrep, m2: int) -> np.ndarray:
        key = (irrep, int(m2))
        out = basis_cache.get(key)
        if out is None:
            out = component_basis(groups, irrep, int(m2))
            basis_cache[key] = out
        return out

    dim = 0
    for mp in multiplets:
        for vec in mp.states.values():
            dim = max(dim, int(np.asarray(vec).size))
    out = {int(q2): np.zeros((dim, dim), dtype=complex) for q2 in q2_values}
    if dim == 0 or not reduced_adjoint.blocks:
        return out

    _, rank2 = reduced_adjoint.op.charge
    for bra_irrep in site.irreps:
        _, bra_j2 = bra_irrep.charge
        for ket_irrep in site.irreps:
            _, ket_j2 = ket_irrep.charge
            adj_block = reduced_adjoint.blocks.get((bra_irrep, ket_irrep))
            if adj_block is None:
                continue

            pieces = []
            for q2 in out:
                for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
                    bra_m2 = ket_m2 + q2
                    if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                        continue
                    coeff = cg(ket_j2, ket_m2, rank2, q2, bra_j2, bra_m2)
                    if abs(coeff) <= 1.0e-12:
                        continue
                    pieces.append((q2, ket_m2, bra_m2, coeff))
            if not pieces:
                continue

            weight = 1.0 / float(len(pieces))
            norm = np.sqrt(bra_j2 + 1.0)
            for q2, ket_m2, bra_m2, coeff in pieces:
                bra_basis = basis(bra_irrep, bra_m2)
                ket_basis = basis(ket_irrep, ket_m2)
                scale = np.conjugate(weight * norm / coeff)
                out[q2] += scale * (bra_basis @ adj_block @ ket_basis.conj().T)
    return out


def _expanded_operator_from_reduced_adjoint(
    states,
    template: ReducedSU2Tensor,
    q2: int,
    component_adjoint: np.ndarray,
) -> ReducedSU2Tensor:
    """Adjoint of ``expanded_operator_from_reduced`` wrt reduced matrix blocks."""
    from .su2_core import cg

    blocks = {}
    if component_adjoint.size == 0:
        return _zero_reduced_like(template)

    dnelec, rank2 = template.op.charge
    grouped: dict[tuple[Irrep, int], list[tuple[int, int]]] = {}
    for pos, state in enumerate(states):
        grouped.setdefault((state.irrep, int(state.m2)), []).append(
            (pos, int(state.local_index))
        )

    for bra_irrep, ket_irrep in template.blocks:
        bra_nelec, bra_j2 = bra_irrep.charge
        ket_nelec, ket_j2 = ket_irrep.charge
        if bra_nelec != ket_nelec + dnelec:
            continue
        block_shape = template.block(bra_irrep, ket_irrep).shape
        block_adj = np.zeros(block_shape, dtype=complex)
        norm = np.sqrt(bra_j2 + 1.0)
        for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
            bra_m2 = ket_m2 + int(q2)
            if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                continue
            bra_group = grouped.get((bra_irrep, bra_m2))
            ket_group = grouped.get((ket_irrep, ket_m2))
            if not bra_group or not ket_group:
                continue
            coeff = cg(ket_j2, ket_m2, rank2, int(q2), bra_j2, bra_m2)
            if abs(coeff) <= 1.0e-14:
                continue
            bra_pos, bra_local = zip(*bra_group)
            ket_pos, ket_local = zip(*ket_group)
            scale = np.conjugate(coeff / norm)
            block_adj[np.ix_(bra_local, ket_local)] += (
                scale * component_adjoint[np.ix_(bra_pos, ket_pos)]
            )
        if np.any(np.abs(block_adj) > 0.0):
            blocks[(bra_irrep, ket_irrep)] = block_adj
    return ReducedSU2Tensor(
        IrrepTensor(template.site, template.site, template.op, blocks)
    )


def _pair_annihilate_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    cache,
    xcache,
    ycache,
    xycache,
    i,
    j,
):
    key = (int(i), int(j))
    if key in xycache:
        return xycache[key]
    tensor = coupled_reduced_product(
        operators[("Ctilde", key[0])],
        operators[("Ctilde", key[1])],
        rank2=0,
        scale=-1.0 / np.sqrt(2.0),
    )
    _pair_annihilate_tangent(operators, xoperators, cache, xcache, key[0], key[1])
    _pair_annihilate_tangent(operators, yoperators, cache, ycache, key[0], key[1])
    xytensor = _coupled_product_bilinear(
        operators[("Ctilde", key[0])],
        xoperators.get(("Ctilde", key[0])),
        yoperators.get(("Ctilde", key[0])),
        xyoperators.get(("Ctilde", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        0,
        scale=-1.0 / np.sqrt(2.0),
    )
    cache[key] = tensor
    xycache[key] = xytensor
    return xytensor


def _pair_annihilate_bilinear_adjoint_x(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    tangent_adjoint,
    i,
    j,
):
    key = (int(i), int(j))
    out_x = {}
    out_xy = {}
    left_x, right_x, left_xy, right_xy = _coupled_product_bilinear_adjoint_x(
        operators[("Ctilde", key[0])],
        xoperators.get(("Ctilde", key[0])),
        yoperators.get(("Ctilde", key[0])),
        xyoperators.get(("Ctilde", key[0])),
        operators[("Ctilde", key[1])],
        xoperators.get(("Ctilde", key[1])),
        yoperators.get(("Ctilde", key[1])),
        xyoperators.get(("Ctilde", key[1])),
        tangent_adjoint,
        0,
        scale=-1.0 / np.sqrt(2.0),
    )
    _add_reduced_adjoint(out_x, ("Ctilde", key[0]), left_x)
    _add_reduced_adjoint(out_x, ("Ctilde", key[1]), right_x)
    _add_reduced_adjoint(out_xy, ("Ctilde", key[0]), left_xy)
    _add_reduced_adjoint(out_xy, ("Ctilde", key[1]), right_xy)
    return out_x, out_xy


def _cdag_density_tangent(operators, doperators, density_cache, ddensity_cache, cache, dcache, k, j, i):
    key = (int(k), int(j), int(i))
    if key in dcache:
        return dcache[key]
    op_key = ("CdagDensity", key[0], key[1], key[2])
    tensor = operators.get(op_key)
    direct = doperators.get(op_key)
    if tensor is not None and direct is not None:
        dcache[key] = direct
        cache[key] = tensor
        return direct
    density = density_cache.get((key[1], key[2]))
    if density is None:
        density = operators.get(("Density", key[1], key[2]))
        if density is None:
            density = coupled_reduced_product(
                operators[("Cdag", key[1])],
                operators[("Ctilde", key[2])],
                rank2=0,
                scale=np.sqrt(2.0),
            )
        density_cache[(key[1], key[2])] = density
    ddensity = _density_tangent(
        operators,
        doperators,
        density_cache,
        ddensity_cache,
        key[1],
        key[2],
    )
    if tensor is None:
        tensor = coupled_reduced_product(operators[("Cdag", key[0])], density, rank2=1)
    dtensor = _coupled_product_tangent(
        operators[("Cdag", key[0])],
        doperators.get(("Cdag", key[0])),
        density,
        ddensity,
        1,
    )
    cache[key] = tensor
    dcache[key] = dtensor
    return dtensor


def _cdag_density_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    density_cache,
    xdensity_cache,
    ydensity_cache,
    xydensity_cache,
    cache,
    xcache,
    ycache,
    xycache,
    k,
    j,
    i,
):
    key = (int(k), int(j), int(i))
    if key in xycache:
        return xycache[key]
    op_key = ("CdagDensity", key[0], key[1], key[2])
    tensor = operators.get(op_key)
    direct = xyoperators.get(op_key)
    if tensor is not None and direct is not None:
        xycache[key] = direct
        cache[key] = tensor
        return direct
    density = density_cache.get((key[1], key[2]))
    if density is None:
        density = operators.get(("Density", key[1], key[2]))
        if density is None:
            density = coupled_reduced_product(
                operators[("Cdag", key[1])],
                operators[("Ctilde", key[2])],
                rank2=0,
                scale=np.sqrt(2.0),
            )
        density_cache[(key[1], key[2])] = density
    xdensity = _density_tangent(
        operators,
        xoperators,
        density_cache,
        xdensity_cache,
        key[1],
        key[2],
    )
    ydensity = _density_tangent(
        operators,
        yoperators,
        density_cache,
        ydensity_cache,
        key[1],
        key[2],
    )
    xydensity = _density_bilinear(
        operators,
        xoperators,
        yoperators,
        xyoperators,
        density_cache,
        xdensity_cache,
        ydensity_cache,
        xydensity_cache,
        key[1],
        key[2],
    )
    if tensor is None:
        tensor = coupled_reduced_product(operators[("Cdag", key[0])], density, rank2=1)
    xytensor = _coupled_product_bilinear(
        operators[("Cdag", key[0])],
        xoperators.get(("Cdag", key[0])),
        yoperators.get(("Cdag", key[0])),
        xyoperators.get(("Cdag", key[0])),
        density,
        xdensity,
        ydensity,
        xydensity,
        1,
    )
    cache[key] = tensor
    xycache[key] = xytensor
    return xytensor


def _weighted_packages_tangent(
    operators,
    doperators,
    h1e,
    dh1e,
    eri,
    deri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = True,
):
    """Analytic tangent of ``weighted_packages_from_operators``."""
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    if (
        not _operator_dict_has_nonzero_blocks(doperators)
        and not _weighted_coefficients_have_support(
            dh1e,
            deri,
            site_count,
            future_sites,
            build_v1=bool(build_v1),
        )
    ):
        return {}
    out = {}
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    cdag_density_cache = {}
    ddensity_cache = {}
    dspin_density_cache = {}
    dpair_cache = {}
    dcdag_density_cache = {}

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v1_terms = []
        v3_terms = []

        if build_v1:
            for i in range(site_count):
                tensor = operators[("Cdag", i)]
                dtensor = doperators.get(("Cdag", i), _zero_reduced_like(tensor))
                v1_terms.append((tensor, dtensor, h1e[i, q], dh1e[i, q]))

        for i in range(site_count):
            tensor = operators[("Cdag", i)]
            dtensor = doperators.get(("Cdag", i), _zero_reduced_like(tensor))
            v3_terms.append((tensor, dtensor, eri[i, q, q, q], deri[i, q, q, q]))

            for j in range(site_count):
                density = density_cache.get((i, j))
                if density is None:
                    density = operators.get(("Density", i, j))
                    if density is None:
                        density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=np.sqrt(2.0),
                        )
                    density_cache[(i, j)] = density
                ddensity = _density_tangent(
                    operators,
                    doperators,
                    density_cache,
                    ddensity_cache,
                    i,
                    j,
                )
                density_terms.append((density, ddensity, eri[i, j, q, q], deri[i, j, q, q]))
                exchange_density_terms.append((density, ddensity, eri[i, q, q, j], deri[i, q, q, j]))

                spin_density = spin_density_cache.get((i, j))
                if spin_density is None:
                    spin_density = operators.get(("SpinDensity", i, j))
                    if spin_density is None:
                        spin_density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=2,
                        )
                    spin_density_cache[(i, j)] = spin_density
                dspin = _spin_density_tangent(
                    operators,
                    doperators,
                    spin_density_cache,
                    dspin_density_cache,
                    i,
                    j,
                )
                exchange_spin_terms.append((spin_density, dspin, eri[i, q, q, j], deri[i, q, q, j]))

                pair = pair_cache.get((i, j))
                if pair is None:
                    pair = coupled_reduced_product(
                        operators[("Ctilde", i)],
                        operators[("Ctilde", j)],
                        rank2=0,
                        scale=-1.0 / np.sqrt(2.0),
                    )
                    pair_cache[(i, j)] = pair
                dpair = _pair_annihilate_tangent(
                    operators,
                    doperators,
                    pair_cache,
                    dpair_cache,
                    i,
                    j,
                )
                pair_terms.append((pair, dpair, eri[q, i, q, j], deri[q, i, q, j]))

        if build_v1:
            for i in range(site_count):
                for j in range(site_count):
                    for k in range(site_count):
                        cdag_density = cdag_density_cache.get((k, j, i))
                        if cdag_density is None:
                            density = density_cache.get((j, i))
                            if density is None:
                                density = operators.get(("Density", j, i))
                                if density is None:
                                    density = coupled_reduced_product(
                                        operators[("Cdag", j)],
                                        operators[("Ctilde", i)],
                                        rank2=0,
                                        scale=np.sqrt(2.0),
                                    )
                                density_cache[(j, i)] = density
                            cdag_density = coupled_reduced_product(
                                operators[("Cdag", k)],
                                density,
                                rank2=1,
                            )
                            cdag_density_cache[(k, j, i)] = cdag_density
                        dcd = _cdag_density_tangent(
                            operators,
                            doperators,
                            density_cache,
                            ddensity_cache,
                            cdag_density_cache,
                            dcdag_density_cache,
                            k,
                            j,
                            i,
                        )
                        v1_terms.append((cdag_density, dcd, eri[k, q, j, i], deri[k, q, j, i]))

        packages = {
            ("NextDensity", q): _weighted_tangent_terms(density_terms),
            ("NextExchangeDensity", q): _weighted_tangent_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): _weighted_tangent_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): _weighted_tangent_terms(pair_terms),
            ("NextV1Spinor", q): _weighted_tangent_terms(v1_terms),
            ("NextV3Cdag", q): _weighted_tangent_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return out


def _weighted_packages_tangent_adjoint(
    operators,
    doperators,
    package_adjoint,
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = True,
):
    """Adjoint of ``_weighted_packages_tangent`` wrt tangent operators/integrals."""
    if build_v1:
        raise NotImplementedError(
            "weighted-package V1 adjoint is handled by projected component V1"
        )
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    doperator_adjoint = {}
    dh1_adj = np.zeros_like(h1e, dtype=float)
    deri_adj = np.zeros_like(eri, dtype=float)
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    ddensity_cache = {}
    dspin_density_cache = {}
    dpair_cache = {}

    def direct_term(adjoint, tensor, dtensor_key, coeff, coeff_array, coeff_index):
        coeff_array[coeff_index] += _reduced_tensor_pairing(adjoint, tensor)
        dtensor = doperators.get(dtensor_key)
        if dtensor is not None:
            _add_reduced_adjoint(
                doperator_adjoint,
                dtensor_key,
                _scale_adjoint(adjoint, coeff),
            )

    def composite_term(adjoint, tensor, tangent, coeff, coeff_array, coeff_index, adj_fn, *args):
        coeff_array[coeff_index] += _reduced_tensor_pairing(adjoint, tensor)
        if tangent.blocks:
            _merge_reduced_adjoint(
                doperator_adjoint,
                adj_fn(operators, doperators, _scale_adjoint(adjoint, coeff), *args),
            )

    for q in future_sites:
        q = int(q)
        v3_adj = package_adjoint.get(("NextV3Cdag", q))
        density_adj = package_adjoint.get(("NextDensity", q))
        exchange_density_adj = package_adjoint.get(("NextExchangeDensity", q))
        exchange_spin_adj = package_adjoint.get(("NextExchangeSpinDensity", q))
        pair_adj = package_adjoint.get(("NextPairAnnihilate", q))

        if v3_adj is not None:
            for i in range(site_count):
                tensor = operators[("Cdag", i)]
                direct_term(
                    v3_adj,
                    tensor,
                    ("Cdag", i),
                    eri[i, q, q, q],
                    deri_adj,
                    (i, q, q, q),
                )

        for i in range(site_count):
            for j in range(site_count):
                if (
                    density_adj is not None
                    or exchange_density_adj is not None
                    or exchange_spin_adj is not None
                ):
                    density = density_cache.get((i, j))
                    if density is None:
                        density = operators.get(("Density", i, j))
                        if density is None:
                            density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=0,
                                scale=np.sqrt(2.0),
                            )
                        density_cache[(i, j)] = density
                    ddensity = _density_tangent(
                        operators,
                        doperators,
                        density_cache,
                        ddensity_cache,
                        i,
                        j,
                    )
                    if density_adj is not None:
                        composite_term(
                            density_adj,
                            density,
                            ddensity,
                            eri[i, j, q, q],
                            deri_adj,
                            (i, j, q, q),
                            _density_tangent_adjoint,
                            i,
                            j,
                        )
                    if exchange_density_adj is not None:
                        composite_term(
                            exchange_density_adj,
                            density,
                            ddensity,
                            eri[i, q, q, j],
                            deri_adj,
                            (i, q, q, j),
                            _density_tangent_adjoint,
                            i,
                            j,
                        )

                if exchange_spin_adj is not None:
                    spin_density = spin_density_cache.get((i, j))
                    if spin_density is None:
                        spin_density = operators.get(("SpinDensity", i, j))
                        if spin_density is None:
                            spin_density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=2,
                            )
                        spin_density_cache[(i, j)] = spin_density
                    dspin = _spin_density_tangent(
                        operators,
                        doperators,
                        spin_density_cache,
                        dspin_density_cache,
                        i,
                        j,
                    )
                    composite_term(
                        exchange_spin_adj,
                        spin_density,
                        dspin,
                        eri[i, q, q, j],
                        deri_adj,
                        (i, q, q, j),
                        _spin_density_tangent_adjoint,
                        i,
                        j,
                    )

                if pair_adj is not None:
                    pair = pair_cache.get((i, j))
                    if pair is None:
                        pair = coupled_reduced_product(
                            operators[("Ctilde", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=-1.0 / np.sqrt(2.0),
                        )
                        pair_cache[(i, j)] = pair
                    dpair = _pair_annihilate_tangent(
                        operators,
                        doperators,
                        pair_cache,
                        dpair_cache,
                        i,
                        j,
                    )
                    composite_term(
                        pair_adj,
                        pair,
                        dpair,
                        eri[q, i, q, j],
                        deri_adj,
                        (q, i, q, j),
                        _pair_annihilate_tangent_adjoint,
                        i,
                        j,
                    )

    return doperator_adjoint, dh1_adj, deri_adj


def _weighted_packages_bilinear_adjoint_x(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    package_adjoint,
    h1e,
    eri,
    yeri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = True,
):
    """Adjoint of ``_weighted_packages_bilinear`` wrt ``x``/``xy`` inputs."""
    if build_v1:
        raise NotImplementedError(
            "weighted-package V1 adjoint is handled by projected component V1"
        )
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    xoperator_adjoint = {}
    xyoperator_adjoint = {}
    xh1_adj = np.zeros_like(h1e, dtype=float)
    xyh1_adj = np.zeros_like(h1e, dtype=float)
    xeri_adj = np.zeros_like(eri, dtype=float)
    xyeri_adj = np.zeros_like(eri, dtype=float)
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    ydensity_cache = {}
    yspin_density_cache = {}
    ypair_cache = {}

    def direct_term(adjoint, tensor, ytensor, tensor_key, coeff, ycoeff, index):
        xeri_adj[index] += _reduced_tensor_pairing(adjoint, ytensor)
        xyeri_adj[index] += _reduced_tensor_pairing(adjoint, tensor)
        if abs(ycoeff) > 0.0:
            _add_reduced_adjoint(
                xoperator_adjoint,
                tensor_key,
                _scale_adjoint(adjoint, ycoeff),
            )
        if abs(coeff) > 0.0:
            _add_reduced_adjoint(
                xyoperator_adjoint,
                tensor_key,
                _scale_adjoint(adjoint, coeff),
            )

    def composite_term(
        adjoint,
        tensor,
        ytensor,
        coeff,
        ycoeff,
        index,
        tangent_adj_fn,
        bilinear_adj_fn,
        *args,
    ):
        xeri_adj[index] += _reduced_tensor_pairing(adjoint, ytensor)
        xyeri_adj[index] += _reduced_tensor_pairing(adjoint, tensor)
        if abs(ycoeff) > 0.0:
            _merge_reduced_adjoint(
                xoperator_adjoint,
                tangent_adj_fn(
                    operators,
                    xoperators,
                    _scale_adjoint(adjoint, ycoeff),
                    *args,
                ),
            )
        if abs(coeff) > 0.0:
            bx_adj, bxy_adj = bilinear_adj_fn(
                operators,
                xoperators,
                yoperators,
                xyoperators,
                _scale_adjoint(adjoint, coeff),
                *args,
            )
            _merge_reduced_adjoint(xoperator_adjoint, bx_adj)
            _merge_reduced_adjoint(xyoperator_adjoint, bxy_adj)

    for q in future_sites:
        q = int(q)
        v3_adj = package_adjoint.get(("NextV3Cdag", q))
        density_adj = package_adjoint.get(("NextDensity", q))
        exchange_density_adj = package_adjoint.get(("NextExchangeDensity", q))
        exchange_spin_adj = package_adjoint.get(("NextExchangeSpinDensity", q))
        pair_adj = package_adjoint.get(("NextPairAnnihilate", q))

        if v3_adj is not None:
            for i in range(site_count):
                tensor = operators[("Cdag", i)]
                ytensor = yoperators.get(("Cdag", i), _zero_reduced_like(tensor))
                direct_term(
                    v3_adj,
                    tensor,
                    ytensor,
                    ("Cdag", i),
                    eri[i, q, q, q],
                    yeri[i, q, q, q],
                    (i, q, q, q),
                )

        for i in range(site_count):
            for j in range(site_count):
                if (
                    density_adj is not None
                    or exchange_density_adj is not None
                    or exchange_spin_adj is not None
                ):
                    density = density_cache.get((i, j))
                    if density is None:
                        density = operators.get(("Density", i, j))
                        if density is None:
                            density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=0,
                                scale=np.sqrt(2.0),
                            )
                        density_cache[(i, j)] = density
                    ydensity = _density_tangent(
                        operators,
                        yoperators,
                        density_cache,
                        ydensity_cache,
                        i,
                        j,
                    )
                    if density_adj is not None:
                        composite_term(
                            density_adj,
                            density,
                            ydensity,
                            eri[i, j, q, q],
                            yeri[i, j, q, q],
                            (i, j, q, q),
                            _density_tangent_adjoint,
                            _density_bilinear_adjoint_x,
                            i,
                            j,
                        )
                    if exchange_density_adj is not None:
                        composite_term(
                            exchange_density_adj,
                            density,
                            ydensity,
                            eri[i, q, q, j],
                            yeri[i, q, q, j],
                            (i, q, q, j),
                            _density_tangent_adjoint,
                            _density_bilinear_adjoint_x,
                            i,
                            j,
                        )

                if exchange_spin_adj is not None:
                    spin_density = spin_density_cache.get((i, j))
                    if spin_density is None:
                        spin_density = operators.get(("SpinDensity", i, j))
                        if spin_density is None:
                            spin_density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=2,
                            )
                        spin_density_cache[(i, j)] = spin_density
                    yspin = _spin_density_tangent(
                        operators,
                        yoperators,
                        spin_density_cache,
                        yspin_density_cache,
                        i,
                        j,
                    )
                    composite_term(
                        exchange_spin_adj,
                        spin_density,
                        yspin,
                        eri[i, q, q, j],
                        yeri[i, q, q, j],
                        (i, q, q, j),
                        _spin_density_tangent_adjoint,
                        _spin_density_bilinear_adjoint_x,
                        i,
                        j,
                    )

                if pair_adj is not None:
                    pair = pair_cache.get((i, j))
                    if pair is None:
                        pair = coupled_reduced_product(
                            operators[("Ctilde", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=-1.0 / np.sqrt(2.0),
                        )
                        pair_cache[(i, j)] = pair
                    ypair = _pair_annihilate_tangent(
                        operators,
                        yoperators,
                        pair_cache,
                        ypair_cache,
                        i,
                        j,
                    )
                    composite_term(
                        pair_adj,
                        pair,
                        ypair,
                        eri[q, i, q, j],
                        yeri[q, i, q, j],
                        (q, i, q, j),
                        _pair_annihilate_tangent_adjoint,
                        _pair_annihilate_bilinear_adjoint_x,
                        i,
                        j,
                    )

    return (
        xoperator_adjoint,
        xyoperator_adjoint,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _weighted_packages_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    h1e,
    xh1e,
    yh1e,
    xyh1e,
    eri,
    xeri,
    yeri,
    xyeri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = True,
):
    """Mixed tangent of ``weighted_packages_from_operators``."""
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    if (
        not _operator_dict_has_nonzero_blocks(xoperators)
        and not _operator_dict_has_nonzero_blocks(xyoperators)
        and not _weighted_coefficients_have_support(
            xh1e,
            xeri,
            site_count,
            future_sites,
            build_v1=bool(build_v1),
        )
        and not _weighted_coefficients_have_support(
            xyh1e,
            xyeri,
            site_count,
            future_sites,
            build_v1=bool(build_v1),
        )
    ):
        return {}
    out = {}
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    cdag_density_cache = {}
    xdensity_cache = {}
    ydensity_cache = {}
    xydensity_cache = {}
    xspin_density_cache = {}
    yspin_density_cache = {}
    xyspin_density_cache = {}
    xpair_cache = {}
    ypair_cache = {}
    xypair_cache = {}
    xcdag_density_cache = {}
    ycdag_density_cache = {}
    xycdag_density_cache = {}

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v1_terms = []
        v3_terms = []

        if build_v1:
            for i in range(site_count):
                tensor = operators[("Cdag", i)]
                xtensor = xoperators.get(("Cdag", i), _zero_reduced_like(tensor))
                ytensor = yoperators.get(("Cdag", i), _zero_reduced_like(tensor))
                xytensor = xyoperators.get(("Cdag", i), _zero_reduced_like(tensor))
                v1_terms.append(
                    (
                        tensor,
                        xtensor,
                        ytensor,
                        xytensor,
                        h1e[i, q],
                        xh1e[i, q],
                        yh1e[i, q],
                        xyh1e[i, q],
                    )
                )

        for i in range(site_count):
            tensor = operators[("Cdag", i)]
            xtensor = xoperators.get(("Cdag", i), _zero_reduced_like(tensor))
            ytensor = yoperators.get(("Cdag", i), _zero_reduced_like(tensor))
            xytensor = xyoperators.get(("Cdag", i), _zero_reduced_like(tensor))
            v3_terms.append(
                (
                    tensor,
                    xtensor,
                    ytensor,
                    xytensor,
                    eri[i, q, q, q],
                    xeri[i, q, q, q],
                    yeri[i, q, q, q],
                    xyeri[i, q, q, q],
                )
            )

            for j in range(site_count):
                density = density_cache.get((i, j))
                if density is None:
                    density = operators.get(("Density", i, j))
                    if density is None:
                        density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=np.sqrt(2.0),
                        )
                    density_cache[(i, j)] = density
                xdensity = _density_tangent(
                    operators,
                    xoperators,
                    density_cache,
                    xdensity_cache,
                    i,
                    j,
                )
                ydensity = _density_tangent(
                    operators,
                    yoperators,
                    density_cache,
                    ydensity_cache,
                    i,
                    j,
                )
                xydensity = _density_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    density_cache,
                    xdensity_cache,
                    ydensity_cache,
                    xydensity_cache,
                    i,
                    j,
                )
                density_terms.append(
                    (
                        density,
                        xdensity,
                        ydensity,
                        xydensity,
                        eri[i, j, q, q],
                        xeri[i, j, q, q],
                        yeri[i, j, q, q],
                        xyeri[i, j, q, q],
                    )
                )
                exchange_density_terms.append(
                    (
                        density,
                        xdensity,
                        ydensity,
                        xydensity,
                        eri[i, q, q, j],
                        xeri[i, q, q, j],
                        yeri[i, q, q, j],
                        xyeri[i, q, q, j],
                    )
                )

                spin_density = spin_density_cache.get((i, j))
                if spin_density is None:
                    spin_density = operators.get(("SpinDensity", i, j))
                    if spin_density is None:
                        spin_density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=2,
                        )
                    spin_density_cache[(i, j)] = spin_density
                xspin = _spin_density_tangent(
                    operators,
                    xoperators,
                    spin_density_cache,
                    xspin_density_cache,
                    i,
                    j,
                )
                yspin = _spin_density_tangent(
                    operators,
                    yoperators,
                    spin_density_cache,
                    yspin_density_cache,
                    i,
                    j,
                )
                xyspin = _spin_density_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    spin_density_cache,
                    xspin_density_cache,
                    yspin_density_cache,
                    xyspin_density_cache,
                    i,
                    j,
                )
                exchange_spin_terms.append(
                    (
                        spin_density,
                        xspin,
                        yspin,
                        xyspin,
                        eri[i, q, q, j],
                        xeri[i, q, q, j],
                        yeri[i, q, q, j],
                        xyeri[i, q, q, j],
                    )
                )

                pair = pair_cache.get((i, j))
                if pair is None:
                    pair = coupled_reduced_product(
                        operators[("Ctilde", i)],
                        operators[("Ctilde", j)],
                        rank2=0,
                        scale=-1.0 / np.sqrt(2.0),
                    )
                    pair_cache[(i, j)] = pair
                xpair = _pair_annihilate_tangent(
                    operators,
                    xoperators,
                    pair_cache,
                    xpair_cache,
                    i,
                    j,
                )
                ypair = _pair_annihilate_tangent(
                    operators,
                    yoperators,
                    pair_cache,
                    ypair_cache,
                    i,
                    j,
                )
                xypair = _pair_annihilate_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    pair_cache,
                    xpair_cache,
                    ypair_cache,
                    xypair_cache,
                    i,
                    j,
                )
                pair_terms.append(
                    (
                        pair,
                        xpair,
                        ypair,
                        xypair,
                        eri[q, i, q, j],
                        xeri[q, i, q, j],
                        yeri[q, i, q, j],
                        xyeri[q, i, q, j],
                    )
                )

        if build_v1:
            for i in range(site_count):
                for j in range(site_count):
                    for k in range(site_count):
                        cdag_density = cdag_density_cache.get((k, j, i))
                        if cdag_density is None:
                            density = density_cache.get((j, i))
                            if density is None:
                                density = operators.get(("Density", j, i))
                                if density is None:
                                    density = coupled_reduced_product(
                                        operators[("Cdag", j)],
                                        operators[("Ctilde", i)],
                                        rank2=0,
                                        scale=np.sqrt(2.0),
                                    )
                                density_cache[(j, i)] = density
                            cdag_density = coupled_reduced_product(
                                operators[("Cdag", k)],
                                density,
                                rank2=1,
                            )
                            cdag_density_cache[(k, j, i)] = cdag_density
                        xcd = _cdag_density_tangent(
                            operators,
                            xoperators,
                            density_cache,
                            xdensity_cache,
                            cdag_density_cache,
                            xcdag_density_cache,
                            k,
                            j,
                            i,
                        )
                        ycd = _cdag_density_tangent(
                            operators,
                            yoperators,
                            density_cache,
                            ydensity_cache,
                            cdag_density_cache,
                            ycdag_density_cache,
                            k,
                            j,
                            i,
                        )
                        xycd = _cdag_density_bilinear(
                            operators,
                            xoperators,
                            yoperators,
                            xyoperators,
                            density_cache,
                            xdensity_cache,
                            ydensity_cache,
                            xydensity_cache,
                            cdag_density_cache,
                            xcdag_density_cache,
                            ycdag_density_cache,
                            xycdag_density_cache,
                            k,
                            j,
                            i,
                        )
                        v1_terms.append(
                            (
                                cdag_density,
                                xcd,
                                ycd,
                                xycd,
                                eri[k, q, j, i],
                                xeri[k, q, j, i],
                                yeri[k, q, j, i],
                                xyeri[k, q, j, i],
                            )
                        )

        packages = {
            ("NextDensity", q): _weighted_bilinear_terms(density_terms),
            ("NextExchangeDensity", q): _weighted_bilinear_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): _weighted_bilinear_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): _weighted_bilinear_terms(pair_terms),
            ("NextV1Spinor", q): _weighted_bilinear_terms(v1_terms),
            ("NextV3Cdag", q): _weighted_bilinear_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return out


def _scale_irrep_tensor(tensor: IrrepTensor, factor: complex, *, atol: float = 1.0e-14):
    blocks = {
        key: factor * block
        for key, block in tensor.blocks.items()
        if np.any(np.abs(factor * block) > atol)
    }
    return IrrepTensor(tensor.bra, tensor.ket, tensor.op, blocks)


def _add_irrep_tensors(*tensors: IrrepTensor, atol: float = 1.0e-14):
    if not tensors:
        raise ValueError("at least one IrrepTensor is required")
    bra = tensors[0].bra
    ket = tensors[0].ket
    op = tensors[0].op
    blocks = {}
    keys = set().union(*(tensor.blocks.keys() for tensor in tensors))
    for key in keys:
        block = None
        for tensor in tensors:
            term = tensor.blocks.get(key)
            if term is None:
                continue
            block = np.array(term, copy=True) if block is None else block + term
        if block is not None and np.any(np.abs(block) > atol):
            blocks[key] = block
    return IrrepTensor(bra, ket, op, blocks)


def _make_tangent_block(source_block, d_hamiltonian, d_reduced_operators):
    from .su2_two_site import RenormalizedSU2Block

    tangent_truncated = replace(source_block.truncated, hamiltonian=d_hamiltonian)
    block = RenormalizedSU2Block(
        truncated=tangent_truncated,
        hamiltonian=d_hamiltonian,
        transform=source_block.transform,
        operators={},
        reduced_operators=d_reduced_operators,
        parity=None,
    )
    block._su2_multiplets = getattr(source_block, "_su2_multiplets", [])
    for name in (
        "_su2_allowed_final_nelec",
        "_su2_requested_D",
        "_su2_chosen_D",
    ):
        if hasattr(source_block, name):
            setattr(block, name, getattr(source_block, name))
    return block


def _two_site_source_data(narg):
    data = getattr(narg, "_su2_response_two_site_source_data", None)
    if data is not None:
        return data
    from .su2_core import asarray, full_jw_model

    multiplets = [state.multiplet for state in narg.branch_states]
    model = full_jw_model(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), nelec=2)
    pair_ops = {
        "Cdu": [asarray(op) for op in model.Cdu],
        "Cdd": [asarray(op) for op in model.Cdd],
    }
    data = multiplets, model, pair_ops
    narg._su2_response_two_site_source_data = data
    return data


def _two_site_source_reduced_tensor(narg, key):
    cache = getattr(narg, "_su2_response_source_tensor_cache", None)
    if cache is None:
        cache = {}
        narg._su2_response_source_tensor_cache = cache
    if key in cache:
        return cache[key]

    multiplets, model, pair_ops = _two_site_source_data(narg)
    name = key[0]
    tensor = None
    if name == "Cdag":
        site = int(key[1])
        tensor = reduced_tensor_from_components(
            multiplets,
            {1: model.Cdu[site], -1: model.Cdd[site]},
            OpIrrep((1, 1)),
        )
    elif name == "Ctilde":
        site = int(key[1])
        tensor = reduced_tensor_from_components(
            multiplets,
            {-1: model.Cu[site], 1: -model.Cd[site]},
            OpIrrep((-1, 1)),
        )
    elif name == "Density":
        i = int(key[1])
        j = int(key[2])
        density = model.Cdu[i] @ model.Cu[j] + model.Cdd[i] @ model.Cd[j]
        tensor = reduced_tensor_from_components(
            multiplets,
            {0: density},
            OpIrrep((0, 0)),
        )
    elif name == "SpinDensity":
        i = int(key[1])
        j = int(key[2])
        tensor = reduced_tensor_from_components(
            multiplets,
            {
                -2: model.Cdd[i] @ model.Cu[j],
                0: (model.Cdu[i] @ model.Cu[j] - model.Cdd[i] @ model.Cd[j])
                / np.sqrt(2.0),
                2: -(model.Cdu[i] @ model.Cd[j]),
            },
            OpIrrep((0, 2)),
        )
    elif name in {"PairCreate0", "PairCreate2"}:
        from .su2_chain import _pair_create_component_ops

        i = int(key[1])
        j = int(key[2])
        rank2 = 0 if name == "PairCreate0" else 2
        components = _pair_create_component_ops(pair_ops, i, j, rank2)
        if components:
            tensor = reduced_tensor_from_components(
                multiplets,
                components,
                OpIrrep((2, rank2)),
            )
    cache[key] = tensor
    return tensor


def _component_v1_packages_tangent(
    block,
    dops,
    h1e,
    dh1e,
    eri,
    deri,
    site_count: int,
    future_sites,
):
    """Tangent of ``seed_component_v1_packages`` in the retained component basis."""
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    if (
        not _operator_dict_has_nonzero_blocks(dops)
        and not _component_v1_coefficients_have_support(
            dh1e,
            deri,
            site_count,
            future_sites,
        )
    ):
        return {}

    from .su2_chain import block_component_fermion_ops, retained_component_multiplets
    from .su2_three_site import expanded_operator_from_reduced

    states, multiplets = retained_component_multiplets(block)
    _, ops = block_component_fermion_ops(block, site_count)
    dim = ops["Cdu"][0].shape[0] if ops["Cdu"] else 0
    dop_components = {name: [] for name in ("Cdu", "Cdd", "Cu", "Cd")}
    for site_index in range(site_count):
        cdag = block.reduced_operators[("Cdag", site_index)]
        ctilde = block.reduced_operators[("Ctilde", site_index)]
        dcdag = dops.get(("Cdag", site_index), _zero_reduced_like(cdag))
        dctilde = dops.get(("Ctilde", site_index), _zero_reduced_like(ctilde))
        dop_components["Cdu"].append(
            expanded_operator_from_reduced(states, dcdag, q2=1)
        )
        dop_components["Cdd"].append(
            expanded_operator_from_reduced(states, dcdag, q2=-1)
        )
        dop_components["Cu"].append(
            expanded_operator_from_reduced(states, dctilde, q2=-1)
        )
        dop_components["Cd"].append(
            -expanded_operator_from_reduced(states, dctilde, q2=1)
        )

    out = {}
    for q in future_sites:
        q = int(q)
        dv1u = np.zeros((dim, dim), dtype=complex)
        dv1d = np.zeros((dim, dim), dtype=complex)
        for i in range(int(site_count)):
            coeff = h1e[i, q]
            dcoeff = dh1e[i, q]
            if abs(coeff) > 0.0 or abs(dcoeff) > 0.0:
                dv1u += dcoeff * ops["Cdu"][i] + coeff * dop_components["Cdu"][i]
                dv1d += dcoeff * ops["Cdd"][i] + coeff * dop_components["Cdd"][i]

        for i in range(int(site_count)):
            for j in range(int(site_count)):
                residual = (
                    ops["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdd"][j] @ ops["Cd"][i]
                )
                dresidual = (
                    dop_components["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdu"][j] @ dop_components["Cu"][i]
                    + dop_components["Cdd"][j] @ ops["Cd"][i]
                    + ops["Cdd"][j] @ dop_components["Cd"][i]
                )
                for k in range(int(site_count)):
                    coeff = eri[k, q, j, i]
                    dcoeff = deri[k, q, j, i]
                    if abs(coeff) <= 0.0 and abs(dcoeff) <= 0.0:
                        continue
                    term_u = ops["Cdu"][k] @ residual
                    term_d = ops["Cdd"][k] @ residual
                    dterm_u = (
                        dop_components["Cdu"][k] @ residual
                        + ops["Cdu"][k] @ dresidual
                    )
                    dterm_d = (
                        dop_components["Cdd"][k] @ residual
                        + ops["Cdd"][k] @ dresidual
                    )
                    dv1u += dcoeff * term_u + coeff * dterm_u
                    dv1d += dcoeff * term_d + coeff * dterm_d

        if np.any(np.abs(dv1u) > 1.0e-14) or np.any(np.abs(dv1d) > 1.0e-14):
            tensor = reduced_tensor_from_components(
                multiplets,
                {1: dv1u, -1: dv1d},
                OpIrrep((1, 1)),
            )
            if tensor.blocks:
                out[("NextV1Spinor", q)] = tensor
    return out


def _component_v1_packages_tangent_adjoint(
    block,
    dops,
    package_adjoint,
    h1e,
    eri,
    site_count: int,
    future_sites,
):
    """Adjoint of ``_component_v1_packages_tangent``."""
    from .su2_chain import block_component_fermion_ops, retained_component_multiplets
    from .su2_three_site import expanded_operator_from_reduced

    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    states, multiplets = retained_component_multiplets(block)
    _, ops = block_component_fermion_ops(block, site_count)
    dim = ops["Cdu"][0].shape[0] if ops["Cdu"] else 0
    op_adjoint = {}
    dh1_adj = np.zeros_like(h1e, dtype=float)
    deri_adj = np.zeros_like(eri, dtype=float)
    if dim == 0:
        return op_adjoint, dh1_adj, deri_adj

    def component_pairing(left, right) -> float:
        return float(np.real(np.vdot(left, right)))

    component_adjoints = {
        name: [np.zeros((dim, dim), dtype=complex) for _ in range(site_count)]
        for name in ("Cdu", "Cdd", "Cu", "Cd")
    }

    for q in future_sites:
        adj = package_adjoint.get(("NextV1Spinor", q))
        if adj is None:
            continue
        comp_adj = _reduced_tensor_from_components_adjoint(
            multiplets,
            adj,
            (1, -1),
        )
        au = comp_adj[1]
        ad = comp_adj[-1]

        for i in range(site_count):
            coeff = h1e[i, q]
            dh1_adj[i, q] += component_pairing(au, ops["Cdu"][i])
            dh1_adj[i, q] += component_pairing(ad, ops["Cdd"][i])
            if abs(coeff) > 0.0:
                component_adjoints["Cdu"][i] += np.conjugate(coeff) * au
                component_adjoints["Cdd"][i] += np.conjugate(coeff) * ad

        for i in range(site_count):
            for j in range(site_count):
                residual = (
                    ops["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdd"][j] @ ops["Cd"][i]
                )
                for k in range(site_count):
                    coeff = eri[k, q, j, i]
                    term_u = ops["Cdu"][k] @ residual
                    term_d = ops["Cdd"][k] @ residual
                    deri_adj[k, q, j, i] += component_pairing(au, term_u)
                    deri_adj[k, q, j, i] += component_pairing(ad, term_d)
                    if abs(coeff) <= 0.0:
                        continue

                    scaled_u = np.conjugate(coeff) * au
                    scaled_d = np.conjugate(coeff) * ad
                    component_adjoints["Cdu"][k] += scaled_u @ residual.conj().T
                    component_adjoints["Cdd"][k] += scaled_d @ residual.conj().T
                    residual_adj = ops["Cdu"][k].conj().T @ scaled_u
                    residual_adj += ops["Cdd"][k].conj().T @ scaled_d
                    component_adjoints["Cdu"][j] += residual_adj @ ops["Cu"][i].conj().T
                    component_adjoints["Cu"][i] += ops["Cdu"][j].conj().T @ residual_adj
                    component_adjoints["Cdd"][j] += residual_adj @ ops["Cd"][i].conj().T
                    component_adjoints["Cd"][i] += ops["Cdd"][j].conj().T @ residual_adj

    for site_index in range(site_count):
        cdag = block.reduced_operators[("Cdag", site_index)]
        ctilde = block.reduced_operators[("Ctilde", site_index)]
        cdag_adj = _expanded_operator_from_reduced_adjoint(
            states,
            cdag,
            1,
            component_adjoints["Cdu"][site_index],
        )
        cdag_adj = add_reduced_tensors(
            cdag_adj,
            _expanded_operator_from_reduced_adjoint(
                states,
                cdag,
                -1,
                component_adjoints["Cdd"][site_index],
            ),
        )
        ctilde_adj = _expanded_operator_from_reduced_adjoint(
            states,
            ctilde,
            -1,
            component_adjoints["Cu"][site_index],
        )
        ctilde_adj = add_reduced_tensors(
            ctilde_adj,
            _expanded_operator_from_reduced_adjoint(
                states,
                ctilde,
                1,
                -component_adjoints["Cd"][site_index],
            ),
        )
        _add_reduced_adjoint(op_adjoint, ("Cdag", site_index), cdag_adj)
        _add_reduced_adjoint(op_adjoint, ("Ctilde", site_index), ctilde_adj)

    del expanded_operator_from_reduced
    return op_adjoint, dh1_adj, deri_adj


def _component_v1_packages_bilinear(
    block,
    xops,
    yops,
    xyops,
    h1e,
    xh1e,
    yh1e,
    xyh1e,
    eri,
    xeri,
    yeri,
    xyeri,
    site_count: int,
    future_sites,
):
    """Mixed tangent of ``seed_component_v1_packages`` in component space."""
    from .su2_chain import block_component_fermion_ops, retained_component_multiplets
    from .su2_three_site import expanded_operator_from_reduced

    states, multiplets = retained_component_multiplets(block)
    _, ops = block_component_fermion_ops(block, int(site_count))
    dim = ops["Cdu"][0].shape[0] if ops["Cdu"] else 0

    def component_dict(source_ops):
        out = {name: [] for name in ("Cdu", "Cdd", "Cu", "Cd")}
        for site_index in range(int(site_count)):
            cdag = block.reduced_operators[("Cdag", site_index)]
            ctilde = block.reduced_operators[("Ctilde", site_index)]
            dcdag = source_ops.get(("Cdag", site_index), _zero_reduced_like(cdag))
            dctilde = source_ops.get(("Ctilde", site_index), _zero_reduced_like(ctilde))
            out["Cdu"].append(expanded_operator_from_reduced(states, dcdag, q2=1))
            out["Cdd"].append(expanded_operator_from_reduced(states, dcdag, q2=-1))
            out["Cu"].append(expanded_operator_from_reduced(states, dctilde, q2=-1))
            out["Cd"].append(-expanded_operator_from_reduced(states, dctilde, q2=1))
        return out

    xcomp = component_dict(xops)
    ycomp = component_dict(yops)
    xycomp = component_dict(xyops)

    out = {}
    for q in future_sites:
        q = int(q)
        dv1u = np.zeros((dim, dim), dtype=complex)
        dv1d = np.zeros((dim, dim), dtype=complex)
        for i in range(int(site_count)):
            coeff = h1e[i, q]
            xcoeff = xh1e[i, q]
            ycoeff = yh1e[i, q]
            xycoeff = xyh1e[i, q]
            if (
                abs(coeff) > 0.0
                or abs(xcoeff) > 0.0
                or abs(ycoeff) > 0.0
                or abs(xycoeff) > 0.0
            ):
                dv1u += (
                    xycoeff * ops["Cdu"][i]
                    + xcoeff * ycomp["Cdu"][i]
                    + ycoeff * xcomp["Cdu"][i]
                    + coeff * xycomp["Cdu"][i]
                )
                dv1d += (
                    xycoeff * ops["Cdd"][i]
                    + xcoeff * ycomp["Cdd"][i]
                    + ycoeff * xcomp["Cdd"][i]
                    + coeff * xycomp["Cdd"][i]
                )

        for i in range(int(site_count)):
            for j in range(int(site_count)):
                residual = (
                    ops["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdd"][j] @ ops["Cd"][i]
                )
                xresidual = (
                    xcomp["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdu"][j] @ xcomp["Cu"][i]
                    + xcomp["Cdd"][j] @ ops["Cd"][i]
                    + ops["Cdd"][j] @ xcomp["Cd"][i]
                )
                yresidual = (
                    ycomp["Cdu"][j] @ ops["Cu"][i]
                    + ops["Cdu"][j] @ ycomp["Cu"][i]
                    + ycomp["Cdd"][j] @ ops["Cd"][i]
                    + ops["Cdd"][j] @ ycomp["Cd"][i]
                )
                xyresidual = (
                    xycomp["Cdu"][j] @ ops["Cu"][i]
                    + xcomp["Cdu"][j] @ ycomp["Cu"][i]
                    + ycomp["Cdu"][j] @ xcomp["Cu"][i]
                    + ops["Cdu"][j] @ xycomp["Cu"][i]
                    + xycomp["Cdd"][j] @ ops["Cd"][i]
                    + xcomp["Cdd"][j] @ ycomp["Cd"][i]
                    + ycomp["Cdd"][j] @ xcomp["Cd"][i]
                    + ops["Cdd"][j] @ xycomp["Cd"][i]
                )
                for k in range(int(site_count)):
                    coeff = eri[k, q, j, i]
                    xcoeff = xeri[k, q, j, i]
                    ycoeff = yeri[k, q, j, i]
                    xycoeff = xyeri[k, q, j, i]
                    if (
                        abs(coeff) <= 0.0
                        and abs(xcoeff) <= 0.0
                        and abs(ycoeff) <= 0.0
                        and abs(xycoeff) <= 0.0
                    ):
                        continue
                    term_u = ops["Cdu"][k] @ residual
                    term_d = ops["Cdd"][k] @ residual
                    xterm_u = xcomp["Cdu"][k] @ residual + ops["Cdu"][k] @ xresidual
                    xterm_d = xcomp["Cdd"][k] @ residual + ops["Cdd"][k] @ xresidual
                    yterm_u = ycomp["Cdu"][k] @ residual + ops["Cdu"][k] @ yresidual
                    yterm_d = ycomp["Cdd"][k] @ residual + ops["Cdd"][k] @ yresidual
                    xyterm_u = (
                        xycomp["Cdu"][k] @ residual
                        + xcomp["Cdu"][k] @ yresidual
                        + ycomp["Cdu"][k] @ xresidual
                        + ops["Cdu"][k] @ xyresidual
                    )
                    xyterm_d = (
                        xycomp["Cdd"][k] @ residual
                        + xcomp["Cdd"][k] @ yresidual
                        + ycomp["Cdd"][k] @ xresidual
                        + ops["Cdd"][k] @ xyresidual
                    )
                    dv1u += (
                        xycoeff * term_u
                        + xcoeff * yterm_u
                        + ycoeff * xterm_u
                        + coeff * xyterm_u
                    )
                    dv1d += (
                        xycoeff * term_d
                        + xcoeff * yterm_d
                        + ycoeff * xterm_d
                        + coeff * xyterm_d
                    )

        if np.any(np.abs(dv1u) > 1.0e-14) or np.any(np.abs(dv1d) > 1.0e-14):
            tensor = reduced_tensor_from_components(
                multiplets,
                {1: dv1u, -1: dv1d},
                OpIrrep((1, 1)),
            )
            if tensor.blocks:
                out[("NextV1Spinor", q)] = tensor
    return out


def _component_v1_packages_bilinear_adjoint_x(
    block,
    xops,
    yops,
    xyops,
    package_adjoint,
    h1e,
    eri,
    yh1e,
    yeri,
    site_count: int,
    future_sites,
):
    """Adjoint of ``_component_v1_packages_bilinear`` wrt ``x``/``xy`` inputs."""
    from .su2_chain import block_component_fermion_ops, retained_component_multiplets
    from .su2_three_site import expanded_operator_from_reduced

    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    states, multiplets = retained_component_multiplets(block)
    _, ops = block_component_fermion_ops(block, site_count)
    dim = ops["Cdu"][0].shape[0] if ops["Cdu"] else 0
    xoperator_adjoint = {}
    xyoperator_adjoint = {}
    xh1_adj = np.zeros_like(h1e, dtype=float)
    xyh1_adj = np.zeros_like(h1e, dtype=float)
    xeri_adj = np.zeros_like(eri, dtype=float)
    xyeri_adj = np.zeros_like(eri, dtype=float)
    if dim == 0:
        return (
            xoperator_adjoint,
            xyoperator_adjoint,
            xh1_adj,
            xeri_adj,
            xyh1_adj,
            xyeri_adj,
        )

    def component_dict(source_ops):
        out = {name: [] for name in ("Cdu", "Cdd", "Cu", "Cd")}
        for site_index in range(site_count):
            cdag = block.reduced_operators[("Cdag", site_index)]
            ctilde = block.reduced_operators[("Ctilde", site_index)]
            dcdag = source_ops.get(("Cdag", site_index), _zero_reduced_like(cdag))
            dctilde = source_ops.get(("Ctilde", site_index), _zero_reduced_like(ctilde))
            out["Cdu"].append(expanded_operator_from_reduced(states, dcdag, q2=1))
            out["Cdd"].append(expanded_operator_from_reduced(states, dcdag, q2=-1))
            out["Cu"].append(expanded_operator_from_reduced(states, dctilde, q2=-1))
            out["Cd"].append(-expanded_operator_from_reduced(states, dctilde, q2=1))
        return out

    xcomp = component_dict(xops)
    ycomp = component_dict(yops)
    xycomp = component_dict(xyops)
    xcomp_adj = {
        name: [np.zeros((dim, dim), dtype=complex) for _ in range(site_count)]
        for name in ("Cdu", "Cdd", "Cu", "Cd")
    }
    xycomp_adj = {
        name: [np.zeros((dim, dim), dtype=complex) for _ in range(site_count)]
        for name in ("Cdu", "Cdd", "Cu", "Cd")
    }

    def component_pairing(left, right) -> float:
        return float(np.real(np.vdot(left, right)))

    def residual(i, j):
        return ops["Cdu"][j] @ ops["Cu"][i] + ops["Cdd"][j] @ ops["Cd"][i]

    def tangent_residual(comp, i, j):
        return (
            comp["Cdu"][j] @ ops["Cu"][i]
            + ops["Cdu"][j] @ comp["Cu"][i]
            + comp["Cdd"][j] @ ops["Cd"][i]
            + ops["Cdd"][j] @ comp["Cd"][i]
        )

    def residual_adjoint(target, adjoint, i, j):
        target["Cdu"][j] += adjoint @ ops["Cu"][i].conj().T
        target["Cu"][i] += ops["Cdu"][j].conj().T @ adjoint
        target["Cdd"][j] += adjoint @ ops["Cd"][i].conj().T
        target["Cd"][i] += ops["Cdd"][j].conj().T @ adjoint

    def xyresidual_x_adjoint(adjoint, i, j):
        xcomp_adj["Cdu"][j] += adjoint @ ycomp["Cu"][i].conj().T
        xcomp_adj["Cu"][i] += ycomp["Cdu"][j].conj().T @ adjoint
        xcomp_adj["Cdd"][j] += adjoint @ ycomp["Cd"][i].conj().T
        xcomp_adj["Cd"][i] += ycomp["Cdd"][j].conj().T @ adjoint

    def xterm_adjoint(target, adjoint, spin_name, k, i, j, base_residual):
        target[spin_name][k] += adjoint @ base_residual.conj().T
        residual_adj = ops[spin_name][k].conj().T @ adjoint
        residual_adjoint(target, residual_adj, i, j)

    def xyterm_adjoint(adjoint, spin_name, k, i, j, base_residual, y_residual):
        xcomp_adj[spin_name][k] += adjoint @ y_residual.conj().T
        residual_adjoint(
            xcomp_adj,
            ycomp[spin_name][k].conj().T @ adjoint,
            i,
            j,
        )
        xycomp_adj[spin_name][k] += adjoint @ base_residual.conj().T
        xyresidual_adj = ops[spin_name][k].conj().T @ adjoint
        residual_adjoint(xycomp_adj, xyresidual_adj, i, j)
        xyresidual_x_adjoint(xyresidual_adj, i, j)

    for q in future_sites:
        q = int(q)
        adj = package_adjoint.get(("NextV1Spinor", q))
        if adj is None:
            continue
        comp_adj = _reduced_tensor_from_components_adjoint(
            multiplets,
            adj,
            (1, -1),
        )
        spin_adjoints = (("Cdu", comp_adj[1]), ("Cdd", comp_adj[-1]))

        for i in range(site_count):
            for spin_name, adjoint in spin_adjoints:
                xh1_adj[i, q] += component_pairing(adjoint, ycomp[spin_name][i])
                xyh1_adj[i, q] += component_pairing(adjoint, ops[spin_name][i])
                if abs(yh1e[i, q]) > 0.0:
                    xcomp_adj[spin_name][i] += np.conjugate(yh1e[i, q]) * adjoint
                if abs(h1e[i, q]) > 0.0:
                    xycomp_adj[spin_name][i] += np.conjugate(h1e[i, q]) * adjoint

        for i in range(site_count):
            for j in range(site_count):
                base_residual = residual(i, j)
                y_residual = tangent_residual(ycomp, i, j)
                for k in range(site_count):
                    coeff = eri[k, q, j, i]
                    ycoeff = yeri[k, q, j, i]
                    if abs(coeff) <= 0.0 and abs(ycoeff) <= 0.0:
                        continue
                    for spin_name, adjoint in spin_adjoints:
                        yterm = (
                            ycomp[spin_name][k] @ base_residual
                            + ops[spin_name][k] @ y_residual
                        )
                        term = ops[spin_name][k] @ base_residual
                        xeri_adj[k, q, j, i] += component_pairing(adjoint, yterm)
                        xyeri_adj[k, q, j, i] += component_pairing(adjoint, term)
                        if abs(ycoeff) > 0.0:
                            xterm_adjoint(
                                xcomp_adj,
                                np.conjugate(ycoeff) * adjoint,
                                spin_name,
                                k,
                                i,
                                j,
                                base_residual,
                            )
                        if abs(coeff) > 0.0:
                            xyterm_adjoint(
                                np.conjugate(coeff) * adjoint,
                                spin_name,
                                k,
                                i,
                                j,
                                base_residual,
                                y_residual,
                            )

    def emit_component_adjoints(component_adjoint, target):
        for site_index in range(site_count):
            cdag = block.reduced_operators[("Cdag", site_index)]
            ctilde = block.reduced_operators[("Ctilde", site_index)]
            cdag_adj = _expanded_operator_from_reduced_adjoint(
                states,
                cdag,
                1,
                component_adjoint["Cdu"][site_index],
            )
            cdag_adj = add_reduced_tensors(
                cdag_adj,
                _expanded_operator_from_reduced_adjoint(
                    states,
                    cdag,
                    -1,
                    component_adjoint["Cdd"][site_index],
                ),
            )
            ctilde_adj = _expanded_operator_from_reduced_adjoint(
                states,
                ctilde,
                -1,
                component_adjoint["Cu"][site_index],
            )
            ctilde_adj = add_reduced_tensors(
                ctilde_adj,
                _expanded_operator_from_reduced_adjoint(
                    states,
                    ctilde,
                    1,
                    -component_adjoint["Cd"][site_index],
                ),
            )
            _add_reduced_adjoint(target, ("Cdag", site_index), cdag_adj)
            _add_reduced_adjoint(target, ("Ctilde", site_index), ctilde_adj)

    emit_component_adjoints(xcomp_adj, xoperator_adjoint)
    emit_component_adjoints(xycomp_adj, xyoperator_adjoint)
    del expanded_operator_from_reduced
    return (
        xoperator_adjoint,
        xyoperator_adjoint,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _seed_two_site_tangent_block(
    block,
    dh1e,
    deri,
    *,
    h1e_full,
    eri_full,
    dh1e_full,
    deri_full,
    final_size,
    project_v1_packages=False,
    include_retained_mixing=True,
    return_response: bool = False,
):
    from .su2_two_site import build_two_site_su2_narg

    narg = block.truncated.source
    local_is_zero = (
        not np.any(np.abs(dh1e) > 0.0)
        and not np.any(np.abs(deri) > 0.0)
    )
    if local_is_zero:
        response = _zero_truncation_tangent(narg, block.truncated)
    else:
        dnarg = build_two_site_su2_narg(dh1e, deri)
        response = truncation_tangent(
            narg,
            block.truncated,
            dnarg.hamiltonian,
            include_retained_mixing=include_retained_mixing,
        )

    dops = {}
    if not local_is_zero:
        for key, tensor in block.reduced_operators.items():
            source_tensor = _two_site_source_reduced_tensor(narg, key)
            if source_tensor is None:
                continue
            dops[key] = rotate_reduced_tensor_tangent(
                block.truncated,
                source_tensor,
                _zero_reduced_like(source_tensor),
                response,
            )

    # Seed exact two-operator composites are algebraic functions of Cdag/Ctilde.
    if not local_is_zero:
        for i in range(2):
            for j in range(2):
                if ("Density", i, j) in block.reduced_operators:
                    key = ("Density", i, j)
                    if key not in dops:
                        dops[key] = _density_tangent(
                            block.reduced_operators,
                            dops,
                            {},
                            {},
                            i,
                            j,
                        )
                if ("SpinDensity", i, j) in block.reduced_operators:
                    key = ("SpinDensity", i, j)
                    if key not in dops:
                        dops[key] = _spin_density_tangent(
                            block.reduced_operators,
                            dops,
                            {},
                            {},
                            i,
                            j,
                        )
    future_sites = tuple(range(2, int(final_size)))
    dpackages = _weighted_packages_tangent(
        block.reduced_operators,
        dops,
        h1e_full,
        dh1e_full,
        eri_full,
        deri_full,
        2,
        future_sites,
        build_v1=not bool(project_v1_packages),
    )
    dops.update(dpackages)
    if project_v1_packages:
        dops.update(
            _component_v1_packages_tangent(
                block,
                dops,
                h1e_full,
                dh1e_full,
                eri_full,
                deri_full,
                2,
                future_sites,
            )
        )
    source_tangent = _make_tangent_block(block, response.d_hamiltonian, dops)
    if return_response:
        return source_tangent, response.min_gap, response
    return source_tangent, response.min_gap


def _two_site_hamiltonian_integral_adjoint(seed_adjoint: IrrepTensor):
    from .su2_two_site import build_two_site_su2_narg

    h_adj = np.zeros((2, 2), dtype=float)
    g_adj = np.zeros((2, 2, 2, 2), dtype=float)
    zeros_h = np.zeros((2, 2))
    zeros_g = np.zeros((2, 2, 2, 2))
    for p in range(2):
        for q in range(2):
            unit_h = np.zeros((2, 2))
            unit_h[p, q] = 1.0
            tensor = build_two_site_su2_narg(unit_h, zeros_g).hamiltonian
            h_adj[p, q] = _irrep_tensor_pairing(seed_adjoint, tensor)
    for p in range(2):
        for q in range(2):
            for r in range(2):
                for s in range(2):
                    unit_g = np.zeros((2, 2, 2, 2))
                    unit_g[p, q, r, s] = 1.0
                    tensor = build_two_site_su2_narg(zeros_h, unit_g).hamiltonian
                    g_adj[p, q, r, s] = _irrep_tensor_pairing(seed_adjoint, tensor)
    return h_adj, g_adj


def _seed_two_site_tangent_adjoint(
    block,
    source_tangent,
    source_hamiltonian_adjoint,
    source_operator_adjoint,
    h1e_full,
    eri_full,
    *,
    final_size,
    project_v1_packages=False,
    include_retained_mixing=True,
):
    """Adjoint of ``_seed_two_site_tangent_block`` wrt active integrals."""
    if not project_v1_packages:
        raise NotImplementedError(
            "seed adjoint is currently implemented for projected V1 packages"
        )
    narg = block.truncated.source
    future_sites = tuple(range(2, int(final_size)))
    dops_adjoint = {}
    dh1_adj = np.zeros_like(h1e_full, dtype=float)
    deri_adj = np.zeros_like(eri_full, dtype=float)

    weighted_adjoint = {
        key: adj
        for key, adj in (source_operator_adjoint or {}).items()
        if isinstance(key, tuple)
        and key[0]
        in {
            "NextDensity",
            "NextExchangeDensity",
            "NextExchangeSpinDensity",
            "NextPairAnnihilate",
            "NextV3Cdag",
        }
    }
    if weighted_adjoint:
        op_adj, h_adj, g_adj = _weighted_packages_tangent_adjoint(
            block.reduced_operators,
            source_tangent.reduced_operators,
            weighted_adjoint,
            h1e_full,
            eri_full,
            2,
            future_sites,
            build_v1=False,
        )
        _merge_reduced_adjoint(dops_adjoint, op_adj)
        dh1_adj += h_adj
        deri_adj += g_adj

    v1_adjoint = {
        key: adj
        for key, adj in (source_operator_adjoint or {}).items()
        if isinstance(key, tuple) and key[0] == "NextV1Spinor"
    }
    if v1_adjoint:
        op_adj, h_adj, g_adj = _component_v1_packages_tangent_adjoint(
            block,
            source_tangent.reduced_operators,
            v1_adjoint,
            h1e_full,
            eri_full,
            2,
            future_sites,
        )
        _merge_reduced_adjoint(dops_adjoint, op_adj)
        dh1_adj += h_adj
        deri_adj += g_adj

    for key, adj in (source_operator_adjoint or {}).items():
        if isinstance(key, tuple) and str(key[0]).startswith("Next"):
            continue
        _add_reduced_adjoint(dops_adjoint, key, adj)

    source_tensors = {}
    for key in block.reduced_operators:
        tensor = _two_site_source_reduced_tensor(narg, key)
        if tensor is not None:
            source_tensors[key] = tensor
    _, transform_adjoint = rotate_reduced_tensors_tangent_adjoint(
        block.truncated,
        source_tensors,
        dops_adjoint,
    )
    seed_h_adjoint = truncation_tangent_adjoint(
        narg,
        block.truncated,
        transform_adjoint_blocks=transform_adjoint,
        hamiltonian_adjoint=source_hamiltonian_adjoint,
        include_retained_mixing=include_retained_mixing,
    )
    h_seed_adj, g_seed_adj = _two_site_hamiltonian_integral_adjoint(seed_h_adjoint)
    dh1_adj[:2, :2] += h_seed_adj
    deri_adj[:2, :2, :2, :2] += g_seed_adj
    return dh1_adj, deri_adj


def _seed_two_site_bilinear_block(
    block,
    xdh1e,
    xderi,
    ydh1e,
    yderi,
    xydh1e,
    xyderi,
    *,
    h1e_full,
    eri_full,
    xdh1e_full,
    xderi_full,
    ydh1e_full,
    yderi_full,
    xydh1e_full,
    xyderi_full,
    final_size,
    project_v1_packages=False,
    include_retained_mixing=True,
    x_path: RecursiveTangentPath | None = None,
    y_path: RecursiveTangentPath | None = None,
):
    from .su2_two_site import build_two_site_su2_narg

    narg = block.truncated.source
    tangent_x = (
        x_path.responses.get(2)
        if x_path is not None and 2 in x_path.responses
        else None
    )
    tangent_y = (
        y_path.responses.get(2)
        if y_path is not None and 2 in y_path.responses
        else None
    )
    source_x_cached = (
        x_path.sources.get(2)
        if x_path is not None and 2 in x_path.sources
        else None
    )
    source_y_cached = (
        y_path.sources.get(2)
        if y_path is not None and 2 in y_path.sources
        else None
    )
    x_hamiltonian = build_two_site_su2_narg(xdh1e, xderi).hamiltonian
    y_hamiltonian = build_two_site_su2_narg(ydh1e, yderi).hamiltonian
    xy_is_zero = (
        not np.any(np.abs(xydh1e) > 0.0)
        and not np.any(np.abs(xyderi) > 0.0)
    )
    xy_hamiltonian = (
        _zero_irrep_tensor_like(narg.hamiltonian)
        if xy_is_zero
        else build_two_site_su2_narg(xydh1e, xyderi).hamiltonian
    )
    response = truncation_bilinear_tangent(
        narg,
        block.truncated,
        x_hamiltonian,
        y_hamiltonian,
        xy_hamiltonian,
        include_retained_mixing=include_retained_mixing,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
    )

    xops = (
        dict(source_x_cached.reduced_operators)
        if source_x_cached is not None
        else {}
    )
    yops = (
        dict(source_y_cached.reduced_operators)
        if source_y_cached is not None
        else {}
    )
    xyops = {}
    for key, tensor in block.reduced_operators.items():
        source_tensor = _two_site_source_reduced_tensor(narg, key)
        if source_tensor is None:
            continue
        zero = _zero_reduced_like(source_tensor)
        if source_x_cached is None:
            xops[key] = rotate_reduced_tensor_tangent(
                block.truncated,
                source_tensor,
                zero,
                response.x,
            )
        if source_y_cached is None:
            yops[key] = rotate_reduced_tensor_tangent(
                block.truncated,
                source_tensor,
                zero,
                response.y,
            )
        xyops[key] = rotate_reduced_tensor_bilinear(
            block.truncated,
            source_tensor,
            zero,
            zero,
            zero,
            response,
        )

    for i in range(2):
        for j in range(2):
            for name, builder in (
                ("Density", _density_bilinear),
                ("SpinDensity", _spin_density_bilinear),
            ):
                key = (name, i, j)
                if key not in block.reduced_operators or key in xyops:
                    continue
                xyops[key] = builder(
                    block.reduced_operators,
                    xops,
                    yops,
                    xyops,
                    {},
                    {},
                    {},
                    {},
                    i,
                    j,
                )
                if source_x_cached is None and key not in xops:
                    if name == "Density":
                        xops[key] = _density_tangent(block.reduced_operators, xops, {}, {}, i, j)
                    else:
                        xops[key] = _spin_density_tangent(block.reduced_operators, xops, {}, {}, i, j)
                if source_y_cached is None and key not in yops:
                    if name == "Density":
                        yops[key] = _density_tangent(block.reduced_operators, yops, {}, {}, i, j)
                    else:
                        yops[key] = _spin_density_tangent(block.reduced_operators, yops, {}, {}, i, j)

    future_sites = tuple(range(2, int(final_size)))
    if source_x_cached is None:
        xpackages = _weighted_packages_tangent(
            block.reduced_operators,
            xops,
            h1e_full,
            xdh1e_full,
            eri_full,
            xderi_full,
            2,
            future_sites,
            build_v1=not bool(project_v1_packages),
        )
        xops.update(xpackages)
    if source_y_cached is None:
        ypackages = _weighted_packages_tangent(
            block.reduced_operators,
            yops,
            h1e_full,
            ydh1e_full,
            eri_full,
            yderi_full,
            2,
            future_sites,
            build_v1=not bool(project_v1_packages),
        )
        yops.update(ypackages)
    if project_v1_packages and source_x_cached is None:
        xops.update(
            _component_v1_packages_tangent(
                block,
                xops,
                h1e_full,
                xdh1e_full,
                eri_full,
                xderi_full,
                2,
                future_sites,
            )
        )
    if project_v1_packages and source_y_cached is None:
        yops.update(
            _component_v1_packages_tangent(
                block,
                yops,
                h1e_full,
                ydh1e_full,
                eri_full,
                yderi_full,
                2,
                future_sites,
            )
        )
    xypackages = _weighted_packages_bilinear(
        block.reduced_operators,
        xops,
        yops,
        xyops,
        h1e_full,
        xdh1e_full,
        ydh1e_full,
        xydh1e_full,
        eri_full,
        xderi_full,
        yderi_full,
        xyderi_full,
        2,
        future_sites,
        build_v1=not bool(project_v1_packages),
    )
    xyops.update(xypackages)
    if project_v1_packages:
        xyops.update(
            _component_v1_packages_bilinear(
                block,
                xops,
                yops,
                xyops,
                h1e_full,
                xdh1e_full,
                ydh1e_full,
                xydh1e_full,
                eri_full,
                xderi_full,
                yderi_full,
                xyderi_full,
                2,
                future_sites,
            )
        )
    source_x = (
        source_x_cached
        if source_x_cached is not None
        else _make_tangent_block(block, response.x.d_hamiltonian, xops)
    )
    source_y = (
        source_y_cached
        if source_y_cached is not None
        else _make_tangent_block(block, response.y.d_hamiltonian, yops)
    )
    source_xy = _make_tangent_block(block, response.dxy_hamiltonian, xyops)
    return source_x, source_y, source_xy, response.min_gap


def _seed_two_site_bilinear_adjoint_x(
    block,
    source_x,
    source_y,
    source_xy,
    source_x_hamiltonian_adjoint,
    source_x_operator_adjoint,
    source_xy_hamiltonian_adjoint,
    source_xy_operator_adjoint,
    h1e_full,
    eri_full,
    ydh1e_full,
    yderi_full,
    response: TruncationBilinearTangent,
    *,
    final_size,
    project_v1_packages=False,
    include_retained_mixing=True,
):
    """Adjoint of ``_seed_two_site_bilinear_block`` wrt ``x``/``xy`` integrals."""
    if not project_v1_packages:
        raise NotImplementedError(
            "seed bilinear adjoint is currently implemented for projected V1 packages"
        )
    from .su2_two_site import build_two_site_su2_narg

    narg = block.truncated.source
    future_sites = tuple(range(2, int(final_size)))
    if source_x_hamiltonian_adjoint is None:
        source_x_hamiltonian_adjoint = _zero_irrep_tensor_like(source_x.hamiltonian)
    xh1_adj = np.zeros_like(h1e_full, dtype=float)
    xeri_adj = np.zeros_like(eri_full, dtype=float)
    xyh1_adj = np.zeros_like(h1e_full, dtype=float)
    xyeri_adj = np.zeros_like(eri_full, dtype=float)

    xops_adjoint = dict(source_x_operator_adjoint or {})
    xyops_adjoint = dict(source_xy_operator_adjoint or {})
    weighted_adjoint = {}
    v1_adjoint = {}
    next_prefixes = {
        "NextDensity",
        "NextExchangeDensity",
        "NextExchangeSpinDensity",
        "NextPairAnnihilate",
        "NextV1Spinor",
        "NextV3Cdag",
    }
    for key in list(xyops_adjoint):
        if not isinstance(key, tuple) or key[0] not in next_prefixes:
            continue
        adj = xyops_adjoint.pop(key)
        if key[0] == "NextV1Spinor":
            v1_adjoint[key] = adj
        else:
            weighted_adjoint[key] = adj

    if weighted_adjoint:
        (
            xop_adj,
            xyop_adj,
            h_adj,
            g_adj,
            xy_h_adj,
            xy_g_adj,
        ) = _weighted_packages_bilinear_adjoint_x(
            block.reduced_operators,
            source_x.reduced_operators,
            source_y.reduced_operators,
            source_xy.reduced_operators,
            weighted_adjoint,
            h1e_full,
            eri_full,
            yderi_full,
            2,
            future_sites,
            build_v1=False,
        )
        _merge_reduced_adjoint(xops_adjoint, xop_adj)
        _merge_reduced_adjoint(xyops_adjoint, xyop_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
        xyh1_adj += xy_h_adj
        xyeri_adj += xy_g_adj

    if v1_adjoint:
        (
            xop_adj,
            xyop_adj,
            h_adj,
            g_adj,
            xy_h_adj,
            xy_g_adj,
        ) = _component_v1_packages_bilinear_adjoint_x(
            block,
            source_x.reduced_operators,
            source_y.reduced_operators,
            source_xy.reduced_operators,
            v1_adjoint,
            h1e_full,
            eri_full,
            ydh1e_full,
            yderi_full,
            2,
            future_sites,
        )
        _merge_reduced_adjoint(xops_adjoint, xop_adj)
        _merge_reduced_adjoint(xyops_adjoint, xyop_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
        xyh1_adj += xy_h_adj
        xyeri_adj += xy_g_adj

    source_tensors = {}
    for key in block.reduced_operators:
        tensor = _two_site_source_reduced_tensor(narg, key)
        if tensor is not None:
            source_tensors[key] = tensor

    while True:
        algebraic = [
            key
            for key in xyops_adjoint
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] in {"Density", "SpinDensity"}
                and key not in source_tensors
            )
        ]
        if not algebraic:
            break
        for key in algebraic:
            adj = xyops_adjoint.pop(key)
            xy_without_direct = dict(source_xy.reduced_operators)
            xy_without_direct.pop(key, None)
            if key[0] == "Density":
                xop_adj, xyop_adj = _density_bilinear_adjoint_x(
                    block.reduced_operators,
                    source_x.reduced_operators,
                    source_y.reduced_operators,
                    xy_without_direct,
                    adj,
                    key[1],
                    key[2],
                )
            else:
                xop_adj, xyop_adj = _spin_density_bilinear_adjoint_x(
                    block.reduced_operators,
                    source_x.reduced_operators,
                    source_y.reduced_operators,
                    xy_without_direct,
                    adj,
                    key[1],
                    key[2],
                )
            _merge_reduced_adjoint(xops_adjoint, xop_adj)
            _merge_reduced_adjoint(xyops_adjoint, xyop_adj)

    direct_adjoint = {
        key: adj for key, adj in xyops_adjoint.items() if key in source_tensors
    }
    zero_ytensors = {
        key: _zero_reduced_like(tensor) for key, tensor in source_tensors.items()
    }
    (
        _tensor_x_adjoint,
        _tensor_xy_adjoint,
        transform_x_adjoint,
        transform_xy_adjoint,
    ) = rotate_reduced_tensors_bilinear_adjoint_x(
        block.truncated,
        source_tensors,
        zero_ytensors,
        direct_adjoint,
        response,
    )

    perturbation_x_adjoint, perturbation_xy_adjoint = (
        truncation_bilinear_tangent_adjoint_x(
            narg,
            block.truncated,
            response,
            build_two_site_su2_narg(
                ydh1e_full[:2, :2],
                yderi_full[:2, :2, :2, :2],
            ).hamiltonian,
            transform_x_adjoint_blocks=transform_x_adjoint,
            transform_xy_adjoint_blocks=transform_xy_adjoint,
            hamiltonian_xy_adjoint=source_xy_hamiltonian_adjoint,
            include_retained_mixing=include_retained_mixing,
        )
    )
    h_seed_adj, g_seed_adj = _two_site_hamiltonian_integral_adjoint(
        perturbation_x_adjoint
    )
    xh1_adj[:2, :2] += h_seed_adj
    xeri_adj[:2, :2, :2, :2] += g_seed_adj
    h_seed_adj, g_seed_adj = _two_site_hamiltonian_integral_adjoint(
        perturbation_xy_adjoint
    )
    xyh1_adj[:2, :2] += h_seed_adj
    xyeri_adj[:2, :2, :2, :2] += g_seed_adj

    x_from_source_h, x_from_source_g = _seed_two_site_tangent_adjoint(
        block,
        source_x,
        source_x_hamiltonian_adjoint,
        xops_adjoint,
        h1e_full,
        eri_full,
        final_size=final_size,
        project_v1_packages=project_v1_packages,
        include_retained_mixing=include_retained_mixing,
    )
    xh1_adj += x_from_source_h
    xeri_adj += x_from_source_g
    return xh1_adj, xeri_adj, xyh1_adj, xyeri_adj


def _grown_coupling_operators_tangent(
    narg,
    source_tangent,
    *,
    include_even_composites=True,
    even_composites=None,
    cache_result: bool = False,
):
    if not _operator_dict_has_nonzero_blocks(
        getattr(source_tangent, "reduced_operators", {})
    ):
        return {}

    from .su2_chain import block_identity_reduced_tensor
    from .su2_three_site import local_reduced_operator, reduced_product_tensor_irrep

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")
    even_key = None if even_composites is None else tuple(sorted(even_composites))
    cache_key = (id(source_tangent), bool(include_even_composites), even_key)
    if cache_result:
        cache = getattr(source_block, "_su2_response_grown_coupling_tangent_cache", None)
        if cache is None:
            cache = {}
            setattr(source_block, "_su2_response_grown_coupling_tangent_cache", cache)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is source_tangent:
            return dict(cached[1])
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    nsites = old_site_indices[-1] + 2
    new_site_index = nsites - 1
    out = {}
    local_jw = local_reduced_operator("JW")
    for site_index in old_site_indices:
        for name in ("Cdag", "Ctilde"):
            key = (name, site_index)
            dtensor = source_tangent.reduced_operators.get(key)
            if dtensor is None:
                continue
            out[key] = reduced_product_tensor_irrep(
                source_block,
                dtensor,
                local_jw,
                total_rank2=1,
            )
    if include_even_composites:
        if even_composites is None:
            even_composites = {"Density", "SpinDensity", "PairCreate0", "PairCreate2"}
        else:
            even_composites = set(even_composites)
        local_identity = local_reduced_operator("I")
        for key, dtensor in source_tangent.reduced_operators.items():
            if not isinstance(key, tuple) or key[0] not in even_composites:
                continue
            out[key] = reduced_product_tensor_irrep(
                source_block,
                dtensor,
                local_identity,
                total_rank2=dtensor.op.charge[1],
            )
    # New local-site creation/annihilation operators are identity-block products;
    # U^\dag I U is constant, so their tangent is zero.
    for name, op_key in (("Cdag", "Cdag"), ("Ctilde", "Ctilde")):
        del name, op_key
    del block_identity_reduced_tensor, new_site_index
    if cache_result:
        cache[cache_key] = (source_tangent, dict(out))
    return out


def _grown_coupling_operators_tangent_adjoint(
    narg,
    source_tangent,
    grown_tangent_adjoint,
    *,
    include_even_composites=True,
    even_composites=None,
):
    from .su2_three_site import local_reduced_operator

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")
    out = {}

    def add(key, value):
        if not value.blocks:
            return
        if key in out:
            out[key] = add_reduced_tensors(out[key], value)
        else:
            out[key] = value

    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    local_jw = local_reduced_operator("JW")
    for site_index in old_site_indices:
        for name in ("Cdag", "Ctilde"):
            key = (name, site_index)
            adj = grown_tangent_adjoint.get(key)
            tensor = source_tangent.reduced_operators.get(key)
            if adj is None or tensor is None:
                continue
            add(
                key,
                reduced_product_tensor_block_adjoint(
                    source_block,
                    tensor,
                    local_jw,
                    adj,
                    total_rank2=1,
                ),
            )

    if include_even_composites:
        if even_composites is None:
            even_composites = {"Density", "SpinDensity", "PairCreate0", "PairCreate2"}
        else:
            even_composites = set(even_composites)
        local_identity = local_reduced_operator("I")
        for key, adj in grown_tangent_adjoint.items():
            if not isinstance(key, tuple) or key[0] not in even_composites:
                continue
            tensor = source_tangent.reduced_operators.get(key)
            if tensor is None:
                continue
            add(
                key,
                reduced_product_tensor_block_adjoint(
                    source_block,
                    tensor,
                    local_identity,
                    adj,
                    total_rank2=tensor.op.charge[1],
                ),
            )
    return out


def _product_with_local_tangent(source_block, tensor, dtensor, local, rank2):
    if dtensor is None or not dtensor.blocks:
        if tensor is None:
            return None
        return _zero_reduced(
            tensor.site,
            OpIrrep((tensor.op.charge[0] + local.op.charge[0], int(rank2))),
        )
    from .su2_three_site import reduced_product_tensor_irrep

    return reduced_product_tensor_irrep(
        source_block,
        dtensor,
        local,
        total_rank2=int(rank2),
    )


def _product_with_local_bilinear(source_block, tensor, xytensor, local, rank2):
    return _product_with_local_tangent(
        source_block,
        tensor,
        xytensor,
        local,
        int(rank2),
    )


def _cached_grown_reduced_v1_packages(source_block, grown, h1e, eri, future_sites):
    """Cache base projected V1 packages reused by recursive response tangents."""
    from .su2_chain import grown_reduced_v1_packages

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    key = (int(site_count), tuple(int(q) for q in future_sites), id(h1e), id(eri))
    cache = getattr(source_block, "_su2_response_grown_v1_cache", None)
    if cache is None:
        cache = {}
        setattr(source_block, "_su2_response_grown_v1_cache", cache)
    actual = cache.get(key)
    if actual is None:
        actual = grown_reduced_v1_packages(
            source_block,
            grown,
            h1e,
            eri,
            future_sites,
        )
        cache[key] = actual
    return actual


_NEW_SITE_WEIGHTED_PACKAGE_CACHE = {}


def _cached_new_site_weighted_packages(
    operators,
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = False,
):
    """Cache base new-site weighted packages reused by response tangents."""
    from .su2_chain import new_site_weighted_packages_from_operators

    future_key = tuple(int(q) for q in future_sites)
    operator_signature = tuple(
        (repr(key), id(tensor))
        for key, tensor in sorted(operators.items(), key=lambda item: repr(item[0]))
    )
    key = (
        operator_signature,
        id(h1e),
        id(eri),
        int(site_count),
        future_key,
        bool(build_v1),
    )
    owner = next(iter(operators.values()), None)
    cache = None
    if owner is not None:
        try:
            cache = getattr(owner, "_su2_response_new_site_weighted_cache", None)
            if cache is None:
                cache = {}
                setattr(owner, "_su2_response_new_site_weighted_cache", cache)
        except Exception:
            cache = None
    if cache is None:
        cache = _NEW_SITE_WEIGHTED_PACKAGE_CACHE
    cached = cache.get(key)
    if cached is None:
        cached = new_site_weighted_packages_from_operators(
            operators,
            h1e,
            eri,
            int(site_count),
            future_key,
            build_v1=bool(build_v1),
        )
        cache[key] = cached
    return dict(cached)


def _cached_context_new_site_weighted_packages(
    owner,
    cache_key,
    operators,
    h1e,
    eri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = False,
):
    """Cache base new-site packages with a caller-provided context key."""
    from .su2_chain import new_site_weighted_packages_from_operators

    future_key = tuple(int(q) for q in future_sites)
    key = (
        cache_key,
        id(h1e),
        id(eri),
        int(site_count),
        future_key,
        bool(build_v1),
    )
    cache = getattr(owner, "_su2_response_context_new_site_weighted_cache", None)
    if cache is None:
        cache = {}
        setattr(owner, "_su2_response_context_new_site_weighted_cache", cache)
    cached = cache.get(key)
    if cached is None:
        cached = new_site_weighted_packages_from_operators(
            operators,
            h1e,
            eri,
            int(site_count),
            future_key,
            build_v1=bool(build_v1),
        )
        cache[key] = cached
    return dict(cached)


def _cached_grow_source_tensor(source_block, tensor, *, local_name: str):
    """Cache repeated source-block growth of carried response packages."""
    from .su2_chain import grow_source_tensor

    cache = getattr(source_block, "_su2_response_grown_source_tensor_cache", None)
    if cache is None:
        cache = {}
        setattr(source_block, "_su2_response_grown_source_tensor_cache", cache)
    key = (str(local_name), id(tensor))
    cached = cache.get(key)
    if cached is not None and cached[0] is tensor:
        return cached[1]
    grown = grow_source_tensor(source_block, tensor, local_name=local_name)
    cache[key] = (tensor, grown)
    return grown


def _grown_reduced_v1_packages_tangent(
    source_block,
    source_tangent,
    grown,
    h1e,
    dh1e,
    eri,
    deri,
    future_sites,
):
    """Tangent of ``grown_reduced_v1_packages`` for projected V1 packages."""
    actual = _cached_grown_reduced_v1_packages(
        source_block,
        grown,
        h1e,
        eri,
        future_sites,
    )
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        return actual, {}

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_site = site_count - 1
    if old_site_indices[-1] + 1 != new_site:
        raise ValueError("grown V1 tangent source/grown site counts disagree")
    if (
        not _operator_dict_has_nonzero_blocks(source_tangent.reduced_operators)
        and not _grown_v1_coefficients_have_support(
            dh1e,
            deri,
            site_count,
            future_sites,
        )
    ):
        return actual, {}

    from .su2_chain import (
        _local_v1_reduced_operator,
        block_identity_reduced_tensor,
    )
    from .su2_three_site import local_reduced_operator, reduced_product_tensor_irrep

    block_identity = block_identity_reduced_tensor(source_block)
    local_cdag = local_reduced_operator("Cdag")
    local_ctilde = local_reduced_operator("Ctilde")
    local_pair_create = local_reduced_operator("PairCreate")
    local_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeDensity")
    local_spin_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeSpinDensity")
    local_jw_density = _local_v1_reduced_operator("JWDensity")
    local_cdag_density = _local_v1_reduced_operator("CdagDensity")

    identity_cdag = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag,
        total_rank2=1,
    )
    identity_cdag_density = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag_density,
        total_rank2=1,
    )
    zero_identity_cdag = _zero_reduced_like(identity_cdag)
    zero_identity_cdag_density = _zero_reduced_like(identity_cdag_density)

    product_cache: dict[
        tuple,
        tuple[ReducedSU2Tensor | None, ReducedSU2Tensor | None, bool],
    ] = {}

    def source_tensor(key: tuple) -> ReducedSU2Tensor | None:
        return source_block.reduced_operators.get(key)

    def dsource_tensor(key: tuple) -> ReducedSU2Tensor | None:
        return source_tangent.reduced_operators.get(key)

    def product(
        key: tuple,
        tensor_key: tuple,
        local: ReducedSU2Tensor,
        rank2: int,
        *,
        need_actual: bool,
    ):
        tensor = source_tensor(tensor_key)
        if key in product_cache:
            actual_tensor, tangent_tensor, actual_ready = product_cache[key]
            if need_actual and not actual_ready and tensor is not None:
                actual_tensor = reduced_product_tensor_irrep(
                    source_block,
                    tensor,
                    local,
                    total_rank2=int(rank2),
                )
                actual_ready = True
                product_cache[key] = (actual_tensor, tangent_tensor, actual_ready)
            return actual_tensor, tangent_tensor
        dtensor = dsource_tensor(tensor_key)
        actual_ready = False
        actual_tensor = tensor
        if need_actual and tensor is not None:
            actual_tensor = reduced_product_tensor_irrep(
                source_block,
                tensor,
                local,
                total_rank2=int(rank2),
            )
            actual_ready = True
        tangent_tensor = _product_with_local_tangent(
            source_block,
            tensor,
            dtensor,
            local,
            int(rank2),
        )
        product_cache[key] = (actual_tensor, tangent_tensor, actual_ready)
        return actual_tensor, tangent_tensor

    out = {}
    for q in future_sites:
        q = int(q)
        terms = []

        coeff = h1e[new_site, q]
        dcoeff = dh1e[new_site, q]
        if abs(coeff) > 0.0 or abs(dcoeff) > 0.0:
            terms.append((identity_cdag, zero_identity_cdag, coeff, dcoeff))

        for i in range(site_count):
            for j in range(site_count):
                for k in range(site_count):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    dcoeff = deri[k, q, j, i]
                    if abs(coeff) <= 0.0 and abs(dcoeff) <= 0.0:
                        continue

                    if k == new_site and j < new_site and i < new_site:
                        tensor, dtensor = product(
                            ("DensityCdag", j, i),
                            ("Density", j, i),
                            local_cdag,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append((tensor, dtensor, coeff, dcoeff))
                    elif k < new_site and j == new_site and i < new_site:
                        tensor, dtensor = product(
                            ("HopDensityCdag", k, i),
                            ("Density", k, i),
                            local_cdag,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append((tensor, dtensor, -0.5 * coeff, -0.5 * dcoeff))
                        tensor, dtensor = product(
                            ("HopSpinDensityCdag", k, i),
                            ("SpinDensity", k, i),
                            local_cdag,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append(
                                (
                                    tensor,
                                    dtensor,
                                    np.sqrt(1.5) * coeff,
                                    np.sqrt(1.5) * dcoeff,
                                )
                            )
                    elif k < new_site and j < new_site and i == new_site:
                        tensor, dtensor = product(
                            ("Pair0Ctilde", k, j),
                            ("PairCreate0", k, j),
                            local_ctilde,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append(
                                (
                                    tensor,
                                    dtensor,
                                    -coeff / np.sqrt(2.0),
                                    -dcoeff / np.sqrt(2.0),
                                )
                            )
                        tensor, dtensor = product(
                            ("Pair2Ctilde", k, j),
                            ("PairCreate2", k, j),
                            local_ctilde,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append(
                                (
                                    tensor,
                                    dtensor,
                                    np.sqrt(1.5) * coeff,
                                    np.sqrt(1.5) * dcoeff,
                                )
                            )
                    elif k == new_site and j == new_site and i < new_site:
                        tensor, dtensor = product(
                            ("CtildePairCreate", i),
                            ("Ctilde", i),
                            local_pair_create,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append((tensor, dtensor, -coeff, -dcoeff))
                    elif k == new_site and j < new_site and i == new_site:
                        tensor, dtensor = product(
                            ("CdagDensityMidJW", j),
                            ("Cdag", j),
                            local_density_mid_jw,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append((tensor, dtensor, 0.5 * coeff, 0.5 * dcoeff))
                        tensor, dtensor = product(
                            ("CdagSpinDensityMidJW", j),
                            ("Cdag", j),
                            local_spin_density_mid_jw,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append(
                                (
                                    tensor,
                                    dtensor,
                                    np.sqrt(1.5) * coeff,
                                    np.sqrt(1.5) * dcoeff,
                                )
                            )
                    elif k < new_site and j == new_site and i == new_site:
                        tensor, dtensor = product(
                            ("CdagJWDensity", k),
                            ("Cdag", k),
                            local_jw_density,
                            1,
                            need_actual=abs(dcoeff) > 0.0,
                        )
                        if tensor is not None:
                            terms.append((tensor, dtensor, coeff, dcoeff))
                    else:
                        terms.append(
                            (
                                identity_cdag_density,
                                zero_identity_cdag_density,
                                coeff,
                                dcoeff,
                            )
                        )

        total = _weighted_tangent_terms(terms)
        if total is not None:
            out[("NextV1Spinor", q)] = total
    return actual, out


def _grown_reduced_v1_packages_tangent_adjoint(
    source_block,
    source_tangent,
    grown,
    package_adjoint,
    h1e,
    eri,
    future_sites,
):
    """Adjoint of ``_grown_reduced_v1_packages_tangent``."""
    from .su2_chain import (
        _local_v1_reduced_operator,
        block_identity_reduced_tensor,
    )
    from .su2_three_site import local_reduced_operator, reduced_product_tensor_irrep

    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    source_operator_adjoint = {}
    dh1_adj = np.zeros_like(h1e, dtype=float)
    deri_adj = np.zeros_like(eri, dtype=float)
    if not old_site_indices:
        return source_operator_adjoint, dh1_adj, deri_adj

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_site = site_count - 1
    if old_site_indices[-1] + 1 != new_site:
        raise ValueError("grown V1 adjoint source/grown site counts disagree")

    block_identity = block_identity_reduced_tensor(source_block)
    local_cdag = local_reduced_operator("Cdag")
    local_ctilde = local_reduced_operator("Ctilde")
    local_pair_create = local_reduced_operator("PairCreate")
    local_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeDensity")
    local_spin_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeSpinDensity")
    local_jw_density = _local_v1_reduced_operator("JWDensity")
    local_cdag_density = _local_v1_reduced_operator("CdagDensity")

    identity_cdag = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag,
        total_rank2=1,
    )
    identity_cdag_density = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag_density,
        total_rank2=1,
    )
    product_cache = {}

    def product_tensor(cache_key, tensor_key, local, rank2):
        if cache_key in product_cache:
            return product_cache[cache_key]
        tensor = source_block.reduced_operators.get(tensor_key)
        if tensor is None:
            product_cache[cache_key] = None
            return None
        out = reduced_product_tensor_irrep(
            source_block,
            tensor,
            local,
            total_rank2=int(rank2),
        )
        product_cache[cache_key] = out
        return out

    def product_term(adj, cache_key, tensor_key, local, rank2, coeff, scale, index):
        tensor = product_tensor(cache_key, tensor_key, local, rank2)
        if tensor is None:
            return
        deri_adj[index] += scale * _reduced_tensor_pairing(adj, tensor)
        dtensor = source_tangent.reduced_operators.get(tensor_key)
        if dtensor is None:
            return
        _add_reduced_adjoint(
            source_operator_adjoint,
            tensor_key,
            reduced_product_tensor_block_adjoint(
                source_block,
                dtensor,
                local,
                _scale_adjoint(adj, scale * coeff),
                total_rank2=int(rank2),
            ),
        )

    future_sites = tuple(int(q) for q in future_sites)
    for q in future_sites:
        adj = package_adjoint.get(("NextV1Spinor", q))
        if adj is None:
            continue
        dh1_adj[new_site, q] += _reduced_tensor_pairing(adj, identity_cdag)

        for i in range(site_count):
            for j in range(site_count):
                for k in range(site_count):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    index = (k, q, j, i)

                    if k == new_site and j < new_site and i < new_site:
                        product_term(
                            adj,
                            ("DensityCdag", j, i),
                            ("Density", j, i),
                            local_cdag,
                            1,
                            coeff,
                            1.0,
                            index,
                        )
                    elif k < new_site and j == new_site and i < new_site:
                        product_term(
                            adj,
                            ("HopDensityCdag", k, i),
                            ("Density", k, i),
                            local_cdag,
                            1,
                            coeff,
                            -0.5,
                            index,
                        )
                        product_term(
                            adj,
                            ("HopSpinDensityCdag", k, i),
                            ("SpinDensity", k, i),
                            local_cdag,
                            1,
                            coeff,
                            np.sqrt(1.5),
                            index,
                        )
                    elif k < new_site and j < new_site and i == new_site:
                        product_term(
                            adj,
                            ("Pair0Ctilde", k, j),
                            ("PairCreate0", k, j),
                            local_ctilde,
                            1,
                            coeff,
                            -1.0 / np.sqrt(2.0),
                            index,
                        )
                        product_term(
                            adj,
                            ("Pair2Ctilde", k, j),
                            ("PairCreate2", k, j),
                            local_ctilde,
                            1,
                            coeff,
                            np.sqrt(1.5),
                            index,
                        )
                    elif k == new_site and j == new_site and i < new_site:
                        product_term(
                            adj,
                            ("CtildePairCreate", i),
                            ("Ctilde", i),
                            local_pair_create,
                            1,
                            coeff,
                            -1.0,
                            index,
                        )
                    elif k == new_site and j < new_site and i == new_site:
                        product_term(
                            adj,
                            ("CdagDensityMidJW", j),
                            ("Cdag", j),
                            local_density_mid_jw,
                            1,
                            coeff,
                            0.5,
                            index,
                        )
                        product_term(
                            adj,
                            ("CdagSpinDensityMidJW", j),
                            ("Cdag", j),
                            local_spin_density_mid_jw,
                            1,
                            coeff,
                            np.sqrt(1.5),
                            index,
                        )
                    elif k < new_site and j == new_site and i == new_site:
                        product_term(
                            adj,
                            ("CdagJWDensity", k),
                            ("Cdag", k),
                            local_jw_density,
                            1,
                            coeff,
                            1.0,
                            index,
                        )
                    else:
                        deri_adj[index] += _reduced_tensor_pairing(
                            adj,
                            identity_cdag_density,
                        )

    return source_operator_adjoint, dh1_adj, deri_adj


def _grown_reduced_v1_packages_bilinear(
    source_block,
    source_x,
    source_y,
    source_xy,
    grown,
    h1e,
    xh1e,
    yh1e,
    xyh1e,
    eri,
    xeri,
    yeri,
    xyeri,
    future_sites,
):
    """Mixed tangent of projected grown V1 packages."""
    from .su2_chain import (
        _local_v1_reduced_operator,
        block_identity_reduced_tensor,
    )
    from .su2_three_site import local_reduced_operator, reduced_product_tensor_irrep

    actual = _cached_grown_reduced_v1_packages(
        source_block,
        grown,
        h1e,
        eri,
        future_sites,
    )
    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    if not old_site_indices:
        return actual, {}

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_site = site_count - 1
    if old_site_indices[-1] + 1 != new_site:
        raise ValueError("grown V1 bilinear source/grown site counts disagree")

    block_identity = block_identity_reduced_tensor(source_block)
    local_cdag = local_reduced_operator("Cdag")
    local_ctilde = local_reduced_operator("Ctilde")
    local_pair_create = local_reduced_operator("PairCreate")
    local_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeDensity")
    local_spin_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeSpinDensity")
    local_jw_density = _local_v1_reduced_operator("JWDensity")
    local_cdag_density = _local_v1_reduced_operator("CdagDensity")

    identity_cdag = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag,
        total_rank2=1,
    )
    identity_cdag_density = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag_density,
        total_rank2=1,
    )
    zero_identity_cdag = _zero_reduced_like(identity_cdag)
    zero_identity_cdag_density = _zero_reduced_like(identity_cdag_density)

    product_cache: dict[
        tuple,
        tuple[
            ReducedSU2Tensor | None,
            ReducedSU2Tensor | None,
            ReducedSU2Tensor | None,
            ReducedSU2Tensor | None,
            tuple[bool, bool, bool, bool],
        ],
    ] = {}

    def source_tensor(key: tuple) -> ReducedSU2Tensor | None:
        return source_block.reduced_operators.get(key)

    def product(
        key: tuple,
        tensor_key: tuple,
        local: ReducedSU2Tensor,
        rank2: int,
        *,
        need_actual: bool,
        need_x: bool,
        need_y: bool,
        need_xy: bool,
    ):
        tensor = source_tensor(tensor_key)
        if key in product_cache:
            actual_tensor, xproduct, yproduct, xyproduct, ready = product_cache[key]
            actual_ready, x_ready, y_ready, xy_ready = ready
            changed = False
            if need_actual and not actual_ready and tensor is not None:
                actual_tensor = reduced_product_tensor_irrep(
                    source_block,
                    tensor,
                    local,
                    total_rank2=int(rank2),
                )
                actual_ready = True
                changed = True
            if need_x and not x_ready:
                xproduct = _product_with_local_tangent(
                    source_block,
                    tensor,
                    source_x.reduced_operators.get(tensor_key),
                    local,
                    int(rank2),
                )
                x_ready = True
                changed = True
            if need_y and not y_ready:
                yproduct = _product_with_local_tangent(
                    source_block,
                    tensor,
                    source_y.reduced_operators.get(tensor_key),
                    local,
                    int(rank2),
                )
                y_ready = True
                changed = True
            if need_xy and not xy_ready:
                xyproduct = _product_with_local_bilinear(
                    source_block,
                    tensor,
                    source_xy.reduced_operators.get(tensor_key),
                    local,
                    int(rank2),
                )
                xy_ready = True
                changed = True
            if changed:
                product_cache[key] = (
                    actual_tensor,
                    xproduct,
                    yproduct,
                    xyproduct,
                    (actual_ready, x_ready, y_ready, xy_ready),
                )
            return actual_tensor, xproduct, yproduct, xyproduct
        xtensor = source_x.reduced_operators.get(tensor_key)
        ytensor = source_y.reduced_operators.get(tensor_key)
        xytensor = source_xy.reduced_operators.get(tensor_key)
        actual_tensor = tensor
        actual_ready = False
        if need_actual and tensor is not None:
            actual_tensor = reduced_product_tensor_irrep(
                source_block,
                tensor,
                local,
                total_rank2=int(rank2),
            )
            actual_ready = True
        xproduct = (
            _product_with_local_tangent(
                source_block,
                tensor,
                xtensor,
                local,
                int(rank2),
            )
            if need_x
            else None
        )
        yproduct = (
            _product_with_local_tangent(
                source_block,
                tensor,
                ytensor,
                local,
                int(rank2),
            )
            if need_y
            else None
        )
        xyproduct = (
            _product_with_local_bilinear(
                source_block,
                tensor,
                xytensor,
                local,
                int(rank2),
            )
            if need_xy
            else None
        )
        product_cache[key] = (
            actual_tensor,
            xproduct,
            yproduct,
            xyproduct,
            (actual_ready, bool(need_x), bool(need_y), bool(need_xy)),
        )
        return actual_tensor, xproduct, yproduct, xyproduct

    def product_for_coeffs(
        key: tuple,
        tensor_key: tuple,
        local: ReducedSU2Tensor,
        rank2: int,
        coeff,
        xcoeff,
        ycoeff,
        xycoeff,
    ):
        return product(
            key,
            tensor_key,
            local,
            rank2,
            need_actual=abs(xycoeff) > 0.0,
            need_x=abs(ycoeff) > 0.0,
            need_y=abs(xcoeff) > 0.0,
            need_xy=abs(coeff) > 0.0,
        )

    out = {}
    for q in future_sites:
        q = int(q)
        terms = []

        coeff = h1e[new_site, q]
        xcoeff = xh1e[new_site, q]
        ycoeff = yh1e[new_site, q]
        xycoeff = xyh1e[new_site, q]
        if (
            abs(coeff) > 0.0
            or abs(xcoeff) > 0.0
            or abs(ycoeff) > 0.0
            or abs(xycoeff) > 0.0
        ):
            terms.append(
                (
                    identity_cdag,
                    zero_identity_cdag,
                    zero_identity_cdag,
                    zero_identity_cdag,
                    coeff,
                    xcoeff,
                    ycoeff,
                    xycoeff,
                )
            )

        for i in range(site_count):
            for j in range(site_count):
                for k in range(site_count):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    xcoeff = xeri[k, q, j, i]
                    ycoeff = yeri[k, q, j, i]
                    xycoeff = xyeri[k, q, j, i]
                    if (
                        abs(coeff) <= 0.0
                        and abs(xcoeff) <= 0.0
                        and abs(ycoeff) <= 0.0
                        and abs(xycoeff) <= 0.0
                    ):
                        continue

                    if k == new_site and j < new_site and i < new_site:
                        tensors = product_for_coeffs(
                            ("DensityCdag", j, i),
                            ("Density", j, i),
                            local_cdag,
                            1,
                            coeff,
                            xcoeff,
                            ycoeff,
                            xycoeff,
                        )
                        if tensors[0] is not None:
                            terms.append((*tensors, coeff, xcoeff, ycoeff, xycoeff))
                    elif k < new_site and j == new_site and i < new_site:
                        tensors = product_for_coeffs(
                            ("HopDensityCdag", k, i),
                            ("Density", k, i),
                            local_cdag,
                            1,
                            -0.5 * coeff,
                            -0.5 * xcoeff,
                            -0.5 * ycoeff,
                            -0.5 * xycoeff,
                        )
                        if tensors[0] is not None:
                            terms.append(
                                (
                                    *tensors,
                                    -0.5 * coeff,
                                    -0.5 * xcoeff,
                                    -0.5 * ycoeff,
                                    -0.5 * xycoeff,
                                )
                            )
                        tensors = product_for_coeffs(
                            ("HopSpinDensityCdag", k, i),
                            ("SpinDensity", k, i),
                            local_cdag,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * xcoeff,
                            np.sqrt(1.5) * ycoeff,
                            np.sqrt(1.5) * xycoeff,
                        )
                        if tensors[0] is not None:
                            scale = np.sqrt(1.5)
                            terms.append(
                                (
                                    *tensors,
                                    scale * coeff,
                                    scale * xcoeff,
                                    scale * ycoeff,
                                    scale * xycoeff,
                                )
                            )
                    elif k < new_site and j < new_site and i == new_site:
                        tensors = product_for_coeffs(
                            ("Pair0Ctilde", k, j),
                            ("PairCreate0", k, j),
                            local_ctilde,
                            1,
                            -coeff / np.sqrt(2.0),
                            -xcoeff / np.sqrt(2.0),
                            -ycoeff / np.sqrt(2.0),
                            -xycoeff / np.sqrt(2.0),
                        )
                        if tensors[0] is not None:
                            scale = -1.0 / np.sqrt(2.0)
                            terms.append(
                                (
                                    *tensors,
                                    scale * coeff,
                                    scale * xcoeff,
                                    scale * ycoeff,
                                    scale * xycoeff,
                                )
                            )
                        tensors = product_for_coeffs(
                            ("Pair2Ctilde", k, j),
                            ("PairCreate2", k, j),
                            local_ctilde,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * xcoeff,
                            np.sqrt(1.5) * ycoeff,
                            np.sqrt(1.5) * xycoeff,
                        )
                        if tensors[0] is not None:
                            scale = np.sqrt(1.5)
                            terms.append(
                                (
                                    *tensors,
                                    scale * coeff,
                                    scale * xcoeff,
                                    scale * ycoeff,
                                    scale * xycoeff,
                                )
                            )
                    elif k == new_site and j == new_site and i < new_site:
                        tensors = product_for_coeffs(
                            ("CtildePairCreate", i),
                            ("Ctilde", i),
                            local_pair_create,
                            1,
                            -coeff,
                            -xcoeff,
                            -ycoeff,
                            -xycoeff,
                        )
                        if tensors[0] is not None:
                            terms.append(
                                (*tensors, -coeff, -xcoeff, -ycoeff, -xycoeff)
                            )
                    elif k == new_site and j < new_site and i == new_site:
                        tensors = product_for_coeffs(
                            ("CdagDensityMidJW", j),
                            ("Cdag", j),
                            local_density_mid_jw,
                            1,
                            0.5 * coeff,
                            0.5 * xcoeff,
                            0.5 * ycoeff,
                            0.5 * xycoeff,
                        )
                        if tensors[0] is not None:
                            terms.append(
                                (
                                    *tensors,
                                    0.5 * coeff,
                                    0.5 * xcoeff,
                                    0.5 * ycoeff,
                                    0.5 * xycoeff,
                                )
                            )
                        tensors = product_for_coeffs(
                            ("CdagSpinDensityMidJW", j),
                            ("Cdag", j),
                            local_spin_density_mid_jw,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * xcoeff,
                            np.sqrt(1.5) * ycoeff,
                            np.sqrt(1.5) * xycoeff,
                        )
                        if tensors[0] is not None:
                            scale = np.sqrt(1.5)
                            terms.append(
                                (
                                    *tensors,
                                    scale * coeff,
                                    scale * xcoeff,
                                    scale * ycoeff,
                                    scale * xycoeff,
                                )
                            )
                    elif k < new_site and j == new_site and i == new_site:
                        tensors = product_for_coeffs(
                            ("CdagJWDensity", k),
                            ("Cdag", k),
                            local_jw_density,
                            1,
                            coeff,
                            xcoeff,
                            ycoeff,
                            xycoeff,
                        )
                        if tensors[0] is not None:
                            terms.append((*tensors, coeff, xcoeff, ycoeff, xycoeff))
                    else:
                        terms.append(
                            (
                                identity_cdag_density,
                                zero_identity_cdag_density,
                                zero_identity_cdag_density,
                                zero_identity_cdag_density,
                                coeff,
                                xcoeff,
                                ycoeff,
                                xycoeff,
                            )
                        )

        total = _weighted_bilinear_terms(terms)
        if total is not None:
            out[("NextV1Spinor", q)] = total
    return actual, out


def _grown_reduced_v1_packages_bilinear_adjoint_x(
    source_block,
    source_x,
    source_y,
    source_xy,
    grown,
    package_adjoint,
    h1e,
    eri,
    yh1e,
    yeri,
    future_sites,
):
    """Adjoint of projected grown-V1 bilinear packages wrt ``x``/``xy`` inputs."""
    from .su2_chain import (
        _local_v1_reduced_operator,
        block_identity_reduced_tensor,
    )
    from .su2_three_site import local_reduced_operator, reduced_product_tensor_irrep

    old_site_indices = sorted(
        key[1]
        for key in source_block.reduced_operators
        if len(key) == 2 and key[0] == "Cdag"
    )
    xoperator_adjoint = {}
    xyoperator_adjoint = {}
    xh1_adj = np.zeros_like(h1e, dtype=float)
    xyh1_adj = np.zeros_like(h1e, dtype=float)
    xeri_adj = np.zeros_like(eri, dtype=float)
    xyeri_adj = np.zeros_like(eri, dtype=float)
    if not old_site_indices:
        return (
            xoperator_adjoint,
            xyoperator_adjoint,
            xh1_adj,
            xeri_adj,
            xyh1_adj,
            xyeri_adj,
        )

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    new_site = site_count - 1
    if old_site_indices[-1] + 1 != new_site:
        raise ValueError("grown V1 bilinear adjoint source/grown site counts disagree")

    block_identity = block_identity_reduced_tensor(source_block)
    local_cdag = local_reduced_operator("Cdag")
    local_ctilde = local_reduced_operator("Ctilde")
    local_pair_create = local_reduced_operator("PairCreate")
    local_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeDensity")
    local_spin_density_mid_jw = _local_v1_reduced_operator("CdagJWCtildeSpinDensity")
    local_jw_density = _local_v1_reduced_operator("JWDensity")
    local_cdag_density = _local_v1_reduced_operator("CdagDensity")

    identity_cdag = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag,
        total_rank2=1,
    )
    identity_cdag_density = reduced_product_tensor_irrep(
        source_block,
        block_identity,
        local_cdag_density,
        total_rank2=1,
    )
    product_cache = {}

    def product_tensor(cache_key, tensor_key, local, rank2):
        if cache_key in product_cache:
            return product_cache[cache_key]
        tensor = source_block.reduced_operators.get(tensor_key)
        if tensor is None:
            product_cache[cache_key] = None
            return None
        out = reduced_product_tensor_irrep(
            source_block,
            tensor,
            local,
            total_rank2=int(rank2),
        )
        product_cache[cache_key] = out
        return out

    def product_adjoint(source_tangent, tensor_key, local, rank2, adj):
        tensor = source_tangent.reduced_operators.get(tensor_key)
        if tensor is None:
            return None
        return reduced_product_tensor_block_adjoint(
            source_block,
            tensor,
            local,
            adj,
            total_rank2=int(rank2),
        )

    def product_term(
        adj,
        cache_key,
        tensor_key,
        local,
        rank2,
        coeff,
        ycoeff,
        index,
        *,
        scale=1.0,
    ):
        actual = product_tensor(cache_key, tensor_key, local, rank2)
        if actual is None:
            return
        ytensor = source_y.reduced_operators.get(tensor_key)
        yproduct = _product_with_local_tangent(
            source_block,
            source_block.reduced_operators.get(tensor_key),
            ytensor,
            local,
            int(rank2),
        )
        if yproduct is not None:
            xeri_adj[index] += scale * _reduced_tensor_pairing(adj, yproduct)
        xyeri_adj[index] += scale * _reduced_tensor_pairing(adj, actual)
        if abs(ycoeff) > 0.0:
            x_adj = product_adjoint(
                source_x,
                tensor_key,
                local,
                rank2,
                _scale_adjoint(adj, ycoeff),
            )
            _add_reduced_adjoint(xoperator_adjoint, tensor_key, x_adj)
        if abs(coeff) > 0.0:
            xy_adj = product_adjoint(
                source_xy,
                tensor_key,
                local,
                rank2,
                _scale_adjoint(adj, coeff),
            )
            _add_reduced_adjoint(xyoperator_adjoint, tensor_key, xy_adj)

    future_sites = tuple(int(q) for q in future_sites)
    for q in future_sites:
        adj = package_adjoint.get(("NextV1Spinor", q))
        if adj is None:
            continue
        xyh1_adj[new_site, q] += _reduced_tensor_pairing(adj, identity_cdag)

        for i in range(site_count):
            for j in range(site_count):
                for k in range(site_count):
                    if new_site not in (i, j, k):
                        continue
                    coeff = eri[k, q, j, i]
                    ycoeff = yeri[k, q, j, i]
                    index = (k, q, j, i)

                    if k == new_site and j < new_site and i < new_site:
                        product_term(
                            adj,
                            ("DensityCdag", j, i),
                            ("Density", j, i),
                            local_cdag,
                            1,
                            coeff,
                            ycoeff,
                            index,
                        )
                    elif k < new_site and j == new_site and i < new_site:
                        product_term(
                            adj,
                            ("HopDensityCdag", k, i),
                            ("Density", k, i),
                            local_cdag,
                            1,
                            -0.5 * coeff,
                            -0.5 * ycoeff,
                            index,
                            scale=-0.5,
                        )
                        product_term(
                            adj,
                            ("HopSpinDensityCdag", k, i),
                            ("SpinDensity", k, i),
                            local_cdag,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * ycoeff,
                            index,
                            scale=np.sqrt(1.5),
                        )
                    elif k < new_site and j < new_site and i == new_site:
                        product_term(
                            adj,
                            ("Pair0Ctilde", k, j),
                            ("PairCreate0", k, j),
                            local_ctilde,
                            1,
                            -coeff / np.sqrt(2.0),
                            -ycoeff / np.sqrt(2.0),
                            index,
                            scale=-1.0 / np.sqrt(2.0),
                        )
                        product_term(
                            adj,
                            ("Pair2Ctilde", k, j),
                            ("PairCreate2", k, j),
                            local_ctilde,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * ycoeff,
                            index,
                            scale=np.sqrt(1.5),
                        )
                    elif k == new_site and j == new_site and i < new_site:
                        product_term(
                            adj,
                            ("CtildePairCreate", i),
                            ("Ctilde", i),
                            local_pair_create,
                            1,
                            -coeff,
                            -ycoeff,
                            index,
                            scale=-1.0,
                        )
                    elif k == new_site and j < new_site and i == new_site:
                        product_term(
                            adj,
                            ("CdagDensityMidJW", j),
                            ("Cdag", j),
                            local_density_mid_jw,
                            1,
                            0.5 * coeff,
                            0.5 * ycoeff,
                            index,
                            scale=0.5,
                        )
                        product_term(
                            adj,
                            ("CdagSpinDensityMidJW", j),
                            ("Cdag", j),
                            local_spin_density_mid_jw,
                            1,
                            np.sqrt(1.5) * coeff,
                            np.sqrt(1.5) * ycoeff,
                            index,
                            scale=np.sqrt(1.5),
                        )
                    elif k < new_site and j == new_site and i == new_site:
                        product_term(
                            adj,
                            ("CdagJWDensity", k),
                            ("Cdag", k),
                            local_jw_density,
                            1,
                            coeff,
                            ycoeff,
                            index,
                        )
                    else:
                        xyeri_adj[index] += _reduced_tensor_pairing(
                            adj,
                            identity_cdag_density,
                        )

    return (
        xoperator_adjoint,
        xyoperator_adjoint,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _new_site_weighted_packages_tangent(
    operators,
    doperators,
    h1e,
    dh1e,
    eri,
    deri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = False,
    actual=None,
):
    """Tangent of ``new_site_weighted_packages_from_operators``."""
    if build_v1:
        raise NotImplementedError(
            "new-site V1 tangent is handled by the projected V1 recurrence"
        )
    if actual is None:
        actual = _cached_new_site_weighted_packages(
            operators,
            h1e,
            eri,
            int(site_count),
            future_sites,
            build_v1=False,
        )
    else:
        actual = dict(actual)
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    if (
        not _operator_dict_has_nonzero_blocks(doperators)
        and not _new_site_weighted_coefficients_have_support(
            deri,
            site_count,
            future_sites,
        )
    ):
        return actual, {}
    out = {}
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    ddensity_cache = {}
    dspin_density_cache = {}
    dpair_cache = {}
    new_site = site_count - 1

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v3_terms = []

        tensor = operators[("Cdag", new_site)]
        dtensor = doperators.get(("Cdag", new_site), _zero_reduced_like(tensor))
        v3_terms.append(
            (tensor, dtensor, eri[new_site, q, q, q], deri[new_site, q, q, q])
        )

        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue

                density = density_cache.get((i, j))
                if density is None:
                    density = operators.get(("Density", i, j))
                    if density is None:
                        density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=np.sqrt(2.0),
                        )
                    density_cache[(i, j)] = density
                ddensity = _density_tangent(
                    operators,
                    doperators,
                    density_cache,
                    ddensity_cache,
                    i,
                    j,
                )
                density_terms.append(
                    (density, ddensity, eri[i, j, q, q], deri[i, j, q, q])
                )
                exchange_density_terms.append(
                    (density, ddensity, eri[i, q, q, j], deri[i, q, q, j])
                )

                spin_density = spin_density_cache.get((i, j))
                if spin_density is None:
                    spin_density = operators.get(("SpinDensity", i, j))
                    if spin_density is None:
                        spin_density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=2,
                        )
                    spin_density_cache[(i, j)] = spin_density
                dspin = _spin_density_tangent(
                    operators,
                    doperators,
                    spin_density_cache,
                    dspin_density_cache,
                    i,
                    j,
                )
                exchange_spin_terms.append(
                    (spin_density, dspin, eri[i, q, q, j], deri[i, q, q, j])
                )

                pair = pair_cache.get((i, j))
                if pair is None:
                    pair = coupled_reduced_product(
                        operators[("Ctilde", i)],
                        operators[("Ctilde", j)],
                        rank2=0,
                        scale=-1.0 / np.sqrt(2.0),
                    )
                    pair_cache[(i, j)] = pair
                dpair = _pair_annihilate_tangent(
                    operators,
                    doperators,
                    pair_cache,
                    dpair_cache,
                    i,
                    j,
                )
                pair_terms.append((pair, dpair, eri[q, i, q, j], deri[q, i, q, j]))

        packages = {
            ("NextDensity", q): _weighted_tangent_terms(density_terms),
            ("NextExchangeDensity", q): _weighted_tangent_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): _weighted_tangent_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): _weighted_tangent_terms(pair_terms),
            ("NextV3Cdag", q): _weighted_tangent_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return actual, out


def _new_site_weighted_packages_tangent_adjoint(
    operators,
    doperators,
    package_adjoint,
    h1e,
    eri,
    site_count: int,
    future_sites,
):
    """Adjoint of ``_new_site_weighted_packages_tangent``."""
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    doperator_adjoint = {}
    dh1_adj = np.zeros_like(h1e, dtype=float)
    deri_adj = np.zeros_like(eri, dtype=float)
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    ddensity_cache = {}
    dspin_density_cache = {}
    dpair_cache = {}
    new_site = site_count - 1

    def direct_term(adjoint, tensor, dtensor_key, coeff, coeff_array, coeff_index):
        coeff_array[coeff_index] += _reduced_tensor_pairing(adjoint, tensor)
        dtensor = doperators.get(dtensor_key)
        if dtensor is not None:
            _add_reduced_adjoint(
                doperator_adjoint,
                dtensor_key,
                _scale_adjoint(adjoint, coeff),
            )

    def composite_term(adjoint, tensor, tangent, coeff, coeff_array, coeff_index, adj_fn, *args):
        coeff_array[coeff_index] += _reduced_tensor_pairing(adjoint, tensor)
        if tangent.blocks:
            _merge_reduced_adjoint(
                doperator_adjoint,
                adj_fn(operators, doperators, _scale_adjoint(adjoint, coeff), *args),
            )

    for q in future_sites:
        q = int(q)
        adj = package_adjoint.get(("NextV3Cdag", q))
        if adj is not None:
            tensor = operators[("Cdag", new_site)]
            direct_term(
                adj,
                tensor,
                ("Cdag", new_site),
                eri[new_site, q, q, q],
                deri_adj,
                (new_site, q, q, q),
            )

        density_adj = package_adjoint.get(("NextDensity", q))
        exchange_density_adj = package_adjoint.get(("NextExchangeDensity", q))
        exchange_spin_adj = package_adjoint.get(("NextExchangeSpinDensity", q))
        pair_adj = package_adjoint.get(("NextPairAnnihilate", q))
        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue
                if (
                    density_adj is not None
                    or exchange_density_adj is not None
                    or exchange_spin_adj is not None
                ):
                    density = density_cache.get((i, j))
                    if density is None:
                        density = operators.get(("Density", i, j))
                        if density is None:
                            density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=0,
                                scale=np.sqrt(2.0),
                            )
                        density_cache[(i, j)] = density
                    ddensity = _density_tangent(
                        operators,
                        doperators,
                        density_cache,
                        ddensity_cache,
                        i,
                        j,
                    )
                    if density_adj is not None:
                        composite_term(
                            density_adj,
                            density,
                            ddensity,
                            eri[i, j, q, q],
                            deri_adj,
                            (i, j, q, q),
                            _density_tangent_adjoint,
                            i,
                            j,
                        )
                    if exchange_density_adj is not None:
                        composite_term(
                            exchange_density_adj,
                            density,
                            ddensity,
                            eri[i, q, q, j],
                            deri_adj,
                            (i, q, q, j),
                            _density_tangent_adjoint,
                            i,
                            j,
                        )

                if exchange_spin_adj is not None:
                    spin_density = spin_density_cache.get((i, j))
                    if spin_density is None:
                        spin_density = operators.get(("SpinDensity", i, j))
                        if spin_density is None:
                            spin_density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=2,
                            )
                        spin_density_cache[(i, j)] = spin_density
                    dspin = _spin_density_tangent(
                        operators,
                        doperators,
                        spin_density_cache,
                        dspin_density_cache,
                        i,
                        j,
                    )
                    composite_term(
                        exchange_spin_adj,
                        spin_density,
                        dspin,
                        eri[i, q, q, j],
                        deri_adj,
                        (i, q, q, j),
                        _spin_density_tangent_adjoint,
                        i,
                        j,
                    )

                if pair_adj is not None:
                    pair = pair_cache.get((i, j))
                    if pair is None:
                        pair = coupled_reduced_product(
                            operators[("Ctilde", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=-1.0 / np.sqrt(2.0),
                        )
                        pair_cache[(i, j)] = pair
                    dpair = _pair_annihilate_tangent(
                        operators,
                        doperators,
                        pair_cache,
                        dpair_cache,
                        i,
                        j,
                    )
                    composite_term(
                        pair_adj,
                        pair,
                        dpair,
                        eri[q, i, q, j],
                        deri_adj,
                        (q, i, q, j),
                        _pair_annihilate_tangent_adjoint,
                        i,
                        j,
                    )

    return doperator_adjoint, dh1_adj, deri_adj


def _new_site_weighted_packages_bilinear(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    h1e,
    xh1e,
    yh1e,
    xyh1e,
    eri,
    xeri,
    yeri,
    xyeri,
    site_count: int,
    future_sites,
    *,
    build_v1: bool = False,
    actual=None,
):
    """Mixed tangent of ``new_site_weighted_packages_from_operators``."""
    if build_v1:
        raise NotImplementedError(
            "new-site V1 bilinear is handled by the projected V1 recurrence"
        )
    if actual is None:
        actual = _cached_new_site_weighted_packages(
            operators,
            h1e,
            eri,
            int(site_count),
            future_sites,
            build_v1=False,
        )
    else:
        actual = dict(actual)
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    if (
        not _operator_dict_has_nonzero_blocks(xoperators)
        and not _operator_dict_has_nonzero_blocks(xyoperators)
        and not _new_site_weighted_coefficients_have_support(
            xeri,
            site_count,
            future_sites,
        )
        and not _new_site_weighted_coefficients_have_support(
            xyeri,
            site_count,
            future_sites,
        )
    ):
        return actual, {}
    out = {}
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    xdensity_cache = {}
    ydensity_cache = {}
    xydensity_cache = {}
    xspin_density_cache = {}
    yspin_density_cache = {}
    xyspin_density_cache = {}
    xpair_cache = {}
    ypair_cache = {}
    xypair_cache = {}
    new_site = site_count - 1

    for q in future_sites:
        density_terms = []
        exchange_density_terms = []
        exchange_spin_terms = []
        pair_terms = []
        v3_terms = []

        tensor = operators[("Cdag", new_site)]
        xtensor = xoperators.get(("Cdag", new_site), _zero_reduced_like(tensor))
        ytensor = yoperators.get(("Cdag", new_site), _zero_reduced_like(tensor))
        xytensor = xyoperators.get(("Cdag", new_site), _zero_reduced_like(tensor))
        v3_terms.append(
            (
                tensor,
                xtensor,
                ytensor,
                xytensor,
                eri[new_site, q, q, q],
                xeri[new_site, q, q, q],
                yeri[new_site, q, q, q],
                xyeri[new_site, q, q, q],
            )
        )

        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue

                density = density_cache.get((i, j))
                if density is None:
                    density = operators.get(("Density", i, j))
                    if density is None:
                        density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=np.sqrt(2.0),
                        )
                    density_cache[(i, j)] = density
                xdensity = _density_tangent(
                    operators,
                    xoperators,
                    density_cache,
                    xdensity_cache,
                    i,
                    j,
                )
                ydensity = _density_tangent(
                    operators,
                    yoperators,
                    density_cache,
                    ydensity_cache,
                    i,
                    j,
                )
                xydensity = _density_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    density_cache,
                    xdensity_cache,
                    ydensity_cache,
                    xydensity_cache,
                    i,
                    j,
                )
                density_terms.append(
                    (
                        density,
                        xdensity,
                        ydensity,
                        xydensity,
                        eri[i, j, q, q],
                        xeri[i, j, q, q],
                        yeri[i, j, q, q],
                        xyeri[i, j, q, q],
                    )
                )
                exchange_density_terms.append(
                    (
                        density,
                        xdensity,
                        ydensity,
                        xydensity,
                        eri[i, q, q, j],
                        xeri[i, q, q, j],
                        yeri[i, q, q, j],
                        xyeri[i, q, q, j],
                    )
                )

                spin_density = spin_density_cache.get((i, j))
                if spin_density is None:
                    spin_density = operators.get(("SpinDensity", i, j))
                    if spin_density is None:
                        spin_density = coupled_reduced_product(
                            operators[("Cdag", i)],
                            operators[("Ctilde", j)],
                            rank2=2,
                        )
                    spin_density_cache[(i, j)] = spin_density
                xspin = _spin_density_tangent(
                    operators,
                    xoperators,
                    spin_density_cache,
                    xspin_density_cache,
                    i,
                    j,
                )
                yspin = _spin_density_tangent(
                    operators,
                    yoperators,
                    spin_density_cache,
                    yspin_density_cache,
                    i,
                    j,
                )
                xyspin = _spin_density_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    spin_density_cache,
                    xspin_density_cache,
                    yspin_density_cache,
                    xyspin_density_cache,
                    i,
                    j,
                )
                exchange_spin_terms.append(
                    (
                        spin_density,
                        xspin,
                        yspin,
                        xyspin,
                        eri[i, q, q, j],
                        xeri[i, q, q, j],
                        yeri[i, q, q, j],
                        xyeri[i, q, q, j],
                    )
                )

                pair = pair_cache.get((i, j))
                if pair is None:
                    pair = coupled_reduced_product(
                        operators[("Ctilde", i)],
                        operators[("Ctilde", j)],
                        rank2=0,
                        scale=-1.0 / np.sqrt(2.0),
                    )
                    pair_cache[(i, j)] = pair
                xpair = _pair_annihilate_tangent(
                    operators,
                    xoperators,
                    pair_cache,
                    xpair_cache,
                    i,
                    j,
                )
                ypair = _pair_annihilate_tangent(
                    operators,
                    yoperators,
                    pair_cache,
                    ypair_cache,
                    i,
                    j,
                )
                xypair = _pair_annihilate_bilinear(
                    operators,
                    xoperators,
                    yoperators,
                    xyoperators,
                    pair_cache,
                    xpair_cache,
                    ypair_cache,
                    xypair_cache,
                    i,
                    j,
                )
                pair_terms.append(
                    (
                        pair,
                        xpair,
                        ypair,
                        xypair,
                        eri[q, i, q, j],
                        xeri[q, i, q, j],
                        yeri[q, i, q, j],
                        xyeri[q, i, q, j],
                    )
                )

        packages = {
            ("NextDensity", q): _weighted_bilinear_terms(density_terms),
            ("NextExchangeDensity", q): _weighted_bilinear_terms(exchange_density_terms),
            ("NextExchangeSpinDensity", q): _weighted_bilinear_terms(exchange_spin_terms),
            ("NextPairAnnihilate", q): _weighted_bilinear_terms(pair_terms),
            ("NextV3Cdag", q): _weighted_bilinear_terms(v3_terms),
        }
        out.update({key: tensor for key, tensor in packages.items() if tensor is not None})
    return actual, out


def _new_site_weighted_packages_bilinear_adjoint_x(
    operators,
    xoperators,
    yoperators,
    xyoperators,
    package_adjoint,
    h1e,
    eri,
    yeri,
    site_count: int,
    future_sites,
):
    """Adjoint of new-site bilinear packages wrt ``x`` and ``xy`` inputs."""
    site_count = int(site_count)
    future_sites = tuple(int(q) for q in future_sites)
    xoperator_adjoint = {}
    xyoperator_adjoint = {}
    xh1_adj = np.zeros_like(h1e, dtype=float)
    xyh1_adj = np.zeros_like(h1e, dtype=float)
    xeri_adj = np.zeros_like(eri, dtype=float)
    xyeri_adj = np.zeros_like(eri, dtype=float)
    density_cache = {}
    spin_density_cache = {}
    pair_cache = {}
    ydensity_cache = {}
    yspin_density_cache = {}
    ypair_cache = {}
    new_site = site_count - 1

    def merge(target, source):
        _merge_reduced_adjoint(target, source)

    def direct_term(adjoint, tensor, ytensor, tensor_key, coeff, ycoeff, index):
        xeri_adj[index] += _reduced_tensor_pairing(adjoint, ytensor)
        xyeri_adj[index] += _reduced_tensor_pairing(adjoint, tensor)
        if abs(ycoeff) > 0.0:
            _add_reduced_adjoint(
                xoperator_adjoint,
                tensor_key,
                _scale_adjoint(adjoint, ycoeff),
            )
        if abs(coeff) > 0.0:
            _add_reduced_adjoint(
                xyoperator_adjoint,
                tensor_key,
                _scale_adjoint(adjoint, coeff),
            )

    def composite_term(
        adjoint,
        tensor,
        ytensor,
        coeff,
        ycoeff,
        index,
        tangent_adj_fn,
        bilinear_adj_fn,
        *args,
    ):
        xeri_adj[index] += _reduced_tensor_pairing(adjoint, ytensor)
        xyeri_adj[index] += _reduced_tensor_pairing(adjoint, tensor)
        if abs(ycoeff) > 0.0:
            merge(
                xoperator_adjoint,
                tangent_adj_fn(
                    operators,
                    xoperators,
                    _scale_adjoint(adjoint, ycoeff),
                    *args,
                ),
            )
        if abs(coeff) > 0.0:
            bx_adj, bxy_adj = bilinear_adj_fn(
                operators,
                xoperators,
                yoperators,
                xyoperators,
                _scale_adjoint(adjoint, coeff),
                *args,
            )
            merge(xoperator_adjoint, bx_adj)
            merge(xyoperator_adjoint, bxy_adj)

    for q in future_sites:
        q = int(q)
        adj = package_adjoint.get(("NextV3Cdag", q))
        if adj is not None:
            tensor = operators[("Cdag", new_site)]
            ytensor = yoperators.get(("Cdag", new_site), _zero_reduced_like(tensor))
            direct_term(
                adj,
                tensor,
                ytensor,
                ("Cdag", new_site),
                eri[new_site, q, q, q],
                yeri[new_site, q, q, q],
                (new_site, q, q, q),
            )

        density_adj = package_adjoint.get(("NextDensity", q))
        exchange_density_adj = package_adjoint.get(("NextExchangeDensity", q))
        exchange_spin_adj = package_adjoint.get(("NextExchangeSpinDensity", q))
        pair_adj = package_adjoint.get(("NextPairAnnihilate", q))
        for i in range(site_count):
            for j in range(site_count):
                if i != new_site and j != new_site:
                    continue
                if (
                    density_adj is not None
                    or exchange_density_adj is not None
                    or exchange_spin_adj is not None
                ):
                    density = density_cache.get((i, j))
                    if density is None:
                        density = operators.get(("Density", i, j))
                        if density is None:
                            density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=0,
                                scale=np.sqrt(2.0),
                            )
                        density_cache[(i, j)] = density
                    ydensity = _density_tangent(
                        operators,
                        yoperators,
                        density_cache,
                        ydensity_cache,
                        i,
                        j,
                    )
                    if density_adj is not None:
                        composite_term(
                            density_adj,
                            density,
                            ydensity,
                            eri[i, j, q, q],
                            yeri[i, j, q, q],
                            (i, j, q, q),
                            _density_tangent_adjoint,
                            _density_bilinear_adjoint_x,
                            i,
                            j,
                        )
                    if exchange_density_adj is not None:
                        composite_term(
                            exchange_density_adj,
                            density,
                            ydensity,
                            eri[i, q, q, j],
                            yeri[i, q, q, j],
                            (i, q, q, j),
                            _density_tangent_adjoint,
                            _density_bilinear_adjoint_x,
                            i,
                            j,
                        )

                if exchange_spin_adj is not None:
                    spin_density = spin_density_cache.get((i, j))
                    if spin_density is None:
                        spin_density = operators.get(("SpinDensity", i, j))
                        if spin_density is None:
                            spin_density = coupled_reduced_product(
                                operators[("Cdag", i)],
                                operators[("Ctilde", j)],
                                rank2=2,
                            )
                        spin_density_cache[(i, j)] = spin_density
                    yspin = _spin_density_tangent(
                        operators,
                        yoperators,
                        spin_density_cache,
                        yspin_density_cache,
                        i,
                        j,
                    )
                    composite_term(
                        exchange_spin_adj,
                        spin_density,
                        yspin,
                        eri[i, q, q, j],
                        yeri[i, q, q, j],
                        (i, q, q, j),
                        _spin_density_tangent_adjoint,
                        _spin_density_bilinear_adjoint_x,
                        i,
                        j,
                    )

                if pair_adj is not None:
                    pair = pair_cache.get((i, j))
                    if pair is None:
                        pair = coupled_reduced_product(
                            operators[("Ctilde", i)],
                            operators[("Ctilde", j)],
                            rank2=0,
                            scale=-1.0 / np.sqrt(2.0),
                        )
                        pair_cache[(i, j)] = pair
                    ypair = _pair_annihilate_tangent(
                        operators,
                        yoperators,
                        pair_cache,
                        ypair_cache,
                        i,
                        j,
                    )
                    composite_term(
                        pair_adj,
                        pair,
                        ypair,
                        eri[q, i, q, j],
                        yeri[q, i, q, j],
                        (q, i, q, j),
                        _pair_annihilate_tangent_adjoint,
                        _pair_annihilate_bilinear_adjoint_x,
                        i,
                        j,
                    )

    return (
        xoperator_adjoint,
        xyoperator_adjoint,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _pre_rotation_tensors_and_tangents(
    narg,
    source_tangent,
    h1e_full,
    dh1e_full,
    eri_full,
    deri_full,
    future_sites,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
    cache_tangent_grown: bool = False,
    return_parts: bool = False,
):
    from .su2_chain import (
        add_optional_reduced_terms,
        complete_density_composites,
        complete_pair_composites,
        grown_coupling_operators,
        reduced_coupling_operators_from_growth,
        weighted_packages_from_operators,
    )

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    future_sites = tuple(future_sites)
    carry_pair_composites = bool(project_v1_packages and len(future_sites) > 1)
    grown = grown_coupling_operators(
        narg,
        include_even_composites=(
            (not project_v1_packages)
            or carry_pair_composites
            or carry_rdm_operators
        ),
        even_composites=(
            {"Density"}
            if carry_rdm_operators and project_v1_packages and not carry_pair_composites
            else None
        ),
    )
    dgrown = _grown_coupling_operators_tangent(
        narg,
        source_tangent,
        include_even_composites=(
            (not project_v1_packages)
            or carry_pair_composites
            or carry_rdm_operators
        ),
        even_composites=(
            {"Density"}
            if carry_rdm_operators and project_v1_packages and not carry_pair_composites
            else None
        ),
        cache_result=cache_tangent_grown,
    )
    if carry_rdm_operators:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        grown = complete_density_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                key = ("Density", i, j)
                if key in grown and key not in dgrown:
                    dgrown[key] = _density_tangent(grown, dgrown, {}, {}, i, j)
    if carry_pair_composites:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        grown = complete_pair_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                for name in ("Density", "SpinDensity"):
                    key = (name, i, j)
                    if key in grown and key not in dgrown:
                        if name == "Density":
                            dgrown[key] = _density_tangent(grown, dgrown, {}, {}, i, j)
                        else:
                            dgrown[key] = _spin_density_tangent(grown, dgrown, {}, {}, i, j)
    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    v1_actual = None
    dv1_new = None
    if project_v1_packages:
        weighted = _cached_context_new_site_weighted_packages(
            source_block,
            (
                "tangent_project_v1",
                id(narg),
                bool(carry_rdm_operators),
                bool(carry_pair_composites),
            ),
            grown,
            h1e_full,
            eri_full,
            site_count,
            future_sites,
            build_v1=False,
        )
        _, dweighted = _new_site_weighted_packages_tangent(
            grown,
            dgrown,
            h1e_full,
            dh1e_full,
            eri_full,
            deri_full,
            site_count,
            future_sites,
            build_v1=False,
            actual=weighted,
        )
        v1_actual, dv1_new = _grown_reduced_v1_packages_tangent(
            source_block,
            source_tangent,
            grown,
            h1e_full,
            dh1e_full,
            eri_full,
            deri_full,
            future_sites,
        )
        carried_terms = {}
        dcarried_terms = {}
        even_prefixes = {
            "NextDensity",
            "NextExchangeDensity",
            "NextExchangeSpinDensity",
            "NextPairAnnihilate",
        }
        odd_prefixes = {"NextV1Spinor", "NextV3Cdag"}
        for q in future_sites:
            q = int(q)
            for prefix in even_prefixes:
                key = (prefix, q)
                terms = []
                dterms = []
                carried = source_block.reduced_operators.get(key)
                dcarried = source_tangent.reduced_operators.get(key)
                if carried is not None:
                    terms.append(
                        _cached_grow_source_tensor(
                            source_block, carried, local_name="I"
                        )
                    )
                if dcarried is not None:
                    dterms.append(
                        _cached_grow_source_tensor(
                            source_block, dcarried, local_name="I"
                        )
                    )
                if key in weighted:
                    terms.append(weighted[key])
                if key in dweighted:
                    dterms.append(dweighted[key])
                total = add_optional_reduced_terms(terms)
                dtotal = add_optional_reduced_terms(dterms)
                if total is not None:
                    carried_terms[key] = total
                if dtotal is not None:
                    dcarried_terms[key] = dtotal
            for prefix in odd_prefixes:
                key = (prefix, q)
                terms = []
                dterms = []
                carried = source_block.reduced_operators.get(key)
                dcarried = source_tangent.reduced_operators.get(key)
                if carried is not None:
                    terms.append(
                        _cached_grow_source_tensor(
                            source_block, carried, local_name="JW"
                        )
                    )
                if dcarried is not None:
                    dterms.append(
                        _cached_grow_source_tensor(
                            source_block, dcarried, local_name="JW"
                        )
                    )
                if prefix == "NextV1Spinor":
                    if key in v1_actual:
                        terms.append(v1_actual[key])
                    if key in dv1_new:
                        dterms.append(dv1_new[key])
                else:
                    if key in weighted:
                        terms.append(weighted[key])
                    if key in dweighted:
                        dterms.append(dweighted[key])
                total = add_optional_reduced_terms(terms)
                dtotal = add_optional_reduced_terms(dterms)
                if total is not None:
                    carried_terms[key] = total
                if dtotal is not None:
                    dcarried_terms[key] = dtotal
        weighted = carried_terms
        dweighted = dcarried_terms
    else:
        weighted = weighted_packages_from_operators(
            grown,
            h1e_full,
            eri_full,
            site_count,
            future_sites,
            build_v1=True,
        )
        dweighted = _weighted_packages_tangent(
            grown,
            dgrown,
            h1e_full,
            dh1e_full,
            eri_full,
            deri_full,
            site_count,
            future_sites,
            build_v1=True,
        )
    del reduced_coupling_operators_from_growth
    tensors = {**grown, **weighted}
    dtensors = {**dgrown, **dweighted}
    if not return_parts:
        return tensors, dtensors
    parts = {
        "grown": dict(grown),
        "dgrown": dict(dgrown),
        "weighted": dict(weighted),
        "dweighted": dict(dweighted),
        "v1_actual": None if v1_actual is None else dict(v1_actual),
        "dv1_new": None if dv1_new is None else dict(dv1_new),
    }
    return tensors, dtensors, parts


def _grow_source_tensor_tangent_adjoint(
    source_block,
    source_tangent,
    key,
    adjoint,
    *,
    local_name: str,
):
    """Adjoint of growing a source tangent tensor by a fixed local operator."""
    from .su2_three_site import local_reduced_operator

    tensor = source_tangent.reduced_operators.get(key)
    if tensor is None:
        return None
    local = local_reduced_operator(local_name)
    return reduced_product_tensor_block_adjoint(
        source_block,
        tensor,
        local,
        adjoint,
        total_rank2=tensor.op.charge[1],
    )


def _pre_rotation_tensors_and_tangents_adjoint(
    narg,
    source_tangent,
    tensor_adjoint,
    h1e_full,
    eri_full,
    future_sites,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
    parts=None,
):
    """Adjoint of ``_pre_rotation_tensors_and_tangents`` wrt tangent inputs."""
    if not project_v1_packages:
        raise NotImplementedError(
            "pre-rotation adjoint is currently implemented for projected V1 packages"
        )
    if parts is None:
        raise ValueError("pre-rotation adjoint requires forward parts")

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")
    future_sites = tuple(int(q) for q in future_sites)
    grown = dict(parts["grown"])
    dgrown = dict(parts["dgrown"])
    source_operator_adjoint = {}
    grown_tangent_adjoint = {}
    weighted_adjoint = {}
    v1_adjoint = {}
    dh1_adj = np.zeros_like(h1e_full, dtype=float)
    deri_adj = np.zeros_like(eri_full, dtype=float)

    even_prefixes = {
        "NextDensity",
        "NextExchangeDensity",
        "NextExchangeSpinDensity",
        "NextPairAnnihilate",
    }
    odd_prefixes = {"NextV1Spinor", "NextV3Cdag"}
    next_prefixes = even_prefixes | odd_prefixes

    for key, adj in tensor_adjoint.items():
        if not isinstance(key, tuple) or key[0] not in next_prefixes:
            _add_reduced_adjoint(grown_tangent_adjoint, key, adj)

    for q in future_sites:
        for prefix in even_prefixes:
            key = (prefix, q)
            adj = tensor_adjoint.get(key)
            if adj is None:
                continue
            carried_adj = _grow_source_tensor_tangent_adjoint(
                source_block,
                source_tangent,
                key,
                adj,
                local_name="I",
            )
            if carried_adj is not None:
                _add_reduced_adjoint(source_operator_adjoint, key, carried_adj)
            weighted_adjoint[key] = adj

        for prefix in odd_prefixes:
            key = (prefix, q)
            adj = tensor_adjoint.get(key)
            if adj is None:
                continue
            carried_adj = _grow_source_tensor_tangent_adjoint(
                source_block,
                source_tangent,
                key,
                adj,
                local_name="JW",
            )
            if carried_adj is not None:
                _add_reduced_adjoint(source_operator_adjoint, key, carried_adj)
            if prefix == "NextV1Spinor":
                v1_adjoint[key] = adj
            else:
                weighted_adjoint[key] = adj

    if weighted_adjoint:
        op_adj, _, g_adj = _new_site_weighted_packages_tangent_adjoint(
            grown,
            dgrown,
            weighted_adjoint,
            h1e_full,
            eri_full,
            max(key[1] for key in grown if key[0] == "Cdag") + 1,
            future_sites,
        )
        _merge_reduced_adjoint(grown_tangent_adjoint, op_adj)
        deri_adj += g_adj

    if v1_adjoint:
        op_adj, h_adj, g_adj = _grown_reduced_v1_packages_tangent_adjoint(
            source_block,
            source_tangent,
            grown,
            v1_adjoint,
            h1e_full,
            eri_full,
            future_sites,
        )
        _merge_reduced_adjoint(source_operator_adjoint, op_adj)
        dh1_adj += h_adj
        deri_adj += g_adj

    if grown_tangent_adjoint:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        new_site = site_count - 1
        algebraic_keys = [
            key
            for key in grown_tangent_adjoint
            if (
                isinstance(key, tuple)
                and len(key) == 3
                and key[0] in {"Density", "SpinDensity"}
                and new_site in (int(key[1]), int(key[2]))
            )
        ]
        for key in algebraic_keys:
            adj = grown_tangent_adjoint.pop(key)
            dops_without_direct = dict(dgrown)
            dops_without_direct.pop(key, None)
            if key[0] == "Density":
                _merge_reduced_adjoint(
                    grown_tangent_adjoint,
                    _density_tangent_adjoint(
                        grown,
                        dops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    ),
                )
            else:
                _merge_reduced_adjoint(
                    grown_tangent_adjoint,
                    _spin_density_tangent_adjoint(
                        grown,
                        dops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    ),
                )

        carry_pair_composites = bool(project_v1_packages and len(future_sites) > 1)
        even_composites = (
            {"Density"}
            if carry_rdm_operators and project_v1_packages and not carry_pair_composites
            else None
        )
        op_adj = _grown_coupling_operators_tangent_adjoint(
            narg,
            source_tangent,
            grown_tangent_adjoint,
            include_even_composites=(
                (not project_v1_packages)
                or carry_pair_composites
                or carry_rdm_operators
            ),
            even_composites=even_composites,
        )
        _merge_reduced_adjoint(source_operator_adjoint, op_adj)

    return source_operator_adjoint, dh1_adj, deri_adj


def _recursive_growth_step_tangent_adjoint(
    source_block,
    grown,
    block,
    source_tangent,
    next_hamiltonian_adjoint,
    next_operator_adjoint,
    h1e_full,
    eri_full,
    future_sites,
    response,
    parts,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
):
    """Adjoint of one nonterminal recursive tangent growth/truncation step."""
    site_index = max(key[1] for key in parts["grown"] if key[0] == "Cdag")
    nsites = site_index + 1
    tensors = {**parts["grown"], **parts["weighted"]}

    tensor_adjoint, transform_adjoint = rotate_reduced_tensors_tangent_adjoint(
        block.truncated,
        tensors,
        next_operator_adjoint,
    )
    pre_op_adj, pre_h_adj, pre_g_adj = _pre_rotation_tensors_and_tangents_adjoint(
        grown,
        source_tangent,
        tensor_adjoint,
        h1e_full,
        eri_full,
        future_sites,
        project_v1_packages=project_v1_packages,
        carry_rdm_operators=carry_rdm_operators,
        parts=parts,
    )
    grown_hamiltonian_adjoint = truncation_tangent_adjoint(
        grown,
        block.truncated,
        transform_adjoint_blocks=transform_adjoint,
        hamiltonian_adjoint=next_hamiltonian_adjoint,
    )
    h_adj, op_adj, step_h_adj, step_g_adj = _grown_hamiltonian_tangent_adjoint(
        source_block,
        source_tangent,
        grown_hamiltonian_adjoint,
        np.asarray(h1e_full)[:nsites, :nsites],
        np.asarray(eri_full)[:nsites, :nsites, :nsites, :nsites],
        site_index=site_index,
    )

    source_operator_adjoint = {}
    _merge_reduced_adjoint(source_operator_adjoint, pre_op_adj)
    _merge_reduced_adjoint(source_operator_adjoint, op_adj)
    dh1_adj = np.array(pre_h_adj, copy=True)
    deri_adj = np.array(pre_g_adj, copy=True)
    dh1_adj[:nsites, :nsites] += step_h_adj
    deri_adj[:nsites, :nsites, :nsites, :nsites] += step_g_adj
    return h_adj, source_operator_adjoint, dh1_adj, deri_adj


def _recursive_growth_step_bilinear_adjoint_x(
    source_block,
    grown,
    block,
    source_x,
    source_y,
    source_xy,
    next_x_hamiltonian_adjoint,
    next_x_operator_adjoint,
    next_xy_hamiltonian_adjoint,
    next_xy_operator_adjoint,
    h1e_full,
    eri_full,
    ydh1e_full,
    yderi_full,
    future_sites,
    response: TruncationBilinearTangent,
    perturbation_y,
    x_pre_rotation_parts,
    y_pre_rotation_parts,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
):
    """Adjoint of one recursive mixed growth/truncation step wrt ``x``/``xy``."""

    def add_hamiltonian_adjoint(left, right):
        if right is None or not right.blocks:
            return left
        if left is None or not left.blocks:
            return right
        return _add_irrep_tensors(left, right)

    site_index = max(key[1] for key in x_pre_rotation_parts["grown"] if key[0] == "Cdag")
    nsites = int(site_index) + 1
    tensors = {**x_pre_rotation_parts["grown"], **x_pre_rotation_parts["weighted"]}
    ytensors = {
        **y_pre_rotation_parts["dgrown"],
        **y_pre_rotation_parts["dweighted"],
    }

    prev_x_h_adj = None
    prev_xy_h_adj = None
    prev_x_op_adj = {}
    prev_xy_op_adj = {}
    xh1_adj = np.zeros_like(h1e_full, dtype=float)
    xeri_adj = np.zeros_like(eri_full, dtype=float)
    xyh1_adj = np.zeros_like(h1e_full, dtype=float)
    xyeri_adj = np.zeros_like(eri_full, dtype=float)

    x_grown_h_adj = None
    if next_x_operator_adjoint:
        tensor_x_adjoint, transform_x_adjoint = rotate_reduced_tensors_tangent_adjoint(
            block.truncated,
            tensors,
            next_x_operator_adjoint,
        )
        op_adj, h_adj, g_adj = _pre_rotation_tensors_and_tangents_adjoint(
            grown,
            source_x,
            tensor_x_adjoint,
            h1e_full,
            eri_full,
            future_sites,
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=carry_rdm_operators,
            parts=x_pre_rotation_parts,
        )
        _merge_reduced_adjoint(prev_x_op_adj, op_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
    else:
        transform_x_adjoint = {}

    if next_x_hamiltonian_adjoint is not None or transform_x_adjoint:
        x_grown_h_adj = truncation_tangent_adjoint(
            grown,
            block.truncated,
            transform_adjoint_blocks=transform_x_adjoint,
            hamiltonian_adjoint=next_x_hamiltonian_adjoint,
            include_retained_mixing=True,
        )

    if x_grown_h_adj is not None and x_grown_h_adj.blocks:
        h_adj, op_adj, step_h_adj, step_g_adj = _grown_hamiltonian_tangent_adjoint(
            source_block,
            source_x,
            x_grown_h_adj,
            np.asarray(h1e_full)[:nsites, :nsites],
            np.asarray(eri_full)[:nsites, :nsites, :nsites, :nsites],
            site_index=site_index,
        )
        prev_x_h_adj = add_hamiltonian_adjoint(prev_x_h_adj, h_adj)
        _merge_reduced_adjoint(prev_x_op_adj, op_adj)
        xh1_adj[:nsites, :nsites] += step_h_adj
        xeri_adj[:nsites, :nsites, :nsites, :nsites] += step_g_adj

    if next_xy_operator_adjoint:
        (
            tensor_x_adjoint,
            tensor_xy_adjoint,
            transform_x_adjoint,
            transform_xy_adjoint,
        ) = rotate_reduced_tensors_bilinear_adjoint_x(
            block.truncated,
            tensors,
            ytensors,
            next_xy_operator_adjoint,
            response,
        )
        (
            op_x_adj,
            op_xy_adj,
            h_adj,
            g_adj,
            xy_h_adj,
            xy_g_adj,
        ) = _pre_rotation_tensors_and_bilinears_adjoint_x(
            grown,
            source_x,
            source_y,
            source_xy,
            tensor_x_adjoint,
            tensor_xy_adjoint,
            h1e_full,
            eri_full,
            ydh1e_full,
            yderi_full,
            future_sites,
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=carry_rdm_operators,
            x_pre_rotation_parts=x_pre_rotation_parts,
            y_pre_rotation_parts=y_pre_rotation_parts,
        )
        _merge_reduced_adjoint(prev_x_op_adj, op_x_adj)
        _merge_reduced_adjoint(prev_xy_op_adj, op_xy_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
        xyh1_adj += xy_h_adj
        xyeri_adj += xy_g_adj
    else:
        transform_x_adjoint = {}
        transform_xy_adjoint = {}

    if (
        next_xy_hamiltonian_adjoint is not None
        or transform_x_adjoint
        or transform_xy_adjoint
    ):
        perturbation_x_adjoint, perturbation_xy_adjoint = (
            truncation_bilinear_tangent_adjoint_x(
                grown,
                block.truncated,
                response,
                perturbation_y,
                transform_x_adjoint_blocks=transform_x_adjoint,
                transform_xy_adjoint_blocks=transform_xy_adjoint,
                hamiltonian_xy_adjoint=next_xy_hamiltonian_adjoint,
                include_retained_mixing=True,
            )
        )
        if perturbation_x_adjoint.blocks:
            h_adj, op_adj, step_h_adj, step_g_adj = (
                _grown_hamiltonian_tangent_adjoint(
                    source_block,
                    source_x,
                    perturbation_x_adjoint,
                    np.asarray(h1e_full)[:nsites, :nsites],
                    np.asarray(eri_full)[:nsites, :nsites, :nsites, :nsites],
                    site_index=site_index,
                )
            )
            prev_x_h_adj = add_hamiltonian_adjoint(prev_x_h_adj, h_adj)
            _merge_reduced_adjoint(prev_x_op_adj, op_adj)
            xh1_adj[:nsites, :nsites] += step_h_adj
            xeri_adj[:nsites, :nsites, :nsites, :nsites] += step_g_adj
        if perturbation_xy_adjoint.blocks:
            h_adj, op_adj, step_h_adj, step_g_adj = (
                _grown_hamiltonian_bilinear_adjoint(
                    source_block,
                    source_xy,
                    perturbation_xy_adjoint,
                    np.asarray(h1e_full)[:nsites, :nsites],
                    np.asarray(eri_full)[:nsites, :nsites, :nsites, :nsites],
                    site_index=site_index,
                )
            )
            prev_xy_h_adj = add_hamiltonian_adjoint(prev_xy_h_adj, h_adj)
            _merge_reduced_adjoint(prev_xy_op_adj, op_adj)
            xyh1_adj[:nsites, :nsites] += step_h_adj
            xyeri_adj[:nsites, :nsites, :nsites, :nsites] += step_g_adj

    if prev_x_h_adj is None:
        prev_x_h_adj = _zero_irrep_tensor_like(source_x.hamiltonian)
    if prev_xy_h_adj is None:
        prev_xy_h_adj = _zero_irrep_tensor_like(source_xy.hamiltonian)
    return (
        prev_x_h_adj,
        prev_x_op_adj,
        prev_xy_h_adj,
        prev_xy_op_adj,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _rotate_reduced_tensors_tangent(truncated, tensors, dtensors, truncation_response):
    out = {}
    keys = set(tensors) | set(dtensors)
    for key in keys:
        tensor = tensors.get(key)
        dtensor = dtensors.get(key)
        if tensor is None:
            tensor = _zero_reduced_like(dtensor)
        if dtensor is None:
            dtensor = _zero_reduced_like(tensor)
        rotated = rotate_reduced_tensor_tangent(
            truncated,
            tensor,
            dtensor,
            truncation_response,
        )
        if rotated.blocks:
            out[key] = rotated
    return out


def rotate_reduced_tensors_tangent_adjoint(
    truncated,
    tensors,
    rotated_adjoint,
):
    """Adjoint of ``_rotate_reduced_tensors_tangent`` wrt tangent tensors/response."""
    tensor_adjoint = {}
    transform_adjoint_blocks = {}
    for key, adjoint_tensor in (rotated_adjoint or {}).items():
        tensor = tensors.get(key)
        if tensor is None:
            continue
        op_adj, transform_adj = rotate_irrep_tensor_tangent_adjoint(
            truncated,
            tensor.tensor,
            adjoint_tensor.tensor,
        )
        reduced_adj = ReducedSU2Tensor(op_adj)
        if reduced_adj.blocks:
            if key in tensor_adjoint:
                tensor_adjoint[key] = add_reduced_tensors(
                    tensor_adjoint[key],
                    reduced_adj,
                )
            else:
                tensor_adjoint[key] = reduced_adj
        for tkey, value in transform_adj.items():
            if tkey in transform_adjoint_blocks:
                transform_adjoint_blocks[tkey] = (
                    transform_adjoint_blocks[tkey] + value
                )
            else:
                transform_adjoint_blocks[tkey] = value
    return tensor_adjoint, transform_adjoint_blocks


def _rotate_reduced_tensors_bilinear(
    truncated,
    tensors,
    xtensors,
    ytensors,
    xytensors,
    truncation_response,
):
    out = {}
    keys = set(tensors) | set(xtensors) | set(ytensors) | set(xytensors)
    zero_cache = {}

    def zero_like(tensor):
        cached = zero_cache.get(id(tensor))
        if cached is None:
            cached = _zero_reduced_like(tensor)
            zero_cache[id(tensor)] = cached
        return cached

    for key in keys:
        tensor = tensors.get(key)
        xtensor = xtensors.get(key)
        ytensor = ytensors.get(key)
        xytensor = xytensors.get(key)
        if tensor is None:
            template = xtensor or ytensor or xytensor
            tensor = zero_like(template)
        if xtensor is None:
            xtensor = zero_like(tensor)
        if ytensor is None:
            ytensor = zero_like(tensor)
        if xytensor is None:
            xytensor = zero_like(tensor)
        rotated = rotate_reduced_tensor_bilinear(
            truncated,
            tensor,
            xtensor,
            ytensor,
            xytensor,
            truncation_response,
        )
        if rotated.blocks:
            out[key] = rotated
    return out


def rotate_reduced_tensors_bilinear_adjoint_x(
    truncated,
    tensors,
    ytensors,
    rotated_adjoint,
    truncation_response,
):
    """Adjoint of ``_rotate_reduced_tensors_bilinear`` wrt ``x`` tensors/response."""
    tensor_x_adjoint = {}
    tensor_xy_adjoint = {}
    transform_x_adjoint_blocks = {}
    transform_xy_adjoint_blocks = {}
    zero_cache = {}

    def zero_like(tensor):
        cached = zero_cache.get(id(tensor))
        if cached is None:
            cached = _zero_reduced_like(tensor)
            zero_cache[id(tensor)] = cached
        return cached

    def zero_source_like(adjoint_tensor):
        key = ("source", id(adjoint_tensor))
        cached = zero_cache.get(key)
        if cached is None:
            cached = _zero_reduced(truncated.source.site, adjoint_tensor.op)
            zero_cache[key] = cached
        return cached

    def add_tensor(target, key, tensor):
        if tensor is None or not tensor.blocks:
            return
        if key in target:
            target[key] = add_reduced_tensors(target[key], tensor)
        else:
            target[key] = tensor

    def add_blocks(target, source):
        for block_key, value in source.items():
            if block_key in target:
                target[block_key] = target[block_key] + value
            else:
                target[block_key] = value

    for key, adjoint_tensor in (rotated_adjoint or {}).items():
        tensor = tensors.get(key)
        ytensor = ytensors.get(key)
        if tensor is None:
            tensor = (
                zero_like(ytensor)
                if ytensor is not None
                else zero_source_like(adjoint_tensor)
            )
        if ytensor is None:
            ytensor = zero_like(tensor)
        (
            tensor_x,
            tensor_xy,
            transform_x,
            transform_xy,
        ) = rotate_irrep_tensor_bilinear_adjoint_x(
            truncated,
            tensor.tensor,
            ytensor.tensor,
            adjoint_tensor.tensor,
            truncation_response,
        )
        add_tensor(tensor_x_adjoint, key, ReducedSU2Tensor(tensor_x))
        add_tensor(tensor_xy_adjoint, key, ReducedSU2Tensor(tensor_xy))
        add_blocks(transform_x_adjoint_blocks, transform_x)
        add_blocks(transform_xy_adjoint_blocks, transform_xy)

    return (
        tensor_x_adjoint,
        tensor_xy_adjoint,
        transform_x_adjoint_blocks,
        transform_xy_adjoint_blocks,
    )


def _pre_rotation_tensors_and_bilinears(
    narg,
    source_x,
    source_y,
    source_xy,
    h1e_full,
    xh1e_full,
    yh1e_full,
    xyh1e_full,
    eri_full,
    xeri_full,
    yeri_full,
    xyeri_full,
    future_sites,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
    cache_y_tangent_grown: bool = False,
    x_is_zero: bool = False,
    x_pre_rotation_parts=None,
    y_pre_rotation_parts=None,
):
    from .su2_chain import (
        add_optional_reduced_terms,
        complete_density_composites,
        complete_pair_composites,
        grown_coupling_operators,
        weighted_packages_from_operators,
    )

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    future_sites = tuple(future_sites)
    carry_pair_composites = bool(project_v1_packages and len(future_sites) > 1)
    include_even = (
        (not project_v1_packages)
        or carry_pair_composites
        or carry_rdm_operators
    )
    even_composites = (
        {"Density"}
        if carry_rdm_operators and project_v1_packages and not carry_pair_composites
        else None
    )
    even_key = None if even_composites is None else tuple(sorted(even_composites))
    cached_base_parts = x_pre_rotation_parts or y_pre_rotation_parts
    grown_from_parts = (
        cached_base_parts is not None
        and cached_base_parts.get("grown") is not None
    )
    if grown_from_parts:
        grown = dict(cached_base_parts["grown"])
    else:
        grown = grown_coupling_operators(
            narg,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    xgrown = (
        {}
        if x_is_zero
        else dict(x_pre_rotation_parts["dgrown"])
        if x_pre_rotation_parts is not None
        and x_pre_rotation_parts.get("dgrown") is not None
        else _grown_coupling_operators_tangent(
            narg,
            source_x,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    )
    y_completed_cached = False
    y_completed_cache_key = None
    y_completed_cache = None
    if (
        y_pre_rotation_parts is not None
        and y_pre_rotation_parts.get("dgrown") is not None
    ):
        ygrown = dict(y_pre_rotation_parts["dgrown"])
        y_completed_cached = True
    elif cache_y_tangent_grown:
        y_completed_cache_key = (
            id(narg),
            id(source_y),
            bool(include_even),
            even_key,
            bool(carry_rdm_operators),
            bool(carry_pair_composites),
        )
        y_completed_cache = getattr(
            source_block,
            "_su2_response_completed_grown_tangent_cache",
            None,
        )
        if y_completed_cache is None:
            y_completed_cache = {}
            setattr(
                source_block,
                "_su2_response_completed_grown_tangent_cache",
                y_completed_cache,
            )
        cached = y_completed_cache.get(y_completed_cache_key)
        if cached is not None and cached[0] is source_y:
            ygrown = dict(cached[1])
            y_completed_cached = True
        else:
            ygrown = _grown_coupling_operators_tangent(
                narg,
                source_y,
                include_even_composites=include_even,
                even_composites=even_composites,
                cache_result=True,
            )
    else:
        ygrown = _grown_coupling_operators_tangent(
            narg,
            source_y,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    xygrown = _grown_coupling_operators_tangent(
        narg,
        source_xy,
        include_even_composites=include_even,
        even_composites=even_composites,
    )
    if carry_rdm_operators:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        if not grown_from_parts:
            grown = complete_density_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                key = ("Density", i, j)
                if key not in grown:
                    continue
                if not x_is_zero and key not in xgrown:
                    xgrown[key] = _density_tangent(grown, xgrown, {}, {}, i, j)
                if not y_completed_cached and key not in ygrown:
                    ygrown[key] = _density_tangent(grown, ygrown, {}, {}, i, j)
                if key not in xygrown:
                    xygrown[key] = _density_bilinear(
                        grown,
                        xgrown,
                        ygrown,
                        xygrown,
                        {},
                        {},
                        {},
                        {},
                        i,
                        j,
                    )
    if carry_pair_composites:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        if not grown_from_parts:
            grown = complete_pair_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                for name in ("Density", "SpinDensity"):
                    key = (name, i, j)
                    if key not in grown:
                        continue
                    if not x_is_zero and key not in xgrown:
                        if name == "Density":
                            xgrown[key] = _density_tangent(grown, xgrown, {}, {}, i, j)
                        else:
                            xgrown[key] = _spin_density_tangent(grown, xgrown, {}, {}, i, j)
                    if not y_completed_cached and key not in ygrown:
                        if name == "Density":
                            ygrown[key] = _density_tangent(grown, ygrown, {}, {}, i, j)
                        else:
                            ygrown[key] = _spin_density_tangent(grown, ygrown, {}, {}, i, j)
                    if key not in xygrown:
                        if name == "Density":
                            xygrown[key] = _density_bilinear(
                                grown,
                                xgrown,
                                ygrown,
                                xygrown,
                                {},
                                {},
                                {},
                                {},
                                i,
                                j,
                            )
                        else:
                            xygrown[key] = _spin_density_bilinear(
                                grown,
                                xgrown,
                                ygrown,
                                xygrown,
                                {},
                                {},
                                {},
                                {},
                                i,
                                j,
                            )
    if (
        cache_y_tangent_grown
        and not y_completed_cached
        and y_completed_cache is not None
        and y_completed_cache_key is not None
    ):
        y_completed_cache[y_completed_cache_key] = (source_y, dict(ygrown))
    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    if project_v1_packages:
        weighted_is_final = (
            cached_base_parts is not None
            and cached_base_parts.get("weighted") is not None
        )
        if (
            cached_base_parts is not None
            and cached_base_parts.get("weighted") is not None
        ):
            weighted = dict(cached_base_parts["weighted"])
        else:
            weighted = _cached_context_new_site_weighted_packages(
                source_block,
                (
                    "bilinear_project_v1",
                    id(narg),
                    bool(include_even),
                    even_key,
                    bool(carry_rdm_operators),
                    bool(carry_pair_composites),
                ),
                grown,
                h1e_full,
                eri_full,
                site_count,
                future_sites,
                build_v1=False,
            )
        xweighted_is_final = (
            (not x_is_zero)
            and x_pre_rotation_parts is not None
            and x_pre_rotation_parts.get("dweighted") is not None
        )
        if x_is_zero:
            xweighted = {}
        elif (
            x_pre_rotation_parts is not None
            and x_pre_rotation_parts.get("dweighted") is not None
        ):
            xweighted = dict(x_pre_rotation_parts["dweighted"])
        else:
            _, xweighted = _new_site_weighted_packages_tangent(
                grown,
                xgrown,
                h1e_full,
                xh1e_full,
                eri_full,
                xeri_full,
                site_count,
                future_sites,
                build_v1=False,
                actual=weighted,
            )
        _, xyweighted = _new_site_weighted_packages_bilinear(
            grown,
            xgrown,
            ygrown,
            xygrown,
            h1e_full,
            xh1e_full,
            yh1e_full,
            xyh1e_full,
            eri_full,
            xeri_full,
            yeri_full,
            xyeri_full,
            site_count,
            future_sites,
            build_v1=False,
            actual=weighted,
        )
        v1_actual = (
            None
            if cached_base_parts is None
            else cached_base_parts.get("v1_actual")
        )
        if v1_actual is not None:
            v1_actual = dict(v1_actual)
        else:
            v1_actual = _cached_grown_reduced_v1_packages(
                source_block,
                grown,
                h1e_full,
                eri_full,
                future_sites,
            )
        if x_is_zero:
            xv1_new = {}
        elif (
            x_pre_rotation_parts is not None
            and x_pre_rotation_parts.get("dv1_new") is not None
        ):
            xv1_new = dict(x_pre_rotation_parts["dv1_new"])
        else:
            v1_actual, xv1_new = _grown_reduced_v1_packages_tangent(
                source_block,
                source_x,
                grown,
                h1e_full,
                xh1e_full,
                eri_full,
                xeri_full,
                future_sites,
            )
        y_package_cache_key = None
        y_package_cache = None
        y_package_cached = False
        yweighted_is_final = (
            y_pre_rotation_parts is not None
            and y_pre_rotation_parts.get("dweighted") is not None
            and y_pre_rotation_parts.get("dv1_new") is not None
        )
        if (
            y_pre_rotation_parts is not None
            and y_pre_rotation_parts.get("dweighted") is not None
            and y_pre_rotation_parts.get("dv1_new") is not None
        ):
            yweighted = dict(y_pre_rotation_parts["dweighted"])
            yv1_new = dict(y_pre_rotation_parts["dv1_new"])
            y_package_cached = True
        elif cache_y_tangent_grown:
            y_package_cache_key = (
                id(narg),
                id(source_y),
                tuple(int(q) for q in future_sites),
                id(h1e_full),
                id(yh1e_full),
                id(eri_full),
                id(yeri_full),
            )
            y_package_cache = getattr(
                source_block,
                "_su2_response_y_pre_rotation_package_cache",
                None,
            )
            if y_package_cache is None:
                y_package_cache = {}
                setattr(
                    source_block,
                    "_su2_response_y_pre_rotation_package_cache",
                    y_package_cache,
                )
            cached = y_package_cache.get(y_package_cache_key)
            if cached is not None and cached[0] is source_y:
                yweighted = dict(cached[1])
                yv1_new = dict(cached[2])
                y_package_cached = True
        if not y_package_cached:
            _, yweighted = _new_site_weighted_packages_tangent(
                grown,
                ygrown,
                h1e_full,
                yh1e_full,
                eri_full,
                yeri_full,
                site_count,
                future_sites,
                build_v1=False,
                actual=weighted,
            )
            _, yv1_new = _grown_reduced_v1_packages_tangent(
                source_block,
                source_y,
                grown,
                h1e_full,
                yh1e_full,
                eri_full,
                yeri_full,
                future_sites,
            )
            if (
                cache_y_tangent_grown
                and y_package_cache is not None
                and y_package_cache_key is not None
            ):
                y_package_cache[y_package_cache_key] = (
                    source_y,
                    dict(yweighted),
                    dict(yv1_new),
                )
        _, xyv1_new = _grown_reduced_v1_packages_bilinear(
            source_block,
            source_x,
            source_y,
            source_xy,
            grown,
            h1e_full,
            xh1e_full,
            yh1e_full,
            xyh1e_full,
            eri_full,
            xeri_full,
            yeri_full,
            xyeri_full,
            future_sites,
        )
        carried_terms = {}
        xcarried_terms = {}
        ycarried_terms = {}
        xycarried_terms = {}
        even_prefixes = {
            "NextDensity",
            "NextExchangeDensity",
            "NextExchangeSpinDensity",
            "NextPairAnnihilate",
        }
        odd_prefixes = {"NextV1Spinor", "NextV3Cdag"}
        for q in future_sites:
            q = int(q)
            for prefix in even_prefixes:
                key = (prefix, q)
                terms = []
                xterms = []
                yterms = []
                xyterms = []
                carried = source_block.reduced_operators.get(key)
                xcarried = source_x.reduced_operators.get(key)
                ycarried = source_y.reduced_operators.get(key)
                xycarried = source_xy.reduced_operators.get(key)
                if weighted_is_final:
                    if key in weighted:
                        terms.append(weighted[key])
                elif carried is not None:
                    terms.append(
                        _cached_grow_source_tensor(
                            source_block, carried, local_name="I"
                        )
                    )
                if xweighted_is_final:
                    if key in xweighted:
                        xterms.append(xweighted[key])
                elif xcarried is not None:
                    xterms.append(
                        _cached_grow_source_tensor(
                            source_block, xcarried, local_name="I"
                        )
                    )
                if yweighted_is_final:
                    if key in yweighted:
                        yterms.append(yweighted[key])
                elif ycarried is not None:
                    yterms.append(
                        _cached_grow_source_tensor(
                            source_block, ycarried, local_name="I"
                        )
                    )
                if xycarried is not None:
                    xyterms.append(
                        _cached_grow_source_tensor(
                            source_block, xycarried, local_name="I"
                        )
                    )
                if (not weighted_is_final) and key in weighted:
                    terms.append(weighted[key])
                if (not xweighted_is_final) and key in xweighted:
                    xterms.append(xweighted[key])
                if (not yweighted_is_final) and key in yweighted:
                    yterms.append(yweighted[key])
                if key in xyweighted:
                    xyterms.append(xyweighted[key])
                total = add_optional_reduced_terms(terms)
                xtotal = add_optional_reduced_terms(xterms)
                ytotal = add_optional_reduced_terms(yterms)
                xytotal = add_optional_reduced_terms(xyterms)
                if total is not None:
                    carried_terms[key] = total
                if xtotal is not None:
                    xcarried_terms[key] = xtotal
                if ytotal is not None:
                    ycarried_terms[key] = ytotal
                if xytotal is not None:
                    xycarried_terms[key] = xytotal
            for prefix in odd_prefixes:
                key = (prefix, q)
                terms = []
                xterms = []
                yterms = []
                xyterms = []
                carried = source_block.reduced_operators.get(key)
                xcarried = source_x.reduced_operators.get(key)
                ycarried = source_y.reduced_operators.get(key)
                xycarried = source_xy.reduced_operators.get(key)
                if weighted_is_final:
                    if key in weighted:
                        terms.append(weighted[key])
                elif carried is not None:
                    terms.append(
                        _cached_grow_source_tensor(
                            source_block, carried, local_name="JW"
                        )
                    )
                if xweighted_is_final:
                    if key in xweighted:
                        xterms.append(xweighted[key])
                elif xcarried is not None:
                    xterms.append(
                        _cached_grow_source_tensor(
                            source_block, xcarried, local_name="JW"
                        )
                    )
                if yweighted_is_final:
                    if key in yweighted:
                        yterms.append(yweighted[key])
                elif ycarried is not None:
                    yterms.append(
                        _cached_grow_source_tensor(
                            source_block, ycarried, local_name="JW"
                        )
                    )
                if xycarried is not None:
                    xyterms.append(
                        _cached_grow_source_tensor(
                            source_block, xycarried, local_name="JW"
                        )
                    )
                if prefix == "NextV1Spinor":
                    if (not weighted_is_final) and key in v1_actual:
                        terms.append(v1_actual[key])
                    if (not xweighted_is_final) and key in xv1_new:
                        xterms.append(xv1_new[key])
                    if (not yweighted_is_final) and key in yv1_new:
                        yterms.append(yv1_new[key])
                    if key in xyv1_new:
                        xyterms.append(xyv1_new[key])
                else:
                    if (not weighted_is_final) and key in weighted:
                        terms.append(weighted[key])
                    if (not xweighted_is_final) and key in xweighted:
                        xterms.append(xweighted[key])
                    if (not yweighted_is_final) and key in yweighted:
                        yterms.append(yweighted[key])
                    if key in xyweighted:
                        xyterms.append(xyweighted[key])
                total = add_optional_reduced_terms(terms)
                xtotal = add_optional_reduced_terms(xterms)
                ytotal = add_optional_reduced_terms(yterms)
                xytotal = add_optional_reduced_terms(xyterms)
                if total is not None:
                    carried_terms[key] = total
                if xtotal is not None:
                    xcarried_terms[key] = xtotal
                if ytotal is not None:
                    ycarried_terms[key] = ytotal
                if xytotal is not None:
                    xycarried_terms[key] = xytotal
        weighted = carried_terms
        xweighted = xcarried_terms
        yweighted = ycarried_terms
        xyweighted = xycarried_terms
    else:
        if (
            cached_base_parts is not None
            and cached_base_parts.get("weighted") is not None
        ):
            weighted = dict(cached_base_parts["weighted"])
        else:
            weighted = weighted_packages_from_operators(
                grown,
                h1e_full,
                eri_full,
                site_count,
                future_sites,
                build_v1=True,
            )
        if x_is_zero:
            xweighted = {}
        elif (
            x_pre_rotation_parts is not None
            and x_pre_rotation_parts.get("dweighted") is not None
        ):
            xweighted = dict(x_pre_rotation_parts["dweighted"])
        else:
            xweighted = _weighted_packages_tangent(
                grown,
                xgrown,
                h1e_full,
                xh1e_full,
                eri_full,
                xeri_full,
                site_count,
                future_sites,
                build_v1=True,
            )
        if (
            y_pre_rotation_parts is not None
            and y_pre_rotation_parts.get("dweighted") is not None
        ):
            yweighted = dict(y_pre_rotation_parts["dweighted"])
        else:
            yweighted = _weighted_packages_tangent(
                grown,
                ygrown,
                h1e_full,
                yh1e_full,
                eri_full,
                yeri_full,
                site_count,
                future_sites,
                build_v1=True,
            )
        xyweighted = _weighted_packages_bilinear(
            grown,
            xgrown,
            ygrown,
            xygrown,
            h1e_full,
            xh1e_full,
            yh1e_full,
            xyh1e_full,
            eri_full,
            xeri_full,
            yeri_full,
            xyeri_full,
            site_count,
            future_sites,
            build_v1=True,
        )
    return (
        {**grown, **weighted},
        {**xgrown, **xweighted},
        {**ygrown, **yweighted},
        {**xygrown, **xyweighted},
    )


def _pre_rotation_tensors_and_bilinears_adjoint_x(
    narg,
    source_x,
    source_y,
    source_xy,
    tensor_x_adjoint,
    tensor_xy_adjoint,
    h1e_full,
    eri_full,
    yh1e_full,
    yeri_full,
    future_sites,
    *,
    project_v1_packages=False,
    carry_rdm_operators=False,
    x_pre_rotation_parts=None,
    y_pre_rotation_parts=None,
):
    """Adjoint of bilinear pre-rotation tensor assembly wrt ``x``/``xy`` inputs."""
    if not project_v1_packages:
        raise NotImplementedError(
            "bilinear pre-rotation adjoint is implemented for projected V1 packages"
        )

    from .su2_chain import (
        complete_density_composites,
        complete_pair_composites,
        grown_coupling_operators,
    )

    source_block = getattr(narg, "_su2_source_renormalized_block", None)
    if source_block is None:
        raise ValueError("grown NARG object does not carry its source block")
    future_sites = tuple(int(q) for q in future_sites)
    carry_pair_composites = bool(project_v1_packages and len(future_sites) > 1)
    include_even = (
        (not project_v1_packages)
        or carry_pair_composites
        or carry_rdm_operators
    )
    even_composites = (
        {"Density"}
        if carry_rdm_operators and project_v1_packages and not carry_pair_composites
        else None
    )

    cached_base_parts = x_pre_rotation_parts or y_pre_rotation_parts
    if cached_base_parts is not None and cached_base_parts.get("grown") is not None:
        grown = dict(cached_base_parts["grown"])
    else:
        grown = grown_coupling_operators(
            narg,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    if x_pre_rotation_parts is not None and x_pre_rotation_parts.get("dgrown") is not None:
        xgrown = dict(x_pre_rotation_parts["dgrown"])
    else:
        xgrown = _grown_coupling_operators_tangent(
            narg,
            source_x,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    if y_pre_rotation_parts is not None and y_pre_rotation_parts.get("dgrown") is not None:
        ygrown = dict(y_pre_rotation_parts["dgrown"])
    else:
        ygrown = _grown_coupling_operators_tangent(
            narg,
            source_y,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
    xygrown = _grown_coupling_operators_tangent(
        narg,
        source_xy,
        include_even_composites=include_even,
        even_composites=even_composites,
    )

    if carry_rdm_operators:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        if cached_base_parts is None or cached_base_parts.get("grown") is None:
            grown = complete_density_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                key = ("Density", i, j)
                if key not in grown:
                    continue
                if key not in xgrown:
                    xgrown[key] = _density_tangent(grown, xgrown, {}, {}, i, j)
                if key not in ygrown:
                    ygrown[key] = _density_tangent(grown, ygrown, {}, {}, i, j)
                if key not in xygrown:
                    xygrown[key] = _density_bilinear(
                        grown,
                        xgrown,
                        ygrown,
                        xygrown,
                        {},
                        {},
                        {},
                        {},
                        i,
                        j,
                    )
    if carry_pair_composites:
        site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
        if cached_base_parts is None or cached_base_parts.get("grown") is None:
            grown = complete_pair_composites(grown, site_count, source_block=source_block)
        for i in range(site_count):
            for j in range(site_count):
                for name in ("Density", "SpinDensity"):
                    key = (name, i, j)
                    if key not in grown:
                        continue
                    if key not in xgrown:
                        if name == "Density":
                            xgrown[key] = _density_tangent(grown, xgrown, {}, {}, i, j)
                        else:
                            xgrown[key] = _spin_density_tangent(grown, xgrown, {}, {}, i, j)
                    if key not in ygrown:
                        if name == "Density":
                            ygrown[key] = _density_tangent(grown, ygrown, {}, {}, i, j)
                        else:
                            ygrown[key] = _spin_density_tangent(grown, ygrown, {}, {}, i, j)
                    if key not in xygrown:
                        if name == "Density":
                            xygrown[key] = _density_bilinear(
                                grown,
                                xgrown,
                                ygrown,
                                xygrown,
                                {},
                                {},
                                {},
                                {},
                                i,
                                j,
                            )
                        else:
                            xygrown[key] = _spin_density_bilinear(
                                grown,
                                xgrown,
                                ygrown,
                                xygrown,
                                {},
                                {},
                                {},
                                {},
                                i,
                                j,
                            )

    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    source_x_operator_adjoint = {}
    source_xy_operator_adjoint = {}
    xgrown_adjoint = {}
    xygrown_adjoint = {}
    xweighted_adjoint = {}
    xyweighted_adjoint = {}
    xv1_adjoint = {}
    xyv1_adjoint = {}
    xh1_adj = np.zeros_like(h1e_full, dtype=float)
    xeri_adj = np.zeros_like(eri_full, dtype=float)
    xyh1_adj = np.zeros_like(h1e_full, dtype=float)
    xyeri_adj = np.zeros_like(eri_full, dtype=float)

    even_prefixes = {
        "NextDensity",
        "NextExchangeDensity",
        "NextExchangeSpinDensity",
        "NextPairAnnihilate",
    }
    odd_prefixes = {"NextV1Spinor", "NextV3Cdag"}
    next_prefixes = even_prefixes | odd_prefixes

    if tensor_x_adjoint:
        if x_pre_rotation_parts is None:
            raise ValueError("x pre-rotation parts are required for x adjoints")
        op_adj, h_adj, g_adj = _pre_rotation_tensors_and_tangents_adjoint(
            narg,
            source_x,
            tensor_x_adjoint,
            h1e_full,
            eri_full,
            future_sites,
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=carry_rdm_operators,
            parts=x_pre_rotation_parts,
        )
        _merge_reduced_adjoint(source_x_operator_adjoint, op_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj

    for key, adj in (tensor_xy_adjoint or {}).items():
        if not isinstance(key, tuple) or key[0] not in next_prefixes:
            _add_reduced_adjoint(xygrown_adjoint, key, adj)
            continue
        local_name = "JW" if key[0] in odd_prefixes else "I"
        carried_adj = _grow_source_tensor_tangent_adjoint(
            source_block,
            source_xy,
            key,
            adj,
            local_name=local_name,
        )
        _add_reduced_adjoint(source_xy_operator_adjoint, key, carried_adj)
        if key[0] == "NextV1Spinor":
            _add_reduced_adjoint(xyv1_adjoint, key, adj)
        else:
            _add_reduced_adjoint(xyweighted_adjoint, key, adj)

    if xweighted_adjoint:
        op_adj, h_adj, g_adj = _new_site_weighted_packages_tangent_adjoint(
            grown,
            xgrown,
            xweighted_adjoint,
            h1e_full,
            eri_full,
            site_count,
            future_sites,
        )
        _merge_reduced_adjoint(xgrown_adjoint, op_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj

    if xyweighted_adjoint:
        (
            xop_adj,
            xyop_adj,
            h_adj,
            g_adj,
            xy_h_adj,
            xy_g_adj,
        ) = _new_site_weighted_packages_bilinear_adjoint_x(
            grown,
            xgrown,
            ygrown,
            xygrown,
            xyweighted_adjoint,
            h1e_full,
            eri_full,
            yeri_full,
            site_count,
            future_sites,
        )
        _merge_reduced_adjoint(xgrown_adjoint, xop_adj)
        _merge_reduced_adjoint(xygrown_adjoint, xyop_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
        xyh1_adj += xy_h_adj
        xyeri_adj += xy_g_adj

    if xv1_adjoint:
        op_adj, h_adj, g_adj = _grown_reduced_v1_packages_tangent_adjoint(
            source_block,
            source_x,
            grown,
            xv1_adjoint,
            h1e_full,
            eri_full,
            future_sites,
        )
        _merge_reduced_adjoint(source_x_operator_adjoint, op_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj

    if xyv1_adjoint:
        (
            xop_adj,
            xyop_adj,
            h_adj,
            g_adj,
            xy_h_adj,
            xy_g_adj,
        ) = _grown_reduced_v1_packages_bilinear_adjoint_x(
            source_block,
            source_x,
            source_y,
            source_xy,
            grown,
            xyv1_adjoint,
            h1e_full,
            eri_full,
            yh1e_full,
            yeri_full,
            future_sites,
        )
        _merge_reduced_adjoint(source_x_operator_adjoint, xop_adj)
        _merge_reduced_adjoint(source_xy_operator_adjoint, xyop_adj)
        xh1_adj += h_adj
        xeri_adj += g_adj
        xyh1_adj += xy_h_adj
        xyeri_adj += xy_g_adj

    def reduce_xgrown_algebraic():
        new_site = site_count - 1
        while True:
            keys = [
                key
                for key in xgrown_adjoint
                if (
                    isinstance(key, tuple)
                    and len(key) == 3
                    and key[0] in {"Density", "SpinDensity"}
                    and new_site in (int(key[1]), int(key[2]))
                )
            ]
            if not keys:
                return
            for key in keys:
                adj = xgrown_adjoint.pop(key)
                dops_without_direct = dict(xgrown)
                dops_without_direct.pop(key, None)
                if key[0] == "Density":
                    op_adj = _density_tangent_adjoint(
                        grown,
                        dops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    )
                else:
                    op_adj = _spin_density_tangent_adjoint(
                        grown,
                        dops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    )
                _merge_reduced_adjoint(xgrown_adjoint, op_adj)

    def reduce_xygrown_algebraic():
        new_site = site_count - 1
        while True:
            keys = [
                key
                for key in xygrown_adjoint
                if (
                    isinstance(key, tuple)
                    and len(key) == 3
                    and key[0] in {"Density", "SpinDensity"}
                    and new_site in (int(key[1]), int(key[2]))
                )
            ]
            if not keys:
                return
            for key in keys:
                adj = xygrown_adjoint.pop(key)
                xyops_without_direct = dict(xygrown)
                xyops_without_direct.pop(key, None)
                if key[0] == "Density":
                    xop_adj, xyop_adj = _density_bilinear_adjoint_x(
                        grown,
                        xgrown,
                        ygrown,
                        xyops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    )
                else:
                    xop_adj, xyop_adj = _spin_density_bilinear_adjoint_x(
                        grown,
                        xgrown,
                        ygrown,
                        xyops_without_direct,
                        adj,
                        key[1],
                        key[2],
                    )
                _merge_reduced_adjoint(xgrown_adjoint, xop_adj)
                _merge_reduced_adjoint(xygrown_adjoint, xyop_adj)

    reduce_xygrown_algebraic()
    reduce_xgrown_algebraic()
    if xgrown_adjoint:
        op_adj = _grown_coupling_operators_tangent_adjoint(
            narg,
            source_x,
            xgrown_adjoint,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
        _merge_reduced_adjoint(source_x_operator_adjoint, op_adj)
    if xygrown_adjoint:
        op_adj = _grown_coupling_operators_tangent_adjoint(
            narg,
            source_xy,
            xygrown_adjoint,
            include_even_composites=include_even,
            even_composites=even_composites,
        )
        _merge_reduced_adjoint(source_xy_operator_adjoint, op_adj)

    return (
        source_x_operator_adjoint,
        source_xy_operator_adjoint,
        xh1_adj,
        xeri_adj,
        xyh1_adj,
        xyeri_adj,
    )


def _grown_hamiltonian_tangent(source_block, source_tangent, h1e_block, dh1e_block, eri_block, deri_block, *, site_index):
    from .su2_three_site import direct_reduced_base_tensor, direct_reduced_full_hamiltonian_tensor

    zeros_h = np.zeros_like(h1e_block)
    zeros_g = np.zeros_like(eri_block)
    local_part = _add_irrep_tensors(
        direct_reduced_base_tensor(source_block, dh1e_block, deri_block, site_index=site_index),
        _scale_irrep_tensor(
            direct_reduced_base_tensor(source_block, zeros_h, zeros_g, site_index=site_index),
            -1.0,
        ),
    )
    source_part = direct_reduced_full_hamiltonian_tensor(
        source_tangent,
        zeros_h,
        zeros_g,
        site_index=site_index,
    )
    return _add_irrep_tensors(local_part, source_part)


def _grown_hamiltonian_tangent_adjoint(
    source_block,
    source_tangent,
    scalar_adjoint,
    h1e_block,
    eri_block,
    *,
    site_index,
):
    """Adjoint of ``_grown_hamiltonian_tangent``."""
    from .su2_three_site import direct_reduced_base_tensor

    source_h_adj, source_op_adj = direct_reduced_full_hamiltonian_tangent_adjoint(
        source_tangent,
        scalar_adjoint,
        site_index=site_index,
    )
    dh1_adj = np.zeros_like(h1e_block, dtype=float)
    deri_adj = np.zeros_like(eri_block, dtype=float)

    site_index = int(site_index)
    zeros_h = np.zeros_like(h1e_block)
    zeros_g = np.zeros_like(eri_block)

    unit_h = np.zeros_like(h1e_block)
    unit_h[site_index, site_index] = 1.0
    h_tensor = _add_irrep_tensors(
        direct_reduced_base_tensor(
            source_block,
            unit_h,
            zeros_g,
            site_index=site_index,
        ),
        _scale_irrep_tensor(
            direct_reduced_base_tensor(
                source_block,
                zeros_h,
                zeros_g,
                site_index=site_index,
            ),
            -1.0,
        ),
    )
    dh1_adj[site_index, site_index] = _irrep_tensor_pairing(
        scalar_adjoint,
        h_tensor,
    )

    unit_g = np.zeros_like(eri_block)
    unit_g[site_index, site_index, site_index, site_index] = 1.0
    g_tensor = _add_irrep_tensors(
        direct_reduced_base_tensor(
            source_block,
            zeros_h,
            unit_g,
            site_index=site_index,
        ),
        _scale_irrep_tensor(
            direct_reduced_base_tensor(
                source_block,
                zeros_h,
                zeros_g,
                site_index=site_index,
            ),
            -1.0,
        ),
    )
    deri_adj[site_index, site_index, site_index, site_index] = _irrep_tensor_pairing(
        scalar_adjoint,
        g_tensor,
    )

    return source_h_adj, source_op_adj, dh1_adj, deri_adj


def _grown_hamiltonian_bilinear(
    source_block,
    source_xy,
    h1e_block,
    xydh1e_block,
    eri_block,
    xyderi_block,
    *,
    site_index,
):
    from .su2_three_site import direct_reduced_base_tensor, direct_reduced_full_hamiltonian_tensor

    zeros_h = np.zeros_like(h1e_block)
    zeros_g = np.zeros_like(eri_block)
    local_part = _add_irrep_tensors(
        direct_reduced_base_tensor(
            source_block,
            xydh1e_block,
            xyderi_block,
            site_index=site_index,
        ),
        _scale_irrep_tensor(
            direct_reduced_base_tensor(
                source_block,
                zeros_h,
                zeros_g,
                site_index=site_index,
            ),
            -1.0,
        ),
    )
    source_part = direct_reduced_full_hamiltonian_tensor(
        source_xy,
        zeros_h,
        zeros_g,
        site_index=site_index,
    )
    return _add_irrep_tensors(local_part, source_part)


def _grown_hamiltonian_bilinear_adjoint(
    source_block,
    source_xy,
    scalar_adjoint,
    h1e_block,
    eri_block,
    *,
    site_index,
):
    """Adjoint of ``_grown_hamiltonian_bilinear`` wrt ``xy`` inputs."""
    return _grown_hamiltonian_tangent_adjoint(
        source_block,
        source_xy,
        scalar_adjoint,
        h1e_block,
        eri_block,
        site_index=site_index,
    )


def recursive_perturbation_for_active_integrals(solver, dh1e, deri, *, state_id: int = 0):
    """Propagate an active Hamiltonian perturbation through all SU2-NARG truncations."""
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")
    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    dh1e = np.asarray(dh1e)
    deri = np.asarray(deri)
    final_size = int(h1e.shape[0])
    target_nelec, target_j2 = solver.target_irrep
    del state_id

    if final_size < 2:
        raise ValueError("recursive response requires at least two active sites")
    source_tangent, min_gap = _seed_two_site_tangent_block(
        solver.chain.blocks[2],
        dh1e[:2, :2],
        deri[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        dh1e_full=dh1e,
        deri_full=deri,
        final_size=final_size,
        project_v1_packages=bool(
            solver.timings.get("project_v1_packages", False)
            if solver.timings is not None
            else False
        ),
        include_retained_mixing=final_size != 2,
    )
    if final_size == 2:
        target_irrep = Irrep((int(target_nelec), int(target_j2)))
        block = source_tangent.hamiltonian.block(target_irrep, target_irrep)
        return RecursivePerturbation(
            tensor=source_tangent.hamiltonian,
            block=block,
            min_gap=float(min_gap),
            block_count=1,
        )

    block_count = 1
    for nsites in range(3, final_size + 1):
        source_block = solver.chain.blocks[nsites - 1]
        grown = solver.chain.final if nsites == final_size else solver.chain.blocks[nsites].truncated.source
        h1e_n = h1e[:nsites, :nsites]
        eri_n = eri[:nsites, :nsites, :nsites, :nsites]
        dh1e_n = dh1e[:nsites, :nsites]
        deri_n = deri[:nsites, :nsites, :nsites, :nsites]
        d_grown_h = _grown_hamiltonian_tangent(
            source_block,
            source_tangent,
            h1e_n,
            dh1e_n,
            eri_n,
            deri_n,
            site_index=nsites - 1,
        )
        if nsites == final_size:
            target_irrep = Irrep((int(target_nelec), int(target_j2)))
            return RecursivePerturbation(
                tensor=d_grown_h,
                block=d_grown_h.block(target_irrep, target_irrep),
                min_gap=float(min_gap),
                block_count=int(block_count),
            )

        block = solver.chain.blocks[nsites]
        response = truncation_tangent(
            grown,
            block.truncated,
            d_grown_h,
            include_retained_mixing=True,
        )
        min_gap = min(float(min_gap), float(response.min_gap))
        tensors, dtensors = _pre_rotation_tensors_and_tangents(
            grown,
            source_tangent,
            h1e,
            dh1e,
            eri,
            deri,
            tuple(range(nsites, final_size)),
            project_v1_packages=bool(
                solver.timings.get("project_v1_packages", False)
                if solver.timings is not None
                else False
            ),
            carry_rdm_operators=bool(
                solver.timings.get("carry_rdm_operators", False)
                if solver.timings is not None
                else False
            ),
        )
        dops = _rotate_reduced_tensors_tangent(
            block.truncated,
            tensors,
            dtensors,
            response,
        )
        source_tangent = _make_tangent_block(block, response.d_hamiltonian, dops)
        block_count += 1

    raise RuntimeError("recursive response did not reach the final NARG block")


def recursive_tangent_path_for_active_integrals(
    solver,
    dh1e,
    deri,
    *,
    state_id: int = 0,
):
    """Return the intermediate first-order recursive path for one perturbation."""
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")
    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    dh1e = np.asarray(dh1e)
    deri = np.asarray(deri)
    final_size = int(h1e.shape[0])
    del state_id

    if final_size < 2:
        raise ValueError("recursive response requires at least two active sites")
    source_tangent, min_gap, seed_response = _seed_two_site_tangent_block(
        solver.chain.blocks[2],
        dh1e[:2, :2],
        deri[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        dh1e_full=dh1e,
        deri_full=deri,
        final_size=final_size,
        project_v1_packages=bool(
            solver.timings.get("project_v1_packages", False)
            if solver.timings is not None
            else False
        ),
        include_retained_mixing=final_size != 2,
        return_response=True,
    )
    sources = {2: source_tangent}
    grown_hamiltonians = {}
    responses = {2: seed_response}
    pre_rotation_parts = {}
    block_count = 1
    if final_size == 2:
        return RecursiveTangentPath(
            sources=sources,
            grown_hamiltonians=grown_hamiltonians,
            responses=responses,
            min_gap=float(min_gap),
            block_count=int(block_count),
            pre_rotation_parts=pre_rotation_parts,
        )

    for nsites in range(3, final_size + 1):
        source_block = solver.chain.blocks[nsites - 1]
        grown = (
            solver.chain.final
            if nsites == final_size
            else solver.chain.blocks[nsites].truncated.source
        )
        h1e_n = h1e[:nsites, :nsites]
        eri_n = eri[:nsites, :nsites, :nsites, :nsites]
        dh1e_n = dh1e[:nsites, :nsites]
        deri_n = deri[:nsites, :nsites, :nsites, :nsites]
        d_grown_h = _grown_hamiltonian_tangent(
            source_block,
            source_tangent,
            h1e_n,
            dh1e_n,
            eri_n,
            deri_n,
            site_index=nsites - 1,
        )
        grown_hamiltonians[nsites] = d_grown_h
        if nsites == final_size:
            return RecursiveTangentPath(
                sources=sources,
                grown_hamiltonians=grown_hamiltonians,
                responses=responses,
                min_gap=float(min_gap),
                block_count=int(block_count),
                pre_rotation_parts=pre_rotation_parts,
            )

        block = solver.chain.blocks[nsites]
        response = truncation_tangent(
            grown,
            block.truncated,
            d_grown_h,
            include_retained_mixing=True,
        )
        responses[nsites] = response
        min_gap = min(float(min_gap), float(response.min_gap))
        tensors, dtensors, parts = _pre_rotation_tensors_and_tangents(
            grown,
            source_tangent,
            h1e,
            dh1e,
            eri,
            deri,
            tuple(range(nsites, final_size)),
            project_v1_packages=bool(
                solver.timings.get("project_v1_packages", False)
                if solver.timings is not None
                else False
            ),
            carry_rdm_operators=bool(
                solver.timings.get("carry_rdm_operators", False)
                if solver.timings is not None
                else False
            ),
            cache_tangent_grown=True,
            return_parts=True,
        )
        pre_rotation_parts[nsites] = parts
        dops = _rotate_reduced_tensors_tangent(
            block.truncated,
            tensors,
            dtensors,
            response,
        )
        source_tangent = _make_tangent_block(block, response.d_hamiltonian, dops)
        sources[nsites] = source_tangent
        block_count += 1

    raise RuntimeError("recursive response path did not reach the final NARG block")


def recursive_bilinear_perturbation_for_active_integrals(
    solver,
    xdh1e,
    xderi,
    ydh1e,
    yderi,
    xydh1e,
    xyderi,
    *,
    state_id: int = 0,
    x_path: RecursiveTangentPath | None = None,
    y_path: RecursiveTangentPath | None = None,
):
    """Propagate a mixed active Hamiltonian perturbation analytically."""
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")
    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    xdh1e = np.asarray(xdh1e)
    xderi = np.asarray(xderi)
    ydh1e = np.asarray(ydh1e)
    yderi = np.asarray(yderi)
    xydh1e = np.asarray(xydh1e)
    xyderi = np.asarray(xyderi)
    final_size = int(h1e.shape[0])
    target_nelec, target_j2 = solver.target_irrep
    x_is_zero = (
        not np.any(np.abs(xdh1e) > 0.0)
        and not np.any(np.abs(xderi) > 0.0)
    )
    del state_id

    if final_size < 2:
        raise ValueError("recursive response requires at least two active sites")
    source_x, source_y, source_xy, min_gap = _seed_two_site_bilinear_block(
        solver.chain.blocks[2],
        xdh1e[:2, :2],
        xderi[:2, :2, :2, :2],
        ydh1e[:2, :2],
        yderi[:2, :2, :2, :2],
        xydh1e[:2, :2],
        xyderi[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        xdh1e_full=xdh1e,
        xderi_full=xderi,
        ydh1e_full=ydh1e,
        yderi_full=yderi,
        xydh1e_full=xydh1e,
        xyderi_full=xyderi,
        final_size=final_size,
        project_v1_packages=bool(
            solver.timings.get("project_v1_packages", False)
            if solver.timings is not None
            else False
        ),
        include_retained_mixing=final_size != 2,
        x_path=x_path,
        y_path=y_path,
    )
    if x_path is not None:
        source_x = x_path.sources.get(2, source_x)
        min_gap = min(float(min_gap), float(x_path.min_gap))
    if y_path is not None:
        source_y = y_path.sources.get(2, source_y)
        min_gap = min(float(min_gap), float(y_path.min_gap))

    target_irrep = Irrep((int(target_nelec), int(target_j2)))
    if final_size == 2:
        block = source_xy.hamiltonian.block(target_irrep, target_irrep)
        return RecursiveBilinearPerturbation(
            tensor=source_xy.hamiltonian,
            block=block,
            min_gap=float(min_gap),
            block_count=1,
        )

    block_count = 1
    for nsites in range(3, final_size + 1):
        source_block = solver.chain.blocks[nsites - 1]
        grown = solver.chain.final if nsites == final_size else solver.chain.blocks[nsites].truncated.source
        if x_path is not None:
            source_x = x_path.sources.get(nsites - 1, source_x)
        if y_path is not None:
            source_y = y_path.sources.get(nsites - 1, source_y)
        h1e_n = h1e[:nsites, :nsites]
        eri_n = eri[:nsites, :nsites, :nsites, :nsites]
        xdh1e_n = xdh1e[:nsites, :nsites]
        xderi_n = xderi[:nsites, :nsites, :nsites, :nsites]
        ydh1e_n = ydh1e[:nsites, :nsites]
        yderi_n = yderi[:nsites, :nsites, :nsites, :nsites]
        xydh1e_n = xydh1e[:nsites, :nsites]
        xyderi_n = xyderi[:nsites, :nsites, :nsites, :nsites]
        if x_path is not None and nsites in x_path.grown_hamiltonians:
            x_grown_h = x_path.grown_hamiltonians[nsites]
        else:
            x_grown_h = _grown_hamiltonian_tangent(
                source_block,
                source_x,
                h1e_n,
                xdh1e_n,
                eri_n,
                xderi_n,
                site_index=nsites - 1,
            )
        if y_path is not None and nsites in y_path.grown_hamiltonians:
            y_grown_h = y_path.grown_hamiltonians[nsites]
        else:
            y_grown_h = _grown_hamiltonian_tangent(
                source_block,
                source_y,
                h1e_n,
                ydh1e_n,
                eri_n,
                yderi_n,
                site_index=nsites - 1,
            )
        xy_grown_h = _grown_hamiltonian_bilinear(
            source_block,
            source_xy,
            h1e_n,
            xydh1e_n,
            eri_n,
            xyderi_n,
            site_index=nsites - 1,
        )
        if nsites == final_size:
            return RecursiveBilinearPerturbation(
                tensor=xy_grown_h,
                block=xy_grown_h.block(target_irrep, target_irrep),
                min_gap=float(min_gap),
                block_count=int(block_count),
            )

        block = solver.chain.blocks[nsites]
        tangent_x = (
            x_path.responses.get(nsites)
            if x_path is not None and nsites in x_path.responses
            else None
        )
        tangent_y = (
            y_path.responses.get(nsites)
            if y_path is not None and nsites in y_path.responses
            else None
        )
        response = truncation_bilinear_tangent(
            grown,
            block.truncated,
            x_grown_h,
            y_grown_h,
            xy_grown_h,
            include_retained_mixing=True,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
        )
        min_gap = min(float(min_gap), float(response.min_gap))
        tensors, xtensors, ytensors, xytensors = _pre_rotation_tensors_and_bilinears(
            grown,
            source_x,
            source_y,
            source_xy,
            h1e,
            xdh1e,
            ydh1e,
            xydh1e,
            eri,
            xderi,
            yderi,
            xyderi,
            tuple(range(nsites, final_size)),
            project_v1_packages=bool(
                solver.timings.get("project_v1_packages", False)
                if solver.timings is not None
                else False
            ),
            carry_rdm_operators=bool(
                solver.timings.get("carry_rdm_operators", False)
                if solver.timings is not None
                else False
            ),
            cache_y_tangent_grown=y_path is not None,
            x_is_zero=x_is_zero,
            x_pre_rotation_parts=(
                x_path.pre_rotation_parts.get(nsites)
                if x_path is not None
                and x_path.pre_rotation_parts is not None
                else None
            ),
            y_pre_rotation_parts=(
                y_path.pre_rotation_parts.get(nsites)
                if y_path is not None
                and y_path.pre_rotation_parts is not None
                else None
            ),
        )
        if x_path is not None and nsites in x_path.sources:
            xops = None
            source_x = x_path.sources[nsites]
        else:
            xops = (
                {}
                if x_is_zero
                else _rotate_reduced_tensors_tangent(
                    block.truncated,
                    tensors,
                    xtensors,
                    response.x,
                )
            )
        xyops = _rotate_reduced_tensors_bilinear(
            block.truncated,
            tensors,
            xtensors,
            ytensors,
            xytensors,
            response,
        )
        if xops is not None:
            source_x = _make_tangent_block(block, response.x.d_hamiltonian, xops)
        if y_path is not None and nsites in y_path.sources:
            source_y = y_path.sources[nsites]
        else:
            yops = _rotate_reduced_tensors_tangent(
                block.truncated,
                tensors,
                ytensors,
                response.y,
            )
            source_y = _make_tangent_block(block, response.y.d_hamiltonian, yops)
        source_xy = _make_tangent_block(block, response.dxy_hamiltonian, xyops)
        block_count += 1

    raise RuntimeError("recursive mixed response did not reach the final NARG block")


def _symmetric_h1_keys(nsites: int) -> tuple[tuple[int, int], ...]:
    nsites = int(nsites)
    return tuple((p, q) for p in range(nsites) for q in range(p, nsites))


def _canonical_pair(p: int, q: int) -> tuple[int, int]:
    p = int(p)
    q = int(q)
    return (p, q) if p <= q else (q, p)


def _symmetric_eri_keys(nsites: int) -> tuple[tuple[int, int, int, int], ...]:
    pairs = _symmetric_h1_keys(nsites)
    keys = []
    for left_index, (p, q) in enumerate(pairs):
        for r, s in pairs[left_index:]:
            keys.append((p, q, r, s))
    return tuple(keys)


def symmetric_active_integral_basis_size(nsites: int) -> int:
    """Return the number of independent real symmetric CAS integral components."""
    nsite = int(nsites)
    npair = nsite * (nsite + 1) // 2
    return npair + npair * (npair + 1) // 2


def _eri_symmetry_positions(key) -> tuple[tuple[int, int, int, int], ...]:
    p, q, r, s = (int(x) for x in key)
    left = {(p, q), (q, p)}
    right = {(r, s), (s, r)}
    positions = set()
    for a, b in left:
        for c, d in right:
            positions.add((a, b, c, d))
            positions.add((c, d, a, b))
    return tuple(sorted(positions))


def _symmetrized_h1_component(h1e, key):
    p, q = key
    if p == q:
        return h1e[p, q]
    return 0.5 * (h1e[p, q] + h1e[q, p])


def _symmetrized_eri_component(eri, key):
    positions = _eri_symmetry_positions(key)
    return sum(eri[pos] for pos in positions) / float(len(positions))


def active_symmetric_integral_coefficients(dh1e, deri, basis: RecursiveActiveResponseBasis):
    """Project symmetric active integral perturbations onto a response basis."""
    dh1e = np.asarray(dh1e)
    deri = np.asarray(deri)
    h_coeff = np.asarray(
        [_symmetrized_h1_component(dh1e, key) for key in basis.h1_keys],
        dtype=np.result_type(dh1e, deri, complex),
    )
    eri_coeff = np.asarray(
        [_symmetrized_eri_component(deri, key) for key in basis.eri_keys],
        dtype=np.result_type(dh1e, deri, complex),
    )
    return h_coeff, eri_coeff


def active_symmetric_adjoint_arrays_from_values(
    nsites: int,
    h1_keys,
    eri_keys,
    h_values,
    eri_values,
):
    """Expand symmetric-basis adjoint values into full active adjoint arrays."""
    nsites = int(nsites)
    h_adj = np.zeros((nsites, nsites), dtype=np.result_type(h_values, eri_values, complex))
    g_adj = np.zeros(
        (nsites, nsites, nsites, nsites),
        dtype=np.result_type(h_values, eri_values, complex),
    )
    for value, key in zip(np.asarray(h_values), h1_keys):
        value = np.conjugate(value)
        p, q = (int(x) for x in key)
        if p == q:
            h_adj[p, q] += value
        else:
            h_adj[p, q] += 0.5 * value
            h_adj[q, p] += 0.5 * value
    for value, key in zip(np.asarray(eri_values), eri_keys):
        value = np.conjugate(value)
        positions = _eri_symmetry_positions(key)
        if not positions:
            continue
        share = value / float(len(positions))
        for pos in positions:
            g_adj[pos] += share
    return h_adj, g_adj


def _active_basis_component_perturbation(
    nsites: int,
    h1_keys,
    eri_keys,
    component: int,
):
    """Return the active integral perturbation for one symmetric basis component."""
    nsites = int(nsites)
    component = int(component)
    dh1 = np.zeros((nsites, nsites), dtype=float)
    deri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    if component < len(h1_keys):
        p, q = h1_keys[component]
        dh1[p, q] = 1.0
        dh1[q, p] = 1.0
        return dh1, deri
    eri_key = eri_keys[component - len(h1_keys)]
    for position in _eri_symmetry_positions(eri_key):
        deri[position] = 1.0
    return dh1, deri


def _terminal_block_from_recursive_tangent_path(solver, path: RecursiveTangentPath):
    final_size = int(np.asarray(solver.h1e).shape[0])
    target_nelec, target_j2 = solver.target_irrep
    target = Irrep((int(target_nelec), int(target_j2)))
    if final_size == 2:
        tensor = path.sources[2].hamiltonian
    else:
        tensor = path.grown_hamiltonians[final_size]
    return tensor.block(target, target)


def recursive_active_integral_adjoint_from_path(
    solver,
    path: RecursiveTangentPath,
    terminal_block_adjoint,
):
    """Reverse one recursive tangent path back to active-integral adjoints."""
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")
    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    final_size = int(h1e.shape[0])
    target_nelec, target_j2 = solver.target_irrep
    target = Irrep((int(target_nelec), int(target_j2)))
    dh1_adj = np.zeros_like(h1e, dtype=float)
    deri_adj = np.zeros_like(eri, dtype=float)
    block_adjoint = np.asarray(terminal_block_adjoint, dtype=complex)

    if final_size == 2:
        tensor = path.sources[2].hamiltonian
        terminal_adjoint = IrrepTensor(
            tensor.bra,
            tensor.ket,
            tensor.op,
            {(target, target): block_adjoint},
        )
        return _seed_two_site_tangent_adjoint(
            solver.chain.blocks[2],
            path.sources[2],
            terminal_adjoint,
            {},
            h1e,
            eri,
            final_size=final_size,
            project_v1_packages=bool(
                solver.timings.get("project_v1_packages", False)
                if solver.timings is not None
                else False
            ),
            include_retained_mixing=False,
        )

    final_tensor = path.grown_hamiltonians[final_size]
    terminal_adjoint = IrrepTensor(
        final_tensor.bra,
        final_tensor.ket,
        final_tensor.op,
        {(target, target): block_adjoint},
    )
    source_block = solver.chain.blocks[final_size - 1]
    source_tangent = path.sources[final_size - 1]
    h_adj, op_adj, step_h_adj, step_g_adj = _grown_hamiltonian_tangent_adjoint(
        source_block,
        source_tangent,
        terminal_adjoint,
        h1e[:final_size, :final_size],
        eri[:final_size, :final_size, :final_size, :final_size],
        site_index=final_size - 1,
    )
    dh1_adj[:final_size, :final_size] += step_h_adj
    deri_adj[:final_size, :final_size, :final_size, :final_size] += step_g_adj

    project_v1_packages = bool(
        solver.timings.get("project_v1_packages", False)
        if solver.timings is not None
        else False
    )
    carry_rdm_operators = bool(
        solver.timings.get("carry_rdm_operators", False)
        if solver.timings is not None
        else False
    )
    for nsites in range(final_size - 1, 2, -1):
        h_adj, op_adj, step_h_adj, step_g_adj = _recursive_growth_step_tangent_adjoint(
            solver.chain.blocks[nsites - 1],
            solver.chain.blocks[nsites].truncated.source,
            solver.chain.blocks[nsites],
            path.sources[nsites - 1],
            h_adj,
            op_adj,
            h1e,
            eri,
            tuple(range(nsites, final_size)),
            path.responses[nsites],
            path.pre_rotation_parts[nsites],
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=carry_rdm_operators,
        )
        dh1_adj += step_h_adj
        deri_adj += step_g_adj

    seed_h_adj, seed_g_adj = _seed_two_site_tangent_adjoint(
        solver.chain.blocks[2],
        path.sources[2],
        h_adj,
        op_adj,
        h1e,
        eri,
        final_size=final_size,
        project_v1_packages=project_v1_packages,
        include_retained_mixing=final_size != 2,
    )
    dh1_adj += seed_h_adj
    deri_adj += seed_g_adj
    return dh1_adj, deri_adj


def recursive_active_integral_adjoint_arrays(
    solver,
    bra_vector,
    ket_vector,
    *,
    state_id: int = 0,
    factor=1.0,
    path: RecursiveTangentPath | None = None,
):
    """Return active integral adjoint arrays from one reverse recursive sweep."""
    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    bra = np.asarray(bra_vector, dtype=complex).reshape(-1)
    ket = np.asarray(ket_vector, dtype=complex).reshape(-1)
    terminal_adjoint = factor * np.outer(bra, np.conjugate(ket))
    if path is None:
        dh1_struct = np.ones_like(h1e, dtype=float)
        deri_struct = np.ones_like(eri, dtype=float)
        path = recursive_tangent_path_for_active_integrals(
            solver,
            dh1_struct,
            deri_struct,
            state_id=state_id,
        )
    dh1_adj, deri_adj = recursive_active_integral_adjoint_from_path(
        solver,
        path,
        terminal_adjoint,
    )
    return dh1_adj, deri_adj, path


def recursive_bilinear_active_integral_adjoint_arrays_x(
    solver,
    ydh1e,
    yderi,
    bra_vector,
    ket_vector,
    *,
    state_id: int = 0,
    factor=1.0,
    y_path: RecursiveTangentPath | None = None,
    x_path: RecursiveTangentPath | None = None,
    xdh1e=None,
    xderi=None,
):
    r"""Return reverse-mode active adjoints for ``B_{xy}(x,y,xy)``.

    The returned arrays satisfy, within the fixed recursive truncation pattern,

    $$
    \langle b|B_{xy}(x,y,xy)|k\rangle
      = \langle A_x^h, xh\rangle
      + \langle A_x^g, xg\rangle
      + \langle A_{xy}^h, xyh\rangle
      + \langle A_{xy}^g, xyg\rangle .
    $$
    """
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")

    from .su2_two_site import build_two_site_su2_narg

    h1e = np.asarray(solver.h1e)
    eri = np.asarray(solver.eri)
    ydh1e = np.asarray(ydh1e)
    yderi = np.asarray(yderi)
    final_size = int(h1e.shape[0])
    if xdh1e is None:
        xdh1e = np.ones_like(h1e, dtype=float)
    else:
        xdh1e = np.asarray(xdh1e)
    if xderi is None:
        xderi = np.ones_like(eri, dtype=float)
    else:
        xderi = np.asarray(xderi)
    zero_h = np.zeros_like(h1e)
    zero_g = np.zeros_like(eri)
    project_v1_packages = bool(
        solver.timings.get("project_v1_packages", False)
        if solver.timings is not None
        else False
    )
    carry_rdm_operators = bool(
        solver.timings.get("carry_rdm_operators", False)
        if solver.timings is not None
        else False
    )
    if y_path is None:
        y_path = recursive_tangent_path_for_active_integrals(
            solver,
            ydh1e,
            yderi,
            state_id=state_id,
        )
    if x_path is None:
        x_path = recursive_tangent_path_for_active_integrals(
            solver,
            xdh1e,
            xderi,
            state_id=state_id,
        )

    source_x, source_y, source_xy, seed_min_gap = _seed_two_site_bilinear_block(
        solver.chain.blocks[2],
        xdh1e[:2, :2],
        xderi[:2, :2, :2, :2],
        ydh1e[:2, :2],
        yderi[:2, :2, :2, :2],
        zero_h[:2, :2],
        zero_g[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        xdh1e_full=xdh1e,
        xderi_full=xderi,
        ydh1e_full=ydh1e,
        yderi_full=yderi,
        xydh1e_full=zero_h,
        xyderi_full=zero_g,
        final_size=final_size,
        project_v1_packages=project_v1_packages,
        include_retained_mixing=final_size != 2,
        x_path=x_path,
        y_path=y_path,
    )
    source_xy_by_size = {2: source_xy}
    response_by_size = {
        2: truncation_bilinear_tangent(
            solver.chain.blocks[2].truncated.source,
            solver.chain.blocks[2].truncated,
            build_two_site_su2_narg(
                xdh1e[:2, :2],
                xderi[:2, :2, :2, :2],
            ).hamiltonian,
            build_two_site_su2_narg(
                ydh1e[:2, :2],
                yderi[:2, :2, :2, :2],
            ).hamiltonian,
            _zero_irrep_tensor_like(
                solver.chain.blocks[2].truncated.source.hamiltonian
            ),
            include_retained_mixing=final_size != 2,
            tangent_x=x_path.responses.get(2),
            tangent_y=y_path.responses.get(2),
        )
    }
    y_grown_by_size = {}
    min_gap = min(float(seed_min_gap), float(x_path.min_gap), float(y_path.min_gap))
    block_count = max(int(x_path.block_count), int(y_path.block_count), 1)

    for nsites in range(3, final_size):
        source_block = solver.chain.blocks[nsites - 1]
        grown = solver.chain.blocks[nsites].truncated.source
        block = solver.chain.blocks[nsites]
        source_x = x_path.sources[nsites - 1]
        source_y = y_path.sources[nsites - 1]
        source_xy = source_xy_by_size[nsites - 1]
        x_grown_h = x_path.grown_hamiltonians[nsites]
        y_grown_h = y_path.grown_hamiltonians[nsites]
        xy_grown_h = _grown_hamiltonian_bilinear(
            source_block,
            source_xy,
            h1e[:nsites, :nsites],
            zero_h[:nsites, :nsites],
            eri[:nsites, :nsites, :nsites, :nsites],
            zero_g[:nsites, :nsites, :nsites, :nsites],
            site_index=nsites - 1,
        )
        response = truncation_bilinear_tangent(
            grown,
            block.truncated,
            x_grown_h,
            y_grown_h,
            xy_grown_h,
            include_retained_mixing=True,
            tangent_x=x_path.responses.get(nsites),
            tangent_y=y_path.responses.get(nsites),
        )
        min_gap = min(float(min_gap), float(response.min_gap))
        tensors, xtensors, ytensors, xytensors = _pre_rotation_tensors_and_bilinears(
            grown,
            source_x,
            source_y,
            source_xy,
            h1e,
            xdh1e,
            ydh1e,
            zero_h,
            eri,
            xderi,
            yderi,
            zero_g,
            tuple(range(nsites, final_size)),
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=carry_rdm_operators,
            x_pre_rotation_parts=(
                x_path.pre_rotation_parts.get(nsites)
                if x_path.pre_rotation_parts is not None
                else None
            ),
            y_pre_rotation_parts=(
                y_path.pre_rotation_parts.get(nsites)
                if y_path.pre_rotation_parts is not None
                else None
            ),
        )
        xyops = _rotate_reduced_tensors_bilinear(
            block.truncated,
            tensors,
            xtensors,
            ytensors,
            xytensors,
            response,
        )
        source_xy_by_size[nsites] = _make_tangent_block(
            block,
            response.dxy_hamiltonian,
            xyops,
        )
        response_by_size[nsites] = response
        y_grown_by_size[nsites] = y_grown_h
        block_count = max(int(block_count), nsites - 1)

    bra = np.asarray(bra_vector, dtype=complex).reshape(-1)
    ket = np.asarray(ket_vector, dtype=complex).reshape(-1)
    terminal_adjoint = factor * np.outer(bra, np.conjugate(ket))
    target_nelec, target_j2 = solver.target_irrep
    target = Irrep((int(target_nelec), int(target_j2)))
    xh1_adj = np.zeros_like(h1e, dtype=float)
    xeri_adj = np.zeros_like(eri, dtype=float)
    xyh1_adj = np.zeros_like(h1e, dtype=float)
    xyeri_adj = np.zeros_like(eri, dtype=float)

    if final_size == 2:
        source_xy = source_xy_by_size[2]
        tensor_adj = IrrepTensor(
            source_xy.hamiltonian.bra,
            source_xy.hamiltonian.ket,
            source_xy.hamiltonian.op,
            {(target, target): terminal_adjoint},
        )
        source_x_h_adj = _zero_irrep_tensor_like(source_x.hamiltonian)
        source_x_op_adj = {}
        source_xy_h_adj = tensor_adj
        source_xy_op_adj = {}
    else:
        final_tensor = solver.chain.final.hamiltonian
        tensor_adj = IrrepTensor(
            final_tensor.bra,
            final_tensor.ket,
            final_tensor.op,
            {(target, target): terminal_adjoint},
        )
        source_block = solver.chain.blocks[final_size - 1]
        source_xy = source_xy_by_size[final_size - 1]
        (
            source_xy_h_adj,
            source_xy_op_adj,
            step_h_adj,
            step_g_adj,
        ) = _grown_hamiltonian_bilinear_adjoint(
            source_block,
            source_xy,
            tensor_adj,
            h1e[:final_size, :final_size],
            eri[:final_size, :final_size, :final_size, :final_size],
            site_index=final_size - 1,
        )
        xyh1_adj[:final_size, :final_size] += step_h_adj
        xyeri_adj[:final_size, :final_size, :final_size, :final_size] += step_g_adj
        source_x_h_adj = _zero_irrep_tensor_like(x_path.sources[final_size - 1].hamiltonian)
        source_x_op_adj = {}

        for nsites in range(final_size - 1, 2, -1):
            (
                source_x_h_adj,
                source_x_op_adj,
                source_xy_h_adj,
                source_xy_op_adj,
                step_xh,
                step_xg,
                step_xyh,
                step_xyg,
            ) = _recursive_growth_step_bilinear_adjoint_x(
                solver.chain.blocks[nsites - 1],
                solver.chain.blocks[nsites].truncated.source,
                solver.chain.blocks[nsites],
                x_path.sources[nsites - 1],
                y_path.sources[nsites - 1],
                source_xy_by_size[nsites - 1],
                source_x_h_adj,
                source_x_op_adj,
                source_xy_h_adj,
                source_xy_op_adj,
                h1e,
                eri,
                ydh1e,
                yderi,
                tuple(range(nsites, final_size)),
                response_by_size[nsites],
                y_grown_by_size[nsites],
                x_path.pre_rotation_parts[nsites],
                y_path.pre_rotation_parts[nsites],
                project_v1_packages=project_v1_packages,
                carry_rdm_operators=carry_rdm_operators,
            )
            xh1_adj += step_xh
            xeri_adj += step_xg
            xyh1_adj += step_xyh
            xyeri_adj += step_xyg

    step_xh, step_xg, step_xyh, step_xyg = _seed_two_site_bilinear_adjoint_x(
        solver.chain.blocks[2],
        x_path.sources[2],
        y_path.sources[2],
        source_xy_by_size[2],
        source_x_h_adj,
        source_x_op_adj,
        source_xy_h_adj,
        source_xy_op_adj,
        h1e,
        eri,
        ydh1e,
        yderi,
        response_by_size[2],
        final_size=final_size,
        project_v1_packages=project_v1_packages,
        include_retained_mixing=final_size != 2,
    )
    xh1_adj += step_xh
    xeri_adj += step_xg
    xyh1_adj += step_xyh
    xyeri_adj += step_xyg
    info = {
        "min_gap": float(min_gap),
        "block_count": int(block_count),
        "evaluation_count": 1,
        "x_path": x_path,
        "y_path": y_path,
    }
    return xh1_adj, xeri_adj, xyh1_adj, xyeri_adj, info


def recursive_active_integral_response_basis(
    solver,
    *,
    state_id: int = 0,
    include_paths: bool = True,
):
    """Build/cache recursive response blocks for symmetric active integrals.

    The basis spans the physical CAS perturbations used by orbital rotations:
    symmetric one-electron terms and two-electron terms obeying
    ``(pq|rs)=(qp|rs)=(pq|sr)=(rs|pq)``.  This gives an exact reusable
    representation for all orbital-pair recursive responses inside the current
    fixed NARG truncation pattern.
    """
    if solver.chain is None or solver.target_irrep is None:
        raise ValueError("recursive response is unavailable before SU2-NARG run().")
    h1e = np.asarray(solver.h1e)
    nsites = int(h1e.shape[0])
    h1_keys = _symmetric_h1_keys(nsites)
    eri_keys = _symmetric_eri_keys(nsites)
    flags = solver.timings or {}
    key = (
        id(solver.chain),
        id(solver.h1e),
        id(solver.eri),
        nsites,
        int(state_id),
        bool(include_paths),
        bool(flags.get("project_v1_packages", False)),
        bool(flags.get("carry_rdm_operators", False)),
    )
    cached = getattr(solver, "_su2_recursive_active_response_basis_cache", None)
    if cached is not None and cached.get("key") == key:
        return cached["basis"]

    from time import perf_counter

    ncomponents = len(h1_keys) + len(eri_keys)
    flags = solver.timings or {}
    workers = int(
        getattr(
            solver,
            "recursive_response_workers",
            flags.get("recursive_response_workers", 1),
        )
        or 1
    )
    workers = max(1, workers)
    build_start = perf_counter()

    def build_component(component):
        dh1, deri = _active_basis_component_perturbation(
            nsites,
            h1_keys,
            eri_keys,
            component,
        )
        if include_paths:
            path = recursive_tangent_path_for_active_integrals(
                solver,
                dh1,
                deri,
                state_id=state_id,
            )
            block = np.asarray(
                _terminal_block_from_recursive_tangent_path(solver, path),
                dtype=complex,
            )
            min_gap = float(path.min_gap)
            block_count = int(path.block_count)
        else:
            perturbation = recursive_perturbation_for_active_integrals(
                solver,
                dh1,
                deri,
                state_id=state_id,
            )
            path = None
            block = np.asarray(perturbation.block, dtype=complex)
            min_gap = float(perturbation.min_gap)
            block_count = int(perturbation.block_count)
        return (
            int(component),
            np.asarray(dh1, dtype=float),
            np.asarray(deri, dtype=float),
            path,
            block,
            min_gap,
            block_count,
        )

    if workers <= 1 or ncomponents <= 1:
        results = [build_component(component) for component in range(ncomponents)]
        worker_count = 1
    else:
        from concurrent.futures import ThreadPoolExecutor

        # Warm shared spectral/operator caches before concurrent component reads.
        first = build_component(0)
        with ThreadPoolExecutor(max_workers=min(workers, ncomponents)) as executor:
            results = [first]
            results.extend(executor.map(build_component, range(1, ncomponents)))
        results.sort(key=lambda item: item[0])
        worker_count = min(workers, ncomponents)

    blocks = []
    paths = []
    h1_components = []
    eri_components = []
    min_gap = np.inf
    block_count = 0
    for (
        _component,
        dh1,
        deri,
        path,
        block,
        component_min_gap,
        component_block_count,
    ) in results:
        h1_components.append(dh1)
        eri_components.append(deri)
        if path is not None:
            paths.append(path)
        blocks.append(block)
        min_gap = min(float(min_gap), float(component_min_gap))
        block_count = max(int(block_count), int(component_block_count))

    if blocks:
        block_stack = np.stack(blocks, axis=0)
    else:
        target_nelec, target_j2 = solver.target_irrep
        target = Irrep((int(target_nelec), int(target_j2)))
        dim = solver.chain.final.hamiltonian.block(target, target).shape[0]
        block_stack = np.zeros((0, dim, dim), dtype=complex)
    basis = RecursiveActiveResponseBasis(
        h1_keys=tuple(h1_keys),
        eri_keys=tuple(eri_keys),
        blocks=block_stack,
        min_gap=float(min_gap),
        block_count=int(block_count),
        paths=tuple(paths) if include_paths else None,
        h1_components=np.stack(h1_components, axis=0),
        eri_components=np.stack(eri_components, axis=0),
        build_seconds=float(perf_counter() - build_start),
        worker_count=int(worker_count),
    )
    solver._su2_recursive_active_response_basis_cache = {
        "key": key,
        "basis": basis,
    }
    return basis


def recursive_response_block_from_active_basis(
    solver,
    dh1e,
    deri,
    *,
    state_id: int = 0,
    basis: RecursiveActiveResponseBasis | None = None,
):
    """Assemble a recursive perturbation block from the active response basis."""
    if basis is None:
        basis = recursive_active_integral_response_basis(solver, state_id=state_id)
    h_coeff, eri_coeff = active_symmetric_integral_coefficients(dh1e, deri, basis)
    coeff = np.concatenate([h_coeff, eri_coeff])
    return np.tensordot(coeff, basis.blocks, axes=(0, 0))


def recursive_active_integral_adjoint_coefficients(
    solver,
    bra_vector,
    ket_vector,
    *,
    state_id: int = 0,
    basis: RecursiveActiveResponseBasis | None = None,
):
    r"""Return active-basis adjoint coefficients for ``bra^\dagger B ket``."""
    if basis is None:
        basis = recursive_active_integral_response_basis(solver, state_id=state_id)
    bra = np.asarray(bra_vector, dtype=complex).reshape(-1)
    ket = np.asarray(ket_vector, dtype=complex).reshape(-1)
    if basis.blocks.shape[1:] != (bra.size, ket.size):
        raise ValueError("response basis block shape is inconsistent with vectors")
    values = np.einsum("i,kij,j->k", np.conjugate(bra), basis.blocks, ket, optimize=True)
    nh = len(basis.h1_keys)
    return values[:nh], values[nh:], basis


def recursive_bilinear_active_integral_adjoint_coefficients(
    solver,
    ydh1e,
    yderi,
    bra_vector,
    ket_vector,
    *,
    state_id: int = 0,
    basis: RecursiveActiveResponseBasis | None = None,
    workers: int | None = None,
    y_path: RecursiveTangentPath | None = None,
    xy_values=None,
):
    r"""Return adjoint coefficients for fixed-pattern mixed responses.

    For fixed ``y`` this builds the scalar adjoints

    $$
    b^x_k = \langle l|B_{xy}(e_k, y, 0)|r\rangle,\qquad
    b^{xy}_k = \langle l|B_{xy}(0, y, e_k)|r\rangle,
    $$

    where ``e_k`` runs over the symmetric active-integral basis.  Any orbital
    pair ``i`` can then be contracted as

    $$
    \sum_k c^x_{ik} b^x_k + \sum_k c^{xy}_{ik} b^{xy}_k.
    $$
    """
    ydh1e = np.asarray(ydh1e)
    yderi = np.asarray(yderi)
    bra = np.asarray(bra_vector, dtype=complex).reshape(-1)
    ket = np.asarray(ket_vector, dtype=complex).reshape(-1)

    nsites = int(ydh1e.shape[0])
    if basis is None:
        h1_keys = _symmetric_h1_keys(nsites)
        eri_keys = _symmetric_eri_keys(nsites)
        h1_components = None
        eri_components = None
        x_paths = None
        block_count = 0
        min_gap = np.inf
        if xy_values is None:
            h_xy_adj, g_xy_adj, _ = recursive_active_integral_adjoint_arrays(
                solver,
                bra,
                ket,
                state_id=state_id,
            )
            xy_values = []
            for component in range(len(h1_keys) + len(eri_keys)):
                dh1_component, deri_component = _active_basis_component_perturbation(
                    nsites,
                    h1_keys,
                    eri_keys,
                    component,
                )
                value = np.vdot(h_xy_adj, dh1_component)
                value += np.vdot(g_xy_adj, deri_component)
                xy_values.append(value)
            xy_values = np.asarray(xy_values, dtype=complex)
    else:
        if basis.blocks.shape[1:] != (bra.size, ket.size):
            raise ValueError("response basis block shape is inconsistent with vectors")
        h1_keys = tuple(basis.h1_keys)
        eri_keys = tuple(basis.eri_keys)
        h1_components = getattr(basis, "h1_components", None)
        eri_components = getattr(basis, "eri_components", None)
        x_paths = basis.paths
        block_count = int(basis.block_count)
        min_gap = min(np.inf, float(basis.min_gap))
        if xy_values is None:
            xy_values = np.einsum(
                "i,kij,j->k",
                np.conjugate(bra),
                basis.blocks,
                ket,
                optimize=True,
            )
    zero_h = np.zeros_like(ydh1e)
    zero_g = np.zeros_like(yderi)
    ncomponents = len(h1_keys) + len(eri_keys)
    x_values = np.zeros(ncomponents, dtype=complex)
    evaluation_count = 0
    if y_path is None:
        y_path = recursive_tangent_path_for_active_integrals(
            solver,
            ydh1e,
            yderi,
            state_id=state_id,
        )
    min_gap = min(float(min_gap), float(y_path.min_gap))

    if workers is None:
        workers = int(
            getattr(
                solver,
                "recursive_response_workers",
                (solver.timings or {}).get("recursive_response_workers", 1),
            )
            or 1
        )
    workers = max(1, int(workers))

    def evaluate_component(component):
        if h1_components is not None and eri_components is not None:
            dh1_component = h1_components[component]
            deri_component = eri_components[component]
        else:
            dh1_component, deri_component = _active_basis_component_perturbation(
                nsites,
                h1_keys,
                eri_keys,
                component,
            )
        x_path = (
            x_paths[component]
            if x_paths is not None and component < len(x_paths)
            else None
        )
        x_perturbation = recursive_bilinear_perturbation_for_active_integrals(
            solver,
            dh1_component,
            deri_component,
            ydh1e,
            yderi,
            zero_h,
            zero_g,
            state_id=state_id,
            x_path=x_path,
            y_path=y_path,
        )
        value = np.vdot(bra, x_perturbation.block @ ket)
        return (
            int(component),
            value,
            float(x_perturbation.min_gap),
            int(x_perturbation.block_count),
        )

    if workers <= 1 or ncomponents <= 1:
        results = [evaluate_component(component) for component in range(ncomponents)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        # Warm shared caches on the current thread before concurrent reads.
        first = evaluate_component(0)
        with ThreadPoolExecutor(max_workers=min(workers, ncomponents)) as executor:
            results = [first]
            results.extend(executor.map(evaluate_component, range(1, ncomponents)))

    for component, value, component_min_gap, component_block_count in results:
        x_values[component] = value
        min_gap = min(float(min_gap), float(component_min_gap))
        block_count = max(int(block_count), int(component_block_count))
        evaluation_count += 1

    return RecursiveBilinearActiveAdjoint(
        h1_keys=tuple(h1_keys),
        eri_keys=tuple(eri_keys),
        x_values=x_values,
        xy_values=np.asarray(xy_values, dtype=complex),
        min_gap=float(min_gap),
        block_count=int(block_count),
        evaluation_count=int(evaluation_count),
        worker_count=int(min(workers, max(1, ncomponents))),
    )


def active_symmetric_pair_response_matrix(
    h1_mo,
    eri_mo,
    pairs,
    *,
    ncore: int,
    ncas: int,
    basis: RecursiveActiveResponseBasis,
):
    """Return pair-to-active-basis coefficients for orbital rotations."""
    pairs = list(pairs)
    ncomponents = int(basis.blocks.shape[0])
    dtype = np.result_type(h1_mo, eri_mo, complex)
    matrix = np.zeros((len(pairs), ncomponents), dtype=dtype)
    for row, pair in enumerate(pairs):
        dh1_i, deri_i = cas_integral_response_from_pair(
            h1_mo,
            eri_mo,
            pair,
            ncore=ncore,
            ncas=ncas,
        )
        h_coeff, eri_coeff = active_symmetric_integral_coefficients(
            dh1_i,
            deri_i,
            basis,
        )
        matrix[row, : len(h_coeff)] = h_coeff
        matrix[row, len(h_coeff) :] = eri_coeff
    return matrix


def recursive_response_pair_components_from_active_basis(
    solver,
    h1_mo,
    eri_mo,
    pairs,
    bra_vector,
    ket_vector,
    *,
    ncore: int,
    ncas: int,
    state_id: int = 0,
    basis: RecursiveActiveResponseBasis | None = None,
    pair_coefficients=None,
    factor: float = 2.0,
):
    """Project an adjoint recursive response onto packed orbital-pair variables."""
    h_adj, eri_adj, basis = recursive_active_integral_adjoint_coefficients(
        solver,
        bra_vector,
        ket_vector,
        state_id=state_id,
        basis=basis,
    )
    adjoint = np.concatenate([h_adj, eri_adj])
    if pair_coefficients is None:
        pair_coefficients = active_symmetric_pair_response_matrix(
            h1_mo,
            eri_mo,
            pairs,
            ncore=ncore,
            ncas=ncas,
            basis=basis,
        )
    return np.asarray(factor * np.real(pair_coefficients @ adjoint), dtype=float)


def _normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1.0e-14:
        raise ValueError("terminal response requires a nonzero vector")
    return vector / norm


def solve_terminal_response(
    hamiltonian,
    vector,
    perturbation,
    *,
    energy: float | None = None,
    root_index: int | None = None,
    gap_tol: float = 1.0e-10,
    spectrum=None,
) -> TerminalResponse:
    r"""Solve the projected tangent equation for one retained-sector eigenstate.

    The response vector ``z`` satisfies

    $$
    Q (H - E) Q z = -Q V |\psi\rangle,\qquad
    \langle \psi | z\rangle = 0,
    $$

    where ``V`` is the perturbation block and ``Q = 1 - |\psi\rangle\langle\psi|``.
    """
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    perturbation = np.asarray(perturbation, dtype=complex)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    if perturbation.shape != hamiltonian.shape:
        raise ValueError("perturbation must have the same shape as hamiltonian")

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    perturbation = 0.5 * (perturbation + perturbation.conj().T)
    psi = _normalize(vector)
    if psi.size != hamiltonian.shape[0]:
        raise ValueError("vector dimension is inconsistent with hamiltonian")

    if spectrum is None:
        evals, evecs = np.linalg.eigh(hamiltonian)
    else:
        evals, evecs = spectrum
        evals = np.asarray(evals, dtype=float)
        evecs = np.asarray(evecs, dtype=complex)
        if evals.ndim != 1 or evecs.shape != hamiltonian.shape:
            raise ValueError("terminal-response spectrum is inconsistent with hamiltonian")
    overlaps = evecs.conj().T @ psi
    if root_index is None:
        root_index = int(np.argmax(np.abs(overlaps)))
    else:
        root_index = int(root_index)
    if root_index < 0 or root_index >= evals.size:
        raise IndexError("root_index is outside the retained-sector spectrum")

    phase_overlap = overlaps[root_index]
    if abs(phase_overlap) > 1.0e-14:
        psi = evecs[:, root_index] * phase_overlap / abs(phase_overlap)
    eig_energy = float(evals[root_index])
    if energy is None:
        energy = eig_energy
    else:
        energy = float(energy)

    source = perturbation @ psi
    first_order = np.vdot(psi, source)
    rhs = source - first_order * psi
    rhs_coeff = evecs.conj().T @ rhs

    z = np.zeros_like(psi, dtype=complex)
    gaps = []
    for idx, value in enumerate(evals):
        if idx == root_index:
            continue
        gap = float(value - energy)
        gaps.append(abs(gap))
        coeff = rhs_coeff[idx]
        if abs(gap) <= gap_tol:
            if abs(coeff) > 10.0 * gap_tol:
                raise np.linalg.LinAlgError(
                    "terminal response is singular in a near-degenerate sector"
                )
            continue
        z -= evecs[:, idx] * coeff / gap

    z -= psi * np.vdot(psi, z)
    residual = (hamiltonian - energy * np.eye(hamiltonian.shape[0])) @ z + rhs
    residual -= psi * np.vdot(psi, residual)
    min_gap = min(gaps) if gaps else np.inf
    return TerminalResponse(
        vector=z,
        energy=energy,
        first_order_energy=float(np.real(first_order)),
        root_index=root_index,
        residual_norm=float(np.linalg.norm(residual)),
        min_gap=float(min_gap),
    )


def density_operator_blocks(final, *, vector, nelec: int, j2: int, site_count: int):
    """Return final-sector matrices for all spin-free density operators ``E[p,q]``."""
    builder = build_su2_rdms(
        final,
        vector,
        nelec=int(nelec),
        j2=int(j2),
        site_count=int(site_count),
    )
    n = int(site_count)
    return [[builder.density_component_block(p, q) for q in range(n)] for p in range(n)]


def hamiltonian_block_from_density(density_blocks, h1e, eri):
    r"""Assemble a scalar Hamiltonian block from final-sector density matrices.

    This uses

    $$
    H = \sum_{pq} h_{pq} E_{pq}
      + {1\over 2}\sum_{pqrs} v_{pqrs}
        \left(E_{pq}E_{rs} - \delta_{qr}E_{ps}\right).
    $$

    For a complete retained sector this reproduces the exact active Hamiltonian;
    for a truncated sector it is the Hamiltonian induced by the final projected
    density algebra.
    """
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    n = int(h1e.shape[0])
    if h1e.shape != (n, n) or eri.shape != (n, n, n, n):
        raise ValueError("h1e/eri shapes are inconsistent")
    if len(density_blocks) != n or any(len(row) != n for row in density_blocks):
        raise ValueError("density_blocks are inconsistent with h1e")
    dim = np.asarray(density_blocks[0][0]).shape[0]
    out = np.zeros((dim, dim), dtype=complex)
    for p in range(n):
        for q in range(n):
            out += h1e[p, q] * density_blocks[p][q]
    for p in range(n):
        for q in range(n):
            epq = density_blocks[p][q]
            for r in range(n):
                for s in range(n):
                    op = epq @ density_blocks[r][s]
                    if q == r:
                        op = op - density_blocks[p][s]
                    out += 0.5 * eri[p, q, r, s] * op
    return 0.5 * (out + out.conj().T)


def active_perturbation_block_from_density(solver, kappa, *, state_id: int = 0):
    """Build an active orbital-rotation perturbation in the final density algebra."""
    if solver.chain is None or solver.root_vectors is None or solver.target_irrep is None:
        raise ValueError("SU2-NARG terminal response is unavailable before run().")
    state_id = int(state_id)
    vector = np.asarray(solver.root_vectors[:, state_id], dtype=complex)
    nelec, j2 = solver.target_irrep
    site_count = int(solver.ncas or np.asarray(solver.h1e).shape[0])
    density = density_operator_blocks(
        solver.chain.final,
        vector=vector,
        nelec=int(nelec),
        j2=int(j2),
        site_count=site_count,
    )
    dh1, deri = active_integral_response(solver.h1e, solver.eri, kappa)
    return hamiltonian_block_from_density(density, dh1, deri)


def terminal_response_for_active_kappa(solver, kappa, *, state_id: int = 0):
    """Return the analytic terminal eigenvector response to an active rotation."""
    if solver.block is None or solver.root_vectors is None:
        raise ValueError("SU2-NARG terminal response is unavailable before run().")
    state_id = int(state_id)
    perturbation = active_perturbation_block_from_density(
        solver,
        kappa,
        state_id=state_id,
    )
    return solve_terminal_response(
        solver.block,
        solver.root_vectors[:, state_id],
        perturbation,
    )


def active_pair_kappa(ncas: int, p: int, q: int, value: float = 1.0):
    """Return an active-space anti-Hermitian generator for one pair."""
    kappa = np.zeros((int(ncas), int(ncas)), dtype=float)
    p = int(p)
    q = int(q)
    if p == q:
        return kappa
    kappa[p, q] = float(value)
    kappa[q, p] = -float(value)
    return kappa
