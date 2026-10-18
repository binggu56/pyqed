"""Malmqvist single-orbital counter-transformations of complete CAS CI tensors."""

from functools import lru_cache
from math import comb

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.linalg import lu, solve_triangular

from pyqed.parallel import resolve_workers
from pyqed.linalg.biorthogonal import orbital_factors


class UnsupportedMalmqvistError(ValueError):
    """A valid CI representation outside the complete-CAS transformation domain."""


@lru_cache(maxsize=8)
def _string_plan(shape, data):
    from .direct_ci import _single_string_links

    strings = np.frombuffer(data, dtype=np.int8).reshape(shape)
    counts = strings.sum(axis=1)
    if (not len(strings) or np.any((strings != 0) & (strings != 1))
            or np.any(counts != counts[0]) or len(strings) != comb(shape[1], int(counts[0]))
            or len(np.unique(strings, axis=0)) != len(strings)):
        raise UnsupportedMalmqvistError('Counter-transformation requires complete fixed-electron spin-string spaces.')
    target, source, p, q, phase = _single_string_links(strings)
    # Direct-CI connectivity uses the opposite off-diagonal operator ordering;
    # orbital replacements require the physical a_p^dagger a_q phase.
    phase = -phase
    order = np.argsort(q, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(q, minlength=shape[1]))]
    occupied = (np.concatenate([np.flatnonzero(strings[:, j]) for j in range(shape[1])])
                if shape[1] else np.empty(0, dtype=np.intp))
    occupied_offsets = np.r_[0, np.cumsum(strings.sum(axis=0))]
    return (target[order], source[order], p[order], phase[order], offsets,
            occupied, occupied_offsets)


@njit(parallel=True, nogil=True, cache=True)
def _replace_orbitals(coefficients, t, target, source, p, phase, offsets, occupied, occupied_offsets):
    # Each task owns spectator columns for the full orbital sequence. Source
    # strings contain j and target strings do not: updates cannot feed back.
    width = 64
    for block in prange((coefficients.shape[1]+width-1)//width):
        start = block*width
        stop = min(start+width, coefficients.shape[1])
        for j in range(t.shape[1]):
            for link in range(offsets[j], offsets[j+1]):
                value = t[p[link], j]*phase[link]
                if value != 0:
                    dst, src = target[link], source[link]
                    for column in range(start, stop):
                        coefficients[dst, column] += value*coefficients[src, column]
            if t[j, j] != 1:
                for link in range(occupied_offsets[j], occupied_offsets[j+1]):
                    row = occupied[link]
                    for column in range(start, stop):
                        coefficients[row, column] *= t[j, j]


def _permuted_strings(strings, permutation):
    """Input-to-output string map and fermionic phase for an orbital permutation."""
    lookup = {tuple(row): i for i, row in enumerate(strings)}
    target, phases = [], []
    for row in strings:
        orbitals = permutation[np.flatnonzero(row)]
        result = np.zeros_like(row)
        result[orbitals] = 1
        inversions = np.count_nonzero(np.triu(orbitals[:, None] > orbitals[None, :], 1))
        target.append(lookup[tuple(result)])
        phases.append((-1)**inversions)
    return np.asarray(target), np.asarray(phases)


def counter_transform_ci(ci, orbital_transform, alpha_strings, beta_strings, *, workers=None):
    """Counter-transform complete CAS CI tensors after an orbital basis change.

    If new orbitals are ``old @ orbital_transform``, apply the induced action
    of its inverse to ``ci[roots, alpha, beta]``. Pivoted LU gives ``X=P L U``;
    first apply the exact signed orbital permutation, then single-orbital
    replacements with ``t_U=U^{-1}`` and ``t_L=I-L`` (Malmqvist's equations).
    Each spin action is linear in the replacement column. Applying both spins
    retains the mixed double replacement exactly; no exponential truncation
    or determinant-space matrix/inverse is constructed.

    Adaptation of P.-A. Malmqvist, Int. J. Quantum Chem. 30, 479-494 (1986),
    doi:10.1002/qua.560300404; see also Knecht et al., JCTC 12, 5881-5894
    (2016), doi:10.1021/acs.jctc.6b00889, Eqs. (35)-(46). The pivot permutation
    and blocked parallel spectator updates are implementation choices, not
    a reproduction of Molcas. Restricted spatial orbitals and complete fixed
    alpha/beta electron spaces only; selected CI/RAS and transition densities
    are not implemented. Maps must be numerically nonsingular. Every nonzero
    coefficient is retained; accuracy is limited by floating-point roundoff
    and conditioning. Inputs are not modified or renormalized.
    """
    alpha, beta = (np.ascontiguousarray(s, dtype=np.int8) for s in (alpha_strings, beta_strings))
    x = np.asarray(orbital_transform)
    ci = np.asarray(ci)
    if (x.ndim != 2 or x.shape[0] != x.shape[1] or alpha.ndim != 2 or beta.ndim != 2
            or alpha.shape[1] != len(x) or beta.shape[1] != len(x)
            or ci.ndim != 3 or ci.shape[1:] != (len(alpha), len(beta))):
        raise ValueError('Orbital maps, spin strings and root-by-alpha-by-beta CI dimensions disagree.')
    plans = [_string_plan(s.shape, s.tobytes()) for s in (alpha, beta)]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(ci)):
        raise ValueError('Orbital maps and CI coefficients must be finite.')
    if not len(x):
        return np.array(ci, dtype=np.result_type(x, ci, float), copy=True)
    singular = np.linalg.svd(x, compute_uv=False)
    if len(x) and singular[-1] <= np.finfo(float).eps*len(x)*singular[0]:
        raise np.linalg.LinAlgError('A singular orbital map cannot be counter-transformed.')
    dtype = np.result_type(x, ci, float)
    perm, lower, upper = lu(np.asarray(x, dtype=dtype))
    t = solve_triangular(upper, np.eye(len(x), dtype=dtype)) + np.eye(len(x))-lower
    permutation = np.argmax(perm, axis=1)
    out = np.array(ci, dtype=dtype, copy=True)
    previous_threads = get_num_threads()
    try:
        set_num_threads(resolve_workers(workers))
        for axis, (strings, plan) in enumerate(zip((alpha, beta), plans), start=1):
            layout = np.moveaxis(out, axis, 0)
            rows = np.ascontiguousarray(layout).reshape(len(strings), -1)
            if not np.array_equal(permutation, np.arange(len(x))):
                target, phases = _permuted_strings(strings, permutation)
                permuted = np.empty_like(rows)
                permuted[target] = phases[:, None]*rows
                rows = permuted
            _replace_orbitals(rows, t, *plan)
            out = np.moveaxis(rows.reshape(layout.shape), 0, axis)
    finally:
        set_num_threads(previous_threads)
    return np.ascontiguousarray(out)


def malmqvist_overlap(bra, ket, s=None, *, workers=None, return_info=False):
    """Exact CASCI overlap using Malmqvist single-orbital counter-transformations.

    Frozen cores use the existing exact Schur complement. The active metric
    uses the existing balanced pivoted LU orbital factors ``A^dagger B=s``;
    both CI tensors are counter-transformed into biorthonormal orbital bases.
    Adaptation of P.-A. Malmqvist, Int. J. Quantum Chem. 30, 479-494 (1986),
    doi:10.1002/qua.560300404, and Knecht et al., JCTC 12, 5881-5894 (2016),
    doi:10.1021/acs.jctc.6b00889. This changes the orbital factorization, not
    the single-orbital replacement algebra; it does not reproduce Molcas.
    See :func:`counter_transform_ci` for transformation details and fidelity.
    Only complete product CAS bases with matching active orbital and spin
    electron counts are supported. Singular active metrics raise explicitly;
    use the exact direct overlap path for those cases. No silent fallback,
    screening, coefficient truncation or state renormalization is used.
    """
    import importlib
    c = importlib.import_module('pyqed.qchem.mcscf.casci')

    def tensors(frame):
        a, b = c._basis_spin_strings(frame.binary)
        coefficients = c._as_state_ci_matrix(frame.ci, frame.binary.shape[0])
        if len(a)*len(b) != coefficients.shape[1]:
            raise UnsupportedMalmqvistError('Malmqvist overlap requires a complete product CAS basis.')
        if not isinstance(frame.binary, c.FCIStringBasis):
            grid = np.asarray(frame.binary).reshape(len(a), len(b), 2, frame.ncas)
            if not (np.all(grid[:, :, 0] == a[:, None, :]) and np.all(grid[:, :, 1] == b[None, :, :])):
                raise UnsupportedMalmqvistError('Malmqvist overlap requires product-ordered determinants.')
        return a, b, coefficients.reshape(len(coefficients), len(a), len(b))

    if bra.ncas != ket.ncas:
        raise UnsupportedMalmqvistError('Malmqvist overlap requires matching active orbital spaces.')
    a, b, left = tensors(bra)
    u, v, right = tensors(ket)
    if (a.shape != u.shape or b.shape != v.shape
            or not np.array_equal(a.sum(axis=1), u.sum(axis=1))
            or not np.array_equal(b.sum(axis=1), v.sum(axis=1))):
        raise UnsupportedMalmqvistError('Malmqvist overlap requires matching alpha/beta electron counts.')
    s = c.mo_overlap(bra, ket, s=s)
    dtype = np.result_type(s, left, right)
    core, active = c._effective_active_overlap(s, bra.ncore, ket.ncore, bra.ncas, ket.ncas, dtype)
    if bra.ncas:
        xl, xr, _, _ = orbital_factors(active)
    else:
        xl = xr = active.copy()
    left = counter_transform_ci(left, xl, a, b, workers=workers)
    right = counter_transform_ci(right, xr, u, v, workers=workers)
    if not np.array_equal(a, u) or not np.array_equal(b, v):
        def order(target, source):
            lookup = {tuple(row): i for i, row in enumerate(source)}
            return np.array([lookup[tuple(row)] for row in target])
        right = right[:, order(a, u)][:, :, order(b, v)]
    overlap = core*(left.conj().reshape(len(left), -1) @ right.reshape(len(right), -1).T)
    if return_info:
        return overlap, dict(method='malmqvist', workers=resolve_workers(workers),
                             biorthogonality_residual=float(np.linalg.norm(xl.conj().T @ active @ xr-np.eye(bra.ncas))),
                             active_condition=float(np.linalg.cond(active)) if bra.ncas else 1.)
    return overlap
