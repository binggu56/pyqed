"""Shared symbolic spatial-orbital term helpers for qchem DMRG builders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSet
from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators


_SPATIAL_LOCAL_OPS = None


def spatial_local_ops():
    """Shared local operator table for the spatial d=4 site basis."""
    global _SPATIAL_LOCAL_OPS
    if _SPATIAL_LOCAL_OPS is None:
        ops = {name: np.asarray(op, dtype=complex) for name, op in SpinHalfFermionOperators().items()}
        _SPATIAL_LOCAL_OPS = {
            "I": np.eye(4, dtype=complex),
            "JW": ops["JW"],
            "cu": ops["Cu"],
            "cdu": ops["Cdu"],
            "cd": ops["Cd"],
            "cdd": ops["Cdd"],
            "nu": ops["Nu"],
            "nd": ops["Nd"],
            "n": ops["Ntot"],
        }
    return _SPATIAL_LOCAL_OPS


def canonical_spatial_local_symbol(symbols, *, tol=1.0e-14):
    """Collapse an ordered same-site operator product into one local symbol."""
    aliases = spatial_local_ops()
    mat = np.eye(4, dtype=complex)
    for symbol in symbols:
        mat = mat @ aliases[symbol]

    if np.linalg.norm(mat) <= tol:
        return None, 0.0

    for name, ref in aliases.items():
        if np.allclose(mat, ref, atol=tol, rtol=0.0):
            return name, 1.0
        if np.allclose(mat, -ref, atol=tol, rtol=0.0):
            return name, -1.0

    name = f"spop{len(aliases)}"
    aliases[name] = mat
    return name, 1.0


class BasisSpatialFermion(BasisSet):
    """One spatial orbital with local states empty, up, down, double."""

    is_electron = True

    def __init__(self, dof):
        super().__init__(dof, 4, [(0, 0), (1, 1), (1, -1), (2, 0)])
        self._op_aliases = spatial_local_ops()

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        mat = np.eye(4, dtype=complex)
        for symbol in op.split_symbol:
            try:
                mat = mat @ self._op_aliases[symbol]
            except KeyError as exc:
                raise ValueError(f"op_symbol:{symbol} is not supported for BasisSpatialFermion") from exc
        return mat * op.factor

    def copy(self, new_dof):
        return BasisSpatialFermion(new_dof)


def spatial_jw_term_spec(local_symbols, sites, factor):
    """
    Convert global spatial fermion products into site-local symbolic terms.

    Each global fermion operator contributes a site-level parity string on all
    earlier spatial orbitals and a local d=4 creation/annihilation operator on
    its own orbital.
    """
    grouped = defaultdict(list)
    for symbol, site in zip(local_symbols, sites):
        site = int(site)
        for k in range(site):
            grouped[k].append("JW")
        grouped[site].append(symbol)

    final_symbols = []
    final_sites = []
    final_factor = factor
    for site in sorted(grouped):
        symbol, local_factor = canonical_spatial_local_symbol(grouped[site])
        if symbol is None:
            return "", [], 0.0
        if symbol == "I":
            continue
        final_symbols.append(symbol)
        final_sites.append(site)
        final_factor *= local_factor
    return " ".join(final_symbols), final_sites, final_factor


def accumulate_symbolic_term(term_map, symbol, dofs, factor, tol=1.0e-14):
    """Accumulate a symbolic operator term without instantiating ``Op`` eagerly."""
    if abs(factor) <= tol or not symbol:
        return
    key = (str(symbol), tuple(int(site) for site in dofs))
    term_map[key] = term_map.get(key, 0.0) + complex(factor)
    if abs(term_map[key]) <= tol:
        term_map.pop(key, None)


def accumulate_spatial_jw_term(term_map, local_symbols, sites, factor, tol=1.0e-14):
    symbol, dofs, term_factor = spatial_jw_term_spec(local_symbols, sites, factor)
    accumulate_symbolic_term(term_map, symbol, dofs, term_factor, tol=tol)


def spatial_one_body_term_map(h_spatial, *, cutoff=1.0e-14):
    """Build symbolic spatial-site terms for a spin-summed one-electron Hamiltonian."""
    h_spatial = np.asarray(h_spatial)
    if h_spatial.ndim != 2 or h_spatial.shape[0] != h_spatial.shape[1]:
        raise ValueError("h_spatial must be a square matrix.")
    term_map = {}
    for p, q in np.argwhere(np.abs(h_spatial) > cutoff):
        val = h_spatial[p, q]
        accumulate_spatial_jw_term(term_map, ["cdu", "cu"], [p, q], val, tol=cutoff)
        accumulate_spatial_jw_term(term_map, ["cdd", "cd"], [p, q], val, tol=cutoff)
    return term_map


def spatial_two_body_term_map(eri_spatial, *, cutoff=1.0e-14, include_half=True):
    """
    Build symbolic spatial-site terms for restricted two-electron integrals.

    ``eri_spatial`` is interpreted as ``(pq|rs)``.  By default the conventional
    second-quantized prefactor ``1/2`` is applied internally to match the qchem
    DMRG builders.
    """
    eri_spatial = np.asarray(eri_spatial)
    if eri_spatial.ndim != 4:
        raise ValueError("eri_spatial must have shape (n, n, n, n).")
    values = 0.5 * eri_spatial if include_half else eri_spatial
    term_map = {}
    for p, q, r, s in np.argwhere(np.abs(values) > cutoff):
        val = values[p, q, r, s]
        if p != r and s != q:
            accumulate_spatial_jw_term(
                term_map,
                ["cdu", "cdu", "cu", "cu"],
                [p, r, s, q],
                val,
                tol=cutoff,
            )
            accumulate_spatial_jw_term(
                term_map,
                ["cdd", "cdd", "cd", "cd"],
                [p, r, s, q],
                val,
                tol=cutoff,
            )
        accumulate_spatial_jw_term(
            term_map,
            ["cdu", "cdd", "cd", "cu"],
            [p, r, s, q],
            val,
            tol=cutoff,
        )
        accumulate_spatial_jw_term(
            term_map,
            ["cdd", "cdu", "cu", "cd"],
            [p, r, s, q],
            val,
            tol=cutoff,
        )
    return term_map


def spatial_two_body_spinfree_term_map(eri_spatial, *, cutoff=1.0e-14, include_half=True):
    """
    Build spin-free scalar two-electron terms in the spatial-site basis.

    The restricted two-electron operator is represented with spin-summed
    generators

        E_pq = sum_sigma c^dagger[p,sigma] c[q,sigma]

    using the standard identity

        c^dagger_p c^dagger_r c_s c_q = E_pq E_rs - delta_qr E_ps.

    This exposes the spin-scalar structure before the local Jordan-Wigner
    canonicalization collapses repeated spatial sites.
    """
    eri_spatial = np.asarray(eri_spatial)
    if eri_spatial.ndim != 4:
        raise ValueError("eri_spatial must have shape (n, n, n, n).")
    values = 0.5 * eri_spatial if include_half else eri_spatial
    term_map = {}
    spin_terms = (("cdu", "cu"), ("cdd", "cd"))
    for p, q, r, s in np.argwhere(np.abs(values) > cutoff):
        val = values[p, q, r, s]
        for left_create, left_destroy in spin_terms:
            for right_create, right_destroy in spin_terms:
                accumulate_spatial_jw_term(
                    term_map,
                    [left_create, left_destroy, right_create, right_destroy],
                    [p, q, r, s],
                    val,
                    tol=cutoff,
                )
        if q == r:
            for create, destroy in spin_terms:
                accumulate_spatial_jw_term(
                    term_map,
                    [create, destroy],
                    [p, s],
                    -val,
                    tol=cutoff,
                )
    return term_map


def merge_term_maps(*term_maps, cutoff=1.0e-14):
    """Merge symbolic term maps using the same cancellation rule as accumulation."""
    merged = {}
    for term_map in term_maps:
        for (symbol, dofs), factor in term_map.items():
            accumulate_symbolic_term(merged, symbol, dofs, factor, tol=cutoff)
    return merged
