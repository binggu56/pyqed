"""Shared symbolic spatial-orbital term helpers for qchem DMRG builders."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSet
from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators


_SPATIAL_LOCAL_OPS = None
_CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE = {}
_SPATIAL_JW_TERM_SPEC_CACHE = {}
_SPATIAL_JW_PATTERN_SPEC_CACHE = {}


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
    key = (tuple(str(symbol) for symbol in symbols), float(tol))
    cached = _CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE.get(key)
    if cached is not None:
        return cached
    aliases = spatial_local_ops()
    mat = np.eye(4, dtype=complex)
    for symbol in symbols:
        mat = mat @ aliases[symbol]

    if np.linalg.norm(mat) <= tol:
        result = (None, 0.0)
        _CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE[key] = result
        return result

    for name, ref in aliases.items():
        if np.allclose(mat, ref, atol=tol, rtol=0.0):
            result = (name, 1.0)
            _CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE[key] = result
            return result
        if np.allclose(mat, -ref, atol=tol, rtol=0.0):
            result = (name, -1.0)
            _CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE[key] = result
            return result

    name = f"spop{len(aliases)}"
    aliases[name] = mat
    result = (name, 1.0)
    _CANONICAL_SPATIAL_LOCAL_SYMBOL_CACHE[key] = result
    return result


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
    key = (
        tuple(str(symbol) for symbol in local_symbols),
        tuple(int(site) for site in sites),
    )
    cached = _SPATIAL_JW_TERM_SPEC_CACHE.get(key)
    if cached is not None:
        symbol, final_sites, local_factor = cached
        return symbol, list(final_sites), factor * local_factor
    grouped = defaultdict(list)
    for symbol, site in zip(key[0], key[1]):
        for k in range(site):
            grouped[k].append("JW")
        grouped[site].append(symbol)

    final_symbols = []
    final_sites = []
    local_factor = 1.0
    for site in sorted(grouped):
        symbol, site_factor = canonical_spatial_local_symbol(grouped[site])
        if symbol is None:
            _SPATIAL_JW_TERM_SPEC_CACHE[key] = ("", (), 0.0)
            return "", [], 0.0
        if symbol == "I":
            continue
        final_symbols.append(symbol)
        final_sites.append(site)
        local_factor *= site_factor
    result = (" ".join(final_symbols), tuple(final_sites), local_factor)
    _SPATIAL_JW_TERM_SPEC_CACHE[key] = result
    return result[0], list(result[1]), factor * result[2]


def spatial_jw_pattern_spec(local_symbols, sites, n_sites):
    """Return the full site pattern without constructing symbolic dof lists."""

    key = (
        tuple(str(symbol) for symbol in local_symbols),
        tuple(int(site) for site in sites),
        int(n_sites),
    )
    cached = _SPATIAL_JW_PATTERN_SPEC_CACHE.get(key)
    if cached is not None:
        return cached
    pattern = ["I"] * int(n_sites)
    factor = 1.0
    for site in range(int(n_sites)):
        site_symbols = []
        for symbol, op_site in zip(key[0], key[1]):
            if site < op_site:
                site_symbols.append("JW")
            elif site == op_site:
                site_symbols.append(symbol)
        if not site_symbols:
            continue
        local_symbol, local_factor = canonical_spatial_local_symbol(site_symbols)
        if local_symbol is None:
            result = ((), 0.0)
            _SPATIAL_JW_PATTERN_SPEC_CACHE[key] = result
            return result
        pattern[site] = local_symbol
        factor *= local_factor
    result = (tuple(pattern), factor)
    _SPATIAL_JW_PATTERN_SPEC_CACHE[key] = result
    return result


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


def spatial_generator_family_term_map(entries, *, cutoff=1.0e-14):
    """
    Build a spin-summed one-generator family term map.

    ``entries`` maps ``(p, q)`` to the coefficient multiplying
    ``E_pq = sum_sigma c^dagger[p,sigma] c[q,sigma]``.
    """
    term_map = {}
    spin_terms = (("cdu", "cu"), ("cdd", "cd"))
    for (p, q), value in dict(entries or {}).items():
        p = int(p)
        q = int(q)
        value = complex(value)
        if abs(value) <= cutoff:
            continue
        for create, destroy in spin_terms:
            accumulate_spatial_jw_term(
                term_map,
                [create, destroy],
                [p, q],
                value,
                tol=cutoff,
            )
    return term_map


def spatial_two_generator_family_term_map(entries, *, cutoff=1.0e-14):
    """
    Build a spin-free two-generator family term map.

    ``entries`` maps ``(p, q, r, s)`` to the coefficient multiplying
    ``E_pq E_rs`` with ``E`` spin-summed.
    """
    term_map = {}
    spin_terms = (("cdu", "cu"), ("cdd", "cd"))
    for (p, q, r, s), value in dict(entries or {}).items():
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)
        value = complex(value)
        if abs(value) <= cutoff:
            continue
        for left_create, left_destroy in spin_terms:
            for right_create, right_destroy in spin_terms:
                accumulate_spatial_jw_term(
                    term_map,
                    [left_create, left_destroy, right_create, right_destroy],
                    [p, q, r, s],
                    value,
                    tol=cutoff,
                )
    return term_map


def spatial_complementary_family_term_maps(families, *, cutoff=1.0e-14):
    """
    Build Hamiltonian term maps from block2-style complementary families.

    The current Abelian spatial Hamiltonian uses the spin-free identity

        H = sum_ps R_ps E_ps + sum_pqrs P_pqrs E_pq E_rs

    where ``R`` already includes the one-body ``Q`` correction.  Structural
    families ``S/A/B/Q`` remain available as metadata/operator-build inputs,
    but only ``R`` and ``P`` carry scalar Hamiltonian coefficients here.
    """
    if families is None:
        return {}
    r_family = families.get("R") if hasattr(families, "get") else None
    p_family = families.get("P") if hasattr(families, "get") else None
    maps = {
        "R": spatial_generator_family_term_map(
            getattr(r_family, "entries", {}) if r_family is not None else {},
            cutoff=cutoff,
        ),
        "P": spatial_two_generator_family_term_map(
            getattr(p_family, "entries", {}) if p_family is not None else {},
            cutoff=cutoff,
        ),
    }
    return {name: term_map for name, term_map in maps.items() if term_map}


def spatial_complementary_family_hamiltonian_term_map(families, *, cutoff=1.0e-14):
    """Return the merged Hamiltonian term map represented by ``R`` and ``P``."""
    return merge_term_maps(
        *spatial_complementary_family_term_maps(families, cutoff=cutoff).values(),
        cutoff=cutoff,
    )


def merge_term_maps(*term_maps, cutoff=1.0e-14):
    """Merge symbolic term maps using the same cancellation rule as accumulation."""
    merged = {}
    for term_map in term_maps:
        for (symbol, dofs), factor in term_map.items():
            accumulate_symbolic_term(merged, symbol, dofs, factor, tol=cutoff)
    return merged


def _kron_all(operators):
    out = np.asarray(operators[0], dtype=complex)
    for operator in operators[1:]:
        out = np.kron(out, np.asarray(operator, dtype=complex))
    return out


def dense_from_spatial_term_map(term_map, nsites):
    """Materialize a spatial-site symbolic term map as a dense operator."""
    ops = spatial_local_ops()
    ident = ops["I"]
    dense = np.zeros((4 ** int(nsites), 4 ** int(nsites)), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        local = [ident.copy() for _ in range(int(nsites))]
        for piece, site in zip(str(symbol).split(), dofs):
            local[int(site)] = ops[piece]
        dense += complex(factor) * _kron_all(local)
    return dense


def spatial_complementary_local_term_maps(families, bond, *, cutoff=1.0e-14):
    """
    Build channel-resolved purely two-site R/P complementary terms.

    This is the local physical-site piece of the block2-style family
    decomposition.  It intentionally includes only terms whose orbital indices
    are fully inside the active two-site window ``(bond, bond + 1)``.  Boundary
    terms crossing into the left or right renormalized blocks require separate
    S/R/A/P/B/Q operator tables.
    """
    if families is None:
        return {}
    active = {int(bond), int(bond) + 1}
    maps = {"R": {}, "P": {}}
    spin_terms = (("cdu", "cu"), ("cdd", "cd"))

    r_family = families.get("R") if hasattr(families, "get") else None
    r_entries = getattr(r_family, "entries", {}) if r_family is not None else {}
    for (p, q), value in r_entries.items():
        p = int(p)
        q = int(q)
        if {p, q} - active:
            continue
        for create, destroy in spin_terms:
            symbol, dofs, factor = spatial_jw_term_spec(
                [create, destroy],
                [p, q],
                complex(value),
            )
            if set(dofs) <= active:
                shifted = [int(site) - int(bond) for site in dofs]
                accumulate_symbolic_term(maps["R"], symbol, shifted, factor, tol=cutoff)

    p_family = families.get("P") if hasattr(families, "get") else None
    p_entries = getattr(p_family, "entries", {}) if p_family is not None else {}
    for (p, q, r, s), value in p_entries.items():
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)
        if {p, q, r, s} - active:
            continue
        for left_create, left_destroy in spin_terms:
            for right_create, right_destroy in spin_terms:
                symbol, dofs, factor = spatial_jw_term_spec(
                    [left_create, left_destroy, right_create, right_destroy],
                    [p, q, r, s],
                    complex(value),
                )
                if set(dofs) <= active:
                    shifted = [int(site) - int(bond) for site in dofs]
                    accumulate_symbolic_term(maps["P"], symbol, shifted, factor, tol=cutoff)

    return {name: term_map for name, term_map in maps.items() if term_map}


def spatial_complementary_local_term_map(families, bond, *, cutoff=1.0e-14):
    """Build the summed purely two-site R/P complementary contribution."""
    return merge_term_maps(
        *spatial_complementary_local_term_maps(families, bond, cutoff=cutoff).values(),
        cutoff=cutoff,
    )


def spatial_complementary_local_matrices(families, bond, *, cutoff=1.0e-14):
    """Return dense two-site matrices for local R and P channels."""
    return {
        name: dense_from_spatial_term_map(term_map, 2)
        for name, term_map in spatial_complementary_local_term_maps(
            families,
            bond,
            cutoff=cutoff,
        ).items()
    }


def spatial_complementary_local_matrix(families, bond, *, cutoff=1.0e-14):
    """Return the dense two-site matrix for local R/P complementary terms."""
    return dense_from_spatial_term_map(
        spatial_complementary_local_term_map(families, bond, cutoff=cutoff),
        2,
    )
