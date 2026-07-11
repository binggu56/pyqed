#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builtin Gaussian derivative integrals.

This module differentiates the existing builtin Cartesian Gaussian integrals
with respect to nuclear center coordinates.  It intentionally stays independent
of libcint/PySCF integral engines.
"""

from functools import lru_cache

import numpy as np

from .basis import _basis_cy, electron_repulsion, kinetic, nuclear_attraction, overlap


def _basis_and_transform(mol):
    basis = getattr(mol, "_bas_cart", None)
    transform = getattr(mol, "_ao_cart2sph", None)
    if basis is None:
        basis = getattr(mol, "_bas", None)
        transform = None
    if basis is None:
        raise ValueError("Build the molecule with driver='builtin' before requesting derivative integrals.")
    return tuple(basis), transform


def _atom_ids_for_basis(basis, atom_coords, tol=1e-10):
    coords = np.asarray(atom_coords, dtype=float)
    ids = []
    for fn in basis:
        delta = np.linalg.norm(coords - np.asarray(fn.origin, dtype=float), axis=1)
        idx = int(np.argmin(delta))
        if delta[idx] > tol:
            raise ValueError("Could not assign a basis function to a nuclear center.")
        ids.append(idx)
    return np.asarray(ids, dtype=int)


def _as_order(order):
    out = tuple(int(x) for x in order)
    if len(out) != 3 or any(x < 0 for x in out):
        raise ValueError("Derivative orders must be a length-3 tuple of nonnegative integers.")
    if sum(out) > 2:
        raise NotImplementedError("Builtin derivative integrals currently support orders through second order.")
    return out


def _raise_axis(shell, axis, delta):
    shell = list(shell)
    shell[axis] += int(delta)
    if shell[axis] < 0:
        return None
    return tuple(shell)


@lru_cache(maxsize=16384)
def _primitive_derivative_terms(expnt, shell, order):
    """
    Terms for center derivatives of a primitive Cartesian Gaussian.

    d/dA_x x_A^l exp(-a x_A^2) = 2a x_A^(l+1) exp(...) - l x_A^(l-1) exp(...)
    """
    order = _as_order(order)
    terms = [(1.0, tuple(int(x) for x in shell))]
    for axis, count in enumerate(order):
        for _ in range(count):
            new_terms = []
            for coeff, lmn in terms:
                raised = _raise_axis(lmn, axis, 1)
                new_terms.append((coeff * 2.0 * float(expnt), raised))
                power = lmn[axis]
                lowered = _raise_axis(lmn, axis, -1)
                if power > 0 and lowered is not None:
                    new_terms.append((coeff * -float(power), lowered))
            terms = new_terms
    return tuple((float(c), tuple(lmn)) for c, lmn in terms if c != 0.0)


def _contracted_one_deriv(fn_a, fn_b, kernel, order_a=(0, 0, 0), order_b=(0, 0, 0), center=None):
    order_a = _as_order(order_a)
    order_b = _as_order(order_b)
    value = 0.0
    for ia, wa in enumerate(fn_a.prim_weights):
        exp_a = float(fn_a.exps[ia])
        terms_a = _primitive_derivative_terms(exp_a, tuple(fn_a.shell), order_a)
        for ib, wb in enumerate(fn_b.prim_weights):
            exp_b = float(fn_b.exps[ib])
            pref = float(wa) * float(wb)
            terms_b = _primitive_derivative_terms(exp_b, tuple(fn_b.shell), order_b)
            for ca, shell_a in terms_a:
                for cb, shell_b in terms_b:
                    if kernel == "overlap":
                        value += pref * ca * cb * S_primitive(
                            exp_a, shell_a, fn_a.origin, exp_b, shell_b, fn_b.origin
                        )
                    elif kernel == "kinetic":
                        value += pref * ca * cb * kinetic(
                            exp_a, shell_a, fn_a.origin, exp_b, shell_b, fn_b.origin
                        )
                    elif kernel == "nuclear":
                        value += pref * ca * cb * nuclear_attraction(
                            exp_a, shell_a, fn_a.origin, exp_b, shell_b, fn_b.origin, center
                        )
                    else:
                        raise ValueError("kernel must be 'overlap', 'kinetic', or 'nuclear'.")
    return value


def _primitive_moment(exp_a, shell_a, origin_a, exp_b, shell_b, origin_b, axis, center):
    raised = _raise_axis(shell_a, axis, 1)
    return overlap(exp_a, raised, origin_a, exp_b, shell_b, origin_b) + (
        float(origin_a[axis]) - float(center[axis])
    ) * overlap(exp_a, shell_a, origin_a, exp_b, shell_b, origin_b)


def _contracted_position_deriv(
    fn_a,
    fn_b,
    moment_axis,
    order_a=(0, 0, 0),
    order_b=(0, 0, 0),
    center=None,
):
    order_a = _as_order(order_a)
    order_b = _as_order(order_b)
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)
    value = 0.0
    for ia, wa in enumerate(fn_a.prim_weights):
        exp_a = float(fn_a.exps[ia])
        terms_a = _primitive_derivative_terms(exp_a, tuple(fn_a.shell), order_a)
        for ib, wb in enumerate(fn_b.prim_weights):
            exp_b = float(fn_b.exps[ib])
            pref = float(wa) * float(wb)
            terms_b = _primitive_derivative_terms(exp_b, tuple(fn_b.shell), order_b)
            for ca, shell_a in terms_a:
                for cb, shell_b in terms_b:
                    value += pref * ca * cb * _primitive_moment(
                        exp_a,
                        shell_a,
                        fn_a.origin,
                        exp_b,
                        shell_b,
                        fn_b.origin,
                        moment_axis,
                        center,
                    )
    return value


def S_primitive(exp_a, shell_a, origin_a, exp_b, shell_b, origin_b):
    return overlap(exp_a, shell_a, origin_a, exp_b, shell_b, origin_b)


def _basis_signature(fn):
    return (
        tuple(int(x) for x in fn.shell),
        tuple(float(x) for x in fn.origin),
        tuple(float(x) for x in fn.exps),
        tuple(float(x) for x in fn.prim_weights),
    )


@lru_cache(maxsize=32768)
def _derivative_signatures_from_signature(sig, order):
    order = _as_order(order)
    shell, origin, exps, weights = sig
    grouped = {}
    for idx, (expnt, weight) in enumerate(zip(exps, weights)):
        for coeff, term_shell in _primitive_derivative_terms(expnt, shell, order):
            if term_shell not in grouped:
                grouped[term_shell] = [0.0] * len(weights)
            grouped[term_shell][idx] += float(weight) * coeff

    out = []
    for term_shell, term_weights in grouped.items():
        if any(weight != 0.0 for weight in term_weights):
            out.append(
                (
                    tuple(int(x) for x in term_shell),
                    origin,
                    exps,
                    tuple(float(x) for x in term_weights),
                )
            )
    return tuple(out)


def _derivative_signatures(fn, order):
    return _derivative_signatures_from_signature(_basis_signature(fn), order)


@lru_cache(maxsize=4096)
def _cartesian_shell_components(l):
    out = []
    for ix in range(l, -1, -1):
        rem = l - ix
        for iy in range(rem, -1, -1):
            out.append((ix, iy, rem - iy))
    return tuple(out)


def _signature_shell_family(sig):
    shell, origin, exps, weights = sig
    return (int(sum(shell)), origin, exps, weights)


@lru_cache(maxsize=16384)
def _compiled_eri_shell_block(fam_a, fam_b, fam_c, fam_d):
    if _basis_cy is None:
        return None

    families = (fam_a, fam_b, fam_c, fam_d)
    starts = []
    stops = []
    shells = []
    origins = []
    exps = []
    weights = []

    for l, origin, sig_exps, sig_weights in families:
        starts.append(len(shells))
        components = _cartesian_shell_components(int(l))
        for component in components:
            shells.append(component)
            origins.append(origin)
            exps.append(sig_exps)
            weights.append(sig_weights)
        stops.append(len(shells))

    max_prim = max(len(row) for row in exps)
    shell_arr = np.asarray(shells, dtype=np.int64)
    origin_arr = np.asarray(origins, dtype=np.float64)
    exp_arr = np.zeros((len(shells), max_prim), dtype=np.float64)
    weight_arr = np.zeros((len(shells), max_prim), dtype=np.float64)
    nprim = np.empty(len(shells), dtype=np.int64)
    for idx, (sig_exps, sig_weights) in enumerate(zip(exps, weights)):
        n = len(sig_exps)
        nprim[idx] = n
        exp_arr[idx, :n] = sig_exps
        weight_arr[idx, :n] = sig_weights

    block = _basis_cy.compute_cartesian_shell_quartet_block(
        shell_arr,
        origin_arr,
        exp_arr,
        weight_arr,
        nprim,
        starts[0],
        stops[0],
        starts[1],
        stops[1],
        starts[2],
        stops[2],
        starts[3],
        stops[3],
    )
    return np.asarray(block, dtype=float)


def _compiled_eri_component(sig_a, sig_b, sig_c, sig_d):
    if _basis_cy is None:
        return None

    signatures = (sig_a, sig_b, sig_c, sig_d)
    families = tuple(_signature_shell_family(sig) for sig in signatures)
    component_indices = []
    for sig, family in zip(signatures, families):
        shell = tuple(int(x) for x in sig[0])
        components = _cartesian_shell_components(family[0])
        try:
            component_indices.append(components.index(shell))
        except ValueError:
            return None

    block = _compiled_eri_shell_block(*families)
    if block is None:
        return None
    return float(
        block[
            component_indices[0],
            component_indices[1],
            component_indices[2],
            component_indices[3],
        ]
    )


def _canonical_eri_signatures(sig_a, sig_b, sig_c, sig_d):
    pair_ab = (sig_a, sig_b) if sig_a <= sig_b else (sig_b, sig_a)
    pair_cd = (sig_c, sig_d) if sig_c <= sig_d else (sig_d, sig_c)
    if pair_ab <= pair_cd:
        return pair_ab + pair_cd
    return pair_cd + pair_ab


def _contracted_eri_from_signatures(sig_a, sig_b, sig_c, sig_d):
    return _contracted_eri_from_signatures_cached(
        *_canonical_eri_signatures(sig_a, sig_b, sig_c, sig_d)
    )


@lru_cache(maxsize=262144)
def _contracted_eri_from_signatures_cached(sig_a, sig_b, sig_c, sig_d):
    compiled = _compiled_eri_component(sig_a, sig_b, sig_c, sig_d)
    if compiled is not None:
        return compiled

    shell_a, origin_a, exps_a, weights_a = sig_a
    shell_b, origin_b, exps_b, weights_b = sig_b
    shell_c, origin_c, exps_c, weights_c = sig_c
    shell_d, origin_d, exps_d, weights_d = sig_d
    value = 0.0
    for ia, wa in enumerate(weights_a):
        for ib, wb in enumerate(weights_b):
            for ic, wc in enumerate(weights_c):
                for id_, wd in enumerate(weights_d):
                    value += wa * wb * wc * wd * electron_repulsion(
                        exps_a[ia],
                        shell_a,
                        origin_a,
                        exps_b[ib],
                        shell_b,
                        origin_b,
                        exps_c[ic],
                        shell_c,
                        origin_c,
                        exps_d[id_],
                        shell_d,
                        origin_d,
                    )
    return value


def _contracted_eri_deriv(
    fn_a,
    fn_b,
    fn_c,
    fn_d,
    order_a=(0, 0, 0),
    order_b=(0, 0, 0),
    order_c=(0, 0, 0),
    order_d=(0, 0, 0),
):
    order_a = _as_order(order_a)
    order_b = _as_order(order_b)
    order_c = _as_order(order_c)
    order_d = _as_order(order_d)
    if _basis_cy is not None:
        value = 0.0
        for sig_a in _derivative_signatures(fn_a, order_a):
            for sig_b in _derivative_signatures(fn_b, order_b):
                for sig_c in _derivative_signatures(fn_c, order_c):
                    for sig_d in _derivative_signatures(fn_d, order_d):
                        value += _contracted_eri_from_signatures(sig_a, sig_b, sig_c, sig_d)
        return value

    value = 0.0
    for ia, wa in enumerate(fn_a.prim_weights):
        exp_a = float(fn_a.exps[ia])
        terms_a = _primitive_derivative_terms(exp_a, tuple(fn_a.shell), order_a)
        for ib, wb in enumerate(fn_b.prim_weights):
            exp_b = float(fn_b.exps[ib])
            terms_b = _primitive_derivative_terms(exp_b, tuple(fn_b.shell), order_b)
            for ic, wc in enumerate(fn_c.prim_weights):
                exp_c = float(fn_c.exps[ic])
                terms_c = _primitive_derivative_terms(exp_c, tuple(fn_c.shell), order_c)
                for id_, wd in enumerate(fn_d.prim_weights):
                    exp_d = float(fn_d.exps[id_])
                    pref = float(wa) * float(wb) * float(wc) * float(wd)
                    terms_d = _primitive_derivative_terms(exp_d, tuple(fn_d.shell), order_d)
                    for ca, shell_a in terms_a:
                        for cb, shell_b in terms_b:
                            for cc, shell_c in terms_c:
                                for cd, shell_d in terms_d:
                                    value += pref * ca * cb * cc * cd * electron_repulsion(
                                        exp_a,
                                        shell_a,
                                        fn_a.origin,
                                        exp_b,
                                        shell_b,
                                        fn_b.origin,
                                        exp_c,
                                        shell_c,
                                        fn_c.origin,
                                        exp_d,
                                        shell_d,
                                        fn_d.origin,
                                    )
    return value


def _axis_order(axis):
    out = [0, 0, 0]
    out[int(axis)] = 1
    return tuple(out)


def _second_order(axis_a, axis_b):
    out = [0, 0, 0]
    out[int(axis_a)] += 1
    out[int(axis_b)] += 1
    return tuple(out)


def _transform_one(mat, transform):
    if transform is None:
        return mat
    return np.einsum("pi,...pq,qj->...ij", transform, mat, transform, optimize=True)


def _transform_eri(eri, transform):
    if transform is None:
        return eri
    return np.einsum(
        "pa,qb,rc,sd,...pqrs->...abcd",
        transform,
        transform,
        transform,
        transform,
        eri,
        optimize=True,
    )


def _pack_eri_s2kl(eri):
    """Pack the last two AO indices of an ERI-like tensor."""

    nao = eri.shape[-1]
    pairs = _ao_pair_indices(nao)
    out = np.empty(eri.shape[:-2] + (len(pairs),), dtype=eri.dtype)
    for pair, (r, s) in enumerate(pairs):
        out[..., pair] = eri[..., r, s]
    return out


def _ao_pair_indices(nao):
    return [(p, q) for p in range(nao) for q in range(p + 1)]


def _ao_pair_lookup(nao, pairs=None):
    if pairs is None:
        pairs = _ao_pair_indices(nao)
    lookup = np.empty((nao, nao), dtype=int)
    for idx, (p, q) in enumerate(pairs):
        lookup[p, q] = idx
        lookup[q, p] = idx
    return lookup


def _eri_permutations(p, q, r, s):
    return {
        (p, q, r, s),
        (q, p, r, s),
        (p, q, s, r),
        (q, p, s, r),
        (r, s, p, q),
        (s, r, p, q),
        (r, s, q, p),
        (s, r, q, p),
    }


def _eri_first_derivative_value(centers, eval_orders, atom, axis):
    value = 0.0
    for slot, center_atom in enumerate(centers):
        if center_atom != atom:
            continue
        orders = [(0, 0, 0)] * 4
        orders[slot] = _axis_order(axis)
        value += eval_orders(orders)
    return value


def _eri_second_derivative_value(centers, eval_orders, atom_a, axis_a, atom_b, axis_b):
    value = 0.0
    for slot_a, center_a in enumerate(centers):
        if center_a != atom_a:
            continue
        for slot_b, center_b in enumerate(centers):
            if center_b != atom_b:
                continue
            orders = [(0, 0, 0)] * 4
            if slot_a == slot_b:
                orders[slot_a] = _second_order(axis_a, axis_b)
            else:
                orders[slot_a] = _axis_order(axis_a)
                orders[slot_b] = _axis_order(axis_b)
            value += eval_orders(orders)
    return value


class CompactERIDerivatives:
    """
    Pair-packed derivative ERIs in a Cartesian AO-pair basis.

    ``data`` stores pair-pair matrices with shape ``(..., npair, npair)``.
    If ``transform`` is not ``None``, it maps Cartesian AOs to the molecule AO
    basis as ``T[cart, ao]``.
    """

    def __init__(self, data, pairs, nao_cart, nao, transform=None):
        self.data = np.asarray(data, dtype=float)
        self.pairs = tuple((int(p), int(q)) for p, q in pairs)
        self.nao_cart = int(nao_cart)
        self.nao = int(nao)
        self.transform = None if transform is None else np.asarray(transform, dtype=float)
        self.pair_lookup = _ao_pair_lookup(self.nao_cart, self.pairs)

    def block(self, *index):
        if not index:
            return self.data
        return self.data[index]

    def veff(self, dm, *index):
        return compact_eri_veff(self, dm, *index)


def _transform_dm_to_cart(dm, transform):
    dm = np.asarray(dm, dtype=float)
    if transform is None:
        return dm
    return np.einsum("pa,ab,qb->pq", transform, dm, transform, optimize=True)


def _transform_mat_from_cart(mat, transform):
    if transform is None:
        return mat
    return np.einsum("pa,pq,qb->ab", transform, mat, transform, optimize=True)


def _pack_symmetric_density(dm, pairs):
    packed = np.empty(len(pairs), dtype=float)
    for idx, (p, q) in enumerate(pairs):
        packed[idx] = dm[p, p] if p == q else dm[p, q] + dm[q, p]
    return packed


def _unpack_symmetric_pair_values(values, pairs, nao):
    mat = np.zeros((nao, nao), dtype=float)
    for idx, (p, q) in enumerate(pairs):
        mat[p, q] = values[idx]
        mat[q, p] = values[idx]
    return mat


def compact_eri_jk(compact, dm, *index):
    """
    Contract pair-packed ERIs with a density matrix.
    """
    eri_pair = compact.block(*index)
    transform = compact.transform
    dm_cart = _transform_dm_to_cart(dm, transform)
    pairs = compact.pairs
    lookup = compact.pair_lookup
    nao = compact.nao_cart

    dm_pair = _pack_symmetric_density(dm_cart, pairs)
    vj_cart = _unpack_symmetric_pair_values(eri_pair @ dm_pair, pairs, nao)

    exchange = eri_pair[
        lookup[:, :, None, None],
        lookup[None, None, :, :],
    ]
    vk_cart = np.einsum("rs,prqs->pq", dm_cart, exchange, optimize=True)

    return (
        _transform_mat_from_cart(vj_cart, transform),
        _transform_mat_from_cart(vk_cart, transform),
    )


def compact_eri_veff_many(compact, dm):
    """
    Contract every leading block of a compact derivative ERI with one density.
    """
    data = compact.data.reshape(-1, compact.data.shape[-2], compact.data.shape[-1])
    transform = compact.transform
    dm_cart = _transform_dm_to_cart(dm, transform)
    pairs = compact.pairs
    lookup = compact.pair_lookup
    nao = compact.nao_cart
    nblock = data.shape[0]

    dm_pair = _pack_symmetric_density(dm_cart, pairs)
    vj_pair = np.einsum("bpq,q->bp", data, dm_pair, optimize=True)
    vj_cart = np.zeros((nblock, nao, nao), dtype=float)
    for idx, (p, q) in enumerate(pairs):
        vj_cart[:, p, q] = vj_pair[:, idx]
        vj_cart[:, q, p] = vj_pair[:, idx]

    exchange = data[
        :,
        lookup[:, :, None, None],
        lookup[None, None, :, :],
    ]
    vk_cart = np.einsum("rs,bprqs->bpq", dm_cart, exchange, optimize=True)
    veff_cart = vj_cart - 0.5 * vk_cart
    if transform is not None:
        veff = np.einsum("pa,bpq,qc->bac", transform, veff_cart, transform, optimize=True)
    else:
        veff = veff_cart
    return veff.reshape(compact.data.shape[:-2] + (compact.nao, compact.nao))


def compact_eri_veff(compact, dm, *index):
    vj, vk = compact_eri_jk(compact, dm, *index)
    return vj - 0.5 * vk


def _eri_scalar_weights(perms, dm_left, dm_right):
    j_weight = 0.0
    k_weight = 0.0
    for a, b, c, d in perms:
        j_weight += dm_left[a, b] * dm_right[c, d]
        k_weight += dm_left[a, c] * dm_right[b, d]
    return j_weight - 0.5 * k_weight


def _one_center_coeffs(kernel, atom, atom_p, atom_q, charge_atom=None):
    ca = 1.0 if atom_p == atom else 0.0
    cb = 1.0 if atom_q == atom else 0.0
    if kernel == "nuclear" and charge_atom == atom:
        ca -= 1.0
        cb -= 1.0
    return ca, cb


def _one_deriv_element(fn_p, fn_q, atom_p, atom_q, atom, axis, kernel, charges, coords):
    ca, cb = _one_center_coeffs(kernel, atom, atom_p, atom_q)
    if kernel == "nuclear":
        value = 0.0
        for charge_atom, charge in enumerate(charges):
            va, vb = _one_center_coeffs(kernel, atom, atom_p, atom_q, charge_atom)
            if va:
                value -= charge * va * _contracted_one_deriv(
                    fn_p, fn_q, "nuclear", order_a=_axis_order(axis), center=coords[charge_atom]
                )
            if vb:
                value -= charge * vb * _contracted_one_deriv(
                    fn_p, fn_q, "nuclear", order_b=_axis_order(axis), center=coords[charge_atom]
                )
        return value

    value = 0.0
    if ca:
        value += ca * _contracted_one_deriv(fn_p, fn_q, kernel, order_a=_axis_order(axis))
    if cb:
        value += cb * _contracted_one_deriv(fn_p, fn_q, kernel, order_b=_axis_order(axis))
    return value


def _one_second_element(fn_p, fn_q, atom_p, atom_q, atom_a, axis_a, atom_b, axis_b, kernel, charges, coords):
    if kernel == "nuclear":
        value = 0.0
        for charge_atom, charge in enumerate(charges):
            ca1, cb1 = _one_center_coeffs(kernel, atom_a, atom_p, atom_q, charge_atom)
            ca2, cb2 = _one_center_coeffs(kernel, atom_b, atom_p, atom_q, charge_atom)
            if ca1 and ca2:
                value -= charge * ca1 * ca2 * _contracted_one_deriv(
                    fn_p, fn_q, "nuclear", order_a=_second_order(axis_a, axis_b), center=coords[charge_atom]
                )
            if ca1 and cb2:
                value -= charge * ca1 * cb2 * _contracted_one_deriv(
                    fn_p,
                    fn_q,
                    "nuclear",
                    order_a=_axis_order(axis_a),
                    order_b=_axis_order(axis_b),
                    center=coords[charge_atom],
                )
            if cb1 and ca2:
                value -= charge * cb1 * ca2 * _contracted_one_deriv(
                    fn_p,
                    fn_q,
                    "nuclear",
                    order_a=_axis_order(axis_b),
                    order_b=_axis_order(axis_a),
                    center=coords[charge_atom],
                )
            if cb1 and cb2:
                value -= charge * cb1 * cb2 * _contracted_one_deriv(
                    fn_p, fn_q, "nuclear", order_b=_second_order(axis_a, axis_b), center=coords[charge_atom]
                )
        return value

    ca1, cb1 = _one_center_coeffs(kernel, atom_a, atom_p, atom_q)
    ca2, cb2 = _one_center_coeffs(kernel, atom_b, atom_p, atom_q)
    value = 0.0
    if ca1 and ca2:
        value += ca1 * ca2 * _contracted_one_deriv(
            fn_p, fn_q, kernel, order_a=_second_order(axis_a, axis_b)
        )
    if ca1 and cb2:
        value += ca1 * cb2 * _contracted_one_deriv(
            fn_p, fn_q, kernel, order_a=_axis_order(axis_a), order_b=_axis_order(axis_b)
        )
    if cb1 and ca2:
        value += cb1 * ca2 * _contracted_one_deriv(
            fn_p, fn_q, kernel, order_a=_axis_order(axis_b), order_b=_axis_order(axis_a)
        )
    if cb1 and cb2:
        value += cb1 * cb2 * _contracted_one_deriv(
            fn_p, fn_q, kernel, order_b=_second_order(axis_a, axis_b)
        )
    return value


def one_electron_derivatives(mol, kernel="hcore", order=1):
    """
    Return builtin one-electron integral derivatives in the molecule AO basis.

    Parameters
    ----------
    mol
        Built ``Molecule`` object.
    kernel : {'overlap', 'kinetic', 'nuclear', 'hcore'}
        Integral family. ``'nuclear'`` includes all electron-nuclear attraction
        centers and charges. ``'hcore'`` is kinetic plus nuclear attraction.
    order : {1, 2}
        Nuclear derivative order.

    Returns
    -------
    ndarray
        First order: ``(natm, 3, nao, nao)``.
        Second order: ``(natm, 3, natm, 3, nao, nao)``.
    """
    kernel = str(kernel).lower()
    if kernel not in {"overlap", "kinetic", "nuclear", "hcore"}:
        raise ValueError("kernel must be 'overlap', 'kinetic', 'nuclear', or 'hcore'.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if kernel == "hcore":
        return (
            one_electron_derivatives(mol, "kinetic", order=order)
            + one_electron_derivatives(mol, "nuclear", order=order)
        )

    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)

    if order == 1:
        out = np.zeros((natm, 3, nao_cart, nao_cart), dtype=float)
        for p, fn_p in enumerate(basis):
            for q, fn_q in enumerate(basis[: p + 1]):
                for atom in range(natm):
                    for axis in range(3):
                        value = _one_deriv_element(
                            fn_p, fn_q, atom_ids[p], atom_ids[q], atom, axis, kernel, charges, coords
                        )
                        out[atom, axis, p, q] = value
                        out[atom, axis, q, p] = value
        return _transform_one(out, transform)

    out = np.zeros((natm, 3, natm, 3, nao_cart, nao_cart), dtype=float)
    perturbations = [(atom, axis) for atom in range(natm) for axis in range(3)]
    for p, fn_p in enumerate(basis):
        for q, fn_q in enumerate(basis[: p + 1]):
            for pert_a, (atom_a, axis_a) in enumerate(perturbations):
                for atom_b, axis_b in perturbations[: pert_a + 1]:
                    value = _one_second_element(
                        fn_p,
                        fn_q,
                        atom_ids[p],
                        atom_ids[q],
                        atom_a,
                        axis_a,
                        atom_b,
                        axis_b,
                        kernel,
                        charges,
                        coords,
                    )
                    out[atom_a, axis_a, atom_b, axis_b, p, q] = value
                    out[atom_b, axis_b, atom_a, axis_a, p, q] = value
                    out[atom_a, axis_a, atom_b, axis_b, q, p] = value
                    out[atom_b, axis_b, atom_a, axis_a, q, p] = value
    return _transform_one(out, transform)


def one_index_one_electron_derivatives(mol, kernel="overlap", index="ket"):
    """
    Return one-index first derivatives of one-electron AO integrals.

    Only one AO center is differentiated.  For ``index='ket'`` this returns
    ``<chi_p | d chi_q / d R_A>`` for basis functions ``q`` on atom ``A``;
    ``index='bra'`` differentiates ``chi_p`` instead.

    Parameters
    ----------
    mol
        Built ``Molecule`` object.
    kernel : {'overlap', 'kinetic'}
        Integral family.  One-index nuclear-attraction derivatives are not
        included here because nuclear attraction also depends explicitly on
        nuclear charge centers.
    index : {'bra', 'ket'}
        AO index whose Gaussian center is differentiated.

    Returns
    -------
    ndarray
        Shape ``(natm, 3, nao, nao)``.
    """

    kernel = str(kernel).lower()
    if kernel not in {"overlap", "kinetic"}:
        raise ValueError("kernel must be 'overlap' or 'kinetic'.")
    index = str(index).lower()
    if index not in {"bra", "ket"}:
        raise ValueError("index must be 'bra' or 'ket'.")

    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)

    out = np.zeros((natm, 3, nao_cart, nao_cart), dtype=float)
    for p, fn_p in enumerate(basis):
        for q, fn_q in enumerate(basis):
            atom = atom_ids[p] if index == "bra" else atom_ids[q]
            for axis in range(3):
                order = _axis_order(axis)
                if index == "bra":
                    value = _contracted_one_deriv(fn_p, fn_q, kernel, order_a=order)
                else:
                    value = _contracted_one_deriv(fn_p, fn_q, kernel, order_b=order)
                out[atom, axis, p, q] = value
    return _transform_one(out, transform)


def position_derivatives(mol, center=None):
    """
    Return first nuclear derivatives of AO position integrals.

    The returned tensor has shape ``(natm, 3, 3, nao, nao)`` with indices
    ``(atom, nuclear_axis, moment_axis, ao, ao)``.
    """
    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)

    out = np.zeros((natm, 3, 3, nao_cart, nao_cart), dtype=float)
    for p, fn_p in enumerate(basis):
        for q, fn_q in enumerate(basis):
            for atom in range(natm):
                for axis in range(3):
                    ca = 1.0 if atom_ids[p] == atom else 0.0
                    cb = 1.0 if atom_ids[q] == atom else 0.0
                    if not ca and not cb:
                        continue
                    for moment_axis in range(3):
                        value = 0.0
                        if ca:
                            value += _contracted_position_deriv(
                                fn_p,
                                fn_q,
                                moment_axis,
                                order_a=_axis_order(axis),
                                center=center,
                            )
                        if cb:
                            value += _contracted_position_deriv(
                                fn_p,
                                fn_q,
                                moment_axis,
                                order_b=_axis_order(axis),
                                center=center,
                            )
                        out[atom, axis, moment_axis, p, q] = value
    return _transform_one(out, transform)


def _eri_center_coeff(atom, center_atom):
    return 1.0 if atom == center_atom else 0.0


def _eri_derivatives_compact(mol, order=1):
    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)
    nao = int(getattr(mol, "nao", nao_cart))
    pairs = _ao_pair_indices(nao_cart)
    npert = natm * 3
    npair = len(pairs)

    if order == 1:
        data = np.zeros((npert, npair, npair), dtype=float)
        for pair_pq, (p, q) in enumerate(pairs):
            for pair_rs, (r, s) in enumerate(pairs[: pair_pq + 1]):
                centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
                unique_centers = sorted(set(centers))
                if len(unique_centers) == 1:
                    continue
                fns = (basis[p], basis[q], basis[r], basis[s])
                cache = {}

                def eval_orders(orders):
                    key = tuple(orders)
                    if key not in cache:
                        cache[key] = _contracted_eri_deriv(*fns, *orders)
                    return cache[key]

                ref_atom = unique_centers[-1]
                ref_sum = np.zeros(3, dtype=float)
                for atom in unique_centers[:-1]:
                    for axis in range(3):
                        value = _eri_first_derivative_value(centers, eval_orders, atom, axis)
                        pert = atom * 3 + axis
                        data[pert, pair_pq, pair_rs] = value
                        data[pert, pair_rs, pair_pq] = value
                        ref_sum[axis] += value
                for axis in range(3):
                    value = -ref_sum[axis]
                    pert = ref_atom * 3 + axis
                    data[pert, pair_pq, pair_rs] = value
                    data[pert, pair_rs, pair_pq] = value
        return CompactERIDerivatives(data, pairs, nao_cart, nao, transform=transform)

    data = np.zeros((npert, npert, npair, npair), dtype=float)
    for pair_pq, (p, q) in enumerate(pairs):
        for pair_rs, (r, s) in enumerate(pairs[: pair_pq + 1]):
            centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
            unique_centers = sorted(set(centers))
            if len(unique_centers) == 1:
                continue
            fns = (basis[p], basis[q], basis[r], basis[s])
            ref_atom = unique_centers[-1]
            independent_perturbations = [(atom, axis) for atom in unique_centers[:-1] for axis in range(3)]
            all_perturbations = [(atom, axis) for atom in unique_centers for axis in range(3)]
            cache = {}

            def eval_orders(orders):
                key = tuple(orders)
                if key not in cache:
                    cache[key] = _contracted_eri_deriv(*fns, *orders)
                return cache[key]

            block = {}
            for pert_a, (atom_a, axis_a) in enumerate(independent_perturbations):
                for atom_b, axis_b in independent_perturbations[: pert_a + 1]:
                    value = _eri_second_derivative_value(
                        centers,
                        eval_orders,
                        atom_a,
                        axis_a,
                        atom_b,
                        axis_b,
                    )
                    block[(atom_a, axis_a, atom_b, axis_b)] = value
                    block[(atom_b, axis_b, atom_a, axis_a)] = value

            def value_from_sum_rules(atom_a, axis_a, atom_b, axis_b):
                if atom_a != ref_atom and atom_b != ref_atom:
                    return block[(atom_a, axis_a, atom_b, axis_b)]
                if atom_a == ref_atom and atom_b != ref_atom:
                    return -sum(
                        block[(other, axis_a, atom_b, axis_b)]
                        for other in unique_centers[:-1]
                    )
                if atom_a != ref_atom and atom_b == ref_atom:
                    return -sum(
                        block[(atom_a, axis_a, other, axis_b)]
                        for other in unique_centers[:-1]
                    )
                return sum(
                    block[(other_a, axis_a, other_b, axis_b)]
                    for other_a in unique_centers[:-1]
                    for other_b in unique_centers[:-1]
                )

            for pert_a, (atom_a, axis_a) in enumerate(all_perturbations):
                pidx_a = atom_a * 3 + axis_a
                for atom_b, axis_b in all_perturbations[: pert_a + 1]:
                    pidx_b = atom_b * 3 + axis_b
                    value = value_from_sum_rules(atom_a, axis_a, atom_b, axis_b)
                    data[pidx_a, pidx_b, pair_pq, pair_rs] = value
                    data[pidx_a, pidx_b, pair_rs, pair_pq] = value
                    if pidx_a != pidx_b:
                        data[pidx_b, pidx_a, pair_pq, pair_rs] = value
                        data[pidx_b, pidx_a, pair_rs, pair_pq] = value
    return CompactERIDerivatives(data, pairs, nao_cart, nao, transform=transform)


def eri_derivative_veff_scalar(mol, dm_left, dm_right, order=2):
    """
    Direct scalar contractions of derivative ERIs with two densities.

    Returns ``Tr[dm_left * G_x(dm_right)]`` for first derivatives, or
    ``Tr[dm_left * G_xy(dm_right)]`` for second derivatives, without storing
    the derivative ERI tensor.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)
    pairs = _ao_pair_indices(nao_cart)
    npert = natm * 3
    dm_l = _transform_dm_to_cart(dm_left, transform)
    dm_r = _transform_dm_to_cart(dm_right, transform)

    if order == 1:
        out = np.zeros(npert, dtype=float)
        for pair_pq, (p, q) in enumerate(pairs):
            for r, s in pairs[: pair_pq + 1]:
                centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
                unique_centers = sorted(set(centers))
                if len(unique_centers) == 1:
                    continue
                fns = (basis[p], basis[q], basis[r], basis[s])
                perms = tuple(_eri_permutations(p, q, r, s))
                weight = _eri_scalar_weights(perms, dm_l, dm_r)
                if weight == 0.0:
                    continue
                cache = {}

                def eval_orders(orders):
                    key = tuple(orders)
                    if key not in cache:
                        cache[key] = _contracted_eri_deriv(*fns, *orders)
                    return cache[key]

                ref_atom = unique_centers[-1]
                ref_sum = np.zeros(3, dtype=float)
                for atom in unique_centers[:-1]:
                    for axis in range(3):
                        value = _eri_first_derivative_value(centers, eval_orders, atom, axis)
                        out[atom * 3 + axis] += value * weight
                        ref_sum[axis] += value
                for axis in range(3):
                    out[ref_atom * 3 + axis] -= ref_sum[axis] * weight
        return out

    out = np.zeros((npert, npert), dtype=float)
    for pair_pq, (p, q) in enumerate(pairs):
        for r, s in pairs[: pair_pq + 1]:
            centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
            unique_centers = sorted(set(centers))
            if len(unique_centers) == 1:
                continue
            fns = (basis[p], basis[q], basis[r], basis[s])
            perms = tuple(_eri_permutations(p, q, r, s))
            weight = _eri_scalar_weights(perms, dm_l, dm_r)
            if weight == 0.0:
                continue
            ref_atom = unique_centers[-1]
            independent_perturbations = [(atom, axis) for atom in unique_centers[:-1] for axis in range(3)]
            all_perturbations = [(atom, axis) for atom in unique_centers for axis in range(3)]
            cache = {}

            def eval_orders(orders):
                key = tuple(orders)
                if key not in cache:
                    cache[key] = _contracted_eri_deriv(*fns, *orders)
                return cache[key]

            block = {}
            for pert_a, (atom_a, axis_a) in enumerate(independent_perturbations):
                for atom_b, axis_b in independent_perturbations[: pert_a + 1]:
                    value = _eri_second_derivative_value(
                        centers,
                        eval_orders,
                        atom_a,
                        axis_a,
                        atom_b,
                        axis_b,
                    )
                    block[(atom_a, axis_a, atom_b, axis_b)] = value
                    block[(atom_b, axis_b, atom_a, axis_a)] = value

            def value_from_sum_rules(atom_a, axis_a, atom_b, axis_b):
                if atom_a != ref_atom and atom_b != ref_atom:
                    return block[(atom_a, axis_a, atom_b, axis_b)]
                if atom_a == ref_atom and atom_b != ref_atom:
                    return -sum(
                        block[(other, axis_a, atom_b, axis_b)]
                        for other in unique_centers[:-1]
                    )
                if atom_a != ref_atom and atom_b == ref_atom:
                    return -sum(
                        block[(atom_a, axis_a, other, axis_b)]
                        for other in unique_centers[:-1]
                    )
                return sum(
                    block[(other_a, axis_a, other_b, axis_b)]
                    for other_a in unique_centers[:-1]
                    for other_b in unique_centers[:-1]
                )

            for pert_a, (atom_a, axis_a) in enumerate(all_perturbations):
                pidx_a = atom_a * 3 + axis_a
                for atom_b, axis_b in all_perturbations[: pert_a + 1]:
                    pidx_b = atom_b * 3 + axis_b
                    value = value_from_sum_rules(atom_a, axis_a, atom_b, axis_b) * weight
                    out[pidx_a, pidx_b] += value
                    if pidx_a != pidx_b:
                        out[pidx_b, pidx_a] += value
    return out


def one_index_eri_derivatives(mol, aosym="s1", convention="center"):
    """
    Return one-index AO ERI derivatives in the molecule AO basis.

    Only the first AO center of ``(pq|rs)`` is differentiated.  The returned
    tensor is useful for AO-gradient contractions that use PySCF's
    ``int2e_ip1`` primitive.

    Parameters
    ----------
    mol
        Built ``Molecule`` object.
    aosym : {'s1', 's2kl'}
        ``'s1'`` returns ``(natm, 3, nao, nao, nao, nao)``. ``'s2kl'`` packs
        the last two AO indices and returns ``(natm, 3, nao, nao, nao_pair)``.
    convention : {'center', 'ip1'}
        ``'center'`` returns derivatives with respect to the Gaussian center
        of the first AO.  ``'ip1'`` returns the opposite sign, matching the
        usual libcint/PySCF ``int2e_ip1`` electron-coordinate derivative.
    """

    aosym = str(aosym).lower()
    if aosym not in {"s1", "s2kl"}:
        raise ValueError("aosym must be 's1' or 's2kl'.")
    convention = str(convention).lower().replace("-", "_")
    if convention not in {"center", "ip1"}:
        raise ValueError("convention must be 'center' or 'ip1'.")

    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)

    out = np.zeros((natm, 3, nao_cart, nao_cart, nao_cart, nao_cart), dtype=float)
    ao_pairs = _ao_pair_indices(nao_cart)
    for p, fn_p in enumerate(basis):
        atom = atom_ids[p]
        for q, fn_q in enumerate(basis):
            for pair_rs, (r, s) in enumerate(ao_pairs):
                fns = (fn_p, fn_q, basis[r], basis[s])
                cache = {}
                for axis in range(3):
                    orders = [_axis_order(axis), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
                    key = tuple(orders)
                    if key not in cache:
                        cache[key] = _contracted_eri_deriv(*fns, *orders)
                    value = cache[key]
                    out[atom, axis, p, q, r, s] = value
                    out[atom, axis, p, q, s, r] = value

    if convention == "ip1":
        out = -out
    out = _transform_eri(out, transform)
    if aosym == "s2kl":
        out = _pack_eri_s2kl(out)
    return out


def eri_derivatives(mol, order=1, compact=False):
    """
    Return builtin electron-repulsion integral derivatives in the molecule AO basis.

    First order shape is ``(natm, 3, nao, nao, nao, nao)``.
    Second order shape is ``(natm, 3, natm, 3, nao, nao, nao, nao)``.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if compact:
        return _eri_derivatives_compact(mol, order=order)

    basis, transform = _basis_and_transform(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, coords)
    natm = coords.shape[0]
    nao_cart = len(basis)
    ao_pairs = _ao_pair_indices(nao_cart)

    if order == 1:
        out = np.zeros((natm, 3, nao_cart, nao_cart, nao_cart, nao_cart), dtype=float)
        for pair_pq, (p, q) in enumerate(ao_pairs):
            for pair_rs, (r, s) in enumerate(ao_pairs[: pair_pq + 1]):
                centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
                fns = (basis[p], basis[q], basis[r], basis[s])
                perms = tuple(_eri_permutations(p, q, r, s))
                cache = {}

                def eval_orders(orders):
                    key = tuple(orders)
                    if key not in cache:
                        cache[key] = _contracted_eri_deriv(*fns, *orders)
                    return cache[key]

                for atom in sorted(set(centers)):
                    for axis in range(3):
                        value = 0.0
                        for slot, center_atom in enumerate(centers):
                            if center_atom != atom:
                                continue
                            orders = [(0, 0, 0)] * 4
                            orders[slot] = _axis_order(axis)
                            value += eval_orders(orders)
                        for idx in perms:
                            out[(atom, axis) + idx] = value
        return _transform_eri(out, transform)

    out = np.zeros((natm, 3, natm, 3, nao_cart, nao_cart, nao_cart, nao_cart), dtype=float)
    for pair_pq, (p, q) in enumerate(ao_pairs):
        for pair_rs, (r, s) in enumerate(ao_pairs[: pair_pq + 1]):
            centers = (atom_ids[p], atom_ids[q], atom_ids[r], atom_ids[s])
            fns = (basis[p], basis[q], basis[r], basis[s])
            perms = tuple(_eri_permutations(p, q, r, s))
            perturbations = [(atom, axis) for atom in sorted(set(centers)) for axis in range(3)]
            cache = {}

            def eval_orders(orders):
                key = tuple(orders)
                if key not in cache:
                    cache[key] = _contracted_eri_deriv(*fns, *orders)
                return cache[key]

            for pert_a, (atom_a, axis_a) in enumerate(perturbations):
                for atom_b, axis_b in perturbations[: pert_a + 1]:
                    value = 0.0
                    for slot_a, center_a in enumerate(centers):
                        if center_a != atom_a:
                            continue
                        for slot_b, center_b in enumerate(centers):
                            if center_b != atom_b:
                                continue
                            orders = [(0, 0, 0)] * 4
                            if slot_a == slot_b:
                                orders[slot_a] = _second_order(axis_a, axis_b)
                            else:
                                orders[slot_a] = _axis_order(axis_a)
                                orders[slot_b] = _axis_order(axis_b)
                            value += eval_orders(orders)
                    for idx in perms:
                        out[(atom_a, axis_a, atom_b, axis_b) + idx] = value
                        if atom_a != atom_b or axis_a != axis_b:
                            out[(atom_b, axis_b, atom_a, axis_a) + idx] = value
    return _transform_eri(out, transform)
