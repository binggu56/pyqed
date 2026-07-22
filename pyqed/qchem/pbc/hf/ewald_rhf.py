#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

from pyqed.qchem.basis import S, T
from pyqed.qchem.fourier import (
    AOBlockPairFTPlan,
    gaussian_pair_ft_batch,
    has_periodic_pair_ft_backend,
)
from pyqed.qchem.pbc.ewald import (
    ewald_nuclear_repulsion,
    ewald_nuclear_repulsion_1d_inf_vacuum,
    inf_vacuum_1d_gv_weights,
    reciprocal_vectors,
    short_range_eri_s,
    short_range_point_charge_s,
)


def _shifted_gaussian(fn, shift):
    shifted = object.__new__(fn.__class__)
    shifted.__dict__ = dict(fn.__dict__)
    shifted.origin = np.asarray(fn.origin, dtype=float) + np.asarray(shift, dtype=float)
    return shifted


def _symmetrize(mat):
    return 0.5 * (mat + mat.conj().T)


def _gaussian_pair_ft_decay_bound(a, b, shift):
    a_origin = np.asarray(a.origin, dtype=float)
    b_origin = np.asarray(b.origin, dtype=float) + np.asarray(shift, dtype=float)
    distance = float(np.linalg.norm(a_origin - b_origin))
    angular = int(sum(a.shell) + sum(b.shell))
    bound = 0.0
    for ia, wa in enumerate(a.prim_weights):
        alpha = float(a.exps[ia])
        for ib, wb in enumerate(b.prim_weights):
            beta = float(b.exps[ib])
            p = alpha + beta
            q = alpha * beta / p
            moment = (distance + 1.0 / math.sqrt(p) + 1.0) ** angular
            bound += (
                abs(float(wa) * float(wb))
                * (math.pi / p) ** 1.5
                * math.exp(-q * distance * distance)
                * moment
            )
    return bound


def _normalize_kpts(kpts):
    if kpts is None:
        return np.zeros((1, 3), dtype=float)
    arr = np.asarray(kpts, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.shape[-1] != 3:
        raise ValueError("kpts must have shape (nk, 3) or (3,).")
    return arr


def _matrix_norm(mats_a, mats_b):
    return float(sum(np.linalg.norm(a - b) for a, b in zip(mats_a, mats_b)))


def _infer_kmesh(lattice, kpts):
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if len(kpts) == 1:
        return np.ones(3, dtype=int)
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T
    scaled = kpts @ np.linalg.inv(recip)
    mesh = []
    tol = max(1.0e-5, 10.0 * np.finfo(float).eps)
    for axis_values in scaled.T:
        unique = []
        for value in np.sort(axis_values):
            if not unique or abs(value - unique[-1]) > tol:
                unique.append(float(value))
        mesh.append(max(1, len(unique)))
    return np.asarray(mesh, dtype=int)


def _scaled_from_cartesian(kpts, lattice):
    recip = 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T
    return np.asarray(kpts, dtype=float) @ np.linalg.inv(recip)


def _wrap_scaled(values):
    return ((np.asarray(values, dtype=float) + 0.5) % 1.0) - 0.5


def _unique_sorted(values, tol=1.0e-8):
    unique = []
    for value in np.sort(np.asarray(values, dtype=float)):
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(float(value))
    return np.asarray(unique, dtype=float)


def _interpolate_periodic_1d_bands(mesh_kpts, mesh_energies, target_kpts, lattice, tol=1.0e-8):
    mesh_kpts = _normalize_kpts(mesh_kpts)
    target_kpts = _normalize_kpts(target_kpts)
    mesh_energies = np.asarray(mesh_energies, dtype=float)
    if mesh_energies.ndim != 2 or mesh_energies.shape[0] != len(mesh_kpts):
        raise ValueError("mesh_energies must have shape (nk, nband).")
    if len(mesh_kpts) == 1:
        return np.repeat(mesh_energies, len(target_kpts), axis=0)

    mesh_scaled = _wrap_scaled(_scaled_from_cartesian(mesh_kpts, lattice))
    target_scaled = _wrap_scaled(_scaled_from_cartesian(target_kpts, lattice))
    unique_by_axis = [_unique_sorted(mesh_scaled[:, axis], tol=tol) for axis in range(3)]
    varying = [axis for axis, values in enumerate(unique_by_axis) if len(values) > 1]
    if len(varying) != 1:
        raise NotImplementedError(
            "mesh_interpolate currently supports one-dimensional k meshes only."
        )

    axis = varying[0]
    fixed_axes = [other for other in range(3) if other != axis]
    for fixed_axis in fixed_axes:
        fixed_values = unique_by_axis[fixed_axis]
        if len(fixed_values) != 1:
            raise NotImplementedError(
                "mesh_interpolate currently supports one-dimensional k meshes only."
            )
        if np.max(np.abs(target_scaled[:, fixed_axis] - fixed_values[0])) > tol:
            raise ValueError("target k-points leave the one-dimensional SCF k mesh line.")

    order = np.argsort(mesh_scaled[:, axis])
    x_mesh = mesh_scaled[order, axis]
    y_mesh = mesh_energies[order]
    x_ext = np.concatenate([x_mesh - 1.0, x_mesh, x_mesh + 1.0])
    y_ext = np.concatenate([y_mesh, y_mesh, y_mesh], axis=0)
    out = np.empty((len(target_kpts), mesh_energies.shape[1]), dtype=float)
    x_target = target_scaled[:, axis]
    for band in range(mesh_energies.shape[1]):
        out[:, band] = np.interp(x_target, x_ext, y_ext[:, band])
    return out


class EwaldRHF:
    """
    Dense native Ewald RHF/KRHF for small Cartesian Gaussian PBC cells.

    The implementation is correctness-first and intentionally explicit.  It
    supports closed-shell neutral 1D chains and 3D cells with modest real- and
    reciprocal-space cutoffs.  K-point calculations use k-dependent one-body
    matrices and the cell-averaged density for the dense Coulomb/exchange step.
    """

    def __init__(
        self,
        cell,
        kpts=None,
        nk=None,
        eta=0.5,
        real_cut=3,
        recip_cut=6,
        mesh=None,
        damping=0.5,
        nuclear_background=True,
        eri_screen_tol=1.0e-12,
        jk_builder="ewald",
        pair_cut=None,
        pair_ft_screen_tol=1.0e-14,
        occupation_tol=1.0e-10,
    ):
        self.cell = cell
        if kpts is not None and nk is not None:
            raise ValueError("Specify either kpts or nk, not both.")
        if nk is not None:
            if not self.cell.built:
                self.cell.build()
            kpts = self.cell.make_kpts(nk)
        self.kpts = _normalize_kpts(kpts)
        self.eta = float(eta)
        self.real_cut = int(real_cut)
        self.recip_cut = int(recip_cut)
        self.mesh = None if mesh is None else tuple(int(x) for x in mesh)
        self.damping = float(damping)
        self.nuclear_background = bool(nuclear_background)
        self.eri_screen_tol = float(eri_screen_tol)
        self.jk_builder = str(jk_builder).lower()
        if self.jk_builder not in ("ewald", "reciprocal"):
            raise ValueError("jk_builder must be 'ewald' or 'reciprocal'.")
        self.pair_cut = self.real_cut if pair_cut is None else int(pair_cut)
        self.pair_ft_screen_tol = float(pair_ft_screen_tol)
        self.occupation_tol = float(occupation_tol)

        self.e_tot = None
        self.e_elec = None
        self.e_nuc = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.dm = None
        self.fock = None
        self.converged = False
        self.nkpts = int(len(self.kpts))

        self.overlap = None
        self.hcore = None
        self.eri = None
        self.madelung = None
        self._basis = None
        self._shift_keys = None
        self._shift_vectors = None
        self._shifted_basis = None
        self._pair_shift_keys = None
        self._pair_shift_vectors = None
        self._pair_shifted_basis = None
        self._pair_ft_terms = None
        self._pair_ft_block_plan = None
        self._pair_ft_origins = None
        self._pair_ft_shift_array = None
        self._pair_ft_right_origins_batch = None
        self._pair_ft_image_pair_mask = None
        self._pair_ft_pair_image_starts = None
        self._pair_ft_pair_image_indices = None
        self._pair_ft_primitive_terms = None
        self._s_r = None
        self._t_r = None
        self._vne_sr_r = None
        self._overlap_k = None
        self._hcore_k = None
        self._exchange_eri_k = None
        self._pair_ft_cache = None
        self._pair_ft_batch_cache = None
        self._g_weight_cache = None
        self._one_body_cache = None
        self._band_veff_cache = None

    def _vector_key(self, vec):
        arr = np.asarray(vec, dtype=float)
        arr = np.where(np.abs(arr) < 1e-14, 0.0, arr)
        return tuple(np.round(arr, 12))

    def _validate(self):
        if not self.cell.built:
            self.cell.build()
        if self.cell.dimension not in (1, 3):
            raise NotImplementedError("EwaldRHF supports dimension=1 and dimension=3 cells.")
        if self.cell.charge != 0:
            raise NotImplementedError("EwaldRHF v1 requires neutral periodic cells.")
        if self.cell.spin != 0 or int(self.cell.nelectron) % 2:
            raise NotImplementedError("EwaldRHF v1 supports closed-shell even-electron cells only.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive for method='ewald'.")
        if self.real_cut < 0 or self.recip_cut < 0 or self.pair_cut < 0:
            raise ValueError("real_cut, recip_cut, and pair_cut must be non-negative.")
        if self.pair_ft_screen_tol < 0.0:
            raise ValueError("pair_ft_screen_tol must be non-negative.")
        if self.occupation_tol < 0.0:
            raise ValueError("occupation_tol must be non-negative.")

    def _use_inf_vacuum_1d(self):
        return (
            int(self.cell.dimension) == 1
            and str(getattr(self.cell, "low_dim_ft_type", "")).lower() == "inf_vacuum"
        )

    def _reciprocal_mesh(self):
        if self.mesh is not None:
            return self.mesh
        return (31, 38, 38) if self._use_inf_vacuum_1d() else None

    def _periodic_setup(self):
        self._basis = tuple(self.cell.unit_molecule._bas)
        if len(self._basis) != int(self.cell.nao):
            raise NotImplementedError(
                "EwaldRHF currently supports Cartesian AO builds only. "
                "Use integral_options={'coord_type': 'cartesian'} for p/d basis tests."
            )

        self._shift_keys = list(self.cell.image_keys(self.real_cut))
        self._shift_vectors = {
            key: self.cell.translation_vector(key)
            for key in self._shift_keys
        }
        self._shifted_basis = {
            key: [_shifted_gaussian(fn, self._shift_vectors[key]) for fn in self._basis]
            for key in self._shift_keys
        }
        self._pair_shift_keys = list(self.cell.image_keys(self.pair_cut))
        self._pair_shift_vectors = {
            key: self.cell.translation_vector(key)
            for key in self._pair_shift_keys
        }
        self._pair_shifted_basis = {
            key: [_shifted_gaussian(fn, self._pair_shift_vectors[key]) for fn in self._basis]
            for key in self._pair_shift_keys
        }
        self._pair_ft_block_plan = None
        self._pair_ft_origins = np.ascontiguousarray(
            [np.asarray(fn.origin, dtype=float) for fn in self._basis]
        )
        self._pair_ft_shift_array = np.ascontiguousarray(
            [self._pair_shift_vectors[key] for key in self._pair_shift_keys],
            dtype=float,
        )
        self._pair_ft_right_origins_batch = np.ascontiguousarray(
            self._pair_ft_origins[None, :, :] + self._pair_ft_shift_array[:, None, :]
        )
        if has_periodic_pair_ft_backend():
            self._pair_ft_block_plan = AOBlockPairFTPlan(self._basis, self._basis)
        self._build_pair_ft_terms()
        self._pair_ft_primitive_terms = None
        if self._pair_ft_block_plan is not None:
            self._pair_ft_primitive_terms = self._pair_ft_block_plan.periodic_primitive_terms(
                self._pair_ft_origins,
                self._pair_ft_right_origins_batch,
                image_pair_mask=self._pair_ft_image_pair_mask,
            )
        self._pair_ft_cache = {}
        self._pair_ft_batch_cache = {}
        self._g_weight_cache = {}
        self._one_body_cache = {}
        self._band_veff_cache = {}

    def _build_pair_ft_terms(self):
        terms = []
        nao = len(self._basis)
        mask = np.zeros((len(self._pair_shift_keys), nao * nao), dtype=np.bool_)
        for image, key in enumerate(self._pair_shift_keys):
            shift = self._pair_shift_vectors[key]
            shifted_basis = self._pair_shifted_basis[key]
            for p, bp in enumerate(self._basis):
                for q, bq in enumerate(shifted_basis):
                    keep = self.pair_ft_screen_tol == 0.0 or (
                        _gaussian_pair_ft_decay_bound(bp, self._basis[q], shift)
                        > self.pair_ft_screen_tol
                    )
                    if not keep:
                        continue
                    terms.append((shift, p, q, bp, bq))
                    mask[image, p * nao + q] = True
        self._pair_ft_terms = terms
        self._pair_ft_image_pair_mask = np.ascontiguousarray(mask)
        counts = np.count_nonzero(mask, axis=0).astype(np.int64)
        starts = np.empty(nao * nao + 1, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])
        indices = np.empty(int(starts[-1]), dtype=np.int64)
        for pair_idx in range(nao * nao):
            start = int(starts[pair_idx])
            stop = int(starts[pair_idx + 1])
            indices[start:stop] = np.nonzero(mask[:, pair_idx])[0]
        self._pair_ft_pair_image_starts = np.ascontiguousarray(starts)
        self._pair_ft_pair_image_indices = np.ascontiguousarray(indices)

    def _build_one_body_blocks(self):
        charges = np.asarray(self.cell.unit_molecule.atom_charges(), dtype=float)
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        nao = len(self._basis)
        self._s_r = {}
        self._t_r = {}
        self._vne_sr_r = {}

        for key in self._shift_keys:
            shifted_basis = self._shifted_basis[key]
            s_block = np.zeros((nao, nao), dtype=float)
            t_block = np.zeros((nao, nao), dtype=float)
            vne_block = np.zeros((nao, nao), dtype=float)

            for p, bp in enumerate(self._basis):
                for q, bq in enumerate(shifted_basis):
                    s_block[p, q] = S(bp, bq)
                    t_block[p, q] = T(bp, bq)
                    value = 0.0
                    for nuc_key in self._shift_keys:
                        nuc_shift = self._shift_vectors[nuc_key]
                        for charge, coord in zip(charges, coords):
                            value -= charge * short_range_point_charge_s(
                                bp,
                                bq,
                                coord + nuc_shift,
                                self.eta,
                            )
                    vne_block[p, q] = value

            self._s_r[key] = s_block
            self._t_r[key] = t_block
            self._vne_sr_r[key] = vne_block

    def _fourier_sum(self, blocks, kvec):
        acc = np.zeros_like(next(iter(blocks.values())), dtype=np.complex128)
        for key, block in blocks.items():
            phase = np.exp(1j * np.dot(kvec, self._shift_vectors[key]))
            acc += phase * block
        return _symmetrize(acc)

    def _periodic_pair_ft(self, gvec, kvec):
        cache_key = (self._vector_key(gvec), self._vector_key(kvec))
        if self._pair_ft_cache is not None and cache_key in self._pair_ft_cache:
            return self._pair_ft_cache[cache_key]

        pair = self._periodic_pair_ft_batch(
            np.asarray(gvec, dtype=float).reshape(1, 3), kvec
        )[0]
        if self._pair_ft_cache is not None:
            self._pair_ft_cache[cache_key] = pair
        return pair

    def _periodic_pair_ft_batch_direct(self, gvecs, kvec):
        nao = len(self._basis)
        out = np.zeros((len(gvecs), nao, nao), dtype=np.complex128)
        kvec = np.asarray(kvec, dtype=float)
        for shift, p, q, bp, bq in self._pair_ft_terms:
            phase = np.exp(1j * np.dot(kvec, shift))
            out[:, p, q] += phase * gaussian_pair_ft_batch(bp, bq, gvecs)
        return out

    def _periodic_pair_ft_batch(self, gvecs, kvec):
        gvecs = np.asarray(gvecs, dtype=float)
        if gvecs.ndim != 2 or gvecs.shape[1] != 3:
            raise ValueError("gvecs must have shape (ng, 3).")
        kkey = self._vector_key(kvec)
        rounded_gvecs = np.where(np.abs(gvecs) < 1e-14, 0.0, gvecs)
        rounded_gvecs = np.round(rounded_gvecs, 12)
        cache_key = (kkey, rounded_gvecs.shape, rounded_gvecs.tobytes())
        if self._pair_ft_batch_cache is not None and cache_key in self._pair_ft_batch_cache:
            return self._pair_ft_batch_cache[cache_key]

        kvec = np.asarray(kvec, dtype=float)
        if self._pair_ft_block_plan is None:
            out = self._periodic_pair_ft_batch_direct(gvecs, kvec)
        else:
            phases = np.exp(1j * (self._pair_ft_shift_array @ kvec))
            out = self._pair_ft_block_plan.periodic_sum(
                gvecs,
                left_origins=self._pair_ft_origins,
                right_origins_batch=self._pair_ft_right_origins_batch,
                phases=phases,
                image_pair_mask=self._pair_ft_image_pair_mask,
                pair_image_starts=self._pair_ft_pair_image_starts,
                pair_image_indices=self._pair_ft_pair_image_indices,
                primitive_terms=self._pair_ft_primitive_terms,
                compiled=True,
            )

        if self._pair_ft_batch_cache is not None:
            self._pair_ft_batch_cache[cache_key] = out
        return out

    def _pair_overlap_at_k(self, kvec):
        return _symmetrize(self._periodic_pair_ft(np.zeros(3), kvec))

    def _reciprocal_g_weights(self, include_zero=False):
        cache_key = bool(include_zero)
        if self._g_weight_cache is not None and cache_key in self._g_weight_cache:
            return self._g_weight_cache[cache_key]

        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        if self._use_inf_vacuum_1d():
            gvecs, weights = inf_vacuum_1d_gv_weights(lattice, self._reciprocal_mesh())
            mask = np.ones(len(gvecs), dtype=bool)
            if not include_zero:
                mask = np.einsum("gi,gi->g", gvecs, gvecs) > 1e-16
            values = list(zip(gvecs[mask], weights[mask]))
            if self._g_weight_cache is not None:
                self._g_weight_cache[cache_key] = values
            return values

        volume = abs(float(np.linalg.det(lattice)))
        values = [
            (gvec, 1.0 / volume)
            for _h, _k, _l, gvec in reciprocal_vectors(
                lattice,
                self.recip_cut,
                include_zero=include_zero,
            )
        ]
        if self._g_weight_cache is not None:
            self._g_weight_cache[cache_key] = values
        return values

    def _reciprocal_nuclear_attraction(self, kvec):
        charges = np.asarray(self.cell.unit_molecule.atom_charges(), dtype=float)
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        values = self._reciprocal_g_weights()
        if not values:
            return mat
        gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
        weights = np.asarray([weight for _gvec, weight in values], dtype=float)
        g2 = np.einsum("gi,gi->g", gvecs, gvecs)
        mask = g2 > 0.0
        if not np.any(mask):
            return mat
        gvecs = gvecs[mask]
        weights = weights[mask]
        g2 = g2[mask]
        damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
        rho_nuc = np.einsum("a,ag->g", charges, np.exp(-1j * (coords @ gvecs.T)))
        pair_plus_g = self._periodic_pair_ft_batch(-gvecs, kvec)
        coeff = -(4.0 * np.pi) * weights * damping * rho_nuc / g2
        mat += np.einsum("g,gpq->pq", coeff, pair_plus_g, optimize=True)
        return _symmetrize(mat)

    def _nuclear_background_hcore(self, overlap):
        if self._use_inf_vacuum_1d() or not self.nuclear_background:
            return np.zeros_like(overlap, dtype=np.complex128)
        volume = abs(float(np.linalg.det(np.asarray(self.cell.lattice_vectors, dtype=float))))
        q_nuc = float(np.sum(self.cell.unit_molecule.atom_charges()))
        coeff = np.pi * q_nuc / (self.eta * self.eta * volume)
        return coeff * overlap

    def _coulomb_background_eri(self):
        if self._use_inf_vacuum_1d() or not self.nuclear_background:
            return 0.0
        volume = abs(float(np.linalg.det(np.asarray(self.cell.lattice_vectors, dtype=float))))
        coeff = np.pi / (self.eta * self.eta * volume)
        overlap = self._fourier_sum(self._s_r, np.zeros(3))
        return -coeff * np.einsum("pq,rs->pqrs", overlap, overlap, optimize=True).real

    def _periodic_short_range_eri(self):
        nao = len(self._basis)
        eri = np.zeros((nao, nao, nao, nao), dtype=float)
        pair_overlap = {}
        for key in self._shift_keys:
            shifted_basis = self._shifted_basis[key]
            for p, bp in enumerate(self._basis):
                for q, bq in enumerate(shifted_basis):
                    pair_overlap[(key, p, q)] = abs(S(bp, bq))
        image_pair_overlap = {}
        for r_key in self._shift_keys:
            r_basis = self._shifted_basis[r_key]
            for s_key in self._shift_keys:
                s_basis = self._shifted_basis[s_key]
                for r, br in enumerate(r_basis):
                    for s, bs in enumerate(s_basis):
                        image_pair_overlap[(r_key, s_key, r, s)] = abs(S(br, bs))

        for q_key in self._shift_keys:
            for r_key in self._shift_keys:
                for s_key in self._shift_keys:
                    q_basis = self._shifted_basis[q_key]
                    r_basis = self._shifted_basis[r_key]
                    s_basis = self._shifted_basis[s_key]
                    for p, bp in enumerate(self._basis):
                        for q, bq in enumerate(q_basis):
                            pq_bound = pair_overlap[(q_key, p, q)]
                            for r, br in enumerate(r_basis):
                                for s, bs in enumerate(s_basis):
                                    rs_bound = image_pair_overlap[(r_key, s_key, r, s)]
                                    if pq_bound * rs_bound < self.eri_screen_tol:
                                        continue
                                    eri[p, q, r, s] += short_range_eri_s(
                                        bp,
                                        bq,
                                        br,
                                        bs,
                                        self.eta,
                                    )
        return eri

    def _periodic_reciprocal_eri(self):
        nao = len(self._basis)
        eri = np.zeros((nao, nao, nao, nao), dtype=np.complex128)
        gamma = np.zeros(3)
        for gvec, weight in self._reciprocal_g_weights():
            g2 = float(np.dot(gvec, gvec))
            if g2 <= 0.0:
                continue
            damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
            pair_g = self._periodic_pair_ft(gvec, gamma)
            pair_minus_g = self._periodic_pair_ft(-gvec, gamma)
            eri += (
                (4.0 * np.pi)
                * weight
                * damping
                / g2
                * np.einsum("pq,rs->pqrs", pair_g, pair_minus_g, optimize=True)
            )
        eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2).conj())
        return np.asarray(eri.real, dtype=float)

    def _build_exchange_eri_k(self):
        if self.nkpts == 1 or self._use_inf_vacuum_1d():
            self._exchange_eri_k = None
            return

        nao = len(self._basis)
        xeri = np.zeros((self.nkpts, self.nkpts, nao, nao, nao, nao), dtype=np.complex128)
        for i, ki in enumerate(self.kpts):
            for j, kj in enumerate(self.kpts):
                xeri[i, j] = self._exchange_eri_block(ki, kj)

        self._exchange_eri_k = xeri

    def _exchange_eri_block(self, ki, kj):
        ki = np.asarray(ki, dtype=float)
        kj = np.asarray(kj, dtype=float)
        nao = len(self._basis)
        block = np.zeros((nao, nao, nao, nao), dtype=np.complex128)
        pair_overlap = {}
        for key in self._shift_keys:
            shifted_basis = self._shifted_basis[key]
            for p, bp in enumerate(self._basis):
                for q, bq in enumerate(shifted_basis):
                    pair_overlap[(key, p, q)] = abs(S(bp, bq))

        for r_key in self._shift_keys:
            r_shift = self._shift_vectors[r_key]
            r_basis = self._shifted_basis[r_key]
            phase_r = np.exp(1j * np.dot(kj, r_shift))
            for s_key in self._shift_keys:
                s_shift = self._shift_vectors[s_key]
                s_basis = self._shifted_basis[s_key]
                phase_pair_origin = np.exp(1j * np.dot(ki - kj, s_shift))
                for t_key in self._shift_keys:
                    t_shift = self._shift_vectors[t_key]
                    q_basis = [
                        _shifted_gaussian(fn, s_shift + t_shift)
                        for fn in self._basis
                    ]
                    phase = (
                        phase_r
                        * phase_pair_origin
                        * np.exp(1j * np.dot(ki, t_shift))
                    )
                    for p, bp in enumerate(self._basis):
                        for r, br in enumerate(r_basis):
                            pr_bound = pair_overlap[(r_key, p, r)]
                            for s, bs in enumerate(s_basis):
                                for q, bq in enumerate(q_basis):
                                    sq_bound = pair_overlap[(t_key, s, q)]
                                    if pr_bound * sq_bound < self.eri_screen_tol:
                                        continue
                                    block[p, q, r, s] += phase * short_range_eri_s(
                                        bp,
                                        br,
                                        bs,
                                        bq,
                                        self.eta,
                                    )

        qvec = kj - ki
        for gvec, weight in self._reciprocal_g_weights(include_zero=True):
            gq = gvec + qvec
            g2 = float(np.dot(gq, gq))
            if g2 <= 1e-16:
                continue
            damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
            pair_pr = self._periodic_pair_ft(gq, kj)
            pair_sq = self._periodic_pair_ft(-gq, ki)
            block += (
                (4.0 * np.pi)
                * weight
                * damping
                / g2
                * np.einsum("pr,sq->pqrs", pair_pr, pair_sq, optimize=True)
            )

        if (
            self.nuclear_background
            and not self._use_inf_vacuum_1d()
            and float(np.dot(qvec, qvec)) <= 1e-16
        ):
            volume = abs(float(np.linalg.det(np.asarray(self.cell.lattice_vectors, dtype=float))))
            coeff = np.pi / (self.eta * self.eta * volume)
            overlap_pr = self._periodic_pair_ft(np.zeros(3), kj)
            overlap_sq = self._periodic_pair_ft(np.zeros(3), ki)
            block -= coeff * np.einsum(
                "pr,sq->pqrs",
                overlap_pr,
                overlap_sq,
                optimize=True,
            )

        return block

    def _madelung(self):
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        if self._use_inf_vacuum_1d():
            energy = ewald_nuclear_repulsion_1d_inf_vacuum(
                np.asarray([1.0]),
                np.zeros((1, 3), dtype=float),
                lattice,
                eta=self.eta,
                real_cut=self.real_cut,
                mesh=self._reciprocal_mesh(),
            )
            return -2.0 * energy

        kmesh = _infer_kmesh(lattice, self.kpts)
        probe_lattice = np.einsum("xi,x->xi", lattice, kmesh)
        energy = ewald_nuclear_repulsion(
            np.asarray([1.0]),
            np.zeros((1, 3), dtype=float),
            probe_lattice,
            eta=self.eta,
            real_cut=self.real_cut,
            recip_cut=self.recip_cut,
            neutralizing_background=True,
        )
        return -2.0 * energy

    def _build_integrals(self):
        self._validate()
        self._periodic_setup()
        self._build_one_body_blocks()

        overlap_k = []
        hcore_k = []
        for kvec in self.kpts:
            overlap, hcore = self._one_body_at_k(kvec)
            overlap_k.append(overlap)
            hcore_k.append(hcore)

        if self.jk_builder == "reciprocal":
            self.eri = None
            self._exchange_eri_k = None
        else:
            eri = self._periodic_short_range_eri()
            eri += self._periodic_reciprocal_eri()
            eri += self._coulomb_background_eri()
            self.eri = np.asarray(eri, dtype=float)
            self._build_exchange_eri_k()
        self._overlap_k = overlap_k
        self._hcore_k = hcore_k
        self.overlap = overlap_k[0] if self.nkpts == 1 else overlap_k
        self.hcore = hcore_k[0] if self.nkpts == 1 else hcore_k

        if self._use_inf_vacuum_1d():
            self.e_nuc = float(
                ewald_nuclear_repulsion_1d_inf_vacuum(
                    self.cell.unit_molecule.atom_charges(),
                    self.cell._atom_coords,
                    self.cell.lattice_vectors,
                    eta=self.eta,
                    real_cut=self.real_cut,
                    mesh=self._reciprocal_mesh(),
                )
            )
        else:
            self.e_nuc = float(
                self.cell.ewald_nuclear_repulsion(
                    eta=self.eta,
                    real_cut=self.real_cut,
                    recip_cut=self.recip_cut,
                    neutralizing_background=self.nuclear_background,
                )
            )
        self.madelung = self._madelung()

    def _one_body_at_k(self, kvec):
        kvec = np.asarray(kvec, dtype=float)
        cache_key = self._vector_key(kvec)
        if self._one_body_cache is not None and cache_key in self._one_body_cache:
            return self._one_body_cache[cache_key]
        overlap = self._fourier_sum(self._s_r, kvec)
        kinetic = self._fourier_sum(self._t_r, kvec)
        vne = self._fourier_sum(self._vne_sr_r, kvec)
        vne += self._reciprocal_nuclear_attraction(kvec)
        vne += self._nuclear_background_hcore(overlap)
        value = (overlap, _symmetrize(kinetic + vne))
        if self._one_body_cache is not None:
            self._one_body_cache[cache_key] = value
        return value

    def _occupations_from_energies(self, mo_energy_k):
        mo_energy_k = [np.asarray(energy, dtype=float) for energy in mo_energy_k]
        nk = len(mo_energy_k)
        total_electrons = int(self.cell.nelectron) * nk
        if total_electrons % 2:
            raise NotImplementedError("KRHF occupations require an even total electron count.")
        total_pairs = total_electrons // 2
        total_orbitals = sum(len(energy) for energy in mo_energy_k)
        if total_pairs > total_orbitals:
            raise ValueError("Not enough periodic orbitals for closed-shell occupation.")

        mo_occ = [np.zeros_like(energy, dtype=float) for energy in mo_energy_k]
        if total_pairs == 0:
            return mo_occ
        if total_pairs == total_orbitals:
            return [np.full_like(energy, 2.0, dtype=float) for energy in mo_energy_k]

        flat = []
        for ik, energy in enumerate(mo_energy_k):
            for ib, value in enumerate(energy):
                flat.append((float(value), ik, ib))
        flat.sort(key=lambda item: item[0])

        fermi = flat[total_pairs - 1][0]
        tol = self.occupation_tol
        full = [item for item in flat if item[0] < fermi - tol]
        frontier = [item for item in flat if abs(item[0] - fermi) <= tol]
        remaining_pairs = total_pairs - len(full)

        for _energy, ik, ib in full:
            mo_occ[ik][ib] = 2.0
        if frontier:
            frontier_occ = 2.0 * remaining_pairs / len(frontier)
            for _energy, ik, ib in frontier:
                mo_occ[ik][ib] = frontier_occ
        return mo_occ

    @staticmethod
    def _density_from_mo_occ(mo_coeff, mo_occ):
        coeff = np.asarray(mo_coeff, dtype=np.complex128)
        occ = np.asarray(mo_occ, dtype=float)
        mask = occ > 1.0e-12
        if not np.any(mask):
            return np.zeros((coeff.shape[0], coeff.shape[0]), dtype=np.complex128)
        cocc = coeff[:, mask]
        dm = (cocc * occ[mask]) @ cocc.conj().T
        return _symmetrize(dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_energy is None:
            raise ValueError("MO energies are not available.")

        energy_arr = np.asarray(mo_energy, dtype=object)
        single_k = energy_arr.ndim == 1 and not isinstance(mo_energy, (list, tuple))
        if single_k:
            occ_k = self._occupations_from_energies([np.asarray(mo_energy, dtype=float)])
            return occ_k[0]

        occ_k = self._occupations_from_energies(
            [np.asarray(energy, dtype=float) for energy in mo_energy]
        )
        return occ_k[0] if self.nkpts == 1 and len(occ_k) == 1 else occ_k

    def _solve_fock(self, fock_k, overlap_k):
        mo_energy = []
        mo_coeff = []
        for fock, overlap in zip(fock_k, overlap_k):
            evals, evecs = eigh(_symmetrize(fock), _symmetrize(overlap))
            mo_energy.append(evals)
            mo_coeff.append(evecs)
        mo_occ = self._occupations_from_energies(mo_energy)
        dm_k = [
            self._density_from_mo_occ(coeff, occ)
            for coeff, occ in zip(mo_coeff, mo_occ)
        ]
        return mo_energy, mo_coeff, mo_occ, dm_k

    def _average_density(self, dm_k):
        return sum(dm_k) / len(dm_k)

    def get_veff(self, dm, overlap=None):
        dm = np.asarray(dm, dtype=np.complex128)
        if self.eri is None:
            vj = self._reciprocal_j(dm)
            vk = self._reciprocal_k_gamma(dm, overlap)
            return _symmetrize(vj - 0.5 * vk)
        vj = np.einsum("pqrs,rs->pq", self.eri, dm, optimize=True)
        vk = np.einsum("prqs,rs->pq", self.eri, dm, optimize=True)
        if self.madelung is not None and overlap is not None:
            vk = vk + self.madelung * (overlap @ dm @ overlap)
        return _symmetrize(vj - 0.5 * vk)

    def _reciprocal_j(self, dm):
        dm = np.asarray(dm, dtype=np.complex128)
        return self._reciprocal_j_at_k([dm], np.zeros(3), [np.zeros(3)])

    def _reciprocal_j_at_k(self, dm_k, kvec, source_kpts=None):
        if source_kpts is None:
            source_kpts = self.kpts
        dm_k = [np.asarray(dm, dtype=np.complex128) for dm in dm_k]
        source_kpts = [np.asarray(k, dtype=float) for k in source_kpts]
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        kvec = np.asarray(kvec, dtype=float)
        values = self._reciprocal_g_weights(include_zero=False)
        if not values:
            return mat
        gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
        weights = np.asarray([weight for _gvec, weight in values], dtype=float)
        g2 = np.einsum("gi,gi->g", gvecs, gvecs)
        mask = g2 > 1e-16
        if not np.any(mask):
            return mat
        gvecs = gvecs[mask]
        weights = weights[mask]
        g2 = g2[mask]
        rho = np.zeros(len(gvecs), dtype=np.complex128)
        for dm, source_kvec in zip(dm_k, source_kpts):
            pair_minus = self._periodic_pair_ft_batch(gvecs, source_kvec)
            rho += np.einsum("gpq,qp->g", pair_minus, dm, optimize=True)
        rho /= len(dm_k)
        pair_plus = self._periodic_pair_ft_batch(-gvecs, kvec)
        coeff = (4.0 * np.pi) * weights * rho / g2
        mat += np.einsum("g,gpq->pq", coeff, pair_plus, optimize=True)
        return _symmetrize(mat)

    def _reciprocal_k_gamma(self, dm, overlap):
        dm = np.asarray(dm, dtype=np.complex128)
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        gamma = np.zeros(3)
        values = self._reciprocal_g_weights(include_zero=False)
        if values:
            gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
            weights = np.asarray([weight for _gvec, weight in values], dtype=float)
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            mask = g2 > 1e-16
            if np.any(mask):
                gvecs = gvecs[mask]
                weights = weights[mask]
                g2 = g2[mask]
                pair_pr = self._periodic_pair_ft_batch(gvecs, gamma)
                pair_sq = self._periodic_pair_ft_batch(-gvecs, gamma)
                coeff = (4.0 * np.pi) * weights / g2
                mat += np.einsum(
                    "g,gpr,rs,gsq->pq",
                    coeff,
                    pair_pr,
                    dm,
                    pair_sq,
                    optimize=True,
                )
        if self.madelung is not None and overlap is not None:
            pair_overlap = self._pair_overlap_at_k(gamma)
            mat = mat + self.madelung * (pair_overlap @ dm @ pair_overlap)
        return _symmetrize(mat)

    def _reciprocal_k_at_k(self, dm_k, kvec, overlap):
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        values = self._reciprocal_g_weights(include_zero=True)
        if not values:
            return mat
        base_gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
        base_weights = np.asarray([weight for _gvec, weight in values], dtype=float)
        for kj, dm in zip(self.kpts, dm_k):
            qvec = kj - kvec
            gq = base_gvecs + qvec
            g2 = np.einsum("gi,gi->g", gq, gq)
            mask = g2 > 1e-16
            if not np.any(mask):
                continue
            gq = gq[mask]
            weights = base_weights[mask]
            g2 = g2[mask]
            pair_pr = self._periodic_pair_ft_batch(gq, kj)
            pair_sq = self._periodic_pair_ft_batch(-gq, kvec)
            coeff = (4.0 * np.pi) * weights / g2
            mat += np.einsum(
                "g,gpr,rs,gsq->pq",
                coeff,
                pair_pr,
                dm,
                pair_sq,
                optimize=True,
            )
        mat /= self.nkpts

        match = self._matching_k_index(kvec)
        if self.madelung is not None and match is not None:
            pair_overlap = self._pair_overlap_at_k(kvec)
            mat = mat + self.madelung * (pair_overlap @ dm_k[match] @ pair_overlap)
        return _symmetrize(mat)

    def _get_veff_k(self, dm_k, k_index):
        dm_avg = self._average_density(dm_k)
        if self.jk_builder == "reciprocal":
            vj = self._reciprocal_j_at_k(dm_k, self.kpts[k_index])
            vk = self._reciprocal_k_at_k(dm_k, self.kpts[k_index], self._overlap_k[k_index])
            return _symmetrize(vj - 0.5 * vk)

        if self.nkpts > 1 and not self._use_inf_vacuum_1d():
            vj = self._reciprocal_j_at_k(dm_k, self.kpts[k_index])
        else:
            vj = np.einsum("pqrs,rs->pq", self.eri, dm_avg, optimize=True)

        if self._exchange_eri_k is None:
            vk = np.einsum("prqs,rs->pq", self.eri, dm_avg, optimize=True)
            if self.madelung is not None:
                overlap = self._overlap_k[k_index]
                vk = vk + self.madelung * (overlap @ dm_avg @ overlap)
        else:
            vk = np.zeros_like(vj, dtype=np.complex128)
            for j, dm in enumerate(dm_k):
                xeri = self._exchange_eri_k[k_index, j]
                vk += np.einsum("pqrs,rs->pq", xeri, dm, optimize=True)
            vk /= self.nkpts
            if self.madelung is not None:
                overlap = self._overlap_k[k_index]
                dm = dm_k[k_index]
                vk = vk + self.madelung * (overlap @ dm @ overlap)

        return _symmetrize(vj - 0.5 * vk)

    def _matching_k_index(self, kvec, tol=1e-10):
        kvec = np.asarray(kvec, dtype=float)
        for index, ref in enumerate(self.kpts):
            if np.linalg.norm(kvec - ref) <= tol:
                return index
        return None

    def _get_veff_at_k(self, dm_k, kvec, overlap):
        kvec = np.asarray(kvec, dtype=float)
        cache_key = None
        if self._band_veff_cache is not None:
            cache_key = ("finite_q", self._vector_key(kvec))
            if cache_key in self._band_veff_cache:
                return self._band_veff_cache[cache_key]
        dm_avg = self._average_density(dm_k)
        if self.jk_builder == "reciprocal" or (
            self.nkpts > 1 and not self._use_inf_vacuum_1d()
        ):
            vj = self._reciprocal_j_at_k(dm_k, kvec)
        else:
            vj = np.einsum("pqrs,rs->pq", self.eri, dm_avg, optimize=True)

        if self._use_inf_vacuum_1d():
            vk = np.einsum("prqs,rs->pq", self.eri, dm_avg, optimize=True)
            if self.madelung is not None:
                vk = vk + self.madelung * (overlap @ dm_avg @ overlap)
        elif self.jk_builder == "reciprocal":
            vk = self._reciprocal_k_at_k(dm_k, kvec, overlap)
        else:
            vk = np.zeros_like(vj, dtype=np.complex128)
            for j, dm in enumerate(dm_k):
                block = self._exchange_eri_block(kvec, self.kpts[j])
                vk += np.einsum("pqrs,rs->pq", block, dm, optimize=True)
            vk /= self.nkpts

            match = self._matching_k_index(kvec)
            if self.madelung is not None and match is not None:
                vk = vk + self.madelung * (overlap @ dm_k[match] @ overlap)

        veff = _symmetrize(vj - 0.5 * vk)
        if self._band_veff_cache is not None and cache_key is not None:
            self._band_veff_cache[cache_key] = veff
        return veff

    def _build_fock_k(self, dm_k):
        fock_k = []
        for k_index, hcore in enumerate(self._hcore_k):
            fock_k.append(_symmetrize(hcore + self._get_veff_k(dm_k, k_index)))
        return fock_k

    def band_structure(
        self,
        kpts=None,
        nk=None,
        scaled_kpts=None,
        reference="fermi",
        exchange="mesh_interpolate",
        sort_bands="energy",
    ):
        if self.dm is None:
            raise RuntimeError("Run SCF before calling band_structure().")
        if kpts is not None and nk is not None:
            raise ValueError("Specify either kpts or nk, not both.")
        if scaled_kpts is not None and (kpts is not None or nk is not None):
            raise ValueError("Specify scaled_kpts alone, not with kpts/nk.")
        if nk is not None:
            kpts = self.cell.make_kpts(nk)
        elif scaled_kpts is not None:
            scaled = np.asarray(scaled_kpts, dtype=float)
            if scaled.ndim == 1:
                scaled = scaled.reshape(1, 3)
            if scaled.shape[-1] != 3:
                raise ValueError("scaled_kpts must have shape (nk, 3) or (3,).")
            recip = 2.0 * np.pi * np.linalg.inv(
                np.asarray(self.cell.lattice_vectors, dtype=float)
            ).T
            kpts = scaled @ recip
        elif kpts is None:
            kpts = self.kpts
        kpts = _normalize_kpts(kpts)

        dm_k = [self.dm] if self.nkpts == 1 else list(self.dm)
        mo_energy = []
        mo_coeff = []
        fock_k = []
        overlap_k = []
        exchange_key = str(exchange).lower()
        if exchange_key in ("interpolate", "mesh-interpolate"):
            exchange_key = "mesh_interpolate"
        if exchange_key not in ("average", "finite_q", "mesh", "mesh_interpolate"):
            raise ValueError(
                "exchange must be 'mesh_interpolate', 'average', 'finite_q', or 'mesh'."
            )
        sort_key = str(sort_bands).lower()
        if sort_key not in ("energy", "overlap"):
            raise ValueError("sort_bands must be 'energy' or 'overlap'.")

        if self.nkpts == 1:
            scf_energy = np.asarray(self.mo_energy).reshape(1, -1)
            scf_occ = np.asarray(self.mo_occ).reshape(1, -1)
        else:
            scf_energy = np.asarray(self.mo_energy)
            scf_occ = np.asarray(self.mo_occ)
        occupied = scf_energy[scf_occ > 1e-12]
        e_fermi = float(np.max(occupied)) if occupied.size else None

        if exchange_key == "mesh_interpolate":
            mo_energy = _interpolate_periodic_1d_bands(
                self.kpts,
                scf_energy,
                kpts,
                self.cell.lattice_vectors,
            )
            mo_coeff = [None] * len(kpts)
            fock_k = [None] * len(kpts)
            overlap_k = [None] * len(kpts)
        else:
            dm_avg = self._average_density(dm_k)
            prev_coeff = None
            for kvec in kpts:
                overlap, hcore = self._one_body_at_k(kvec)
                if exchange_key == "average":
                    veff = self.get_veff(dm_avg, overlap)
                elif exchange_key == "mesh":
                    match = self._matching_k_index(kvec)
                    if match is None:
                        raise ValueError(
                            "exchange='mesh' only accepts self-consistent SCF k-points. "
                            "Use exchange='mesh_interpolate' for plotting between mesh points."
                        )
                    veff = self._get_veff_k(dm_k, match)
                else:
                    veff = self._get_veff_at_k(dm_k, kvec, overlap)
                fock = _symmetrize(hcore + veff)
                evals, evecs = eigh(_symmetrize(fock), _symmetrize(overlap))
                if sort_key == "overlap" and prev_coeff is not None:
                    score = np.abs(prev_coeff.conj().T @ overlap @ evecs)
                    _rows, order = linear_sum_assignment(-score)
                    evals = evals[order]
                    evecs = evecs[:, order]
                mo_energy.append(evals)
                mo_coeff.append(evecs)
                fock_k.append(fock)
                overlap_k.append(overlap)
                prev_coeff = evecs

            mo_energy = np.asarray(mo_energy)

        reference_key = str(reference).lower()
        if reference_key in ("fermi", "homo") and e_fermi is not None:
            mo_energy_ref = mo_energy - e_fermi
        elif reference_key in ("none", "absolute"):
            mo_energy_ref = mo_energy.copy()
        else:
            raise ValueError("reference must be 'fermi' or 'none'.")

        return {
            "kpts": kpts,
            "mo_energy": mo_energy,
            "mo_energy_reference": mo_energy_ref,
            "mo_coeff": mo_coeff,
            "fock": fock_k,
            "overlap": overlap_k,
            "e_fermi": e_fermi,
            "exchange": exchange_key,
            "interpolated": exchange_key == "mesh_interpolate",
        }

    def bands(self, *args, **kwargs):
        return self.band_structure(*args, **kwargs)

    def get_fock(self, dm=None):
        if dm is None:
            return self.fock
        dm_k = [dm] if self.nkpts == 1 else list(dm)
        fock_k = self._build_fock_k(dm_k)
        return fock_k[0] if self.nkpts == 1 else fock_k

    def _electronic_energy(self, dm_k, fock_k):
        e = 0.0
        for dm, hcore, fock in zip(dm_k, self._hcore_k, fock_k):
            e += 0.5 * np.trace(dm @ (hcore + fock)).real
        return float(e / len(dm_k))

    def run(self, max_cycle=50, conv_tol=1e-8, conv_tol_dm=1e-6):
        self._build_integrals()
        mo_energy, mo_coeff, mo_occ, dm_k = self._solve_fock(self._hcore_k, self._overlap_k)

        e_last = None
        converged = False
        fock_k = self._hcore_k
        for cycle in range(int(max_cycle)):
            fock_k = self._build_fock_k(dm_k)
            mo_energy_new, mo_coeff_new, mo_occ_new, dm_k_new = self._solve_fock(
                fock_k,
                self._overlap_k,
            )
            if cycle > 0 and self.damping > 0.0:
                dm_k_new = [
                    (1.0 - self.damping) * d_old + self.damping * d_new
                    for d_old, d_new in zip(dm_k, dm_k_new)
                ]

            e_elec = self._electronic_energy(dm_k_new, fock_k)
            if e_last is not None:
                de = abs(e_elec - e_last)
                ddm = _matrix_norm(dm_k_new, dm_k)
                if de < conv_tol and ddm < conv_tol_dm:
                    converged = True
                    dm_k = dm_k_new
                    mo_energy = mo_energy_new
                    mo_coeff = mo_coeff_new
                    mo_occ = mo_occ_new
                    break

            e_last = e_elec
            dm_k = dm_k_new
            mo_energy = mo_energy_new
            mo_coeff = mo_coeff_new
            mo_occ = mo_occ_new

        fock_k = self._build_fock_k(dm_k)
        e_elec = self._electronic_energy(dm_k, fock_k)
        self.e_elec = float(e_elec)
        self.e_tot = float(e_elec + self.e_nuc)
        self.mo_energy = mo_energy[0] if self.nkpts == 1 else mo_energy
        self.mo_coeff = mo_coeff[0] if self.nkpts == 1 else mo_coeff
        self.mo_occ = mo_occ[0] if self.nkpts == 1 else mo_occ
        self.dm = dm_k[0] if self.nkpts == 1 else dm_k
        self.fock = fock_k[0] if self.nkpts == 1 else fock_k
        self.converged = bool(converged)
        self._band_veff_cache = {}
        return self

    def kernel(self, **kwargs):
        return self.run(**kwargs).e_tot

    def get_hcore(self):
        return self.hcore

    def get_ovlp(self):
        return self.overlap

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            return self.dm
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_occ is None:
            raise ValueError("MO occupations are not available.")

        if self.nkpts == 1 and np.asarray(mo_coeff).ndim == 2:
            return self._density_from_mo_occ(mo_coeff, mo_occ)
        dm = [
            self._density_from_mo_occ(coeff_k, occ_k)
            for coeff_k, occ_k in zip(mo_coeff, mo_occ)
        ]
        return dm[0] if self.nkpts == 1 and len(dm) == 1 else dm


KRHF = EwaldRHF
