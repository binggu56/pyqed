#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import math
import os
import time

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
from pyqed.qchem.pbc.pseudo import local_gaussian_overlap, projector_overlap


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


def _gaussian_pair_ft_decay_bounds(a, b, shifts):
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 3)
    a_origin = np.asarray(a.origin, dtype=float)
    b_origins = np.asarray(b.origin, dtype=float)[None, :] + shifts
    distances = np.linalg.norm(a_origin[None, :] - b_origins, axis=1)
    angular = int(sum(a.shell) + sum(b.shell))
    bounds = np.zeros(len(shifts), dtype=float)
    for alpha, wa in zip(a.exps, a.prim_weights):
        alpha = float(alpha)
        for beta, wb in zip(b.exps, b.prim_weights):
            beta = float(beta)
            p = alpha + beta
            q = alpha * beta / p
            moment = (distances + 1.0 / math.sqrt(p) + 1.0) ** angular
            bounds += (
                abs(float(wa) * float(wb))
                * (math.pi / p) ** 1.5
                * np.exp(-q * distances * distances)
                * moment
            )
    return bounds


def _gaussian_pair_ft_decay_radius(a, b, tol):
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("Automatic image domains require a positive tolerance.")
    angular = int(sum(a.shell) + sum(b.shell))
    terms = []
    monotone_radius = 0.0
    for alpha, wa in zip(a.exps, a.prim_weights):
        alpha = float(alpha)
        for beta, wb in zip(b.exps, b.prim_weights):
            beta = float(beta)
            p = alpha + beta
            q = alpha * beta / p
            offset = 1.0 / math.sqrt(p) + 1.0
            prefactor = abs(float(wa) * float(wb)) * (math.pi / p) ** 1.5
            terms.append((prefactor, q, offset))
            if angular:
                monotone_radius = max(
                    monotone_radius,
                    0.5
                    * (
                        math.sqrt(offset * offset + 2.0 * angular / q)
                        - offset
                    ),
                )

    def radial_bound(distance):
        return sum(
            prefactor
            * math.exp(-q * distance * distance)
            * (distance + offset) ** angular
            for prefactor, q, offset in terms
        )

    lower = monotone_radius
    upper = max(1.0, lower)
    while radial_bound(upper) > tol:
        upper *= 2.0
        if upper > 1.0e6:
            raise RuntimeError("Unable to resolve an automatic AO image radius.")
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        if radial_bound(midpoint) > tol:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _screened_image_domain(cell, basis, tol, cut=None):
    tol = float(tol)
    if cut is None:
        radius = max(
            _gaussian_pair_ft_decay_radius(left, right, tol)
            + float(
                np.linalg.norm(
                    np.asarray(left.origin, dtype=float)
                    - np.asarray(right.origin, dtype=float)
                )
            )
            for left in basis
            for right in basis
        )
        if cell.dimension == 1:
            bounds = (
                int(math.ceil(radius / np.linalg.norm(cell.lattice_vectors[0]))),
            )
        else:
            inverse = np.linalg.inv(np.asarray(cell.lattice_vectors, dtype=float))
            bounds = tuple(
                int(math.ceil(radius * np.linalg.norm(inverse[:, axis])))
                for axis in range(3)
            )
    else:
        cut = int(cut)
        bounds = (cut,) if cell.dimension == 1 else (cut, cut, cut)

    ranges = [range(-bound, bound + 1) for bound in bounds]
    if cell.dimension == 1:
        keys = [(i,) for i in ranges[0]]
    else:
        keys = [
            (i, j, k)
            for i in ranges[0]
            for j in ranges[1]
            for k in ranges[2]
        ]
    if tol == 0.0:
        return keys

    shifts = np.asarray([cell.translation_vector(key) for key in keys], dtype=float)
    keep = np.zeros(len(keys), dtype=bool)
    for left in basis:
        for right in basis:
            keep |= _gaussian_pair_ft_decay_bounds(left, right, shifts) > tol
    retained = [key for key, retain in zip(keys, keep) if retain]
    zero = (0,) if cell.dimension == 1 else (0, 0, 0)
    if zero not in retained:
        retained.append(zero)
    return sorted(retained)


def _is_auto_cut(value):
    return value is None or str(value).strip().lower() == "auto"


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


class _KPointDIIS:
    def __init__(self, space=8, start_cycle=2):
        self.space = max(2, int(space))
        self.start_cycle = max(0, int(start_cycle))
        self.errors = []
        self.focks = []

    def update(self, fock_k, dm_k, overlap_k, cycle):
        errors = [
            fock @ dm @ overlap - overlap @ dm @ fock
            for fock, dm, overlap in zip(fock_k, dm_k, overlap_k)
        ]
        error_norm = max(float(np.max(np.abs(error))) for error in errors)
        self.errors.append([np.asarray(error).copy() for error in errors])
        self.focks.append([np.asarray(fock).copy() for fock in fock_k])
        if len(self.errors) > self.space:
            self.errors.pop(0)
            self.focks.pop(0)
        if int(cycle) < self.start_cycle or len(self.errors) < 2:
            return fock_k, error_norm

        size = len(self.errors)
        system = np.empty((size + 1, size + 1), dtype=float)
        system[:size, size] = -1.0
        system[size, :size] = -1.0
        system[size, size] = 0.0
        rhs = np.zeros(size + 1, dtype=float)
        rhs[size] = -1.0
        for i in range(size):
            for j in range(size):
                system[i, j] = sum(
                    float(np.vdot(left, right).real)
                    for left, right in zip(self.errors[i], self.errors[j])
                )
        try:
            coeff = np.linalg.solve(system, rhs)[:size]
        except np.linalg.LinAlgError:
            return fock_k, error_norm
        extrapolated = [
            _symmetrize(sum(coeff[i] * self.focks[i][k] for i in range(size)))
            for k in range(len(fock_k))
        ]
        return extrapolated, error_norm


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
    ``recip_bounds`` optionally gives three reciprocal index bounds for the
    one-electron AO build while ``recip_cut`` remains the scalar Ewald control.
    """

    def __init__(
        self,
        cell,
        kpts=None,
        nk=None,
        gamma_centered=False,
        eta=0.5,
        real_cut=3,
        recip_cut="auto",
        recip_bounds=None,
        recip_precision=1.0e-8,
        recip_max_cut=30,
        mesh=None,
        damping=0.5,
        nuclear_background=True,
        eri_screen_tol=1.0e-12,
        jk_builder=None,
        pair_cut=None,
        pair_ft_screen_tol=1.0e-14,
        occupation_mode="aufbau",
        occupation_tol=1.0e-10,
        pseudo_cut=None,
        pseudo_local_screen_tol=1.0e-13,
        one_body_screen_tol=1.0e-14,
        one_body_nuclear_cut=4,
        one_body_workers=None,
        diis=True,
        diis_space=8,
        diis_start_cycle=2,
    ):
        self.cell = cell
        if kpts is not None and nk is not None:
            raise ValueError("Specify either kpts or nk, not both.")
        if nk is not None:
            if not self.cell.built:
                self.cell.build()
            kpts = self.cell.make_kpts(nk, gamma_centered=gamma_centered)
        self.kpts = _normalize_kpts(kpts)
        self.eta = float(eta)
        self.real_cut = "auto" if _is_auto_cut(real_cut) else int(real_cut)
        self._recip_cut_request = None
        self._resolved_recip_cut = None
        self.recip_auto_info = None
        self._reciprocal_domain_seconds = 0.0
        self.recip_cut = recip_cut
        if recip_bounds is None:
            self.recip_bounds = None
        else:
            bounds = np.asarray(recip_bounds, dtype=int)
            if bounds.shape != (3,) or np.any(bounds < 0):
                raise ValueError("recip_bounds must contain three non-negative integers.")
            self.recip_bounds = tuple(int(value) for value in bounds)
        self.recip_precision = float(recip_precision)
        self.recip_max_cut = int(recip_max_cut)
        self.mesh = None if mesh is None else tuple(int(x) for x in mesh)
        self.damping = float(damping)
        self.nuclear_background = bool(nuclear_background)
        self.eri_screen_tol = float(eri_screen_tol)
        if jk_builder is None:
            jk_builder = "gdf" if getattr(self.cell, "pseudo", None) is not None else "ewald"
        self.jk_builder = str(jk_builder).lower()
        if self.jk_builder not in ("ewald", "reciprocal", "gdf"):
            raise ValueError("jk_builder must be 'ewald', 'reciprocal', or 'gdf'.")
        pair_cut = self.real_cut if pair_cut is None else pair_cut
        self.pair_cut = "auto" if _is_auto_cut(pair_cut) else int(pair_cut)
        self.pair_ft_screen_tol = float(pair_ft_screen_tol)
        self.occupation_mode = str(occupation_mode).strip().lower()
        if self.occupation_mode not in ("aufbau", "fractional"):
            raise ValueError("occupation_mode must be 'aufbau' or 'fractional'.")
        self.occupation_tol = float(occupation_tol)
        self.pseudo_cut = (
            1
            if pseudo_cut is None and getattr(self.cell, "pseudo", None) is not None
            else (
                0
                if pseudo_cut is None and _is_auto_cut(self.pair_cut)
                else (self.pair_cut if pseudo_cut is None else int(pseudo_cut))
            )
        )
        self.pseudo_local_screen_tol = float(pseudo_local_screen_tol)
        self.one_body_screen_tol = float(one_body_screen_tol)
        self.one_body_nuclear_cut = int(one_body_nuclear_cut)
        if one_body_workers is None:
            value = os.environ.get("PYQED_PBC_ONE_BODY_WORKERS")
            try:
                cap = 12 if value is None else max(1, int(value))
            except (TypeError, ValueError):
                cap = 12
            one_body_workers = min(cap, os.cpu_count() or 1)
        self.one_body_workers = max(1, int(one_body_workers))
        self.diis = bool(diis)
        self.diis_space = max(2, int(diis_space))
        self.diis_start_cycle = max(0, int(diis_start_cycle))

        self.e_tot = None
        self.e_elec = None
        self.e_nuc = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.dm = None
        self.fock = None
        self.converged = False
        self.niter = 0
        self.scf_history = []
        self.integral_build_timings = {}
        self.nkpts = int(len(self.kpts))

        self.overlap = None
        self.hcore = None
        self.eri = None
        self.madelung = None
        self._basis = None
        self._periodic_setup_key = None
        self._shift_keys = None
        self._shift_vectors = None
        self._resolved_shift_keys = None
        self._shifted_basis = None
        self._pair_shift_keys = None
        self._pair_shift_vectors = None
        self._resolved_pair_shift_keys = None
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
        self._vpp_local_r = None
        self._overlap_k = None
        self._hcore_k = None
        self._exchange_eri_k = None
        self._pair_ft_cache = None
        self._pair_ft_batch_cache = None
        self._g_weight_cache = None
        self._one_body_cache = None
        self._one_body_image_pair_count = 0
        self._pseudo_projector_terms = None
        self._pseudo_shift_vectors = None
        self._band_veff_cache = None
        self.with_df = None

    @property
    def recip_cut(self):
        if self._resolved_recip_cut is not None:
            return self._resolved_recip_cut
        return self._recip_cut_request

    @recip_cut.setter
    def recip_cut(self, value):
        if _is_auto_cut(value):
            self._recip_cut_request = "auto"
            self._resolved_recip_cut = None
        else:
            cut = int(value)
            self._recip_cut_request = cut
            self._resolved_recip_cut = cut
        if hasattr(self, "recip_auto_info"):
            self.recip_auto_info = None

    @staticmethod
    def _resolved_cut_from_keys(keys):
        return max((max(abs(value) for value in key) for key in keys), default=0)

    def _resolve_periodic_image_domains(self):
        if not self.cell.built:
            self.cell.build()
        basis = tuple(self.cell.unit_molecule._bas)
        real_auto = _is_auto_cut(self.real_cut)
        pair_auto = _is_auto_cut(self.pair_cut)
        if real_auto and self.one_body_screen_tol <= 0.0:
            raise ValueError(
                "real_cut='auto' requires one_body_screen_tol to be positive."
            )
        pair_domain_tol = min(
            self.pair_ft_screen_tol,
            self.one_body_screen_tol,
        )
        if pair_auto and pair_domain_tol <= 0.0:
            raise ValueError(
                "pair_cut='auto' requires a positive AO-pair screen tolerance."
            )

        real_envelope = None if real_auto else int(self.real_cut)
        pair_envelope = None if pair_auto else int(self.pair_cut)
        real_keys = _screened_image_domain(
            self.cell,
            basis,
            self.one_body_screen_tol,
            cut=real_envelope,
        )
        pair_keys = _screened_image_domain(
            self.cell,
            basis,
            pair_domain_tol,
            cut=pair_envelope,
        )
        resolved_real_cut = (
            self._resolved_cut_from_keys(real_keys)
            if real_auto
            else real_envelope
        )
        resolved_pair_cut = (
            self._resolved_cut_from_keys(pair_keys)
            if pair_auto
            else pair_envelope
        )
        if resolved_pair_cut < resolved_real_cut:
            raise ValueError(
                "pair_cut must be at least real_cut so the reciprocal "
                "one-electron AO-pair domain is not truncated."
            )

        pair_keys = sorted(set(pair_keys).union(real_keys))
        self.real_cut = int(resolved_real_cut)
        self.pair_cut = int(max(resolved_pair_cut, resolved_real_cut))
        self._resolved_shift_keys = real_keys
        self._resolved_pair_shift_keys = pair_keys

    def _vector_key(self, vec):
        arr = np.asarray(vec, dtype=float)
        arr = np.where(np.abs(arr) < 1e-14, 0.0, arr)
        return tuple(np.round(arr, 12))

    def _reciprocal_cut_configuration_key(self):
        if _is_auto_cut(self._recip_cut_request):
            return (
                "auto",
                float(self.recip_precision),
                int(self.recip_max_cut),
            )
        return ("explicit", int(self._recip_cut_request))

    def _periodic_setup_configuration_key(self):
        basis = tuple(self.cell.unit_molecule._bas)
        basis_key = tuple(
            (
                tuple(np.asarray(fn.origin, dtype=float)),
                tuple(int(value) for value in fn.shell),
                tuple(np.asarray(fn.exps, dtype=float)),
                tuple(np.asarray(fn.prim_weights, dtype=float)),
            )
            for fn in basis
        )
        return (
            basis_key,
            np.ascontiguousarray(self.cell._atom_coords, dtype=float).tobytes(),
            np.ascontiguousarray(self.cell.lattice_vectors, dtype=float).tobytes(),
            np.ascontiguousarray(self.kpts, dtype=float).tobytes(),
            float(self.eta),
            self.jk_builder,
            int(self.real_cut),
            self._reciprocal_cut_configuration_key(),
            self.recip_bounds,
            int(self.pair_cut),
            int(self.pseudo_cut),
            float(self.pair_ft_screen_tol),
            float(self.pseudo_local_screen_tol),
            float(self.one_body_screen_tol),
            int(self.one_body_nuclear_cut),
            tuple(self._resolved_shift_keys or ()),
            tuple(self._resolved_pair_shift_keys or ()),
            self.mesh,
            id(getattr(self.cell, "_pseudos_by_atom", None)),
        )

    def _validate(self):
        if not self.cell.built:
            self.cell.build()
        if self.recip_bounds is not None:
            bounds = np.asarray(self.recip_bounds, dtype=int)
            if bounds.shape != (3,) or np.any(bounds < 0):
                raise ValueError(
                    "recip_bounds must contain three non-negative integers."
                )
            self.recip_bounds = tuple(int(value) for value in bounds)
        if (
            self.cell.has_pseudo
            and not _is_auto_cut(self.pair_cut)
            and int(self.pair_cut) < 2
        ):
            raise ValueError(
                "Native GTH pseudopotentials require pair_cut >= 2 for a "
                "converged periodic electron-ion matrix."
            )
        self._resolve_periodic_image_domains()
        if self.cell.dimension not in (1, 3):
            raise NotImplementedError("EwaldRHF supports dimension=1 and dimension=3 cells.")
        if self.cell.has_pseudo and self.cell.dimension != 3:
            raise NotImplementedError(
                "Native GTH pseudopotentials currently require dimension=3."
            )
        if self.cell.has_pseudo and self.jk_builder == "ewald":
            raise NotImplementedError(
                "Native GTH pseudopotentials require jk_builder='gdf' or 'reciprocal'."
            )
        if self.cell.has_pseudo and self.pair_cut < 2:
            raise ValueError(
                "Native GTH pseudopotentials require pair_cut >= 2 for a "
                "converged periodic electron-ion matrix."
            )
        if self.pair_cut < self.real_cut:
            raise ValueError(
                "pair_cut must be at least real_cut so the reciprocal "
                "one-electron AO-pair domain is not truncated."
            )
        if self.cell.charge != 0:
            raise NotImplementedError("EwaldRHF v1 requires neutral periodic cells.")
        if self.cell.spin != 0 or int(self.cell.nelectron) % 2:
            raise NotImplementedError("EwaldRHF v1 supports closed-shell even-electron cells only.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive for method='ewald'.")
        if not np.isfinite(self.recip_precision) or self.recip_precision <= 0.0:
            raise ValueError("recip_precision must be a positive finite value.")
        if self.recip_max_cut < 2:
            raise ValueError("recip_max_cut must be at least 2.")
        cut_values = [
            self.real_cut,
            self.pair_cut,
            self.pseudo_cut,
            self.one_body_nuclear_cut,
        ]
        if not _is_auto_cut(self._recip_cut_request):
            cut_values.append(int(self._recip_cut_request))
        if min(cut_values) < 0:
            raise ValueError("Periodic real/reciprocal cutoffs must be non-negative.")
        if self.pair_ft_screen_tol < 0.0:
            raise ValueError("pair_ft_screen_tol must be non-negative.")
        if self.occupation_tol < 0.0:
            raise ValueError("occupation_tol must be non-negative.")
        if self.pseudo_local_screen_tol < 0.0:
            raise ValueError("pseudo_local_screen_tol must be non-negative.")
        if self.one_body_screen_tol < 0.0:
            raise ValueError("one_body_screen_tol must be non-negative.")

    def _use_inf_vacuum_1d(self):
        return (
            int(self.cell.dimension) == 1
            and str(getattr(self.cell, "low_dim_ft_type", "")).lower() == "inf_vacuum"
        )

    def _reciprocal_mesh(self):
        if self.mesh is not None:
            return self.mesh
        return (31, 38, 38) if self._use_inf_vacuum_1d() else None

    @staticmethod
    def _reciprocal_shell_indices(cut):
        values = range(-int(cut), int(cut) + 1)
        return np.asarray(
            [
                (h, k, n3)
                for h in values
                for k in values
                for n3 in values
                if max(abs(h), abs(k), abs(n3)) == cut
            ],
            dtype=float,
        )

    def _reciprocal_kpoint_pad(self):
        if self.nkpts <= 1:
            return 0
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        scaled = np.asarray(self.kpts, dtype=float) @ np.linalg.inv(reciprocal)
        transfers = scaled[:, None, :] - scaled[None, :, :]
        largest = float(np.max(np.abs(transfers)))
        return max(0, int(math.ceil(largest - 1.0e-12)))

    def _reciprocal_shell_bound(self, cut, block_size=512):
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        indices = self._reciprocal_shell_indices(cut)
        volume = abs(float(np.linalg.det(lattice)))
        nao = len(self._basis)
        electronic_matrix = np.zeros((self.nkpts, nao, nao), dtype=float)
        nuclear_matrix = np.zeros_like(electronic_matrix)
        ionic = 0.0
        charge_l1 = float(np.sum(np.abs(self.cell.ionic_charges)))

        for start in range(0, len(indices), int(block_size)):
            gvecs = np.ascontiguousarray(
                indices[start : start + int(block_size)] @ reciprocal,
                dtype=float,
            )
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            coulomb = 4.0 * np.pi / (volume * g2)
            damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
            pair_abs = np.abs(self._periodic_pair_ft_batch_many(gvecs, self.kpts))

            if self.jk_builder in ("ewald", "reciprocal"):
                electronic_damping = (
                    damping if self.jk_builder == "ewald" else np.ones_like(damping)
                )
                electronic_matrix += np.einsum(
                    "g,kgpq->kpq",
                    coulomb * electronic_damping,
                    pair_abs * pair_abs,
                    optimize=True,
                )
            nuclear_matrix += np.einsum(
                "g,kgpq->kpq",
                coulomb * damping * charge_l1,
                pair_abs,
                optimize=True,
            )
            ionic += float(
                np.sum(0.5 * coulomb * damping * charge_l1 * charge_l1)
            )

        components = {
            "electronic": float(np.max(electronic_matrix, initial=0.0)),
            "electron_nuclear": float(np.max(nuclear_matrix, initial=0.0)),
            "ion_ion": float(ionic),
        }
        return float(sum(components.values())), components

    def _resolve_reciprocal_domain(self):
        started = time.perf_counter()
        try:
            if self._use_inf_vacuum_1d():
                resolved = (
                    0
                    if _is_auto_cut(self._recip_cut_request)
                    else int(self._recip_cut_request)
                )
                self._resolved_recip_cut = resolved
                self.recip_auto_info = {
                    "mode": "mesh",
                    "requested": self._recip_cut_request,
                    "resolved_cut": resolved,
                    "mesh": self._reciprocal_mesh(),
                }
                return

            if not _is_auto_cut(self._recip_cut_request):
                resolved = int(self._recip_cut_request)
                self._resolved_recip_cut = resolved
                self.recip_auto_info = {
                    "mode": "explicit",
                    "requested": resolved,
                    "resolved_cut": resolved,
                }
                return

            kpoint_pad = self._reciprocal_kpoint_pad()
            base_max_cut = self.recip_max_cut - kpoint_pad
            if base_max_cut < 2:
                raise ValueError(
                    "recip_max_cut is too small for the reciprocal k-point transfer pad."
                )

            history = []
            ratios = []
            previous = None
            selected = None
            estimated_tail = math.inf
            for cut in range(1, base_max_cut + 1):
                shell_bound, components = self._reciprocal_shell_bound(cut)
                ratio = None
                if previous is not None and previous > 0.0:
                    ratio = shell_bound / previous
                    ratios.append(ratio)
                if shell_bound == 0.0:
                    estimated_tail = 0.0
                elif cut >= 2 and ratios:
                    recent = ratios[-3:]
                    if all(0.0 <= value < 1.0 for value in recent):
                        tail_ratio = max(recent)
                        estimated_tail = (
                            shell_bound * tail_ratio / (1.0 - tail_ratio)
                        )
                    else:
                        estimated_tail = math.inf
                history.append(
                    {
                        "cut": cut,
                        "shell_bound": shell_bound,
                        "estimated_tail": estimated_tail,
                        **components,
                    }
                )
                if cut >= 2 and estimated_tail <= self.recip_precision:
                    selected = cut
                    break
                previous = shell_bound

            if selected is None:
                last = history[-1]
                raise RuntimeError(
                    "Automatic reciprocal cutoff did not reach recip_precision="
                    f"{self.recip_precision:.3e} by recip_max_cut={self.recip_max_cut}; "
                    f"last estimated tail was {last['estimated_tail']:.3e}."
                )

            resolved = selected + kpoint_pad
            self._resolved_recip_cut = resolved
            self.recip_auto_info = {
                "mode": "auto",
                "requested": "auto",
                "precision": self.recip_precision,
                "base_cut": selected,
                "kpoint_pad": kpoint_pad,
                "resolved_cut": resolved,
                "shell_bound": history[-1]["shell_bound"],
                "estimated_tail": history[-1]["estimated_tail"],
                "history": history,
            }
        finally:
            self._reciprocal_domain_seconds = time.perf_counter() - started

    def _periodic_setup(self):
        self._resolve_periodic_image_domains()
        setup_key = self._periodic_setup_configuration_key()
        if self._basis is not None and setup_key == self._periodic_setup_key:
            self._reciprocal_domain_seconds = 0.0
            return False
        self._basis = tuple(self.cell.unit_molecule._bas)
        if len(self._basis) != int(self.cell.nao):
            raise NotImplementedError(
                "EwaldRHF currently supports Cartesian AO builds only. "
                "Use integral_options={'coord_type': 'cartesian'} for p/d basis tests."
            )

        self._shift_keys = list(self._resolved_shift_keys)
        self._shift_vectors = {
            key: self.cell.translation_vector(key)
            for key in self._shift_keys
        }
        self._shifted_basis = {
            key: [_shifted_gaussian(fn, self._shift_vectors[key]) for fn in self._basis]
            for key in self._shift_keys
        }
        self._pair_shift_keys = list(self._resolved_pair_shift_keys)
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
        self._resolve_reciprocal_domain()
        self._one_body_cache = {}
        self._build_pseudo_projector_terms()
        self._band_veff_cache = {}
        self._periodic_setup_key = setup_key
        return True

    def _build_pseudo_projector_terms(self):
        self._pseudo_projector_terms = []
        if not self.cell.has_pseudo:
            self._pseudo_shift_vectors = np.zeros((0, 3), dtype=float)
            return

        keys = list(self.cell.image_keys(self.pseudo_cut))
        shifts = np.asarray(
            [self.cell.translation_vector(key) for key in keys],
            dtype=float,
        )
        self._pseudo_shift_vectors = shifts
        nao = len(self._basis)
        for atom_index, pseudo in enumerate(self.cell._pseudos_by_atom):
            if pseudo is None:
                continue
            center = np.asarray(self.cell._atom_coords[atom_index], dtype=float)
            for angular_momentum, projector in enumerate(pseudo.projectors):
                if projector.nproj == 0:
                    continue
                overlaps = np.zeros(
                    (
                        len(shifts),
                        projector.nproj,
                        2 * angular_momentum + 1,
                        nao,
                    ),
                    dtype=np.complex128,
                )
                for image, shift in enumerate(shifts):
                    shifted_basis = [
                        _shifted_gaussian(function, shift)
                        for function in self._basis
                    ]
                    for projector_index in range(projector.nproj):
                        for magnetic_index, magnetic_number in enumerate(
                            range(-angular_momentum, angular_momentum + 1)
                        ):
                            for ao, function in enumerate(shifted_basis):
                                overlaps[
                                    image,
                                    projector_index,
                                    magnetic_index,
                                    ao,
                                ] = projector_overlap(
                                    function,
                                    center,
                                    angular_momentum,
                                    magnetic_number,
                                    projector_index,
                                    projector.radius,
                                )
                self._pseudo_projector_terms.append(
                    (projector.coupling, overlaps)
                )

    def _build_pair_ft_terms(self):
        terms = []
        nao = len(self._basis)
        mask = np.zeros((len(self._pair_shift_keys), nao * nao), dtype=np.bool_)
        shifts = np.asarray(
            [self._pair_shift_vectors[key] for key in self._pair_shift_keys],
            dtype=float,
        )
        screen_tol = min(self.pair_ft_screen_tol, self.one_body_screen_tol)
        for p, left in enumerate(self._basis):
            for q, right in enumerate(self._basis):
                keep = (
                    np.ones(len(shifts), dtype=bool)
                    if screen_tol == 0.0
                    else _gaussian_pair_ft_decay_bounds(left, right, shifts)
                    > screen_tol
                )
                mask[:, p * nao + q] = keep
                for image in np.nonzero(keep)[0]:
                    key = self._pair_shift_keys[int(image)]
                    terms.append(
                        (
                            shifts[image],
                            p,
                            q,
                            left,
                            self._pair_shifted_basis[key][q],
                        )
                    )
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
        charges = self.cell.ionic_charges
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        nao = len(self._basis)
        self._s_r = {}
        self._t_r = {}
        self._vne_sr_r = {}
        self._vpp_local_r = {
            key: np.zeros((nao, nao), dtype=float)
            for key in self._shift_keys
        }

        try:
            from pyqed.qchem import basis as basis_module

            basis_cy = basis_module._basis_cy
            compiled = getattr(basis_cy, "compute_periodic_one_electron", None)
        except (AttributeError, ImportError):
            compiled = None
        if compiled is not None:
            signatures = [basis_module._basis_signature(fn) for fn in self._basis]
            shells, origins, exps, weights, nprim = (
                basis_module._pack_signatures_for_numba(signatures)
            )
            shifts = np.asarray(
                [self._shift_vectors[key] for key in self._shift_keys], dtype=float
            )
            right_origins = origins[None, :, :] + shifts[:, None, :]
            nuclear_image_keys = list(
                self.cell.image_keys(self.one_body_nuclear_cut)
            )
            use_lattice_images = self.cell.dimension == 3
            if use_lattice_images:
                nuclear_coords = coords
                nuclear_charges = charges
            else:
                nuclear_shifts = [
                    self.cell.translation_vector(key)
                    for key in nuclear_image_keys
                ]
                nuclear_coords = np.concatenate(
                    [coords + shift for shift in nuclear_shifts], axis=0
                )
                nuclear_charges = np.tile(charges, len(nuclear_shifts))
            image_pair_mask = np.zeros(
                (len(shifts), nao, nao),
                dtype=np.uint8,
            )
            for p, left in enumerate(self._basis):
                for q, right in enumerate(self._basis):
                    if self.one_body_screen_tol == 0.0:
                        image_pair_mask[:, p, q] = 1
                    else:
                        image_pair_mask[:, p, q] = (
                            _gaussian_pair_ft_decay_bounds(left, right, shifts)
                            > self.one_body_screen_tol
                        )
            packed = (
                np.ascontiguousarray(shells, dtype=np.int64),
                np.ascontiguousarray(origins, dtype=np.float64),
                np.ascontiguousarray(exps, dtype=np.float64),
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(nprim, dtype=np.int64),
                np.ascontiguousarray(nuclear_coords, dtype=np.float64),
                np.ascontiguousarray(nuclear_charges, dtype=np.float64),
            )
            self._one_body_image_pair_count = int(
                np.count_nonzero(image_pair_mask)
            )
            worker_count = min(self.one_body_workers, len(right_origins))
            image_chunks = [
                chunk
                for chunk in np.array_split(
                    np.arange(len(right_origins)),
                    worker_count,
                )
                if len(chunk)
            ]

            def build_chunk(chunk):
                start = int(chunk[0])
                stop = int(chunk[-1]) + 1
                args = (
                    packed[0],
                    packed[1],
                    np.ascontiguousarray(right_origins[start:stop]),
                    packed[2],
                    packed[3],
                    packed[4],
                    packed[5],
                    packed[6],
                    float(self.eta),
                    np.ascontiguousarray(image_pair_mask[start:stop]),
                    float(self.one_body_screen_tol),
                )
                if use_lattice_images:
                    args += (
                        np.ascontiguousarray(
                            self.cell.lattice_vectors,
                            dtype=np.float64,
                        ),
                        np.ascontiguousarray(nuclear_image_keys, dtype=np.int64),
                    )
                return compiled(*args)

            if worker_count <= 1:
                chunks = [build_chunk(image_chunks[0])]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    chunks = list(executor.map(build_chunk, image_chunks))
            overlap = np.concatenate([chunk[0] for chunk in chunks], axis=0)
            kinetic = np.concatenate([chunk[1] for chunk in chunks], axis=0)
            vnuc = np.concatenate([chunk[2] for chunk in chunks], axis=0)
            for image, key in enumerate(self._shift_keys):
                self._s_r[key] = np.asarray(overlap[image])
                self._t_r[key] = np.asarray(kinetic[image])
                self._vne_sr_r[key] = np.asarray(vnuc[image])
            self._build_local_pseudo_blocks(
                compiled=compiled,
                packed=(shells, origins, exps, weights, nprim, right_origins),
            )
            return

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
        self._one_body_image_pair_count = len(self._shift_keys) * nao * nao
        self._build_local_pseudo_blocks()

    def _build_local_pseudo_blocks(self, compiled=None, packed=None):
        if not self.cell.has_pseudo:
            return
        pseudo_keys = list(self.cell.image_keys(self.pseudo_cut))
        pseudo_shifts = [self.cell.translation_vector(key) for key in pseudo_keys]

        compiled_gaussian = False
        if compiled is not None and packed is not None:
            shells, origins, exps, weights, nprim, right_origins = packed
            groups = {}
            for atom_index, pseudo in enumerate(self.cell._pseudos_by_atom):
                if pseudo is None:
                    continue
                groups.setdefault(round(float(pseudo.local_radius), 14), []).append(
                    (atom_index, pseudo)
                )
            for entries in groups.values():
                nuclear_coords = []
                nuclear_charges = []
                for shift in pseudo_shifts:
                    for atom_index, pseudo in entries:
                        nuclear_coords.append(self.cell._atom_coords[atom_index] + shift)
                        nuclear_charges.append(-float(pseudo.ionic_charge))
                _overlap, _kinetic, correction = compiled(
                    np.ascontiguousarray(shells, dtype=np.int64),
                    np.ascontiguousarray(origins, dtype=np.float64),
                    np.ascontiguousarray(right_origins, dtype=np.float64),
                    np.ascontiguousarray(exps, dtype=np.float64),
                    np.ascontiguousarray(weights, dtype=np.float64),
                    np.ascontiguousarray(nprim, dtype=np.int64),
                    np.ascontiguousarray(nuclear_coords, dtype=np.float64),
                    np.ascontiguousarray(nuclear_charges, dtype=np.float64),
                    1.0 / (np.sqrt(2.0) * float(entries[0][1].local_radius)),
                )
                for image, key in enumerate(self._shift_keys):
                    self._vpp_local_r[key] += np.asarray(correction[image])

            try:
                from pyqed.qchem import basis as basis_module

                gaussian_kernel = getattr(
                    basis_module._basis_cy,
                    "compute_periodic_gth_local_gaussian",
                    None,
                )
            except (AttributeError, ImportError):
                gaussian_kernel = None
            if gaussian_kernel is not None:
                pseudo_coords = []
                pseudo_radii = []
                local_coefficients = []
                nlocal = []
                for shift in pseudo_shifts:
                    for atom_index, pseudo in enumerate(self.cell._pseudos_by_atom):
                        if pseudo is None:
                            continue
                        pseudo_coords.append(self.cell._atom_coords[atom_index] + shift)
                        pseudo_radii.append(float(pseudo.local_radius))
                        row = np.zeros(4, dtype=float)
                        row[:len(pseudo.local_coefficients)] = pseudo.local_coefficients
                        local_coefficients.append(row)
                        nlocal.append(len(pseudo.local_coefficients))
                gaussian = gaussian_kernel(
                    np.ascontiguousarray(shells, dtype=np.int64),
                    np.ascontiguousarray(origins, dtype=np.float64),
                    np.ascontiguousarray(right_origins, dtype=np.float64),
                    np.ascontiguousarray(exps, dtype=np.float64),
                    np.ascontiguousarray(weights, dtype=np.float64),
                    np.ascontiguousarray(nprim, dtype=np.int64),
                    np.ascontiguousarray(pseudo_coords, dtype=np.float64),
                    np.ascontiguousarray(pseudo_radii, dtype=np.float64),
                    np.ascontiguousarray(local_coefficients, dtype=np.float64),
                    np.ascontiguousarray(nlocal, dtype=np.int64),
                )
                for image, key in enumerate(self._shift_keys):
                    self._vpp_local_r[key] += np.asarray(gaussian[image])
                compiled_gaussian = True
        else:
            for key in self._shift_keys:
                shifted_basis = self._shifted_basis[key]
                for p, left in enumerate(self._basis):
                    for q, right in enumerate(shifted_basis):
                        for shift in pseudo_shifts:
                            for atom_index, pseudo in enumerate(self.cell._pseudos_by_atom):
                                if pseudo is None:
                                    continue
                                center = self.cell._atom_coords[atom_index] + shift
                                eta = 1.0 / (np.sqrt(2.0) * pseudo.local_radius)
                                self._vpp_local_r[key][p, q] += (
                                    pseudo.ionic_charge
                                    * short_range_point_charge_s(left, right, center, eta)
                                )

        if compiled_gaussian:
            return

        for key in self._shift_keys:
            shifted_basis = self._shifted_basis[key]
            for p, left in enumerate(self._basis):
                for q, right in enumerate(shifted_basis):
                    if self.pair_ft_screen_tol > 0.0 and (
                        _gaussian_pair_ft_decay_bound(left, self._basis[q], self._shift_vectors[key])
                        <= self.pair_ft_screen_tol
                    ):
                        continue
                    for shift in pseudo_shifts:
                        for atom_index, pseudo in enumerate(self.cell._pseudos_by_atom):
                            if pseudo is None:
                                continue
                            center = self.cell._atom_coords[atom_index] + shift
                            if self._local_pseudo_gaussian_bound(
                                left, right, center, pseudo
                            ) <= self.pseudo_local_screen_tol:
                                continue
                            self._vpp_local_r[key][p, q] += local_gaussian_overlap(
                                left,
                                right,
                                center,
                                pseudo,
                            )

    @staticmethod
    def _local_pseudo_gaussian_bound(left, right, center, pseudo):
        center = np.asarray(center, dtype=float)
        left_center = np.asarray(left.origin, dtype=float)
        right_center = np.asarray(right.origin, dtype=float)
        angular_left = int(sum(left.shell))
        angular_right = int(sum(right.shell))
        exponent_c = 0.5 / float(pseudo.local_radius) ** 2
        radial_scale = sum(
            abs(float(coefficient))
            for coefficient in pseudo.local_coefficients
        )
        if radial_scale == 0.0:
            return 0.0
        bound = 0.0
        for exponent_a, weight_a in zip(left.exps, left.prim_weights):
            for exponent_b, weight_b in zip(right.exps, right.prim_weights):
                exponents = np.asarray(
                    [float(exponent_a), float(exponent_b), exponent_c]
                )
                centers = np.asarray([left_center, right_center, center])
                total = float(np.sum(exponents))
                product_center = np.einsum("i,ix->x", exponents, centers) / total
                decay = np.exp(
                    -np.einsum("i,ix,ix->", exponents, centers, centers)
                    + total * float(np.dot(product_center, product_center))
                )
                width = 1.0 / np.sqrt(total)
                polynomial = (
                    (np.linalg.norm(product_center - left_center) + width) ** angular_left
                    * (np.linalg.norm(product_center - right_center) + width) ** angular_right
                )
                radial = 1.0
                reduced_radius = (
                    np.linalg.norm(product_center - center) + width
                ) / float(pseudo.local_radius)
                for power, coefficient in enumerate(pseudo.local_coefficients):
                    radial += abs(float(coefficient)) * reduced_radius ** (2 * power)
                bound += (
                    abs(float(weight_a) * float(weight_b))
                    * decay
                    * polynomial
                    * radial_scale
                    * radial
                    * (np.pi / total) ** 1.5
                )
        return float(bound)

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

    def _periodic_pair_ft_batch_many(self, gvecs, kvecs):
        gvecs = np.asarray(gvecs, dtype=float)
        kvecs = np.asarray(kvecs, dtype=float)
        if gvecs.ndim != 2 or gvecs.shape[1] != 3:
            raise ValueError("gvecs must have shape (ng, 3).")
        if kvecs.ndim == 1:
            kvecs = kvecs.reshape(1, 3)
        if kvecs.ndim != 2 or kvecs.shape[1] != 3:
            raise ValueError("kvecs must have shape (nk, 3).")
        if self._pair_ft_block_plan is None:
            return np.stack(
                [self._periodic_pair_ft_batch(gvecs, kvec) for kvec in kvecs],
                axis=0,
            )

        phases = np.ascontiguousarray(
            np.exp(1.0j * (kvecs @ self._pair_ft_shift_array.T)),
            dtype=np.complex128,
        )
        return self._pair_ft_block_plan.periodic_sum_many(
            gvecs,
            left_origins=self._pair_ft_origins,
            right_origins_batch=self._pair_ft_right_origins_batch,
            phases=phases,
            image_pair_mask=self._pair_ft_image_pair_mask,
            pair_image_starts=self._pair_ft_pair_image_starts,
            pair_image_indices=self._pair_ft_pair_image_indices,
            primitive_terms=self._pair_ft_primitive_terms,
            compiled=True,
            threads=self.one_body_workers,
        )

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
        reciprocal_domain = (
            self.recip_cut if self.recip_bounds is None else self.recip_bounds
        )
        values = [
            (gvec, 1.0 / volume)
            for _h, _k, _l, gvec in reciprocal_vectors(
                lattice,
                reciprocal_domain,
                include_zero=include_zero,
            )
        ]
        if self._g_weight_cache is not None:
            self._g_weight_cache[cache_key] = values
        return values

    def _reciprocal_nuclear_attraction(self, kvec):
        return self._reciprocal_nuclear_attraction_many(
            np.asarray(kvec, dtype=float).reshape(1, 3)
        )[0]

    def _reciprocal_nuclear_attraction_many(self, kvecs):
        kvecs = np.asarray(kvecs, dtype=float)
        if kvecs.ndim == 1:
            kvecs = kvecs.reshape(1, 3)
        if kvecs.ndim != 2 or kvecs.shape[1] != 3:
            raise ValueError("kvecs must have shape (nk, 3).")
        charges = self.cell.ionic_charges
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        mats = np.zeros(
            (len(kvecs), len(self._basis), len(self._basis)),
            dtype=np.complex128,
        )
        values = self._reciprocal_g_weights()
        if not values:
            return mats
        gvecs = np.asarray([gvec for gvec, _weight in values], dtype=float)
        weights = np.asarray([weight for _gvec, weight in values], dtype=float)
        g2 = np.einsum("gi,gi->g", gvecs, gvecs)
        mask = g2 > 0.0
        if not np.any(mask):
            return mats
        gvecs = gvecs[mask]
        weights = weights[mask]
        g2 = g2[mask]
        damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
        rho_nuc = np.einsum("a,ag->g", charges, np.exp(-1j * (coords @ gvecs.T)))
        pair_plus_g = self._periodic_pair_ft_batch_many(-gvecs, kvecs)
        coeff = -(4.0 * np.pi) * weights * damping * rho_nuc / g2
        mats += np.einsum("g,kgpq->kpq", coeff, pair_plus_g, optimize=True)
        return np.asarray([_symmetrize(mat) for mat in mats])

    def _local_pseudopotential(self, kvec):
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        if not self.cell.has_pseudo:
            return mat
        return self._fourier_sum(self._vpp_local_r, kvec)

    def _nonlocal_pseudopotential(self, kvec):
        mat = np.zeros((len(self._basis), len(self._basis)), dtype=np.complex128)
        if not self._pseudo_projector_terms:
            return mat
        phases = np.exp(1j * (self._pseudo_shift_vectors @ np.asarray(kvec, dtype=float)))
        for coupling, image_overlaps in self._pseudo_projector_terms:
            bloch_overlaps = np.einsum(
                "r,rimp->imp",
                phases,
                image_overlaps,
                optimize=True,
            )
            mat += np.einsum(
                "imp,ij,jmq->pq",
                bloch_overlaps.conj(),
                coupling,
                bloch_overlaps,
                optimize=True,
            )
        return _symmetrize(mat)

    def _nuclear_background_hcore(self, overlap):
        if self._use_inf_vacuum_1d() or not self.nuclear_background:
            return np.zeros_like(overlap, dtype=np.complex128)
        volume = abs(float(np.linalg.det(np.asarray(self.cell.lattice_vectors, dtype=float))))
        q_nuc = float(np.sum(self.cell.ionic_charges))
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
        target = 1.0e-12
        decay = math.sqrt(-math.log(target))
        direct_scale = float(np.min(np.linalg.svd(probe_lattice, compute_uv=False)))
        probe_recip = 2.0 * np.pi * np.linalg.inv(probe_lattice).T
        reciprocal_scale = float(
            np.min(np.linalg.svd(probe_recip, compute_uv=False))
        )
        real_cut = max(
            self.real_cut,
            int(math.ceil(decay / (self.eta * direct_scale))),
        )
        recip_cut = max(
            self.recip_cut,
            int(math.ceil(2.0 * self.eta * decay / reciprocal_scale)),
        )
        energy = ewald_nuclear_repulsion(
            np.asarray([1.0]),
            np.zeros((1, 3), dtype=float),
            probe_lattice,
            eta=self.eta,
            real_cut=real_cut,
            recip_cut=recip_cut,
            neutralizing_background=True,
        )
        return -2.0 * energy

    def _build_integrals(self):
        total_started = time.perf_counter()
        setup_started = time.perf_counter()
        self._validate()
        setup_changed = self._periodic_setup()
        setup_seconds = time.perf_counter() - setup_started
        one_body_started = time.perf_counter()
        if setup_changed or self._s_r is None:
            self._build_one_body_blocks()
            one_body_reused = False
        else:
            one_body_reused = True
        one_body_seconds = time.perf_counter() - one_body_started

        missing_indices = [
            index
            for index, kvec in enumerate(self.kpts)
            if self._one_body_cache is None
            or self._vector_key(kvec) not in self._one_body_cache
        ]
        reciprocal_nuclear_by_index = {}
        reciprocal_started = time.perf_counter()
        if missing_indices:
            missing_kpts = np.asarray(self.kpts, dtype=float)[missing_indices]
            reciprocal_nuclear_by_index.update(
                zip(
                    missing_indices,
                    self._reciprocal_nuclear_attraction_many(missing_kpts),
                )
            )
        reciprocal_seconds = time.perf_counter() - reciprocal_started
        assembly_started = time.perf_counter()
        overlap_k = []
        hcore_k = []
        for index, kvec in enumerate(self.kpts):
            overlap, hcore = self._one_body_at_k(
                kvec,
                reciprocal_nuclear=reciprocal_nuclear_by_index.get(index),
            )
            overlap_k.append(overlap)
            hcore_k.append(hcore)
        assembly_seconds = time.perf_counter() - assembly_started

        electronic_started = time.perf_counter()
        if self.jk_builder in ("reciprocal", "gdf"):
            self.eri = None
            self._exchange_eri_k = None
        else:
            eri = self._periodic_short_range_eri()
            eri += self._periodic_reciprocal_eri()
            eri += self._coulomb_background_eri()
            self.eri = np.asarray(eri, dtype=float)
            self._build_exchange_eri_k()
        electronic_seconds = time.perf_counter() - electronic_started
        self._overlap_k = overlap_k
        self._hcore_k = hcore_k
        self.integral_build_timings = {
            "periodic_setup_seconds": float(setup_seconds),
            "reciprocal_domain_seconds": float(self._reciprocal_domain_seconds),
            "one_body_real_space_seconds": float(one_body_seconds),
            "one_body_reused": bool(one_body_reused),
            "reciprocal_nuclear_seconds": float(reciprocal_seconds),
            "one_body_k_assembly_seconds": float(assembly_seconds),
            "electronic_eri_seconds": float(electronic_seconds),
            "total_seconds": float(time.perf_counter() - total_started),
            "real_image_count": int(len(self._shift_keys)),
            "pair_image_count": int(len(self._pair_shift_keys)),
            "one_body_pair_image_count": int(self._one_body_image_pair_count),
            "pair_ft_term_count": int(len(self._pair_ft_terms)),
            "recip_cut": int(self.recip_cut),
            "recip_bounds": self.recip_bounds,
            "recip_cut_auto": self.recip_auto_info["mode"] == "auto",
            "recip_precision": float(self.recip_precision),
            "recip_auto_base_cut": self.recip_auto_info.get("base_cut"),
            "recip_auto_kpoint_pad": self.recip_auto_info.get("kpoint_pad", 0),
            "recip_auto_shell_bound": self.recip_auto_info.get("shell_bound"),
            "recip_auto_estimated_tail": self.recip_auto_info.get("estimated_tail"),
        }
        self.overlap = overlap_k[0] if self.nkpts == 1 else overlap_k
        self.hcore = hcore_k[0] if self.nkpts == 1 else hcore_k

        if self._use_inf_vacuum_1d():
            self.e_nuc = float(
                ewald_nuclear_repulsion_1d_inf_vacuum(
                    self.cell.ionic_charges,
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

    def _one_body_at_k(self, kvec, reciprocal_nuclear=None):
        kvec = np.asarray(kvec, dtype=float)
        cache_key = self._vector_key(kvec)
        if self._one_body_cache is not None and cache_key in self._one_body_cache:
            return self._one_body_cache[cache_key]
        overlap = self._fourier_sum(self._s_r, kvec)
        kinetic = self._fourier_sum(self._t_r, kvec)
        vne = self._fourier_sum(self._vne_sr_r, kvec)
        if reciprocal_nuclear is None:
            reciprocal_nuclear = self._reciprocal_nuclear_attraction(kvec)
        vne += reciprocal_nuclear
        vne += self._local_pseudopotential(kvec)
        vne += self._nonlocal_pseudopotential(kvec)
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

        if self.occupation_mode == "aufbau":
            for _energy, ik, ib in flat[:total_pairs]:
                mo_occ[ik][ib] = 2.0
            return mo_occ

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
        if self.jk_builder == "gdf":
            if self.with_df is None:
                from pyqed.qchem.pbc.gdf import PeriodicGDF

                self.with_df = PeriodicGDF(self)
            vj, vk = self.with_df.get_jk(dm_k)
            fock_k = []
            for k_index, hcore in enumerate(self._hcore_k):
                veff_ao = vj[k_index] - 0.5 * vk[k_index]
                if self.madelung is not None:
                    overlap = self._overlap_k[k_index]
                    exchange_correction = self.madelung * (
                        overlap @ dm_k[k_index] @ overlap
                    )
                    veff_ao -= 0.5 * exchange_correction
                fock_k.append(_symmetrize(hcore + veff_ao))
            return fock_k

        fock_k = []
        for k_index, hcore in enumerate(self._hcore_k):
            fock_k.append(_symmetrize(hcore + self._get_veff_k(dm_k, k_index)))
        return fock_k

    def density_fit(self, auxbasis=None, **kwargs):
        """Attach the native periodic GDF backend and use it for SCF J/K."""

        from pyqed.qchem.pbc.gdf import PeriodicGDF

        self.jk_builder = "gdf"
        self.with_df = PeriodicGDF(self, auxbasis=auxbasis, **kwargs)
        return self

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
        if self.jk_builder == "gdf" and exchange_key in ("average", "finite_q"):
            raise NotImplementedError(
                "GDF band Fock matrices are available on the self-consistent k mesh. "
                "Use exchange='mesh' there or exchange='mesh_interpolate' between "
                "mesh points."
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
                    if self.jk_builder == "gdf":
                        overlap = self._overlap_k[match]
                        fock = self.fock if self.nkpts == 1 else self.fock[match]
                        veff = None
                    else:
                        veff = self._get_veff_k(dm_k, match)
                else:
                    veff = self._get_veff_at_k(dm_k, kvec, overlap)
                if veff is not None:
                    fock = _symmetrize(hcore + veff)
                else:
                    fock = _symmetrize(fock)
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

    def get_init_guess(self, key="auto"):
        """Build a core or valence-configuration SAD density guess."""
        key = str(key).lower().replace("-", "_")
        if key == "auto":
            key = "pseudo_sad" if self.cell.has_pseudo else "core"
        if key == "core":
            return self._solve_fock(self._hcore_k, self._overlap_k)[3]
        if key not in ("pseudo_sad", "sad"):
            raise ValueError("init_guess must be 'auto', 'core', or 'pseudo_sad'.")
        if not self.cell.has_pseudo or any(
            pseudo is None for pseudo in self.cell._pseudos_by_atom
        ):
            raise ValueError("pseudo_sad requires a pseudopotential on every atom.")

        diagonal = np.zeros(self.cell.nao, dtype=float)
        for center, pseudo in zip(
            self.cell._atom_coords,
            self.cell._pseudos_by_atom,
        ):
            center = np.asarray(center, dtype=float)
            for angular_momentum, electron_count in enumerate(
                pseudo.valence_configuration
            ):
                if electron_count == 0.0:
                    continue
                indices = [
                    ao
                    for ao, function in enumerate(self._basis)
                    if int(sum(function.shell)) == angular_momentum
                    and np.linalg.norm(np.asarray(function.origin) - center) < 1.0e-9
                ]
                if not indices:
                    raise ValueError(
                        f"No l={angular_momentum} AO functions are available for "
                        f"the {pseudo.symbol} pseudopotential SAD guess."
                    )
                diagonal[indices] += float(electron_count) / len(indices)

        guess = np.diag(diagonal).astype(np.complex128)
        guesses = []
        for overlap in self._overlap_k:
            electron_count = np.trace(guess @ overlap).real
            if electron_count <= 0.0:
                raise ValueError("The pseudopotential SAD density has zero population.")
            guesses.append(guess * (float(self.cell.nelectron) / electron_count))
        return guesses

    def run(
        self,
        max_cycle=50,
        conv_tol=1e-8,
        conv_tol_dm=1e-6,
        dm0=None,
        init_guess="auto",
    ):
        self._build_integrals()
        mo_energy, mo_coeff, mo_occ, core_dm_k = self._solve_fock(
            self._hcore_k,
            self._overlap_k,
        )
        if dm0 is None:
            guess_key = str(init_guess).lower().replace("-", "_")
            if guess_key == "core" or (guess_key == "auto" and not self.cell.has_pseudo):
                dm_k = core_dm_k
            else:
                dm_k = self.get_init_guess(init_guess)
        else:
            if str(init_guess).lower() != "auto":
                raise ValueError("Specify either dm0 or a non-default init_guess, not both.")
            if self.nkpts == 1 and np.asarray(dm0).ndim == 2:
                dm_k = [np.asarray(dm0, dtype=np.complex128)]
            else:
                dm_k = [np.asarray(dm, dtype=np.complex128) for dm in dm0]
            if len(dm_k) != self.nkpts:
                raise ValueError("dm0 must provide one AO density matrix per k-point.")
            for density in dm_k:
                if density.shape != (self.cell.nao, self.cell.nao):
                    raise ValueError("Each dm0 matrix must have shape (nao, nao).")
            dm_k = [_symmetrize(density) for density in dm_k]
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        e_last = None
        converged = False
        fock_k = self._hcore_k
        diis = (
            _KPointDIIS(self.diis_space, self.diis_start_cycle)
            if self.diis
            else None
        )
        self.scf_history = []
        for cycle in range(int(max_cycle)):
            fock_k = self._build_fock_k(dm_k)
            diis_error = None
            if diis is not None:
                fock_k, diis_error = diis.update(
                    fock_k,
                    dm_k,
                    self._overlap_k,
                    cycle,
                )
            mo_energy_new, mo_coeff_new, mo_occ_new, dm_k_new = self._solve_fock(
                fock_k,
                self._overlap_k,
            )
            if (
                cycle > 0
                and self.damping > 0.0
                and (diis is None or cycle < self.diis_start_cycle)
            ):
                dm_k_new = [
                    (1.0 - self.damping) * d_old + self.damping * d_new
                    for d_old, d_new in zip(dm_k, dm_k_new)
                ]

            e_elec = self._electronic_energy(dm_k_new, fock_k)
            de = None if e_last is None else abs(e_elec - e_last)
            ddm = _matrix_norm(dm_k_new, dm_k)
            self.scf_history.append(
                {
                    "cycle": int(cycle + 1),
                    "energy": float(e_elec),
                    "delta_energy": None if de is None else float(de),
                    "density_norm": float(ddm),
                    "diis_error": None if diis_error is None else float(diis_error),
                }
            )
            if e_last is not None:
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
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ

        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
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
        self.niter = int(len(self.scf_history))
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

    def nuc_grad_method(self):
        """Return the native fixed-cell KRHF nuclear-gradient driver."""
        from pyqed.qchem.pbc.grad import KRHFGradients

        return KRHFGradients(self)

    def CPHF(self):
        """Return the native static periodic CPHF response driver."""
        from pyqed.qchem.pbc.scf import KRHFResponse

        return KRHFResponse(self)

    def response(self):
        """Return the native static periodic CPHF response driver."""
        return self.CPHF()

    def Hessian(self):
        """Return the CPHF-relaxed fixed-cell periodic Hessian driver."""
        from pyqed.qchem.pbc.hessian import KRHFHessian

        return KRHFHessian(self)

    def forces(self, atmlst=None):
        """Return native fixed-cell nuclear forces in Hartree/Bohr."""
        return self.nuc_grad_method().forces(atmlst=atmlst)


KRHF = EwaldRHF
