"""Analytic fixed-cell gradients for native Ewald KRHF."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyqed.qchem.basis import S, T
from pyqed.qchem.basis_derivatives import (
    _atom_ids_for_basis,
    _derivative_signatures,
)
from pyqed.qchem.fourier import (
    AOBlockPairFTPlan,
    gaussian_pair_ft_batch,
    has_periodic_pair_ft_backend,
)
from pyqed.qchem.pbc.ewald import (
    ewald_nuclear_gradient,
    short_range_eri_s,
    short_range_point_charge_s,
    short_range_three_center_eri,
    short_range_two_center_coulomb,
)
from pyqed.qchem.pbc.hf.ewald_rhf import (
    _gaussian_pair_ft_decay_bound,
    _shifted_gaussian,
    _symmetrize,
)
from pyqed.qchem.pbc.pseudo import local_gaussian_overlap, projector_overlap


def _gaussian_from_signature(template, signature):
    shell, origin, exps, weights = signature
    function = object.__new__(template.__class__)
    function.__dict__ = dict(template.__dict__)
    function.shell = tuple(int(value) for value in shell)
    function.origin = np.asarray(origin, dtype=float)
    function.exps = np.asarray(exps, dtype=float)
    function.prim_weights = np.asarray(weights, dtype=float)
    return function


class Gradients:
    """Analytic nuclear gradients for the native Ewald KRHF energy."""

    def __init__(self, mean_field):
        self.base = mean_field
        self.cell = mean_field.cell
        self.de = None
        self.de_elec = None
        self.de_nuc = None
        self.components = None
        self.timings = {}
        self.gdf_response_info = None
        self._basis_derivative_cache = {}
        self._basis_second_derivative_cache = {}
        self._pair_ft_derivative_cache = {}
        self._pair_ft_second_derivative_cache = {}
        self._pair_ft_gradient_data = {}
        self._pair_ft_hessian_data = {}
        self._gdf_raw_response_cache = None
        self._atom_ids = None

    def _validate(self, *, require_scf=True):
        mf = self.base
        if require_scf:
            if getattr(mf, "dm", None) is None:
                raise ValueError("Run the KRHF calculation before requesting gradients.")
            if not getattr(mf, "converged", False):
                raise RuntimeError("Analytic gradients require a converged KRHF reference.")
        if int(self.cell.dimension) != 3:
            raise NotImplementedError("Native KRHF gradients currently require dimension=3.")
        if int(mf.nkpts) != 1:
            raise NotImplementedError(
                "Native KRHF gradients currently support one k point only."
            )
        if str(mf.jk_builder) not in ("ewald", "reciprocal", "gdf"):
            raise NotImplementedError(
                "Native KRHF gradients require an Ewald, reciprocal, or GDF J/K builder."
            )
        if str(mf.jk_builder) == "ewald" and mf.eri is None:
            raise RuntimeError("The Ewald ERI tensor is unavailable.")
        if self.cell.has_pseudo and str(mf.jk_builder) not in ("reciprocal", "gdf"):
            raise NotImplementedError(
                "Native GTH pseudopotential gradients currently require "
                "jk_builder='reciprocal' or 'gdf'."
            )
        if str(mf.jk_builder) == "gdf":
            if mf.with_df is None:
                raise RuntimeError("The periodic GDF backend is unavailable.")
            from pyqed.pbc.gw.integrals import _gdf_backend_settings

            kernel = _gdf_backend_settings(mf.with_df._space.reference)[4]
            if kernel not in ("full", "range_separated"):
                raise NotImplementedError(
                    "Native GDF gradients require a full or range-separated "
                    "Coulomb kernel."
                )
        self._atom_ids = _atom_ids_for_basis(
            mf._basis,
            np.asarray(self.cell._atom_coords, dtype=float),
        )

    def _basis_derivatives(self, function, axis):
        key = (
            tuple(int(value) for value in function.shell),
            np.asarray(function.origin, dtype=float).tobytes(),
            np.asarray(function.exps, dtype=float).tobytes(),
            np.asarray(function.prim_weights, dtype=float).tobytes(),
            int(axis),
        )
        cached = self._basis_derivative_cache.get(key)
        if cached is None:
            order = [0, 0, 0]
            order[int(axis)] = 1
            cached = tuple(
                _gaussian_from_signature(function, signature)
                for signature in _derivative_signatures(function, tuple(order))
            )
            self._basis_derivative_cache[key] = cached
        return cached

    def _pair_center_derivative(self, kernel, left, right, axis, slot):
        if slot == 0:
            return sum(kernel(function, right) for function in self._basis_derivatives(left, axis))
        return sum(kernel(left, function) for function in self._basis_derivatives(right, axis))

    def _basis_second_derivatives(self, function, axis_a, axis_b):
        key = (
            tuple(int(value) for value in function.shell),
            np.asarray(function.origin, dtype=float).tobytes(),
            np.asarray(function.exps, dtype=float).tobytes(),
            np.asarray(function.prim_weights, dtype=float).tobytes(),
            int(axis_a),
            int(axis_b),
        )
        cached = self._basis_second_derivative_cache.get(key)
        if cached is None:
            order = [0, 0, 0]
            order[int(axis_a)] += 1
            order[int(axis_b)] += 1
            cached = tuple(
                _gaussian_from_signature(function, signature)
                for signature in _derivative_signatures(function, tuple(order))
            )
            self._basis_second_derivative_cache[key] = cached
        return cached

    def _pair_second_center_derivative(
        self,
        kernel,
        left,
        right,
        axis_a,
        slot_a,
        axis_b,
        slot_b,
    ):
        if slot_a == slot_b == 0:
            return sum(
                kernel(function, right)
                for function in self._basis_second_derivatives(
                    left,
                    axis_a,
                    axis_b,
                )
            )
        if slot_a == slot_b == 1:
            return sum(
                kernel(left, function)
                for function in self._basis_second_derivatives(
                    right,
                    axis_a,
                    axis_b,
                )
            )
        if slot_a == 0:
            left_derivatives = self._basis_derivatives(left, axis_a)
            right_derivatives = self._basis_derivatives(right, axis_b)
        else:
            left_derivatives = self._basis_derivatives(left, axis_b)
            right_derivatives = self._basis_derivatives(right, axis_a)
        return sum(
            kernel(left_derivative, right_derivative)
            for left_derivative in left_derivatives
            for right_derivative in right_derivatives
        )

    def _pair_ft_derivatives(self, gvec):
        vector = np.asarray(gvec, dtype=float)
        vector = np.where(np.abs(vector) < 1.0e-14, 0.0, vector)
        cache_key = tuple(np.round(vector, 12))
        cached = self._pair_ft_derivative_cache.get(cache_key)
        if cached is not None:
            return cached
        out = self._pair_ft_derivatives_many(vector[None, :])[:, :, 0]
        self._pair_ft_derivative_cache[cache_key] = out
        return out

    def _pair_ft_derivatives_many(self, gvecs):
        mf = self.base
        gvecs = np.asarray(gvecs, dtype=float)
        if gvecs.ndim != 2 or gvecs.shape[1] != 3:
            raise ValueError("gvecs must have shape (ng, 3).")
        if mf._pair_ft_block_plan is not None and has_periodic_pair_ft_backend():
            return self._pair_ft_derivatives_many_compiled(gvecs)
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, len(gvecs), nao, nao), dtype=np.complex128)
        for shift, p, q, left, right in mf._pair_ft_terms:
            phase = np.exp(1.0j * np.dot(mf.kpts[0], shift))
            for axis in range(3):
                atom_p = int(self._atom_ids[p])
                atom_q = int(self._atom_ids[q])
                out[atom_p, axis, :, p, q] += phase * sum(
                    (
                        gaussian_pair_ft_batch(function, right, gvecs)
                        for function in self._basis_derivatives(left, axis)
                    ),
                    start=np.zeros(len(gvecs), dtype=np.complex128),
                )
                out[atom_q, axis, :, p, q] += phase * sum(
                    (
                        gaussian_pair_ft_batch(left, function, gvecs)
                        for function in self._basis_derivatives(right, axis)
                    ),
                    start=np.zeros(len(gvecs), dtype=np.complex128),
                )
        return out

    def _pair_ft_second_derivatives_many(self, gvecs):
        mf = self.base
        gvecs = np.asarray(gvecs, dtype=float)
        if gvecs.ndim != 2 or gvecs.shape[1] != 3:
            raise ValueError("gvecs must have shape (ng, 3).")
        if mf._pair_ft_block_plan is not None and has_periodic_pair_ft_backend():
            return self._pair_ft_second_derivatives_many_compiled(gvecs)
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros(
            (natom, 3, natom, 3, len(gvecs), nao, nao),
            dtype=np.complex128,
        )
        for shift, p, q, left, right in mf._pair_ft_terms:
            phase = np.exp(1.0j * np.dot(mf.kpts[0], shift))
            atoms = (int(self._atom_ids[p]), int(self._atom_ids[q]))
            for slot_a in range(2):
                atom_a = atoms[slot_a]
                for slot_b in range(2):
                    atom_b = atoms[slot_b]
                    for axis_a in range(3):
                        for axis_b in range(3):
                            value = self._pair_second_center_derivative(
                                lambda a, b: gaussian_pair_ft_batch(a, b, gvecs),
                                left,
                                right,
                                axis_a,
                                slot_a,
                                axis_b,
                                slot_b,
                            )
                            out[
                                atom_a,
                                axis_a,
                                atom_b,
                                axis_b,
                                :,
                                p,
                                q,
                            ] += phase * value
        return out

    def _build_pair_ft_hessian_data(self):
        mf = self.base
        plan_key = ("scf", id(mf._pair_ft_block_plan))
        cached = self._pair_ft_hessian_data.get(plan_key)
        if cached is not None:
            return cached

        first = self._build_pair_ft_gradient_data()
        second_basis = []
        second_parents = []
        second_axes_a = []
        second_axes_b = []
        for parent, function in enumerate(mf._basis):
            for axis_a in range(3):
                for axis_b in range(3):
                    for derivative in self._basis_second_derivatives(
                        function,
                        axis_a,
                        axis_b,
                    ):
                        second_basis.append(derivative)
                        second_parents.append(parent)
                        second_axes_a.append(axis_a)
                        second_axes_b.append(axis_b)

        second_basis = tuple(second_basis)
        second_parents = np.asarray(second_parents, dtype=int)
        second_axes_a = np.asarray(second_axes_a, dtype=int)
        second_axes_b = np.asarray(second_axes_b, dtype=int)
        second_origins = np.ascontiguousarray(
            [function.origin for function in second_basis],
            dtype=float,
        )
        right_second_origins = (
            second_origins[None, :, :]
            + first["shift_array"][:, None, :]
        )
        nimage = len(first["shift_array"])
        nao = len(mf._basis)
        base_mask = mf._pair_ft_image_pair_mask.reshape(nimage, nao, nao)
        left_second_mask = np.ascontiguousarray(
            np.take(base_mask, second_parents, axis=1).reshape(nimage, -1)
        )
        right_second_mask = np.ascontiguousarray(
            np.take(base_mask, second_parents, axis=2).reshape(nimage, -1)
        )
        cross_mask = np.ascontiguousarray(
            np.take(
                np.take(base_mask, first["parents"], axis=1),
                first["parents"],
                axis=2,
            ).reshape(nimage, -1)
        )

        left_second_plan = AOBlockPairFTPlan(second_basis, mf._basis)
        right_second_plan = AOBlockPairFTPlan(mf._basis, second_basis)
        cross_plan = AOBlockPairFTPlan(first["basis"], first["basis"])
        left_second_terms = left_second_plan.periodic_primitive_terms(
            second_origins,
            first["base_right_origins"],
            image_pair_mask=left_second_mask,
        )
        right_second_terms = right_second_plan.periodic_primitive_terms(
            first["base_origins"],
            right_second_origins,
            image_pair_mask=right_second_mask,
        )
        cross_terms = cross_plan.periodic_primitive_terms(
            first["origins"],
            first["right_origins"],
            image_pair_mask=cross_mask,
        )
        data = {
            "first": first,
            "second_basis": second_basis,
            "second_parents": second_parents,
            "second_axes_a": second_axes_a,
            "second_axes_b": second_axes_b,
            "second_origins": second_origins,
            "right_second_origins": right_second_origins,
            "left_second_mask": left_second_mask,
            "right_second_mask": right_second_mask,
            "cross_mask": cross_mask,
            "left_second_plan": left_second_plan,
            "right_second_plan": right_second_plan,
            "cross_plan": cross_plan,
            "left_second_terms": left_second_terms,
            "right_second_terms": right_second_terms,
            "cross_terms": cross_terms,
        }
        self._pair_ft_hessian_data[plan_key] = data
        return data

    def _pair_ft_second_derivatives_many_compiled(self, gvecs):
        mf = self.base
        data = self._build_pair_ft_hessian_data()
        first = data["first"]
        phases = np.exp(
            1.0j
            * (
                np.asarray(mf.kpts[0], dtype=float).reshape(1, 3)
                @ first["shift_array"].T
            )
        )
        common = {
            "gvecs": gvecs,
            "phases": phases,
            "compiled": True,
            "threads": mf.one_body_workers,
        }
        left_second = data["left_second_plan"].periodic_sum_many(
            left_origins=data["second_origins"],
            right_origins_batch=first["base_right_origins"],
            image_pair_mask=data["left_second_mask"],
            primitive_terms=data["left_second_terms"],
            **common,
        )[0]
        right_second = data["right_second_plan"].periodic_sum_many(
            left_origins=first["base_origins"],
            right_origins_batch=data["right_second_origins"],
            image_pair_mask=data["right_second_mask"],
            primitive_terms=data["right_second_terms"],
            **common,
        )[0]
        cross = data["cross_plan"].periodic_sum_many(
            left_origins=first["origins"],
            right_origins_batch=first["right_origins"],
            image_pair_mask=data["cross_mask"],
            primitive_terms=data["cross_terms"],
            **common,
        )[0]

        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros(
            (natom, 3, natom, 3, len(gvecs), nao, nao),
            dtype=np.complex128,
        )
        for derivative, (parent, axis_a, axis_b) in enumerate(
            zip(
                data["second_parents"],
                data["second_axes_a"],
                data["second_axes_b"],
            )
        ):
            atom = int(self._atom_ids[parent])
            out[atom, axis_a, atom, axis_b, :, parent, :] += left_second[
                :, derivative, :
            ]
            out[atom, axis_a, atom, axis_b, :, :, parent] += right_second[
                :, :, derivative
            ]

        for left_derivative, (left_parent, left_axis) in enumerate(
            zip(first["parents"], first["axes"])
        ):
            left_atom = int(self._atom_ids[left_parent])
            for right_derivative, (right_parent, right_axis) in enumerate(
                zip(first["parents"], first["axes"])
            ):
                right_atom = int(self._atom_ids[right_parent])
                value = cross[:, left_derivative, right_derivative]
                out[
                    left_atom,
                    left_axis,
                    right_atom,
                    right_axis,
                    :,
                    left_parent,
                    right_parent,
                ] += value
                out[
                    right_atom,
                    right_axis,
                    left_atom,
                    left_axis,
                    :,
                    left_parent,
                    right_parent,
                ] += value
        return out

    def _build_pair_ft_gradient_data(self, plan_data=None):
        mf = self.base
        if plan_data is None:
            plan_key = ("scf", id(mf._pair_ft_block_plan))
            origins = mf._pair_ft_origins
            shift_array = mf._pair_ft_shift_array
            right_origins_batch = mf._pair_ft_right_origins_batch
            image_pair_mask = mf._pair_ft_image_pair_mask
            coeff_tol = 0.0
            factor_screen_tol = 0.0
        else:
            plan_key = ("gdf", id(plan_data["plan"]))
            origins = np.asarray(plan_data["origins"], dtype=float)
            shift_array = np.asarray(plan_data["shift_array"], dtype=float)
            right_origins_batch = np.asarray(
                plan_data["right_origins_batch"], dtype=float
            )
            image_pair_mask = np.asarray(
                plan_data["image_pair_mask"], dtype=np.bool_
            )
            coeff_tol = float(plan_data.get("pair_ft_coeff_tol", 0.0))
            factor_screen_tol = float(
                plan_data.get("pair_ft_factor_screen_tol", 0.0)
            )
        cached = self._pair_ft_gradient_data.get(plan_key)
        if cached is not None:
            return cached

        derivative_basis = []
        parents = []
        axes = []
        for parent, function in enumerate(mf._basis):
            for axis in range(3):
                for derivative in self._basis_derivatives(function, axis):
                    derivative_basis.append(derivative)
                    parents.append(parent)
                    axes.append(axis)

        derivative_basis = tuple(derivative_basis)
        parents = np.asarray(parents, dtype=int)
        axes = np.asarray(axes, dtype=int)
        derivative_origins = np.ascontiguousarray(
            [function.origin for function in derivative_basis], dtype=float
        )
        right_derivative_origins = (
            derivative_origins[None, :, :]
            + shift_array[:, None, :]
        )
        base_mask = image_pair_mask.reshape(
            len(shift_array), len(mf._basis), len(mf._basis)
        )
        left_mask = np.ascontiguousarray(
            base_mask[:, parents, :].reshape(len(base_mask), -1)
        )
        right_mask = np.ascontiguousarray(
            base_mask[:, :, parents].reshape(len(base_mask), -1)
        )
        left_plan = AOBlockPairFTPlan(derivative_basis, mf._basis)
        right_plan = AOBlockPairFTPlan(mf._basis, derivative_basis)
        left_terms = left_plan.periodic_primitive_terms(
            derivative_origins,
            right_origins_batch,
            image_pair_mask=left_mask,
            coeff_tol=coeff_tol,
        )
        right_terms = right_plan.periodic_primitive_terms(
            origins,
            right_derivative_origins,
            image_pair_mask=right_mask,
            coeff_tol=coeff_tol,
        )
        left_terms["factor_screen_tol"] = factor_screen_tol
        right_terms["factor_screen_tol"] = factor_screen_tol
        data = {
            "basis": derivative_basis,
            "parents": parents,
            "axes": axes,
            "base_origins": origins,
            "shift_array": shift_array,
            "base_right_origins": right_origins_batch,
            "origins": derivative_origins,
            "right_origins": right_derivative_origins,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "left_plan": left_plan,
            "right_plan": right_plan,
            "left_terms": left_terms,
            "right_terms": right_terms,
        }
        self._pair_ft_gradient_data[plan_key] = data
        return data

    def _pair_ft_derivatives_many_compiled(self, gvecs):
        return self._pair_ft_derivatives_from_plan_many(
            gvecs,
            self.base.kpts[0],
        )

    def _pair_ft_derivatives_from_plan_many(self, gvecs, kvec, plan_data=None):
        mf = self.base
        data = self._build_pair_ft_gradient_data(plan_data)
        phases = np.exp(
            1.0j
            * (
                np.asarray(kvec, dtype=float).reshape(1, 3)
                @ data["shift_array"].T
            )
        )
        left = data["left_plan"].periodic_sum_many(
            gvecs,
            left_origins=data["origins"],
            right_origins_batch=data["base_right_origins"],
            phases=phases,
            image_pair_mask=data["left_mask"],
            primitive_terms=data["left_terms"],
            compiled=True,
            threads=mf.one_body_workers,
        )[0]
        right = data["right_plan"].periodic_sum_many(
            gvecs,
            left_origins=data["base_origins"],
            right_origins_batch=data["right_origins"],
            phases=phases,
            image_pair_mask=data["right_mask"],
            primitive_terms=data["right_terms"],
            compiled=True,
            threads=mf.one_body_workers,
        )[0]

        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, len(gvecs), nao, nao), dtype=np.complex128)
        for derivative, (parent, axis) in enumerate(
            zip(data["parents"], data["axes"])
        ):
            atom = int(self._atom_ids[parent])
            out[atom, axis, :, parent, :] += left[:, derivative, :]
            out[atom, axis, :, :, parent] += right[:, :, derivative]
        return out

    def _real_space_one_body_derivatives(self):
        try:
            from pyqed.qchem import basis as basis_module

            compiled = getattr(
                basis_module._basis_cy,
                "compute_periodic_one_electron",
                None,
            )
        except (AttributeError, ImportError):
            compiled = None
        if compiled is None:
            self.one_body_derivative_backend = "python"
            return self._real_space_one_body_derivatives_python()
        self.one_body_derivative_backend = "compiled"
        return self._real_space_one_body_derivatives_compiled(
            compiled,
            basis_module,
        )

    def _real_space_one_body_derivatives_compiled(self, compiled, basis_module):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = np.asarray(self.cell.ionic_charges, dtype=float)
        natom = len(coords)
        nao = len(mf._basis)
        shifts = np.asarray(
            [mf._shift_vectors[key] for key in mf._shift_keys],
            dtype=float,
        )
        phases = np.exp(1.0j * (shifts @ np.asarray(mf.kpts[0], dtype=float)))

        derivative_basis = []
        parents = []
        axes = []
        for parent, function in enumerate(mf._basis):
            for axis in range(3):
                for derivative in self._basis_derivatives(function, axis):
                    derivative_basis.append(derivative)
                    parents.append(parent)
                    axes.append(axis)
        derivative_basis = tuple(derivative_basis)
        parents = np.asarray(parents, dtype=int)
        axes = np.asarray(axes, dtype=int)

        base_mask = np.zeros((len(shifts), nao, nao), dtype=np.uint8)
        for p, left in enumerate(mf._basis):
            for q, right in enumerate(mf._basis):
                if mf.one_body_screen_tol == 0.0:
                    base_mask[:, p, q] = 1
                else:
                    for image, shift in enumerate(shifts):
                        base_mask[image, p, q] = (
                            _gaussian_pair_ft_decay_bound(left, right, shift)
                            > mf.one_body_screen_tol
                        )
        left_mask = np.ascontiguousarray(
            base_mask[:, parents, :],
            dtype=np.uint8,
        )
        right_mask = np.ascontiguousarray(
            base_mask[:, :, parents],
            dtype=np.uint8,
        )

        def pack(functions):
            signatures = [basis_module._basis_signature(fn) for fn in functions]
            values = basis_module._pack_signatures_for_numba(signatures)
            dtypes = (np.int64, np.float64, np.float64, np.float64, np.int64)
            return tuple(
                np.ascontiguousarray(value, dtype=dtype)
                for value, dtype in zip(values, dtypes)
            )

        base_packed = pack(mf._basis)
        derivative_packed = pack(derivative_basis)
        nuclear_image_keys = np.ascontiguousarray(
            list(self.cell.image_keys(mf.one_body_nuclear_cut)),
            dtype=np.int64,
        )
        lattice = np.ascontiguousarray(
            self.cell.lattice_vectors,
            dtype=np.float64,
        )
        sectors = {
            "left": (derivative_packed, base_packed, left_mask),
            "right": (base_packed, derivative_packed, right_mask),
        }

        def evaluate(task):
            name, atom = task
            left, right, mask = sectors[name]
            right_origins = np.ascontiguousarray(
                right[1][None, :, :] + shifts[:, None, :],
                dtype=np.float64,
            )
            values = compiled(
                left[0],
                left[1],
                right_origins,
                left[2],
                left[3],
                left[4],
                np.ascontiguousarray(coords[atom : atom + 1]),
                np.ascontiguousarray(charges[atom : atom + 1]),
                float(mf.eta),
                mask,
                0.0,
                lattice,
                nuclear_image_keys,
                right[0],
                right[2],
                right[3],
                right[4],
            )
            return name, atom, values

        tasks = [(name, atom) for name in sectors for atom in range(natom)]
        worker_count = min(max(1, int(mf.one_body_workers)), len(tasks))
        if worker_count == 1:
            evaluated = map(evaluate, tasks)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            evaluated = executor.map(evaluate, tasks)
        blocks = {name: {} for name in sectors}
        try:
            for name, atom, values in evaluated:
                blocks[name][atom] = values
        finally:
            if worker_count > 1:
                executor.shutdown()

        def bloch_sum(values):
            return np.einsum("i,ipq->pq", phases, values, optimize=True)

        left_s = bloch_sum(blocks["left"][0][0])
        left_t = bloch_sum(blocks["left"][0][1])
        right_s = bloch_sum(blocks["right"][0][0])
        right_t = bloch_sum(blocks["right"][0][1])
        left_v = {
            atom: bloch_sum(blocks["left"][atom][2])
            for atom in range(natom)
        }
        right_v = {
            atom: bloch_sum(blocks["right"][atom][2])
            for atom in range(natom)
        }

        shape = (natom, 3, nao, nao)
        s1 = np.zeros(shape, dtype=np.complex128)
        t1 = np.zeros_like(s1)
        v1 = np.zeros_like(s1)
        for derivative, (parent, axis) in enumerate(zip(parents, axes)):
            atom_p = int(self._atom_ids[parent])
            s1[atom_p, axis, parent, :] += left_s[derivative, :]
            s1[atom_p, axis, :, parent] += right_s[:, derivative]
            t1[atom_p, axis, parent, :] += left_t[derivative, :]
            t1[atom_p, axis, :, parent] += right_t[:, derivative]
            for atom_n in range(natom):
                v1[atom_p, axis, parent, :] += left_v[atom_n][derivative, :]
                v1[atom_n, axis, parent, :] -= left_v[atom_n][derivative, :]
                v1[atom_p, axis, :, parent] += right_v[atom_n][:, derivative]
                v1[atom_n, axis, :, parent] -= right_v[atom_n][:, derivative]

        for atom in range(natom):
            for axis in range(3):
                s1[atom, axis] = _symmetrize(s1[atom, axis])
                t1[atom, axis] = _symmetrize(t1[atom, axis])
                v1[atom, axis] = _symmetrize(v1[atom, axis])
        return s1, t1, v1

    def _real_space_one_body_derivatives_python(self):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = self.cell.ionic_charges
        natom = len(coords)
        nao = len(mf._basis)
        s1 = np.zeros((natom, 3, nao, nao), dtype=np.complex128)
        t1 = np.zeros_like(s1)
        v1 = np.zeros_like(s1)
        nuclear_keys = self.cell.image_keys(mf.one_body_nuclear_cut)

        for key in mf._shift_keys:
            shift = mf._shift_vectors[key]
            shifted_basis = mf._shifted_basis[key]
            phase = np.exp(1.0j * np.dot(mf.kpts[0], shift))
            for p, left in enumerate(mf._basis):
                atom_p = int(self._atom_ids[p])
                for q, right in enumerate(shifted_basis):
                    if (
                        mf.one_body_screen_tol > 0.0
                        and _gaussian_pair_ft_decay_bound(left, mf._basis[q], shift)
                        <= mf.one_body_screen_tol
                    ):
                        continue
                    atom_q = int(self._atom_ids[q])
                    for axis in range(3):
                        ds_left = self._pair_center_derivative(S, left, right, axis, 0)
                        ds_right = self._pair_center_derivative(S, left, right, axis, 1)
                        dt_left = self._pair_center_derivative(T, left, right, axis, 0)
                        dt_right = self._pair_center_derivative(T, left, right, axis, 1)
                        s1[atom_p, axis, p, q] += phase * ds_left
                        s1[atom_q, axis, p, q] += phase * ds_right
                        t1[atom_p, axis, p, q] += phase * dt_left
                        t1[atom_q, axis, p, q] += phase * dt_right

                        for nuclear_key in nuclear_keys:
                            nuclear_shift = self.cell.translation_vector(nuclear_key)
                            for atom_n, (charge, center) in enumerate(zip(charges, coords)):
                                image_center = center + nuclear_shift
                                dleft = self._pair_center_derivative(
                                    lambda a, b: short_range_point_charge_s(
                                        a, b, image_center, mf.eta
                                    ),
                                    left,
                                    right,
                                    axis,
                                    0,
                                )
                                dright = self._pair_center_derivative(
                                    lambda a, b: short_range_point_charge_s(
                                        a, b, image_center, mf.eta
                                    ),
                                    left,
                                    right,
                                    axis,
                                    1,
                                )
                                factor = phase * float(charge)
                                v1[atom_p, axis, p, q] -= factor * dleft
                                v1[atom_q, axis, p, q] -= factor * dright
                                v1[atom_n, axis, p, q] += factor * (dleft + dright)

        for atom in range(natom):
            for axis in range(3):
                s1[atom, axis] = _symmetrize(s1[atom, axis])
                t1[atom, axis] = _symmetrize(t1[atom, axis])
                v1[atom, axis] = _symmetrize(v1[atom, axis])
        return s1, t1, v1

    def _real_space_one_body_second_derivatives(self):
        try:
            from pyqed.qchem import basis as basis_module

            compiled = getattr(
                basis_module._basis_cy,
                "compute_periodic_one_electron",
                None,
            )
        except (AttributeError, ImportError):
            compiled = None
        if compiled is None:
            self.one_body_second_derivative_backend = "python"
            return self._real_space_one_body_second_derivatives_python()
        self.one_body_second_derivative_backend = "compiled"
        return self._real_space_one_body_second_derivatives_compiled(
            compiled,
            basis_module,
        )

    def _real_space_one_body_second_derivatives_compiled(
        self,
        compiled,
        basis_module,
    ):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = np.asarray(self.cell.ionic_charges, dtype=float)
        natom = len(coords)
        nao = len(mf._basis)
        shifts = np.asarray(
            [mf._shift_vectors[key] for key in mf._shift_keys],
            dtype=float,
        )
        phases = np.exp(1.0j * (shifts @ np.asarray(mf.kpts[0], dtype=float)))

        first_basis = []
        first_parents = []
        first_axes = []
        second_basis = []
        second_parents = []
        second_axes_a = []
        second_axes_b = []
        for parent, function in enumerate(mf._basis):
            for axis in range(3):
                for derivative in self._basis_derivatives(function, axis):
                    first_basis.append(derivative)
                    first_parents.append(parent)
                    first_axes.append(axis)
            for axis_a in range(3):
                for axis_b in range(3):
                    for derivative in self._basis_second_derivatives(
                        function,
                        axis_a,
                        axis_b,
                    ):
                        second_basis.append(derivative)
                        second_parents.append(parent)
                        second_axes_a.append(axis_a)
                        second_axes_b.append(axis_b)

        first_basis = tuple(first_basis)
        first_parents = np.asarray(first_parents, dtype=int)
        first_axes = np.asarray(first_axes, dtype=int)
        second_basis = tuple(second_basis)
        second_parents = np.asarray(second_parents, dtype=int)
        second_axes_a = np.asarray(second_axes_a, dtype=int)
        second_axes_b = np.asarray(second_axes_b, dtype=int)

        base_mask = np.zeros((len(shifts), nao, nao), dtype=np.uint8)
        for p, left in enumerate(mf._basis):
            for q, right in enumerate(mf._basis):
                if mf.one_body_screen_tol == 0.0:
                    base_mask[:, p, q] = 1
                else:
                    for image, shift in enumerate(shifts):
                        base_mask[image, p, q] = (
                            _gaussian_pair_ft_decay_bound(left, right, shift)
                            > mf.one_body_screen_tol
                        )
        left_second_mask = np.ascontiguousarray(
            base_mask[:, second_parents, :],
            dtype=np.uint8,
        )
        right_second_mask = np.ascontiguousarray(
            base_mask[:, :, second_parents],
            dtype=np.uint8,
        )
        cross_mask = np.ascontiguousarray(
            base_mask[:, first_parents, :][:, :, first_parents],
            dtype=np.uint8,
        )

        def pack(functions):
            signatures = [basis_module._basis_signature(fn) for fn in functions]
            values = basis_module._pack_signatures_for_numba(signatures)
            dtypes = (np.int64, np.float64, np.float64, np.float64, np.int64)
            return tuple(
                np.ascontiguousarray(value, dtype=dtype)
                for value, dtype in zip(values, dtypes)
            )

        base_packed = pack(mf._basis)
        first_packed = pack(first_basis)
        second_packed = pack(second_basis)
        nuclear_image_keys = np.ascontiguousarray(
            list(self.cell.image_keys(mf.one_body_nuclear_cut)),
            dtype=np.int64,
        )
        lattice = np.ascontiguousarray(
            self.cell.lattice_vectors,
            dtype=np.float64,
        )
        sectors = {
            "left_second": (second_packed, base_packed, left_second_mask),
            "right_second": (base_packed, second_packed, right_second_mask),
            "cross": (first_packed, first_packed, cross_mask),
        }

        def evaluate(task):
            name, atom = task
            left, right, mask = sectors[name]
            right_origins = np.ascontiguousarray(
                right[1][None, :, :] + shifts[:, None, :],
                dtype=np.float64,
            )
            values = compiled(
                left[0],
                left[1],
                right_origins,
                left[2],
                left[3],
                left[4],
                np.ascontiguousarray(coords[atom : atom + 1]),
                np.ascontiguousarray(charges[atom : atom + 1]),
                float(mf.eta),
                mask,
                0.0,
                lattice,
                nuclear_image_keys,
                right[0],
                right[2],
                right[3],
                right[4],
            )
            return name, atom, values

        tasks = [(name, atom) for name in sectors for atom in range(natom)]
        worker_count = min(max(1, int(mf.one_body_workers)), len(tasks))
        if worker_count == 1:
            evaluated = map(evaluate, tasks)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            evaluated = executor.map(evaluate, tasks)
        blocks = {name: {} for name in sectors}
        try:
            for name, atom, values in evaluated:
                blocks[name][atom] = values
        finally:
            if worker_count > 1:
                executor.shutdown()

        def bloch_sum(values):
            return np.einsum("i,ipq->pq", phases, values, optimize=True)

        left_s = bloch_sum(blocks["left_second"][0][0])
        left_t = bloch_sum(blocks["left_second"][0][1])
        right_s = bloch_sum(blocks["right_second"][0][0])
        right_t = bloch_sum(blocks["right_second"][0][1])
        cross_s = bloch_sum(blocks["cross"][0][0])
        cross_t = bloch_sum(blocks["cross"][0][1])
        left_v = {
            atom: bloch_sum(blocks["left_second"][atom][2])
            for atom in range(natom)
        }
        right_v = {
            atom: bloch_sum(blocks["right_second"][atom][2])
            for atom in range(natom)
        }
        cross_v = {
            atom: bloch_sum(blocks["cross"][atom][2])
            for atom in range(natom)
        }

        shape = (natom, 3, natom, 3, nao, nao)
        s2 = np.zeros(shape, dtype=np.complex128)
        t2 = np.zeros_like(s2)
        v2 = np.zeros_like(s2)
        for derivative, (parent, axis_a, axis_b) in enumerate(
            zip(second_parents, second_axes_a, second_axes_b)
        ):
            atom_p = int(self._atom_ids[parent])
            s2[atom_p, axis_a, atom_p, axis_b, parent, :] += left_s[
                derivative, :
            ]
            t2[atom_p, axis_a, atom_p, axis_b, parent, :] += left_t[
                derivative, :
            ]
            s2[atom_p, axis_a, atom_p, axis_b, :, parent] += right_s[
                :, derivative
            ]
            t2[atom_p, axis_a, atom_p, axis_b, :, parent] += right_t[
                :, derivative
            ]
            for atom_n in range(natom):
                left_targets = ((atom_p, 1.0), (atom_n, -1.0))
                right_targets = ((atom_p, 1.0), (atom_n, -1.0))
                for atom_a, sign_a in left_targets:
                    for atom_b, sign_b in left_targets:
                        v2[atom_a, axis_a, atom_b, axis_b, parent, :] += (
                            sign_a * sign_b * left_v[atom_n][derivative, :]
                        )
                for atom_a, sign_a in right_targets:
                    for atom_b, sign_b in right_targets:
                        v2[atom_a, axis_a, atom_b, axis_b, :, parent] += (
                            sign_a * sign_b * right_v[atom_n][:, derivative]
                        )

        for left_derivative, (left_parent, left_axis) in enumerate(
            zip(first_parents, first_axes)
        ):
            left_atom = int(self._atom_ids[left_parent])
            for right_derivative, (right_parent, right_axis) in enumerate(
                zip(first_parents, first_axes)
            ):
                right_atom = int(self._atom_ids[right_parent])
                s_value = cross_s[left_derivative, right_derivative]
                t_value = cross_t[left_derivative, right_derivative]
                s2[
                    left_atom,
                    left_axis,
                    right_atom,
                    right_axis,
                    left_parent,
                    right_parent,
                ] += s_value
                s2[
                    right_atom,
                    right_axis,
                    left_atom,
                    left_axis,
                    left_parent,
                    right_parent,
                ] += s_value
                t2[
                    left_atom,
                    left_axis,
                    right_atom,
                    right_axis,
                    left_parent,
                    right_parent,
                ] += t_value
                t2[
                    right_atom,
                    right_axis,
                    left_atom,
                    left_axis,
                    left_parent,
                    right_parent,
                ] += t_value
                for atom_n in range(natom):
                    value = cross_v[atom_n][left_derivative, right_derivative]
                    left_targets = ((left_atom, 1.0), (atom_n, -1.0))
                    right_targets = ((right_atom, 1.0), (atom_n, -1.0))
                    for atom_a, sign_a in left_targets:
                        for atom_b, sign_b in right_targets:
                            contribution = sign_a * sign_b * value
                            v2[
                                atom_a,
                                left_axis,
                                atom_b,
                                right_axis,
                                left_parent,
                                right_parent,
                            ] += contribution
                            v2[
                                atom_b,
                                right_axis,
                                atom_a,
                                left_axis,
                                left_parent,
                                right_parent,
                            ] += contribution

        transpose = (2, 3, 0, 1, 5, 4)
        s2 = 0.5 * (s2 + s2.conj().transpose(transpose))
        t2 = 0.5 * (t2 + t2.conj().transpose(transpose))
        v2 = 0.5 * (v2 + v2.conj().transpose(transpose))
        return s2, t2, v2

    def _real_space_one_body_second_derivatives_python(self):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = self.cell.ionic_charges
        natom = len(coords)
        nao = len(mf._basis)
        shape = (natom, 3, natom, 3, nao, nao)
        s2 = np.zeros(shape, dtype=np.complex128)
        t2 = np.zeros_like(s2)
        v2 = np.zeros_like(s2)
        nuclear_keys = self.cell.image_keys(mf.one_body_nuclear_cut)

        for key in mf._shift_keys:
            shift = mf._shift_vectors[key]
            shifted_basis = mf._shifted_basis[key]
            phase = np.exp(1.0j * np.dot(mf.kpts[0], shift))
            for p, left in enumerate(mf._basis):
                atom_p = int(self._atom_ids[p])
                for q, right in enumerate(shifted_basis):
                    if (
                        mf.one_body_screen_tol > 0.0
                        and _gaussian_pair_ft_decay_bound(
                            left,
                            mf._basis[q],
                            shift,
                        )
                        <= mf.one_body_screen_tol
                    ):
                        continue
                    atom_q = int(self._atom_ids[q])
                    atoms = (atom_p, atom_q)
                    for slot_a in range(2):
                        for slot_b in range(2):
                            for axis_a in range(3):
                                for axis_b in range(3):
                                    s2[
                                        atoms[slot_a],
                                        axis_a,
                                        atoms[slot_b],
                                        axis_b,
                                        p,
                                        q,
                                    ] += phase * self._pair_second_center_derivative(
                                        S,
                                        left,
                                        right,
                                        axis_a,
                                        slot_a,
                                        axis_b,
                                        slot_b,
                                    )
                                    t2[
                                        atoms[slot_a],
                                        axis_a,
                                        atoms[slot_b],
                                        axis_b,
                                        p,
                                        q,
                                    ] += phase * self._pair_second_center_derivative(
                                        T,
                                        left,
                                        right,
                                        axis_a,
                                        slot_a,
                                        axis_b,
                                        slot_b,
                                    )

                    for nuclear_key in nuclear_keys:
                        nuclear_shift = self.cell.translation_vector(nuclear_key)
                        for atom_n, (charge, center) in enumerate(
                            zip(charges, coords)
                        ):
                            image_center = center + nuclear_shift
                            operators = (
                                (atom_p, 0, 1.0),
                                (atom_n, 0, -1.0),
                                (atom_q, 1, 1.0),
                                (atom_n, 1, -1.0),
                            )
                            def kernel(a, b):
                                return short_range_point_charge_s(
                                    a,
                                    b,
                                    image_center,
                                    mf.eta,
                                )
                            factor = -phase * float(charge)
                            for atom_a, slot_a, sign_a in operators:
                                for atom_b, slot_b, sign_b in operators:
                                    for axis_a in range(3):
                                        for axis_b in range(3):
                                            v2[
                                                atom_a,
                                                axis_a,
                                                atom_b,
                                                axis_b,
                                                p,
                                                q,
                                            ] += (
                                                factor
                                                * sign_a
                                                * sign_b
                                                * self._pair_second_center_derivative(
                                                    kernel,
                                                    left,
                                                    right,
                                                    axis_a,
                                                    slot_a,
                                                    axis_b,
                                                    slot_b,
                                                )
                                            )

        transpose = (2, 3, 0, 1, 5, 4)
        s2 = 0.5 * (s2 + s2.conj().transpose(transpose))
        t2 = 0.5 * (t2 + t2.conj().transpose(transpose))
        v2 = 0.5 * (v2 + v2.conj().transpose(transpose))
        return s2, t2, v2

    def _reciprocal_nuclear_derivatives(self):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = self.cell.ionic_charges
        natom = len(coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, nao, nao), dtype=np.complex128)
        values = mf._reciprocal_g_weights()
        block_size = 128
        for start in range(0, len(values), block_size):
            block = values[start:start + block_size]
            gvecs = np.asarray([gvec for gvec, _weight in block], dtype=float)
            weights = np.asarray([weight for _gvec, weight in block], dtype=float)
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            mask = g2 > 0.0
            if not np.any(mask):
                continue
            gvecs = gvecs[mask]
            weights = weights[mask]
            g2 = g2[mask]
            nuclear_phases = np.exp(-1.0j * (coords @ gvecs.T))
            rho_nuc = np.einsum("a,ag->g", charges, nuclear_phases)
            drho = (
                -1.0j
                * charges[:, None, None]
                * nuclear_phases[:, None, :]
                * gvecs.T[None, :, :]
            )
            pair = mf._periodic_pair_ft_batch(-gvecs, mf.kpts[0])
            pair1 = self._pair_ft_derivatives_many(-gvecs)
            coefficient = (
                -(4.0 * np.pi)
                * weights
                * np.exp(-g2 / (4.0 * mf.eta * mf.eta))
                / g2
            )
            out += np.einsum(
                "g,Axg,gpq->Axpq", coefficient, drho, pair, optimize=True
            )
            out += np.einsum(
                "g,g,Axgpq->Axpq", coefficient, rho_nuc, pair1, optimize=True
            )
        for atom in range(natom):
            for axis in range(3):
                out[atom, axis] = _symmetrize(out[atom, axis])
        return out

    def _reciprocal_nuclear_second_derivatives(self):
        mf = self.base
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        charges = self.cell.ionic_charges
        natom = len(coords)
        nao = len(mf._basis)
        out = np.zeros(
            (natom, 3, natom, 3, nao, nao),
            dtype=np.complex128,
        )
        values = mf._reciprocal_g_weights()
        block_size = 128
        for start in range(0, len(values), block_size):
            block = values[start:start + block_size]
            gvecs = np.asarray([gvec for gvec, _weight in block], dtype=float)
            weights = np.asarray([weight for _gvec, weight in block], dtype=float)
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            mask = g2 > 0.0
            if not np.any(mask):
                continue
            gvecs = gvecs[mask]
            weights = weights[mask]
            g2 = g2[mask]
            nuclear_phases = np.exp(-1.0j * (coords @ gvecs.T))
            rho_nuc = np.einsum("a,ag->g", charges, nuclear_phases)
            drho = (
                -1.0j
                * charges[:, None, None]
                * nuclear_phases[:, None, :]
                * gvecs.T[None, :, :]
            )
            d2rho = np.zeros(
                (natom, 3, natom, 3, len(gvecs)),
                dtype=np.complex128,
            )
            for atom in range(natom):
                d2rho[atom, :, atom, :, :] = -(
                    float(charges[atom])
                    * nuclear_phases[atom][None, None, :]
                    * np.einsum("ga,gb->abg", gvecs, gvecs)
                )
            pair = mf._periodic_pair_ft_batch(-gvecs, mf.kpts[0])
            pair1 = self._pair_ft_derivatives_many(-gvecs)
            pair2 = self._pair_ft_second_derivatives_many(-gvecs)
            coefficient = (
                -(4.0 * np.pi)
                * weights
                * np.exp(-g2 / (4.0 * mf.eta * mf.eta))
                / g2
            )
            out += np.einsum(
                "g,AxByg,gpq->AxBypq",
                coefficient,
                d2rho,
                pair,
                optimize=True,
            )
            out += np.einsum(
                "g,Axg,Bygpq->AxBypq",
                coefficient,
                drho,
                pair1,
                optimize=True,
            )
            out += np.einsum(
                "g,Byg,Axgpq->AxBypq",
                coefficient,
                drho,
                pair1,
                optimize=True,
            )
            out += np.einsum(
                "g,g,AxBygpq->AxBypq",
                coefficient,
                rho_nuc,
                pair2,
                optimize=True,
            )
        return 0.5 * (out + out.conj().transpose(2, 3, 0, 1, 5, 4))

    def _local_pseudopotential_derivatives(self):
        mf = self.base
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, nao, nao), dtype=np.complex128)
        if not self.cell.has_pseudo:
            return out

        pseudo_shifts = [
            self.cell.translation_vector(key)
            for key in self.cell.image_keys(mf.pseudo_cut)
        ]
        for key in mf._shift_keys:
            shift = mf._shift_vectors[key]
            shifted_basis = mf._shifted_basis[key]
            phase = np.exp(1.0j * np.dot(mf.kpts[0], shift))
            for p, left in enumerate(mf._basis):
                atom_p = int(self._atom_ids[p])
                for q, right in enumerate(shifted_basis):
                    atom_q = int(self._atom_ids[q])
                    for pseudo_shift in pseudo_shifts:
                        for atom_i, pseudo in enumerate(self.cell._pseudos_by_atom):
                            if pseudo is None:
                                continue
                            center = self.cell._atom_coords[atom_i] + pseudo_shift
                            eta = 1.0 / (np.sqrt(2.0) * pseudo.local_radius)
                            include_gaussian = (
                                (
                                    mf.pair_ft_screen_tol == 0.0
                                    or _gaussian_pair_ft_decay_bound(
                                        left, mf._basis[q], shift
                                    )
                                    > mf.pair_ft_screen_tol
                                )
                                and
                                mf._local_pseudo_gaussian_bound(
                                    left, right, center, pseudo
                                )
                                > mf.pseudo_local_screen_tol
                            )

                            def kernel(a, b):
                                value = pseudo.ionic_charge * short_range_point_charge_s(
                                    a, b, center, eta
                                )
                                if include_gaussian:
                                    value += local_gaussian_overlap(a, b, center, pseudo)
                                return value

                            for axis in range(3):
                                dleft = self._pair_center_derivative(
                                    kernel, left, right, axis, 0
                                )
                                dright = self._pair_center_derivative(
                                    kernel, left, right, axis, 1
                                )
                                out[atom_p, axis, p, q] += phase * dleft
                                out[atom_q, axis, p, q] += phase * dright
                                out[atom_i, axis, p, q] -= phase * (dleft + dright)

        for atom in range(natom):
            for axis in range(3):
                out[atom, axis] = _symmetrize(out[atom, axis])
        return out

    def _nonlocal_pseudopotential_derivatives(self):
        mf = self.base
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, nao, nao), dtype=np.complex128)
        if not self.cell.has_pseudo:
            return out

        term_index = 0
        phases = np.exp(1.0j * (mf._pseudo_shift_vectors @ mf.kpts[0]))
        for atom_i, pseudo in enumerate(self.cell._pseudos_by_atom):
            if pseudo is None:
                continue
            center = np.asarray(self.cell._atom_coords[atom_i], dtype=float)
            for angular_momentum, projector in enumerate(pseudo.projectors):
                if projector.nproj == 0:
                    continue
                coupling, image_overlaps = mf._pseudo_projector_terms[term_index]
                term_index += 1
                bloch = np.einsum(
                    "r,rimp->imp", phases, image_overlaps, optimize=True
                )
                bloch1 = np.zeros(
                    (
                        natom,
                        3,
                        projector.nproj,
                        2 * angular_momentum + 1,
                        nao,
                    ),
                    dtype=np.complex128,
                )
                for image, shift in enumerate(mf._pseudo_shift_vectors):
                    phase = phases[image]
                    for ao, function in enumerate(mf._basis):
                        shifted = _shifted_gaussian(function, shift)
                        atom_ao = int(self._atom_ids[ao])
                        for projector_index in range(projector.nproj):
                            for magnetic_index, magnetic_number in enumerate(
                                range(-angular_momentum, angular_momentum + 1)
                            ):
                                for axis in range(3):
                                    derivative = sum(
                                        projector_overlap(
                                            derived,
                                            center,
                                            angular_momentum,
                                            magnetic_number,
                                            projector_index,
                                            projector.radius,
                                        )
                                        for derived in self._basis_derivatives(
                                            shifted, axis
                                        )
                                    )
                                    value = phase * derivative
                                    bloch1[
                                        atom_ao,
                                        axis,
                                        projector_index,
                                        magnetic_index,
                                        ao,
                                    ] += value
                                    bloch1[
                                        atom_i,
                                        axis,
                                        projector_index,
                                        magnetic_index,
                                        ao,
                                    ] -= value

                out += np.einsum(
                    "Aximp,ij,jmq->Axpq",
                    bloch1.conj(),
                    coupling,
                    bloch,
                    optimize=True,
                )
                out += np.einsum(
                    "imp,ij,Axjmq->Axpq",
                    bloch.conj(),
                    coupling,
                    bloch1,
                    optimize=True,
                )

        for atom in range(natom):
            for axis in range(3):
                out[atom, axis] = _symmetrize(out[atom, axis])
        return out

    def one_electron_derivatives(self):
        s1, t1, short_range = self._real_space_one_body_derivatives()
        reciprocal = self._reciprocal_nuclear_derivatives()
        volume = abs(float(np.linalg.det(self.cell.lattice_vectors)))
        background_coefficient = (
            np.pi
            * float(np.sum(self.cell.ionic_charges))
            / (self.base.eta * self.base.eta * volume)
            if self.base.nuclear_background
            else 0.0
        )
        h1 = t1 + short_range + reciprocal + background_coefficient * s1
        h1 += self._local_pseudopotential_derivatives()
        h1 += self._nonlocal_pseudopotential_derivatives()
        return s1, h1

    def one_electron_second_derivatives(self):
        """Return analytic overlap and core-Hamiltonian second derivatives."""

        if self.cell.has_pseudo:
            raise NotImplementedError(
                "Second-order periodic GTH pseudopotential derivatives are not "
                "implemented yet."
            )
        s2, t2, short_range2 = self._real_space_one_body_second_derivatives()
        reciprocal2 = self._reciprocal_nuclear_second_derivatives()
        volume = abs(float(np.linalg.det(self.cell.lattice_vectors)))
        background_coefficient = (
            np.pi
            * float(np.sum(self.cell.ionic_charges))
            / (self.base.eta * self.base.eta * volume)
            if self.base.nuclear_background
            else 0.0
        )
        h2 = t2 + short_range2 + reciprocal2 + background_coefficient * s2
        return s2, h2

    def _short_range_eri_derivatives(self):
        mf = self.base
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros((natom, 3, nao, nao, nao, nao), dtype=float)
        pair_overlap = {}
        for key in mf._shift_keys:
            shifted_basis = mf._shifted_basis[key]
            for p, left in enumerate(mf._basis):
                for q, right in enumerate(shifted_basis):
                    pair_overlap[(key, p, q)] = abs(S(left, right))
        image_pair_overlap = {}
        for r_key in mf._shift_keys:
            r_basis = mf._shifted_basis[r_key]
            for s_key in mf._shift_keys:
                s_basis = mf._shifted_basis[s_key]
                for r, left in enumerate(r_basis):
                    for s, right in enumerate(s_basis):
                        image_pair_overlap[(r_key, s_key, r, s)] = abs(S(left, right))

        for q_key in mf._shift_keys:
            q_basis = mf._shifted_basis[q_key]
            for r_key in mf._shift_keys:
                r_basis = mf._shifted_basis[r_key]
                for s_key in mf._shift_keys:
                    s_basis = mf._shifted_basis[s_key]
                    for p, fn_p in enumerate(mf._basis):
                        for q, fn_q in enumerate(q_basis):
                            pq_bound = pair_overlap[(q_key, p, q)]
                            for r, fn_r in enumerate(r_basis):
                                for s, fn_s in enumerate(s_basis):
                                    if (
                                        pq_bound
                                        * image_pair_overlap[(r_key, s_key, r, s)]
                                        < mf.eri_screen_tol
                                    ):
                                        continue
                                    functions = (fn_p, fn_q, fn_r, fn_s)
                                    indices = (p, q, r, s)
                                    for slot, (function, ao) in enumerate(
                                        zip(functions, indices)
                                    ):
                                        atom = int(self._atom_ids[ao])
                                        for axis in range(3):
                                            value = 0.0
                                            for derivative in self._basis_derivatives(
                                                function, axis
                                            ):
                                                args = list(functions)
                                                args[slot] = derivative
                                                value += short_range_eri_s(*args, mf.eta)
                                            out[atom, axis, p, q, r, s] += value
        return out

    def _reciprocal_eri_derivatives(self, s1, *, damped):
        mf = self.base
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        out = np.zeros(
            (natom, 3, nao, nao, nao, nao),
            dtype=np.complex128,
        )
        for gvec, weight in mf._reciprocal_g_weights():
            g2 = float(np.dot(gvec, gvec))
            if g2 <= 0.0:
                continue
            damping = (
                np.exp(-g2 / (4.0 * mf.eta * mf.eta))
                if damped
                else 1.0
            )
            coefficient = (4.0 * np.pi) * weight * damping / g2
            pair_g = mf._periodic_pair_ft(gvec, mf.kpts[0])
            pair_minus_g = mf._periodic_pair_ft(-gvec, mf.kpts[0])
            pair1_g = self._pair_ft_derivatives(gvec)
            pair1_minus_g = self._pair_ft_derivatives(-gvec)
            out += coefficient * (
                np.einsum("Axpq,rs->Axpqrs", pair1_g, pair_minus_g, optimize=True)
                + np.einsum("pq,Axrs->Axpqrs", pair_g, pair1_minus_g, optimize=True)
            )
        out = 0.5 * (out + out.transpose(0, 1, 3, 2, 5, 4).conj())

        if damped and mf.nuclear_background:
            volume = abs(float(np.linalg.det(self.cell.lattice_vectors)))
            coefficient = np.pi / (mf.eta * mf.eta * volume)
            overlap = np.asarray(mf._overlap_k[0])
            out -= coefficient * (
                np.einsum("Axpq,rs->Axpqrs", s1, overlap, optimize=True)
                + np.einsum("pq,Axrs->Axpqrs", overlap, s1, optimize=True)
            )
        return np.asarray(out.real, dtype=float)

    def two_electron_derivatives(self, s1):
        if str(self.base.jk_builder) == "reciprocal":
            return self._reciprocal_eri_derivatives(s1, damped=False)
        return self._short_range_eri_derivatives() + self._reciprocal_eri_derivatives(
            s1,
            damped=True,
        )

    def _reciprocal_veff_derivatives(self, dm):
        return self._reciprocal_veff_derivatives_many(
            np.asarray(dm, dtype=np.complex128)[None, :, :]
        )[0]

    def _reciprocal_veff_derivatives_many(self, dms):
        mf = self.base
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        dms = np.asarray(dms, dtype=np.complex128)
        if dms.ndim != 3 or dms.shape[1:] != (nao, nao):
            raise ValueError(f"dms must have shape (ndm, {nao}, {nao}).")
        out = np.zeros((len(dms), natom, 3, nao, nao), dtype=np.complex128)
        values = mf._reciprocal_g_weights()
        block_size = 128
        for start in range(0, len(values), block_size):
            block = values[start:start + block_size]
            gvecs = np.asarray([gvec for gvec, _weight in block], dtype=float)
            weights = np.asarray([weight for _gvec, weight in block], dtype=float)
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            mask = g2 > 0.0
            if not np.any(mask):
                continue
            gvecs = gvecs[mask]
            weights = weights[mask]
            g2 = g2[mask]
            coefficient = (4.0 * np.pi) * weights / g2
            pair_g = mf._periodic_pair_ft_batch(gvecs, mf.kpts[0])
            pair_minus_g = mf._periodic_pair_ft_batch(-gvecs, mf.kpts[0])
            pair1_g = self._pair_ft_derivatives_many(gvecs)
            pair1_minus_g = self._pair_ft_derivatives_many(-gvecs)

            rho_minus = np.einsum(
                "grs,Drs->Dg", pair_minus_g, dms, optimize=True
            )
            rho1_minus = np.einsum(
                "Axgrs,Drs->DAxg", pair1_minus_g, dms, optimize=True
            )
            vj1 = np.einsum(
                "g,Axgpq,Dg->DAxpq",
                coefficient,
                pair1_g,
                rho_minus,
                optimize=True,
            )
            vj1 += np.einsum(
                "g,gpq,DAxg->DAxpq",
                coefficient,
                pair_g,
                rho1_minus,
                optimize=True,
            )
            vk1 = np.einsum(
                "g,Axgpr,Drs,gsq->DAxpq",
                coefficient,
                pair1_g,
                dms,
                pair_minus_g,
                optimize=True,
            )
            vk1 += np.einsum(
                "g,gpr,Drs,Axgsq->DAxpq",
                coefficient,
                pair_g,
                dms,
                pair1_minus_g,
                optimize=True,
            )
            out += vj1 - 0.5 * vk1
        for density in range(len(dms)):
            for atom in range(natom):
                for axis in range(3):
                    out[density, atom, axis] = _symmetrize(
                        out[density, atom, axis]
                    )
        return out

    def reciprocal_veff_second_scalar(self, dm):
        """Return ``Tr[P G[2](P)]`` for the reciprocal J/K backend."""

        if str(self.base.jk_builder) != "reciprocal":
            raise NotImplementedError(
                "Analytic reciprocal J/K second derivatives require "
                "jk_builder='reciprocal'."
            )
        mf = self.base
        dm = np.asarray(dm, dtype=np.complex128)
        natom = len(self.cell._atom_coords)
        nao = len(mf._basis)
        if dm.shape != (nao, nao):
            raise ValueError(f"dm must have shape ({nao}, {nao}).")
        out = np.zeros(
            (natom, 3, natom, 3, nao, nao),
            dtype=np.complex128,
        )
        values = mf._reciprocal_g_weights()
        block_size = 64
        for start in range(0, len(values), block_size):
            block = values[start:start + block_size]
            gvecs = np.asarray([gvec for gvec, _weight in block], dtype=float)
            weights = np.asarray([weight for _gvec, weight in block], dtype=float)
            g2 = np.einsum("gi,gi->g", gvecs, gvecs)
            mask = g2 > 0.0
            if not np.any(mask):
                continue
            gvecs = gvecs[mask]
            weights = weights[mask]
            g2 = g2[mask]
            coefficient = (4.0 * np.pi) * weights / g2
            pair_g = mf._periodic_pair_ft_batch(gvecs, mf.kpts[0])
            pair_minus = mf._periodic_pair_ft_batch(-gvecs, mf.kpts[0])
            pair1_g = self._pair_ft_derivatives_many(gvecs)
            pair1_minus = self._pair_ft_derivatives_many(-gvecs)
            pair2_g = self._pair_ft_second_derivatives_many(gvecs)
            pair2_minus = self._pair_ft_second_derivatives_many(-gvecs)

            rho_minus = np.einsum("grs,rs->g", pair_minus, dm, optimize=True)
            rho1_minus = np.einsum(
                "Axgrs,rs->Axg",
                pair1_minus,
                dm,
                optimize=True,
            )
            rho2_minus = np.einsum(
                "AxBygrs,rs->AxByg",
                pair2_minus,
                dm,
                optimize=True,
            )
            vj2 = np.einsum(
                "g,AxBygpq,g->AxBypq",
                coefficient,
                pair2_g,
                rho_minus,
                optimize=True,
            )
            vj2 += np.einsum(
                "g,Axgpq,Byg->AxBypq",
                coefficient,
                pair1_g,
                rho1_minus,
                optimize=True,
            )
            vj2 += np.einsum(
                "g,Bygpq,Axg->AxBypq",
                coefficient,
                pair1_g,
                rho1_minus,
                optimize=True,
            )
            vj2 += np.einsum(
                "g,gpq,AxByg->AxBypq",
                coefficient,
                pair_g,
                rho2_minus,
                optimize=True,
            )

            vk2 = np.einsum(
                "g,AxBygpr,rs,gsq->AxBypq",
                coefficient,
                pair2_g,
                dm,
                pair_minus,
                optimize=True,
            )
            vk2 += np.einsum(
                "g,Axgpr,rs,Bygsq->AxBypq",
                coefficient,
                pair1_g,
                dm,
                pair1_minus,
                optimize=True,
            )
            vk2 += np.einsum(
                "g,Bygpr,rs,Axgsq->AxBypq",
                coefficient,
                pair1_g,
                dm,
                pair1_minus,
                optimize=True,
            )
            vk2 += np.einsum(
                "g,gpr,rs,AxBygsq->AxBypq",
                coefficient,
                pair_g,
                dm,
                pair2_minus,
                optimize=True,
            )
            out += vj2 - 0.5 * vk2

        if mf.madelung is not None:
            zero = np.zeros((1, 3), dtype=float)
            overlap = mf._pair_overlap_at_k(np.zeros(3))
            overlap1 = self._pair_ft_derivatives_many(zero)[:, :, 0]
            overlap2 = self._pair_ft_second_derivatives_many(zero)[:, :, :, :, 0]
            out -= 0.5 * mf.madelung * (
                np.einsum(
                    "AxBypr,rs,sq->AxBypq",
                    overlap2,
                    dm,
                    overlap,
                    optimize=True,
                )
                + np.einsum(
                    "Axpr,rs,Bysq->AxBypq",
                    overlap1,
                    dm,
                    overlap1,
                    optimize=True,
                )
                + np.einsum(
                    "Bypr,rs,Axsq->AxBypq",
                    overlap1,
                    dm,
                    overlap1,
                    optimize=True,
                )
                + np.einsum(
                    "pr,rs,AxBysq->AxBypq",
                    overlap,
                    dm,
                    overlap2,
                    optimize=True,
                )
            )
        return np.einsum("AxBypq,qp->AxBy", out, dm, optimize=True).real

    def _reciprocal_two_electron_contribution(self, dm):
        veff1 = self._reciprocal_veff_derivatives(dm)
        return 0.5 * np.einsum(
            "Axpq,qp->Ax", veff1, dm, optimize=True
        ).real

    def effective_potential_derivatives(
        self,
        dm,
        *,
        s1=None,
        require_scf=True,
    ):
        """Return fixed-density J/K nuclear derivatives in the AO basis."""

        self._validate(require_scf=require_scf)
        dm = np.asarray(dm, dtype=np.complex128)
        nao = int(self.cell.nao)
        if dm.shape != (nao, nao):
            raise ValueError(f"dm must have shape ({nao}, {nao}).")
        builder = str(self.base.jk_builder)
        if builder == "reciprocal":
            veff1 = self._reciprocal_veff_derivatives(dm)
        elif builder == "ewald":
            if s1 is None:
                s1 = self.one_electron_derivatives()[0]
            s1 = np.asarray(s1, dtype=np.complex128)
            derivative_shape = (len(self.cell._atom_coords), 3, nao, nao)
            if s1.shape == (3 * len(self.cell._atom_coords), nao, nao):
                s1 = s1.reshape(derivative_shape)
            if s1.shape != derivative_shape:
                raise ValueError(
                    f"s1 must have shape {derivative_shape} or its flattened "
                    "nuclear-coordinate form."
                )
            eri1 = self.two_electron_derivatives(s1)
            vj1 = np.einsum("Axpqrs,rs->Axpq", eri1, dm, optimize=True)
            vk1 = np.einsum("Axprqs,rs->Axpq", eri1, dm, optimize=True)
            veff1 = vj1 - 0.5 * vk1
        elif builder == "gdf":
            veff1 = self._gdf_veff_derivatives(dm)
        else:
            raise NotImplementedError(
                f"Explicit {builder!r} Fock nuclear derivatives are not implemented."
            )

        if self.base.madelung is not None:
            if builder == "reciprocal":
                overlap = self.base._pair_overlap_at_k(np.zeros(3))
                overlap1 = self._pair_ft_derivatives(np.zeros(3))
            else:
                if s1 is None:
                    s1 = self.one_electron_derivatives()[0]
                overlap = np.asarray(self.base._overlap_k[0])
                overlap1 = np.asarray(s1, dtype=np.complex128)
            veff1 -= 0.5 * self.base.madelung * (
                np.einsum(
                    "Axpr,rs,sq->Axpq",
                    overlap1,
                    dm,
                    overlap,
                    optimize=True,
                )
                + np.einsum(
                    "pr,rs,Axsq->Axpq",
                    overlap,
                    dm,
                    overlap1,
                    optimize=True,
                )
            )
        return np.asarray(veff1, dtype=np.complex128)

    def effective_potential_derivatives_many(
        self,
        dms,
        *,
        s1=None,
        require_scf=True,
    ):
        """Return fixed-density J/K derivatives for a batch of AO densities."""

        self._validate(require_scf=require_scf)
        dms = np.asarray(dms, dtype=np.complex128)
        nao = int(self.cell.nao)
        if dms.ndim != 3 or dms.shape[1:] != (nao, nao):
            raise ValueError(f"dms must have shape (ndm, {nao}, {nao}).")
        builder = str(self.base.jk_builder)
        if builder == "gdf":
            veff1 = self._gdf_veff_derivatives_many(dms)
            if self.base.madelung is not None:
                if s1 is None:
                    s1 = self.one_electron_derivatives()[0]
                overlap = np.asarray(self.base._overlap_k[0])
                overlap1 = np.asarray(s1, dtype=np.complex128)
                veff1 -= 0.5 * self.base.madelung * (
                    np.einsum(
                        "Axpr,Drs,sq->DAxpq",
                        overlap1,
                        dms,
                        overlap,
                        optimize=True,
                    )
                    + np.einsum(
                        "pr,Drs,Axsq->DAxpq",
                        overlap,
                        dms,
                        overlap1,
                        optimize=True,
                    )
                )
            return np.asarray(veff1, dtype=np.complex128)
        if builder != "reciprocal":
            return np.asarray(
                [
                    self.effective_potential_derivatives(
                        dm,
                        s1=s1,
                        require_scf=False,
                    )
                    for dm in dms
                ],
                dtype=np.complex128,
            )

        veff1 = self._reciprocal_veff_derivatives_many(dms)
        if self.base.madelung is not None:
            overlap = self.base._pair_overlap_at_k(np.zeros(3))
            overlap1 = self._pair_ft_derivatives(np.zeros(3))
            veff1 -= 0.5 * self.base.madelung * (
                np.einsum(
                    "Axpr,Drs,sq->DAxpq",
                    overlap1,
                    dms,
                    overlap,
                    optimize=True,
                )
                + np.einsum(
                    "pr,Drs,Axsq->DAxpq",
                    overlap,
                    dms,
                    overlap1,
                    optimize=True,
                )
            )
        return np.asarray(veff1, dtype=np.complex128)

    def explicit_integral_derivatives(self, dm=None, *, require_scf=True):
        """Return overlap, core, and fixed-density effective-potential derivatives.

        The returned arrays have shape ``(natom, 3, nao, nao)``.  The effective
        potential derivative contains the explicit nuclear derivative at fixed
        AO density, including the Madelung exchange correction.
        """

        self._validate(require_scf=require_scf)
        if dm is None:
            if not require_scf:
                raise ValueError("dm is required when no converged SCF is available.")
            dm = self.base.make_rdm1()
        dm = np.asarray(dm, dtype=np.complex128)
        nao = int(self.cell.nao)
        if dm.shape != (nao, nao):
            raise ValueError(f"dm must have shape ({nao}, {nao}).")

        s1, h1 = self.one_electron_derivatives()
        veff1 = self.effective_potential_derivatives(
            dm,
            s1=s1,
            require_scf=False,
        )
        return (
            np.asarray(s1, dtype=np.complex128),
            np.asarray(h1, dtype=np.complex128),
            np.asarray(veff1, dtype=np.complex128),
        )

    def directional_integral_derivatives(
        self,
        weights,
        dm=None,
        *,
        require_scf=True,
    ):
        r"""Return analytic AO derivatives contracted with one direction.

        ``weights[A, x]`` may be complex, as required for a traveling-wave
        phonon.  The method is algebraically equivalent to contracting
        :meth:`explicit_integral_derivatives`, but releases each full
        Cartesian derivative tensor after use to lower peak retained memory.
        """

        self._validate(require_scf=require_scf)
        natom = len(self.cell._atom_coords)
        weights = np.asarray(weights, dtype=np.complex128)
        if weights.shape != (natom, 3):
            raise ValueError(f"weights must have shape ({natom}, 3).")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite.")
        if dm is None:
            if not require_scf:
                raise ValueError("dm is required when no converged SCF is available.")
            dm = self.base.make_rdm1()
        dm = np.asarray(dm, dtype=np.complex128)
        nao = int(self.cell.nao)
        if dm.shape != (nao, nao):
            raise ValueError(f"dm must have shape ({nao}, {nao}).")

        s1, h1 = self.one_electron_derivatives()
        directional_s1 = np.einsum("Ax,Axpq->pq", weights, s1, optimize=True)
        directional_h1 = np.einsum("Ax,Axpq->pq", weights, h1, optimize=True)
        del h1
        veff1 = self.effective_potential_derivatives(
            dm,
            s1=s1,
            require_scf=False,
        )
        directional_veff1 = np.einsum(
            "Ax,Axpq->pq",
            weights,
            veff1,
            optimize=True,
        )
        full_tensor_bytes = int(natom * 3 * nao * nao * 16)
        self.directional_response_info = {
            "full_cartesian_tensor_bytes": full_tensor_bytes,
            "retained_peak_tensor_count": 2,
            "legacy_retained_peak_tensor_count": 3,
            "retained_peak_bytes": 2 * full_tensor_bytes,
        }
        return (
            np.asarray(directional_s1, dtype=np.complex128),
            np.asarray(directional_h1, dtype=np.complex128),
            np.asarray(directional_veff1, dtype=np.complex128),
        )

    @staticmethod
    def _spectral_pseudoinverse_response(metric, metric1, threshold):
        metric = _symmetrize(np.asarray(metric, dtype=np.complex128))
        evals, evecs = np.linalg.eigh(metric)
        scale = max(float(np.max(np.abs(evals))) if evals.size else 0.0, 1.0)
        keep = evals > float(threshold) * scale
        if not np.any(keep):
            raise np.linalg.LinAlgError("The GDF metric response has zero retained rank.")

        inverse = np.zeros_like(evals)
        inverse[keep] = 1.0 / evals[keep]
        response_kernel = np.zeros((len(evals), len(evals)), dtype=float)
        for left in range(len(evals)):
            for right in range(len(evals)):
                separation = evals[left] - evals[right]
                tolerance = 1.0e-12 * max(
                    1.0, abs(evals[left]), abs(evals[right])
                )
                if abs(separation) > tolerance:
                    response_kernel[left, right] = (
                        inverse[left] - inverse[right]
                    ) / separation
                elif keep[left] and keep[right]:
                    average = 0.5 * (evals[left] + evals[right])
                    response_kernel[left, right] = -1.0 / (average * average)
                elif not keep[left] and not keep[right]:
                    response_kernel[left, right] = 0.0
                else:
                    raise np.linalg.LinAlgError(
                        "A GDF metric eigenvalue crosses the retained-rank threshold."
                    )

        metric1_eigen = np.einsum(
            "Pi,AxPQ,Qj->Axij",
            evecs.conj(),
            metric1,
            evecs,
            optimize=True,
        )
        inverse1_eigen = response_kernel[None, None, :, :] * metric1_eigen
        inverse_metric = (evecs * inverse[None, :]) @ evecs.conj().T
        inverse_metric1 = np.einsum(
            "Pi,Axij,Qj->AxPQ",
            evecs,
            inverse1_eigen,
            evecs.conj(),
            optimize=True,
        )
        return inverse_metric, inverse_metric1, evals, keep

    def _gdf_short_range_metric_response(
        self,
        aux,
        omega,
        image_keys,
        screen_tol,
    ):
        from pyqed.pbc.gw.integrals import _gdf_sr_aux_screen_data

        mf = self.base
        natom = len(self.cell._atom_coords)
        cart_atom_ids = _atom_ids_for_basis(
            aux.cart_basis,
            np.asarray(self.cell._atom_coords, dtype=float),
        )
        ncart = aux.ncart
        metric_cart = np.zeros((ncart, ncart), dtype=np.complex128)
        metric_cart1 = np.zeros(
            (natom, 3, ncart, ncart), dtype=np.complex128
        )
        aux_centers, aux_scales = _gdf_sr_aux_screen_data(aux)
        for image_key in image_keys:
            shift = mf.cell.translation_vector(image_key)
            shifted_aux = tuple(
                _shifted_gaussian(function, shift)
                for function in aux.cart_basis
            )
            keep_mask = None
            if screen_tol != 0.0:
                shifted_centers = aux_centers + shift[None, :]
                distances = np.linalg.norm(
                    aux_centers[:, None, :] - shifted_centers[None, :, :],
                    axis=2,
                )
                damping = np.empty_like(distances)
                near = distances <= 1.0e-12
                damping[near] = 2.0 * float(omega) / np.sqrt(np.pi)
                far = ~near
                damping[far] = np.fromiter(
                    (
                        math.erfc(float(omega) * float(distance))
                        / float(distance)
                        for distance in distances[far]
                    ),
                    dtype=float,
                    count=int(np.count_nonzero(far)),
                )
                keep_mask = (
                    aux_scales[:, None] * aux_scales[None, :] * damping
                    > screen_tol
                )
            for left_index, left in enumerate(aux.cart_basis):
                atom_left = int(cart_atom_ids[left_index])
                for right_index, right in enumerate(shifted_aux):
                    if keep_mask is not None and not keep_mask[left_index, right_index]:
                        continue
                    atom_right = int(cart_atom_ids[right_index])
                    metric_cart[left_index, right_index] += (
                        short_range_two_center_coulomb(left, right, omega)
                    )
                    for axis in range(3):
                        metric_cart1[
                            atom_left, axis, left_index, right_index
                        ] += sum(
                            short_range_two_center_coulomb(
                                derivative, right, omega
                            )
                            for derivative in self._basis_derivatives(left, axis)
                        )
                        metric_cart1[
                            atom_right, axis, left_index, right_index
                        ] += sum(
                            short_range_two_center_coulomb(
                                left, derivative, omega
                            )
                            for derivative in self._basis_derivatives(right, axis)
                        )

        transform = np.asarray(aux.transform)
        metric = np.einsum(
            "aP,ab,bQ->PQ", transform, metric_cart, transform, optimize=True
        )
        metric1 = np.einsum(
            "aP,Axab,bQ->AxPQ",
            transform,
            metric_cart1,
            transform,
            optimize=True,
        )
        return _symmetrize(metric), 0.5 * (
            metric1 + metric1.conj().transpose(0, 1, 3, 2)
        )

    def _gdf_short_range_three_center_response(
        self,
        aux,
        omega,
        image_keys,
        pair_screen_tol,
        short_range_screen_tol,
        allowed_pair_mask=None,
    ):
        from pyqed.pbc.gw.integrals import (
            _gdf_sr3c_aux_indices,
            _gdf_sr_aux_screen_data,
        )

        mf = self.base
        basis = tuple(mf._basis)
        nao = len(basis)
        if allowed_pair_mask is not None:
            allowed_pair_mask = np.asarray(allowed_pair_mask, dtype=np.bool_)
            if allowed_pair_mask.shape != (nao, nao):
                raise ValueError(
                    f"allowed_pair_mask must have shape ({nao}, {nao})."
                )
        natom = len(self.cell._atom_coords)
        cart_atom_ids = _atom_ids_for_basis(
            aux.cart_basis,
            np.asarray(self.cell._atom_coords, dtype=float),
        )
        aux_centers, aux_scales = _gdf_sr_aux_screen_data(aux)
        image_data = tuple(
            (
                mf.cell.translation_vector(image_key),
                tuple(
                    _shifted_gaussian(
                        function,
                        mf.cell.translation_vector(image_key),
                    )
                    for function in basis
                ),
            )
            for image_key in image_keys
        )
        three_center_cart = np.zeros(
            (aux.ncart, nao, nao), dtype=np.complex128
        )
        three_center_cart1 = np.zeros(
            (natom, 3, aux.ncart, nao, nao), dtype=np.complex128
        )
        for left_shift, left_basis in image_data:
            for right_shift, right_basis in image_data:
                relative_shift = right_shift - left_shift
                for left_index, left in enumerate(left_basis):
                    atom_left = int(self._atom_ids[left_index])
                    for right_index, right in enumerate(right_basis):
                        if (
                            allowed_pair_mask is not None
                            and not allowed_pair_mask[left_index, right_index]
                        ):
                            continue
                        atom_right = int(self._atom_ids[right_index])
                        pair_bound = _gaussian_pair_ft_decay_bound(
                            basis[left_index],
                            basis[right_index],
                            relative_shift,
                        )
                        if pair_screen_tol != 0.0 and pair_bound <= pair_screen_tol:
                            continue
                        aux_indices = _gdf_sr3c_aux_indices(
                            left,
                            right,
                            aux_centers,
                            aux_scales,
                            omega,
                            pair_bound,
                            short_range_screen_tol,
                        )
                        for auxiliary in aux_indices:
                            aux_function = aux.cart_basis[int(auxiliary)]
                            atom_aux = int(cart_atom_ids[int(auxiliary)])
                            three_center_cart[auxiliary, left_index, right_index] += (
                                short_range_three_center_eri(
                                    left,
                                    right,
                                    aux_function,
                                    omega,
                                )
                            )
                            for axis in range(3):
                                three_center_cart1[
                                    atom_left,
                                    axis,
                                    auxiliary,
                                    left_index,
                                    right_index,
                                ] += sum(
                                    short_range_three_center_eri(
                                        derivative,
                                        right,
                                        aux_function,
                                        omega,
                                    )
                                    for derivative in self._basis_derivatives(
                                        left, axis
                                    )
                                )
                                three_center_cart1[
                                    atom_right,
                                    axis,
                                    auxiliary,
                                    left_index,
                                    right_index,
                                ] += sum(
                                    short_range_three_center_eri(
                                        left,
                                        derivative,
                                        aux_function,
                                        omega,
                                    )
                                    for derivative in self._basis_derivatives(
                                        right, axis
                                    )
                                )
                                three_center_cart1[
                                    atom_aux,
                                    axis,
                                    auxiliary,
                                    left_index,
                                    right_index,
                                ] += sum(
                                    short_range_three_center_eri(
                                        left,
                                        right,
                                        derivative,
                                        omega,
                                    )
                                    for derivative in self._basis_derivatives(
                                        aux_function, axis
                                    )
                                )

        transform = np.asarray(aux.transform)
        three_center = np.einsum(
            "aP,apq->Ppq", transform, three_center_cart, optimize=True
        )
        three_center1 = np.einsum(
            "aP,Axapq->AxPpq",
            transform,
            three_center_cart1,
            optimize=True,
        )
        return three_center, three_center1

    def _gdf_raw_response(self):
        if self._gdf_raw_response_cache is not None:
            return self._gdf_raw_response_cache
        from pyqed.pbc.gw.integrals import (
            _gdf_aux_coord_type,
            _gdf_auxbasis_name,
            _gdf_auxiliary_basis,
            _gdf_auxiliary_charge,
            _gdf_auxiliary_ft,
            _gdf_backend_settings,
            _gdf_g_block_size,
            _gdf_metric_relative_tol,
            _gdf_metric_tol,
            _gdf_pair_ft_batch,
            _gdf_pair_ft_plan_data,
            _gdf_pair_screen_tol,
            _gdf_reciprocal_coulomb_blocks,
            _gdf_rs_aux_engine,
            _gdf_rs_compact_auxiliary_basis,
            _gdf_rs_shell_engine,
            _gdf_short_range_cut,
            _gdf_short_range_image_keys,
            _gdf_short_range_screen_tol,
        )

        mf = self.base
        backend = mf.with_df
        space = backend._space
        ref = space.reference
        q_index = space.q0_index
        qvec = np.asarray(space.qpts[q_index], dtype=float)
        (
            recip_cut,
            pair_cut,
            mesh,
            _recip_key,
            kernel,
            omega,
            _kernel_key,
            _auto_info,
        ) = _gdf_backend_settings(ref)
        if kernel not in ("full", "range_separated"):
            raise NotImplementedError(
                "The analytic GDF response requires a full or range-separated kernel."
            )
        pair_screen_tol = _gdf_pair_screen_tol(ref)
        auxbasis = _gdf_auxbasis_name(ref)
        aux = _gdf_auxiliary_basis(
            space,
            auxbasis,
            _gdf_aux_coord_type(ref),
        )
        rs_pair_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
        rs_aux_engine = _gdf_rs_aux_engine(ref, aux, kernel, omega, mesh)
        pair_partition_active = bool(
            rs_pair_engine is not None and rs_pair_engine.partition_active
        )
        aux_partition_active = bool(
            rs_aux_engine is not None and rs_aux_engine.partition_active
        )
        compensated_partition = pair_partition_active or aux_partition_active
        compact_pair_mask = (
            np.asarray(rs_pair_engine.compact_pair_mask, dtype=np.bool_)
            if pair_partition_active
            else None
        )
        compact_aux_mask = (
            np.asarray(rs_aux_engine.compact_aux_mask, dtype=np.bool_)
            if rs_aux_engine is not None
            else None
        )
        plan_data = _gdf_pair_ft_plan_data(
            mf,
            pair_cut,
            pair_screen_tol,
        )
        g_block_size = _gdf_g_block_size(
            mf,
            mesh=mesh,
            naux=aux.naux,
            nao_pair=mf.cell.nao * mf.cell.nao,
            nkpts=1,
        )
        g_block_size = max(1, int(g_block_size or 128))

        coords = np.asarray(self.cell._atom_coords, dtype=float)
        cart_atom_ids = _atom_ids_for_basis(aux.cart_basis, coords)
        transform = np.asarray(aux.transform)
        aux_atom_ids = np.empty(aux.naux, dtype=int)
        for column in range(aux.naux):
            support = np.flatnonzero(np.abs(transform[:, column]) > 1.0e-14)
            atoms = np.unique(cart_atom_ids[support])
            if len(atoms) != 1:
                raise RuntimeError("An auxiliary function mixes centers in the GDF transform.")
            aux_atom_ids[column] = int(atoms[0])

        natom = len(coords)
        nao = int(mf.cell.nao)
        metric = np.zeros((aux.naux, aux.naux), dtype=np.complex128)
        metric1 = np.zeros(
            (natom, 3, aux.naux, aux.naux), dtype=np.complex128
        )
        three_center = np.zeros(
            (aux.naux, nao, nao), dtype=np.complex128
        )
        three_center1 = np.zeros(
            (natom, 3, aux.naux, nao, nao), dtype=np.complex128
        )
        block_count = 0
        vector_count = 0
        reciprocal_kernel = "full" if compensated_partition else kernel
        reciprocal_omega = None if compensated_partition else omega
        for gvecs, weights in _gdf_reciprocal_coulomb_blocks(
            mf,
            qvec,
            backend.g2_tol,
            recip_cut=recip_cut,
            mesh=mesh,
            kernel=reciprocal_kernel,
            omega=reciprocal_omega,
            block_size=g_block_size,
        ):
            block_count += 1
            vector_count += len(gvecs)
            aux_ft = _gdf_auxiliary_ft(
                space,
                aux,
                gvecs,
                cache_enabled=False,
            )
            pair_ft = _gdf_pair_ft_batch(
                space,
                gvecs,
                mf.kpts[0],
                pair_cut,
                pair_screen_tol,
                cache_enabled=False,
            )
            pair_ft1 = self._pair_ft_derivatives_from_plan_many(
                gvecs,
                mf.kpts[0],
                plan_data=plan_data,
            )
            aux_ft1 = np.zeros(
                (natom, 3, len(gvecs), aux.naux), dtype=np.complex128
            )
            phase_derivative = -1.0j * gvecs.T
            for auxiliary, atom in enumerate(aux_atom_ids):
                aux_ft1[atom, :, :, auxiliary] = (
                    phase_derivative * aux_ft[:, auxiliary][None, :]
                )

            metric += np.einsum(
                "gP,g,gQ->PQ", aux_ft.conj(), weights, aux_ft, optimize=True
            )
            metric1 += np.einsum(
                "AxgP,g,gQ->AxPQ",
                aux_ft1.conj(),
                weights,
                aux_ft,
                optimize=True,
            )
            metric1 += np.einsum(
                "gP,g,AxgQ->AxPQ",
                aux_ft.conj(),
                weights,
                aux_ft1,
                optimize=True,
            )
            three_center += np.einsum(
                "gP,g,gmn->Pmn",
                aux_ft.conj(),
                weights,
                pair_ft,
                optimize=True,
            )
            three_center1 += np.einsum(
                "AxgP,g,gmn->AxPmn",
                aux_ft1.conj(),
                weights,
                pair_ft,
                optimize=True,
            )
            three_center1 += np.einsum(
                "gP,g,Axgmn->AxPmn",
                aux_ft.conj(),
                weights,
                pair_ft1,
                optimize=True,
            )
            if compensated_partition:
                g2 = np.einsum("gi,gi->g", gvecs, gvecs)
                short_weights = weights * (
                    1.0
                    - np.exp(
                        -g2 / (4.0 * float(omega) * float(omega))
                    )
                )
                metric_aux_mask = (
                    compact_aux_mask
                    if aux_partition_active
                    else np.ones(aux.naux, dtype=np.bool_)
                )
                metric_aux_ft = aux_ft * metric_aux_mask[None, :]
                metric_aux_ft1 = (
                    aux_ft1 * metric_aux_mask[None, None, None, :]
                )
                metric -= np.einsum(
                    "gP,g,gQ->PQ",
                    metric_aux_ft.conj(),
                    short_weights,
                    metric_aux_ft,
                    optimize=True,
                )
                metric1 -= (
                    np.einsum(
                        "AxgP,g,gQ->AxPQ",
                        metric_aux_ft1.conj(),
                        short_weights,
                        metric_aux_ft,
                        optimize=True,
                    )
                    + np.einsum(
                        "gP,g,AxgQ->AxPQ",
                        metric_aux_ft.conj(),
                        short_weights,
                        metric_aux_ft1,
                        optimize=True,
                    )
                )

                reciprocal_pair = pair_ft
                reciprocal_pair1 = pair_ft1
                if compact_pair_mask is not None:
                    reciprocal_pair = (
                        pair_ft * compact_pair_mask[None, :, :]
                    )
                    reciprocal_pair1 = (
                        pair_ft1 * compact_pair_mask[None, None, None, :, :]
                    )
                reciprocal_aux = aux_ft
                reciprocal_aux1 = aux_ft1
                if aux_partition_active:
                    reciprocal_aux = aux_ft * compact_aux_mask[None, :]
                    reciprocal_aux1 = (
                        aux_ft1 * compact_aux_mask[None, None, None, :]
                    )
                three_center -= np.einsum(
                    "gP,g,gmn->Pmn",
                    reciprocal_aux.conj(),
                    short_weights,
                    reciprocal_pair,
                    optimize=True,
                )
                three_center1 -= np.einsum(
                    "AxgP,g,gmn->AxPmn",
                    reciprocal_aux1.conj(),
                    short_weights,
                    reciprocal_pair,
                    optimize=True,
                )
                three_center1 -= np.einsum(
                    "gP,g,Axgmn->AxPmn",
                    reciprocal_aux.conj(),
                    short_weights,
                    reciprocal_pair1,
                    optimize=True,
                )

        if kernel == "range_separated":
            short_range_cut = _gdf_short_range_cut(ref)
            image_keys = _gdf_short_range_image_keys(
                ref,
                short_range_cut,
                omega,
            )
            short_range_screen_tol = _gdf_short_range_screen_tol(ref)
            short_range_aux = _gdf_rs_compact_auxiliary_basis(
                aux,
                rs_aux_engine,
            )
            has_compact_aux = short_range_aux.ncart > 0
            has_compact_pairs = (
                compact_pair_mask is None or bool(np.any(compact_pair_mask))
            )
            if has_compact_aux:
                short_metric, short_metric1 = (
                    self._gdf_short_range_metric_response(
                        short_range_aux,
                        omega,
                        image_keys,
                        short_range_screen_tol,
                    )
                )
                metric += short_metric
                metric1 += short_metric1
            if has_compact_aux and has_compact_pairs:
                short_three_center, short_three_center1 = (
                    self._gdf_short_range_three_center_response(
                        short_range_aux,
                        omega,
                        image_keys,
                        pair_screen_tol,
                        short_range_screen_tol,
                        allowed_pair_mask=compact_pair_mask,
                    )
                )
                three_center += short_three_center
                three_center1 += short_three_center1

            if has_compact_aux:
                volume = abs(float(np.linalg.det(self.cell.lattice_vectors)))
                g0 = np.pi / (float(omega) * float(omega) * volume)
                auxiliary_charge = _gdf_auxiliary_charge(space, aux)
                if compact_aux_mask is not None:
                    auxiliary_charge = auxiliary_charge * compact_aux_mask
                metric -= g0 * np.outer(auxiliary_charge, auxiliary_charge)
                if has_compact_pairs:
                    overlap = _gdf_pair_ft_batch(
                        space,
                        np.zeros((1, 3), dtype=float),
                        mf.kpts[0],
                        pair_cut,
                        pair_screen_tol,
                        cache_enabled=False,
                    )[0]
                    overlap1 = self._pair_ft_derivatives_from_plan_many(
                        np.zeros((1, 3), dtype=float),
                        mf.kpts[0],
                        plan_data=plan_data,
                    )[:, :, 0]
                    if compact_pair_mask is not None:
                        overlap = overlap * compact_pair_mask
                        overlap1 = overlap1 * compact_pair_mask[None, None, :, :]
                    three_center -= (
                        g0
                        * auxiliary_charge[:, None, None]
                        * overlap[None, :, :]
                    )
                    three_center1 -= (
                        g0
                        * auxiliary_charge[None, None, :, None, None]
                        * overlap1[:, :, None, :, :]
                    )
        metric = _symmetrize(metric)
        metric1 = 0.5 * (
            metric1 + metric1.conj().transpose(0, 1, 3, 2)
        )
        metric_scale = max(float(np.max(np.abs(np.linalg.eigvalsh(metric)))), 1.0)
        spectral_threshold = max(
            float(_gdf_metric_tol(ref)),
            float(backend.g2_tol),
            float(_gdf_metric_relative_tol(ref)) * metric_scale,
        ) / metric_scale
        inverse_metric, inverse_metric1, evals, keep = (
            self._spectral_pseudoinverse_response(
                metric,
                metric1,
                spectral_threshold,
            )
        )
        self.gdf_response_info = {
            "kernel": kernel,
            "q_index": int(q_index),
            "auxbasis": auxbasis,
            "naux": int(aux.naux),
            "metric_rank": int(np.count_nonzero(keep)),
            "metric_eigenvalues": np.array(evals, copy=True),
            "g_blocks": int(block_count),
            "g_vectors": int(vector_count),
            "rs_pair_partition": (
                None if rs_pair_engine is None else rs_pair_engine.mode
            ),
            "rs_aux_partition": (
                None if rs_aux_engine is None else rs_aux_engine.mode
            ),
            "rs_pair_partition_active": bool(pair_partition_active),
            "rs_aux_partition_active": bool(aux_partition_active),
            "rs_compact_pairs": (
                int(nao * nao)
                if compact_pair_mask is None
                else int(np.count_nonzero(compact_pair_mask))
            ),
            "rs_compact_aux": (
                int(aux.naux)
                if compact_aux_mask is None
                else int(np.count_nonzero(compact_aux_mask))
            ),
        }
        result = three_center, three_center1, inverse_metric, inverse_metric1
        self._gdf_raw_response_cache = result
        return result

    def _gdf_veff_derivatives_many(self, dms):
        """Return fixed-density native GDF J/K derivative matrices."""

        dms = np.asarray(dms, dtype=np.complex128)
        nao = int(self.cell.nao)
        if dms.ndim != 3 or dms.shape[1:] != (nao, nao):
            raise ValueError(f"dms must have shape (ndm, {nao}, {nao}).")
        three_center, three_center1, inverse_metric, inverse_metric1 = (
            self._gdf_raw_response()
        )
        density_aux = np.einsum(
            "Pij,Dji->DP",
            three_center,
            dms,
            optimize=True,
        )
        density_aux1 = np.einsum(
            "AxPij,Dji->DAxP",
            three_center1,
            dms,
            optimize=True,
        )
        vj1 = np.einsum(
            "AxPij,PQ,DQ->DAxij",
            three_center1,
            inverse_metric,
            density_aux.conj(),
            optimize=True,
        )
        vj1 += np.einsum(
            "Pij,AxPQ,DQ->DAxij",
            three_center,
            inverse_metric1,
            density_aux.conj(),
            optimize=True,
        )
        vj1 += np.einsum(
            "Pij,PQ,DAxQ->DAxij",
            three_center,
            inverse_metric,
            density_aux1.conj(),
            optimize=True,
        )

        interaction = inverse_metric.T
        interaction1 = inverse_metric1.transpose(0, 1, 3, 2)
        vk1 = np.einsum(
            "AxPim,Dmn,PQ,Qjn->DAxij",
            three_center1,
            dms,
            interaction,
            three_center.conj(),
            optimize=True,
        )
        vk1 += np.einsum(
            "Pim,Dmn,AxPQ,Qjn->DAxij",
            three_center,
            dms,
            interaction1,
            three_center.conj(),
            optimize=True,
        )
        vk1 += np.einsum(
            "Pim,Dmn,PQ,AxQjn->DAxij",
            three_center,
            dms,
            interaction,
            three_center1.conj(),
            optimize=True,
        )
        veff1 = vj1 - 0.5 * vk1
        return 0.5 * (veff1 + veff1.conj().transpose(0, 1, 2, 4, 3))

    def _gdf_veff_derivatives(self, dm):
        return self._gdf_veff_derivatives_many(
            np.asarray(dm, dtype=np.complex128)[None, :, :]
        )[0]

    def gdf_derivative_factors(self, *, require_scf=True):
        r"""Return analytic one-k GDF AO tensors and their derivatives.

        The mapping contains the unwhitened three-center tensor
        :math:`B_{P\mu\nu}`, its Cartesian nuclear derivative, the inverse
        auxiliary metric :math:`M^{-1}`, and its derivative.  These are the
        shared primitives used by analytic GDF Fock and response-kernel
        derivatives.
        """

        self._validate(require_scf=require_scf)
        three_center, three_center1, inverse_metric, inverse_metric1 = (
            self._gdf_raw_response()
        )
        return {
            "three_center": three_center,
            "three_center1": three_center1,
            "inverse_metric": inverse_metric,
            "inverse_metric1": inverse_metric1,
            "info": dict(self.gdf_response_info),
        }

    def _gdf_two_electron_contribution(self, dm):
        three_center, three_center1, inverse_metric, inverse_metric1 = (
            self._gdf_raw_response()
        )
        density_aux = np.einsum(
            "Pij,ji->P", three_center, dm, optimize=True
        )
        density_aux1 = np.einsum(
            "AxPij,ji->AxP", three_center1, dm, optimize=True
        )
        coulomb = 0.5 * (
            np.einsum(
                "AxP,PQ,Q->Ax",
                density_aux1.conj(),
                inverse_metric,
                density_aux,
                optimize=True,
            )
            + np.einsum(
                "P,AxPQ,Q->Ax",
                density_aux.conj(),
                inverse_metric1,
                density_aux,
                optimize=True,
            )
            + np.einsum(
                "P,PQ,AxQ->Ax",
                density_aux.conj(),
                inverse_metric,
                density_aux1,
                optimize=True,
            )
        )

        interaction = inverse_metric.T
        interaction1 = inverse_metric1.transpose(0, 1, 3, 2)
        exchange = -0.25 * (
            np.einsum(
                "ji,AxPim,mn,PQ,Qjn->Ax",
                dm,
                three_center1,
                dm,
                interaction,
                three_center.conj(),
                optimize=True,
            )
            + np.einsum(
                "ji,Pim,mn,AxPQ,Qjn->Ax",
                dm,
                three_center,
                dm,
                interaction1,
                three_center.conj(),
                optimize=True,
            )
            + np.einsum(
                "ji,Pim,mn,PQ,AxQjn->Ax",
                dm,
                three_center,
                dm,
                interaction,
                three_center1.conj(),
                optimize=True,
            )
        )
        return np.asarray((coulomb + exchange).real, dtype=float)

    def make_rdm1e(self):
        mo_energy = np.asarray(self.base.mo_energy)
        mo_coeff = np.asarray(self.base.mo_coeff)
        mo_occ = np.asarray(self.base.mo_occ)
        occupied = mo_occ > 0.0
        coefficients = mo_coeff[:, occupied]
        return np.einsum(
            "pi,qi,i->pq",
            coefficients,
            coefficients.conj(),
            mo_energy[occupied] * mo_occ[occupied],
            optimize=True,
        )

    def electronic(self, atmlst=None):
        self._validate()
        started = time.perf_counter()
        dm = np.asarray(self.base.make_rdm1(), dtype=np.complex128)
        weighted_dm = self.make_rdm1e()
        s1, h1 = self.one_electron_derivatives()
        one_body_seconds = time.perf_counter() - started

        started = time.perf_counter()
        eri1 = None
        if str(self.base.jk_builder) == "reciprocal":
            two_electron = self._reciprocal_two_electron_contribution(dm)
        elif str(self.base.jk_builder) == "gdf":
            two_electron = self._gdf_two_electron_contribution(dm)
        else:
            eri1 = self.two_electron_derivatives(s1)
            two_electron = np.zeros((len(self.cell._atom_coords), 3), dtype=float)
            for atom in range(len(two_electron)):
                for axis in range(3):
                    derivative = eri1[atom, axis]
                    vj = np.einsum("pqrs,rs->pq", derivative, dm, optimize=True)
                    vk = np.einsum("prqs,rs->pq", derivative, dm, optimize=True)
                    two_electron[atom, axis] = (
                        0.5 * np.trace(dm @ (vj - 0.5 * vk)).real
                    )
        two_body_seconds = time.perf_counter() - started

        one_electron = np.einsum("Axpq,qp->Ax", h1, dm, optimize=True).real
        pulay = -np.einsum("Axpq,qp->Ax", s1, weighted_dm, optimize=True).real

        madelung = np.zeros_like(one_electron)
        if self.base.madelung is not None:
            if str(self.base.jk_builder) == "reciprocal":
                overlap = self.base._pair_overlap_at_k(np.zeros(3))
                overlap1 = self._pair_ft_derivatives(np.zeros(3))
            else:
                overlap = np.asarray(self.base._overlap_k[0])
                overlap1 = s1
            for atom in range(len(one_electron)):
                for axis in range(3):
                    dveff = -0.5 * self.base.madelung * (
                        overlap1[atom, axis] @ dm @ overlap
                        + overlap @ dm @ overlap1[atom, axis]
                    )
                    madelung[atom, axis] = 0.5 * np.trace(dm @ dveff).real

        self.components = {
            "one_electron": one_electron,
            "two_electron": two_electron,
            "pulay": pulay,
            "madelung": madelung,
        }
        self.timings = {
            "one_electron_seconds": float(one_body_seconds),
            "two_electron_seconds": float(two_body_seconds),
        }
        gradient = one_electron + two_electron + pulay + madelung
        if atmlst is not None:
            gradient = gradient[np.asarray(tuple(atmlst), dtype=int)]
        return np.asarray(gradient, dtype=float)

    def nuclear(self, atmlst=None):
        self._validate()
        gradient = ewald_nuclear_gradient(
            self.cell.ionic_charges,
            self.cell._atom_coords,
            self.cell.lattice_vectors,
            eta=self.base.eta,
            real_cut=self.base.real_cut,
            recip_cut=self.base.recip_cut,
        )
        if atmlst is not None:
            gradient = gradient[np.asarray(tuple(atmlst), dtype=int)]
        return gradient

    def run(self, atmlst=None):
        self.de_elec = self.electronic(atmlst=atmlst)
        self.de_nuc = self.nuclear(atmlst=atmlst)
        self.de = self.de_elec + self.de_nuc
        return self.de

    def kernel(self, atmlst=None):
        return self.run(atmlst=atmlst)

    def forces(self, atmlst=None):
        return -self.run(atmlst=atmlst)


KRHFGradients = Gradients


__all__ = ["Gradients", "KRHFGradients"]
