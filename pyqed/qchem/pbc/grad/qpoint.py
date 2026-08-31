"""Analytic commensurate-q nuclear perturbations for periodic KRHF."""

from __future__ import annotations

import copy
import time

import numpy as np

from pyqed.qchem.basis_derivatives import _atom_ids_for_basis
from pyqed.qchem.fourier import has_periodic_pair_ft_backend
from pyqed.qchem.pbc.hf.ewald_rhf import _infer_kmesh
from pyqed.qchem.pbc.supercell import CommensurateSupercell
from pyqed.units import amu_to_au


class PrimitiveGDFQDerivativeEngine:
    r"""Build analytic finite-:math:`q` GDF derivatives in the primitive cell.

    For an auxiliary transfer :math:`Q`, a traveling-wave displacement
    :math:`u_{A\alpha}(R)=u_{A\alpha}e^{iq\cdot R}` produces the off-diagonal
    primitive-cell factors

    .. math::

       B^{(q)}_{Q+q,Q}
       = \langle \chi_{Q+q}|v|\rho_Q^{(q)}\rangle
       + \langle \chi_{Q+q}^{(-q)}|v|\rho_Q\rangle,

    and

    .. math::

       (M^{-1})^{(q)}_{Q+q,Q}
       =-M^{-1}_{Q+q}M^{(q)}_{Q+q,Q}M^{-1}_Q.

    Reciprocal vectors, AO-pair Fourier derivatives, and auxiliary Fourier
    derivatives are evaluated directly in the primitive cell.  No Born-von
    Karman supercell AO or auxiliary tensor is constructed.  The current exact
    implementation supports the full reciprocal Coulomb GDF kernel; the
    range-separated short-range derivative remains with the commensurate
    reference implementation.

    The GDF architecture follows Q. Sun et al., J. Chem. Phys. 147, 164119
    (2017), DOI: 10.1063/1.4998644.  The finite-momentum convention follows
    F. Giustino, Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003.  This is an analytic primitive-cell
    adaptation for Gaussian periodic density fitting, not a reproduction of
    either reference's complete DFPT implementation.
    """

    def __init__(self, mean_field, qpoint, cartesian_mode):
        self.base = mean_field
        self.qpoint = np.asarray(qpoint, dtype=float)
        self.cartesian_mode = np.asarray(cartesian_mode, dtype=np.complex128)
        self._reciprocal_cache = {}
        self._zero_ao_cache = {}
        self._derivative_ao_cache = {}
        self._inverse_derivative_cache = {}
        self._pair_plan_data = None
        self._one_body_pair_plan_data = None
        self._gradient_helper = None
        self.info = {}
        self._setup()

    def _setup(self):
        from pyqed.pbc.gw.integrals import (
            _gdf_aux_coord_type,
            _gdf_auxbasis_name,
            _gdf_auxiliary_basis,
            _gdf_backend_settings,
            _gdf_pair_ft_plan_data,
            _gdf_pair_screen_tol,
        )
        from pyqed.qchem.pbc.grad.rhf import Gradients

        mf = self.base
        if self.qpoint.shape != (3,):
            raise ValueError("qpoint must contain three Cartesian components.")
        natom = len(mf.cell._atom_coords)
        if self.cartesian_mode.size != 3 * natom:
            raise ValueError(f"cartesian_mode must contain {3 * natom} components.")
        self.cartesian_mode = np.ascontiguousarray(
            self.cartesian_mode.reshape(natom, 3)
        )
        if int(mf.cell.dimension) != 3:
            raise NotImplementedError(
                "Primitive q-resolved GDF derivatives require dimension=3."
            )
        if str(mf.jk_builder) != "gdf" or mf.with_df is None:
            raise NotImplementedError(
                "Primitive q-resolved derivatives require jk_builder='gdf'."
            )
        self.q_index = int(mf.with_df.find_qpoint_index(self.qpoint))
        self.minus_q_index = int(mf.with_df.find_qpoint_index(-self.qpoint))
        self.space = mf.with_df._space
        self.reference = self.space.reference
        settings = _gdf_backend_settings(self.reference)
        (
            self.recip_cut,
            self.pair_cut,
            self.mesh,
            _recip_key,
            self.kernel,
            self.omega,
            _kernel_key,
            _auto_info,
        ) = settings
        if self.kernel != "full":
            raise NotImplementedError(
                "Primitive q-resolved GDF derivatives currently require "
                "reciprocal_kernel='full'."
            )
        self.pair_screen_tol = _gdf_pair_screen_tol(self.reference)
        auxbasis = _gdf_auxbasis_name(self.reference)
        self.aux = _gdf_auxiliary_basis(
            self.space,
            auxbasis,
            _gdf_aux_coord_type(self.reference),
        )
        self._pair_plan_data = _gdf_pair_ft_plan_data(
            mf,
            self.pair_cut,
            self.pair_screen_tol,
        )
        self._one_body_pair_plan_data = _gdf_pair_ft_plan_data(
            mf,
            mf.pair_cut,
            mf.pair_ft_screen_tol,
        )
        helper = Gradients(mf)
        helper._atom_ids = _atom_ids_for_basis(
            mf._basis,
            np.asarray(mf.cell._atom_coords, dtype=float),
        )
        self._gradient_helper = helper
        cart_atom_ids = _atom_ids_for_basis(
            self.aux.cart_basis,
            np.asarray(mf.cell._atom_coords, dtype=float),
        )
        transform = np.asarray(self.aux.transform)
        self.aux_atom_ids = np.empty(self.aux.naux, dtype=int)
        for column in range(self.aux.naux):
            support = np.flatnonzero(np.abs(transform[:, column]) > 1.0e-14)
            atoms = np.unique(cart_atom_ids[support])
            if len(atoms) != 1:
                raise RuntimeError(
                    "An auxiliary function mixes centers in the GDF transform."
                )
            self.aux_atom_ids[column] = int(atoms[0])
        self.info = {
            "backend": "primitive_cell_reciprocal_gdf",
            "kernel": self.kernel,
            "qpoint": np.array(self.qpoint, copy=True),
            "q_index": int(self.q_index),
            "naux": int(self.aux.naux),
            "nao": int(mf.cell.nao),
            "nkpts": int(mf.nkpts),
            "temporary_supercell_nao": 0,
            "temporary_supercell_naux": 0,
            "cached_bytes": 0,
        }

    def _refresh_memory_info(self):
        arrays = []
        for cache in (
            self._reciprocal_cache,
            self._zero_ao_cache,
            self._derivative_ao_cache,
            self._inverse_derivative_cache,
        ):
            for value in cache.values():
                values = value if isinstance(value, tuple) else (value,)
                arrays.extend(item for item in values if isinstance(item, np.ndarray))
        self.info["cached_bytes"] = int(sum(array.nbytes for array in arrays))

    def _shift_q_index(self, source_q_index, sign):
        source = np.asarray(self.space.qpts[int(source_q_index)], dtype=float)
        return int(
            self.base.with_df.find_qpoint_index(
                source + int(sign) * self.qpoint
            )
        )

    def _reciprocal_data(self, q_index):
        from pyqed.pbc.gw.integrals import (
            _gdf_auxiliary_ft,
            _gdf_metric_invsqrt,
            _gdf_metric_relative_tol,
            _gdf_metric_tol,
            _gdf_precision,
            _gdf_reciprocal_coulomb_vectors,
            _periodic_gdf_aux_metric,
        )

        q_index = int(q_index)
        cached = self._reciprocal_cache.get(q_index)
        if cached is not None:
            return cached
        qvec = np.asarray(self.space.qpts[q_index], dtype=float)
        vectors, weights = _gdf_reciprocal_coulomb_vectors(
            self.base,
            qvec,
            self.base.with_df.g2_tol,
            recip_cut=self.recip_cut,
            mesh=self.mesh,
            kernel="full",
            omega=None,
        )
        auxiliary_ft = _gdf_auxiliary_ft(
            self.space,
            self.aux,
            vectors,
            cache_enabled=False,
        )
        metric = _periodic_gdf_aux_metric(
            self.space,
            q_index,
            self.aux,
            self.base.with_df.g2_tol,
        )
        threshold = max(
            float(self.base.with_df.g2_tol),
            float(_gdf_metric_tol(self.reference)),
        )
        inverse_sqrt, _evals = _gdf_metric_invsqrt(
            metric,
            threshold,
            self.aux.name,
            precision=_gdf_precision(self.reference),
            relative_threshold=_gdf_metric_relative_tol(self.reference),
        )
        inverse_metric = inverse_sqrt @ inverse_sqrt.conj().T
        cached = (
            np.ascontiguousarray(vectors),
            np.ascontiguousarray(weights),
            np.ascontiguousarray(auxiliary_ft),
            np.ascontiguousarray(inverse_metric),
        )
        self._reciprocal_cache[q_index] = cached
        self._refresh_memory_info()
        return cached

    def inverse_metric(self, q_index):
        """Return the primitive auxiliary inverse metric for one transfer."""
        return self._reciprocal_data(int(q_index))[3]

    def _directional_pair_ft(
        self,
        vectors,
        right_k_index,
        sign,
        *,
        plan_data=None,
    ):
        data = self._gradient_helper._build_pair_ft_gradient_data(
            self._pair_plan_data if plan_data is None else plan_data
        )
        sign = int(sign)
        mode_q = sign * self.qpoint
        mode = self.cartesian_mode if sign > 0 else self.cartesian_mode.conj()
        shifts = np.asarray(data["shift_array"], dtype=float)
        right_k = np.asarray(self.base.kpts[int(right_k_index)], dtype=float)
        left_phases = np.exp(1.0j * (right_k @ shifts.T))[None, :]
        right_phases = np.exp(
            1.0j * ((right_k + mode_q) @ shifts.T)
        )[None, :]
        common = {
            "gvecs": vectors,
            "compiled": bool(has_periodic_pair_ft_backend()),
            "threads": self.base.one_body_workers,
        }
        left_values = data["left_plan"].periodic_sum_many(
            left_origins=data["origins"],
            right_origins_batch=data["base_right_origins"],
            phases=left_phases,
            image_pair_mask=data["left_mask"],
            primitive_terms=data["left_terms"],
            **common,
        )[0]
        right_values = data["right_plan"].periodic_sum_many(
            left_origins=data["base_origins"],
            right_origins_batch=data["right_origins"],
            phases=right_phases,
            image_pair_mask=data["right_mask"],
            primitive_terms=data["right_terms"],
            **common,
        )[0]
        nao = int(self.base.cell.nao)
        out = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
        for derivative, (parent, axis) in enumerate(
            zip(data["parents"], data["axes"])
        ):
            atom = int(self._gradient_helper._atom_ids[parent])
            weight = mode[atom, axis]
            out[:, parent, :] += weight * left_values[:, derivative, :]
            out[:, :, parent] += weight * right_values[:, :, derivative]
        return out

    def _zero_pair_ao(self, left_k_index, right_k_index):
        from pyqed.pbc.gw.integrals import _periodic_gdf_three_center_ao

        key = (int(left_k_index), int(right_k_index))
        cached = self._zero_ao_cache.get(key)
        if cached is not None:
            return cached
        q_index = int(
            self.base.with_df.find_qpoint_index(
                self.base.kpts[key[1]] - self.base.kpts[key[0]]
            )
        )
        cached = np.ascontiguousarray(
            _periodic_gdf_three_center_ao(
                self.space,
                q_index,
                key[0],
                key[1],
                self.aux,
                self.base.with_df.g2_tol,
            )
        )
        self._zero_ao_cache[key] = cached
        self._refresh_memory_info()
        return cached

    def _derivative_pair_ao(self, left_k_index, right_k_index, sign):
        from pyqed.pbc.gw.integrals import _gdf_pair_ft_batch

        key = (int(left_k_index), int(right_k_index), int(sign))
        cached = self._derivative_ao_cache.get(key)
        if cached is not None:
            return cached
        source_q_index = int(
            self.base.with_df.find_qpoint_index(
                self.base.kpts[key[1]] - self.base.kpts[key[0]]
            )
        )
        target_q_index = self._shift_q_index(source_q_index, key[2])
        source_vectors, source_weights, source_aux, _source_inverse = (
            self._reciprocal_data(source_q_index)
        )
        target_vectors, target_weights, target_aux, _target_inverse = (
            self._reciprocal_data(target_q_index)
        )
        pair_source = _gdf_pair_ft_batch(
            self.space,
            source_vectors,
            self.base.kpts[key[1]],
            self.pair_cut,
            self.pair_screen_tol,
            cache_enabled=False,
        )
        pair_derivative = self._directional_pair_ft(
            target_vectors,
            key[1],
            key[2],
        )
        mode = self.cartesian_mode if key[2] > 0 else self.cartesian_mode.conj()
        phase_dot = np.einsum(
            "gi,Pi->gP",
            source_vectors,
            mode[self.aux_atom_ids],
            optimize=True,
        )
        auxiliary_bra_derivative = 1.0j * phase_dot * source_aux.conj()
        cached = np.einsum(
            "gP,g,gmn->Pmn",
            target_aux.conj(),
            target_weights,
            pair_derivative,
            optimize=True,
        )
        cached += np.einsum(
            "gP,g,gmn->Pmn",
            auxiliary_bra_derivative,
            source_weights,
            pair_source,
            optimize=True,
        )
        cached = np.ascontiguousarray(cached)
        self._derivative_ao_cache[key] = cached
        self._refresh_memory_info()
        return cached

    def pair_ao_factors(self, left_k_index, right_k_index):
        r"""Return :math:`(B,D_qB,D_{-q}B)` and their transfer indices."""
        left_k_index = int(left_k_index)
        right_k_index = int(right_k_index)
        source_q_index = int(
            self.base.with_df.find_qpoint_index(
                self.base.kpts[right_k_index] - self.base.kpts[left_k_index]
            )
        )
        return (
            self._zero_pair_ao(left_k_index, right_k_index),
            self._derivative_pair_ao(left_k_index, right_k_index, 1),
            self._derivative_pair_ao(left_k_index, right_k_index, -1),
            (
                source_q_index,
                self._shift_q_index(source_q_index, 1),
                self._shift_q_index(source_q_index, -1),
            ),
        )

    def inverse_metric_derivative(self, source_q_index, sign=1):
        r"""Return :math:`D_{\pm q}M^{-1}_{Q\pm q,Q}`."""
        source_q_index = int(source_q_index)
        sign = int(sign)
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")
        key = (source_q_index, sign)
        cached = self._inverse_derivative_cache.get(key)
        if cached is not None:
            return cached
        target_q_index = self._shift_q_index(source_q_index, sign)
        source_vectors, source_weights, source_aux, source_inverse = (
            self._reciprocal_data(source_q_index)
        )
        target_vectors, target_weights, target_aux, target_inverse = (
            self._reciprocal_data(target_q_index)
        )
        mode = self.cartesian_mode if sign > 0 else self.cartesian_mode.conj()
        source_dot = np.einsum(
            "gi,Pi->gP",
            source_vectors,
            mode[self.aux_atom_ids],
            optimize=True,
        )
        target_dot = np.einsum(
            "gi,Pi->gP",
            target_vectors,
            mode[self.aux_atom_ids],
            optimize=True,
        )
        bra_derivative = 1.0j * source_dot * source_aux.conj()
        ket_derivative = -1.0j * target_dot * target_aux
        metric_derivative = np.einsum(
            "gP,g,gQ->PQ",
            bra_derivative,
            source_weights,
            source_aux,
            optimize=True,
        )
        metric_derivative += np.einsum(
            "gP,g,gQ->PQ",
            target_aux.conj(),
            target_weights,
            ket_derivative,
            optimize=True,
        )
        cached = np.ascontiguousarray(
            -target_inverse @ metric_derivative @ source_inverse
        )
        self._inverse_derivative_cache[key] = cached
        self._refresh_memory_info()
        return cached

    def _compiled_real_space_one_body_blocks(self):
        from pyqed.qchem import basis as basis_module
        from pyqed.qchem.pbc.hf.ewald_rhf import (
            _gaussian_pair_ft_decay_bound,
        )

        compiled = getattr(
            getattr(basis_module, "_basis_cy", None),
            "compute_periodic_one_electron",
            None,
        )
        if compiled is None:
            return None

        mf = self.base
        helper = self._gradient_helper
        nao = int(mf.cell.nao)
        shifts = np.asarray(
            [mf._shift_vectors[key] for key in mf._shift_keys],
            dtype=float,
        )
        derivative_basis = []
        parents = []
        axes = []
        for parent, function in enumerate(mf._basis):
            for axis in range(3):
                for derivative in helper._basis_derivatives(function, axis):
                    derivative_basis.append(derivative)
                    parents.append(parent)
                    axes.append(axis)
        derivative_basis = tuple(derivative_basis)
        parents = np.asarray(parents, dtype=int)
        axes = np.asarray(axes, dtype=int)

        mask = np.zeros((len(shifts), nao, nao), dtype=np.uint8)
        for left_index, left in enumerate(mf._basis):
            for right_index, right in enumerate(mf._basis):
                if mf.one_body_screen_tol == 0.0:
                    mask[:, left_index, right_index] = 1
                    continue
                for image, shift in enumerate(shifts):
                    mask[image, left_index, right_index] = (
                        _gaussian_pair_ft_decay_bound(left, right, shift)
                        > mf.one_body_screen_tol
                    )

        def pack(functions):
            signatures = [
                basis_module._basis_signature(function)
                for function in functions
            ]
            values = basis_module._pack_signatures_for_numba(signatures)
            dtypes = (
                np.int64,
                np.float64,
                np.float64,
                np.float64,
                np.int64,
            )
            return tuple(
                np.ascontiguousarray(value, dtype=dtype)
                for value, dtype in zip(values, dtypes)
            )

        base = pack(mf._basis)
        derivative = pack(derivative_basis)
        sectors = {
            "left": (
                derivative,
                base,
                np.ascontiguousarray(mask[:, parents, :]),
            ),
            "right": (
                base,
                derivative,
                np.ascontiguousarray(mask[:, :, parents]),
            ),
        }
        lattice = np.ascontiguousarray(
            mf.cell.lattice_vectors,
            dtype=np.float64,
        )

        def evaluate(
            name,
            atom_coords,
            atom_charges,
            nuclear_keys,
            *,
            absolute_centers=False,
        ):
            left, right, sector_mask = sectors[name]
            right_origins = np.ascontiguousarray(
                right[1][None, :, :] + shifts[:, None, :],
                dtype=np.float64,
            )
            if absolute_centers:
                return compiled(
                    left[0],
                    left[1],
                    right_origins,
                    left[2],
                    left[3],
                    left[4],
                    np.ascontiguousarray(atom_coords, dtype=np.float64),
                    np.ascontiguousarray(atom_charges, dtype=np.float64),
                    float(mf.eta),
                    image_pair_mask=sector_mask,
                    nuclear_screen_tol=0.0,
                    right_shells=right[0],
                    right_exps=right[2],
                    right_weights=right[3],
                    right_nprim=right[4],
                )
            return compiled(
                left[0],
                left[1],
                right_origins,
                left[2],
                left[3],
                left[4],
                np.ascontiguousarray(atom_coords, dtype=np.float64),
                np.ascontiguousarray(atom_charges, dtype=np.float64),
                float(mf.eta),
                sector_mask,
                0.0,
                lattice,
                np.ascontiguousarray(nuclear_keys, dtype=np.int64),
                right[0],
                right[2],
                right[3],
                right[4],
            )

        coords = np.asarray(mf.cell._atom_coords, dtype=float)
        charges = np.asarray(mf.cell.ionic_charges, dtype=float)
        nuclear_keys = np.ascontiguousarray(
            list(mf.cell.image_keys(mf.one_body_nuclear_cut)),
            dtype=np.int64,
        )
        total_left = evaluate("left", coords, charges, nuclear_keys)
        total_right = evaluate("right", coords, charges, nuclear_keys)
        left_phases = np.exp(1.0j * (np.asarray(mf.kpts) @ shifts.T))
        right_phases = np.exp(
            1.0j
            * (
                (np.asarray(mf.kpts) + self.qpoint[None, :])
                @ shifts.T
            )
        )
        shape = (mf.nkpts, nao, nao)
        overlap = np.zeros(shape, dtype=np.complex128)
        kinetic = np.zeros_like(overlap)
        short_range = np.zeros_like(overlap)
        atom_ids = np.asarray(helper._atom_ids, dtype=int)

        for k_index in range(mf.nkpts):
            left_s, left_t, left_v = (
                np.einsum(
                    "i,ipq->pq",
                    left_phases[k_index],
                    values,
                    optimize=True,
                )
                for values in total_left
            )
            right_s, right_t, right_v = (
                np.einsum(
                    "i,ipq->pq",
                    right_phases[k_index],
                    values,
                    optimize=True,
                )
                for values in total_right
            )
            for derivative_index, (parent, axis) in enumerate(
                zip(parents, axes)
            ):
                coefficient = self.cartesian_mode[
                    atom_ids[parent], axis
                ]
                overlap[k_index, parent, :] += (
                    coefficient * left_s[derivative_index, :]
                )
                overlap[k_index, :, parent] += (
                    coefficient * right_s[:, derivative_index]
                )
                kinetic[k_index, parent, :] += (
                    coefficient * left_t[derivative_index, :]
                )
                kinetic[k_index, :, parent] += (
                    coefficient * right_t[:, derivative_index]
                )
                short_range[k_index, parent, :] += (
                    coefficient * left_v[derivative_index, :]
                )
                short_range[k_index, :, parent] += (
                    coefficient * right_v[:, derivative_index]
                )

        nuclear_calls = 0
        for atom, (coord, charge) in enumerate(zip(coords, charges)):
            for key in nuclear_keys:
                one_key = np.asarray(key, dtype=np.int64).reshape(1, 3)
                image_shift = np.asarray(key, dtype=float) @ lattice
                image_coord = coord + image_shift
                left_image = evaluate(
                    "left",
                    image_coord.reshape(1, 3),
                    np.asarray([charge]),
                    one_key,
                    absolute_centers=True,
                )[2]
                right_image = evaluate(
                    "right",
                    image_coord.reshape(1, 3),
                    np.asarray([charge]),
                    one_key,
                    absolute_centers=True,
                )[2]
                nuclear_calls += 2
                image_phase = np.exp(1.0j * np.dot(self.qpoint, image_shift))
                for k_index in range(mf.nkpts):
                    left_v = np.einsum(
                        "i,ipq->pq",
                        left_phases[k_index],
                        left_image,
                        optimize=True,
                    )
                    right_v = np.einsum(
                        "i,ipq->pq",
                        left_phases[k_index],
                        right_image,
                        optimize=True,
                    )
                    for derivative_index, (parent, axis) in enumerate(
                        zip(parents, axes)
                    ):
                        coefficient = (
                            image_phase * self.cartesian_mode[atom, axis]
                        )
                        short_range[k_index, parent, :] -= (
                            coefficient * left_v[derivative_index, :]
                        )
                        short_range[k_index, :, parent] -= (
                            coefficient * right_v[:, derivative_index]
                        )

        self.info["one_body_real_space_backend"] = "compiled_shell_integrals"
        self.info["one_body_nuclear_image_calls"] = int(nuclear_calls)
        return overlap, kinetic, short_range

    def _python_real_space_one_body_blocks(self):
        from pyqed.qchem.basis import S, T
        from pyqed.qchem.pbc.ewald import short_range_point_charge_s
        from pyqed.qchem.pbc.hf.ewald_rhf import (
            _gaussian_pair_ft_decay_bound,
        )

        mf = self.base
        helper = self._gradient_helper
        coords = np.asarray(mf.cell._atom_coords, dtype=float)
        charges = np.asarray(mf.cell.ionic_charges, dtype=float)
        nuclear_keys = mf.cell.image_keys(mf.one_body_nuclear_cut)
        shape = (mf.nkpts, mf.cell.nao, mf.cell.nao)
        overlap = np.zeros(shape, dtype=np.complex128)
        kinetic = np.zeros_like(overlap)
        short_range = np.zeros_like(overlap)
        for k_index, kpoint in enumerate(mf.kpts):
            for key in mf._shift_keys:
                shift = mf._shift_vectors[key]
                shifted_basis = mf._shifted_basis[key]
                left_phase = np.exp(1.0j * np.dot(kpoint, shift))
                right_phase = np.exp(
                    1.0j * np.dot(kpoint + self.qpoint, shift)
                )
                for left_index, left in enumerate(mf._basis):
                    left_atom = helper._atom_ids[left_index]
                    for right_index, right in enumerate(shifted_basis):
                        if (
                            mf.one_body_screen_tol > 0.0
                            and _gaussian_pair_ft_decay_bound(
                                left,
                                mf._basis[right_index],
                                shift,
                            )
                            <= mf.one_body_screen_tol
                        ):
                            continue
                        right_atom = helper._atom_ids[right_index]
                        for axis in range(3):
                            left_weight = self.cartesian_mode[left_atom, axis]
                            right_weight = self.cartesian_mode[right_atom, axis]
                            overlap[k_index, left_index, right_index] += (
                                left_weight
                                * left_phase
                                * helper._pair_center_derivative(
                                    S, left, right, axis, 0
                                )
                                + right_weight
                                * right_phase
                                * helper._pair_center_derivative(
                                    S, left, right, axis, 1
                                )
                            )
                            kinetic[k_index, left_index, right_index] += (
                                left_weight
                                * left_phase
                                * helper._pair_center_derivative(
                                    T, left, right, axis, 0
                                )
                                + right_weight
                                * right_phase
                                * helper._pair_center_derivative(
                                    T, left, right, axis, 1
                                )
                            )
                            for nuclear_key in nuclear_keys:
                                nuclear_shift = mf.cell.translation_vector(
                                    nuclear_key
                                )
                                image_phase = np.exp(
                                    1.0j
                                    * np.dot(self.qpoint, nuclear_shift)
                                )
                                for atom, (charge, center) in enumerate(
                                    zip(charges, coords)
                                ):
                                    image_center = center + nuclear_shift

                                    def kernel(a, b):
                                        return short_range_point_charge_s(
                                            a,
                                            b,
                                            image_center,
                                            mf.eta,
                                        )

                                    dleft = helper._pair_center_derivative(
                                        kernel, left, right, axis, 0
                                    )
                                    dright = helper._pair_center_derivative(
                                        kernel, left, right, axis, 1
                                    )
                                    factor = float(charge)
                                    short_range[
                                        k_index, left_index, right_index
                                    ] -= factor * (
                                        left_weight * left_phase * dleft
                                        + right_weight * right_phase * dright
                                    )
                                    short_range[
                                        k_index, left_index, right_index
                                    ] += (
                                        factor
                                        * self.cartesian_mode[atom, axis]
                                        * image_phase
                                        * left_phase
                                        * (dleft + dright)
                                    )
        self.info["one_body_real_space_backend"] = "python_shell_integrals"
        return overlap, kinetic, short_range

    def _reciprocal_nuclear_derivative_blocks(self):
        mf = self.base
        coords = np.asarray(mf.cell._atom_coords, dtype=float)
        charges = np.asarray(mf.cell.ionic_charges, dtype=float)
        values = mf._reciprocal_g_weights(include_zero=True)
        out = np.zeros(
            (mf.nkpts, mf.cell.nao, mf.cell.nao),
            dtype=np.complex128,
        )
        if not values:
            return out
        base_vectors = np.asarray(
            [vector for vector, _weight in values], dtype=float
        )
        base_weights = np.asarray(
            [weight for _vector, weight in values], dtype=float
        )

        g2 = np.einsum("gi,gi->g", base_vectors, base_vectors)
        keep = g2 > 1.0e-16
        if np.any(keep):
            vectors = base_vectors[keep]
            weights = base_weights[keep]
            g2_kept = g2[keep]
            nuclear_density = np.einsum(
                "a,ag->g",
                charges,
                np.exp(-1.0j * (coords @ vectors.T)),
                optimize=True,
            )
            coefficient = (
                -4.0
                * np.pi
                * weights
                * np.exp(-g2_kept / (4.0 * mf.eta * mf.eta))
                / g2_kept
            )
            for k_index in range(mf.nkpts):
                pair_derivative = self._directional_pair_ft(
                    -vectors,
                    k_index,
                    1,
                    plan_data=self._one_body_pair_plan_data,
                )
                out[k_index] += np.einsum(
                    "g,g,gpq->pq",
                    coefficient,
                    nuclear_density,
                    pair_derivative,
                    optimize=True,
                )

        shifted_vectors = base_vectors + self.qpoint[None, :]
        shifted2 = np.einsum(
            "gi,gi->g", shifted_vectors, shifted_vectors
        )
        keep = shifted2 > 1.0e-16
        if np.any(keep):
            vectors = shifted_vectors[keep]
            weights = base_weights[keep]
            shifted2 = shifted2[keep]
            phases = np.exp(-1.0j * (coords @ vectors.T))
            displacement_dot = np.einsum(
                "gi,ai->ag",
                vectors,
                self.cartesian_mode,
                optimize=True,
            )
            nuclear_derivative = np.einsum(
                "a,ag,ag->g",
                -1.0j * charges,
                phases,
                displacement_dot,
                optimize=True,
            )
            coefficient = (
                -4.0
                * np.pi
                * weights
                * np.exp(-shifted2 / (4.0 * mf.eta * mf.eta))
                / shifted2
            )
            for k_index, kpoint in enumerate(mf.kpts):
                pair = mf._periodic_pair_ft_batch(-vectors, kpoint)
                out[k_index] += np.einsum(
                    "g,g,gpq->pq",
                    coefficient,
                    nuclear_derivative,
                    pair,
                    optimize=True,
                )
        return out

    def one_electron_derivative_blocks(self):
        r"""Return primitive :math:`k\rightarrow k+q` overlap and core blocks."""
        if self.base.cell.has_pseudo:
            raise NotImplementedError(
                "Primitive finite-q GTH pseudopotential derivatives are not "
                "implemented yet."
            )
        started = time.perf_counter()
        real_space = self._compiled_real_space_one_body_blocks()
        if real_space is None:
            real_space = self._python_real_space_one_body_blocks()
        overlap, kinetic, short_range = real_space
        reciprocal = self._reciprocal_nuclear_derivative_blocks()
        volume = abs(float(np.linalg.det(self.base.cell.lattice_vectors)))
        background = (
            np.pi
            * float(np.sum(self.base.cell.ionic_charges))
            / (self.base.eta * self.base.eta * volume)
            if self.base.nuclear_background
            else 0.0
        )
        hcore = kinetic + short_range + reciprocal + background * overlap
        self.info["one_body_seconds"] = float(time.perf_counter() - started)
        return (
            tuple(np.ascontiguousarray(block) for block in overlap),
            tuple(np.ascontiguousarray(block) for block in hcore),
        )

    def fixed_density_veff_derivative_blocks(self, densities, overlap_blocks):
        r"""Return fixed-density primitive GDF :math:`J-\tfrac12K` blocks."""
        mf = self.base
        densities = tuple(
            np.asarray(block, dtype=np.complex128) for block in densities
        )
        if len(densities) != mf.nkpts:
            raise ValueError("densities must provide one block per k point.")
        nao = int(mf.cell.nao)
        if any(block.shape != (nao, nao) for block in densities):
            raise ValueError(f"Each density block must have shape ({nao}, {nao}).")
        overlap_blocks = tuple(
            np.asarray(block, dtype=np.complex128)
            for block in overlap_blocks
        )
        pair_by_k = {
            int(k): int(kq) for k, kq in mf.with_df.pair_keys(self.q_index)
        }
        zero_q_index = int(mf.with_df.find_qpoint_index(np.zeros(3)))
        rho_zero = np.zeros(self.aux.naux, dtype=np.complex128)
        rho_minus = np.zeros_like(rho_zero)
        for k_index, density in enumerate(densities):
            zero, _plus, minus, transfers = self.pair_ao_factors(
                k_index, k_index
            )
            if transfers[0] != zero_q_index or transfers[2] != self.minus_q_index:
                raise RuntimeError("Inconsistent diagonal GDF momentum sectors.")
            rho_zero += np.einsum(
                "Pij,ji->P", zero, density, optimize=True
            )
            rho_minus += np.einsum(
                "Pij,ji->P", minus, density, optimize=True
            )

        inverse_zero = self.inverse_metric(zero_q_index)
        inverse_q = self.inverse_metric(self.q_index)
        inverse_q_zero = self.inverse_metric_derivative(
            zero_q_index,
            sign=1,
        )
        blocks = []
        for k_index in range(mf.nkpts):
            kq_index = pair_by_k[k_index]
            external_zero, external_plus, _external_minus, transfers = (
                self.pair_ao_factors(kq_index, k_index)
            )
            if (
                transfers[0] != self.minus_q_index
                or transfers[1] != zero_q_index
            ):
                raise RuntimeError("Inconsistent Hartree GDF momentum sectors.")
            hartree = np.einsum(
                "Pij,PQ,Q->ij",
                external_plus,
                inverse_zero,
                rho_zero.conj(),
                optimize=True,
            )
            hartree += np.einsum(
                "Pij,PQ,Q->ij",
                external_zero,
                inverse_q_zero,
                rho_zero.conj(),
                optimize=True,
            )
            hartree += np.einsum(
                "Pij,PQ,Q->ij",
                external_zero,
                inverse_q,
                rho_minus.conj(),
                optimize=True,
            )

            exchange = np.zeros((nao, nao), dtype=np.complex128)
            for source_k, density in enumerate(densities):
                left_zero, left_plus, _left_minus, left_transfers = (
                    self.pair_ao_factors(kq_index, source_k)
                )
                right_zero, _right_plus, right_minus, right_transfers = (
                    self.pair_ao_factors(k_index, source_k)
                )
                if left_transfers[1] != right_transfers[0]:
                    raise RuntimeError(
                        "Inconsistent differentiated exchange momentum sectors."
                    )
                if right_transfers[2] != left_transfers[0]:
                    raise RuntimeError(
                        "Inconsistent conjugate exchange momentum sectors."
                    )
                exchange += np.einsum(
                    "Pim,mn,PQ,Qjn->ij",
                    left_plus,
                    density,
                    self.inverse_metric(right_transfers[0]).T,
                    right_zero.conj(),
                    optimize=True,
                )
                exchange += np.einsum(
                    "Pim,mn,PQ,Qjn->ij",
                    left_zero,
                    density,
                    self.inverse_metric_derivative(
                        left_transfers[0],
                        sign=1,
                    ).T,
                    right_zero.conj(),
                    optimize=True,
                )
                exchange += np.einsum(
                    "Pim,mn,PQ,Qjn->ij",
                    left_zero,
                    density,
                    self.inverse_metric(left_transfers[0]).T,
                    right_minus.conj(),
                    optimize=True,
                )

            veff = (hartree - 0.5 * exchange) / float(mf.nkpts)
            if mf.madelung is not None:
                overlap1 = overlap_blocks[k_index]
                veff -= 0.5 * mf.madelung * (
                    overlap1 @ densities[k_index] @ mf._overlap_k[k_index]
                    + mf._overlap_k[kq_index]
                    @ densities[kq_index]
                    @ overlap1
                )
            blocks.append(np.ascontiguousarray(veff))
        self._refresh_memory_info()
        self.info["fixed_density_pair_blocks"] = int(
            len(self._zero_ao_cache)
        )
        return tuple(blocks)


class CommensurateGDFQDerivative:
    r"""Build an analytic finite-q KRHF perturbation through a supercell.

    For a diagonal Born-von Karman supercell containing :math:`N` primitive
    cells, one-k GDF nuclear derivatives at the common folded twist are
    contracted as

    .. math::

       F_{q\nu}^{[1],\mathrm{SC}}=
       \sum_{R A\alpha} e^{i q\cdot R}
       \frac{e_{A\alpha,\nu}(q)}{\sqrt{M_A}}
       \frac{\partial F^{\mathrm{SC}}}{\partial R_{A\alpha}},

    then folded to primitive :math:`k\rightarrow k+q` AO blocks.  Static
    periodic CPHF adds the induced Hartree--Fock potential.  The method is an
    exact commensurate-supercell formulation within the native integral and
    GDF cutoffs; it is not a primitive-cell DFPT kernel and its temporary
    memory grows with the commensurate supercell.  ``info['reference_residuals']``
    reports zero-order overlap, core-Hamiltonian, Fock, and density equality;
    converge the real/pair and reciprocal domains before using the derivative
    quantitatively.
    """

    def __init__(
        self,
        mean_field,
        qpoint,
        mode_vector,
        *,
        mesh=None,
        cphf_tol=1.0e-9,
        cphf_max_cycle=80,
        cphf_level_shift=0.0,
    ):
        self.base = mean_field
        self.qpoint = np.asarray(qpoint, dtype=float)
        if self.qpoint.shape != (3,):
            raise ValueError("qpoint must contain three Cartesian components.")
        self.mesh = (
            tuple(
                int(value)
                for value in _infer_kmesh(
                    mean_field.cell.lattice_vectors,
                    mean_field.kpts,
                )
            )
            if mesh is None
            else tuple(int(value) for value in mesh)
        )
        self.mode_vector_input = np.asarray(mode_vector, dtype=np.complex128)
        self.cphf_tol = float(cphf_tol)
        self.cphf_max_cycle = int(cphf_max_cycle)
        self.cphf_level_shift = float(cphf_level_shift)

        self.transform = None
        self.supercell_mean_field = None
        self.supercell_density = None
        self.mode_vector = None
        self.cartesian_mode = None
        self.explicit_fock_derivative = None
        self.overlap_derivative = None
        self.fock_derivative = None
        self.induced_fock_derivative = None
        self.response = None
        self.gradient = None
        self.info = None
        self.success = False
        self.message = "not run"
        self.star_symmetry_residuals = {}

    def _validate(self):
        mf = self.base
        if not getattr(mf, "converged", False):
            raise RuntimeError("Run and converge KRHF before building a q derivative.")
        if int(mf.cell.dimension) != 3:
            raise NotImplementedError("Commensurate q derivatives require dimension=3.")
        if str(mf.jk_builder) != "gdf" or mf.with_df is None:
            raise NotImplementedError(
                "Commensurate analytic q derivatives require jk_builder='gdf'."
            )
        if int(np.prod(self.mesh)) != int(mf.nkpts):
            raise ValueError("The inferred supercell mesh does not match the SCF k mesh.")
        transform = CommensurateSupercell(mf.cell, self.mesh)
        transform.validate_qpoint(self.qpoint)
        q_index = int(mf.with_df.find_qpoint_index(self.qpoint))
        minus_q_index = int(mf.with_df.find_qpoint_index(-self.qpoint))

        natom = len(mf.cell._atom_coords)
        if self.mode_vector_input.size != 3 * natom:
            raise ValueError(f"mode_vector must contain {3 * natom} components.")
        mode = np.asarray(
            self.mode_vector_input.reshape(natom, 3),
            dtype=np.complex128,
        )
        norm = float(np.linalg.norm(mode))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("mode_vector must have finite nonzero norm.")
        mode /= norm
        if q_index == minus_q_index and np.max(np.abs(mode.imag)) > 1.0e-12:
            raise ValueError(
                "Self-opposite q-point mode vectors must be real; choose the "
                "equivalent real phonon gauge."
            )
        if q_index == minus_q_index:
            mode = mode.real.astype(np.complex128)
        masses = np.asarray(
            mf.cell.unit_molecule.atom_mass_list(),
            dtype=float,
        ) * amu_to_au
        self.transform = transform
        self.mode_vector = mode
        self.cartesian_mode = mode / np.sqrt(masses)[:, None]
        self.mode_input_norm = norm

    def _scf_options(self, mean_field):
        recip_cut = getattr(mean_field, "_recip_cut_request", mean_field.recip_cut)
        recip_bounds = mean_field.recip_bounds
        if isinstance(recip_cut, (int, np.integer)):
            primitive_bounds = (
                (int(recip_cut),) * 3
                if recip_bounds is None
                else tuple(int(value) for value in recip_bounds)
            )
            recip_bounds = tuple(
                int(mesh) * int(bound)
                for mesh, bound in zip(self.mesh, primitive_bounds)
            )
            recip_cut = max(recip_bounds)
        return {
            "eta": mean_field.eta,
            "real_cut": mean_field.real_cut,
            "pair_cut": mean_field.pair_cut,
            "recip_cut": recip_cut,
            "recip_bounds": recip_bounds,
            "recip_precision": mean_field.recip_precision,
            "recip_max_cut": mean_field.recip_max_cut,
            "damping": mean_field.damping,
            "nuclear_background": mean_field.nuclear_background,
            "eri_screen_tol": mean_field.eri_screen_tol,
            "jk_builder": "gdf",
            "pair_ft_screen_tol": mean_field.pair_ft_screen_tol,
            "occupation_mode": mean_field.occupation_mode,
            "occupation_tol": mean_field.occupation_tol,
            "pseudo_cut": mean_field.pseudo_cut,
            "pseudo_local_screen_tol": mean_field.pseudo_local_screen_tol,
            "one_body_screen_tol": mean_field.one_body_screen_tol,
            "one_body_nuclear_cut": mean_field.one_body_nuclear_cut,
            "one_body_workers": mean_field.one_body_workers,
            "diis": mean_field.diis,
            "diis_space": mean_field.diis_space,
            "diis_start_cycle": mean_field.diis_start_cycle,
        }

    def _gdf_options(self, mean_field):
        backend = mean_field.with_df
        reciprocal_mesh = backend.mesh
        if reciprocal_mesh is None and backend.recip_cut is not None:
            cut = int(backend.recip_cut)
            reciprocal_mesh = tuple(
                2 * (int(mesh) * cut + int(mesh) // 2) + 1
                for mesh in self.mesh
            )
        elif reciprocal_mesh not in (None, "auto"):
            primitive_mesh = np.asarray(reciprocal_mesh, dtype=int).reshape(3)
            reciprocal_mesh = tuple(
                2
                * (
                    int(mesh) * (int(size) // 2)
                    + int(mesh) // 2
                )
                + 1
                for mesh, size in zip(self.mesh, primitive_mesh)
            )
        return {
            "auxbasis": backend.auxbasis,
            "precision": backend.precision,
            "reciprocal_kernel": backend.reciprocal_kernel,
            "recip_cut": backend.recip_cut,
            "omega": backend.omega,
            "mesh": reciprocal_mesh,
            "pair_cut": backend.pair_cut,
            "pair_screen_tol": backend.pair_screen_tol,
            "image_cut": backend.image_cut,
            "metric_tol": backend.metric_tol,
            "g2_tol": backend.g2_tol,
            "storage": "memory",
            "max_memory_mb": backend.max_memory_mb,
            "release_raw_ao": backend.release_raw_ao,
            "stream_pairs": backend.stream_pairs,
            "stream_pair_batch_size": backend.stream_pair_batch_size,
            "stream_pair_batch_mb": backend.stream_pair_batch_mb,
        }

    def _build_supercell_reference(self):
        super_cell = self.transform.build_cell()
        supercell_twist = self.transform.common_twist(self.base.kpts)
        super_mf = super_cell.KRHF(
            kpts=supercell_twist.reshape(1, 3),
            **self._scf_options(self.base),
        )
        for name, value in vars(self.base).items():
            if name.startswith("gdf_") or name.startswith("df_"):
                setattr(super_mf, name, copy.deepcopy(value))
        super_mf.density_fit(**self._gdf_options(self.base))
        super_mf._build_integrals()
        return super_mf

    def _induced_fock(self, density_derivative):
        mf = self.base
        q_index = int(mf.with_df.find_qpoint_index(self.qpoint))
        pair_by_k = {
            int(k): int(kq) for k, kq in mf.with_df.pair_keys(q_index)
        }
        densities = [
            np.asarray(block, dtype=np.complex128)
            for block in density_derivative
        ]
        vj, vk = mf.with_df.get_jk_response(densities, q_index)
        induced = [
            np.asarray(vj[k]) - 0.5 * np.asarray(vk[k])
            for k in range(mf.nkpts)
        ]
        if mf.madelung is not None:
            for k_index, kq_index in pair_by_k.items():
                induced[k_index] -= 0.5 * mf.madelung * (
                    mf._overlap_k[kq_index]
                    @ densities[k_index]
                    @ mf._overlap_k[k_index]
                )
        return tuple(np.ascontiguousarray(block) for block in induced)

    def _enforce_self_opposite_star(self, blocks, name):
        """Project q=-q blocks onto their exact Hermitian star relation."""
        mf = self.base
        q_index = int(mf.with_df.find_qpoint_index(self.qpoint))
        minus_q_index = int(mf.with_df.find_qpoint_index(-self.qpoint))
        blocks = [np.asarray(block, dtype=np.complex128) for block in blocks]
        if q_index != minus_q_index:
            self.star_symmetry_residuals[name] = None
            return tuple(np.ascontiguousarray(block) for block in blocks)
        pair_by_k = {
            int(k): int(kq) for k, kq in mf.with_df.pair_keys(q_index)
        }
        residual = 0.0
        projected = [np.array(block, copy=True) for block in blocks]
        visited = set()
        for k_index, kq_index in pair_by_k.items():
            residual = max(
                residual,
                float(np.max(np.abs(blocks[kq_index] - blocks[k_index].conj().T))),
            )
            pair = tuple(sorted((k_index, kq_index)))
            if pair in visited:
                continue
            visited.add(pair)
            average = 0.5 * (blocks[k_index] + blocks[kq_index].conj().T)
            projected[k_index] = average
            projected[kq_index] = average.conj().T
        self.star_symmetry_residuals[name] = residual
        return tuple(np.ascontiguousarray(block) for block in projected)

    def _reference_residuals(self, super_mf, super_dm):
        """Measure zero-order primitive/supercell representation equality."""

        primitive_dm = [
            np.asarray(self.base.dm)
        ] if self.base.nkpts == 1 else [
            np.asarray(block) for block in self.base.dm
        ]
        primitive_overlap = self.transform.embed_operator(
            self.base._overlap_k,
            self.base.kpts,
            np.zeros(3),
        )
        primitive_hcore = self.transform.embed_operator(
            self.base._hcore_k,
            self.base.kpts,
            np.zeros(3),
        )
        primitive_fock = self.transform.embed_operator(
            self.base._build_fock_k(primitive_dm),
            self.base.kpts,
            np.zeros(3),
        )
        super_overlap = np.asarray(super_mf._overlap_k[0])
        super_hcore = np.asarray(super_mf._hcore_k[0])
        super_fock = np.asarray(super_mf._build_fock_k([super_dm])[0])

        def relative_error(actual, expected):
            difference = float(np.linalg.norm(actual - expected))
            scale = max(float(np.linalg.norm(expected)), 1.0e-30)
            return difference, difference / scale

        overlap_error = relative_error(super_overlap, primitive_overlap)
        hcore_error = relative_error(super_hcore, primitive_hcore)
        fock_error = relative_error(super_fock, primitive_fock)
        stationary_dm = super_mf._solve_fock(
            [super_fock],
            [super_overlap],
        )[3][0]
        density_error = relative_error(stationary_dm, super_dm)
        return {
            "overlap_absolute": overlap_error[0],
            "overlap_relative": overlap_error[1],
            "hcore_absolute": hcore_error[0],
            "hcore_relative": hcore_error[1],
            "fock_absolute": fock_error[0],
            "fock_relative": fock_error[1],
            "density_absolute": density_error[0],
            "density_relative": density_error[1],
        }

    def run(self):
        started = time.perf_counter()
        self._validate()
        super_mf = self._build_supercell_reference()
        primitive_dm = (
            [np.asarray(self.base.dm)]
            if int(self.base.nkpts) == 1
            else [np.asarray(block) for block in self.base.dm]
        )
        super_dm = self.transform.embed_density(primitive_dm, self.base.kpts)
        reference_residuals = self._reference_residuals(super_mf, super_dm)
        gradient = super_mf.nuc_grad_method()
        mode_weights = self.transform.mode_weights(
            self.cartesian_mode,
            self.qpoint,
        ).reshape(self.transform.ncell * self.transform.natom, 3)
        mode_s1, mode_h1, mode_veff1 = (
            gradient.directional_integral_derivatives(
                mode_weights,
                super_dm,
                require_scf=False,
            )
        )
        mode_explicit_fock1 = mode_h1 + mode_veff1
        overlap_q = self.transform.fold_operator(
            mode_s1,
            self.base.kpts,
            self.qpoint,
        )
        explicit_fock_q = self.transform.fold_operator(
            mode_explicit_fock1,
            self.base.kpts,
            self.qpoint,
        )
        overlap_q = self._enforce_self_opposite_star(overlap_q, "overlap")
        explicit_fock_q = self._enforce_self_opposite_star(
            explicit_fock_q,
            "explicit_fock",
        )
        response = self.base.response().kernel(
            explicit_fock_q,
            s1=overlap_q,
            qpoint=self.qpoint,
            tol=self.cphf_tol,
            max_cycle=self.cphf_max_cycle,
            level_shift=self.cphf_level_shift,
        )
        if not response.converged:
            raise RuntimeError("Periodic finite-q CPHF did not converge.")
        density_q = (
            [np.asarray(response.dm1[0])]
            if int(self.base.nkpts) == 1
            else [np.asarray(block[0]) for block in response.dm1]
        )
        induced_fock_q = self._induced_fock(density_q)
        induced_fock_q = self._enforce_self_opposite_star(
            induced_fock_q,
            "induced_fock",
        )
        total_fock_q = tuple(
            np.ascontiguousarray(explicit + induced)
            for explicit, induced in zip(explicit_fock_q, induced_fock_q)
        )

        self.supercell_mean_field = super_mf
        self.supercell_density = super_dm
        self.explicit_fock_derivative = explicit_fock_q
        self.overlap_derivative = overlap_q
        self.induced_fock_derivative = induced_fock_q
        self.fock_derivative = total_fock_q
        self.response = response
        self.gradient = gradient
        self.info = {
            "backend": "commensurate_twisted_supercell_gdf",
            "mesh": tuple(self.mesh),
            "ncell": int(self.transform.ncell),
            "qpoint": np.array(self.qpoint, copy=True),
            "mode_input_norm": float(self.mode_input_norm),
            "cphf_residual_norm": float(response.residual_norm),
            "cphf_iterations": int(response.niter),
            "seconds": float(time.perf_counter() - started),
            "temporary_supercell_nao": int(self.transform.super_nao),
            "supercell_twist": np.array(super_mf.kpts[0], copy=True),
            "supercell_recip_cut": super_mf.recip_cut,
            "supercell_recip_bounds": super_mf.recip_bounds,
            "supercell_gdf_mesh": (
                super_mf.with_df.mesh if super_mf.with_df is not None else None
            ),
            "directional_response": dict(gradient.directional_response_info),
            "star_symmetry_residuals": dict(self.star_symmetry_residuals),
            "reference_residuals": reference_residuals,
            "approximation": "analytic_commensurate_supercell_not_primitive_dfpt",
        }
        self.success = True
        self.message = "analytic commensurate-q derivative built"
        return self

    kernel = run


class PrimitiveGDFQDerivative(CommensurateGDFQDerivative):
    r"""Build a full-reciprocal finite-:math:`q` KRHF perturbation directly.

    The driver evaluates primitive :math:`k\rightarrow k+q` overlap,
    core-Hamiltonian, and fixed-density GDF Fock blocks, then solves the
    existing primitive finite-momentum CPHF equations.  No supercell AO or
    auxiliary tensor is constructed.  The current implementation is an
    analytic primitive-cell adaptation of periodic GDF and supports
    all-electron three-dimensional cells with the full reciprocal Coulomb
    kernel.  GTH pseudopotential and range-separated short-range derivatives
    remain on the commensurate reference path.

    The GDF architecture follows Q. Sun et al., J. Chem. Phys. 147, 164119
    (2017), DOI: 10.1063/1.4998644.  The finite-momentum convention follows
    F. Giustino, Rev. Mod. Phys. 89, 015003 (2017),
    DOI: 10.1103/RevModPhys.89.015003.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primitive_engine = None

    def run(self):
        started = time.perf_counter()
        self._validate()
        engine = PrimitiveGDFQDerivativeEngine(
            self.base,
            self.qpoint,
            self.cartesian_mode,
        )
        primitive_dm = (
            [np.asarray(self.base.dm)]
            if int(self.base.nkpts) == 1
            else [np.asarray(block) for block in self.base.dm]
        )
        overlap_q, hcore_q = engine.one_electron_derivative_blocks()
        veff_q = engine.fixed_density_veff_derivative_blocks(
            primitive_dm,
            overlap_q,
        )
        explicit_fock_q = tuple(
            np.ascontiguousarray(hcore + veff)
            for hcore, veff in zip(hcore_q, veff_q)
        )
        overlap_q = self._enforce_self_opposite_star(overlap_q, "overlap")
        explicit_fock_q = self._enforce_self_opposite_star(
            explicit_fock_q,
            "explicit_fock",
        )
        response = self.base.response().kernel(
            explicit_fock_q,
            s1=overlap_q,
            qpoint=self.qpoint,
            tol=self.cphf_tol,
            max_cycle=self.cphf_max_cycle,
            level_shift=self.cphf_level_shift,
        )
        if not response.converged:
            raise RuntimeError("Periodic finite-q CPHF did not converge.")
        density_q = (
            [np.asarray(response.dm1[0])]
            if int(self.base.nkpts) == 1
            else [np.asarray(block[0]) for block in response.dm1]
        )
        induced_fock_q = self._induced_fock(density_q)
        induced_fock_q = self._enforce_self_opposite_star(
            induced_fock_q,
            "induced_fock",
        )
        total_fock_q = tuple(
            np.ascontiguousarray(explicit + induced)
            for explicit, induced in zip(explicit_fock_q, induced_fock_q)
        )

        self.primitive_engine = engine
        self.explicit_fock_derivative = explicit_fock_q
        self.overlap_derivative = overlap_q
        self.induced_fock_derivative = induced_fock_q
        self.fock_derivative = total_fock_q
        self.response = response
        self.gradient = None
        self.info = {
            "backend": "primitive_cell_full_reciprocal_gdf",
            "mesh": tuple(self.mesh),
            "ncell": int(self.transform.ncell),
            "qpoint": np.array(self.qpoint, copy=True),
            "mode_input_norm": float(self.mode_input_norm),
            "cphf_residual_norm": float(response.residual_norm),
            "cphf_iterations": int(response.niter),
            "seconds": float(time.perf_counter() - started),
            "temporary_supercell_nao": 0,
            "temporary_supercell_naux": 0,
            "star_symmetry_residuals": dict(self.star_symmetry_residuals),
            "primitive_engine": dict(engine.info),
            "approximation": (
                "analytic_primitive_full_reciprocal_gdf_fixed_orbital_"
                "integrals_plus_cphf"
            ),
        }
        self.success = True
        self.message = "analytic primitive-cell q derivative built"
        return self

    kernel = run


def commensurate_gdf_q_derivative(mean_field, qpoint, mode_vector, **kwargs):
    """Build and run :class:`CommensurateGDFQDerivative`."""
    return CommensurateGDFQDerivative(
        mean_field,
        qpoint,
        mode_vector,
        **kwargs,
    ).run()


def gdf_q_derivative(mean_field, qpoint, mode_vector, **kwargs):
    """Build the production finite-q GDF derivative for one KRHF reference."""
    try:
        return PrimitiveGDFQDerivative(
            mean_field,
            qpoint,
            mode_vector,
            **kwargs,
        ).run()
    except NotImplementedError:
        return commensurate_gdf_q_derivative(
            mean_field,
            qpoint,
            mode_vector,
            **kwargs,
        )


__all__ = [
    "CommensurateGDFQDerivative",
    "PrimitiveGDFQDerivative",
    "PrimitiveGDFQDerivativeEngine",
    "commensurate_gdf_q_derivative",
    "gdf_q_derivative",
]
