"""SU(2) native-system and local-action kernel helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


_USE_CYTHON_PARENT_BLOCK_BATCH = False


@lru_cache(maxsize=1)
def _cpp_module():
    try:
        from pyqed.mps.nonabelian import _su2_kernel as module
    except Exception:
        return None
    return module


def cpp_available():
    """Return whether the compiled SU(2) engine module exists."""

    return _cpp_module() is not None


def cython_available():
    """Compatibility alias for :func:`cpp_available`."""

    return cpp_available()


def resolve_backend(requested):
    """Resolve a requested SU(2) kernel backend."""

    requested = str(requested or "auto").lower().replace("-", "_")
    if requested == "default":
        requested = "auto"
    if requested == "cython":
        requested = "cpp"
    if requested not in {"auto", "cpp", "python"}:
        raise ValueError("su2_kernel_backend must be 'auto', 'cpp', or 'python'.")
    if requested == "cpp":
        if not cpp_available():
            raise RuntimeError(
                "su2_kernel_backend='cpp' requested, but "
                "pyqed.mps.nonabelian._su2_kernel is not importable."
            )
        return "cpp"
    if requested == "auto":
        return "cpp" if cpp_available() else "python"
    return "python"


def _resolve_helper_backend(requested, helper_name):
    actual = resolve_backend(requested)
    if actual == "python":
        return actual, None
    module = _cpp_module()
    helper = None if module is None else getattr(module, helper_name, None)
    if helper is None:
        if str(requested or "auto").lower().replace("-", "_") in {"cpp", "cython"}:
            raise RuntimeError(
                f"su2_kernel_backend='cpp' requested, but {helper_name} "
                "is not available from pyqed.mps.nonabelian._su2_kernel."
            )
        return "python", None
    return actual, helper


def build_component_parent_blocks(plan, component_dims, *, backend="auto"):
    """Build component parent blocks with the optional compiled helper."""

    actual, helper = _resolve_helper_backend(backend, "build_component_parent_blocks")
    if helper is None:
        return None, actual
    return helper(plan, tuple(int(dim) for dim in component_dims)), actual


def project_component_orthonormal_blocks(parent_blocks, transforms, max_elements, *, backend="auto"):
    """Project component parent blocks with the optional compiled helper."""

    if parent_blocks is None:
        return None, resolve_backend(backend)
    parent_blocks = tuple(parent_blocks)
    total_elements = sum(
        int(transforms[int(out_comp)].shape[1])
        * int(transforms[int(in_comp)].shape[1])
        for in_comp, out_comp, _block in parent_blocks
    )
    if total_elements > int(max_elements):
        return None, resolve_backend(backend)
    requested = str(backend or "auto").lower().replace("-", "_")
    if requested in {"auto", "default"}:
        return (
            _project_component_orthonormal_blocks_python(
                parent_blocks,
                transforms,
                max_elements,
            ),
            "python",
        )
    actual, helper = _resolve_helper_backend(
        backend,
        "project_component_orthonormal_blocks",
    )
    if helper is None:
        return None, actual
    packed_transforms = tuple(np.asarray(transform, dtype=complex) for transform in transforms)
    return helper(parent_blocks, packed_transforms, int(max_elements)), actual


def _project_component_orthonormal_blocks_python(parent_blocks, transforms, max_elements):
    """Project parent blocks, choosing the cheaper dense-multiply order."""

    if parent_blocks is None:
        return None
    parent_blocks = tuple(parent_blocks)
    transforms = tuple(np.asarray(transform, dtype=complex) for transform in transforms)
    total_elements = 0
    for in_comp, out_comp, _block in parent_blocks:
        total_elements += (
            int(transforms[int(out_comp)].shape[1])
            * int(transforms[int(in_comp)].shape[1])
        )
        if total_elements > int(max_elements):
            return None
    block_by_pair = {
        (int(in_comp), int(out_comp)): np.asarray(block, dtype=complex)
        for in_comp, out_comp, block in parent_blocks
    }

    def project_block(in_comp, out_comp, block):
        X_in = transforms[int(in_comp)]
        X_out = transforms[int(out_comp)]
        m_out, m_in = (int(dim) for dim in block.shape)
        r_out = int(X_out.shape[1])
        r_in = int(X_in.shape[1])
        left_first_cost = r_out * m_out * m_in + r_out * m_in * r_in
        right_first_cost = m_out * m_in * r_in + r_out * m_out * r_in
        if right_first_cost < left_first_cost:
            return X_out.conj().T @ (block @ X_in)
        return (X_out.conj().T @ block) @ X_in

    out = []
    visited = set()
    for in_comp, out_comp, parent_block in parent_blocks:
        in_comp = int(in_comp)
        out_comp = int(out_comp)
        key = (in_comp, out_comp)
        if key in visited:
            continue
        reverse_key = (out_comp, in_comp)
        reverse_block = block_by_pair.get(reverse_key)
        use_reverse_pair = (
            in_comp != out_comp
            and reverse_block is not None
            and reverse_key not in visited
            and tuple(int(dim) for dim in reverse_block.shape)
            == (int(parent_block.shape[1]), int(parent_block.shape[0]))
        )
        transformed = project_block(in_comp, out_comp, np.asarray(parent_block, dtype=complex))
        if np.any(np.abs(transformed) > 1.0e-15):
            out.append((int(in_comp), int(out_comp), np.ascontiguousarray(transformed)))
        visited.add(key)
        if use_reverse_pair:
            reverse = transformed.conj().T
            if np.any(np.abs(reverse) > 1.0e-15):
                out.append(
                    (
                        int(out_comp),
                        int(in_comp),
                        np.ascontiguousarray(reverse),
                    )
                )
            visited.add(reverse_key)
    return tuple(out)


@dataclass(frozen=True)
class _BatchPlan:
    entries: tuple
    left_mats: np.ndarray
    right_mats: np.ndarray
    tmp_shape: tuple
    output_shape: tuple


@dataclass(frozen=True)
class _ParentBlockBatch:
    in_comps: np.ndarray
    out_comps: np.ndarray
    blocks: np.ndarray


class SU2LocalAction:
    """Packed local action for SU(2) component bases."""

    def __init__(
        self,
        component_basis,
        family_table=None,
        *,
        parent_blocks=(),
        packed_terms=None,
        backend="auto",
    ):
        actual = resolve_backend(backend)
        self.backend = actual
        self.component_basis = component_basis
        self.family_table = family_table
        self.dim = int(component_basis.orthonormal_dim)
        self.transforms = tuple(
            (
                transform
                if hasattr(transform, "stored_elements")
                else np.asarray(transform, dtype=complex)
            )
            for transform in component_basis.component_transforms
        )
        self.orth_slices = tuple(
            component_basis._orth_slice(idx)
            for idx in range(int(component_basis.n_components))
        )
        self.parent_dims = tuple(
            int(np.asarray(indices).size)
            for indices in component_basis.component_indices
        )
        self.parent_blocks = tuple(
            (
                int(in_comp),
                int(out_comp),
                np.asarray(block, dtype=complex),
            )
            for in_comp, out_comp, block in tuple(parent_blocks or ())
        )
        self.parent_block_batches, self.parent_block_singles = (
            self._compile_parent_blocks(self.parent_blocks)
        )
        self._cpp_parent_block_table = None
        if actual == "cpp" and self.parent_blocks:
            try:
                from pyqed.mps import cpp_davidson

                table_cls = getattr(cpp_davidson, "SU2ParentBlockTable", None)
                if table_cls is not None:
                    self._cpp_parent_block_table = table_cls(
                        self.parent_block_batches,
                        self.parent_block_singles,
                    )
            except Exception:
                self._cpp_parent_block_table = None
        module = (
            _cpp_module()
            if actual == "cpp" and _USE_CYTHON_PARENT_BLOCK_BATCH
            else None
        )
        self._parent_block_batch_helper = (
            None if module is None else getattr(module, "apply_parent_block_batch", None)
        )
        self._parent_block_batches_helper = (
            None
            if module is None
            else getattr(module, "apply_parent_block_batches", None)
        )
        self.batch_plans, self.single_entries = (
            self._compile_entries(family_table.family_blocks)
            if family_table is not None
            else ((), ())
        )
        self._cpp_family_table = (
            self._build_cpp_packed_family_table(packed_terms)
            if packed_terms is not None
            else self._build_cpp_family_table(family_table)
        )
        if self._cpp_family_table is not None:
            self.family_table = None
            self.batch_plans = ()
            self.single_entries = ()

    @classmethod
    def from_family_table(cls, component_basis, family_table, *, backend="auto"):
        """Build a packed local action from a family tensor table."""

        if family_table is None:
            return None
        actual = resolve_backend(backend)
        if actual == "python":
            return None
        return cls(component_basis, family_table, backend=backend)

    @classmethod
    def from_parent_blocks(cls, component_basis, parent_blocks, *, backend="auto"):
        """Build a packed local action from parent component blocks."""

        if not parent_blocks:
            return None
        actual = resolve_backend(backend)
        if actual == "python":
            return None
        return cls(
            component_basis,
            None,
            parent_blocks=tuple(parent_blocks),
            backend=backend,
        )

    @classmethod
    def from_packed_family_terms(
        cls,
        component_basis,
        packed_terms,
        *,
        backend="auto",
    ):
        """Build a native action directly from packed qchem factor routes."""

        if packed_terms is None or resolve_backend(backend) == "python":
            return None
        action = cls(
            component_basis,
            None,
            packed_terms=packed_terms,
            backend=backend,
        )
        return action if action._cpp_family_table is not None else None

    def _compile_parent_blocks(self, parent_blocks):
        groups = {}
        for in_comp, out_comp, block in parent_blocks:
            groups.setdefault(tuple(block.shape), []).append((in_comp, out_comp, block))
        batches = []
        singles = []
        for entries in groups.values():
            if len(entries) < 4:
                singles.extend(entries)
                continue
            batches.append(
                _ParentBlockBatch(
                    in_comps=np.asarray(
                        [entry[0] for entry in entries],
                        dtype=np.int64,
                    ),
                    out_comps=np.asarray(
                        [entry[1] for entry in entries],
                        dtype=np.int64,
                    ),
                    blocks=np.ascontiguousarray(
                        np.stack([entry[2] for entry in entries], axis=0)
                    ),
                )
            )
        return tuple(batches), tuple(singles)

    def _compile_entries(self, family_blocks):
        batches = {}
        singles = []
        for _family, entries in tuple(family_blocks or ()):
            for entry in tuple(entries or ()):
                signature = entry.factor_batch_signature
                if signature is None:
                    singles.append(entry)
                else:
                    batches.setdefault(signature, []).append(entry)
        plans = []
        for entries in batches.values():
            if len(entries) < 4:
                singles.extend(entries)
                continue
            kernels = [entry.factor_kernel for entry in entries]
            first = kernels[0]
            plans.append(
                _BatchPlan(
                    entries=tuple(entries),
                    left_mats=np.ascontiguousarray(
                        np.stack([kernel.left_matrix for kernel in kernels], axis=0)
                    ),
                    right_mats=np.ascontiguousarray(
                        np.stack([kernel.right_matrix for kernel in kernels], axis=0)
                    ),
                    tmp_shape=tuple(first.tmp_shape),
                    output_shape=tuple(first.output_shape),
                )
            )
        return tuple(plans), tuple(singles)

    def _cpp_transform_descriptor(self, index, transform):
        """Pack one compact orthonormal transform for the native table."""

        orth_start = int(self.orth_slices[index].start)
        name = type(transform).__name__
        if name == "DiagonalMetricTransform":
            return (
                "diagonal",
                orth_start,
                int(transform.parent_dim),
                np.ascontiguousarray(transform.rows, dtype=np.int64),
                np.ascontiguousarray(transform.values, dtype=complex),
            )
        if name == "KroneckerMetricTransform":
            return (
                "kronecker",
                orth_start,
                np.ascontiguousarray(transform.left, dtype=complex),
                np.ascontiguousarray(transform.right, dtype=complex),
                int(transform.phys_dims[0]),
                int(transform.phys_dims[1]),
                int(transform.shape[0]),
            )
        dense = np.ascontiguousarray(np.asarray(transform, dtype=complex))
        if dense.ndim != 2:
            raise ValueError("SU2 component transform must be rank two.")
        return ("dense", orth_start, dense)

    def _build_cpp_family_table(self, family_table):
        """Build the persistent C++ factorized-family action when possible."""

        if self.backend != "cpp" or family_table is None:
            return None
        try:
            from pyqed.mps import cpp_davidson

            table_cls = getattr(cpp_davidson, "SU2FactorizedFamilyTable", None)
            if table_cls is None:
                return None
            transforms = tuple(
                self._cpp_transform_descriptor(index, transform)
                for index, transform in enumerate(self.transforms)
            )
            entries = []
            for _family, family_entries in tuple(family_table.family_blocks or ()):
                for entry in tuple(family_entries or ()):
                    kernel = entry.factor_kernel
                    if (
                        kernel is None
                        or not bool(kernel.matmul_two_step)
                        or kernel.left_matrix is None
                        or kernel.right_matrix is None
                    ):
                        return None
                    entries.append(
                        (
                            int(entry.in_comp),
                            int(entry.out_comp),
                            int(entry.in_slice.start),
                            int(entry.in_slice.stop - entry.in_slice.start),
                            int(entry.out_slice.start),
                            int(entry.out_slice.stop - entry.out_slice.start),
                            np.ascontiguousarray(kernel.left_matrix, dtype=complex),
                            np.ascontiguousarray(kernel.right_matrix, dtype=complex),
                            tuple(int(dim) for dim in kernel.tmp_shape),
                            tuple(int(dim) for dim in kernel.output_shape),
                            tuple(int(dim) for dim in kernel.input_shape),
                            int(kernel.output_size),
                        )
                    )
            if not entries:
                return None
            return table_cls(transforms, tuple(entries), self.dim)
        except (ImportError, TypeError, ValueError, RuntimeError):
            return None

    def _build_cpp_packed_family_table(self, packed_terms):
        """Build a persistent table from packed qchem routes without expansion."""

        if self.backend != "cpp" or packed_terms is None:
            return None
        try:
            from pyqed.mps import cpp_davidson

            table_cls = getattr(
                cpp_davidson,
                "SU2PackedFactorizedFamilyTable",
                None,
            )
            if table_cls is None:
                return None
            transforms = tuple(
                self._cpp_transform_descriptor(index, transform)
                for index, transform in enumerate(self.transforms)
            )
            component_indices = tuple(
                np.ascontiguousarray(indices, dtype=np.int64)
                for indices in self.component_basis.component_indices
            )
            basis_offsets, basis_shapes = packed_terms._cpp_factor_route_basis()
            left = packed_terms.plan.left_factor_table
            right = packed_terms.plan.right_factor_table
            left_pool = left.factor_pool
            right_pool = right.factor_pool
            if np.iscomplexobj(left_pool.data) or np.iscomplexobj(right_pool.data):
                return None
            route_arrays = (
                np.ascontiguousarray(packed_terms.in_indices, dtype=np.int64),
                np.ascontiguousarray(packed_terms.out_indices, dtype=np.int64),
                np.ascontiguousarray(packed_terms.left_indices, dtype=np.int64),
                np.ascontiguousarray(packed_terms.right_indices, dtype=np.int64),
                np.ascontiguousarray(basis_offsets, dtype=np.int64),
                np.ascontiguousarray(basis_shapes, dtype=np.int64),
            )
            raw_sources = all(
                getattr(table, name, None) is not None
                for table in (left, right)
                for name in (
                    "factor_boundary_pool",
                    "factor_w_pool",
                    "factor_boundary_array_ids",
                    "factor_w_block_ids",
                )
            )
            if raw_sources:
                def raw_side(table):
                    boundary = table.factor_boundary_pool
                    w_pool = table.factor_w_pool
                    return (
                        np.ascontiguousarray(table.factor_indices, dtype=np.int64),
                        np.ascontiguousarray(
                            table.factor_boundary_array_ids, dtype=np.int64
                        ),
                        np.ascontiguousarray(table.factor_w_block_ids, dtype=np.int64),
                        np.ascontiguousarray(boundary.offsets, dtype=np.int64),
                        np.ascontiguousarray(boundary.shape_offsets, dtype=np.int64),
                        np.ascontiguousarray(boundary.shapes, dtype=np.int64),
                        np.ascontiguousarray(boundary.data, dtype=float),
                        np.ascontiguousarray(w_pool.offsets, dtype=np.int64),
                        np.ascontiguousarray(w_pool.shape_offsets, dtype=np.int64),
                        np.ascontiguousarray(w_pool.shapes, dtype=np.int64),
                        np.ascontiguousarray(w_pool.data, dtype=float),
                    )

                packed_arrays = route_arrays + raw_side(left) + raw_side(right)
            else:
                packed_arrays = route_arrays + (
                np.ascontiguousarray(left.factor_indices, dtype=np.int64),
                np.ascontiguousarray(left_pool.offsets, dtype=np.int64),
                np.ascontiguousarray(left_pool.shape_offsets, dtype=np.int64),
                np.ascontiguousarray(left_pool.shapes, dtype=np.int64),
                np.ascontiguousarray(left_pool.data, dtype=float),
                np.ascontiguousarray(right.factor_indices, dtype=np.int64),
                np.ascontiguousarray(right_pool.offsets, dtype=np.int64),
                np.ascontiguousarray(right_pool.shape_offsets, dtype=np.int64),
                np.ascontiguousarray(right_pool.shapes, dtype=np.int64),
                np.ascontiguousarray(right_pool.data, dtype=float),
                )
            table = table_cls(
                transforms,
                component_indices,
                packed_arrays,
                int(packed_terms.total_dim),
                self.dim,
            )
            if raw_sources:
                left.release_materialized_factors()
                right.release_materialized_factors()
            return table
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def matvec(self, vector):
        """Apply the packed local action."""

        vector = np.asarray(vector, dtype=complex).reshape(self.dim)
        if self._cpp_family_table is not None:
            return np.asarray(
                self._cpp_family_table.matvec(vector),
                dtype=complex,
            ).reshape(self.dim)
        out = np.zeros(self.dim, dtype=complex)
        if self.backend == "cpp" and not self.parent_block_batches:
            module = _cpp_module()
            if module is not None and hasattr(module, "apply_su2_block2_action"):
                module.apply_su2_block2_action(self, vector, out)
                return out
        return self._matvec_python(vector, out)

    def matmat(self, vectors):
        """Apply the packed local action to several column vectors."""

        vectors = np.asarray(vectors, dtype=complex)
        if vectors.ndim != 2 or int(vectors.shape[0]) != self.dim:
            raise ValueError(
                f"Expected a ({self.dim}, nvec) block, got {vectors.shape}."
            )
        nvec = int(vectors.shape[1])
        if nvec == 1:
            return self.matvec(vectors[:, 0]).reshape(self.dim, 1)
        parent_inputs = []
        parent_outputs = []
        for idx, transform in enumerate(self.transforms):
            parent_inputs.append(transform @ vectors[self.orth_slices[idx], :])
            parent_outputs.append(
                np.zeros((self.parent_dims[idx], nvec), dtype=complex)
            )
        for in_comp, out_comp, block in self.parent_blocks:
            parent_outputs[int(out_comp)] += block @ parent_inputs[int(in_comp)]
        for plan in self.batch_plans:
            self._apply_batch_columns(plan, parent_inputs, parent_outputs, nvec)
        for entry in self.single_entries:
            shape = tuple(entry.input_entry.shape) + (nvec,)
            block_inputs = parent_inputs[int(entry.in_comp)][entry.in_slice, :].reshape(
                shape
            )
            parent_outputs[int(entry.out_comp)][entry.out_slice, :] += (
                entry.apply_blocks(block_inputs)
            )
        out = np.zeros((self.dim, nvec), dtype=complex)
        for idx, parent_out in enumerate(parent_outputs):
            out[self.orth_slices[idx], :] = self.transforms[idx].conj().T @ parent_out
        return out

    def _matvec_python(self, vector, out=None):
        """Apply the packed local action with the NumPy fallback kernel."""

        vector = np.asarray(vector, dtype=complex).reshape(self.dim)
        if out is None:
            out = np.zeros(self.dim, dtype=complex)
        else:
            out[...] = 0.0
        parent_inputs, parent_outputs = self._parent_buffers(vector)
        native_parent_blocks = bool(
            self._cpp_parent_block_table is not None
            and self._cpp_parent_block_table.apply(parent_inputs, parent_outputs)
        )
        if not native_parent_blocks:
            if not self._apply_parent_block_batches(parent_inputs, parent_outputs):
                for batch in self.parent_block_batches:
                    self._apply_parent_block_batch(batch, parent_inputs, parent_outputs)
            for in_comp, out_comp, block in self.parent_block_singles:
                parent_outputs[int(out_comp)] += block @ parent_inputs[int(in_comp)]
        for plan in self.batch_plans:
            self._apply_batch(plan, parent_inputs, parent_outputs)
        for entry in self.single_entries:
            block_in = parent_inputs[int(entry.in_comp)][entry.in_slice].reshape(
                entry.input_entry.shape
            )
            parent_outputs[int(entry.out_comp)][entry.out_slice] += entry.apply_block(
                block_in
            )
        return self._from_parent_outputs(parent_outputs, out)

    def _apply_parent_block_batches(self, parent_inputs, parent_outputs):
        helper = self._parent_block_batches_helper
        return bool(
            helper is not None
            and self.parent_block_batches
            and helper(
                self.parent_block_batches,
                parent_inputs,
                parent_outputs,
            )
        )

    def _apply_parent_block_batch(self, batch, parent_inputs, parent_outputs):
        if self._parent_block_batch_helper is not None and self._parent_block_batch_helper(
            batch.blocks,
            batch.in_comps,
            batch.out_comps,
            parent_inputs,
            parent_outputs,
        ):
            return
        inputs = np.stack(
            [parent_inputs[int(idx)] for idx in batch.in_comps],
            axis=0,
        )
        outputs = np.matmul(batch.blocks, inputs[..., None]).reshape(
            len(batch.in_comps),
            batch.blocks.shape[1],
        )
        for out_comp, contrib in zip(batch.out_comps, outputs):
            parent_outputs[int(out_comp)] += contrib

    def _parent_buffers(self, vector):
        parent_inputs = []
        parent_outputs = []
        for idx, transform in enumerate(self.transforms):
            slc = self.orth_slices[idx]
            parent_inputs.append(transform @ vector[slc])
            parent_outputs.append(np.zeros(self.parent_dims[idx], dtype=complex))
        return parent_inputs, parent_outputs

    def _from_parent_outputs(self, parent_outputs, out):
        for idx, parent_out in enumerate(parent_outputs):
            transform = self.transforms[idx]
            out[self.orth_slices[idx]] = transform.conj().T @ parent_out
        return out

    def _apply_batch(self, plan, parent_inputs, parent_outputs):
        entries = plan.entries
        input_mats = np.stack(
            [
                parent_inputs[int(entry.in_comp)][entry.in_slice]
                .reshape(entry.input_entry.shape)
                .reshape(
                    int(np.prod(entry.input_entry.shape[:2], dtype=int)),
                    int(np.prod(entry.input_entry.shape[2:], dtype=int)),
                )
                for entry in entries
            ],
            axis=0,
        )
        tmp = np.matmul(plan.left_mats, input_mats).reshape(
            (len(entries),) + tuple(plan.tmp_shape)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in plan.output_shape)
        tmp_mats = np.ascontiguousarray(
            tmp.transpose(0, 2, 4, 1, 3, 6, 5).reshape(
                len(entries),
                ldim * adim,
                -1,
            )
        )
        contribs = np.matmul(tmp_mats, plan.right_mats).reshape(
            len(entries),
            ldim * adim * ddim * qdim,
        )
        for entry, contrib in zip(entries, contribs):
            parent_outputs[int(entry.out_comp)][entry.out_slice] += contrib

    def _apply_batch_columns(self, plan, parent_inputs, parent_outputs, nvec):
        entries = plan.entries
        input_mats = np.stack(
            [
                parent_inputs[int(entry.in_comp)][entry.in_slice, :]
                .reshape(tuple(entry.input_entry.shape) + (nvec,))
                .reshape(
                    int(np.prod(entry.input_entry.shape[:2], dtype=int)),
                    int(np.prod(entry.input_entry.shape[2:], dtype=int)) * nvec,
                )
                for entry in entries
            ],
            axis=0,
        )
        tmp = np.matmul(plan.left_mats, input_mats).reshape(
            (len(entries),) + tuple(plan.tmp_shape) + (nvec,)
        )
        ldim, adim, ddim, qdim = (int(dim) for dim in plan.output_shape)
        tmp_mats = np.ascontiguousarray(
            tmp.transpose(0, 7, 2, 4, 1, 3, 6, 5).reshape(
                len(entries),
                nvec,
                ldim * adim,
                -1,
            )
        )
        contribs = np.matmul(
            tmp_mats,
            plan.right_mats[:, None, :, :],
        ).transpose(0, 2, 3, 1).reshape(
            len(entries),
            ldim * adim * ddim * qdim,
            nvec,
        )
        for entry, contrib in zip(entries, contribs):
            parent_outputs[int(entry.out_comp)][entry.out_slice, :] += contrib

    @property
    def stats(self):
        """Return local-action diagnostics."""

        return {
            "backend": str(self.backend),
            "n_parent_blocks": int(len(self.parent_blocks)),
            "n_parent_block_batch_groups": int(len(self.parent_block_batches)),
            "cpp_parent_block_table": bool(self._cpp_parent_block_table is not None),
            "cpp_parent_block_table_stats": (
                None
                if self._cpp_parent_block_table is None
                else dict(self._cpp_parent_block_table.stats)
            ),
            "cpp_factorized_family_table": bool(
                self._cpp_family_table is not None
            ),
            "cpp_factorized_family_table_stats": (
                None
                if self._cpp_family_table is None
                else dict(self._cpp_family_table.stats)
            ),
            "n_parent_block_batched_entries": int(
                sum(len(batch.in_comps) for batch in self.parent_block_batches)
            ),
            "n_parent_block_single_entries": int(len(self.parent_block_singles)),
            "parent_block_elements": int(
                sum(block.size for _in_comp, _out_comp, block in self.parent_blocks)
            ),
            "n_batch_groups": int(len(self.batch_plans)),
            "n_batched_entries": int(
                sum(len(plan.entries) for plan in self.batch_plans)
            ),
            "n_single_entries": int(len(self.single_entries)),
            "left_matrix_elements": int(
                sum(plan.left_mats.size for plan in self.batch_plans)
            ),
            "right_matrix_elements": int(
                sum(plan.right_mats.size for plan in self.batch_plans)
            ),
        }
