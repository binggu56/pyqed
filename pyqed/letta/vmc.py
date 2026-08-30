r"""Variational Monte Carlo for unrestricted physically tied LETTA states.

The routines in this module contract only single configurations.  They do not
construct a many-body basis, a dense state vector, or a frontier environment.
For a configuration ``s``, a physically tied LETTA has amplitude

.. math::

    \psi(s) = A_0(s_{P_0}) A_1(s_{P_1}) \cdots A_{N-1}(s_{P_{N-1}}),

where every selected ``A_i`` is a matrix on the virtual bonds.  This makes
Metropolis sampling useful when the exact frontier width is too large.

The public entry point is :class:`pyqed.letta.VMC`; the lower-level
sampling and stochastic-reconfiguration records are also exported there.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix

from pyqed.tn import Hamiltonian


def _configuration(configuration, dims):
    value = np.asarray(configuration, dtype=np.intp)
    if value.shape != (len(dims),):
        raise ValueError(f"configuration must have shape {(len(dims),)}.")
    if any(
        np.any(value[site] < 0) or np.any(value[site] >= dim)
        for site, dim in enumerate(dims)
    ):
        raise ValueError("configuration contains an out-of-range local state.")
    return value


def _combine_products(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return left @ right


class _MatrixProductTree:
    """Ordered matrix-product tree with logarithmic point updates/queries."""

    def __init__(self, matrices):
        self.nleaves = len(matrices)
        size = 1
        while size < self.nleaves:
            size *= 2
        self.size = size
        self.nodes = [None] * (2 * size)
        for index, matrix in enumerate(matrices):
            self.nodes[size + index] = np.asarray(matrix)
        for node in range(size - 1, 0, -1):
            self.nodes[node] = _combine_products(
                self.nodes[2 * node], self.nodes[2 * node + 1]
            )

    def update(self, index, matrix):
        node = self.size + int(index)
        self.nodes[node] = np.asarray(matrix)
        node //= 2
        while node:
            self.nodes[node] = _combine_products(
                self.nodes[2 * node], self.nodes[2 * node + 1]
            )
            node //= 2

    def range_product(self, start, stop):
        """Return the ordered product over ``[start, stop)`` or ``None``."""
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > self.nleaves:
            raise IndexError("invalid matrix-product range.")
        left_product = None
        right_product = None
        left = start + self.size
        right = stop + self.size
        while left < right:
            if left & 1:
                left_product = _combine_products(left_product, self.nodes[left])
                left += 1
            if right & 1:
                right -= 1
                right_product = _combine_products(self.nodes[right], right_product)
            left //= 2
            right //= 2
        return _combine_products(left_product, right_product)

    @property
    def product(self):
        return self.nodes[1]


class LETTAWavefunction:
    """Configuration-amplitude view of FrontierTiedLETTA-like tensors."""

    def __init__(self, tensors, physical_groups, dims=None, *, copy=True):
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        physical_groups = tuple(
            tuple(int(site) for site in sites) for sites in physical_groups
        )
        if not tensors:
            raise ValueError("tensors must contain at least one tensor.")
        if len(tensors) != len(physical_groups):
            raise ValueError("tensors and physical_groups must have equal length.")
        nsites = len(tensors)
        for tensor_site, (tensor, sites) in enumerate(zip(tensors, physical_groups)):
            if tensor.ndim != 2 + len(sites):
                raise ValueError(
                    f"tensor {tensor_site} must have {2 + len(sites)} axes."
                )
            if any(axis == 0 for axis in tensor.shape):
                raise ValueError(f"tensor {tensor_site} cannot have an empty axis.")
            if not np.issubdtype(tensor.dtype, np.number) or np.any(
                ~np.isfinite(tensor)
            ):
                raise ValueError(f"tensor {tensor_site} must be finite and numeric.")
            if len(set(sites)) != len(sites):
                raise ValueError(f"physical_groups[{tensor_site}] contains duplicates.")
            if any(site < 0 or site >= nsites for site in sites):
                raise ValueError(
                    f"physical_groups[{tensor_site}] contains an invalid site."
                )
            if tensor_site not in sites:
                raise ValueError(
                    f"tensor {tensor_site} must depend on its own physical site."
                )
            if tensor_site and tensors[tensor_site - 1].shape[1] != tensor.shape[0]:
                raise ValueError(f"virtual bond mismatch before tensor {tensor_site}.")
        if tensors[0].shape[0] != 1 or tensors[-1].shape[1] != 1:
            raise ValueError("the virtual matrix product must have scalar boundaries.")

        inferred_dims = [None] * nsites
        for tensor_site, (tensor, sites) in enumerate(zip(tensors, physical_groups)):
            for axis, physical_site in enumerate(sites, start=2):
                dim = int(tensor.shape[axis])
                previous = inferred_dims[physical_site]
                if previous is not None and previous != dim:
                    raise ValueError(
                        f"inconsistent dimension for physical site {physical_site}."
                    )
                inferred_dims[physical_site] = dim
        if any(dim is None for dim in inferred_dims):
            raise ValueError("every physical site must occur in physical_groups.")
        inferred_dims = tuple(inferred_dims)
        if dims is None:
            dims = inferred_dims
        else:
            dims = tuple(int(dim) for dim in dims)
            if dims != inferred_dims:
                raise ValueError("dims are inconsistent with tensor physical axes.")

        self.dims = dims
        self.physical_groups = physical_groups
        parameter_dtype = np.dtype(
            np.result_type(*[tensor.dtype for tensor in tensors])
        )
        if not np.issubdtype(parameter_dtype, np.inexact):
            parameter_dtype = np.dtype(np.float64)
        self.tensors = [
            np.asarray(tensor, dtype=parameter_dtype).copy()
            if copy
            else np.asarray(tensor, dtype=parameter_dtype)
            for tensor in tensors
        ]
        self.dtype = parameter_dtype
        dependencies = [[] for _ in range(nsites)]
        for tensor_site, sites in enumerate(physical_groups):
            for physical_site in sites:
                dependencies[physical_site].append(tensor_site)
        self.dependent_tensors = tuple(tuple(sites) for sites in dependencies)
        self._sizes = tuple(tensor.size for tensor in self.tensors)
        self._offsets = np.cumsum((0,) + self._sizes)
        self.version = 0

    @classmethod
    def from_state(cls, state, *, copy=True):
        """Construct from an object exposing tensors, physical_groups, and dims."""
        if getattr(state, "autoregressive", False):
            raise TypeError(
                "autoregressive FutureLETTA uses normalized amplitudes; call "
                "state.sample(...) for independent samples. Its SR derivative "
                "adapter is not implemented yet."
            )
        if all(
            hasattr(state, name)
            for name in ("factors", "factor_masks", "physical_groups", "dims")
        ):
            return ConditionalLETTAWavefunction.from_state(state, copy=copy)
        missing = [
            name
            for name in ("tensors", "physical_groups", "dims")
            if not hasattr(state, name)
        ]
        if missing:
            raise TypeError(
                "state must expose tensors, physical_groups, and dims; missing "
                + ", ".join(missing)
            )
        return cls(
            state.tensors,
            state.physical_groups,
            state.dims,
            copy=copy,
        )

    @property
    def nsites(self):
        return len(self.dims)

    @property
    def nparameters(self):
        return int(self._offsets[-1])

    def selected_matrix(self, tensor_site, configuration):
        configuration = np.asarray(configuration)
        physical_groups = self.physical_groups[tensor_site]
        index = (slice(None), slice(None)) + tuple(
            int(configuration[site]) for site in physical_groups
        )
        return self.tensors[tensor_site][index]

    def selected_matrices(self, configuration):
        configuration = _configuration(configuration, self.dims)
        return tuple(
            self.selected_matrix(site, configuration) for site in range(self.nsites)
        )

    def amplitude(self, configuration):
        matrices = self.selected_matrices(configuration)
        product = matrices[0]
        for matrix in matrices[1:]:
            product = product @ matrix
        return np.asarray(product).reshape(()).item()

    def amplitudes(self, configurations):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        return np.asarray(
            [self.amplitude(configuration) for configuration in configurations],
            dtype=self.dtype,
        )

    def product_cache(self, configuration):
        return LETTAProductCache(self, configuration)

    def parameter_vector(self, *, copy=True):
        vector = np.concatenate([tensor.reshape(-1) for tensor in self.tensors])
        return vector.copy() if copy else vector

    def tensors_from_parameters(self, parameters):
        parameters = np.asarray(parameters)
        if parameters.shape != (self.nparameters,):
            raise ValueError(f"parameters must have shape {(self.nparameters,)}.")
        dtype = np.result_type(parameters.dtype, self.dtype)
        return tuple(
            np.asarray(parameters[start:stop], dtype=dtype).reshape(tensor.shape).copy()
            for tensor, start, stop in zip(
                self.tensors, self._offsets[:-1], self._offsets[1:]
            )
        )

    def parameters_from_tensors(self, tensors):
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        if len(tensors) != len(self.tensors) or any(
            tensor.shape != reference.shape
            for tensor, reference in zip(tensors, self.tensors)
        ):
            raise ValueError("proposal tensor shapes differ from the wavefunction.")
        return np.concatenate([tensor.reshape(-1) for tensor in tensors])

    def set_parameter_vector(self, parameters):
        self.tensors = list(self.tensors_from_parameters(parameters))
        self.dtype = np.dtype(
            np.result_type(*[tensor.dtype for tensor in self.tensors])
        )
        self.version += 1

    def _log_derivative_terms(self, configuration):
        """Yield active parameter indices and holomorphic derivatives by tensor."""
        configuration = _configuration(configuration, self.dims)
        matrices = self.selected_matrices(configuration)
        left = [None] * (self.nsites + 1)
        right = [None] * (self.nsites + 1)
        left[0] = np.ones(1, dtype=self.dtype)
        for site, matrix in enumerate(matrices):
            left[site + 1] = left[site] @ matrix
        right[-1] = np.ones(1, dtype=self.dtype)
        for site in range(self.nsites - 1, -1, -1):
            right[site] = matrices[site] @ right[site + 1]
        amplitude = left[-1][0]
        if not np.isfinite(amplitude) or abs(amplitude) <= np.finfo(float).tiny:
            raise ValueError("log derivatives are undefined at zero amplitude.")

        for site, tensor in enumerate(self.tensors):
            left_dim, right_dim = tensor.shape[:2]
            local_dims = tuple(self.dims[index] for index in self.physical_groups[site])
            physical_index = np.ravel_multi_index(
                tuple(configuration[index] for index in self.physical_groups[site]),
                local_dims,
            )
            local_size = int(np.prod(local_dims))
            derivative = np.outer(left[site], right[site + 1]) / amplitude
            flat_indices = (
                self._offsets[site]
                + np.arange(left_dim * right_dim, dtype=np.intp) * local_size
                + physical_index
            )
            yield flat_indices, derivative.reshape(-1)

    def log_derivative(self, configuration):
        r"""Return the holomorphic derivatives ``partial log(psi)/partial theta``."""
        result = np.zeros(self.nparameters, dtype=self.dtype)
        for indices, values in self._log_derivative_terms(configuration):
            result[indices] = values
        return result

    def log_derivatives(self, configurations):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        return np.asarray(
            [self.log_derivative(configuration) for configuration in configurations]
        )

    def sparse_log_derivatives(self, configurations, *, batch_size=256):
        """Return the sampled log-derivative operator in CSR form."""
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        active_per_sample = int(
            sum(tensor.shape[0] * tensor.shape[1] for tensor in self.tensors)
        )
        nsamples = len(configurations)
        indices = np.empty(nsamples * active_per_sample, dtype=np.intp)
        data = np.empty(nsamples * active_per_sample, dtype=self.dtype)
        cursor = 0
        for start in range(0, nsamples, batch_size):
            for configuration in configurations[start : start + batch_size]:
                for local_indices, local_values in self._log_derivative_terms(
                    configuration
                ):
                    stop = cursor + len(local_indices)
                    indices[cursor:stop] = local_indices
                    data[cursor:stop] = local_values
                    cursor = stop
        if cursor != len(data):
            raise RuntimeError("log-derivative sparsity changed between samples.")
        indptr = np.arange(
            0,
            (nsamples + 1) * active_per_sample,
            active_per_sample,
            dtype=np.intp,
        )
        return csr_matrix(
            (data, indices, indptr),
            shape=(nsamples, self.nparameters),
        )


class ConditionalLETTAWavefunction:
    """Factor-native amplitude and derivative view of conditional LETTA."""

    def __init__(
        self,
        factors,
        factor_masks,
        physical_groups,
        dims,
        *,
        copy=True,
    ):
        dims = tuple(int(dim) for dim in dims)
        groups = tuple(tuple(int(site) for site in group) for group in physical_groups)
        if not dims or len(groups) != len(dims) or len(factors) != len(dims):
            raise ValueError(
                "factors, physical_groups, and dims must contain one entry per site."
            )
        if len(factor_masks) != len(factors):
            raise ValueError("factor_masks must match factors.")

        source_factors = tuple(tuple(site_factors) for site_factors in factors)
        source_masks = tuple(tuple(site_masks) for site_masks in factor_masks)
        flat_source = [factor for site_factors in source_factors for factor in site_factors]
        parameter_dtype = np.dtype(
            np.result_type(*[np.asarray(factor).dtype for factor in flat_source])
        )
        if not np.issubdtype(parameter_dtype, np.inexact):
            parameter_dtype = np.dtype(np.float64)
        self.dims = dims
        self.physical_groups = groups
        self.dtype = parameter_dtype
        self.factors = []
        self.factor_masks = []
        self._site_factor_slices = []
        self.tensors = []
        cursor = 0
        for site, (group, site_factors, site_masks) in enumerate(
            zip(groups, source_factors, source_masks)
        ):
            if not group or group[0] != site or len(group) != len(site_factors):
                raise ValueError(
                    f"site {site} must have one B/C factor per ordered physical leg."
                )
            if len(site_masks) != len(site_factors):
                raise ValueError(f"factor masks do not match site {site}.")
            copied_factors = []
            copied_masks = []
            for factor, mask in zip(site_factors, site_masks):
                factor = np.asarray(factor, dtype=parameter_dtype)
                mask = np.asarray(mask, dtype=bool)
                if factor.shape != mask.shape:
                    raise ValueError("each factor mask must match its factor shape.")
                value = factor.copy() if copy else factor
                value[~mask] = 0
                copied_factors.append(value)
                copied_masks.append(mask.copy())
                self.tensors.append(value)
            self.factors.append(copied_factors)
            self.factor_masks.append(copied_masks)
            self._site_factor_slices.append(
                slice(cursor, cursor + len(copied_factors))
            )
            cursor += len(copied_factors)

        dependencies = [[] for _ in dims]
        for tensor_site, sites in enumerate(groups):
            for physical_site in sites:
                dependencies[physical_site].append(tensor_site)
        self.dependent_tensors = tuple(tuple(sites) for sites in dependencies)
        self._active_flat = tuple(
            np.flatnonzero(mask.reshape(-1))
            for site_masks in self.factor_masks
            for mask in site_masks
        )
        self._parameter_lookups = []
        for factor, active in zip(self.tensors, self._active_flat):
            lookup = np.full(factor.size, -1, dtype=np.intp)
            lookup[active] = np.arange(active.size, dtype=np.intp)
            self._parameter_lookups.append(lookup)
        self._sizes = tuple(active.size for active in self._active_flat)
        self._offsets = np.cumsum((0,) + self._sizes)
        self.version = 0

    @classmethod
    def from_state(cls, state, *, copy=True):
        return cls(
            state.factors,
            state.factor_masks,
            state.physical_groups,
            state.dims,
            copy=copy,
        )

    @property
    def nsites(self):
        return len(self.dims)

    @property
    def nparameters(self):
        return int(self._offsets[-1])

    def _selected_factor_matrices(self, configuration):
        configuration = _configuration(configuration, self.dims)
        matrices = []
        flat_positions = []
        for site, (group, site_factors) in enumerate(
            zip(self.physical_groups, self.factors)
        ):
            physical = int(configuration[site])
            first = site_factors[0]
            positions = np.arange(first.size, dtype=np.intp).reshape(first.shape)
            matrices.append(first[:, physical, :])
            flat_positions.append(positions[:, physical, :].reshape(-1))
            for parent, factor in zip(group[1:], site_factors[1:]):
                parent_state = int(configuration[parent])
                positions = np.arange(factor.size, dtype=np.intp).reshape(factor.shape)
                matrices.append(factor[:, physical, parent_state, :])
                flat_positions.append(
                    positions[:, physical, parent_state, :].reshape(-1)
                )
        return tuple(matrices), tuple(flat_positions)

    def selected_matrix(self, tensor_site, configuration):
        tensor_site = int(tensor_site)
        if tensor_site < 0 or tensor_site >= self.nsites:
            raise IndexError("tensor_site is out of range.")
        configuration = _configuration(configuration, self.dims)
        group = self.physical_groups[tensor_site]
        factors = self.factors[tensor_site]
        physical = int(configuration[tensor_site])
        value = factors[0][:, physical, :]
        for parent, factor in zip(group[1:], factors[1:]):
            value = value @ factor[
                :, physical, int(configuration[parent]), :
            ]
        return value

    def selected_matrices(self, configuration):
        configuration = _configuration(configuration, self.dims)
        return tuple(
            self.selected_matrix(site, configuration) for site in range(self.nsites)
        )

    def amplitude(self, configuration):
        matrices = self.selected_matrices(configuration)
        value = matrices[0]
        for matrix in matrices[1:]:
            value = value @ matrix
        return np.asarray(value).reshape(()).item()

    def amplitudes(self, configurations):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        return np.asarray(
            [self.amplitude(configuration) for configuration in configurations],
            dtype=self.dtype,
        )

    def product_cache(self, configuration):
        return LETTAProductCache(self, configuration)

    def parameter_vector(self, *, copy=True):
        vector = np.concatenate(
            [
                factor.reshape(-1)[active]
                for factor, active in zip(self.tensors, self._active_flat)
            ]
        )
        return vector.copy() if copy else vector

    def tensors_from_parameters(self, parameters):
        parameters = np.asarray(parameters)
        if parameters.shape != (self.nparameters,):
            raise ValueError(f"parameters must have shape {(self.nparameters,)}.")
        dtype = np.result_type(parameters.dtype, self.dtype)
        result = []
        for factor, active, start, stop in zip(
            self.tensors,
            self._active_flat,
            self._offsets[:-1],
            self._offsets[1:],
        ):
            value = np.zeros(factor.shape, dtype=dtype)
            value.reshape(-1)[active] = parameters[start:stop]
            result.append(value)
        return tuple(result)

    def parameters_from_tensors(self, tensors):
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        if len(tensors) != len(self.tensors) or any(
            tensor.shape != reference.shape
            for tensor, reference in zip(tensors, self.tensors)
        ):
            raise ValueError("proposal factor shapes differ from the wavefunction.")
        return np.concatenate(
            [
                tensor.reshape(-1)[active]
                for tensor, active in zip(tensors, self._active_flat)
            ]
        )

    def set_parameter_vector(self, parameters):
        self.tensors = list(self.tensors_from_parameters(parameters))
        self.factors = [
            self.tensors[site_slice]
            for site_slice in self._site_factor_slices
        ]
        self.dtype = np.dtype(
            np.result_type(*[factor.dtype for factor in self.tensors])
        )
        self.version += 1

    def _log_derivative_terms(self, configuration):
        matrices, flat_positions = self._selected_factor_matrices(configuration)
        nfactors = len(matrices)
        left = [None] * (nfactors + 1)
        right = [None] * (nfactors + 1)
        left[0] = np.ones(1, dtype=self.dtype)
        for index, matrix in enumerate(matrices):
            left[index + 1] = left[index] @ matrix
        right[-1] = np.ones(1, dtype=self.dtype)
        for index in range(nfactors - 1, -1, -1):
            right[index] = matrices[index] @ right[index + 1]
        amplitude = left[-1][0]
        if not np.isfinite(amplitude) or abs(amplitude) <= np.finfo(float).tiny:
            raise ValueError("log derivatives are undefined at zero amplitude.")

        for index, positions in enumerate(flat_positions):
            local_indices = self._parameter_lookups[index][positions]
            supported = local_indices >= 0
            if not np.any(supported):
                continue
            derivative = np.outer(left[index], right[index + 1]).reshape(-1)
            yield (
                self._offsets[index] + local_indices[supported],
                derivative[supported] / amplitude,
            )

    def log_derivative(self, configuration):
        result = np.zeros(self.nparameters, dtype=self.dtype)
        for indices, values in self._log_derivative_terms(configuration):
            result[indices] = values
        return result

    def log_derivatives(self, configurations):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        return np.asarray(
            [self.log_derivative(configuration) for configuration in configurations]
        )

    def sparse_log_derivatives(self, configurations, *, batch_size=256):
        configurations = np.asarray(configurations, dtype=np.intp)
        if configurations.ndim != 2 or configurations.shape[1] != self.nsites:
            raise ValueError(
                f"configurations must have shape (nsamples, {self.nsites})."
            )
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        indices = []
        data = []
        indptr = [0]
        for start in range(0, len(configurations), batch_size):
            for configuration in configurations[start : start + batch_size]:
                for local_indices, local_values in self._log_derivative_terms(
                    configuration
                ):
                    indices.extend(local_indices)
                    data.extend(local_values)
                indptr.append(len(indices))
        return csr_matrix(
            (
                np.asarray(data, dtype=self.dtype),
                np.asarray(indices, dtype=np.intp),
                np.asarray(indptr, dtype=np.intp),
            ),
            shape=(len(configurations), self.nparameters),
        )

    def sync_to_state(self, state, *, copy=True):
        if tuple(state.dims) != self.dims or tuple(
            tuple(group) for group in state.physical_groups
        ) != self.physical_groups:
            raise ValueError("state physical layout differs from the VMC wavefunction.")
        target = [factor for site in state.factors for factor in site]
        if len(target) != len(self.tensors) or any(
            np.shape(left) != right.shape
            for left, right in zip(target, self.tensors)
        ):
            raise ValueError("state factor shapes differ from the VMC wavefunction.")
        state.factors = [
            [np.array(factor, copy=bool(copy)) for factor in site_factors]
            for site_factors in self.factors
        ]
        if hasattr(state, "energy"):
            state.energy = None
        if hasattr(state, "converged"):
            state.converged = False
        if hasattr(state, "history"):
            state.history = []
        return state


class LETTAProductCache:
    """Cached amplitude and range products for one configuration."""

    def __init__(self, wavefunction: LETTAWavefunction, configuration):
        self.wavefunction = wavefunction
        self.configuration = _configuration(configuration, wavefunction.dims).copy()
        self.version = wavefunction.version
        self.matrices = list(wavefunction.selected_matrices(self.configuration))
        self._tree = _MatrixProductTree(self.matrices)

    def _check_version(self):
        if self.version != self.wavefunction.version:
            raise RuntimeError(
                "the wavefunction parameters changed; refresh the product cache."
            )

    @property
    def amplitude(self):
        self._check_version()
        return np.asarray(self._tree.product).reshape(()).item()

    def _replacement_product(self, replacements):
        self._check_version()
        if not replacements:
            return self.amplitude
        product = None
        cursor = 0
        for tensor_site in sorted(replacements):
            product = _combine_products(
                product, self._tree.range_product(cursor, tensor_site)
            )
            product = _combine_products(product, replacements[tensor_site])
            cursor = tensor_site + 1
        product = _combine_products(
            product, self._tree.range_product(cursor, self.wavefunction.nsites)
        )
        return np.asarray(product).reshape(()).item()

    def amplitude_for(self, configuration):
        configuration = _configuration(configuration, self.wavefunction.dims)
        changed_sites = np.flatnonzero(configuration != self.configuration)
        affected = {
            tensor_site
            for physical_site in changed_sites
            for tensor_site in self.wavefunction.dependent_tensors[physical_site]
        }
        replacements = {
            tensor_site: self.wavefunction.selected_matrix(tensor_site, configuration)
            for tensor_site in affected
        }
        return self._replacement_product(replacements)

    def amplitude_after_local_update(self, physical_site, value):
        physical_site = int(physical_site)
        value = int(value)
        if physical_site < 0 or physical_site >= self.wavefunction.nsites:
            raise IndexError("physical_site is out of range.")
        if value < 0 or value >= self.wavefunction.dims[physical_site]:
            raise ValueError("local state is out of range.")
        configuration = self.configuration.copy()
        configuration[physical_site] = value
        return self.amplitude_for(configuration)

    def accept_local_update(self, physical_site, value):
        physical_site = int(physical_site)
        value = int(value)
        configuration = self.configuration.copy()
        configuration[physical_site] = value
        return self.accept_configuration(configuration)

    def accept_configuration(self, configuration):
        """Atomically accept a configuration and update every affected tensor."""
        self._check_version()
        configuration = _configuration(configuration, self.wavefunction.dims).copy()
        changed_sites = np.flatnonzero(configuration != self.configuration)
        affected = {
            tensor_site
            for physical_site in changed_sites
            for tensor_site in self.wavefunction.dependent_tensors[physical_site]
        }
        for tensor_site in sorted(affected):
            matrix = self.wavefunction.selected_matrix(tensor_site, configuration)
            self.matrices[tensor_site] = matrix
            self._tree.update(tensor_site, matrix)
        self.configuration = configuration
        return self.amplitude


class ConfigurationActionOperator(Protocol):
    """Protocol for Hamiltonians acting directly on configurations."""

    dims: tuple[int, ...]

    def configuration_actions(self, configuration):
        """Yield ``(matrix_element, ket_configuration)`` for one bra state."""


class LocalHamiltonianActions:
    """Configuration-action adapter for :class:`Hamiltonian`."""

    def __init__(self, hamiltonian: Hamiltonian, *, matrix_element_tolerance=0.0):
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a Hamiltonian.")
        tolerance = float(matrix_element_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("matrix_element_tolerance must be nonnegative.")
        self.hamiltonian = hamiltonian
        self.dims = hamiltonian.dims
        self.dtype = hamiltonian.dtype
        self.matrix_element_tolerance = tolerance
        self.hamiltonian._validate_products_hermitian()

    def _nonzero_row(self, operator, state):
        row = np.asarray(operator)[int(state)]
        if self.matrix_element_tolerance == 0.0:
            columns = np.flatnonzero(row != 0.0)
        else:
            columns = np.flatnonzero(
                np.abs(row) > self.matrix_element_tolerance
            )
        return tuple((int(column), row[column]) for column in columns)

    def configuration_actions(self, configuration):
        configuration = _configuration(configuration, self.dims)
        if self.hamiltonian.constant != 0.0:
            yield self.hamiltonian.constant, configuration
        for term in self.hamiltonian.local_terms:
            support_dims = tuple(self.dims[site] for site in term.sites)
            row_index = np.ravel_multi_index(
                tuple(configuration[site] for site in term.sites), support_dims
            )
            row = term.operator[row_index]
            if self.matrix_element_tolerance == 0.0:
                columns = np.flatnonzero(row != 0.0)
            else:
                columns = np.flatnonzero(np.abs(row) > self.matrix_element_tolerance)
            for column in columns:
                value = row[column]
                target = configuration.copy()
                target_states = np.unravel_index(int(column), support_dims)
                for site, state in zip(term.sites, target_states):
                    target[site] = state
                yield value, target
        for operator_string in self.hamiltonian.products:
            choices = tuple(
                self._nonzero_row(operator, configuration[site])
                for site, operator in zip(
                    operator_string.sites,
                    operator_string.operators,
                )
            )
            if any(not site_choices for site_choices in choices):
                continue
            for selected in product(*choices):
                target = configuration.copy()
                value = operator_string.coefficient
                for site, (state, matrix_element) in zip(
                    operator_string.sites,
                    selected,
                ):
                    target[site] = state
                    value = value * matrix_element
                if (
                    self.matrix_element_tolerance == 0.0
                    or abs(value) > self.matrix_element_tolerance
                ):
                    yield value, target


@dataclass(frozen=True)
class MetropolisDiagnostics:
    attempts: int
    accepted: int
    acceptance_rate: float
    zero_amplitude_rejections: int
    initialization_attempts: int
    site_attempts: tuple[int, ...]
    site_accepts: tuple[int, ...]
    single_site_attempts: int = 0
    single_site_accepts: int = 0
    exchange_attempts: int = 0
    exchange_accepts: int = 0

    @property
    def single_site_acceptance_rate(self):
        return (
            float(self.single_site_accepts / self.single_site_attempts)
            if self.single_site_attempts
            else 0.0
        )

    @property
    def exchange_acceptance_rate(self):
        return (
            float(self.exchange_accepts / self.exchange_attempts)
            if self.exchange_attempts
            else 0.0
        )


@dataclass(frozen=True)
class VMCSamples:
    configurations: np.ndarray
    amplitudes: np.ndarray
    local_energies: np.ndarray
    log_derivatives: np.ndarray | None
    diagnostics: MetropolisDiagnostics

    @property
    def nsamples(self):
        return int(len(self.configurations))


@dataclass(frozen=True)
class EnergyEstimate:
    """Energy statistics with naive and autocorrelation-aware errors."""

    energy: complex
    variance: float
    real_variance: float
    standard_error: float
    nsamples: int
    diagnostics: MetropolisDiagnostics
    integrated_autocorrelation_time: float = 0.5
    effective_sample_size: float = 0.0
    autocorrelation_standard_error: float = 0.0


def _autocorrelation_statistics(values):
    """Estimate real-part autocorrelation with Geyer's positive-pair window."""

    values = np.asarray(values).real.reshape(-1)
    nsamples = int(values.size)
    if nsamples < 2:
        return 0.5, float(nsamples), 0.0
    centered = values - np.mean(values)
    variance = float(np.mean(centered**2))
    if not np.isfinite(variance) or variance <= np.finfo(float).tiny:
        return 0.5, float(nsamples), 0.0
    transform_size = 1 << (2 * nsamples - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_size)
    autocovariance = np.fft.irfft(spectrum.conj() * spectrum, n=transform_size)[
        :nsamples
    ]
    autocovariance /= np.arange(nsamples, 0, -1)
    correlation = autocovariance / autocovariance[0]
    positive_pairs = []
    lag = 0
    while lag + 1 < nsamples:
        pair = correlation[lag] + correlation[lag + 1]
        if not np.isfinite(pair) or pair <= 0.0:
            break
        positive_pairs.append(float(pair))
        lag += 2
    tau = sum(positive_pairs) - 0.5
    tau = max(0.5, float(tau))
    effective = min(float(nsamples), float(nsamples / (2.0 * tau)))
    standard_error = float(np.sqrt(variance / max(effective, 1.0)))
    return tau, effective, standard_error


@dataclass(frozen=True)
class SRDirection:
    """Solution of the sampled, regularized SR system without forming its metric."""

    direction: np.ndarray
    force: np.ndarray
    metric_diagonal: np.ndarray
    diagonal_shift: float
    diagonal_floor: float
    iterations: int
    converged: bool
    residual_norm: float
    force_norm: float
    derivative_backend: str = "dense"
    stored_derivative_elements: int = 0


@dataclass(frozen=True)
class SRProposal:
    tensors: tuple[np.ndarray, ...]
    delta: np.ndarray
    direction: SRDirection
    step_size: float
    applied_scale: float
    base_version: int


class MetropolisSampler:
    """Metropolis sampler with tie-aware single-site and exchange proposals."""

    def __init__(
        self,
        wavefunction: LETTAWavefunction,
        *,
        seed=None,
        initial_configuration=None,
        max_initialization_attempts=10_000,
        proposal="single_site",
        exchange_probability=0.5,
        exchange_pairs=None,
        site_charges=None,
    ):
        self.wavefunction = wavefunction
        self.rng = np.random.default_rng(seed)
        proposal = str(proposal)
        if proposal not in {
            "single_site",
            "exchange",
            "charge_pair",
            "heat_bath",
            "mixed",
        }:
            raise ValueError(
                "proposal must be 'single_site', 'exchange', 'charge_pair', "
                "'heat_bath', or 'mixed'."
            )
        exchange_probability = float(exchange_probability)
        if not np.isfinite(exchange_probability) or not 0.0 <= exchange_probability <= 1.0:
            raise ValueError("exchange_probability must lie in [0, 1].")
        self.proposal = proposal
        self.exchange_probability = exchange_probability
        if site_charges is None:
            self.site_charges = None
        else:
            normalized_charges = tuple(
                tuple(tuple(int(value) for value in charge) for charge in site)
                for site in site_charges
            )
            if len(normalized_charges) != wavefunction.nsites or any(
                len(site) != dim
                for site, dim in zip(normalized_charges, wavefunction.dims)
            ):
                raise ValueError(
                    "site_charges must provide one charge per local basis state."
                )
            ranks = {
                len(charge)
                for site in normalized_charges
                for charge in site
            }
            if len(ranks) != 1:
                raise ValueError("site_charges must have a common charge rank.")
            self.site_charges = normalized_charges
        if proposal == "charge_pair" and self.site_charges is None:
            raise ValueError("charge_pair proposals require site_charges.")
        if exchange_pairs is None:
            self.exchange_pairs = tuple(combinations(range(wavefunction.nsites), 2))
        else:
            normalized_pairs = set()
            for pair in exchange_pairs:
                if len(pair) != 2:
                    raise ValueError("each exchange pair must contain two sites.")
                first, second = sorted(int(site) for site in pair)
                if first < 0 or second >= wavefunction.nsites or first == second:
                    raise ValueError("exchange pairs must contain distinct valid sites.")
                normalized_pairs.add((first, second))
            if not normalized_pairs:
                raise ValueError("exchange_pairs cannot be empty.")
            self.exchange_pairs = tuple(sorted(normalized_pairs))
        self.max_initialization_attempts = int(max_initialization_attempts)
        if self.max_initialization_attempts < 1:
            raise ValueError("max_initialization_attempts must be positive.")
        self.attempts = 0
        self.accepted = 0
        self.zero_amplitude_rejections = 0
        self.initialization_attempts = 0
        self.site_attempts = np.zeros(wavefunction.nsites, dtype=np.int64)
        self.site_accepts = np.zeros(wavefunction.nsites, dtype=np.int64)
        self.single_site_attempts = 0
        self.single_site_accepts = 0
        self.exchange_attempts = 0
        self.exchange_accepts = 0
        self.cache = self._initialize(initial_configuration)

    def _random_configuration(self):
        return np.asarray(
            [self.rng.integers(dim) for dim in self.wavefunction.dims],
            dtype=np.intp,
        )

    def _initialize(self, initial_configuration=None):
        for attempt in range(self.max_initialization_attempts):
            configuration = (
                _configuration(initial_configuration, self.wavefunction.dims)
                if attempt == 0 and initial_configuration is not None
                else self._random_configuration()
            )
            cache = self.wavefunction.product_cache(configuration)
            self.initialization_attempts += 1
            amplitude = cache.amplitude
            if np.isfinite(amplitude) and abs(amplitude) > np.finfo(float).tiny:
                return cache
        raise ValueError(
            "could not find a finite nonzero-amplitude configuration by random search."
        )

    def refresh(self):
        """Refresh cached products after a parameter update."""
        configuration = self.cache.configuration.copy()
        cache = self.wavefunction.product_cache(configuration)
        if np.isfinite(cache.amplitude) and abs(cache.amplitude) > np.finfo(float).tiny:
            self.cache = cache
        else:
            self.cache = self._initialize()
        return self

    def _metropolis_accept(self, configuration, sites, proposal):
        old_amplitude = self.cache.amplitude
        new_amplitude = self.cache.amplitude_for(configuration)
        if not np.isfinite(new_amplitude) or abs(new_amplitude) <= np.finfo(float).tiny:
            self.zero_amplitude_rejections += 1
            return False
        log_ratio = 2.0 * (np.log(abs(new_amplitude)) - np.log(abs(old_amplitude)))
        accepted = log_ratio >= 0.0 or np.log(self.rng.random()) < log_ratio
        if accepted:
            self.cache.accept_configuration(configuration)
            self.accepted += 1
            self.site_accepts[list(sites)] += 1
            if proposal == "single_site":
                self.single_site_accepts += 1
            else:
                self.exchange_accepts += 1
        return bool(accepted)

    def _record_acceptance(self, configuration, sites, proposal):
        self.cache.accept_configuration(configuration)
        self.accepted += 1
        self.site_accepts[list(sites)] += 1
        if proposal == "single_site":
            self.single_site_accepts += 1
        else:
            self.exchange_accepts += 1
        return True

    def _single_site_step(self, site=None):
        if self.cache.version != self.wavefunction.version:
            self.refresh()
        if site is None:
            site = int(self.rng.integers(self.wavefunction.nsites))
        else:
            site = int(site)
        if site < 0 or site >= self.wavefunction.nsites:
            raise IndexError("site is out of range.")
        self.attempts += 1
        self.single_site_attempts += 1
        self.site_attempts[site] += 1
        dim = self.wavefunction.dims[site]
        if dim == 1:
            return False
        old_value = int(self.cache.configuration[site])
        proposal = int(self.rng.integers(dim - 1))
        if proposal >= old_value:
            proposal += 1
        configuration = self.cache.configuration.copy()
        configuration[site] = proposal
        return self._metropolis_accept(configuration, (site,), "single_site")

    def _exchange_step(self, pair=None):
        if self.cache.version != self.wavefunction.version:
            self.refresh()
        nsites = self.wavefunction.nsites
        if pair is None:
            if nsites < 2:
                pair = (0, 0)
            else:
                pair = tuple(int(site) for site in self.rng.choice(nsites, 2, replace=False))
        else:
            if len(pair) != 2:
                raise ValueError("pair must contain exactly two sites.")
            pair = tuple(int(site) for site in pair)
        first, second = pair
        if first < 0 or first >= nsites or second < 0 or second >= nsites:
            raise IndexError("exchange site is out of range.")
        if first == second and nsites >= 2:
            raise ValueError("exchange sites must be distinct.")
        self.attempts += 1
        self.exchange_attempts += 1
        self.site_attempts[list(dict.fromkeys(pair))] += 1
        if first == second:
            return False
        first_value = int(self.cache.configuration[first])
        second_value = int(self.cache.configuration[second])
        if first_value == second_value:
            return False
        if (
            second_value >= self.wavefunction.dims[first]
            or first_value >= self.wavefunction.dims[second]
        ):
            return False
        configuration = self.cache.configuration.copy()
        configuration[first], configuration[second] = second_value, first_value
        return self._metropolis_accept(configuration, pair, "exchange")

    def _charge_pair_step(self, pair=None):
        """Propose a two-site basis change with exactly conserved charges."""
        if self.cache.version != self.wavefunction.version:
            self.refresh()
        if self.site_charges is None:
            raise ValueError("charge_pair proposals require site_charges.")
        if pair is None:
            pair = self.exchange_pairs[
                int(self.rng.integers(len(self.exchange_pairs)))
            ]
        elif len(pair) != 2:
            raise ValueError("pair must contain exactly two sites.")
        first, second = (int(site) for site in pair)
        nsites = self.wavefunction.nsites
        if first < 0 or first >= nsites or second < 0 or second >= nsites:
            raise IndexError("charge-pair site is out of range.")
        if first == second:
            raise ValueError("charge-pair sites must be distinct.")

        self.attempts += 1
        self.exchange_attempts += 1
        self.site_attempts[[first, second]] += 1
        current_first = int(self.cache.configuration[first])
        current_second = int(self.cache.configuration[second])
        total = tuple(
            left + right
            for left, right in zip(
                self.site_charges[first][current_first],
                self.site_charges[second][current_second],
            )
        )
        candidates = []
        for first_state, first_charge in enumerate(self.site_charges[first]):
            for second_state, second_charge in enumerate(self.site_charges[second]):
                if (first_state, second_state) == (current_first, current_second):
                    continue
                if tuple(
                    left + right
                    for left, right in zip(first_charge, second_charge)
                ) == total:
                    candidates.append((first_state, second_state))
        if not candidates:
            return False
        proposed_first, proposed_second = candidates[
            int(self.rng.integers(len(candidates)))
        ]
        configuration = self.cache.configuration.copy()
        configuration[first] = proposed_first
        configuration[second] = proposed_second
        return self._metropolis_accept(configuration, (first, second), "charge_pair")

    def _eligible_exchange_pairs(self, configuration):
        pairs = []
        for first, second in self.exchange_pairs:
            first_value = int(configuration[first])
            second_value = int(configuration[second])
            if first_value == second_value:
                continue
            if (
                second_value < self.wavefunction.dims[first]
                and first_value < self.wavefunction.dims[second]
            ):
                pairs.append((first, second))
        return tuple(pairs)

    def _heat_bath_step(self, pair=None):
        """Sample a two-state exchange conditional with a Hastings correction."""
        if self.cache.version != self.wavefunction.version:
            self.refresh()
        configuration = self.cache.configuration
        explicit_pair = pair is not None
        eligible = self._eligible_exchange_pairs(configuration)
        if pair is None:
            pair = (
                eligible[int(self.rng.integers(len(eligible)))]
                if eligible
                else None
            )
        else:
            if len(pair) != 2:
                raise ValueError("pair must contain exactly two sites.")
            pair = tuple(int(site) for site in pair)
        self.attempts += 1
        self.exchange_attempts += 1
        if pair is None:
            return False
        first, second = pair
        nsites = self.wavefunction.nsites
        if first < 0 or first >= nsites or second < 0 or second >= nsites:
            raise IndexError("exchange site is out of range.")
        if first == second:
            raise ValueError("exchange sites must be distinct.")
        self.site_attempts[[first, second]] += 1
        first_value = int(configuration[first])
        second_value = int(configuration[second])
        if first_value == second_value:
            return False
        if (
            second_value >= self.wavefunction.dims[first]
            or first_value >= self.wavefunction.dims[second]
        ):
            return False

        target = configuration.copy()
        target[first], target[second] = second_value, first_value
        old_amplitude = self.cache.amplitude
        new_amplitude = self.cache.amplitude_for(target)
        if not np.isfinite(new_amplitude) or abs(new_amplitude) <= np.finfo(float).tiny:
            self.zero_amplitude_rejections += 1
            return False
        log_ratio = 2.0 * (np.log(abs(new_amplitude)) - np.log(abs(old_amplitude)))
        if log_ratio >= 0.0:
            move_probability = 1.0 / (1.0 + np.exp(-min(log_ratio, 700.0)))
        else:
            exponential = np.exp(max(log_ratio, -700.0))
            move_probability = exponential / (1.0 + exponential)
        if self.rng.random() >= move_probability:
            return False

        if not explicit_pair:
            reverse_count = len(self._eligible_exchange_pairs(target))
            if reverse_count == 0:
                return False
            hastings = min(1.0, len(eligible) / reverse_count)
            if self.rng.random() >= hastings:
                return False
        return self._record_acceptance(target, pair, "heat_bath")

    def step(self, site=None, *, pair=None, proposal=None):
        """Attempt one configured move; explicit arguments aid diagnostics/tests."""
        proposal = self.proposal if proposal is None else str(proposal)
        if proposal == "mixed":
            proposal = (
                "exchange"
                if self.rng.random() < self.exchange_probability
                else "single_site"
            )
        if proposal == "single_site":
            if pair is not None:
                raise ValueError("pair is only valid for exchange proposals.")
            return self._single_site_step(site)
        if proposal == "exchange":
            if site is not None:
                raise ValueError("site is only valid for single-site proposals.")
            return self._exchange_step(pair)
        if proposal == "charge_pair":
            if site is not None:
                raise ValueError("site is only valid for single-site proposals.")
            return self._charge_pair_step(pair)
        if proposal == "heat_bath":
            if site is not None:
                raise ValueError("site is only valid for single-site proposals.")
            return self._heat_bath_step(pair)
        raise ValueError(
            "proposal must be 'single_site', 'exchange', 'charge_pair', "
            "'heat_bath', or 'mixed'."
        )

    def sweep(self, nsteps=None):
        if nsteps is None:
            nsteps = self.wavefunction.nsites
        for _ in range(int(nsteps)):
            self.step()
        return self

    def _diagnostics_since(self, start):
        attempts = self.attempts - start[0]
        accepted = self.accepted - start[1]
        return MetropolisDiagnostics(
            attempts=int(attempts),
            accepted=int(accepted),
            acceptance_rate=float(accepted / attempts) if attempts else 0.0,
            zero_amplitude_rejections=int(self.zero_amplitude_rejections - start[2]),
            initialization_attempts=int(self.initialization_attempts),
            site_attempts=tuple(int(value) for value in self.site_attempts - start[3]),
            site_accepts=tuple(int(value) for value in self.site_accepts - start[4]),
            single_site_attempts=int(self.single_site_attempts - start[5]),
            single_site_accepts=int(self.single_site_accepts - start[6]),
            exchange_attempts=int(self.exchange_attempts - start[7]),
            exchange_accepts=int(self.exchange_accepts - start[8]),
        )

    def draw(self, nsamples, *, burn_in=100, sweeps_between=1):
        nsamples = int(nsamples)
        burn_in = int(burn_in)
        sweeps_between = int(sweeps_between)
        if nsamples < 1:
            raise ValueError("nsamples must be positive.")
        if burn_in < 0 or sweeps_between < 1:
            raise ValueError("burn_in must be nonnegative and sweeps_between positive.")
        start = (
            self.attempts,
            self.accepted,
            self.zero_amplitude_rejections,
            self.site_attempts.copy(),
            self.site_accepts.copy(),
            self.single_site_attempts,
            self.single_site_accepts,
            self.exchange_attempts,
            self.exchange_accepts,
        )
        for _ in range(burn_in):
            self.sweep()
        configurations = np.empty((nsamples, self.wavefunction.nsites), dtype=np.intp)
        amplitudes = np.empty(nsamples, dtype=self.wavefunction.dtype)
        for sample in range(nsamples):
            for _ in range(sweeps_between):
                self.sweep()
            configurations[sample] = self.cache.configuration
            amplitudes[sample] = self.cache.amplitude
        return configurations, amplitudes, self._diagnostics_since(start)


class VMC:
    """Sampling, local-energy, and stochastic-reconfiguration driver.

    SR applies its covariance matrix without forming the quadratic
    parameter-by-parameter matrix.  Its sparse backend stores only active
    configuration derivatives and is selected automatically when samples do
    not include a dense log-derivative array.
    """

    def __init__(
        self,
        state,
        *,
        seed=None,
        initial_configuration=None,
        matrix_element_tolerance=0.0,
        copy_tensors=True,
        proposal="single_site",
        exchange_probability=0.5,
    ):
        if not hasattr(state, "hamiltonian"):
            raise TypeError(
                "state must own a Hamiltonian; use VMC.from_tensors(...) for "
                "a bare tensor sequence."
            )
        self.graph = getattr(state, "graph", None)
        wavefunction = LETTAWavefunction.from_state(state, copy=copy_tensors)
        self._initialize(
            wavefunction,
            state.hamiltonian,
            source_state=state,
            seed=seed,
            initial_configuration=initial_configuration,
            matrix_element_tolerance=matrix_element_tolerance,
            proposal=proposal,
            exchange_probability=exchange_probability,
        )

    @classmethod
    def from_tensors(
        cls,
        tensors,
        hamiltonian,
        *,
        graph,
        seed=None,
        initial_configuration=None,
        matrix_element_tolerance=0.0,
        copy_tensors=True,
        proposal="single_site",
        exchange_probability=0.5,
    ):
        """Construct a detached VMC driver from tensors and an undirected tie graph."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("hamiltonian must be a Hamiltonian.")
        from .frontier import _normalize_graph, _parents_from_graph

        edges = _normalize_graph(hamiltonian, graph)
        parents = _parents_from_graph(len(hamiltonian.sites), edges)
        physical_groups = tuple(
            (site,) + tuple(parent_sites)
            for site, parent_sites in enumerate(parents)
        )
        result = cls.__new__(cls)
        wavefunction = LETTAWavefunction(
            tensors,
            physical_groups,
            hamiltonian.dims,
            copy=copy_tensors,
        )
        result.graph = edges
        result._initialize(
            wavefunction,
            hamiltonian,
            source_state=None,
            seed=seed,
            initial_configuration=initial_configuration,
            matrix_element_tolerance=matrix_element_tolerance,
            proposal=proposal,
            exchange_probability=exchange_probability,
        )
        return result

    def _initialize(
        self,
        wavefunction,
        hamiltonian,
        *,
        source_state,
        seed,
        initial_configuration,
        matrix_element_tolerance,
        proposal,
        exchange_probability,
    ):
        self.wavefunction = wavefunction
        self._source_state = source_state
        if isinstance(hamiltonian, Hamiltonian):
            self.action_operator = LocalHamiltonianActions(
                hamiltonian,
                matrix_element_tolerance=matrix_element_tolerance,
            )
            self.hamiltonian = hamiltonian
        elif hasattr(hamiltonian, "configuration_actions") and hasattr(
            hamiltonian, "dims"
        ):
            self.action_operator = hamiltonian
            self.hamiltonian = None
        else:
            raise TypeError(
                "hamiltonian must be Hamiltonian or implement "
                "configuration_actions."
            )
        if tuple(self.action_operator.dims) != self.wavefunction.dims:
            raise ValueError("Hamiltonian and wavefunction dimensions differ.")
        exchange_pairs = getattr(self, "graph", None)
        site_charges = None
        if self.hamiltonian is not None:
            if not exchange_pairs:
                exchange_pairs = sorted(
                    {
                        tuple(sorted(pair))
                        for support in self.hamiltonian.supports
                        for pair in combinations(support, 2)
                    }
                )
            if all(site.charges is not None for site in self.hamiltonian.sites):
                site_charges = tuple(
                    site.charges for site in self.hamiltonian.sites
                )
        self.sampler = MetropolisSampler(
            self.wavefunction,
            seed=seed,
            initial_configuration=initial_configuration,
            proposal=proposal,
            exchange_probability=exchange_probability,
            exchange_pairs=exchange_pairs or None,
            site_charges=site_charges,
        )

    @property
    def tensors(self):
        return self.wavefunction.tensors

    @property
    def physical_groups(self):
        return self.wavefunction.physical_groups

    @property
    def dims(self):
        return self.wavefunction.dims

    def amplitude(self, configuration):
        return self.wavefunction.amplitude(configuration)

    def amplitudes(self, configurations):
        return self.wavefunction.amplitudes(configurations)

    def log_derivative(self, configuration):
        return self.wavefunction.log_derivative(configuration)

    def log_derivatives(self, configurations):
        return self.wavefunction.log_derivatives(configurations)

    def local_energy(self, configuration, *, cache=None):
        configuration = _configuration(configuration, self.dims)
        if cache is None:
            cache = self.wavefunction.product_cache(configuration)
        elif cache.wavefunction is not self.wavefunction:
            raise ValueError("cache belongs to a different wavefunction.")
        elif not np.array_equal(cache.configuration, configuration):
            raise ValueError("cache configuration does not match configuration.")
        amplitude = cache.amplitude
        if not np.isfinite(amplitude) or abs(amplitude) <= np.finfo(float).tiny:
            raise ValueError("local energy is undefined at zero amplitude.")
        dtype = np.result_type(
            self.wavefunction.dtype, getattr(self.action_operator, "dtype", complex)
        )
        energy = np.zeros((), dtype=dtype)
        for matrix_element, target in self.action_operator.configuration_actions(
            configuration
        ):
            if np.array_equal(target, configuration):
                ratio = 1.0
            else:
                ratio = cache.amplitude_for(target) / amplitude
            energy = energy + matrix_element * ratio
        return energy.item()

    def sample(
        self,
        nsamples,
        *,
        burn_in=100,
        sweeps_between=1,
        include_log_derivatives=False,
    ):
        configurations, amplitudes, diagnostics = self.sampler.draw(
            nsamples,
            burn_in=burn_in,
            sweeps_between=sweeps_between,
        )
        local_energies = np.asarray(
            [self.local_energy(configuration) for configuration in configurations]
        )
        derivatives = (
            self.log_derivatives(configurations) if include_log_derivatives else None
        )
        return VMCSamples(
            configurations=configurations,
            amplitudes=amplitudes,
            local_energies=local_energies,
            log_derivatives=derivatives,
            diagnostics=diagnostics,
        )

    @staticmethod
    def estimate_from_samples(samples: VMCSamples):
        if not isinstance(samples, VMCSamples) or samples.nsamples < 1:
            raise TypeError("samples must be a nonempty VMCSamples object.")
        energy = np.mean(samples.local_energies)
        variance = float(np.mean(np.abs(samples.local_energies - energy) ** 2))
        real_values = np.asarray(samples.local_energies).real
        real_variance = float(np.mean((real_values - np.mean(real_values)) ** 2))
        autocorrelation_time, effective_samples, autocorrelation_error = (
            _autocorrelation_statistics(samples.local_energies)
        )
        return EnergyEstimate(
            energy=complex(energy),
            variance=variance,
            real_variance=real_variance,
            standard_error=float(np.sqrt(real_variance / samples.nsamples)),
            nsamples=samples.nsamples,
            diagnostics=samples.diagnostics,
            integrated_autocorrelation_time=autocorrelation_time,
            effective_sample_size=effective_samples,
            autocorrelation_standard_error=autocorrelation_error,
        )

    def estimate(
        self,
        nsamples,
        *,
        burn_in=100,
        sweeps_between=1,
        return_samples=False,
    ):
        samples = self.sample(
            nsamples,
            burn_in=burn_in,
            sweeps_between=sweeps_between,
        )
        estimate = self.estimate_from_samples(samples)
        return (estimate, samples) if return_samples else estimate

    def sr_direction(
        self,
        samples: VMCSamples,
        *,
        diagonal_shift=1.0e-3,
        diagonal_floor=1.0e-8,
        tolerance=1.0e-8,
        max_iterations=None,
        derivative_backend="auto",
        derivative_batch_size=256,
    ):
        """Compute a regularized SR direction using dense or sparse Jacobian matvecs."""
        if not isinstance(samples, VMCSamples):
            raise TypeError("samples must be a VMCSamples object.")
        derivative_backend = str(derivative_backend).lower().replace("-", "_")
        if derivative_backend not in {"auto", "dense", "sparse"}:
            raise ValueError("derivative_backend must be 'auto', 'dense', or 'sparse'.")
        derivatives = samples.log_derivatives
        if derivative_backend == "auto":
            derivative_backend = "dense" if derivatives is not None else "sparse"
        if derivative_backend == "dense":
            if derivatives is None:
                derivatives = self.log_derivatives(samples.configurations)
            derivatives = np.asarray(derivatives)
        else:
            derivatives = self.wavefunction.sparse_log_derivatives(
                samples.configurations,
                batch_size=derivative_batch_size,
            )
        if derivatives.shape != (samples.nsamples, self.wavefunction.nparameters):
            raise ValueError("sample log-derivative shape is inconsistent.")
        diagonal_shift = float(diagonal_shift)
        diagonal_floor = float(diagonal_floor)
        tolerance = float(tolerance)
        if diagonal_shift < 0.0 or diagonal_floor <= 0.0 or tolerance <= 0.0:
            raise ValueError(
                "diagonal_shift must be nonnegative; diagonal_floor and tolerance "
                "must be positive."
            )
        energy_centered = samples.local_energies - np.mean(samples.local_energies)
        if derivative_backend == "dense":
            centered = derivatives - np.mean(derivatives, axis=0, keepdims=True)
            force = centered.T.conj() @ energy_centered / samples.nsamples
            metric_diagonal = np.mean(np.abs(centered) ** 2, axis=0).real

            def covariance_action(vector):
                return centered.T.conj() @ (centered @ vector) / samples.nsamples

            stored_derivative_elements = int(derivatives.size)
        else:
            mean_derivative = np.asarray(derivatives.mean(axis=0)).reshape(-1)
            squared = derivatives.copy()
            squared.data = np.abs(squared.data) ** 2
            mean_abs_squared = np.asarray(squared.mean(axis=0)).reshape(-1).real
            metric_diagonal = np.maximum(
                mean_abs_squared - np.abs(mean_derivative) ** 2,
                0.0,
            )
            mean_energy_centered = np.mean(energy_centered)
            force = np.asarray(
                derivatives.conj().T @ energy_centered
            ).reshape(-1) / samples.nsamples
            force = force - mean_derivative.conj() * mean_energy_centered

            def covariance_action(vector):
                projected = np.asarray(derivatives @ vector).reshape(-1)
                projected = projected - mean_derivative @ vector
                result = np.asarray(
                    derivatives.conj().T @ projected
                ).reshape(-1) / samples.nsamples
                return result - mean_derivative.conj() * np.mean(projected)

            stored_derivative_elements = int(derivatives.nnz)
        regularizer = diagonal_shift * metric_diagonal + diagonal_floor

        def action(vector):
            return covariance_action(vector) + regularizer * vector

        right_hand_side = -force
        nparameters = self.wavefunction.nparameters
        if max_iterations is None:
            max_iterations = min(nparameters, 1000)
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        direction = np.zeros(nparameters, dtype=np.result_type(force.dtype, complex))
        residual = right_hand_side.copy()
        preconditioner = metric_diagonal + regularizer
        preconditioned = residual / preconditioner
        search = preconditioned.copy()
        residual_preconditioned = np.vdot(residual, preconditioned)
        force_norm = float(np.linalg.norm(force))
        target = tolerance * max(force_norm, 1.0)
        converged = float(np.linalg.norm(residual)) <= target
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            if converged:
                iterations -= 1
                break
            action_search = action(search)
            denominator = np.vdot(search, action_search)
            if abs(denominator) <= np.finfo(float).tiny:
                break
            alpha = residual_preconditioned / denominator
            direction = direction + alpha * search
            residual = residual - alpha * action_search
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= target:
                converged = True
                break
            preconditioned = residual / preconditioner
            next_residual_preconditioned = np.vdot(residual, preconditioned)
            if abs(residual_preconditioned) <= np.finfo(float).tiny:
                break
            beta = next_residual_preconditioned / residual_preconditioned
            search = preconditioned + beta * search
            residual_preconditioned = next_residual_preconditioned
        residual_norm = float(np.linalg.norm(action(direction) - right_hand_side))
        return SRDirection(
            direction=direction,
            force=force,
            metric_diagonal=metric_diagonal,
            diagonal_shift=diagonal_shift,
            diagonal_floor=diagonal_floor,
            iterations=int(iterations),
            converged=bool(converged),
            residual_norm=residual_norm,
            force_norm=force_norm,
            derivative_backend=derivative_backend,
            stored_derivative_elements=stored_derivative_elements,
        )

    def propose_sr(
        self,
        samples: VMCSamples,
        *,
        step_size=0.05,
        max_relative_update=0.1,
        **direction_options,
    ):
        step_size = float(step_size)
        max_relative_update = float(max_relative_update)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite.")
        if not np.isfinite(max_relative_update) or max_relative_update <= 0.0:
            raise ValueError("max_relative_update must be positive and finite.")
        direction = self.sr_direction(samples, **direction_options)
        parameters = self.wavefunction.parameter_vector()
        delta = step_size * direction.direction
        limit = max_relative_update * max(float(np.linalg.norm(parameters)), 1.0e-12)
        delta_norm = float(np.linalg.norm(delta))
        applied_scale = min(1.0, limit / delta_norm) if delta_norm > 0.0 else 1.0
        delta = applied_scale * delta
        proposed = parameters + delta
        return SRProposal(
            tensors=self.wavefunction.tensors_from_parameters(proposed),
            delta=delta,
            direction=direction,
            step_size=step_size,
            applied_scale=float(applied_scale),
            base_version=self.wavefunction.version,
        )

    def sync_to_state(self, state=None, *, copy_tensors=True):
        """Copy current VMC parameters into a compatible LETTA state."""

        if state is None:
            state = self._source_state
        if state is None:
            raise ValueError(
                "no source state is available; pass a compatible state explicitly."
            )
        if isinstance(self.wavefunction, ConditionalLETTAWavefunction):
            return self.wavefunction.sync_to_state(
                state,
                copy=copy_tensors,
            )
        missing = [
            name
            for name in ("tensors", "physical_groups", "dims")
            if not hasattr(state, name)
        ]
        if missing:
            raise TypeError(
                "state must expose tensors, physical_groups, and dims; missing "
                + ", ".join(missing)
            )
        if tuple(state.dims) != self.dims or tuple(
            tuple(sites) for sites in state.physical_groups
        ) != self.physical_groups:
            raise ValueError("state physical layout differs from the VMC wavefunction.")
        if len(state.tensors) != len(self.tensors) or any(
            np.shape(target) != source.shape
            for target, source in zip(state.tensors, self.tensors)
        ):
            raise ValueError("state tensor shapes differ from the VMC wavefunction.")
        state.tensors = [
            np.array(tensor, copy=bool(copy_tensors)) for tensor in self.tensors
        ]
        if hasattr(state, "energy"):
            state.energy = None
        if hasattr(state, "converged"):
            state.converged = False
        if hasattr(state, "history"):
            state.history = []
        return state

    def apply_sr(self, proposal: SRProposal, *, sync_to_state=False):
        if not isinstance(proposal, SRProposal):
            raise TypeError("proposal must be an SRProposal.")
        if proposal.base_version != self.wavefunction.version:
            raise ValueError("cannot apply an SR proposal made for stale parameters.")
        parameters = self.wavefunction.parameters_from_tensors(proposal.tensors)
        self.wavefunction.set_parameter_vector(parameters)
        self.sampler.refresh()
        if sync_to_state:
            self.sync_to_state()
        return self


__all__ = [
    "ConditionalLETTAWavefunction",
    "ConfigurationActionOperator",
    "EnergyEstimate",
    "LETTAProductCache",
    "LETTAWavefunction",
    "LocalHamiltonianActions",
    "MetropolisDiagnostics",
    "MetropolisSampler",
    "SRDirection",
    "SRProposal",
    "VMC",
    "VMCSamples",
]
