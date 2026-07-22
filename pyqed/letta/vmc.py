r"""Variational Monte Carlo for unrestricted physically tied LETTA states.

The routines in this module contract only single configurations.  They do not
construct a many-body basis, a dense state vector, or a frontier environment.
For a configuration ``s``, a physically tied LETTA has amplitude

.. math::

    \psi(s) = A_0(s_{P_0}) A_1(s_{P_1}) \cdots A_{N-1}(s_{P_{N-1}}),

where every selected ``A_i`` is a matrix on the virtual bonds.  This makes
Metropolis sampling useful when the exact frontier width is too large.

The public entry point is :class:`pyqed.letta.LETTAVMC`; the lower-level
sampling and stochastic-reconfiguration records are also exported there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .local_terms import LocalHamiltonian


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

    def __init__(self, tensors, physical_sites, dims=None, *, copy=True):
        tensors = tuple(np.asarray(tensor) for tensor in tensors)
        physical_sites = tuple(
            tuple(int(site) for site in sites) for sites in physical_sites
        )
        if not tensors:
            raise ValueError("tensors must contain at least one tensor.")
        if len(tensors) != len(physical_sites):
            raise ValueError("tensors and physical_sites must have equal length.")
        nsites = len(tensors)
        for tensor_site, (tensor, sites) in enumerate(zip(tensors, physical_sites)):
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
                raise ValueError(f"physical_sites[{tensor_site}] contains duplicates.")
            if any(site < 0 or site >= nsites for site in sites):
                raise ValueError(
                    f"physical_sites[{tensor_site}] contains an invalid site."
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
        for tensor_site, (tensor, sites) in enumerate(zip(tensors, physical_sites)):
            for axis, physical_site in enumerate(sites, start=2):
                dim = int(tensor.shape[axis])
                previous = inferred_dims[physical_site]
                if previous is not None and previous != dim:
                    raise ValueError(
                        f"inconsistent dimension for physical site {physical_site}."
                    )
                inferred_dims[physical_site] = dim
        if any(dim is None for dim in inferred_dims):
            raise ValueError("every physical site must occur in physical_sites.")
        inferred_dims = tuple(inferred_dims)
        if dims is None:
            dims = inferred_dims
        else:
            dims = tuple(int(dim) for dim in dims)
            if dims != inferred_dims:
                raise ValueError("dims are inconsistent with tensor physical axes.")

        self.dims = dims
        self.physical_sites = physical_sites
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
        for tensor_site, sites in enumerate(physical_sites):
            for physical_site in sites:
                dependencies[physical_site].append(tensor_site)
        self.dependent_tensors = tuple(tuple(sites) for sites in dependencies)
        self._sizes = tuple(tensor.size for tensor in self.tensors)
        self._offsets = np.cumsum((0,) + self._sizes)
        self.version = 0

    @classmethod
    def from_state(cls, state, *, copy=True):
        """Construct from an object exposing tensors, physical_sites, and dims."""
        missing = [
            name
            for name in ("tensors", "physical_sites", "dims")
            if not hasattr(state, name)
        ]
        if missing:
            raise TypeError(
                "state must expose tensors, physical_sites, and dims; missing "
                + ", ".join(missing)
            )
        return cls(
            state.tensors,
            state.physical_sites,
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
        sites = self.physical_sites[tensor_site]
        index = (slice(None), slice(None)) + tuple(
            int(configuration[site]) for site in sites
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

    def set_parameter_vector(self, parameters):
        self.tensors = list(self.tensors_from_parameters(parameters))
        self.dtype = np.dtype(
            np.result_type(*[tensor.dtype for tensor in self.tensors])
        )
        self.version += 1

    def log_derivative(self, configuration):
        r"""Return the holomorphic derivatives ``partial log(psi)/partial theta``."""
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

        result = np.zeros(self.nparameters, dtype=self.dtype)
        for site, tensor in enumerate(self.tensors):
            left_dim, right_dim = tensor.shape[:2]
            local_dims = tuple(self.dims[index] for index in self.physical_sites[site])
            physical_index = np.ravel_multi_index(
                tuple(configuration[index] for index in self.physical_sites[site]),
                local_dims,
            )
            local_size = int(np.prod(local_dims))
            derivative = np.outer(left[site], right[site + 1]) / amplitude
            flat_indices = (
                self._offsets[site]
                + np.arange(left_dim * right_dim, dtype=np.intp) * local_size
                + physical_index
            )
            result[flat_indices] = derivative.reshape(-1)
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
    """Configuration-action adapter for :class:`LocalHamiltonian`."""

    def __init__(self, hamiltonian: LocalHamiltonian, *, matrix_element_tolerance=0.0):
        if not isinstance(hamiltonian, LocalHamiltonian):
            raise TypeError("hamiltonian must be a LocalHamiltonian.")
        tolerance = float(matrix_element_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("matrix_element_tolerance must be nonnegative.")
        self.hamiltonian = hamiltonian
        self.dims = hamiltonian.dims
        self.dtype = hamiltonian.dtype
        self.matrix_element_tolerance = tolerance

    def configuration_actions(self, configuration):
        configuration = _configuration(configuration, self.dims)
        if self.hamiltonian.constant != 0.0:
            yield self.hamiltonian.constant, configuration
        for term in self.hamiltonian.terms:
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
    ):
        self.wavefunction = wavefunction
        self.rng = np.random.default_rng(seed)
        proposal = str(proposal)
        if proposal not in {"single_site", "exchange", "mixed"}:
            raise ValueError("proposal must be 'single_site', 'exchange', or 'mixed'.")
        exchange_probability = float(exchange_probability)
        if not np.isfinite(exchange_probability) or not 0.0 <= exchange_probability <= 1.0:
            raise ValueError("exchange_probability must lie in [0, 1].")
        self.proposal = proposal
        self.exchange_probability = exchange_probability
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
        raise ValueError("proposal must be 'single_site', 'exchange', or 'mixed'.")

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


class LETTAVMC:
    """Sampling, local-energy, and stochastic-reconfiguration driver.

    The SR implementation stores the sample-by-parameter log-derivative array,
    but applies its covariance matrix without forming the quadratic
    parameter-by-parameter matrix.  Callers should batch or thin samples when
    that rectangular array becomes the memory bottleneck.
    """

    def __init__(
        self,
        state_or_tensors,
        hamiltonian,
        physical_sites=None,
        dims=None,
        *,
        seed=None,
        initial_configuration=None,
        matrix_element_tolerance=0.0,
        copy_tensors=True,
        proposal="single_site",
        exchange_probability=0.5,
    ):
        self._source_state = None
        if hasattr(state_or_tensors, "tensors"):
            if physical_sites is not None or dims is not None:
                raise ValueError(
                    "physical_sites and dims must be omitted when a state is supplied."
                )
            self.wavefunction = LETTAWavefunction.from_state(
                state_or_tensors, copy=copy_tensors
            )
            self._source_state = state_or_tensors
        else:
            if physical_sites is None:
                raise ValueError(
                    "physical_sites are required when supplying a tensor sequence."
                )
            self.wavefunction = LETTAWavefunction(
                state_or_tensors, physical_sites, dims, copy=copy_tensors
            )

        if isinstance(hamiltonian, LocalHamiltonian):
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
                "hamiltonian must be LocalHamiltonian or implement "
                "configuration_actions."
            )
        if tuple(self.action_operator.dims) != self.wavefunction.dims:
            raise ValueError("Hamiltonian and wavefunction dimensions differ.")
        self.sampler = MetropolisSampler(
            self.wavefunction,
            seed=seed,
            initial_configuration=initial_configuration,
            proposal=proposal,
            exchange_probability=exchange_probability,
        )

    @property
    def tensors(self):
        return self.wavefunction.tensors

    @property
    def physical_sites(self):
        return self.wavefunction.physical_sites

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
    ):
        """Compute a matrix-free regularized stochastic-reconfiguration direction."""
        if not isinstance(samples, VMCSamples):
            raise TypeError("samples must be a VMCSamples object.")
        derivatives = samples.log_derivatives
        if derivatives is None:
            derivatives = self.log_derivatives(samples.configurations)
        derivatives = np.asarray(derivatives)
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
        centered = derivatives - np.mean(derivatives, axis=0, keepdims=True)
        energy_centered = samples.local_energies - np.mean(samples.local_energies)
        force = centered.T.conj() @ energy_centered / samples.nsamples
        metric_diagonal = np.mean(np.abs(centered) ** 2, axis=0).real
        regularizer = diagonal_shift * metric_diagonal + diagonal_floor

        def action(vector):
            return (
                centered.T.conj() @ (centered @ vector) / samples.nsamples
                + regularizer * vector
            )

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
        missing = [
            name
            for name in ("tensors", "physical_sites", "dims")
            if not hasattr(state, name)
        ]
        if missing:
            raise TypeError(
                "state must expose tensors, physical_sites, and dims; missing "
                + ", ".join(missing)
            )
        if tuple(state.dims) != self.dims or tuple(
            tuple(sites) for sites in state.physical_sites
        ) != self.physical_sites:
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
        parameters = np.concatenate([tensor.reshape(-1) for tensor in proposal.tensors])
        self.wavefunction.set_parameter_vector(parameters)
        self.sampler.refresh()
        if sync_to_state:
            self.sync_to_state()
        return self


__all__ = [
    "ConfigurationActionOperator",
    "EnergyEstimate",
    "LETTAProductCache",
    "LETTAVMC",
    "LETTAWavefunction",
    "LocalHamiltonianActions",
    "MetropolisDiagnostics",
    "MetropolisSampler",
    "SRDirection",
    "SRProposal",
    "VMCSamples",
]
