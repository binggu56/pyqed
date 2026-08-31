"""Continuum embedding for geometric quantum dynamics."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def _positive_eta(eta):
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be positive and finite")
    return eta


def _hermitian_matrix(matrix, *, name):
    matrix = matrix.astype(np.complex128) if sp.issparse(matrix) else np.asarray(
        matrix,
        dtype=np.complex128,
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        delta = matrix - matrix.getH()
        error = float(np.max(np.abs(delta.data), initial=0.0))
    else:
        error = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    if error > 1.0e-10:
        raise ValueError(f"{name} must be Hermitian; error={error:.3e}")
    return matrix


class DiagonalElectronicContinuum:
    r"""Quadrature representation of an electronic continuum.

    The retarded self-energy is evaluated as

    .. math::

       \Sigma^R(z)=\sum_c
       \frac{W_c W_c^\dagger}{z-\epsilon_c},
       \qquad z=E+i\eta.

    ``couplings[:, c]`` is :math:`W_c`. If quadrature ``weights`` are
    supplied, their square roots are absorbed into the couplings.

    This is an adaptation of Feshbach projection to a discretized electronic
    continuum. It is exact for the supplied nodes, weights, and couplings, but
    it does not construct a thermodynamic-limit continuum, dynamical
    screening, or a phonon bath. See H. Feshbach, Ann. Phys. 5, 357 (1958),
    DOI: 10.1016/0003-4916(58)90007-1.
    """

    def __init__(self, energies, couplings, *, weights=None):
        energies = np.asarray(energies, dtype=float)
        couplings = np.asarray(couplings, dtype=np.complex128)
        if energies.ndim != 1 or energies.size == 0:
            raise ValueError("energies must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(energies)):
            raise ValueError("continuum energies must be finite")
        if couplings.ndim != 2 or couplings.shape[1] != energies.size:
            raise ValueError(
                "couplings must have shape (nactive, ncontinuum)"
            )
        if not np.all(np.isfinite(couplings)):
            raise ValueError("continuum couplings must be finite")
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != energies.shape:
                raise ValueError("weights must match continuum energies")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError("quadrature weights must be positive and finite")
            couplings = couplings * np.sqrt(weights)[None, :]

        self.energies = energies
        self.couplings = couplings
        self.weights = None if weights is None else weights
        self.nactive, self.ncontinuum = couplings.shape

    def self_energy(self, energy, *, eta=1.0e-3):
        """Return the retarded active-space self-energy at one energy."""

        eta = _positive_eta(eta)
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError("energy must be finite")
        denominator = energy + 1.0j * eta - self.energies
        return (self.couplings / denominator[None, :]) @ self.couplings.conj().T

    def self_energy_operator(self, energy, *, eta=1.0e-3):
        """Return a matrix-free retarded self-energy operator."""

        eta = _positive_eta(eta)
        energy = float(energy)
        denominator = energy + 1.0j * eta - self.energies
        coupling = self.couplings

        def matvec(vector):
            return coupling @ (coupling.conj().T @ vector / denominator)

        return sla.LinearOperator(
            (self.nactive, self.nactive),
            matvec=matvec,
            dtype=np.complex128,
        )

    def hybridization(self, energy, *, eta=1.0e-3):
        r"""Return :math:`\Gamma(E)=-2\operatorname{Im}\Sigma^R(E)`."""

        sigma = self.self_energy(energy, eta=eta)
        return 1.0j * (sigma - sigma.conj().T)

    def memory_kernel(self, times):
        r"""Return :math:`K(t)=W\exp(-iH_Qt)W^\dagger` for ``t >= 0``."""

        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or np.any(times < 0.0) or not np.all(
            np.isfinite(times)
        ):
            raise ValueError("times must be a finite nonnegative 1D array")
        phases = np.exp(-1.0j * np.outer(times, self.energies))
        return np.einsum(
            "ac,tc,bc->tab",
            self.couplings,
            phases,
            self.couplings.conj(),
            optimize=True,
        )


class MatrixElectronicContinuum:
    r"""Finite matrix representation of an electronic continuum.

    This backend evaluates

    .. math::

       \Sigma^R(z)=W(z-H_{QQ})^{-1}W^\dagger

    by a dense or sparse linear solve. Use :meth:`diagonalize` before a dense
    energy scan when the finite continuum is small enough to diagonalize.
    """

    def __init__(self, hamiltonian, coupling):
        hamiltonian = _hermitian_matrix(
            hamiltonian,
            name="continuum_hamiltonian",
        )
        coupling = coupling.astype(np.complex128) if sp.issparse(
            coupling
        ) else np.asarray(coupling, dtype=np.complex128)
        if coupling.ndim != 2 or coupling.shape[1] != hamiltonian.shape[0]:
            raise ValueError(
                "coupling must have shape (nactive, ncontinuum)"
            )
        self.hamiltonian = hamiltonian
        self.coupling = coupling.tocsr() if sp.issparse(coupling) else coupling
        self.nactive, self.ncontinuum = coupling.shape

    def self_energy(self, energy, *, eta=1.0e-3):
        """Return the retarded active-space self-energy at one energy."""

        eta = _positive_eta(eta)
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError("energy must be finite")
        z = energy + 1.0j * eta
        coupling = self.coupling
        right = coupling.conj().T
        if sp.issparse(self.hamiltonian):
            shifted = z * sp.eye(
                self.ncontinuum,
                dtype=np.complex128,
                format="csc",
            ) - self.hamiltonian.tocsc()
            right = right.toarray() if sp.issparse(right) else right
            solution = sla.spsolve(shifted, right)
        else:
            shifted = z * np.eye(self.ncontinuum) - self.hamiltonian
            solution = scipy.linalg.solve(
                shifted,
                right,
                assume_a="gen",
                check_finite=False,
            )
        value = coupling @ solution
        return value.toarray() if sp.issparse(value) else np.asarray(value)

    def hybridization(self, energy, *, eta=1.0e-3):
        r"""Return :math:`\Gamma(E)=-2\operatorname{Im}\Sigma^R(E)`."""

        sigma = self.self_energy(energy, eta=eta)
        return 1.0j * (sigma - sigma.conj().T)

    def diagonalize(self):
        """Return an equivalent diagonal-continuum backend."""

        matrix = (
            self.hamiltonian.toarray()
            if sp.issparse(self.hamiltonian)
            else self.hamiltonian
        )
        energies, frames = np.linalg.eigh(matrix)
        coupling = self.coupling @ frames
        if sp.issparse(coupling):
            coupling = coupling.toarray()
        return DiagonalElectronicContinuum(energies, coupling)


class FeshbachEmbedding:
    r"""Active Hamiltonian embedded in an eliminated continuum.

    Given active and continuum projectors :math:`P` and :math:`Q=1-P`, this
    class evaluates the exact finite-dimensional Feshbach resolvent

    .. math::

       G_P^R(E)=
       [E+i\eta-H_{PP}-\Sigma^R(E)]^{-1},

    .. math::

       \Sigma^R(E)=
       H_{PQ}[E+i\eta-H_{QQ}]^{-1}H_{QP}.

    :meth:`from_ldr` is an adapter that partitions a complete overlap-dressed
    LDR Hamiltonian, so geometric kinetic coupling between active and
    eliminated electronic subspaces is retained. The Feshbach reduction is
    exact for the finite parent Hamiltonian. It is not a thermodynamic-limit
    solid-state continuum solver: continuum interpolation, many-body
    screening, and tensor-network time propagation remain external
    responsibilities.

    Reference: H. Feshbach, Ann. Phys. 5, 357-390 (1958),
    DOI: 10.1016/0003-4916(58)90007-1.
    """

    def __init__(self, active_hamiltonian, continuum):
        active_hamiltonian = _hermitian_matrix(
            active_hamiltonian,
            name="active_hamiltonian",
        )
        if not hasattr(continuum, "self_energy"):
            raise TypeError("continuum must provide self_energy(energy, eta=...)")
        if int(continuum.nactive) != active_hamiltonian.shape[0]:
            raise ValueError("continuum and active Hamiltonian dimensions differ")

        self.active_hamiltonian = active_hamiltonian
        self.continuum = continuum
        self.nactive = active_hamiltonian.shape[0]
        self.ncontinuum = int(continuum.ncontinuum)

        self.source_solver = None
        self.local_active_states = None
        self.active_indices = None
        self.continuum_indices = None
        self.minimum_projector_overlap = None
        self.maximum_projector_leakage = None
        self.continuum_coupling_norm = None

        self.energy_grid = None
        self.spectral_density = None
        self.self_energy_trace = None
        self.hybridization_trace = None
        self.self_energies = None
        self.green_functions = None
        self.success = False
        self.message = "not run"

    @staticmethod
    def _local_states(solver, active_states):
        if np.isscalar(active_states):
            count = int(active_states)
            if float(active_states) != count:
                raise ValueError("active state count must be an integer")
            if not 0 < count < solver.nstates:
                raise ValueError("active state count must lie in [1, nstates)")
            states = np.tile(np.arange(count), (solver.ngrid, 1))
        else:
            states = np.asarray(active_states)
            if not np.issubdtype(states.dtype, np.integer):
                if not np.all(np.equal(states, np.asarray(states, dtype=int))):
                    raise ValueError("active state indices must be integers")
            if states.ndim == 1:
                states = np.tile(states.astype(int), (solver.ngrid, 1))
            elif states.ndim == 2 and states.shape[0] == solver.ngrid:
                states = states.astype(int)
            else:
                raise ValueError(
                    "active_states must be a count, a state list, or an "
                    "(ngrid, nactive_local) array"
                )
        if states.shape[1] == 0 or states.shape[1] >= solver.nstates:
            raise ValueError("retain at least one state and eliminate at least one")
        if np.any(states < 0) or np.any(states >= solver.nstates):
            raise ValueError("active state index is outside the local state space")
        if any(np.unique(row).size != row.size for row in states):
            raise ValueError("active state indices must be unique at each grid point")
        return states

    @classmethod
    def from_ldr(
        cls,
        solver,
        active_states,
        *,
        time=0.0,
        diagonalize_continuum=True,
    ):
        """Partition a complete finite LDR Hamiltonian into ``P`` and ``Q``."""

        required = ("ngrid", "nstates", "shape", "hamiltonian")
        if any(not hasattr(solver, name) for name in required):
            raise TypeError("solver is not an LDR-compatible object")
        local_states = cls._local_states(solver, active_states)
        offsets = solver.nstates * np.arange(solver.ngrid)[:, None]
        active_indices = (offsets + local_states).reshape(-1)
        mask = np.ones(solver.ngrid * solver.nstates, dtype=bool)
        mask[active_indices] = False
        continuum_indices = np.flatnonzero(mask)

        full = solver.hamiltonian(time=time, sparse=True).tocsr()
        active_hamiltonian = full[active_indices][:, active_indices].tocsr()
        continuum_hamiltonian = full[continuum_indices][
            :,
            continuum_indices,
        ].tocsr()
        coupling = full[active_indices][:, continuum_indices].tocsr()
        continuum = MatrixElectronicContinuum(
            continuum_hamiltonian,
            coupling,
        )
        if diagonalize_continuum:
            continuum = continuum.diagonalize()

        embedded = cls(active_hamiltonian, continuum)
        embedded.source_solver = solver
        embedded.local_active_states = local_states
        embedded.active_indices = active_indices
        embedded.continuum_indices = continuum_indices
        embedded.continuum_coupling_norm = float(sla.norm(coupling))
        embedded._set_projector_diagnostics()
        return embedded

    def _set_projector_diagnostics(self):
        links = getattr(self.source_solver, "links", None)
        if links is None:
            return
        singular_values = []
        shape = self.source_solver.shape
        for (axis, index), link in links.items():
            right = list(index)
            right[axis] += 1
            left_flat = np.ravel_multi_index(index, shape)
            right_flat = np.ravel_multi_index(tuple(right), shape)
            left_states = self.local_active_states[left_flat]
            right_states = self.local_active_states[right_flat]
            block = np.asarray(link)[np.ix_(left_states, right_states)]
            singular_values.extend(np.linalg.svd(block, compute_uv=False))
        if singular_values:
            minimum = float(np.min(singular_values))
            self.minimum_projector_overlap = minimum
            self.maximum_projector_leakage = float(
                np.sqrt(max(0.0, 1.0 - min(1.0, minimum) ** 2))
            )

    def self_energy(self, energy, *, eta=1.0e-3):
        """Return the retarded continuum self-energy."""

        return self.continuum.self_energy(energy, eta=eta)

    def green_function(self, energy, *, eta=1.0e-3):
        """Return the dense retarded active-space Green function."""

        eta = _positive_eta(eta)
        active = (
            self.active_hamiltonian.toarray()
            if sp.issparse(self.active_hamiltonian)
            else self.active_hamiltonian
        )
        shifted = (
            (float(energy) + 1.0j * eta) * np.eye(self.nactive)
            - active
            - self.self_energy(energy, eta=eta)
        )
        return scipy.linalg.solve(
            shifted,
            np.eye(self.nactive),
            assume_a="gen",
            check_finite=False,
        )

    @staticmethod
    def _projected_trace(green, probe):
        if probe is None:
            return np.trace(green)
        probe = np.asarray(probe, dtype=np.complex128)
        if probe.ndim == 1:
            probe = probe[:, None]
        if probe.ndim != 2 or probe.shape[0] != green.shape[0]:
            raise ValueError("probe must have shape (nactive,) or (nactive, nprobe)")
        return np.trace(probe.conj().T @ green @ probe)

    def run_spectrum(
        self,
        energies,
        *,
        eta=1.0e-3,
        probe=None,
        store_matrices=False,
    ):
        r"""Evaluate :math:`-\pi^{-1}\operatorname{Im}\operatorname{Tr}G_P^R`."""

        eta = _positive_eta(eta)
        energies = np.asarray(energies, dtype=float)
        if energies.ndim != 1 or energies.size < 2:
            raise ValueError("energies must be a one-dimensional grid")
        if not np.all(np.isfinite(energies)) or np.any(np.diff(energies) <= 0.0):
            raise ValueError("energies must be finite and strictly increasing")

        spectrum = np.empty(energies.size)
        sigma_trace = np.empty(energies.size, dtype=np.complex128)
        green_store = [] if store_matrices else None
        sigma_store = [] if store_matrices else None
        active = (
            self.active_hamiltonian.toarray()
            if sp.issparse(self.active_hamiltonian)
            else self.active_hamiltonian
        )
        identity = np.eye(self.nactive)
        for index, energy in enumerate(energies):
            sigma = self.self_energy(energy, eta=eta)
            shifted = (
                (energy + 1.0j * eta) * identity
                - active
                - sigma
            )
            green = scipy.linalg.solve(
                shifted,
                identity,
                assume_a="gen",
                check_finite=False,
            )
            spectrum[index] = -np.imag(self._projected_trace(green, probe)) / np.pi
            sigma_trace[index] = np.trace(sigma)
            if store_matrices:
                green_store.append(green)
                sigma_store.append(sigma)

        self.energy_grid = energies
        self.spectral_density = spectrum
        self.self_energy_trace = sigma_trace
        self.hybridization_trace = -2.0 * np.imag(sigma_trace)
        self.green_functions = None if green_store is None else np.asarray(green_store)
        self.self_energies = None if sigma_store is None else np.asarray(sigma_store)
        self.success = True
        self.message = "computed the continuum-embedded active-space spectrum"
        return self


__all__ = [
    "DiagonalElectronicContinuum",
    "FeshbachEmbedding",
    "MatrixElectronicContinuum",
]
