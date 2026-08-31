"""Projected periodic TDA continua and total-momentum bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import linalg as sla

from pyqed.units import kelvin


def _positive_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def bose_occupation(frequency, temperature):
    r"""Return :math:`N(\omega,T)=[e^{\omega/(k_BT)}-1]^{-1}`.

    ``frequency`` is in Hartree and ``temperature`` is in kelvin.  The exact
    zero-temperature limit is returned without evaluating the exponential.
    """

    frequency = _positive_finite(frequency, "frequency")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be nonnegative and finite")
    if temperature == 0.0:
        return 0.0
    exponent = frequency / (temperature * float(kelvin))
    if exponent > 700.0:
        return 0.0
    return float(1.0 / np.expm1(exponent))


class TotalMomentumSector:
    r"""One conserved exciton-plus-phonon crystal-momentum sector.

    For an exciton momentum :math:`K` and phonon occupations
    :math:`n_{q\nu}`, the represented configurations obey

    .. math::

       P_{\mathrm{tot}} = K + \sum_{q\nu} n_{q\nu}q \pmod G.

    The transition space supplies the reciprocal-cell wrapping and the
    commensurate momentum mesh.
    """

    def __init__(self, transition_space, total_q_index=0):
        required = ("qpts", "normalize_q_index", "find_qpoint_index")
        if any(not hasattr(transition_space, name) for name in required):
            raise TypeError("transition_space does not provide q-point bookkeeping")
        self.transition_space = transition_space
        self.total_q_index = transition_space.normalize_q_index(total_q_index)
        self.total_qvec = np.asarray(
            transition_space.qpts[self.total_q_index],
            dtype=float,
        )

    @staticmethod
    def _occupations(q_indices, occupations):
        q_indices = np.asarray(q_indices, dtype=object)
        occupations = np.asarray(occupations, dtype=object)
        if q_indices.ndim != 1 or occupations.ndim != 1:
            raise ValueError("phonon q indices and occupations must be 1D")
        if q_indices.shape != occupations.shape:
            raise ValueError("phonon q indices and occupations must have equal length")
        normalized_occupations = []
        for value in occupations:
            integer = int(value)
            if integer != value or integer < 0:
                raise ValueError("phonon occupations must be nonnegative integers")
            normalized_occupations.append(integer)
        return list(q_indices), np.asarray(normalized_occupations, dtype=int)

    def exciton_q_index(self, phonon_q_indices=(), occupations=()):
        """Return the exciton momentum required by total-momentum conservation."""

        q_indices, occupations = self._occupations(
            phonon_q_indices,
            occupations,
        )
        phonon_momentum = np.zeros(3)
        for q_index, occupation in zip(q_indices, occupations):
            index = self.transition_space.normalize_q_index(q_index)
            phonon_momentum += occupation * self.transition_space.qpts[index]
        return self.transition_space.find_qpoint_index(
            self.total_qvec - phonon_momentum
        )

    def contains(self, exciton_q_index, phonon_q_indices=(), occupations=()):
        """Return whether a configuration belongs to this total-momentum sector."""

        exciton_q_index = self.transition_space.normalize_q_index(exciton_q_index)
        expected = self.exciton_q_index(phonon_q_indices, occupations)
        return exciton_q_index == expected


class ExcitonPhononCoupling:
    r"""One mass-weighted derivative of a momentum-resolved TDA operator.

    The derivative maps a source exciton block at :math:`K` into a target
    block at :math:`K+q`. In atomic units the quantized one-phonon coupling is

    .. math::

       M_{q\nu} = (2\omega_{q\nu})^{-1/2}
       \frac{\partial H_{\mathrm{TDA}}}{\partial Q_{q\nu}}.

    This is an interface for finite-difference or analytic derivatives. It
    does not itself differentiate the quasiparticle energies, screened BSE
    kernel, or orbital basis.
    """

    def __init__(
        self,
        derivative,
        frequency,
        *,
        phonon_q_index,
        source_q_index,
        target_q_index,
        branch=None,
    ):
        derivative = sla.aslinearoperator(derivative)
        self.derivative = derivative
        self.frequency = _positive_finite(frequency, "frequency")
        self.zero_point_amplitude = 1.0 / np.sqrt(2.0 * self.frequency)
        self.phonon_q_index = int(phonon_q_index)
        self.source_q_index = int(source_q_index)
        self.target_q_index = int(target_q_index)
        self.branch = None if branch is None else int(branch)

    @classmethod
    def from_finite_difference(
        cls,
        plus_operator,
        minus_operator,
        displacement,
        frequency,
        **kwargs,
    ):
        r"""Build :math:`\partial H/\partial Q` from central differences."""

        displacement = _positive_finite(displacement, "displacement")
        plus = sla.aslinearoperator(plus_operator)
        minus = sla.aslinearoperator(minus_operator)
        if plus.shape != minus.shape:
            raise ValueError("displaced TDA operators must have equal shape")
        scale = 1.0 / (2.0 * displacement)

        def matvec(vector):
            return scale * (plus.matvec(vector) - minus.matvec(vector))

        def rmatvec(vector):
            return scale * (plus.rmatvec(vector) - minus.rmatvec(vector))

        derivative = sla.LinearOperator(
            plus.shape,
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.result_type(plus.dtype, minus.dtype, np.complex128),
        )
        return cls(derivative, frequency, **kwargs)

    def validate_momentum(self, transition_space):
        """Validate the declared :math:`K\rightarrow K+q` selection rule."""

        source = transition_space.normalize_q_index(self.source_q_index)
        phonon = transition_space.normalize_q_index(self.phonon_q_index)
        target = transition_space.normalize_q_index(self.target_q_index)
        expected = transition_space.find_qpoint_index(
            transition_space.qpts[source] + transition_space.qpts[phonon]
        )
        if target != expected:
            raise ValueError(
                "exciton-phonon coupling violates K -> K+q momentum conservation"
            )
        return self

    def _apply(self, vectors):
        vectors = np.asarray(vectors, dtype=np.complex128)
        vector_input = vectors.ndim == 1
        if vector_input:
            vectors = vectors[:, None]
        if vectors.ndim != 2 or vectors.shape[0] != self.derivative.shape[1]:
            raise ValueError(
                "source vectors must match the derivative source dimension"
            )
        result = np.column_stack(
            [self.derivative.matvec(vectors[:, column]) for column in range(vectors.shape[1])]
        )
        result *= self.zero_point_amplitude
        return result[:, 0] if vector_input else result

    def between(self, target_vectors, source_vectors):
        r"""Return :math:`\langle A_{K+q}|M_{q\nu}|A_K\rangle`."""

        target_vectors = np.asarray(target_vectors, dtype=np.complex128)
        if target_vectors.ndim == 1:
            target_vectors = target_vectors[:, None]
        if target_vectors.ndim != 2 or target_vectors.shape[0] != self.derivative.shape[0]:
            raise ValueError(
                "target vectors must match the derivative target dimension"
            )
        return target_vectors.conj().T @ self._apply(source_vectors)

    def active_to_target(self, source_vectors):
        """Return active-to-target-basis coupling rows for continuum embedding."""

        return self._apply(source_vectors).conj().T


class ProjectedTDAContinuum:
    r"""Matrix-free TDA continuum with selected exciton poles removed.

    Let the columns of :math:`A` span poles excluded from this target momentum
    sector and define :math:`Q=1-AA^\dagger`. Given source-active-to-target-
    transition coupling :math:`V`,
    this class evaluates

    .. math::

       \Sigma^R(E)
       = VQ\left[E+i\eta-QH_{\mathrm{TDA}}Q\right]^{-1}QV^\dagger.

    The projected linear systems are solved with GMRES. The construction is
    exact up to the iterative tolerance for the supplied finite TDA operator;
    it is an adaptation of Feshbach projection, not a phonon self-energy or a
    thermodynamic-limit continuum model.

    References
    ----------
    H. Feshbach, Ann. Phys. 5, 357-390 (1958),
    DOI: 10.1016/0003-4916(58)90007-1.

    M. Rohlfing and S. G. Louie, Phys. Rev. B 62, 4927-4944 (2000),
    DOI: 10.1103/PhysRevB.62.4927.
    """

    def __init__(
        self,
        operator,
        excluded_vectors,
        coupling,
        *,
        solver_tol=1.0e-10,
        maxiter=None,
        use_diagonal_preconditioner=True,
    ):
        source_operator = operator
        if hasattr(operator, "aslinearoperator"):
            operator = operator.aslinearoperator()
        operator = sla.aslinearoperator(operator)
        if operator.shape[0] != operator.shape[1]:
            raise ValueError("TDA operator must be square")

        excluded_vectors = np.asarray(excluded_vectors, dtype=np.complex128)
        if excluded_vectors.ndim == 1:
            excluded_vectors = excluded_vectors[:, None]
        if excluded_vectors.ndim != 2 or excluded_vectors.shape[0] != operator.shape[0]:
            raise ValueError(
                "excluded_vectors must have shape (ntransition, nexcluded)"
            )
        if excluded_vectors.shape[1] >= operator.shape[0]:
            raise ValueError("at least one target continuum direction is required")
        if excluded_vectors.shape[1]:
            singular_values = np.linalg.svd(excluded_vectors, compute_uv=False)
            if singular_values[-1] <= 1.0e-12 * singular_values[0]:
                raise ValueError("excluded_vectors must be linearly independent")
            orthonormality_error = float(
                np.linalg.norm(
                    excluded_vectors.conj().T @ excluded_vectors
                    - np.eye(excluded_vectors.shape[1])
                )
            )
            excluded_vectors, _triangular = np.linalg.qr(excluded_vectors)
        else:
            orthonormality_error = 0.0

        coupling = np.asarray(coupling, dtype=np.complex128)
        if coupling.ndim != 2 or coupling.shape[1] != operator.shape[0]:
            raise ValueError(
                "coupling must have shape (nactive, ntransition)"
            )
        if coupling.shape[0] == 0:
            raise ValueError("coupling must contain at least one active source state")
        if not np.all(np.isfinite(coupling)):
            raise ValueError("coupling must be finite")

        self.source_operator = source_operator
        self.operator = operator
        self.excluded_vectors = excluded_vectors
        self.ntransition = int(operator.shape[0])
        self.nactive = int(coupling.shape[0])
        self.nexcluded = int(excluded_vectors.shape[1])
        self.ncontinuum = self.ntransition - self.nexcluded
        self.solver_tol = _positive_finite(solver_tol, "solver_tol")
        if maxiter is not None:
            maxiter = int(maxiter)
            if maxiter < 1:
                raise ValueError("maxiter must be positive")
        self.maxiter = maxiter
        self.use_diagonal_preconditioner = bool(use_diagonal_preconditioner)
        self.excluded_orthonormality_error = orthonormality_error
        self.removed_pole_coupling_norm = float(
            np.linalg.norm(coupling @ excluded_vectors)
        )
        self.coupling = self._project_q(coupling.conj().T).conj().T
        self.projected_coupling_norm = float(np.linalg.norm(self.coupling))
        self.excluded_hamiltonian = self._excluded_hamiltonian()
        self.excluded_residual_norms = self._excluded_residual_norms()
        self.last_solve_info = None

    def _project_p(self, value):
        if self.nexcluded == 0:
            return np.zeros_like(value)
        return self.excluded_vectors @ (self.excluded_vectors.conj().T @ value)

    def _project_q(self, value):
        return value - self._project_p(value)

    def _hamiltonian_matmat(self, values):
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim == 1:
            return np.asarray(self.operator.matvec(values), dtype=np.complex128)
        return np.column_stack(
            [self.operator.matvec(values[:, column]) for column in range(values.shape[1])]
        )

    def _excluded_hamiltonian(self):
        if self.nexcluded == 0:
            return np.zeros((0, 0), dtype=np.complex128)
        applied = self._hamiltonian_matmat(self.excluded_vectors)
        matrix = self.excluded_vectors.conj().T @ applied
        return 0.5 * (matrix + matrix.conj().T)

    def _excluded_residual_norms(self):
        if self.nexcluded == 0:
            return np.zeros(0)
        applied = self._hamiltonian_matmat(self.excluded_vectors)
        residual = self._project_q(applied)
        return np.linalg.norm(residual, axis=0)

    def _shifted_operator(self, z):
        def matvec(vector):
            vector = np.asarray(vector, dtype=np.complex128)
            qvector = self._project_q(vector)
            shifted_q = z * qvector - self._hamiltonian_matmat(qvector)
            return self._project_q(shifted_q) + self._project_p(vector)

        return sla.LinearOperator(
            (self.ntransition, self.ntransition),
            matvec=matvec,
            dtype=np.complex128,
        )

    def _projected_hamiltonian_operator(self):
        def matvec(vector):
            qvector = self._project_q(np.asarray(vector, dtype=np.complex128))
            return self._project_q(self._hamiltonian_matmat(qvector))

        def matmat(vectors):
            qvectors = self._project_q(np.asarray(vectors, dtype=np.complex128))
            return self._project_q(self._hamiltonian_matmat(qvectors))

        return sla.LinearOperator(
            (self.ntransition, self.ntransition),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            dtype=np.complex128,
        )

    def hamiltonian_operator(self):
        """Return the pole-subtracted target-sector Hamiltonian operator."""

        return self._projected_hamiltonian_operator()

    def _projected_trace(self):
        diagonal = getattr(self.source_operator, "diagonal", None)
        if callable(diagonal):
            diagonal = diagonal()
        if diagonal is None:
            return None
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (self.ntransition,):
            return None
        return complex(np.sum(diagonal) - np.trace(self.excluded_hamiltonian))

    def _preconditioner(self, z):
        if not self.use_diagonal_preconditioner:
            return None
        diagonal = getattr(self.source_operator, "diagonal", None)
        if callable(diagonal):
            diagonal = diagonal()
        if diagonal is None:
            return None
        diagonal = np.asarray(diagonal, dtype=np.complex128)
        if diagonal.shape != (self.ntransition,):
            return None
        denominator = z - diagonal
        floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(denominator))))
        denominator = np.where(
            np.abs(denominator) < floor,
            denominator + 1.0j * floor,
            denominator,
        )

        def matvec(vector):
            pvector = self._project_p(vector)
            qvector = self._project_q(vector)
            return pvector + self._project_q(qvector / denominator)

        return sla.LinearOperator(
            (self.ntransition, self.ntransition),
            matvec=matvec,
            dtype=np.complex128,
        )

    def solve_continuum(self, energy, right, *, eta=1.0e-3):
        r"""Apply :math:`[E+i\eta-QHQ]^{-1}` to continuum vectors."""

        energy = float(energy)
        eta = _positive_finite(eta, "eta")
        if not np.isfinite(energy):
            raise ValueError("energy must be finite")
        right = np.asarray(right, dtype=np.complex128)
        vector_input = right.ndim == 1
        if vector_input:
            right = right[:, None]
        if right.ndim != 2 or right.shape[0] != self.ntransition:
            raise ValueError("right must have shape (ntransition,) or (ntransition, nrhs)")
        right = self._project_q(right)
        z = energy + 1.0j * eta
        shifted = self._shifted_operator(z)
        preconditioner = self._preconditioner(z)
        solutions = np.zeros_like(right)
        iterations = []
        residual_norms = []
        for column in range(right.shape[1]):
            rhs = right[:, column]
            if np.linalg.norm(rhs) == 0.0:
                iterations.append(0)
                residual_norms.append(0.0)
                continue
            history = []
            solution, info = sla.gmres(
                shifted,
                rhs,
                rtol=self.solver_tol,
                atol=0.0,
                maxiter=self.maxiter,
                M=preconditioner,
                callback=history.append,
                callback_type="pr_norm",
            )
            if info != 0:
                raise RuntimeError(
                    "Projected TDA continuum solve did not converge; "
                    f"GMRES info={info}."
                )
            solution = self._project_q(solution)
            solutions[:, column] = solution
            iterations.append(len(history))
            residual_norms.append(
                float(np.linalg.norm(shifted.matvec(solution) - rhs))
            )
        self.last_solve_info = {
            "energy": energy,
            "eta": eta,
            "iterations": tuple(iterations),
            "residual_norms": tuple(residual_norms),
            "converged": True,
        }
        return solutions[:, 0] if vector_input else solutions

    def self_energy(self, energy, *, eta=1.0e-3):
        """Return the pole-free retarded self-energy in the active space."""

        right = self.coupling.conj().T
        solution = self.solve_continuum(energy, right, eta=eta)
        return self.coupling @ solution

    def self_energy_operator(self, energy, *, eta=1.0e-3):
        """Return a matrix-free active-space self-energy operator."""

        def matvec(vector):
            vector = np.asarray(vector, dtype=np.complex128)
            if vector.shape != (self.nactive,):
                raise ValueError(
                    f"active vector must have shape ({self.nactive},)"
                )
            right = self.coupling.conj().T @ vector
            return self.coupling @ self.solve_continuum(
                energy,
                right,
                eta=eta,
            )

        return sla.LinearOperator(
            (self.nactive, self.nactive),
            matvec=matvec,
            dtype=np.complex128,
        )

    def hybridization(self, energy, *, eta=1.0e-3):
        r"""Return :math:`\Gamma(E)=i[\Sigma^R(E)-\Sigma^A(E)]`."""

        sigma = self.self_energy(energy, eta=eta)
        return 1.0j * (sigma - sigma.conj().T)

    def memory_kernel(self, times):
        r"""Return :math:`K(t)=VQe^{-iQHQt}QV^\dagger` for ``t >= 0``.

        The action is evaluated with :func:`scipy.sparse.linalg.expm_multiply`.
        This is an exact finite-operator reference path; long-time production
        propagation should compress the resulting kernel.
        """

        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or np.any(times < 0.0) or not np.all(
            np.isfinite(times)
        ):
            raise ValueError("times must be a finite nonnegative 1D array")
        hamiltonian = self._projected_hamiltonian_operator()
        trace_hamiltonian = self._projected_trace()
        right = self.coupling.conj().T
        kernel = np.empty(
            (times.size, self.nactive, self.nactive),
            dtype=np.complex128,
        )
        for index, time in enumerate(times):
            if time == 0.0:
                propagated = right
            else:
                generator = (-1.0j * time) * hamiltonian
                trace_generator = (
                    None
                    if trace_hamiltonian is None
                    else -1.0j * time * trace_hamiltonian
                )
                propagated = sla.expm_multiply(
                    generator,
                    right,
                    traceA=trace_generator,
                )
            kernel[index] = self.coupling @ propagated
        return kernel


@dataclass(frozen=True)
class ExcitonPhononChannel:
    """One phonon-assisted target-momentum continuum channel.

    Use :meth:`from_coupling` to connect an
    :class:`ExcitonPhononCoupling` to the pole-subtracted transition space of
    its target finite-momentum TDA operator.
    """

    continuum: object
    frequency: float
    occupation: float = 0.0
    phonon_q_index: int | None = None
    branch: int | None = None

    def __post_init__(self):
        required = ("self_energy", "nactive", "ncontinuum")
        if any(not hasattr(self.continuum, name) for name in required):
            raise TypeError("continuum does not provide the embedding interface")
        object.__setattr__(
            self,
            "frequency",
            _positive_finite(self.frequency, "frequency"),
        )
        occupation = float(self.occupation)
        if not np.isfinite(occupation) or occupation < 0.0:
            raise ValueError("occupation must be nonnegative and finite")
        object.__setattr__(self, "occupation", occupation)

    @classmethod
    def from_coupling(
        cls,
        coupling,
        target_operator,
        source_vectors,
        *,
        excluded_vectors=None,
        occupation=0.0,
        **continuum_options,
    ):
        r"""Construct a channel from :math:`M_{q\nu}` and BSE vectors.

        ``source_vectors`` are retained excitons in the source momentum
        block. ``excluded_vectors`` are retained excitons in the target block
        and are removed from its continuum projector. Passing no excluded
        target vectors retains the entire target transition space.
        """

        if not isinstance(coupling, ExcitonPhononCoupling):
            raise TypeError("coupling must be an ExcitonPhononCoupling")
        target_q_index = getattr(target_operator, "q_index", None)
        if (
            target_q_index is not None
            and int(target_q_index) != coupling.target_q_index
        ):
            raise ValueError(
                "target TDA operator does not match the coupling target momentum"
            )
        target = (
            target_operator.aslinearoperator()
            if hasattr(target_operator, "aslinearoperator")
            else sla.aslinearoperator(target_operator)
        )
        if target.shape[0] != coupling.derivative.shape[0]:
            raise ValueError(
                "target TDA operator and coupling derivative dimensions differ"
            )
        if excluded_vectors is None:
            excluded_vectors = np.zeros((target.shape[0], 0), dtype=np.complex128)
        active_to_target = coupling.active_to_target(source_vectors)
        if hasattr(target_operator, "projected_continuum"):
            continuum = target_operator.projected_continuum(
                active_to_target,
                excluded_vectors=excluded_vectors,
                **continuum_options,
            )
        else:
            continuum = ProjectedTDAContinuum(
                target_operator,
                excluded_vectors,
                active_to_target,
                **continuum_options,
            )
        return cls(
            continuum=continuum,
            frequency=coupling.frequency,
            occupation=occupation,
            phonon_q_index=coupling.phonon_q_index,
            branch=coupling.branch,
        )

    @classmethod
    def thermal_from_coupling(
        cls,
        coupling,
        target_operator,
        source_vectors,
        *,
        temperature,
        excluded_vectors=None,
        **continuum_options,
    ):
        r"""Construct a channel with its Bose occupation evaluated at ``T``.

        The one-phonon Fan weights are :math:`N_{q\nu}+1` for emission and
        :math:`N_{q\nu}` for absorption.  ``temperature`` is in kelvin.
        """

        return cls.from_coupling(
            coupling,
            target_operator,
            source_vectors,
            excluded_vectors=excluded_vectors,
            occupation=bose_occupation(coupling.frequency, temperature),
            **continuum_options,
        )


class ExcitonPhononContinuum:
    r"""Second-order phonon-assisted sum of projected TDA continua.

    Each channel is assumed to contain the quantized coupling
    :math:`g_{S\lambda}^{q\nu}`. The retarded self-energy is

    .. math::

       \Sigma^R(E)=\sum_{q\nu}\left[
       (N_{q\nu}+1)\Sigma_{q\nu}(E-\omega_{q\nu})
       +N_{q\nu}\Sigma_{q\nu}(E+\omega_{q\nu})\right].

    This is the one-phonon Fan contribution adapted from the first-principles
    exciton-phonon formulation of H.-Y. Chen, D. Sangalli, and M. Bernardi,
    Phys. Rev. Lett. 125, 107401 (2020),
    DOI: 10.1103/PhysRevLett.125.107401. It omits Debye-Waller terms,
    multiphonon processes, and self-consistent vertex corrections.
    """

    def __init__(self, channels):
        channels = tuple(channels)
        if not channels:
            raise ValueError("at least one exciton-phonon channel is required")
        channels = tuple(
            channel
            if isinstance(channel, ExcitonPhononChannel)
            else ExcitonPhononChannel(*channel)
            for channel in channels
        )
        nactive = int(channels[0].continuum.nactive)
        if any(int(channel.continuum.nactive) != nactive for channel in channels):
            raise ValueError("all channels must share one active exciton dimension")
        self.channels = channels
        self.nactive = nactive
        self.ncontinuum = int(
            sum(int(channel.continuum.ncontinuum) for channel in channels)
        )
        self.last_channel_self_energies = None
        self.times = None
        self.active_states = None
        self.active_populations = None
        self.channel_populations = None
        self.continuum_population = None
        self.total_norm = None
        self.success = False
        self.message = "not run"

    def _auxiliary_sectors(self):
        sectors = []
        for channel_index, channel in enumerate(self.channels):
            sectors.append(
                (
                    channel_index,
                    "emission",
                    channel,
                    channel.frequency,
                    np.sqrt(channel.occupation + 1.0),
                )
            )
            if channel.occupation > 0.0:
                sectors.append(
                    (
                        channel_index,
                        "absorption",
                        channel,
                        -channel.frequency,
                        np.sqrt(channel.occupation),
                    )
                )
        return tuple(sectors)

    def self_energy(self, energy, *, eta=1.0e-3):
        """Return the summed emission and absorption self-energy."""

        energy = float(energy)
        eta = _positive_finite(eta, "eta")
        if not np.isfinite(energy):
            raise ValueError("energy must be finite")
        total = np.zeros((self.nactive, self.nactive), dtype=np.complex128)
        details = []
        for channel in self.channels:
            emission = (channel.occupation + 1.0) * channel.continuum.self_energy(
                energy - channel.frequency,
                eta=eta,
            )
            absorption = channel.occupation * channel.continuum.self_energy(
                energy + channel.frequency,
                eta=eta,
            )
            total += emission + absorption
            details.append((emission, absorption))
        self.last_channel_self_energies = tuple(details)
        return total

    def self_energy_operator(self, energy, *, eta=1.0e-3):
        """Return the active-space self-energy as a linear operator."""

        return sla.aslinearoperator(self.self_energy(energy, eta=eta))

    def hybridization(self, energy, *, eta=1.0e-3):
        r"""Return :math:`\Gamma(E)=i[\Sigma^R(E)-\Sigma^A(E)]`."""

        sigma = self.self_energy(energy, eta=eta)
        return 1.0j * (sigma - sigma.conj().T)

    def memory_kernel(self, times):
        """Return the thermal one-phonon continuum memory kernel."""

        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or np.any(times < 0.0) or not np.all(
            np.isfinite(times)
        ):
            raise ValueError("times must be a finite nonnegative 1D array")
        total = np.zeros(
            (times.size, self.nactive, self.nactive),
            dtype=np.complex128,
        )
        for channel in self.channels:
            electronic = channel.continuum.memory_kernel(times)
            phonon = (
                (channel.occupation + 1.0)
                * np.exp(-1.0j * channel.frequency * times)
                + channel.occupation
                * np.exp(1.0j * channel.frequency * times)
            )
            total += phonon[:, None, None] * electronic
        return total

    def feshbach_embedding(self, active_hamiltonian):
        """Return a :class:`pyqed.ldr.FeshbachEmbedding` for these channels."""

        from pyqed.ldr import FeshbachEmbedding

        return FeshbachEmbedding(active_hamiltonian, self)

    def run_spectrum(
        self,
        active_hamiltonian,
        energies,
        *,
        eta=1.0e-3,
        probe=None,
        store_matrices=False,
    ):
        r"""Build and evaluate the finite-temperature Feshbach spectrum.

        This convenience route retains the same one-phonon Fan approximation
        as :meth:`self_energy`; it does not add Debye--Waller or multiphonon
        terms.
        """

        embedding = self.feshbach_embedding(active_hamiltonian)
        embedding.run_spectrum(
            energies,
            eta=eta,
            probe=probe,
            store_matrices=store_matrices,
        )
        self.embedding = embedding
        return embedding

    def augmented_hamiltonian(self, active_hamiltonian):
        """Return the exact finite one-phonon auxiliary Hamiltonian."""

        active_hamiltonian = np.asarray(active_hamiltonian, dtype=np.complex128)
        if active_hamiltonian.shape != (self.nactive, self.nactive):
            raise ValueError("active_hamiltonian has the wrong shape")
        if np.linalg.norm(active_hamiltonian - active_hamiltonian.conj().T) > 1.0e-10:
            raise ValueError("active_hamiltonian must be Hermitian")
        sectors = self._auxiliary_sectors()
        offsets = [self.nactive]
        for _index, _kind, channel, _shift, _scale in sectors:
            offsets.append(offsets[-1] + channel.continuum.ntransition)
        dimension = offsets[-1]

        def matvec(vector):
            vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
            if vector.size != dimension:
                raise ValueError(f"state must have shape ({dimension},)")
            active = vector[: self.nactive]
            result = np.zeros_like(vector)
            result[: self.nactive] = active_hamiltonian @ active
            for sector_index, sector in enumerate(sectors):
                _channel_index, _kind, channel, shift, scale = sector
                start, stop = offsets[sector_index : sector_index + 2]
                continuum = channel.continuum
                target = continuum._project_q(vector[start:stop])
                coupling = scale * continuum.coupling
                result[: self.nactive] += coupling @ target
                result[start:stop] = continuum._project_q(
                    continuum.hamiltonian_operator().matvec(target)
                    + shift * target
                    + coupling.conj().T @ active
                )
            return result

        def matmat(vectors):
            vectors = np.asarray(vectors, dtype=np.complex128)
            if vectors.ndim != 2 or vectors.shape[0] != dimension:
                raise ValueError(
                    f"states must have shape ({dimension}, nvector)"
                )
            return np.column_stack(
                [matvec(vectors[:, column]) for column in range(vectors.shape[1])]
            )

        operator = sla.LinearOperator(
            (dimension, dimension),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            dtype=np.complex128,
        )
        operator.active_dimension = self.nactive
        operator.sectors = sectors
        operator.offsets = tuple(offsets)
        operator.active_hamiltonian = active_hamiltonian
        trace = complex(np.trace(active_hamiltonian))
        for _index, _kind, channel, shift, _scale in sectors:
            continuum = channel.continuum
            continuum_trace = continuum._projected_trace()
            if continuum_trace is None:
                trace = None
                break
            trace += continuum_trace + shift * continuum.ncontinuum
        operator.trace = trace
        return operator

    def run_dynamics(self, active_hamiltonian, initial_state, times):
        """Propagate exact finite one-phonon auxiliary-space dynamics."""

        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or abs(times[0]) > 1.0e-14
        ):
            raise ValueError("times must start at zero and increase strictly")
        initial_state = np.asarray(initial_state, dtype=np.complex128)
        if initial_state.shape != (self.nactive,):
            raise ValueError(
                f"initial_state must have shape ({self.nactive},)"
            )
        norm = float(np.linalg.norm(initial_state))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("initial_state must have finite nonzero norm")
        initial_state = initial_state / norm
        hamiltonian = self.augmented_hamiltonian(active_hamiltonian)
        complete_initial = np.zeros(hamiltonian.shape[0], dtype=np.complex128)
        complete_initial[: self.nactive] = initial_state
        generator = (-1.0j) * hamiltonian
        trace_generator = (
            None if hamiltonian.trace is None else -1.0j * hamiltonian.trace
        )
        steps = np.diff(times)
        if np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-14):
            states = sla.expm_multiply(
                generator,
                complete_initial,
                start=0.0,
                stop=times[-1],
                num=times.size,
                endpoint=True,
                traceA=trace_generator,
            )
        else:
            states = np.empty(
                (times.size, hamiltonian.shape[0]),
                dtype=np.complex128,
            )
            states[0] = complete_initial
            for index, time in enumerate(times[1:], start=1):
                states[index] = sla.expm_multiply(
                    time * generator,
                    complete_initial,
                    traceA=(
                        None
                        if trace_generator is None
                        else time * trace_generator
                    ),
                )
        active_states = states[:, : self.nactive]
        channel_populations = np.empty(
            (times.size, len(hamiltonian.sectors)),
            dtype=float,
        )
        for sector_index, sector in enumerate(hamiltonian.sectors):
            start, stop = hamiltonian.offsets[sector_index : sector_index + 2]
            continuum = sector[2].continuum
            projected = np.column_stack(
                [continuum._project_q(state[start:stop]) for state in states]
            ).T
            channel_populations[:, sector_index] = np.sum(
                np.abs(projected) ** 2,
                axis=1,
            )
        active_populations = np.abs(active_states) ** 2
        continuum_population = np.sum(channel_populations, axis=1)
        self.times = times
        self.active_states = active_states
        self.active_populations = active_populations
        self.channel_populations = channel_populations
        self.continuum_population = continuum_population
        self.total_norm = np.sum(active_populations, axis=1) + continuum_population
        self.success = True
        self.message = "propagated the finite one-phonon embedded dynamics"
        return self


__all__ = [
    "ExcitonPhononChannel",
    "ExcitonPhononContinuum",
    "ExcitonPhononCoupling",
    "ProjectedTDAContinuum",
    "TotalMomentumSector",
    "bose_occupation",
]
