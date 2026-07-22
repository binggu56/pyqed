"""Continuum fixed-density and few-pair benchmarks for p+ip pairing.

The model is the two-particle sector of a separable two-dimensional p-wave
pairing Hamiltonian.  Radial kinetic energy is the outer ordered coordinate;
the angular dependence is the exact antipodal ``p+ip`` pair tie

    P^dagger(E) = integral dtheta/(2 pi) exp(i theta)
                  c^dagger(E, theta) c^dagger(E, theta + pi).

The thermodynamic calculation below keeps the microscopic pair orbitals
hard-core and fixes fermion density directly through energy integrals.  The
few-pair classes that follow it are separate dilute cLETTA prototypes and are
retained for comparisons of virtual and memory constructions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar

from .cletta import cletta_memory_matrices


@dataclass
class ContinuumPipPairingModel:
    r"""Separable continuum ``p+ip`` pairing model in the one-pair sector.

    Energies lie in ``[0, energy_cutoff]`` and the two-dimensional density of
    radial states is constant.  The dimensionless radial p-wave form factor is

    $$
    f(E)=\sqrt{E/E_c}.
    $$

    With an attractive interaction ``-G P^dagger P``, a bound state of energy
    ``-B`` satisfies

    $$
    \frac{1}{G}=\rho\int_0^{E_c}
    \frac{f(E)^2}{2E+B}\,dE.
    $$

    The hard energy cutoff keeps this first benchmark finite and makes the
    integral analytic.  A bound state exists for ``G * density_of_states > 2``.
    """

    coupling: float = 3.0
    energy_cutoff: float = 1.0
    density_of_states: float = 1.0

    def __post_init__(self):
        self.coupling = float(self.coupling)
        self.energy_cutoff = float(self.energy_cutoff)
        self.density_of_states = float(self.density_of_states)
        if self.coupling <= 0.0:
            raise ValueError("coupling must be positive.")
        if self.energy_cutoff <= 0.0:
            raise ValueError("energy_cutoff must be positive.")
        if self.density_of_states <= 0.0:
            raise ValueError("density_of_states must be positive.")

    @property
    def dimensionless_coupling(self) -> float:
        return self.coupling * self.density_of_states

    @property
    def critical_coupling(self) -> float:
        return 2.0 / self.density_of_states

    @property
    def maximum_fermion_density(self) -> float:
        r"""Return the density when every hard-core pair orbital is filled.

        ``density_of_states`` counts independent ``(k, -k)`` pair orbitals
        per area and per energy.  Each occupied orbital contains two
        fermions, hence ``n_max = 2 rho E_c``.
        """
        return 2.0 * self.density_of_states * self.energy_cutoff

    def form_factor(self, energy):
        """Return the radial p-wave form factor ``sqrt(E / E_c)``."""
        energy = np.asarray(energy, dtype=float)
        if np.any((energy < 0.0) | (energy > self.energy_cutoff)):
            raise ValueError("energy must lie inside the model cutoff.")
        values = np.sqrt(energy / self.energy_cutoff)
        return float(values) if values.ndim == 0 else values

    def pair_susceptibility(self, binding_energy: float) -> float:
        r"""Return ``rho integral f(E)^2 / (2 E + B) dE`` analytically."""
        binding_energy = float(binding_energy)
        if binding_energy <= 0.0:
            raise ValueError("binding_energy must be positive.")
        cutoff = self.energy_cutoff
        ratio = 2.0 * cutoff / binding_energy
        integral = 0.5 - binding_energy * np.log1p(ratio) / (4.0 * cutoff)
        return self.density_of_states * integral

    def binding_energy(self) -> float:
        """Return the positive binding energy ``B`` of the exact pair."""
        if self.coupling <= self.critical_coupling:
            raise ValueError(
                "this cutoff p-wave model binds only when "
                "coupling * density_of_states > 2."
            )

        target = 1.0 / self.coupling

        def equation(binding):
            return self.pair_susceptibility(binding) - target

        lower = np.finfo(float).tiny * self.energy_cutoff
        upper = self.energy_cutoff
        while equation(upper) > 0.0:
            upper *= 2.0
        return float(brentq(equation, lower, upper, xtol=1.0e-14, rtol=1.0e-13))


@dataclass
class ThermodynamicPipBCS:
    r"""Direct fixed-density continuum calculation for hard-core pairs.

    This is the thermodynamic hard-core Gaussian (BCS) state

    $$
    |\Psi\rangle=\prod_{\mathbf k\in\mathcal K_+}
    \left[u(E_{\mathbf k})+
    v(E_{\mathbf k})e^{i\theta_{\mathbf k}}S^+_{\mathbf k}\right]|0\rangle,
    $$

    where ``K_+`` contains one member of every antipodal momentum pair and
    ``(S_k^+)^2 = 0``.  The continuum measure is

    $$
    \frac{1}{A}\sum_{\mathbf k\in\mathcal K_+}
    \longrightarrow \rho\int_0^{E_c}dE
    \int\frac{d\theta}{2\pi}.
    $$

    Thus the calculation has no finite number of physical orbitals.  The
    angular integral is exact because only the single chiral harmonic
    ``exp(i theta)`` occurs, and adaptive quadrature is used only for the
    radial continuum integral.

    The chemical potential is varied together with the gap so the requested
    fermion density, rather than a particle number in a finite box, is fixed.
    Number projection changes no intensive observable in the thermodynamic
    limit.  This class is the ``D=1`` hard-core Gaussian reference for a later
    interacting cLETTA calculation; it does not claim a nontrivial virtual or
    memory bond.
    """

    model: ContinuumPipPairingModel
    fermion_density: float
    chemical_potential: float
    gap: float
    energy_density: float
    kinetic_energy_density: float
    interaction_energy_density: float
    success: bool = True
    message: str = "continuum gap and number equations converged"

    bond_dim: int = 1
    num_memory_modes: int = 0

    @classmethod
    def solve(
        cls,
        model: ContinuumPipPairingModel,
        fermion_density: float,
        *,
        initial_guess=None,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
    ) -> "ThermodynamicPipBCS":
        """Solve the radial gap and number integrals at fixed density."""
        density = float(fermion_density)
        maximum_density = model.maximum_fermion_density
        if not (0.0 < density < maximum_density):
            raise ValueError(
                "fermion_density must lie strictly between zero and "
                f"{maximum_density:g}."
            )

        cutoff = model.energy_cutoff
        rho = model.density_of_states
        normal_fermi_energy = density / (2.0 * rho)

        def integrate(function, chemical_potential):
            points = None
            if 0.0 < chemical_potential < cutoff:
                points = [chemical_potential]
            value, _ = quad(
                function,
                0.0,
                cutoff,
                points=points,
                epsabs=float(epsabs),
                epsrel=float(epsrel),
                limit=400,
            )
            return float(value)

        def observables(parameters):
            chemical_potential = float(parameters[0])
            gap = float(np.exp(parameters[1]))

            def quasiparticle_energy(energy):
                return np.sqrt(
                    (energy - chemical_potential) ** 2
                    + gap**2 * model.form_factor(energy) ** 2
                )

            gap_integral = integrate(
                lambda energy: (
                    model.form_factor(energy) ** 2
                    / (2.0 * quasiparticle_energy(energy))
                ),
                chemical_potential,
            )
            calculated_density = rho * integrate(
                lambda energy: (
                    1.0
                    - (energy - chemical_potential)
                    / quasiparticle_energy(energy)
                ),
                chemical_potential,
            )
            return gap_integral, calculated_density

        def residual(parameters):
            gap_integral, calculated_density = observables(parameters)
            return np.array(
                [
                    model.coupling * rho * gap_integral - 1.0,
                    (calculated_density - density) / maximum_density,
                ]
            )

        if initial_guess is None:
            starts = (
                (normal_fermi_energy, 0.1 * cutoff),
                (normal_fermi_energy, 0.5 * cutoff),
                (0.0, cutoff),
                (-0.5 * cutoff, cutoff),
            )
        else:
            chemical_potential, gap = map(float, initial_guess)
            if gap <= 0.0:
                raise ValueError("the initial gap must be positive.")
            starts = ((chemical_potential, gap),)

        lower = np.array([-100.0 * cutoff, np.log(1.0e-13 * cutoff)])
        upper = np.array([101.0 * cutoff, np.log(100.0 * cutoff)])
        best = None
        for chemical_potential, gap in starts:
            result = least_squares(
                residual,
                np.array([chemical_potential, np.log(gap)]),
                bounds=(lower, upper),
                xtol=1.0e-13,
                ftol=1.0e-13,
                gtol=1.0e-13,
                max_nfev=1000,
            )
            error = float(np.linalg.norm(residual(result.x), ord=np.inf))
            if best is None or error < best[0]:
                best = (error, result)

        error, result = best
        if not result.success or error > 2.0e-8:
            raise RuntimeError(
                "fixed-density continuum solve did not converge: "
                f"maximum residual {error:.3e}; {result.message}"
            )

        chemical_potential = float(result.x[0])
        gap = float(np.exp(result.x[1]))

        def quasiparticle_energy(energy):
            return np.sqrt(
                (energy - chemical_potential) ** 2
                + gap**2 * model.form_factor(energy) ** 2
            )

        kinetic = 2.0 * rho * integrate(
            lambda energy: (
                0.5
                * energy
                * (
                    1.0
                    - (energy - chemical_potential)
                    / quasiparticle_energy(energy)
                )
            ),
            chemical_potential,
        )
        interaction = -gap**2 / model.coupling
        return cls(
            model=model,
            fermion_density=density,
            chemical_potential=chemical_potential,
            gap=gap,
            energy_density=float(kinetic + interaction),
            kinetic_energy_density=float(kinetic),
            interaction_energy_density=float(interaction),
        )

    @property
    def pair_filling(self) -> float:
        """Return occupied-pair fraction ``n_f / (2 rho E_c)``."""
        return self.fermion_density / self.model.maximum_fermion_density

    @property
    def phase(self) -> str:
        """Classify the Gaussian state by the sign of the chemical potential."""
        tolerance = 1.0e-10 * self.model.energy_cutoff
        if self.chemical_potential > tolerance:
            return "weak pairing"
        if self.chemical_potential < -tolerance:
            return "strong pairing"
        return "Read-Green line"

    def quasiparticle_energy(self, energy):
        """Return the positive Bogoliubov dispersion in the radial continuum."""
        energy = np.asarray(energy, dtype=float)
        values = np.sqrt(
            (energy - self.chemical_potential) ** 2
            + self.gap**2 * self.model.form_factor(energy) ** 2
        )
        return float(values) if values.ndim == 0 else values

    def pair_occupation(self, energy):
        r"""Return ``v(E)^2``, the hard-core pair occupation probability."""
        energy = np.asarray(energy, dtype=float)
        values = 0.5 * (
            1.0
            - (energy - self.chemical_potential)
            / self.quasiparticle_energy(energy)
        )
        return float(values) if values.ndim == 0 else values

    def empty_occupation(self, energy):
        """Return ``u(E)^2 = 1 - v(E)^2``."""
        values = 1.0 - self.pair_occupation(energy)
        return float(values) if np.ndim(values) == 0 else values

    def anomalous_amplitude(self, energy, theta=0.0):
        r"""Return ``u(E) v(E) exp(i theta)`` for the chiral pair tie."""
        energy = np.asarray(energy, dtype=float)
        theta = np.asarray(theta, dtype=float)
        values = (
            self.gap
            * self.model.form_factor(energy)
            / (2.0 * self.quasiparticle_energy(energy))
            * np.exp(1.0j * theta)
        )
        return complex(values) if values.ndim == 0 else values

    def local_hard_core_amplitudes(self, energy, theta=0.0):
        r"""Return local amplitudes ``(u, v exp(i theta))``.

        The two entries correspond to empty and occupied hard-core pair
        states.  There is no local double-pair state.
        """
        empty = np.sqrt(self.empty_occupation(energy))
        occupied = np.sqrt(self.pair_occupation(energy)) * np.exp(
            1.0j * np.asarray(theta, dtype=float)
        )
        return np.stack(np.broadcast_arrays(empty, occupied), axis=-1)

    def integrated_fermion_density(self) -> float:
        """Recompute the fixed density by direct adaptive quadrature."""
        value, _ = quad(
            lambda energy: (
                2.0
                * self.model.density_of_states
                * self.pair_occupation(energy)
            ),
            0.0,
            self.model.energy_cutoff,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=400,
        )
        return float(value)

    def gap_equation_residual(self) -> float:
        """Return the direct-continuum gap-equation residual."""
        value, _ = quad(
            lambda energy: (
                self.model.form_factor(energy) ** 2
                / (2.0 * self.quasiparticle_energy(energy))
            ),
            0.0,
            self.model.energy_cutoff,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=400,
        )
        return float(
            self.model.coupling * self.model.density_of_states * value - 1.0
        )


@dataclass
class ThermodynamicPipCLETTA:
    r"""Hard-core frequency-ordered ``D=2, M=1`` continuum cLETTA.

    This is a continuous matrix-product *correlator* on the hard-core BCS
    reference, not a canonical composite-boson cMPS.  At each radial energy
    the physical shell has exactly two states, empty and occupied.  For a
    radial quadrature cell ``dE`` its tensors are

    $$
    A^0(E)=u(E)[I+dE\,Q_c]-\sqrt{dE}\,v(E)R_c,
    $$

    $$
    A^1(E)=v(E)[I+dE\,Q_c]+\sqrt{dE}\,u(E)R_c.
    $$

    Thus there is no double occupation at a shell point.  The centered form
    also has a finite continuum transfer generator because the terms linear
    in ``sqrt(dE)`` cancel between the two physical states.

    The structured cLETTA matrices are

    $$
    Q_c=I_m\otimes Q-\kappa N_m\otimes I_D,
    $$

    $$
    R_c=\sqrt{\kappa}\,a_m\otimes I_D+a_m^\dagger\otimes S,
    $$

    with a two-dimensional virtual space and one memory mode truncated to
    occupations zero and one.  ``quadrature_points`` are integration nodes,
    not physical orbitals.

    For the Kac-scaled reduced pairing Hamiltonian, only one-shell reduced
    density matrices enter the thermodynamic energy density.  Entangling
    different shells can only reduce their local pairing coherence.  The
    optimized finite-quadrature cLETTA therefore returns zero tie strength and
    exactly the BCS energy density.  Nonzero ties remain genuine, contractible
    fluctuation states, but their effect on the intensive energy vanishes as
    the radial integration is refined.  Finite-size Richardson corrections
    are subextensive and are intentionally not folded into this
    direct-continuum energy density.
    """

    reference: ThermodynamicPipBCS
    radial_decay: float
    tie_strength: float
    memory_decay: float
    fugacity_shift: float
    energy_density: float
    kinetic_energy_density: float
    interaction_energy_density: float
    pairing_amplitude_density: complex
    fermion_density: float
    norm: float
    quadrature_points: int
    success: bool = True
    message: str = "hard-core continuum cLETTA contraction converged"

    bond_dim: int = 2
    num_memory_modes: int = 1
    memory_depth: int = 1
    effective_bond_dim: int = 4

    @property
    def model(self) -> ContinuumPipPairingModel:
        return self.reference.model

    @classmethod
    def evaluate(
        cls,
        reference: ThermodynamicPipBCS,
        *,
        radial_decay: float,
        tie_strength: float,
        memory_decay: float,
        quadrature_points=128,
    ) -> "ThermodynamicPipCLETTA":
        """Contract one hard-core cLETTA trial state at fixed density."""
        radial_decay = float(radial_decay)
        tie_strength = float(tie_strength)
        memory_decay = float(memory_decay)
        quadrature_points = int(quadrature_points)
        if radial_decay < 0.0:
            raise ValueError("radial_decay must be non-negative.")
        if memory_decay <= 0.0:
            raise ValueError("memory_decay must be positive.")
        if quadrature_points < 8:
            raise ValueError("quadrature_points must be at least 8.")

        def density_residual(fugacity_shift):
            values = cls._contract(
                reference,
                radial_decay,
                tie_strength,
                memory_decay,
                float(fugacity_shift),
                quadrature_points,
            )
            return values[1] - reference.fermion_density

        lower = -20.0
        upper = 20.0
        lower_value = density_residual(lower)
        upper_value = density_residual(upper)
        if lower_value > 0.0 or upper_value < 0.0:
            raise RuntimeError("could not bracket the fixed-density fugacity shift.")
        fugacity_shift = float(
            brentq(
                density_residual,
                lower,
                upper,
                xtol=2.0e-12,
                rtol=2.0e-12,
            )
        )
        norm, density, kinetic, pairing = cls._contract(
            reference,
            radial_decay,
            tie_strength,
            memory_decay,
            fugacity_shift,
            quadrature_points,
        )
        interaction = -reference.model.coupling * abs(pairing) ** 2
        return cls(
            reference=reference,
            radial_decay=radial_decay,
            tie_strength=tie_strength,
            memory_decay=memory_decay,
            fugacity_shift=fugacity_shift,
            energy_density=float(kinetic + interaction),
            kinetic_energy_density=float(kinetic),
            interaction_energy_density=float(interaction),
            pairing_amplitude_density=complex(pairing),
            fermion_density=float(density),
            norm=float(norm),
            quadrature_points=quadrature_points,
        )

    @classmethod
    def optimize(
        cls,
        reference: ThermodynamicPipBCS,
        *,
        quadrature_points=48,
        validation_points=160,
    ) -> "ThermodynamicPipCLETTA":
        """Vary the virtual decay, tie, and memory decay at fixed density."""
        quadrature_points = int(quadrature_points)
        validation_points = int(validation_points)
        if quadrature_points < 8 or validation_points < quadrature_points:
            raise ValueError(
                "quadrature_points must be at least 8 and validation_points "
                "must not be smaller."
            )

        def objective(parameters):
            radial_decay, tie_strength, log_memory_decay = parameters
            try:
                state = cls.evaluate(
                    reference,
                    radial_decay=radial_decay,
                    tie_strength=tie_strength,
                    memory_decay=np.exp(log_memory_decay),
                    quadrature_points=quadrature_points,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                return 1.0e6
            return state.energy_density

        bounds = (
            (0.0, 20.0),
            (-5.0, 5.0),
            (np.log(1.0e-3), np.log(20.0)),
        )
        candidates = [(reference.energy_density, np.array([1.0, 0.0, 0.0]))]
        for start in (
            (1.0, 0.2, 0.0),
            (3.0, -0.5, np.log(0.5)),
            (8.0, 1.0, np.log(2.0)),
        ):
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1.0e-13, "gtol": 1.0e-9, "maxiter": 300},
            )
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.fun), np.asarray(result.x)))

        _, parameters = min(candidates, key=lambda item: item[0])
        if abs(parameters[1]) < 2.0e-6:
            parameters = np.array([1.0, 0.0, 0.0])
        state = cls.evaluate(
            reference,
            radial_decay=float(parameters[0]),
            tie_strength=float(parameters[1]),
            memory_decay=float(np.exp(parameters[2])),
            quadrature_points=validation_points,
        )
        if state.energy_density < reference.energy_density - 2.0e-8:
            raise FloatingPointError(
                "cLETTA fell below the exact thermodynamic BCS energy; "
                "increase quadrature resolution."
            )
        return state

    def base_matrices(self):
        """Return the ``D=2`` virtual matrices ``(Q, R, S)``."""
        cutoff = self.model.energy_cutoff
        q_matrix = np.diag([0.0, -self.radial_decay / cutoff]).astype(
            np.complex128
        )
        r_matrix = np.zeros((2, 2), dtype=np.complex128)
        tie_matrix = (
            self.tie_strength
            / np.sqrt(2.0 * cutoff)
            * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
        )
        return q_matrix, r_matrix, tie_matrix

    def combined_matrices(self):
        """Return the explicit ``4 x 4`` virtual-memory cLETTA matrices."""
        q_matrix, r_matrix, tie_matrix = self.base_matrices()
        return cletta_memory_matrices(
            q_matrix,
            r_matrix,
            tie_matrix,
            self.memory_decay / self.model.energy_cutoff,
            memory_dim=2,
        )

    def boundary_vectors(self):
        """Return memory-vacuum, virtual-vacuum boundaries."""
        right = np.zeros(self.effective_bond_dim, dtype=np.complex128)
        left = np.zeros(self.effective_bond_dim, dtype=np.complex128)
        right[0] = 1.0
        left[0] = 1.0
        return left, right

    def hard_core_tensors(
        self,
        energy,
        cell_width,
        *,
        theta=0.0,
        fugacity_shift=None,
    ):
        r"""Return the two physical shell tensors ``(A0, A1)``.

        ``A1`` carries the chiral factor ``exp(i theta)``.  Radial
        contractions use the co-rotating ``theta=0`` basis because the single
        angular harmonic is integrated analytically.
        """
        energy = float(energy)
        cell_width = float(cell_width)
        if not (0.0 <= energy <= self.model.energy_cutoff):
            raise ValueError("energy must lie inside the model cutoff.")
        if cell_width <= 0.0:
            raise ValueError("cell_width must be positive.")
        if fugacity_shift is None:
            fugacity_shift = self.fugacity_shift
        probability = self._shifted_pair_probability(
            self.reference.pair_occupation(energy),
            float(fugacity_shift),
        )
        empty = np.sqrt(1.0 - probability)
        occupied = np.sqrt(probability)
        q_matrix, r_matrix = self.combined_matrices()
        drift = np.eye(self.effective_bond_dim) + cell_width * q_matrix
        noise = np.sqrt(cell_width) * r_matrix
        return np.asarray(
            [
                empty * drift - occupied * noise,
                np.exp(1.0j * float(theta))
                * (occupied * drift + empty * noise),
            ]
        )

    @staticmethod
    def _shifted_pair_probability(probability, fugacity_shift):
        probability = np.asarray(probability, dtype=float)
        tiny = np.finfo(float).tiny
        probability = np.clip(probability, tiny, 1.0 - np.finfo(float).eps)
        log_odds = (
            np.log(probability)
            - np.log1p(-probability)
            + 2.0 * float(fugacity_shift)
        )
        shifted = np.empty_like(log_odds)
        positive = log_odds >= 0.0
        shifted[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
        exponential = np.exp(log_odds[~positive])
        shifted[~positive] = exponential / (1.0 + exponential)
        return float(shifted) if shifted.ndim == 0 else shifted

    @classmethod
    def _contract(
        cls,
        reference,
        radial_decay,
        tie_strength,
        memory_decay,
        fugacity_shift,
        points,
    ):
        prototype = cls(
            reference=reference,
            radial_decay=float(radial_decay),
            tie_strength=float(tie_strength),
            memory_decay=float(memory_decay),
            fugacity_shift=float(fugacity_shift),
            energy_density=np.nan,
            kinetic_energy_density=np.nan,
            interaction_energy_density=np.nan,
            pairing_amplitude_density=np.nan,
            fermion_density=np.nan,
            norm=np.nan,
            quadrature_points=int(points),
        )
        nodes, weights = leggauss(int(points))
        cutoff = reference.model.energy_cutoff
        energies = 0.5 * cutoff * (nodes + 1.0)
        widths = 0.5 * cutoff * weights
        left, right = prototype.boundary_vectors()
        environment = np.outer(right, right.conj())
        density_environment = np.zeros_like(environment)
        kinetic_environment = np.zeros_like(environment)
        pairing_environment = np.zeros_like(environment)
        identity = np.eye(2, dtype=np.complex128)
        occupation = np.array([[0.0, 0.0], [0.0, 1.0]])
        lowering = np.array([[0.0, 1.0], [0.0, 0.0]])

        def transfer(tensors, matrix, operator):
            result = np.zeros_like(matrix)
            for ket in range(2):
                for bra in range(2):
                    result += (
                        operator[bra, ket]
                        * tensors[ket]
                        @ matrix
                        @ tensors[bra].conj().T
                    )
            return result

        rho = reference.model.density_of_states
        for energy, width in zip(energies, widths):
            tensors = prototype.hard_core_tensors(
                energy,
                width,
                fugacity_shift=fugacity_shift,
            )
            previous = environment
            previous_density = density_environment
            previous_kinetic = kinetic_environment
            previous_pairing = pairing_environment
            environment = transfer(tensors, previous, identity)
            density_environment = transfer(
                tensors, previous_density, identity
            ) + transfer(tensors, previous, 2.0 * rho * width * occupation)
            kinetic_environment = transfer(
                tensors, previous_kinetic, identity
            ) + transfer(
                tensors,
                previous,
                2.0 * rho * width * energy * occupation,
            )
            pairing_environment = transfer(
                tensors, previous_pairing, identity
            ) + transfer(
                tensors,
                previous,
                rho
                * width
                * reference.model.form_factor(energy)
                * lowering,
            )

        def boundary_value(matrix):
            return np.vdot(left, matrix @ left)

        norm = float(np.real(boundary_value(environment)))
        if not np.isfinite(norm) or norm <= 1.0e-14:
            raise FloatingPointError("cLETTA norm is non-positive or non-finite.")
        density = float(np.real(boundary_value(density_environment)) / norm)
        kinetic = float(np.real(boundary_value(kinetic_environment)) / norm)
        pairing = complex(boundary_value(pairing_environment) / norm)
        return norm, density, kinetic, pairing


@dataclass
class ExactOnePairPipState:
    r"""Exact continuous ``p+ip`` pair used only as a reference.

    The outer energy-ordered insertion is

    $$
    R(E)=\phi(E)|1\rangle\langle0|,
    $$

    with right boundary ``|0>`` and left boundary ``<1|``.  Since ``R(E)`` is
    nilpotent, the ordered exponential contains exactly one physical pair
    insertion.  That physical insertion is the single antipodal angular tie
    ``P^dagger(E)``.  The energy-dependent insertion below can encode the
    exact wavefunction directly, so this class is not presented as a
    compressed cLETTA result.
    """

    model: ContinuumPipPairingModel
    binding_energy: float
    normalization: float

    bond_dim: int = 2
    @classmethod
    def from_model(cls, model: ContinuumPipPairingModel) -> "ExactOnePairPipState":
        """Construct the normalized exact one-pair state."""
        binding = model.binding_energy()

        def norm_integrand(energy):
            form = model.form_factor(energy)
            return (
                model.density_of_states
                * form
                * form
                / (2.0 * energy + binding) ** 2
            )

        norm_squared, _ = quad(
            norm_integrand,
            0.0,
            model.energy_cutoff,
            epsabs=1.0e-13,
            epsrel=1.0e-13,
            limit=300,
        )
        return cls(
            model=model,
            binding_energy=binding,
            normalization=1.0 / np.sqrt(norm_squared),
        )

    @property
    def energy(self) -> float:
        """Return the exact pair energy relative to the vacuum."""
        return -float(self.binding_energy)

    def radial_amplitude(self, energy):
        r"""Return ``phi(E) = N f(E) / (2 E + B)``."""
        energy_array = np.asarray(energy, dtype=float)
        values = (
            self.normalization
            * self.model.form_factor(energy_array)
            / (2.0 * energy_array + self.binding_energy)
        )
        return float(values) if values.ndim == 0 else values

    @staticmethod
    def angular_amplitude(theta):
        r"""Return the ``p+ip`` winding factor ``exp(i theta)``."""
        theta = np.asarray(theta, dtype=float)
        values = np.exp(1.0j * theta)
        return complex(values) if values.ndim == 0 else values

    def pair_wavefunction(self, energy, theta):
        """Return the separable radial-angular pair wavefunction."""
        return self.radial_amplitude(energy) * self.angular_amplitude(theta)

    def outer_matrices(self, energy):
        r"""Return an exact but unrestricted energy-dependent ``D=2`` form.

        ``R`` multiplies the composite angular pair insertion
        ``P^dagger(E)`` rather than a one-fermion creation operator.
        """
        q_matrix = np.zeros((2, 2), dtype=np.complex128)
        r_matrix = np.zeros((2, 2), dtype=np.complex128)
        r_matrix[1, 0] = self.radial_amplitude(energy)
        return q_matrix, r_matrix

    def norm(self) -> float:
        """Evaluate the continuum norm by adaptive quadrature."""
        value, _ = quad(
            lambda energy: self.model.density_of_states
            * abs(self.radial_amplitude(energy)) ** 2,
            0.0,
            self.model.energy_cutoff,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=300,
        )
        return float(value)

    def energy_expectation(self) -> float:
        r"""Evaluate ``<H>`` directly from kinetic and separable terms."""
        model = self.model

        kinetic, _ = quad(
            lambda energy: model.density_of_states
            * 2.0
            * energy
            * abs(self.radial_amplitude(energy)) ** 2,
            0.0,
            model.energy_cutoff,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=300,
        )
        pair_overlap, _ = quad(
            lambda energy: model.density_of_states
            * model.form_factor(energy)
            * self.radial_amplitude(energy),
            0.0,
            model.energy_cutoff,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=300,
        )
        return float(kinetic - model.coupling * abs(pair_overlap) ** 2)


@dataclass
class OneScalePipCLETTA:
    r"""Restricted continuous ``D=2, M=1`` variational pair state.

    A constant outer generator

    $$
    Q=\operatorname{diag}(-\lambda/E_c,0),\qquad
    R=\mathcal N|1\rangle\langle0|
    $$

    produces the radial amplitude

    $$
    \phi_\lambda(E)=\mathcal N f(E)e^{-\lambda E/E_c}.
    $$

    This is a genuine one-scale restriction: unlike
    :class:`ExactOnePairPipState`, it cannot insert an arbitrary function of
    energy into ``R(E)``.  The single tied channel is still the analytically
    contracted antipodal angular pair ``P^dagger(E)``.
    """

    model: ContinuumPipPairingModel
    decay_rate: float
    normalization: float
    energy: float

    bond_dim: int = 2
    num_tie_channels: int = 1
    num_memory_scales: int = 1

    @classmethod
    def optimize(
        cls,
        model: ContinuumPipPairingModel,
        *,
        rate_bounds=(0.0, 20.0),
    ) -> "OneScalePipCLETTA":
        """Variationally optimize the single dimensionless decay rate."""
        lower, upper = map(float, rate_bounds)
        if not (0.0 <= lower < upper):
            raise ValueError("rate_bounds must satisfy 0 <= lower < upper.")

        def normalization(rate):
            value, _ = quad(
                lambda energy: model.density_of_states
                * model.form_factor(energy) ** 2
                * np.exp(-2.0 * rate * energy / model.energy_cutoff),
                0.0,
                model.energy_cutoff,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=300,
            )
            return 1.0 / np.sqrt(value)

        def energy_for_rate(rate):
            norm = normalization(rate)

            def amplitude(energy):
                return (
                    norm
                    * model.form_factor(energy)
                    * np.exp(-rate * energy / model.energy_cutoff)
                )

            kinetic, _ = quad(
                lambda energy: model.density_of_states
                * 2.0
                * energy
                * amplitude(energy) ** 2,
                0.0,
                model.energy_cutoff,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=300,
            )
            overlap, _ = quad(
                lambda energy: model.density_of_states
                * model.form_factor(energy)
                * amplitude(energy),
                0.0,
                model.energy_cutoff,
                epsabs=1.0e-12,
                epsrel=1.0e-12,
                limit=300,
            )
            return float(kinetic - model.coupling * overlap**2)

        result = minimize_scalar(
            energy_for_rate,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        rate = float(result.x)
        return cls(
            model=model,
            decay_rate=rate,
            normalization=normalization(rate),
            energy=energy_for_rate(rate),
        )

    def radial_amplitude(self, energy):
        """Return the restricted one-scale radial amplitude."""
        energy_array = np.asarray(energy, dtype=float)
        values = (
            self.normalization
            * self.model.form_factor(energy_array)
            * np.exp(
                -self.decay_rate
                * energy_array
                / self.model.energy_cutoff
            )
        )
        return float(values) if values.ndim == 0 else values

    @staticmethod
    def angular_amplitude(theta):
        """Return the exact angular ``p+ip`` winding factor."""
        theta = np.asarray(theta, dtype=float)
        values = np.exp(1.0j * theta)
        return complex(values) if values.ndim == 0 else values

    def outer_matrices(self):
        r"""Return the constant energy-space matrices ``(Q, R)``."""
        q_matrix = np.diag(
            [-self.decay_rate / self.model.energy_cutoff, 0.0]
        ).astype(np.complex128)
        r_matrix = np.zeros((2, 2), dtype=np.complex128)
        r_matrix[1, 0] = self.normalization
        return q_matrix, r_matrix

    def norm(self) -> float:
        """Evaluate the state norm independently by quadrature."""
        value, _ = quad(
            lambda energy: self.model.density_of_states
            * abs(self.radial_amplitude(energy)) ** 2,
            0.0,
            self.model.energy_cutoff,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=300,
        )
        return float(value)


@dataclass
class TwoPairPipCLETTA:
    r"""Genuine ``D=2, M=1, L=1`` cLETTA for two composite p-wave pairs.

    The physical continuum field is the dilute composite pair field
    ``P^dagger(E)``.  The base field matrix vanishes and the cLETTA matrices
    are built exactly as

    $$
    Q_c=I\otimes Q-\kappa N\otimes I,
    $$

    $$
    R_c=\sqrt{\kappa}\,a\otimes I+a^\dagger\otimes G.
    $$

    Starting and ending in the memory vacuum forces one open and one close.
    The nilpotent ``D=2`` tie matrix ``G proportional |1><0|`` removes the
    vacuum and every sector with more than two pair-field insertions.  This is
    therefore the first prototype here in which the memory mode is explicit
    and dynamically propagated.

    The effective pair field is treated as canonical.  Pauli corrections
    between overlapping microscopic Cooper pairs are not included in this
    dilute-pair benchmark.
    """

    model: ContinuumPipPairingModel
    radial_decay: float
    memory_decay: float
    normalization: float
    energy: float
    quadrature_points: int

    bond_dim: int = 2
    num_memory_modes: int = 1
    memory_depth: int = 1

    @classmethod
    def optimize(
        cls,
        model: ContinuumPipPairingModel,
        *,
        quadrature_points=64,
        validation_points=160,
        radial_decay_bounds=(0.0, 20.0),
        memory_decay_bounds=(1.0e-5, 20.0),
    ) -> "TwoPairPipCLETTA":
        """Optimize the radial and explicit-memory decay scales."""
        quadrature_points = int(quadrature_points)
        validation_points = int(validation_points)
        if quadrature_points < 8 or validation_points < quadrature_points:
            raise ValueError(
                "quadrature_points must be at least 8 and validation_points "
                "must not be smaller."
            )
        radial_bounds = tuple(map(float, radial_decay_bounds))
        memory_bounds = tuple(map(float, memory_decay_bounds))
        if not (0.0 <= radial_bounds[0] < radial_bounds[1]):
            raise ValueError("radial_decay_bounds must satisfy 0 <= lower < upper.")
        if not (0.0 < memory_bounds[0] < memory_bounds[1]):
            raise ValueError("memory_decay_bounds must satisfy 0 < lower < upper.")

        def objective(parameters):
            radial_decay, memory_decay = parameters
            return cls._energy_for_parameters(
                model,
                radial_decay,
                memory_decay,
                quadrature_points,
            )[0]

        starts = (
            (1.0, 0.5),
            (2.0, 1.0),
            (4.0, 2.0),
        )
        best = None
        for start in starts:
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=(radial_bounds, memory_bounds),
                options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 400},
            )
            if best is None or result.fun < best.fun:
                best = result

        radial_decay, memory_decay = map(float, best.x)
        energy, normalization = cls._energy_for_parameters(
            model,
            radial_decay,
            memory_decay,
            validation_points,
        )
        return cls(
            model=model,
            radial_decay=radial_decay,
            memory_decay=memory_decay,
            normalization=normalization,
            energy=energy,
            quadrature_points=validation_points,
        )

    @staticmethod
    def _quadrature(model, points):
        nodes, weights = leggauss(int(points))
        energies = 0.5 * model.energy_cutoff * (nodes + 1.0)
        weights = (
            0.5
            * model.energy_cutoff
            * model.density_of_states
            * weights
        )
        return energies, weights

    @classmethod
    def _energy_for_parameters(
        cls,
        model,
        radial_decay,
        memory_decay,
        points,
    ):
        energies, weights = cls._quadrature(model, points)
        form = model.form_factor(energies)
        lower = np.minimum(energies[:, np.newaxis], energies[np.newaxis, :])
        upper = np.maximum(energies[:, np.newaxis], energies[np.newaxis, :])
        raw = (
            form[:, np.newaxis]
            * form[np.newaxis, :]
            * np.exp(
                -float(radial_decay) * lower / model.energy_cutoff
                -float(memory_decay) * (upper - lower) / model.energy_cutoff
            )
        )
        weighted = (
            np.sqrt(weights[:, np.newaxis] * weights[np.newaxis, :])
            * raw
            / np.sqrt(2.0)
        )
        raw_norm = float(np.linalg.norm(weighted))
        coefficient = weighted / raw_norm

        interaction_vector = np.sqrt(weights) * form
        one_pair_hamiltonian = np.diag(2.0 * energies)
        one_pair_hamiltonian -= model.coupling * np.outer(
            interaction_vector,
            interaction_vector,
        )
        energy = 2.0 * np.real(
            np.trace(
                coefficient.conj().T
                @ one_pair_hamiltonian
                @ coefficient
            )
        )
        return float(energy), 1.0 / raw_norm

    def ordered_amplitude(self, first_energy, second_energy):
        """Return the normalized amplitude for ``first_energy <= second_energy``."""
        first = np.asarray(first_energy, dtype=float)
        second = np.asarray(second_energy, dtype=float)
        if np.any(first > second):
            raise ValueError("ordered_amplitude requires first_energy <= second_energy.")
        return (
            self.normalization
            * self.model.form_factor(first)
            * self.model.form_factor(second)
            * np.exp(
                -self.radial_decay * first / self.model.energy_cutoff
                -self.memory_decay
                * (second - first)
                / self.model.energy_cutoff
            )
        )

    def combined_matrices(self):
        """Return the explicit ``4 x 4`` finite-depth cLETTA matrices."""
        cutoff = self.model.energy_cutoff
        q_matrix = np.diag([-self.radial_decay / cutoff, 0.0])
        r_matrix = np.zeros((2, 2), dtype=float)
        tie_matrix = np.zeros((2, 2), dtype=float)
        kappa = self.memory_decay / cutoff
        tie_matrix[1, 0] = self.normalization / np.sqrt(kappa)
        return cletta_memory_matrices(
            q_matrix,
            r_matrix,
            tie_matrix,
            kappa,
            memory_dim=2,
        )

    def boundary_vectors(self):
        """Return combined memory-vacuum boundary vectors."""
        right = np.zeros(4, dtype=np.complex128)
        left = np.zeros(4, dtype=np.complex128)
        right[0] = 1.0  # |memory=0, outer=0>
        left[1] = 1.0  # <memory=0, outer=1|
        return left, right

    def contracted_ordered_amplitude(self, first_energy, second_energy):
        """Contract the explicit cLETTA matrices for two ordered insertions."""
        first = float(first_energy)
        second = float(second_energy)
        if not (0.0 <= first <= second <= self.model.energy_cutoff):
            raise ValueError("energies must satisfy 0 <= first <= second <= cutoff.")
        q_matrix, r_matrix = self.combined_matrices()
        left, right = self.boundary_vectors()
        coefficient = (
            left
            @ expm(q_matrix * (self.model.energy_cutoff - second))
            @ r_matrix
            @ expm(q_matrix * (second - first))
            @ r_matrix
            @ expm(q_matrix * first)
            @ right
        )
        return (
            self.model.form_factor(first)
            * self.model.form_factor(second)
            * coefficient
        )

    @property
    def exact_dilute_pair_energy(self):
        """Return the exact energy of two noninteracting composite bound pairs."""
        return -2.0 * self.model.binding_energy()


@dataclass
class TwoPairPipD3CLETTA:
    r"""Fixed-two-pair ``D=3, M=1, L=1`` cLETTA.

    The extra virtual state enlarges the post-opening charge sector without
    changing the physical pair number.  Its ordered amplitude is

    $$
    A(E_1,E_2)=\mathcal N f(E_1)f(E_2)e^{-\beta(E_2-E_1)/E_c}
    \left[c_1e^{-\alpha_1E_1/E_c}+c_2e^{-\alpha_2E_1/E_c}\right],
    $$

    for ``E_1 <= E_2``, with ``c_1=cos(theta)`` and
    ``c_2=sin(theta)``.  Thus ``D=3`` is compared with ``D=2`` in the same
    two-pair sector rather than being used as a particle-number ladder.
    """

    model: ContinuumPipPairingModel
    radial_decays: np.ndarray
    memory_decay: float
    mixing_angle: float
    normalization: float
    energy: float
    quadrature_points: int

    bond_dim: int = 3
    num_memory_modes: int = 1
    memory_depth: int = 1

    def __post_init__(self):
        self.radial_decays = np.asarray(self.radial_decays, dtype=float)
        if self.radial_decays.shape != (2,):
            raise ValueError("radial_decays must contain two values.")

    @classmethod
    def optimize(
        cls,
        model: ContinuumPipPairingModel,
        *,
        quadrature_points=64,
        validation_points=160,
    ) -> "TwoPairPipD3CLETTA":
        """Optimize two radial channels, their mixing, and one memory rate."""
        quadrature_points = int(quadrature_points)
        validation_points = int(validation_points)
        if quadrature_points < 8 or validation_points < quadrature_points:
            raise ValueError(
                "quadrature_points must be at least 8 and validation_points "
                "must not be smaller."
            )

        def objective(parameters):
            return cls._energy_for_parameters(
                model,
                parameters[:2],
                parameters[2],
                parameters[3],
                quadrature_points,
            )[0]

        starts = (
            (3.1, 3.1, 1.39, 0.0),
            (1.0, 5.0, 1.0, 0.7),
            (2.5, 10.0, 1.5, 0.9),
            (5.0, 15.0, 0.8, -0.5),
        )
        bounds = (
            (0.0, 30.0),
            (0.0, 30.0),
            (1.0e-5, 20.0),
            (-np.pi, np.pi),
        )
        best = None
        for start in starts:
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 600},
            )
            if best is None or result.fun < best.fun:
                best = result

        radial_decays = np.asarray(best.x[:2], dtype=float)
        memory_decay = float(best.x[2])
        mixing_angle = float(best.x[3])
        if radial_decays[0] > radial_decays[1]:
            radial_decays = radial_decays[::-1]
            first = np.cos(mixing_angle)
            second = np.sin(mixing_angle)
            mixing_angle = float(np.arctan2(first, second))
        energy, normalization = cls._energy_for_parameters(
            model,
            radial_decays,
            memory_decay,
            mixing_angle,
            validation_points,
        )
        return cls(
            model=model,
            radial_decays=radial_decays,
            memory_decay=memory_decay,
            mixing_angle=mixing_angle,
            normalization=normalization,
            energy=energy,
            quadrature_points=validation_points,
        )

    @classmethod
    def _energy_for_parameters(
        cls,
        model,
        radial_decays,
        memory_decay,
        mixing_angle,
        points,
    ):
        energies, weights = TwoPairPipCLETTA._quadrature(model, points)
        form = model.form_factor(energies)
        lower = np.minimum(energies[:, np.newaxis], energies[np.newaxis, :])
        upper = np.maximum(energies[:, np.newaxis], energies[np.newaxis, :])
        coefficients = np.array(
            [np.cos(mixing_angle), np.sin(mixing_angle)]
        )
        radial = sum(
            coefficient
            * np.exp(-decay * lower / model.energy_cutoff)
            for coefficient, decay in zip(coefficients, radial_decays)
        )
        raw = (
            form[:, np.newaxis]
            * form[np.newaxis, :]
            * np.exp(
                -float(memory_decay)
                * (upper - lower)
                / model.energy_cutoff
            )
            * radial
        )
        weighted = (
            np.sqrt(weights[:, np.newaxis] * weights[np.newaxis, :])
            * raw
            / np.sqrt(2.0)
        )
        raw_norm = float(np.linalg.norm(weighted))
        coefficient_matrix = weighted / raw_norm

        interaction_vector = np.sqrt(weights) * form
        one_pair_hamiltonian = np.diag(2.0 * energies)
        one_pair_hamiltonian -= model.coupling * np.outer(
            interaction_vector,
            interaction_vector,
        )
        energy = 2.0 * np.real(
            np.trace(
                coefficient_matrix.conj().T
                @ one_pair_hamiltonian
                @ coefficient_matrix
            )
        )
        return float(energy), 1.0 / raw_norm

    @property
    def mixing_coefficients(self):
        return np.array(
            [np.cos(self.mixing_angle), np.sin(self.mixing_angle)]
        )

    def ordered_amplitude(self, first_energy, second_energy):
        """Return the normalized amplitude for ``first_energy <= second_energy``."""
        first = np.asarray(first_energy, dtype=float)
        second = np.asarray(second_energy, dtype=float)
        if np.any(first > second):
            raise ValueError("ordered_amplitude requires first_energy <= second_energy.")
        radial = sum(
            coefficient
            * np.exp(-decay * first / self.model.energy_cutoff)
            for coefficient, decay in zip(
                self.mixing_coefficients,
                self.radial_decays,
            )
        )
        return (
            self.normalization
            * self.model.form_factor(first)
            * self.model.form_factor(second)
            * np.exp(
                -self.memory_decay
                * (second - first)
                / self.model.energy_cutoff
            )
            * radial
        )

    def combined_matrices(self):
        """Return the explicit ``6 x 6`` finite-depth cLETTA matrices."""
        cutoff = self.model.energy_cutoff
        q_matrix = np.diag(
            [0.0, self.radial_decays[0] / cutoff, self.radial_decays[1] / cutoff]
        )
        r_matrix = np.zeros((3, 3), dtype=float)
        tie_matrix = np.zeros((3, 3), dtype=float)
        kappa = self.memory_decay / cutoff
        for row, (coefficient, decay) in enumerate(
            zip(self.mixing_coefficients, self.radial_decays),
            start=1,
        ):
            tie_matrix[row, 0] = (
                self.normalization
                * coefficient
                * np.exp(-decay)
                / np.sqrt(kappa)
            )
        return cletta_memory_matrices(
            q_matrix,
            r_matrix,
            tie_matrix,
            kappa,
            memory_dim=2,
        )

    def boundary_vectors(self):
        """Return memory-vacuum boundaries for the two post-opening channels."""
        right = np.zeros(6, dtype=np.complex128)
        left = np.zeros(6, dtype=np.complex128)
        right[0] = 1.0
        left[1:3] = 1.0
        return left, right

    def contracted_ordered_amplitude(self, first_energy, second_energy):
        """Contract the explicit ``6 x 6`` matrices for two insertions."""
        first = float(first_energy)
        second = float(second_energy)
        if not (0.0 <= first <= second <= self.model.energy_cutoff):
            raise ValueError("energies must satisfy 0 <= first <= second <= cutoff.")
        q_matrix, r_matrix = self.combined_matrices()
        left, right = self.boundary_vectors()
        coefficient = (
            left
            @ expm(q_matrix * (self.model.energy_cutoff - second))
            @ r_matrix
            @ expm(q_matrix * (second - first))
            @ r_matrix
            @ expm(q_matrix * first)
            @ right
        )
        return (
            self.model.form_factor(first)
            * self.model.form_factor(second)
            * coefficient
        )

    @property
    def exact_dilute_pair_energy(self):
        return -2.0 * self.model.binding_energy()


__all__ = [
    "ContinuumPipPairingModel",
    "ThermodynamicPipBCS",
    "ThermodynamicPipCLETTA",
    "ExactOnePairPipState",
    "OneScalePipCLETTA",
    "TwoPairPipCLETTA",
    "TwoPairPipD3CLETTA",
]
