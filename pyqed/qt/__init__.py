"""Quantum-trajectory and quantum-hydrodynamic methods."""

from .jastrow_1d import (
    LegacyPolynomialQTM1D,
    ProjectedJastrow1D,
    exact_quartic_ground_state,
    quartic_force,
    quartic_potential,
)
from .jastrow_two_particle import (
    ProjectedTwoParticleJastrow1D,
    exact_two_particle_ground_state,
)
from .transport_basis import (
    SharedRadialTransportBasis,
    select_three_body_features,
    weak_poisson_objective,
)
from .neural_transport import InvariantNeuralTransportPotential
from .jastrow_three_particle import (
    ProjectedThreeParticleJastrow1D,
    exact_three_particle_ground_state,
)
from .score_corrections import (
    InvariantNeuralScoreCorrection1D,
    SharedLinearScoreCorrection1D,
    global_polynomial_jastrow_terms,
)
from .direct_score_flow import (
    DirectOverdampedScoreFlow1D,
    exact_double_well_three_particle_ground_state,
    optimize_global_double_well_jastrow,
    tilted_double_well_force,
    tilted_double_well_potential,
)
from .proximal_score_flow import ProximalLinearScoreFlow1D
from .transport_proximal_flow import JacobianProximalFlow1D
from .double_well_vmc import (
    ThreeParticleDoubleWellVMC,
    double_well_jastrow_terms,
    double_well_local_energy,
    integrated_autocorrelation_time,
    occupation_probabilities,
    optimize_symmetric_double_well_jastrow,
    three_body_gaussian_terms,
    three_particle_double_well_potential,
)
from .tdvmc import (
    ComplexJastrowTDVMC1D,
    anharmonic_double_well,
    split_operator_step,
)

__all__ = [
    "LegacyPolynomialQTM1D",
    "ProjectedJastrow1D",
    "exact_quartic_ground_state",
    "quartic_force",
    "quartic_potential",
    "ProjectedTwoParticleJastrow1D",
    "exact_two_particle_ground_state",
    "SharedRadialTransportBasis",
    "select_three_body_features",
    "weak_poisson_objective",
    "InvariantNeuralTransportPotential",
    "ProjectedThreeParticleJastrow1D",
    "exact_three_particle_ground_state",
    "InvariantNeuralScoreCorrection1D",
    "SharedLinearScoreCorrection1D",
    "global_polynomial_jastrow_terms",
    "DirectOverdampedScoreFlow1D",
    "exact_double_well_three_particle_ground_state",
    "optimize_global_double_well_jastrow",
    "tilted_double_well_force",
    "tilted_double_well_potential",
    "ProximalLinearScoreFlow1D",
    "JacobianProximalFlow1D",
    "ThreeParticleDoubleWellVMC",
    "double_well_jastrow_terms",
    "double_well_local_energy",
    "integrated_autocorrelation_time",
    "occupation_probabilities",
    "optimize_symmetric_double_well_jastrow",
    "three_body_gaussian_terms",
    "three_particle_double_well_potential",
    "ComplexJastrowTDVMC1D",
    "anharmonic_double_well",
    "split_operator_step",
]
