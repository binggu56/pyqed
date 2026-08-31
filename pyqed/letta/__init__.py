"""LETTA tensor ansatz algorithms."""

from .core import (
    LETTA,
    LETTAOperatorPackage,
    SequentialLETTA,
)
from .range import NNNLETTA
from .uniform import ULETTA, UniformLETTA
from .xletta import AbelianXLETTA, XLETTA
from .abelian import Layout, TiedFrontierLayout, XLayout
from .physical_tying import (
    PhysicalTieState,
    PhysicalTieStep,
    VariationalPhysicalTie,
    compress_physical_ties,
    fixed_range_parent_sets,
)
from .cp import CPDecomposition, cp_als
from .conditional_cp import ConditionalCPDecomposition, conditional_cp_decompose
from .cp_tying import CPBlockUpdate, CPTiedLETTA
from .dense_tying import DenseSiteUpdate, DenseTiedLETTA
from .block_mpo_frontier import BlockFrontierMessage, BlockMPOFrontier
from .renormalized_frontier import (
    TermRenormalizedFrontier,
    renormalized_operator_mpo,
)
from .frontier_tying import (
    FrontierBondExpansion,
    FrontierGaugeUpdate,
    FrontierMergedSolveDiagnostics,
    FrontierNaturalGradientUpdate,
    FrontierPairEnvironment,
    FrontierSiteEnvironment,
    FrontierSiteUpdate,
    FrontierTiedLETTA,
    FrontierTwoSiteUpdate,
    GraphLETTA,
)
from .projected_frontier import ProjectedLETTA, SectorProjectedLETTA, SectorProjection
from .local_terms import (
    LocalHamiltonian,
    LocalMPO,
    LocalMPOProduct,
    LocalTerm,
    local_charges_from_sites,
    fixed_charge_projector_mpo,
    validate_charge_conservation,
)
from .initialization import (
    frontier_tensors_from_mps,
    frontier_tied_letta_from_mps,
)
from .matrix_free import (
    BlockDavidsonDiagnostics,
    DavidsonDiagnostics,
    lowest_generalized_davidson,
    lowest_recycled_block_davidson,
)
from .physical_blocks import (
    PhysicalBlockGeneralizedProblem,
    PhysicalBlockLayout,
    PhysicalBlockLinearOperator,
    PhysicalBlockSolveDiagnostics,
    hamiltonian_physical_connectivity,
)
from .tt_frontier import (
    TermwiseTTMPOFrontier,
    TTAdvanceDiagnostics,
    TTContractionDiagnostics,
    TTFrontier,
    TTHoleDiagnostics,
    TTMPOFrontier,
    TTRoundDiagnostics,
)
from .vmc import (
    ConfigurationActionOperator,
    EnergyEstimate,
    LETTAProductCache,
    LETTAVMC,
    LETTAWavefunction,
    LocalHamiltonianActions,
    MetropolisDiagnostics,
    MetropolisSampler,
    SRDirection,
    SRProposal,
    VMCSamples,
)
from .ordering import (
    heuristic_heisenberg_block_order,
    heuristic_heisenberg_order,
    heisenberg_block_frontier_profile,
    heisenberg_frontier_profile,
    optimize_heisenberg_block_order,
    optimize_heisenberg_order,
)
from .tdvp import (
    LETTATDVPEngine,
    NumPyTDVP,
    Window2Hamiltonian,
    letta_structural_rank_caps,
    nearest_neighbor_hamiltonian,
    one_site_tdvp_step,
    two_site_tdvp_step,
    window2_hamiltonian_from_mpo,
    window2_product_state,
)
from .dynamics import LETTAEvolution, TDVP, resolve_letta_backend
from .nnn_tdvp import (
    NNNLETTATDVPEngine,
    nnn_product_state,
    nnn_structural_rank_caps,
    one_site_nnn_tdvp_step,
)
from .observables import (
    nnn_system_reduced_density_matrix,
    site_reduced_density_matrix,
    system_reduced_density_matrix,
)
from .adaptive_graph import (
    AdaptiveTieGraphRun,
    AdaptiveTieGraphStep,
    TieFrontierCut,
    TieGraphCost,
    TieGraphEvaluation,
    TieGraphProposal,
    TieSignal,
    TieSignalBatch,
    adapt_tie_graph,
    adaptive_tie_graph_step,
    evaluate_tie_graph_proposal,
    graph_signals_from_samples,
    rank_tie_graph_proposals,
    sample_tie_signals,
    state_with_tie_graph_proposal,
    tie_edges,
    tie_frontier_cost,
)


def __getattr__(name):
    if name in {"NonAbelianFrontierLETTA", "SU2LETTA"}:
        from . import su2_qchem

        return getattr(su2_qchem, name)
    torch_exports = {
        "TorchLETTATDVPEngine",
        "TorchTDVP",
        "TorchWindow2Hamiltonian",
        "TorchWindow2State",
        "torch_one_site_tdvp_step",
        "torch_backend_capabilities",
        "torch_site_reduced_density_matrix",
        "torch_system_reduced_density_matrix",
        "torch_two_site_tdvp_step",
    }
    if name in torch_exports:
        from . import torch_tdvp

        return getattr(torch_tdvp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdaptiveTieGraphRun",
    "AdaptiveTieGraphStep",
    "LETTA",
    "LETTAOperatorPackage",
    "NonAbelianFrontierLETTA",
    "SequentialLETTA",
    "SU2LETTA",
    "NNNLETTA",
    "NNNLETTATDVPEngine",
    "CPBlockUpdate",
    "BlockFrontierMessage",
    "BlockMPOFrontier",
    "CPDecomposition",
    "ConditionalCPDecomposition",
    "CPTiedLETTA",
    "ConfigurationActionOperator",
    "DenseSiteUpdate",
    "DenseTiedLETTA",
    "DavidsonDiagnostics",
    "BlockDavidsonDiagnostics",
    "FrontierSiteEnvironment",
    "FrontierSiteUpdate",
    "FrontierTiedLETTA",
    "GraphLETTA",
    "FrontierTwoSiteUpdate",
    "SectorProjectedLETTA",
    "ProjectedLETTA",
    "SectorProjection",
    "TiedFrontierLayout",
    "FrontierBondExpansion",
    "FrontierGaugeUpdate",
    "FrontierMergedSolveDiagnostics",
    "FrontierNaturalGradientUpdate",
    "FrontierPairEnvironment",
    "EnergyEstimate",
    "LETTAProductCache",
    "LETTAVMC",
    "LETTAWavefunction",
    "LETTATDVPEngine",
    "NumPyTDVP",
    "LETTAEvolution",
    "TDVP",
    "resolve_letta_backend",
    "LocalHamiltonian",
    "LocalMPO",
    "LocalMPOProduct",
    "LocalTerm",
    "Window2Hamiltonian",
    "local_charges_from_sites",
    "LocalHamiltonianActions",
    "MetropolisDiagnostics",
    "MetropolisSampler",
    "PhysicalTieState",
    "PhysicalTieStep",
    "PhysicalBlockGeneralizedProblem",
    "PhysicalBlockLayout",
    "PhysicalBlockLinearOperator",
    "PhysicalBlockSolveDiagnostics",
    "SRDirection",
    "SRProposal",
    "TermwiseTTMPOFrontier",
    "TermRenormalizedFrontier",
    "TTAdvanceDiagnostics",
    "TTContractionDiagnostics",
    "TTFrontier",
    "TTHoleDiagnostics",
    "TTMPOFrontier",
    "TTRoundDiagnostics",
    "TieFrontierCut",
    "TieGraphCost",
    "TieGraphEvaluation",
    "TieGraphProposal",
    "TieSignal",
    "TieSignalBatch",
    "VariationalPhysicalTie",
    "ULETTA",
    "UniformLETTA",
    "AbelianXLETTA",
    "Layout",
    "XLayout",
    "XLETTA",
    "VMCSamples",
    "adapt_tie_graph",
    "adaptive_tie_graph_step",
    "compress_physical_ties",
    "cp_als",
    "conditional_cp_decompose",
    "evaluate_tie_graph_proposal",
    "fixed_range_parent_sets",
    "frontier_tensors_from_mps",
    "frontier_tied_letta_from_mps",
    "fixed_charge_projector_mpo",
    "graph_signals_from_samples",
    "hamiltonian_physical_connectivity",
    "heuristic_heisenberg_block_order",
    "heuristic_heisenberg_order",
    "heisenberg_block_frontier_profile",
    "heisenberg_frontier_profile",
    "lowest_generalized_davidson",
    "lowest_recycled_block_davidson",
    "letta_structural_rank_caps",
    "nearest_neighbor_hamiltonian",
    "nnn_system_reduced_density_matrix",
    "nnn_product_state",
    "nnn_structural_rank_caps",
    "one_site_nnn_tdvp_step",
    "one_site_tdvp_step",
    "optimize_heisenberg_block_order",
    "optimize_heisenberg_order",
    "rank_tie_graph_proposals",
    "renormalized_operator_mpo",
    "sample_tie_signals",
    "site_reduced_density_matrix",
    "state_with_tie_graph_proposal",
    "system_reduced_density_matrix",
    "tie_edges",
    "tie_frontier_cost",
    "two_site_tdvp_step",
    "validate_charge_conservation",
    "window2_hamiltonian_from_mpo",
    "window2_product_state",
]
