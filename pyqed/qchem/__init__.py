from .mol import *
# from .casci import *
# from .fci import *

try:
    from .dft import *
except (ImportError, OSError):
    pass

try:
    from .hf import *
except (ImportError, OSError):
    pass

try:
    from .orbital_clustering import (
        cluster_mf_orbitals,
        cluster_orbitals,
        graph_cut_ratio,
        orbital_interaction_graph,
        spectral_orbital_clusters,
    )
except (ImportError, OSError):
    cluster_mf_orbitals = None
    cluster_orbitals = None
    graph_cut_ratio = None
    orbital_interaction_graph = None
    spectral_orbital_clusters = None

try:
    from .tddft import *
except (ImportError, OSError):
    pass

try:
    from .rttdhf import *
except (ImportError, OSError):
    pass

try:
    from .rttddft import *
except (ImportError, OSError):
    pass

try:
    from .soc import *
except (ImportError, OSError):
    pass

try:
    from .relativistic import *
except (ImportError, OSError):
    pass

try:
    from .symmetry import *
except (ImportError, OSError):
    pass

try:
    from .geometric import (
        BOHamiltonianDerivatives,
        GeometricFGTerms,
        bo_hamiltonian_derivatives,
        dipole_exponential_ci_overlap,
        dipole_orbital_rotation_unitary,
        orbital_rotation_ci_overlap,
    )
except (ImportError, OSError):
    BOHamiltonianDerivatives = None
    GeometricFGTerms = None
    bo_hamiltonian_derivatives = None
    dipole_exponential_ci_overlap = None
    dipole_orbital_rotation_unitary = None
    orbital_rotation_ci_overlap = None

try:
    from .ci.cisd import *
except (ImportError, OSError):
    pass

try:
    from .ci.fci import *
except (ImportError, OSError):
    pass

try:
    from .mp.mp2 import COMP2, MP2, UMP2
except (ImportError, OSError):
    COMP2 = None
    MP2 = None
    UMP2 = None

try:
    from .cd import CD, CDResult
except (ImportError, OSError):
    CD = None
    CDResult = None

try:
    from .membrane_cd import MembraneCD, MembraneCDFrame, MembraneCDResult
except (ImportError, OSError):
    MembraneCD = None
    MembraneCDFrame = None
    MembraneCDResult = None

try:
    from .xas import XAS, XASResult
except (ImportError, OSError):
    XAS = None
    XASResult = None

try:
    from .ir import IR
except (ImportError, OSError):
    IR = None

try:
    from .vibronic import (
        LVC,
        QVC,
        build_lvc,
        compare_lvc_to_sharc,
        load_sharc_lvc_template,
        lvc_from_sharc_template,
        mode_derivative_couplings_from_overlaps,
        project_cartesian_to_modes,
        vibronic_couplings_from_derivative_couplings,
    )
except (ImportError, OSError):
    LVC = None
    QVC = None
    build_lvc = None
    compare_lvc_to_sharc = None
    load_sharc_lvc_template = None
    lvc_from_sharc_template = None
    mode_derivative_couplings_from_overlaps = None
    project_cartesian_to_modes = None
    vibronic_couplings_from_derivative_couplings = None

try:
    from .semiempirical import (
        DEFAULT_OM2_PARAMETERS,
        MRCI as SemiempiricalMRCI,
        OM2,
        OM2AtomicParameters,
        OM2HamiltonianData,
        OM2MRCIScanner,
        OM2ParameterError,
        OM2ParameterSet,
        OM2Reference,
        SemiempiricalMolecule,
        SemiempiricalMethodNotAvailable,
        ValenceOrbital,
    )
except (ImportError, OSError):
    DEFAULT_OM2_PARAMETERS = None
    SemiempiricalMRCI = None
    OM2 = None
    OM2AtomicParameters = None
    OM2HamiltonianData = None
    OM2MRCIScanner = None
    OM2ParameterError = None
    OM2ParameterSet = None
    OM2Reference = None
    SemiempiricalMolecule = None
    SemiempiricalMethodNotAvailable = None
    ValenceOrbital = None

# Optional modules can be temporarily unavailable while adjacent APIs evolve.
try:
    from .dmrg.dmrg import DMRG, QCDMRG
    from .dmrg.dmrgscf import DMRGSCF
    from .dmrg.tddmrg import TDDMRG, gaussian_pulse
except (ImportError, OSError):
    DMRG = None
    DMRGSCF = None
    QCDMRG = None
    TDDMRG = None
    gaussian_pulse = None

try:
    from .letta import LETTA
except (ImportError, OSError):
    LETTA = None

try:
    from .dmrg.overlap import overlap as dmrg_overlap
    from .dmrg.overlap import unitary_overlap as dmrg_unitary_overlap
    from .dmrg.overlap import biorthogonal_overlap as dmrg_biorthogonal_overlap
    from .dmrg.overlap import su2_biorthogonal_overlap as dmrg_su2_biorthogonal_overlap
    from .dmrg.overlap import biorthogonal_overlap_diagnostics as dmrg_biorthogonal_overlap_diagnostics
    from .dmrg.overlap import automatic_overlap as dmrg_automatic_overlap
except (ImportError, OSError):
    dmrg_overlap = None
    dmrg_unitary_overlap = None
    dmrg_biorthogonal_overlap = None
    dmrg_su2_biorthogonal_overlap = None
    dmrg_biorthogonal_overlap_diagnostics = None
    dmrg_automatic_overlap = None

try:
    from .mcscf.cocas import COCAS, COCASCI
except (ImportError, OSError):
    COCAS = None
    COCASCI = None

try:
    from .mcscf.casci import CASCI
except (ImportError, OSError):
    CASCI = None

try:
    from .mcscf.rasscf import RASCI, RASSCF, FirstOrderRASSCF, SecondOrderRASSCF
except (ImportError, OSError):
    RASCI = None
    RASSCF = None
    FirstOrderRASSCF = None
    SecondOrderRASSCF = None

try:
    from .mcscf.nevpt2 import NEVPT2, SCNEVPT2, NEVPT2Component
except (ImportError, OSError):
    NEVPT2 = None
    SCNEVPT2 = None
    NEVPT2Component = None

try:
    from .mcscf.caspt2 import (
        CASPT2,
        DiagonalCASPT2,
        MSCASPT2,
        XMSCASPT2,
        CASPT2Component,
    )
except (ImportError, OSError):
    CASPT2 = None
    DiagonalCASPT2 = None
    MSCASPT2 = None
    XMSCASPT2 = None
    CASPT2Component = None

try:
    from .mcscf import AVAS, avas
except (ImportError, OSError):
    AVAS = None
    avas = None

try:
    from .tdcasci import TDCASCI, TDCASCITrajectory
    from .tdcis import TDCIS, cis_determinant_basis
    from .mctdhf import (
        DenseCIDensityProvider,
        DMRGDensityProvider,
        MCTDHF,
        MCTDHFTrajectory,
        RDM12DensityProvider,
    )
except (ImportError, OSError):
    TDCASCI = None
    TDCASCITrajectory = None
    TDCIS = None
    cis_determinant_basis = None
    MCTDHF = None
    MCTDHFTrajectory = None
    DenseCIDensityProvider = None
    DMRGDensityProvider = None
    RDM12DensityProvider = None

try:
    from . import nac
except (ImportError, OSError):
    nac = None

try:
    from .mcscf.zvector import MCSCFZVector, MCSCFZVectorResult, NACRHS, PropertyRHS
except (ImportError, OSError):
    MCSCFZVectorResult = None
    MCSCFZVector = None
    NACRHS = None
    PropertyRHS = None

try:
    from .mcscf.casscf import CASSCF, FirstOrderCASSCF, SecondOrderCASSCF
except (ImportError, OSError):
    CASSCF = None
    FirstOrderCASSCF = None
    SecondOrderCASSCF = None

try:
    from pyqed.narg.qchem import NARGOpt, NARGSCF
except (ImportError, OSError):
    pass


def __getattr__(name):
    if name in {"NARGOpt", "NARGSCF"}:
        from pyqed.narg.qchem import NARGOpt, NARGSCF

        globals().update(NARGOpt=NARGOpt, NARGSCF=NARGSCF)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from .mcscf.soc_si import (
        SOCStateInteractionResult,
        SingletTripletSOCResult,
        align_triplet_multiplet_phases,
        st_soc,
        soc_state_interaction,
        spin_lower,
    )
except (ImportError, OSError):
    SOCStateInteractionResult = None
    SingletTripletSOCResult = None
    align_triplet_multiplet_phases = None
    st_soc = None
    soc_state_interaction = None
    spin_lower = None

try:
    from .qmmm import (
        PointChargeEmbeddedPostSCF,
        PointChargeEmbeddedSCF,
        embed_point_charges,
    )
except (ImportError, OSError):
    PointChargeEmbeddedPostSCF = None
    PointChargeEmbeddedSCF = None
    embed_point_charges = None
