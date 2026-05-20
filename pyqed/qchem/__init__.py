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
    from .geometric import BOHamiltonianDerivatives, GeometricFGTerms, bo_hamiltonian_derivatives
except (ImportError, OSError):
    BOHamiltonianDerivatives = None
    GeometricFGTerms = None
    bo_hamiltonian_derivatives = None

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
    from .ir import IR
except (ImportError, OSError):
    IR = None

try:
    from .vibronic import (
        LVC,
        build_linear_vibronic_model,
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
    build_linear_vibronic_model = None
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
    from .dmrg.overlap import overlap as dmrg_overlap
    from .dmrg.overlap import unitary_overlap as dmrg_unitary_overlap
    from .dmrg.overlap import biorthogonal_overlap as dmrg_biorthogonal_overlap
    from .dmrg.overlap import biorthogonal_overlap_diagnostics as dmrg_biorthogonal_overlap_diagnostics
    from .dmrg.overlap import automatic_overlap as dmrg_automatic_overlap
except (ImportError, OSError):
    dmrg_overlap = None
    dmrg_unitary_overlap = None
    dmrg_biorthogonal_overlap = None
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
    from .mcscf.casscf import CASSCF, FirstOrderCASSCF, SecondOrderCASSCF
except (ImportError, OSError):
    CASSCF = None
    FirstOrderCASSCF = None
    SecondOrderCASSCF = None

try:
    from .mcscf.soc_si import SOCStateInteractionResult, soc_state_interaction
except (ImportError, OSError):
    SOCStateInteractionResult = None
    soc_state_interaction = None

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
