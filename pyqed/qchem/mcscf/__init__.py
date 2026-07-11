from .casci import CASCI
try:
    from .cocas import COCAS, COCASCI
except (ImportError, OSError, TimeoutError):
    COCAS = None
    COCASCI = None

try:
    from .casscf import CASSCF, FirstOrderCASSCF, SecondOrderCASSCF
except (ImportError, OSError, TimeoutError):
    CASSCF = None
    FirstOrderCASSCF = None
    SecondOrderCASSCF = None

try:
    from .reduced_ci import ReducedCISubspace
except (ImportError, OSError, TimeoutError):
    ReducedCISubspace = None

try:
    from .soc_si import (
        SOCStateInteractionResult,
        SingletTripletSOCResult,
        st_soc,
        soc_state_interaction,
    )
except (ImportError, OSError, TimeoutError):
    SOCStateInteractionResult = None
    SingletTripletSOCResult = None
    st_soc = None
    soc_state_interaction = None
