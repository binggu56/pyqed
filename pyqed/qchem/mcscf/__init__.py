from .casci import CASCI
from . import avas
from .avas import AVAS
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
    from .rasscf import RASCI, RASSCF, FirstOrderRASSCF, SecondOrderRASSCF
except (ImportError, OSError, TimeoutError):
    RASCI = None
    RASSCF = None
    FirstOrderRASSCF = None
    SecondOrderRASSCF = None

try:
    from .reduced_ci import ReducedCISubspace
except (ImportError, OSError, TimeoutError):
    ReducedCISubspace = None

try:
    from .nevpt2 import NEVPT2, SCNEVPT2, NEVPT2Component
except (ImportError, OSError, TimeoutError):
    NEVPT2 = None
    SCNEVPT2 = None
    NEVPT2Component = None

try:
    from .caspt2 import (
        CASPT2,
        DiagonalCASPT2,
        MSCASPT2,
        XMSCASPT2,
        CASPT2Component,
    )
except (ImportError, OSError, TimeoutError):
    CASPT2 = None
    DiagonalCASPT2 = None
    MSCASPT2 = None
    XMSCASPT2 = None
    CASPT2Component = None

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
