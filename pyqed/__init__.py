"""
Top-level convenience imports for :mod:`pyqed`.

Historically this package eagerly imported most submodules (physics, optics,
qchem, MPS, etc.).  That makes interactive use convenient, but it also makes
``import pyqed`` fragile because a single optional dependency (e.g. SciPy) can
break imports for users that only need a small part of the library such as the
MPS/DMRG code.

To keep backwards compatibility *when dependencies are present* while allowing
lightweight imports in minimal environments, we guard optional star-imports.
"""

from __future__ import annotations

# Keep lightweight pieces unconditional.
from .units import *  # noqa: F401,F403

# Optional convenience imports.
_OPTIONAL_IMPORT_ERRORS = (ModuleNotFoundError, ImportError, OSError, TimeoutError)
_OPTIONAL_QCHEM_IMPORT_ERRORS = _OPTIONAL_IMPORT_ERRORS + (RuntimeError,)

try:  # pragma: no cover
    from .phys import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    # Most of pyqed does not require SciPy; allow import to succeed.
    pass

try:  # pragma: no cover
    from .mol import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from .style import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from .wpd import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from .qchem import Molecule  # noqa: F401
except _OPTIONAL_QCHEM_IMPORT_ERRORS:
    # qchem pulls in large dependencies and historically also referenced
    # top-level symbols during import.  Some optional accelerators can also
    # raise RuntimeError at decoration/cache setup time, so allow this
    # convenience import to fail softly.
    pass

try:  # pragma: no cover
    from .optics import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from pyqed.polariton.cavity import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from pyqed.dvr.dvr_1d import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from pyqed.mps.mps import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass

try:  # pragma: no cover
    from pyqed.qip import *  # noqa: F401,F403
except _OPTIONAL_IMPORT_ERRORS:
    pass
