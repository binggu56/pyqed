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
try:  # pragma: no cover
    from .phys import *  # noqa: F401,F403
except ModuleNotFoundError:
    # Most of pyqed does not require SciPy; allow import to succeed.
    pass

try:  # pragma: no cover
    from .mol import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from .style import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from .wpd import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from .qchem import Molecule  # noqa: F401
except (ModuleNotFoundError, ImportError):
    # qchem pulls in large dependencies and historically also referenced
    # top-level symbols during import, so allow import cycles to fail softly.
    pass

try:  # pragma: no cover
    from .optics import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from pyqed.polariton.cavity import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from pyqed.dvr.dvr_1d import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from pyqed.mps.mps import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

try:  # pragma: no cover
    from pyqed.qip import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass
