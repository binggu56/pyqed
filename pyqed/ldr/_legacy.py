"""Shared helpers for marking legacy LDR modules.

These modules are kept for compatibility, but the preferred APIs now live in
the streamlined LDR stack (:class:`LDR`, ``coarse_grained``, and ``core``).
"""

from __future__ import annotations

import warnings

_warned_modules: set[str] = set()


def warn_legacy_module(module_name: str, *, replacement: str | None = None):
    """Emit a single deprecation warning for a legacy module.

    Parameters
    ----------
    module_name:
        Fully qualified module name.
    replacement:
        Optional recommended replacement module or API.
    """
    if module_name in _warned_modules:
        return
    _warned_modules.add(module_name)

    message = (
        f"{module_name!r} is a legacy LDR implementation kept only for compatibility. "
        "Prefer the cleaner runtime APIs in ``pyqed.ldr`` and ``pyqed.ldr.core`` "
        "for active development."
    )
    if replacement is not None:
        message = (
            f"{module_name!r} is a legacy module kept only for compatibility. "
            f"Use {replacement!r} instead."
        )

    warnings.warn(message, DeprecationWarning, stacklevel=2)


