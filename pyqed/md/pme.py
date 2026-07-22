"""PME setup helpers for native MD workflows."""

from __future__ import annotations

import numpy as np


PME_ACCURACY_SPACING_NM = {
    "balanced": 0.10,
    "high": 0.075,
}


def pme_mesh_for_accuracy(cell_lengths_nm, accuracy, multiple=8, minimum=16):
    """Return an orthorhombic PME mesh for a named accuracy level.

    The mesh is chosen from a target real-space grid spacing and rounded up to
    a small FFT-friendly multiple.  ``high`` intentionally mirrors the DPPC
    membrane parity setting that closes PyQED/OpenMM reciprocal-space energy.
    """
    accuracy = str(accuracy).lower()
    if accuracy not in PME_ACCURACY_SPACING_NM:
        raise ValueError("PME accuracy must be 'balanced' or 'high'.")
    lengths = np.asarray(cell_lengths_nm, dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("cell_lengths_nm must contain three positive lengths.")
    multiple = int(multiple)
    minimum = int(minimum)
    if multiple <= 0 or minimum <= 0:
        raise ValueError("multiple and minimum must be positive.")
    raw = np.ceil(lengths / PME_ACCURACY_SPACING_NM[accuracy]).astype(int)
    mesh = np.ceil(raw / multiple).astype(int) * multiple
    return tuple(int(max(value, minimum)) for value in mesh)
