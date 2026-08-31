"""Stable local paths for the OneDrive-backed phenol SA-CASSCF data."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHENOL_5D_PRODUCTION = PROJECT_ROOT / "dataset" / "phenol_5d_production"
DEFAULT_PHENOL_SA6_DATABASE = Path(
    os.environ.get(
        "PYQED_PHENOL_SA6_DATABASE",
        PROJECT_ROOT
        / "dataset"
        / "phenol_sa6_casscf_production"
        / "electronic.sqlite",
    )
).expanduser()
DEFAULT_PHENOL_5D_DATA = (
    PHENOL_5D_PRODUCTION
    / "inputs"
    / "inward"
    / "phenol_sa6_5d_p_gauge_inward.npz"
)
DEFAULT_PHENOL_5D_RADIAL_CORRECTION = (
    PHENOL_5D_PRODUCTION
    / "model"
    / "radial_correction"
    / "phenol_sa6_5d_radial_delta.npz"
)
DEFAULT_PHENOL_5D_CHECKPOINT = (
    PHENOL_5D_PRODUCTION
    / "model"
    / "mace_y"
    / "phenol_sa6_5d_mace_y.pt"
)
DEFAULT_PHENOL_5D_DISTILLATION_CACHE = (
    PHENOL_5D_PRODUCTION / "cache" / "distillation_49x7x5x9x7_r24"
)
DEFAULT_PHENOL_5D_OPERATOR_CACHE = (
    PHENOL_5D_PRODUCTION / "cache" / "dressed_operator_65x9x7x11x9_r64"
)
DEFAULT_PHENOL_5D_50FS = PHENOL_5D_PRODUCTION / "dynamics" / "050fs"
