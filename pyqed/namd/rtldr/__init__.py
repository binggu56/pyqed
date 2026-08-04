"""Real-time local diabatic representation."""

from .core import RetainedStateRTLDR, RetainedStateTrajectory, frames_from_overlap
from .rttdhf import RTLDR, RTLDRTrajectory, RTTDHFFrame, det_overlap

__all__ = [
    "GDVRFrame",
    "GDVRSolver",
    "GDVRTrajectory",
    "RTLDR",
    "RTLDRTrajectory",
    "RTTDHFFrame",
    "RetainedStateRTLDR",
    "RetainedStateTrajectory",
    "det_overlap",
    "frames_from_overlap",
    "gdvr_det_overlap",
]


def __getattr__(name):
    if name in {"GDVRFrame", "GDVRSolver", "GDVRTrajectory", "gdvr_det_overlap"}:
        from . import gdvr

        mapping = {
            "GDVRFrame": gdvr.GDVRFrame,
            "GDVRSolver": gdvr.Solver,
            "GDVRTrajectory": gdvr.Trajectory,
            "gdvr_det_overlap": gdvr.gdvr_det_overlap,
        }
        value = mapping[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
