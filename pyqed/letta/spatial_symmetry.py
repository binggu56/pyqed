"""Spatial multiplets for graph-tied, spin-reduced LETTA tensors.

The shared finite-group contraction algorithm lives in the MPS package.
See SpatialMultipletSweep for references, exactness, and limitations.
"""
from pyqed.mps.nonabelian.spatial_symmetry import SpatialMultipletSweep


class SpatialSymmetryLETTA(SpatialMultipletSweep):
    """Spatial group leg on a LETTA seed, retaining its graph-tied parameters."""
