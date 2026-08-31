"""Build a Jacobi-tree polyspherical KEO as an MPO.

The tree describes AB + CD bond breaking. Its root coordinate ``r0`` is the
distance between the AB and CD centers of mass. A production calculation
should use radial/angular DVRs tailored to the desired boundary conditions;
the two-point sine grids here keep the example inexpensive.
"""

import numpy as np

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.namd.polyspherical import (
    PolysphericalTree,
    build_keo_mpo,
    sample_analytic_metric,
)


def main():
    masses = np.array([1.0, 1.0, 1.0, 1.0])
    tree = PolysphericalTree(((0, 1), (2, 3)), masses)
    domains = [
        (2.5, 5.0),             # r0: AB--CD separation
        (1.0, 2.0),             # r1: AB bond
        (0.4, np.pi - 0.4),     # theta1
        (1.0, 2.0),             # r2: CD bond
        (0.4, np.pi - 0.4),     # theta2
        (-np.pi, np.pi),        # phi2
    ]
    dvrs = [SineDVR(lower, upper, 2) for lower, upper in domains]

    keo = build_keo_mpo(tree, dvrs, method="analytic")
    metric, pseudopotential = sample_analytic_metric(
        tree, dvrs
    )
    dense = _mpo_to_dense_operator(keo)
    print("coordinates:", tree.coordinate_labels)
    print("Jacobi reduced masses:", tree.reduced_masses)
    print("metric grid shape:", metric.shape)
    print("pseudopotential grid shape:", pseudopotential.shape)
    print("MPO bonds:", keo.bond_orders())
    print("dense validation shape:", dense.shape)
    print("Hermiticity error:", np.max(np.abs(dense - dense.conj().T)))


if __name__ == "__main__":
    main()
