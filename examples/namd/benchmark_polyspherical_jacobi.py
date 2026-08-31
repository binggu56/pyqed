"""Benchmark the generic polyspherical KEO against analytical Jacobi form."""

import numpy as np

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.namd.polyspherical import (
    PolysphericalTree,
    build_keo_mpo,
)
from pyqed.namd.triatomic import Triatom


def reorder_r_R_gamma_to_R_r_gamma(matrix, shape):
    nr, nR, ng = shape
    return matrix.reshape(nr, nR, ng, nr, nR, ng).transpose(
        1, 0, 2, 4, 3, 5
    ).reshape(nR * nr * ng, nR * nr * ng)


def run(npts):
    masses = np.array([1.0, 16.0, 1.0])
    radial_R = SineDVR(2.0, 3.0, npts)
    radial_r = SineDVR(1.4, 2.2, npts)
    gamma = SineDVR(0.7, 1.5, npts)
    tree = PolysphericalTree(((1, 2), 0), masses)
    dvrs = [radial_R, radial_r, gamma]

    mpo, metric, pseudopotential = build_keo_mpo(
        tree, dvrs, field_rtol=0.0, return_fields=True
    )
    dense = _mpo_to_dense_operator(mpo)

    triatom = Triatom.__new__(Triatom)
    triatom.mass = masses
    triatom.J = 0
    triatom.dvrs = [radial_r, radial_R, gamma]
    triatom.x = [dvr.x for dvr in triatom.dvrs]
    triatom.nx = [dvr.npts for dvr in triatom.dvrs]
    analytic = reorder_r_R_gamma_to_R_r_gamma(
        triatom._buildK_jacobi_h_oh(), triatom.nx
    )

    radial_R_grid, radial_r_grid, gamma_grid = np.meshgrid(
        *(dvr.x for dvr in dvrs), indexing="ij"
    )
    mu_r = masses[1] * masses[2] / (masses[1] + masses[2])
    mu_R = masses[0] * (masses[1] + masses[2]) / masses.sum()
    angular = (
        1.0 / (mu_R * radial_R_grid**2)
        + 1.0 / (mu_r * radial_r_grid**2)
    )
    expected_metric = np.zeros_like(metric)
    expected_metric[..., 0, 0] = 1.0 / mu_R
    expected_metric[..., 1, 1] = 1.0 / mu_r
    expected_metric[..., 2, 2] = angular
    expected_pseudo = (
        -0.125 * angular * (1.0 + 1.0 / np.sin(gamma_grid) ** 2)
    )

    return {
        "npts": npts,
        "metric_error": float(np.max(np.abs(metric - expected_metric))),
        "pseudo_error": float(
            np.max(np.abs(pseudopotential - expected_pseudo))
        ),
        "operator_relative_error": float(
            np.linalg.norm(dense - analytic) / np.linalg.norm(analytic)
        ),
        "hermiticity_error": float(
            np.max(np.abs(dense - dense.conj().T))
        ),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(dense)[0]),
        "maximum_mpo_bond": max(mpo.bond_orders()),
    }


def main():
    print(
        "n  metric_err  pseudo_err  operator_rel  herm_err  min_eig  max_bond"
    )
    for npts in (2, 3, 4):
        result = run(npts)
        print(
            f"{result['npts']:d} "
            f"{result['metric_error']:.3e} "
            f"{result['pseudo_error']:.3e} "
            f"{result['operator_relative_error']:.3e} "
            f"{result['hermiticity_error']:.3e} "
            f"{result['minimum_eigenvalue']:.6e} "
            f"{result['maximum_mpo_bond']:d}"
        )


if __name__ == "__main__":
    main()
