#!/usr/bin/env python3
"""Compare D3+ TDVP2 with exact dense propagation of the deployed 9^3 MPO."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import Coord, keo
from pyqed.ml import MACE
from pyqed.namd import TNLDR
from pyqed.units import au2fs

from h3plus_fci_mace_dynamics import (
    BOUNDS,
    COVARIANCE_BOHR2,
    H3Masses,
    ISOTOPE_MASSES_AMU,
    MAX_METRIC_CONDITION,
    geometry,
    mace_geometry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--tmax-fs", type=float, default=0.3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid = DVR.from_axes(
        tuple(SineDVR(lower, upper, 9) for lower, upper in BOUNDS),
        names=("Qs", "Qx", "Qy"),
    )
    coord = Coord(to_cartesian=geometry, bounds=BOUNDS)
    fit = MACE.load(args.checkpoint, mace_geometry, distill=False)
    fit.grids = tuple(np.asarray(axis) for axis in grid.x)
    fit.shape = tuple(grid.shape)
    fit.distill_y(rank=32, degree=8, method="grid", seed=31)
    nuclear_keo = keo.podolsky(
        max_metric_condition=MAX_METRIC_CONDITION
    ).bind(coord, grid=grid, molecule=H3Masses("D"))
    driver = TNLDR(
        fit, grid=grid, coord=coord, keo=nuclear_keo,
        overlap_rank=16, operator_rank=32,
    ).build()

    mesh = np.meshgrid(*grid.x, indexing="ij")
    coordinates = np.stack(mesh, axis=-1)
    covariance = (
        np.sqrt(ISOTOPE_MASSES_AMU["H"] / ISOTOPE_MASSES_AMU["D"])
        * COVARIANCE_BOHR2
    )
    exponent = np.einsum(
        "...i,ij,...j->...", coordinates, np.linalg.inv(covariance), coordinates
    )
    packet = np.zeros((*grid.shape, 2), dtype=complex)
    packet[..., 0] = np.exp(-0.25 * exponent)
    initial = driver.state(packet, max_rank=64, physical=False)
    tdvp_state = initial.copy()
    projectors = tuple(
        driver.adiabatic_projector(state, method="dense", max_rank=24)[0]
        for state in range(2)
    )
    dense_projectors = tuple(projector.to_dense() for projector in projectors)
    hamiltonian = csr_matrix(driver.hamiltonian.to_dense())
    exact = driver.dense(initial, physical=False).reshape(-1)
    exact /= np.linalg.norm(exact)
    steps = round(args.tmax_fs / args.dt_fs)
    dt = args.dt_fs / au2fs
    exact_populations = [
        [float(np.real(exact.conj() @ (projector @ exact)))
         for projector in dense_projectors]
    ]
    exact_norms = [float(np.vdot(exact, exact).real)]
    for _step in range(steps):
        exact = expm_multiply((-1j * dt) * hamiltonian, exact)
        exact_populations.append([
            float(np.real(exact.conj() @ (projector @ exact)))
            for projector in dense_projectors
        ])
        exact_norms.append(float(np.vdot(exact, exact).real))

    driver.run(
        tdvp_state, dt=dt, steps=steps, interval=1, max_bond=64,
        integrator="tdvp2", cutoff=1.0e-11, e_ops=projectors,
        normalize=False, progress=False,
    )
    tdvp = driver.dense(driver.final_state, physical=False).reshape(-1)
    tdvp /= np.linalg.norm(tdvp)
    fidelity = float(abs(np.vdot(exact, tdvp)) ** 2)
    exact_populations = np.asarray(exact_populations)
    population_error = np.abs(driver.populations - exact_populations)
    times = np.arange(steps + 1) * args.dt_fs
    metrics = {
        "grid": [9, 9, 9],
        "dt_fs": args.dt_fs,
        "tmax_fs": args.tmax_fs,
        "reference": "exact dense propagation of the deployed MACE/FTT/LPA MPO",
        "final_state_fidelity": fidelity,
        "maximum_population_error": float(np.max(population_error)),
        "final_s1_population_error": float(population_error[-1, 0]),
        "maximum_exact_norm_error": float(
            np.max(np.abs(np.asarray(exact_norms) - exact_norms[0]))
        ),
        "maximum_tdvp_norm_error": float(
            np.max(np.abs(driver.norms - driver.norms[0]))
        ),
    }
    (args.output_dir / "dense_reference_report.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "dense_reference.npz", time_fs=times,
        exact_populations=exact_populations,
        tdvp_populations=driver.populations,
        population_error=population_error,
    )

    figure, panels = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    panels[0].plot(times, exact_populations[:, 0], label="exact dense")
    panels[0].plot(times, driver.populations[:, 0], "--", label="TDVP2")
    panels[0].set(
        xlabel="time / fs", ylabel=r"$S_1$ population",
        title=r"D$_3^+$ $9^3$ reference",
    )
    panels[0].legend(frameon=False)
    panels[1].semilogy(times, np.maximum(population_error[:, 0], 1.0e-16))
    panels[1].set(
        xlabel="time / fs", ylabel=r"$|\Delta P_{S_1}|$",
        title="TDVP2 propagation error",
    )
    for panel in panels:
        panel.grid(alpha=0.2)
    output = args.output_dir / "dense_reference.png"
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
