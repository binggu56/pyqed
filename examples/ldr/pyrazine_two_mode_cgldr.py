#!/usr/bin/env python3
"""Exact two-mode pyrazine benchmark for CGLDR."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import DVR
from pyqed.ldr import CGLDR, ElectronicPartition
from pyqed.ldr.coarse_grained import CGLDRElectronicData
from pyqed.ldr.observables import (
    mps_to_array,
    nuclear_density_distance,
    nuclear_observables,
)
from pyqed.models.pyrazine import lvc as literature_lvc
from pyqed.mps.mps import gaussian_state
from pyqed.units import au2fs


STATE_IDS = (1, 2)
COORDINATE_NAMES = ("nu_6a", "nu_10a")


def two_state_parameters():
    """Return the literature S1/S2 parameters in tuning/coupling order."""
    model = literature_lvc()
    energies = np.asarray(model.E)[list(STATE_IDS)]
    energies = energies - energies[0]
    frequencies = np.asarray(model.omega)[[1, 0]]
    couplings = np.asarray(model.linear_couplings)[
        np.ix_(STATE_IDS, STATE_IDS, (1, 0))
    ]
    return energies, frequencies, couplings


def build_dvr(
    npts=(64, 40),
    domains=((-8.0, 8.0), (-8.0, 8.0)),
):
    _, frequencies, _ = two_state_parameters()
    return DVR(
        domains=domains,
        npts=npts,
        mass=tuple(1.0 / frequencies),
        names=COORDINATE_NAMES,
    )


def electronic_hamiltonian(tuning, coupling):
    """Evaluate the exact two-state diabatic vibronic potential."""
    energies, frequencies, linear = two_state_parameters()
    hamiltonian = np.diag(energies).astype(float)
    hamiltonian += tuning * linear[..., 0]
    hamiltonian += coupling * linear[..., 1]
    hamiltonian += (
        0.5
        * (
            frequencies[0] * tuning**2
            + frequencies[1] * coupling**2
        )
        * np.eye(2)
    )
    return hamiltonian


def build_cgldr(dvr, *, max_rank=64):
    """Build CGLDR data whose secondary expansion is exact for the LVC model."""
    energies, frequencies, linear = two_state_parameters()
    tuning_grid = np.asarray(dvr.x[0])
    sampled_energies = (
        energies[None, :]
        + tuning_grid[:, None]
        * np.diagonal(linear[..., 0])[None, :]
        + 0.5 * frequencies[0] * tuning_grid[:, None] ** 2
    )
    overlaps = np.broadcast_to(
        np.eye(2)[None, :, None, :],
        (tuning_grid.size, 2, tuning_grid.size, 2),
    ).copy()
    gradients = np.broadcast_to(
        linear[..., 1],
        (tuning_grid.size, 1, 2, 2),
    ).copy()
    hessians = np.broadcast_to(
        frequencies[1] * np.eye(2),
        (tuning_grid.size, 1, 1, 2, 2),
    ).copy()
    data = CGLDRElectronicData(
        energies=sampled_energies,
        overlaps=overlaps,
        hamiltonian_gradients=gradients,
        hamiltonian_hessians=hessians,
        reactive_grids=(tuning_grid,),
        metadata={
            "model": "literature_pyrazine_two_mode_lvc",
            "sampled_mode": "nu_6a",
            "expanded_mode": "nu_10a",
        },
    )
    dynamics = CGLDR(
        dvr,
        ElectronicPartition(
            sampled=("nu_6a",),
            expanded=("nu_10a",),
            center=(0.0,),
        ),
        state_ids=(0, 1),
        tt_options={"max_rank": max_rank},
    )
    dynamics.set_electronic_data(data)
    return dynamics


def build_full_hamiltonian(dvr):
    """Return the exact sparse two-dimensional diabatic Hamiltonian."""
    local = np.asarray(
        [
            electronic_hamiltonian(tuning, coupling)
            for tuning, coupling in dvr.points
        ]
    )
    potential = sp.block_diag(local, format="csr")
    kinetic = sp.kron(
        dvr.kinetic(),
        sp.identity(2, dtype=complex, format="csr"),
        format="csr",
    )
    return kinetic + potential


def initial_wavepacket(dvr):
    """Return the dimensionless harmonic ground packet on diabatic S2."""
    return gaussian_state(
        dvr.x,
        state=1,
        nstates=2,
        center=(0.0, 0.0),
        width=(1.0, 1.0),
    )


def propagate_full(hamiltonian, initial, times):
    vector = np.moveaxis(mps_to_array(initial), 0, -1).reshape(-1)
    states = expm_multiply(
        -1j * hamiltonian,
        vector,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=-1j * hamiltonian.diagonal().sum(),
    )
    return states.reshape(len(times), *initial.dims[1:], 2)


def run_benchmark(
    *,
    npts=(64, 40),
    domains=((-8.0, 8.0), (-8.0, 8.0)),
    time_step=2.0,
    steps=1000,
    output_every=10,
    max_rank=64,
):
    dvr = build_dvr(npts=npts, domains=domains)
    dynamics = build_cgldr(dvr, max_rank=max_rank)
    initial = initial_wavepacket(dvr)
    dynamics.run(
        initial,
        time_step=time_step,
        steps=steps,
        output_every=output_every,
        save_data=False,
    )
    cg_states = np.asarray([mps_to_array(state) for state in dynamics.states])
    times = np.linspace(0.0, time_step * steps, len(cg_states))

    hamiltonian = build_full_hamiltonian(dvr)
    full_states = propagate_full(hamiltonian, initial, times)
    full_observables = nuclear_observables(
        full_states,
        dvr.x,
        electronic_axis=-1,
    )
    cg_observables = nuclear_observables(
        cg_states,
        dvr.x,
        electronic_axis=1,
    )
    distance = nuclear_density_distance(
        full_observables["nuclear_density"],
        cg_observables["nuclear_density"],
    )
    full_populations = np.sum(
        np.abs(full_states) ** 2,
        axis=(1, 2),
    )
    cg_populations = np.sum(
        np.abs(cg_states) ** 2,
        axis=(2, 3),
    )
    return {
        "times_au": times,
        "times_fs": times * au2fs,
        "full_populations": full_populations,
        "cg_populations": cg_populations,
        "full_nuclear_density": full_observables["nuclear_density"],
        "cg_nuclear_density": cg_observables["nuclear_density"],
        "full_coordinate_means": full_observables["coordinate_means"],
        "cg_coordinate_means": cg_observables["coordinate_means"],
        "full_coordinate_covariance": full_observables[
            "coordinate_covariance"
        ],
        "cg_coordinate_covariance": cg_observables[
            "coordinate_covariance"
        ],
        "full_survival_probability": full_observables[
            "survival_probability"
        ],
        "cg_survival_probability": cg_observables[
            "survival_probability"
        ],
        "full_norms": full_observables["norms"],
        "cg_norms": cg_observables["norms"],
        **distance,
    }


def plot_results(results, output):
    times = results["times_fs"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    for state, color in enumerate(("tab:blue", "tab:orange"), start=1):
        axes[0].plot(
            times,
            results["full_populations"][:, state - 1],
            color=color,
            label=rf"full $S_{state}$",
        )
        axes[0].plot(
            times,
            results["cg_populations"][:, state - 1],
            color=color,
            linestyle="--",
            label=rf"CGLDR $S_{state}$",
        )
    for coordinate, label, color in (
        (0, r"$\langle Q_{6a}\rangle$", "tab:purple"),
        (1, r"$\langle Q_{10a}\rangle$", "tab:green"),
    ):
        axes[1].plot(
            times,
            results["full_coordinate_means"][:, coordinate],
            color=color,
            label=f"full {label}",
        )
        axes[1].plot(
            times,
            results["cg_coordinate_means"][:, coordinate],
            color=color,
            linestyle="--",
            label=f"CGLDR {label}",
        )
    axes[2].plot(
        times,
        results["total_variation"],
        color="tab:red",
    )
    axes[0].set(ylabel="Diabatic population", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Coordinate mean")
    axes[2].set(
        xlabel="Time / fs",
        ylabel="Nuclear-density\nTV distance",
        ylim=(
            -0.02 * max(float(np.max(results["total_variation"])), 1.0e-12),
            1.08 * max(float(np.max(results["total_variation"])), 1.0e-12),
        ),
    )
    axes[0].legend(ncols=2, fontsize="small")
    axes[1].legend(ncols=2, fontsize="small")
    for axis in axes:
        axis.grid(alpha=0.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tuning", type=int, default=64)
    parser.add_argument("--n-coupling", type=int, default=40)
    parser.add_argument("--domain", type=float, default=8.0)
    parser.add_argument("--time-step", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-every", type=int, default=10)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyrazine_two_mode_cgldr.npz"),
    )
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    results = run_benchmark(
        npts=(args.n_tuning, args.n_coupling),
        domains=((-args.domain, args.domain),) * 2,
        time_step=args.time_step,
        steps=args.steps,
        output_every=args.output_every,
        max_rank=args.max_rank,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **results)
    print("final full populations:", results["full_populations"][-1])
    print("final CGLDR populations:", results["cg_populations"][-1])
    print("maximum density TV:", np.max(results["total_variation"]))
    print("maximum mean error:", np.max(np.abs(
        results["full_coordinate_means"]
        - results["cg_coordinate_means"]
    )))
    print("full norm range:", np.min(results["full_norms"]), np.max(results["full_norms"]))
    print("CGLDR norm range:", np.min(results["cg_norms"]), np.max(results["cg_norms"]))
    print("saved:", args.output)
    if args.plot is not None:
        plot_results(results, args.plot)
        print("plot:", args.plot)


if __name__ == "__main__":
    main()
