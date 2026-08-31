#!/usr/bin/env python3
"""Compare full-LDR and explicitly propagated multipatch-gauge SO2 dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from scipy.linalg import eigh
import ultraplot as uplt

from examples.ldr.so2_casci_cgldr import DEFAULT_SCAN_DIR, load_so2_linked_scan
from examples.ldr.so2_casci_cgldr_dense import (
    dense_kinetic,
    nuclear_packet,
    observables,
)
from examples.ldr.so2_casci_full_ldr import full_hamiltonian, path_overlap
from examples.ldr.so2_procrustes_gauge import gauged_hamiltonian
from pyqed.ldr.overlap import unpack
from pyqed.units import au2fs


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_GAUGE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
    "procrustes_gauge.npz"
)


def propagate(hamiltonian, state, times_fs):
    values, vectors = eigh(hamiltonian, overwrite_a=True, check_finite=False)
    coefficients = vectors.conj().T @ state
    phases = np.exp(-1j * np.outer(times_fs / au2fs, values))
    return (phases * coefficients[None, :]) @ vectors.conj().T


def transform_states(states, gauge):
    """Transform patch-gauge coefficients back to the original local frames."""
    gauge = np.asarray(gauge, dtype=complex).reshape(-1, *gauge.shape[-2:])
    reshaped = states.reshape(len(states), len(gauge), gauge.shape[-1])
    return np.einsum("gia,tga->tgi", gauge, reshaped, optimize=True).reshape(
        states.shape
    )


def plot_dynamics(output_dir, times, exact, atlas, errors, *, gauge_label, stem):
    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
            "savefig.transparent": False,
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=3,
        width=8.2,
        height=4.65,
        share=False,
        wspace=1.5,
        hspace=2.0,
    )
    panels = (
        (exact[1][:, 1], atlas[1][:, 1], r"$P_1$"),
        (exact[1][:, 2], atlas[1][:, 2], r"$P_2$"),
        (exact[2][:, 0], atlas[2][:, 0], r"$\langle q_s\rangle$ (bohr)"),
        (
            np.rad2deg(exact[2][:, 1]),
            np.rad2deg(atlas[2][:, 1]),
            r"$\langle\theta\rangle$ (deg)",
        ),
        (exact[2][:, 2], atlas[2][:, 2], r"$\langle q_a\rangle$ (bohr)"),
    )
    for axis, (reference, patched, ylabel) in zip(axes[:5], panels):
        axis.plot(times, reference, color="black", label="Full LDR")
        axis.plot(
            times,
            patched,
            color="#D55E00",
            linestyle="--",
            label=gauge_label,
        )
        axis.format(ylabel=ylabel, grid=False, tickdir="out")
    axes[5].semilogy(
        times,
        np.maximum(errors["state"], 1.0e-16),
        color="#0072B2",
        label=r"$\|\Delta\Psi\|_\infty$",
    )
    axes[5].semilogy(
        times,
        np.maximum(errors["population"], 1.0e-16),
        color="#D55E00",
        linestyle="--",
        label=r"$\|\Delta P\|_\infty$",
    )
    axes[5].semilogy(
        times,
        np.maximum(errors["coordinate"], 1.0e-16),
        color="#009E73",
        linestyle=":",
        label="scaled coordinate error",
    )
    axes[5].format(ylabel="Absolute error", grid=False, tickdir="out")
    axes[5].legend(frame=False, loc="best")
    for axis in axes[3:]:
        axis.format(xlabel="Time (fs)")
    axes[2].format(xlabel="Time (fs)")
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="top", ncols=2, frame=False)
    for panel, axis in enumerate(axes):
        axis.text(
            0.025,
            0.965,
            "abcdef"[panel],
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
    output = output_dir / stem
    figure.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(output.with_suffix(".png"))
    print(output.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GAUGE.parent)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        grids = tuple(np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)
        patch_boundary = int(archive["patch_boundary_theta_index"])
    gauge_label = (
        "Two-patch gauge" if patch_boundary >= 0 else "Single-patch gauge"
    )
    stem = (
        "so2_two_patch_dynamics"
        if patch_boundary >= 0
        else "so2_single_patch_dynamics"
    )
    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    ngrid = int(np.prod(shape))
    overlap = path_overlap(shape, links).reshape(ngrid, nstates, ngrid, nstates)
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, axes = dense_kinetic(scan, *grids)
    exact_hamiltonian = full_hamiltonian(kinetic, overlap, energies)
    atlas_hamiltonian, aligned_overlap, local = gauged_hamiltonian(
        kinetic,
        overlap,
        energies,
        gauge.reshape(ngrid, nstates, nstates),
    )
    del aligned_overlap, local

    packet = nuclear_packet(*grids, axes)
    original_initial = (
        packet[..., None] * primary_gauge[..., args.initial_state]
    ).reshape(-1)
    original_initial /= np.linalg.norm(original_initial)
    patch_initial = np.einsum(
        "...ia,...i->...a",
        gauge.conj(),
        original_initial.reshape(*shape, nstates),
        optimize=True,
    ).reshape(-1)
    times = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)
    exact_states = propagate(exact_hamiltonian, original_initial, times)
    atlas_states = propagate(atlas_hamiltonian, patch_initial, times)
    physical_atlas_states = transform_states(atlas_states, gauge)

    transport = primary_gauge
    exact_observables = observables(exact_states, grids, transport)
    atlas_observables = observables(physical_atlas_states, grids, transport)
    state_error = np.max(np.abs(exact_states - physical_atlas_states), axis=1)
    population_error = np.max(
        np.abs(exact_observables[1] - atlas_observables[1]), axis=1
    )
    spans = np.asarray([grid[-1] - grid[0] for grid in grids])
    coordinate_error = np.max(
        np.abs(exact_observables[2] - atlas_observables[2]) / spans[None, :],
        axis=1,
    )
    state_overlap = np.abs(
        np.einsum(
            "ti,ti->t",
            exact_states.conj(),
            physical_atlas_states,
            optimize=True,
        )
    ) ** 2
    exact_state_norm = np.sum(np.abs(exact_states) ** 2, axis=1)
    atlas_state_norm = np.sum(np.abs(physical_atlas_states) ** 2, axis=1)
    fidelity = np.clip(
        state_overlap / (exact_state_norm * atlas_state_norm),
        0.0,
        1.0,
    )
    summary = {
        "method": f"explicit full-LDR versus {gauge_label.lower()} propagation",
        "grid": list(shape),
        "time_fs": float(args.time_fs),
        "dt_fs": float(args.dt_fs),
        "max_state_coefficient_error": float(np.max(state_error)),
        "max_reference_population_error": float(np.max(population_error)),
        "max_scaled_coordinate_error": float(np.max(coordinate_error)),
        "minimum_wavefunction_fidelity": float(np.min(fidelity)),
        "max_exact_norm_error": float(np.max(np.abs(exact_observables[4] - 1.0))),
        "max_atlas_norm_error": float(np.max(np.abs(atlas_observables[4] - 1.0))),
    }
    np.savez(
        args.output_dir / f"{stem}.npz",
        times_fs=times,
        exact_reference_populations=exact_observables[1],
        atlas_reference_populations=atlas_observables[1],
        exact_means=exact_observables[2],
        atlas_means=atlas_observables[2],
        exact_norms=exact_observables[4],
        atlas_norms=atlas_observables[4],
        state_error=state_error,
        population_error=population_error,
        coordinate_error=coordinate_error,
        fidelity=fidelity,
    )
    with (args.output_dir / f"{stem}_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    plot_dynamics(
        args.output_dir,
        times,
        exact_observables,
        atlas_observables,
        {
            "state": state_error,
            "population": population_error,
            "coordinate": coordinate_error,
        },
        gauge_label=gauge_label,
        stem=stem,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
