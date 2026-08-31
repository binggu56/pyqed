#!/usr/bin/env python3
"""Plot timing and energy accuracy for the phenol SA-CASSCF optimization."""

from pyqed.units import au2ev
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = au2ev


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("hessian", type=Path)
    parser.add_argument("native_hessian", type=Path)
    parser.add_argument("rdm", type=Path)
    parser.add_argument("--native-repeat", action="append", type=Path, default=[])
    parser.add_argument("--hessian-repeat", action="append", type=Path, default=[])
    parser.add_argument("--native-hessian-repeat", action="append", type=Path, default=[])
    parser.add_argument("--rdm-repeat", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    baseline = load(args.baseline)
    previous = load(args.previous)
    native = load(args.native)
    hessian = load(args.hessian)
    native_hessian = load(args.native_hessian)
    rdm = load(args.rdm)
    native_runs = [native] + [load(path) for path in args.native_repeat]
    hessian_runs = [hessian] + [load(path) for path in args.hessian_repeat]
    native_hessian_runs = [native_hessian] + [
        load(path) for path in args.native_hessian_repeat
    ]
    rdm_runs = [rdm] + [load(path) for path in args.rdm_repeat]
    records = (reference, baseline, previous, native, hessian, native_hessian, rdm)
    labels = (
        "PySCF",
        "PyQED\noriginal",
        "PyQED\npre-native",
        "PyQED\nnative\nCI",
        "PyQED\nbatched\nAH",
        "PyQED\nnative\nHessian",
        "PyQED\nnative\nRDM",
    )
    colors = (
        "#4c78a8",
        "#e45756",
        "#7b6fd0",
        "#2a9d8f",
        "#e9a23b",
        "#264653",
        "#6a4c93",
    )
    times = np.asarray([float(record["wall_seconds"]) for record in records])
    native_times = np.asarray([float(record["wall_seconds"]) for record in native_runs])
    hessian_times = np.asarray([float(record["wall_seconds"]) for record in hessian_runs])
    native_hessian_times = np.asarray(
        [float(record["wall_seconds"]) for record in native_hessian_runs]
    )
    rdm_times = np.asarray([float(record["wall_seconds"]) for record in rdm_runs])
    times[3] = np.mean(native_times)
    times[4] = np.mean(hessian_times)
    times[5] = np.mean(native_hessian_times)
    times[6] = np.mean(rdm_times)

    reference_excitation = reference["energies"] - reference["energies"][0]
    errors = []
    for record in (baseline, previous, native, hessian, native_hessian, rdm):
        excitation = record["energies"] - record["energies"][0]
        errors.append((excitation - reference_excitation) * HARTREE_TO_EV * 1000.0)
    errors = np.asarray(errors)

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
    axes[0].bar(np.arange(7), times, color=colors, width=0.66)
    axes[0].errorbar(
        3,
        times[3],
        yerr=np.asarray([[times[3] - np.min(native_times)], [np.max(native_times) - times[3]]]),
        fmt="none",
        color="0.15",
        capsize=3,
        lw=1.0,
    )
    axes[0].errorbar(
        4,
        times[4],
        yerr=np.asarray(
            [[times[4] - np.min(hessian_times)], [np.max(hessian_times) - times[4]]]
        ),
        fmt="none",
        color="0.15",
        capsize=3,
        lw=1.0,
    )
    axes[0].errorbar(
        5,
        times[5],
        yerr=np.asarray(
            [
                [times[5] - np.min(native_hessian_times)],
                [np.max(native_hessian_times) - times[5]],
            ]
        ),
        fmt="none",
        color="0.15",
        capsize=3,
        lw=1.0,
    )
    axes[0].errorbar(
        6,
        times[6],
        yerr=np.asarray(
            [[times[6] - np.min(rdm_times)], [np.max(rdm_times) - times[6]]]
        ),
        fmt="none",
        color="0.15",
        capsize=3,
        lw=1.0,
    )
    axes[0].set_yscale("log")
    axes[0].set(
        xticks=np.arange(7),
        xticklabels=labels,
        ylabel="Wall time (s, log scale)",
        title="Equilibrium SA(6)-CASSCF benchmark",
    )
    axes[0].set_ylim(7.0, 650.0)
    for index, value in enumerate(times):
        axes[0].text(index, value * 1.08, f"{value:.1f} s", ha="center", va="bottom")
    axes[0].text(
        0.98,
        0.86,
        f"native RDM: {times[5] / times[6]:.2f}$\\times$\n"
        f"total: {times[1] / times[6]:.2f}$\\times$",
        ha="right",
        va="top",
        transform=axes[0].transAxes,
    )

    states = np.arange(reference["energies"].size)
    axes[1].axhline(0.0, color="0.45", lw=0.9)
    axes[1].plot(states, errors[0], "o--", color=colors[1], label="Original")
    axes[1].plot(
        states,
        errors[5],
        "x-",
        color=colors[6],
        markersize=7,
        markeredgewidth=1.5,
        label="Native Hessian + RDM",
    )
    axes[1].set(
        xticks=states,
        xticklabels=[f"S{state}" for state in states],
        xlabel="Electronic state",
        ylabel="Excitation-energy error (meV)",
        title="Agreement with PySCF",
    )
    axes[1].set_ylim(-0.18, 0.18)
    axes[1].text(
        0.98,
        0.95,
        f"latest max |error| = {np.max(np.abs(errors[5])):.2f} meV\n"
        f"(<1 meV target)",
        ha="right",
        va="top",
        transform=axes[1].transAxes,
    )
    axes[1].legend(frameon=False, loc="lower left")
    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.13, 1.04, label, weight="bold", transform=axis.transAxes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=360)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)

    summary = {
        "wall_seconds": dict(
            zip(
                (
                    "pyscf",
                    "pyqed_original",
                    "pyqed_pre_native",
                    "pyqed_native",
                    "pyqed_batched_hessian",
                    "pyqed_native_hessian",
                    "pyqed_native_rdm",
                ),
                times,
            )
        ),
        "native_incremental_speedup": float(times[2] / times[3]),
        "hessian_incremental_speedup": float(times[3] / times[4]),
        "native_hessian_incremental_speedup": float(times[4] / times[5]),
        "native_rdm_incremental_speedup": float(times[5] / times[6]),
        "total_pyqed_speedup": float(times[1] / times[6]),
        "remaining_pyqed_pyscf_ratio": float(times[6] / times[0]),
        "maximum_error_mev_original": float(np.max(np.abs(errors[0]))),
        "maximum_error_mev_pre_native": float(np.max(np.abs(errors[1]))),
        "maximum_error_mev_native": float(np.max(np.abs(errors[2]))),
        "maximum_error_mev_batched_hessian": float(np.max(np.abs(errors[3]))),
        "maximum_error_mev_native_hessian": float(np.max(np.abs(errors[4]))),
        "maximum_error_mev_native_rdm": float(np.max(np.abs(errors[5]))),
        "native_wall_seconds_runs": native_times.tolist(),
        "hessian_wall_seconds_runs": hessian_times.tolist(),
        "native_hessian_wall_seconds_runs": native_hessian_times.tolist(),
        "native_rdm_wall_seconds_runs": rdm_times.tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
