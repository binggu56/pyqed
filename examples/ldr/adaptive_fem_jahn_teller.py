"""Benchmark a pilot-adapted P2 FEM mesh on Jahn--Teller dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.ldr.graph_ldr_jahn_teller import (
    adaptive_fem_ldr_dynamics,
    fem_ldr_dynamics,
    full_ldr_dynamics,
)


def _timed(function, **kwargs):
    start = perf_counter()
    result = function(**kwargs)
    return result, perf_counter() - start


def run_benchmark(
    *,
    extent=5.0,
    dt=0.02,
    final_time=15.0,
    nout=10,
    adaptive_nr=6,
    adaptive_ntheta=12,
    adaptive_cycles=1,
    matched_nr=8,
    matched_ntheta=16,
    uniform_nr=11,
    uniform_ntheta=24,
    reference_ncart=31,
):
    """Run adaptive P2, two uniform P2 meshes, and full-LDR dynamics."""

    nsteps = int(round(float(final_time) / float(dt)))
    common = {
        "extent": extent,
        "dt": dt,
        "nsteps": nsteps,
        "nout": nout,
    }
    adaptive, adaptive_seconds = _timed(
        adaptive_fem_ldr_dynamics,
        **common,
        nr=adaptive_nr,
        ntheta=adaptive_ntheta,
        cycles=adaptive_cycles,
    )
    matched, matched_seconds = _timed(
        fem_ldr_dynamics,
        **common,
        nr=matched_nr,
        ntheta=matched_ntheta,
        order=2,
        geometry="polar",
    )
    uniform, uniform_seconds = _timed(
        fem_ldr_dynamics,
        **common,
        nr=uniform_nr,
        ntheta=uniform_ntheta,
        order=2,
        geometry="polar",
    )
    reference, reference_seconds = _timed(
        full_ldr_dynamics,
        **common,
        ncart=reference_ncart,
    )

    adaptive_population_error = (
        adaptive["adiabatic_populations"] - reference["adiabatic_populations"]
    )
    uniform_population_error = (
        uniform["adiabatic_populations"] - reference["adiabatic_populations"]
    )
    matched_population_error = (
        matched["adiabatic_populations"] - reference["adiabatic_populations"]
    )
    adaptive_x_error = (
        adaptive["coordinate_means"][:, 0] - reference["coordinate_means"][:, 0]
    )
    uniform_x_error = (
        uniform["coordinate_means"][:, 0] - reference["coordinate_means"][:, 0]
    )
    matched_x_error = (
        matched["coordinate_means"][:, 0] - reference["coordinate_means"][:, 0]
    )
    summary = {
        "adaptive_nodes": adaptive["mesh"].size,
        "adaptive_vertices": adaptive["mesh"].vertex_count,
        "adaptive_triangles": len(adaptive["mesh"].vertex_triangles),
        "adaptive_population_rmse": float(
            np.sqrt(np.mean(adaptive_population_error**2))
        ),
        "adaptive_x_rmse": float(np.sqrt(np.mean(adaptive_x_error**2))),
        "adaptive_max_norm_error": adaptive["max_norm_error"],
        "adaptive_seconds_including_pilot": adaptive_seconds,
        "matched_uniform_nodes": matched["mesh"].size,
        "matched_uniform_population_rmse": float(
            np.sqrt(np.mean(matched_population_error**2))
        ),
        "matched_uniform_x_rmse": float(np.sqrt(np.mean(matched_x_error**2))),
        "matched_uniform_max_norm_error": matched["max_norm_error"],
        "matched_uniform_seconds": matched_seconds,
        "uniform_nodes": uniform["mesh"].size,
        "uniform_population_rmse": float(
            np.sqrt(np.mean(uniform_population_error**2))
        ),
        "uniform_x_rmse": float(np.sqrt(np.mean(uniform_x_error**2))),
        "uniform_max_norm_error": uniform["max_norm_error"],
        "uniform_seconds": uniform_seconds,
        "reference_nodes": reference["solver"].ngrid,
        "reference_max_norm_error": reference["max_norm_error"],
        "reference_seconds": reference_seconds,
        "dt": dt,
        "final_time": final_time,
    }
    return adaptive, matched, uniform, reference, summary


def plot_benchmark(adaptive, matched, uniform, reference, summary, filename):
    """Plot the adapted mesh, dynamics, and reference errors."""

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh = adaptive["mesh"]
    triangulation = mtri.Triangulation(
        mesh.nodes[: mesh.vertex_count, 0],
        mesh.nodes[: mesh.vertex_count, 1],
        mesh.vertex_triangles,
    )
    times = reference["solver"].times
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)

    axes[0, 0].triplot(triangulation, color="0.55", linewidth=0.45)
    axes[0, 0].scatter(
        mesh.nodes[: mesh.vertex_count, 0],
        mesh.nodes[: mesh.vertex_count, 1],
        s=7,
        label="vertices",
    )
    axes[0, 0].scatter(
        mesh.nodes[mesh.vertex_count :, 0],
        mesh.nodes[mesh.vertex_count :, 1],
        s=3,
        alpha=0.65,
        label="P2 midside nodes",
    )
    axes[0, 0].set(
        xlabel="x",
        ylabel="y",
        title=(
            f"Adapted mesh: {mesh.size} P2 nodes, "
            f"{len(mesh.vertex_triangles)} triangles"
        ),
    )
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(frameon=False, markerscale=1.8)

    axes[0, 1].plot(
        times,
        reference["adiabatic_populations"][:, 1],
        color="black",
        linewidth=2.0,
        label=f"full LDR ({summary['reference_nodes']})",
    )
    axes[0, 1].plot(
        times,
        matched["adiabatic_populations"][:, 1],
        ":",
        color="tab:green",
        label=f"uniform P2 ({summary['matched_uniform_nodes']})",
    )
    axes[0, 1].plot(
        times,
        uniform["adiabatic_populations"][:, 1],
        "--",
        color="tab:blue",
        label=f"uniform P2 ({summary['uniform_nodes']})",
    )
    axes[0, 1].plot(
        times,
        adaptive["adiabatic_populations"][:, 1],
        color="tab:orange",
        label=f"adaptive P2 ({summary['adaptive_nodes']})",
    )
    axes[0, 1].set(
        xlabel="time",
        ylabel="upper adiabatic population",
        title="Nonadiabatic population transfer",
        ylim=(-0.02, 1.02),
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        times,
        reference["coordinate_means"][:, 0],
        color="black",
        linewidth=2.0,
        label="full LDR",
    )
    axes[1, 0].plot(
        times,
        matched["coordinate_means"][:, 0],
        ":",
        color="tab:green",
        label="matched uniform P2",
    )
    axes[1, 0].plot(
        times,
        uniform["coordinate_means"][:, 0],
        "--",
        color="tab:blue",
        label="uniform P2",
    )
    axes[1, 0].plot(
        times,
        adaptive["coordinate_means"][:, 0],
        color="tab:orange",
        label="adaptive P2",
    )
    axes[1, 0].set(
        xlabel="time",
        ylabel=r"$\langle x\rangle$",
        title="Wavepacket motion",
    )
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)

    adaptive_error = np.abs(
        adaptive["adiabatic_populations"][:, 1]
        - reference["adiabatic_populations"][:, 1]
    )
    uniform_error = np.abs(
        uniform["adiabatic_populations"][:, 1]
        - reference["adiabatic_populations"][:, 1]
    )
    matched_error = np.abs(
        matched["adiabatic_populations"][:, 1]
        - reference["adiabatic_populations"][:, 1]
    )
    floor = np.finfo(float).eps
    axes[1, 1].semilogy(
        times,
        np.maximum(adaptive_error, floor),
        color="tab:orange",
        label=(
            "adaptive P2, RMSE="
            f"{summary['adaptive_population_rmse']:.3f}"
        ),
    )
    axes[1, 1].semilogy(
        times,
        np.maximum(matched_error, floor),
        ":",
        color="tab:green",
        label=(
            "matched uniform, RMSE="
            f"{summary['matched_uniform_population_rmse']:.3f}"
        ),
    )
    axes[1, 1].semilogy(
        times,
        np.maximum(uniform_error, floor),
        "--",
        color="tab:blue",
        label=f"uniform P2, RMSE={summary['uniform_population_rmse']:.3f}",
    )
    axes[1, 1].set(
        xlabel="time",
        ylabel="absolute population error",
        title="Error relative to full LDR",
    )
    axes[1, 1].grid(alpha=0.2, which="both")
    axes[1, 1].legend(frameon=False)

    figure.suptitle(
        "Pilot-adapted quadratic FEM for conical-intersection dynamics"
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=190)
    plt.close(figure)
    return filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/adaptive_p2_jahn_teller"),
    )
    parser.add_argument("--final-time", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adaptive, matched, uniform, reference, summary = run_benchmark(
        dt=args.dt,
        final_time=args.final_time,
        adaptive_cycles=args.cycles,
    )
    figure = plot_benchmark(
        adaptive,
        matched,
        uniform,
        reference,
        summary,
        args.output_dir / "adaptive_p2_jahn_teller_dynamics.png",
    )
    np.savez_compressed(
        args.output_dir / "adaptive_p2_jahn_teller_dynamics.npz",
        times=reference["solver"].times,
        adaptive_populations=adaptive["adiabatic_populations"],
        matched_uniform_populations=matched["adiabatic_populations"],
        uniform_populations=uniform["adiabatic_populations"],
        reference_populations=reference["adiabatic_populations"],
        adaptive_means=adaptive["coordinate_means"],
        matched_uniform_means=matched["coordinate_means"],
        uniform_means=uniform["coordinate_means"],
        reference_means=reference["coordinate_means"],
        adaptive_nodes=adaptive["mesh"].nodes,
        adaptive_triangles=adaptive["mesh"].vertex_triangles,
        adaptive_vertex_count=adaptive["mesh"].vertex_count,
    )
    with (args.output_dir / "adaptive_p2_jahn_teller_summary.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
