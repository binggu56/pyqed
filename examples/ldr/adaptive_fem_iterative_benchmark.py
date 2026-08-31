"""Test iterative solve--estimate--refine at a fixed P2 node budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.ldr.adaptive_fem_residual_convergence import METHODS
from examples.ldr.graph_ldr_jahn_teller import (
    adaptive_fem_ldr_dynamics,
    fem_ldr_dynamics,
    full_ldr_dynamics,
)


def _errors(result, reference):
    population = result["adiabatic_populations"] - reference[
        "adiabatic_populations"
    ]
    x_mean = result["coordinate_means"][:, 0] - reference[
        "coordinate_means"
    ][:, 0]
    return float(np.sqrt(np.mean(population**2))), float(
        np.sqrt(np.mean(x_mean**2))
    )


def run_benchmark(*, target_nodes=650, cycles=(1, 2, 3, 4), final_time=15.0):
    """Compare repeated pilot adaptation at one final mesh budget."""

    dt = 0.02
    common = {
        "extent": 5.0,
        "dt": dt,
        "nsteps": int(round(final_time / dt)),
        "nout": 10,
    }
    start = perf_counter()
    reference = full_ldr_dynamics(**common, ncart=31)
    reference_seconds = perf_counter() - start
    uniform = fem_ldr_dynamics(
        **common,
        nr=8,
        ntheta=24,
        order=2,
        geometry="polar",
    )
    uniform_population_rmse, uniform_x_rmse = _errors(uniform, reference)

    results = {}
    records = []
    for method, options in METHODS.items():
        results[method] = []
        for cycle_count in cycles:
            start = perf_counter()
            result = adaptive_fem_ldr_dynamics(
                **common,
                nr=6,
                ntheta=12,
                cycles=int(cycle_count),
                target_nodes=int(target_nodes),
                electronic_weight=options["electronic_weight"],
            )
            seconds = perf_counter() - start
            population_rmse, x_rmse = _errors(result, reference)
            results[method].append(result)
            records.append(
                {
                    "method": method,
                    "cycles": int(cycle_count),
                    "nodes": result["mesh"].size,
                    "population_rmse": population_rmse,
                    "x_rmse": x_rmse,
                    "max_norm_error": result["max_norm_error"],
                    "seconds_including_pilots": seconds,
                }
            )
    metadata = {
        "target_nodes": int(target_nodes),
        "final_time": final_time,
        "dt": dt,
        "reference_nodes": reference["solver"].ngrid,
        "reference_seconds": reference_seconds,
        "reference_max_norm_error": reference["max_norm_error"],
        "uniform_nodes": uniform["mesh"].size,
        "uniform_population_rmse": uniform_population_rmse,
        "uniform_x_rmse": uniform_x_rmse,
        "uniform_max_norm_error": uniform["max_norm_error"],
        "records": records,
    }
    return results, uniform, reference, metadata


def plot_benchmark(results, uniform, reference, metadata, filename):
    """Plot iteration dependence, dynamics, and equal-budget meshes."""

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    records = metadata["records"]
    figure, axes = plt.subplot_mosaic(
        [
            ["population", "population", "coordinate", "coordinate"],
            ["dynamics", "dynamics", "one_mesh", "best_mesh"],
        ],
        figsize=(13.2, 7.4),
        constrained_layout=True,
    )
    for axis_name, error_name, ylabel in (
        ("population", "population_rmse", "population RMSE"),
        ("coordinate", "x_rmse", r"$\langle x\rangle$ RMSE"),
    ):
        axis = axes[axis_name]
        for method, options in METHODS.items():
            selected = [record for record in records if record["method"] == method]
            axis.plot(
                [record["cycles"] for record in selected],
                [record[error_name] for record in selected],
                marker=options["marker"],
                color=options["color"],
                label=method,
            )
        uniform_error = metadata[
            "uniform_population_rmse" if error_name == "population_rmse" else "uniform_x_rmse"
        ]
        axis.axhline(
            uniform_error,
            color="tab:blue",
            linestyle="--",
            label=f"uniform P2 ({metadata['uniform_nodes']})",
        )
        axis.set(xlabel="adaptation cycles", ylabel=ylabel, xticks=(1, 2, 3, 4))
        axis.grid(alpha=0.25)
    axes["population"].set_title("Electronic-population error")
    axes["coordinate"].set_title("Wavepacket-position error")
    axes["population"].legend(frameon=False, ncol=2)

    best_record = min(records, key=lambda record: record["population_rmse"])
    best_index = best_record["cycles"] - 1
    best = results[best_record["method"]][best_index]
    one_shot = results[best_record["method"]][0]
    times = reference["solver"].times
    dynamics = axes["dynamics"]
    dynamics.plot(
        times,
        reference["adiabatic_populations"][:, 1],
        color="black",
        linewidth=2.0,
        label="full LDR",
    )
    dynamics.plot(
        times,
        uniform["adiabatic_populations"][:, 1],
        "--",
        color="tab:blue",
        label=f"uniform P2 ({uniform['mesh'].size})",
    )
    dynamics.plot(
        times,
        one_shot["adiabatic_populations"][:, 1],
        color="tab:orange",
        alpha=0.65,
        label=f"one-shot {best_record['method']} ({one_shot['mesh'].size})",
    )
    dynamics.plot(
        times,
        best["adiabatic_populations"][:, 1],
        color="tab:green",
        label=(
            f"{best_record['cycles']}-cycle {best_record['method']} "
            f"({best['mesh'].size})"
        ),
    )
    dynamics.set(
        xlabel="time",
        ylabel="upper adiabatic population",
        title="Best iterative result at fixed budget",
        ylim=(-0.02, 0.58),
    )
    dynamics.grid(alpha=0.2)
    dynamics.legend(frameon=False, ncol=2)

    for result, axis_name, title in (
        (one_shot, "one_mesh", "one-shot mesh"),
        (best, "best_mesh", f"{best_record['cycles']}-cycle mesh"),
    ):
        mesh = result["mesh"]
        triangulation = mtri.Triangulation(
            mesh.nodes[: mesh.vertex_count, 0],
            mesh.nodes[: mesh.vertex_count, 1],
            mesh.vertex_triangles,
        )
        axis = axes[axis_name]
        axis.triplot(triangulation, color="0.35", linewidth=0.42)
        axis.set(
            xlabel="x",
            ylabel="y",
            title=f"{title}: {mesh.size} nodes",
            xlim=(-5.3, 5.3),
            ylim=(-5.3, 5.3),
        )
        axis.set_aspect("equal")

    figure.suptitle(
        "Iterative solve--estimate--refine for quadratic FEM dynamics"
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=190)
    plt.close(figure)
    return filename, best_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/adaptive_p2_iterative"),
    )
    parser.add_argument("--target-nodes", type=int, default=650)
    parser.add_argument("--final-time", type=float, default=15.0)
    args = parser.parse_args(argv)

    results, uniform, reference, metadata = run_benchmark(
        target_nodes=args.target_nodes,
        final_time=args.final_time,
    )
    figure, best_record = plot_benchmark(
        results,
        uniform,
        reference,
        metadata,
        args.output_dir / "adaptive_p2_iterative_benchmark.png",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata["best"] = best_record
    summary = args.output_dir / "adaptive_p2_iterative_benchmark.json"
    with summary.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    np.savez_compressed(
        args.output_dir / "adaptive_p2_iterative_benchmark.npz",
        times=reference["solver"].times,
        method=np.asarray([record["method"] for record in metadata["records"]]),
        cycles=np.asarray([record["cycles"] for record in metadata["records"]]),
        nodes=np.asarray([record["nodes"] for record in metadata["records"]]),
        population_rmse=np.asarray(
            [record["population_rmse"] for record in metadata["records"]]
        ),
        x_rmse=np.asarray([record["x_rmse"] for record in metadata["records"]]),
        reference_populations=reference["adiabatic_populations"],
        uniform_populations=uniform["adiabatic_populations"],
    )
    print(json.dumps(metadata, indent=2))
    print(f"Figure: {figure}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
