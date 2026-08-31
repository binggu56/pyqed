"""Compare residual, hybrid, and projector P2 adaptation by node budget."""

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


METHODS = {
    "residual": {"electronic_weight": 0.0, "color": "tab:red", "marker": "o"},
    "hybrid": {"electronic_weight": 0.5, "color": "tab:purple", "marker": "s"},
    "projector": {
        "electronic_weight": 1.0,
        "color": "tab:orange",
        "marker": "^",
    },
}
UNIFORM_MESHES = ((5, 12), (6, 16), (8, 16), (8, 24), (11, 24))


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


def run_study(*, budgets=(300, 450, 650, 850), dt=0.02, final_time=15.0):
    """Run one full-LDR reference and all adaptive and uniform meshes."""

    nsteps = int(round(float(final_time) / float(dt)))
    common = {"dt": dt, "nsteps": nsteps, "nout": 10, "extent": 5.0}
    start = perf_counter()
    reference = full_ldr_dynamics(**common, ncart=31)
    reference_seconds = perf_counter() - start

    adaptive = {}
    records = []
    for method, options in METHODS.items():
        adaptive[method] = []
        for budget in budgets:
            start = perf_counter()
            result = adaptive_fem_ldr_dynamics(
                **common,
                nr=6,
                ntheta=12,
                cycles=1,
                target_nodes=int(budget),
                electronic_weight=options["electronic_weight"],
            )
            seconds = perf_counter() - start
            population_rmse, x_rmse = _errors(result, reference)
            adaptive[method].append(result)
            records.append(
                {
                    "kind": "adaptive",
                    "method": method,
                    "target_nodes": int(budget),
                    "nodes": result["mesh"].size,
                    "population_rmse": population_rmse,
                    "x_rmse": x_rmse,
                    "max_norm_error": result["max_norm_error"],
                    "seconds_including_pilot": seconds,
                }
            )

    uniform = []
    for nr, ntheta in UNIFORM_MESHES:
        start = perf_counter()
        result = fem_ldr_dynamics(
            **common,
            nr=nr,
            ntheta=ntheta,
            order=2,
            geometry="polar",
        )
        seconds = perf_counter() - start
        population_rmse, x_rmse = _errors(result, reference)
        uniform.append(result)
        records.append(
            {
                "kind": "uniform",
                "method": "uniform",
                "nr": nr,
                "ntheta": ntheta,
                "nodes": result["mesh"].size,
                "population_rmse": population_rmse,
                "x_rmse": x_rmse,
                "max_norm_error": result["max_norm_error"],
                "seconds": seconds,
            }
        )
    metadata = {
        "dt": dt,
        "final_time": final_time,
        "reference_nodes": reference["solver"].ngrid,
        "reference_seconds": reference_seconds,
        "reference_max_norm_error": reference["max_norm_error"],
        "records": records,
    }
    return adaptive, uniform, reference, metadata


def plot_study(adaptive, uniform, reference, metadata, filename):
    """Plot convergence, representative dynamics, and adapted meshes."""

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    records = metadata["records"]
    figure, axes = plt.subplot_mosaic(
        [
            ["population", "population", "coordinate", "coordinate"],
            ["dynamics", "dynamics", "residual_mesh", "projector_mesh"],
        ],
        figsize=(13.2, 7.4),
        constrained_layout=True,
    )

    uniform_records = [record for record in records if record["kind"] == "uniform"]
    uniform_nodes = np.asarray([record["nodes"] for record in uniform_records])
    ordering = np.argsort(uniform_nodes)
    for axis_name, error_name, ylabel in (
        ("population", "population_rmse", "population RMSE"),
        ("coordinate", "x_rmse", r"$\langle x\rangle$ RMSE"),
    ):
        axis = axes[axis_name]
        axis.semilogy(
            uniform_nodes[ordering],
            np.asarray([record[error_name] for record in uniform_records])[ordering],
            "D--",
            color="tab:blue",
            label="uniform P2",
        )
        for method, options in METHODS.items():
            selected = [
                record
                for record in records
                if record["kind"] == "adaptive" and record["method"] == method
            ]
            nodes = np.asarray([record["nodes"] for record in selected])
            values = np.asarray([record[error_name] for record in selected])
            order = np.argsort(nodes)
            axis.semilogy(
                nodes[order],
                values[order],
                marker=options["marker"],
                color=options["color"],
                label=f"{method} adaptive",
            )
        axis.set(xlabel="P2 nodes", ylabel=ylabel)
        axis.grid(alpha=0.25, which="both")
    axes["population"].set_title("Electronic-population convergence")
    axes["coordinate"].set_title("Wavepacket-position convergence")
    axes["population"].legend(frameon=False, ncol=2)

    budgets = [result["mesh"].size for result in adaptive["hybrid"]]
    representative = int(np.argmin(np.abs(np.asarray(budgets) - 450)))
    times = reference["solver"].times
    dynamics_axis = axes["dynamics"]
    dynamics_axis.plot(
        times,
        reference["adiabatic_populations"][:, 1],
        color="black",
        linewidth=2.0,
        label="full LDR",
    )
    matched_uniform = min(uniform, key=lambda result: abs(result["mesh"].size - 450))
    dynamics_axis.plot(
        times,
        matched_uniform["adiabatic_populations"][:, 1],
        "--",
        color="tab:blue",
        label=f"uniform P2 ({matched_uniform['mesh'].size})",
    )
    for method, options in METHODS.items():
        result = adaptive[method][representative]
        dynamics_axis.plot(
            times,
            result["adiabatic_populations"][:, 1],
            color=options["color"],
            label=f"{method} ({result['mesh'].size})",
        )
    dynamics_axis.set(
        xlabel="time",
        ylabel="upper adiabatic population",
        title="Representative 450-node dynamics",
        ylim=(-0.02, 0.58),
    )
    dynamics_axis.grid(alpha=0.2)
    dynamics_axis.legend(frameon=False, ncol=2)

    for method, axis_name in (
        ("residual", "residual_mesh"),
        ("projector", "projector_mesh"),
    ):
        result = adaptive[method][representative]
        mesh = result["mesh"]
        triangulation = mtri.Triangulation(
            mesh.nodes[: mesh.vertex_count, 0],
            mesh.nodes[: mesh.vertex_count, 1],
            mesh.vertex_triangles,
        )
        axis = axes[axis_name]
        axis.triplot(triangulation, color=METHODS[method]["color"], linewidth=0.45)
        axis.set(
            xlabel="x",
            ylabel="y",
            title=f"{method}: {mesh.size} nodes",
            xlim=(-5.3, 5.3),
            ylim=(-5.3, 5.3),
        )
        axis.set_aspect("equal")

    figure.suptitle(
        "Residual and electronic-projector adaptation for P2 FEM dynamics"
    )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=190)
    plt.close(figure)
    return filename


def save_outputs(adaptive, uniform, reference, metadata, output_dir):
    """Save summary, convergence arrays, and representative trajectories."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "adaptive_p2_residual_convergence.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    records = metadata["records"]
    np.savez_compressed(
        output_dir / "adaptive_p2_residual_convergence.npz",
        times=reference["solver"].times,
        record_kind=np.asarray([record["kind"] for record in records]),
        record_method=np.asarray([record["method"] for record in records]),
        nodes=np.asarray([record["nodes"] for record in records]),
        population_rmse=np.asarray([record["population_rmse"] for record in records]),
        x_rmse=np.asarray([record["x_rmse"] for record in records]),
        reference_populations=reference["adiabatic_populations"],
        reference_means=reference["coordinate_means"],
    )
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/adaptive_p2_residual_convergence"),
    )
    parser.add_argument("--budgets", default="300,450,650,850")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=15.0)
    args = parser.parse_args(argv)
    budgets = tuple(int(value) for value in args.budgets.split(","))

    adaptive, uniform, reference, metadata = run_study(
        budgets=budgets,
        dt=args.dt,
        final_time=args.final_time,
    )
    figure = plot_study(
        adaptive,
        uniform,
        reference,
        metadata,
        args.output_dir / "adaptive_p2_residual_convergence.png",
    )
    summary = save_outputs(
        adaptive,
        uniform,
        reference,
        metadata,
        args.output_dir,
    )
    print(json.dumps(metadata, indent=2))
    print(f"Figure: {figure}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
