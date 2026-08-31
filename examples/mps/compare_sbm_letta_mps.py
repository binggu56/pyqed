#!/usr/bin/env python3
"""Compare MPS and window-2 LETTA dynamics for the Drude spin-boson model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SBM = Path(
    "/Users/gugroup/Library/CloudStorage/OneDrive-西湖大学/research/SBM"
)


def load_backend(sbm_dir: Path):
    sys.path.insert(0, str(sbm_dir))
    import model
    import tdvp_dynamics
    from pyqed.mps.mps import MPO

    # research/SBM/tdvp.py uses the historical MPO ``cores`` interface.
    if not hasattr(MPO, "cores"):
        MPO.cores = property(
            lambda self: self.factors,
            lambda self, value: setattr(self, "factors", value),
        )
    return model, tdvp_dynamics


def configure(model, args):
    model.MODEL_TAG = (
        f"letta_mps_N{args.nmodes}_d{args.local_dim}_w{args.omega_max:g}"
        f"_dt{args.dt:g}_T{args.tmax:g}"
    )
    model.NMODES = args.nmodes
    model.LOCAL_DIM = args.local_dim
    model.QUADRATURE_ORDER = args.quadrature_order
    model.OMEGA_MAX = args.omega_max
    model.TN_DT = args.dt
    model.TMAX = args.tmax
    model.TDVP_REPORT_STEPS = max(1, int(round(0.5 / args.dt)))
    model.TDVP_FORCE_SWITCH_TIME = args.tmax + args.dt
    model.TDVP_SATURATION_STEPS = int(round(args.tmax / args.dt)) + 2


def load_case(path: Path):
    table = np.genfromtxt(path / "TDVP_observables.csv", delimiter=",", names=True)
    if table.ndim == 0:
        table = table.reshape(1)
    with (path / "TDVP_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(path / "TDVP_bond_diagnostics.npz") as payload:
        ranks = np.asarray(payload["ranks"], dtype=int)
    return table, metadata, ranks


def parameter_count(backend: str, dims: list[int], ranks: np.ndarray) -> int:
    left = np.r_[1, ranks]
    right = np.r_[ranks, 1]
    if backend == "mps":
        return int(sum(l * d * r for l, d, r in zip(left, dims, right)))
    total = 0
    for site, (l, r) in enumerate(zip(left, right)):
        physical = dims[site]
        if site + 1 < len(dims):
            physical *= dims[site + 1]
        total += int(l * physical * r)
    return total


def analyze(output: Path, cases: dict, dims: list[int], reference_key: str):
    reference, _, _ = cases[reference_key]
    ref_rho = reference["rho01_real"] + 1j * reference["rho01_imag"]
    rows = []
    for key, (table, metadata, ranks) in cases.items():
        rho = table["rho01_real"] + 1j * table["rho01_imag"]
        delta_sz = table["sigma_z"] - reference["sigma_z"]
        delta_rho = rho - ref_rho
        backend, rank_text = key.split("_d")
        rows.append(
            {
                "case": key,
                "backend": backend,
                "rank_cap": int(rank_text),
                "peak_parameters": max(
                    parameter_count(backend, dims, item) for item in ranks
                ),
                "final_max_rank": int(ranks[-1].max()),
                "max_sigma_z_error": float(np.max(np.abs(delta_sz))),
                "rms_sigma_z_error": float(np.sqrt(np.mean(delta_sz**2))),
                "max_rho01_error": float(np.max(np.abs(delta_rho))),
                "max_trace_error": float(np.max(np.abs(table["trace_error"]))),
                "wall_seconds": float(metadata["wall_seconds"]),
                "mean_seconds_per_step": float(metadata["mean_seconds_per_step"]),
            }
        )

    fields = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"reference": reference_key, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    return rows


def plot_trajectories(output: Path, cases: dict, reference_key: str):
    reference = cases[reference_key][0]
    ref_rho = reference["rho01_real"] + 1j * reference["rho01_imag"]
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for index, (key, (table, _, ranks)) in enumerate(cases.items()):
        style = "-" if key.startswith("mps") else "--"
        color = colors(index % 10)
        rho = table["rho01_real"] + 1j * table["rho01_imag"]
        axes[0, 0].plot(table["time"], table["sigma_z"], style, color=color, label=key)
        axes[0, 1].plot(table["time"], np.abs(rho), style, color=color, label=key)
        if key != reference_key:
            error = np.maximum(np.abs(table["sigma_z"] - reference["sigma_z"]), 1e-16)
            axes[1, 0].semilogy(table["time"], error, style, color=color, label=key)
        axes[1, 1].step(table["time"], ranks.max(axis=1), where="post", linestyle=style,
                        color=color, label=key)
    labels = (
        (r"Spin population", r"$\langle\sigma_z\rangle$"),
        (r"Spin coherence", r"$|\rho_{01}|$"),
        (f"Error relative to {reference_key}", r"$|\Delta\langle\sigma_z\rangle|$"),
        (r"Realized tensor rank", r"$\max\chi$"),
    )
    for axis, (title, ylabel) in zip(axes.flat, labels):
        axis.set_title(title)
        axis.set_xlabel("time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="outside lower center", frameon=False)
    fig.savefig(output / "sbm_letta_mps_trajectories.png", dpi=180)
    fig.savefig(output / "sbm_letta_mps_trajectories.pdf")
    plt.close(fig)


def plot_efficiency(output: Path, rows: list[dict], reference_key: str):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for backend, marker in (("mps", "o"), ("letta", "s")):
        selected = [row for row in rows if row["backend"] == backend and row["case"] != reference_key]
        selected.sort(key=lambda row: row["peak_parameters"])
        parameters = [row["peak_parameters"] for row in selected]
        errors = [max(row["max_sigma_z_error"], 1e-16) for row in selected]
        walls = [row["wall_seconds"] for row in selected]
        labels = [row["rank_cap"] for row in selected]
        axes[0].loglog(parameters, errors, marker=marker, label=backend.upper())
        axes[1].loglog(walls, errors, marker=marker, label=backend.upper())
        for x, y, rank in zip(parameters, errors, labels):
            axes[0].annotate(f"D={rank}", (x, y), xytext=(4, 4), textcoords="offset points")
        for x, y, rank in zip(walls, errors, labels):
            axes[1].annotate(f"D={rank}", (x, y), xytext=(4, 4), textcoords="offset points")
    axes[0].set(xlabel="peak complex parameters", ylabel=r"max $|\Delta\langle\sigma_z\rangle|$",
                title="Accuracy versus state size")
    axes[1].set(xlabel="wall time (s)", ylabel=r"max $|\Delta\langle\sigma_z\rangle|$",
                title="Accuracy versus runtime")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(frameon=False)
    fig.savefig(output / "sbm_letta_mps_efficiency.png", dpi=180)
    fig.savefig(output / "sbm_letta_mps_efficiency.pdf")
    plt.close(fig)


def half_dt_check(output: Path, model, dynamics, args, reference_rank: int, coarse):
    original_dt = model.TN_DT
    model.TN_DT = 0.5 * original_dt
    model.TDVP_REPORT_STEPS = max(1, int(round(0.5 / model.TN_DT)))
    model.TDVP_FORCE_SWITCH_TIME = args.tmax + model.TN_DT
    model.TDVP_SATURATION_STEPS = int(round(args.tmax / model.TN_DT)) + 2
    path = output / f"mps_d{reference_rank}_dt_half"
    dynamics.run(
        "mps", path, rank=reference_rank, force=args.force, save_states=False,
        heom_path=None,
    )
    fine, metadata, _ = load_case(path)
    fine_on_coarse = fine[::2]
    if len(fine_on_coarse) != len(coarse) or not np.allclose(
        fine_on_coarse["time"], coarse["time"], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Half-step trajectory does not align with the main grid")
    coarse_rho = coarse["rho01_real"] + 1j * coarse["rho01_imag"]
    fine_rho = fine_on_coarse["rho01_real"] + 1j * fine_on_coarse["rho01_imag"]
    result = {
        "coarse_dt": original_dt,
        "fine_dt": model.TN_DT,
        "rank": reference_rank,
        "max_sigma_z_difference": float(
            np.max(np.abs(coarse["sigma_z"] - fine_on_coarse["sigma_z"]))
        ),
        "max_rho01_difference": float(np.max(np.abs(coarse_rho - fine_rho))),
        "fine_wall_seconds": float(metadata["wall_seconds"]),
    }
    (output / "time_step_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    model.TN_DT = original_dt
    return result


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbm-dir", type=Path, default=DEFAULT_SBM)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nmodes", type=int, default=16)
    parser.add_argument("--local-dim", type=int, default=4)
    parser.add_argument("--quadrature-order", type=int, default=256)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--mps-ranks", type=int, nargs="+", default=(2, 4, 8, 16, 24))
    parser.add_argument("--letta-ranks", type=int, nargs="+", default=(1, 2, 3, 4, 6))
    parser.add_argument("--half-dt-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    model, dynamics = load_backend(args.sbm_dir.resolve())
    configure(model, args)
    cases = {}
    for backend, ranks in (("mps", args.mps_ranks), ("letta", args.letta_ranks)):
        for rank in ranks:
            key = f"{backend}_d{rank}"
            case = args.output / key
            dynamics.run(
                backend, case, rank=rank, force=args.force, save_states=False,
                heom_path=None,
            )
            cases[key] = load_case(case)

    reference_key = f"mps_d{max(args.mps_ranks)}"
    dimensions = [2] + [args.local_dim] * args.nmodes
    rows = analyze(args.output, cases, dimensions, reference_key)
    plot_trajectories(args.output, cases, reference_key)
    plot_efficiency(args.output, rows, reference_key)
    time_step = None
    if args.half_dt_check:
        time_step = half_dt_check(
            args.output, model, dynamics, args, max(args.mps_ranks),
            cases[reference_key][0],
        )
    print(f"Reference: {reference_key}")
    for row in rows:
        print(
            f"{row['case']:>10s} params={row['peak_parameters']:>8d} "
            f"max_dsz={row['max_sigma_z_error']:.3e} "
            f"max_drho={row['max_rho01_error']:.3e} wall={row['wall_seconds']:.2f}s"
        )
    if time_step is not None:
        print(
            f"half-dt check: max_dsz={time_step['max_sigma_z_difference']:.3e} "
            f"max_drho={time_step['max_rho01_difference']:.3e}"
        )


if __name__ == "__main__":
    cli()
