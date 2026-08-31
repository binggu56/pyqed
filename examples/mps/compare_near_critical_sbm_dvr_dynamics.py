#!/usr/bin/env python3
"""Native MPS versus window-2 LETTA dynamics for a DVR spin-boson chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.letta import (
    LETTAEvolution,
    letta_structural_rank_caps,
    nearest_neighbor_hamiltonian,
    system_reduced_density_matrix,
    window2_product_state,
)
from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_wilson_chain,
    spin_boson_bond_hamiltonians,
    spin_boson_product_factors,
)
from pyqed.mps.mpo import nearest_neighbor_mpo
from pyqed.mps.mps import MPS
from pyqed.mps.tdvp import TDVPEngine
from pyqed.narg.spin_boson import local_boson_operators


OBSERVABLE_FIELDS = (
    "time", "mode", "sigma_z", "rho01_real", "rho01_imag", "rho01_abs",
    "raw_trace", "trace_error", "hermiticity_error", "minimum_eigenvalue",
    "max_rank", "step_seconds", "discarded_weight", "krylov_residual_max",
)


def build_problem(args):
    chain = log_discretized_spin_boson_wilson_chain(
        args.nmodes,
        alpha=args.alpha,
        Lambda=args.Lambda,
        s=args.s,
        omegac=args.omegac,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    identity, annihilation, creation, oscillator = local_boson_operators(
        args.local_dim, basis="sine-dvr", dvr_qmax=args.qmax
    )
    bonds, dims = spin_boson_bond_hamiltonians(
        chain, identity, annihilation, creation, oscillator
    )
    factors = spin_boson_product_factors(chain, oscillator, spin_state=1)
    oscillator_values, oscillator_vectors = np.linalg.eigh(oscillator)
    vacuum = oscillator_vectors[:, 0]
    metadata = {
        "model": "zero-temperature sub-Ohmic spin-boson Wilson chain",
        "Hamiltonian": (
            "H=(epsilon/2)sigma_z-(Delta/2)sigma_x+sum_n e_n a_n^dagger a_n"
            "+(t0/2)sigma_z(a_0+a_0^dagger)+sum_n t_n(a_n^dagger a_{n+1}+h.c.)"
        ),
        "spectral_density": "J(w)=2*pi*alpha*omega_c^(1-s)*w^s",
        "initial_state": "|sigma_z=-1> tensor sine-DVR oscillator vacua",
        "basis": "sine-DVR",
        "dvr_qmax": args.qmax,
        "dvr_oscillator_low_eigenvalues": oscillator_values[:6].tolist(),
        "dvr_vacuum_edge_probability": float(
            abs(vacuum[0]) ** 2 + abs(vacuum[-1]) ** 2
        ),
        "alpha": args.alpha,
        "Lambda": args.Lambda,
        "s": args.s,
        "omega_c": args.omegac,
        "Delta": args.delta,
        "epsilon": args.epsilon,
        "N": args.nmodes,
        "d": args.local_dim,
        "dt": args.dt,
        "tmax": args.tmax,
        "estimated_chain_roundtrip_time": float(
            2.0 * args.nmodes / max(np.max(np.abs(chain.hopping)), 1.0e-15)
        ),
    }
    return bonds, dims, factors, metadata


def _rescale(value, log_scale, context):
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError(f"{context}: invalid contraction scale {scale}")
    return value / scale, log_scale + math.log(scale)


def mps_system_rdm(state):
    cores = [
        np.asarray(state._get_std_B(site), dtype=complex)
        for site in range(state.L)
    ]
    environment = np.ones((1, 1), dtype=complex)
    log_scale = 0.0
    for site in reversed(range(1, len(cores))):
        core = cores[site]
        environment = np.einsum(
            "asb,csd,bd->ac", core.conj(), core, environment, optimize=True
        )
        environment, log_scale = _rescale(
            environment, log_scale, f"MPS environment {site}"
        )
    first = cores[0]
    rho = np.einsum(
        "aud,avb,bd->uv", first, first.conj(), environment, optimize=True
    )
    rho, log_scale = _rescale(rho, log_scale, "MPS system RDM")
    trace = np.trace(rho)
    log_norm = log_scale + math.log(float(trace.real))
    rho /= trace.real
    return rho, {
        "log_norm": log_norm,
        "trace_error": abs(math.expm1(log_norm)),
        "hermiticity_error": float(np.max(np.abs(rho - rho.conj().T))),
        "minimum_eigenvalue": float(
            np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))[0]
        ),
    }


def state_ranks(backend, state):
    if backend == "mps":
        return tuple(
            int(state._get_std_B(site).shape[-1]) for site in range(state.L - 1)
        )
    if hasattr(state, "ranks"):
        return tuple(int(value) for value in state.ranks)
    return tuple(int(tensor.shape[-1]) for tensor in state.tensors[:-1])


def mps_structural_rank_caps(dims, maximum):
    def capped_product(values):
        result = 1
        for value in values:
            result *= int(value)
            if result >= maximum:
                return maximum
        return result

    return tuple(
        min(
            maximum,
            capped_product(dims[: bond + 1]),
            capped_product(dims[bond + 1 :]),
        )
        for bond in range(len(dims) - 1)
    )


def observe(backend, state, time_value, mode, elapsed, diagnostics):
    if backend == "mps":
        rho, rdm_info = mps_system_rdm(state)
    else:
        rho, rdm_info = system_reduced_density_matrix(state, return_info=True)
    ranks = state_ranks(backend, state)
    return {
        "time": float(time_value),
        "mode": mode,
        "sigma_z": float((rho[0, 0] - rho[1, 1]).real),
        "rho01_real": float(rho[0, 1].real),
        "rho01_imag": float(rho[0, 1].imag),
        "rho01_abs": float(abs(rho[0, 1])),
        "raw_trace": float(math.exp(rdm_info["log_norm"])),
        "trace_error": rdm_info["trace_error"],
        "hermiticity_error": rdm_info["hermiticity_error"],
        "minimum_eigenvalue": rdm_info["minimum_eigenvalue"],
        "max_rank": max(ranks, default=1),
        "step_seconds": float(elapsed),
        "discarded_weight": float(diagnostics.get("truncation_error", 0.0)),
        "krylov_residual_max": float(
            diagnostics.get("krylov_residual_max", 0.0)
        ),
    }


def _write_case(path, rows, rank_history, discarded_history, modes, caps, metadata):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "TDVP_observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        path / "TDVP_bond_diagnostics.npz",
        times=np.asarray([row["time"] for row in rows]),
        modes=np.asarray(modes),
        ranks=np.asarray(rank_history, dtype=int),
        discarded_weights=np.asarray(discarded_history, dtype=float),
        structural_caps=np.asarray(caps, dtype=int),
    )
    (path / "TDVP_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def run_case(backend, path, rank, bonds, dims, factors, physical_metadata, args):
    metadata_path = path / "TDVP_metadata.json"
    requested_tensor_backend = "numpy" if backend == "mps" else args.letta_backend
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("complete")
            and metadata.get("implementation") == "pyqed-native"
            and metadata.get("requested_tensor_backend") == requested_tensor_backend
        ):
            print(f"SKIP existing native result {path}", flush=True)
            return

    if backend == "mps":
        state = MPS([factor.reshape(1, factor.size, 1) for factor in factors])
        engine = TDVPEngine(
            nearest_neighbor_mpo(bonds, dims),
            integrator="tdvp2",
            max_bond=rank,
            cutoff=0.0,
            krylov_dim=args.krylov_maxdim,
            krylov_tol=args.krylov_tolerance,
            canonicalize_first=True,
            canonicalize_each_step=False,
        )
        caps = mps_structural_rank_caps(dims, rank)
        mode = "tdvp2"
        saturation_streak = 0
        switch_step = switch_time = switch_reason = None
        tensor_backend = "numpy"
        tensor_device = "cpu"
        tensor_threads = None
        tensor_channel_mode = "dense"
    else:
        state = window2_product_state(factors, max_bond=rank)
        engine = LETTAEvolution(
            nearest_neighbor_hamiltonian(bonds, dims),
            max_bond=rank,
            cutoff=0.0,
            krylov_dim=args.krylov_maxdim,
            krylov_tol=args.krylov_tolerance,
            saturation_steps=args.saturation_steps,
            force_switch_time=args.force_switch_time,
            backend=args.letta_backend,
            device=args.torch_device,
            torch_num_threads=args.torch_threads,
        )
        caps = letta_structural_rank_caps(dims, rank)
        mode = "tdvp2"
        tensor_backend = engine.backend
        tensor_device = str(getattr(engine.engine, "device", "cpu"))
        tensor_threads = getattr(engine.engine, "num_threads", None)
        tensor_channel_mode = getattr(engine.engine, "channel_mode", None)

    requested_steps = int(round(args.tmax / args.dt))
    report_steps = max(1, int(round(args.report_interval / args.dt)))
    zero_info = {"truncation_error": 0.0, "krylov_residual_max": 0.0}
    rows = [observe(backend, state, 0.0, mode, 0.0, zero_info)]
    rank_history = [state_ranks(backend, state)]
    discarded_history = [[0.0] * len(caps)]
    modes = [mode]
    step_times = []
    started = time.perf_counter()
    complete = True
    step = 0

    for step in range(1, requested_steps + 1):
        if (
            args.max_wall_seconds is not None
            and time.perf_counter() - started >= args.max_wall_seconds
        ):
            complete = False
            break
        tic = time.perf_counter()
        if backend == "mps":
            mode_used = mode
            state, info = engine.step(
                state, args.dt, normalize=False, return_info=True
            )
            ranks = state_ranks(backend, state)
            if mode == "tdvp2":
                saturated = all(value >= cap for value, cap in zip(ranks, caps))
                saturation_streak = saturation_streak + 1 if saturated else 0
                reason = None
                if saturation_streak >= args.saturation_steps:
                    reason = "rank saturation"
                if (
                    args.force_switch_time is not None
                    and step * args.dt >= args.force_switch_time
                ):
                    reason = "forced switch time"
                if reason is not None:
                    mode = "tdvp1"
                    engine.integrator = "tdvp"
                    switch_step = step
                    switch_time = step * args.dt
                    switch_reason = reason
        else:
            state, info = engine.step(state, args.dt, normalize=False)
            mode_used = info["mode_used"]
            mode = info["next_mode"]
        elapsed = time.perf_counter() - tic
        step_times.append(elapsed)
        if step % report_steps == 0 or step == requested_steps:
            rows.append(
                observe(backend, state, step * args.dt, mode_used, elapsed, info)
            )
            rank_history.append(state_ranks(backend, state))
            discarded = info.get("discarded_weights")
            if discarded is None:
                discarded = [info.get("truncation_error", 0.0)] * len(caps)
            discarded_history.append(discarded)
            modes.append(mode_used)

    if backend == "letta":
        switch_step = engine.switch_step
        switch_time = engine.switch_time
        switch_reason = engine.switch_reason
    wall = time.perf_counter() - started
    metadata = {
        **physical_metadata,
        "implementation": "pyqed-native",
        "backend": backend,
        "requested_tensor_backend": requested_tensor_backend,
        "tensor_backend": tensor_backend,
        "tensor_device": tensor_device,
        "tensor_threads": tensor_threads,
        "tensor_channel_mode": tensor_channel_mode,
        "representation": "MPS" if backend == "mps" else "LETTA(window=2)",
        "evolution_driver": "hybrid_symmetric_TDVP2_to_TDVP1",
        "rank_cap": rank,
        "structural_rank_caps": caps,
        "switch_step": switch_step,
        "switch_time": switch_time,
        "switch_reason": switch_reason,
        "krylov_target": args.krylov_tolerance,
        "krylov_max_dimension": args.krylov_maxdim,
        "requested_number_of_steps": requested_steps,
        "completed_number_of_steps": min(step, requested_steps),
        "complete": complete,
        "wall_seconds": wall,
        "mean_seconds_per_step": float(np.mean(step_times)) if step_times else 0.0,
    }
    _write_case(
        path, rows, rank_history, discarded_history, modes, caps, metadata
    )


def load_case(path):
    table = np.genfromtxt(path / "TDVP_observables.csv", delimiter=",", names=True)
    if table.ndim == 0:
        table = table.reshape(1)
    metadata = json.loads(
        (path / "TDVP_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(path / "TDVP_bond_diagnostics.npz") as payload:
        ranks = np.asarray(payload["ranks"], dtype=int)
    return table, metadata, ranks


def load_validated_mps_reference(path, args):
    """Load an existing MPS trajectory after matching its physical model."""
    case = load_case(path)
    metadata = case[1]
    expected = {
        "N": args.nmodes,
        "d": args.local_dim,
        "alpha": args.alpha,
        "Lambda": args.Lambda,
        "s": args.s,
        "omega_c": args.omegac,
        "Delta": args.delta,
        "epsilon": args.epsilon,
        "tmax": args.tmax,
    }
    for field, value in expected.items():
        actual = metadata.get(field)
        if actual is None or not np.isclose(float(actual), float(value)):
            raise ValueError(
                f"MPS reference mismatch for {field}: {actual!r} != {value!r}"
            )
    reference_dt = metadata.get("dt", metadata.get("tn_dt"))
    if reference_dt is None or not np.isclose(float(reference_dt), args.dt):
        raise ValueError(
            f"MPS reference mismatch for dt: {reference_dt!r} != {args.dt!r}"
        )
    if not metadata.get("complete"):
        raise ValueError("The supplied MPS reference trajectory is incomplete.")
    rank = metadata.get("rank_cap", metadata.get("max_rank"))
    if rank is None:
        raise ValueError("The supplied MPS reference does not record its rank cap.")
    return f"mps_d{int(rank)}", case


def load_validated_mps_suite(path, args):
    """Load the selected reference and matching sibling MPS trajectories."""
    reference_key, reference_case = load_validated_mps_reference(path, args)
    cases = {reference_key: reference_case}
    for sibling in path.parent.glob("mps_d*"):
        if not sibling.is_dir() or sibling == path:
            continue
        key, case = load_validated_mps_reference(sibling, args)
        cases[key] = case
    return reference_key, cases


def parameter_count(backend, dims, ranks):
    left, right = np.r_[1, ranks], np.r_[ranks, 1]
    if backend == "mps":
        return int(sum(a * d * b for a, d, b in zip(left, dims, right)))
    return int(
        sum(
            a
            * dims[site]
            * (dims[site + 1] if site + 1 < len(dims) else 1)
            * b
            for site, (a, b) in enumerate(zip(left, right))
        )
    )


def analyze(output, cases, dims, reference_key):
    reference = cases[reference_key][0]
    reference_rho = reference["rho01_real"] + 1.0j * reference["rho01_imag"]
    rows = []
    for key, (table, metadata, ranks) in cases.items():
        backend, rank_text = key.split("_d")
        rho = table["rho01_real"] + 1.0j * table["rho01_imag"]
        rows.append(
            {
                "case": key,
                "backend": backend,
                "tensor_backend": metadata.get("tensor_backend", "numpy"),
                "rank_cap": int(rank_text),
                "peak_parameters": max(
                    parameter_count(backend, dims, item) for item in ranks
                ),
                "final_max_rank": int(ranks[-1].max()),
                "max_sigma_z_error": float(
                    np.max(np.abs(table["sigma_z"] - reference["sigma_z"]))
                ),
                "max_rho01_error": float(np.max(np.abs(rho - reference_rho))),
                "max_trace_error": float(np.max(np.abs(table["trace_error"]))),
                "wall_seconds": float(metadata["wall_seconds"]),
                "mean_seconds_per_step": float(metadata["mean_seconds_per_step"]),
                "complete": bool(metadata["complete"]),
            }
        )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"reference": reference_key, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    return rows


def plot_results(output, cases, rows, reference_key):
    reference = cases[reference_key][0]
    reference_rank = reference_key.split("_d")[1]
    colors = plt.get_cmap("tab10")

    ordered_cases = sorted(
        cases.items(),
        key=lambda item: (
            item[0].split("_d")[0] != "mps",
            int(item[0].split("_d")[1]),
        ),
    )
    mps_keys = [key for key, _ in ordered_cases if key.startswith("mps")]
    letta_keys = [key for key, _ in ordered_cases if key.startswith("letta")]
    mps_colors = {
        key: plt.get_cmap("Blues")(0.42 + 0.46 * index / max(len(mps_keys) - 1, 1))
        for index, key in enumerate(mps_keys)
    }
    letta_colors = {
        key: plt.get_cmap("Oranges")(0.58 + 0.30 * index / max(len(letta_keys) - 1, 1))
        for index, key in enumerate(letta_keys)
    }
    fig, axis = plt.subplots(figsize=(9.6, 6.0), constrained_layout=True)
    inset = axis.inset_axes([0.51, 0.12, 0.46, 0.39])
    for key, (table, _, _) in ordered_cases:
        backend, rank = key.split("_d")
        if key == reference_key:
            color, linestyle, linewidth, zorder = "black", ":", 2.4, 2
        elif backend == "mps":
            color, linestyle, linewidth, zorder = mps_colors[key], "-", 1.8, 3
        else:
            color, linestyle, linewidth, zorder = (
                letta_colors[key], "--", 2.2, 4
            )
        label = f"{backend.upper()} $D={rank}$"
        axis.plot(
            table["time"],
            table["sigma_z"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )
        if key != reference_key:
            inset.semilogy(
                table["time"],
                np.maximum(
                    np.abs(table["sigma_z"] - reference["sigma_z"]), 1.0e-16
                ),
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
    axis.set(
        title="All population dynamics",
        xlabel="time",
        ylabel=r"$\langle\sigma_z\rangle$",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    inset.set(
        title=rf"absolute deviation from MPS $D={reference_rank}$",
        xlabel="time",
        ylabel=r"$|\Delta\langle\sigma_z\rangle|$",
    )
    inset.tick_params(labelsize=8)
    inset.title.set_fontsize(9)
    inset.xaxis.label.set_fontsize(8)
    inset.yaxis.label.set_fontsize(8)
    inset.grid(alpha=0.2, which="both")
    fig.savefig(output / "near_critical_sbm_all_population_dynamics.png", dpi=180)
    fig.savefig(output / "near_critical_sbm_all_population_dynamics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    displayed = [
        item
        for item in cases.items()
        if item[0] == reference_key or item[0].startswith("letta")
    ]
    for index, (key, (table, _, ranks)) in enumerate(displayed):
        style = "-" if key.startswith("mps") else "--"
        color = colors(index % 10)
        axes[0, 0].plot(
            table["time"], table["sigma_z"], style, color=color, label=key
        )
        axes[0, 1].plot(
            table["time"], table["rho01_abs"], style, color=color, label=key
        )
        if key != reference_key:
            axes[1, 0].semilogy(
                table["time"],
                np.maximum(
                    np.abs(table["sigma_z"] - reference["sigma_z"]), 1.0e-16
                ),
                style,
                color=color,
            )
        axes[1, 1].step(
            table["time"], ranks.max(axis=1), where="post", linestyle=style,
            color=color,
        )
    labels = (
        ("Population dynamics", r"$\langle\sigma_z\rangle$"),
        ("Spin coherence", r"$|\rho_{01}|$"),
        (f"Error relative to {reference_key}", r"$|\Delta\langle\sigma_z\rangle|$"),
        ("Realized tensor rank", r"$\max\chi$"),
    )
    for axis, (title, ylabel) in zip(axes.flat, labels):
        axis.set(title=title, xlabel="time", ylabel=ylabel)
        axis.grid(alpha=0.25, which="both")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=4, loc="outside lower center", frameon=False)
    fig.savefig(output / "near_critical_sbm_dvr_dynamics.png", dpi=180)
    fig.savefig(output / "near_critical_sbm_dvr_dynamics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    backend_styles = {
        "mps": ("o", "tab:blue", "-"),
        "letta": ("s", "tab:orange", "--"),
    }
    for backend, (marker, color, linestyle) in backend_styles.items():
        selected = [
            row
            for row in rows
            if row["backend"] == backend and row["case"] != reference_key
        ]
        selected.sort(key=lambda row: row["peak_parameters"])
        if not selected:
            continue
        for row_index, field in enumerate(
            ("max_sigma_z_error", "max_rho01_error")
        ):
            errors = [max(row[field], 1.0e-16) for row in selected]
            xsets = (
                [row["peak_parameters"] for row in selected],
                [row["wall_seconds"] for row in selected],
            )
            for axis, xvalues in zip(axes[row_index], xsets):
                axis.loglog(
                    xvalues,
                    errors,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                    markersize=7,
                    label=backend.upper(),
                )
                for row, xvalue, error in zip(selected, xvalues, errors):
                    axis.annotate(
                        f"$D={row['rank_cap']}$",
                        (xvalue, error),
                        xytext=(5, 6 if backend == "mps" else -13),
                        textcoords="offset points",
                        color=color,
                        fontsize=9,
                    )
    for row_index, ylabel in enumerate(
        (r"max $|\Delta\langle\sigma_z\rangle|$", r"max $|\Delta\rho_{01}|$")
    ):
        axes[row_index, 0].set(xlabel="peak complex parameters", ylabel=ylabel)
        axes[row_index, 1].set(xlabel="wall time (s)", ylabel=ylabel)
    for axis in axes.flat:
        axis.grid(alpha=0.25, which="both")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)
    fig.suptitle(
        "LETTA–MPS accuracy/resource Pareto comparison\n"
        "lower and further left is better",
        fontsize=14,
    )
    fig.savefig(output / "near_critical_sbm_dvr_efficiency.png", dpi=180)
    fig.savefig(output / "near_critical_sbm_dvr_efficiency.pdf")
    plt.close(fig)

    mps_rows = [
        row
        for row in rows
        if row["backend"] == "mps" and row["case"] != reference_key
    ]
    letta_rows = [row for row in rows if row["backend"] == "letta"]
    pairs = []
    for letta in sorted(letta_rows, key=lambda row: row["rank_cap"]):
        dominated = [
            mps
            for mps in mps_rows
            if mps["max_sigma_z_error"] >= letta["max_sigma_z_error"]
            and mps["max_rho01_error"] >= letta["max_rho01_error"]
        ]
        if dominated:
            comparator = min(
                dominated,
                key=lambda row: abs(
                    math.log(row["wall_seconds"] / letta["wall_seconds"])
                ),
            )
            pairs.append((letta, comparator))
    if pairs:
        fig, axes = plt.subplots(
            1, 2, figsize=(11.2, 4.4), constrained_layout=True
        )
        labels = [
            f"LETTA $D={letta['rank_cap']}$\nvs MPS $D={mps['rank_cap']}$"
            for letta, mps in pairs
        ]
        metrics = (
            ("runtime speedup", "wall_seconds"),
            (r"$\sigma_z$ accuracy gain", "max_sigma_z_error"),
            (r"$\rho_{01}$ accuracy gain", "max_rho01_error"),
        )
        xvalues = np.arange(len(pairs), dtype=float)
        width = 0.23
        for index, (label, field) in enumerate(metrics):
            values = [mps[field] / letta[field] for letta, mps in pairs]
            bars = axes[0].bar(
                xvalues + (index - 1) * width,
                values,
                width,
                label=label,
            )
            value_labels = [
                f"{value:.2f}×" if value < 2.0 else f"{value:.1f}×"
                for value in values
            ]
            axes[0].bar_label(
                bars, labels=value_labels, padding=2, fontsize=9
            )
        axes[0].axhline(1.0, color="black", linewidth=1.0)
        axes[0].set(
            title="LETTA gains over the nearest MPS point it dominates",
            ylabel="improvement factor (higher is better)",
            xticks=xvalues,
            xticklabels=labels,
            yscale="log",
        )
        axes[0].legend(frameon=False, fontsize=9)

        width = 0.34
        mps_parameters = [mps["peak_parameters"] for _, mps in pairs]
        letta_parameters = [letta["peak_parameters"] for letta, _ in pairs]
        mps_bars = axes[1].bar(
            xvalues - width / 2,
            mps_parameters,
            width,
            label="MPS",
            color="tab:blue",
        )
        letta_bars = axes[1].bar(
            xvalues + width / 2,
            letta_parameters,
            width,
            label="LETTA",
            color="tab:orange",
        )
        axes[1].bar_label(mps_bars, fmt="%d", padding=2, fontsize=9)
        axes[1].bar_label(letta_bars, fmt="%d", padding=2, fontsize=9)
        axes[1].set(
            title="Parameter-count tradeoff",
            ylabel="peak complex parameters",
            xticks=xvalues,
            xticklabels=labels,
        )
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.grid(axis="y", alpha=0.25, which="both")
        fig.savefig(output / "near_critical_sbm_dvr_advantage.png", dpi=180)
        fig.savefig(output / "near_critical_sbm_dvr_advantage.pdf")
        plt.close(fig)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nmodes", type=int, default=20)
    parser.add_argument("--local-dim", type=int, default=12)
    parser.add_argument("--qmax", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.075)
    parser.add_argument("--Lambda", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--omegac", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--mps-ranks", type=int, nargs="+", default=(4, 8, 12, 16, 24))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("mps", "letta"),
        default=("mps", "letta"),
        help="ansatz families to evolve; a reused MPS reference may still be loaded",
    )
    parser.add_argument(
        "--mps-reference-dir",
        type=Path,
        help="reuse a completed, physically matching MPS case instead of rerunning MPS",
    )
    parser.add_argument("--letta-ranks", type=int, nargs="+", default=(2, 4))
    parser.add_argument(
        "--letta-backend", choices=("auto", "numpy", "torch"), default="auto"
    )
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--saturation-steps", type=int, default=4)
    parser.add_argument("--force-switch-time", type=float, default=2.0)
    parser.add_argument("--krylov-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--krylov-maxdim", type=int, default=24)
    parser.add_argument("--report-interval", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    bonds, dims, factors, physical_metadata = build_problem(args)
    (args.output / "physical_metadata.json").write_text(
        json.dumps(physical_metadata, indent=2), encoding="utf-8"
    )
    cases = {}
    if args.mps_reference_dir is not None:
        reference_key, reference_cases = load_validated_mps_suite(
            args.mps_reference_dir, args
        )
        cases.update(reference_cases)
    else:
        if "mps" not in args.backends:
            raise ValueError(
                "An MPS reference is required when the MPS backend is omitted."
            )
        reference_key = f"mps_d{max(args.mps_ranks)}"
    backends = []
    if args.mps_reference_dir is None and "mps" in args.backends:
        backends.append(("mps", args.mps_ranks))
    if "letta" in args.backends:
        backends.append(("letta", args.letta_ranks))
    for backend, ranks in backends:
        for rank in ranks:
            key = f"{backend}_d{rank}"
            path = args.output / key
            run_case(
                backend, path, rank, bonds, dims, factors, physical_metadata, args
            )
            cases[key] = load_case(path)
            if not cases[key][1]["complete"]:
                raise RuntimeError(f"Incomplete trajectory for {key}")
    rows = analyze(args.output, cases, dims, reference_key)
    plot_results(args.output, cases, rows, reference_key)
    for row in rows:
        print(
            f"{row['case']:>10s} params={row['peak_parameters']:>8d} "
            f"tensor={row['tensor_backend']:<5s} "
            f"max_dsz={row['max_sigma_z_error']:.3e} "
            f"max_drho={row['max_rho01_error']:.3e} "
            f"wall={row['wall_seconds']:.2f}s"
        )
    print(args.output)


if __name__ == "__main__":
    cli()
