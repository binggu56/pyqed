#!/usr/bin/env python3
"""Fit single- and two-patch SO2 Procrustes-gauge LDR models by TT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import numpy as np

from examples.ldr.so2_casci_cgldr import (
    DEFAULT_SCAN_DIR,
    casci_overlap_active,
    load_so2_linked_scan,
)
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet, observables
from examples.ldr.so2_casci_full_ldr import (
    STATE_IDS,
    _electronic_point,
    full_hamiltonian,
    path_overlap,
)
from examples.ldr.so2_procrustes_dynamics import propagate, transform_states
from examples.ldr.so2_procrustes_gauge import reference_index, rotate_kernel
from pyqed.ldr.oracle import Frames, ProcrustesOracle
from pyqed.ldr.overlap import unpack
from pyqed.ldr.ttfit import (
    FiberSampler,
    HamiltonianSampler,
    assemble,
    fiber_shape,
    fit_cross,
    fit_svd,
    group_kinetic_terms,
    kernel_fiber,
)


DEFAULT_REFERENCE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_full_ldr_9x9x9_20fs/"
    "electronic_reference.npz"
)
DEFAULT_SINGLE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/"
    "procrustes_gauge.npz"
)
DEFAULT_TWO = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_two_patch_9x9x9/"
    "procrustes_gauge.npz"
)


class ArrayGaugeOracle:
    def __init__(self, local, overlap):
        self.local = np.asarray(local)
        self.overlap = np.asarray(overlap)
        self.shape = self.local.shape[:-2]

    def hamiltonian_many(self, indices):
        return np.asarray([self.local[index] for index in indices])

    def overlap_many(self, pairs):
        blocks = []
        for left, right in pairs:
            i = np.ravel_multi_index(left, self.shape)
            j = np.ravel_multi_index(right, self.shape)
            blocks.append(self.overlap[i, :, j, :])
        return np.asarray(blocks)


@dataclass(frozen=True)
class SO2Builder:
    grids: tuple
    basis: str
    derivative_workers: int = 1

    def __call__(self, index):
        return _electronic_point(
            (
                tuple(index),
                float(self.grids[0][index[0]]),
                float(self.grids[1][index[1]]),
                float(self.grids[2][index[2]]),
                self.basis,
                int(self.derivative_workers),
            )
        )


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def field_error(fitted, exact):
    fitted = np.asarray(fitted)
    exact = np.asarray(exact)
    scale = max(float(np.linalg.norm(exact)), np.finfo(float).tiny)
    return {
        "relative_error": float(np.linalg.norm(fitted - exact) / scale),
        "max_abs_error": float(np.max(np.abs(fitted - exact))),
    }


def fit_svd_model(local, fibers, groups, rank, *, electronic_mode):
    local_field = local.reshape(*local.shape[:-2], -1)

    def fit_field(exact):
        if electronic_mode == "coupled":
            _cores, fitted, info = fit_svd(exact, rank)
            return fitted, info
        fitted = np.empty_like(exact)
        element_info = []
        for flat in range(exact.shape[-1]):
            _cores, values, info = fit_svd(exact[..., flat], rank)
            fitted[..., flat] = values
            element_info.append(info)
        errors = field_error(fitted, exact)
        info = {
            "backend": "svd-blockwise",
            "samples": int(sum(item["samples"] for item in element_info)),
            **errors,
            "ranks": [item["ranks"] for item in element_info],
        }
        return fitted, info

    fitted_local, local_info = fit_field(local_field)
    fitted_fibers = {}
    infos = {"local": local_info}
    for active, values in fibers.items():
        fitted, info = fit_field(values)
        fitted_fibers[active] = fitted
        infos["S_" + "".join(map(str, active))] = info
    hamiltonian = assemble(
        groups,
        fitted_fibers,
        fitted_local.reshape(local.shape),
    )
    return hamiltonian, infos


def fit_cross_model(
    local,
    overlap,
    fibers,
    groups,
    rank,
    *,
    sweeps,
    rtol,
    validation,
    seed,
    start_rank,
    kick_rank,
    electronic_mode,
):
    shape = local.shape[:-2]
    nstates = local.shape[-1]
    oracle = ArrayGaugeOracle(local, overlap)
    if electronic_mode not in {"coupled", "blockwise"}:
        raise ValueError("electronic_mode must be 'coupled' or 'blockwise'")

    def fit_field(exact, sampler_factory, field_seed):
        if electronic_mode == "coupled":
            sampler = sampler_factory(None)
            _cores, fitted, info = fit_cross(
                exact.shape,
                sampler,
                batch_evaluator=sampler.batch,
                max_rank=rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=field_seed,
                start_rank=start_rank,
                kick_rank=kick_rank,
            )
            return fitted, info, [sampler]
        fitted = np.empty_like(exact)
        element_info = []
        samplers = []
        for flat in range(nstates * nstates):
            element = divmod(flat, nstates)
            sampler = sampler_factory(element)
            _cores, values, info = fit_cross(
                exact.shape[:-1],
                sampler,
                batch_evaluator=sampler.batch,
                max_rank=rank,
                sweeps=sweeps,
                rtol=rtol,
                validation=validation,
                seed=field_seed + flat,
                start_rank=start_rank,
                kick_rank=kick_rank,
            )
            fitted[..., flat] = values
            element_info.append(info)
            samplers.append(sampler)
        info = {
            "backend": "native-blockwise",
            "samples": int(sum(item["samples"] for item in element_info)),
            "sweeps": int(max(item["sweeps"] for item in element_info)),
            "validation_error": float(
                max(item["validation_error"] for item in element_info)
            ),
            "ranks": [item["ranks"] for item in element_info],
        }
        return fitted, info, samplers

    local_exact = local.reshape(*shape, nstates * nstates)
    fitted_local, local_info, local_samplers = fit_field(
        local_exact,
        lambda element: HamiltonianSampler(oracle, nstates, element=element),
        seed,
    )
    local_info.update(field_error(fitted_local, local_exact))
    local_points = set().union(*(sampler.points for sampler in local_samplers))
    local_info["unique_geometries"] = len(local_points)
    fitted_fibers = {}
    infos = {"local": local_info}
    all_pairs = set()
    for offset, (active, exact) in enumerate(fibers.items(), start=1):
        fitted, info, samplers = fit_field(
            exact,
            lambda element, active=active: FiberSampler(
                oracle,
                shape,
                nstates,
                active,
                element=element,
            ),
            seed + 20 * offset,
        )
        info.update(field_error(fitted, exact))
        pairs = set().union(*(sampler.pairs for sampler in samplers))
        info["unique_overlap_blocks"] = len(pairs)
        fitted_fibers[active] = fitted
        infos["S_" + "".join(map(str, active))] = info
        all_pairs.update(pairs)
    hamiltonian = assemble(
        groups,
        fitted_fibers,
        fitted_local.reshape(local.shape),
    )
    geometries = set(local_points)
    geometries.update(index for pair in all_pairs for index in pair)
    sampling = {
        "scalar_samples": int(sum(info["samples"] for info in infos.values())),
        "unique_overlap_blocks": len(all_pairs),
        "unique_geometries": len(geometries),
        "full_scalar_entries": int(
            local.size + sum(values.size for values in fibers.values())
        ),
        "electronic_mode": electronic_mode,
    }
    return hamiltonian, infos, sampling


def dynamics_metrics(
    hamiltonian,
    gauge,
    original_initial,
    exact_states,
    exact_observables,
    grids,
    transport,
    times,
):
    initial = np.einsum(
        "...ia,...i->...a",
        gauge.conj(),
        original_initial.reshape(*gauge.shape[:-2], gauge.shape[-1]),
        optimize=True,
    ).reshape(-1)
    states = propagate(hamiltonian, initial, times)
    physical = transform_states(states, gauge)
    measured = observables(physical, grids, transport)
    state_error = np.max(np.abs(exact_states - physical), axis=1)
    population_error = np.max(
        np.abs(exact_observables[1] - measured[1]), axis=1
    )
    spans = np.asarray([grid[-1] - grid[0] for grid in grids])
    coordinate_error = np.max(
        np.abs(exact_observables[2] - measured[2]) / spans[None, :], axis=1
    )
    overlap = np.abs(np.einsum("ti,ti->t", exact_states.conj(), physical)) ** 2
    norms = np.sum(np.abs(physical) ** 2, axis=1)
    exact_norms = np.sum(np.abs(exact_states) ** 2, axis=1)
    fidelity = np.clip(overlap / (norms * exact_norms), 0.0, 1.0)
    summary = {
        "max_state_error": float(np.max(state_error)),
        "max_population_error": float(np.max(population_error)),
        "max_scaled_coordinate_error": float(np.max(coordinate_error)),
        "minimum_fidelity": float(np.min(fidelity)),
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
    }
    curves = {
        "populations": measured[1],
        "means": measured[2],
        "state_error": state_error,
        "population_error": population_error,
        "coordinate_error": coordinate_error,
        "fidelity": fidelity,
    }
    return summary, curves


def plot_rank_profile(output, ranks, summaries):
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    colors = {"single": "#0072B2", "two": "#D55E00"}
    for name in ("single", "two"):
        errors = [summaries[name][str(rank)]["hamiltonian_relative_error"] for rank in ranks]
        ax.loglog(ranks, errors, "o-", color=colors[name], label=f"{name}-patch")
    ax.set(xlabel="Maximum TT rank", ylabel="Relative Hamiltonian error")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(rank) for rank in ranks])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(False)
    ax.legend(frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=350)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_dynamics(output, times, exact, curves):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    colors = {"single": "#0072B2", "two": "#D55E00"}
    styles = {"svd": ":", "cross": "--"}
    for state, axis in zip((1, 2), axes[0]):
        axis.plot(times, exact[1][:, state], color="black", lw=1.8, label="Full LDR")
        for patch in ("single", "two"):
            for method in ("svd", "cross"):
                axis.plot(
                    times,
                    curves[patch][method]["populations"][:, state],
                    color=colors[patch],
                    linestyle=styles[method],
                    label=f"{patch} {method.upper()}",
                )
        axis.set(xlabel="Time (fs)", ylabel=rf"$P_{state}$")
    for patch in ("single", "two"):
        for method in ("svd", "cross"):
            label = f"{patch} {method.upper()}"
            axes[1, 0].semilogy(
                times,
                np.maximum(curves[patch][method]["population_error"], 1.0e-16),
                color=colors[patch],
                linestyle=styles[method],
                label=label,
            )
            axes[1, 1].semilogy(
                times,
                np.maximum(1.0 - curves[patch][method]["fidelity"], 1.0e-16),
                color=colors[patch],
                linestyle=styles[method],
                label=label,
            )
    axes[1, 0].set(xlabel="Time (fs)", ylabel="Maximum population error")
    axes[1, 1].set(xlabel="Time (fs)", ylabel=r"$1-\mathcal{F}$")
    axes[1, 0].set_ylim(1.0e-8, 1.0e-2)
    axes[1, 1].set_ylim(1.0e-9, 2.0e-3)
    axes[1, 0].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[1, 1].yaxis.set_major_formatter(LogFormatterMathtext())
    for label, axis in zip("abcd", axes.flat):
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def oracle_check(args, grids, energies, local, center):
    builder = SO2Builder(grids, args.basis, args.integral_workers)
    frames = Frames(
        energies.shape[:-1],
        builder,
        cache_dir=args.point_cache,
        workers=args.workers,
    )
    oracle = ProcrustesOracle(
        frames,
        center,
        frame=lambda record: record[1],
        energies=lambda record: record[2],
        overlap=lambda left, right: casci_overlap_active(left, right, STATE_IDS),
        energy_shift=float(np.min(energies)),
    )
    probes = [center]
    for axis in range(len(center)):
        for step in (-1, 1):
            index = list(center)
            index[axis] = min(max(index[axis] + step, 0), energies.shape[axis] - 1)
            probes.append(tuple(index))
    values = oracle.hamiltonian_many(probes)
    expected = np.asarray([local[index] for index in probes])
    forward = [(probes[i], probes[i + 1]) for i in range(len(probes) - 1)]
    blocks = oracle.overlap_many(forward)
    reverse = oracle.overlap_many([(right, left) for left, right in forward])
    result = {
        "local_max_abs_error": float(np.max(np.abs(values - expected))),
        "overlap_adjoint_max_abs_error": float(
            np.max(np.abs(blocks - reverse.swapaxes(-1, -2).conj()))
        ),
        "frames": frames.stats,
    }
    frames.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--single-gauge", type=Path, default=DEFAULT_SINGLE)
    parser.add_argument("--two-gauge", type=Path, default=DEFAULT_TWO)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_procrustes_tt"))
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--cross-rank", type=int, default=None)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-8)
    parser.add_argument("--cross-validation", type=int, default=1024)
    parser.add_argument("--cross-start-rank", type=int, default=1)
    parser.add_argument("--cross-kick-rank", type=int, default=2)
    parser.add_argument(
        "--cross-electronic-mode",
        choices=("blockwise", "coupled"),
        default="blockwise",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--integral-workers", type=int, default=1)
    parser.add_argument("--point-cache", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.point_cache is None:
        args.point_cache = args.reference.parent / "point_cache"
    ranks = tuple(sorted(set(int(rank) for rank in args.ranks)))
    cross_rank = max(ranks) if args.cross_rank is None else int(args.cross_rank)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    ngrid = int(np.prod(shape))
    overlap = path_overlap(shape, links).reshape(ngrid, nstates, ngrid, nstates)
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, axes = dense_kinetic(scan, *grids)
    terms = scan.solver.buildK_qsqa_terms(axes, symmetrize=True, svd_tol=0.0)
    groups = group_kinetic_terms(terms, shape)
    grouped_kinetic = sum(groups.values())
    kinetic_error = float(np.max(np.abs(grouped_kinetic - kinetic)))
    if kinetic_error > 1.0e-10:
        raise RuntimeError(f"Grouped KEO mismatch: {kinetic_error:.3e}")
    exact_hamiltonian = full_hamiltonian(kinetic, overlap, energies)

    with np.load(args.single_gauge) as archive:
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)
    packet = nuclear_packet(*grids, axes)
    original_initial = (
        packet[..., None] * primary_gauge[..., args.initial_state]
    ).reshape(-1)
    original_initial /= np.linalg.norm(original_initial)
    times = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)
    print("[TT] propagating full-LDR reference", flush=True)
    exact_states = propagate(exact_hamiltonian, original_initial, times)
    exact_observables = observables(exact_states, grids, primary_gauge)

    summary = {
        "grid": list(shape),
        "active_keo_patterns": [list(active) for active in groups if active],
        "kinetic_reconstruction_max_abs": kinetic_error,
        "ranks": list(ranks),
        "cross_rank": cross_rank,
        "electronic_mode": args.cross_electronic_mode,
        "svd": {},
        "cross": {},
        "dynamics": {},
    }
    curves = {"single": {}, "two": {}}
    saved = {"times_fs": times, "exact_populations": exact_observables[1]}

    for patch, gauge_path in (("single", args.single_gauge), ("two", args.two_gauge)):
        print(f"[TT] preparing {patch}-patch fields", flush=True)
        with np.load(gauge_path) as archive:
            gauge = np.asarray(archive["gauge"], dtype=complex)
            local = np.asarray(archive["aligned_local_hamiltonian"], dtype=complex)
        aligned = rotate_kernel(overlap, gauge.reshape(ngrid, nstates, nstates))
        fibers = {
            active: kernel_fiber(aligned, shape, active)
            for active in groups
            if active
        }
        exact_gauge_h = assemble(groups, fibers, local)
        covariance = float(
            np.linalg.norm(
                exact_gauge_h
                - rotate_kernel(
                    exact_hamiltonian.reshape(ngrid, nstates, ngrid, nstates),
                    gauge.reshape(ngrid, nstates, nstates),
                ).reshape(exact_gauge_h.shape)
            )
            / np.linalg.norm(exact_gauge_h)
        )
        patch_svd = {}
        selected_svd_h = None
        selected_svd_info = None
        for rank in ranks:
            started = time.perf_counter()
            fitted_h, infos = fit_svd_model(
                local,
                fibers,
                groups,
                rank,
                electronic_mode=args.cross_electronic_mode,
            )
            entry = field_error(fitted_h, exact_gauge_h)
            entry["hamiltonian_relative_error"] = entry.pop("relative_error")
            entry["hamiltonian_max_abs_error"] = entry.pop("max_abs_error")
            entry["fields"] = infos
            entry["seconds"] = time.perf_counter() - started
            patch_svd[str(rank)] = entry
            print(
                f"[TT-SVD] {patch} rank={rank} "
                f"H error={entry['hamiltonian_relative_error']:.3e}",
                flush=True,
            )
            if rank == cross_rank:
                selected_svd_h = fitted_h
                selected_svd_info = entry
        if selected_svd_h is None:
            selected_svd_h, infos = fit_svd_model(
                local,
                fibers,
                groups,
                cross_rank,
                electronic_mode=args.cross_electronic_mode,
            )
            selected_svd_info = field_error(selected_svd_h, exact_gauge_h)

        print(f"[TT-cross] fitting {patch}-patch fields", flush=True)
        started = time.perf_counter()
        crossed_h, cross_infos, sampling = fit_cross_model(
            local,
            aligned,
            fibers,
            groups,
            cross_rank,
            sweeps=args.cross_sweeps,
            rtol=args.cross_rtol,
            validation=args.cross_validation,
            seed=args.seed,
            start_rank=args.cross_start_rank,
            kick_rank=args.cross_kick_rank,
            electronic_mode=args.cross_electronic_mode,
        )
        cross_entry = field_error(crossed_h, exact_gauge_h)
        cross_entry["hamiltonian_relative_error"] = cross_entry.pop("relative_error")
        cross_entry["hamiltonian_max_abs_error"] = cross_entry.pop("max_abs_error")
        cross_entry["fields"] = cross_infos
        cross_entry["sampling"] = sampling
        cross_entry["seconds"] = time.perf_counter() - started
        print(
            f"[TT-cross] {patch} H error={cross_entry['hamiltonian_relative_error']:.3e} "
            f"samples={sampling['scalar_samples']}/{sampling['full_scalar_entries']}",
            flush=True,
        )

        summary["svd"][patch] = patch_svd
        summary["cross"][patch] = cross_entry
        summary.setdefault("exact_covariance_relative_error", {})[patch] = covariance
        summary["dynamics"][patch] = {}
        for method, hamiltonian in (("svd", selected_svd_h), ("cross", crossed_h)):
            print(f"[TT] propagating {patch} {method}", flush=True)
            metrics, method_curves = dynamics_metrics(
                hamiltonian,
                gauge,
                original_initial,
                exact_states,
                exact_observables,
                grids,
                primary_gauge,
                times,
            )
            summary["dynamics"][patch][method] = metrics
            curves[patch][method] = method_curves
            prefix = f"{patch}_{method}"
            saved[f"{prefix}_populations"] = method_curves["populations"]
            saved[f"{prefix}_means"] = method_curves["means"]
            saved[f"{prefix}_population_error"] = method_curves["population_error"]
            saved[f"{prefix}_fidelity"] = method_curves["fidelity"]
        del aligned, fibers, exact_gauge_h, selected_svd_h, crossed_h

    with np.load(args.single_gauge) as archive:
        single_local = np.asarray(archive["aligned_local_hamiltonian"], dtype=complex)
    summary["oracle"] = oracle_check(
        args,
        grids,
        energies,
        single_local,
        reference_index(grids),
    )
    np.savez(args.output_dir / "so2_procrustes_tt_dynamics.npz", **saved)
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(_json(summary), stream, indent=2)
        stream.write("\n")
    plot_rank_profile(args.output_dir / "so2_procrustes_tt_ranks", ranks, summary["svd"])
    plot_dynamics(
        args.output_dir / "so2_procrustes_tt_dynamics",
        times,
        exact_observables,
        curves,
    )
    print(json.dumps(_json(summary), indent=2))


if __name__ == "__main__":
    main()
