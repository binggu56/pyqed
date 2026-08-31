#!/usr/bin/env python3
"""Run cached SO2 Procrustes-gauge dynamics with a genuine TT/MPO backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.so2_casci_cgldr import DEFAULT_SCAN_DIR, load_so2_linked_scan
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet, observables
from examples.ldr.so2_casci_full_ldr import full_hamiltonian, path_overlap
from examples.ldr.so2_procrustes_dynamics import propagate
from examples.ldr.so2_procrustes_overlap_mpo import (
    DEFAULT_LINK_DIR,
    LABELS,
)
from examples.ldr.so2_procrustes_tt import ArrayGaugeOracle, DEFAULT_REFERENCE, DEFAULT_TWO
from examples.ldr.so2_procrustes_gauge import rotate_kernel
from pyqed.ldr.overlap import unpack
from pyqed.ldr import AbInitioFit
from pyqed.ldr.ttfit import (
    field_mpo,
    fit_ey,
    fit_mpo,
)
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.functional import FunctionalTT
from pyqed.mps.mps import MPS
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.tdvp import TDVPEngine
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2fs


DEFAULT_ENERGY_DIR = Path("/private/tmp/so2_procrustes_functional_cross")


def fitted_field_signature(directory):
    """Return a cheap cache key for one saved AbInitioFit artifact set."""
    if directory is None:
        return None
    directory = Path(directory)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return (("summary.json", None, None),)
    summary = json.loads(summary_path.read_text())
    names = [
        "summary.json",
        summary["grids"],
        summary["energy_model"],
        *summary["link_models"],
    ]
    signature = []
    for name in names:
        path = directory / name
        stat = path.stat()
        signature.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


def matrix_field_mpo(values, rank, operator_rank=None):
    values = np.asarray(values, dtype=complex)
    shape = values.shape[:-2]
    nstates = values.shape[-1]
    cores = {
        (alpha, beta): decompose(values[..., alpha, beta], rank=int(rank))
        for alpha in range(nstates)
        for beta in range(nstates)
    }
    operator = field_mpo(cores, shape, nstates)
    operator = 0.5 * (operator + operator.adjoint())
    if operator_rank is not None and max(operator.bond_orders()) > int(operator_rank):
        operator = operator.compress_hermitian(int(operator_rank))
    return operator


def physical_projectors(gauge, primary_gauge, rank, operator_rank):
    nstates = gauge.shape[-1]
    operators = []
    for state in range(nstates):
        vector = primary_gauge[..., :, state]
        projector = np.einsum(
            "...ia,...i,...j,...jb->...ab",
            gauge.conj(),
            vector,
            vector.conj(),
            gauge,
            optimize=True,
        )
        operators.append(matrix_field_mpo(projector, rank, operator_rank))
    return operators


def initial_state(packet, primary_gauge, gauge, state, rank):
    physical = packet[..., None] * primary_gauge[..., :, int(state)]
    aligned = np.einsum("...ia,...i->...a", gauge.conj(), physical, optimize=True)
    return MPS(decompose(aligned, rank=int(rank))).normalize(), physical.reshape(-1)


def feature_anchors(shape, scheme):
    center = tuple(size // 2 for size in shape)
    if scheme == "center":
        return (center,)
    if scheme == "axes":
        anchors = [center]
        for axis, size in enumerate(shape):
            for position in (0, size - 1):
                index = list(center)
                index[axis] = position
                anchors.append(tuple(index))
        return tuple(dict.fromkeys(anchors))
    if scheme == "cuts":
        anchors = [center]
        for axis, size in enumerate(shape):
            for position in range(size):
                index = list(center)
                index[axis] = position
                anchors.append(tuple(index))
        return tuple(dict.fromkeys(anchors))
    if scheme == "corners":
        corners = [
            tuple((size - 1) * bit for size, bit in zip(shape, bits))
            for bits in np.ndindex(*(2,) * len(shape))
        ]
        return tuple(dict.fromkeys([center, *corners]))
    if scheme == "grid3":
        axes = [tuple(dict.fromkeys((0, size // 2, size - 1))) for size in shape]
        return tuple(
            tuple(axes[axis][position] for axis, position in enumerate(index))
            for index in np.ndindex(*(len(axis) for axis in axes))
        )
    raise ValueError(f"unknown feature-anchor scheme {scheme}")


def plot_results(path, times, reference, tensor):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for state in (1, 2):
        line = axes[0].plot(
            times,
            reference[:, state],
            lw=1.8,
            label=rf"Full LDR $P_{state}$",
        )[0]
        axes[0].plot(
            times,
            tensor[:, state],
            "--",
            color=line.get_color(),
            label=rf"TT/TDVP $P_{state}$",
        )
    axes[1].semilogy(
        times,
        np.maximum(np.max(np.abs(tensor - reference), axis=1), 1.0e-16),
        color="#D55E00",
    )
    axes[0].set(xlabel="Time (fs)", ylabel="Population", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Time (fs)", ylabel="Maximum population error")
    axes[0].legend(frameon=False, fontsize=8)
    for label, axis in zip("ab", axes):
        axis.text(0.02, 0.96, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.grid(False)
    fig.savefig(path.with_suffix(".png"), dpi=350)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_TWO)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_procrustes_tdvp"))
    parser.add_argument("--fit-rank", type=int, default=32)
    parser.add_argument(
        "--fit",
        choices=("ey", "direct", "dressed"),
        default="ey",
    )
    parser.add_argument("--link-dir", type=Path, default=DEFAULT_LINK_DIR)
    parser.add_argument(
        "--field-dir",
        type=Path,
        help="Directory written by AbInitioFit.save().",
    )
    parser.add_argument("--link-rank", type=int, default=32)
    parser.add_argument("--link-patch", choices=("single", "two"), default="two")
    parser.add_argument("--path-order", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--energy-dir", type=Path, default=DEFAULT_ENERGY_DIR)
    parser.add_argument("--energy-rank", type=int, default=32)
    parser.add_argument(
        "--feature-anchors",
        choices=("center", "axes", "cuts", "corners", "grid3"),
        default="cuts",
    )
    parser.add_argument("--feature-rank", type=int)
    parser.add_argument("--operator-rank", type=int, default=96)
    parser.add_argument("--state-rank", type=int, default=48)
    parser.add_argument("--tdvp-workers", type=int, default=6)
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--projector-rank", type=int, default=32)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-validation", type=int, default=1024)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-8)
    parser.add_argument("--cross-start-rank", type=int, default=1)
    parser.add_argument("--cross-kick-rank", type=int, default=2)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--integrator", choices=("tdvp2", "tdvp", "hybrid"), default="tdvp2")
    parser.add_argument(
        "--split-mpo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep KEO groups as separate MPO components during TDVP2.",
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-11)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)
        local = np.asarray(archive["aligned_local_hamiltonian"], dtype=complex)

    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    ngrid = int(np.prod(shape))
    overlap = path_overlap(shape, links).reshape(ngrid, nstates, ngrid, nstates)
    aligned_overlap = rotate_kernel(overlap, gauge.reshape(ngrid, nstates, nstates))
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, axes = dense_kinetic(scan, *grids)
    terms = scan.solver.buildK_qsqa_terms(axes, symmetrize=True, svd_tol=0.0)

    fit_signature = {
        "grid": tuple(shape),
        "fit_rank": args.fit_rank,
        "fit": args.fit,
        "feature_anchors": args.feature_anchors,
        "feature_rank": args.feature_rank,
        "operator_rank": args.operator_rank,
        "cross_sweeps": args.cross_sweeps,
        "cross_validation": args.cross_validation,
        "cross_rtol": args.cross_rtol,
        "cross_start_rank": args.cross_start_rank,
        "cross_kick_rank": args.cross_kick_rank,
        "seed": args.seed,
        "split_mpo": args.split_mpo,
        "gauge": str(args.gauge.resolve()),
        "link_dir": str(args.link_dir.resolve()),
        "field_dir": None if args.field_dir is None else str(args.field_dir.resolve()),
        "field_artifacts": fitted_field_signature(args.field_dir),
        "link_rank": args.link_rank,
        "link_patch": args.link_patch,
        "path_order": tuple(args.path_order),
        "energy_dir": str(args.energy_dir.resolve()),
        "energy_rank": args.energy_rank,
    }
    fit_cache = args.output_dir / "hamiltonian.pkl"
    restored_fit = False
    started = time.perf_counter()
    if args.reuse_fit and fit_cache.exists():
        with fit_cache.open("rb") as stream:
            cached = pickle.load(stream)
        if cached.get("signature") == fit_signature:
            hamiltonian = cached["hamiltonian"]
            fit_info = cached["info"]
            restored_fit = True
    if not restored_fit:
        oracle = ArrayGaugeOracle(local, aligned_overlap)
        common = {
            "max_rank": args.fit_rank,
            "operator_rank": args.operator_rank,
            "sweeps": args.cross_sweeps,
            "rtol": args.cross_rtol,
            "validation": args.cross_validation,
            "seed": args.seed,
            "start_rank": args.cross_start_rank,
            "kick_rank": args.cross_kick_rank,
            "split": args.split_mpo,
        }
        if args.fit == "ey":
            hamiltonian, fit_info = fit_ey(
                oracle,
                terms,
                shape,
                nstates,
                feature_anchors(shape, args.feature_anchors),
                feature_rank=args.feature_rank,
                **common,
            )
        elif args.fit == "direct":
            hamiltonian, fit_info = fit_mpo(
                oracle,
                terms,
                shape,
                nstates,
                **common,
            )
        else:
            if not args.split_mpo:
                raise ValueError("the dressed backend requires --split-mpo")
            fitted = None
            if args.field_dir is None:
                link_paths = tuple(
                    args.link_dir
                    / f"link_{label}_{args.link_patch}_rank{args.link_rank}.npz"
                    for label in LABELS
                )
                energy_path = (
                    args.energy_dir
                    / f"ebar_{args.link_patch}_cross_rank{args.energy_rank}.npz"
                )
                models = tuple(FunctionalTT.load(path) for path in link_paths)
                energy_model = FunctionalTT.load(energy_path)
            else:
                fitted = AbInitioFit.load(args.field_dir)
                if fitted.labels != tuple(LABELS):
                    raise ValueError(
                        f"fitted coordinate labels {fitted.labels} != {tuple(LABELS)}"
                    )
                if len(fitted.grids) != len(grids):
                    raise ValueError("fitted field has the wrong number of coordinates")
                for expected, actual in zip(grids, fitted.grids):
                    np.testing.assert_allclose(expected, actual)
                models = fitted.links
                energy_model = fitted.energy
                energy_path = fitted.paths["energy"]
            options = dict(
                path_order=tuple(args.path_order),
                overlap_rank=args.fit_rank,
                overlap_sweeps=args.cross_sweeps,
                overlap_rtol=args.cross_rtol,
                overlap_validation=args.cross_validation,
                cross_start=args.cross_start_rank,
                cross_kick=args.cross_kick_rank,
                operator_rank=args.operator_rank,
                potential_rank=args.operator_rank,
                seed=args.seed,
            )
            driver = (
                TTLDR(
                    energy=energy_model,
                    links=models,
                    grids=grids,
                    keo=terms,
                    **options,
                )
                if fitted is None
                else TTLDR.from_fit(fitted, keo=terms, **options)
            )
            if fitted is not None:
                fitted.close()
            hamiltonian = driver.components
            fit_info = {
                "backend": "ttldr-fitted-fields",
                "kinetic": driver.overlap_info,
                "potential": {
                    **driver.potential_info,
                    "source": str(energy_path),
                    "functional_ranks": tuple(energy_model.ranks_),
                    "electronic_structure_calls": 0,
                },
                "components": len(hamiltonian),
                "operator_ranks": [
                    tuple(component.bond_orders()) for component in hamiltonian
                ],
            }
        with fit_cache.open("wb") as stream:
            pickle.dump(
                {
                    "signature": fit_signature,
                    "hamiltonian": hamiltonian,
                    "info": fit_info,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    fit_seconds = time.perf_counter() - started
    projectors = physical_projectors(
        gauge,
        primary_gauge,
        args.projector_rank,
        args.operator_rank,
    )
    packet = nuclear_packet(*grids, axes)
    state, physical_initial = initial_state(
        packet,
        primary_gauge,
        gauge,
        args.initial_state,
        args.state_rank,
    )

    steps = int(round(args.time_fs / args.dt_fs))
    dt = args.dt_fs / au2fs
    initial_populations = np.asarray([state.expectation(op) for op in projectors]).real
    started = time.perf_counter()
    if args.split_mpo:
        if args.integrator != "tdvp2":
            raise ValueError("split-MPO propagation currently requires TDVP2")
        engine = TDVPEngine(
            hamiltonian,
            max_bond=args.state_rank,
            cutoff=args.cutoff,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            workers=args.tdvp_workers,
        )
        current = state
        measured = []
        truncation_errors = []
        bond_dimensions = [tuple(current.bond_orders())]
        for step in range(steps):
            current, info = engine.step(current, dt, normalize=True)
            measured.append([current.expectation(op) for op in projectors])
            truncation_errors.append(info["truncation_error"])
            bond_dimensions.append(tuple(current.bond_orders()))
            if args.progress:
                print(f"[TDVP2] {step + 1}/{steps}", flush=True)
        final_state = current
        measured = np.asarray(measured, dtype=complex)
    else:
        dynamics = TDMPS(hamiltonian, D=args.state_rank, normalize=True)
        dynamics.run(
            state,
            dt=dt,
            steps=steps,
            e_ops=projectors,
            interval=1,
            integrator=args.integrator,
            cutoff=args.cutoff,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            progress=args.progress,
            track_energy=False,
        )
        final_state = dynamics.final_state
        measured = dynamics.observables
        truncation_errors = dynamics.tdvp_truncation_errors
        bond_dimensions = dynamics.bond_dimensions
    tdvp_seconds = time.perf_counter() - started
    times = np.arange(steps + 1, dtype=float) * args.dt_fs
    tt_populations = np.vstack((initial_populations, measured.real))

    exact_hamiltonian = full_hamiltonian(kinetic, overlap, energies)
    started = time.perf_counter()
    exact_states = propagate(exact_hamiltonian, physical_initial, times)
    reference_seconds = time.perf_counter() - started
    reference_populations = observables(exact_states, grids, primary_gauge)[1]

    final_aligned = np.asarray(
        tt_to_tensor(
            [final_state._get_std_B(site) for site in range(final_state.L)]
        )
    ).reshape(*shape, nstates)
    final_physical = np.einsum("...ia,...a->...i", gauge, final_aligned, optimize=True).reshape(-1)
    final_fidelity = float(
        abs(np.vdot(exact_states[-1], final_physical)) ** 2
        / (np.vdot(exact_states[-1], exact_states[-1]).real * np.vdot(final_physical, final_physical).real)
    )
    population_error = float(np.max(np.abs(tt_populations - reference_populations)))
    summary = {
        "grid": list(shape),
        "fit_rank": args.fit_rank,
        "fit_backend": args.fit,
        "operator_rank": args.operator_rank,
        "state_rank": args.state_rank,
        "tdvp_workers": args.tdvp_workers,
        "split_mpo": args.split_mpo,
        "fit_seconds": fit_seconds,
        "restored_fit": restored_fit,
        "tdvp_seconds": tdvp_seconds,
        "full_ldr_seconds": reference_seconds,
        "final_fidelity": final_fidelity,
        "max_population_error": population_error,
        "max_tdvp_truncation_error": float(np.max(truncation_errors, initial=0.0)),
        "max_state_bond": int(np.max(bond_dimensions)),
        "fit": fit_info,
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    np.savez(
        args.output_dir / "dynamics.npz",
        times_fs=times,
        reference_populations=reference_populations,
        tt_populations=tt_populations,
    )
    plot_results(
        args.output_dir / "so2_procrustes_tdvp",
        times,
        reference_populations,
        tt_populations,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
