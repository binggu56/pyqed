#!/usr/bin/env python3
"""Build and propagate a matched spin-pure SO2 CASCI full-LDR reference."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from examples.ldr.so2_casci_cgldr import (
    DEFAULT_SCAN_DIR,
    REFERENCE_BOND,
    REFERENCE_THETA_DEG,
    SQRT2,
    active_space_gaps,
    casci_overlap_active,
    casci_reference_point,
    load_so2_linked_scan,
    require_smooth_active_space,
    so2_qs_theta_body_frame,
)
from examples.ldr.so2_casci_cgldr_dense import (
    dense_kinetic,
    nuclear_packet,
    observables,
)
from pyqed.ldr import CGLDRElectronicData
from pyqed.units import au2fs


NSTATES = 3
STATE_IDS = (0, 1, 2)


def _electronic_point(task):
    if len(task) == 6:
        index, qs, theta, qa, basis, derivative_workers = task
        nstates = NSTATES
    else:
        index, qs, theta, qa, basis, derivative_workers, nstates = task[:7]
    spin_root_cushion = int(task[7]) if len(task) > 7 else 8
    direct_ci_eigensolver = task[8] if len(task) > 8 else None
    direct_ci_auto_spin0 = task[9] if len(task) > 9 else None
    state_ids = tuple(range(int(nstates)))
    point = casci_reference_point(
        so2_qs_theta_body_frame(qs, theta, qa),
        basis=basis,
        charge=0,
        spin=0,
        unit="bohr",
        ncas=6,
        nelecas=6,
        nstates=nstates,
        scf_tol=1.0e-8,
        scf_max_cycle=80,
        multiplicity=1,
        eri_workers=derivative_workers,
        spin_root_cushion=spin_root_cushion,
        direct_ci_eigensolver=direct_ci_eigensolver,
        direct_ci_auto_spin0=direct_ci_auto_spin0,
    )
    gaps = require_smooth_active_space(point)
    energies = np.asarray(point.e_tot, dtype=float)
    spin_square = np.asarray([point.spin_square(i) for i in state_ids])
    return index, point.frame(), energies, spin_square, np.asarray(gaps)


def _overlap_link(task):
    if len(task) == 4:
        axis, index, left, right = task
        state_ids = STATE_IDS
    else:
        axis, index, left, right, state_ids = task
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = casci_overlap_active(left, right, state_ids)
    return axis, index, overlap


def cache_file(cache_dir, index):
    return cache_dir / ("point_" + "_".join(map(str, index)) + ".pkl")


def save_cached(path, result):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def acquire_points(
    grids,
    basis,
    output,
    *,
    workers,
    reuse,
    derivative_workers,
    nstates=NSTATES,
    spin_root_cushion=8,
    direct_ci_eigensolver=None,
    direct_ci_auto_spin0=None,
):
    shape = tuple(len(grid) for grid in grids)
    nstates = int(nstates)
    frames = np.empty(shape, dtype=object)
    energies = np.empty((*shape, nstates), dtype=float)
    spin_square = np.empty_like(energies)
    gaps = np.empty((*shape, 2), dtype=float)
    cache_dir = output / "point_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            index,
            float(grids[0][index[0]]),
            float(grids[1][index[1]]),
            float(grids[2][index[2]]),
            basis,
            int(derivative_workers),
            nstates,
            int(spin_root_cushion),
            direct_ci_eigensolver,
            direct_ci_auto_spin0,
        )
        for index in np.ndindex(shape)
    ]

    def assign(result):
        index, frame, point_energies, point_s2, point_gaps = result
        frames[index] = frame
        energies[index] = point_energies
        spin_square[index] = np.real(np.real_if_close(point_s2))
        gaps[index] = point_gaps

    pending = []
    completed = 0
    for task in tasks:
        path = cache_file(cache_dir, task[0])
        if reuse and path.is_file():
            with path.open("rb") as stream:
                assign(pickle.load(stream))
            completed += 1
        else:
            pending.append(task)
    if completed:
        print(f"[full LDR] restored {completed}/{len(tasks)} points", flush=True)
    report_every = max(1, len(tasks) // 20)

    def report(count):
        if count % report_every == 0 or count == len(tasks):
            suffix = f" ({workers} workers)" if workers > 1 else ""
            print(f"[full LDR] CASCI {count}/{len(tasks)}{suffix}", flush=True)

    if workers == 1:
        for count, task in enumerate(pending, start=completed + 1):
            result = _electronic_point(task)
            assign(result)
            save_cached(cache_file(cache_dir, task[0]), result)
            report(count)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(_electronic_point, task): task for task in pending
            }
            for count, future in enumerate(
                as_completed(futures), start=completed + 1
            ):
                result = future.result()
                assign(result)
                task = futures[future]
                save_cached(cache_file(cache_dir, task[0]), result)
                report(count)
    return frames, energies, spin_square, gaps


def build_links(frames, *, workers, nstates=NSTATES):
    shape = frames.shape
    state_ids = tuple(range(int(nstates)))
    tasks = []
    for index in np.ndindex(shape):
        for axis in range(len(shape)):
            if index[axis] + 1 >= shape[axis]:
                continue
            neighbor = list(index)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            tasks.append((axis, index, frames[index], frames[neighbor], state_ids))
    links = {}
    if workers == 1:
        results = map(_overlap_link, tasks)
        for count, result in enumerate(results, start=1):
            axis, index, overlap = result
            links[(axis, index)] = overlap
            if count % 100 == 0 or count == len(tasks):
                print(f"[full LDR] overlap links {count}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_overlap_link, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                axis, index, overlap = future.result()
                links[(axis, index)] = overlap
                if count % 100 == 0 or count == len(tasks):
                    print(
                        f"[full LDR] overlap links {count}/{len(tasks)}",
                        flush=True,
                    )
    return links


def path_overlap(shape, links, nstates=None):
    from pyqed.namd.triatomic import Triatom

    if nstates is None:
        nstates = np.asarray(next(iter(links.values()))).shape[-1]
    solver = object.__new__(Triatom)
    solver.nx = np.asarray(shape, dtype=int)
    solver.ndim = len(shape)
    solver.overlap_path_average = True
    return solver._build_linked_overlap_from_links(links, int(nstates))


def full_hamiltonian(kinetic, overlap, energies):
    nuclear_size = kinetic.shape[0]
    nstates = np.asarray(energies).shape[-1]
    matrix = (
        kinetic[:, None, :, None]
        * overlap.reshape(nuclear_size, nstates, nuclear_size, nstates)
    ).reshape(nuclear_size * nstates, -1)
    shifted = energies - float(np.min(energies))
    for point, values in enumerate(shifted.reshape(-1, nstates)):
        begin = point * nstates
        matrix[begin : begin + nstates, begin : begin + nstates] += np.diag(values)
    return 0.5 * (matrix + matrix.conj().T)


def reference_transport(overlap, grids):
    center = (
        int(np.argmin(np.abs(grids[0] - SQRT2 * REFERENCE_BOND))),
        int(np.argmin(np.abs(grids[1] - np.deg2rad(REFERENCE_THETA_DEG)))),
        int(np.argmin(np.abs(grids[2]))),
    )
    selection = (
        (slice(None),) * 3
        + (slice(None),)
        + center
        + (slice(None),)
    )
    blocks = overlap[selection]
    left, _singular, right = np.linalg.svd(blocks, full_matrices=False)
    return left @ right, center


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--integral-workers", type=int, default=1)
    parser.add_argument("--nstates", type=int, default=NSTATES)
    parser.add_argument("--spin-root-cushion", type=int, default=8)
    parser.add_argument(
        "--direct-ci-eigensolver",
        choices=("auto", "davidson", "eigsh"),
        default="auto",
    )
    parser.add_argument("--disable-auto-spin0", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with np.load(args.grid_data, allow_pickle=False) as archive:
        simple_grid = all(name in archive for name in ("qs", "theta", "qa"))
        grids = (
            tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
            if simple_grid
            else None
        )
    if grids is None:
        data = CGLDRElectronicData.from_npz(args.grid_data)
        grids = tuple(np.asarray(grid) for grid in data.reactive_grids)
        grids += tuple(np.asarray(grid) for grid in data.expanded_grids)
        if tuple(data.metadata["sampled_coordinates"]) + tuple(
            data.metadata["expanded_coordinates"]
        ) != ("qs", "theta", "qa"):
            raise ValueError("grid_data must use coordinate order (qs, theta, qa)")

    frames, energies, spin_square, gaps = acquire_points(
        grids,
        args.basis,
        args.output_dir,
        workers=args.workers,
        reuse=args.reuse,
        derivative_workers=args.integral_workers,
        nstates=args.nstates,
        spin_root_cushion=args.spin_root_cushion,
        direct_ci_eigensolver=args.direct_ci_eigensolver,
        direct_ci_auto_spin0=False if args.disable_auto_spin0 else None,
    )
    electronic_seconds = time.perf_counter() - started
    if np.max(np.abs(spin_square)) > 1.0e-7:
        raise RuntimeError("Spin-pure singlet selection failed")
    links = build_links(frames, workers=args.workers, nstates=args.nstates)
    axes_index = np.asarray([key[0] for key in links], dtype=int)
    link_indices = np.asarray([key[1] for key in links], dtype=int)
    link_data = np.asarray(list(links.values()), dtype=complex)
    np.savez(
        args.output_dir / "electronic_reference.npz",
        energies=energies,
        spin_square=spin_square,
        active_space_gaps=gaps,
        qs=grids[0],
        theta=grids[1],
        qa=grids[2],
        link_axes=axes_index,
        link_indices=link_indices,
        link_data=link_data,
        nstates=int(args.nstates),
    )

    if args.build_only:
        elapsed = time.perf_counter() - started
        singular = np.asarray(
            [np.linalg.svd(block, compute_uv=False) for block in links.values()]
        )
        summary = {
            "method": f"RHF/CASCI(6e,6o)/{args.basis} electronic reference",
            "grid": list(energies.shape[:3]),
            "nstates": int(args.nstates),
            "electronic_points": int(np.prod(energies.shape[:3])),
            "overlap_links": len(links),
            "workers": args.workers,
            "spin_root_cushion": int(args.spin_root_cushion),
            "direct_ci_eigensolver": args.direct_ci_eigensolver,
            "direct_ci_auto_spin0": not args.disable_auto_spin0,
            "max_abs_s2": float(np.max(np.abs(spin_square))),
            "minimum_active_gap_eh": float(np.min(gaps)),
            "minimum_link_singular_value": float(np.min(singular)),
            "electronic_seconds": electronic_seconds,
            "total_seconds": elapsed,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    overlap = path_overlap(energies.shape[:3], links, nstates=args.nstates)
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, dvr_axes = dense_kinetic(scan, *grids)
    hamiltonian = full_hamiltonian(kinetic, overlap, energies)
    print(
        f"[full LDR] H={hamiltonian.shape}, Hermitian error="
        f"{np.max(np.abs(hamiltonian - hamiltonian.conj().T)):.3e}",
        flush=True,
    )

    packet = nuclear_packet(*grids, dvr_axes)
    transport, center = reference_transport(overlap, grids)
    psi0 = (packet[..., None] * transport[..., 2]).reshape(-1)
    psi0 /= np.linalg.norm(psi0)
    eigenvalues, eigenvectors = eigh(
        hamiltonian, overwrite_a=True, check_finite=False
    )
    times_fs = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)
    coefficients = eigenvectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times_fs / au2fs, eigenvalues))
    states = (phases * coefficients[None, :]) @ eigenvectors.conj().T
    values = observables(states, grids, transport)
    elapsed = time.perf_counter() - started
    np.savez(
        args.output_dir / "full_ldr_dynamics.npz",
        times_fs=times_fs,
        coordinate_names=np.asarray(("qs", "theta", "qa")),
        populations=values[0],
        reference_populations=values[1],
        means=values[2],
        variances=values[3],
        norms=values[4],
        qs=grids[0],
        theta=grids[1],
        qa=grids[2],
    )
    summary = {
        "method": f"RHF/CASCI(6e,6o)/{args.basis} full LDR",
        "grid": list(energies.shape[:3]),
        "electronic_points": int(np.prod(energies.shape[:3])),
        "workers": args.workers,
        "reference_index_qs_theta_qa": list(center),
        "max_abs_s2": float(np.max(np.abs(spin_square))),
        "minimum_active_gap_eh": float(np.min(gaps)),
        "final_reference_populations": values[1][-1].tolist(),
        "final_means_qs_theta_qa": values[2][-1].tolist(),
        "max_norm_error": float(np.max(np.abs(values[4] - 1.0))),
        "electronic_seconds": electronic_seconds,
        "total_seconds": elapsed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
