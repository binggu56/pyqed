#!/usr/bin/env python3
"""Minimal ab initio LDR wavepacket simulation for H3+.

This example builds a tiny body-fixed H3+ grid in internal coordinates
``(r1, r2, theta)``, computes native PyQED RHF/CASCI adiabatic energies and
direct CASCI overlaps, and runs a short LDR propagation with the triatomic
solver.

The example deliberately does not do state tracking or phase matching.  The
LDR overlap matrix is built directly from the electronic wavefunctions in the
same body-fixed geometry convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.phys import gwp
from pyqed.qchem import CASCI, Molecule
from pyqed.qchem.mcscf.casci import overlap
from pyqed.units import au2fs


def h3plus_body_frame(r1: float, r2: float, theta: float):
    """Return body-fixed H3+ atom specification in bohr.

    Atom 1 is on the +x axis, atom 2 is at the origin, and atom 3 is in the
    xy plane.  This fixes the rotational gauge before every electronic
    structure call.
    """
    return [
        ["H", (float(r1), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r2) * np.cos(theta), float(r2) * np.sin(theta), 0.0)],
    ]


def run_casci_point(r1, r2, theta, basis, nstates, ncas, nelecas, verbose=0):
    mol = Molecule(
        atom=h3plus_body_frame(r1, r2, theta),
        basis=basis,
        charge=1,
        spin=0,
        unit="bohr",
    )
    mol.build()
    mf = mol.RHF(verbose=verbose).run(max_cycle=80)
    mc = CASCI(mf, ncas=ncas, nelecas=nelecas, verbose=verbose).run(nstates=nstates)
    return mc


def _set_worker_thread_limits(worker_threads):
    if worker_threads is None:
        return
    value = str(int(worker_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def _h3plus_scan_point_worker(task):
    idx, r1, r2, theta, basis, nstates, ncas, nelecas, worker_threads = task
    _set_worker_thread_limits(worker_threads)
    mc = run_casci_point(r1, r2, theta, basis, nstates, ncas, nelecas)
    energies = np.asarray(mc.e_tot[:nstates], dtype=float)
    return idx, energies, mc


def _scan_tasks(solver, basis, nstates, ncas, nelecas, worker_threads):
    tasks = []
    for idx in np.ndindex(*solver.nx):
        tasks.append((
            idx,
            float(solver.x[0][idx[0]]),
            float(solver.x[1][idx[1]]),
            float(solver.x[2][idx[2]]),
            basis,
            nstates,
            ncas,
            nelecas,
            worker_threads,
        ))
    return tasks


def _run_scan_tasks(tasks, nx, nstates, n_workers):
    apes = np.zeros((*nx, nstates), dtype=float)
    mc_grid = np.empty(nx, dtype=object)
    total = len(tasks)

    if n_workers <= 1:
        for count, task in enumerate(tasks, start=1):
            idx, energies, mc = _h3plus_scan_point_worker(task)
            mc_grid[idx] = mc
            apes[idx] = energies
            print_scan_progress(count, total, task)
        return apes, mc_grid

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_h3plus_scan_point_worker, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            idx, energies, mc = future.result()
            mc_grid[idx] = mc
            apes[idx] = energies
            print(f"[scan] {count:3d}/{total}: completed idx={idx}")

    return apes, mc_grid


def print_scan_progress(count, total, task):
    idx, r1, r2, theta, *_ = task
    print(
        f"[scan] {count:3d}/{total}: "
        f"idx={idx} r1={r1:.6f} r2={r2:.6f} theta={theta:.6f}"
    )


def build_full_overlap(mc_grid, nstates):
    flat = mc_grid.ravel()
    ngrid = flat.size
    A = np.zeros((ngrid, nstates, ngrid, nstates), dtype=float)
    for a in range(ngrid):
        A[a, :, a, :] = np.eye(nstates)
        for b in range(a + 1, ngrid):
            sab = np.asarray(overlap(flat[a], flat[b]), dtype=float)
            A[a, :, b, :] = sab
            A[b, :, a, :] = sab.T
    return A


def build_linked_overlap(solver, mc_grid, nstates):
    links = {}
    for idx in np.ndindex(*solver.nx):
        for axis in range(solver.ndim):
            if idx[axis] + 1 >= solver.nx[axis]:
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt = tuple(nxt)
            links[(axis, idx)] = np.asarray(overlap(mc_grid[idx], mc_grid[nxt]), dtype=float)
    return solver._build_linked_overlap_from_links(links, nstates).reshape(
        int(np.prod(solver.nx)),
        nstates,
        int(np.prod(solver.nx)),
        nstates,
    )


def scan_h3plus_grid(
    solver,
    basis,
    nstates,
    ncas,
    nelecas,
    overlap_method="full",
    n_workers=1,
    worker_threads=1,
):
    """Compute APES and CASCI overlaps on the DVR grid."""
    nx = tuple(solver.nx)
    tasks = _scan_tasks(solver, basis, nstates, ncas, nelecas, worker_threads)
    apes, mc_grid = _run_scan_tasks(tasks, nx, nstates, int(n_workers))

    overlap_method = overlap_method.lower()
    if overlap_method == "full":
        A = build_full_overlap(mc_grid, nstates)
    elif overlap_method == "linked":
        A = build_linked_overlap(solver, mc_grid, nstates)
    else:
        raise ValueError("--overlap-method must be 'full' or 'linked'.")

    return apes, A.reshape((*nx, nstates, *nx, nstates))


def make_initial_wavepacket(
    solver,
    state=1,
    width=18.0,
    widths=None,
    project_reference_state=False,
    reference_index=None,
):
    """Build a normalized Gaussian packet on one adiabatic state."""
    psi_values = np.zeros((*solver.nx, solver.nrot, solver.nstates), dtype=complex)
    center = np.array([axis[len(axis) // 2] for axis in solver.x])
    if widths is None:
        widths = np.full(solver.ndim, float(width))
    else:
        widths = np.asarray(widths, dtype=float)
        if widths.shape != (solver.ndim,):
            raise ValueError("widths must have one value per internal coordinate.")
    if reference_index is None:
        reference_index = tuple(len(axis) // 2 for axis in solver.x)
    else:
        reference_index = tuple(reference_index)

    if project_reference_state:
        if solver.overlap_matrix is None:
            raise RuntimeError("Reference-state projection requires solver.overlap_matrix.")
        if len(reference_index) != solver.ndim:
            raise ValueError("reference_index must have one entry per internal coordinate.")
        ref_flat = np.ravel_multi_index(reference_index, solver.nx)
        ng = int(np.prod(solver.nx))
        overlap_matrix = solver.overlap_matrix.reshape(
            ng,
            solver.nstates,
            ng,
            solver.nstates,
        )

    for idx in np.ndindex(*solver.nx):
        q = np.array([solver.x[axis][idx[axis]] for axis in range(solver.ndim)])
        amp = np.exp(-np.sum(widths * (q - center) ** 2))
        if project_reference_state:
            flat = np.ravel_multi_index(idx, solver.nx)
            psi_values[idx + (0, slice(None))] = amp * overlap_matrix[
                flat,
                :,
                ref_flat,
                state,
            ]
        else:
            psi_values[idx + (0, state)] = amp

    psi = solver.to_quadrature_normalized(psi_values)
    norm = solver.norm(psi)
    if norm == 0:
        raise RuntimeError("Initial wavepacket norm is zero.")
    return psi / norm


def rotational_shape(solver):
    if solver.nrot == 1:
        return (*solver.nx, solver.nstates)
    return (*solver.nx, solver.nrot, solver.nstates)


def maybe_drop_rot_axis(solver, psi):
    """Use the simpler vibronic shape when J=0."""
    if solver.nrot == 1:
        return psi[..., 0, :]
    return psi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--npts", type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument("--r-min", type=float, default=1.45)
    parser.add_argument("--r-max", type=float, default=1.75)
    parser.add_argument("--theta-min", type=float, default=0.95)
    parser.add_argument("--theta-max", type=float, default=1.15)
    parser.add_argument("--dvr-type", default="default", choices=["default", "sine"])
    parser.add_argument("--J", type=int, default=0)
    parser.add_argument("--Jz", type=int, default=None)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--nt", type=int, default=2)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument("--width", type=float, default=18.0)
    parser.add_argument(
        "--widths",
        type=float,
        nargs=3,
        default=None,
        metavar=("R1", "R2", "THETA"),
        help="Anisotropic Gaussian widths for r1, r2, theta. Overrides --width.",
    )
    parser.add_argument("--overlap-method", choices=["full", "linked"], default="full")
    parser.add_argument(
        "--kinetic-propagator",
        choices=["dense", "expm_multiply", "chebyshev"],
        default="dense",
        help="Use dense expm precomputation, expm_multiply, or Chebyshev action propagation.",
    )
    parser.add_argument(
        "--kinetic-action",
        choices=["dense", "matrix-free"],
        default="dense",
        help="Use a dense flat kinetic matrix or a LinearOperator kinetic action.",
    )
    parser.add_argument("--chebyshev-tol", type=float, default=1e-12)
    parser.add_argument("--chebyshev-max-order", type=int, default=4096)
    parser.add_argument(
        "--chebyshev-bounds",
        choices=["endpoints", "exact", "eigsh", "gershgorin"],
        default="gershgorin",
        help="Method used to estimate spectral bounds for Chebyshev scaling.",
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument(
        "--project-reference-state",
        action="store_true",
        help="Project the selected state at the central grid point into each local LDR basis.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).with_name("h3plus_abinitio_ldr_cache.npz"),
    )
    parser.add_argument("--force-scan", action="store_true")
    args = parser.parse_args()

    solver = Triatom(
        h3plus_body_frame(1.60, 1.60, np.pi / 3.0),
        basis=args.basis,
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
        J=args.J,
        Jz=args.Jz,
    )
    solver.set_dvr(
        domains=[
            [args.r_min, args.r_max],
            [args.r_min, args.r_max],
            [args.theta_min, args.theta_max],
        ],
        npts=args.npts,
        dvr_type=args.dvr_type,
    )

    if args.cache.exists() and not args.force_scan:
        data = np.load(args.cache)
        solver.apes = data["apes"]
        solver.overlap_matrix = data["overlap_matrix"]
        print(f"[cache] Loaded {args.cache}")
    else:
        solver.apes, solver.overlap_matrix = scan_h3plus_grid(
            solver,
            basis=args.basis,
            nstates=args.nstates,
            ncas=args.ncas,
            nelecas=args.nelecas,
            overlap_method=args.overlap_method,
            n_workers=args.n_workers,
            worker_threads=args.worker_threads,
        )
        np.savez(
            args.cache,
            apes=solver.apes,
            overlap_matrix=solver.overlap_matrix,
            npts=np.asarray(args.npts, dtype=int),
            domains=np.asarray(
                [
                    [args.r_min, args.r_max],
                    [args.r_min, args.r_max],
                    [args.theta_min, args.theta_max],
                ],
                dtype=float,
            ),
            dvr_type=np.asarray(args.dvr_type),
        )
        print(f"[cache] Saved {args.cache}")

    if args.initial_state < 0 or args.initial_state >= args.nstates:
        raise ValueError("--initial-state must be between 0 and nstates-1.")

    psi0 = make_initial_wavepacket(
        solver,
        state=args.initial_state,
        width=args.width,
        widths=args.widths,
        project_reference_state=args.project_reference_state,
    )
    psi0 = maybe_drop_rot_axis(solver, psi0)
    print(f"[setup] psi0 shape = {psi0.shape}, expected = {rotational_shape(solver)}")
    print(f"[setup] APES min/max = {solver.apes.min():.8f} / {solver.apes.max():.8f} Eh")

    result = solver.run(
        psi0,
        dt=args.dt_fs / au2fs,
        nt=args.nt,
        nout=args.nout,
        kinetic_propagator=args.kinetic_propagator,
        chebyshev_tol=args.chebyshev_tol,
        chebyshev_max_order=args.chebyshev_max_order,
        chebyshev_bounds=args.chebyshev_bounds,
        kinetic_action=args.kinetic_action,
    )
    pops = solver.get_population(result, plot=False)
    print("[result] times/fs =", result["times"] * au2fs)
    print("[result] populations =")
    print(pops)
    print("[result] total population =", pops.sum(axis=1))

    out = args.cache.with_name(args.cache.stem + "_dynamics.npz")
    np.savez(out, times=result["times"], populations=pops)
    print(f"[result] Saved dynamics to {out}")


if __name__ == "__main__":
    main()
