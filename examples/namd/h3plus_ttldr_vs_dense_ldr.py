#!/usr/bin/env python3
"""Benchmark genuine H3+ TT-LDR/TDVP dynamics against dense LDR."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs


def h3plus_body_frame(r=1.65, theta=np.pi / 3.0):
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


@contextlib.contextmanager
def pushd(path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def build_solver(args):
    theta0 = np.deg2rad(0.5 * (args.theta_min_deg + args.theta_max_deg))
    solver = Triatom(
        h3plus_body_frame(theta=theta0),
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.set_dvr(
        domains=[
            [args.r_min, args.r_max],
            [args.r_min, args.r_max],
            [np.deg2rad(args.theta_min_deg), np.deg2rad(args.theta_max_deg)],
        ],
        npts=[args.n_r, args.n_r, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )
    return solver


def load_or_scan(solver, args):
    apes_path = args.outdir / "apes.npz"
    overlap_path = args.outdir / "overlap_links.npz"
    if args.reuse_cache and apes_path.exists() and overlap_path.exists():
        solver.apes = np.load(apes_path)["data"]
        packed = np.load(overlap_path)
        solver.overlap_links = solver._unpack_overlap_links(
            packed["axes"], packed["indices"], packed["data"]
        )
        solver.overlap_matrix = None
        return "cache", 0.0
    start = time.perf_counter()
    with pushd(args.outdir):
        solver.scan_pes(
            electronic_method="am1/meci",
            nstates=args.nstates,
            ncas=args.ncas,
            nelecas=2,
            overlap_method="link-only",
            unitarize_overlap_links=False,
            n_workers=args.n_workers,
            worker_threads=1,
        )
    return "scan", time.perf_counter() - start


def initial_packet(solver, state, width):
    values = np.zeros((*solver.nx, solver.nstates), dtype=complex)
    center = np.asarray([axis[len(axis) // 2] for axis in solver.x])
    for index in np.ndindex(*solver.nx):
        point = np.asarray([solver.x[axis][index[axis]] for axis in range(solver.ndim)])
        values[index + (state,)] = np.exp(-width * np.sum((point - center) ** 2))
    values = solver.to_quadrature_normalized(values)
    return values / solver.norm(values)


def dense_propagate(state, hamiltonian, dt, steps, interval, nstates):
    states = [state.reshape(-1)]
    current = states[0]
    for step in range(1, steps + 1):
        current = expm_multiply(-1j * dt * hamiltonian, current)
        if step % interval == 0 or step == steps:
            states.append(current.copy())
    populations = np.asarray(
        [np.sum(np.abs(item.reshape(*state.shape)) ** 2, axis=(0, 1, 2)) for item in states]
    ).reshape(len(states), nstates)
    return states, populations


def plot_populations(times, dense, tensor, outpath):
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for state in range(dense.shape[1]):
        line = ax.plot(times, dense[:, state], lw=2.0, label=f"dense S{state}")[0]
        ax.plot(times, tensor[:, state], "--", color=line.get_color(), label=f"TT S{state}")
    ax.set(xlabel="time / fs", ylabel="population", ylim=(-0.04, 1.04))
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=2)
    parser.add_argument("--n-theta", type=int, default=2)
    parser.add_argument("--r-min", type=float, default=1.48)
    parser.add_argument("--r-max", type=float, default=1.82)
    parser.add_argument("--theta-min-deg", type=float, default=56.0)
    parser.add_argument("--theta-max-deg", type=float, default=64.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--packet-width", type=float, default=80.0)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--overlap-method", choices=("cross", "dense"), default="cross")
    parser.add_argument("--overlap-rank", type=int, default=8)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument(
        "--gauge-sync", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h3plus_ttldr_vs_dense_ldr"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    solver = build_solver(args)
    source, scan_time = load_or_scan(solver, args)
    psi0 = initial_packet(solver, min(args.initial_state, args.nstates - 1), args.packet_width)
    dt = args.dt_fs / au2fs

    start = time.perf_counter()
    tt = solver.ttldr(
        overlap_method=args.overlap_method,
        overlap_rank=args.overlap_rank,
        operator_rank=args.operator_rank,
        gauge_sync=args.gauge_sync,
    )
    tt_build = time.perf_counter() - start
    state = tt.state(psi0, max_rank=args.state_rank)
    start = time.perf_counter()
    tt.run(
        state,
        dt=dt,
        steps=args.steps,
        interval=args.interval,
        max_bond=args.state_rank,
        integrator="tdvp2",
        progress=False,
    )
    tt_time = time.perf_counter() - start

    kinetic = solver.buildK(sparse=False)
    dense_h = solver._build_flat_kinetic_matrix(kinetic)
    dense_h += np.diag(solver.apes.reshape(-1))
    start = time.perf_counter()
    dense_states, dense_populations = dense_propagate(
        psi0, dense_h, dt, args.steps, args.interval, args.nstates
    )
    dense_time = time.perf_counter() - start
    dense_final = dense_states[-1] / np.linalg.norm(dense_states[-1])
    tt_final = tt.dense(tt.final_state).reshape(-1)
    fidelity = float(abs(np.vdot(dense_final, tt_final)) ** 2)
    population_error = float(np.max(np.abs(dense_populations - tt.populations)))

    times_fs = tt.times * au2fs
    plot_path = args.outdir / "h3plus_ttldr_vs_dense_ldr_populations.png"
    plot_populations(times_fs, dense_populations, tt.populations, plot_path)
    np.savez(
        args.outdir / "h3plus_ttldr_vs_dense_ldr_results.npz",
        times_fs=times_fs,
        dense_populations=dense_populations,
        tt_populations=tt.populations,
        fidelity=fidelity,
        population_error=population_error,
        operator_ranks=np.asarray(tt.operator_ranks),
    )
    print(f"[grid] nx={solver.nx}, dim={int(np.prod(tt.dims))}")
    print(f"[scan] source={source}, time={scan_time:.3f} s")
    print(f"[TT] build={tt_build:.3f} s, propagate={tt_time:.3f} s, ranks={tt.operator_ranks}")
    print(f"[gauge] {tt.gauge_info}")
    print(f"[TT-cross] {tt.overlap_info}")
    print(f"[dense] propagate={dense_time:.3f} s")
    print(f"[error] final fidelity={fidelity:.12f}, population max={population_error:.3e}")
    print(f"[plot] {plot_path}")


if __name__ == "__main__":
    main()
