#!/usr/bin/env python3
"""Benchmark H3+ AM1/MECI TT-LDR action against dense LDR.

The grid is deliberately tiny by default so that a full pairwise LDR overlap
matrix and dense kinetic matrix can be used as the reference.
"""

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
import scipy.linalg
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs


def h3plus_body_frame(r: float = 1.65, theta: float = np.pi / 3.0):
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


@contextlib.contextmanager
def pushd(path: Path):
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
    overlap_path = args.outdir / "overlap_matrix.npz"
    if args.reuse_cache and apes_path.exists() and overlap_path.exists():
        solver.apes = np.load(apes_path)["data"]
        solver.overlap_matrix = np.load(overlap_path)["data"]
        solver.overlap_links = None
        return "cache", 0.0

    t0 = time.perf_counter()
    with pushd(args.outdir):
        solver.scan_pes(
            electronic_method="am1/meci",
            nstates=args.nstates,
            ncas=args.ncas,
            nelecas=2,
            overlap_method="full",
            n_workers=args.n_workers,
            worker_threads=1,
            scf_tol=args.scf_tol,
            max_cycle=args.max_cycle,
            damping=args.damping,
        )
    return "scan", time.perf_counter() - t0


def initial_packet(solver: Triatom, state: int, width: float):
    values = np.zeros((*solver.nx, solver.nstates), dtype=complex)
    center = np.array([axis[len(axis) // 2] for axis in solver.x])
    for idx in np.ndindex(*solver.nx):
        q = np.array([solver.x[axis][idx[axis]] for axis in range(solver.ndim)])
        values[idx + (state,)] = np.exp(-width * np.sum((q - center) ** 2))
    psi = solver.to_quadrature_normalized(values)
    norm = solver.norm(psi)
    if norm == 0:
        raise RuntimeError("Initial packet norm is zero.")
    return psi / norm


def populations(states, nstates):
    return np.array(
        [np.sum(np.abs(psi) ** 2, axis=tuple(range(psi.ndim - 1))) for psi in states]
    ).reshape(len(states), nstates)


def dense_split_propagate(psi0, K_dense, apes, dt, nt, nout):
    exp_t = scipy.linalg.expm(-1j * K_dense * dt)
    exp_v_half = np.exp(-0.5j * apes * dt)
    states = [psi0.copy()]
    times = [0.0]
    psi = psi0.copy()
    for step in range(1, nt + 1):
        psi = exp_v_half * psi
        psi = (exp_t @ psi.reshape(-1)).reshape(psi0.shape)
        psi = exp_v_half * psi
        if step % nout == 0:
            states.append(psi.copy())
            times.append(step * dt)
    return np.asarray(times), states


def ttldr_split_propagate(psi0, action, apes, dt, nt, nout, trace_k):
    exp_v_half = np.exp(-0.5j * apes * dt)
    k_op = action.linear("k")
    states = [psi0.copy()]
    times = [0.0]
    psi = psi0.copy()
    for step in range(1, nt + 1):
        psi = exp_v_half * psi
        psi = expm_multiply(-1j * dt * k_op, psi.reshape(-1), traceA=-1j * dt * trace_k)
        psi = psi.reshape(psi0.shape)
        psi = exp_v_half * psi
        if step % nout == 0:
            states.append(psi.copy())
            times.append(step * dt)
    return np.asarray(times), states


def plot_population_overlay(times_fs, dense_pops, tt_pops, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for state in range(dense_pops.shape[1]):
        color = colors[state % len(colors)]
        ax.plot(times_fs, dense_pops[:, state], color=color, lw=2.0, label=f"dense S{state}")
        ax.plot(
            times_fs,
            tt_pops[:, state],
            color=color,
            lw=1.7,
            ls="--",
            label=f"TT-LDR S{state}",
        )
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("H3+ AM1/MECI dense LDR vs TT-LDR")
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
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--packet-width", type=float, default=80.0)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--nt", type=int, default=10)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/private/tmp/h3plus_ttldr_vs_dense_ldr"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    solver = build_solver(args)
    source, scan_time = load_or_scan(solver, args)

    t0 = time.perf_counter()
    T = solver.buildK(sparse=False)
    T = 0.5 * (T + T.conj().T)
    K_dense = solver._build_flat_kinetic_matrix(T)
    H_dense = K_dense + np.diag(solver.apes.reshape(-1))
    action = solver.build_ttldr_action(T_total=T, prefer_links=False)
    build_time = time.perf_counter() - t0

    rng = np.random.default_rng(args.seed)
    psi_rand = rng.normal(size=action.shape) + 1j * rng.normal(size=action.shape)
    psi_rand = psi_rand / np.linalg.norm(psi_rand.reshape(-1))

    dense_k = K_dense @ psi_rand.reshape(-1)
    dense_h = H_dense @ psi_rand.reshape(-1)
    tt_k = action.k(psi_rand).reshape(-1)
    tt_h = action.h(psi_rand).reshape(-1)
    k_abs = float(np.max(np.abs(tt_k - dense_k)))
    h_abs = float(np.max(np.abs(tt_h - dense_h)))
    k_rel = float(np.linalg.norm(tt_k - dense_k) / np.linalg.norm(dense_k))
    h_rel = float(np.linalg.norm(tt_h - dense_h) / np.linalg.norm(dense_h))

    psi0 = initial_packet(
        solver,
        state=min(args.initial_state, args.nstates - 1),
        width=args.packet_width,
    )
    dt = args.dt_fs / au2fs
    trace_k = solver._kinetic_trace_from_nuclear_operator(T)

    t0 = time.perf_counter()
    dense_times, dense_states = dense_split_propagate(
        psi0, K_dense, solver.apes, dt, args.nt, args.nout
    )
    dense_prop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    tt_times, tt_states = ttldr_split_propagate(
        psi0, action, solver.apes, dt, args.nt, args.nout, trace_k
    )
    tt_prop_time = time.perf_counter() - t0

    dense_pops = populations(dense_states, args.nstates)
    tt_pops = populations(tt_states, args.nstates)
    state_diffs = np.asarray(
        [
            np.linalg.norm((tt - dense).reshape(-1))
            for tt, dense in zip(tt_states, dense_states)
        ]
    )
    pop_max_abs = float(np.max(np.abs(tt_pops - dense_pops)))
    final_l2 = float(state_diffs[-1])

    times_fs = dense_times * au2fs
    pop_png = args.outdir / "h3plus_ttldr_vs_dense_ldr_populations.png"
    plot_population_overlay(times_fs, dense_pops, tt_pops, pop_png)

    data_path = args.outdir / "h3plus_ttldr_vs_dense_ldr_results.npz"
    np.savez(
        data_path,
        times_au=dense_times,
        times_fs=times_fs,
        dense_pops=dense_pops,
        tt_pops=tt_pops,
        state_l2_diffs=state_diffs,
        apes=solver.apes,
        overlap_matrix=solver.overlap_matrix,
        nx=np.asarray(solver.nx),
        metrics=np.array(
            [
                k_abs,
                k_rel,
                h_abs,
                h_rel,
                pop_max_abs,
                final_l2,
                scan_time,
                build_time,
                dense_prop_time,
                tt_prop_time,
            ]
        ),
        metric_names=np.array(
            [
                "k_max_abs",
                "k_rel_l2",
                "h_max_abs",
                "h_rel_l2",
                "population_max_abs",
                "final_state_l2",
                "scan_time_s",
                "build_time_s",
                "dense_split_time_s",
                "tt_split_time_s",
            ]
        ),
    )

    print(f"[grid] nx={solver.nx}, ngrid={int(np.prod(solver.nx))}, dim={action.size}")
    print(f"[scan] source={source}, time={scan_time:.3f} s")
    print(f"[build] dense+TT action time={build_time:.3f} s")
    print(f"[action] K max abs={k_abs:.3e}, rel L2={k_rel:.3e}")
    print(f"[action] H max abs={h_abs:.3e}, rel L2={h_rel:.3e}")
    print(f"[prop] dense split time={dense_prop_time:.3f} s")
    print(f"[prop] TT-LDR split time={tt_prop_time:.3f} s")
    print(f"[prop] population max abs diff={pop_max_abs:.3e}")
    print(f"[prop] final state L2 diff={final_l2:.3e}")
    print(f"[plot] {pop_png}")
    print(f"[data] {data_path}")


if __name__ == "__main__":
    main()
