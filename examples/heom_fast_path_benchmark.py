#!/usr/bin/env python3
"""Benchmark native HEOM DOP853 fast RHS against the legacy native RHS."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import sympy as sp

from pyqed import pauli
import pyqed.heom.deom as heom_deom
from pyqed.heom import Bath, HEOM
from pyqed.heom.deom import decompose_spectrum_pade


def build_spin_boson_solver(args):
    _, sx, _, sz = pauli()
    omega, lam_sym, gam_sym = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam_sym * gam_sym * omega / (gam_sym**2 + omega**2)).subs(
        {lam_sym: args.reorganization, gam_sym: args.cutoff}
    )
    bath = Bath(
        [spectrum],
        omega,
        [args.beta],
        [args.npsd],
        [0] * (args.npsd + 1),
        [decompose_spectrum_pade],
    )
    hamiltonian = -0.5 * args.delta * sx - 0.5 * args.bias * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0
    solver = HEOM(
        system=hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=args.lmax,
        hierarchy_truncation="total",
    )
    return solver, rho0


def native_args(solver, ddos):
    expn, etal, etar, etaa = solver._native_bath_list
    return [
        ddos,
        solver._native_keys,
        solver._native_minus_index,
        solver._native_plus_index,
        expn,
        etal,
        etar,
        etaa,
        solver._native_mode,
        np.ascontiguousarray(solver.system, dtype=np.complex128),
        np.ascontiguousarray(solver.system_dipole, dtype=np.complex128),
        np.ascontiguousarray(solver.coupling, dtype=np.complex128),
        np.ascontiguousarray(solver.coupling_dipole, dtype=np.complex128),
        None,
        None,
    ]


def run_native_direct(heom_cpp, solver, rho0, args, *, use_edges):
    ddos = np.zeros((solver.nmax, solver.nsys, solver.nsys), dtype=np.complex128)
    ddos[0] = rho0
    call_args = native_args(solver, ddos)
    if args.fixed_output:
        nt = int(round(args.tmax / args.dt))
        t_eval = np.linspace(0.0, args.tmax, nt + 1)
        call_args.extend([t_eval, args.rtol, args.atol, args.threads])
        function = heom_cpp.dop853_by_index
    else:
        call_args.extend([0.0, args.tmax, args.rtol, args.atol, args.threads])
        function = heom_cpp.dop853_adaptive_by_index
    if use_edges:
        call_args.extend(solver._native_edge_tables)

    start = perf_counter()
    if args.fixed_output:
        rhos, nfev, n_steps, n_rejected = function(*call_args)
        times = t_eval
    else:
        times, rhos, nfev, n_steps, n_rejected = function(*call_args)
    elapsed = perf_counter() - start
    return elapsed, times, rhos, int(nfev), int(n_steps), int(n_rejected), ddos


def best_direct_run(heom_cpp, solver, rho0, args, *, use_edges):
    best = None
    for _ in range(args.trials):
        result = run_native_direct(heom_cpp, solver, rho0, args, use_edges=use_edges)
        if best is None or result[0] < best[0]:
            best = result
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npsd", type=int, default=11, help="Pade poles; nexp is npsd + 1")
    parser.add_argument("--lmax", type=int, default=6, help="total hierarchy depth")
    parser.add_argument("--tmax", type=float, default=0.05, help="final propagation time")
    parser.add_argument("--dt", type=float, default=0.005, help="fixed output spacing for --fixed-output")
    parser.add_argument("--threads", type=int, default=1, help="native HEOM threads; 0 means hardware count")
    parser.add_argument("--rtol", type=float, default=1.0e-7, help="DOP853 relative tolerance")
    parser.add_argument("--atol", type=float, default=1.0e-9, help="DOP853 absolute tolerance")
    parser.add_argument("--trials", type=int, default=3, help="direct old/fast timing trials")
    parser.add_argument("--beta", type=float, default=1.0, help="inverse temperature")
    parser.add_argument("--reorganization", type=float, default=0.2, help="Drude reorganization energy")
    parser.add_argument("--cutoff", type=float, default=1.0, help="Drude cutoff frequency")
    parser.add_argument("--delta", type=float, default=1.0, help="spin-boson tunneling")
    parser.add_argument("--bias", type=float, default=0.2, help="spin-boson bias")
    parser.add_argument("--fixed-output", action="store_true", help="save on a fixed output grid instead of accepted steps")
    parser.add_argument("--skip-legacy", action="store_true", help="only time the default fast path")
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.tmax <= 0.0:
        raise ValueError("--tmax must be positive")
    if args.fixed_output:
        nt = int(round(args.tmax / args.dt))
        if nt < 1 or not np.isclose(nt * args.dt, args.tmax):
            raise ValueError("--tmax must be a positive integer multiple of --dt")

    heom_cpp = heom_deom._get_heom_cpp()
    if heom_cpp is None or not hasattr(heom_cpp, "dop853_by_index"):
        raise SystemExit("native HEOM extension is not built; run `python setup.py build_ext --inplace`")
    if not args.fixed_output and not hasattr(heom_cpp, "dop853_adaptive_by_index"):
        raise SystemExit("native HEOM extension does not provide adaptive-step DOP853 output")

    default_solver, rho0 = build_spin_boson_solver(args)
    run_kwargs = dict(
        rho0=rho0.copy(),
        method="dop853",
        rtol=args.rtol,
        atol=args.atol,
        threads=args.threads,
    )
    if args.fixed_output:
        run_kwargs.update(dt=args.dt, nt=nt)
    else:
        run_kwargs.update(t_span=(0.0, args.tmax))

    start = perf_counter()
    t_eval, default_rhos = default_solver.run(**run_kwargs)
    default_elapsed = perf_counter() - start

    print(
        f"model: nsys={default_solver.nsys} nexp={default_solver.nexp} "
        f"lmax={default_solver.lmax} nado={default_solver.nmax}"
    )
    print(
        'HEOM.run(method="dop853"): '
        f"method={default_solver.method} fast_edges={default_solver._native_edge_tables is not None} "
        f"output={'fixed' if args.fixed_output else 'accepted'} "
        f"nout={len(t_eval)} threads={default_solver.threads} time={default_elapsed:.6f}s "
        f"nfev={default_solver.nfev} steps={default_solver.n_steps}"
    )

    if args.skip_legacy:
        return

    direct_solver, _ = build_spin_boson_solver(args)
    direct_solver.check_()
    direct_solver.init_()
    if direct_solver._native_edge_tables is None:
        raise RuntimeError("native edge tables were not built")

    legacy = best_direct_run(heom_cpp, direct_solver, rho0, args, use_edges=False)
    fast = best_direct_run(heom_cpp, direct_solver, rho0, args, use_edges=True)
    legacy_time, legacy_t, legacy_rhos, legacy_nfev, legacy_steps, legacy_rejected, _ = legacy
    fast_time, fast_t, fast_rhos, fast_nfev, fast_steps, fast_rejected, _ = fast
    speedup = legacy_time / fast_time if fast_time > 0.0 else float("inf")
    final_diff = np.max(np.abs(legacy_rhos[-1] - fast_rhos[-1]))
    default_diff = np.max(np.abs(default_rhos[-1] - fast_rhos[-1]))
    same_grid = legacy_t.shape == fast_t.shape and np.allclose(legacy_t, fast_t)
    max_diff = np.max(np.abs(legacy_rhos - fast_rhos)) if same_grid else final_diff

    print(
        "legacy native RHS: "
        f"time={legacy_time:.6f}s nout={len(legacy_t)} nfev={legacy_nfev} "
        f"steps={legacy_steps} rejected={legacy_rejected}"
    )
    print(
        "edge-table RHS: "
        f"time={fast_time:.6f}s nout={len(fast_t)} nfev={fast_nfev} "
        f"steps={fast_steps} rejected={fast_rejected}"
    )
    print(f"speedup: {speedup:.2f}x")
    print(f"max |edge - legacy|: {max_diff:.3e}")
    print(f"final |edge - legacy|: {final_diff:.3e}")
    print(f"final |default - edge|: {default_diff:.3e}")


if __name__ == "__main__":
    main()
