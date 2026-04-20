#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-system exactness benchmark for q-coordinate MPS code.

Benchmark layer #1 (gold standard):
1) Ground-state energy from DMRG vs exact diagonalization
2) Real-time dynamics (TDMPS) vs exact dense propagation

The Hamiltonian is built in q-space with the same local basis and operator
rules used by the current q-coordinate first-quantized scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from matplotlib.ticker import ScalarFormatter

from pyqed.mps.dmrg import DMRG
from pyqed.mps.mps import MPS, MPO, _mpo_to_dense_operator
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.first_quantization import Chain


def build_q_hamiltonian_mpo(n_electrons=3, qmax=4, t=1.0, t2=0.0, v=0.0):
    """
    Build q-space MPO for t-V model with N MPS sites = N electrons.

    Physical constraints in this model:
    - q_n >= 1 for all n (including q1 in this implementation)
    """
    d = qmax + 1
    q_op = np.diag(np.arange(d, dtype=float))
    model = Chain(
        nsites=n_electrons,
        local_dim=d,
        local_operator_mats={"Q": q_op},
    )

    min_q = [None] + [1] * n_electrons  # 1-indexed

    hop_ranges = [(1, float(t)), (2, float(t2))]

    # Neighbor-coupled kinetic terms for each hopping range r
    for r, tr in hop_ranges:
        if abs(tr) == 0.0:
            continue
        for n in range(1, n_electrons):
            qn_min = min_q[n]
            qnp1_min = min_q[n + 1]

            for qn in range(qn_min, d):
                for qnp1 in range(qnp1_min, d):
                    # Right move for particle n by r sites
                    qn_out = qn + r
                    qnp1_out = qnp1 - r
                    if qn_out < d and qnp1_out >= qnp1_min:
                        model.add_term(
                            -tr,
                            [
                                (f"E{qn_out}_{qn}", n - 1),
                                (f"E{qnp1_out}_{qnp1}", n),
                            ],
                        )

                    # Left move for particle n by r sites
                    qn_out = qn - r
                    qnp1_out = qnp1 + r
                    if qn_out >= qn_min and qnp1_out < d:
                        model.add_term(
                            -tr,
                            [
                                (f"E{qn_out}_{qn}", n - 1),
                                (f"E{qnp1_out}_{qnp1}", n),
                            ],
                        )

    # Last particle free boundary moves: q_N -> q_N +/- r
    qn_min = min_q[n_electrons]
    for r, tr in hop_ranges:
        if abs(tr) == 0.0:
            continue
        for qn in range(qn_min, d):
            qn_out = qn + r
            if qn_out < d:
                model.add_term(-tr, [(f"E{qn_out}_{qn}", n_electrons - 1)])
            qn_out = qn - r
            if qn_out >= qn_min:
                model.add_term(-tr, [(f"E{qn_out}_{qn}", n_electrons - 1)])

    # Interaction V * sum_{n=2}^N P^n_{1}
    if abs(v) > 0.0:
        for n in range(2, n_electrons + 1):
            model.add_term(v, [("P1", n - 1)])

    return model.build_mpo(algo="qr"), d, len(model.terms)


def build_product_q_state(n_electrons, d, q0=1):
    """Product initial state q_n = q0 as MPS."""
    factors = []
    for _ in range(n_electrons):
        a = np.zeros((1, d, 1), dtype=complex)
        a[0, q0, 0] = 1.0
        factors.append(a)
    return MPS(factors, labels=["lv", "p", "rv"])


def mps_to_dense_state(psi: MPS) -> np.ndarray:
    """Contract an MPS to a dense state vector."""
    t = psi.factors[0]
    for a in psi.factors[1:]:
        t = np.tensordot(t, a, axes=([-1], [0]))
    vec = np.squeeze(t, axis=(0, -1)).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec = vec / norm
    return vec


def local_q_ops_dense(n_sites, d):
    """Dense local q operators for each site."""
    q = np.diag(np.arange(d, dtype=float))
    eye = np.eye(d, dtype=float)
    ops = []
    for site in range(n_sites):
        op = np.array([[1.0]], dtype=float)
        for i in range(n_sites):
            op = np.kron(op, q if i == site else eye)
        ops.append(op)
    return ops


@dataclass
class GroundStateBench:
    e_exact: float
    e_dmrg: float
    abs_err: float
    overlap: float
    rayleigh_dmrg: float
    rayleigh_abs_err: float


@dataclass
class DynamicsBench:
    steps: int
    dt: float
    order: int
    max_abs_q_err: float
    max_infidelity: float
    max_norm_drift_mps: float


@dataclass
class DSweepPoint:
    D: int
    e_dmrg: float
    abs_err: float
    success: bool
    message: str


def benchmark_ground_state(H_mpo, n_sites, d, chi=32, nsweeps=8):
    h_dense = _mpo_to_dense_operator(H_mpo)
    evals, evecs = np.linalg.eigh(h_dense)
    e_exact = float(np.real(evals[0]))
    psi_exact = evecs[:, 0]
    psi_exact = psi_exact / np.linalg.norm(psi_exact)

    # Real, slightly perturbed product-state init avoids zero-vector issues
    # in local ARPACK solves for some tightly constrained product states.
    rng = np.random.default_rng(123)
    psi0 = []
    for _ in range(n_sites):
        a = np.zeros((1, d, 1), dtype=float)
        a[0, 1, 0] = 1.0
        a += 1e-8 * rng.standard_normal(a.shape)
        psi0.append(a)
    dmrg = DMRG(H_mpo.factors, D=chi, nsweeps=nsweeps, opt="2site")
    dmrg.init_guess = psi0
    dmrg.run()

    e_dmrg = float(np.real(dmrg.e_tot))
    psi_dmrg = mps_to_dense_state(dmrg.ground_state)
    overlap = float(np.abs(np.vdot(psi_exact, psi_dmrg)))
    rayleigh = float(np.real(np.vdot(psi_dmrg, h_dense @ psi_dmrg)))

    return GroundStateBench(
        e_exact=e_exact,
        e_dmrg=e_dmrg,
        abs_err=abs(e_dmrg - e_exact),
        overlap=overlap,
        rayleigh_dmrg=rayleigh,
        rayleigh_abs_err=abs(rayleigh - e_exact),
    )


def benchmark_dynamics(H_mpo, n_sites, d, chi=32, dt=0.05, steps=6, order=8):
    h_dense = _mpo_to_dense_operator(H_mpo)
    u = expm(-1j * dt * h_dense)

    psi_mps = build_product_q_state(n_sites, d, q0=1)
    psi_exact = mps_to_dense_state(psi_mps)

    td = TDMPS(H_mpo, D=chi)
    td.build_propagator(dt=dt, order=order, scale=0)

    q_ops = local_q_ops_dense(n_sites, d)
    q_diag = np.diag(np.arange(d, dtype=float))

    max_abs_q_err = 0.0
    max_infidelity = 0.0
    max_norm_drift = 0.0

    for _ in range(steps):
        psi_mps = td.step(psi_mps)
        psi_exact = u @ psi_exact
        psi_exact = psi_exact / np.linalg.norm(psi_exact)

        psi_mps_vec = mps_to_dense_state(psi_mps)

        q_exact = np.array(
            [np.real(np.vdot(psi_exact, op @ psi_exact)) for op in q_ops], dtype=float
        )
        q_mps = np.real_if_close(
            psi_mps.copy().right_canonicalize().site_expectation_value(q_diag)
        ).astype(float)

        q_err = float(np.max(np.abs(q_exact - q_mps)))
        max_abs_q_err = max(max_abs_q_err, q_err)

        overlap = np.abs(np.vdot(psi_exact, psi_mps_vec))
        infidelity = float(max(0.0, 1.0 - overlap * overlap))
        max_infidelity = max(max_infidelity, infidelity)

        mps_norm = float(np.linalg.norm(psi_mps_vec))
        max_norm_drift = max(max_norm_drift, abs(mps_norm - 1.0))

    return DynamicsBench(
        steps=steps,
        dt=dt,
        order=order,
        max_abs_q_err=max_abs_q_err,
        max_infidelity=max_infidelity,
        max_norm_drift_mps=max_norm_drift,
    )


def parse_D_list(text):
    vals = [s.strip() for s in str(text).split(",") if s.strip()]
    if not vals:
        raise ValueError("Empty D list.")
    out = []
    for v in vals:
        iv = int(v)
        if iv <= 0:
            raise ValueError(f"Invalid D value: {iv}")
        out.append(iv)
    return out


def sweep_ground_state_vs_D(H_mpo, n_sites, d, nsweeps, D_values):
    h_dense = _mpo_to_dense_operator(H_mpo)
    e_exact = float(np.real(np.linalg.eigvalsh(h_dense)[0]))

    rows = []
    for D in D_values:
        try:
            gs = benchmark_ground_state(H_mpo, n_sites=n_sites, d=d, chi=D, nsweeps=nsweeps)
            rows.append(
                DSweepPoint(
                    D=int(D),
                    e_dmrg=float(gs.e_dmrg),
                    abs_err=float(abs(gs.e_dmrg - e_exact)),
                    success=True,
                    message="ok",
                )
            )
        except Exception as exc:
            rows.append(
                DSweepPoint(
                    D=int(D),
                    e_dmrg=np.nan,
                    abs_err=np.nan,
                    success=False,
                    message=str(exc),
                )
            )
    return e_exact, rows


def save_d_sweep(prefix, e_exact, rows):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    dvals = np.array([r.D for r in rows], dtype=int)
    ed = np.array([r.e_dmrg for r in rows], dtype=float)
    err = np.array([r.abs_err for r in rows], dtype=float)
    ok = np.array([r.success for r in rows], dtype=bool)

    data_path = prefix.with_suffix(".npz")
    np.savez(data_path, D=dvals, e_dmrg=ed, abs_err=err, success=ok, e_exact=e_exact)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=150)

    if np.any(ok):
        ax1.plot(dvals[ok], ed[ok], "o-", lw=1.5, ms=4.0, label="DMRG")
        eplot = ed[ok]
        span = max(float(np.ptp(eplot)), 1.0e-10)
        mid = float(np.mean(eplot))
        ax1.set_ylim(mid - 0.6 * span, mid + 0.6 * span)
    ax1.axhline(e_exact, color="k", ls="--", lw=1.0, label="Exact")
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.set_xlabel("Bond dimension D")
    ax1.set_ylabel("Ground-state energy")
    ax1.set_title("Energy vs D")
    ax1.legend(loc="best")

    eps = 1e-18
    if np.any(ok):
        ax2.semilogy(dvals[ok], np.maximum(err[ok], eps), "o-", lw=1.5, ms=4.0)
    ax2.set_xlabel("Bond dimension D")
    ax2.set_ylabel(r"$|E_D - E_{\mathrm{exact}}|$")
    ax2.set_title("Absolute energy error")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig_path = prefix.with_name(prefix.name + "_energy_vs_D.png")
    fig.savefig(fig_path)
    plt.close(fig)

    txt_path = prefix.with_name(prefix.name + "_energy_vs_D.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("# D e_dmrg abs_err success message\n")
        for r in rows:
            fh.write(f"{r.D:4d} {r.e_dmrg:.16e} {r.abs_err:.6e} {int(r.success)} {r.message}\n")

    return data_path, fig_path, txt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark q-MPS against exact dense results.")
    parser.add_argument("--N", type=int, default=3, help="number of q-sites (electrons)")
    parser.add_argument("--qmax", type=int, default=4, help="q cutoff, local dim = qmax+1")
    parser.add_argument("--t", type=float, default=1.0, help="hopping amplitude")
    parser.add_argument("--t2", type=float, default=0.0, help="next-nearest hopping amplitude")
    parser.add_argument("--V", type=float, default=0.0, help="interaction strength")
    parser.add_argument("--chi", type=int, default=64, help="MPS bond dimension")
    parser.add_argument("--nsweeps", type=int, default=8, help="DMRG sweeps")
    parser.add_argument("--dt", type=float, default=0.005, help="time step for dynamics benchmark")
    parser.add_argument("--steps", type=int, default=6, help="number of propagation steps")
    parser.add_argument("--order", type=int, default=8, help="Taylor order for MPS propagator")
    parser.add_argument(
        "--energy-tol", type=float, default=1e-8, help="ground-state energy tolerance"
    )
    parser.add_argument(
        "--q-tol", type=float, default=1e-3, help="max absolute <q_n(t)> tolerance"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if tolerances are not met",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON output",
    )
    parser.add_argument(
        "--sweep-D",
        type=str,
        default=None,
        help="comma-separated D list for energy sweep plot, e.g. '4,8,16,32,64'",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="examples/qchem/qmps_benchmark",
        help="output prefix for sweep data/figures",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    args = parse_args()
    H_mpo, d, n_terms = build_q_hamiltonian_mpo(
        n_electrons=args.N, qmax=args.qmax, t=args.t, t2=args.t2, v=args.V
    )

    if args.sweep_D is not None:
        D_values = parse_D_list(args.sweep_D)
        e_exact, rows = sweep_ground_state_vs_D(
            H_mpo, n_sites=args.N, d=d, nsweeps=args.nsweeps, D_values=D_values
        )
        data_path, fig_path, txt_path = save_d_sweep(args.plot_prefix, e_exact, rows)
        print("=== Ground-State Energy vs D Sweep ===")
        print(
            f"N={args.N}, qmax={args.qmax} (d={d}), t={args.t}, t2={args.t2}, V={args.V}, "
            f"nsweeps={args.nsweeps}"
        )
        print(f"E_exact = {e_exact:.12f}")
        for r in rows:
            status = "ok" if r.success else "fail"
            print(f"D={r.D:4d}  E={r.e_dmrg:.12f}  |dE|={r.abs_err:.3e}  [{status}]")
        print(f"Saved data: {data_path}")
        print(f"Saved text: {txt_path}")
        print(f"Saved fig : {fig_path}")
        return

    gs = benchmark_ground_state(
        H_mpo, n_sites=args.N, d=d, chi=args.chi, nsweeps=args.nsweeps
    )
    dyn = benchmark_dynamics(
        H_mpo,
        n_sites=args.N,
        d=d,
        chi=args.chi,
        dt=args.dt,
        steps=args.steps,
        order=args.order,
    )

    result = {
        "settings": {
            "N": args.N,
            "qmax": args.qmax,
            "d": d,
            "t": args.t,
            "t2": args.t2,
            "V": args.V,
            "chi": args.chi,
            "nsweeps": args.nsweeps,
            "dt": args.dt,
            "steps": args.steps,
            "order": args.order,
            "n_terms": n_terms,
        },
        "ground_state": asdict(gs),
        "dynamics": asdict(dyn),
        "thresholds": {"energy_tol": args.energy_tol, "q_tol": args.q_tol},
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=== Small-System Exactness Benchmark ===")
        print(
            f"N={args.N}, qmax={args.qmax} (d={d}), t={args.t}, t2={args.t2}, V={args.V}, "
            f"chi={args.chi}, terms={n_terms}"
        )
        print("--- Ground state ---")
        print(f"E_exact               = {gs.e_exact:.12f}")
        print(f"E_dmrg                = {gs.e_dmrg:.12f}")
        print(f"|E_dmrg - E_exact|    = {gs.abs_err:.3e}")
        print(f"overlap(|psi0>)       = {gs.overlap:.12f}")
        print(f"Rayleigh(psi_dmrg)    = {gs.rayleigh_dmrg:.12f}")
        print(f"|Rayleigh - E_exact|  = {gs.rayleigh_abs_err:.3e}")
        print("--- Dynamics ---")
        print(f"dt={dyn.dt}, steps={dyn.steps}, order={dyn.order}")
        print(f"max_t,n |<q_n>_mps - <q_n>_exact| = {dyn.max_abs_q_err:.3e}")
        print(f"max_t infidelity                  = {dyn.max_infidelity:.3e}")
        print(f"max_t norm drift (MPS)            = {dyn.max_norm_drift_mps:.3e}")

    if args.check:
        ok = (gs.abs_err <= args.energy_tol) and (dyn.max_abs_q_err <= args.q_tol)
        if not ok:
            raise SystemExit(
                "Benchmark failed: "
                f"|dE|={gs.abs_err:.3e} (tol={args.energy_tol:.3e}), "
                f"|dq|max={dyn.max_abs_q_err:.3e} (tol={args.q_tol:.3e})"
            )


if __name__ == "__main__":
    main()
