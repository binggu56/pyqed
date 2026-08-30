#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain-wall dynamics in first-quantized q-coordinates using MPS/MPO.

This script implements the free/interacting spinless-fermion t-V model in the
q-basis used in arXiv:2404.07105:

    q_1 = x_1,  q_n = x_n - x_{n-1} (n > 1)

with the ordered sector enforced by q_n >= 1. We keep the user-requested
mapping "number of MPS sites = number of electrons", i.e. one q_n per site.

Outputs:
1) NPZ file with times, <q_n>(t), and max entanglement entropy.
2) PNG figures for entropy growth and q-profile heatmap.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.operator_mpo.operator import Op
from pyqed.operator_mpo.basis import BasisSet
from pyqed.operator_mpo.model_mpo import ModelMPO as ModelMPO
from pyqed.operator_mpo.model import Model
from pyqed.mps.mps import MPS
from pyqed.tn import MPO
from pyqed.mps.tdmps import TDMPS


class QBasis(BasisSet):
    """
    Local basis |q> with q = 0..qmax.

    Supported operators:
    - I       : identity
    - Pk      : projector |k><k|
    - Ei_j    : transition |i><j|
    - Q       : diagonal q operator
    """

    def __init__(self, dof: int, d: int):
        super().__init__(dof, d, [0] * d)

    def op_mat(self, op):
        if not isinstance(op, Op):
            op = Op(op, None)
        symbol = op.symbol
        mat = np.zeros((self.nbas, self.nbas), dtype=float)

        if symbol == "I":
            mat = np.eye(self.nbas, dtype=float)
        elif symbol == "Q":
            mat = np.diag(np.arange(self.nbas, dtype=float))
        elif symbol.startswith("P"):
            idx = int(symbol[1:])
            if idx < 0 or idx >= self.nbas:
                raise ValueError(f"Projector index out of range: {symbol}")
            mat[idx, idx] = 1.0
        elif symbol.startswith("E"):
            body = symbol[1:]
            parts = body.split("_")
            if len(parts) != 2:
                raise ValueError(f"Unsupported transition operator: {symbol}")
            i = int(parts[0])
            j = int(parts[1])
            if i < 0 or i >= self.nbas or j < 0 or j >= self.nbas:
                raise ValueError(f"Transition index out of range: {symbol}")
            mat[i, j] = 1.0
        else:
            raise ValueError(f"Unsupported operator symbol: {symbol}")

        return mat * op.factor

    def copy(self, new_dof):
        return self.__class__(new_dof, self.nbas)


def build_q_hamiltonian_mpo(n_electrons=20, qmax=10, t=1.0, v=0.0):
    """
    Build q-space MPO for t-V model with N sites = N electrons.

    Physical constraints are enforced exactly by only generating hopping terms
    that map physical states (q_n >= 1) to physical states. Here we use:
    - q_1 >= 1 (left boundary x_1 >= 1)
    - q_n >= 1 for n >= 2 (Pauli ordering x_n > x_{n-1})
    """
    d = qmax + 1
    basis = [QBasis(site, d) for site in range(n_electrons)]
    terms = []

    # 1-indexed minimum physical q values for convenience.
    min_q = [None] + [1] * n_electrons

    # Kinetic term in q-space:
    # -t * sum_{n=1}^{N-1} (T_n^\dag T_{n+1} + h.c.)
    # plus last electron free hop term on q_N.
    for n in range(1, n_electrons):
        qn_min = min_q[n]
        qnp1_min = min_q[n + 1]

        for qn in range(qn_min, d):
            for qnp1 in range(qnp1_min, d):
                # particle n moves right: q_n -> q_n+1, q_{n+1} -> q_{n+1}-1
                qn_out = qn + 1
                qnp1_out = qnp1 - 1
                if qn_out < d and qnp1_out >= qnp1_min:
                    terms.append(
                        (-t)
                        * Op(f"E{qn_out}_{qn}", n - 1)
                        * Op(f"E{qnp1_out}_{qnp1}", n)
                    )

                # particle n moves left: q_n -> q_n-1, q_{n+1} -> q_{n+1}+1
                qn_out = qn - 1
                qnp1_out = qnp1 + 1
                if qn_out >= qn_min and qnp1_out < d:
                    terms.append(
                        (-t)
                        * Op(f"E{qn_out}_{qn}", n - 1)
                        * Op(f"E{qnp1_out}_{qnp1}", n)
                    )

    # Last particle hopping: x_N -> x_N +/- 1 => q_N -> q_N +/- 1.
    qn_min = min_q[n_electrons]
    for qn in range(qn_min, d):
        qn_out = qn + 1
        if qn_out < d:
            terms.append((-t) * Op(f"E{qn_out}_{qn}", n_electrons - 1))

        qn_out = qn - 1
        if qn_out >= qn_min:
            terms.append((-t) * Op(f"E{qn_out}_{qn}", n_electrons - 1))

    # Interaction term: V * sum_{n=2}^N P^n_{1}
    if abs(v) > 0.0:
        for n in range(2, n_electrons + 1):
            terms.append(v * Op("P1", n - 1))

    model = Model(basis=basis, ham_terms=terms)
    auto_mpo = ModelMPO(model, algo="qr")

    factors = []
    for w in auto_mpo.matrices:
        arr = np.asarray(w)
        if np.max(np.abs(np.imag(arr))) < 1e-12:
            arr = np.real(arr)
        factors.append(arr.transpose(0, 3, 1, 2))

    return MPO(factors), d, len(terms)


def build_domain_wall_initial_state(n_electrons, d, q0=1):
    """
    Domain-wall initial product state in q-basis: q_n = q0 for all n.

    For the half-filled left-domain-wall benchmark in the paper, use q0=1.
    """
    if q0 < 0 or q0 >= d:
        raise ValueError(f"q0={q0} out of basis range [0, {d - 1}]")

    factors = []
    for _ in range(n_electrons):
        a = np.zeros((1, d, 1), dtype=complex)
        a[0, q0, 0] = 1.0
        factors.append(a)
    return MPS(factors, labels=["lv", "p", "rv"])


def measure_q_and_entropy(psi, q_diag):
    """Return (<q_n> array, max bond entropy) from a copy of MPS state."""
    psi_c = psi.copy().right_canonicalize()
    q_expect = np.real_if_close(psi_c.site_expectation_value(q_diag)).astype(float)

    if psi_c.singular_values is None:
        smax = 0.0
    else:
        ent = psi_c.entanglement_entropy()
        smax = float(np.max(ent)) if ent.size else 0.0
    return q_expect, smax


def run_domain_wall(
    n_electrons=20,
    L=40,
    qmax=10,
    t=1.0,
    v=0.0,
    dt=0.02,
    tmax=2.0,
    chi=80,
    order=2,
    interval=10,
):
    H_mpo, d, n_terms = build_q_hamiltonian_mpo(
        n_electrons=n_electrons, qmax=qmax, t=t, v=v
    )
    psi = build_domain_wall_initial_state(n_electrons=n_electrons, d=d, q0=1)

    solver = TDMPS(H_mpo, D=chi)
    solver.build_propagator(dt=dt, order=order, scale=0)

    q_diag = np.diag(np.arange(d, dtype=float))
    steps = int(round(tmax / dt))
    checkpoints = list(range(interval, steps + 1, interval))
    if steps > 0 and (not checkpoints or checkpoints[-1] != steps):
        checkpoints.append(steps)

    times = []
    q_profiles = []
    smax_list = []

    for step in range(1, steps + 1):
        psi = solver.step(psi)
        if step in checkpoints:
            qexp, smax = measure_q_and_entropy(psi, q_diag)
            times.append(step * dt)
            q_profiles.append(qexp)
            smax_list.append(smax)

    times = np.asarray(times, dtype=float)
    q_profiles = np.asarray(q_profiles, dtype=float)
    smax = np.asarray(smax_list, dtype=float)

    # x_N = sum_{n=1}^N q_n in this coordinate convention.
    xN_mean = q_profiles.sum(axis=1) if q_profiles.size else np.array([], dtype=float)

    info = {
        "n_terms": n_terms,
        "n_electrons": n_electrons,
        "L": L,
        "qmax": qmax,
        "t": t,
        "v": v,
        "dt": dt,
        "tmax": tmax,
        "chi": chi,
        "order": order,
    }
    return times, q_profiles, smax, xN_mean, info


def save_and_plot(prefix, times, q_profiles, smax, xN_mean, info):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = prefix.with_suffix(".npz")
    np.savez(
        npz_path,
        times=times,
        q_profiles=q_profiles,
        smax=smax,
        xN_mean=xN_mean,
        **info,
    )

    # Entropy growth
    fig1, ax1 = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    ax1.plot(times, smax, "o-", lw=1.5, ms=3.0)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\max_n S_n$")
    ax1.set_title("Domain-wall quench: max entanglement entropy")
    fig1.tight_layout()
    entropy_path = prefix.with_name(prefix.name + "_entropy.png")
    fig1.savefig(entropy_path)
    plt.close(fig1)

    # q_n profile heatmap
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    im = ax2.imshow(
        q_profiles,
        origin="lower",
        aspect="auto",
        extent=[1, q_profiles.shape[1], times[0], times[-1]],
        cmap="viridis",
    )
    ax2.set_xlabel("particle index n")
    ax2.set_ylabel("time")
    ax2.set_title(r"$\langle q_n(t)\rangle$")
    cbar = fig2.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r"$\langle q_n\rangle$")
    fig2.tight_layout()
    qmap_path = prefix.with_name(prefix.name + "_q_profile.png")
    fig2.savefig(qmap_path)
    plt.close(fig2)

    return npz_path, entropy_path, qmap_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="q-coordinate domain-wall dynamics with TDMPS (sites = n_electrons)"
    )
    parser.add_argument("--L", type=int, default=16, help="box size used for diagnostics")
    parser.add_argument("--N", type=int, default=8, help="number of electrons (= number of MPS sites)")
    parser.add_argument("--qmax", type=int, default=8, help="local q cutoff (d = qmax+1)")
    parser.add_argument("--t", type=float, default=1.0, help="hopping amplitude")
    parser.add_argument("--V", type=float, default=0.0, help="nearest-neighbor interaction")
    parser.add_argument("--dt", type=float, default=0.02, help="time step")
    parser.add_argument("--tmax", type=float, default=1.0, help="maximum propagation time")
    parser.add_argument("--chi", type=int, default=48, help="MPS bond dimension cutoff")
    parser.add_argument("--order", type=int, default=2, help="Taylor order for exp(MPO)")
    parser.add_argument("--interval", type=int, default=10, help="measurement interval in steps")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="examples/qchem/domain_wall_q_tdmps_demo",
        help="output path prefix (without extension)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Keep run logs compact.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    times, q_profiles, smax, xN_mean, info = run_domain_wall(
        n_electrons=args.N,
        L=args.L,
        qmax=args.qmax,
        t=args.t,
        v=args.V,
        dt=args.dt,
        tmax=args.tmax,
        chi=args.chi,
        order=args.order,
        interval=args.interval,
    )
    npz_path, entropy_path, qmap_path = save_and_plot(
        args.output_prefix, times, q_profiles, smax, xN_mean, info
    )

    print("=== q-coordinate domain-wall dynamics ===")
    print(f"N={args.N}, L={args.L}, qmax={args.qmax}, t={args.t}, V={args.V}")
    print(f"dt={args.dt}, tmax={args.tmax}, chi={args.chi}, interval={args.interval}")
    print(f"MPO terms: {info['n_terms']}")
    if times.size > 0:
        print(f"Recorded {times.size} checkpoints up to t={times[-1]:.3f}")
        print(f"max_n,t <q_n> = {float(np.max(q_profiles)):.6f}")
        print(f"max_t max_n S_n = {float(np.max(smax)):.6f}")
        print(f"max_t <x_N> = max_t sum_n <q_n> = {float(np.max(xN_mean)):.6f}")
        if np.max(xN_mean) < args.L:
            print(f"Boundary check: max <x_N> < L ({args.L}), right boundary not yet reached.")
        else:
            print(f"Boundary check: max <x_N> >= L ({args.L}), boundary effects may appear.")
    print(f"Saved data: {npz_path}")
    print(f"Saved fig : {entropy_path}")
    print(f"Saved fig : {qmap_path}")


if __name__ == "__main__":
    main()
