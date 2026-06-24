#!/usr/bin/env python3
"""Calibrate Taylor MPO propagation against two-site TDVP for GDVR H2."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.mps import expect_mps
from pyqed.mps.tdmps import TDMPS
from pyqed.qchem.gdvr import AtomicChain, TDDMRG
from pyqed.qchem.rttdhf import gaussian_pulse


def _parse_orders(value):
    return tuple(int(item) for item in str(value).replace(" ", "").split(",") if item)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lz", type=float, default=4.0)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--bond", type=int, default=24)
    parser.add_argument("--td-bond", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--orders", default="2,4,6,8")
    parser.add_argument("--time-reversal-steps", type=int, default=0)
    parser.add_argument("--field", type=float, default=1.0e-3)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--center", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.35)
    parser.add_argument("--out", default="/private/tmp/gdvr_taylor_calibration.png")
    parser.add_argument("--data", default=None)
    return parser


def _run_case(td, psi0, mu0, mu_mpo, pulse, args, *, integrator, order):
    label = "TDVP2" if integrator == "tdvp2" else f"Taylor-{order}"
    print(f"Running {label}...")
    start = time.perf_counter()
    td.run(
        psi0=psi0.copy(),
        dt=args.dt,
        steps=args.steps,
        e_ops=[mu_mpo],
        field=pulse,
        order=order,
        integrator=integrator,
    )
    runtime = time.perf_counter() - start
    print(f"  {label} propagation finished in {runtime:.2f} s")

    reversal = None
    if args.time_reversal_steps > 0:
        print(f"  {label} time-reversal check ({args.time_reversal_steps} steps)...")
        reversal = td.time_reversal_error(
            psi0=psi0.copy(),
            dt=args.dt,
            steps=args.time_reversal_steps,
            field=pulse,
            order=order,
            integrator=integrator,
        )
        print(f"  {label} time-reversal error: {reversal['state_error']:.6e}")

    times = np.concatenate(([0.0], np.asarray(td.times, dtype=float)))
    mu = np.concatenate(([mu0], np.real(td.observables[:, 0])))
    energy_drift = np.asarray(td.energy_drift, dtype=complex)
    norm_error = np.abs(np.asarray(td.pre_normalization_norms, dtype=float) - 1.0)
    truncation = np.asarray(td.tdvp_truncation_errors, dtype=float)

    return {
        "integrator": integrator,
        "order": int(order),
        "runtime": float(runtime),
        "times": times,
        "delta_mu": mu - mu[0],
        "norm_error": norm_error,
        "energy_times": np.asarray(td.energy_times, dtype=float),
        "energy_drift": np.abs(np.real(energy_drift)),
        "truncation": truncation,
        "time_reversal": np.nan if reversal is None else reversal["state_error"],
        "final_state": td.final_state.copy(),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    orders = _parse_orders(args.orders)
    if not orders:
        raise ValueError("At least one Taylor order is required.")

    mol = AtomicChain(["H", "H"], coords=[(0.0, 0.0, -0.7), (0.0, 0.0, 0.7)])
    mol.build(Lz=args.lz, Nz=args.nz, M=args.m, verbose=False)
    mf = mol.RHF().run(conv=1.0e-8, max_iter=80, verbose=False)

    pulse = gaussian_pulse(
        amplitude=args.field,
        center=args.center,
        width=args.width,
        omega=args.omega,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(
        mf,
        D=args.bond,
        td_bond_dim=args.bond if args.td_bond is None else args.td_bond,
        init_guess="random",
    ).build()
    td.optimize_ground_state(
        nstates=1,
        nsweeps=args.sweeps,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
        davidson_tol=1.0e-8,
    )

    psi0 = td.export_initial_guess(dense=True)
    mu_mpo = td.get_interaction_mpo(axis=2)
    mu0 = float(np.real(expect_mps(psi0.factors, mu_mpo.factors)))

    reference = _run_case(td, psi0, mu0, mu_mpo, pulse, args, integrator="tdvp2", order=max(orders))
    results = [reference]
    for order in orders:
        results.append(_run_case(td, psi0, mu0, mu_mpo, pulse, args, integrator="taylor", order=order))

    ref_mu = reference["delta_mu"]
    ref_state = reference["final_state"]
    rows = []
    for result in results:
        delta = result["delta_mu"] - ref_mu
        if result is reference:
            state_error = 0.0
        else:
            diagnostic = TDMPS.overlap_diagnostic(
                TDMPS.state_overlap(ref_state, result["final_state"]),
                ref_state.norm(),
                result["final_state"].norm(),
            )
            state_error = diagnostic["state_error"]
        rows.append({
            "label": "TDVP2" if result is reference else f"Taylor-{result['order']}",
            "runtime": result["runtime"],
            "max_mu_error": float(np.max(np.abs(delta))),
            "rms_mu_error": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
            "max_norm_error": float(np.nanmax(result["norm_error"])),
            "max_energy_drift": float(np.nanmax(result["energy_drift"])),
            "max_truncation": float(np.nanmax(result["truncation"])),
            "time_reversal": float(result["time_reversal"]),
            "state_error": float(state_error),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else out.with_suffix(".npz")

    fig, axes = plt.subplots(4, 1, figsize=(7.4, 9.2), sharex=False)
    field_times = reference["times"]
    axes[0].plot(field_times, [pulse(t)[2] for t in field_times], color="tab:red", lw=1.8)
    axes[0].set_ylabel(r"$E_z(t)$")

    axes[1].plot(reference["times"], reference["delta_mu"], color="black", lw=2.0, label="TDVP2")
    for result in results[1:]:
        axes[1].plot(result["times"], result["delta_mu"], lw=1.5, ls="--", label=f"Taylor-{result['order']}")
    axes[1].set_ylabel(r"$\Delta\mu_z(t)$")
    axes[1].legend(frameon=False, ncol=2)

    for result in results[1:]:
        err = np.maximum(np.abs(result["delta_mu"] - ref_mu), 1.0e-18)
        axes[2].plot(result["times"], err, lw=1.5, label=f"Taylor-{result['order']}")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$|\Delta\mu-\Delta\mu_\mathrm{TDVP2}|$")
    axes[2].legend(frameon=False, ncol=2)

    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.2
    axes[3].bar(x - 1.5 * width, [max(row["max_norm_error"], 1.0e-18) for row in rows], width, label=r"$|N_\mathrm{pre}-1|$")
    axes[3].bar(x - 0.5 * width, [max(row["max_energy_drift"], 1.0e-18) for row in rows], width, label=r"$|\Delta\langle H_0\rangle|$")
    axes[3].bar(x + 0.5 * width, [max(row["time_reversal"], 1.0e-18) for row in rows], width, label="TR error")
    axes[3].bar(x + 1.5 * width, [max(row["state_error"], 1.0e-18) for row in rows], width, label="final-state error")
    axes[3].set_yscale("log")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=25, ha="right")
    axes[3].set_ylabel("diagnostics")
    axes[3].legend(frameon=False, ncol=2)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"H2 GDVR Taylor calibration, Nz={args.nz}, M={args.m}, D={td.td_bond_dim}\n"
        f"E_RHF={mf.e_tot:.8f} Ha, E_DMRG={td.e_tot:.8f} Ha, steps={args.steps}, dt={args.dt}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)

    np.savez(
        data_path,
        rows=np.array(rows, dtype=object),
        orders=np.asarray(orders, dtype=int),
        tdvp2_times=reference["times"],
        tdvp2_delta_mu=reference["delta_mu"],
        **{
            f"taylor_{result['order']}_delta_mu": result["delta_mu"]
            for result in results[1:]
        },
    )

    print(f"RHF energy:      {mf.e_tot:.12f} Ha")
    print(f"DMRG energy:     {td.e_tot:.12f} Ha")
    print(f"GDVR MPO terms:  {td._active_integral_build_info['symbolic_terms']}")
    print(f"max MPO bond:    {td._active_integral_build_info['mpo_max_bond']}")
    print("case        runtime(s)   max|dmu-ref|   rms|dmu-ref|   max|Npre-1|   max|dE|      TR error     state err")
    for row in rows:
        print(
            f"{row['label']:<10s} {row['runtime']:10.2f} "
            f"{row['max_mu_error']:13.6e} {row['rms_mu_error']:13.6e} "
            f"{row['max_norm_error']:12.6e} {row['max_energy_drift']:11.6e} "
            f"{row['time_reversal']:11.6e} {row['state_error']:11.6e}"
        )
    print(f"Saved figure:    {out}")
    print(f"Saved data:      {data_path}")


if __name__ == "__main__":
    main()
