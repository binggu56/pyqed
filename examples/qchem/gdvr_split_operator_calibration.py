#!/usr/bin/env python3
"""Prototype GDVR split-operator propagation against two-site TDVP."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.mps import expect_mps
from pyqed.mps.tdmps import TDMPS
from pyqed.qchem.gdvr import AtomicChain
from pyqed.qchem.gdvr.tddmrg import (
    GDVRSpatialDensityPhase,
    GDVRSpatialFactorizedDensityPhase,
    GDVRSpatialGroupedPairDensityPhase,
    GDVRSpatialHybridDensityPhase,
    GDVRSpatialOneBodyRotation,
    GDVRSpatialPronyDensityPhase,
    GDVRSpatialSVDDensityPhase,
    GDVRSpatialTaylorDensityPhase,
)
from pyqed.qchem.rttdhf import gaussian_pulse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lz", type=float, default=4.0)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--bond", type=int, default=24)
    parser.add_argument("--td-bond", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--density-method",
        choices=("pair", "grouped-pair", "factorized", "taylor", "prony", "hybrid", "svd"),
        default="pair",
    )
    parser.add_argument(
        "--pair-compress-mode",
        choices=("pair", "color", "distance", "end"),
        default="color",
    )
    parser.add_argument("--pair-direct-adjacent", action="store_true")
    parser.add_argument(
        "--pair-distance-order",
        choices=("ascending", "descending"),
        default="ascending",
    )
    parser.add_argument("--prony-rank", type=int, default=8)
    parser.add_argument("--prony-residual-rank", type=int, default=0)
    parser.add_argument("--hybrid-prony-rank", type=int, default=8)
    parser.add_argument("--hybrid-residual-rank", type=int, default=8)
    parser.add_argument("--factorized-rank", type=int, default=None)
    parser.add_argument("--factorized-tt-rank", type=int, default=None)
    parser.add_argument("--factorized-max-sites", type=int, default=12)
    parser.add_argument("--taylor-order", type=int, default=3)
    parser.add_argument("--taylor-rank", type=int, default=None)
    parser.add_argument("--svd-rank", type=int, default=8)
    parser.add_argument("--reference-integrator", choices=("tdvp", "tdvp2"), default="tdvp")
    parser.add_argument("--density-integrator", choices=("tdvp", "tdvp2"), default="tdvp2")
    parser.add_argument("--tdvp-krylov-dim", type=int, default=12)
    parser.add_argument("--tdvp-krylov-tol", type=float, default=1.0e-13)
    parser.add_argument("--tdvp-diagonal-fast-path", action="store_true")
    parser.add_argument("--field", type=float, default=1.0e-3)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--center", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.35)
    parser.add_argument("--out", default="/private/tmp/gdvr_split_operator_calibration.png")
    return parser


def _compress_normalize(psi, bond_dim):
    psi = psi.compress(bond_dim)
    norm2 = float(np.real(psi.norm()))
    psi.normalize()
    return psi, float(np.sqrt(max(norm2, 0.0)))


def _run_split(td, psi0, mu_mpo, mu0, pulse, args):
    mol = td.gdvr_mf.mol
    hcore = np.asarray(mol.hcore, dtype=complex)
    bond_dim = int(td.td_bond_dim)
    dt = float(args.dt)

    start = time.perf_counter()
    if args.density_method == "prony":
        density_half = GDVRSpatialPronyDensityPhase(
            mol,
            0.5 * dt,
            rank=args.prony_rank,
            residual_rank=args.prony_residual_rank,
        )
    elif args.density_method == "hybrid":
        density_half = GDVRSpatialHybridDensityPhase(
            mol,
            0.5 * dt,
            prony_rank=args.hybrid_prony_rank,
            residual_rank=args.hybrid_residual_rank,
        )
    elif args.density_method == "svd":
        density_half = GDVRSpatialSVDDensityPhase(
            mol,
            0.5 * dt,
            rank=args.svd_rank,
        )
    elif args.density_method == "factorized":
        density_half = GDVRSpatialFactorizedDensityPhase(
            mol,
            0.5 * dt,
            rank=args.factorized_rank,
            tt_rank=args.factorized_tt_rank,
            max_sites=args.factorized_max_sites,
        )
    elif args.density_method == "taylor":
        density_half = GDVRSpatialTaylorDensityPhase(
            mol,
            0.5 * dt,
            order=args.taylor_order,
            rank=args.taylor_rank,
        )
    elif args.density_method == "grouped-pair":
        density_half = GDVRSpatialGroupedPairDensityPhase(
            mol,
            0.5 * dt,
            compress_mode=args.pair_compress_mode,
            direct_adjacent=args.pair_direct_adjacent,
            distance_order=args.pair_distance_order,
        )
    else:
        density_half = GDVRSpatialDensityPhase(mol, 0.5 * dt)
    one_body = GDVRSpatialOneBodyRotation(hcore, dt)
    setup_time = time.perf_counter() - start

    psi = psi0.copy()
    times = [0.0]
    delta_mu = [0.0]
    norm_errors = []
    static_energies = [TDMPS(td._get_td_hamiltonian(), D=bond_dim).static_energy(psi)]

    start = time.perf_counter()
    for step in range(args.steps):
        time_mid = (step + 0.5) * dt
        field_z = pulse(time_mid)[2]

        step_norms = []
        density_kwargs = {"field_z": field_z, "max_bond": bond_dim}
        if args.density_method in {"prony", "hybrid", "svd"}:
            density_kwargs["integrator"] = args.density_integrator
            density_kwargs["krylov_dim"] = args.tdvp_krylov_dim
            density_kwargs["krylov_tol"] = args.tdvp_krylov_tol
            density_kwargs["diagonal_fast_path"] = args.tdvp_diagonal_fast_path
        psi = density_half.apply(psi, **density_kwargs)
        psi, norm = _compress_normalize(psi, bond_dim)
        step_norms.append(norm)

        psi = one_body.apply(
            psi,
            max_bond=bond_dim,
        )
        psi, norm = _compress_normalize(psi, bond_dim)
        step_norms.append(norm)

        psi = density_half.apply(psi, **density_kwargs)
        psi, norm = _compress_normalize(psi, bond_dim)
        step_norms.append(norm)

        times.append((step + 1) * dt)
        delta_mu.append(float(np.real(expect_mps(psi.factors, mu_mpo.factors))) - mu0)
        norm_errors.append(abs(step_norms[-1] - 1.0))
        static_energies.append(TDMPS(td._get_td_hamiltonian(), D=bond_dim).static_energy(psi))

    runtime = time.perf_counter() - start
    return {
        "setup_time": float(setup_time),
        "runtime": float(runtime),
        "times": np.asarray(times, dtype=float),
        "delta_mu": np.asarray(delta_mu, dtype=float),
        "norm_error": np.asarray(norm_errors, dtype=float),
        "energy_drift": np.asarray(static_energies, dtype=complex) - static_energies[0],
        "final_state": psi.copy(),
        "density_fit": getattr(density_half, "fit_info", None),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)

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

    td = mf.TDDMRG(
        D=args.bond,
        td_bond_dim=args.bond if args.td_bond is None else args.td_bond,
    ).build()
    td.optimize_ground_state(
        nstates=1,
        nsweeps=args.sweeps,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
        davidson_tol=1.0e-8,
    )

    psi0 = td.export_ground_state(dense=True)
    mu_mpo = td.get_interaction_mpo(axis=2)
    mu0 = float(np.real(expect_mps(psi0.factors, mu_mpo.factors)))

    ref_start = time.perf_counter()
    td.run(
        psi0=psi0.copy(),
        dt=args.dt,
        steps=args.steps,
        e_ops=[mu_mpo],
        field=pulse,
        order=4,
        integrator=args.reference_integrator,
        krylov_dim=args.tdvp_krylov_dim,
        krylov_tol=args.tdvp_krylov_tol,
        diagonal_fast_path=args.tdvp_diagonal_fast_path,
    )
    ref_runtime = time.perf_counter() - ref_start
    ref_times = np.concatenate(([0.0], np.asarray(td.times, dtype=float)))
    ref_delta_mu = np.concatenate(([0.0], np.real(td.observables[:, 0]) - mu0))

    split = _run_split(td, psi0, mu_mpo, mu0, pulse, args)
    state_diag = TDMPS.overlap_diagnostic(
        TDMPS.state_overlap(td.final_state, split["final_state"]),
        td.final_state.norm(),
        split["final_state"].norm(),
    )
    mu_error = split["delta_mu"] - ref_delta_mu
    split_energy_drift = np.abs(np.real(split["energy_drift"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 8.8), sharex=True)
    axes[0].plot(ref_times, [pulse(t)[2] for t in ref_times], color="tab:red", lw=1.8)
    axes[0].set_ylabel(r"$E_z(t)$")

    ref_label = args.reference_integrator.upper()
    axes[1].plot(ref_times, ref_delta_mu, color="black", lw=2.0, label=ref_label)
    axes[1].plot(split["times"], split["delta_mu"], color="tab:blue", ls="--", lw=1.8, label="split")
    axes[1].set_ylabel(r"$\Delta\mu_z(t)$")
    axes[1].legend(frameon=False)

    axes[2].plot(split["times"], np.maximum(np.abs(mu_error), 1.0e-18), color="tab:purple", lw=1.8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(rf"$|\Delta\mu_\mathrm{{split}}-\Delta\mu_\mathrm{{{ref_label}}}|$")

    step_times = np.arange(1, args.steps + 1, dtype=float) * args.dt
    axes[3].plot(step_times, np.maximum(split["norm_error"], 1.0e-18), lw=1.6, label=r"$|N_\mathrm{pre}-1|$")
    axes[3].plot(split["times"], np.maximum(split_energy_drift, 1.0e-18), lw=1.6, label=r"$|\Delta\langle H_0\rangle|$")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("time (a.u.)")
    axes[3].set_ylabel("split diagnostics")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"H2 GDVR split-operator prototype, Nz={args.nz}, M={args.m}, D={td.td_bond_dim}\n"
        f"E_RHF={mf.e_tot:.8f} Ha, E_DMRG={td.e_tot:.8f} Ha, steps={args.steps}, dt={args.dt}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)

    print(f"RHF energy:       {mf.e_tot:.12f} Ha")
    print(f"DMRG energy:      {td.e_tot:.12f} Ha")
    print(f"GDVR MPO terms:   {td._active_integral_build_info['symbolic_terms']}")
    print(f"max MPO bond:     {td._active_integral_build_info['mpo_max_bond']}")
    print(f"{ref_label} runtime:    {ref_runtime:.2f} s")
    print(f"density method:   {args.density_method}")
    if split["density_fit"] is not None:
        fit = split["density_fit"]
        if args.density_method == "prony":
            print(f"Prony rank:       {args.prony_rank}")
            print(f"Residual rank:    {fit['residual_retained_rank']}")
            print(f"Prony rel error:  {fit['rel_error']:.6e}")
            if fit["full_kernel_rel_error"] is not None:
                print(f"Full kernel err:  {fit['full_kernel_rel_error']:.6e}")
            print(f"Toeplitz spread:  {fit['toeplitz_max_rel_spread']:.6e}")
        elif args.density_method == "hybrid":
            print(f"Prony rank:       {args.hybrid_prony_rank}")
            print(f"Residual rank:    {fit['residual_retained_rank']}")
            print(f"Prony rel error:  {fit['rel_error']:.6e}")
            print(f"Full kernel err:  {fit['full_kernel_rel_error']:.6e}")
            print(f"Toeplitz spread:  {fit['toeplitz_max_rel_spread']:.6e}")
        elif args.density_method == "svd":
            print(f"SVD rank:         {fit['retained_rank']}")
            print(f"Full kernel err:  {fit['full_kernel_rel_error']:.6e}")
        elif args.density_method == "factorized":
            print(f"Factor rank:      {fit['retained_rank']}")
            print(f"Phase MPO bond:   {fit['max_mpo_bond']}")
            print(f"Full kernel err:  {fit['full_kernel_rel_error']:.6e}")
        elif args.density_method == "taylor":
            print(f"Taylor order:     {args.taylor_order}")
            print(f"Density rank:     {fit['retained_rank']}")
            print(f"Full kernel err:  {fit['full_kernel_rel_error']:.6e}")
        elif args.density_method == "grouped-pair":
            print(f"Pair gates:       {fit['pair_gates']}")
            print(f"Color groups:     {fit['color_groups']}")
            print(f"Distance groups:  {fit['distance_groups']}")
            print(f"Compress mode:    {fit['compress_mode']}")
            print(f"Direct adjacent:  {fit['direct_adjacent']}")
            print(f"Distance order:   {fit['distance_order']}")
    print(f"split setup time: {split['setup_time']:.2f} s")
    print(f"split runtime:    {split['runtime']:.2f} s")
    print(f"max |dmu error|:  {np.max(np.abs(mu_error)):.6e}")
    print(f"rms |dmu error|:  {np.sqrt(np.mean(np.abs(mu_error) ** 2)):.6e}")
    print(f"max |Npre - 1|:   {np.nanmax(split['norm_error']):.6e}")
    print(f"max |d<H0>|:      {np.nanmax(split_energy_drift):.6e} Ha")
    print(f"state error:      {state_diag['state_error']:.6e}")
    print(f"Saved figure:     {out}")


if __name__ == "__main__":
    main()
