#!/usr/bin/env python3
"""Compare direct GDVR-TDDMRG and GDVR RT-TDHF dipole dynamics for H2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem.gdvr import AtomicChain, RTTDHF as GDVRRTTDHF
from pyqed.qchem.rttdhf import gaussian_pulse
from pyqed.units import au2fs


AU_TIME_FS = au2fs


def optical_period(omega):
    return 2.0 * np.pi / float(omega)


def sin2_pulse(amplitude, omega, cycles, phase=0.0):
    duration = float(cycles) * optical_period(omega)

    def field(t):
        t = float(t)
        out = np.zeros(3)
        if 0.0 <= t <= duration:
            envelope = np.sin(np.pi * t / duration) ** 2
            out[2] = float(amplitude) * envelope * np.sin(float(omega) * t + float(phase))
        return out

    field.duration = duration
    field.shape = "sin2"
    field.cycles = float(cycles)
    field.ramp_cycles = 0.0
    field.flat_cycles = 0.0
    return field


def flat_top_pulse(amplitude, omega, cycles, ramp_cycles=1.0, flat_cycles=None, phase=0.0):
    period = optical_period(omega)
    ramp_cycles = float(ramp_cycles)
    if flat_cycles is None:
        flat_cycles = float(cycles) - 2.0 * ramp_cycles
    flat_cycles = float(flat_cycles)
    if ramp_cycles < 0.0:
        raise ValueError("ramp_cycles must be non-negative.")
    if flat_cycles <= 0.0:
        raise ValueError("flat-top pulse needs a positive flat_cycles window.")

    ramp = ramp_cycles * period
    flat = flat_cycles * period
    duration = 2.0 * ramp + flat

    def envelope(t):
        if t < 0.0 or t > duration:
            return 0.0
        if ramp > 0.0 and t < ramp:
            return float(np.sin(0.5 * np.pi * t / ramp) ** 2)
        if t <= ramp + flat:
            return 1.0
        if ramp > 0.0:
            return float(np.sin(0.5 * np.pi * (duration - t) / ramp) ** 2)
        return 0.0

    def field(t):
        t = float(t)
        out = np.zeros(3)
        env = envelope(t)
        if env != 0.0:
            out[2] = float(amplitude) * env * np.sin(float(omega) * t + float(phase))
        return out

    field.duration = duration
    field.shape = "flat-top"
    field.cycles = duration / period
    field.ramp_cycles = ramp_cycles
    field.flat_cycles = flat_cycles
    return field


def build_pulse(args):
    if args.pulse_shape == "gaussian":
        pulse = gaussian_pulse(
            amplitude=args.field,
            center=args.center,
            width=args.width,
            omega=args.omega,
            phase=args.phase,
            polarization=(0.0, 0.0, 1.0),
        )
        pulse.shape = "gaussian"
        pulse.cycles = None
        return pulse
    if args.pulse_shape == "sin2":
        return sin2_pulse(args.field, args.omega, args.cycles, phase=args.phase)
    return flat_top_pulse(
        args.field,
        args.omega,
        args.cycles,
        ramp_cycles=args.ramp_cycles,
        flat_cycles=args.flat_cycles,
        phase=args.phase,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lz", type=float, default=4.0)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--h2-bond", type=float, default=1.4)
    parser.add_argument("--bond", type=int, default=32)
    parser.add_argument("--td-bond", type=int, default=None)
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--no-dmrg-symmetry", action="store_true")
    parser.add_argument("--symbolic-algo", choices=("qr", "Hopcroft-Karp", "Hungarian"), default="qr")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--integrator", choices=("tdvp", "tdvp2", "taylor"), default="tdvp")
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-13)
    parser.add_argument("--diagonal-fast-path", action="store_true")
    parser.add_argument("--tdvp-dynamic-mode", choices=("split", "midpoint"), default="split")
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--pulse-shape", choices=("gaussian", "sin2", "flat-top"), default="gaussian")
    parser.add_argument("--time-reversal-steps", type=int, default=None)
    parser.add_argument("--no-track-energy", action="store_true")
    parser.add_argument("--field", type=float, default=1.0e-3)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--center", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.35)
    parser.add_argument("--cycles", type=float, default=8.0)
    parser.add_argument("--ramp-cycles", type=float, default=1.0)
    parser.add_argument("--flat-cycles", type=float, default=None)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--out", default="/private/tmp/gdvr_tddmrg_vs_rttdhf.png")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    z0 = 0.5 * float(args.h2_bond)
    mol = AtomicChain(["H", "H"], coords=[(0.0, 0.0, -z0), (0.0, 0.0, z0)])
    mol.build(Lz=args.lz, Nz=args.nz, M=args.m, verbose=False)
    mf = mol.RHF().run(conv=1.0e-8, max_iter=80, verbose=False)

    pulse = build_pulse(args)
    if args.steps is None:
        args.steps = 160 if args.pulse_shape == "gaussian" else int(np.ceil(pulse.duration / float(args.dt)))

    td = mf.TDDMRG(
        D=args.bond,
        td_bond_dim=args.bond if args.td_bond is None else args.td_bond,
        symbolic_algo=args.symbolic_algo,
    ).build()
    if args.skip_dmrg:
        td.e_tot = np.nan
    else:
        td.optimize_ground_state(
            nstates=1,
            nsweeps=args.sweeps,
            symmetry_list=None if args.no_dmrg_symmetry else ["charge", "sz"],
            compute_s2=False,
            davidson_tol=1.0e-8,
        )

    psi0 = td.default_initial_condition() if args.skip_dmrg else td.export_ground_state(dense=True)
    mu_mpo = td.get_interaction_mpo(axis=2)
    mu0_dmrg = float(np.real(psi0.expectation(mu_mpo)))
    td.run(
        psi0=psi0,
        dt=args.dt,
        steps=args.steps,
        e_ops=["mu_z"],
        field=pulse,
        order=args.order,
        integrator=args.integrator,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        diagonal_fast_path=args.diagonal_fast_path,
        tdvp_dynamic_mode=args.tdvp_dynamic_mode,
        track_energy=not args.no_track_energy,
    )
    times_dmrg = np.concatenate(([0.0], td.times))
    mu_dmrg = np.concatenate(([mu0_dmrg], np.real(td.observables[:, 0])))

    reversal_steps = min(args.steps, 20) if args.time_reversal_steps is None else args.time_reversal_steps
    reversal = None
    if reversal_steps > 0:
        reversal = td.time_reversal_error(
            psi0=psi0,
            dt=args.dt,
            steps=reversal_steps,
            field=pulse,
            order=args.order,
            integrator=args.integrator,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            diagonal_fast_path=args.diagonal_fast_path,
            tdvp_dynamic_mode=args.tdvp_dynamic_mode,
        )

    rt = GDVRRTTDHF(
        mf,
        interaction=mol.dipole_operator("z"),
        field=pulse,
    ).run(dt=args.dt, nsteps=args.steps, method="orbital")

    field_z = np.array([pulse(t)[2] for t in rt.times])
    mu_tdhf = np.asarray(rt.dipoles[:, 2], dtype=float)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.8), sharex=True)
    axes[0].plot(rt.times, field_z, color="tab:red", lw=1.8)
    axes[0].set_ylabel(r"$E_z(t)$")

    axes[1].plot(times_dmrg, mu_dmrg - mu_dmrg[0], color="tab:blue", lw=1.8, label="GDVR-TDDMRG")
    axes[1].plot(rt.times, mu_tdhf - mu_tdhf[0], color="tab:green", lw=1.8, ls="--", label="GDVR RT-TDHF")
    axes[1].set_ylabel(r"$\Delta\mu_z(t)$")
    axes[1].legend(frameon=False)

    axes[2].plot(
        times_dmrg,
        (mu_dmrg - mu_dmrg[0]) - (mu_tdhf - mu_tdhf[0]),
        color="black",
        lw=1.6,
    )
    axes[2].set_ylabel("TDDMRG - TDHF")

    norm_times = np.arange(1, args.steps + 1, dtype=float) * args.dt
    norm_error = np.abs(np.asarray(td.pre_normalization_norms, dtype=float) - 1.0)
    energy_change = np.abs(np.real(td.energy_drift))
    axes[3].plot(norm_times, np.maximum(norm_error, 1.0e-16), color="tab:purple", lw=1.4, label=r"$|N_\mathrm{pre}-1|$")
    axes[3].plot(td.energy_times, np.maximum(energy_change, 1.0e-16), color="tab:orange", lw=1.4, label=r"$|\Delta\langle H_0\rangle|$")
    if td.tdvp_truncation_errors is not None:
        axes[3].plot(
            norm_times,
            np.maximum(td.tdvp_truncation_errors, 1.0e-16),
            color="tab:brown",
            lw=1.2,
            label="TDVP trunc.",
        )
    axes[3].set_yscale("log")
    axes[3].set_xlabel("time (a.u.)")
    axes[3].set_ylabel("diagnostics")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"H2 GDVR direct-MPO comparison, Nz={args.nz}, M={args.m}\n"
        f"{args.pulse_shape} pulse, E0={args.field:g} au, omega={args.omega:g} au, "
        f"dt={args.dt:g} au\n"
        f"E_RHF={mf.e_tot:.8f} Ha, E_DMRG={td.e_tot:.8f} Ha, D={td.td_bond_dim}, "
        f"{args.integrator}, {args.tdvp_dynamic_mode}, Krylov={args.krylov_dim}, Taylor={args.order}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    np.savez_compressed(
        out.with_suffix(".npz"),
        time_tddmrg_au=times_dmrg,
        mu_tddmrg_z=mu_dmrg,
        induced_mu_tddmrg_z=mu_dmrg - mu_dmrg[0],
        time_tdhf_au=rt.times,
        mu_tdhf_z=mu_tdhf,
        induced_mu_tdhf_z=mu_tdhf - mu_tdhf[0],
        field_z=field_z,
        pre_normalization_norms=td.pre_normalization_norms,
        tdvp_truncation_errors=td.tdvp_truncation_errors,
        energy_times=td.energy_times,
        energy_drift=td.energy_drift,
    )
    np.savetxt(
        out.with_suffix(".trace.csv"),
        np.column_stack(
            [
                rt.times,
                field_z,
                mu_tdhf,
                mu_tdhf - mu_tdhf[0],
                times_dmrg,
                mu_dmrg,
                mu_dmrg - mu_dmrg[0],
            ]
        ),
        delimiter=",",
        header=(
            "time_au,field_z,mu_tdhf_z,induced_mu_tdhf_z,"
            "time_tddmrg_au,mu_tddmrg_z,induced_mu_tddmrg_z"
        ),
        comments="",
    )

    if hasattr(pulse, "duration"):
        print(
            f"Pulse:           {pulse.shape}, cycles={pulse.cycles:g}, "
            f"T={pulse.duration * AU_TIME_FS:.6f} fs"
        )
    else:
        print(f"Pulse:           {pulse.shape}")
    print(f"Steps/dt:        {args.steps} / {args.dt:g} au")
    print(f"RHF energy:      {mf.e_tot:.12f} Ha")
    if args.skip_dmrg:
        print("DMRG energy:     skipped")
    else:
        print(f"DMRG energy:     {td.e_tot:.12f} Ha")
    if (
        td.static_energies is not None
        and len(td.static_energies) > 0
        and np.any(np.isfinite(td.static_energies))
    ):
        print(f"Initial <H0>:    {np.real(td.static_energies[0]):.12f} Ha")
    print(f"GDVR MPO terms:  {td.build_info['symbolic_terms']}")
    print(f"max |dmu_TDHF|:  {np.max(np.abs(mu_tdhf - mu_tdhf[0])):.6e}")
    print(f"max |dmu_DMRG|:  {np.max(np.abs(mu_dmrg - mu_dmrg[0])):.6e}")
    print(f"max |Npre - 1|:  {np.nanmax(norm_error):.6e}")
    if np.any(np.isfinite(energy_change)):
        print(f"max |d<H0>|:     {np.nanmax(energy_change):.6e} Ha")
    else:
        print("max |d<H0>|:     skipped")
    if td.tdvp_truncation_errors is not None:
        print(f"max TDVP trunc:  {np.nanmax(td.tdvp_truncation_errors):.6e}")
    if reversal is not None:
        print(f"time reversal:   {reversal['state_error']:.6e} over {reversal_steps} steps")
    print(f"Saved figure:    {out}")
    print(f"Saved trace:     {out.with_suffix('.trace.csv')}")
    print(f"Saved npz:       {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
