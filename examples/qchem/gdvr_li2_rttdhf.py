#!/usr/bin/env python3
"""GDVR RT-TDHF probe for all-electron Li2 in a z-polarized pulse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem.gdvr import AtomicChain, RTTDHF, cap_operator_from_z


AU_TIME_FS = 0.02418884326505
FIELD_TO_INTENSITY_W_CM2 = 3.50944506e16


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
    return field


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=5.05, help="Li-Li distance in bohr.")
    parser.add_argument("--lz", type=float, default=16.0)
    parser.add_argument("--nz", type=int, default=31)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--transverse-basis", default="sto3g")
    parser.add_argument("--dvr-method", choices=("sine", "exp", "sinc"), default="sine")
    parser.add_argument("--scf-conv", type=float, default=1.0e-8)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--field", type=float, default=0.08)
    parser.add_argument("--omega", type=float, default=0.057)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cap-strength", type=float, default=0.005)
    parser.add_argument("--cap-width", type=float, default=4.0)
    parser.add_argument("--cap-order", type=int, default=3)
    parser.add_argument("--method", choices=("density", "orbital"), default="orbital")
    parser.add_argument("--out", type=Path, default=Path("/private/tmp/gdvr_li2_rttdhf.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    z0 = 0.5 * float(args.bond)
    mol = AtomicChain(["Li", "Li"], coords=[(0.0, 0.0, -z0), (0.0, 0.0, z0)])
    mol.build(
        Lz=float(args.lz),
        Nz=int(args.nz),
        M=int(args.m),
        transverse_basis=args.transverse_basis,
        verbose=bool(args.verbose),
        dvr_method=args.dvr_method,
    )
    mf = mol.RHF().run(conv=args.scf_conv, max_iter=args.scf_max_iter, verbose=args.verbose)

    pulse = sin2_pulse(args.field, args.omega, args.cycles, phase=args.phase)
    steps = int(args.steps) if args.steps is not None else int(np.ceil(pulse.duration / float(args.dt)))

    cap = None
    if args.cap_strength > 0.0:
        cap = cap_operator_from_z(
            mol.z,
            M=int(args.m),
            width=float(args.cap_width),
            strength=float(args.cap_strength),
            order=int(args.cap_order),
        )
    rt = RTTDHF(mf, interaction=mol.dipole_operator("z"), field=pulse, cap=cap)
    rt.run(dt=float(args.dt), nsteps=steps, store_dm=False, method=args.method)

    field_z = np.asarray([pulse(t)[2] for t in rt.times], dtype=float)
    mu_z = np.asarray(rt.dipoles[:, 2], dtype=float)
    dmu_z = mu_z - float(mu_z[0])
    electron_counts = np.asarray(rt.electron_counts, dtype=float)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
    json_path = args.json if args.json is not None else args.out.with_suffix(".summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        csv_path,
        np.column_stack((rt.times, rt.times * AU_TIME_FS, field_z, mu_z, dmu_z, electron_counts)),
        delimiter=",",
        header="time_au,time_fs,field_z_au,dipole_z_au,induced_dipole_z_au,electron_count",
        comments="",
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True)
    axes[0].plot(rt.times, field_z, color="tab:red", lw=1.8)
    axes[0].set_ylabel(r"$E_z(t)$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(rt.times, dmu_z, color="tab:blue", lw=1.8)
    axes[1].set_ylabel(r"$\Delta\mu_z(t)$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(rt.times, electron_counts, color="tab:purple", lw=1.8)
    axes[2].set_ylabel("electron count")
    axes[2].set_xlabel("time (a.u.)")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "Li2 GDVR RT-TDHF, "
        f"R={args.bond:g} bohr, Nz={args.nz}, M={args.m}, "
        f"E0={args.field:g} au, omega={args.omega:g} au"
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "grid": {
            "Lz_bohr": float(args.lz),
            "Nz": int(args.nz),
            "M": int(args.m),
            "dz_bohr": float(mol.dz),
            "transverse_basis": args.transverse_basis,
        },
        "molecule": {
            "bond_bohr": float(args.bond),
            "nelec": int(mol.nelec),
            "nuclear_repulsion_ha": float(mol.nuclear_repulsion_energy()),
        },
        "scf": {
            "energy_ha": float(mf.e_tot),
            "occupied_orbitals": int(np.count_nonzero(np.asarray(mf.mo_occ) > 1.0e-8)),
            "mo_energy_ha": np.asarray(mf.mo_energy[: min(8, len(mf.mo_energy))], dtype=float).tolist(),
        },
        "pulse": {
            "shape": "sin2",
            "E0_au": float(args.field),
            "omega_au": float(args.omega),
            "cycles": float(args.cycles),
            "duration_fs": float(pulse.duration * AU_TIME_FS),
            "dt_au": float(args.dt),
            "nsteps": int(steps),
            "intensity_w_cm2": float(FIELD_TO_INTENSITY_W_CM2 * float(args.field) ** 2),
        },
        "observables": {
            "max_abs_induced_dipole_z_au": float(np.max(np.abs(dmu_z))),
            "final_induced_dipole_z_au": float(dmu_z[-1]),
            "min_electron_count": float(np.min(electron_counts)),
            "final_electron_count": float(electron_counts[-1]),
        },
        "files": {
            "plot_png": str(args.out),
            "timeseries_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Li2 GDVR RT-TDHF R={args.bond:g} bohr Nz={args.nz} M={args.m}")
    print(f"RHF energy:      {mf.e_tot:.12f} Ha")
    print(f"Pulse:           sin2, cycles={args.cycles:g}, T={pulse.duration * AU_TIME_FS:.6f} fs")
    print(f"Steps/dt:        {steps} / {args.dt:g} au")
    print(f"max |dmu_z|:     {np.max(np.abs(dmu_z)):.6e}")
    print(f"min N_e:         {np.min(electron_counts):.6f}")
    print(f"final N_e:       {electron_counts[-1]:.6f}")
    print(f"Saved figure:    {args.out}")
    print(f"Saved CSV:       {csv_path}")
    print(f"Saved summary:   {json_path}")


if __name__ == "__main__":
    main()
