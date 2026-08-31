#!/usr/bin/env python3
"""Compare H2 GDVR RT-TDHF against PySCF/GTO references.

The comparison has two regimes:

1. Clean validation: RHF energy, HOMO/LUMO, finite-field electronic
   polarizability, PySCF TDHF roots, and weak-kick real-time polarization.
2. Strong-field sanity: the same IR pulse propagated with GDVR and GTO, then
   compared through a shared finite-difference HHG spectrum.
"""

from pathlib import Path
import argparse
import csv
import json
import sys
import time as walltime

import numpy as np
from pyscf import gto, scf, tdscf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import Molecule as PyQEDMolecule
from pyqed.qchem import RHF as PyQEDRHF
from pyqed.qchem import RTTDHF as PyQEDRTTDHF
from pyqed.qchem.gdvr import AtomicChain, RTTDHF as GDVRRT
from pyqed.qchem.gdvr.rhf import scf_rhf_method2

from gdvr_h2_hhg import (
    AU_TIME_FS,
    FIELD_TO_INTENSITY_W_CM2,
    flat_top_pulse,
    hhg_spectrum,
    optical_period,
    symmetry_diagnostics,
)


def h2_atom_string(bond):
    bond = float(bond)
    return f"H 0 0 {-0.5 * bond:.16g}; H 0 0 {0.5 * bond:.16g}"


def z_operator_gdvr(mol):
    return mol.position_operator("z")


def build_gdvr_reference(args):
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[[0.0, 0.0, -0.5 * args.bond], [0.0, 0.0, 0.5 * args.bond]],
    )
    t0 = walltime.perf_counter()
    mol.build(
        Lz=args.gdvr_lz,
        Nz=args.gdvr_nz,
        M=args.gdvr_m,
        verbose=args.verbose,
        dvr_method=args.gdvr_method,
    )
    build_seconds = walltime.perf_counter() - t0
    t0 = walltime.perf_counter()
    mf = mol.RHF().run(
        conv=args.scf_conv,
        max_iter=args.scf_max_iter,
        verbose=args.verbose,
    )
    scf_seconds = walltime.perf_counter() - t0
    return mol, mf, {"build": build_seconds, "scf": scf_seconds}


def run_gdvr_static_field(mol, field, args, dm0=None):
    nz = int(mol.shapes["Nz"])
    m = int(mol.shapes["M"])
    z_op = z_operator_gdvr(mol)
    # Electronic dipole is mu_z = -z, so H - E mu_z = H + E z.
    hcore = np.asarray(mol.hcore, dtype=float) + float(field) * z_op
    etot, eps, cmo, dm, info = scf_rhf_method2(
        hcore,
        mol.eri_j,
        mol.eri_k,
        nz,
        m,
        mol.nelec,
        Enuc=mol.nuclear_repulsion_energy(),
        conv=args.scf_conv,
        max_iter=args.scf_max_iter,
        verbose=False,
        dm0=dm0,
    )
    z_expect = float(np.einsum("ij,ji->", z_op, dm, optimize=True).real)
    return {
        "energy_ha": float(etot),
        "mo_energy_ha": np.asarray(eps, dtype=float),
        "dm": np.asarray(dm),
        "position_z_expect_au": z_expect,
        "electronic_dipole_z_au": -z_expect,
        "info": dict(info),
    }


def gdvr_static_polarizability(mol, mf, args):
    plus = run_gdvr_static_field(mol, args.static_field, args, dm0=mf.dm)
    minus = run_gdvr_static_field(mol, -args.static_field, args, dm0=mf.dm)
    alpha = (
        plus["electronic_dipole_z_au"] - minus["electronic_dipole_z_au"]
    ) / (2.0 * float(args.static_field))
    return float(alpha), plus, minus


def build_gto_reference(bond, basis, args):
    mol = gto.M(
        atom=h2_atom_string(bond),
        unit="Bohr",
        basis=basis,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv
    mf.max_cycle = args.scf_max_iter
    t0 = walltime.perf_counter()
    mf.kernel()
    return mol, mf, walltime.perf_counter() - t0


def gto_electronic_z(mol, dm_ao):
    z_ao = mol.intor("int1e_r", comp=3)[2]
    return float(np.einsum("ij,ji->", z_ao, dm_ao, optimize=True).real)


def run_gto_static_field(bond, basis, field, args, dm0=None):
    mol = gto.M(
        atom=h2_atom_string(bond),
        unit="Bohr",
        basis=basis,
        verbose=0,
    )
    base = scf.RHF(mol)
    h0 = base.get_hcore()
    z_ao = mol.intor("int1e_r", comp=3)[2]
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv
    mf.max_cycle = args.scf_max_iter
    # Electronic dipole is mu_z = -z, so H - E mu_z = H + E z.
    mf.get_hcore = lambda mol_arg=None: h0 + float(field) * z_ao
    energy = mf.kernel(dm0=dm0)
    dm = mf.make_rdm1()
    z_expect = gto_electronic_z(mol, dm)
    return {
        "energy_ha": float(energy),
        "mo_energy_ha": np.asarray(mf.mo_energy, dtype=float),
        "dm": np.asarray(dm),
        "position_z_expect_au": z_expect,
        "electronic_dipole_z_au": -z_expect,
        "converged": bool(mf.converged),
    }


def gto_static_polarizability(bond, basis, mf, args):
    dm0 = mf.make_rdm1()
    plus = run_gto_static_field(bond, basis, args.static_field, args, dm0=dm0)
    minus = run_gto_static_field(bond, basis, -args.static_field, args, dm0=dm0)
    alpha = (
        plus["electronic_dipole_z_au"] - minus["electronic_dipole_z_au"]
    ) / (2.0 * float(args.static_field))
    return float(alpha), plus, minus


def gto_tdhf_roots(mf, nstates):
    td = tdscf.TDHF(mf)
    td.verbose = 0
    roots, _xy = td.kernel(nstates=int(nstates))
    return np.asarray(roots, dtype=float)


def build_pyqed_gto_reference(bond, basis):
    mol = PyQEDMolecule(atom=h2_atom_string(bond), unit="bohr", basis=basis)
    mol.build()
    return PyQEDRHF(mol).run()


def run_gdvr_kick(mf, args):
    rt = GDVRRT(mf, interaction=mf.mol.dipole_operator("z")).run(
        dt=args.weak_dt,
        nsteps=args.weak_steps,
        store_dm=False,
        method=args.gdvr_propagator,
        kick={"strength": args.kick_strength, "axis": "z"},
    )
    return rt


def run_gto_kick(mf, args):
    return PyQEDRTTDHF(mf).run(
        dt=args.weak_dt,
        nsteps=args.weak_steps,
        store_dm=False,
        kick={"strength": args.kick_strength, "axis": "z"},
    )


def simple_spectrum(time, signal, zero_pad=8, max_omega=2.0):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    dt = float(time[1] - time[0])
    y = signal - signal[0]
    y = y - np.mean(y)
    window = np.hanning(y.size)
    nfft = int(max(1, zero_pad) * y.size)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)
    spec = np.abs(np.fft.rfft(y * window, n=nfft)) ** 2
    if np.max(spec[omega > 0.0]) > 0.0:
        spec = spec / np.max(spec[omega > 0.0])
    mask = (omega >= 0.03) & (omega <= float(max_omega))
    peak = float(omega[mask][np.argmax(spec[mask])]) if np.any(mask) else None
    return {"omega": omega, "spectrum": spec, "peak_omega_ha": peak}


def build_strong_pulse(args):
    return flat_top_pulse(
        args.strong_field,
        args.strong_omega,
        cycles=args.strong_cycles,
        ramp_cycles=args.strong_ramp_cycles,
        flat_cycles=args.strong_flat_cycles,
        phase=args.strong_phase,
    )


def run_gdvr_strong(mol, mf, pulse, args):
    cap = None
    if args.gdvr_cap_strength > 0.0:
        cap = mol.cap(
            width=args.gdvr_cap_width,
            strength=args.gdvr_cap_strength,
            order=args.gdvr_cap_order,
        )
    rt = GDVRRT(
        mf,
        interaction=mol.dipole_operator("z"),
        field=pulse,
        cap=cap,
    )
    nsteps = int(np.ceil(pulse.duration / float(args.strong_dt)))
    return rt.run(
        dt=args.strong_dt,
        nsteps=nsteps,
        store_dm=False,
        method=args.gdvr_propagator,
    )


def run_gto_strong(mf, pulse, args):
    nsteps = int(np.ceil(pulse.duration / float(args.strong_dt)))
    return PyQEDRTTDHF(mf, field=pulse).run(
        dt=args.strong_dt,
        nsteps=nsteps,
        store_dm=False,
    )


def analyze_hhg(rt, pulse, args):
    analysis = hhg_spectrum(
        rt.times,
        rt.dipoles[:, 2],
        args.strong_omega,
        acceleration=None,
        acceleration_source="finite-difference",
        zero_pad=args.zero_pad,
        harmonic_window=args.harmonic_window,
        max_harmonic=args.max_harmonic,
        analysis_start=pulse.analysis_start,
        analysis_stop=pulse.analysis_stop,
        symmetrize_half_wave=args.symmetrize_half_wave,
    )
    return analysis


def harmonic_strength(analysis, harmonic, center=True):
    table = np.asarray(analysis["harmonics"], dtype=float)
    row = table[table[:, 0].astype(int) == int(harmonic)]
    if row.size == 0:
        return None
    return float(row[0, 4 if center else 1])


def rt_electron_counts(rt):
    stored = getattr(rt, "electron_counts", None)
    if stored is not None:
        return np.asarray(stored, dtype=float)
    if hasattr(rt, "electron_count"):
        count = float(rt.electron_count(rt.dm))
    else:
        count = np.nan
    return np.full(np.asarray(rt.times).shape, count, dtype=float)


def rt_electron_loss(rt):
    counts = rt_electron_counts(rt)
    return float(counts[0] - counts[-1])


def row_for_method(
    method,
    basis_or_grid,
    energy,
    homo,
    lumo,
    alpha,
    tdhf_roots=None,
    weak_peak=None,
    weak_pmax=None,
    strong_analysis=None,
    electron_loss=None,
):
    roots = [] if tdhf_roots is None else [float(x) for x in np.asarray(tdhf_roots).reshape(-1)]
    return {
        "method": str(method),
        "basis_or_grid": str(basis_or_grid),
        "energy_ha": float(energy),
        "homo_ha": float(homo),
        "lumo_ha": float(lumo),
        "gap_ha": float(lumo - homo),
        "alpha_z_electronic_au": float(alpha),
        "tdhf_root1_ha": roots[0] if roots else None,
        "tdhf_root2_ha": roots[1] if len(roots) > 1 else None,
        "weak_rt_peak_ha": weak_peak,
        "weak_peak_abs_mu_z_au": weak_pmax,
        "hhg_h3_center": None if strong_analysis is None else harmonic_strength(strong_analysis, 3),
        "hhg_h5_center": None if strong_analysis is None else harmonic_strength(strong_analysis, 5),
        "hhg_h7_center": None if strong_analysis is None else harmonic_strength(strong_analysis, 7),
        "hhg_h9_center": None if strong_analysis is None else harmonic_strength(strong_analysis, 9),
        "hhg_even_over_odd_center": None
        if strong_analysis is None
        else symmetry_diagnostics(strong_analysis["harmonics"], value_col=4)["max_even_over_max_odd"],
        "electron_loss": electron_loss,
    }


def write_rows_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path, weak, strong, analyses):
    np.savez_compressed(
        path,
        gdvr_weak_time_au=weak["gdvr"].times,
        gdvr_weak_mu_z=weak["gdvr"].dipoles[:, 2] - weak["gdvr"].dipoles[0, 2],
        gto_weak_time_au=weak["gto"].times,
        gto_weak_mu_z=weak["gto"].dipoles[:, 2] - weak["gto"].dipoles[0, 2],
        gdvr_strong_time_au=strong["gdvr"].times,
        gdvr_strong_mu_z=strong["gdvr"].dipoles[:, 2] - strong["gdvr"].dipoles[0, 2],
        gdvr_strong_field_z=strong["gdvr"].fields[:, 2],
        gdvr_strong_electron_count=rt_electron_counts(strong["gdvr"]),
        gto_strong_time_au=strong["gto"].times,
        gto_strong_mu_z=strong["gto"].dipoles[:, 2] - strong["gto"].dipoles[0, 2],
        gto_strong_field_z=strong["gto"].fields[:, 2],
        gto_strong_electron_count=rt_electron_counts(strong["gto"]),
        gdvr_hhg_order=analyses["gdvr"]["harmonic_order"],
        gdvr_hhg_norm=analyses["gdvr"]["dipole_norm"],
        gto_hhg_order=analyses["gto"]["harmonic_order"],
        gto_hhg_norm=analyses["gto"]["dipole_norm"],
    )


def plot_comparison(path, rows, weak, weak_spec, strong, analyses, args):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), dpi=180)

    labels = [row["method"] for row in rows]
    energies = [row["energy_ha"] for row in rows]
    axes[0, 0].bar(np.arange(len(rows)), energies, color=["C0"] + ["C1"] * (len(rows) - 1))
    axes[0, 0].set_xticks(np.arange(len(rows)), labels, rotation=25, ha="right")
    axes[0, 0].set_ylabel("RHF energy (Ha)")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].plot(
        weak["gdvr"].times * AU_TIME_FS,
        weak["gdvr"].dipoles[:, 2] - weak["gdvr"].dipoles[0, 2],
        lw=1.5,
        label="GDVR",
    )
    axes[0, 1].plot(
        weak["gto"].times * AU_TIME_FS,
        weak["gto"].dipoles[:, 2] - weak["gto"].dipoles[0, 2],
        lw=1.2,
        label=f"GTO {args.rt_gto_basis}",
    )
    axes[0, 1].set_xlabel("time (fs)")
    axes[0, 1].set_ylabel("weak-kick induced electronic dipole (a.u.)")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        strong["gdvr"].times * AU_TIME_FS,
        strong["gdvr"].dipoles[:, 2] - strong["gdvr"].dipoles[0, 2],
        lw=1.3,
        label="GDVR",
    )
    axes[1, 0].plot(
        strong["gto"].times * AU_TIME_FS,
        strong["gto"].dipoles[:, 2] - strong["gto"].dipoles[0, 2],
        lw=1.1,
        label=f"GTO {args.rt_gto_basis}",
    )
    axes[1, 0].set_xlabel("time (fs)")
    axes[1, 0].set_ylabel("strong-field induced electronic dipole (a.u.)")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].semilogy(
        analyses["gdvr"]["harmonic_order"],
        np.maximum(analyses["gdvr"]["dipole_norm"], 1e-18),
        lw=1.35,
        label="GDVR",
    )
    axes[1, 1].semilogy(
        analyses["gto"]["harmonic_order"],
        np.maximum(analyses["gto"]["dipole_norm"], 1e-18),
        lw=1.15,
        label=f"GTO {args.rt_gto_basis}",
    )
    axes[1, 1].set_xlim(0, args.max_harmonic)
    axes[1, 1].set_ylim(1e-14, 2.0)
    axes[1, 1].set_xlabel("harmonic order")
    axes[1, 1].set_ylabel("normalized HHG yield")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=1.4)
    parser.add_argument("--gto-basis", nargs="+", default=["sto-3g", "aug-cc-pvdz"])
    parser.add_argument("--rt-gto-basis", default=None)
    parser.add_argument("--gdvr-lz", type=float, default=8.0)
    parser.add_argument("--gdvr-nz", type=int, default=41)
    parser.add_argument("--gdvr-m", type=int, default=2)
    parser.add_argument("--gdvr-method", choices=("sine", "exp", "sinc"), default="sine")
    parser.add_argument("--gdvr-propagator", choices=("density", "orbital"), default="orbital")
    parser.add_argument("--gdvr-cap-strength", type=float, default=0.002)
    parser.add_argument("--gdvr-cap-width", type=float, default=2.0)
    parser.add_argument("--gdvr-cap-order", type=int, default=3)
    parser.add_argument("--scf-conv", type=float, default=1e-9)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--static-field", type=float, default=1e-4)
    parser.add_argument("--tdhf-roots", type=int, default=2)
    parser.add_argument("--kick-strength", type=float, default=1e-4)
    parser.add_argument("--weak-dt", type=float, default=0.1)
    parser.add_argument("--weak-steps", type=int, default=1200)
    parser.add_argument("--strong-field", type=float, default=0.05)
    parser.add_argument("--strong-omega", type=float, default=0.057)
    parser.add_argument("--strong-cycles", type=float, default=4.0)
    parser.add_argument("--strong-ramp-cycles", type=float, default=0.5)
    parser.add_argument("--strong-flat-cycles", type=float, default=None)
    parser.add_argument("--strong-phase", type=float, default=0.0)
    parser.add_argument("--strong-dt", type=float, default=0.25)
    parser.add_argument("--zero-pad", type=int, default=8)
    parser.add_argument("--harmonic-window", type=float, default=0.2)
    parser.add_argument("--max-harmonic", type=int, default=40)
    parser.add_argument("--symmetrize-half-wave", dest="symmetrize_half_wave", action="store_true", default=True)
    parser.add_argument("--no-symmetrize-half-wave", dest="symmetrize_half_wave", action="store_false")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/gdvr_h2_vs_gto"))
    parser.add_argument("--tag", default="gdvr_h2_vs_gto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.rt_gto_basis = args.rt_gto_basis or args.gto_basis[-1]
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[gdvr] Lz={args.gdvr_lz:g} Nz={args.gdvr_nz} M={args.gdvr_m} "
        f"bond={args.bond:g}"
    )
    gdvr_mol, gdvr_mf, gdvr_timing = build_gdvr_reference(args)
    gdvr_alpha, gdvr_plus, gdvr_minus = gdvr_static_polarizability(gdvr_mol, gdvr_mf, args)
    gdvr_grid_label = (
        f"L{args.gdvr_lz:g}/N{args.gdvr_nz}/M{args.gdvr_m}/"
        f"dz={gdvr_mol.dz:.3f}"
    )
    print(f"[gdvr] E={gdvr_mf.e_tot:.12f} alpha_z={gdvr_alpha:.6f}")

    gto_refs = {}
    for basis in args.gto_basis:
        print(f"[gto] basis={basis}")
        mol, mf, scf_seconds = build_gto_reference(args.bond, basis, args)
        alpha, plus, minus = gto_static_polarizability(args.bond, basis, mf, args)
        roots = gto_tdhf_roots(mf, args.tdhf_roots)
        gto_refs[basis] = {
            "mol": mol,
            "mf": mf,
            "scf_seconds": scf_seconds,
            "alpha": alpha,
            "static_plus": plus,
            "static_minus": minus,
            "tdhf_roots": roots,
        }
        print(
            f"[gto] {basis} E={mf.e_tot:.12f} alpha_z={alpha:.6f} "
            f"root1={roots[0]:.6f}"
        )

    print(f"[pyqed-gto] basis={args.rt_gto_basis} for RT-TDHF")
    pyqed_gto_rt_mf = build_pyqed_gto_reference(args.bond, args.rt_gto_basis)
    pyscf_rt_energy = float(gto_refs[args.rt_gto_basis]["mf"].e_tot)
    print(
        f"[pyqed-gto] E={pyqed_gto_rt_mf.e_tot:.12f} "
        f"delta_vs_pyscf={pyqed_gto_rt_mf.e_tot - pyscf_rt_energy:+.3e}"
    )

    print("[weak] kick RT")
    weak_gdvr = run_gdvr_kick(gdvr_mf, args)
    weak_gto = run_gto_kick(pyqed_gto_rt_mf, args)
    weak_spec = {
        "gdvr": simple_spectrum(
            weak_gdvr.times,
            weak_gdvr.dipoles[:, 2] - weak_gdvr.dipoles[0, 2],
            zero_pad=args.zero_pad,
        ),
        "gto": simple_spectrum(
            weak_gto.times,
            weak_gto.dipoles[:, 2] - weak_gto.dipoles[0, 2],
            zero_pad=args.zero_pad,
        ),
    }
    print(
        f"[weak] peak GDVR={weak_spec['gdvr']['peak_omega_ha']:.6f} "
        f"GTO={weak_spec['gto']['peak_omega_ha']:.6f}"
    )

    print(
        f"[strong] E0={args.strong_field:g} omega={args.strong_omega:g} "
        f"I={FIELD_TO_INTENSITY_W_CM2 * args.strong_field ** 2:.3e} W/cm^2"
    )
    pulse = build_strong_pulse(args)
    strong_gdvr = run_gdvr_strong(gdvr_mol, gdvr_mf, pulse, args)
    strong_gto = run_gto_strong(pyqed_gto_rt_mf, pulse, args)
    analyses = {
        "gdvr": analyze_hhg(strong_gdvr, pulse, args),
        "gto": analyze_hhg(strong_gto, pulse, args),
    }

    rows = [
        row_for_method(
            "GDVR",
            gdvr_grid_label,
            gdvr_mf.e_tot,
            gdvr_mf.mo_energy[0],
            gdvr_mf.mo_energy[1],
            gdvr_alpha,
            tdhf_roots=None,
            weak_peak=weak_spec["gdvr"]["peak_omega_ha"],
            weak_pmax=float(np.max(np.abs(weak_gdvr.dipoles[:, 2] - weak_gdvr.dipoles[0, 2]))),
            strong_analysis=analyses["gdvr"],
            electron_loss=rt_electron_loss(strong_gdvr),
        )
    ]
    for basis, ref in gto_refs.items():
        mf = ref["mf"]
        rows.append(
            row_for_method(
                f"GTO:{basis}",
                basis,
                mf.e_tot,
                mf.mo_energy[mf.mol.nelectron // 2 - 1],
                mf.mo_energy[mf.mol.nelectron // 2],
                ref["alpha"],
                tdhf_roots=ref["tdhf_roots"],
                weak_peak=weak_spec["gto"]["peak_omega_ha"] if basis == args.rt_gto_basis else None,
                weak_pmax=float(np.max(np.abs(weak_gto.dipoles[:, 2] - weak_gto.dipoles[0, 2])))
                if basis == args.rt_gto_basis
                else None,
                strong_analysis=analyses["gto"] if basis == args.rt_gto_basis else None,
                electron_loss=rt_electron_loss(strong_gto) if basis == args.rt_gto_basis else None,
            )
        )

    prefix = args.outdir / args.tag
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    png_path = prefix.with_suffix(".png")
    write_rows_csv(csv_path, rows)
    save_npz(
        npz_path,
        {"gdvr": weak_gdvr, "gto": weak_gto},
        {"gdvr": strong_gdvr, "gto": strong_gto},
        analyses,
    )
    plot_comparison(
        png_path,
        rows,
        {"gdvr": weak_gdvr, "gto": weak_gto},
        weak_spec,
        {"gdvr": strong_gdvr, "gto": strong_gto},
        analyses,
        args,
    )

    summary = {
        "settings": {
            "bond_bohr": float(args.bond),
            "gdvr": {
                "Lz_bohr": float(args.gdvr_lz),
                "Nz": int(args.gdvr_nz),
                "M": int(args.gdvr_m),
                "dz_bohr": float(gdvr_mol.dz),
                "cap_strength": float(args.gdvr_cap_strength),
                "cap_width": float(args.gdvr_cap_width),
            },
            "gto_bases": list(args.gto_basis),
            "rt_gto_basis": str(args.rt_gto_basis),
            "static_field_au": float(args.static_field),
            "weak": {
                "kick_strength": float(args.kick_strength),
                "dt_au": float(args.weak_dt),
                "nsteps": int(args.weak_steps),
            },
            "strong": {
                "E0_au": float(args.strong_field),
                "omega_au": float(args.strong_omega),
                "period_fs": float(optical_period(args.strong_omega) * AU_TIME_FS),
                "duration_fs": float(pulse.duration * AU_TIME_FS),
                "dt_au": float(args.strong_dt),
                "nsteps": int(np.ceil(pulse.duration / float(args.strong_dt))),
                "intensity_w_cm2": float(FIELD_TO_INTENSITY_W_CM2 * args.strong_field ** 2),
                "analysis_start_fs": float(pulse.analysis_start * AU_TIME_FS),
                "analysis_stop_fs": float(pulse.analysis_stop * AU_TIME_FS),
                "symmetrize_half_wave": bool(args.symmetrize_half_wave),
            },
        },
        "rows": rows,
        "gdvr_static_field": {
            "plus": {k: v for k, v in gdvr_plus.items() if k not in {"dm", "mo_energy_ha"}},
            "minus": {k: v for k, v in gdvr_minus.items() if k not in {"dm", "mo_energy_ha"}},
        },
        "gto_static_field": {
            basis: {
                "plus": {k: v for k, v in ref["static_plus"].items() if k not in {"dm", "mo_energy_ha"}},
                "minus": {k: v for k, v in ref["static_minus"].items() if k not in {"dm", "mo_energy_ha"}},
            }
            for basis, ref in gto_refs.items()
        },
        "hhg_symmetry": {
            "gdvr": symmetry_diagnostics(analyses["gdvr"]["harmonics"], value_col=4),
            "gto": symmetry_diagnostics(analyses["gto"]["harmonics"], value_col=4),
        },
        "timing_seconds": {
            "gdvr": gdvr_timing,
            "gto_scf": {basis: float(ref["scf_seconds"]) for basis, ref in gto_refs.items()},
            "pyqed_gto_rt_reference_energy_ha": float(pyqed_gto_rt_mf.e_tot),
        },
        "files": {
            "csv": str(csv_path),
            "json": str(json_path),
            "npz": str(npz_path),
            "plot_png": str(png_path),
        },
    }
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[files] {csv_path}")
    print(f"[files] {json_path}")
    print(f"[files] {png_path}")
    print("[summary]")
    for row in rows:
        print(
            "{method:18s} E={energy_ha:.10f} alpha={alpha_z_electronic_au:+.6f} "
            "weak={weak_rt_peak_ha} H3={hhg_h3_center} loss={electron_loss}".format(**row)
        )


if __name__ == "__main__":
    main()
