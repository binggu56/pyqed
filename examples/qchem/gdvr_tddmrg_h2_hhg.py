#!/usr/bin/env python3
"""Direct GDVR-TDDMRG HHG prototype for linear hydrogen chains.

This driver reuses the HHG spectrum analysis from ``gdvr_h2_hhg.py`` but
propagates the dipole signal with the direct GDVR TDDMRG path.
"""

from __future__ import annotations

import argparse
import json
import time as walltime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.qchem.gdvr_h2_hhg import (
    AU_TIME_FS,
    FIELD_TO_INTENSITY_W_CM2,
    analysis_bounds,
    flat_top_pulse,
    hhg_spectrum,
    sin2_pulse,
    summarize_harmonics,
    symmetry_diagnostics,
    write_csv,
)
from pyqed.mps.mps import expect_mps
from pyqed.mps.tdmps import TDMPS
from pyqed.qchem.gdvr import (
    acceleration_mpo,
    AtomicChain,
    force_mpo,
    RTTDHF,
)


def _clean_float(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_pulse(args):
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


def chain_spacing(args):
    if args.spacing is not None:
        return float(args.spacing)
    return float(args.h2_bond)


def chain_geometry(args):
    natoms = int(args.natoms)
    spacing = chain_spacing(args)
    z = (np.arange(natoms, dtype=float) - 0.5 * (natoms - 1)) * spacing
    coords = [(0.0, 0.0, float(zi)) for zi in z]
    extent = 0.0 if natoms <= 1 else float(z[-1] - z[0])
    margin = 0.5 * (float(args.lz) - extent)
    return ["H"] * natoms, coords, spacing, extent, margin


def build_hchain(args):
    elements, coords, _spacing, _extent, _margin = chain_geometry(args)
    mol = AtomicChain(elements, coords=coords)
    mol.build(
        Lz=args.lz,
        Nz=args.nz,
        M=args.m,
        transverse_basis=args.transverse_basis,
        dvr_method=args.dvr_method,
        verbose=args.verbose,
    )
    return mol


def _field_z(field, times):
    return np.asarray([field(float(t))[2] for t in np.asarray(times, dtype=float)], dtype=float)


def _tdvp_projection_backend(args):
    backend = getattr(args, "tdvp_projection_backend", None)
    if backend is None:
        return None
    backend = str(backend).lower().replace("_", "-")
    return None if backend == "none" else backend


def _expect_initial(td, psi, mpo, args):
    if psi.factors and hasattr(psi.factors[0], "qns"):
        sector_kwargs = td._tdvp_sector_settings()
        helper = TDMPS(
            td._get_td_hamiltonian(),
            D=args.td_bond or args.bond,
            local_sectors=sector_kwargs["local_sectors"],
            target_sector=sector_kwargs["target_sector"],
            tdvp_projection_backend=_tdvp_projection_backend(args),
        )
        return helper._expectation(psi, mpo)
    factors = mpo.factors if hasattr(mpo, "factors") else mpo
    return expect_mps(psi.factors, factors)


def _tddmrg_trace(td, psi0, pulse, args, acc_mpo=None):
    projection_backend = _tdvp_projection_backend(args)
    if psi0 is None:
        psi0 = td._initial_state_for_run(
            None,
            tdvp_projection_backend=projection_backend,
        )
    mu_mpo = td.get_interaction_mpo(axis=2)
    mu0 = float(np.real(_expect_initial(td, psi0, mu_mpo, args)))
    e_ops = ["mu_z"]
    acc0 = None
    if acc_mpo is not None:
        e_ops.append(acc_mpo)
        acc0 = float(np.real(_expect_initial(td, psi0, acc_mpo, args)))
    td.run(
        psi0=psi0,
        dt=args.dt,
        steps=args.steps,
        e_ops=e_ops,
        field=pulse,
        order=args.order,
        integrator=args.integrator,
        tdvp_projection_backend=projection_backend,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        diagonal_fast_path=args.diagonal_fast_path,
        tdvp_dynamic_mode=args.tdvp_dynamic_mode,
        track_energy=not args.no_track_energy,
        progress=not args.quiet,
    )
    times = np.concatenate(([0.0], np.asarray(td.times, dtype=float)))
    mu = np.concatenate(([mu0], np.real(np.asarray(td.observables[:, 0]))))
    acc = None
    if acc_mpo is not None:
        acc = np.concatenate(([acc0], np.real(np.asarray(td.observables[:, 1]))))
    return times, mu, acc


def _run_tdhf_reference(mf, mol, pulse, args):
    rt = RTTDHF(mf, interaction=mol.dipole_operator("z"), field=pulse)
    rt.run(dt=args.dt, nsteps=args.steps, store_dm=False, method="orbital")
    return rt


def plot_case(path, trace, analysis, summary, tdhf=None, tdhf_analysis=None):
    times = trace["time_au"]
    time_fs = times * AU_TIME_FS
    induced = trace["mu_z"] - float(trace["mu_z"][0])
    field_z = trace["field_z"]
    norm_error = trace["norm_error"]
    trunc_error = trace["tdvp_truncation"]
    energy_drift = trace["energy_drift"]

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.2), dpi=180, sharex=False)
    field_scale = 1.0
    if np.max(np.abs(field_z)) > 0.0 and np.max(np.abs(induced)) > 0.0:
        field_scale = float(np.max(np.abs(induced)) / np.max(np.abs(field_z)))

    axes[0].plot(time_fs, induced, lw=1.5, label="TDDMRG")
    if tdhf is not None:
        axes[0].plot(
            np.asarray(tdhf.times) * AU_TIME_FS,
            np.asarray(tdhf.dipoles[:, 2]) - float(tdhf.dipoles[0, 2]),
            lw=1.2,
            ls="--",
            label="RT-TDHF",
        )
    axes[0].plot(time_fs, field_z * field_scale, lw=1.0, alpha=0.75, label="field scaled")
    axes[0].axvspan(
        analysis["analysis_start_au"] * AU_TIME_FS,
        analysis["analysis_stop_au"] * AU_TIME_FS,
        color="0.7",
        alpha=0.18,
        lw=0.0,
    )
    axes[0].set_ylabel("Pz (a.u.)")
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        analysis["harmonic_order"],
        np.maximum(analysis["accel_norm"], 1.0e-18),
        lw=1.4,
        label=f"TDDMRG {analysis['acceleration_source']} acceleration",
    )
    axes[1].semilogy(
        analysis["harmonic_order"],
        np.maximum(analysis["fd_accel_norm"], 1.0e-18),
        lw=0.9,
        alpha=0.65,
        label="TDDMRG finite-difference acceleration",
    )
    axes[1].semilogy(
        analysis["harmonic_order"],
        np.maximum(analysis["dipole_norm"], 1.0e-18),
        lw=1.0,
        alpha=0.75,
        label=r"TDDMRG $\omega^4|\mathrm{FFT}(P)|^2$",
    )
    if tdhf_analysis is not None:
        axes[1].semilogy(
            tdhf_analysis["harmonic_order"],
            np.maximum(tdhf_analysis["fd_accel_norm"], 1.0e-18),
            lw=1.1,
            ls="--",
            label="RT-TDHF finite-difference acceleration",
        )
    axes[1].axvline(summary["hhg"]["cutoff_estimate_harmonic_order"], color="0.2", ls="--", lw=1.0)
    axes[1].set_xlim(0.0, float(summary["settings"]["max_harmonic"]))
    axes[1].set_ylim(1.0e-16, 2.0)
    axes[1].set_xlabel("harmonic order")
    axes[1].set_ylabel("normalized HHG yield")
    axes[1].legend(frameon=False)

    axes[2].plot(time_fs[1:], np.maximum(norm_error, 1.0e-16), lw=1.3, label=r"$|N_\mathrm{pre}-1|$")
    if energy_drift is not None:
        axes[2].plot(time_fs, np.maximum(np.abs(energy_drift), 1.0e-16), lw=1.2, label=r"$|\Delta\langle H_0\rangle|$")
    if trunc_error is not None:
        axes[2].plot(time_fs[1:], np.maximum(trunc_error, 1.0e-16), lw=1.1, label="TDVP trunc.")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("diagnostics")
    axes[2].legend(frameon=False)

    harmonics = analysis["harmonics"]
    axes[3].semilogy(
        harmonics[:, 0],
        np.maximum(harmonics[:, 2], 1.0e-18),
        marker="o",
        ms=3.0,
        lw=1.1,
        label="TDDMRG harmonics",
    )
    if tdhf_analysis is not None:
        axes[3].semilogy(
            tdhf_analysis["harmonics"][:, 0],
            np.maximum(tdhf_analysis["harmonics"][:, 2], 1.0e-18),
            marker="s",
            ms=2.6,
            lw=0.9,
            ls="--",
            label="RT-TDHF harmonics",
        )
    axes[3].set_xlim(1, float(summary["settings"]["max_harmonic"]))
    axes[3].set_ylim(1.0e-16, 2.0)
    axes[3].set_xlabel("harmonic order")
    axes[3].set_ylabel("window max")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{summary['grid']['system']} direct GDVR-TDDMRG HHG prototype\n"
        f"Nz={summary['grid']['Nz']}, D={summary['tddmrg']['D']}, "
        f"E0={summary['pulse']['E0_au']:g} au, cycles={summary['pulse']['actual_cycles']:.3g}"
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_case(args):
    if bool(args.transverse_opt) and int(args.m) != 1:
        raise ValueError("--transverse-opt currently requires --m 1.")

    elements, _coords, spacing, chain_extent, edge_margin = chain_geometry(args)
    system_label = f"H{len(elements)}"
    tag = (
        f"gdvr_tddmrg_{system_label.lower()}_hhg_r{_clean_float(spacing)}_"
        f"lz{_clean_float(args.lz)}_nz{int(args.nz)}_m{int(args.m)}_"
        f"d{int(args.bond)}_dt{_clean_float(args.dt)}_e{_clean_float(args.field)}_"
        f"{args.pulse_shape}"
    )
    if bool(args.transverse_opt):
        tag += "_too"
    prefix = Path(args.outdir) / tag
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[build] {system_label} GDVR R={spacing:g} bohr "
        f"extent={chain_extent:g} margin={edge_margin:g} Lz={args.lz:g} Nz={args.nz} M={args.m}"
    )
    start = walltime.perf_counter()
    mol = build_hchain(args)
    build_seconds = walltime.perf_counter() - start

    print("[scf] RHF")
    start = walltime.perf_counter()
    mf = mol.RHF().run(conv=args.scf_conv, max_iter=args.scf_max_iter, verbose=args.verbose)
    scf_seconds = walltime.perf_counter() - start
    e_before_transverse_opt = None
    transverse_opt_seconds = 0.0
    if bool(args.transverse_opt):
        print("[too] transverse orbital optimization")
        e_before_transverse_opt = float(mf.e_tot)
        start = walltime.perf_counter()
        mf.newton(
            tol=float(args.transverse_opt_tol),
            max_cycles=int(args.transverse_opt_cycles),
            sweep_iterations=int(args.transverse_opt_sweeps),
            ridge=float(args.transverse_opt_ridge),
            trust_step=float(args.transverse_opt_step),
            trust_radius=float(args.transverse_opt_radius),
            scf_conv=float(args.scf_conv),
            scf_max_iter=int(args.scf_max_iter),
            verbose=bool(args.verbose),
        )
        transverse_opt_seconds = walltime.perf_counter() - start

    pulse = build_pulse(args)
    if args.steps is None:
        args.steps = int(np.ceil(float(pulse.duration) / float(args.dt)))
    analysis_start, analysis_stop, analysis_window_name = analysis_bounds(pulse, args.analysis_window)

    print(
        f"[tddmrg] D={args.bond} tdD={args.td_bond or args.bond} "
        f"integrator={args.integrator} steps={args.steps} dt={args.dt:g}"
    )
    td = mf.TDDMRG(
        D=args.bond,
        td_bond_dim=args.td_bond or args.bond,
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
            davidson_tol=args.dmrg_tol,
        )

    projection_backend = _tdvp_projection_backend(args)
    if args.skip_dmrg:
        psi0 = td._initial_state_for_run(
            None,
            tdvp_projection_backend=projection_backend,
        )
    else:
        block_sparse = projection_backend in {"block", "blocks", "block-sparse", "abelian", "abelian-block"}
        psi0 = td.export_initial_guess(dense=not block_sparse)
    acc_mpo = None
    acceleration_source = "finite-difference"
    accel_mpo_build_seconds = None
    if args.acceleration_observable == "force":
        print("[accel] building GDVR slice-force MPO")
        start = walltime.perf_counter()
        acc_mpo = force_mpo(
            mol,
            symbolic_algo=args.symbolic_algo,
        )
        accel_mpo_build_seconds = walltime.perf_counter() - start
        acceleration_source = "gdvr-slice-force"
    elif args.acceleration_observable == "commutator":
        print(f"[accel] building field-free commutator MPO chi={args.accel_mpo_bond}")
        start = walltime.perf_counter()
        acc_mpo = acceleration_mpo(
            td._get_td_hamiltonian(),
            td.get_interaction_mpo(axis=2),
            chi_max=args.accel_mpo_bond,
        )
        accel_mpo_build_seconds = walltime.perf_counter() - start
        acceleration_source = "field-free-commutator"

    start = walltime.perf_counter()
    times, mu, acceleration = _tddmrg_trace(td, psi0, pulse, args, acc_mpo=acc_mpo)
    tddmrg_seconds = walltime.perf_counter() - start
    if acceleration is not None and args.acceleration_observable == "force":
        acceleration = acceleration + float(mol.nelec) * _field_z(pulse, times)

    analysis = hhg_spectrum(
        times,
        mu,
        args.omega,
        acceleration=acceleration,
        acceleration_source=acceleration_source,
        zero_pad=args.zero_pad,
        harmonic_window=args.harmonic_window,
        max_harmonic=args.max_harmonic,
        analysis_start=analysis_start,
        analysis_stop=analysis_stop,
        symmetrize_half_wave=args.symmetrize_half_wave,
    )

    tdhf = None
    tdhf_analysis = None
    tdhf_seconds = None
    if args.compare_tdhf:
        print("[tdhf] reference")
        start = walltime.perf_counter()
        tdhf = _run_tdhf_reference(mf, mol, pulse, args)
        tdhf_seconds = walltime.perf_counter() - start
        tdhf_analysis = hhg_spectrum(
            tdhf.times,
            tdhf.dipoles[:, 2],
            args.omega,
            acceleration=None,
            acceleration_source="finite-difference",
            zero_pad=args.zero_pad,
            harmonic_window=args.harmonic_window,
            max_harmonic=args.max_harmonic,
            analysis_start=analysis_start,
            analysis_stop=analysis_stop,
            symmetrize_half_wave=args.symmetrize_half_wave,
        )

    field_z = _field_z(pulse, times)
    norm_error = np.abs(np.asarray(td.pre_normalization_norms, dtype=float) - 1.0)
    trunc_error = None if td.tdvp_truncation_errors is None else np.asarray(td.tdvp_truncation_errors, dtype=float)
    energy_drift = None
    if td.energy_drift is not None:
        energy_drift = np.asarray(td.energy_drift, dtype=complex).real
        if not np.any(np.isfinite(energy_drift)):
            energy_drift = None

    write_csv(
        prefix.with_suffix(".trace.csv"),
        [
            "time_au",
            "time_fs",
            "field_z_au",
            "dipole_z_au",
            "induced_dipole_z_au",
            "dipole_acceleration_observable_au",
        ],
        [
            times,
            times * AU_TIME_FS,
            field_z,
            mu,
            mu - float(mu[0]),
            np.full_like(times, np.nan) if acceleration is None else acceleration,
        ],
    )
    write_csv(
        prefix.with_suffix(".spectrum.csv"),
        [
            "omega_au",
            "harmonic_order",
            "energy_ev",
            "hhg_finite_difference_accel_norm",
            "hhg_selected_accel_norm",
            "hhg_omega4_dipole_norm",
        ],
        [
            analysis["omega"],
            analysis["harmonic_order"],
            analysis["energy_ev"],
            analysis["fd_accel_norm"],
            analysis["accel_norm"],
            analysis["dipole_norm"],
        ],
    )
    write_csv(
        prefix.with_suffix(".harmonics.csv"),
        [
            "harmonic_order",
            "max_accel_spectrum_norm_pm_window",
            "max_finite_difference_accel_spectrum_norm_pm_window",
            "max_dipole_spectrum_norm_pm_window",
            "center_accel_spectrum_norm",
            "center_finite_difference_accel_spectrum_norm",
            "center_dipole_spectrum_norm",
            "nearest_bin_harmonic_order",
        ],
        [
            analysis["harmonics"][:, 0],
            analysis["harmonics"][:, 1],
            analysis["harmonics"][:, 2],
            analysis["harmonics"][:, 3],
            analysis["harmonics"][:, 4],
            analysis["harmonics"][:, 5],
            analysis["harmonics"][:, 6],
            analysis["harmonics"][:, 7],
        ],
    )

    up = float(args.field) ** 2 / (4.0 * float(args.omega) ** 2)
    cutoff_harmonic = (float(args.ip) + 3.17 * up) / float(args.omega)
    summary = {
        "prefix": str(prefix),
        "grid": {
            "system": system_label,
            "n_atoms": int(len(elements)),
            "Lz_bohr": float(args.lz),
            "Nz": int(args.nz),
            "M": int(args.m),
            "dz_bohr": float(mol.dz),
            "transverse_basis": None if args.transverse_basis is None else str(args.transverse_basis),
            "spacing_bohr": float(spacing),
            "bond_bohr": float(spacing),
            "chain_extent_bohr": float(chain_extent),
            "edge_margin_bohr": float(edge_margin),
        },
        "pulse": {
            "shape": str(pulse.shape),
            "actual_cycles": float(pulse.cycles),
            "omega_au": float(args.omega),
            "E0_au": float(args.field),
            "peak_intensity_w_cm2_from_E0": float(FIELD_TO_INTENSITY_W_CM2 * float(args.field) ** 2),
            "duration_fs": float(times[-1] * AU_TIME_FS),
            "dt_au": float(args.dt),
            "nsteps": int(args.steps),
        },
        "tddmrg": {
            "D": int(args.bond),
            "td_bond_dim": int(args.td_bond or args.bond),
            "integrator": str(args.integrator),
            "tdvp_dynamic_mode": str(args.tdvp_dynamic_mode),
            "tdvp_projection_backend": None if projection_backend is None else str(projection_backend),
            "skip_dmrg": bool(args.skip_dmrg),
            "RHF_energy_ha": float(mf.e_tot),
            "RHF_energy_before_transverse_opt_ha": e_before_transverse_opt,
            "transverse_opt": {
                "enabled": bool(args.transverse_opt),
                "seconds": float(transverse_opt_seconds),
                "cycles": int(mf.info.get("newton_cycles", 0)) if bool(args.transverse_opt) else 0,
                "converged": bool(mf.info.get("newton_converged", False)) if bool(args.transverse_opt) else False,
                "history_Ha": [
                    float(x) for x in mf.info.get("newton_energy_history", [])
                ] if bool(args.transverse_opt) else [],
            },
            "DMRG_energy_ha": None if args.skip_dmrg else float(td.e_tot),
            "max_abs_induced_dipole_z_au": float(np.max(np.abs(mu - float(mu[0])))),
            "acceleration_observable": str(args.acceleration_observable),
            "acceleration_mpo_bond": None if acc_mpo is None else int(max(acc_mpo.bond_orders())),
            "acceleration_source": str(acceleration_source),
            "max_norm_error": float(np.nanmax(norm_error)) if norm_error.size else 0.0,
            "max_tdvp_truncation": None if trunc_error is None else float(np.nanmax(trunc_error)),
            "max_abs_energy_drift_ha": None if energy_drift is None else float(np.nanmax(np.abs(energy_drift))),
            "symbolic_terms": int(td._active_integral_build_info["symbolic_terms"]),
            "mpo_max_bond": int(td._active_integral_build_info["mpo_max_bond"]),
        },
        "analysis": {
            "window": str(analysis_window_name),
            "analysis_start_fs": float(analysis["analysis_start_au"] * AU_TIME_FS),
            "analysis_stop_fs": float(analysis["analysis_stop_au"] * AU_TIME_FS),
            "symmetrize_half_wave": bool(args.symmetrize_half_wave),
        },
        "settings": {
            "max_harmonic": int(args.max_harmonic),
            "harmonic_window": float(args.harmonic_window),
        },
        "hhg": {
            "ponderomotive_energy_ha": float(up),
            "cutoff_estimate_harmonic_order": float(cutoff_harmonic),
            **summarize_harmonics(analysis["harmonics"], value_col=1),
            "symmetry": symmetry_diagnostics(analysis["harmonics"], value_col=1),
        },
        "timing_seconds": {
            "build": float(build_seconds),
            "scf": float(scf_seconds),
            "transverse_opt": float(transverse_opt_seconds),
            "acceleration_mpo": accel_mpo_build_seconds,
            "tddmrg": float(tddmrg_seconds),
            "tdhf": None if tdhf_seconds is None else float(tdhf_seconds),
        },
        "files": {
            "trace_csv": str(prefix.with_suffix(".trace.csv")),
            "spectrum_csv": str(prefix.with_suffix(".spectrum.csv")),
            "harmonics_csv": str(prefix.with_suffix(".harmonics.csv")),
            "npz": str(prefix.with_suffix(".npz")),
            "plot_png": str(prefix.with_suffix(".png")),
            "summary_json": str(prefix.with_suffix(".summary.json")),
        },
    }
    if tdhf_analysis is not None:
        summary["tdhf"] = {
            "max_abs_induced_dipole_z_au": float(
                np.max(np.abs(np.asarray(tdhf.dipoles[:, 2]) - float(tdhf.dipoles[0, 2])))
            ),
            "hhg": {
                **summarize_harmonics(tdhf_analysis["harmonics"], value_col=2),
                "symmetry": symmetry_diagnostics(tdhf_analysis["harmonics"], value_col=2),
            },
        }

    np.savez_compressed(
        prefix.with_suffix(".npz"),
        time_au=times,
        time_fs=times * AU_TIME_FS,
        field_z_au=field_z,
        dipole_z_au=mu,
        induced_dipole_z_au=mu - float(mu[0]),
        dipole_acceleration_observable_au=np.array([]) if acceleration is None else acceleration,
        omega_au=analysis["omega"],
        harmonic_order=analysis["harmonic_order"],
        hhg_accel_norm=analysis["accel_norm"],
        hhg_finite_difference_accel_norm=analysis["fd_accel_norm"],
        hhg_omega4_dipole_norm=analysis["dipole_norm"],
        harmonic_table=analysis["harmonics"],
        pre_normalization_norms=td.pre_normalization_norms,
        tdvp_truncation_errors=td.tdvp_truncation_errors,
        energy_drift=np.array([]) if energy_drift is None else energy_drift,
        tdhf_time_au=np.array([]) if tdhf is None else tdhf.times,
        tdhf_dipole_z_au=np.array([]) if tdhf is None else tdhf.dipoles[:, 2],
    )
    with open(prefix.with_suffix(".summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    trace = {
        "time_au": times,
        "field_z": field_z,
        "mu_z": mu,
        "norm_error": norm_error,
        "tdvp_truncation": trunc_error,
        "energy_drift": energy_drift,
    }
    plot_case(prefix.with_suffix(".png"), trace, analysis, summary, tdhf=tdhf, tdhf_analysis=tdhf_analysis)

    print(
        "[done] Hmax>1e-6={h6} Hmax>1e-8={h8} even/odd={eo:.3e} "
        "max|dmu|={dmu:.3e} norm={norm:.3e} plot={plot}".format(
            h6=summary["hhg"]["max_odd_harmonic_accel_gt_1e_minus_6"],
            h8=summary["hhg"]["max_odd_harmonic_accel_gt_1e_minus_8"],
            eo=summary["hhg"]["symmetry"]["max_even_over_max_odd"],
            dmu=summary["tddmrg"]["max_abs_induced_dipole_z_au"],
            norm=summary["tddmrg"]["max_norm_error"],
            plot=prefix.with_suffix(".png"),
        )
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natoms", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=None)
    parser.add_argument("--h2-bond", type=float, default=1.4)
    parser.add_argument("--lz", type=float, default=10.0)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--transverse-basis", default=None)
    parser.add_argument("--dvr-method", choices=("sine", "exp", "sinc"), default="sine")
    parser.add_argument("--scf-conv", type=float, default=1.0e-8)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--transverse-opt", action="store_true")
    parser.add_argument("--transverse-opt-cycles", type=int, default=10)
    parser.add_argument("--transverse-opt-sweeps", type=int, default=1)
    parser.add_argument("--transverse-opt-tol", type=float, default=1.0e-7)
    parser.add_argument("--transverse-opt-ridge", type=float, default=0.5)
    parser.add_argument("--transverse-opt-step", type=float, default=0.5)
    parser.add_argument("--transverse-opt-radius", type=float, default=1.0)
    parser.add_argument("--bond", type=int, default=64)
    parser.add_argument("--td-bond", type=int, default=None)
    parser.add_argument("--skip-dmrg", action="store_true")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--dmrg-tol", type=float, default=1.0e-8)
    parser.add_argument("--no-dmrg-symmetry", action="store_true")
    parser.add_argument("--symbolic-algo", choices=("qr", "Hopcroft-Karp", "Hungarian"), default="qr")
    parser.add_argument("--field", type=float, default=0.08)
    parser.add_argument("--omega", type=float, default=0.057)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--pulse-shape", choices=("flat-top", "sin2"), default="sin2")
    parser.add_argument("--ramp-cycles", type=float, default=0.5)
    parser.add_argument("--flat-cycles", type=float, default=None)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ip", type=float, default=0.57)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--integrator", choices=("tdvp", "tdvp2", "taylor"), default="tdvp")
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-13)
    parser.add_argument("--diagonal-fast-path", action="store_true")
    parser.add_argument("--tdvp-dynamic-mode", choices=("split", "midpoint"), default="split")
    parser.add_argument(
        "--tdvp-projection-backend",
        choices=("none", "dense", "dense-sector", "block-sparse"),
        default="block-sparse",
    )
    parser.add_argument("--no-track-energy", action="store_true")
    parser.add_argument("--acceleration-observable", choices=("finite-difference", "force", "commutator"), default="force")
    parser.add_argument("--accel-mpo-bond", type=int, default=96)
    parser.add_argument("--compare-tdhf", action="store_true")
    parser.add_argument("--analysis-window", choices=("auto", "full", "flat-top", "central"), default="auto")
    parser.add_argument("--symmetrize-half-wave", dest="symmetrize_half_wave", action="store_true", default=True)
    parser.add_argument("--no-symmetrize-half-wave", dest="symmetrize_half_wave", action="store_false")
    parser.add_argument("--zero-pad", type=int, default=8)
    parser.add_argument("--harmonic-window", type=float, default=0.2)
    parser.add_argument("--max-harmonic", type=int, default=60)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/gdvr_tddmrg_hchain_hhg"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run_case(args)


if __name__ == "__main__":
    main()
