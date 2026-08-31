#!/usr/bin/env python3
"""GDVR real-time TDHF HHG spectrum for H2 in a strong IR pulse."""

from pathlib import Path
import argparse
import json
import sys
import time as walltime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.units import au2ev, au2fs, au2watt_per_centimeter_squared
from pyqed.qchem.gdvr import AtomicChain, RTTDHF


AU_TIME_FS = au2fs
AU_ENERGY_EV = au2ev
FIELD_TO_INTENSITY_W_CM2 = au2watt_per_centimeter_squared


def _clean_float(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


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
    field.analysis_start = 0.0
    field.analysis_stop = duration
    return field


def flat_top_pulse(amplitude, omega, cycles, ramp_cycles=1.0, flat_cycles=None, phase=0.0):
    period = optical_period(omega)
    ramp_cycles = float(ramp_cycles)
    if ramp_cycles < 0.0:
        raise ValueError("ramp_cycles must be non-negative")
    if flat_cycles is None:
        flat_cycles = float(cycles) - 2.0 * ramp_cycles
    flat_cycles = float(flat_cycles)
    if flat_cycles <= 0.0:
        raise ValueError("flat-top pulse needs a positive flat_cycles window")

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
    field.analysis_start = ramp
    field.analysis_stop = ramp + flat
    return field


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


def analysis_bounds(field, mode):
    if mode == "auto":
        mode = "flat-top" if getattr(field, "shape", "") == "flat-top" else "full"
    if mode == "full":
        return 0.0, float(field.duration), mode
    if mode == "flat-top":
        start = float(getattr(field, "analysis_start", 0.0))
        stop = float(getattr(field, "analysis_stop", field.duration))
        if stop <= start:
            raise ValueError("selected flat-top analysis window is empty")
        return start, stop, mode
    if mode == "central":
        start = 0.25 * float(field.duration)
        stop = 0.75 * float(field.duration)
        return start, stop, mode
    raise ValueError(f"unknown analysis window {mode!r}")


def normalize_spectrum(values, omega):
    values = np.asarray(values, dtype=float).copy()
    mask = omega > 0.0
    scale = float(np.max(values[mask])) if np.any(mask) else 0.0
    if scale > 0.0:
        values /= scale
    return values


def _finite_difference_acceleration(time_au, polarization):
    dt = float(time_au[1] - time_au[0])
    signal = np.asarray(polarization, dtype=float) - float(polarization[0])
    centered = signal - float(np.mean(signal))
    return np.gradient(np.gradient(centered, dt), dt)


def half_wave_antisymmetrize(time_au, values, omega0):
    time_au = np.asarray(time_au, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_au.size != values.size:
        raise ValueError("time and signal arrays must have the same length")
    if time_au.size < 3:
        raise ValueError("need at least three samples for half-wave projection")
    dt = float(time_au[1] - time_au[0])
    half_period = np.pi / float(omega0)
    half_steps = int(round(half_period / dt))
    if half_steps < 1 or time_au.size <= half_steps:
        raise ValueError("analysis window is too short for half-wave projection")
    n = time_au.size - half_steps
    projected = 0.5 * (values[:n] - values[half_steps : half_steps + n])
    return (
        time_au[:n],
        projected,
        {
            "half_period_steps": int(half_steps),
            "half_period_au": float(half_period),
            "sampled_half_period_au": float(half_steps * dt),
            "half_period_error_au": float(half_steps * dt - half_period),
        },
    )


def hhg_spectrum(
    time_au,
    polarization,
    omega0,
    acceleration=None,
    acceleration_source="finite-difference",
    zero_pad=8,
    harmonic_window=0.35,
    max_harmonic=80,
    analysis_start=None,
    analysis_stop=None,
    symmetrize_half_wave=False,
):
    time_au = np.asarray(time_au, dtype=float)
    polarization = np.asarray(polarization, dtype=float)
    if acceleration is None:
        accel = None
    else:
        accel = np.asarray(acceleration, dtype=float)

    mask = np.ones(time_au.size, dtype=bool)
    if analysis_start is not None:
        mask &= time_au >= float(analysis_start)
    if analysis_stop is not None:
        mask &= time_au <= float(analysis_stop)
    time_au = time_au[mask]
    polarization = polarization[mask]
    if accel is not None:
        accel = accel[mask]
    if time_au.size < 4:
        raise ValueError("analysis window has too few samples")

    projection = None
    if symmetrize_half_wave:
        original_time = time_au
        time_au, polarization, projection = half_wave_antisymmetrize(
            original_time, polarization, omega0
        )
        if accel is not None:
            _, accel, _ = half_wave_antisymmetrize(original_time, accel, omega0)

    dt = float(time_au[1] - time_au[0])
    signal = np.asarray(polarization, dtype=float) - float(polarization[0])
    centered = signal - float(np.mean(signal))
    fd_accel = _finite_difference_acceleration(time_au, polarization)
    if accel is None:
        accel = fd_accel
        acceleration_source = "finite-difference"
    else:
        accel = accel - float(np.mean(accel))
    window = np.hanning(signal.size)
    nfft = int(max(1, zero_pad) * signal.size)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)

    accel_spec = np.abs(np.fft.rfft(accel * window, n=nfft)) ** 2
    fd_accel_spec = np.abs(np.fft.rfft(fd_accel * window, n=nfft)) ** 2
    dipole_spec = omega**4 * np.abs(np.fft.rfft(centered * window, n=nfft)) ** 2
    accel_norm = normalize_spectrum(accel_spec, omega)
    fd_accel_norm = normalize_spectrum(fd_accel_spec, omega)
    dipole_norm = normalize_spectrum(dipole_spec, omega)
    harmonic_order = omega / float(omega0)

    harmonics = []
    for order in range(1, int(max_harmonic) + 1):
        mask = np.abs(harmonic_order - order) <= float(harmonic_window)
        if np.any(mask):
            h_accel = float(np.max(accel_norm[mask]))
            h_fd_accel = float(np.max(fd_accel_norm[mask]))
            h_dipole = float(np.max(dipole_norm[mask]))
        else:
            h_accel = 0.0
            h_fd_accel = 0.0
            h_dipole = 0.0
        center = int(np.argmin(np.abs(harmonic_order - order)))
        harmonics.append(
            (
                order,
                h_accel,
                h_fd_accel,
                h_dipole,
                float(accel_norm[center]),
                float(fd_accel_norm[center]),
                float(dipole_norm[center]),
                float(harmonic_order[center]),
            )
        )

    return {
        "time_au": time_au,
        "induced_polarization": signal,
        "acceleration": accel,
        "finite_difference_acceleration": fd_accel,
        "omega": omega,
        "harmonic_order": harmonic_order,
        "energy_ev": omega * AU_ENERGY_EV,
        "accel_norm": accel_norm,
        "fd_accel_norm": fd_accel_norm,
        "dipole_norm": dipole_norm,
        "harmonics": np.asarray(harmonics, dtype=float),
        "acceleration_source": str(acceleration_source),
        "analysis_start_au": float(time_au[0]),
        "analysis_stop_au": float(time_au[-1]),
        "symmetrize_half_wave": bool(symmetrize_half_wave),
        "half_wave_projection": projection,
    }


def write_csv(path, header, columns):
    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def plot_case(path, rt, analysis, cutoff_harmonic, max_harmonic=80, induced_polarization=None):
    time_fs = rt.times * AU_TIME_FS
    induced = (
        np.asarray(induced_polarization, dtype=float)
        if induced_polarization is not None
        else analysis["induced_polarization"]
    )
    field_z = rt.fields[:, 2]
    field_scale = 1.0
    if np.max(np.abs(field_z)) > 0.0 and np.max(np.abs(induced)) > 0.0:
        field_scale = float(np.max(np.abs(induced)) / np.max(np.abs(field_z)))

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.2), dpi=180, sharex=False)
    axes[0].plot(time_fs, induced, lw=1.5, label="induced electronic polarization")
    axes[0].plot(time_fs, field_z * field_scale, lw=1.0, alpha=0.8, label="field scaled")
    axes[0].axvspan(
        analysis["analysis_start_au"] * AU_TIME_FS,
        analysis["analysis_stop_au"] * AU_TIME_FS,
        color="0.7",
        alpha=0.18,
        lw=0.0,
    )
    axes[0].set_ylabel("Pz (a.u.)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_fs, rt.electron_counts, lw=1.5, color="C2")
    axes[1].set_ylabel("electron count")
    axes[1].grid(alpha=0.25)

    axes[2].semilogy(
        analysis["harmonic_order"],
        np.maximum(analysis["accel_norm"], 1e-18),
        lw=1.4,
        label=f"{analysis['acceleration_source']} acceleration",
    )
    axes[2].semilogy(
        analysis["harmonic_order"],
        np.maximum(analysis["dipole_norm"], 1e-18),
        lw=1.1,
        alpha=0.75,
        label=r"$\omega^4|\mathrm{FFT}(P)|^2$",
    )
    axes[2].axvline(cutoff_harmonic, color="0.2", ls="--", lw=1.0, label="simple cutoff estimate")
    axes[2].set_xlim(0.0, float(max_harmonic))
    axes[2].set_ylim(1e-16, 2.0)
    axes[2].set_xlabel("harmonic order")
    axes[2].set_ylabel("normalized HHG yield")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_h2(args, nz):
    bond = float(args.bond)
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[
            [0.0, 0.0, -0.5 * bond],
            [0.0, 0.0, 0.5 * bond],
        ],
    )
    mol.build(
        Lz=float(args.lz),
        Nz=int(nz),
        M=int(args.m),
        verbose=bool(args.verbose),
        dvr_method=args.dvr_method,
    )
    return mol


def summarize_harmonics(harmonics, value_col=1):
    odd = [
        (int(row[0]), float(row[value_col]))
        for row in harmonics
        if int(row[0]) > 1 and int(row[0]) % 2 == 1
    ]
    even = [
        (int(row[0]), float(row[value_col]))
        for row in harmonics
        if int(row[0]) > 0 and int(row[0]) % 2 == 0
    ]
    return {
        "max_odd_harmonic_accel_gt_1e_minus_6": max((n for n, s in odd if s > 1e-6), default=None),
        "max_odd_harmonic_accel_gt_1e_minus_8": max((n for n, s in odd if s > 1e-8), default=None),
        "strongest_odd_harmonics_accel": [
            {"harmonic": int(n), "accel_strength": float(s)}
            for n, s in sorted(odd, key=lambda item: item[1], reverse=True)[:12]
        ],
        "strongest_even_harmonics_accel": [
            {"harmonic": int(n), "accel_strength": float(s)}
            for n, s in sorted(even, key=lambda item: item[1], reverse=True)[:8]
        ],
    }


def symmetry_diagnostics(harmonics, value_col=1):
    order = harmonics[:, 0].astype(int)
    accel = harmonics[:, value_col].astype(float)
    odd = (order > 1) & (order % 2 == 1)
    even = (order > 0) & (order % 2 == 0)
    max_odd = float(np.max(accel[odd])) if np.any(odd) else 0.0
    max_even = float(np.max(accel[even])) if np.any(even) else 0.0
    ratio = max_even / max_odd if max_odd > 0.0 else np.nan
    return {
        "max_even_harmonic_accel": max_even,
        "max_odd_harmonic_accel": max_odd,
        "max_even_over_max_odd": float(ratio),
        "even_harmonics_accel_gt_1e_minus_4": [
            int(n) for n, s in zip(order[even], accel[even]) if s > 1e-4
        ],
        "even_harmonics_accel_gt_1e_minus_6": [
            int(n) for n, s in zip(order[even], accel[even]) if s > 1e-6
        ],
    }


def run_case(args, nz):
    sym_tag = "hwsym" if args.symmetrize_half_wave else "raw"
    tag = (
        f"gdvr_h2_hhg_lz{_clean_float(args.lz)}_nz{int(nz)}_m{int(args.m)}_"
        f"dt{_clean_float(args.dt)}_e{_clean_float(args.field)}_"
        f"cap{_clean_float(args.cap_strength)}_w{_clean_float(args.cap_width)}_"
        f"prop{args.propagator}_acc{args.spectrum_acceleration}_"
        f"{args.pulse_shape}_{sym_tag}"
    )
    prefix = Path(args.outdir) / tag
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build] H2 GDVR Lz={args.lz:g} Nz={int(nz)} M={int(args.m)}")
    start = walltime.perf_counter()
    mol = build_h2(args, nz)
    build_seconds = walltime.perf_counter() - start

    print("[scf] RHF")
    start = walltime.perf_counter()
    mf = mol.RHF().run(
        newton=False,
        conv=args.scf_conv,
        max_iter=args.scf_max_iter,
        verbose=args.verbose,
    )
    scf_seconds = walltime.perf_counter() - start

    if args.newton_sweeps > 0:
        print(f"[newton] sweeps={args.newton_sweeps}")
        mf.newton(
            tol=args.scf_conv,
            sweeps=args.newton_sweeps,
            ridge=0.5,
            trust_step=1.0,
            trust_radius=2.0,
            scf_conv=args.scf_conv,
            scf_max_iter=args.scf_max_iter,
            verbose=args.verbose,
        )

    field = build_pulse(args)
    analysis_start, analysis_stop, analysis_window_name = analysis_bounds(
        field, args.analysis_window
    )
    nsteps = int(np.ceil(field.duration / float(args.dt)))
    cap = None
    if args.cap_strength > 0.0:
        cap = mol.cap(
            width=args.cap_width,
            strength=args.cap_strength,
            order=args.cap_order,
        )

    rt = RTTDHF(mf, interaction=mol.dipole_operator("z"), field=field, cap=cap)

    print(
        f"[rt] pulse={field.shape} cycles={field.cycles:g} dt={args.dt:g} au "
        f"nsteps={nsteps} T={field.duration * AU_TIME_FS:.3f} fs"
    )
    print(
        f"[spectrum] window={analysis_window_name} "
        f"{analysis_start * AU_TIME_FS:.3f}-{analysis_stop * AU_TIME_FS:.3f} fs "
        f"half_wave_sym={args.symmetrize_half_wave}"
    )
    start = walltime.perf_counter()
    rt.run(dt=args.dt, nsteps=nsteps, store_dm=False, method=args.propagator)
    rt_seconds = walltime.perf_counter() - start

    induced_full = rt.dipoles[:, 2] - float(rt.dipoles[0, 2])
    force_accel = rt.dipole_accelerations[:, 2]
    fd_accel_full = _finite_difference_acceleration(rt.times, rt.dipoles[:, 2])
    selected_accel_full = fd_accel_full
    spectrum_accel = None
    if args.spectrum_acceleration == "force":
        spectrum_accel = force_accel
        selected_accel_full = force_accel
    analysis = hhg_spectrum(
        rt.times,
        rt.dipoles[:, 2],
        args.omega,
        acceleration=spectrum_accel,
        acceleration_source=args.spectrum_acceleration,
        zero_pad=args.zero_pad,
        harmonic_window=args.harmonic_window,
        max_harmonic=args.max_harmonic,
        analysis_start=analysis_start,
        analysis_stop=analysis_stop,
        symmetrize_half_wave=args.symmetrize_half_wave,
    )

    write_csv(
        prefix.with_suffix(".trace.csv"),
        [
            "time_au",
            "time_fs",
            "field_z_au",
            "dipole_z_au",
            "induced_dipole_z_au",
            "dipole_accel_au",
            "force_accel_z_au",
            "finite_difference_accel_z_au",
            "electron_count",
            "field_free_energy_ha",
        ],
        [
            rt.times,
            rt.times * AU_TIME_FS,
            rt.fields[:, 2],
            rt.dipoles[:, 2],
            induced_full,
            selected_accel_full,
            force_accel,
            fd_accel_full,
            rt.electron_counts,
            rt.energies,
        ],
    )
    write_csv(
        prefix.with_suffix(".spectrum.csv"),
        [
            "omega_au",
            "harmonic_order",
            "energy_ev",
            "hhg_accel_norm",
            "hhg_finite_difference_accel_norm",
            "hhg_omega4_dipole_norm",
        ],
        [
            analysis["omega"],
            analysis["harmonic_order"],
            analysis["energy_ev"],
            analysis["accel_norm"],
            analysis["fd_accel_norm"],
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
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        time_au=rt.times,
        time_fs=rt.times * AU_TIME_FS,
        field_z_au=rt.fields[:, 2],
        dipole_z_au=rt.dipoles[:, 2],
        induced_dipole_z_au=induced_full,
        dipole_accel_au=selected_accel_full,
        force_accel_z_au=force_accel,
        finite_difference_accel_z_au=fd_accel_full,
        electron_count=rt.electron_counts,
        energy_ha=rt.energies,
        analysis_time_au=analysis["time_au"],
        analysis_time_fs=analysis["time_au"] * AU_TIME_FS,
        analysis_induced_dipole_z_au=analysis["induced_polarization"],
        analysis_dipole_accel_au=analysis["acceleration"],
        analysis_finite_difference_accel_z_au=analysis["finite_difference_acceleration"],
        omega_au=analysis["omega"],
        harmonic_order=analysis["harmonic_order"],
        energy_ev=analysis["energy_ev"],
        hhg_accel_norm=analysis["accel_norm"],
        hhg_finite_difference_accel_norm=analysis["fd_accel_norm"],
        hhg_omega4_dipole_norm=analysis["dipole_norm"],
        harmonic_table=analysis["harmonics"],
        z=mol.z,
        cap_diag=np.zeros(rt.size) if cap is None else np.diag(cap).real,
        dm_final=rt.dm,
    )

    up = float(args.field) ** 2 / (4.0 * float(args.omega) ** 2)
    cutoff_harmonic = (float(args.ip) + 3.17 * up) / float(args.omega)
    plot_case(
        prefix.with_suffix(".png"),
        rt,
        analysis,
        cutoff_harmonic,
        max_harmonic=args.max_harmonic,
        induced_polarization=induced_full,
    )

    active = np.where(np.abs(rt.fields[:, 2]) > max(np.max(np.abs(rt.fields[:, 2])) * 1e-3, 1e-12))[0]
    post_start = int(active[-1] + 1) if active.size and active[-1] + 1 < rt.energies.size else rt.energies.size - 1
    summary = {
        "prefix": str(prefix),
        "grid": {
            "Lz_bohr": float(args.lz),
            "Nz": int(nz),
            "M": int(args.m),
            "dz_bohr": float(mol.dz),
            "bond_bohr": float(args.bond),
        },
        "pulse": {
            "shape": str(field.shape),
            "cycles": float(args.cycles),
            "actual_cycles": float(field.cycles),
            "ramp_cycles": float(getattr(field, "ramp_cycles", 0.0)),
            "flat_cycles": float(getattr(field, "flat_cycles", 0.0)),
            "omega_au": float(args.omega),
            "E0_au": float(args.field),
            "peak_sampled_field_au": float(np.max(np.abs(rt.fields[:, 2]))),
            "peak_intensity_w_cm2_from_E0": float(FIELD_TO_INTENSITY_W_CM2 * float(args.field) ** 2),
            "duration_fs": float(rt.times[-1] * AU_TIME_FS),
            "dt_au": float(args.dt),
            "nsteps": int(nsteps),
        },
        "propagation": {
            "method": rt.propagation_method,
            "spectrum_acceleration": analysis["acceleration_source"],
            "analysis_window": str(analysis_window_name),
            "analysis_start_fs_requested": float(analysis_start * AU_TIME_FS),
            "analysis_stop_fs_requested": float(analysis_stop * AU_TIME_FS),
            "analysis_start_fs_actual": float(analysis["analysis_start_au"] * AU_TIME_FS),
            "analysis_stop_fs_actual": float(analysis["analysis_stop_au"] * AU_TIME_FS),
            "symmetrize_half_wave": bool(args.symmetrize_half_wave),
            "half_wave_projection": analysis["half_wave_projection"],
        },
        "cap": {
            "strength_ha": float(args.cap_strength),
            "width_bohr": float(args.cap_width),
            "order": int(args.cap_order),
            "max_w_ha": 0.0 if cap is None else float(np.max(np.diag(cap).real)),
        },
        "observables": {
            "initial_energy_ha": float(rt.energies[0]),
            "final_energy_ha": float(rt.energies[-1]),
            "postpulse_energy_drift_ha": float(np.max(np.abs(rt.energies[post_start:] - rt.energies[post_start]))),
            "initial_electron_count": float(rt.electron_counts[0]),
            "final_electron_count": float(rt.electron_counts[-1]),
            "absorbed_electrons": float(rt.electron_counts[0] - rt.electron_counts[-1]),
            "peak_abs_induced_polarization_z_au": float(np.max(np.abs(induced_full))),
            "time_peak_abs_induced_polarization_fs": float(rt.times[np.argmax(np.abs(induced_full))] * AU_TIME_FS),
            "peak_abs_analysis_induced_polarization_z_au": float(
                np.max(np.abs(analysis["induced_polarization"]))
            ),
        },
        "hhg": {
            "ponderomotive_energy_ha": float(up),
            "cutoff_estimate_harmonic_order": float(cutoff_harmonic),
            "harmonic_window": float(args.harmonic_window),
            **summarize_harmonics(analysis["harmonics"]),
            "symmetry": symmetry_diagnostics(analysis["harmonics"]),
            "symmetry_center_bin": symmetry_diagnostics(analysis["harmonics"], value_col=4),
        },
        "timing_seconds": {
            "build": float(build_seconds),
            "scf": float(scf_seconds),
            "rt": float(rt_seconds),
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
    with open(prefix.with_suffix(".summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "[done] Nz={Nz} Hmax>1e-6={h6} Hmax>1e-8={h8} "
        "even/odd={eo:.3e} absorbed={loss:.3e} plot={plot}".format(
            Nz=int(nz),
            h6=summary["hhg"]["max_odd_harmonic_accel_gt_1e_minus_6"],
            h8=summary["hhg"]["max_odd_harmonic_accel_gt_1e_minus_8"],
            eo=summary["hhg"]["symmetry_center_bin"]["max_even_over_max_odd"],
            loss=summary["observables"]["absorbed_electrons"],
            plot=prefix.with_suffix(".png"),
        )
    )
    return summary


def plot_convergence(path, summaries):
    selected = np.arange(3, 46, 2)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), dpi=180)
    for summary in summaries:
        table = np.genfromtxt(summary["files"]["harmonics_csv"], delimiter=",", names=True)
        values = []
        for harmonic in selected:
            row = table[np.asarray(table["harmonic_order"], dtype=int) == harmonic]
            values.append(float(row["max_accel_spectrum_norm_pm_window"][0]) if row.size else np.nan)
        label = (
            f"M={summary['grid']['M']}, Nz={summary['grid']['Nz']}, "
            f"dz={summary['grid']['dz_bohr']:.3f}, {summary.get('propagation', {}).get('method', 'density')}"
        )
        axes[0].semilogy(selected, np.maximum(values, 1e-18), marker="o", ms=3.2, lw=1.2, label=label)

        trace = np.genfromtxt(summary["files"]["trace_csv"], delimiter=",", names=True)
        axes[1].plot(trace["time_fs"], trace["electron_count"], lw=1.3, label=label)

    cutoff = summaries[0]["hhg"]["cutoff_estimate_harmonic_order"]
    axes[0].axvline(cutoff, color="0.25", ls="--", lw=1.0, label="cutoff estimate")
    axes[0].set_xlim(2, 46)
    axes[0].set_ylim(1e-8, 2.0)
    axes[0].set_xlabel("harmonic order")
    axes[0].set_ylabel("normalized acceleration yield")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("time (fs)")
    axes[1].set_ylabel("electron count")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=1.4)
    parser.add_argument("--lz", type=float, nargs="+", default=[10.0])
    parser.add_argument("--nz", type=int, nargs="+", default=[61])
    parser.add_argument("--m", type=int, nargs="+", default=[1])
    parser.add_argument("--dvr-method", choices=("sine", "exp", "sinc"), default="sine")
    parser.add_argument("--scf-conv", type=float, default=1e-8)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--newton-sweeps", type=int, default=0)
    parser.add_argument("--field", type=float, default=0.08)
    parser.add_argument("--omega", type=float, default=0.057)
    parser.add_argument("--cycles", type=float, default=8.0)
    parser.add_argument("--pulse-shape", choices=("flat-top", "sin2"), default="flat-top")
    parser.add_argument("--ramp-cycles", type=float, default=1.0)
    parser.add_argument("--flat-cycles", type=float, default=None)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--ip", type=float, default=0.57)
    parser.add_argument("--cap-strength", type=float, nargs="+", default=[0.01])
    parser.add_argument("--cap-width", type=float, nargs="+", default=[2.5])
    parser.add_argument("--cap-order", type=int, default=2)
    parser.add_argument("--propagator", choices=("density", "orbital"), default="orbital")
    parser.add_argument("--spectrum-acceleration", choices=("force", "finite-difference"), default="force")
    parser.add_argument("--analysis-window", choices=("auto", "full", "flat-top", "central"), default="auto")
    parser.add_argument("--symmetrize-half-wave", dest="symmetrize_half_wave", action="store_true", default=True)
    parser.add_argument("--no-symmetrize-half-wave", dest="symmetrize_half_wave", action="store_false")
    parser.add_argument("--zero-pad", type=int, default=8)
    parser.add_argument("--harmonic-window", type=float, default=0.2)
    parser.add_argument("--max-harmonic", type=int, default=80)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/gdvr_h2_hhg"))
    parser.add_argument("--tag", default="gdvr_h2_hhg_sweep")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    summaries = []
    for lz_value in args.lz:
        for cap_strength in args.cap_strength:
            for cap_width in args.cap_width:
                for m_value in args.m:
                    case_args = argparse.Namespace(**vars(args))
                    case_args.lz = float(lz_value)
                    case_args.cap_strength = float(cap_strength)
                    case_args.cap_width = float(cap_width)
                    case_args.m = int(m_value)
                    for nz in args.nz:
                        summaries.append(run_case(case_args, nz))
    aggregate = {
        "cases": summaries,
        "files": {},
    }
    if len(summaries) > 1:
        conv_plot = Path(args.outdir) / f"{args.tag}_convergence.png"
        plot_convergence(conv_plot, summaries)
        aggregate["files"]["convergence_plot_png"] = str(conv_plot)
        print(f"[convergence] {conv_plot}")
    aggregate_json = Path(args.outdir) / f"{args.tag}_summary.json"
    with open(aggregate_json, "w") as handle:
        json.dump(aggregate, handle, indent=2)
    print(f"[summary] {aggregate_json}")


if __name__ == "__main__":
    main()
