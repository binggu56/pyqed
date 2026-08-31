#!/usr/bin/env python3
"""Three-coordinate overlap-only LDR model of phenol photodissociation.

The coordinates are OH stretch, COH bend, and periodic CCOH torsion.  The
electronic Hamiltonian is the repository's published three-state
stretch/torsion model plus the configurable reduced bend extension in
``Phenol3D``.  This is a numerical LDR benchmark, not a new ab initio PES.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import ExponentialDVR, SineDVR
from pyqed.models.phenol import Phenol3D
from pyqed.units import au2angstrom, au2fs, wavenum2au


def build_model(nr=49, nbend=9, ntorsion=9, rmax_angstrom=6.0):
    if ntorsion < 3 or ntorsion % 2 == 0:
        raise ValueError("ntorsion must be an odd integer of at least 3.")

    r_bounds = np.array([0.65, rmax_angstrom]) / au2angstrom
    bend_bounds = np.deg2rad([75.0, 140.0])
    radial_mass = Phenol3D(
        [Phenol3D.r_eq],
        [Phenol3D.theta_eq],
        [0.0],
    ).radial_mass
    probe = Phenol3D([Phenol3D.r_eq], [Phenol3D.theta_eq], [0.0])
    r_dvr = SineDVR(*r_bounds, nr, mass=radial_mass)
    bend_dvr = SineDVR(*bend_bounds, nbend, mass=probe.bend_inertia)
    torsion_dvr = ExponentialDVR(
        (ntorsion - 1) // 2,
        L=2.0 * np.pi,
        x0=np.pi / ntorsion,
        mass=probe.torsional_inertia,
    )
    model = Phenol3D(r_dvr.x, bend_dvr.x, torsion_dvr.x)
    return model, (r_dvr, bend_dvr, torsion_dvr)


def wrapped_angle(phi):
    return np.angle(np.exp(1j * phi))


def vertical_packet(model, frames, bright_diabatic=2):
    r, bend, torsion = np.meshgrid(
        model.r,
        model.bend,
        model.torsion,
        indexing="ij",
    )
    radial = np.exp(-18.0 * (r - model.r_eq) ** 2)
    bend_alpha = model.bend_inertia * 1200.0 * wavenum2au
    bending = np.exp(-0.5 * bend_alpha * (bend - model.theta_eq) ** 2)
    torsion_alpha = model.torsional_inertia * 500.0 * wavenum2au
    torsional = np.exp(-0.5 * torsion_alpha * wrapped_angle(torsion) ** 2)
    envelope = radial * bending * torsional

    # U_g^dagger |pi pi*> expresses one fixed diabatic bright state in each
    # local adiabatic frame without choosing eigenvector phases by hand.
    electronic = frames[..., bright_diabatic, :].conj()
    psi = envelope[..., None] * electronic
    return (psi / np.linalg.norm(psi)).reshape(-1)


def observables(states, model, frames, dissociation_angstrom=3.0):
    shape = model.shape
    psi_ad = states.reshape(len(states), *shape, model.nstates)
    psi_diab = np.einsum("...da,t...a->t...d", frames, psi_ad, optimize=True)
    density = np.sum(np.abs(psi_ad) ** 2, axis=-1)
    adiabatic_populations = np.sum(np.abs(psi_ad) ** 2, axis=(1, 2, 3))
    diabatic_populations = np.sum(np.abs(psi_diab) ** 2, axis=(1, 2, 3))

    r_mean = np.einsum("trbp,r->t", density, model.r, optimize=True) * au2angstrom
    r2_mean = np.einsum("trbp,r->t", density, model.r**2, optimize=True) * au2angstrom**2
    r_std = np.sqrt(np.maximum(0.0, r2_mean - r_mean**2))
    bend_mean = np.rad2deg(np.einsum("trbp,b->t", density, model.bend, optimize=True))
    bend2_mean = np.einsum("trbp,b->t", density, model.bend**2, optimize=True)
    bend_std = np.rad2deg(
        np.sqrt(np.maximum(0.0, bend2_mean - np.deg2rad(bend_mean) ** 2))
    )
    cos_phi = np.einsum("trbp,p->t", density, np.cos(model.torsion), optimize=True)
    sin_phi = np.einsum("trbp,p->t", density, np.sin(model.torsion), optimize=True)
    torsion_mean = np.rad2deg(np.arctan2(sin_phi, cos_phi))
    resultant = np.clip(np.sqrt(cos_phi**2 + sin_phi**2), 1.0e-15, 1.0)
    torsion_std = np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))
    dissociation = density[:, model.r * au2angstrom >= dissociation_angstrom].sum(axis=(1, 2, 3))
    radial_density = density.sum(axis=(2, 3))
    return {
        "adiabatic_populations": adiabatic_populations,
        "diabatic_populations": diabatic_populations,
        "r_mean_angstrom": r_mean,
        "r_std_angstrom": r_std,
        "bend_mean_deg": bend_mean,
        "bend_std_deg": bend_std,
        "torsion_mean_deg": torsion_mean,
        "torsion_std_deg": torsion_std,
        "dissociation": dissociation,
        "radial_density": radial_density,
    }


def representation_error(model, dvrs, adiabatic_bundle):
    diabatic = model.hamiltonian(dvrs, representation="diabatic")
    frames = adiabatic_bundle["frames"].reshape(-1, model.nstates, model.nstates)
    rng = np.random.default_rng(7)
    psi_ad = rng.normal(size=(frames.shape[0], model.nstates))
    psi_ad = psi_ad + 1j * rng.normal(size=psi_ad.shape)
    psi_ad /= np.linalg.norm(psi_ad)
    psi_diab = np.einsum("gda,ga->gd", frames, psi_ad, optimize=True)
    lhs = diabatic["hamiltonian"] @ psi_diab.reshape(-1)
    rhs_ad = adiabatic_bundle["hamiltonian"] @ psi_ad.reshape(-1)
    rhs = np.einsum(
        "gda,ga->gd",
        frames,
        rhs_ad.reshape(frames.shape[0], model.nstates),
        optimize=True,
    ).reshape(-1)
    return np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)


def plot_result(times_fs, model, bundle, obs, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    labels = ("S0", "pi sigma*", "pi pi*")
    colors = ("0.25", "#d06434", "#2673b8")
    for state, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(times_fs, obs["diabatic_populations"][:, state], color=color, label=label)
    axes[0, 0].plot(times_fs, obs["dissociation"], color="#27845d", ls="--", label="R > 3 A")
    axes[0, 0].set(ylabel="population", ylim=(-0.02, 1.02))
    axes[0, 0].legend(frameon=False, ncol=2)

    axes[0, 1].plot(times_fs, obs["r_mean_angstrom"], color="#2673b8")
    axes[0, 1].fill_between(
        times_fs,
        obs["r_mean_angstrom"] - obs["r_std_angstrom"],
        obs["r_mean_angstrom"] + obs["r_std_angstrom"],
        color="#2673b8",
        alpha=0.16,
        linewidth=0,
    )
    axes[0, 1].set(ylabel="<R_OH> / A")
    axes[1, 0].plot(times_fs, obs["bend_mean_deg"], color="#27845d")
    axes[1, 0].fill_between(
        times_fs,
        obs["bend_mean_deg"] - obs["bend_std_deg"],
        obs["bend_mean_deg"] + obs["bend_std_deg"],
        color="#27845d",
        alpha=0.16,
        linewidth=0,
    )
    axes[1, 0].set(xlabel="time / fs", ylabel="COH bend / degree")
    torsion_axis = axes[1, 0].twinx()
    torsion_axis.plot(times_fs, obs["torsion_std_deg"], color="#d06434")
    torsion_axis.set_ylabel("CCOH torsion width / degree", color="#d06434")
    torsion_axis.tick_params(axis="y", colors="#d06434")

    density = obs["radial_density"]
    image = axes[1, 1].pcolormesh(
        times_fs,
        model.r * au2angstrom,
        density.T,
        shading="auto",
        cmap="magma",
    )
    axes[1, 1].axhline(3.0, color="white", lw=0.8, ls="--")
    axes[1, 1].set(xlabel="time / fs", ylabel="R_OH / A")
    fig.colorbar(image, ax=axes[1, 1], label="radial probability")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def run(args):
    model, dvrs = build_model(args.nr, args.nbend, args.ntorsion, args.rmax_angstrom)
    bundle = model.hamiltonian(dvrs, representation="adiabatic")
    psi0 = vertical_packet(model, bundle["frames"], args.bright_diabatic)
    times_fs = np.linspace(0.0, args.tmax_fs, args.nsnapshots)
    states = expm_multiply(
        -1j * bundle["hamiltonian"],
        psi0,
        start=0.0,
        stop=float(args.tmax_fs / au2fs),
        num=args.nsnapshots,
        traceA=-1j * bundle["hamiltonian"].diagonal().sum(),
    )
    obs = observables(states, model, bundle["frames"], args.dissociation_angstrom)
    return model, dvrs, bundle, times_fs, states, obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr", type=int, default=49)
    parser.add_argument("--nbend", type=int, default=9)
    parser.add_argument("--ntorsion", type=int, default=9)
    parser.add_argument("--rmax-angstrom", type=float, default=6.0)
    parser.add_argument("--tmax-fs", type=float, default=100.0)
    parser.add_argument("--nsnapshots", type=int, default=201)
    parser.add_argument("--bright-diabatic", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--dissociation-angstrom", type=float, default=3.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/phenol_3d_ldr"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    model, dvrs, bundle, times_fs, states, obs = run(args)
    error = np.nan if args.skip_validation else representation_error(model, dvrs, bundle)

    figure = args.outdir / "phenol_3d_ldr.png"
    data = args.outdir / "phenol_3d_ldr.npz"
    plot_result(times_fs, model, bundle, obs, figure)
    np.savez_compressed(
        data,
        times_fs=times_fs,
        r_bohr=model.r,
        bend_rad=model.bend,
        torsion_rad=model.torsion,
        states=states,
        energies=bundle["energies"],
        representation_error=error,
        **obs,
    )

    print(f"[grid] {model.shape}; vibronic dimension={bundle['hamiltonian'].shape[0]}")
    print(f"[Hamiltonian] nnz={bundle['hamiltonian'].nnz}; Hermiticity="
          f"{np.linalg.norm((bundle['hamiltonian'] - bundle['hamiltonian'].getH()).data):.3e}")
    print(f"[representation error] {error:.3e}")
    print("[final diabatic populations]", np.array2string(obs["diabatic_populations"][-1], precision=6))
    print(f"[final <R_OH>] {obs['r_mean_angstrom'][-1]:.6f} A")
    print(f"[final dissociation] {obs['dissociation'][-1]:.6f}")
    print(f"[figure] {figure}")
    print(f"[data] {data}")


if __name__ == "__main__":
    main()
