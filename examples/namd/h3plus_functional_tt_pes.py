#!/usr/bin/env python3
"""Fit the cached H3+ APH PES with a symmetry-adapted functional TT."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import ExponentialDVR, LegendreDVR, SineDVR
from pyqed.ldr import keo
from pyqed.mps import FunctionalTT
from pyqed.mps.cross import tt_cross, tt_value
from pyqed.units import au2ev, proton_mass


def symmetry_coordinate(phi, order):
    """Invariant angular coordinate for identical H nuclei."""
    return np.cos(int(order) * np.asarray(phi))


def unique_training_data(rho, theta, phi, potential, order):
    """Average symmetry-equivalent angular samples into scattered FT data."""
    angular = symmetry_coordinate(phi, order)
    labels, inverse = np.unique(np.round(angular, decimals=12), return_inverse=True)
    reduced = np.empty((len(rho), len(theta), len(labels)))
    for index in range(len(labels)):
        reduced[:, :, index] = np.mean(potential[:, :, inverse == index], axis=2)
    mesh = np.meshgrid(rho, theta, labels, indexing="ij")
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    return coordinates, reduced.reshape(-1), labels


def full_coordinates(rho, theta, phi, order):
    mesh = np.meshgrid(rho, theta, symmetry_coordinate(phi, order), indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1)


def short_range_mask(rho, theta, phi, cutoff):
    """Identify geometries assigned to the analytic repulsive wall."""
    aph = keo.APH(("H", "H", "H"), (proton_mass,) * 3)
    mask = np.empty((len(rho), len(theta), len(phi)), dtype=bool)
    for index in np.ndindex(mask.shape):
        coordinates = (rho[index[0]], theta[index[1]], phi[index[2]])
        mask[index] = np.min(aph.pair_distances(coordinates)) < float(cutoff)
    return mask


def retained_training_points(coordinates, cutoff, symmetry_order):
    """Apply the short-range screen to symmetry-reduced FT coordinates."""
    aph = keo.APH(("H", "H", "H"), (proton_mass,) * 3)
    retained = np.empty(len(coordinates), dtype=bool)
    for index, (rho, theta, angular) in enumerate(coordinates):
        phi = np.arccos(np.clip(angular, -1.0, 1.0)) / int(symmetry_order)
        retained[index] = np.min(aph.pair_distances((rho, theta, phi))) >= float(cutoff)
    return retained


def canonical_angular_index(index, size, symmetry_order):
    period = int(size) // int(symmetry_order)
    if period * int(symmetry_order) != int(size):
        raise ValueError("angular grid size must be divisible by the symmetry order")
    reduced = int(index) % period
    return min(reduced, (-reduced) % period)


def cross_fit(potential, screened, rank, sweeps, validation, symmetry_order, seed):
    """Emulate TT-cross QC sampling with H3+ symmetry-aware caching."""
    sampled_geometries = set()

    def oracle(index):
        angular = canonical_angular_index(index[2], potential.shape[2], symmetry_order)
        if not screened[index]:
            sampled_geometries.add((int(index[0]), int(index[1]), angular))
        return potential[index]

    cores, info = tt_cross(
        potential.shape,
        oracle,
        max_rank=rank,
        sweeps=sweeps,
        rtol=1.0e-8,
        validation=validation,
        seed=seed,
        start_rank=1,
        kick_rank=2,
    )
    fitted = np.empty_like(potential)
    for index in np.ndindex(potential.shape):
        fitted[index] = tt_value(cores, index)
    return fitted, cores, info, sampled_geometries


def sampled_grid_mask(shape, samples, symmetry_order):
    mask = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        canonical = canonical_angular_index(index[2], shape[2], symmetry_order)
        mask[index] = (index[0], index[1], canonical) in samples
    return mask


def channel_audit(rho, theta, phi, exact, fitted, rho_target):
    """Compare low fixed-rho APH angular channels for exact and fitted PESs."""
    aph = keo.APH(("H", "H", "H"), (proton_mass,) * 3)
    dvrs = (
        SineDVR(1.75, 7.75, len(rho), mass=aph.mu),
        LegendreDVR(0.0, 0.5 * np.pi, len(theta)),
        ExponentialDVR(npts=len(phi), L=2.0 * np.pi, x0=np.pi),
    )
    for name, expected, dvr in zip(("rho", "theta", "phi"), (rho, theta, phi), dvrs):
        if not np.allclose(expected, dvr.x):
            raise ValueError(f"cached {name} axis does not match the APH DVR")
    rho_index = int(np.argmin(np.abs(rho - float(rho_target))))
    angular_keo = aph.angular_hamiltonian(
        rho[rho_index], dvrs[1:], np.zeros((len(theta), len(phi)))
    )
    exact_energy, exact_vectors = np.linalg.eigh(
        angular_keo + np.diag(exact[rho_index].reshape(-1))
    )
    fitted_energy, fitted_vectors = np.linalg.eigh(
        angular_keo + np.diag(fitted[rho_index].reshape(-1))
    )
    states = min(12, len(exact_energy))
    exact_gaps = exact_energy[:states] - exact_energy[0]
    fitted_gaps = fitted_energy[:states] - fitted_energy[0]
    cross = exact_vectors[:, :states].conj().T @ fitted_vectors[:, :states]
    return {
        "rho": float(rho[rho_index]),
        "exact_energy": exact_energy[:states],
        "fitted_energy": fitted_energy[:states],
        "exact_gaps": exact_gaps,
        "fitted_gaps": fitted_gaps,
        "matched_overlaps": np.max(np.abs(cross), axis=1),
        "subspace_min_singular": float(np.linalg.svd(cross, compute_uv=False)[-1]),
    }


def plot_comparison(rho, exact, fitted, theta_index, outpath, title):
    exact_slice = exact[:, theta_index, :]
    fitted_slice = fitted[:, theta_index, :]
    error = fitted_slice - exact_slice
    vmin = min(float(np.min(exact_slice)), float(np.min(fitted_slice)))
    vmax = max(float(np.max(exact_slice)), float(np.max(fitted_slice)))
    extent = (float(rho[0]), float(rho[-1]), 0.0, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    images = (
        axes[0].imshow(
            exact_slice.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        ),
        axes[1].imshow(
            fitted_slice.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        ),
        axes[2].imshow(
            (error * au2ev * 1000.0).T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-np.max(np.abs(error)) * au2ev * 1000.0,
            vmax=np.max(np.abs(error)) * au2ev * 1000.0,
        ),
    )
    axes[0].set_title("CASCI")
    axes[1].set_title(title)
    axes[2].set_title("Error (meV)")
    for axis in axes:
        axis.set_xlabel(r"$\rho$ (bohr)")
        axis.set_ylabel(r"$\phi/\pi$")
    fig.colorbar(images[0], ax=axes[:2], label="Energy (hartree)", shrink=0.88)
    fig.colorbar(images[2], ax=axes[2], shrink=0.88)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "/private/tmp/hplus_h2_aph_scattering_fine/h3plus_casci_aph_pes.npz"
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("/private/tmp/h3plus_functional_tt")
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--cross-rank", type=int, default=8)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-validation", type=int, default=32)
    parser.add_argument("--refine-sweeps", type=int, default=150)
    parser.add_argument("--refine-rtol", type=float, default=1.0e-4)
    parser.add_argument("--symmetry-order", type=int, default=6)
    parser.add_argument("--wall-height", type=float, default=1.0)
    parser.add_argument("--short-range-cutoff", type=float, default=0.53)
    parser.add_argument("--channel-rho", type=float, default=5.9)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    with np.load(args.cache) as data:
        rho = np.asarray(data["rho"])
        theta = np.asarray(data["theta"])
        phi = np.asarray(data["phi"])
        raw = np.asarray(data["potential"])
    wall = float(np.min(raw) + args.wall_height)
    potential = np.minimum(raw, wall)
    screened = short_range_mask(rho, theta, phi, args.short_range_cutoff)
    potential[screened] = wall
    coordinates, energies, angular = unique_training_data(
        rho, theta, phi, potential, args.symmetry_order
    )
    retained = retained_training_points(
        coordinates, args.short_range_cutoff, args.symmetry_order
    )
    model = FunctionalTT(
        bases=("chebyshev", "legendre", "chebyshev"),
        degrees=(len(rho) - 1, len(theta) - 1, len(angular) - 1),
        rank=args.rank,
        bounds=((1.75, 7.75), (0.0, 0.5 * np.pi), (-1.0, 1.0)),
        regularization=1.0e-12,
        sweeps=args.sweeps,
        rtol=1.0e-9,
        local_rtol=1.0e-11,
        patience=5,
        random_state=args.seed,
    ).fit(coordinates[retained], energies[retained])
    fitted = model.predict(
        full_coordinates(rho, theta, phi, args.symmetry_order)
    ).reshape(potential.shape)
    fitted[screened] = wall
    error = fitted - potential
    accessible = ~screened
    coefficient_count = int(sum(core.size for core in model.cores) + 2)
    channel = channel_audit(rho, theta, phi, potential, fitted, args.channel_rho)
    cross, cross_cores, cross_info, cross_samples = cross_fit(
        potential,
        screened,
        args.cross_rank,
        args.cross_sweeps,
        args.cross_validation,
        args.symmetry_order,
        args.seed,
    )
    cross_qchem_points = len(cross_samples)
    cross_sampled = sampled_grid_mask(
        potential.shape, cross_samples, args.symmetry_order
    )
    held_out = accessible & ~cross_sampled
    cross_error = cross - potential
    cross_channel = channel_audit(rho, theta, phi, potential, cross, args.channel_rho)

    selected = sorted(cross_samples)
    selected_coordinates = np.asarray(
        [
            (
                rho[i],
                theta[j],
                symmetry_coordinate(phi[k], args.symmetry_order),
            )
            for i, j, k in selected
        ]
    )
    selected_energies = np.asarray([potential[index] for index in selected])
    hybrid = FunctionalTT(
        bases=("chebyshev", "legendre", "chebyshev"),
        degrees=(len(rho) - 1, len(theta) - 1, len(angular) - 1),
        rank=args.cross_rank,
        bounds=((1.75, 7.75), (0.0, 0.5 * np.pi), (-1.0, 1.0)),
        random_state=args.seed,
    ).fit_from_cross(
        (rho, theta, symmetry_coordinate(phi, args.symmetry_order)),
        cross_cores,
        selected_coordinates,
        selected_energies,
        sweeps=args.refine_sweeps,
        rtol=args.refine_rtol,
        patience=3,
    )
    hybrid_fitted = hybrid.predict(
        full_coordinates(rho, theta, phi, args.symmetry_order)
    ).reshape(potential.shape)
    hybrid_fitted[screened] = wall
    hybrid_error = hybrid_fitted - potential
    hybrid_channel = channel_audit(
        rho, theta, phi, potential, hybrid_fitted, args.channel_rho
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    model_path = args.outdir / "h3plus_functional_tt.npz"
    hybrid_model_path = args.outdir / "h3plus_hybrid_functional_tt.npz"
    result_path = args.outdir / "h3plus_functional_tt_audit.npz"
    figure_path = args.outdir / "h3plus_functional_tt.png"
    hybrid_figure_path = args.outdir / "h3plus_hybrid_functional_tt.png"
    model.save(model_path)
    hybrid.save(hybrid_model_path)
    np.savez(
        result_path,
        rho=rho,
        theta=theta,
        phi=phi,
        exact=potential,
        fitted=fitted,
        error=error,
        wall=wall,
        short_range_cutoff=args.short_range_cutoff,
        screened=screened,
        ranks=model.ranks_,
        coefficient_count=coefficient_count,
        symmetry_unique_points=len(energies),
        independent_qchem_points=np.count_nonzero(retained),
        channel_rho=channel["rho"],
        channel_exact_energy=channel["exact_energy"],
        channel_fitted_energy=channel["fitted_energy"],
        channel_exact_gaps=channel["exact_gaps"],
        channel_fitted_gaps=channel["fitted_gaps"],
        channel_matched_overlaps=channel["matched_overlaps"],
        channel_subspace_min_singular=channel["subspace_min_singular"],
        cross_fitted=cross,
        cross_error=cross_error,
        cross_ranks=tuple([1] + [core.shape[2] for core in cross_cores]),
        cross_scalar_queries=cross_info["samples"],
        cross_qchem_points=cross_qchem_points,
        cross_sampled=cross_sampled,
        cross_samples=np.asarray(sorted(cross_samples), dtype=int),
        cross_channel_exact_energy=cross_channel["exact_energy"],
        cross_channel_fitted_energy=cross_channel["fitted_energy"],
        cross_channel_exact_gaps=cross_channel["exact_gaps"],
        cross_channel_fitted_gaps=cross_channel["fitted_gaps"],
        cross_channel_subspace_min_singular=cross_channel["subspace_min_singular"],
        hybrid_fitted=hybrid_fitted,
        hybrid_error=hybrid_error,
        hybrid_ranks=hybrid.ranks_,
        hybrid_refine_sweeps=hybrid.refinement["sweeps"],
        hybrid_refine_converged=hybrid.refinement["converged"],
        hybrid_refine_message=hybrid.refinement["message"],
        hybrid_refine_train_error=np.asarray(
            [entry["train_error"] for entry in hybrid.history]
        ),
        hybrid_refine_relative_improvement=np.asarray(
            [entry["relative_improvement"] for entry in hybrid.history]
        ),
        hybrid_channel_exact_energy=hybrid_channel["exact_energy"],
        hybrid_channel_fitted_energy=hybrid_channel["fitted_energy"],
        hybrid_channel_exact_gaps=hybrid_channel["exact_gaps"],
        hybrid_channel_fitted_gaps=hybrid_channel["fitted_gaps"],
        hybrid_channel_subspace_min_singular=hybrid_channel["subspace_min_singular"],
    )
    plot_comparison(
        rho,
        potential,
        fitted,
        len(theta) // 2,
        figure_path,
        "Functional TT (ALS)",
    )
    plot_comparison(
        rho,
        potential,
        hybrid_fitted,
        len(theta) // 2,
        hybrid_figure_path,
        "Cross + functional refinement",
    )

    print(f"full APH grid:              {potential.size}")
    print(f"symmetry-unique grid points:{len(energies):>7}")
    print(f"retained QC points:         {np.count_nonzero(retained)}")
    print(f"screened full-grid points:  {np.count_nonzero(screened)}")
    print(f"functional-TT coefficients: {coefficient_count}")
    print(f"TT ranks:                   {model.ranks_}")
    print(f"RMSE:                       {np.sqrt(np.mean(error**2)):.6e} Eh")
    print(f"MAE:                        {np.mean(np.abs(error)):.6e} Eh")
    print(f"maximum error:              {np.max(np.abs(error)):.6e} Eh")
    print(
        f"accessible-region RMSE:     {np.sqrt(np.mean(error[accessible] ** 2)):.6e} Eh"
    )
    gap_error = np.abs(channel["fitted_gaps"] - channel["exact_gaps"])
    energy_error = np.abs(channel["fitted_energy"] - channel["exact_energy"])
    print(f"channel rho:                {channel['rho']:.6f} bohr")
    print(f"ground-channel error:       {energy_error[0] * au2ev * 1000:.6e} meV")
    print(f"max channel-gap error:      {np.max(gap_error) * au2ev * 1000:.6e} meV")
    print(f"channel subspace overlap:   {channel['subspace_min_singular']:.10f}")
    cross_gap_error = np.abs(cross_channel["fitted_gaps"] - cross_channel["exact_gaps"])
    cross_energy_error = np.abs(
        cross_channel["fitted_energy"] - cross_channel["exact_energy"]
    )
    print("\nTT-cross control")
    print(f"scalar oracle queries:       {cross_info['samples']}")
    print(f"symmetry-unique QC points:   {cross_qchem_points}")
    print(f"TT ranks:                    {cross_info['ranks']}")
    print(f"RMSE:                        {np.sqrt(np.mean(cross_error**2)):.6e} Eh")
    print(
        f"ground-channel error:        {cross_energy_error[0] * au2ev * 1000:.6e} meV"
    )
    print(
        f"max channel-gap error:       {np.max(cross_gap_error) * au2ev * 1000:.6e} meV"
    )
    print(f"channel subspace overlap:    {cross_channel['subspace_min_singular']:.10f}")
    hybrid_gap_error = np.abs(
        hybrid_channel["fitted_gaps"] - hybrid_channel["exact_gaps"]
    )
    hybrid_energy_error = np.abs(
        hybrid_channel["fitted_energy"] - hybrid_channel["exact_energy"]
    )
    print("\nTT-cross + continuous functional-core refinement")
    print(f"selected QC points:          {cross_qchem_points}")
    print(f"TT ranks:                    {hybrid.ranks_}")
    print(f"canonical ALS sweeps:        {hybrid.refinement['sweeps']}")
    print(f"optimization converged:      {hybrid.refinement['converged']}")
    print(f"initial fit error:           {hybrid.refinement['initial_error']:.6e}")
    print(f"final fit error:             {hybrid.refinement['final_error']:.6e}")
    print(f"RMSE:                        {np.sqrt(np.mean(hybrid_error**2)):.6e} Eh")
    print(
        "accessible-region RMSE:      "
        f"{np.sqrt(np.mean(hybrid_error[accessible] ** 2)):.6e} Eh"
    )
    print(
        "held-out-region RMSE:        "
        f"{np.sqrt(np.mean(hybrid_error[held_out] ** 2)):.6e} Eh"
    )
    print(f"held-out full-grid points:   {np.count_nonzero(held_out)}")
    print(
        f"ground-channel error:        {hybrid_energy_error[0] * au2ev * 1000:.6e} meV"
    )
    print(
        f"max channel-gap error:       {np.max(hybrid_gap_error) * au2ev * 1000:.6e} meV"
    )
    print(
        f"channel subspace overlap:    {hybrid_channel['subspace_min_singular']:.10f}"
    )
    print(f"model:  {model_path}")
    print(f"hybrid: {hybrid_model_path}")
    print(f"audit:  {result_path}")
    print(f"figure: {figure_path}")
    print(f"hybrid figure: {hybrid_figure_path}")


if __name__ == "__main__":
    main()
