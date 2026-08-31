#!/usr/bin/env python3
"""Coarse ab initio APH wavepacket test for H+ + H2 reactive scattering."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import ExponentialDVR, LegendreDVR, SineDVR
from pyqed.ldr import keo
from pyqed.mps import FunctionalTT
from pyqed.qchem import CASCI, Molecule
from pyqed.units import au2ev, au2fs, proton_mass


def _casci_point(task):
    index, coordinates, basis = task
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    aph = keo.APH(("H", "H", "H"), (proton_mass,) * 3)
    mol = Molecule(
        atom=aph.geometry(coordinates),
        basis=basis,
        charge=1,
        spin=0,
        unit="bohr",
    )
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at APH point {coordinates}")
    mc = CASCI(mf, ncas=3, nelecas=2, verbose=0).run(nstates=1)
    return index, float(np.asarray(mc.e_tot).reshape(-1)[0])


def scan_pes(aph, dvrs, *, basis, workers, cache):
    axes = tuple(np.asarray(dvr.x) for dvr in dvrs)
    if cache.exists():
        data = np.load(cache)
        if (
            all(
                name in data
                and data[name].shape == axis.shape
                and np.allclose(data[name], axis)
                for name, axis in zip(("rho", "theta", "phi"), axes)
            )
            and str(data["basis"]) == basis
        ):
            print(f"[PES] loaded {cache}")
            return np.asarray(data["potential"])

    shape = tuple(len(axis) for axis in axes)
    potential = np.empty(shape)
    tasks = [
        (index, tuple(axes[axis][index[axis]] for axis in range(3)), basis)
        for index in np.ndindex(shape)
    ]
    start = time.perf_counter()
    if workers == 1:
        for count, task in enumerate(tasks, 1):
            index, energy = _casci_point(task)
            potential[index] = energy
            if count % 50 == 0 or count == len(tasks):
                print(f"[PES] {count}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_casci_point, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                index, energy = future.result()
                potential[index] = energy
                if count % 50 == 0 or count == len(tasks):
                    print(f"[PES] {count}/{len(tasks)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        rho=axes[0],
        theta=axes[1],
        phi=axes[2],
        basis=basis,
        potential=potential,
    )
    print(f"[PES] scan {time.perf_counter() - start:.2f} s; cached {cache}")
    return potential


def grid_geometry(aph, dvrs):
    shape = tuple(dvr.npts for dvr in dvrs)
    distances = np.empty((*shape, 3))
    bond = np.empty(shape)
    separation = np.empty(shape)
    for index in np.ndindex(shape):
        q = tuple(dvrs[axis].x[index[axis]] for axis in range(3))
        distances[index] = aph.pair_distances(q)
        x, y = aph.scaled_jacobi(q)
        bond[index] = np.linalg.norm(x) / np.sqrt(aph.mu_r / aph.mu)
        separation[index] = np.linalg.norm(y) / np.sqrt(aph.mu_R / aph.mu)
    return distances, bond, separation


def functional_pes(model_path, dvrs, distances, *, symmetry_order, cutoff):
    model = FunctionalTT.load(model_path)
    mesh = np.meshgrid(
        dvrs[0].x,
        dvrs[1].x,
        np.cos(int(symmetry_order) * np.asarray(dvrs[2].x)),
        indexing="ij",
    )
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    potential = model.predict(coordinates).reshape(distances.shape[:-1])
    screened = np.min(distances, axis=-1) < float(cutoff)
    potential[screened] = np.min(potential[~screened]) + 1.0
    return potential


def channel_packet(
    aph,
    dvrs,
    potential,
    distances,
    bond,
    *,
    collision_ev,
    rho_center=5.9,
    channel_states=3,
):
    """Build an incoming packet from a localized fixed-rho channel state."""
    radial = np.asarray(dvrs[0].x)
    rho_index = int(np.argmin(np.abs(radial - rho_center)))
    rho_reference = float(radial[rho_index])
    h_angular = aph.angular_hamiltonian(
        rho_reference,
        dvrs[1:],
        potential=potential[rho_index],
    )
    energies, vectors = np.linalg.eigh(h_angular)

    theta, phi = np.meshgrid(dvrs[1].x, dvrs[2].x, indexing="ij")
    dphi = np.angle(np.exp(1j * (phi - 0.5 * np.pi)))
    reactant = np.argmin(distances[rho_index], axis=-1) == 2
    target = (
        np.exp(-0.5 * ((bond[rho_index] - 1.40) / 0.20) ** 2)
        * np.exp(-0.5 * (dphi / 0.34) ** 2)
        * reactant
    )
    angular_weight = np.sqrt(
        np.asarray(dvrs[1].w)[:, None] * np.asarray(dvrs[2].w)[None, :]
    )
    target = (target * angular_weight).reshape(-1)
    target /= np.linalg.norm(target)

    count = min(int(channel_states), vectors.shape[1])
    channel_basis = vectors[:, :count]
    channel = channel_basis @ (channel_basis.conj().T @ target)
    if np.linalg.norm(channel) < 1.0e-10:
        raise RuntimeError(
            "low-energy channel subspace does not span the reactant target"
        )
    channel /= np.linalg.norm(channel)

    momentum = np.sqrt(2.0 * aph.mu * collision_ev / au2ev)
    radial_state = np.exp(
        -0.5 * ((radial - rho_reference) / 0.62) ** 2
        - 1j * momentum * (radial - rho_reference)
    )
    radial_state /= np.linalg.norm(radial_state)
    state = np.einsum("r,a->ra", radial_state, channel).reshape(-1)
    return state, rho_index, energies


def cap_field(dvrs, *, strength):
    rho = np.asarray(dvrs[0].x)
    lower_start = 2.05
    upper_start = 6.85
    cap = np.zeros_like(rho)
    lower = rho < lower_start
    upper = rho > upper_start
    cap[lower] = strength * ((lower_start - rho[lower]) / (lower_start - rho[0])) ** 3
    cap[upper] = strength * ((rho[upper] - upper_start) / (rho[-1] - upper_start)) ** 3
    return np.broadcast_to(cap[:, None, None], tuple(dvr.npts for dvr in dvrs))


def observables(states, distances, cap):
    density = np.abs(states.reshape(len(states), *distances.shape[:-1])) ** 2
    channels = np.argmin(distances, axis=-1)
    asymptotic = np.partition(distances, 1, axis=-1)[..., 1] > 1.7 * np.min(
        distances, axis=-1
    )
    populations = np.empty((len(states), 4))
    absorption = np.empty((len(states), 3))
    for time_index, rho in enumerate(density):
        for channel in range(3):
            mask = (channels == channel) & asymptotic
            populations[time_index, channel] = np.sum(rho[mask])
            absorption[time_index, channel] = 2.0 * np.sum(cap[mask] * rho[mask])
        populations[time_index, 3] = np.sum(rho) - np.sum(populations[time_index, :3])
    return density, populations, absorption


def _jacobi_density(values, separation, bond, separation_edges, bond_edges):
    histogram = np.histogram2d(
        separation.reshape(-1),
        bond.reshape(-1),
        bins=(separation_edges, bond_edges),
        weights=np.asarray(values).reshape(-1),
    )[0]
    return gaussian_filter(histogram, sigma=0.65).T


def _jacobi_minimum(values, separation, bond, separation_edges, bond_edges):
    shape = (len(bond_edges) - 1, len(separation_edges) - 1)
    field = np.full(shape, np.inf)
    separation_bin = np.searchsorted(separation_edges, separation, side="right") - 1
    bond_bin = np.searchsorted(bond_edges, bond, side="right") - 1
    valid = (
        (separation_bin >= 0)
        & (separation_bin < shape[1])
        & (bond_bin >= 0)
        & (bond_bin < shape[0])
    )
    np.minimum.at(
        field,
        (bond_bin[valid], separation_bin[valid]),
        np.asarray(values)[valid],
    )
    separation_centers = 0.5 * (separation_edges[1:] + separation_edges[:-1])
    bond_centers = 0.5 * (bond_edges[1:] + bond_edges[:-1])
    mesh = np.meshgrid(separation_centers, bond_centers)
    finite = np.isfinite(field)
    points = np.column_stack((mesh[0][finite], mesh[1][finite]))
    linear = griddata(points, field[finite], mesh, method="linear")
    nearest = griddata(points, field[finite], mesh, method="nearest")
    return gaussian_filter(np.where(np.isfinite(linear), linear, nearest), sigma=0.45)


def plot_results(
    times_fs,
    density,
    populations,
    yields,
    potential,
    bond,
    separation,
    outpath,
):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
        }
    )
    figure = plt.figure(figsize=(11.0, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.05, 1.0))
    ax_pes = figure.add_subplot(grid[0, :2])
    ax_pop = figure.add_subplot(grid[0, 2:])

    separation_edges = np.linspace(0.0, 1.02 * np.max(separation), 33)
    bond_edges = np.linspace(0.95 * np.min(bond), 1.02 * np.max(bond), 33)
    separation_centers = 0.5 * (separation_edges[1:] + separation_edges[:-1])
    bond_centers = 0.5 * (bond_edges[1:] + bond_edges[:-1])
    asymptote = float(np.sum(potential * density[0]) / np.sum(density[0]))
    energy = _jacobi_minimum(
        (potential - asymptote) * au2ev,
        separation,
        bond,
        separation_edges,
        bond_edges,
    )
    image = ax_pes.contourf(
        separation_centers,
        bond_centers,
        energy,
        levels=np.linspace(-3.0, 1.0, 21),
        cmap="RdYlBu_r",
        extend="both",
    )
    initial = _jacobi_density(
        density[0], separation, bond, separation_edges, bond_edges
    )
    initial /= max(float(np.max(initial)), np.finfo(float).tiny)
    ax_pes.contour(
        separation_centers,
        bond_centers,
        initial,
        levels=(0.1, 0.5),
        colors="black",
        linewidths=(0.8, 1.3),
    )
    figure.colorbar(
        image,
        ax=ax_pes,
        label="$V-V_{\\mathrm{in}}$ / eV",
        pad=0.02,
        extend="both",
    )
    ax_pes.set(
        xlabel="$R_{0,(12)}$ / bohr",
        ylabel="$r_{12}$ / bohr",
        xlim=(separation_edges[0], separation_edges[-1]),
        ylim=(bond_edges[0], bond_edges[-1]),
    )

    reactant = int(np.argmax(populations[0, :3]))
    products = np.sum(np.delete(populations[:, :3], reactant, axis=1), axis=1)
    norm = np.sum(density, axis=(1, 2, 3))
    loss = np.maximum(1.0 - norm, 0.0)
    line_reactant = ax_pop.plot(
        times_fs,
        populations[:, reactant],
        color="#0072B2",
        label="reactant",
    )[0]
    line_norm = ax_pop.plot(
        times_fs,
        norm,
        color="0.25",
        linestyle="--",
        label="norm",
    )[0]
    ax_pop.set(
        xlabel="time / fs",
        ylabel="reactant population",
        ylim=(max(0.0, 0.98 * np.min(populations[:, reactant])), 1.01),
    )
    ax_small = ax_pop.twinx()
    line_complex = ax_small.plot(
        times_fs,
        populations[:, 3],
        color="#D55E00",
        label="interaction region",
    )[0]
    line_products = ax_small.plot(
        times_fs,
        products,
        color="#009E73",
        label="products",
    )[0]
    line_loss = ax_small.plot(
        times_fs,
        loss,
        color="#CC79A7",
        linestyle=":",
        label="CAP loss",
    )[0]
    small_max = max(
        float(np.max(populations[:, 3])),
        float(np.max(products)),
        float(np.max(loss)),
        1.0e-4,
    )
    ax_small.set(ylabel="product / complex / loss", ylim=(0.0, 1.12 * small_max))
    lines = (line_reactant, line_norm, line_complex, line_products, line_loss)
    ax_pop.legend(lines, [line.get_label() for line in lines], loc="lower left", ncol=2)

    chosen = np.linspace(0, len(times_fs) - 1, 4, dtype=int)
    mesh = None
    snapshot_axes = []
    for column, index in enumerate(chosen):
        axis = figure.add_subplot(grid[1, column])
        snapshot_axes.append(axis)
        reduced = _jacobi_density(
            density[index], separation, bond, separation_edges, bond_edges
        )
        peak = max(float(np.max(reduced)), np.finfo(float).tiny)
        relative = np.clip(reduced / peak, 1.0e-4, 1.0)
        mesh = axis.pcolormesh(
            separation_edges,
            bond_edges,
            relative,
            shading="flat",
            cmap="magma",
            norm=LogNorm(vmin=1.0e-4, vmax=1.0),
            rasterized=True,
        )
        axis.set(
            xlabel="$R_{0,(12)}$ / bohr",
            xlim=(separation_edges[0], separation_edges[-1]),
            ylim=(bond_edges[0], bond_edges[-1]),
            title=f"{times_fs[index]:.1f} fs   $N={norm[index]:.3f}$",
        )
        if column == 0:
            axis.set_ylabel("$r_{12}$ / bohr")
        else:
            axis.tick_params(labelleft=False)
    figure.colorbar(
        mesh,
        ax=snapshot_axes,
        label="$P(R,r)/P_{\\max}(t)$",
        pad=0.015,
    )

    for label, axis in zip("abcdef", (ax_pes, ax_pop, *snapshot_axes)):
        axis.text(
            0.015,
            0.97,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color="white" if axis in snapshot_axes else "black",
        )
    outpath = Path(outpath)
    figure.savefig(outpath, dpi=350)
    figure.savefig(outpath.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rho", type=int, default=12)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--n-phi", type=int, default=18)
    parser.add_argument("--rho-min", type=float, default=1.75)
    parser.add_argument("--rho-max", type=float, default=7.75)
    parser.add_argument("--collision-ev", type=float, default=0.35)
    parser.add_argument("--channel-states", type=int, default=12)
    parser.add_argument("--tmax-fs", type=float, default=16.0)
    parser.add_argument("--snapshots", type=int, default=81)
    parser.add_argument("--cap-strength", type=float, default=0.03)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--functional-pes", type=Path)
    parser.add_argument("--symmetry-order", type=int, default=6)
    parser.add_argument("--short-range-cutoff", type=float, default=0.53)
    parser.add_argument(
        "--outdir", type=Path, default=Path("/private/tmp/hplus_h2_aph_scattering")
    )
    args = parser.parse_args()
    if args.n_phi < 6 or args.n_phi % 6:
        raise ValueError("n-phi must be a positive multiple of six")
    args.outdir.mkdir(parents=True, exist_ok=True)

    aph = keo.APH(("H", "H", "H"), (proton_mass,) * 3)
    dvrs = (
        SineDVR(args.rho_min, args.rho_max, args.n_rho, mass=aph.mu),
        LegendreDVR(0.0, np.pi / 2.0, args.n_theta),
        ExponentialDVR(npts=args.n_phi, L=2.0 * np.pi, x0=np.pi),
    )
    distances, bond, separation = grid_geometry(aph, dvrs)
    if args.functional_pes is None:
        potential = scan_pes(
            aph,
            dvrs,
            basis=args.basis,
            workers=args.workers,
            cache=args.outdir / "h3plus_casci_aph_pes.npz",
        )
    else:
        potential = functional_pes(
            args.functional_pes,
            dvrs,
            distances,
            symmetry_order=args.symmetry_order,
            cutoff=args.short_range_cutoff,
        )
        print(f"[PES] evaluated functional TT {args.functional_pes}")
    state0, channel_rho_index, channel_energies = channel_packet(
        aph,
        dvrs,
        potential,
        distances,
        bond,
        collision_ev=args.collision_ev,
        channel_states=args.channel_states,
    )
    print(
        "[channel] rho="
        f"{dvrs[0].x[channel_rho_index]:.3f}, "
        f"lowest gaps/eV={(channel_energies[:6] - channel_energies[0]) * au2ev}"
    )
    initial_density = np.abs(state0.reshape(*distances.shape[:-1])) ** 2
    initial_channels = np.argmin(distances, axis=-1)
    print(
        "[channel] initial arrangements="
        f"{[float(np.sum(initial_density[initial_channels == i])) for i in range(3)]}"
    )

    start = time.perf_counter()
    kinetic = aph.matrix(dvrs)
    print(
        f"[KEO] dimension={kinetic.shape[0]}, build={time.perf_counter() - start:.2f} s"
    )
    wall = np.min(potential) + 1.0
    regularized = np.minimum(potential, wall)
    cap = cap_field(dvrs, strength=args.cap_strength)
    shift = 0.5 * (np.min(regularized) + np.max(regularized))
    hamiltonian = kinetic + np.diag((regularized - shift - 1j * cap).reshape(-1))

    times_au = np.linspace(0.0, args.tmax_fs / au2fs, args.snapshots)
    start = time.perf_counter()
    states = expm_multiply(
        -1j * hamiltonian,
        state0,
        start=times_au[0],
        stop=times_au[-1],
        num=len(times_au),
        endpoint=True,
    )
    print(f"[dynamics] {time.perf_counter() - start:.2f} s")
    density, populations, absorption = observables(states, distances, cap)
    dt = np.diff(times_au)
    yields = np.zeros_like(absorption)
    yields[1:] = np.cumsum(
        0.5 * (absorption[1:] + absorption[:-1]) * dt[:, None], axis=0
    )
    times_fs = times_au * au2fs

    result = args.outdir / "hplus_h2_aph_scattering.npz"
    figure = args.outdir / "hplus_h2_aph_scattering.png"
    np.savez(
        result,
        times_fs=times_fs,
        density=density,
        populations=populations,
        cap_yields=yields,
        norms=np.sum(np.abs(states) ** 2, axis=1),
        potential=potential,
        distances=distances,
        bond=bond,
        separation=separation,
        rho=dvrs[0].x,
        theta=dvrs[1].x,
        phi=dvrs[2].x,
        channel_rho=dvrs[0].x[channel_rho_index],
        channel_energies=channel_energies,
    )
    plot_results(
        times_fs,
        density,
        populations,
        yields,
        potential,
        bond,
        separation,
        figure,
    )
    print(f"[final] norm={np.linalg.norm(states[-1]) ** 2:.6f}")
    print(f"[final] arrangements={populations[-1, :3]}")
    print(f"[final] CAP yields={yields[-1]}")
    print(f"[output] {result}")
    print(f"[figure] {figure}")


if __name__ == "__main__":
    main()
