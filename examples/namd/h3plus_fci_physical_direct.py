#!/usr/bin/env python3
"""Direct-product reference for physically prepared 3D H3+ LDR dynamics."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.qchem import Molecule
from pyqed.units import au2fs


preparation_output = Path("/private/tmp/h3plus_fci_augccpvdz_physical")
output = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")
database = output / "electronic.sqlite"
preparation = json.loads(
    (preparation_output / "h3plus_fci_initial_state.json").read_text()
)
equilibrium = float(preparation["equilibrium_bond_bohr"])
covariance = np.asarray(preparation["probability_covariance_bohr2"])
bounds = ((-0.60, 0.60), (-0.70, 0.70), (-0.70, 0.70))


def geometry(q):
    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    stretch = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.diag(jnp.asarray((1.0, -1.0)))
    )
    shear = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.asarray(((0.0, 1.0), (1.0, 0.0)))
    )
    qs, qx, qy = q
    return (equilibrium + qs) * triangle + qx * stretch + qy * shear


def cap(grid, strength=0.08):
    widths = np.sqrt(np.diag(covariance))
    profiles = []
    total = np.zeros(grid.shape)
    for axis, (coordinate, width) in enumerate(zip(grid.x, widths)):
        wall = float(np.max(np.abs(coordinate)))
        start = min(3.5 * float(width), 0.82 * wall)
        scaled = np.clip((np.abs(coordinate) - start) / (wall - start), 0.0, 1.0)
        profile = strength * scaled**4
        profiles.append(profile)
        shape = [1] * grid.ndim
        shape[axis] = len(profile)
        total += profile.reshape(shape)
    return total, profiles


def edge_probability(states):
    density = np.sum(np.abs(states) ** 2, axis=-1)
    mask = np.zeros(density.shape[1:], dtype=bool)
    for axis in range(3):
        for index in (0, -1):
            section = [slice(None)] * 3
            section[axis] = index
            mask[tuple(section)] = True
    return np.sum(density[:, mask], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npts", type=int, default=11)
    args = parser.parse_args()
    if args.npts < 7:
        parser.error("--npts must be at least 7")
    tag = f"h3plus_fci_physical_direct_{args.npts}"
    output.mkdir(parents=True, exist_ok=True)
    grid = DVR.from_axes(
        tuple(SineDVR(lower, upper, args.npts) for lower, upper in bounds),
        names=("Qs", "Qx", "Qy"),
    )
    coord = Coord(to_cartesian=geometry, bounds=bounds)
    mol = Molecule(
        atom=list(zip(("H", "H", "H"), np.asarray(geometry((0.0, 0.0, 0.0))))),
        charge=1,
        spin=0,
        unit="bohr",
        basis="aug-cc-pvdz",
    ).build(eri="dense")
    mf = mol.RHF().run()
    mc = mol.casci(
        mol.nao,
        2,
        nstates=6,
        ms2=0,
        multiplicity=1,
        mf=mf,
    ).run(nstates=6)
    root_s2 = np.asarray([mc.spin_square(root) for root in range(6)])
    if np.max(np.abs(root_s2)) > 1.0e-7:
        raise RuntimeError(f"non-singlet CASCI root detected: S^2={root_s2}")
    fit = AbInitioFit(
        mc,
        coord=coord,
        states=(1, 2),
        nroots=6,
        database=database,
        workers=6,
        progress=False,
    )
    nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=mol)
    started = perf_counter()
    ldr = fit.direct_product(
        grid,
        keo=nuclear_keo,
        workers=6,
        progress=False,
        energy_shift=fit.energy_shift,
    )
    build_seconds = perf_counter() - started
    link_singular_values = np.asarray(
        [np.linalg.svd(block, compute_uv=False) for block in ldr.links.values()]
    )
    minimum_link_singular_value = float(np.min(link_singular_values[:, -1]))
    fraction_links_below_0_9 = float(
        np.mean(link_singular_values[:, -1] < 0.9)
    )
    if minimum_link_singular_value < 0.9:
        raise RuntimeError(
            "refusing dynamics: a nearest-neighbor singlet link leaves the "
            "selected electronic subspace"
        )

    mesh = np.meshgrid(*grid.x, indexing="ij")
    coordinates = np.stack(mesh, axis=-1)
    exponent = np.einsum(
        "...i,ij,...j->...", coordinates, np.linalg.inv(covariance), coordinates
    )
    envelope = np.exp(-0.25 * exponent)
    anchor = tuple(int(np.argmin(np.abs(axis))) for axis in grid.x)
    packet = ldr.wavepacket(envelope, state=1, anchor=anchor)
    absorber, profiles = cap(grid)
    dt_fs = 0.02
    tmax_fs = 5.0
    nout = 5
    dt = dt_fs / au2fs
    started = perf_counter()
    ldr.run(
        packet,
        dt=dt,
        nsteps=round(tmax_fs / dt_fs),
        nout=nout,
        matrix_free=True,
        absorber=absorber,
    )
    propagation_seconds = perf_counter() - started

    order = np.argsort(ldr.energies, axis=-1)
    populations = np.sum(
        np.take_along_axis(np.abs(ldr.states) ** 2, order[None, ...], axis=-1),
        axis=(1, 2, 3),
    )
    time_fs = ldr.times * au2fs
    edge = edge_probability(ldr.states)
    report = {
        "electronic_reference": "spin-pure singlet full CI (2e, 27o)/aug-cc-pVDZ",
        "electronic_multiplicity": 1,
        "electronic_roots_solved": 6,
        "selected_singlet_roots": [1, 2],
        "initial_state": "harmonic S0 nuclear ground state vertically promoted to S2",
        "equilibrium_bond_bohr": equilibrium,
        "probability_widths_bohr": np.sqrt(np.diag(covariance)).tolist(),
        "grid": list(grid.shape),
        "bounds_bohr": [list(value) for value in bounds],
        "keo": "J=0 Podolsky with pseudopotential",
        "links": "raw nonunitary overlaps with LPA",
        "subspace_diagnostics": {
            "links": int(len(link_singular_values)),
            "minimum_link_singular_value": minimum_link_singular_value,
            "fraction_links_below_0_9": fraction_links_below_0_9,
            "maximum_projector_loss": float(
                np.max(1.0 - link_singular_values[:, -1] ** 2)
            ),
        },
        "cap": {
            "axes": [0, 1, 2],
            "strength_hartree": 0.08,
            "starts_bohr": [
                float(np.min(np.abs(axis[profile > 0.0])))
                if np.any(profile > 0.0) else None
                for axis, profile in zip(grid.x, profiles)
            ],
        },
        "tmax_fs": tmax_fs,
        "dt_fs": dt_fs,
        "final_populations": populations[-1].tolist(),
        "final_survival_probability": float(ldr.norm[-1]),
        "maximum_edge_probability": float(np.max(edge)),
        "final_edge_probability": float(edge[-1]),
        "database_writes": int(ldr.direct_product_info["database_writes"]),
        "database_hits": int(ldr.direct_product_info["database_hits"]),
        "database": str(database),
        "build_seconds": build_seconds,
        "propagation_seconds": propagation_seconds,
    }
    (output / f"{tag}_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez(
        output / f"{tag}.npz",
        time_fs=time_fs,
        populations=populations,
        norms=ldr.norm,
        edge_probability=edge,
        states=ldr.states,
        axes=np.asarray(grid.x),
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, panels = plt.subplots(1, 2, figsize=(6.5, 2.8), constrained_layout=True)
    panels[0].plot(time_fs, populations[:, 0], color="#0072B2", label=r"$S_1$")
    panels[0].plot(time_fs, populations[:, 1], color="#D55E00", label=r"$S_2$")
    panels[0].set(
        xlabel="time (fs)", ylabel="adiabatic population",
        title="(a) Vertically promoted packet", ylim=(-0.02, 1.02),
    )
    panels[0].legend(frameon=False)
    panels[1].plot(time_fs, ldr.norm, color="0.15", label="survival")
    panels[1].plot(time_fs, edge, "--", color="#009E73", label="outer DVR layer")
    panels[1].set(
        xlabel="time (fs)", ylabel="probability",
        title="(b) Boundary diagnostics", ylim=(-0.02, 1.02),
    )
    panels[1].legend(frameon=False)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(direction="out")
    population_path = output / f"{tag}_population"
    figure.savefig(population_path.with_suffix(".pdf"))
    figure.savefig(population_path.with_suffix(".png"), dpi=360)
    plt.close(figure)

    snapshot_times = np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    indices = [int(np.argmin(np.abs(time_fs - value))) for value in snapshot_times]
    snapshot_figure, snapshot_panels = plt.subplots(
        2, 3, figsize=(8.0, 5.1), constrained_layout=True, sharex=True, sharey=True
    )
    for panel, index in zip(snapshot_panels.flat, indices):
        density = np.sum(np.abs(ldr.states[index]) ** 2, axis=(0, 3))
        maximum = max(float(np.max(density)), np.finfo(float).tiny)
        image = panel.pcolormesh(
            grid.x[1], grid.x[2], (density / maximum).T,
            shading="auto", cmap="magma", vmin=0.0, vmax=1.0,
        )
        panel.set_title(fr"$t={time_fs[index]:.1f}$ fs, $N={ldr.norm[index]:.3f}$")
        panel.set_aspect("equal")
    for panel in snapshot_panels[-1]:
        panel.set_xlabel(r"$Q_x$ (bohr)")
    for panel in snapshot_panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ (bohr)")
    snapshot_figure.colorbar(
        image, ax=snapshot_panels,
        label=r"$\rho(Q_x,Q_y;t)/\rho_{\max}(t)$", shrink=0.82,
    )
    snapshot_path = output / f"{tag}_snapshots"
    snapshot_figure.savefig(snapshot_path.with_suffix(".pdf"))
    snapshot_figure.savefig(snapshot_path.with_suffix(".png"), dpi=360)
    plt.close(snapshot_figure)

    print(json.dumps(report, indent=2), flush=True)
    print(population_path.with_suffix(".png"), flush=True)
    print(snapshot_path.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
