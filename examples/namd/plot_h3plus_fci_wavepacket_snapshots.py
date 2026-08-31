"""Replay cached H3+ direct LDR dynamics and plot 0--5 fs snapshots."""

from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.qchem import Molecule
from pyqed.units import au2fs


stem = "h3plus_fci_augccpvdz_3d_s3_mace_ftt_vs_direct_7x7x7_20fs"
output = Path("/private/tmp") / stem
bounds = ((-0.05, 0.05), (-0.36, 0.36), (-0.36, 0.36))


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
    return (1.45 + qs) * triangle + qx * stretch + qy * shear


def main():
    grid = DVR.from_axes(
        tuple(
            SineDVR(lower, upper, count)
            for (lower, upper), count in zip(bounds, (7, 23, 23))
        ),
        names=("Qs", "Qx", "Qy"),
    )
    coord = Coord(to_cartesian=geometry, bounds=bounds)
    mol = Molecule(
        atom=list(
            zip(("H", "H", "H"), np.asarray(geometry((0.0, 0.0, 0.0))))
        ),
        charge=1,
        spin=0,
        unit="bohr",
        basis="aug-cc-pvdz",
    ).build(eri="dense")
    mf = mol.RHF().run()
    mc = mol.casci(mol.nao, 2, nstates=3, mf=mf).run(nstates=3)
    sampler = AbInitioFit(
        mc,
        coord=coord,
        states=(1, 2),
        fit_options={"degrees": (6, 22, 22), "rank": 64},
        database=output / "electronic.sqlite",
    )
    direct = sampler.direct_product(
        grid,
        keo=keo.podolsky().bind(coord, grid=grid, molecule=mol),
        workers=6,
        progress=False,
    )

    center = np.asarray((0.0, -0.08, 0.0))
    sigma = np.asarray((0.025, 0.03, 0.03))
    momentum = np.asarray((0.0, 0.70, 0.0))
    factors = tuple(
        np.where(
            np.abs(axis - value) <= 3.0 * width,
            np.exp(-0.25 * ((axis - value) / width) ** 2 + 1j * kick * axis),
            0.0,
        )
        for axis, value, width, kick in zip(grid.x, center, sigma, momentum)
    )
    packet = direct.wavepacket(
        np.einsum("i,j,k->ijk", *factors),
        state=1,
        anchor=tuple(
            int(np.argmin(np.abs(axis - value)))
            for axis, value in zip(grid.x, center)
        ),
        support_threshold=1.0e-12,
    )
    direct.run(
        packet,
        dt=0.02 / au2fs,
        nsteps=250,
        nout=50,
        matrix_free=False,
    )

    density = np.sum(np.abs(direct.states) ** 2, axis=(1, 4))
    density /= np.sum(density, axis=(1, 2), keepdims=True)
    display_density = density / np.max(density, axis=(1, 2), keepdims=True)
    times = direct.times * au2fs

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "savefig.bbox": "tight",
        }
    )
    figure, panels = plt.subplots(
        2,
        3,
        figsize=(8.4, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for panel, values, time in zip(panels.flat, display_density, times):
        image = panel.pcolormesh(
            grid.x[1],
            grid.x[2],
            values.T,
            shading="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        panel.set_title(fr"$t={time:.0f}$ fs")
        panel.set_aspect("equal")
    for panel in panels[-1]:
        panel.set_xlabel(r"$Q_x$ (bohr)")
    for panel in panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ (bohr)")
    colorbar = figure.colorbar(image, ax=panels, shrink=0.88, pad=0.02)
    colorbar.set_label(r"Relative density $\rho(Q_x,Q_y;t)/\rho_{\max}(t)$")

    figure_path = output / f"{stem}_wavepacket_snapshots_wide_7x23x23_0_5fs"
    figure.savefig(figure_path.with_suffix(".pdf"))
    figure.savefig(figure_path.with_suffix(".png"), dpi=350)
    np.savez(
        figure_path.with_suffix(".npz"),
        times_fs=times,
        qx=grid.x[1],
        qy=grid.x[2],
        density=density,
        norms=direct.norm,
    )
    print(f"maximum norm error: {np.max(np.abs(direct.norm - 1.0)):.3e}")
    print(figure_path.with_suffix(".pdf"))
    print(figure_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
