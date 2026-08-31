#!/usr/bin/env python3
"""Native ab initio 2D H3+ test of Coord and the automatic Podolsky KEO."""

from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.namd import TNLDR
from pyqed.qchem import Molecule
from pyqed.units import amu_to_au, au2ev, au2fs, au2wavenumber


theta = np.deg2rad(70.0)


def geometry(q):
    """Fixed-angle H3+ geometry from the two bond lengths, in bohr."""
    r1, r2 = q
    xyz = jnp.stack(
        (
            jnp.stack((r1, 0.0 * r1, 0.0 * r1)),
            jnp.zeros(3),
            jnp.stack((r2 * jnp.cos(theta), r2 * jnp.sin(theta), 0.0 * r2)),
        )
    )
    return xyz - jnp.mean(xyz, axis=0)


def finite_difference_metric(q, masses, step=1.0e-5):
    """Independent constrained inverse metric from Cartesian differences."""
    q = np.asarray(q, dtype=float)
    xyz = np.asarray(geometry(q), dtype=float)
    tangent = []
    for axis in range(q.size):
        displacement = np.zeros_like(q)
        displacement[axis] = step
        tangent.append(
            (
                np.asarray(geometry(q + displacement))
                - np.asarray(geometry(q - displacement))
            )
            / (2.0 * step)
        )
    vibration = np.stack(tangent, axis=2)
    rotation = np.stack(
        [np.cross(direction, xyz) for direction in np.eye(3)],
        axis=2,
    )
    translation = np.broadcast_to(np.eye(3), (xyz.shape[0], 3, 3))
    vectors = np.concatenate((vibration, rotation, translation), axis=2)
    weighted = vectors * np.sqrt(masses)[:, None, None]
    covariant = weighted.reshape(-1, q.size + 6).T @ weighted.reshape(
        -1, q.size + 6
    )
    return np.linalg.inv(covariant)[: q.size, : q.size]


axes = (
    SineDVR(1.0, 2.4, 3),
    SineDVR(1.0, 2.4, 3),
)
dynamics_grid = DVR.from_axes(axes, names=("r1", "r2"))

reference_geometry = np.asarray(geometry(np.asarray((1.7, 1.7))))
mol = Molecule(
    atom=list(zip(("H", "H", "H"), reference_geometry)),
    charge=1,
    spin=0,
    unit="bohr",
    basis="sto-3g",
)
mol.build(eri="dense")
mf = mol.RHF().run()
mc = mol.casci(3, 2, nstates=3, mf=mf).run(nstates=3)

started = perf_counter()
coord = Coord(
    to_cartesian=geometry,
    bounds=((1.0, 2.4), (1.0, 2.4)),
)
fit = AbInitioFit(
    mc,
    coord=coord,
    states=(0, 1),
    progress=True,
).build()
driver = TNLDR(
    fit,
    grid=dynamics_grid,
    coord=coord,
    keo=keo.podolsky(),
).build()
build_seconds = perf_counter() - started

r1, r2 = np.meshgrid(*dynamics_grid.x, indexing="ij")
center = np.asarray((dynamics_grid.x[0][1], dynamics_grid.x[1][1]))
envelope = np.exp(-18.0 * ((r1 - center[0]) ** 2 + (r2 - center[1]) ** 2))
packet = np.zeros((*dynamics_grid.shape, 2), dtype=complex)
packet[..., 1] = envelope

started = perf_counter()
driver.run(
    driver.state(packet, max_rank=18),
    dt=0.02 / au2fs,
    steps=50,
    interval=1,
    max_bond=32,
    e_ops=driver.projectors(),
    progress=False,
)
propagation_seconds = perf_counter() - started

metric = np.asarray(driver.keo.metric)
pseudopotential = np.asarray(driver.keo.pseudopotential)
atomic_masses = np.asarray(mol.atom_mass_list()) * amu_to_au
metric_reference = np.empty_like(metric)
for index in np.ndindex(dynamics_grid.shape):
    point = np.asarray(
        [dynamics_grid.x[axis][index[axis]] for axis in range(dynamics_grid.ndim)]
    )
    metric_reference[index] = finite_difference_metric(point, atomic_masses)
metric_error = float(np.max(np.abs(metric - metric_reference)))
metric_eigenvalues = np.linalg.eigvalsh(metric)
metric_condition = float(
    np.max(metric_eigenvalues[..., -1] / metric_eigenvalues[..., 0])
)
hamiltonian = driver.hamiltonian.to_dense()
hermiticity_error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
potential_matrix = driver.potential.to_dense().reshape(
    dynamics_grid.size, 2, dynamics_grid.size, 2
)
local_potential = np.asarray(
    [potential_matrix[index, :, index, :] for index in range(dynamics_grid.size)]
).reshape(*dynamics_grid.shape, 2, 2)
energies = np.linalg.eigvalsh(local_potential)

figure, panels = plt.subplots(2, 3, figsize=(11.2, 6.7), constrained_layout=True)
surfaces = (
    ((energies[..., 0] - np.min(energies[..., 0])) * au2ev, "$E_0$ (eV)"),
    ((energies[..., 1] - np.min(energies[..., 0])) * au2ev, "$E_1$ (eV)"),
    (metric[..., 0, 1], "$G^{12}$"),
    (pseudopotential * au2wavenumber, "$V_{ps}$ (cm$^{-1}$)"),
)
for panel, (values, title) in zip(panels.flat[:4], surfaces):
    image = panel.pcolormesh(r1, r2, values, shading="nearest", cmap="viridis")
    panel.set(xlabel="$r_1$ (bohr)", ylabel="$r_2$ (bohr)", title=title)
    figure.colorbar(image, ax=panel)

time_fs = driver.times * au2fs
panels[1, 1].plot(time_fs, driver.populations[:, 0], label="$S_0$")
panels[1, 1].plot(time_fs, driver.populations[:, 1], label="$S_1$")
panels[1, 1].set(xlabel="time (fs)", ylabel="population", title="Fitted-core TNLDR")
panels[1, 1].legend()
panels[1, 2].semilogy(
    time_fs,
    np.maximum(np.abs(driver.norms - 1.0), 1.0e-16),
)
panels[1, 2].set(
    xlabel="time (fs)",
    ylabel="$|\\langle\\Psi|\\Psi\\rangle-1|$",
    title="Norm error",
)

figure_path = Path("h3plus_2d_coord_podolsky.png").resolve()
data_path = Path("h3plus_2d_coord_podolsky.npz").resolve()
figure.savefig(figure_path, dpi=180)
np.savez(
    data_path,
    r1=dynamics_grid.x[0],
    r2=dynamics_grid.x[1],
    energies=energies,
    metric=metric,
    pseudopotential=pseudopotential,
    times=driver.times,
    populations=driver.populations,
    norms=driver.norms,
)

print(f"build: {build_seconds:.2f} s")
print(f"propagation: {propagation_seconds:.2f} s")
print(f"metric condition <= {metric_condition:.3e}")
print(f"finite-difference metric error = {metric_error:.3e}")
print(f"Hamiltonian Hermiticity error = {hermiticity_error:.3e}")
print(f"final norm = {driver.norms[-1]:.12f}")
print(
    "electronic cache: "
    f"hits={driver.database_info['hits']} "
    f"writes={driver.database_info['writes']} "
    f"records={driver.database_info['records']}"
)
print(f"database: {driver.database_path}")
print(f"figure: {figure_path}")
print(f"data: {data_path}")
