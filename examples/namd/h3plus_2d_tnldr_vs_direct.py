#!/usr/bin/env python3
"""Compare TNLDR with direct LDR at the H3+ S1/S2 intersection."""

import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.ldr.overlap import synchronize_link_gauge
from pyqed.namd import TNLDR
from pyqed.qchem import Molecule
from pyqed.units import au2fs


output = Path("/private/tmp/h3plus_s1s2_branching_tnldr_vs_direct_9x9_10fs")
output.mkdir(parents=True, exist_ok=True)
tmax_fs = 10.0
dt_fs = 0.02
bond_length = 1.65
breathing_coordinate = -0.20
seam_offset = 0.015


def geometry(q):
    """Rectilinear H3+ branching plane at fixed breathing coordinate."""
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
    qx, qy = q
    return (
        (bond_length + breathing_coordinate) * triangle
        + (seam_offset + qx) * stretch
        + qy * shear
    )


grid = DVR.from_axes(
    (SineDVR(-0.20, 0.20, 9), SineDVR(-0.20, 0.20, 9)),
    names=("Qx", "Qy"),
)
coord = Coord(to_cartesian=geometry, bounds=((-0.20, 0.20), (-0.20, 0.20)))
reference_geometry = np.asarray(geometry((0.0, 0.0)))
mol = Molecule(
    atom=list(zip(("H", "H", "H"), reference_geometry)),
    charge=1,
    spin=0,
    unit="bohr",
    basis="sto-3g",
).build(eri="dense")
mf = mol.RHF().run()
mc = mol.casci(3, 2, nstates=3, mf=mf).run(nstates=3)

started = perf_counter()
fit = AbInitioFit(
    mc,
    coord=coord,
    states=(1, 2),
    fit_options={"degrees": 16},
    progress=True,
).build()
tnldr = TNLDR(fit, grid=grid, coord=coord, keo=keo.podolsky()).build()
direct = fit.direct_product(
    grid,
    keo=keo.podolsky(),
    workers=1,
    progress=True,
)
build_seconds = perf_counter() - started

direct_tnldr = TNLDR.from_ldr(
    direct,
    overlap_method="dense",
    operator_rank=None,
).build()
coordinates = np.stack(
    np.meshgrid(*grid.x, indexing="ij"), axis=-1
).reshape(-1, coord.ndim)
fitted_blocks = np.asarray(fit.energy.predict(coordinates))
fitted_eigenvalues, fitted_eigenvectors = np.linalg.eigh(fitted_blocks)
tracked_to_energy = np.argsort(
    np.argsort(direct.energies.reshape(-1, direct.nstates), axis=-1),
    axis=-1,
)
frame_blocks = np.take_along_axis(
    fitted_eigenvectors,
    tracked_to_energy[:, None, :],
    axis=2,
)
frame = np.zeros((direct.size, direct.size), dtype=complex)
for point, block in enumerate(frame_blocks):
    indices = slice(point * direct.nstates, (point + 1) * direct.nstates)
    frame[indices, indices] = block
direct_hamiltonian = direct_tnldr.hamiltonian.to_dense()
fitted_hamiltonian = tnldr.hamiltonian.to_dense()
transformed_direct_hamiltonian = frame @ direct_hamiltonian @ frame.conj().T
hamiltonian_relative_error = np.linalg.norm(
    fitted_hamiltonian - transformed_direct_hamiltonian
) / np.linalg.norm(transformed_direct_hamiltonian)
spectral_error = np.max(
    np.abs(
        np.linalg.eigvalsh(fitted_hamiltonian)
        - np.linalg.eigvalsh(direct_hamiltonian)
    )
)
feature_values = np.asarray(fit.feature.predict(coordinates))
points = tuple(np.ndindex(grid.shape))
point_ids = {point: index for index, point in enumerate(points)}
pairs = []
exact_links = []
fitted_links = []
for (axis, left), exact_link in sorted(direct.links.items()):
    right = list(left)
    right[axis] += 1
    right = tuple(right)
    pairs.append((left, right))
    exact_links.append(exact_link)
    fitted_links.append(
        feature_values[point_ids[left]].conj().T
        @ feature_values[point_ids[right]]
    )
exact_links = np.asarray(exact_links)
fitted_links = np.asarray(fitted_links)
exact_singular_values = np.linalg.svd(exact_links, compute_uv=False)
fitted_singular_values = np.linalg.svd(fitted_links, compute_uv=False)
link_singular_value_error = np.abs(
    fitted_singular_values - exact_singular_values
)
gauge_anchor = tuple(size // 2 for size in grid.shape)
_exact_gauges, synchronized_exact_links = synchronize_link_gauge(
    points, pairs, exact_links, gauge_anchor
)
_fitted_gauges, synchronized_fitted_links = synchronize_link_gauge(
    points, pairs, fitted_links, gauge_anchor
)
synchronized_link_relative_error = np.linalg.norm(
    synchronized_fitted_links - synchronized_exact_links
) / np.linalg.norm(synchronized_exact_links)

center = np.asarray((-0.12, 0.0))
sigma = np.asarray((0.04, 0.04))
momentum = np.asarray((0.70, 0.0))
factors = tuple(
    np.where(
        np.abs(axis - value) <= 3.0 * width,
        np.exp(-0.25 * ((axis - value) / width) ** 2 + 1j * kick * axis),
        0.0,
    )
    for axis, value, width, kick in zip(grid.x, center, sigma, momentum)
)
envelope = np.multiply.outer(*factors)
packet_anchor = tuple(
    int(np.argmin(np.abs(axis - value)))
    for axis, value in zip(grid.x, center)
)
direct_packet = direct.wavepacket(
    envelope,
    state=1,
    anchor=packet_anchor,
    support_threshold=1.0e-12,
)
tn_packet, projector_1, state_info = tnldr.matched_state(
    factors,
    state=1,
    anchor=packet_anchor,
    max_bond=32,
    projector_rank=24,
    projector_validation=256,
)
projector_0, projector_info = tnldr.adiabatic_projector(
    0,
    max_rank=24,
    validation=256,
)

dt = dt_fs / au2fs
nsteps = round(tmax_fs / dt_fs)
started = perf_counter()
direct.run(direct_packet, dt=dt, nsteps=nsteps, matrix_free=False)
direct_seconds = perf_counter() - started
direct_order = np.argsort(direct.energies, axis=-1)
direct_populations = np.sum(
    np.take_along_axis(
        np.abs(direct.states) ** 2,
        direct_order[None, ...],
        axis=-1,
    ),
    axis=(1, 2),
)

started = perf_counter()
tnldr.run(
    tn_packet,
    dt=dt,
    steps=nsteps,
    interval=1,
    max_bond=64,
    integrator="tdvp2",
    e_ops=(projector_0, projector_1),
    progress=False,
)
tn_seconds = perf_counter() - started

time_fs = direct.times * au2fs
population_error = np.abs(tnldr.populations - direct_populations)
adaptive_validation = next(
    (
        record["validation"]
        for record in reversed(fit.info["history"])
        if "validation" in record
    ),
    None,
)
summary = {
    "grid": list(grid.shape),
    "tmax_fs": tmax_fs,
    "dt_fs": dt_fs,
    "steps": nsteps,
    "electronic_states": [1, 2],
    "initial_physical_state": 2,
    "initial_center": center.tolist(),
    "initial_sigma": sigma.tolist(),
    "initial_momentum": momentum.tolist(),
    "initial_anchor": list(packet_anchor),
    "seam_qx": -seam_offset,
    "direct_dimension": direct.size,
    "adaptive_geometries": len(fit.info["points"]),
    "adaptive_converged": bool(fit.info["converged"]),
    "adaptive_stop_reason": fit.info["stop_reason"],
    "adaptive_validation": adaptive_validation,
    "fit_degrees": 16,
    "fit_feature_rank": int(fit.info["feature_rank"]),
    "fit_gauge": fit.config["gauge"],
    "unitarize_links": fit.config["unitarize_links"],
    "direct_geometries": int(direct.direct_product_info["geometries"]),
    "direct_database_hits": int(direct.direct_product_info["database_hits"]),
    "direct_database_writes": int(direct.direct_product_info["database_writes"]),
    "database_records": int(direct.database_info["records"]),
    "max_population_error": float(np.max(population_error)),
    "final_population_error": population_error[-1].tolist(),
    "max_direct_norm_error": float(np.max(np.abs(direct.norm - 1.0))),
    "max_tnldr_norm_error": float(np.max(np.abs(tnldr.norms - 1.0))),
    "build_seconds": build_seconds,
    "direct_seconds": direct_seconds,
    "tnldr_seconds": tn_seconds,
    "matched_state_validation_error": float(state_info["validation_error"]),
    "projector_validation_error": float(projector_info["validation_error"]),
    "hamiltonian_relative_error": float(hamiltonian_relative_error),
    "max_spectral_error_hartree": float(spectral_error),
    "max_fitted_pes_error_hartree": float(
        np.max(
            np.abs(
                fitted_eigenvalues
                - np.sort(direct.energies.reshape(-1, direct.nstates), axis=-1)
            )
        )
    ),
    "rms_link_singular_value_error": float(
        np.sqrt(np.mean(link_singular_value_error**2))
    ),
    "max_link_singular_value_error": float(np.max(link_singular_value_error)),
    "synchronized_link_relative_error": float(
        synchronized_link_relative_error
    ),
    "database": str(direct.database_path),
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    }
)
figure, panels = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
colors = ("#0072B2", "#D55E00")
for state, color in enumerate(colors):
    physical = state + 1
    panels[0].plot(time_fs, direct_populations[:, state], color=color, label=f"Direct $S_{physical}$")
    panels[0].plot(time_fs, tnldr.populations[:, state], "--", color=color, label=f"TNLDR $S_{physical}$")
    panels[1].semilogy(
        time_fs,
        np.maximum(population_error[:, state], 1.0e-16),
        color=color,
        label=f"$S_{physical}$",
    )
panels[0].set(xlabel="Time (fs)", ylabel="Adiabatic population", ylim=(-0.025, 1.025))
panels[0].legend(ncol=2, frameon=False)
panels[1].set(xlabel="Time (fs)", ylabel="Absolute population error")
panels[1].legend(frameon=False)
for label, panel in zip(("a", "b"), panels):
    panel.grid(alpha=0.2, linewidth=0.6)
    panel.text(-0.16, 1.04, label, transform=panel.transAxes, fontweight="bold")

figure_path = output / "h3plus_s1s2_branching_tnldr_vs_direct_9x9_10fs"
figure.savefig(figure_path.with_suffix(".pdf"))
figure.savefig(figure_path.with_suffix(".png"), dpi=300)
np.savez(
    output / "h3plus_s1s2_branching_tnldr_vs_direct_9x9_10fs.npz",
    times=direct.times,
    direct_populations=direct_populations,
    tnldr_populations=tnldr.populations,
    direct_norms=direct.norm,
    tnldr_norms=tnldr.norms,
)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
print(f"figure: {figure_path.with_suffix('.pdf')}")
print(f"figure: {figure_path.with_suffix('.png')}")
