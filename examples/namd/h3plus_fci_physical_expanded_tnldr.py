#!/usr/bin/env python3
"""Expanded-domain H3+ TNLDR diagnostic from a frozen MACE-Y checkpoint.

This script never constructs an electronic-structure driver.  It evaluates the
saved Procrustes-gauged local Hamiltonian and endpoint field, distills them on a
larger dynamics domain, and propagates their linked-product TNLDR Hamiltonian.
The electronic SQLite counts are checked before and after the run.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import Coord, keo
from pyqed.ml import MACE
from pyqed.mps.functional import FunctionalTT
from pyqed.namd import TNLDR
from pyqed.units import au2ev, au2fs


preparation_output = Path("/private/tmp/h3plus_fci_augccpvdz_physical")
output = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")
database = output / "electronic.sqlite"
checkpoint = output / "physical_s3_mace_y.pt"
history_path = output / "physical_adaptive_history.json"
reference_path = output / "h3plus_fci_physical_direct_15.npz"
distilled_energy_path = output / "physical_s3_mace_y_expanded_energy.npz"
distilled_feature_path = output / "physical_s3_mace_y_expanded_feature.npz"
distilled_info_path = output / "physical_s3_mace_y_expanded_info.json"
component_cache = output / "physical_s3_mace_y_expanded_podolsky_r24_r48.pkl"
tag = "h3plus_fci_physical_tnldr_expanded_21"
propagation_checkpoint = output / f"{tag}_propagation.pkl"

preparation = json.loads(
    (preparation_output / "h3plus_fci_initial_state.json").read_text()
)
equilibrium = float(preparation["equilibrium_bond_bohr"])
covariance = np.asarray(preparation["probability_covariance_bohr2"])
packet_widths = np.sqrt(np.diag(covariance))
bounds = ((-0.80, 0.80), (-1.00, 1.00), (-1.00, 1.00))
npts = 21
dt_fs = 0.02
tmax_fs = 5.0
nout = 5


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


def mace_geometry(q):
    return np.asarray(geometry(np.asarray(q, dtype=float)), dtype=float)


class H3Masses:
    """Mass-only molecule interface for the Podolsky operator."""

    @staticmethod
    def atom_mass_list():
        return np.full(3, 1.008)


def database_counts():
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("records", "overlaps")
        }


def cap_profiles(grid, start_sigma=4.5, strength=0.08):
    profiles = {}
    starts = []
    for axis, (coordinate, width) in enumerate(zip(grid.x, packet_widths)):
        wall = float(np.max(np.abs(coordinate)))
        start = min(float(start_sigma) * float(width), 0.82 * wall)
        scaled = np.clip((np.abs(coordinate) - start) / (wall - start), 0.0, 1.0)
        profiles[axis] = float(strength) * scaled**4
        starts.append(start)
    return profiles, starts


def edge_probability(state, driver):
    values = driver.dense(state, physical=False)
    density = np.sum(np.abs(values) ** 2, axis=-1)
    mask = np.zeros(driver.nx, dtype=bool)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        mask[tuple(lower)] = True
        mask[tuple(upper)] = True
    return float(np.sum(density[mask])), density


if not checkpoint.is_file() or not history_path.is_file():
    raise FileNotFoundError("the frozen MACE-Y checkpoint and validation history are required")

counts_before = database_counts()
history = json.loads(history_path.read_text())
validation = history[-1]
grid = DVR.from_axes(
    tuple(SineDVR(lower, upper, npts) for lower, upper in bounds),
    names=("Qs", "Qx", "Qy"),
)
coord = Coord(to_cartesian=geometry, bounds=bounds)

started = perf_counter()
fit = MACE.load(checkpoint, mace_geometry, distill=False)
training_bounds = np.asarray(fit.chart_bounds, dtype=float)
fit.grids = tuple(np.asarray(axis) for axis in grid.x)
fit.shape = tuple(grid.shape)
if all(
    path.is_file()
    for path in (distilled_energy_path, distilled_feature_path, distilled_info_path)
):
    print("loading cached expanded MACE-Y tensor trains", flush=True)
    fit.energy = FunctionalTT.load(distilled_energy_path)
    fit.feature = FunctionalTT.load(distilled_feature_path)
    fit.links = None
    fit.info["distillation"] = json.loads(distilled_info_path.read_text())
else:
    print("distilling frozen MACE-Y fields with adaptive TT-cross", flush=True)
    fit.distill_y(
        rank=64,
        degree=10,
        method="cross",
        prediction_batch_size=512,
        cross_points=11,
        cross_sweeps=6,
        cross_rtol=1.0e-8,
        cross_validation=128,
        validation_points=512,
        seed=41,
    )
    fit.energy.save(distilled_energy_path)
    fit.feature.save(distilled_feature_path)
    distilled_info_path.write_text(
        json.dumps(fit.info["distillation"], indent=2) + "\n"
    )
distill_seconds = perf_counter() - started

print("building expanded Podolsky TNLDR MPO", flush=True)
nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=H3Masses())
started = perf_counter()
if component_cache.is_file():
    print("loading cached Podolsky-dressed MPO components", flush=True)
    with component_cache.open("rb") as stream:
        cached_components = pickle.load(stream)
    tnldr = TNLDR.from_components(
        cached_components["components"],
        grids=grid.x,
        overlap_info=cached_components["overlap_info"],
        potential_info=cached_components["potential_info"],
    ).build()
    tnldr.energy = fit.energy
    tnldr.feature = fit.feature
    tnldr.electronic = fit
else:
    tnldr = TNLDR(
        fit,
        grid=grid,
        coord=coord,
        keo=nuclear_keo,
        overlap_rank=24,
        operator_rank=48,
    ).build()
    with component_cache.open("wb") as stream:
        pickle.dump(
            {
                "components": tnldr.components,
                "overlap_info": tnldr.overlap_info,
                "potential_info": tnldr.potential_info,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
build_seconds = perf_counter() - started

mesh = np.meshgrid(*grid.x, indexing="ij")
coordinates = np.stack(mesh, axis=-1)
flat_coordinates = coordinates.reshape(-1, 3)
hamiltonians = fit.energy.predict(flat_coordinates).reshape(*grid.shape, 2, 2)
levels, vectors = np.linalg.eigh(hamiltonians)
exponent = np.einsum(
    "...i,ij,...j->...", coordinates, np.linalg.inv(covariance), coordinates
)
envelope = np.exp(-0.25 * exponent)
working_packet = envelope[..., None] * vectors[..., :, 1]
packet = tnldr.state(working_packet, max_rank=96, physical=False)
projectors = tuple(
    tnldr.adiabatic_projector(state, method="dense", max_rank=32)[0]
    for state in range(2)
)
profiles, cap_starts = cap_profiles(grid)

if propagation_checkpoint.is_file():
    with propagation_checkpoint.open("rb") as stream:
        propagation = pickle.load(stream)
    completed_snapshots = int(propagation["completed_snapshots"])
    time_parts = propagation["time_parts"]
    population_parts = propagation["population_parts"]
    norm_parts = propagation["norm_parts"]
    snapshot_times = propagation["snapshot_times"]
    edge_values = propagation["edge_values"]
    snapshot_densities = propagation["snapshot_densities"]
    current = propagation["current"]
    print(f"resuming after {completed_snapshots}/5 fs", flush=True)
else:
    completed_snapshots = 0
    time_parts = [np.asarray((0.0,))]
    population_parts = []
    norm_parts = []
    snapshot_times = [0.0]
    initial_edge, initial_density = edge_probability(packet, tnldr)
    edge_values = [initial_edge]
    snapshot_densities = [initial_density]
    current = packet
steps_per_snapshot = round(1.0 / dt_fs)
started = perf_counter()
print("propagating five 1 fs blocks", flush=True)
for snapshot in range(completed_snapshots + 1, 6):
    tnldr.run(
        current,
        dt=dt_fs / au2fs,
        steps=steps_per_snapshot,
        interval=nout,
        max_bond=64,
        integrator="tdvp2",
        cutoff=1.0e-11,
        e_ops=projectors,
        absorber=profiles,
        normalize=False,
        progress=False,
        workers=6,
    )
    offset = snapshot - 1.0
    time_parts.append(offset + tnldr.times[1:] * au2fs)
    if snapshot == 1:
        population_parts.append(tnldr.populations)
        norm_parts.append(tnldr.norms)
    else:
        population_parts.append(tnldr.populations[1:])
        norm_parts.append(tnldr.norms[1:])
    current = tnldr.final_state.copy()
    edge, density = edge_probability(current, tnldr)
    snapshot_times.append(float(snapshot))
    edge_values.append(edge)
    snapshot_densities.append(density)
    with propagation_checkpoint.open("wb") as stream:
        pickle.dump(
            {
                "completed_snapshots": snapshot,
                "time_parts": time_parts,
                "population_parts": population_parts,
                "norm_parts": norm_parts,
                "snapshot_times": snapshot_times,
                "edge_values": edge_values,
                "snapshot_densities": snapshot_densities,
                "current": current,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"completed {snapshot}/5 fs", flush=True)
propagation_seconds = perf_counter() - started

time_fs = np.concatenate(time_parts)
populations = np.vstack(population_parts)
norms = np.concatenate(norm_parts)
counts_after = database_counts()
if counts_after != counts_before:
    raise RuntimeError(
        f"electronic database changed during cache-only dynamics: "
        f"{counts_before} -> {counts_after}"
    )

outside_training = np.any(
    (flat_coordinates < training_bounds[:, 0])
    | (flat_coordinates > training_bounds[:, 1]),
    axis=1,
)
report = {
    "status": "exploratory-fit-extrapolation",
    "electronic_evaluations": 0,
    "electronic_database_counts_before": counts_before,
    "electronic_database_counts_after": counts_after,
    "checkpoint": str(checkpoint),
    "checkpoint_validation": validation,
    "grid": list(grid.shape),
    "bounds_bohr": [list(value) for value in bounds],
    "training_chart_bounds_bohr": training_bounds.tolist(),
    "fraction_dvr_points_outside_training_chart": float(np.mean(outside_training)),
    "dvr_node_extents_bohr": [
        [float(axis[0]), float(axis[-1])] for axis in grid.x
    ],
    "dvr_spacings_bohr": [float(np.diff(axis)[0]) for axis in grid.x],
    "cap": {
        "strength_hartree": 0.08,
        "start_in_initial_sigma": 4.5,
        "starts_bohr": cap_starts,
    },
    "keo": "J=0 Podolsky with pseudopotential; H masses only, no qchem driver",
    "electronic_action": "MACE-Y links with nonunitary linked-product approximation",
    "unitarize_links": False,
    "distillation": fit.info["distillation"],
    "operator_ranks": tnldr.operator_ranks,
    "tnldr_rank_limits": {
        "overlap_rank": 24,
        "operator_rank": 48,
        "state_max_bond": 64,
        "tdvp_workers": 6,
    },
    "tmax_fs": tmax_fs,
    "dt_fs": dt_fs,
    "final_populations": populations[-1].tolist(),
    "final_survival_probability": float(norms[-1]),
    "snapshot_edge_probabilities": edge_values,
    "distill_seconds": distill_seconds,
    "build_seconds": build_seconds,
    "propagation_seconds": propagation_seconds,
}
(output / f"{tag}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
np.savez(
    output / f"{tag}.npz",
    time_fs=time_fs,
    populations=populations,
    norms=norms,
    axes=np.asarray(grid.x),
    snapshot_times_fs=np.asarray(snapshot_times),
    snapshot_densities=np.asarray(snapshot_densities),
    snapshot_edge_probability=np.asarray(edge_values),
    fitted_levels=levels,
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
figure, panels = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
colors = ("#0072B2", "#D55E00")
for state, color in enumerate(colors):
    panels[0].plot(
        time_fs, populations[:, state], color=color, label=fr"expanded fit $S_{state + 1}$"
    )
if reference_path.is_file():
    reference = np.load(reference_path)
    for state, color in enumerate(colors):
        panels[0].plot(
            reference["time_fs"], reference["populations"][:, state], "--",
            color=color, alpha=0.75, label=fr"15$^3$ direct $S_{state + 1}$",
        )
panels[0].set(
    xlabel="time (fs)", ylabel="adiabatic population",
    title="(a) Domain sensitivity", ylim=(-0.02, 1.02),
)
panels[0].legend(frameon=False, ncol=2)
panels[1].plot(time_fs, norms, color="0.15", label="survival")
panels[1].plot(
    snapshot_times, edge_values, "o--", color="#009E73", label="outer DVR layer"
)
panels[1].set(
    xlabel="time (fs)", ylabel="probability",
    title="(b) Expanded-boundary diagnostic", ylim=(-0.02, 1.02),
)
panels[1].legend(frameon=False)
for panel in panels:
    panel.spines[["top", "right"]].set_visible(False)
    panel.tick_params(direction="out")
population_path = output / f"{tag}_population"
figure.savefig(population_path.with_suffix(".pdf"))
figure.savefig(population_path.with_suffix(".png"), dpi=360)
plt.close(figure)

snapshot_figure, snapshot_panels = plt.subplots(
    2, 3, figsize=(8.0, 5.2), constrained_layout=True, sharex=True, sharey=True
)
for panel, time, density in zip(
    snapshot_panels.flat, snapshot_times, snapshot_densities
):
    projected = np.sum(density, axis=0)
    maximum = max(float(np.max(projected)), np.finfo(float).tiny)
    image = panel.pcolormesh(
        grid.x[1], grid.x[2], (projected / maximum).T,
        shading="auto", cmap="magma", vmin=0.0, vmax=1.0,
    )
    panel.set_title(fr"$t={time:.1f}$ fs")
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
