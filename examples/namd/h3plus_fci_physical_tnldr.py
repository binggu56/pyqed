#!/usr/bin/env python3
"""Physical 3D H3+ FCI -> S3-MACE -> FTT -> TNLDR benchmark."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.ml import MACE
from pyqed.namd import TNLDR
from pyqed.qchem import Molecule
from pyqed.units import au2ev, au2fs


preparation_output = Path("/private/tmp/h3plus_fci_augccpvdz_physical")
output = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")
output.mkdir(parents=True, exist_ok=True)
database = output / "electronic.sqlite"
initial_state = json.loads(
    (preparation_output / "h3plus_fci_initial_state.json").read_text()
)
equilibrium = float(initial_state["equilibrium_bond_bohr"])
covariance = np.asarray(initial_state["probability_covariance_bohr2"])
packet_widths = np.sqrt(np.diag(covariance))
bounds = ((-0.60, 0.60), (-0.70, 0.70), (-0.70, 0.70))
tmax_fs = 5.0
dt_fs = 0.02
nout = 5
distortion_limit = 0.65


def geometry(q):
    """Map symmetry coordinates through a non-folding exponential strain.

    The Jacobian at the origin is the conventional linear H3+ normal-mode
    map.  Saturating only the traceless strain keeps an expanded rectangular
    coordinate chart away from artificial H-H collisions while preserving
    the S3 action on ``(Qx, Qy)``.
    """
    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    qs, qx, qy = q
    radius = jnp.sqrt(qx**2 + qy**2 + 1.0e-16)
    amplitude = distortion_limit * jnp.tanh(radius / distortion_limit)
    traceless = jnp.asarray(((qx, qy), (qy, -qx)))
    strain = (
        jnp.cosh(amplitude / equilibrium) * jnp.eye(2)
        + jnp.sinh(amplitude / equilibrium) / radius * traceless
    )
    transform = jnp.exp(qs / equilibrium) * strain
    return triangle.at[:, :2].set(
        equilibrium * triangle[:, :2] @ transform
    )


def mace_geometry(q):
    return np.asarray(geometry(np.asarray(q, dtype=float)), dtype=float)


def graph_pairs(coordinates, neighbors=3):
    coordinates = np.asarray(coordinates)
    scale = np.ptp(coordinates, axis=0)
    scale[scale < 1.0e-12] = 1.0
    distances = cdist(coordinates / scale, coordinates / scale)
    tree = minimum_spanning_tree(distances).tocoo()
    pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    np.fill_diagonal(distances, np.inf)
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], neighbors - 1)[:neighbors]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    return np.asarray(sorted(pairs), dtype=int)


def physical_coordinates(seed, count, radius=3.5):
    """Independent quasi-random points in the packet-accessible ellipsoid."""
    count = int(count)
    power = int(np.ceil(np.log2(max(16, 4 * count))))
    unit = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    normal = ndtri(np.clip(unit, 1.0e-10, 1.0 - 1.0e-10))
    normal = normal[np.linalg.norm(normal, axis=1) <= float(radius)]
    if len(normal) < count:
        raise RuntimeError("Sobol rejection pool was too small")
    return normal[:count] * packet_widths


def physical_shell(seed, count, radius=3.6):
    random = np.random.default_rng(seed)
    directions = random.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(radius) * directions * packet_widths


def uniform_coordinates(seed, count, margin=0.0):
    """Independent Sobol points spanning the active coordinate chart."""
    box = np.asarray(bounds, dtype=float)
    lower = box[:, 0] + float(margin)
    upper = box[:, 1] - float(margin)
    if np.any(upper <= lower):
        raise ValueError("uniform-coordinate margin leaves an empty chart")
    power = int(np.ceil(np.log2(max(16, int(count)))))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)
    return qmc.scale(unit[: int(count)], lower, upper)


def paired_coordinates(seed, count, step=0.08, physical_fraction=0.5):
    """Return independent local validation pairs over the chart and packet."""
    count = int(count)
    physical_count = int(round(float(physical_fraction) * count))
    uniform_count = count - physical_count
    box = np.asarray(bounds, dtype=float)
    margin = float(step) + 1.0e-8
    physical = physical_coordinates(seed, physical_count, radius=5.0)
    physical = np.clip(
        physical,
        box[:, 0] + margin,
        box[:, 1] - margin,
    )
    centers = np.vstack(
        (physical, uniform_coordinates(seed + 1, uniform_count, margin=margin))
    )
    random = np.random.default_rng(int(seed) + 2)
    directions = random.normal(size=centers.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    endpoints = centers + float(step) * directions
    coordinates = np.empty((2 * count, 3), dtype=float)
    coordinates[0::2] = centers
    coordinates[1::2] = endpoints
    pairs = np.column_stack((2 * np.arange(count), 2 * np.arange(count) + 1))
    return coordinates, pairs


def train_mace(
    sampler,
    grid,
    coordinates,
    finite_group,
    previous=None,
    epochs=None,
    expanded=False,
):
    coordinates = sampler.reduce_coordinates(coordinates)
    pairs = graph_pairs(coordinates)
    fields = sampler.continuous_fields(coordinates, pairs)
    fit = MACE(
        grid.x,
        ("H", "H", "H"),
        mace_geometry,
        2,
        chart_features=True,
        chart_bounds=bounds,
        geometry_units="bohr",
        channels=20 if expanded else 16,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=10 if expanded else 8,
        radial_mlp=(96, 96) if expanded else (64, 64),
        cutoff=4.5,
    ).fit_y(
        (coordinates, fields["hamiltonians"]),
        coordinates,
        pairs,
        fields["links"],
        feature_rank=20 if expanded else 16,
        feature_objective="links-only",
        ambient_representation="full",
        energy_representation="direct",
        energy_objective="trace-traceless",
        finite_group=finite_group,
        hidden=(96, 96) if expanded else (64, 64),
        epochs=(
            (450 if previous is None else 160)
            if expanded
            else (650 if previous is None else 320)
        ) if epochs is None else epochs,
        learning_rate=1.5e-3 if previous is None else 5.0e-4,
        weight_decay=1.0e-8,
        frame_fraction=0.40 if previous is None else 0.0,
        ambient_fraction=0.20 if previous is None else 0.0,
        smoothness=1.0e-5,
        energy_weight=80.0,
        initial_fit=previous,
        seed=19,
        distill=False,
    )
    return fit


def assess_models(energy, feature_model, fields):
    predicted_h = energy.predict(fields["coordinates"])
    feature = feature_model.predict(fields["coordinates"])
    pairs = fields["pairs"]
    predicted_links = (
        feature[pairs[:, 0]].conj().swapaxes(-1, -2) @ feature[pairs[:, 1]]
    )
    h_errors = np.linalg.norm(predicted_h - fields["hamiltonians"], axis=(-2, -1))
    link_errors = np.linalg.norm(
        predicted_links - fields["links"], axis=(-2, -1)
    ) / np.maximum(
        np.linalg.norm(fields["links"], axis=(-2, -1)),
        np.finfo(float).tiny,
    )
    return {
        "maximum_hamiltonian_error_hartree": float(np.max(h_errors)),
        "rms_hamiltonian_error_hartree": float(np.sqrt(np.mean(h_errors**2))),
        "maximum_relative_link_error": float(np.max(link_errors)),
        "relative_link_error": float(
            np.linalg.norm(predicted_links - fields["links"])
            / max(float(np.linalg.norm(fields["links"])), np.finfo(float).tiny)
        ),
    }


def assess(fit, fields):
    return assess_models(fit.neural_energy, fit.neural_feature, fields)


def subspace_diagnostics(fields, max_distance=0.15):
    """Measure loss from the selected electronic subspace on short raw links."""
    coordinates = np.asarray(fields["coordinates"], dtype=float)
    pairs = np.asarray(fields["pairs"], dtype=int)
    distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
    )
    local = distances <= float(max_distance)
    if not np.any(local):
        raise RuntimeError("no short validation links are available")
    singular_values = np.linalg.svd(fields["links"][local], compute_uv=False)
    minimum = singular_values[:, -1]
    return {
        "maximum_link_distance_bohr": float(max_distance),
        "local_links": int(np.count_nonzero(local)),
        "minimum_link_singular_value": float(np.min(minimum)),
        "one_percent_link_singular_value": float(np.quantile(minimum, 0.01)),
        "median_link_singular_value": float(np.median(minimum)),
        "fraction_links_below_0_9": float(np.mean(minimum < 0.9)),
        "maximum_projector_loss": float(np.max(1.0 - minimum**2)),
    }


def state_gap_diagnostics(sampler, coordinates):
    """Measure separation of the selected manifold from excluded singlet roots."""

    roots = []
    for coordinate in np.asarray(coordinates, dtype=float):
        record = sampler.database.get(
            {
                "geometry": sampler.coord.cartesian(tuple(coordinate)),
                "protocol": sampler.protocol,
            }
        )
        if record is None:
            raise RuntimeError("state-gap diagnostic requires sampled coordinates")
        values = record[1] if isinstance(record, tuple) else record["energies"]
        roots.append(np.asarray(values, dtype=float))
    roots = np.asarray(roots)
    first, last = min(sampler.states), max(sampler.states)
    if tuple(sampler.states) != tuple(range(first, last + 1)):
        raise ValueError("state-gap diagnostics require a contiguous state manifold")
    gaps = []
    labels = []
    if first > 0:
        gaps.append(roots[:, first] - roots[:, first - 1])
        labels.append(f"E{first}-E{first - 1}")
    if last + 1 < roots.shape[1]:
        gaps.append(roots[:, last + 1] - roots[:, last])
        labels.append(f"E{last + 1}-E{last}")
    if not gaps:
        raise ValueError("no excluded root brackets the selected manifold")
    gaps = np.column_stack(gaps)
    nearest = np.min(gaps, axis=1)
    return {
        "excluded_gap_labels": labels,
        "minimum_excluded_root_gap_hartree": float(np.min(nearest)),
        "one_percent_excluded_root_gap_hartree": float(
            np.quantile(nearest, 0.01)
        ),
        "median_excluded_root_gap_hartree": float(np.median(nearest)),
        "coordinates": np.asarray(coordinates, dtype=float),
        "excluded_root_gaps": gaps,
        "nearest_excluded_root_gap": nearest,
    }


def plot_state_leakage(fields, gap_diagnostics, suffix=""):
    """Plot selected-subspace retention and separation from excluded roots."""

    coordinates = np.asarray(fields["coordinates"], dtype=float)
    pairs = np.asarray(fields["pairs"], dtype=int)
    distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
    )
    singular_values = np.linalg.svd(fields["links"], compute_uv=False)[:, -1]
    gap_coordinates = gap_diagnostics["coordinates"]
    normalized_radius = np.linalg.norm(gap_coordinates / packet_widths, axis=1)
    gap_ev = gap_diagnostics["nearest_excluded_root_gap"] * au2ev

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, panels = plt.subplots(
        1, 2, figsize=(7.2, 2.8), constrained_layout=True
    )
    ordered_singular_values = np.sort(singular_values)
    percentile = 100.0 * (
        np.arange(len(ordered_singular_values)) + 0.5
    ) / len(ordered_singular_values)
    panels[0].plot(
        percentile,
        ordered_singular_values,
        color="#0072B2",
        linewidth=1.3,
        marker="o",
        markerfacecolor="white",
        markersize=3.5,
    )
    panels[0].axhline(0.9, color="#D55E00", linestyle="--", linewidth=1.0)
    panels[0].set(
        xlabel="link percentile",
        ylabel="minimum link singular value",
        title=rf"(a) Subspace retention, $|\Delta q|={np.median(distances):.2f}$ bohr",
        ylim=(min(0.895, float(np.min(singular_values)) - 0.002), 1.001),
    )
    panels[1].scatter(
        normalized_radius,
        gap_ev,
        s=22,
        facecolor="white",
        edgecolor="#009E73",
        linewidth=0.9,
    )
    panels[1].set(
        xlabel=r"packet-scaled radius $|q/\sigma|$",
        ylabel="nearest excluded-root gap (eV)",
        title="(b) Separation from other singlets",
    )
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(direction="out")
    path = output / f"h3plus_fci_state_leakage{suffix}"
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=360)
    plt.close(figure)
    np.savez(
        path.with_suffix(".npz"),
        pair_distance=distances,
        minimum_link_singular_value=singular_values,
        gap_coordinates=gap_coordinates,
        nearest_excluded_root_gap_hartree=gap_diagnostics[
            "nearest_excluded_root_gap"
        ],
    )
    return path


def plot_production_validation(fit):
    """Plot independent neural errors against the production gates."""
    h_error = np.sort(
        np.asarray(fit.validation["independent_hamiltonian_errors"]) * au2ev
        * 1000.0
    )
    link_error = np.sort(
        np.asarray(fit.validation["independent_link_errors"])
    )
    figure, panels = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    panels[0].plot(h_error, color="#0072B2")
    panels[0].axhline(
        fit.acceptance["hamiltonian_atol"] * au2ev * 1000.0,
        color="#D55E00",
        linestyle="--",
        label="maximum-error gate",
    )
    panels[0].set(
        xlabel="sorted validation point",
        ylabel=r"$||\Delta \bar H||_F$ (meV)",
        title="(a) Gauged Hamiltonian",
    )
    panels[1].plot(link_error, color="#009E73")
    panels[1].axhline(
        fit.acceptance["link_rtol"],
        color="#D55E00",
        linestyle="--",
        label="relative-error gate",
    )
    panels[1].set(
        xlabel="sorted validation link",
        ylabel="relative raw-link error",
        title="(b) Endpoint links",
    )
    for panel in panels:
        panel.legend(frameon=False)
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(direction="out")
    path = output / "h3plus_fci_production_validation_rejected"
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=360)
    plt.close(figure)
    np.savez(
        path.with_suffix(".npz"),
        hamiltonian_error_hartree=h_error / (au2ev * 1000.0),
        relative_link_error=link_error,
    )
    return path


def plot_pes_cuts(sampler, fit, validation_metrics, suffix=""):
    """Plot symmetry-inequivalent FCI cuts against MACE and distilled FTT."""
    expanded = max(abs(value) for bound in bounds for value in bound) > 0.75
    angle15 = np.deg2rad(15.0)
    angle30 = np.deg2rad(30.0)
    mixed = np.asarray((1.0, 0.8 * np.cos(angle15), 0.8 * np.sin(angle15)))
    mixed /= np.linalg.norm(mixed)
    directions = (
        ("breathing", r"breathing $Q_s$", (1.0, 0.0, 0.0), 0.72, 23),
        ("branching", r"branching $\theta=0^\circ$", (0.0, 1.0, 0.0), 0.88, 23),
        (
            "theta15",
            r"branching $\theta=15^\circ$",
            (0.0, np.cos(angle15), np.sin(angle15)),
            0.88,
            15,
        ),
        (
            "theta30",
            r"branching $\theta=30^\circ$",
            (0.0, np.cos(angle30), np.sin(angle30)),
            0.88,
            15,
        ),
        (
            "diagonal",
            r"diagonal $Q_x=Q_y$",
            np.asarray((0.0, 1.0, 1.0)) / np.sqrt(2.0),
            0.88,
            23,
        ),
        ("mixed", r"mixed $Q_s+Q_{15^\circ}$", mixed, 0.92, 15),
    )
    if not expanded:
        directions = tuple(
            (key, name, direction, min(extent, 0.62), points)
            for key, name, direction, extent, points in directions
        )
    raw_abscissa, raw_coordinates = [], []
    dense_abscissa, dense_coordinates = [], []
    for _key, _name, direction, extent, points in directions:
        direction = np.asarray(direction, dtype=float)
        raw_axis = np.linspace(-extent, extent, points)
        dense_axis = np.linspace(-extent, extent, 401)
        raw_abscissa.append(raw_axis)
        raw_coordinates.append(raw_axis[:, None] * direction[None, :])
        dense_abscissa.append(dense_axis)
        dense_coordinates.append(dense_axis[:, None] * direction[None, :])

    raw_hamiltonians = [
        sampler.continuous_fields(coordinates)["hamiltonians"]
        for coordinates in raw_coordinates
    ]
    raw_levels = [np.linalg.eigvalsh(value) for value in raw_hamiltonians]
    mace_raw_levels = [
        np.linalg.eigvalsh(fit.neural_energy.predict(coordinates))
        for coordinates in raw_coordinates
    ]

    def ftt_levels(coordinates):
        fitted_bounds = np.asarray(fit.energy.bounds_, dtype=float)
        inside = np.all(
            (coordinates >= fitted_bounds[:, 0])
            & (coordinates <= fitted_bounds[:, 1]),
            axis=1,
        )
        levels = np.full((len(coordinates), 2), np.nan)
        levels[inside] = np.linalg.eigvalsh(fit.energy.predict(coordinates[inside]))
        return levels

    ftt_raw_levels = [ftt_levels(coordinates) for coordinates in raw_coordinates]
    mace_dense_levels = [
        np.linalg.eigvalsh(fit.neural_energy.predict(coordinates))
        for coordinates in dense_coordinates
    ]
    ftt_dense_levels = [ftt_levels(coordinates) for coordinates in dense_coordinates]

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ("#0072B2", "#D55E00")
    figure, panels = plt.subplots(
        4,
        3,
        figsize=(9.0, 8.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.0, 0.9, 2.0, 0.9)},
    )
    for index, (_key, name, _direction, _extent, _points) in enumerate(directions):
        block, column = divmod(index, 3)
        energy_panel = panels[2 * block, column]
        residual_panel = panels[2 * block + 1, column]
        origin = float(np.min(raw_levels[index]))
        for state, color in enumerate(colors):
            energy_panel.plot(
                dense_abscissa[index],
                (mace_dense_levels[index][:, state] - origin) * au2ev,
                color=color,
                label=fr"MACE $S_{state + 1}$",
            )
            energy_panel.plot(
                dense_abscissa[index],
                (ftt_dense_levels[index][:, state] - origin) * au2ev,
                "--",
                color=color,
                label=fr"FTT $S_{state + 1}$",
            )
            energy_panel.scatter(
                raw_abscissa[index],
                (raw_levels[index][:, state] - origin) * au2ev,
                s=17,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
                label=fr"FCI $S_{state + 1}$",
            )
            residual_panel.plot(
                raw_abscissa[index],
                (mace_raw_levels[index][:, state] - raw_levels[index][:, state])
                * au2ev
                * 1000.0,
                color=color,
                marker="o",
                markersize=2.5,
            )
            residual_panel.plot(
                raw_abscissa[index],
                (ftt_raw_levels[index][:, state] - raw_levels[index][:, state])
                * au2ev
                * 1000.0,
                "--",
                color=color,
            )
        energy_panel.set_title(fr"({chr(97 + index)}) {name}")
        residual_panel.axhspan(-20.0, 20.0, color="0.92", zorder=-2)
        residual_panel.axhline(0.0, color="0.55", linewidth=0.7)
        residual_panel.set_xlabel(r"cut coordinate (bohr)")
        for panel in (energy_panel, residual_panel):
            panel.spines[["top", "right"]].set_visible(False)
            panel.tick_params(direction="out")
    for row in (0, 2):
        panels[row, 0].set_ylabel("energy relative to cut minimum (eV)")
    for row in (1, 3):
        panels[row, 0].set_ylabel("fit - FCI (meV)")
    handles, labels = panels[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=6, loc="outside lower center")
    figure.suptitle(
        "H$_3^+$ spin-pure singlet full CI/aug-cc-pVDZ\n"
        + r"validation max $||\Delta \bar H||_F$="
        + f"{validation_metrics['maximum_hamiltonian_error_hartree'] * au2ev:.3f} eV, "
        + rf"relative link error={validation_metrics['relative_link_error']:.3f}"
    )
    cut_path = output / f"h3plus_fci_physical_fitted_pes_cuts{suffix}"
    figure.savefig(cut_path.with_suffix(".pdf"))
    figure.savefig(cut_path.with_suffix(".png"), dpi=360)
    plt.close(figure)

    arrays = {}
    for index, (key, _name, _direction, _extent, _points) in enumerate(directions):
        arrays[f"{key}_raw_coordinate"] = raw_abscissa[index]
        arrays[f"{key}_raw_fci_hamiltonian"] = raw_hamiltonians[index]
        arrays[f"{key}_raw_fci_levels"] = raw_levels[index]
        arrays[f"{key}_raw_mace_levels"] = mace_raw_levels[index]
        arrays[f"{key}_raw_ftt_levels"] = ftt_raw_levels[index]
        arrays[f"{key}_dense_coordinate"] = dense_abscissa[index]
        arrays[f"{key}_dense_mace_levels"] = mace_dense_levels[index]
        arrays[f"{key}_dense_ftt_levels"] = ftt_dense_levels[index]
    np.savez(cut_path.with_suffix(".npz"), **arrays)
    return cut_path


def harmonic_envelope(grid):
    mesh = np.meshgrid(*grid.x, indexing="ij")
    coordinates = np.stack(mesh, axis=-1)
    exponent = np.einsum(
        "...i,ij,...j->...", coordinates, np.linalg.inv(covariance), coordinates
    )
    return np.exp(-0.25 * exponent)


def fitted_working_packet(fit, grid, state):
    """Return the same fitted local-adiabatic packet on any dynamics grid."""
    coordinates = np.stack(
        np.meshgrid(*grid.x, indexing="ij"), axis=-1
    ).reshape(-1, grid.ndim)
    blocks = np.asarray(fit.energy.predict(coordinates))
    _energies, vectors = np.linalg.eigh(blocks)
    envelope = harmonic_envelope(grid).reshape(-1)
    return (envelope[:, None] * vectors[:, :, int(state)]).reshape(
        *grid.shape, vectors.shape[-2]
    )


def caps(grid):
    profiles = {}
    for axis, (coordinate, width) in enumerate(zip(grid.x, packet_widths)):
        wall = float(np.max(np.abs(coordinate)))
        start = min(3.5 * float(width), 0.82 * wall)
        scaled = np.clip((np.abs(coordinate) - start) / (wall - start), 0.0, 1.0)
        profiles[axis] = 0.08 * scaled**4
    return profiles


def cap_tensor(grid, profiles):
    value = np.zeros(grid.shape)
    for axis, profile in profiles.items():
        shape = [1] * grid.ndim
        shape[axis] = len(profile)
        value += profile.reshape(shape)
    return value


def direct_populations(direct):
    order = np.argsort(direct.energies, axis=-1)
    return np.sum(
        np.take_along_axis(np.abs(direct.states) ** 2, order[None, ...], axis=-1),
        axis=(1, 2, 3),
    )


def edge_probability(states):
    density = np.sum(np.abs(states) ** 2, axis=-1)
    edge = np.zeros(density.shape[1:], dtype=bool)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        edge[tuple(lower)] = True
        edge[tuple(upper)] = True
    return np.sum(density[:, edge], axis=1)


def main():
    global bounds
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-cuts",
        action="store_true",
        help="fit once, plot PES cuts, and stop before the dynamics accuracy gate",
    )
    parser.add_argument(
        "--plot-cuts",
        action="store_true",
        help="plot PES cuts from the final adaptive checkpoint and stop",
    )
    parser.add_argument(
        "--expanded-fit",
        action="store_true",
        help="qualify an adaptive ab initio fit on the expanded 21^3 chart",
    )
    parser.add_argument(
        "--refine-h-fit",
        action="store_true",
        help="continue an existing fit with the balanced trace/traceless objective",
    )
    parser.add_argument("--refine-epochs", type=int, default=1500)
    parser.add_argument(
        "--production-build",
        action="store_true",
        help="use the native adaptive ensemble-MACE/FTT acceptance pipeline",
    )
    parser.add_argument(
        "--database-audit",
        action="store_true",
        help="report reusable symmetry-reduced records and stop",
    )
    args = parser.parse_args()
    if args.expanded_fit:
        bounds = ((-0.80, 0.80), (-1.00, 1.00), (-1.00, 1.00))
    grid_points = 21 if args.expanded_fit else 13
    fine_points = 25 if args.expanded_fit else 17
    grid = DVR.from_axes(
        tuple(SineDVR(lower, upper, grid_points) for lower, upper in bounds),
        names=("Qs", "Qx", "Qy"),
    )
    fine_grid = DVR.from_axes(
        tuple(SineDVR(lower, upper, fine_points) for lower, upper in bounds),
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
    fit_options = (
        {
            "model": "mace",
            "cache_only": True,
            "initial": 320,
            "batch": 24,
            "maximum": 320,
            "rounds": 1,
            "calibration": 64,
            "validation": 96,
            "ensemble": 3,
            "epochs": 300,
            "refinement_epochs": 240,
            "sync_steps": 600,
            "feature_rank": 20,
            "hidden": (128, 128),
            "encoder": {
                "channels": 24,
                "max_ell": 2,
                "interactions": 2,
                "correlation": 2,
                "radial_basis": 10,
                "radial_mlp": (96, 96),
                "cutoff": 5.0,
            },
            "degrees": 12,
            "rank": 64,
            "verbose": True,
            "strict": False,
        }
        if args.production_build
        else {"degrees": (8, 10, 10), "rank": 64}
    )
    sampler = AbInitioFit(
        mc,
        coord=coord,
        states=(1, 2),
        nroots=6,
        fit_options=fit_options,
        database=database,
        workers=6,
        progress=False,
    )
    if args.database_audit:
        reusable = sampler._database_coordinates()
        print(
            json.dumps(
                {
                    "database": str(database),
                    "records": sampler.database.stats["records"],
                    "protocol_matched_symmetry_representatives": len(reusable),
                    "group": sampler.group,
                    "coord_irreps": sampler.coord_irreps,
                    "bounds": bounds,
                },
                indent=2,
            ),
            flush=True,
        )
        sampler.close()
        return
    if args.production_build:
        fit = sampler.build()
        if not fit.success:
            checkpoint = fit.mace.save(output / "production_mace_rejected.pt")
            validation_path = plot_production_validation(fit)
            report = {
                "accepted": False,
                "model": fit.model,
                "group": fit.group,
                "coord_irreps": fit.coord_irreps,
                "state_validation": fit.state_validation,
                "validation": fit.validation,
                "acceptance": fit.acceptance,
                "database": str(database),
                "database_writes": fit.stats["database"]["writes"],
                "checkpoint": str(checkpoint),
                "validation_figure": str(validation_path.with_suffix(".png")),
            }

            def json_default(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)

            report_path = output / "production_fit_rejected.json"
            report_path.write_text(
                json.dumps(report, indent=2, default=json_default) + "\n"
            )
            print(json.dumps(report, indent=2, default=json_default), flush=True)
            print(validation_path.with_suffix(".png"), flush=True)
            return
        fit_directory = output / "production_fit"
        fit.save(fit_directory, labels=("Qs", "Qx", "Qy"))
        independent = fit.validation["independent"]
        metrics = {
            "maximum_hamiltonian_error_hartree": independent[
                "maximum_hamiltonian_error"
            ],
            "rms_hamiltonian_error_hartree": independent[
                "rms_hamiltonian_error"
            ],
            "maximum_relative_link_error": independent[
                "maximum_relative_link_error"
            ],
            "relative_link_error": independent["relative_link_error"],
        }
        cut_path = plot_pes_cuts(
            sampler, fit.mace, metrics, suffix="_production_accepted"
        )
        report = {
            "accepted": fit.acceptance["accepted"],
            "model": fit.model,
            "group": fit.group,
            "coord_irreps": fit.coord_irreps,
            "state_validation": fit.state_validation,
            "validation": fit.validation,
            "acceptance": fit.acceptance,
            "database": str(database),
            "database_writes": fit.stats["database"]["writes"],
            "fit_directory": str(fit_directory),
            "pes_cuts": str(cut_path.with_suffix(".png")),
        }
        (output / "production_fit_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        print(cut_path.with_suffix(".png"), flush=True)
        return

    feature_rank = 20 if args.expanded_fit else 16
    calibration_base = np.asarray((0.0, 0.065, 0.027))
    calibration = sampler.orbit(calibration_base)
    finite_group = sampler.mace_group(feature_rank, tolerance=2.0e-7)
    orbit_budget = sampler.reduced_size
    full_domain_initial_budget = (
        1
        + 6
        + (384 if args.expanded_fit else 192)
        + (96 if args.expanded_fit else 48)
        + (256 if args.expanded_fit else 0)
    )
    training_parts = [
        np.zeros((1, 3)),
        calibration,
        physical_coordinates(
            19,
            orbit_budget(384 if args.expanded_fit else 192),
            radius=5.0 if args.expanded_fit else 3.5,
        ),
        physical_shell(
            29,
            orbit_budget(96 if args.expanded_fit else 48),
            radius=4.5 if args.expanded_fit else 3.6,
        ),
    ]
    if args.expanded_fit:
        training_parts.append(uniform_coordinates(39, orbit_budget(256)))
    requested_training_points = len(np.unique(np.vstack(training_parts), axis=0))
    training = sampler.reduce_coordinates(np.vstack(training_parts))
    reduced_initial_training_points = len(training)
    if args.expanded_fit:
        validation, validation_pairs = paired_coordinates(119, orbit_budget(256))
        requested_validation_pairs = len(validation_pairs)
        validation, validation_pairs = sampler.reduce_pairs(
            validation, validation_pairs
        )
    else:
        validation = physical_coordinates(119, orbit_budget(128))
        requested_validation_pairs = None
        validation = sampler.reduce_coordinates(validation)
        validation_pairs = graph_pairs(validation)
    validation_fields = sampler.continuous_fields(
        validation, validation_pairs
    )
    subspace = subspace_diagnostics(validation_fields)
    state_gaps = state_gap_diagnostics(sampler, validation)
    state_gap_summary = {
        key: value
        for key, value in state_gaps.items()
        if key
        not in {
            "coordinates",
            "excluded_root_gaps",
            "nearest_excluded_root_gap",
        }
    }
    print(
        f"singlet subspace: {subspace}; excluded-root gaps: {state_gap_summary}",
        flush=True,
    )
    if subspace["minimum_link_singular_value"] < 0.9:
        raise RuntimeError(
            "refusing fit: the selected singlet pair leaks from the local "
            "electronic subspace; increase nroots and track the subspace"
        )
    diagnostic_suffix = "_diagnostic" if args.diagnostic_cuts else ""
    if args.expanded_fit:
        diagnostic_suffix += "_curvilinear_expanded_abinitio"
    fit_suffix = "_curvilinear_expanded_abinitio" if args.expanded_fit else ""
    checkpoint = output / f"physical_s3_quotient_mace_y{fit_suffix}.pt"
    history_path = output / f"physical_s3_quotient_history{fit_suffix}.json"
    training_path = output / f"physical_s3_quotient_training{fit_suffix}.npy"
    started = perf_counter()
    if checkpoint.is_file() and history_path.is_file():
        fit = MACE.load(checkpoint, mace_geometry, distill=False)
        history = json.loads(history_path.read_text())
        history = [
            item
            for item in history
            if item.get("stage") != "symmetry-adapted polynomial Hamiltonian head"
        ]
        if training_path.is_file():
            training = np.load(training_path)
    else:
        fit = train_mace(
            sampler,
            grid,
            training,
            finite_group,
            epochs=500 if args.diagnostic_cuts else None,
            expanded=args.expanded_fit,
        )
        history = [{"round": 0, **subspace, **assess(fit, validation_fields)}]
        print(f"adaptive round 0: {history[-1]}", flush=True)
        fit.save(checkpoint)
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        np.save(training_path, training)

    if args.refine_h_fit:
        refinement_pairs = graph_pairs(training)
        refinement_fields = sampler.continuous_fields(
            training, refinement_pairs
        )
        fit.refine_hamiltonian(
            training,
            refinement_fields["hamiltonians"],
            epochs=args.refine_epochs,
            learning_rate=1.0e-3,
            objective="trace-traceless",
            seed=19,
        )
        history.append(
            {
                "round": int(history[-1]["round"]) + 1,
                "stage": "frozen-encoder S3-projected MACE H refinement",
                **subspace,
                **state_gap_summary,
                **assess(fit, validation_fields),
            }
        )
        print(f"Hamiltonian refinement: {history[-1]}", flush=True)
        fit.save(checkpoint)
        history_path.write_text(json.dumps(history, indent=2) + "\n")

    for adaptive_round in range(
        int(history[-1]["round"]) + 1,
        1 if args.diagnostic_cuts else (9 if args.expanded_fit else 5),
    ):
        if (
            history[-1]["maximum_hamiltonian_error_hartree"] <= 7.5e-4
            and history[-1]["relative_link_error"] <= 2.0e-2
        ):
            break
        if args.expanded_fit:
            candidates, candidate_pairs = paired_coordinates(
                200 + adaptive_round,
                orbit_budget(128),
                physical_fraction=0.35,
            )
            candidates, candidate_pairs = sampler.reduce_pairs(
                candidates, candidate_pairs
            )
        else:
            candidates = physical_coordinates(
                200 + adaptive_round, orbit_budget(128)
            )
            candidates = sampler.reduce_coordinates(candidates)
            candidate_pairs = graph_pairs(candidates)
        candidate_fields = sampler.continuous_fields(
            candidates, candidate_pairs
        )
        candidate_h_error = np.linalg.norm(
            fit.neural_energy.predict(candidate_fields["coordinates"])
            - candidate_fields["hamiltonians"],
            axis=(-2, -1),
        )
        candidate_feature = fit.neural_feature.predict(
            candidate_fields["coordinates"]
        )
        predicted_candidate_links = (
            candidate_feature[candidate_pairs[:, 0]].conj().swapaxes(-1, -2)
            @ candidate_feature[candidate_pairs[:, 1]]
        )
        candidate_link_error = np.linalg.norm(
            predicted_candidate_links - candidate_fields["links"],
            axis=(-2, -1),
        ) / np.maximum(
            np.linalg.norm(candidate_fields["links"], axis=(-2, -1)),
            np.finfo(float).tiny,
        )
        pair_h_error = np.maximum(
            candidate_h_error[candidate_pairs[:, 0]],
            candidate_h_error[candidate_pairs[:, 1]],
        )
        acquisition_score = np.maximum(
            pair_h_error / 7.5e-4,
            candidate_link_error / 2.0e-2,
        )
        acquired_pairs = candidate_pairs[
            np.argsort(acquisition_score)[
                -orbit_budget(48 if args.expanded_fit else 16):
            ]
        ]
        training = np.unique(
            np.vstack(
                (
                    training,
                    candidate_fields["coordinates"][np.unique(acquired_pairs)],
                )
            ),
            axis=0,
        )
        training = sampler.reduce_coordinates(training)
        fit = train_mace(
            sampler,
            grid,
            training,
            finite_group,
            previous=fit,
            expanded=args.expanded_fit,
        )
        history.append(
            {
                "round": adaptive_round,
                "acquisition_maximum_hamiltonian_error_hartree": float(
                    np.max(candidate_h_error)
                ),
                "acquisition_maximum_relative_link_error": float(
                    np.max(candidate_link_error)
                ),
                "training_points": int(len(training)),
                **subspace,
                **assess(fit, validation_fields),
            }
        )
        print(f"adaptive round {adaptive_round}: {history[-1]}", flush=True)
        fit.save(checkpoint)
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        np.save(training_path, training)
    fit_seconds = perf_counter() - started

    history[-1].update(assess(fit, validation_fields))
    history_path.write_text(json.dumps(history, indent=2) + "\n")

    if args.diagnostic_cuts or args.plot_cuts:
        if fit.energy is fit.neural_energy:
            fit.distill_y(rank=64, degree=12, method="grid", seed=19)
            fit.save(checkpoint)
        accepted = (
            history[-1]["maximum_hamiltonian_error_hartree"] <= 7.5e-4
            and history[-1]["relative_link_error"] <= 2.0e-2
        )
        cut_path = plot_pes_cuts(
            sampler,
            fit,
            history[-1],
            suffix=(
                ("_diagnostic_" if args.diagnostic_cuts else "_adaptive_")
                + ("accepted" if accepted else "rejected")
                + ("_curvilinear_expanded_abinitio" if args.expanded_fit else "")
            ),
        )
        leakage_path = plot_state_leakage(
            validation_fields,
            state_gaps,
            suffix=("_curvilinear_expanded" if args.expanded_fit else ""),
        )
        print(json.dumps(history[-1], indent=2), flush=True)
        print(cut_path.with_suffix(".png"), flush=True)
        print(leakage_path.with_suffix(".png"), flush=True)
        return

    converged = (
        history[-1]["maximum_hamiltonian_error_hartree"] <= 7.5e-4
        and history[-1]["relative_link_error"] <= 2.0e-2
    )
    if not converged:
        raise RuntimeError(
            "refusing dynamics: the single-patch MACE-Y fit did not pass "
            "independent continuous validation; use a multipatch Y atlas or "
            "a larger tracked electronic manifold"
        )
    if (
        fit.energy is None
        or fit.feature is None
        or fit.energy is fit.neural_energy
        or fit.feature is fit.neural_feature
    ):
        fit.distill_y(rank=64, degree=12, method="grid", seed=19)
        fit.save(checkpoint)
    distilled_validation = assess_models(fit.energy, fit.feature, validation_fields)
    if (
        distilled_validation["maximum_hamiltonian_error_hartree"] > 7.5e-4
        or distilled_validation["relative_link_error"] > 2.0e-2
    ):
        raise RuntimeError(
            "refusing dynamics: the distilled FTT fields did not pass "
            "independent continuous validation"
        )

    started = perf_counter()
    nuclear_keo = keo.podolsky().bind(coord, grid=grid, molecule=mol)
    direct = sampler.direct_product(
        grid,
        keo=nuclear_keo,
        workers=6,
        progress=True,
        energy_shift=sampler.energy_shift,
    )
    direct_build_seconds = perf_counter() - started
    fitted = TNLDR(
        fit,
        grid=grid,
        coord=coord,
        keo=nuclear_keo,
        overlap_rank=40,
        operator_rank=128,
    ).build()

    envelope = harmonic_envelope(grid)
    anchor = tuple(int(np.argmin(np.abs(axis))) for axis in grid.x)
    direct_packet = direct.wavepacket(envelope, state=1, anchor=anchor)
    working_packet = fitted_working_packet(fit, grid, 1)
    fitted_packet = fitted.state(working_packet, max_rank=64, physical=False)
    projectors = tuple(
        fitted.adiabatic_projector(state, method="dense", max_rank=32)[0]
        for state in range(2)
    )
    profiles = caps(grid)
    direct_cap = cap_tensor(grid, profiles)
    dt = dt_fs / au2fs
    steps = round(tmax_fs / dt_fs)

    started = perf_counter()
    direct.run(
        direct_packet,
        dt=dt,
        nsteps=steps,
        nout=nout,
        matrix_free=True,
        absorber=direct_cap,
    )
    direct_seconds = perf_counter() - started
    populations_direct = direct_populations(direct)

    started = perf_counter()
    fitted.run(
        fitted_packet,
        dt=dt,
        steps=steps,
        interval=nout,
        max_bond=96,
        integrator="tdvp2",
        cutoff=1.0e-11,
        e_ops=projectors,
        absorber=profiles,
        normalize=False,
        progress=False,
    )
    fitted_seconds = perf_counter() - started

    fine_keo = keo.podolsky().bind(coord, grid=fine_grid, molecule=mol)
    fine = TNLDR(
        fit,
        grid=fine_grid,
        coord=coord,
        keo=fine_keo,
        overlap_rank=40,
        operator_rank=128,
    ).build()
    fine_working = fitted_working_packet(fit, fine_grid, 1)
    fine_packet = fine.state(fine_working, max_rank=64, physical=False)
    fine_projectors = tuple(
        fine.adiabatic_projector(state, method="dense", max_rank=32)[0]
        for state in range(2)
    )
    fine_profiles = caps(fine_grid)
    started = perf_counter()
    fine.run(
        fine_packet,
        dt=dt,
        steps=steps,
        interval=nout,
        max_bond=96,
        integrator="tdvp2",
        cutoff=1.0e-11,
        e_ops=fine_projectors,
        absorber=fine_profiles,
        normalize=False,
        progress=False,
    )
    fine_seconds = perf_counter() - started

    time_fs = direct.times * au2fs
    population_error = np.abs(fitted.populations - populations_direct)
    grid_error = np.abs(fine.populations - fitted.populations)
    edge = edge_probability(direct.states)
    summary = {
        "electronic_reference": "spin-pure singlet full CI (2e, 27o)/aug-cc-pVDZ",
        "electronic_multiplicity": 1,
        "electronic_roots_solved": 6,
        "selected_singlet_roots": [1, 2],
        "subspace_diagnostics": subspace,
        "states": [1, 2],
        "equilibrium_bond_bohr": equilibrium,
        "initial_probability_covariance_bohr2": covariance.tolist(),
        "initial_preparation": "harmonic S0 vibrational ground state vertically promoted to S2",
        "bounds_bohr": [list(value) for value in bounds],
        "direct_grid": list(grid.shape),
        "fine_tnldr_grid": list(fine_grid.shape),
        "time_fs": tmax_fs,
        "dt_fs": dt_fs,
        "cap": {
            "strength_hartree": 0.08,
            "start_in_initial_sigma": 3.5,
            "axes": [0, 1, 2],
        },
        "gauge": "anchor-Procrustes",
        "fit": "S3-equivariant MACE (H,Y) distilled to FTT",
        "sampling_measure": (
            "one canonical S3 representative per harmonic/CAP-accessible orbit "
            "plus outer shell"
        ),
        "sampling_symmetry": {
            "group": sampler.group,
            "coord_repr": sampler.coord_repr.tolist(),
            "state_repr": sampler.state_repr.tolist(),
            "validation": sampler.symmetry_validation,
            "state_validation": sampler.state_validation,
        },
        "full_domain_initial_sample_budget": int(full_domain_initial_budget),
        "requested_initial_training_points": int(requested_training_points),
        "symmetry_reduced_initial_training_points": int(
            reduced_initial_training_points
        ),
        "requested_validation_pairs": requested_validation_pairs,
        "unitarize_links": False,
        "training_points": int(fit.info["energy_samples"]),
        "adaptive_history": history,
        "adaptive_converged": True,
        "distilled_validation": distilled_validation,
        "distillation": fit.info.get("distillation", {}),
        "maximum_population_error_vs_direct": float(np.max(population_error)),
        "maximum_coarse_to_fine_tnldr_population_change": float(np.max(grid_error)),
        "final_populations_direct": populations_direct[-1].tolist(),
        "final_populations_tnldr_coarse": fitted.populations[-1].tolist(),
        "final_populations_tnldr_fine": fine.populations[-1].tolist(),
        "final_survival_direct": float(direct.norm[-1]),
        "final_survival_tnldr_coarse": float(fitted.norms[-1]),
        "final_survival_tnldr_fine": float(fine.norms[-1]),
        "maximum_tnldr_coarse_absorption_closure": float(
            np.max(np.abs(fitted.absorption_closure))
        ),
        "maximum_tnldr_fine_absorption_closure": float(
            np.max(np.abs(fine.absorption_closure))
        ),
        "final_tnldr_coarse_tdvp_truncation_error": float(
            fitted.tdvp_truncation_errors[-1]
        ),
        "final_tnldr_fine_tdvp_truncation_error": float(
            fine.tdvp_truncation_errors[-1]
        ),
        "maximum_direct_edge_probability": float(np.max(edge)),
        "final_direct_edge_probability": float(edge[-1]),
        "direct_database_writes": int(direct.direct_product_info["database_writes"]),
        "database": str(database),
        "timings_seconds": {
            "fit": fit_seconds,
            "direct_build": direct_build_seconds,
            "direct_propagation": direct_seconds,
            "tnldr_coarse_propagation": fitted_seconds,
            "tnldr_fine_propagation": fine_seconds,
        },
    }
    (output / "physical_dynamics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.savez(
        output / "h3plus_fci_physical_dynamics.npz",
        time_fs=time_fs,
        direct_populations=populations_direct,
        tnldr_coarse_populations=fitted.populations,
        tnldr_fine_populations=fine.populations,
        direct_norms=direct.norm,
        tnldr_coarse_norms=fitted.norms,
        tnldr_fine_norms=fine.norms,
        direct_edge_probability=edge,
        direct_states=direct.states,
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
    colors = ("#0072B2", "#D55E00")
    figure, panels = plt.subplots(1, 3, figsize=(9.4, 2.8), constrained_layout=True)
    for state, color in enumerate(colors):
        panels[0].plot(
            time_fs, populations_direct[:, state], color=color,
            label=fr"Direct $S_{state + 1}$",
        )
        panels[0].plot(
            time_fs, fitted.populations[:, state], "--", color=color,
            label=fr"TNLDR-{grid_points} $S_{state + 1}$",
        )
        panels[0].plot(
            time_fs, fine.populations[:, state], ":", color=color,
            label=fr"TNLDR-{fine_points} $S_{state + 1}$",
        )
    panels[0].set(
        xlabel="time (fs)", ylabel="adiabatic population",
        title="(a) Population dynamics", ylim=(-0.02, 1.02),
    )
    panels[0].legend(frameon=False, ncol=2)
    panels[1].plot(time_fs, direct.norm, color="0.15", label="Direct")
    panels[1].plot(
        time_fs, fitted.norms, "--", color="#009E73",
        label=f"TNLDR-{grid_points}",
    )
    panels[1].plot(
        time_fs, fine.norms, ":", color="#CC79A7",
        label=f"TNLDR-{fine_points}",
    )
    panels[1].set(
        xlabel="time (fs)", ylabel="survival probability",
        title="(b) Three-axis CAP",
    )
    panels[1].legend(frameon=False)
    panels[2].semilogy(
        time_fs, np.maximum(np.max(population_error, axis=1), 1.0e-15),
        color="#7A3E9D", label=f"TNLDR-{grid_points} vs direct",
    )
    panels[2].semilogy(
        time_fs, np.maximum(np.max(grid_error, axis=1), 1.0e-15),
        color="#E69F00", label=f"TNLDR {grid_points} vs {fine_points}",
    )
    panels[2].set(
        xlabel="time (fs)", ylabel="maximum population difference",
        title="(c) Method/grid checks",
    )
    panels[2].legend(frameon=False)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(direction="out")
    dynamics_path = output / "h3plus_fci_physical_population"
    figure.savefig(dynamics_path.with_suffix(".pdf"))
    figure.savefig(dynamics_path.with_suffix(".png"), dpi=360)
    plt.close(figure)

    selected_times = np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    selected = [int(np.argmin(np.abs(time_fs - value))) for value in selected_times]
    snapshot_figure, snapshot_panels = plt.subplots(
        2, 3, figsize=(8.0, 5.1), constrained_layout=True, sharex=True, sharey=True
    )
    for panel, index in zip(snapshot_panels.flat, selected):
        density = np.sum(np.abs(direct.states[index]) ** 2, axis=(0, 3))
        maximum = max(float(np.max(density)), np.finfo(float).tiny)
        image = panel.pcolormesh(
            grid.x[1], grid.x[2], (density / maximum).T,
            shading="auto", cmap="magma", vmin=0.0, vmax=1.0,
        )
        panel.set_title(
            fr"$t={time_fs[index]:.1f}$ fs, $N={direct.norm[index]:.3f}$"
        )
        panel.set_aspect("equal")
    for panel in snapshot_panels[-1]:
        panel.set_xlabel(r"$Q_x$ (bohr)")
    for panel in snapshot_panels[:, 0]:
        panel.set_ylabel(r"$Q_y$ (bohr)")
    snapshot_figure.colorbar(
        image, ax=snapshot_panels, label=r"$\rho(Q_x,Q_y;t)/\rho_{\max}(t)$",
        shrink=0.82,
    )
    snapshot_path = output / "h3plus_fci_physical_wavepacket_snapshots"
    snapshot_figure.savefig(snapshot_path.with_suffix(".pdf"))
    snapshot_figure.savefig(snapshot_path.with_suffix(".png"), dpi=360)
    plt.close(snapshot_figure)

    cut_axis = np.linspace(-0.55, 0.55, 19)
    cut_coordinates = np.column_stack(
        (np.zeros_like(cut_axis), cut_axis, np.zeros_like(cut_axis))
    )
    raw_levels = np.linalg.eigvalsh(
        sampler.continuous_fields(cut_coordinates)["hamiltonians"]
    )
    dense_axis = np.linspace(cut_axis[0], cut_axis[-1], 401)
    dense_coordinates = np.column_stack(
        (np.zeros_like(dense_axis), dense_axis, np.zeros_like(dense_axis))
    )
    mace_levels = np.linalg.eigvalsh(fit.neural_energy.predict(dense_coordinates))
    ftt_levels = np.linalg.eigvalsh(fit.energy.predict(dense_coordinates))
    origin = float(np.min(raw_levels))
    cut_figure, cut_panel = plt.subplots(figsize=(4.2, 3.1), constrained_layout=True)
    for state, color in enumerate(colors):
        cut_panel.plot(
            dense_axis, (mace_levels[:, state] - origin) * au2ev,
            color=color, label=fr"MACE $S_{state + 1}$",
        )
        cut_panel.plot(
            dense_axis, (ftt_levels[:, state] - origin) * au2ev,
            "--", color=color, label=fr"FTT $S_{state + 1}$",
        )
        cut_panel.scatter(
            cut_axis, (raw_levels[:, state] - origin) * au2ev,
            s=18, facecolor="white", edgecolor=color, linewidth=0.8,
            label=fr"FCI $S_{state + 1}$", zorder=3,
        )
    cut_panel.set(
        xlabel=r"branching coordinate $Q_x$ (bohr)",
        ylabel="energy relative to cut minimum (eV)",
        title=r"FCI and fitted $Q_s=Q_y=0$ cut",
    )
    cut_panel.legend(frameon=False, ncol=2)
    cut_panel.spines[["top", "right"]].set_visible(False)
    cut_panel.tick_params(direction="out")
    cut_path = output / "h3plus_fci_physical_pes_cut"
    cut_figure.savefig(cut_path.with_suffix(".pdf"))
    cut_figure.savefig(cut_path.with_suffix(".png"), dpi=360)
    plt.close(cut_figure)

    print(json.dumps(summary, indent=2), flush=True)
    for path in (dynamics_path, snapshot_path, cut_path):
        print(path.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
