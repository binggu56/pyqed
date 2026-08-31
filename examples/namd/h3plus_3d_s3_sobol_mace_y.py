#!/usr/bin/env python3
"""Scalable S3-reduced Sobol CASCI -> MACE-Y -> FTT -> TTLDR benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ultraplot as uplt
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from examples.namd.h3plus_3d_mace_ftt_ttldr import (
    TARGET_STATES,
    align_external_anchor_sign,
    anchor_aligned_fields,
    axes_from_cache,
    evaluate_mace,
    field_errors,
    generate_cache,
    geometry,
    h3plus_s3_group,
    load_cache,
    nuclear_marginal,
    plot_results,
    product_coordinates,
    run_dynamics,
    save_cache,
    symmetry_errors,
)
from examples.namd.h3plus_casci_positive_link_fit import selected_blocks
from pyqed.ldr.overlap import procrustes
from pyqed.ml import MACE
from pyqed.qchem.mcscf.casci import overlap
from examples.namd.h3plus_3d_mace_ftt_ttldr import electronic_point


def reduce_distortion_to_s3_wedge(qx, qy):
    radius = np.hypot(qx, qy)
    angle = np.mod(np.arctan2(qy, qx), 2.0 * np.pi / 3.0)
    angle = np.where(angle > np.pi / 3.0, 2.0 * np.pi / 3.0 - angle, angle)
    return radius * np.cos(angle), radius * np.sin(angle)


def sobol_representatives(count, qmin, qmax, *, seed=37):
    """Return a nested prefix containing the fixed anchor and one point per S3 orbit."""

    count = int(count)
    if count < 4:
        raise ValueError("at least four symmetry-inequivalent samples are required")
    needed = count - 2
    power = int(np.ceil(np.log2(needed)))
    unit = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(power)[:needed]
    values = float(qmin) + (float(qmax) - float(qmin)) * unit
    qx, qy = reduce_distortion_to_s3_wedge(values[:, 1], values[:, 2])
    sobol = np.column_stack((values[:, 0], qx, qy))
    scale = max(abs(float(qmin)), abs(float(qmax)))
    calibration = np.asarray([0.0, 0.46 * scale, 0.19 * scale])
    return np.vstack((np.zeros(3), calibration, sobol))


def calibration_orbit(representatives):
    coordinate_group = h3plus_s3_group(2, np.diag([1.0, -1.0]))[
        "coordinate_representations"
    ]
    base = np.asarray(representatives[1], dtype=float)
    return np.einsum("gij,j->gi", coordinate_group, base, optimize=True)


def dataset_coordinates(count, qmin, qmax, *, seed=37):
    representatives = sobol_representatives(count, qmin, qmax, seed=seed)
    orbit = calibration_orbit(representatives)
    coordinates = np.vstack((representatives[0], orbit, representatives[2:]))
    training_indices = np.concatenate(
        (np.asarray([0, 1], dtype=int), np.arange(7, len(coordinates), dtype=int))
    )
    if len(training_indices) != int(count):
        raise RuntimeError("internal Sobol/calibration indexing error")
    return coordinates, training_indices


def sparse_overlap_graph(coordinates, *, neighbors=3, scale_reference=None):
    coordinates = np.asarray(coordinates, dtype=float)
    reference = (
        coordinates
        if scale_reference is None
        else np.asarray(scale_reference, dtype=float)
    )
    span = np.ptp(reference, axis=0)
    span[span < 1.0e-12] = 1.0
    scaled = (coordinates - reference.mean(axis=0)) / span
    distances = cdist(scaled, scaled)
    finite = distances.copy()
    np.fill_diagonal(finite, 0.0)
    tree = minimum_spanning_tree(finite).tocoo()
    pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    np.fill_diagonal(distances, np.inf)
    neighbors = min(max(int(neighbors), 1), len(coordinates) - 1)
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], neighbors - 1)[:neighbors]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    pairs = np.asarray(sorted(pairs), dtype=int)
    adjacency = np.zeros((len(coordinates), len(coordinates)), dtype=int)
    adjacency[pairs[:, 0], pairs[:, 1]] = 1
    adjacency += adjacency.T
    if connected_components(adjacency, directed=False)[0] != 1:
        raise RuntimeError("the sparse overlap graph is disconnected")
    lengths = np.linalg.norm(
        scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1
    )
    return pairs, lengths


def nested_sparse_overlap_graphs(coordinates, counts, *, neighbors=3):
    """Build cumulative $O(N)$ graphs without deleting an earlier edge."""

    coordinates = np.asarray(coordinates, dtype=float)
    counts = tuple(sorted(set(map(int, counts))))
    if not counts or counts[0] < 2 or counts[-1] > len(coordinates):
        raise ValueError("nested graph counts are incompatible with the coordinates")
    span = np.ptp(coordinates, axis=0)
    span[span < 1.0e-12] = 1.0
    scaled = (coordinates - coordinates.mean(axis=0)) / span
    initial, _lengths = sparse_overlap_graph(
        coordinates[: counts[0]],
        neighbors=neighbors,
        scale_reference=coordinates,
    )
    cumulative = {tuple(map(int, pair)) for pair in initial}
    graphs = {}
    previous = counts[0]
    for count in counts:
        for right in range(previous, count):
            distances = np.linalg.norm(scaled[:right] - scaled[right], axis=1)
            degree = min(max(int(neighbors), 1), right)
            nearest = np.argpartition(distances, degree - 1)[:degree]
            cumulative.update((int(left), right) for left in nearest)
        pairs = np.asarray(sorted(cumulative), dtype=int)
        lengths = np.linalg.norm(
            scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1
        )
        graphs[count] = (pairs, lengths)
        previous = count
    return graphs


def generate_scattered_cache(
    counts, qmin, qmax, *, basis, nroots, neighbors, seed, output
):
    counts = tuple(sorted(set(map(int, counts))))
    coordinates, training_indices = dataset_coordinates(
        counts[-1], qmin, qmax, seed=seed
    )
    points = []
    energies = []
    for index, coordinate in enumerate(coordinates, start=1):
        point = electronic_point(coordinate, basis=basis, nroots=nroots)
        points.append(point)
        energies.append(np.asarray(point.e_tot, dtype=float))
        if index == 1 or index % 8 == 0 or index == len(coordinates):
            print(f"[scattered CASCI] {index}/{len(coordinates)}", flush=True)
    reference_links = np.asarray([overlap(point, points[0]) for point in points])
    payload = {
        "coordinates": coordinates,
        "energies": np.asarray(energies),
        "reference_links": reference_links,
        "counts": np.asarray(counts, dtype=int),
        "qmin": np.asarray(qmin),
        "qmax": np.asarray(qmax),
        "seed": np.asarray(seed),
        "neighbors": np.asarray(neighbors),
        "basis": np.asarray(basis),
        "nroots": np.asarray(nroots),
        "target_states": np.asarray(TARGET_STATES),
    }
    training_coordinates = coordinates[training_indices]
    graphs = nested_sparse_overlap_graphs(
        training_coordinates, counts, neighbors=neighbors
    )
    for count in counts:
        local_global = training_indices[:count]
        pairs, lengths = graphs[count]
        values = np.asarray(
            [
                selected_blocks(overlap(points[local_global[left]], points[local_global[right]]))
                for left, right in pairs
            ]
        )
        payload[f"training_indices_{count}"] = local_global
        payload[f"pairs_{count}"] = pairs
        payload[f"lengths_{count}"] = lengths
        payload[f"links_{count}"] = values
    payload["nested_graph"] = np.asarray(True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return payload


def load_scattered_cache(filename):
    with np.load(filename) as archive:
        data = {key: archive[key] for key in archive.files}
    if tuple(np.asarray(data["target_states"], dtype=int)) != TARGET_STATES:
        raise ValueError("scattered cache has the wrong target-state manifold")
    return data


def aligned_scattered_data(data):
    coordinates = np.asarray(data["coordinates"], dtype=float)
    reference = selected_blocks(data["reference_links"])
    gauges, _positive, singular = procrustes(reference)
    energies = np.asarray(data["energies"], dtype=float)
    shift = float(np.mean(energies[0, list(TARGET_STATES)]))
    selected = energies[:, list(TARGET_STATES)] - shift
    hamiltonian = np.einsum(
        "nia,ni,nib->nab", gauges.conj(), selected, gauges, optimize=True
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.swapaxes(-1, -2).conj())
    return {
        "coordinates": coordinates,
        "hamiltonian": hamiltonian,
        "gauges": gauges,
        "energy_shift": shift,
        "minimum_reference_singular_value": float(np.min(singular)),
    }


def infer_calibrated_s3(hamiltonian, feature_rank):
    """Infer the arbitrary O(2) anchor orientation from one six-point orbit."""

    base = hamiltonian[1]
    rotated = hamiltonian[2]
    reflected = hamiltonian[4]
    identity = np.eye(2)

    def traceless(value):
        return value - 0.5 * np.trace(value) * identity

    base_t = traceless(base)
    reflected_t = traceless(reflected)

    def reflection_objective(angle):
        cosine, sine = np.cos(2.0 * angle), np.sin(2.0 * angle)
        representation = np.asarray([[cosine, sine], [sine, -cosine]])
        return float(np.linalg.norm(representation @ base_t @ representation - reflected_t))

    coarse = np.linspace(0.0, np.pi, 2048, endpoint=False)
    errors = np.asarray([reflection_objective(angle) for angle in coarse])
    center = coarse[int(np.argmin(errors))]
    spacing = np.pi / len(coarse)
    result = minimize_scalar(
        reflection_objective,
        bounds=(center - spacing, center + spacing),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    cosine, sine = np.cos(2.0 * result.x), np.sin(2.0 * result.x)
    reflection = np.asarray([[cosine, sine], [sine, -cosine]])
    angle = 2.0 * np.pi / 3.0
    choices = []
    for sign in (1.0, -1.0):
        value = sign * angle
        rotation = np.asarray(
            [[np.cos(value), -np.sin(value)], [np.sin(value), np.cos(value)]]
        )
        choices.append(
            (float(np.linalg.norm(rotation @ base @ rotation.T - rotated)), rotation)
        )
    rotation_error, rotation = min(choices, key=lambda item: item[0])
    scale = max(float(np.linalg.norm(base_t)), np.finfo(float).tiny)
    return h3plus_s3_group(feature_rank, reflection, rotation), {
        "reflection_relative_error": float(result.fun) / scale,
        "rotation_relative_error": float(rotation_error) / scale,
        "electronic_rotation": rotation.tolist(),
        "electronic_reflection": reflection.tolist(),
    }


def subset_targets(data, aligned, count):
    indices = np.asarray(data[f"training_indices_{count}"], dtype=int)
    pairs = np.asarray(data[f"pairs_{count}"], dtype=int)
    links = np.asarray(data[f"links_{count}"], dtype=complex)
    global_left = indices[pairs[:, 0]]
    global_right = indices[pairs[:, 1]]
    gauges = aligned["gauges"]
    links = np.einsum(
        "nia,nij,njb->nab",
        gauges[global_left].conj(),
        links,
        gauges[global_right],
        optimize=True,
    )
    return {
        "coordinates": aligned["coordinates"][indices],
        "hamiltonian": aligned["hamiltonian"][indices],
        "pairs": pairs,
        "links": links,
        "indices": indices,
        "lengths": np.asarray(data[f"lengths_{count}"], dtype=float),
    }


def fit_subset(
    targets, axes, group, args, *, distill, initial_fit=None, loss_scales=None
):
    return MACE(
        axes,
        ("H", "H", "H"),
        geometry,
        2,
        chart_features=True,
        geometry_units="bohr",
        channels=args.channels,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=6,
        radial_mlp=(args.head_width, args.head_width),
        cutoff=4.0,
    ).fit_y(
        (targets["coordinates"], targets["hamiltonian"]),
        targets["coordinates"],
        targets["pairs"],
        targets["links"],
        feature_rank=args.feature_rank,
        feature_objective="links-only",
        ambient_representation="full",
        energy_representation="direct",
        finite_group=group,
        hidden=(args.head_width, args.head_width),
        epochs=args.epochs,
        learning_rate=(
            args.learning_rate
            if initial_fit is None
            else args.warm_learning_rate
        ),
        weight_decay=1.0e-8,
        frame_fraction=0.35 if initial_fit is None else 0.0,
        ambient_fraction=0.20 if initial_fit is None else 0.0,
        smoothness=1.0e-5,
        sync_steps=args.sync_steps,
        loss_scales=loss_scales,
        initial_fit=initial_fit,
        seed=args.seed,
        distill=distill,
        tt_rank=args.tt_rank,
        tt_degree=args.tt_degree,
    )


def validation_metrics(fit, validation):
    prediction = evaluate_mace(fit, validation)
    aligned = align_external_anchor_sign(validation, prediction[0])
    return {
        "energy": field_errors(prediction[0], aligned["hamiltonian"], energy=True),
        "links": [
            field_errors(predicted, reference)
            for predicted, reference in zip(prediction[1], aligned["links"])
        ],
        "prediction": prediction,
        "aligned_validation": aligned,
    }


def plot_sampling(targets, all_coordinates, calibration_indices, records, output):
    style = {
        "font.family": "sans-serif", "font.size": 8.0,
        "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "axes.linewidth": 0.8,
        "lines.linewidth": 1.25, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, panels = plt.subplots(
            2, 2, figsize=(7.2, 5.25), constrained_layout=True
        )
        points = targets["coordinates"]
        scatter = panels[0, 0].scatter(
            points[:, 1], points[:, 2], c=points[:, 0], cmap="coolwarm",
            s=17, edgecolor="none",
        )
        calibration = all_coordinates[calibration_indices]
        panels[0, 0].scatter(
            calibration[:, 1], calibration[:, 2], marker="x", color="black",
            s=24, linewidth=0.9, label="symmetry calibration",
        )
        panels[0, 0].set(
            xlabel=r"$Q_x$ (bohr)", ylabel=r"$Q_y$ (bohr)",
            title=r"$S_3$-reduced Sobol design",
        )
        panels[0, 0].legend(frameon=False, loc="lower right")
        figure.colorbar(
            scatter, ax=panels[0, 0], label=r"$Q_s$ (bohr)", fraction=0.05
        )

        for left, right in targets["pairs"]:
            panels[0, 1].plot(
                points[[left, right], 1], points[[left, right], 2],
                color="0.76", lw=0.42, zorder=1,
            )
        panels[0, 1].scatter(
            points[:, 1], points[:, 2], color="#0072B2", s=13,
            edgecolor="white", linewidth=0.25, zorder=2,
        )
        panels[0, 1].set(
            xlabel=r"$Q_x$ (bohr)", ylabel=r"$Q_y$ (bohr)",
            title=f"Cumulative graph ({len(targets['pairs'])} links)",
        )

        samples = np.asarray([record["ab_initio_samples"] for record in records])
        energy = np.asarray([record["energy_rms_mev"] for record in records])
        panels[1, 0].plot(samples, energy, "o-", color="#D55E00", ms=4)
        for x, value in zip(samples, energy):
            panels[1, 0].annotate(
                f"{value:.2f}", (x, value), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=6.5,
            )
        panels[1, 0].set(
            xlabel="CASCI geometries", ylabel="Held-out energy RMS (meV)",
            title="Nested-sample convergence",
        )
        labels = (r"$L_s$", r"$L_x$", r"$L_y$")
        colors = ("#0072B2", "#D55E00", "#009E73")
        for axis, (label, color) in enumerate(zip(labels, colors)):
            values = [record["link_relative_errors"][axis] for record in records]
            panels[1, 1].plot(samples, values, "o-", color=color, ms=3.5, label=label)
        panels[1, 1].set_yscale("log")
        panels[1, 1].set(
            xlabel="CASCI geometries", ylabel="Held-out relative link error",
            title=r"Endpoint-field validation",
        )
        panels[1, 1].legend(frameon=False, ncol=3, loc="lower right")
        for label, panel in zip("abcd", panels.flat):
            panel.text(
                -0.16, 1.04, label, transform=panel.transAxes,
                va="bottom", fontweight="bold", fontsize=9,
            )
            panel.tick_params(direction="in", length=3, width=0.7)
            panel.spines[["top", "right"]].set_visible(False)
        output = Path(output)
        for suffix, options in (
            (output.suffix, {"dpi": 600}), (".pdf", {}), (".svg", {})
        ):
            figure.savefig(output.with_suffix(suffix), bbox_inches="tight", **options)
        plt.close(figure)


def plot_long_dynamics(fields, dynamics, output):
    """Create separate UltraPlot figures for populations and observables."""

    reference = np.asarray(dynamics["reference_states"])
    predicted = np.asarray(dynamics["predicted_states"])
    left = reference.reshape(len(reference), -1)
    right = predicted.reshape(len(predicted), -1)
    fidelity = np.abs(np.sum(left.conj() * right, axis=1)) ** 2
    fidelity /= np.sum(np.abs(left) ** 2, axis=1) * np.sum(
        np.abs(right) ** 2, axis=1
    )
    reference_population = dynamics["reference_adiabatic_populations"]
    predicted_population = dynamics["predicted_adiabatic_populations"]
    population_error = np.max(
        np.abs(predicted_population - reference_population),
        axis=1,
    )
    tt_final_population_error = float(
        np.max(
            np.abs(
                dynamics["tt_final_adiabatic_populations"]
                - reference_population[-1]
            )
        )
    )
    output = Path(output)
    root = output.stem
    for suffix in ("_long_dynamics", "_dynamics"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break

    def save_figure(figure, label):
        filename = output.with_name(root + f"_{label}" + output.suffix)
        for suffix, options in (
            (filename.suffix, {"dpi": 600}), (".pdf", {}), (".svg", {})
        ):
            figure.savefig(filename.with_suffix(suffix), **options)
        uplt.close(figure)
        return filename

    time = dynamics["times_fs"]
    colors = ("#0072B2", "#D55E00")
    figure, panels = uplt.subplots(refwidth=3.55, refheight=2.5)
    panel = panels[0]
    for state, color in enumerate(colors):
        panel.plot(
            time, reference_population[:, state], color=color,
            lw=1.1, alpha=0.75, zorder=1,
            label=rf"Reference $S_{state + 1}$",
        )
        panel.plot(
            time, predicted_population[:, state], color=color,
            ls=(0, (5.0, 2.4)), lw=2.2, zorder=3,
            label=rf"This work $S_{state + 1}$",
        )
        panel.plot(
            time[-1], dynamics["tt_final_adiabatic_populations"][state],
            ls="none", marker="o", ms=4.0, mfc="white", mec=color,
            mew=0.9,
        )
    panel.format(
        xlabel="Time (fs)", ylabel="Adiabatic population",
        ylim=(-0.02, 1.02), title=r"H$_3^+$ nonadiabatic population transfer",
        tickdir="in", grid=False,
    )
    panel.legend(frame=False, ncols=2, loc="best")
    population_path = save_figure(figure, "populations")

    reference_observables = dynamics["reference_observables"]
    predicted_observables = dynamics["predicted_observables"]
    coordinate_colors = ("#0072B2", "#D55E00", "#009E73")
    coordinate_labels = (r"$Q_s$", r"$Q_x$", r"$Q_y$")
    figure, panels = uplt.subplots(
        nrows=2, ncols=2, refwidth=2.55, refheight=1.75,
        share=False,
    )
    for axis, (color, label) in enumerate(
        zip(coordinate_colors, coordinate_labels)
    ):
        panels[0].plot(
            time, reference_observables["coordinate_means"][:, axis],
            color=color, lw=1.0, alpha=0.72, zorder=1, label=label,
        )
        panels[0].plot(
            time, predicted_observables["coordinate_means"][:, axis],
            color=color, ls=(0, (5.0, 2.4)), lw=2.0, zorder=3,
        )
        panels[1].plot(
            time, reference_observables["coordinate_widths"][:, axis],
            color=color, lw=1.0, alpha=0.72, zorder=1, label=label,
        )
        panels[1].plot(
            time, predicted_observables["coordinate_widths"][:, axis],
            color=color, ls=(0, (5.0, 2.4)), lw=2.0, zorder=3,
        )
    panels[0].format(
        xlabel="Time (fs)", ylabel=r"$\langle Q\rangle$ (bohr)",
        title="Nuclear centroids",
    )
    panels[1].format(
        xlabel="Time (fs)", ylabel=r"$\sigma_Q$ (bohr)",
        title="Wavepacket widths",
    )
    for panel in panels[:2]:
        panel.legend(frame=False, ncols=3, loc="best")
    panels[0].plot(
        [], [], color="black", lw=1.0, alpha=0.72, label="Reference",
    )
    panels[0].plot(
        [], [], color="black", ls=(0, (5.0, 2.4)), lw=2.0,
        label="This work",
    )
    panels[0].legend(frame=False, ncols=3, loc="best")
    panels[2].plot(
        time, reference_observables["electronic_coherence"],
        color="black", lw=1.0, alpha=0.72, zorder=1, label="Reference",
    )
    panels[2].plot(
        time, predicted_observables["electronic_coherence"],
        color="#D55E00", ls=(0, (5.0, 2.4)), lw=2.0, zorder=3,
        label="This work",
    )
    panels[2].format(
        xlabel="Time (fs)", ylabel=r"$|\rho^{\mathrm{ad}}_{12}|$",
        title="Electronic coherence",
    )
    panels[3].plot(
        time, reference_observables["electronic_purity"],
        color="black", lw=1.0, alpha=0.72, zorder=1, label="Reference",
    )
    panels[3].plot(
        time, predicted_observables["electronic_purity"],
        color="#D55E00", ls=(0, (5.0, 2.4)), lw=2.0, zorder=3,
        label="This work",
    )
    panels[3].format(
        xlabel="Time (fs)", ylabel=r"$\mathrm{Tr}(\rho_\mathrm{e}^2)$",
        title="Electronic purity",
    )
    for panel in panels:
        panel.format(tickdir="in", grid=False)
    panels[2].legend(frame=False, loc="best")
    panels[3].legend(frame=False, loc="best")
    panels.format(abc="a", abcloc="ul")
    observables_path = save_figure(figure, "observables")

    figure, panels = uplt.subplots(
        ncols=3, refwidth=2.25, refheight=2.0, share=False,
    )
    panels[0].semilogy(
        time, np.maximum(1.0 - fidelity, 1.0e-14),
        color="black", label=r"$1-F_\mathrm{dense}$",
    )
    panels[0].semilogy(
        time, np.maximum(population_error, 1.0e-14),
        color="#D55E00", ls=(0, (5.0, 2.4)), lw=2.0,
        label=r"This work $\max|\Delta P|$",
    )
    panels[0].plot(
        time[-1], max(
            1.0 - dynamics["ttldr_final_fidelity_to_reference"], 1.0e-14
        ),
        marker="o", ls="none", ms=4.0, mfc="white", mec="#0072B2",
        mew=0.9, label=r"TTLDR final $1-F$",
    )
    panels[0].format(
        xlabel="Time (fs)", ylabel="Error", title="Propagation error",
    )
    panels[0].legend(frame=False, loc="best")
    final_states = (
        dynamics["reference_states"][-1],
        dynamics["predicted_states"][-1],
        dynamics["tt_final"],
    )
    methods = ("Reference", "This work", "TTLDR")
    method_colors = ("black", "#D55E00", "#0072B2")
    styles = ("-", (0, (5.0, 2.4)), "none")
    for panel, axis, title in (
        (panels[1], 0, r"Final $Q_s$ marginal"),
        (panels[2], 1, r"Final $Q_x$ marginal"),
    ):
        coordinate = fields["axes"][axis]
        for state, method, color, linestyle in zip(
            final_states, methods, method_colors, styles
        ):
            marginal = nuclear_marginal(state, axis)
            panel.plot(
                coordinate, marginal, color=color, ls=linestyle,
                lw=2.0 if linestyle != "-" else 1.0,
                marker="o" if linestyle == "none" else None,
                ms=4.0, mfc="white", mec=color, mew=0.9, label=method,
            )
        panel.format(
            xlabel=rf"${('Q_s', 'Q_x')[axis]}$ (bohr)",
            ylabel="Marginal probability", title=title,
        )
        panel.legend(frame=False, loc="best")
    panels.format(abc="a", abcloc="ul", tickdir="in", grid=False)
    validation_path = save_figure(figure, "validation")
    return {
        "minimum_dense_fidelity": float(np.min(fidelity)),
        "final_dense_fidelity": float(fidelity[-1]),
        "maximum_population_error": float(np.max(population_error)),
        "final_ttldr_population_error": tt_final_population_error,
        "population_figure": str(population_path),
        "observables_figure": str(observables_path),
        "validation_figure": str(validation_path),
    }


def plot_distillation_comparison(grid_errors, cross_errors, cross_info, output):
    """Compare full-grid and MACE-oracle cross compilation on common probes."""

    style = {
        "font.family": "sans-serif", "font.size": 8.0,
        "axes.labelsize": 8.0, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, panels = plt.subplots(
            1, 2, figsize=(7.2, 2.55), constrained_layout=True
        )
        labels = (r"$\bar E$", r"$Y$", r"$Y^\dagger Y-I$")
        positions = np.arange(len(labels))
        width = 0.34
        panels[0].bar(
            positions - width / 2, grid_errors, width,
            color="#999999", label="grid TT-SVD",
        )
        panels[0].bar(
            positions + width / 2, cross_errors, width,
            color="#0072B2", label="MACE-oracle TT-cross",
        )
        panels[0].set_yscale("log")
        panels[0].set_xticks(positions, labels)
        panels[0].set(
            ylabel="Relative error on common probes",
            title="Continuous-field distillation",
        )
        panels[0].legend(frameon=False)

        fields = ("energy", "feature")
        queries = np.asarray(
            [cross_info[field]["geometry_queries"] for field in fields]
        )
        full = np.asarray(
            [cross_info[field]["full_grid_geometries"] for field in fields]
        )
        positions = np.arange(2)
        panels[1].bar(
            positions - width / 2, queries, width,
            color="#009E73", label="queried by TT-cross",
        )
        panels[1].bar(
            positions + width / 2, full, width,
            facecolor="none", edgecolor="black", linewidth=0.8,
            label="full candidate grid",
        )
        for x, value, total in zip(positions, queries, full):
            panels[1].annotate(
                f"{value}/{total}", (x - width / 2, value),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=6.5,
            )
        panels[1].set_xticks(positions, (r"$\bar E$", r"$Y$"))
        panels[1].set(
            ylabel="MACE geometry evaluations",
            title="Adaptive oracle queries",
        )
        panels[1].legend(frameon=False)
        for label, panel in zip("ab", panels):
            panel.text(
                -0.14, 1.04, label, transform=panel.transAxes,
                va="bottom", fontweight="bold", fontsize=9,
            )
            panel.tick_params(direction="in", length=3, width=0.7)
            panel.spines[["top", "right"]].set_visible(False)
        output = Path(output)
        for suffix, options in (
            (output.suffix, {"dpi": 600}), (".pdf", {}), (".svg", {})
        ):
            figure.savefig(output.with_suffix(suffix), bbox_inches="tight", **options)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=(18, 30, 48))
    parser.add_argument("--qmin", type=float, default=-0.12)
    parser.add_argument("--qmax", type=float, default=0.12)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2400)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--warm-learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--feature-rank", type=int, default=6)
    parser.add_argument("--sync-steps", type=int, default=500)
    parser.add_argument("--tt-rank", type=int, default=16)
    parser.add_argument("--tt-degree", type=int, default=8)
    parser.add_argument(
        "--tt-distill-method", choices=("cross", "grid"), default="cross"
    )
    parser.add_argument("--tt-cross-points", type=int, default=9)
    parser.add_argument("--tt-cross-sweeps", type=int, default=8)
    parser.add_argument("--tt-cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--dt-fs", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("/private/tmp/h3plus_s3_sobol_nested_48plus5.npz"),
    )
    parser.add_argument(
        "--reference-cache", type=Path,
        default=Path("/private/tmp/h3plus_centered_s3_casci_s1s2_5x5x5.npz"),
    )
    parser.add_argument(
        "--validation-cache", type=Path,
        default=Path("/private/tmp/h3plus_centered_s3_casci_s1s2_offgrid_4x4x4.npz"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/private/tmp/h3plus_3d_s3_sobol_mace_y"),
    )
    args = parser.parse_args()
    counts = tuple(sorted(set(args.counts)))
    if counts[0] < 4:
        raise ValueError("all sample counts must be at least four")
    if args.cache.exists() and not args.force:
        data = load_scattered_cache(args.cache)
        cached_counts = tuple(np.asarray(data["counts"], dtype=int))
        if cached_counts != counts:
            raise ValueError(f"cache counts {cached_counts} do not match {counts}")
        expected = {
            "qmin": args.qmin,
            "qmax": args.qmax,
            "seed": args.seed,
            "neighbors": args.neighbors,
            "nroots": 6,
        }
        for key, value in expected.items():
            if np.asarray(data[key]).item() != value:
                raise ValueError(
                    f"cache {key}={np.asarray(data[key]).item()} does not match {value}"
                )
        if np.asarray(data["basis"]).item() != args.basis:
            raise ValueError("cache basis does not match the requested basis")
        if not bool(np.asarray(data.get("nested_graph", False)).item()):
            raise ValueError("cache does not contain cumulative nested graphs")
        print(f"[cache] loaded {args.cache}", flush=True)
    else:
        data = generate_scattered_cache(
            counts, args.qmin, args.qmax, basis=args.basis, nroots=6,
            neighbors=args.neighbors, seed=args.seed, output=args.cache,
        )
        print(f"[cache] saved {args.cache}", flush=True)
    aligned = aligned_scattered_data(data)
    group, calibration = infer_calibrated_s3(
        aligned["hamiltonian"], args.feature_rank
    )

    axes = tuple(np.linspace(args.qmin, args.qmax, 5) for _ in range(3))
    if not args.reference_cache.exists():
        reference_data = generate_cache(axes, basis=args.basis, nroots=6)
        save_cache(args.reference_cache, reference_data)
    reference_data = load_cache(args.reference_cache)
    reference = anchor_aligned_fields(
        reference_data, energy_shift=aligned["energy_shift"]
    )
    for cached, requested in zip(axes_from_cache(reference_data), axes):
        np.testing.assert_allclose(cached, requested)
    centers = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in axes)
    if not args.validation_cache.exists():
        validation_data = generate_cache(
            centers, basis=args.basis, nroots=6, reference_coordinate=(0.0, 0.0, 0.0)
        )
        save_cache(args.validation_cache, validation_data)
    validation = anchor_aligned_fields(
        load_cache(args.validation_cache), energy_shift=aligned["energy_shift"]
    )

    records = []
    final_fit = None
    final_targets = None
    final_validation = None
    first_targets = subset_targets(data, aligned, counts[0])
    loss_scales = {
        "energy": float(np.mean(np.abs(first_targets["hamiltonian"]) ** 2)),
        "link": float(np.mean(np.abs(first_targets["links"]) ** 2)),
    }
    previous_fit = None
    for count in counts:
        targets = subset_targets(data, aligned, count)
        fit = fit_subset(
            targets,
            axes,
            group,
            args,
            distill=False,
            initial_fit=previous_fit,
            loss_scales=loss_scales,
        )
        result = validation_metrics(fit, validation)
        record = {
            "symmetry_inequivalent_training_nodes": int(count),
            "symmetry_calibration_overhead": 5,
            "ab_initio_samples": int(count + 5),
            "overlap_links": int(len(targets["pairs"])),
            "energy_rms_mev": 1000.0 * result["energy"]["rms"],
            "energy_max_mev": 1000.0 * result["energy"]["max_abs"],
            "link_relative_errors": [value["relative_frobenius"] for value in result["links"]],
            "minimum_graph_link_singular_value": float(
                np.min(np.linalg.svd(targets["links"], compute_uv=False))
            ),
            "final_loss": float(fit.history[-1]),
            "warm_started": bool(fit.info["warm_started"]),
        }
        records.append(record)
        print(f"[fit {count}] {json.dumps(record)}", flush=True)
        if count == counts[-1]:
            final_fit = fit
            final_targets = targets
            final_validation = result
        previous_fit = fit

    final_fit.distill_y(
        rank=args.tt_rank,
        degree=args.tt_degree,
        method=args.tt_distill_method,
        cross_points=args.tt_cross_points,
        cross_sweeps=args.tt_cross_sweeps,
        cross_rtol=args.tt_cross_rtol,
        seed=args.seed,
    )
    radius = 0.65 * min(abs(args.qmin), abs(args.qmax))
    angles = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    probes = np.asarray(
        [(qs, radius * np.cos(angle), radius * np.sin(angle))
         for qs in (-0.5 * radius, 0.0, 0.5 * radius) for angle in angles]
    )
    reference_prediction = final_fit.energy.predict(
        product_coordinates(axes)
    ).reshape(reference["hamiltonian"].shape)
    reference = align_external_anchor_sign(reference, reference_prediction)
    dynamics = run_dynamics(
        final_fit, reference, dt_fs=args.dt_fs, steps=args.steps,
        state_rank=args.state_rank, operator_rank=args.operator_rank,
    )
    final_metrics = {
        "counts": records,
        "calibration": calibration,
        "minimum_reference_singular_value": aligned["minimum_reference_singular_value"],
        "neural_s3_covariance": symmetry_errors(
            final_fit.neural_energy, final_fit.neural_feature, group, probes
        ),
        "ftt_s3_covariance": symmetry_errors(
            final_fit.energy, final_fit.feature, group, probes
        ),
        "ftt_distillation": final_fit.info["distillation"],
    }
    for key in (
        "hamiltonian_relative_error", "mace_ftt_final_fidelity",
        "ttldr_final_fidelity_to_reference",
        "ttldr_final_fidelity_to_predicted_dense", "maximum_ttldr_density_error",
    ):
        final_metrics[key] = dynamics[key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "h3plus_3d_s3_sobol_mace_y"
    final_fit.save(stem.with_suffix(".pt"))
    publication_figure = stem.with_name(stem.name + "_long_dynamics.png")
    final_metrics["long_time_dynamics"] = plot_long_dynamics(
        reference, dynamics, publication_figure
    )
    stem.with_suffix(".json").write_text(
        json.dumps(final_metrics, indent=2) + "\n"
    )
    np.savez(
        stem.with_suffix(".npz"),
        representative_coordinates=final_targets["coordinates"],
        graph_pairs=final_targets["pairs"],
        times_fs=dynamics["times_fs"],
        reference_populations=dynamics["reference_populations"],
        predicted_populations=dynamics["predicted_populations"],
        ttldr_populations=dynamics["tt_populations"],
        reference_adiabatic_populations=dynamics[
            "reference_adiabatic_populations"
        ],
        predicted_adiabatic_populations=dynamics[
            "predicted_adiabatic_populations"
        ],
        ttldr_final_adiabatic_populations=dynamics[
            "tt_final_adiabatic_populations"
        ],
        reference_coordinate_means=dynamics["reference_observables"][
            "coordinate_means"
        ],
        predicted_coordinate_means=dynamics["predicted_observables"][
            "coordinate_means"
        ],
        reference_coordinate_widths=dynamics["reference_observables"][
            "coordinate_widths"
        ],
        predicted_coordinate_widths=dynamics["predicted_observables"][
            "coordinate_widths"
        ],
        reference_electronic_density=dynamics["reference_observables"][
            "electronic_density"
        ],
        predicted_electronic_density=dynamics["predicted_observables"][
            "electronic_density"
        ],
        reference_electronic_coherence=dynamics["reference_observables"][
            "electronic_coherence"
        ],
        predicted_electronic_coherence=dynamics["predicted_observables"][
            "electronic_coherence"
        ],
        reference_electronic_purity=dynamics["reference_observables"][
            "electronic_purity"
        ],
        predicted_electronic_purity=dynamics["predicted_observables"][
            "electronic_purity"
        ],
        reference_autocorrelation=dynamics["reference_observables"][
            "autocorrelation"
        ],
        predicted_autocorrelation=dynamics["predicted_observables"][
            "autocorrelation"
        ],
    )
    sampling_figure = stem.with_name(stem.name + "_sampling.png")
    plot_sampling(
        final_targets,
        aligned["coordinates"],
        np.arange(1, 7),
        records,
        sampling_figure,
    )
    result_figure = stem.with_name(stem.name + "_dynamics.png")
    rank_figure = plot_results(
        final_fit, reference, final_validation["aligned_validation"],
        final_validation["prediction"], dynamics, result_figure,
    )
    print(json.dumps(final_metrics, indent=2), flush=True)
    print(f"sampling figure: {sampling_figure}", flush=True)
    print(f"publication dynamics figure: {publication_figure}", flush=True)
    print(f"dynamics figure: {result_figure}", flush=True)
    print(f"rank figure: {rank_figure}", flush=True)


if __name__ == "__main__":
    main()
