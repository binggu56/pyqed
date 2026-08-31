#!/usr/bin/env python3
"""Select and synchronize four SO2 states inside an eight-root product grid."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.ldr.overlap import unpack


def polar(matrix):
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right, singular


def neighbors(index, shape, links, axis_order):
    for axis in axis_order:
        if index[axis] + 1 < shape[axis]:
            right = list(index)
            right[axis] += 1
            yield tuple(right), links[axis, index]
        if index[axis] > 0:
            left = list(index)
            left[axis] -= 1
            left = tuple(left)
            yield left, links[axis, left].conj().T


def select_subspaces(shape, links, anchor, candidate_states, selected_states, axis_order):
    isometries = np.empty((*shape, candidate_states, selected_states), complex)
    visited = np.zeros(shape, dtype=bool)
    isometries[anchor] = np.eye(candidate_states, selected_states)
    visited[anchor] = True
    queue = deque((anchor,))
    transport_singular = []
    while queue:
        index = queue.popleft()
        for neighbor, overlap in neighbors(index, shape, links, axis_order):
            if visited[neighbor]:
                continue
            block = isometries[index].conj().T @ overlap
            transport, singular = polar(block)
            isometries[neighbor] = transport.conj().T
            transport_singular.append(singular)
            visited[neighbor] = True
            queue.append(neighbor)
    if not np.all(visited):
        raise RuntimeError("nearest-neighbor grid graph is disconnected")
    return isometries, np.asarray(transport_singular)


def project_links(shape, links, isometries):
    output = {}
    for index in np.ndindex(shape):
        for axis in range(len(shape)):
            if index[axis] + 1 >= shape[axis]:
                continue
            right = list(index)
            right[axis] += 1
            right = tuple(right)
            output[axis, index] = (
                isometries[index].conj().T
                @ links[axis, index]
                @ isometries[right]
            )
    return output


def synchronize(shape, links, isometries, anchor, *, max_cycle=100, rtol=1.0e-10):
    nstates = isometries.shape[-1]
    transports = {}
    weights = {}
    for key, block in links.items():
        transports[key], singular = polar(block)
        weights[key] = max(float(singular[-1]), np.finfo(float).tiny)
    identity_gauges = np.broadcast_to(
        np.eye(nstates, dtype=complex), (*shape, nstates, nstates)
    ).copy()
    gauges = identity_gauges.copy()

    def residual(values):
        errors = []
        for (axis, index), transport in transports.items():
            right = list(index)
            right[axis] += 1
            right = tuple(right)
            errors.append(
                np.linalg.norm(
                    values[index].conj().T @ transport @ values[right]
                    - np.eye(nstates)
                )
                / np.sqrt(nstates)
            )
        return np.asarray(errors)

    before = residual(gauges)
    best_gauges = gauges.copy()
    best_residual = before
    best_cycle = 0
    history = []
    for cycle in range(int(max_cycle)):
        previous = gauges.copy()
        for index in np.ndindex(shape):
            if index == anchor:
                continue
            average = np.zeros((nstates, nstates), complex)
            total = 0.0
            for axis in range(len(shape)):
                if index[axis] + 1 < shape[axis]:
                    right = list(index)
                    right[axis] += 1
                    right = tuple(right)
                    key = (axis, index)
                    average += weights[key] * transports[key] @ previous[right]
                    total += weights[key]
                if index[axis] > 0:
                    left = list(index)
                    left[axis] -= 1
                    left = tuple(left)
                    key = (axis, left)
                    average += (
                        weights[key] * transports[key].conj().T @ previous[left]
                    )
                    total += weights[key]
            gauges[index] = polar(average / total)[0]
        gauges[anchor] = np.eye(nstates)
        change = float(np.max(np.abs(gauges - previous)))
        history.append(change)
        if (cycle + 1) % 5 == 0 or change <= float(rtol):
            current_residual = residual(gauges)
            if float(np.mean(current_residual)) < float(np.mean(best_residual)):
                best_gauges = gauges.copy()
                best_residual = current_residual
                best_cycle = cycle + 1
        if change <= float(rtol):
            break
    gauges = best_gauges
    after = best_residual
    accepted = best_cycle > 0
    if not accepted:
        gauges = identity_gauges
        after = before
    synchronized = isometries @ gauges
    info = {
        "cycles": len(history),
        "converged": bool(history and history[-1] <= float(rtol)),
        "last_change": history[-1] if history else 0.0,
        "best_cycle": best_cycle,
        "mean_residual_before": float(np.mean(before)),
        "mean_residual_after": float(np.mean(after)),
        "max_residual_before": float(np.max(before)),
        "max_residual_after": float(np.max(after)),
        "accepted": accepted,
    }
    return synchronized, info


def link_arrays(shape, links):
    output = []
    nstates = next(iter(links.values())).shape[-1]
    for axis, size in enumerate(shape):
        edge_shape = list(shape)
        edge_shape[axis] = size - 1
        values = np.empty((*edge_shape, nstates, nstates), complex)
        for index in np.ndindex(tuple(edge_shape)):
            values[index] = links[axis, index]
        output.append(values)
    return tuple(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate",
        type=Path,
        nargs="?",
        default=Path(
            "/private/tmp/so2_cas8candidate_631gstar_9x9x9/"
            "electronic_reference.npz"
        ),
    )
    parser.add_argument(
        "--old-four-state",
        type=Path,
        default=Path(
            "/private/tmp/so2_cas4state_three_patch_9x9x9/procrustes_gauge.npz"
        ),
    )
    parser.add_argument("--selected-states", type=int, default=4)
    parser.add_argument("--axis-order", default="0,2,1")
    parser.add_argument("--sync-cycles", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/so2_cas8_to_4_selected_9x9x9"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    axis_order = tuple(int(value) for value in args.axis_order.split(","))

    with np.load(args.candidate, allow_pickle=False) as archive:
        energies = np.asarray(archive["energies"])
        spin_square = np.asarray(archive["spin_square"])
        grids = tuple(np.asarray(archive[name]) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    shape = energies.shape[:-1]
    candidate_states = energies.shape[-1]
    selected_states = int(args.selected_states)
    if sorted(axis_order) != list(range(len(shape))):
        raise ValueError("axis-order must be a permutation of the grid axes")
    if not 0 < selected_states <= candidate_states:
        raise ValueError("selected-states must lie within the candidate space")
    anchor = tuple(size // 2 for size in shape)

    isometries, tree_singular = select_subspaces(
        shape,
        links,
        anchor,
        candidate_states,
        selected_states,
        axis_order,
    )
    projected = project_links(shape, links, isometries)
    isometries, sync_info = synchronize(
        shape, projected, isometries, anchor, max_cycle=args.sync_cycles
    )
    projected = project_links(shape, links, isometries)
    selected_links = link_arrays(shape, projected)
    shifted = energies - float(np.min(energies))
    selected_energy = np.einsum(
        "...ia,...i,...ib->...ab",
        isometries.conj(),
        shifted,
        isometries,
        optimize=True,
    )
    candidate_singular = np.asarray(
        [np.linalg.svd(block, compute_uv=False) for block in links.values()]
    )
    selected_singular = np.concatenate(
        [
            np.linalg.svd(values, compute_uv=False).reshape(-1, selected_states)
            for values in selected_links
        ]
    )
    selected_axis_minima = [
        float(np.min(np.linalg.svd(values, compute_uv=False)[..., -1]))
        for values in selected_links
    ]
    worst_axis = int(np.argmin(selected_axis_minima))
    worst_values = np.linalg.svd(
        selected_links[worst_axis], compute_uv=False
    )[..., -1]
    worst_index = tuple(
        map(int, np.unravel_index(np.argmin(worst_values), worst_values.shape))
    )
    isometry_defect = np.max(
        np.abs(
            np.einsum("...ia,...ib->...ab", isometries.conj(), isometries)
            - np.eye(selected_states)
        )
    )

    with np.load(args.old_four_state, allow_pickle=False) as archive:
        old_energy = np.asarray(archive["aligned_local_hamiltonian"])
        old_links = tuple(np.asarray(archive[f"link_{axis}"]) for axis in range(3))
    old_singular = np.concatenate(
        [
            np.linalg.svd(values, compute_uv=False).reshape(-1, selected_states)
            for values in old_links
        ]
    )
    center = anchor
    selected_theta = selected_energy[center[0], :, center[2]]
    old_theta = old_energy[center[0], :, center[2]]
    selected_theta_links = selected_links[1][center[0], :, center[2]]
    old_theta_links = old_links[1][center[0], :, center[2]]
    theta = np.rad2deg(grids[1])
    edge_theta = 0.5 * (theta[:-1] + theta[1:])

    summary = {
        "method": f"{candidate_states}-candidate to {selected_states}-state graph transport",
        "grid": list(shape),
        "anchor": list(anchor),
        "axis_order": list(axis_order),
        "max_abs_s2": float(np.max(np.abs(spin_square))),
        "tree_min_rectangular_singular": float(np.min(tree_singular)),
        "candidate_min_square_link_singular": float(np.min(candidate_singular)),
        "old_four_state_min_link_singular": float(np.min(old_singular)),
        "selected_min_link_singular": float(np.min(selected_singular)),
        "selected_axis_min_link_singular": selected_axis_minima,
        "selected_worst_link": {"axis": worst_axis, "index": list(worst_index)},
        "selected_isometry_defect": float(isometry_defect),
        "central_theta_old_max_abs_e14_eh": float(np.max(np.abs(old_theta[:, 0, 3]))),
        "central_theta_selected_max_abs_e14_eh": float(
            np.max(np.abs(selected_theta[:, 0, 3]))
        ),
        "synchronization": sync_info,
    }
    np.savez(
        args.output_dir / "selected_reference.npz",
        candidate_energies=energies,
        spin_square=spin_square,
        isometries=isometries,
        aligned_local_hamiltonian=selected_energy,
        qs=grids[0],
        theta=grids[1],
        qa=grids[2],
        center=np.asarray(center),
        **{f"link_{axis}": values for axis, values in enumerate(selected_links)},
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(8.7, 2.8), constrained_layout=True)
    axes[0].plot(
        theta,
        np.abs(old_theta[:, 0, 3]),
        color="#D55E00",
        marker="o",
        lw=1.3,
        label="4-state atlas",
    )
    axes[0].plot(
        theta,
        np.abs(selected_theta[:, 0, 3]),
        color="#0072B2",
        marker="s",
        lw=1.3,
        label=f"{candidate_states}$\\to${selected_states} selected",
    )
    axes[1].plot(
        edge_theta,
        np.linalg.svd(old_theta_links, compute_uv=False)[:, -1],
        color="#D55E00",
        marker="o",
        lw=1.3,
        label="4-state atlas",
    )
    axes[1].plot(
        edge_theta,
        np.linalg.svd(selected_theta_links, compute_uv=False)[:, -1],
        color="#0072B2",
        marker="s",
        lw=1.3,
        label=f"{candidate_states}$\\to${selected_states} selected",
    )
    bins = np.geomspace(
        max(float(np.min(old_singular[:, -1])), 1.0e-15),
        max(float(np.max(selected_singular[:, -1])), 1.0),
        36,
    )
    axes[2].hist(
        old_singular[:, -1], bins=bins, histtype="step", lw=1.3, color="#D55E00", label="4-state atlas"
    )
    axes[2].hist(
        selected_singular[:, -1], bins=bins, histtype="step", lw=1.3, color="#0072B2", label=f"{candidate_states}$\\to${selected_states} selected"
    )
    axes[0].set(xlabel=r"$\theta$ (degree)", ylabel=r"$|\bar E_{14}|$ ($E_h$)")
    axes[1].set(
        xlabel=r"Link midpoint $\theta$ (degree)",
        ylabel="Minimum link singular value",
        yscale="log",
    )
    axes[2].set(
        xlabel="Minimum link singular value",
        ylabel="Count",
        xscale="log",
    )
    for label, axis in zip("abc", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.legend(frameon=False, fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
        axis.grid(axis="y", color="0.9", lw=0.6)
    figure_path = args.output_dir / "so2_selected_state_grid.png"
    figure.savefig(figure_path, dpi=400, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
