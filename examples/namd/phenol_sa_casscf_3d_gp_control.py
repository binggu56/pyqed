#!/usr/bin/env python3
"""Compare geometric-phase-on/off phenol dynamics on the fitted S1 adiabat."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if os.environ.get("PYQED_NO_MATPLOTLIB") == "1":
    plt = None
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.functional import FunctionalTT
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2ev, au2fs

from examples.namd.phenol_sa_casscf_3d_ftt_ttldr import (
    _jsonable,
    _parallel_transport_phase,
    build_dvrs,
    cap_operators,
    cap_profile,
    cumulative_cap_yield,
    kinetic_terms,
    vibrational_ground_state,
)


HARTREE_TO_EV = au2ev
COLORS = {"gp-on": "#0072B2", "gp-off": "#D55E00"}


def product_coordinates(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=1)


def endpoint_links(feature_values, eigenvectors):
    """Return scalar adiabatic links for every forward product-grid edge."""
    shape = eigenvectors.shape[:-2]
    state_vector = eigenvectors[..., :, 1]
    links = []
    for axis in range(len(shape)):
        left = [slice(None)] * len(shape)
        right = [slice(None)] * len(shape)
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        frame_link = np.einsum(
            "...ra,...rb->...ab",
            feature_values[tuple(left)].conj(),
            feature_values[tuple(right)],
            optimize=True,
        )
        scalar = np.einsum(
            "...a,...ab,...b->...",
            state_vector[tuple(left)].conj(),
            frame_link,
            state_vector[tuple(right)],
            optimize=True,
        )
        links.append(scalar)
    return tuple(links)


def maximum_spanning_tree_gauge(links, shape, anchor):
    """Smooth scalar-link phases along a maximum-overlap spanning tree."""
    shape = tuple(map(int, shape))
    anchor = tuple(map(int, anchor))
    phase = np.zeros(shape, dtype=complex)
    phase[anchor] = 1.0
    visited = np.zeros(shape, dtype=bool)
    visited[anchor] = True
    pending = []

    def push(point):
        for axis in range(len(shape)):
            if point[axis] + 1 < shape[axis]:
                neighbor = list(point)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if not visited[neighbor]:
                    heapq.heappush(
                        pending,
                        (-abs(links[axis][point]), point, neighbor, links[axis][point]),
                    )
            if point[axis] > 0:
                neighbor = list(point)
                neighbor[axis] -= 1
                neighbor = tuple(neighbor)
                if not visited[neighbor]:
                    heapq.heappush(
                        pending,
                        (
                            -abs(links[axis][neighbor]),
                            point,
                            neighbor,
                            links[axis][neighbor].conjugate(),
                        ),
                    )

    push(anchor)
    tree_minimum = np.inf
    while pending:
        negative_weight, parent, child, link = heapq.heappop(pending)
        if visited[child]:
            continue
        transported = phase[parent].conjugate() * link
        if abs(transported) <= 1.0e-14:
            raise RuntimeError("the adiabatic overlap graph is disconnected")
        phase[child] = transported.conjugate() / abs(transported)
        tree_minimum = min(tree_minimum, -negative_weight)
        visited[child] = True
        push(child)
    if not np.all(visited):
        raise RuntimeError("the adiabatic overlap graph is disconnected")

    gauged = []
    for axis, values in enumerate(links):
        left = [slice(None)] * len(shape)
        right = [slice(None)] * len(shape)
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        gauged.append(
            phase[tuple(left)].conjugate() * values * phase[tuple(right)]
        )
    return phase, tuple(gauged), float(tree_minimum)


def rectangular_loop_phase(links, lower, upper):
    """Return the phase of a scalar-link loop in the first two coordinates."""
    i0, j0, k = map(int, lower)
    i1, j1, k1 = map(int, upper)
    if k1 != k:
        raise ValueError("the diagnostic loop must lie on one bend cut")
    product = 1.0 + 0.0j
    minimum = 1.0
    for i in range(i0, i1):
        value = links[0][i, j0, k]
        product *= value / abs(value)
        minimum = min(minimum, abs(value))
    for j in range(j0, j1):
        value = links[1][i1, j, k]
        product *= value / abs(value)
        minimum = min(minimum, abs(value))
    for i in range(i1 - 1, i0 - 1, -1):
        value = links[0][i, j1, k].conjugate()
        product *= value / abs(value)
        minimum = min(minimum, abs(value))
    for j in range(j1 - 1, j0 - 1, -1):
        value = links[1][i0, j, k].conjugate()
        product *= value / abs(value)
        minimum = min(minimum, abs(value))
    return float(np.angle(product)), float(minimum)


def fit_scalar_fields(axes, energy, links, output, *, energy_rank, link_rank, label):
    common = {
        "bounds": tuple((float(axis[0]), float(axis[-1])) for axis in axes),
        "normalization": "frobenius",
    }
    energy_model = FunctionalTT(
        degrees=tuple(len(axis) - 1 for axis in axes),
        rank=int(energy_rank),
        hermitian=True,
        **common,
    ).fit_grid(axes, energy[..., None, None])
    energy_path = output / f"{label}_energy.npz"
    energy_model.save(energy_path)
    link_models = []
    diagnostics = []
    for axis, values in enumerate(links):
        edge_axes = list(axes)
        edge_axes[axis] = 0.5 * (edge_axes[axis][:-1] + edge_axes[axis][1:])
        model = FunctionalTT(
            degrees=tuple(len(grid) - 1 for grid in edge_axes),
            rank=int(link_rank),
            bounds=tuple((float(grid[0]), float(grid[-1])) for grid in edge_axes),
            normalization="frobenius",
            hermitian=False,
        ).fit_grid(edge_axes, values[..., None, None])
        coordinates = product_coordinates(edge_axes)
        predicted = model.predict(coordinates).reshape(values.shape)
        difference = np.abs(predicted - values)
        path = output / f"{label}_link_axis{axis}.npz"
        model.save(path)
        link_models.append(model)
        diagnostics.append(
            {
                "axis": axis,
                "ranks": list(map(int, model.ranks_)),
                "absolute_rms": float(np.sqrt(np.mean(difference**2))),
                "absolute_max": float(np.max(difference)),
                "relative_frobenius": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(values), np.finfo(float).tiny)
                ),
                "path": str(path),
            }
        )
    fit = SimpleNamespace(
        success=True,
        grids=tuple(axes),
        energy=energy_model,
        links=tuple(link_models),
        feature=None,
    )
    return fit, diagnostics, str(energy_path)


def run_control(label, fit, axes, dvrs, initial_scalar, args):
    driver = TTLDR.from_fit(
        fit,
        grids=axes,
        keo=kinetic_terms(dvrs),
        overlap_rank=args.overlap_rank,
        potential_rank=args.potential_rank,
        operator_rank=args.operator_rank,
        fitted_kinetic_backend="link-mpo",
        energy_shift=float(args.energy_shift),
    )
    profile = cap_profile(axes[0], args.cap_start, args.cap_strength, args.cap_order)
    cap, channels = cap_operators(axes, profile, nstates=1)
    driver.components = (*driver.components, (-1.0j) * cap)
    driver._hamiltonian = None
    driver.is_hermitian = False
    initial = initial_scalar[..., None]
    state = driver.state(initial, max_rank=args.state_rank)
    observables = (*driver.projectors(), *channels)
    driver.run(
        state,
        dt=float(args.time_fs / args.steps / au2fs),
        steps=args.steps,
        interval=args.interval,
        max_bond=args.state_rank,
        integrator="tdvp2",
        cutoff=args.cutoff,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        normalize=False,
        progress=args.progress,
        e_ops=observables,
    )
    final = driver.dense(driver.final_state)[..., 0]
    radial = np.sum(np.abs(final) ** 2, axis=(1, 2))
    expectations = np.asarray(driver.populations)[:, 1:2]
    yields = cumulative_cap_yield(driver.times, expectations)[:, 0]
    absorbed = 1.0 - np.asarray(driver.norms)
    closure = yields - absorbed
    return {
        "label": label,
        "times_fs": np.asarray(driver.times) * au2fs,
        "norms": np.asarray(driver.norms),
        "absorbed": absorbed,
        "cap_yield": yields,
        "cap_expectation": expectations[:, 0],
        "final_radial": radial,
        "final_absorbed_probability": float(absorbed[-1]),
        "maximum_absorption_closure_defect": float(np.max(np.abs(closure))),
        "final_state_ranks": list(map(int, driver.final_state.bond_orders())),
        "operator_ranks": driver.operator_ranks,
    }


def plot_results(output, axes, results, phases):
    figure, panels = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for result in results:
        label = result["label"]
        color = COLORS[label]
        panels[0, 0].plot(
            result["times_fs"], 100.0 * result["absorbed"], color=color, label=label
        )
        panels[0, 1].plot(
            result["times_fs"], 100.0 * result["cap_yield"], color=color, label=label
        )
        panels[1, 0].plot(
            axes[0], result["final_radial"], color=color, label=label
        )
    panels[0, 0].set(xlabel="time (fs)", ylabel="norm loss (%)", title="Projected $S_1$ survival")
    panels[0, 1].set(xlabel="time (fs)", ylabel="integrated CAP flux (%)", title="Dissociation flux")
    panels[1, 0].set(xlabel=r"$R_{OH}$ ($\AA$)", ylabel="radial probability", title="Final radial distribution")
    names = ("inner $S_1/S_2$", "outer $S_0/S_1$")
    x = np.arange(len(names))
    panels[1, 1].bar(x - 0.18, np.abs(phases["gp-on"]) / np.pi, 0.36, color=COLORS["gp-on"], label="gp-on")
    panels[1, 1].bar(x + 0.18, np.abs(phases["gp-off"]) / np.pi, 0.36, color=COLORS["gp-off"], label="gp-off")
    panels[1, 1].set(xticks=x, xticklabels=names, ylabel=r"$|\gamma|/\pi$", ylim=(0.0, 1.08), title="Connection control")
    for panel in panels.flat:
        panel.grid(alpha=0.2)
        panel.legend(frameon=False)
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
    png = output / "phenol_gp_on_off_200fs.png"
    pdf = output / "phenol_gp_on_off_200fs.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def nearest(axis, value):
    return int(np.argmin(np.abs(np.asarray(axis) - float(value))))


def sine_dvr_bounds(axis):
    """Recover Dirichlet box boundaries from uniformly spaced interior nodes."""
    axis = np.asarray(axis, dtype=float)
    spacing = np.diff(axis)
    if len(spacing) == 0 or not np.allclose(spacing, spacing[0]):
        raise ValueError("phenol control requires uniform sine-DVR nodes")
    return float(axis[0] - spacing[0]), float(axis[-1] + spacing[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_ftt_inward_20260821/phenol_sa6_3d_ftt_ttldr.npz"),
    )
    parser.add_argument(
        "--energy",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_ftt_inward_20260821/energy_rank40.npz"),
    )
    parser.add_argument(
        "--feature",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_ftt_inward_20260821/feature_rank40.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_gp_control_200fs_20260822"),
    )
    parser.add_argument("--time-fs", type=float, default=200.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--state-rank", type=int, default=40)
    parser.add_argument("--energy-rank", type=int, default=40)
    parser.add_argument("--link-rank", type=int, default=64)
    parser.add_argument("--overlap-rank", type=int, default=32)
    parser.add_argument("--potential-rank", type=int, default=32)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--cap-start", type=float, default=2.45)
    parser.add_argument("--cap-strength", type=float, default=0.02)
    parser.add_argument("--cap-order", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reference = np.load(args.reference)
    axes = tuple(np.asarray(reference[name]) for name in ("r_oh", "phi", "theta"))
    bounds = np.asarray([sine_dvr_bounds(axis) for axis in axes])
    rebuilt_axes, dvrs = build_dvrs(*(len(axis) for axis in axes), bounds)
    if any(not np.allclose(left, right) for left, right in zip(axes, rebuilt_axes)):
        raise RuntimeError("the reference axes are incompatible with the phenol DVR builder")

    coordinates = product_coordinates(axes)
    energy_model = FunctionalTT.load(args.energy)
    feature_model = FunctionalTT.load(args.feature)
    shape = tuple(len(axis) for axis in axes)
    hamiltonian = energy_model.predict(coordinates).reshape(*shape, 3, 3)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.swapaxes(-1, -2).conj())
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    feature_shape = tuple(feature_model.output_shape_)
    features = feature_model.predict(coordinates).reshape(*shape, *feature_shape)
    raw_links = endpoint_links(features, eigenvectors)
    anchor = tuple(
        nearest(axis, value)
        for axis, value in zip(axes, (0.96994, 0.0, np.deg2rad(108.8)))
    )
    phase, signed_links, tree_minimum = maximum_spanning_tree_gauge(
        raw_links, shape, anchor
    )
    positive_links = tuple(np.abs(values).astype(complex) for values in signed_links)

    k = nearest(axes[2], np.deg2rad(108.8))
    loops = {
        "inner": (
            (nearest(axes[0], 1.02), nearest(axes[1], -0.15), k),
            (nearest(axes[0], 1.25), nearest(axes[1], 0.15), k),
        ),
        "outer": (
            (nearest(axes[0], 1.75), nearest(axes[1], -0.15), k),
            (nearest(axes[0], 1.98), nearest(axes[1], 0.15), k),
        ),
    }
    phase_on = np.asarray(
        [rectangular_loop_phase(signed_links, *loops[name])[0] for name in ("inner", "outer")]
    )
    phase_off = np.asarray(
        [rectangular_loop_phase(positive_links, *loops[name])[0] for name in ("inner", "outer")]
    )
    if not np.allclose(np.abs(phase_on), np.pi, atol=5.0e-3):
        raise RuntimeError(f"signed control lost the fitted Berry phase: {phase_on}")
    if not np.allclose(phase_off, 0.0, atol=1.0e-10):
        raise RuntimeError(f"positive-link control retained a loop phase: {phase_off}")

    ground_energy, nuclear, ground_residual = vibrational_ground_state(
        tuple(dvr.t() for dvr in dvrs), energies[..., 0]
    )
    smooth_electronic = _parallel_transport_phase(
        eigenvectors[..., :, 1], anchor
    )
    scalar_basis = eigenvectors[..., :, 1] * phase[..., None]
    initial_scalar = nuclear * np.einsum(
        "...a,...a->...", scalar_basis.conj(), smooth_electronic, optimize=True
    )
    initial_scalar /= np.linalg.norm(initial_scalar)
    args.energy_shift = float(np.min(energies[..., 1]))
    fits = {}
    fit_info = {}
    for label, values in (("gp-on", signed_links), ("gp-off", positive_links)):
        fits[label], links_info, energy_path = fit_scalar_fields(
            axes,
            energies[..., 1],
            values,
            args.output,
            energy_rank=args.energy_rank,
            link_rank=args.link_rank,
            label=label,
        )
        fit_info[label] = {"links": links_info, "energy": energy_path}

    results = []
    for label in ("gp-on", "gp-off"):
        print(f"[{label}] starting projected S1 propagation", flush=True)
        result = run_control(label, fits[label], axes, dvrs, initial_scalar, args)
        results.append(result)
        print(
            f"[{label}] absorbed={result['final_absorbed_probability']:.6e}, "
            f"closure={result['maximum_absorption_closure_defect']:.3e}",
            flush=True,
        )

    phases = {"gp-on": phase_on, "gp-off": phase_off}
    png, pdf = plot_results(args.output, axes, results, phases)
    np.savez_compressed(
        args.output / "phenol_gp_on_off_200fs.npz",
        r_oh=axes[0],
        phi=axes[1],
        theta=axes[2],
        initial_nuclear=nuclear,
        initial_scalar=initial_scalar,
        phase_on=phase_on,
        phase_off=phase_off,
        **{
            f"{key}_{result['label'].replace('-', '_')}": result[key]
            for result in results
            for key in ("times_fs", "norms", "absorbed", "cap_yield", "final_radial")
        },
    )
    summary = {
        "passed": bool(
            all(item["maximum_absorption_closure_defect"] <= 5.0e-3 for item in results)
            and all(
                entry["absolute_max"] <= 5.0e-3
                and entry["relative_frobenius"] <= 5.0e-3
                for info in fit_info.values()
                for entry in info["links"]
            )
        ),
        "control": "same fitted S1 adiabatic energy; signed versus magnitude-only scalar links",
        "grid_shape": shape,
        "time_fs": args.time_fs,
        "steps": args.steps,
        "anchor": anchor,
        "minimum_spanning_tree_link_magnitude": tree_minimum,
        "loop_phases_radian": phases,
        "ground_vibrational_energy_hartree": ground_energy,
        "ground_eigenpair_residual": ground_residual,
        "energy_shift_hartree": args.energy_shift,
        "fits": fit_info,
        "results": results,
        "figure": str(png),
        "figure_pdf": str(pdf),
        "data": str(args.output / "phenol_gp_on_off_200fs.npz"),
    }
    (args.output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
