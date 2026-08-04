"""Compare exact, cMPS, and cLETTA Luttinger density correlations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ContinuousMPS,
    ExponentialLuttingerModel,
    cmps_luttinger_density_correlation,
    cmps_luttinger_energy_shift_density,
    optimize_luttinger_cletta,
)


METHODS = (
    ("cmps_d2", "cMPS $D=2$"),
    ("cmps_d3", "cMPS $D=3$"),
    ("cletta_d2_m1", r"cLETTA $D=2,M=1,L=4$"),
    ("cletta_d3_m1", r"cLETTA $D=3,M=1,L=4$"),
)


def _optimize_states(args, model):
    d2 = optimize_luttinger_cletta(
        model,
        bond_dim=2,
        num_modes=0,
        restarts=args.restarts,
        seed=args.seed + 20,
        maxiter=args.maxiter,
        quadrature_points=args.quadrature_points,
    )
    d3 = optimize_luttinger_cletta(
        model,
        bond_dim=3,
        num_modes=0,
        seed_states=[d2],
        restarts=args.restarts,
        seed=args.seed + 30,
        maxiter=args.maxiter,
        quadrature_points=args.quadrature_points,
    )
    cletta = optimize_luttinger_cletta(
        model,
        bond_dim=2,
        num_modes=1,
        depth=1,
        seed_states=[d2],
        restarts=args.restarts,
        seed=args.seed + 21,
        maxiter=args.maxiter,
        quadrature_points=args.quadrature_points,
    )
    cletta_d3 = optimize_luttinger_cletta(
        model,
        bond_dim=3,
        num_modes=1,
        depth=1,
        seed_states=[d3, cletta],
        restarts=args.restarts,
        seed=args.seed + 31,
        maxiter=args.maxiter,
        quadrature_points=args.quadrature_points,
    )
    states = {
        "cmps_d2": d2,
        "cmps_d3": d3,
        "cletta_d2_m1": cletta,
        "cletta_d3_m1": cletta_d3,
    }
    for state in states.values():
        state.energy = cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=args.validation_quadrature_points,
        )
    return states


def _save_states(path, states):
    arrays = {}
    for key, state in states.items():
        arrays[f"{key}_q"] = state.q
        arrays[f"{key}_r"] = state.r
        arrays[f"{key}_energy"] = state.energy
        arrays[f"{key}_success"] = bool(state.success)
        arrays[f"{key}_nfev"] = int(state.nfev)
    np.savez(path, **arrays)


def _load_states(path):
    archive = np.load(path)
    states = {}
    for key, _label in METHODS:
        state = ContinuousMPS(archive[f"{key}_q"], archive[f"{key}_r"])
        state.energy = float(archive[f"{key}_energy"])
        state.success = bool(archive[f"{key}_success"])
        state.nfev = int(archive[f"{key}_nfev"])
        states[key] = state
    return states


def _load_depth_converged_state(path, depth):
    archive = np.load(path)
    state = ContinuousMPS(
        archive[f"L{depth}_q"],
        archive[f"L{depth}_r"],
    )
    state.success = True
    return state


def _free_correlation(distance, cutoff):
    inverse_cutoff = 1.0 / float(cutoff)
    return (
        inverse_cutoff**2 - distance**2
    ) / (
        2.0
        * np.pi**2
        * (inverse_cutoff**2 + distance**2) ** 2
    )


def _relative_l2_error(distance, values, reference):
    numerator = np.trapezoid(np.abs(values - reference) ** 2, distance)
    denominator = np.trapezoid(np.abs(reference) ** 2, distance)
    return float(np.sqrt(numerator / denominator))


def _write_data(path, distance, exact, free, correlations):
    with Path(path).open("w", newline="") as handle:
        fieldnames = ["distance", "exact", "free"] + [key for key, _ in METHODS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, value in enumerate(distance):
            row = {
                "distance": value,
                "exact": exact[index],
                "free": free[index],
            }
            row.update({key: correlations[key][index] for key, _ in METHODS})
            writer.writerow(row)


def _plot(args, distance, exact, free, correlations, errors):
    import ultraplot as uplt
    from matplotlib.ticker import LogFormatterMathtext

    colors = {
        "exact": "#202124",
        "cmps_d2": "#0072B2",
        "cmps_d3": "#009E73",
        "cletta_d2_m1": "#D55E00",
        "cletta_d3_m1": "#CC79A7",
    }
    styles = {
        "cmps_d2": "--",
        "cmps_d3": "-.",
        "cletta_d2_m1": "-",
        "cletta_d3_m1": ":",
    }
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "tick.labelsize": 10.5,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        ncols=2,
        refwidth=3.35,
        refheight=2.75,
        share=False,
        wspace=4.2,
    )

    delta_exact = exact - free
    axes[0].plot(distance, delta_exact, color=colors["exact"], label="exact")
    for key, label in METHODS:
        axes[0].plot(
            distance,
            correlations[key] - free,
            color=colors[key],
            linestyle=styles[key],
            label=label,
        )
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].format(
        xlabel=r"distance $r$",
        ylabel=r"$\Delta C(r)=C(r)-C_0(r)$",
        xlim=(0.0, args.distance_max),
        title="Density correlation",
        grid=False,
    )

    floor = 1.0e-10
    for key, label in METHODS:
        axes[1].semilogy(
            distance,
            np.maximum(np.abs(correlations[key] - exact), floor),
            color=colors[key],
            linestyle=styles[key],
            label=rf"{label}: $\epsilon_C={errors[key]:.2e}$",
        )
    axes[1].format(
        xlabel=r"distance $r$",
        ylabel=r"$|C(r)-C_{\rm exact}(r)|$",
        xlim=(0.0, args.distance_max),
        ylim=(1.0e-7, 5.0e-3),
        title="Pointwise error",
        grid=False,
    )
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="b",
        ncols=3,
        frame=False,
    )
    axes[0].text(
        0.0,
        1.045,
        "a",
        transform=axes[0].transAxes,
        fontsize=13,
        fontweight="bold",
    )
    axes[1].text(
        0.0,
        1.045,
        "b",
        transform=axes[1].transAxes,
        fontsize=13,
        fontweight="bold",
    )

    output = Path(args.figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=400)


def run(args):
    model = ExponentialLuttingerModel(
        decay_rates=[1.0],
        strengths=[args.coupling],
        fermi_velocity=1.0,
    )
    cache = Path(args.state_cache)
    if cache.exists() and not args.force:
        states = _load_states(cache)
    else:
        states = _optimize_states(args, model)
        cache.parent.mkdir(parents=True, exist_ok=True)
        _save_states(cache, states)
    states["cletta_d2_m1"] = _load_depth_converged_state(
        args.cletta_d2_depth_cache,
        args.cletta_depth,
    )
    states["cletta_d3_m1"] = _load_depth_converged_state(
        args.cletta_d3_depth_cache,
        args.cletta_depth,
    )
    for key in ("cletta_d2_m1", "cletta_d3_m1"):
        states[key].energy = cmps_luttinger_energy_shift_density(
            model,
            states[key],
            quadrature_points=args.validation_quadrature_points,
        )

    distance = np.linspace(0.0, args.distance_max, args.distance_points)
    exact = model.density_correlation(
        distance,
        uv_cutoff=args.uv_cutoff,
        points=args.correlation_points,
    )
    free = _free_correlation(distance, args.uv_cutoff)
    correlations = {
        key: cmps_luttinger_density_correlation(
            state,
            distance,
            uv_cutoff=args.uv_cutoff,
            points=args.correlation_points,
        )
        for key, state in states.items()
    }
    delta_exact = exact - free
    errors = {
        key: _relative_l2_error(
            distance,
            values - free,
            delta_exact,
        )
        for key, values in correlations.items()
    }
    exact_energy = model.ground_state_energy_shift_density()[0]
    for key, label in METHODS:
        state = states[key]
        print(
            f"{label}: E={state.energy:.12f}, "
            f"dE={state.energy - exact_energy:.3e}, "
            f"epsilon_C={errors[key]:.3e}, success={state.success}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_data(output, distance, exact, free, correlations)
    _plot(args, distance, exact, free, correlations, errors)
    return states, errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=8.0)
    parser.add_argument("--uv-cutoff", type=float, default=8.0)
    parser.add_argument("--distance-max", type=float, default=6.0)
    parser.add_argument("--distance-points", type=int, default=301)
    parser.add_argument("--correlation-points", type=int, default=16000)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--quadrature-points", type=int, default=180)
    parser.add_argument("--validation-quadrature-points", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cletta-depth", type=int, default=4)
    parser.add_argument(
        "--state-cache",
        default="/private/tmp/nonlocal_luttinger_correlation_states_g8.npz",
    )
    parser.add_argument(
        "--cletta-d2-depth-cache",
        default="/private/tmp/nonlocal_luttinger_depth_convergence_d2_g8.npz",
    )
    parser.add_argument(
        "--cletta-d3-depth-cache",
        default="/private/tmp/nonlocal_luttinger_depth34_g8.npz",
    )
    parser.add_argument(
        "--output",
        default="/private/tmp/nonlocal_luttinger_correlations_g8.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/nonlocal_luttinger_correlations_g8.pdf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
