"""Parameter-matched cMPS/cLETTA benchmark for a Coulomb Luttinger liquid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.fft import dct

from pyqed.mps import (
    ContinuousMPS,
    CoulombLuttingerModel,
    canonical_parameter_size,
    cmps_luttinger_energy_shift_density,
    cmps_luttinger_parameter,
    optimize_luttinger_cletta,
)


def _state_path(directory, kind, bond_dim, depth=None, num_modes=1):
    suffix = (
        f"_M{num_modes}_L{depth}" if kind == "cletta" else ""
    )
    return Path(directory) / (
        f"coulomb_luttinger_{kind}_D{bond_dim}{suffix}.npz"
    )


def _save_cmps(path, state):
    np.savez(
        path,
        theta=state.theta,
        energy=state.energy,
        success=bool(state.success),
        nfev=int(state.nfev),
    )


def _load_cmps(path, bond_dim):
    archive = np.load(path)
    state = ContinuousMPS.from_canonical_parameters(
        archive["theta"],
        bond_dim,
    )
    state.energy = float(archive["energy"])
    state.success = bool(archive["success"])
    state.nfev = int(archive["nfev"])
    return state


def _save_cletta(path, state, depth):
    np.savez(
        path,
        base_theta=state.cletta_base.theta,
        tie=state.cletta_tie_matrices,
        rate=state.cletta_decay_rates,
        depth=int(depth),
        energy=state.energy,
        success=bool(state.success),
        nfev=int(state.nfev),
    )


def _load_cletta(path, bond_dim, depth):
    archive = np.load(path)
    base = ContinuousMPS.from_canonical_parameters(
        archive["base_theta"],
        bond_dim,
    )
    state = base.cletta_memory_state(
        archive["tie"],
        archive["rate"],
        depth=depth,
    )
    state.energy = float(archive["energy"])
    state.success = bool(archive["success"])
    state.nfev = int(archive["nfev"])
    state.luttinger_bond_dim = int(bond_dim)
    state.luttinger_num_modes = int(len(archive["rate"]))
    state.luttinger_depth = int(depth)
    return state


def _best_depth_seed(model, state, depth, args):
    candidates = []
    for factor in (1.0, 0.75, 0.5, 0.25, 0.0):
        candidate = state.cletta_base.cletta_memory_state(
            factor * state.cletta_tie_matrices,
            state.cletta_decay_rates,
            depth=depth,
        )
        energy = cmps_luttinger_energy_shift_density(
            model,
            candidate,
            quadrature_points=args.depth_seed_quadrature_points,
            contraction_backend="hierarchy_iterative",
            iterative_tolerance=args.iterative_tolerance,
            iterative_maxiter=args.iterative_maxiter,
        )
        candidates.append((energy, factor, candidate))
    energy, factor, candidate = min(candidates, key=lambda item: item[0])
    print(
        f"D={state.luttinger_bond_dim} M={state.luttinger_num_modes} "
        f"L={depth}: selected tie attenuation {factor:g} "
        f"(seed E={energy:.10f})",
        flush=True,
    )
    return candidate


def _optimize_or_load(args, model):
    directory = Path(args.result_directory)
    directory.mkdir(parents=True, exist_ok=True)
    cmps = {}
    previous = None
    for bond_dim in (2, 3, 4, 5):
        path = _state_path(directory, "cmps", bond_dim)
        if path.exists() and not args.force:
            state = _load_cmps(path, bond_dim)
        else:
            seeds = [] if previous is None else [previous]
            state = optimize_luttinger_cletta(
                model,
                bond_dim=bond_dim,
                num_modes=0,
                seed_states=seeds,
                restarts=args.cmps_restarts,
                seed=10 * bond_dim,
                maxiter=args.maxiter,
                quadrature_points=args.quadrature_points,
            )
            state.energy = cmps_luttinger_energy_shift_density(
                model,
                state,
                quadrature_points=args.validation_quadrature_points,
            )
            _save_cmps(path, state)
        cmps[bond_dim] = state
        previous = state

    cletta = {}
    for bond_dim in (2, 3):
        previous_depth = None
        for depth in range(1, args.max_depth + 1):
            path = _state_path(directory, "cletta", bond_dim, depth)
            if path.exists() and not args.force:
                state = _load_cletta(path, bond_dim, depth)
            else:
                seeds = [cmps[bond_dim]]
                if previous_depth is not None:
                    seeds.append(previous_depth)
                lower = cletta.get((bond_dim - 1, depth))
                if lower is not None:
                    seeds.append(lower)
                state = optimize_luttinger_cletta(
                    model,
                    bond_dim=bond_dim,
                    num_modes=1,
                    depth=depth,
                    seed_states=seeds,
                    restarts=args.cletta_restarts,
                    seed=100 * bond_dim + depth,
                    maxiter=args.maxiter,
                    quadrature_points=args.quadrature_points,
                )
                state.energy = cmps_luttinger_energy_shift_density(
                    model,
                    state,
                    quadrature_points=args.validation_quadrature_points,
                )
                _save_cletta(path, state, depth)
            cletta[(bond_dim, depth)] = state
            previous_depth = state

    matrix_modes = {}
    matrix_mode_states = {}
    mode_depth_rows = []
    for num_modes in (2, 4, 8):
        previous_depth = None
        previous_energy = None
        for depth in range(1, args.mode_max_depth + 1):
            path = _state_path(
                directory,
                "cletta",
                2,
                depth,
                num_modes,
            )
            if path.exists() and not args.force:
                state = _load_cletta(path, 2, depth)
            else:
                seeds = [cmps[2]]
                if previous_depth is not None:
                    seeds = [
                        _best_depth_seed(
                            model,
                            previous_depth,
                            depth,
                            args,
                        )
                    ]
                state = optimize_luttinger_cletta(
                    model,
                    bond_dim=2,
                    num_modes=num_modes,
                    depth=depth,
                    seed_states=seeds,
                    restarts=args.matrix_mode_restarts,
                    seed=2000 + 10 * num_modes + depth,
                    maxiter=args.matrix_mode_maxiter,
                    quadrature_points=args.matrix_mode_quadrature_points,
                    contraction_backend=(
                        "explicit"
                        if depth == 1
                        else "hierarchy_iterative"
                    ),
                    iterative_tolerance=args.iterative_tolerance,
                    iterative_maxiter=args.iterative_maxiter,
                )
                state.energy = cmps_luttinger_energy_shift_density(
                    model,
                    state,
                    quadrature_points=args.mode_validation_quadrature_points,
                    contraction_backend=(
                        "explicit"
                        if depth == 1
                        else "hierarchy_iterative"
                    ),
                    iterative_tolerance=args.iterative_tolerance,
                    iterative_maxiter=args.iterative_maxiter,
                )
                if not state.success:
                    raise RuntimeError(
                        f"D=2 M={num_modes} L={depth} did not converge: "
                        f"{state.message}"
                    )
                _save_cletta(path, state, depth)
            energy_step = (
                np.nan
                if previous_energy is None
                else abs(state.energy - previous_energy)
                / abs(model.ground_state_energy_shift_density()[0])
            )
            mode_depth_rows.append(
                {
                    "modes": num_modes,
                    "depth": depth,
                    "energy": state.energy,
                    "energy_step": energy_step,
                    "success": bool(state.success),
                }
            )
            matrix_mode_states[(num_modes, depth)] = state
            previous_depth = state
            previous_energy = state.energy
        matrix_modes[num_modes] = previous_depth
    return cmps, cletta, matrix_modes, mode_depth_rows


def _cosine_correlation(momentum, parameter, cutoff):
    spacing = float(momentum[1] - momentum[0])
    integrand = (
        momentum
        * (parameter - 1.0)
        * np.exp(-momentum / cutoff)
        / (2.0 * np.pi**2)
    )
    return 0.5 * spacing * dct(integrand, type=1)


def _log_weighted_error(distance, values, reference, minimum, maximum):
    selected = (distance >= minimum) & (distance <= maximum)
    coordinate = distance[selected]
    numerator = np.trapezoid(
        (values[selected] - reference[selected]) ** 2 / coordinate,
        coordinate,
    )
    denominator = np.trapezoid(
        reference[selected] ** 2 / coordinate,
        coordinate,
    )
    return float(np.sqrt(numerator / denominator))


def _analyze(
    args,
    model,
    cmps,
    cletta,
    matrix_modes,
    mode_depth_rows,
):
    exact_energy = model.ground_state_energy_shift_density()[0]
    momentum_plot = np.geomspace(
        args.momentum_min,
        args.momentum_max,
        args.momentum_points,
    )
    exact_parameter_plot = model.luttinger_parameter(momentum_plot)

    transform_momentum = np.linspace(
        0.0,
        args.transform_momentum_max,
        args.transform_points,
    )
    distance = (
        np.pi
        * np.arange(args.transform_points)
        / args.transform_momentum_max
    )
    exact_parameter_transform = model.luttinger_parameter(
        transform_momentum
    )
    exact_correlation = _cosine_correlation(
        transform_momentum,
        exact_parameter_transform,
        args.uv_cutoff,
    )

    selected_states = [
        ("cMPS D=2", cmps[2], canonical_parameter_size(2)),
        ("cMPS D=3", cmps[3], canonical_parameter_size(3)),
        ("cMPS D=4", cmps[4], canonical_parameter_size(4)),
        ("cMPS D=5", cmps[5], canonical_parameter_size(5)),
        (
            "cLETTA D=2",
            cletta[(2, args.max_depth)],
            canonical_parameter_size(2) + 2**2 + 1,
        ),
        (
            "cLETTA D=3",
            cletta[(3, args.max_depth)],
            canonical_parameter_size(3) + 3**2 + 1,
        ),
        *[
            (
                (
                    f"cLETTA D=2 M={num_modes} "
                    f"L={matrix_modes[num_modes].luttinger_depth}"
                ),
                matrix_modes[num_modes],
                canonical_parameter_size(2) + 5 * num_modes,
            )
            for num_modes in (2, 4, 8)
        ],
    ]
    parameter_curves = {}
    correlation_curves = {}
    rows = []
    infrared = momentum_plot <= args.infrared_max
    for label, state, parameter_count in selected_states:
        parameter_function = (
            lambda momentum: cmps_luttinger_parameter(state, momentum)
        )
        parameter = parameter_function(momentum_plot)
        transform_parameter = parameter_function(transform_momentum)
        correlation = _cosine_correlation(
            transform_momentum,
            transform_parameter,
            args.uv_cutoff,
        )
        parameter_curves[label] = parameter
        correlation_curves[label] = correlation
        rows.append(
            {
                "ansatz": label,
                "parameters": parameter_count,
                "energy": state.energy,
                "energy_error": state.energy - exact_energy,
                "K_at_momentum_min": parameter[0],
                "infrared_log_error": float(
                    np.sqrt(
                        np.mean(
                            np.log(
                                parameter[infrared]
                                / exact_parameter_plot[infrared]
                            )
                            ** 2
                        )
                    )
                ),
                "correlation_log_error": _log_weighted_error(
                    distance,
                    correlation,
                    exact_correlation,
                    args.distance_min,
                    args.distance_max,
                ),
                "success": bool(state.success),
                "nfev": int(state.nfev),
            }
        )

    depth_rows = []
    previous_energy = None
    previous_correlation = None
    for depth in range(1, args.max_depth + 1):
        state = cletta[(3, depth)]
        energy = cmps_luttinger_energy_shift_density(
            model,
            state,
            quadrature_points=args.validation_quadrature_points,
        )
        transform_parameter = cmps_luttinger_parameter(
            state,
            transform_momentum,
        )
        correlation = _cosine_correlation(
            transform_momentum,
            transform_parameter,
            args.uv_cutoff,
        )
        energy_step = (
            np.nan
            if previous_energy is None
            else abs(energy - previous_energy) / abs(exact_energy)
        )
        correlation_step = (
            np.nan
            if previous_correlation is None
            else _log_weighted_error(
                distance,
                correlation,
                previous_correlation,
                args.distance_min,
                args.distance_max,
            )
        )
        depth_rows.append(
            {
                "depth": depth,
                "energy": energy,
                "energy_error": energy - exact_energy,
                "energy_step": energy_step,
                "correlation_step": correlation_step,
                "success": bool(state.success),
            }
        )
        previous_energy = energy
        previous_correlation = correlation

    return {
        "exact_energy": exact_energy,
        "momentum": momentum_plot,
        "exact_parameter": exact_parameter_plot,
        "parameter_curves": parameter_curves,
        "distance": distance,
        "exact_correlation": exact_correlation,
        "correlation_curves": correlation_curves,
        "rows": rows,
        "depth_rows": depth_rows,
        "mode_depth_rows": mode_depth_rows,
        "mode_labels": {
            modes: (
                f"cLETTA D=2 M={modes} "
                f"L={matrix_modes[modes].luttinger_depth}"
            )
            for modes in (2, 4, 8)
        },
    }


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(args, results):
    import ultraplot as uplt
    from matplotlib.ticker import LogFormatterMathtext

    exact_color = "#202124"
    cmps_color = "#0072B2"
    cletta_color = "#D55E00"
    residual_color = "#009E73"
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "tick.labelsize": 10,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=2,
        refwidth=3.25,
        refheight=2.5,
        share=False,
        wspace=5.2,
        hspace=5.5,
    )
    momentum = results["momentum"]
    exact_parameter = results["exact_parameter"]
    cmps_parameter = results["parameter_curves"]["cMPS D=4"]
    cletta_parameter = results["parameter_curves"]["cLETTA D=3"]

    axes[0].semilogx(
        momentum,
        exact_parameter,
        color=exact_color,
        label="exact",
    )
    axes[0].semilogx(
        momentum,
        cmps_parameter,
        color=cmps_color,
        linestyle="--",
        label="cMPS",
    )
    axes[0].semilogx(
        momentum,
        cletta_parameter,
        color=cletta_color,
        label="cLETTA",
    )
    axes[0].format(
        xlabel=r"momentum $ka$",
        ylabel=r"$K(k)$",
        xlim=(args.momentum_min, args.momentum_max),
        ylim=(0.0, 1.04),
        title=r"$K(k)$: matched 22 parameters",
        grid=False,
    )
    axes[0].legend(loc="ul", frame=False)
    axes[0].xaxis.set_major_formatter(LogFormatterMathtext())

    axes[1].loglog(
        momentum,
        np.abs(cmps_parameter / exact_parameter - 1.0),
        color=cmps_color,
        linestyle="--",
    )
    axes[1].loglog(
        momentum,
        np.abs(cletta_parameter / exact_parameter - 1.0),
        color=cletta_color,
    )
    axes[1].format(
        xlabel=r"momentum $ka$",
        ylabel=r"$|K/K_{\rm exact}-1|$",
        xlim=(args.momentum_min, args.infrared_max),
        title="Infrared relative error",
        grid=False,
    )
    axes[1].xaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())

    distance = results["distance"]
    selected = (distance >= 1.0) & (distance <= args.distance_plot_max)
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["exact_correlation"][selected],
        color=exact_color,
    )
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["correlation_curves"]["cMPS D=4"][selected],
        color=cmps_color,
        linestyle="--",
    )
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["correlation_curves"]["cLETTA D=3"][selected],
        color=cletta_color,
    )
    axes[2].format(
        xlabel=r"distance $r/a$",
        ylabel=r"$r^2\Delta C(r)$",
        xlim=(1.0, args.distance_plot_max),
        title="Critical real-space tail",
        grid=False,
    )
    axes[2].xaxis.set_major_formatter(LogFormatterMathtext())

    depth = np.array([row["depth"] for row in results["depth_rows"][1:]])
    energy_step = np.array(
        [row["energy_step"] for row in results["depth_rows"][1:]]
    )
    correlation_step = np.array(
        [row["correlation_step"] for row in results["depth_rows"][1:]]
    )
    axes[3].semilogy(
        depth,
        energy_step,
        color=residual_color,
        marker="o",
        markerfacecolor="white",
        label=r"energy $\delta E_L$",
    )
    axes[3].semilogy(
        depth,
        correlation_step,
        color=cletta_color,
        marker="s",
        markerfacecolor="white",
        label=r"correlation $\delta C_L$",
    )
    axes[3].format(
        xlabel=r"hierarchy depth $L$",
        ylabel="successive change",
        xticks=depth,
        title="Hierarchy convergence",
        grid=False,
    )
    axes[3].legend(loc="ur", frame=False)
    axes[3].yaxis.set_major_formatter(LogFormatterMathtext())

    for label, axis in zip("abcd", axes):
        axis.text(
            -0.13,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
        )

    output = Path(args.figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=400)


def _plot_modes(args, results):
    import ultraplot as uplt
    from matplotlib.ticker import LogFormatterMathtext

    colors = {
        "exact": "#202124",
        "cmps": "#0072B2",
        2: "#6F6F6F",
        4: "#009E73",
        8: "#D55E00",
    }
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.2,
            "tick.labelsize": 10,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        nrows=2,
        ncols=2,
        refwidth=3.25,
        refheight=2.5,
        share=False,
        wspace=5.2,
        hspace=5.5,
    )
    momentum = results["momentum"]
    exact_parameter = results["exact_parameter"]
    cmps_parameter = results["parameter_curves"]["cMPS D=2"]
    cmps_d5_parameter = results["parameter_curves"]["cMPS D=5"]
    mode_parameters = {
        modes: results["parameter_curves"][results["mode_labels"][modes]]
        for modes in (2, 4, 8)
    }

    axes[0].semilogx(
        momentum,
        exact_parameter,
        color=colors["exact"],
        label="exact",
    )
    axes[0].semilogx(
        momentum,
        cmps_parameter,
        color=colors["cmps"],
        linestyle="--",
        label=r"cMPS $D=2$",
    )
    axes[0].semilogx(
        momentum,
        cmps_d5_parameter,
        color=colors["cmps"],
        linestyle=":",
        label=r"cMPS $D=5$",
    )
    for modes in (2, 4, 8):
        axes[0].semilogx(
            momentum,
            mode_parameters[modes],
            color=colors[modes],
            label=rf"$M={modes}$",
        )
    axes[0].format(
        xlabel=r"momentum $ka$",
        ylabel=r"$K(k)$",
        xlim=(args.momentum_min, args.momentum_max),
        ylim=(0.0, 1.04),
        title="Memory-channel resolution",
        grid=False,
    )
    axes[0].legend(loc="ul", ncols=2, frame=False)
    axes[0].xaxis.set_major_formatter(LogFormatterMathtext())

    axes[1].loglog(
        momentum,
        np.abs(cmps_parameter / exact_parameter - 1.0),
        color=colors["cmps"],
        linestyle="--",
    )
    axes[1].loglog(
        momentum,
        np.abs(cmps_d5_parameter / exact_parameter - 1.0),
        color=colors["cmps"],
        linestyle=":",
    )
    for modes in (2, 4, 8):
        axes[1].loglog(
            momentum,
            np.abs(mode_parameters[modes] / exact_parameter - 1.0),
            color=colors[modes],
        )
    axes[1].format(
        xlabel=r"momentum $ka$",
        ylabel=r"$|K/K_{\rm exact}-1|$",
        xlim=(args.momentum_min, args.infrared_max),
        title="Infrared relative error",
        grid=False,
    )
    axes[1].xaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())

    distance = results["distance"]
    selected = (distance >= 1.0) & (distance <= args.distance_plot_max)
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["exact_correlation"][selected],
        color=colors["exact"],
    )
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["correlation_curves"]["cMPS D=2"][selected],
        color=colors["cmps"],
        linestyle="--",
    )
    axes[2].semilogx(
        distance[selected],
        distance[selected] ** 2
        * results["correlation_curves"]["cMPS D=5"][selected],
        color=colors["cmps"],
        linestyle=":",
    )
    for modes in (2, 4, 8):
        axes[2].semilogx(
            distance[selected],
            distance[selected] ** 2
            * results["correlation_curves"][results["mode_labels"][modes]][
                selected
            ],
            color=colors[modes],
        )
    axes[2].format(
        xlabel=r"distance $r/a$",
        ylabel=r"$r^2\Delta C(r)$",
        xlim=(1.0, args.distance_plot_max),
        title="Critical real-space tail",
        grid=False,
    )
    axes[2].xaxis.set_major_formatter(LogFormatterMathtext())

    row_by_name = {row["ansatz"]: row for row in results["rows"]}
    mode_counts = np.array([2, 4, 8])
    energy_error = np.array(
        [
            row_by_name[results["mode_labels"][modes]]["energy_error"]
            / abs(results["exact_energy"])
            for modes in mode_counts
        ]
    )
    correlation_error = np.array(
        [
            row_by_name[results["mode_labels"][modes]][
                "correlation_log_error"
            ]
            for modes in mode_counts
        ]
    )
    axes[3].semilogy(
        mode_counts,
        energy_error,
        color=colors[4],
        marker="o",
        markerfacecolor="white",
        label="energy",
    )
    axes[3].semilogy(
        mode_counts,
        correlation_error,
        color=colors[8],
        marker="s",
        markerfacecolor="white",
        label="correlation",
    )
    cmps_d5 = row_by_name["cMPS D=5"]
    axes[3].axhline(
        cmps_d5["energy_error"] / abs(results["exact_energy"]),
        color=colors["cmps"],
        linestyle="--",
        linewidth=1.1,
    )
    axes[3].text(
        0.98,
        cmps_d5["energy_error"] / abs(results["exact_energy"]),
        r"cMPS $D=5$: energy",
        color=colors["cmps"],
        fontsize=8,
        ha="right",
        va="bottom",
        transform=axes[3].get_yaxis_transform(),
    )
    axes[3].axhline(
        cmps_d5["correlation_log_error"],
        color=colors["cmps"],
        linestyle=":",
        linewidth=1.1,
    )
    axes[3].text(
        0.98,
        cmps_d5["correlation_log_error"],
        r"cMPS $D=5$: correlation",
        color=colors["cmps"],
        fontsize=8,
        ha="right",
        va="bottom",
        transform=axes[3].get_yaxis_transform(),
    )
    final_depths = {
        int(label.rsplit("=", 1)[-1])
        for label in results["mode_labels"].values()
    }
    depth_text = (
        str(next(iter(final_depths)))
        if len(final_depths) == 1
        else "mixed"
    )
    axes[3].format(
        xlabel=r"memory channels $M$",
        ylabel="relative error",
        xticks=mode_counts,
        title=rf"Matrix cLETTA: $D=2$, $L={depth_text}$",
        grid=False,
    )
    axes[3].legend(loc="ur", frame=False)
    axes[3].yaxis.set_major_formatter(LogFormatterMathtext())

    for label, axis in zip("abcd", axes):
        axis.text(
            -0.13,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
        )
    output = Path(args.mode_figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=400)


def run(args):
    model = CoulombLuttingerModel(
        coupling=args.coupling,
        softening=args.softening,
        fermi_velocity=args.fermi_velocity,
    )
    cmps, cletta, matrix_modes, mode_depth_rows = _optimize_or_load(
        args,
        model,
    )
    if args.optimize_only:
        for row in mode_depth_rows:
            print(
                f"D=2 M={row['modes']} L={row['depth']}: "
                f"E={row['energy']:.12f} "
                f"delta_L={row['energy_step']:.3e} "
                f"success={row['success']}"
            )
        return {"mode_depth_rows": mode_depth_rows}
    results = _analyze(
        args,
        model,
        cmps,
        cletta,
        matrix_modes,
        mode_depth_rows,
    )
    _write_csv(args.summary, results["rows"])
    _write_csv(args.depth_output, results["depth_rows"])
    _plot(args, results)
    _plot_modes(args, results)
    for row in results["rows"]:
        print(
            f"{row['ansatz']}: params={row['parameters']} "
            f"dE={row['energy_error']:.3e} "
            f"epsilon_K={row['infrared_log_error']:.3e} "
            f"epsilon_C={row['correlation_log_error']:.3e} "
            f"success={row['success']}"
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=8.0)
    parser.add_argument("--softening", type=float, default=1.0)
    parser.add_argument("--fermi-velocity", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--cmps-restarts", type=int, default=6)
    parser.add_argument("--cletta-restarts", type=int, default=3)
    parser.add_argument("--matrix-mode-restarts", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--matrix-mode-maxiter", type=int, default=350)
    parser.add_argument("--quadrature-points", type=int, default=240)
    parser.add_argument(
        "--matrix-mode-quadrature-points",
        type=int,
        default=80,
    )
    parser.add_argument("--validation-quadrature-points", type=int, default=900)
    parser.add_argument("--momentum-min", type=float, default=1.0e-8)
    parser.add_argument("--momentum-max", type=float, default=10.0)
    parser.add_argument("--momentum-points", type=int, default=1201)
    parser.add_argument("--infrared-max", type=float, default=1.0)
    parser.add_argument("--uv-cutoff", type=float, default=8.0)
    parser.add_argument("--transform-momentum-max", type=float, default=80.0)
    parser.add_argument("--transform-points", type=int, default=131073)
    parser.add_argument("--distance-min", type=float, default=0.1)
    parser.add_argument("--distance-max", type=float, default=5000.0)
    parser.add_argument("--distance-plot-max", type=float, default=1000.0)
    parser.add_argument("--mode-max-depth", type=int, default=2)
    parser.add_argument(
        "--mode-validation-quadrature-points",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--depth-seed-quadrature-points",
        type=int,
        default=16,
    )
    parser.add_argument("--iterative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--iterative-maxiter", type=int, default=1200)
    parser.add_argument("--optimize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--result-directory",
        default="examples/mps/results",
    )
    parser.add_argument(
        "--summary",
        default="examples/mps/results/coulomb_luttinger_summary.csv",
    )
    parser.add_argument(
        "--depth-output",
        default="examples/mps/results/coulomb_luttinger_depth.csv",
    )
    parser.add_argument(
        "--figure",
        default="examples/mps/results/coulomb_luttinger_comparison.pdf",
    )
    parser.add_argument(
        "--mode-figure",
        default=(
            "examples/mps/results/"
            "coulomb_luttinger_mode_convergence.pdf"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
