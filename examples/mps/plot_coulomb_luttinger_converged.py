"""Plot the hierarchy-converged Coulomb-Luttinger cLETTA benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import dct

from pyqed.mps import (
    ContinuousMPS,
    CoulombLuttingerModel,
    cmps_luttinger_parameter,
)


RESULTS = Path("examples/mps/results")
OUTPUT = RESULTS / "coulomb_luttinger_mode_convergence.pdf"
DATA = RESULTS / "coulomb_luttinger_converged_figure_data.npz"


def load_cmps(path, bond_dim):
    archive = np.load(path)
    return ContinuousMPS.from_canonical_parameters(archive["theta"], bond_dim)


def load_cletta(path, bond_dim):
    archive = np.load(path)
    base = ContinuousMPS.from_canonical_parameters(
        archive["base_theta"],
        bond_dim,
    )
    return base.cletta_memory_state(
        archive["tie"],
        archive["rate"],
        depth=int(archive["depth"]),
    )


def cosine_correlation(momentum, parameter, cutoff):
    spacing = float(momentum[1] - momentum[0])
    integrand = (
        momentum
        * (parameter - 1.0)
        * np.exp(-momentum / cutoff)
        / (2.0 * np.pi**2)
    )
    return 0.5 * spacing * dct(integrand, type=1)


def main():
    import ultraplot as uplt
    from matplotlib.ticker import LogFormatterMathtext

    model = CoulombLuttingerModel(
        coupling=8.0,
        softening=1.0,
        fermi_velocity=1.0,
    )
    states = {
        "cmps_d2": load_cmps(RESULTS / "coulomb_luttinger_cmps_D2.npz", 2),
        "cmps_d5": load_cmps(RESULTS / "coulomb_luttinger_cmps_D5.npz", 5),
        "cletta": load_cletta(
            RESULTS / "coulomb_luttinger_cletta_D2_M2_L5_implicit.npz",
            2,
        ),
    }
    momentum = np.geomspace(1.0e-8, 10.0, 1201)
    transform_momentum = np.linspace(0.0, 80.0, 131073)
    combined_momentum = np.concatenate([momentum, transform_momentum])
    split = momentum.size
    exact_combined = model.luttinger_parameter(combined_momentum)
    parameter = {
        label: cmps_luttinger_parameter(state, combined_momentum)
        for label, state in states.items()
    }
    exact_parameter = exact_combined[:split]
    exact_transform = exact_combined[split:]
    parameter_plot = {
        label: values[:split] for label, values in parameter.items()
    }
    parameter_transform = {
        label: values[split:] for label, values in parameter.items()
    }
    distance = (
        np.pi
        * np.arange(transform_momentum.size)
        / transform_momentum[-1]
    )
    correlation = {
        "exact": cosine_correlation(
            transform_momentum,
            exact_transform,
            8.0,
        ),
        **{
            label: cosine_correlation(
                transform_momentum,
                values,
                8.0,
            )
            for label, values in parameter_transform.items()
        },
    }
    np.savez(
        DATA,
        momentum=momentum,
        exact_parameter=exact_parameter,
        cmps_d2_parameter=parameter_plot["cmps_d2"],
        cmps_d5_parameter=parameter_plot["cmps_d5"],
        cletta_parameter=parameter_plot["cletta"],
        distance=distance,
        exact_correlation=correlation["exact"],
        cmps_d2_correlation=correlation["cmps_d2"],
        cmps_d5_correlation=correlation["cmps_d5"],
        cletta_correlation=correlation["cletta"],
    )

    colors = {
        "exact": "#202124",
        "cmps": "#0072B2",
        "cletta": "#D55E00",
        "parameter": "#009E73",
    }
    uplt.rc.update(
        {
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.3,
            "tick.labelsize": 10,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = uplt.subplots(
        ncols=2,
        refwidth=3.25,
        refheight=2.5,
        share=False,
        wspace=5.0,
    )
    curves = (
        ("exact", exact_parameter, colors["exact"], "-", "exact"),
        (
            "cmps_d2",
            parameter_plot["cmps_d2"],
            colors["cmps"],
            "--",
            r"cMPS $D=2$",
        ),
        (
            "cmps_d5",
            parameter_plot["cmps_d5"],
            colors["cmps"],
            ":",
            r"cMPS $D=5$",
        ),
        (
            "cletta",
            parameter_plot["cletta"],
            colors["cletta"],
            "-",
            r"cLETTA $D=2,M=2$",
        ),
    )
    for _key, values, color, linestyle, label in curves:
        axes[0].semilogx(
            momentum,
            values,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0].format(
        xlabel=r"momentum $ka$",
        ylabel=r"$K_{\mathrm{LL}}(k)$",
        xlim=(1.0e-8, 10.0),
        ylim=(0.0, 1.04),
        title="Converged momentum dependence",
        grid=False,
    )
    axes[0].legend(loc="ul", ncols=2, frame=False)
    axes[0].xaxis.set_major_formatter(LogFormatterMathtext())

    selected = (distance >= 1.0) & (distance <= 1000.0)
    correlation_curves = (
        ("exact", colors["exact"], "-"),
        ("cmps_d2", colors["cmps"], "--"),
        ("cmps_d5", colors["cmps"], ":"),
        ("cletta", colors["cletta"], "-"),
    )
    for key, color, linestyle in correlation_curves:
        axes[1].semilogx(
            distance[selected],
            distance[selected] ** 2 * correlation[key][selected],
            color=color,
            linestyle=linestyle,
        )
    axes[1].format(
        xlabel=r"distance $r/a$",
        ylabel=r"$r^2\Delta C(r)$",
        xlim=(1.0, 1000.0),
        title="Critical real-space tail",
        grid=False,
    )
    axes[1].xaxis.set_major_formatter(LogFormatterMathtext())

    for label, axis in zip("ab", axes):
        axis.text(
            -0.13,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=13,
            fontweight="bold",
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=400)


if __name__ == "__main__":
    main()
