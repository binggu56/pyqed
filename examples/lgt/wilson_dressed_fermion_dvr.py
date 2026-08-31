#!/usr/bin/env python3
"""Gauge-covariant fermionic Fourier-DVR calculation with O(N log N) links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from pyqed.lgt import WilsonFourierDVR


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "wilson_fermion_dvr"


def background(npts: int, length: float):
    spacing = length / npts
    sites = -0.5 * length + spacing * np.arange(npts)
    midpoints = sites + 0.5 * spacing
    gauge_potential = (
        0.22
        + 0.34 * np.cos(2.0 * np.pi * midpoints / length)
        + 0.16 * np.sin(4.0 * np.pi * midpoints / length)
    )
    mass = 0.82 + 0.27 * np.cos(2.0 * np.pi * sites / length)
    link_phases = spacing * gauge_potential
    return WilsonFourierDVR(link_phases, length), mass, gauge_potential


def _relative_residual(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def spectrum_calculation(npts=127, length=24.0, nlevels=18, seed=17):
    model, mass, gauge_potential = background(npts, length)
    dense = model.dense_dirac(mass)
    spectrum = scipy.linalg.eigvalsh(dense, check_finite=False)
    positive = spectrum[spectrum > 1.0e-10][:nlevels]

    rng = np.random.default_rng(seed)
    beta = (
        0.63 * np.sin(2.0 * np.pi * model.x / length)
        - 0.31 * np.cos(6.0 * np.pi * model.x / length)
        + 0.08 * rng.normal(size=npts)
    )
    transformed = model.gauge_transform(beta)
    transformed_dense = transformed.dense_dirac(mass)
    transformed_spectrum = scipy.linalg.eigvalsh(
        transformed_dense, check_finite=False
    )
    transformed_positive = transformed_spectrum[
        transformed_spectrum > 1.0e-10
    ][:nlevels]

    state = rng.normal(size=(npts, 2)) + 1j * rng.normal(size=(npts, 2))
    gauge = np.exp(1j * beta)[:, None]
    fft_action = model.apply_dirac(state, mass)
    dense_action = (dense @ state.reshape(-1)).reshape(npts, 2)
    covariance_left = transformed.apply_dirac(gauge * state, mass)
    covariance_right = gauge * fft_action

    diagnostics = {
        "dirac_fft_dense_residual": _relative_residual(fft_action, dense_action),
        "gauge_covariance_residual": _relative_residual(
            covariance_left, covariance_right
        ),
        "hermiticity_residual": _relative_residual(dense.conj().T, dense),
        "spectrum_gauge_residual": _relative_residual(
            transformed_positive, positive
        ),
        "nearest_link_residual": _relative_residual(
            np.asarray(
                [model.wilson_line(site, site + 1) for site in range(npts)]
            ),
            model.link_variables,
        ),
    }
    return {
        "npts": npts,
        "length": length,
        "x": model.x,
        "mass": mass,
        "gauge_potential": gauge_potential,
        "link_phases": model.link_phases,
        "holonomy_phase": model.holonomy_phase,
        "positive_energies": positive,
        "gauge_transformed_positive_energies": transformed_positive,
        "diagnostics": diagnostics,
    }


def _stable_time(function, minimum_batch_seconds=0.04, batches=5):
    function()
    repeats = 1
    while True:
        start = perf_counter()
        for _ in range(repeats):
            function()
        elapsed = perf_counter() - start
        if elapsed >= minimum_batch_seconds:
            break
        repeats *= 2
    samples = []
    for _ in range(batches):
        start = perf_counter()
        for _ in range(repeats):
            function()
        samples.append((perf_counter() - start) / repeats)
    return float(np.median(samples)), repeats


def scaling_calculation(
    # Odd 3/5-smooth sizes avoid prime-length FFT overhead while retaining the
    # unambiguous shortest Wilson path of an odd periodic grid.
    fast_sizes=(45, 81, 135, 243, 405, 729, 1215, 2187, 3645, 6561, 10935, 19683, 32805),
    dense_limit=1215,
    length=24.0,
    seed=23,
):
    rng = np.random.default_rng(seed)
    fast_seconds = []
    fast_repeats = []
    dense_sizes = []
    dense_seconds = []
    dense_repeats = []
    dense_residuals = []
    for npts in fast_sizes:
        model, _, _ = background(npts, length)
        state = rng.normal(size=npts) + 1j * rng.normal(size=npts)
        fast_call = lambda m=model, v=state: m.apply_derivative(v)
        seconds, repeats = _stable_time(fast_call)
        fast_seconds.append(seconds)
        fast_repeats.append(repeats)

        if npts <= dense_limit:
            dense = model.dense_derivative()
            dense_call = lambda matrix=dense, v=state: matrix @ v
            dense_time, repeats = _stable_time(dense_call)
            dense_sizes.append(npts)
            dense_seconds.append(dense_time)
            dense_repeats.append(repeats)
            dense_residuals.append(
                _relative_residual(fast_call(), dense_call())
            )

    fast_sizes = np.asarray(fast_sizes)
    dense_sizes = np.asarray(dense_sizes)
    fast_seconds = np.asarray(fast_seconds)
    dense_seconds = np.asarray(dense_seconds)
    fast_slope = float(
        np.polyfit(np.log(fast_sizes[-6:]), np.log(fast_seconds[-6:]), 1)[0]
    )
    dense_slope = float(
        np.polyfit(np.log(dense_sizes[-4:]), np.log(dense_seconds[-4:]), 1)[0]
    )
    return {
        "fast_sizes": fast_sizes,
        "fast_seconds": fast_seconds,
        "fast_repeats": np.asarray(fast_repeats),
        "dense_sizes": dense_sizes,
        "dense_seconds": dense_seconds,
        "dense_repeats": np.asarray(dense_repeats),
        "dense_residuals": np.asarray(dense_residuals),
        "fast_effective_power": fast_slope,
        "dense_effective_power": dense_slope,
    }


def _style_axis(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_spectrum(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].plot(data["x"], data["mass"], label=r"mass $m(x)$")
    axes[0].plot(
        data["x"],
        data["gauge_potential"],
        label=r"link field $A_1(x)$",
    )
    axes[0].set_xlabel("position x")
    axes[0].set_ylabel("background value")
    axes[0].set_title("Nonuniform fermion and link background")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    level = np.arange(1, len(data["positive_energies"]) + 1)
    axes[1].plot(
        level,
        data["positive_energies"],
        "o",
        label="original links",
    )
    axes[1].plot(
        level,
        data["gauge_transformed_positive_energies"],
        "+",
        markersize=10,
        markeredgewidth=1.8,
        label="locally gauge-transformed links",
    )
    axes[1].set_xlabel("positive-energy level")
    axes[1].set_ylabel("Dirac energy")
    axes[1].set_title("Gauge-invariant Wilson-DVR spectrum")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_scaling(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    fast_n = data["fast_sizes"]
    dense_n = data["dense_sizes"]
    fast_t = data["fast_seconds"]
    dense_t = data["dense_seconds"]

    axes[0].loglog(
        fast_n,
        fast_t,
        "o-",
        label=rf"prefix + FFT (fit $N^{{{data['fast_effective_power']:.2f}}}$)",
    )
    axes[0].loglog(
        dense_n,
        dense_t,
        "s-",
        label=rf"dense Wilson matrix (fit $N^{{{data['dense_effective_power']:.2f}}}$)",
    )
    fft_guide = fast_n * np.log2(fast_n)
    fft_guide *= fast_t[-1] / fft_guide[-1]
    dense_guide = dense_n.astype(float) ** 2
    dense_guide *= dense_t[-1] / dense_guide[-1]
    axes[0].loglog(fast_n, fft_guide, "k--", alpha=0.5, label=r"$N\log_2N$")
    axes[0].loglog(dense_n, dense_guide, "k:", alpha=0.6, label=r"$N^2$")
    axes[0].set_xlabel("DVR points N")
    axes[0].set_ylabel("median derivative matvec time (s)")
    axes[0].set_title("Wilson derivative application")
    axes[0].legend(frameon=False, fontsize=9)
    _style_axis(axes[0])

    axes[1].semilogx(
        fast_n,
        fast_t / (fast_n * np.log2(fast_n)),
        "o-",
        label=r"FFT time / $N\log_2N$",
    )
    axes[1].semilogx(
        dense_n,
        dense_t / dense_n**2,
        "s-",
        label=r"dense time / $N^2$",
    )
    axes[1].set_xlabel("DVR points N")
    axes[1].set_ylabel("complexity-normalized time (s)")
    axes[1].set_title("Asymptotic normalization")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run(output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    spectrum = spectrum_calculation()
    scaling = scaling_calculation()
    spectrum_figure = output_directory / "05_wilson_dressed_fermion_spectrum.png"
    scaling_figure = output_directory / "06_wilson_fft_scaling.png"
    plot_spectrum(spectrum, spectrum_figure)
    plot_scaling(scaling, scaling_figure)

    payload = {
        "scope": (
            "Two-component fermion in a classical nonuniform U(1) link "
            "background. Quantized electric-link dynamics are not included."
        ),
        "factorization": "D_U = S D_uniform-holonomy S^dagger",
        "spectrum": spectrum,
        "scaling": scaling,
        "figures": {
            "spectrum": str(spectrum_figure),
            "scaling": str(scaling_figure),
        },
    }
    data_path = output_directory / "wilson_fermion_data.json"
    data_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print(f"wrote {data_path}")
    print(f"wrote {spectrum_figure}")
    print(f"wrote {scaling_figure}")
    print(json.dumps(spectrum["diagnostics"], indent=2))
    print(
        "effective timing powers: "
        f"FFT={scaling['fast_effective_power']:.3f}, "
        f"dense={scaling['dense_effective_power']:.3f}"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    args = parser.parse_args()
    run(args.output_directory)


if __name__ == "__main__":
    main()
