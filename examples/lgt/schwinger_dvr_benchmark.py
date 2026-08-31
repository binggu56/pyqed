#!/usr/bin/env python3
"""Minimal DVR versus local-lattice benchmarks for the Schwinger model.

The first two calculations isolate the spatial Dirac regulator.  The last two
use the exact bosonized massless Schwinger model,

    H = 1/2 int dx [Pi^2 + (d_x phi)^2 + (g^2/pi) phi^2],

and extract its rest mass from the low-momentum dispersion.  This keeps the
mass-gap comparison exact and reproducible without conflating DVR accuracy
with an approximate interacting many-body solver.
"""

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

from pyqed.dvr import ExponentialDVR, SineDVR


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "schwinger_dvr"
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def _positive_levels(hamiltonian: np.ndarray, count: int) -> np.ndarray:
    energies = scipy.linalg.eigvalsh(hamiltonian, check_finite=False)
    positive = energies[energies > 1.0e-10]
    if positive.size < count:
        raise RuntimeError("not enough positive-energy Dirac levels")
    return positive[:count]


def fourier_dirac_hamiltonian(npts: int, length: float, mass_fn) -> np.ndarray:
    """Two-component periodic Dirac Hamiltonian on an odd Fourier DVR grid."""
    dvr = ExponentialDVR(npts=npts, L=length)
    momentum = dvr.momentum()
    mass = np.asarray(mass_fn(dvr.x), dtype=float)
    hamiltonian = np.kron(momentum, SIGMA_X)
    hamiltonian += np.kron(np.diag(mass), SIGMA_Z)
    return hamiltonian


def staggered_dirac_hamiltonian(n_orbitals: int, length: float, mass_fn) -> np.ndarray:
    """One-component periodic staggered Dirac Hamiltonian."""
    if n_orbitals % 2:
        raise ValueError("a periodic staggered lattice needs an even site count")
    spacing = length / n_orbitals
    sites = np.arange(n_orbitals)
    x = -0.5 * length + spacing * sites
    hamiltonian = np.diag((-1.0) ** sites * mass_fn(x)).astype(complex)
    for site in sites:
        neighbor = (site + 1) % n_orbitals
        hamiltonian[site, neighbor] += -0.5j / spacing
        hamiltonian[neighbor, site] += 0.5j / spacing
    return hamiltonian


def dispersion_benchmark(npts: int = 63, length: float = 20.0, mass: float = 0.7):
    """Return free positive-energy dispersion errors at matched orbital count."""
    if npts % 2 == 0:
        raise ValueError("use an odd Fourier grid to avoid a Nyquist ambiguity")
    dvr = ExponentialDVR(npts=npts, L=length)
    numeric_momenta = scipy.linalg.eigvalsh(dvr.momentum(), check_finite=False)
    numeric_momenta = numeric_momenta[numeric_momenta >= -1.0e-12]

    mode = np.arange((npts + 1) // 2)
    momentum = 2.0 * np.pi * mode / length
    numeric_momenta = numeric_momenta[: momentum.size]
    exact = np.sqrt(mass**2 + momentum**2)
    dvr_energy = np.sqrt(mass**2 + numeric_momenta**2)

    # A staggered lattice with 2*npts one-component sites has the same number
    # of fermionic orbitals as npts two-component DVR points.
    spacing = length / (2 * npts)
    staggered_momentum = np.sin(momentum * spacing) / spacing
    staggered_energy = np.sqrt(mass**2 + staggered_momentum**2)
    floor = np.finfo(float).eps
    return {
        "momentum_fraction": momentum / momentum[-1],
        "dvr_error": np.maximum(np.abs(dvr_energy - exact) / exact, floor),
        "staggered_error": np.maximum(
            np.abs(staggered_energy - exact) / exact, floor
        ),
        "npts": npts,
        "length": length,
        "mass": mass,
    }


def inhomogeneous_mass_benchmark(
    npts_values=(9, 13, 17, 25, 33, 49, 65, 97, 129),
    *,
    reference_npts: int = 257,
    length: float = 18.0,
    mass0: float = 1.0,
    modulation: float = 0.65,
    levels: int = 6,
):
    """Compare low positive Dirac levels for a smooth periodic mass."""

    def mass_fn(x):
        return mass0 + modulation * np.cos(2.0 * np.pi * x / length)

    reference = _positive_levels(
        fourier_dirac_hamiltonian(reference_npts, length, mass_fn), levels
    )
    dvr_error = []
    staggered_error = []
    for npts in npts_values:
        dvr_levels = _positive_levels(
            fourier_dirac_hamiltonian(npts, length, mass_fn), levels
        )
        staggered_levels = _positive_levels(
            staggered_dirac_hamiltonian(2 * npts, length, mass_fn), levels
        )
        dvr_error.append(np.max(np.abs((dvr_levels - reference) / reference)))
        staggered_error.append(
            np.max(np.abs((staggered_levels - reference) / reference))
        )
    return {
        "orbitals": 2 * np.asarray(npts_values),
        "dvr_error": np.maximum(dvr_error, np.finfo(float).eps),
        "staggered_error": np.maximum(staggered_error, np.finfo(float).eps),
        "reference_levels": reference,
        "reference_npts": reference_npts,
        "length": length,
        "mass0": mass0,
        "modulation": modulation,
        "levels": levels,
    }


def _schwinger_gap_dvr(npts: int, length: float, g: float, fit_modes: int):
    """Extract the Schwinger boson rest mass from a sine-DVR dispersion."""
    # SineDVR.t() is -d^2/(2m); multiplying the m=1 operator by two gives
    # the positive Laplacian entering omega^2 = -d_x^2 + g^2/pi.
    laplacian = 2.0 * SineDVR(-0.5 * length, 0.5 * length, npts).t()
    mu_exact = g / np.sqrt(np.pi)
    operator = laplacian + mu_exact**2 * np.eye(npts)
    omega2 = scipy.linalg.eigh(
        operator,
        subset_by_index=(0, fit_modes - 1),
        eigvals_only=True,
        check_finite=False,
    )
    momentum = np.pi * np.arange(1, fit_modes + 1) / length
    mu2 = np.mean(omega2 - momentum**2)
    return float(np.sqrt(max(mu2, 0.0)))


def _schwinger_gap_finite_difference(
    npts: int, length: float, g: float, fit_modes: int
):
    """Extract the same rest mass using a local three-point Laplacian."""
    spacing = length / (npts + 1)
    mu_exact = g / np.sqrt(np.pi)
    diagonal = np.full(npts, 2.0 / spacing**2 + mu_exact**2)
    off_diagonal = np.full(npts - 1, -1.0 / spacing**2)
    omega2 = scipy.linalg.eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, fit_modes - 1),
        eigvals_only=True,
        check_finite=False,
    )
    momentum = np.pi * np.arange(1, fit_modes + 1) / length
    mu2 = np.mean(omega2 - momentum**2)
    return float(np.sqrt(max(mu2, 0.0)))


def _median_runtime(function, repeats: int) -> float:
    function()  # warm up dispatch and allocations
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def schwinger_gap_benchmark(
    npts_values=(7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257),
    *,
    length: float = 20.0,
    g: float = 1.0,
    fit_modes: int = 4,
    timing_repeats: int = 7,
):
    """Massless Schwinger gaps from the exact bosonized Hamiltonian."""
    mu_exact = g / np.sqrt(np.pi)
    dvr_gap = []
    finite_difference_gap = []
    dvr_seconds = []
    finite_difference_seconds = []
    for npts in npts_values:
        dvr_call = lambda n=npts: _schwinger_gap_dvr(n, length, g, fit_modes)
        fd_call = lambda n=npts: _schwinger_gap_finite_difference(
            n, length, g, fit_modes
        )
        dvr_gap.append(dvr_call())
        finite_difference_gap.append(fd_call())
        dvr_seconds.append(_median_runtime(dvr_call, timing_repeats))
        finite_difference_seconds.append(_median_runtime(fd_call, timing_repeats))

    dvr_gap = np.asarray(dvr_gap)
    finite_difference_gap = np.asarray(finite_difference_gap)
    floor = np.finfo(float).eps
    return {
        "npts": np.asarray(npts_values),
        "cutoff_over_g": np.pi * np.asarray(npts_values) / (length * g),
        "dvr_vector_error": np.maximum(np.abs(dvr_gap - mu_exact) / g, floor),
        "fd_vector_error": np.maximum(
            np.abs(finite_difference_gap - mu_exact) / g, floor
        ),
        "dvr_scalar_error": np.maximum(
            np.abs(2.0 * dvr_gap - 2.0 * mu_exact) / g, floor
        ),
        "fd_scalar_error": np.maximum(
            np.abs(2.0 * finite_difference_gap - 2.0 * mu_exact) / g,
            floor,
        ),
        "dvr_seconds": np.asarray(dvr_seconds),
        "fd_seconds": np.asarray(finite_difference_seconds),
        "exact_vector_gap_over_g": 1.0 / np.sqrt(np.pi),
        "exact_scalar_gap_over_g": 2.0 / np.sqrt(np.pi),
        "length_times_g": length * g,
        "fit_modes": fit_modes,
        "timing_repeats": timing_repeats,
    }


def _style_axis(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_dispersion(data, output: Path):
    fig, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    axis.semilogy(
        data["momentum_fraction"],
        data["staggered_error"],
        "o-",
        label="nearest-neighbor staggered",
        markersize=4,
    )
    axis.semilogy(
        data["momentum_fraction"],
        data["dvr_error"],
        "s-",
        label="Fourier DVR",
        markersize=4,
    )
    axis.set_xlabel(r"resolved momentum $k/\Lambda$")
    axis.set_ylabel(r"relative dispersion error $|E_k-E_k^{\rm exact}|/E_k^{\rm exact}$")
    axis.set_title("Free Dirac dispersion at matched orbital count")
    axis.legend(frameon=False)
    _style_axis(axis)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_eigenvalue_convergence(data, output: Path):
    fig, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    axis.loglog(
        data["orbitals"],
        data["staggered_error"],
        "o-",
        label="nearest-neighbor staggered",
    )
    axis.loglog(data["orbitals"], data["dvr_error"], "s-", label="Fourier DVR")
    axis.set_xlabel("fermionic orbitals")
    axis.set_ylabel("maximum relative error in six lowest positive levels")
    axis.set_title(r"Smooth mass $m(x)=m_0+\delta m\cos(2\pi x/L)$")
    axis.legend(frameon=False)
    _style_axis(axis)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_gap_cutoff(data, output: Path):
    fig, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    axis.loglog(
        data["cutoff_over_g"], data["fd_vector_error"], "o-", label=r"FD $M_V$"
    )
    axis.loglog(
        data["cutoff_over_g"], data["fd_scalar_error"], "o--", label=r"FD $M_S$"
    )
    axis.loglog(
        data["cutoff_over_g"], data["dvr_vector_error"], "s-", label=r"DVR $M_V$"
    )
    axis.loglog(
        data["cutoff_over_g"], data["dvr_scalar_error"], "s--", label=r"DVR $M_S$"
    )
    axis.set_xlabel(r"ultraviolet cutoff $\Lambda/g$")
    axis.set_ylabel(r"absolute gap error $|M-M_{\rm exact}|/g$")
    axis.set_title("Bosonized massless Schwinger model")
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_gap_timing(data, output: Path):
    fig, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    axis.loglog(data["fd_seconds"], data["fd_vector_error"], "o-", label=r"FD $M_V$")
    axis.loglog(data["fd_seconds"], data["fd_scalar_error"], "o--", label=r"FD $M_S$")
    axis.loglog(data["dvr_seconds"], data["dvr_vector_error"], "s-", label=r"DVR $M_V$")
    axis.loglog(data["dvr_seconds"], data["dvr_scalar_error"], "s--", label=r"DVR $M_S$")
    axis.set_xlabel("median construction + eigensolve time (s)")
    axis.set_ylabel(r"absolute gap error $|M-M_{\rm exact}|/g$")
    axis.set_title("Accuracy-to-time tradeoff (single CPU thread)")
    axis.legend(frameon=False, loc="center right")
    _style_axis(axis)
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
    dispersion = dispersion_benchmark()
    inhomogeneous = inhomogeneous_mass_benchmark()
    gaps = schwinger_gap_benchmark()

    figures = {
        "dispersion": output_directory / "01_dispersion_error_vs_momentum.png",
        "eigenvalues": output_directory / "02_eigenvalue_error_vs_orbitals.png",
        "gap_cutoff": output_directory / "03_schwinger_gap_error_vs_cutoff.png",
        "gap_timing": output_directory / "04_schwinger_gap_error_vs_time.png",
    }
    plot_dispersion(dispersion, figures["dispersion"])
    plot_eigenvalue_convergence(inhomogeneous, figures["eigenvalues"])
    plot_gap_cutoff(gaps, figures["gap_cutoff"])
    plot_gap_timing(gaps, figures["gap_timing"])

    payload = {
        "description": (
            "DVR versus local spatial regulators; Schwinger gaps use the exact "
            "bosonized massless theory and a four-mode dispersion intercept."
        ),
        "dispersion": dispersion,
        "inhomogeneous_mass": inhomogeneous,
        "schwinger_gaps": gaps,
        "figures": {key: str(path) for key, path in figures.items()},
    }
    data_path = output_directory / "benchmark_data.json"
    data_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    print(f"wrote {data_path}")
    for path in figures.values():
        print(f"wrote {path}")
    print(
        "largest-cutoff errors: "
        f"FD M_V={gaps['fd_vector_error'][-1]:.3e}, "
        f"DVR M_V={gaps['dvr_vector_error'][-1]:.3e}"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"figure/data directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    run(args.output_directory)


if __name__ == "__main__":
    main()
