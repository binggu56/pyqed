"""Finite-size scaling trial using spin-boson susceptibility peaks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import SpinBosonWilsonNARG, log_discretized_spin_boson_wilson_chain


def _magnetization(
    alpha,
    *,
    epsilon,
    nmodes,
    nboson,
    bond_dim,
    Lambda,
    s,
    delta,
    basis,
    dvr_qmax,
):
    chain = log_discretized_spin_boson_wilson_chain(
        int(nmodes),
        alpha=float(alpha),
        Lambda=Lambda,
        s=s,
        omegac=1.0,
        epsilon=float(epsilon),
        delta=delta,
    )
    result = SpinBosonWilsonNARG(
        chain,
        nboson=nboson,
        bond_dim=bond_dim,
        basis=basis,
        displacements=None,
        dvr_qmax=dvr_qmax,
    ).run(nroots=2)
    return float(np.real(result.magnetizations[0])), float(result.energies[1] - result.energies[0])


def _scan_susceptibility(
    nmodes_list,
    alphas,
    *,
    epsilon=1e-7,
    nboson=16,
    bond_dim=64,
    Lambda=1.5,
    s=0.5,
    delta=0.1,
    basis="sine-dvr",
    dvr_qmax=12.0,
):
    chi = np.empty((len(nmodes_list), len(alphas)), dtype=float)
    gaps = np.empty_like(chi)
    for row, nmodes in enumerate(nmodes_list):
        for col, alpha in enumerate(alphas):
            plus, gap_plus = _magnetization(
                alpha,
                epsilon=epsilon,
                nmodes=nmodes,
                nboson=nboson,
                bond_dim=bond_dim,
                Lambda=Lambda,
                s=s,
                delta=delta,
                basis=basis,
                dvr_qmax=dvr_qmax,
            )
            minus, gap_minus = _magnetization(
                alpha,
                epsilon=-epsilon,
                nmodes=nmodes,
                nboson=nboson,
                bond_dim=bond_dim,
                Lambda=Lambda,
                s=s,
                delta=delta,
                basis=basis,
                dvr_qmax=dvr_qmax,
            )
            chi[row, col] = (plus - minus) / (2.0 * epsilon)
            gaps[row, col] = 0.5 * (gap_plus + gap_minus)
            print(
                f"N={int(nmodes):2d} alpha={alpha:.6f} "
                f"chi={chi[row, col]:.10e} gap={gaps[row, col]:.10e}",
                flush=True,
            )
    return chi, gaps


def _peak_quadratic(alphas, values):
    index = int(np.nanargmax(values))
    if index == 0 or index == len(alphas) - 1:
        return float(alphas[index]), float(values[index]), True
    x = alphas[index - 1 : index + 2]
    y = values[index - 1 : index + 2]
    coeffs = np.polyfit(x, y, 2)
    if coeffs[0] >= 0.0:
        return float(alphas[index]), float(values[index]), False
    peak_alpha = -coeffs[1] / (2.0 * coeffs[0])
    if peak_alpha < x[0] or peak_alpha > x[-1]:
        return float(alphas[index]), float(values[index]), False
    peak_value = float(np.polyval(coeffs, peak_alpha))
    return float(peak_alpha), peak_value, False


def _fit_peak_shift(nmodes, alpha_peaks, *, Lambda=1.5):
    best = None
    for alpha_c in np.linspace(0.04, min(alpha_peaks) - 1e-4, 120):
        shifts = alpha_peaks - alpha_c
        mask = shifts > 0.0
        if np.count_nonzero(mask) < 3:
            continue
        x = np.asarray(nmodes, dtype=float)[mask] * np.log(Lambda)
        y = np.log(shifts[mask])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss = float(np.sum((y - pred) ** 2))
        if best is None or ss < best["score"]:
            best = {
                "alpha_c": float(alpha_c),
                "inv_nu": float(-slope),
                "nu": float(-1.0 / slope) if slope != 0.0 else np.inf,
                "score": ss,
                "intercept": float(intercept),
            }
    return best


def _fit_peak_height(nmodes, chi_peaks, *, Lambda=1.5):
    x = np.asarray(nmodes, dtype=float) * np.log(Lambda)
    y = np.log(np.asarray(chi_peaks, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"gamma_over_nu": float(slope), "intercept": float(intercept), "r2": r2}


def main():
    Lambda = 1.5
    nmodes = np.array([10, 12, 14, 16])
    alphas = np.array([0.055, 0.070, 0.080, 0.090, 0.100, 0.115, 0.130, 0.150])
    epsilon = 1e-7

    chi, gaps = _scan_susceptibility(
        nmodes,
        alphas,
        epsilon=epsilon,
        Lambda=Lambda,
    )
    peaks = []
    peak_values = []
    edge_flags = []
    for row in range(len(nmodes)):
        alpha_peak, chi_peak, at_edge = _peak_quadratic(alphas, np.abs(chi[row]))
        peaks.append(alpha_peak)
        peak_values.append(chi_peak)
        edge_flags.append(at_edge)
    peaks = np.asarray(peaks)
    peak_values = np.asarray(peak_values)
    shift_fit = _fit_peak_shift(nmodes, peaks, Lambda=Lambda)
    height_fit = _fit_peak_height(nmodes, peak_values, Lambda=Lambda)

    print()
    print("susceptibility peaks")
    for nmode, peak, value, edge in zip(nmodes, peaks, peak_values, edge_flags):
        suffix = " edge" if edge else ""
        print(f"N={int(nmode):2d} alpha_peak={peak:.6f} chi_peak={value:.8e}{suffix}")
    print()
    if shift_fit is None:
        print("peak shift fit: not available")
    else:
        print(
            "peak shift fit: "
            f"alpha_c={shift_fit['alpha_c']:.6f}, "
            f"1/nu={shift_fit['inv_nu']:.6f}, "
            f"nu={shift_fit['nu']:.6f}, score={shift_fit['score']:.6e}"
        )
    print(
        "peak height fit: "
        f"gamma/nu={height_fit['gamma_over_nu']:.6f}, r2={height_fit['r2']:.6f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)
    for row, nmode in enumerate(nmodes):
        axes[0].plot(alphas, np.abs(chi[row]), marker="o", label=f"N={int(nmode)}")
        axes[1].semilogy(alphas, gaps[row], marker="o", label=f"N={int(nmode)}")
    axes[0].set_title(f"susceptibility, epsilon={epsilon:g}")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel(r"$|\partial m / \partial \epsilon|$")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("finite-chain gap")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("E1 - E0")
    axes[1].grid(True, which="both", alpha=0.25)

    axes[2].plot(nmodes, peaks, marker="o", label=r"$\alpha_{peak}(N)$")
    if shift_fit is not None:
        nline = np.linspace(float(min(nmodes)), float(max(nmodes)), 100)
        aline = shift_fit["alpha_c"] + np.exp(
            shift_fit["intercept"] - shift_fit["inv_nu"] * nline * np.log(Lambda)
        )
        axes[2].plot(nline, aline, "--", label="shift fit")
    axes[2].set_title("pseudocritical drift")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel(r"$\alpha_{peak}$")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.25)

    out = Path(__file__).with_name("spin_boson_susceptibility_fss.png")
    fig.savefig(out, dpi=180)
    print()
    print(out)


if __name__ == "__main__":
    main()
