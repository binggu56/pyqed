"""Finite-size scaling trial for spin-boson order-parameter NARG scans."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import SpinBosonWilsonNARG, log_discretized_spin_boson_wilson_chain


def _scan_order_parameter(
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
    mags = np.empty((len(nmodes_list), len(alphas)), dtype=float)
    gaps = np.empty_like(mags)
    for row, nmodes in enumerate(nmodes_list):
        for col, alpha in enumerate(alphas):
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
            mags[row, col] = abs(float(np.real(result.magnetizations[0])))
            gaps[row, col] = float(result.energies[1] - result.energies[0])
            print(
                f"N={int(nmodes):2d} alpha={alpha:.6f} "
                f"m={mags[row, col]:.10e} gap={gaps[row, col]:.10e}",
                flush=True,
            )
    return mags, gaps


def _pseudocritical_from_slope(alphas, mags):
    peaks = []
    slopes = []
    for row in range(mags.shape[0]):
        slope = np.gradient(mags[row], alphas)
        index = int(np.nanargmax(np.abs(slope)))
        peaks.append(float(alphas[index]))
        slopes.append(float(slope[index]))
    return np.asarray(peaks), np.asarray(slopes)


def _collapse_score(alphas, nmodes, mags, *, alpha_c, beta_over_nu, inv_nu, Lambda):
    xs = []
    ys = []
    for row, nmode in enumerate(nmodes):
        scale = Lambda ** (float(nmode) * inv_nu)
        xs.append((alphas - alpha_c) * scale)
        ys.append(mags[row] * Lambda ** (float(nmode) * beta_over_nu))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    score = 0.0
    count = 0
    bins = np.linspace(np.nanmin(x), np.nanmax(x), 10)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (x >= left) & (x < right)
        if np.count_nonzero(mask) >= 2:
            mean = float(np.mean(y[mask]))
            denom = max(abs(mean), 1e-12)
            score += float(np.mean(((y[mask] - mean) / denom) ** 2))
            count += 1
    if count == 0:
        return np.inf
    return score / count


def _grid_collapse(alphas, nmodes, mags, *, Lambda=1.5):
    best = None
    for alpha_c in np.linspace(0.065, 0.095, 31):
        for beta_over_nu in np.linspace(0.05, 0.6, 56):
            for inv_nu in np.linspace(0.1, 1.4, 53):
                score = _collapse_score(
                    alphas,
                    nmodes,
                    mags,
                    alpha_c=float(alpha_c),
                    beta_over_nu=float(beta_over_nu),
                    inv_nu=float(inv_nu),
                    Lambda=Lambda,
                )
                if best is None or score < best["score"]:
                    best = {
                        "alpha_c": float(alpha_c),
                        "beta_over_nu": float(beta_over_nu),
                        "inv_nu": float(inv_nu),
                        "score": float(score),
                    }
    best["nu"] = 1.0 / best["inv_nu"]
    best["beta"] = best["beta_over_nu"] * best["nu"]
    return best


def main():
    Lambda = 1.5
    nmodes = np.array([10, 12, 14, 16])
    alphas = np.array([0.055, 0.070, 0.080, 0.090, 0.100, 0.115, 0.130])
    epsilon = 1e-7

    mags, gaps = _scan_order_parameter(
        nmodes,
        alphas,
        epsilon=epsilon,
        Lambda=Lambda,
    )
    peaks, slopes = _pseudocritical_from_slope(alphas, mags)
    collapse = _grid_collapse(alphas, nmodes, mags, Lambda=Lambda)

    print()
    print("pseudocritical slope peaks")
    for nmode, peak, slope in zip(nmodes, peaks, slopes):
        print(f"N={int(nmode):2d} alpha_peak={peak:.6f} slope={slope:.8e}")
    print()
    print(
        "rough collapse: "
        f"alpha_c={collapse['alpha_c']:.6f}, "
        f"beta/nu={collapse['beta_over_nu']:.6f}, "
        f"1/nu={collapse['inv_nu']:.6f}, "
        f"nu={collapse['nu']:.6f}, "
        f"beta={collapse['beta']:.6f}, "
        f"score={collapse['score']:.6e}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)
    for row, nmode in enumerate(nmodes):
        axes[0].plot(alphas, mags[row], marker="o", label=f"N={int(nmode)}")
        axes[1].semilogy(alphas, gaps[row], marker="o", label=f"N={int(nmode)}")
    axes[0].set_title(f"order parameter, epsilon={epsilon:g}")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("|<sigma_z>|")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("finite-chain gap")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("E1 - E0")
    axes[1].grid(True, which="both", alpha=0.25)

    for row, nmode in enumerate(nmodes):
        x = (alphas - collapse["alpha_c"]) * Lambda ** (
            float(nmode) * collapse["inv_nu"]
        )
        y = mags[row] * Lambda ** (float(nmode) * collapse["beta_over_nu"])
        axes[2].plot(x, y, marker="o", linestyle="", label=f"N={int(nmode)}")
    axes[2].set_title("rough finite-size collapse")
    axes[2].set_xlabel(r"$(\alpha-\alpha_c)\Lambda^{N/\nu}$")
    axes[2].set_ylabel(r"$m\Lambda^{N\beta/\nu}$")
    axes[2].grid(True, alpha=0.25)

    out = Path(__file__).with_name("spin_boson_order_parameter_fss.png")
    fig.savefig(out, dpi=180)
    print()
    print(out)


if __name__ == "__main__":
    main()
