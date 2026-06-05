"""Order-parameter scan for the sub-Ohmic spin-boson Wilson-chain NARG."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import SpinBosonWilsonNARG, log_discretized_spin_boson_wilson_chain


def _order_parameter_scan(
    alphas,
    *,
    epsilon,
    nmodes=16,
    nboson=16,
    bond_dim=64,
    Lambda=1.5,
    s=0.5,
    delta=0.1,
    basis="sine-dvr",
    dvr_qmax=12.0,
):
    magnetizations = []
    gaps = []
    for alpha in alphas:
        chain = log_discretized_spin_boson_wilson_chain(
            nmodes,
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
        magnetizations.append(float(np.real(result.magnetizations[0])))
        gaps.append(float(result.energies[1] - result.energies[0]))
        print(
            f"{alpha:.6f} {magnetizations[-1]: .10e} {gaps[-1]:.10e}",
            flush=True,
        )
    return np.asarray(magnetizations), np.asarray(gaps)


def _fit_beta_grid(alphas, magnetizations, alpha_min=0.055, alpha_max=0.12):
    alphas = np.asarray(alphas, dtype=float)
    mags = np.abs(np.asarray(magnetizations, dtype=float))
    candidates = np.linspace(alpha_min, alpha_max, 131)
    best = None
    for alpha_c in candidates:
        mask = (alphas > alpha_c) & (mags > 0.0) & np.isfinite(mags)
        if np.count_nonzero(mask) < 3:
            continue
        x = np.log(alphas[mask] - alpha_c)
        y = np.log(mags[mask])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
        score = ss_res
        if best is None or score < best["score"]:
            best = {
                "alpha_c": float(alpha_c),
                "beta": float(slope),
                "intercept": float(intercept),
                "r2": r2,
                "score": score,
                "mask": mask,
            }
    return best


def _fit_beta_fixed_alpha_c(alphas, magnetizations, alpha_c, *, min_alpha=None):
    alphas = np.asarray(alphas, dtype=float)
    mags = np.abs(np.asarray(magnetizations, dtype=float))
    mask = (alphas > float(alpha_c)) & (mags > 0.0) & np.isfinite(mags)
    if min_alpha is not None:
        mask &= alphas >= float(min_alpha)
    if np.count_nonzero(mask) < 2:
        return None
    x = np.log(alphas[mask] - float(alpha_c))
    y = np.log(mags[mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "alpha_c": float(alpha_c),
        "beta": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "mask": mask,
    }


def main():
    alphas = np.array(
        [0.040, 0.055, 0.070, 0.080, 0.085, 0.090, 0.095, 0.105, 0.115, 0.130, 0.150]
    )
    epsilon = 1e-7

    print("alpha magnetization gap")
    magnetizations, gaps = _order_parameter_scan(alphas, epsilon=epsilon)
    fit = _fit_beta_grid(alphas, magnetizations, alpha_min=0.070, alpha_max=0.090)
    fixed_fit = _fit_beta_fixed_alpha_c(alphas, magnetizations, 0.075, min_alpha=0.080)

    print()
    if fixed_fit is None:
        print("fixed-alpha_c beta fit: not enough localized-side points")
    else:
        print(
            "fixed-alpha_c beta fit: "
            f"alpha_c={fixed_fit['alpha_c']:.6f}, "
            f"beta={fixed_fit['beta']:.6f}, r2={fixed_fit['r2']:.6f}"
        )
    if fit is None:
        print("beta fit: not enough localized-side points")
    else:
        print(
            "rough beta grid fit: "
            f"alpha_c={fit['alpha_c']:.6f}, beta={fit['beta']:.6f}, r2={fit['r2']:.6f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(alphas, np.abs(magnetizations), marker="o")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("|<sigma_z>|")
    axes[0].set_title(f"order parameter, epsilon={epsilon:g}")
    axes[0].grid(True, alpha=0.25)

    axes[1].semilogy(alphas, gaps, marker="o")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("E1 - E0")
    axes[1].set_title("finite-chain gap")
    axes[1].grid(True, which="both", alpha=0.25)

    if fit is not None:
        axes[0].axvline(fit["alpha_c"], color="tab:red", linestyle=":", linewidth=1.4)
        fit_alphas = alphas[fit["mask"]]
        fit_curve = np.exp(fit["intercept"]) * (fit_alphas - fit["alpha_c"]) ** fit["beta"]
        axes[0].plot(fit_alphas, fit_curve, "--", color="tab:red")
    if fixed_fit is not None:
        axes[0].axvline(fixed_fit["alpha_c"], color="0.3", linestyle=":", linewidth=1.0)
        fit_alphas = alphas[fixed_fit["mask"]]
        fit_curve = (
            np.exp(fixed_fit["intercept"])
            * (fit_alphas - fixed_fit["alpha_c"]) ** fixed_fit["beta"]
        )
        axes[0].plot(fit_alphas, fit_curve, "--", color="0.3")

    out = Path(__file__).with_name("spin_boson_order_parameter_alpha.png")
    fig.savefig(out, dpi=180)
    print()
    print(out)


if __name__ == "__main__":
    main()
