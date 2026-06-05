"""Finite Wilson-chain critical-exponent diagnostics for spin-boson NARG.

This script uses centered sine-DVR oscillator boxes for the critical scan.
Automatic displacement is deliberately disabled here: it is useful deep in a
localized phase, but it can bias the pseudocritical onset in finite chains.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pyqed.narg import (
    SpinBosonWilsonNARG,
    fit_field_exponent,
    fit_gap_exponent,
    fit_order_parameter_exponent,
    log_discretized_spin_boson_wilson_chain,
    scan_spin_boson_alpha,
    scan_spin_boson_gap_thresholds,
)


def _first_largest_finite(nmodes, values):
    mask = np.isfinite(values)
    if not np.any(mask):
        return None, None
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(nmodes[indices])])
    return index, float(values[index])


def _print_gap_table(scan):
    print("Unbiased centered sine-DVR finite-size gap scan")
    print(f"gap threshold: {scan.threshold:.3e}")
    print()
    print("N    alpha_gap<thr   alpha_min_gap   min_gap")
    for nmode, alpha_thr, alpha_min, min_gap in zip(
        scan.nmodes,
        scan.threshold_alphas,
        scan.minimum_gap_alphas,
        scan.minimum_gaps,
    ):
        alpha_thr_text = f"{alpha_thr:.6f}" if np.isfinite(alpha_thr) else "not crossed"
        print(f"{nmode:2d}   {alpha_thr_text:>12}   {alpha_min:13.6f}   {min_gap:.6e}")

    print()
    print("alpha grid gaps")
    header = "alpha      " + "  ".join(f"N={int(nmode):<8d}" for nmode in scan.nmodes)
    print(header)
    for column, alpha in enumerate(scan.alphas):
        row = "  ".join(f"{scan.gaps[row, column]:.6e}" for row in range(len(scan.nmodes)))
        print(f"{alpha:8.5f}  {row}")


def main():
    # Sub-Ohmic spin-boson baths have ordinary power-law critical behavior.
    # The Ohmic s=1 case is KT-like and should be analyzed differently.
    nmodes = np.array([6, 7, 8])
    alphas = np.linspace(0.30, 0.45, 10)
    nboson = 6
    bond_dims = np.array([40, 44, 48])
    physics = dict(
        Lambda=2.0,
        s=0.5,
        omegac=1.0,
        delta=0.1,
    )
    basis_options = dict(
        basis="sine-dvr",
        displacements=None,
        dvr_qmax=8.0,
    )
    common = dict(
        **physics,
        **basis_options,
    )

    gap_scan = scan_spin_boson_gap_thresholds(
        nmodes,
        alphas,
        nboson=nboson,
        bond_dim=bond_dims,
        gap_threshold=1e-9,
        **common,
    )
    _print_gap_table(gap_scan)

    row, alpha_c = _first_largest_finite(gap_scan.nmodes, gap_scan.threshold_alphas)
    if alpha_c is None:
        print()
        print("No threshold crossing found; expand the alpha window or lower the gap threshold.")
        return

    print()
    print(f"working alpha_c from largest finite threshold: {alpha_c:.6f} (N={gap_scan.nmodes[row]})")

    left = gap_scan.alphas < alpha_c
    if np.count_nonzero(left) >= 2:
        gap_fit = fit_gap_exponent(gap_scan.alphas[left], gap_scan.gaps[row, left], alpha_c)
        print(
            "delocalized-side gap fit: "
            f"nu*z={gap_fit.exponent:.4f}, r2={gap_fit.r2:.4f}"
        )

    localized_alphas = gap_scan.alphas[gap_scan.alphas > alpha_c]
    if len(localized_alphas) >= 2:
        biased = scan_spin_boson_alpha(
            localized_alphas,
            nmodes=int(gap_scan.nmodes[row]),
            nboson=nboson,
            bond_dim=int(bond_dims[row]),
            epsilon=1e-5,
            nroots=2,
            **common,
        )
        beta_fit = fit_order_parameter_exponent(
            biased.alphas,
            biased.magnetizations,
            alpha_c,
        )
        print(
            "biased localized-side order fit: "
            f"beta={beta_fit.exponent:.4f}, r2={beta_fit.r2:.4f}"
        )

    epsilons = np.logspace(-6, -3, 4)
    mags = []
    for epsilon in epsilons:
        chain = log_discretized_spin_boson_wilson_chain(
            int(gap_scan.nmodes[row]),
            alpha=alpha_c,
            epsilon=float(epsilon),
            **physics,
        )
        result = SpinBosonWilsonNARG(
            chain,
            nboson=nboson,
            bond_dim=int(bond_dims[row]),
            **basis_options,
        ).run(nroots=1)
        mags.append(abs(float(np.real(result.magnetizations[0]))))

    field_fit = fit_field_exponent(epsilons, mags)
    print(
        "critical-field fit at working alpha_c: "
        f"1/delta={field_fit.exponent:.4f}, r2={field_fit.r2:.4f}"
    )


if __name__ == "__main__":
    main()
