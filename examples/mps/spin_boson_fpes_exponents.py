"""Extract scaling slopes from spin-boson fixed-PES observables."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import (
    log_discretized_spin_boson_wilson_chain,
    scan_spin_boson_fpes_observables,
)


def _wilson_fit(sites, values, *, Lambda=2.0, first=6, last=10):
    n = np.asarray(sites, dtype=float) + 1.0
    values = np.asarray(values, dtype=float)
    mask = (
        (n >= first)
        & (n <= last)
        & (values > 0.0)
        & np.isfinite(values)
    )
    x = (n[mask] - 1.0) * np.log(Lambda)
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "decay_exponent": float(-slope),
        "intercept": float(intercept),
        "r2": r2,
        "n": n[mask],
        "values": values[mask],
    }


def main():
    alpha = 0.335
    nmodes = 16
    nboson = 5
    bond_dim = 32
    Lambda = 2.0
    fit_window = (6, 10)
    chain = log_discretized_spin_boson_wilson_chain(
        nmodes,
        alpha=alpha,
        Lambda=Lambda,
        s=0.5,
        omegac=1.0,
        epsilon=0.0,
        delta=0.1,
    )
    q = np.linspace(-8.0, 8.0, 321)
    scan = scan_spin_boson_fpes_observables(
        chain,
        np.arange(nmodes),
        q,
        nboson=nboson,
        bond_dim=bond_dim,
        basis="sine-dvr",
        displacements=None,
        dvr_qmax=8.0,
    )

    quantities = {
        "q0": scan.q0,
        "well_separation": scan.well_separations,
        "barrier": scan.barrier_heights,
        "curvature": scan.curvatures,
        "energy_scale": scan.energy_scales,
        "omega": scan.onsite_frequencies,
        "coupling_norm": scan.coupling_norms,
    }
    fits = {
        name: _wilson_fit(
            scan.sites,
            values,
            Lambda=Lambda,
            first=fit_window[0],
            last=fit_window[1],
        )
        for name, values in quantities.items()
    }

    print(f"FPES observable scaling at alpha={alpha}")
    print(f"fit window: N={fit_window[0]}..{fit_window[1]}")
    print("quantity        decay_exponent      slope          r2")
    for name, fit in fits.items():
        print(
            f"{name:15s} {fit['decay_exponent']:15.8f} "
            f"{fit['slope']:13.8f} {fit['r2']:11.6f}"
        )

    print()
    print("N q0 well_sep barrier curvature energy_scale omega coupling_norm")
    for index, site in enumerate(scan.sites):
        print(
            f"{site + 1:2d} "
            f"{scan.q0[index]:.8e} "
            f"{scan.well_separations[index]:.8e} "
            f"{scan.barrier_heights[index]:.8e} "
            f"{scan.curvatures[index]:.8e} "
            f"{scan.energy_scales[index]:.8e} "
            f"{scan.onsite_frequencies[index]:.8e} "
            f"{scan.coupling_norms[index]:.8e}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, name in zip(axes, ["q0", "barrier", "curvature", "energy_scale"]):
        values = quantities[name]
        ax.semilogy(scan.sites + 1, values, marker="o")
        fit = fits[name]
        if len(fit["n"]) >= 2:
            nline = fit["n"]
            yline = np.exp(
                fit["intercept"] + fit["slope"] * (nline - 1.0) * np.log(Lambda)
            )
            ax.semilogy(nline, yline, "--", label=f"y={fit['decay_exponent']:.3f}")
            ax.legend(frameon=False)
        ax.axvspan(fit_window[0], fit_window[1], color="0.8", alpha=0.25)
        ax.set_title(name)
        ax.set_xlabel("added mode N")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Spin-boson FPES observable scaling at alpha={alpha}",
        fontsize=13,
    )
    out = Path(__file__).with_name("spin_boson_fpes_exponents.png")
    fig.savefig(out, dpi=180)
    print()
    print(out)


if __name__ == "__main__":
    main()
