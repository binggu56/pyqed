"""Locate spin-boson alpha_c from fixed-PES well-separation flow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg import scan_spin_boson_fpes_alpha


def main():
    alphas = np.array([0.050, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115])
    sites = np.array([7, 11, 15])
    q = np.linspace(-10.0, 10.0, 201)
    scan = scan_spin_boson_fpes_alpha(
        alphas,
        sites,
        q,
        nmodes=16,
        nboson=16,
        bond_dim=64,
        Lambda=1.5,
        s=0.5,
        delta=0.1,
        q0_threshold=0.5,
        basis="sine-dvr",
        displacements=None,
        dvr_qmax=10.0,
    )

    print("FPES alpha scan from endpoint q0")
    print(f"q0 threshold: {scan.q0_threshold:.6f}")
    if scan.pseudo_critical_alpha is None:
        print("pseudo alpha_c: not bracketed")
    else:
        print(f"pseudo alpha_c: {scan.pseudo_critical_alpha:.6f}")
    print()
    print("alpha endpoint_q0 q0_slope q0_by_N")
    for index, alpha in enumerate(scan.alphas):
        q0_text = " ".join(f"{value:.4f}" for value in scan.q0[index])
        print(
            f"{alpha:.6f} {scan.endpoint_q0[index]:.6f} "
            f"{scan.q0_slopes[index]: .6e} {q0_text}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for index, alpha in enumerate(scan.alphas):
        axes[0].plot(scan.sites + 1, scan.q0[index], marker="o", label=f"{alpha:.3f}")
    axes[0].axhline(scan.q0_threshold, color="0.3", linestyle="--", linewidth=1.0)
    axes[0].set_title("FPES well half-separation q0(N)")
    axes[0].set_xlabel("added mode N")
    axes[0].set_ylabel("q0")
    axes[0].legend(title="alpha", frameon=False, fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(scan.alphas, scan.endpoint_q0, marker="o", label="endpoint q0")
    axes[1].axhline(scan.q0_threshold, color="0.3", linestyle="--", linewidth=1.0)
    if scan.pseudo_critical_alpha is not None:
        axes[1].axvline(scan.pseudo_critical_alpha, color="tab:red", linestyle=":", linewidth=1.4)
    axes[1].set_title("endpoint q0 crossing")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("q0 at largest N")
    axes[1].grid(True, alpha=0.25)

    out = Path(__file__).with_name("spin_boson_fpes_alpha_c.png")
    fig.savefig(out, dpi=180)
    print()
    print(out)


if __name__ == "__main__":
    main()
