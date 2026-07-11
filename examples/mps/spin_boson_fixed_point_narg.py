"""Search for spin-boson NARG fixed-point candidates from spectrum flow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pyqed.narg import scan_spin_boson_fixed_point_flows


def main():
    alphas = np.array([0.30, 0.33, 0.36, 0.39, 0.42])
    scan = scan_spin_boson_fixed_point_flows(
        alphas,
        nmodes=10,
        nboson=5,
        bond_dim=32,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
        nlevels=4,
        late_steps=3,
        basis="sine-dvr",
        displacements=None,
        dvr_qmax=8.0,
    )

    print("Centered sine-DVR spin-boson NARG fixed-point scan")
    if scan.crossover_alpha is None:
        print("crossover candidate: not resolved")
    else:
        print(f"largest endpoint change near alpha: {scan.crossover_alpha:.6f}")
    print()
    print("alpha       N-drift        endpoint rescaled gaps")
    for alpha, score, gaps in zip(scan.alphas, scan.drift_scores, scan.endpoint_spectra):
        gap_text = "  ".join(f"{gap:.6e}" for gap in gaps)
        print(f"{alpha:8.5f}  {score:12.6e}   {gap_text}")

    print()
    print("neighbor endpoint changes")
    for index, change in enumerate(scan.endpoint_changes):
        print(
            f"{scan.alphas[index]:.5f} -> {scan.alphas[index + 1]:.5f}: "
            f"{change:.6e}"
        )

    best = int(np.nanargmin(scan.drift_scores))
    print()
    print(f"lowest-drift flow by Wilson step at alpha={scan.alphas[best]:.6f}")
    for step, row in enumerate(scan.spectra[best]):
        gap_text = "  ".join(f"{gap:.6e}" for gap in row)
        print(f"{step:3d}   {gap_text}")


if __name__ == "__main__":
    main()
