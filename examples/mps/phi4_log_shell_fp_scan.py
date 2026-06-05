"""Small fixed-point diagnostic for k-log phi4 NARG shells."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4LogShellNARG


def main():
    cutoff = 4.0
    log_factor = 2.0
    rows = []
    for nshells in range(1, 5):
        toy = Phi4LogShellNARG(
            cutoff=cutoff,
            log_factor=log_factor,
            nshells=nshells,
            active_shells=1,
            amplitude_npoints=3,
            field_range=4.5,
            mass2=0.5,
            coupling=0.8,
            quadrature_order=max(128, 48 * nshells),
        )
        fit = toy.fit_ir_shell_effective_potential(max_power=4)
        coeff = fit["coefficients"]
        k_ir = toy.shell_representatives[-1]
        omega2_eff = coeff["omega2_eff"]
        lambda_eff = coeff["lambda_eff"]
        rows.append(
            {
                "nshells": nshells,
                "k_ir": k_ir,
                "ir_cutoff": toy.ir_cutoff,
                "omega2_eff": omega2_eff,
                "lambda_eff": lambda_eff,
                "omega2_over_k2": omega2_eff / (k_ir * k_ir),
                "shape_lambda": lambda_eff / (omega2_eff * omega2_eff),
                "rms_error": fit["rms_error"],
            }
        )

    print("Exact small-shell k-log NARG fixed-point diagnostic")
    print("N  k_IR       omega2_eff   lambda_eff   omega2/k^2   lambda/omega2^2   rms")
    for row in rows:
        print(
            f"{row['nshells']:1d}  {row['k_ir']:.6f}  {row['omega2_eff']: .8f}  "
            f"{row['lambda_eff']: .8f}  {row['omega2_over_k2']: .8f}  "
            f"{row['shape_lambda']: .8f}  {row['rms_error']:.3e}"
        )

    steps = np.array([row["nshells"] for row in rows])
    mass_ratio = np.array([row["omega2_over_k2"] for row in rows])
    shape_lambda = np.array([row["shape_lambda"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    axes[0].plot(steps, mass_ratio, "o-", color="#39568c")
    axes[0].set_xlabel("number of k-log shells")
    axes[0].set_ylabel(r"$\omega^2_{\rm eff}/k_{\rm IR}^2$")
    axes[0].set_title("relevant mass direction")
    axes[0].set_xticks(steps)

    axes[1].plot(steps, shape_lambda, "o-", color="#b03a48")
    axes[1].set_xlabel("number of k-log shells")
    axes[1].set_ylabel(r"$\lambda_{\rm eff}/\omega_{\rm eff}^4$")
    axes[1].set_title("shape-normalized quartic")
    axes[1].set_xticks(steps)

    output = Path(__file__).with_name("phi4_log_shell_fp_scan.png")
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
