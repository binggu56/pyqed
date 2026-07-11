"""Iterated k-shell integrate-and-rescale flow for phi4 NARG."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4LogShellNARG


def fit_radial_surface(active_configs, values):
    """Fit ``V(r) = c0 + omega2 r^2/2 + lambda r^4/24``."""
    radius2 = np.sum(active_configs * active_configs, axis=1)
    design = np.column_stack([np.ones_like(radius2), 0.5 * radius2, radius2 * radius2 / 24.0])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fit = design @ coefficients
    return {
        "c0": float(coefficients[0]),
        "omega2_eff": float(coefficients[1]),
        "lambda_eff": float(coefficients[2]),
        "rms_error": float(np.sqrt(np.mean((values - fit) ** 2))),
    }


def _s_diagnostics(s_kernel):
    offdiag = s_kernel[~np.eye(s_kernel.shape[0], dtype=bool)]
    return {
        "s_min": float(np.min(s_kernel)),
        "s_offdiag_rms": float(np.sqrt(np.mean((offdiag - 1.0) ** 2))),
    }


def full_narg_shell_fit(toy):
    """Fit the residual PES and recompute the full NARG overlap kernel."""
    result = toy.narg_effective_hamiltonian(nbranches=1)
    s_kernel = result.kinetic_dressing[:, 0, :, 0]
    residual_surface = np.diag(result.hamiltonian - result.active_kinetic * s_kernel)
    fit = fit_radial_surface(result.active_configs, residual_surface)
    fit["s_kernel"] = s_kernel
    fit.update(_s_diagnostics(s_kernel))
    return fit


def calibrate_shell_quartic(log_factor, amplitude_npoints, field_range, quadrature_order):
    """Return the single-shell fit factor between model coupling and q^4 PES."""
    toy = Phi4LogShellNARG(
        cutoff=np.sqrt(log_factor),
        log_factor=log_factor,
        nshells=1,
        active_shells=1,
        amplitude_npoints=amplitude_npoints,
        field_range=field_range,
        mass2=0.0,
        coupling=1.0,
        quadrature_order=quadrature_order,
    )
    return full_narg_shell_fit(toy)["lambda_eff"]


def rg_step(
    mass2,
    coupling,
    *,
    log_factor,
    amplitude_npoints,
    field_range,
    quadrature_order,
    quartic_factor,
):
    """Integrate one UV cos/sin supersite and rescale the IR supersite.

    The rescaling is the standard mode-coordinate rescaling ``q' = q/sqrt(b)``
    and ``H' = b H``.  The full NARG overlap kernel is recomputed from the
    conditional retained state of the enlarged environment at this split.
    """
    toy = Phi4LogShellNARG(
        cutoff=log_factor**1.5,
        log_factor=log_factor,
        nshells=2,
        active_shells=1,
        amplitude_npoints=amplitude_npoints,
        field_range=field_range,
        mass2=mass2,
        coupling=coupling,
        quadrature_order=quadrature_order,
    )
    fit = full_narg_shell_fit(toy)
    k_ir = toy.mode_wave_numbers[toy.active_modes[0]]

    if not np.allclose(k_ir, 1.0, atol=1e-12):
        raise RuntimeError("The two-shell RG regulator should leave the active shell at k=1.")

    new_mass2 = log_factor**2 * (fit["omega2_eff"] - 1.0)
    new_coupling = log_factor**3 * fit["lambda_eff"] / quartic_factor
    return {
        "mass2": float(new_mass2),
        "coupling": float(new_coupling),
        "omega2_eff": float(fit["omega2_eff"]),
        "lambda_eff": float(fit["lambda_eff"]),
        "s_kernel": fit["s_kernel"].copy(),
        "s_min": float(fit["s_min"]),
        "s_offdiag_rms": float(fit["s_offdiag_rms"]),
        "rms_error": float(fit["rms_error"]),
    }


def main():
    log_factor = 2.0
    amplitude_npoints = 9
    field_range = 4.5
    quadrature_order = 160
    max_steps = 40
    tolerance = 1e-7
    max_abs_coupling = 20.0

    quartic_factor = calibrate_shell_quartic(
        log_factor,
        amplitude_npoints,
        field_range,
        quadrature_order,
    )

    mass2 = 0.5
    coupling = 0.8
    rows = [
        {
            "step": 0,
            "mass2": mass2,
            "coupling": coupling,
            "delta": np.nan,
            "s_min": np.nan,
            "s_offdiag_rms": np.nan,
            "rms_error": np.nan,
        }
    ]

    converged = False
    for step in range(1, max_steps + 1):
        previous = np.array([mass2, coupling], dtype=float)
        mapped = rg_step(
            mass2,
            coupling,
            log_factor=log_factor,
            amplitude_npoints=amplitude_npoints,
            field_range=field_range,
            quadrature_order=quadrature_order,
            quartic_factor=quartic_factor,
        )
        mass2 = mapped["mass2"]
        coupling = mapped["coupling"]
        current = np.array([mass2, coupling], dtype=float)
        delta = float(np.linalg.norm(current - previous))
        rows.append(
            {
                "step": step,
                "mass2": mass2,
                "coupling": coupling,
                "delta": delta,
                "s_min": mapped["s_min"],
                "s_offdiag_rms": mapped["s_offdiag_rms"],
                "rms_error": mapped["rms_error"],
            }
        )
        if delta < tolerance:
            converged = True
            break
        if not np.all(np.isfinite(current)) or np.max(np.abs(current)) > max_abs_coupling:
            break

    print("Iterated k-shell NARG integrate/rescale flow")
    print(f"log factor             : {log_factor:.6f}")
    print(f"quartic calibration    : {quartic_factor:.8f}")
    print(f"tolerance              : {tolerance:.1e}")
    print("step   mass2          coupling       min(S)      rms(S_off-1)   delta          fit_rms")
    for row in rows:
        delta = row["delta"]
        rms = row["rms_error"]
        delta_text = "nan" if np.isnan(delta) else f"{delta:.3e}"
        rms_text = "nan" if np.isnan(rms) else f"{rms:.3e}"
        print(
            f"{row['step']:4d}  {row['mass2']: .10f}  {row['coupling']: .10f}  "
            f"{row['s_min']: .8f}  {row['s_offdiag_rms']: .8f}  "
            f"{delta_text:>11s}  {rms_text:>9s}"
        )
    print(f"converged              : {converged}")

    steps = np.array([row["step"] for row in rows])
    masses = np.array([row["mass2"] for row in rows])
    couplings = np.array([row["coupling"] for row in rows])
    deltas = np.array([row["delta"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].plot(steps, masses, "o-", label=r"$m^2$")
    axes[0].plot(steps, couplings, "s-", label=r"$g$")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_xlabel("RG step")
    axes[0].set_ylabel("rescaled coupling")
    axes[0].set_title("integrate + rescale map")
    axes[0].legend(frameon=False)

    axes[1].semilogy(steps[1:], deltas[1:], "o-", color="#6b3f8f")
    axes[1].axhline(tolerance, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("RG step")
    axes[1].set_ylabel("map distance")
    axes[1].set_title("fixed-point distance")

    output = Path(__file__).with_name("phi4_log_shell_rg_flow.png")
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
