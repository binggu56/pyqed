"""Plot PES panels along an integrate-and-rescale k-shell NARG flow."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.narg.functional import Phi4LogShellNARG


def _fit_radial_surface(active_configs, values):
    radius2 = np.sum(active_configs * active_configs, axis=1)
    design = np.column_stack([np.ones_like(radius2), 0.5 * radius2, radius2 * radius2 / 24.0])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    fit = design @ coefficients
    return {
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


def _full_narg_residual_surface(toy, previous_s=None):
    result = toy.narg_effective_hamiltonian(nbranches=1)
    step_s = result.kinetic_dressing[:, 0, :, 0]
    if previous_s is None:
        previous_s = np.ones_like(step_s)
    total_s = previous_s * step_s
    residual_surface = np.diag(result.hamiltonian - result.active_kinetic * step_s)
    fit = _fit_radial_surface(result.active_configs, residual_surface)
    fit.update(_s_diagnostics(total_s))
    return result.active_configs, residual_surface, total_s, fit


def _quartic_calibration(log_factor, amplitude_npoints, field_range, quadrature_order):
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
    _, _, _, fit = _full_narg_residual_surface(toy)
    return fit["lambda_eff"]


def _single_shell_pes(mass2, coupling, *, log_factor, amplitude_npoints, field_range, quadrature_order):
    toy = Phi4LogShellNARG(
        cutoff=np.sqrt(log_factor),
        log_factor=log_factor,
        nshells=1,
        active_shells=1,
        amplitude_npoints=amplitude_npoints,
        field_range=field_range,
        mass2=mass2,
        coupling=coupling,
        quadrature_order=quadrature_order,
    )
    _, surface, _, _ = _full_narg_residual_surface(toy)
    return toy.amplitude_grid.copy(), surface.reshape(amplitude_npoints, amplitude_npoints)


def _integrate_rescale_pes(
    mass2,
    coupling,
    *,
    log_factor,
    amplitude_npoints,
    field_range,
    quadrature_order,
    quartic_factor,
    previous_s,
):
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
    _, surface, s_kernel, fit = _full_narg_residual_surface(toy, previous_s=previous_s)
    surface = surface.reshape(amplitude_npoints, amplitude_npoints)

    next_mass2 = log_factor**2 * (fit["omega2_eff"] - 1.0)
    next_coupling = log_factor**3 * fit["lambda_eff"] / quartic_factor

    q_rescaled = toy.amplitude_grid / np.sqrt(log_factor)
    surface_rescaled = log_factor * surface
    return q_rescaled, surface_rescaled, next_mass2, next_coupling, s_kernel, fit


def _relative(surface):
    surface = np.asarray(surface, dtype=float)
    return surface - np.min(surface)


def main():
    log_factor = 2.0
    amplitude_npoints = 5
    field_range = 4.5
    quadrature_order = 160
    nsteps = 1

    mass2 = 0.5
    coupling = 0.8
    quartic_factor = _quartic_calibration(
        log_factor,
        amplitude_npoints,
        field_range,
        quadrature_order,
    )

    panels = []
    s_kernel = None
    q, surface = _single_shell_pes(
        mass2,
        coupling,
        log_factor=log_factor,
        amplitude_npoints=amplitude_npoints,
        field_range=field_range,
        quadrature_order=quadrature_order,
    )
    panels.append(
        {
            "step": 0,
            "q": q,
            "surface": _relative(surface),
            "mass2": mass2,
            "coupling": coupling,
            "s_min": np.nan,
            "s_offdiag_rms": np.nan,
            "rms_error": np.nan,
        }
    )

    for step in range(1, nsteps + 1):
        q, surface, mass2, coupling, s_kernel, fit = _integrate_rescale_pes(
            mass2,
            coupling,
            log_factor=log_factor,
            amplitude_npoints=amplitude_npoints,
            field_range=field_range,
            quadrature_order=quadrature_order,
            quartic_factor=quartic_factor,
            previous_s=s_kernel,
        )
        panels.append(
            {
                "step": step,
                "q": q,
                "surface": _relative(surface),
                "mass2": mass2,
                "coupling": coupling,
                "s_min": fit["s_min"],
                "s_offdiag_rms": fit["s_offdiag_rms"],
                "rms_error": fit["rms_error"],
            }
        )

    print("Rescaled PES panels for k-shell NARG flow")
    print("step   mass2          coupling       min(S)      rms(S_off-1)   max PES        fit_rms")
    for panel in panels:
        rms = panel["rms_error"]
        rms_text = "nan" if np.isnan(rms) else f"{rms:.3e}"
        print(
            f"{panel['step']:4d}  {panel['mass2']: .10f}  {panel['coupling']: .10f}  "
            f"{panel['s_min']: .8f}  {panel['s_offdiag_rms']: .8f}  "
            f"{np.max(panel['surface']): .8f}  {rms_text:>9s}"
        )

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.8, 7.3), constrained_layout=True)
    axes = np.ravel(axes)

    for axis, panel in zip(axes, panels):
        q = panel["q"]
        surface = panel["surface"]
        contour = axis.contourf(q, q, surface.T, levels=24, cmap="magma")
        axis.contour(q, q, surface.T, levels=7, colors="black", linewidths=0.35, alpha=0.45)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"rescaled $q_{\cos k}$")
        axis.set_ylabel(r"rescaled $q_{\sin k}$")
        axis.set_title(
            f"step {panel['step']}\n"
            + rf"$m^2={panel['mass2']:.2f},\ g={panel['coupling']:.2f}$"
        )
        fig.colorbar(contour, ax=axis, shrink=0.82)

    for axis in axes[len(panels) :]:
        axis.set_visible(False)

    fig.suptitle(r"Integrate UV $\pm k$ shell, then rescale $k,q,H$")
    output = Path(__file__).with_name("phi4_log_shell_rg_pes_flow.png")
    fig.savefig(output, dpi=220)
    print(output)


if __name__ == "__main__":
    main()
