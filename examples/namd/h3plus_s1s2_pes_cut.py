#!/usr/bin/env python3
"""Plot direct and Procrustes-fitted H3+ S1/S2 PES along Qy=0."""

import json
from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.qchem import Molecule
from pyqed.units import au2ev


output = Path("/private/tmp/h3plus_s1s2_branching_tnldr_vs_direct_9x9_10fs")
output.mkdir(parents=True, exist_ok=True)
bond_length = 1.65
breathing_coordinate = -0.20
seam_qx = -0.015


def geometry(q):
    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    stretch = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.diag(jnp.asarray((1.0, -1.0)))
    )
    shear = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.asarray(((0.0, 1.0), (1.0, 0.0)))
    )
    qx, qy = q
    return (
        (bond_length + breathing_coordinate) * triangle
        + (0.015 + qx) * stretch
        + qy * shear
    )


grid = DVR.from_axes(
    (SineDVR(-0.20, 0.20, 9), SineDVR(-0.20, 0.20, 9)),
    names=("Qx", "Qy"),
)
coord = Coord(to_cartesian=geometry, bounds=((-0.20, 0.20), (-0.20, 0.20)))
mol = Molecule(
    atom=list(zip(("H", "H", "H"), np.asarray(geometry((0.0, 0.0))))),
    charge=1,
    spin=0,
    unit="bohr",
    basis="sto-3g",
).build(eri="dense")
mf = mol.RHF().run()
mc = mol.casci(3, 2, nstates=3, mf=mf).run(nstates=3)
fit = AbInitioFit(
    mc,
    coord=coord,
    states=(1, 2),
    fit_options={"degrees": 8},
).build()
direct = fit.direct_product(grid, keo=keo.podolsky())

qy_index = int(np.argmin(np.abs(grid.x[1])))
qx_data = np.asarray(grid.x[0])
direct_energy = np.sort(np.asarray(direct.energies[:, qy_index]), axis=-1)
fitted_at_data = np.linalg.eigvalsh(
    fit.energy.predict(np.column_stack((qx_data, np.zeros_like(qx_data))))
)
qx_dense = np.linspace(qx_data[0], qx_data[-1], 401)
fitted_dense = np.linalg.eigvalsh(
    fit.energy.predict(np.column_stack((qx_dense, np.zeros_like(qx_dense))))
)
zero = float(np.min(direct_energy))
direct_ev = (direct_energy - zero) * au2ev
fitted_data_ev = (fitted_at_data - zero) * au2ev
fitted_dense_ev = (fitted_dense - zero) * au2ev
error_mev = (fitted_at_data - direct_energy) * au2ev * 1.0e3

summary = {
    "cut": "Qy=0",
    "fit_gauge": fit.config["gauge"],
    "unitarize_links": fit.config["unitarize_links"],
    "adaptive_geometries": len(fit.info["points"]),
    "direct_geometries": direct.direct_product_info["geometries"],
    "max_abs_energy_error_meV": np.max(np.abs(error_mev), axis=0).tolist(),
    "rms_energy_error_meV": np.sqrt(np.mean(error_mev**2, axis=0)).tolist(),
    "max_abs_gap_error_meV": float(
        np.max(np.abs(np.diff(fitted_data_ev, axis=1) - np.diff(direct_ev, axis=1)))
        * 1.0e3
    ),
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
    }
)
figure, panels = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
colors = ("#0072B2", "#D55E00")
for state, color in enumerate(colors):
    physical = state + 1
    panels[0].plot(qx_dense, fitted_dense_ev[:, state], color=color, label=rf"Fit $S_{physical}$")
    panels[0].plot(
        qx_data,
        direct_ev[:, state],
        linestyle="none",
        marker="o" if state == 0 else "s",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=color,
        label=rf"Direct $S_{physical}$",
    )
    panels[1].plot(
        qx_data,
        error_mev[:, state],
        marker="o" if state == 0 else "s",
        markersize=4.0,
        color=color,
        label=rf"$S_{physical}$",
    )
for panel in panels:
    panel.axvline(seam_qx, color="0.35", linestyle=":", linewidth=1.0)
    panel.grid(alpha=0.2, linewidth=0.6)
panels[0].set(xlabel=r"$Q_x$ (bohr)", ylabel="Energy relative to cut minimum (eV)")
panels[0].legend(frameon=False, ncol=2)
panels[1].axhline(0.0, color="black", linewidth=0.8)
panels[1].set(xlabel=r"$Q_x$ (bohr)", ylabel="Fit $-$ direct (meV)")
panels[1].legend(frameon=False)
for label, panel in zip(("a", "b"), panels):
    panel.text(-0.16, 1.04, label, transform=panel.transAxes, fontweight="bold")

stem = output / "h3plus_s1s2_pes_cut_qy0"
figure.savefig(stem.with_suffix(".pdf"))
figure.savefig(stem.with_suffix(".png"), dpi=350)
np.savez(
    stem.with_suffix(".npz"),
    qx_data=qx_data,
    qx_dense=qx_dense,
    direct_energy_ev=direct_ev,
    fitted_energy_at_data_ev=fitted_data_ev,
    fitted_energy_dense_ev=fitted_dense_ev,
    error_mev=error_mev,
)
stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"figure: {stem.with_suffix('.pdf')}")
print(f"figure: {stem.with_suffix('.png')}")

y = np.asarray(
    fit.feature.predict(np.column_stack((qx_dense, np.zeros_like(qx_dense))))
)
gram = np.einsum("xra,xrb->xab", y.conj(), y, optimize=True)
defect = gram - np.eye(y.shape[-1])
singular = np.linalg.svd(y, compute_uv=False)
y_summary = {
    "cut": "Qy=0",
    "shape": list(y.shape),
    "feature_rank": int(y.shape[1]),
    "electronic_states": int(y.shape[2]),
    "max_abs_imaginary_component": float(np.max(np.abs(y.imag))),
    "max_orthogonality_defect": float(
        np.max(np.linalg.norm(defect, axis=(-2, -1)))
    ),
    "singular_value_range": [float(np.min(singular)), float(np.max(singular))],
    "sampled_link_rms_relative_error": float(
        fit.info["synchronization"]["rms_relative_link_error"]
    ),
    "sampled_link_max_relative_error": float(
        fit.info["synchronization"]["maximum_relative_link_error"]
    ),
    "fitted_link_initial_relative_error": float(
        fit.info["variational"]["initial_relative_link_error"]
    ),
    "fitted_link_final_relative_error": float(
        fit.info["variational"].get(
            "candidate_relative_link_error",
            fit.info["variational"].get("rms_relative_link_error"),
        )
    ),
    "fitted_link_optimization_accepted": bool(
        fit.info["variational"]["accepted"]
    ),
}

figure_y, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
component_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
component_styles = ("-", "--", "-.", ":")
for state, panel in enumerate(axes[0]):
    for component, (color, style) in enumerate(
        zip(component_colors, component_styles)
    ):
        panel.plot(
            qx_dense,
            y[:, component, state].real,
            color=color,
            linestyle=style,
            label=rf"$Y_{{{component + 1},{state + 1}}}$",
        )
    panel.set(xlabel=r"$Q_x$ (bohr)", ylabel=rf"$Y_{{r,{state + 1}}}$")
    panel.legend(frameon=False, ncol=2)
for state, color in enumerate(colors):
    axes[1, 0].plot(
        qx_dense,
        singular[:, state],
        color=color,
        label=rf"$\sigma_{state + 1}(Y)$",
    )
axes[1, 0].set(xlabel=r"$Q_x$ (bohr)", ylabel="Singular value")
axes[1, 0].legend(frameon=False)
defect_curves = (
    (defect[:, 0, 0].real, r"$(Y^\dagger Y-I)_{11}$"),
    (defect[:, 1, 1].real, r"$(Y^\dagger Y-I)_{22}$"),
    (defect[:, 0, 1].real, r"Re $(Y^\dagger Y-I)_{12}$"),
)
for (values, label), color, style in zip(
    defect_curves, component_colors, component_styles
):
    axes[1, 1].plot(qx_dense, values, color=color, linestyle=style, label=label)
axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
axes[1, 1].set(xlabel=r"$Q_x$ (bohr)", ylabel="Orthogonality defect")
axes[1, 1].legend(frameon=False)
for label, panel in zip(("a", "b", "c", "d"), axes.flat):
    panel.axvline(seam_qx, color="0.35", linestyle=":", linewidth=1.0)
    panel.grid(alpha=0.2, linewidth=0.6)
    panel.text(-0.16, 1.04, label, transform=panel.transAxes, fontweight="bold")

y_stem = output / "h3plus_s1s2_fitted_y_qy0"
figure_y.savefig(y_stem.with_suffix(".pdf"))
figure_y.savefig(y_stem.with_suffix(".png"), dpi=350)
np.savez(
    y_stem.with_suffix(".npz"),
    qx=qx_dense,
    y=y,
    gram=gram,
    orthogonality_defect=defect,
    singular_values=singular,
)
y_stem.with_suffix(".json").write_text(json.dumps(y_summary, indent=2) + "\n")
print(json.dumps(y_summary, indent=2))
print(f"figure: {y_stem.with_suffix('.pdf')}")
print(f"figure: {y_stem.with_suffix('.png')}")
