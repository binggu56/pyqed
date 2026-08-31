"""Plot the 3D rocksalt-LiH GDF/GW/BSE qualification suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np

from pyqed.units import au2mev


COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#5B5B5B",
}


def _load(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _study(payload):
    return payload["studies"][0]


def _observable_errors(row):
    values = {
        "J": float(row["max_abs_J_error_meV"]),
        "K": float(row["max_abs_K_error_meV"]),
    }
    if "native_krhf" in row:
        values["KRHF"] = abs(
            float(row["native_krhf"]["energy_error_vs_pyscf_gdf_Ha"])
        ) * au2mev
    if "gw" in row:
        values["GW"] = float(row["gw"]["max_abs_qp_error_meV"])
    if "bse" in row:
        values["BSE"] = max(
            float(row["bse"]["max_abs_A_error_Ha"]),
            float(row["bse"]["max_abs_B_error_Ha"]),
        ) * au2mev
    return values


def _format_nk_axis(axis, nkpts):
    axis.xaxis.set_major_locator(FixedLocator(nkpts))
    axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in nkpts]))
    axis.xaxis.set_minor_formatter(NullFormatter())


def plot(args):
    precision = _load(args.precision)["studies"]
    auxiliary = _load(args.auxiliary)["studies"]
    rows = [
        _study(_load(args.mesh_222)),
        _study(_load(args.mesh_422)),
        _study(_load(args.mesh_444)),
    ]
    derivative = _load(args.derivative)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(8.35, 9.1), constrained_layout=True)

    precisions = np.asarray([row["gdf_precision"] for row in precision])
    order = np.argsort(precisions)
    axes[0, 0].loglog(
        precisions[order],
        np.asarray([row["max_abs_J_error_meV"] for row in precision])[order],
        "o-",
        color=COLORS["blue"],
        label=r"$J$",
    )
    axes[0, 0].loglog(
        precisions[order],
        np.asarray([row["max_abs_K_error_meV"] for row in precision])[order],
        "s--",
        color=COLORS["orange"],
        label=r"$K$",
    )
    axes[0, 0].invert_xaxis()
    axes[0, 0].set(
        xlabel="GDF precision",
        ylabel="Maximum error (meV)",
        title="a  Precision control",
    )
    axes[0, 0].legend(frameon=False)

    labels = [
        "SV(P)-JKFIT",
        "universal-JKFIT",
        "SVP-RIFIT",
    ]
    x = np.arange(len(auxiliary))
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [row["max_abs_J_error_meV"] for row in auxiliary],
        width,
        color=COLORS["blue"],
        label=r"$J$",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["max_abs_K_error_meV"] for row in auxiliary],
        width,
        color=COLORS["orange"],
        label=r"$K$",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set(
        ylabel="Maximum error (meV)",
        title="b  Auxiliary basis",
    )
    axes[0, 1].legend(frameon=False)

    nkpts = np.asarray([row["nkpts"] for row in rows])
    errors = [_observable_errors(row) for row in rows]
    styles = {
        "J": ("o-", COLORS["blue"]),
        "K": ("s--", COLORS["orange"]),
        "KRHF": ("^:", COLORS["green"]),
        "GW": ("D-.", COLORS["purple"]),
        "BSE": ("v-", COLORS["gray"]),
    }
    for name, (style, color) in styles.items():
        selected = [
            (nk, values[name])
            for nk, values in zip(nkpts, errors)
            if name in values
        ]
        axes[1, 0].loglog(
            [item[0] for item in selected],
            [item[1] for item in selected],
            style,
            color=color,
            label=name,
        )
    axes[1, 0].set(
        xlabel=r"Number of $k$ points",
        ylabel="Maximum error (meV)",
        title="c  PySCF agreement",
        xticks=nkpts,
    )
    _format_nk_axis(axes[1, 0], nkpts)
    axes[1, 0].legend(frameon=False, ncols=2, fontsize=8)

    axes[1, 1].loglog(
        nkpts,
        [row["native_gdf_seconds"] for row in rows],
        "o-",
        color=COLORS["blue"],
        label="PyQED",
    )
    axes[1, 1].loglog(
        nkpts,
        [row["pyscf_gdf_build_seconds"] for row in rows],
        "s--",
        color=COLORS["orange"],
        label="PySCF",
    )
    axes[1, 1].set(
        xlabel=r"Number of $k$ points",
        ylabel="GDF build time (s)",
        title="d  Factor construction",
        xticks=nkpts,
    )
    _format_nk_axis(axes[1, 1], nkpts)
    axes[1, 1].legend(frameon=False)

    memory_mb = np.asarray([row["native_factor_bytes"] for row in rows]) / 1.0e6
    axes[2, 0].loglog(
        nkpts,
        memory_mb,
        "o-",
        color=COLORS["green"],
        label="materialized factors",
    )
    guide = memory_mb[0] * (nkpts / nkpts[0]) ** 2
    axes[2, 0].loglog(
        nkpts,
        guide,
        "--",
        color=COLORS["gray"],
        label=r"$N_k^2$",
    )
    axes[2, 0].set(
        xlabel=r"Number of $k$ points",
        ylabel="Factor storage (MB)",
        title="e  Memory scaling",
        xticks=nkpts,
    )
    _format_nk_axis(axes[2, 0], nkpts)
    axes[2, 0].legend(frameon=False)

    primitive = derivative["primitive_vs_commensurate"]
    finite = derivative["finite_difference_validation"]
    derivative_labels = (
        r"$S^{[1]}$",
        r"$F^{[1]}_{\rm exp}$",
        r"$F^{[1]}_{\rm CPHF}$",
        r"$F^{[1]}$",
        r"$K^{[1]}$",
        "FD total",
        "FD bare",
        "FD screened",
    )
    derivative_values = (
        primitive["overlap_derivative"]["relative_frobenius"],
        primitive["explicit_fock_derivative"]["relative_frobenius"],
        primitive["induced_fock_derivative"]["relative_frobenius"],
        primitive["fock_derivative"]["relative_frobenius"],
        primitive["screened_bse_kernel_derivative"]["relative_frobenius"],
        finite["relative_error"][0],
        finite["component_relative_error"]["bare"][0],
        finite["component_relative_error"]["screened"][0],
    )
    bar_colors = [COLORS["blue"]] * 5 + [COLORS["orange"]] * 3
    axes[2, 1].bar(
        np.arange(len(derivative_values)),
        derivative_values,
        color=bar_colors,
        width=0.7,
    )
    axes[2, 1].axvline(4.5, color="0.55", lw=0.8)
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xticks(
        np.arange(len(derivative_labels)),
        derivative_labels,
        rotation=28,
        ha="right",
    )
    axes[2, 1].set(
        ylabel="Relative Frobenius difference",
        title=r"f  Finite-$q$ derivative qualification",
    )

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2, lw=0.6, which="both")
        axis.spines[["top", "right"]].set_visible(False)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure: {output}")
    print(f"pdf: {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        type=Path,
        default=Path("/private/tmp/pbc_lih_222_svp_solid_precision_ladder_current.json"),
    )
    parser.add_argument(
        "--auxiliary",
        type=Path,
        default=Path("/private/tmp/pbc_lih_222_svp_solid_aux_ladder_current.json"),
    )
    parser.add_argument(
        "--mesh-222",
        type=Path,
        default=Path("/private/tmp/pbc_lih_svp_solid_222_full_validation_current.json"),
    )
    parser.add_argument(
        "--mesh-422",
        type=Path,
        default=Path("/private/tmp/pbc_lih_svp_solid_422_full_validation.json"),
    )
    parser.add_argument(
        "--mesh-444",
        type=Path,
        default=Path("/private/tmp/pbc_lih_svp_solid_444_factor_validation.json"),
    )
    parser.add_argument(
        "--derivative",
        type=Path,
        default=Path("/private/tmp/pbc_lih_3d_derivative_validation.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_lih_3d_qualification.png"),
    )
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
