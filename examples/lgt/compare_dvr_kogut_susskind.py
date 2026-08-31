#!/usr/bin/env python3
"""Matched-orbital ED comparison of gauge-DVR and staggered Schwinger models.

The comparison uses ``N_KS = 2*N_DVR``: a DVR point carries two Dirac
components, whereas a Kogut--Susskind site carries one staggered component.
The two calculations therefore have the same number of fermion modes and the
same half-filled matter-space dimension at each comparison point.

Both ED calculations are periodic and extract the vector rest mass from the
lowest nonzero density momentum.  This removes the boundary/interpolator
mismatch, but the short grid ladder remains a regulator pilot rather
than a controlled continuum extrapolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.lgt import KogutSusskindED, QuantumSchwingerDVR


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "dvr_vs_kogut_susskind"
SAVED_DVR = {
    9: HERE
    / "results"
    / "dynamical_schwinger_dvr_n9"
    / "dynamical_schwinger_data.json",
}


def dominant_channel(strengths, first_root):
    strengths = np.asarray(strengths, dtype=float)
    if first_root >= strengths.size or np.max(strengths[first_root:]) <= 0.0:
        raise RuntimeError("no excited state has nonzero channel strength")
    return int(first_root + np.argmax(strengths[first_root:]))


def dvr_point(npts, *, length, coupling, mass, flux_cutoff, nroots):
    started = perf_counter()
    model = QuantumSchwingerDVR(
        npts,
        length,
        coupling=coupling,
        mass=mass,
        flux_cutoff=flux_cutoff,
    ).run(nroots=nroots)
    seconds = perf_counter() - started
    vector_root = dominant_channel(model.vector_strengths, model.vacuum_dimension)
    scalar_root = dominant_channel(model.scalar_strengths, model.vacuum_dimension)
    gaps = np.asarray(model.energies) - model.energies[0]
    vector_energy = float(gaps[vector_root])
    vector_mass = np.sqrt(max(vector_energy**2 - model.vector_momentum**2, 0.0))
    return {
        "regulator": "Wilson-DVR",
        "boundary": "periodic",
        "fermion_modes": int(2 * npts),
        "effective_cells": int(npts),
        "spatial_sites": int(npts),
        "dimension": int(model.dimension),
        "seconds": float(seconds),
        "vector_root": vector_root,
        "scalar_root": scalar_root,
        "vector_excitation_energy_over_g": float(vector_energy / coupling),
        "vector_momentum_over_g": float(model.vector_momentum / coupling),
        "M_V_over_g": float(vector_mass / coupling),
        "M_S_over_g": float(gaps[scalar_root] / coupling),
        "vector_strength": float(model.vector_strengths[vector_root]),
        "scalar_strength": float(model.scalar_strengths[scalar_root]),
        "vacuum_dimension": int(model.vacuum_dimension),
    }


def saved_dvr_point(
    path,
    npts,
    *,
    length,
    coupling,
    mass,
    flux_cutoff,
    nroots,
):
    payload = json.loads(Path(path).read_text())
    parameters = payload["parameters"]
    expected = {
        "npts": int(npts),
        "length_times_g": float(length * coupling),
        "fermion_mass_over_g": float(mass / coupling),
        "nroots": int(nroots),
    }
    for key, value in expected.items():
        if not np.isclose(parameters[key], value):
            raise ValueError(
                f"saved DVR parameter {key}={parameters[key]!r} "
                f"does not match {value!r}"
            )
    record = next(
        row
        for row in payload["flux_convergence"]
        if int(row["flux_cutoff"]) == int(flux_cutoff)
    )
    return {
        "regulator": "Wilson-DVR",
        "boundary": "periodic",
        "fermion_modes": int(2 * npts),
        "effective_cells": int(npts),
        "spatial_sites": int(npts),
        "dimension": int(record["dimension"]),
        "seconds": float(record["seconds"]),
        "vector_root": int(record["vector_level"]),
        "scalar_root": int(record["scalar_level"]),
        "vector_excitation_energy_over_g": float(
            record["vector_excitation_energy"] / coupling
        ),
        "vector_momentum_over_g": float(record["vector_momentum"] / coupling),
        "M_V_over_g": float(record["vector_gap"] / coupling),
        "M_S_over_g": float(record["scalar_gap"] / coupling),
        "vector_strength": None,
        "scalar_strength": None,
        "vacuum_dimension": int(record["vacuum_dimension"]),
        "reused_from": str(Path(path)),
    }


def staggered_point(nsites, *, length, coupling, mass, flux_cutoff, nroots):
    started = perf_counter()
    model = KogutSusskindED(
        nsites,
        length,
        coupling=coupling,
        mass=mass,
        flux_cutoff=flux_cutoff,
        boundary="periodic",
    ).run(nroots=nroots)
    seconds = perf_counter() - started
    ground = model.states[:, 0]
    vector_strengths = np.abs(
        model.states.conj().T @ (model.build_vector_operator() @ ground)
    ) ** 2
    scalar_strengths = np.abs(
        model.states.conj().T @ (model.build_scalar_operator() @ ground)
    ) ** 2
    vector_root = dominant_channel(vector_strengths, 1)
    scalar_root = dominant_channel(scalar_strengths, 1)
    gaps = np.asarray(model.energies) - model.energies[0]
    vector_energy = float(gaps[vector_root])
    vector_mass = np.sqrt(max(vector_energy**2 - model.vector_momentum**2, 0.0))
    return {
        "regulator": "Kogut-Susskind",
        "boundary": "periodic",
        "fermion_modes": int(nsites),
        "effective_cells": int(nsites // 2),
        "spatial_sites": int(nsites),
        "dimension": int(model.dimension),
        "seconds": float(seconds),
        "vector_root": vector_root,
        "scalar_root": scalar_root,
        "vector_excitation_energy_over_g": float(vector_energy / coupling),
        "vector_momentum_over_g": float(model.vector_momentum / coupling),
        "M_V_over_g": float(vector_mass / coupling),
        "M_S_over_g": float(gaps[scalar_root] / coupling),
        "vector_strength": float(vector_strengths[vector_root]),
        "scalar_strength": float(scalar_strengths[scalar_root]),
        "vacuum_dimension": 1,
    }


def style(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot(points, output):
    continuum = {"M_V_over_g": 1.0 / np.sqrt(np.pi), "M_S_over_g": 2.0 / np.sqrt(np.pi)}
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    colors = {"Wilson-DVR": "C0", "Kogut-Susskind": "C1"}
    markers = {"Wilson-DVR": "o", "Kogut-Susskind": "s"}
    for regulator in colors:
        rows = [row for row in points if row["regulator"] == regulator]
        cells = np.asarray([row["effective_cells"] for row in rows])
        for axis, key, title in (
            (axes[0, 0], "M_V_over_g", r"vector mass $M_V/g$"),
            (axes[0, 1], "M_S_over_g", r"scalar mass $M_S/g$"),
        ):
            values = np.asarray([row[key] for row in rows])
            axis.plot(
                cells,
                values,
                marker=markers[regulator],
                color=colors[regulator],
                label=regulator,
            )
            axis.axhline(continuum[key], color="0.25", linestyle="--", alpha=0.65)
            axis.set(xlabel="matched Dirac cells", ylabel=title)
            style(axis)

        errors = np.asarray(
            [
                max(
                    abs(row["M_V_over_g"] - continuum["M_V_over_g"]),
                    abs(row["M_S_over_g"] - continuum["M_S_over_g"]),
                )
                for row in rows
            ]
        )
        axes[1, 0].semilogy(
            cells,
            errors,
            marker=markers[regulator],
            color=colors[regulator],
            label=regulator,
        )
        axes[1, 1].semilogy(
            [row["dimension"] for row in rows],
            [row["seconds"] for row in rows],
            marker=markers[regulator],
            color=colors[regulator],
            label=regulator,
        )

    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(
        xlabel="matched Dirac cells",
        ylabel=r"max$(|\Delta M_V|,|\Delta M_S|)/g$",
        title="largest continuum-mass error",
    )
    axes[1, 1].set(
        xlabel="physical ED dimension",
        ylabel="wall time (s)",
        title="physical-sector ED cost",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    style(axes[1, 0])
    style(axes[1, 1])
    figure.suptitle(
        "Matched fermion modes and periodic boundaries",
        fontsize=14,
    )
    figure.savefig(output, dpi=210)
    plt.close(figure)


def run(
    output_directory=DEFAULT_OUTPUT,
    *,
    dvr_grids=(3, 5, 7, 9),
    length=10.0,
    coupling=1.0,
    mass=0.0,
    flux_cutoff=3,
    nroots=24,
    reuse_saved=True,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    points = []
    for npts in map(int, dvr_grids):
        saved = SAVED_DVR.get(npts)
        if reuse_saved and saved is not None and saved.exists():
            dvr = saved_dvr_point(
                saved,
                npts,
                length=length,
                coupling=coupling,
                mass=mass,
                flux_cutoff=flux_cutoff,
                nroots=nroots,
            )
        else:
            dvr = dvr_point(
                npts,
                length=length,
                coupling=coupling,
                mass=mass,
                flux_cutoff=flux_cutoff,
                nroots=nroots,
            )
        staggered = staggered_point(
            2 * npts,
            length=length,
            coupling=coupling,
            mass=mass,
            flux_cutoff=flux_cutoff,
            nroots=nroots,
        )
        points.extend((dvr, staggered))
        print(
            f"cells={npts}: DVR Mv={dvr['M_V_over_g']:.6f} "
            f"Ms={dvr['M_S_over_g']:.6f}; "
            f"KS Mv={staggered['M_V_over_g']:.6f} "
            f"Ms={staggered['M_S_over_g']:.6f}",
            flush=True,
        )

    figure = output_directory / "30_dvr_vs_kogut_susskind.png"
    plot(points, figure)
    payload = {
        "description": (
            "Full-gauge matched-orbital periodic ED pilot comparing the "
            "Wilson-DVR and Kogut-Susskind spatial regulators."
        ),
        "matching": "N_KS = 2*N_DVR, giving the same number of fermion modes",
        "caveat": (
            "Both use periodic boundaries and the lowest nonzero density mode, "
            "but four coarse grids do not define a controlled continuum fit."
        ),
        "parameters": {
            "length_times_g": float(length * coupling),
            "mass_over_g": float(mass / coupling),
            "flux_cutoff": int(flux_cutoff),
            "nroots": int(nroots),
            "dvr_grids": list(map(int, dvr_grids)),
            "reuse_saved": bool(reuse_saved),
        },
        "continuum_massless": {
            "M_V_over_g": float(1.0 / np.sqrt(np.pi)),
            "M_S_over_g": float(2.0 / np.sqrt(np.pi)),
        },
        "points": points,
        "figure": str(figure),
    }
    data_path = output_directory / "dvr_vs_kogut_susskind.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dvr-grids", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--flux-cutoff", type=int, default=3)
    parser.add_argument("--nroots", type=int, default=24)
    parser.add_argument("--recompute-dvr", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    recompute_dvr = values.pop("recompute_dvr")
    payload = run(**values, reuse_saved=not recompute_dvr)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
