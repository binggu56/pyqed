#!/usr/bin/env python3
"""Benchmark the H2O Jacobi LDR/UED workflow.

The benchmark covers:

* coordinate/geometry consistency
* PES equivalence for valence and Jacobi coordinates
* Jacobi kinetic Hermiticity and spectrum
* grid, range, and timestep sensitivity
* near-collinear CI-access diagnostics
* UED scattering-grid sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatomic
from pyqed.ued.ued import UED
from pyqed.units import au2fs

from examples.ued.h2o_s2_vibronic_ued import (
    DEFAULTS,
    fc_packet_on_state,
    h2o_reference_geometry,
    make_driver,
    make_ldr,
    marginals,
    momentum_grid,
    populations,
)


def cfg(**updates):
    data = deepcopy(DEFAULTS)
    data.update(updates)
    return SimpleNamespace(**data)


def angle_at_center(coords, i, j, k):
    vi = coords[i] - coords[j]
    vk = coords[k] - coords[j]
    cosang = np.dot(vi, vk) / np.linalg.norm(vi) / np.linalg.norm(vk)
    return np.rad2deg(np.arccos(np.clip(cosang, -1.0, 1.0)))


def pair_distances(coords):
    return {
        "H1O": float(np.linalg.norm(coords[0] - coords[1])),
        "OH2": float(np.linalg.norm(coords[1] - coords[2])),
        "H1H2": float(np.linalg.norm(coords[0] - coords[2])),
        "HOH_deg": float(angle_at_center(coords, 0, 1, 2)),
    }


def geometry_benchmark():
    tri = Triatomic(h2o_reference_geometry(), unit="bohr", coordinates="jacobi")
    qeq = tri.valence_to_jacobi(1.81, 1.81, np.deg2rad(104.5))
    coords = tri.internal_to_xyz(*qeq)
    return {
        "qeq": {
            "r": float(qeq[0]),
            "R": float(qeq[1]),
            "gamma_deg": float(np.rad2deg(qeq[2])),
        },
        "pairs": pair_distances(coords),
    }


def energies_from_driver(driver, xyz):
    point = driver(np.asarray(xyz, dtype=float))
    energies = getattr(point, "e_tot", point)
    return np.asarray(energies, dtype=float)


def pes_equivalence_benchmark(driver):
    val = Triatomic(h2o_reference_geometry(), unit="bohr", coordinates="valence")
    jac = Triatomic(h2o_reference_geometry(), unit="bohr", coordinates="jacobi")
    points = [
        (1.81, 1.81, np.deg2rad(104.5)),
        (1.95, 1.81, np.deg2rad(104.5)),
        (1.81, 1.95, np.deg2rad(104.5)),
        (1.81, 1.81, np.deg2rad(95.0)),
        (1.81, 1.81, np.deg2rad(140.0)),
    ]
    rows = []
    for qv in points:
        qj = jac.valence_to_jacobi(*qv)
        ev = energies_from_driver(driver, val.internal_to_xyz(*qv))
        ej = energies_from_driver(driver, jac.internal_to_xyz(*qj))
        rows.append(
            {
                "r1": qv[0],
                "r2": qv[1],
                "theta_deg": np.rad2deg(qv[2]),
                "r": qj[0],
                "R": qj[1],
                "gamma_deg": np.rad2deg(qj[2]),
                "max_abs_dE": float(np.max(np.abs(ev - ej))),
                "E_valence": ev.tolist(),
                "E_jacobi": ej.tolist(),
            }
        )
    return rows


def kinetic_benchmark():
    c = cfg(n_r=5, n_R=5, n_theta=9, R_max=5.0)
    ldr = make_ldr(c, driver=None)
    T = ldr.buildK(sparse=False)
    eig = np.linalg.eigvalsh(0.5 * (T + T.conj().T))
    return {
        "shape": list(T.shape),
        "hermitian_max_abs": float(np.max(np.abs(T - T.conj().T))),
        "eig_min": float(eig[0]),
        "eig_max": float(eig[-1]),
    }


def edge_weight(result):
    psi = np.asarray(result["psilist"])
    mask = np.zeros(psi.shape[1:-1], dtype=bool)
    for axis, n in enumerate(mask.shape):
        sl = [slice(None)] * mask.ndim
        sl[axis] = 0
        mask[tuple(sl)] = True
        sl[axis] = n - 1
        mask[tuple(sl)] = True
    prob = np.sum(np.abs(psi) ** 2, axis=-1)
    return np.sum(prob[:, mask], axis=1).real


def run_ldr_case(driver, c, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    ldr = make_ldr(c, driver)
    cwd = Path.cwd()
    try:
        os.chdir(outdir)
        pes, _, _ = ldr.scan_pes(
            nstates=c.nstates,
            overlap_method=c.overlap_method,
            driver=driver,
        )
    finally:
        os.chdir(cwd)

    psi0, fc_energy, ref_idx = fc_packet_on_state(ldr, pes, c)
    result = ldr.run(
        psi0,
        dt=c.dt_fs / au2fs,
        nt=int(round(c.tmax_fs / c.dt_fs)),
        nout=max(1, int(round(c.ued_dt_fs / c.dt_fs))),
        kinetic_propagator=c.kinetic_propagator,
    )
    times_fs = np.asarray(result["times"]) * au2fs
    pops = populations(result)
    _, _, angle_density = marginals(ldr, result)
    angle_deg = np.rad2deg(np.asarray(ldr.x[2]))
    near = (angle_deg < 10.0) | (angle_deg > 170.0)
    near_ci = np.sum(angle_density[:, near], axis=1)
    edge = edge_weight(result)
    return {
        "ldr": ldr,
        "pes": pes,
        "result": result,
        "times_fs": times_fs,
        "pops": pops,
        "near_ci": near_ci,
        "edge": edge,
        "fc_energy": fc_energy,
        "ref_idx": ref_idx,
        "angle_deg": angle_deg,
    }


def summarize_case(name, c, data):
    pops = data["pops"]
    return {
        "case": name,
        "n_r": c.n_r,
        "n_R": c.n_R,
        "n_theta": c.n_theta,
        "r_min": c.r_min,
        "r_max": c.r_max,
        "R_min": c.R_min,
        "R_max": c.R_max,
        "gamma_min_deg": c.gamma_min_deg,
        "gamma_max_deg": c.gamma_max_deg,
        "dt_fs": c.dt_fs,
        "tmax_fs": c.tmax_fs,
        "final_S0": float(pops[-1, 0]),
        "final_S1": float(pops[-1, 1]),
        "final_S2": float(pops[-1, 2]),
        "max_S1": float(np.max(pops[:, 1])),
        "max_near_ci": float(np.max(data["near_ci"])),
        "final_near_ci": float(data["near_ci"][-1]),
        "max_edge": float(np.max(data["edge"])),
        "final_edge": float(data["edge"][-1]),
        "fc_energy": float(data["fc_energy"]),
    }


def ued_benchmark(ldr, result, outdir):
    ldr.ldr_result = result
    ldr.ued_result = result
    rows = []
    for n_s, s_max in [(11, 8.0), (21, 10.0), (31, 10.0)]:
        sx, sy, s = momentum_grid(s_max, n_s)
        signal = UED(ldr, aligned=True).run(s=s, verbose=False)
        I = np.asarray(signal["I_signal"]).reshape(len(signal["times"]), n_s, n_s)
        delta = I - I[0][None, :, :]
        rows.append(
            {
                "n_s": n_s,
                "s_max": s_max,
                "delta_rel_norm_final": float(
                    np.linalg.norm(delta[-1]) / (np.linalg.norm(I[0]) or 1.0)
                ),
                "delta_abs_max_final": float(np.max(np.abs(delta[-1]))),
            }
        )
        np.savez_compressed(
            outdir / f"ued_ns{n_s}_smax{s_max:g}.npz",
            sx=sx,
            sy=sy,
            s=s,
            I_signal=I,
            times=np.asarray(signal["times"]) * au2fs,
        )
    return rows


def plot_case_group(path, title, case_data):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), dpi=180)
    for name, data in case_data:
        t = data["times_fs"]
        axes[0].plot(t, data["pops"][:, 1], lw=1.8, label=name)
        axes[1].plot(t, data["near_ci"], lw=1.8, label=name)
        axes[2].plot(t, data["edge"], lw=1.8, label=name)
    axes[0].set_ylabel("S1 population")
    axes[1].set_ylabel("near-collinear weight")
    axes[2].set_ylabel("edge weight")
    for ax in axes:
        ax.set_xlabel("time (fs)")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ued_rows(path, rows):
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=180)
    labels = [f"{row['n_s']} / {row['s_max']:g}" for row in rows]
    vals = [row["delta_rel_norm_final"] for row in rows]
    ax.bar(labels, vals, color="#2f6f91")
    ax.set_xlabel("n_s / s_max")
    ax.set_ylabel(r"$||\Delta I(t_f)||/||I(0)||$")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_jacobi_benchmark"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    driver = make_driver(cfg())

    geometry = geometry_benchmark()
    pes_rows = pes_equivalence_benchmark(driver)
    kinetic = kinetic_benchmark()

    all_case_rows = []

    grid_specs = [
        ("grid_3x3x5", cfg(n_r=3, n_R=3, n_theta=5, tmax_fs=1.0, ued_dt_fs=0.5)),
        ("grid_5x5x9", cfg(n_r=5, n_R=5, n_theta=9, tmax_fs=3.0, ued_dt_fs=0.5)),
        ("grid_7x7x11", cfg(n_r=7, n_R=7, n_theta=11, tmax_fs=5.0, ued_dt_fs=0.5)),
    ]
    grid_data = []
    for name, c in grid_specs:
        data = run_ldr_case(driver, c, args.outdir / name)
        grid_data.append((name, data))
        all_case_rows.append(summarize_case(name, c, data))
    plot_case_group(args.outdir / "grid_convergence.png", "Grid convergence", grid_data)

    range_specs = [
        ("range_R4", cfg(n_r=5, n_R=5, n_theta=9, R_max=4.0, tmax_fs=3.0, ued_dt_fs=0.5)),
        ("range_R5", cfg(n_r=5, n_R=5, n_theta=9, R_max=5.0, tmax_fs=3.0, ued_dt_fs=0.5)),
        ("range_R6", cfg(n_r=5, n_R=5, n_theta=9, R_max=6.0, tmax_fs=3.0, ued_dt_fs=0.5)),
    ]
    range_data = []
    for name, c in range_specs:
        data = run_ldr_case(driver, c, args.outdir / name)
        range_data.append((name, data))
        all_case_rows.append(summarize_case(name, c, data))
    plot_case_group(args.outdir / "range_convergence.png", "Range convergence", range_data)

    dt_base = cfg(n_r=5, n_R=5, n_theta=9, R_max=5.0, tmax_fs=2.0, ued_dt_fs=0.5)
    dt_seed = run_ldr_case(driver, dt_base, args.outdir / "dt_scan_data")
    dt_data = []
    for dt in (0.2, 0.1, 0.05):
        c = cfg(n_r=5, n_R=5, n_theta=9, R_max=5.0, tmax_fs=2.0, dt_fs=dt, ued_dt_fs=0.5)
        psi0, fc_energy, ref_idx = fc_packet_on_state(dt_seed["ldr"], dt_seed["pes"], c)
        result = dt_seed["ldr"].run(
            psi0,
            dt=c.dt_fs / au2fs,
            nt=int(round(c.tmax_fs / c.dt_fs)),
            nout=max(1, int(round(c.ued_dt_fs / c.dt_fs))),
            kinetic_propagator=c.kinetic_propagator,
        )
        tmp = {
            "ldr": dt_seed["ldr"],
            "pes": dt_seed["pes"],
            "result": result,
            "times_fs": np.asarray(result["times"]) * au2fs,
            "pops": populations(result),
            "near_ci": marginals(dt_seed["ldr"], result)[2][:, (dt_seed["angle_deg"] < 10.0) | (dt_seed["angle_deg"] > 170.0)].sum(axis=1),
            "edge": edge_weight(result),
            "fc_energy": fc_energy,
            "ref_idx": ref_idx,
            "angle_deg": dt_seed["angle_deg"],
        }
        name = f"dt_{dt:g}fs"
        dt_data.append((name, tmp))
        all_case_rows.append(summarize_case(name, c, tmp))
    plot_case_group(args.outdir / "dt_convergence.png", "Timestep convergence", dt_data)

    ued_rows = ued_benchmark(grid_data[-1][1]["ldr"], grid_data[-1][1]["result"], args.outdir)
    plot_ued_rows(args.outdir / "ued_s_convergence.png", ued_rows)

    write_csv(args.outdir / "dynamics_summary.csv", all_case_rows)
    write_csv(args.outdir / "pes_equivalence.csv", pes_rows)
    write_csv(args.outdir / "ued_summary.csv", ued_rows)
    summary = {
        "geometry": geometry,
        "pes_max_abs_dE": max(row["max_abs_dE"] for row in pes_rows),
        "kinetic": kinetic,
        "dynamics_rows": all_case_rows,
        "ued_rows": ued_rows,
    }
    with (args.outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"outdir={args.outdir}")
    print(f"qeq={geometry['qeq']}")
    print(f"pairs={geometry['pairs']}")
    print(f"pes_max_abs_dE={summary['pes_max_abs_dE']:.3e}")
    print(f"kinetic={kinetic}")
    print(f"dynamics_csv={args.outdir / 'dynamics_summary.csv'}")
    print(f"ued_csv={args.outdir / 'ued_summary.csv'}")
    print(f"summary={args.outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
