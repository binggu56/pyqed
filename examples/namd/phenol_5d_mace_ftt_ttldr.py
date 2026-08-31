#!/usr/bin/env python3
"""Five-dimensional phenol MACE -> oracle TT-cross -> TTLDR demonstration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from examples.namd.phenol_abinitio_active import (
    symmetric_basis,
    symmetric_coefficients,
)
from pyqed.dvr import ExponentialDVR, SineDVR
from pyqed.ml import MACE
from pyqed.models.phenol_coordinates import PHENOL_SPECIES, PhenolReactiveChart
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.phenol import build_phenol_5d_keo_mpo
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2ev, au2fs


def sobol_coordinates(count, bounds, seed):
    power = int(np.ceil(np.log2(max(int(count) - 1, 1))))
    unit = qmc.Sobol(len(bounds), scramble=True, seed=int(seed)).random_base2(power)
    values = bounds[:, 0] + unit[: max(int(count) - 1, 0)] * np.ptp(bounds, axis=1)
    anchor = np.clip(
        PhenolReactiveChart().equilibrium, bounds[:, 0], bounds[:, 1]
    )
    return np.vstack((anchor, values))[: int(count)]


def build_dvrs(points, chart):
    bounds = np.asarray(chart.default_bounds, dtype=float).copy()
    bounds[0, 0] = 0.72
    atomic = np.asarray(
        [chart.coordinate_to_atomic(bound) for bound in bounds.T]
    ).T
    # A slightly wider box places a useful DVR node near the 0.970 A
    # equilibrium while retaining the 3.5 A dissociation region.
    radial = SineDVR(*atomic[0], points, mass=1.0)
    torsion = ExponentialDVR(
        npts=points,
        L=2.0 * np.pi,
        x0=np.pi / points,
        mass=1.0,
    )
    bend = SineDVR(*atomic[2], points, mass=1.0)
    co = SineDVR(*atomic[3], points, mass=1.0)
    ring = SineDVR(*atomic[4], points, mass=1.0)
    dvrs = (radial, torsion, bend, co, ring)
    public_scale = chart.coordinate_from_atomic(np.ones(5))
    axes = tuple(dvr.x * public_scale[axis] for axis, dvr in enumerate(dvrs))
    return axes, dvrs


def identity_links(axes, nstates, rank, degree):
    links = []
    for active in range(len(axes)):
        edge_axes = list(axes)
        edge_axes[active] = 0.5 * (edge_axes[active][:-1] + edge_axes[active][1:])
        shape = tuple(len(axis) for axis in edge_axes)
        values = np.broadcast_to(np.eye(nstates), (*shape, nstates, nstates)).copy()
        links.append(
            FunctionalTT(
                degrees=tuple(min(int(degree), len(axis) - 1) for axis in edge_axes),
                rank=int(rank),
                bounds=tuple((float(axis[0]), float(axis[-1])) for axis in edge_axes),
                normalization="frobenius", hermitian=False,
            ).fit_grid(tuple(edge_axes), values)
        )
    return tuple(links)


def initial_packet(axes, chart, state=2):
    widths = np.asarray((0.085, 0.32, np.deg2rad(4.5), 0.055, 0.045))
    factors = [
        np.exp(-0.25 * ((axis - center) / width) ** 2)
        for axis, center, width in zip(axes, chart.equilibrium, widths)
    ]
    nuclear = factors[0]
    for factor in factors[1:]:
        nuclear = np.multiply.outer(nuclear, factor)
    values = np.zeros((*nuclear.shape, 3), dtype=complex)
    values[..., int(state)] = nuclear
    return values / np.linalg.norm(values)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    chart = PhenolReactiveChart()
    axes, dvrs = build_dvrs(args.grid_points, chart)
    bounds = np.asarray([(axis[0], axis[-1]) for axis in axes])
    training = sobol_coordinates(args.samples, bounds, args.seed)
    matrices = np.asarray([chart.model_dpem(point) for point in training])
    fit = MACE(
        axes, PHENOL_SPECIES, chart.geometry, 3,
        chart_features=True, geometry_units="angstrom",
        channels=args.channels, max_ell=2, interactions=2, correlation=2,
        radial_basis=args.radial_basis, radial_mlp=(args.width, args.width),
        cutoff=args.cutoff,
    )
    fit.fit_basis_h(
        training, symmetric_coefficients(matrices), symmetric_basis(3),
        hidden=(args.width, args.width), epochs=args.epochs,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    fit.distill_energy(
        rank=args.tt_rank, degree=args.tt_degree, method="cross",
        points=args.cross_points, sweeps=args.cross_sweeps,
        rtol=args.cross_rtol, cross_validation=args.cross_validation,
        validation_points=args.validation_samples, seed=args.seed,
    )
    fit.links = identity_links(axes, 3, args.link_rank, args.tt_degree)
    validation = sobol_coordinates(args.validation_samples, bounds, args.seed + 1009)
    reference = np.asarray([chart.model_dpem(point) for point in validation])
    mace_values = fit.neural_energy.predict(validation)
    ftt_values = fit.energy.predict(validation)
    mace_rmse = float(np.sqrt(np.mean(np.abs(mace_values - reference) ** 2)) * au2ev * 1000.0)
    ftt_rmse = float(np.sqrt(np.mean(np.abs(ftt_values - reference) ** 2)) * au2ev * 1000.0)
    keo_started = time.perf_counter()
    keo, keo_info = build_phenol_5d_keo_mpo(
        dvrs,
        chart,
        cross_max_rank=args.keo_cross_rank,
        cross_sweeps=args.keo_cross_sweeps,
        cross_rtol=args.keo_cross_rtol,
        cross_validation=args.keo_cross_validation,
        mpo_max_rank=args.keo_mpo_rank,
        seed=args.seed,
        split=True,
        return_info=True,
    )
    keo_seconds = time.perf_counter() - keo_started
    driver = TTLDR.from_fit(
        fit, keo=keo, overlap_rank=args.link_rank,
        potential_rank=args.potential_rank, operator_rank=args.operator_rank,
        fitted_kinetic_backend="link-mpo",
    )
    initial = initial_packet(axes, chart, args.bright_state)
    state = driver.state(initial, max_rank=args.state_rank)
    times_fs = np.linspace(0.0, args.tmax_fs, args.steps + 1)
    driver.run(
        state, dt=float((times_fs[1] - times_fs[0]) / au2fs),
        steps=args.steps, interval=1, max_bond=args.state_rank,
        integrator="tdvp2", cutoff=1.0e-10, progress=False,
        e_ops=driver.projectors(),
    )
    final = driver.dense(driver.final_state)
    probability = np.abs(final) ** 2
    radial = probability.sum(axis=(1, 2, 3, 4, 5))
    radial /= radial.sum()
    r_mean_final = float(np.dot(axes[0], radial))
    distillation = fit.info["distillation"]
    cross = distillation["cross"]
    summary = {
        "dimensions": 5, "training_geometries": len(training),
        "cross_geometry_queries": cross["geometry_queries"],
        "cross_full_grid_geometries": cross["full_grid_geometries"],
        "cross_query_fraction": cross["geometry_query_fraction"],
        "mace_validation_rmse_mev": mace_rmse,
        "mace_ftt_validation_rmse_mev": ftt_rmse,
        "ftt_to_mace_relative_error": distillation["energy_relative_error"],
        "kinetic_model": "AD G-matrix J=0 Podolsky KEO",
        "keo_seconds": keo_seconds,
        "keo_cross": keo_info,
        "keo_component_active_axes": [list(active) for active, _operator in keo],
        "keo_component_ranks": [
            list(map(int, operator.bond_orders())) for _active, operator in keo
        ],
        "dressed_keo_backend": driver.overlap_info["backend"],
        "maximum_norm_drift": float(np.max(np.abs(driver.norms - 1.0))),
        "final_r_oh_mean_angstrom": r_mean_final,
        "operator_ranks": driver.operator_ranks,
        "final_state_ranks": list(map(int, driver.final_state.bond_orders())),
    }
    figure, panels = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    panels[0].semilogy(fit.history, color="#0072b2")
    panels[0].set(xlabel="epoch", ylabel="normalized MACE loss")
    populations = np.asarray(driver.populations)
    for state_id, color in enumerate(("#444444", "#d55e00", "#0072b2")):
        panels[1].plot(times_fs, populations[:, state_id], color=color, label=f"state {state_id}")
    panels[1].set(xlabel="time (fs)", ylabel="diabatic population", ylim=(-0.02, 1.02))
    panels[1].legend(frameon=False, fontsize=8)
    panels[2].plot(axes[0], radial, "o-", color="#0072b2")
    panels[2].axvline(r_mean_final, color="#d55e00", linestyle="--", label=fr"$\langle R\rangle={r_mean_final:.2f}$ A")
    panels[2].set(xlabel=r"$R_{OH}$ (angstrom)", ylabel="final radial probability")
    panels[2].legend(frameon=False, fontsize=8)
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output / "phenol_5d_mace_ftt_ttldr.png", dpi=260)
    figure.savefig(args.output / "phenol_5d_mace_ftt_ttldr.pdf")
    plt.close(figure)
    np.savez(
        args.output / "phenol_5d_mace_ftt_ttldr.npz",
        times_fs=times_fs, populations=populations, final_probability=probability,
        r_oh_angstrom=axes[0], phi_radian=axes[1], theta_radian=axes[2],
        q_co=axes[3], q_ring=axes[4], training_coordinates=training,
    )
    fit.save(args.output / "phenol_5d_mace.pt")
    (args.output / "phenol_5d_mace_ftt_ttldr.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"figure: {args.output / 'phenol_5d_mace_ftt_ttldr.png'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--grid-points", type=int, default=7)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--radial-basis", type=int, default=12)
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--tt-rank", type=int, default=20)
    parser.add_argument("--tt-degree", type=int, default=6)
    parser.add_argument("--cross-points", type=int, default=9)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--cross-validation", type=int, default=128)
    parser.add_argument("--keo-cross-rank", type=int, default=10)
    parser.add_argument("--keo-cross-sweeps", type=int, default=6)
    parser.add_argument("--keo-cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--keo-cross-validation", type=int, default=128)
    parser.add_argument("--keo-mpo-rank", type=int, default=96)
    parser.add_argument("--link-rank", type=int, default=4)
    parser.add_argument("--potential-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=48)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--bright-state", type=int, default=2)
    parser.add_argument("--tmax-fs", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_5d_mace_ftt_ttldr"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
