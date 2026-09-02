#!/usr/bin/env python3
"""Validate and profile link elimination for the open sine--cosine DVR.

The interval Gauss law is solved exactly before the MPS calculation.  This is
the standard open-boundary Schwinger-model reduction of Bañuls et al., JHEP
11, 158 (2013), DOI: 10.1007/JHEP11(2013)158, applied here to the paired
DCT-IV/DST-IV regulator.  The DVR application is an adaptation, not a
reproduction of that staggered-fermion calculation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kogut_susskind_n80_mv_ms import _channel_source, _symmetric_mpo
from pyqed.lgt import OpenSineMatterDVRMPO, OpenSineWilsonDVRMPO
from pyqed.mps import DMRG, TDMPS


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "open_sine_matter_dvr_benchmark"
EXPLICIT_N12 = (
    HERE
    / "results"
    / "open_sine_dvr_n12_d32_mv_ms_pilot"
    / "open_sine_dvr_n12_d32_mv_ms.json"
)
EXPLICIT_N40 = (
    HERE
    / "results"
    / "open_sine_dvr_n40_d128_readiness"
    / "open_sine_dvr_n40_readiness.json"
)
EXPLICIT_STEP = (
    HERE
    / "results"
    / "open_sine_dvr_factorized_tdvp_benchmark"
    / "open_sine_dvr_factorized_tdvp_benchmark.json"
)


def _physical_projection(npts=2, length=5.0, mass=0.2):
    explicit = OpenSineWilsonDVRMPO(
        npts, length, mass=mass, flux_cutoff=2
    )
    reduced = OpenSineMatterDVRMPO(npts, length, mass=mass)
    charges = np.real(np.diag(explicit.matter["q"])).astype(int)
    matter_indices = []
    explicit_indices = []
    for states in itertools.product(range(4), repeat=npts):
        flux = explicit.left_flux
        internal = []
        for state in states[:-1]:
            flux -= int(charges[state])
            internal.append(flux)
        flux -= int(charges[states[-1]])
        if flux != explicit.right_flux:
            continue
        full = []
        for site, state in enumerate(states):
            full.append(state)
            if site < explicit.nlinks:
                full.append(internal[site] + explicit.flux_cutoff)
        matter_indices.append(np.ravel_multi_index(states, reduced.dims))
        explicit_indices.append(np.ravel_multi_index(full, explicit.dims))

    errors = {}
    pairs = {
        "H": (explicit.build_mpo(), reduced.build_mpo()),
        "M_V": (explicit.build_vector_mpo(), reduced.build_vector_mpo()),
        "M_S": (explicit.build_scalar_mpo(), reduced.build_scalar_mpo()),
    }
    for label, (full, matter) in pairs.items():
        lhs = full.to_dense()[np.ix_(explicit_indices, explicit_indices)]
        rhs = matter.to_dense()[np.ix_(matter_indices, matter_indices)]
        errors[label] = float(np.max(np.abs(lhs - rhs)))
    return errors, len(matter_indices)


def _setup(npts, length=20.0):
    builder = OpenSineMatterDVRMPO(npts, length)
    maps, target, manager = builder.gauss_symmetry()
    started = perf_counter()
    raw = builder.build_mpo()
    hamiltonian = _symmetric_mpo(raw, maps, compress=True)
    elapsed = perf_counter() - started
    return builder, maps, target, manager, hamiltonian, {
        "seconds": elapsed,
        "chain_sites": builder.nsites,
        "raw_mpo_bond": max(raw.bond_orders()),
        "compressed_mpo_bond": max(hamiltonian.bond_orders()),
    }


def _n12_calculation(*, bond_dim=32, half_sweeps=2):
    builder, maps, target, manager, hamiltonian, setup = _setup(12)
    sectors = [[site_map[state] for state in sorted(site_map)] for site_map in maps]
    initial = builder.gauss_seed_mps(
        bond_dim=bond_dim, seed=7, native_site_storage=False
    )
    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=bond_dim,
        init_guess=initial,
        nsweeps=half_sweeps,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-9,
        davidson_tol=1.0e-10,
        davidson_max_iter=200,
        noise=1.0e-6,
        performance="symmetric",
    ).run()
    ground_seconds = perf_counter() - started
    vacuum = solver.ground_state

    scalar = builder.build_scalar_mpo()
    scalar_mean = float(np.real(vacuum.expectation(_symmetric_mpo(scalar, maps))))
    sources = {
        "vector": _channel_source(
            builder.build_vector_mpo(), vacuum, maps, bond_dim=bond_dim
        ),
        "scalar": _channel_source(
            scalar, vacuum, maps, mean=scalar_mean, bond_dim=bond_dim
        ),
    }
    step_seconds = {}
    for label, source in sources.items():
        propagator = TDMPS(
            hamiltonian,
            D=bond_dim,
            local_sectors=sectors,
            target_sector=target,
            projection="block-sparse",
        )
        started = perf_counter()
        propagated = propagator.step(
            source,
            dt=0.1,
            integrator="tdvp",
            krylov_dim=8,
            krylov_tol=1.0e-10,
        )
        step_seconds[label] = perf_counter() - started
        np.testing.assert_allclose(propagated.norm_squared(), 1.0, atol=2.0e-12)
    return {
        "bond_dim": int(bond_dim),
        "half_sweeps": int(half_sweeps),
        "ground_energy": float(solver.e_tot),
        "ground_seconds": ground_seconds,
        "scalar_mean": scalar_mean,
        "tdvp_step_seconds": step_seconds,
        "setup": setup,
    }


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def _plot(data, output):
    explicit12 = data["explicit_n12"]
    explicit40 = data["explicit_n40"]
    explicit_step = data["explicit_step"]
    matter12 = data["matter_n12"]
    matter40 = data["matter_n40_setup"]
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.0), constrained_layout=True)

    labels = list(data["physical_sector_max_error"])
    values = [max(data["physical_sector_max_error"][key], 1.0e-16) for key in labels]
    axes[0, 0].bar(labels, values, color=["C0", "C1", "C2"])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(
        ylabel="maximum matrix error",
        title="Exact physical-sector reduction",
    )
    if not any(data["physical_sector_max_error"].values()):
        axes[0, 0].text(
            0.5,
            0.08,
            "all measured errors = 0; bars show log floor",
            transform=axes[0, 0].transAxes,
            ha="center",
            fontsize=8,
        )

    x = np.arange(2)
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [2 * 12 - 1, 2 * 40 - 1],
        width,
        label="explicit links",
        color="0.55",
    )
    axes[0, 1].bar(
        x + width / 2,
        [matter12["setup"]["chain_sites"], matter40["chain_sites"]],
        width,
        label="links eliminated",
        color="C2",
    )
    axes[0, 1].set(
        xticks=x,
        xticklabels=["N=12", "N=40"],
        ylabel="MPS chain sites",
        title="Representation size",
    )
    axes[0, 1].legend(frameon=False)

    explicit_vector = float(explicit_step.get("symmetric_single_seconds", np.nan))
    axes[1, 0].bar(
        ["explicit\nvector", "reduced\nvector", "reduced\nscalar"],
        [
            explicit_vector,
            matter12["tdvp_step_seconds"]["vector"],
            matter12["tdvp_step_seconds"]["scalar"],
        ],
        color=["0.55", "C0", "C1"],
    )
    axes[1, 0].set(
        ylabel="one TDVP step (s)",
        title=r"$N=12,D=32$ channel propagation",
    )

    explicit_half = float(explicit40.get("ground_seconds", np.nan)) / max(
        int(explicit40.get("ground_half_sweeps", 1)), 1
    )
    matter_bound = float(data["n40_half_sweep_lower_bound_seconds"])
    axes[1, 1].bar(
        ["explicit\ncompleted", "reduced\nlower bound"],
        [explicit_half, matter_bound],
        color=["0.55", "C3"],
    )
    axes[1, 1].set(
        ylabel="seconds per half-sweep",
        title=r"$N=40,D=128$ current DMRG path",
    )
    axes[1, 1].text(
        1,
        matter_bound * 1.01,
        ">",
        ha="center",
        va="bottom",
        color="C3",
        fontsize=13,
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.tick_params(direction="in")
    path = output / "36_open_sine_matter_dvr_link_elimination.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def run(
    *,
    output=DEFAULT_OUTPUT,
    n40_half_sweep_lower_bound_seconds=765.0,
    n40_peak_memory_gb=6.56,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors, physical_dimension = _physical_projection()
    matter12 = _n12_calculation()
    _builder, _maps, _target, _manager, _hamiltonian, matter40 = _setup(40)
    explicit12 = _load(EXPLICIT_N12)
    explicit40 = _load(EXPLICIT_N40)
    explicit_step = _load(EXPLICIT_STEP)
    data = {
        "method": "open paired DCT-IV/DST-IV DVR with interval links eliminated",
        "fidelity": (
            "exact open-boundary Gauss-law reduction; paired spectral regulator "
            "is an adaptation; N=40 DMRG timing is a bounded failed attempt"
        ),
        "physical_sector_dimension_n2": physical_dimension,
        "physical_sector_max_error": errors,
        "matter_n12": matter12,
        "matter_n40_setup": matter40,
        "n40_half_sweep_lower_bound_seconds": float(
            n40_half_sweep_lower_bound_seconds
        ),
        "n40_peak_memory_gb_before_interrupt": float(n40_peak_memory_gb),
        "explicit_n12": explicit12,
        "explicit_n40": explicit40,
        "explicit_step": explicit_step,
        "references": [
            "https://doi.org/10.1007/JHEP11(2013)158",
            "https://doi.org/10.1103/PhysRevD.107.054506",
        ],
    }
    data_path = output / "open_sine_matter_dvr_benchmark.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n")
    figure_path = _plot(data, output)
    print(f"[exact] max errors: {errors}", flush=True)
    print(
        f"[N=12] E0={matter12['ground_energy']:.12f}; "
        f"DMRG={matter12['ground_seconds']:.3f} s; "
        f"TDVP={matter12['tdvp_step_seconds']}",
        flush=True,
    )
    print(
        f"[N=40] MPO {matter40['raw_mpo_bond']}->"
        f"{matter40['compressed_mpo_bond']} in {matter40['seconds']:.3f} s; "
        f"half-sweep >{n40_half_sweep_lower_bound_seconds:.0f} s",
        flush=True,
    )
    print(f"[result] JSON: {data_path}", flush=True)
    print(f"[result] figure: {figure_path}", flush=True)
    return data, figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--n40-half-sweep-lower-bound-seconds", type=float, default=765.0
    )
    parser.add_argument("--n40-peak-memory-gb", type=float, default=6.56)
    args = parser.parse_args()
    run(
        output=args.output,
        n40_half_sweep_lower_bound_seconds=args.n40_half_sweep_lower_bound_seconds,
        n40_peak_memory_gb=args.n40_peak_memory_gb,
    )


if __name__ == "__main__":
    main()
