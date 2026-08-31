#!/usr/bin/env python3
"""Lift converged N=11 flux-4 roots into flux-5 and refine the vacuum."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from channel_targeted_mv_ms_mps import style
from pyqed.lgt import AlternatingWilsonDVRMPO
from pyqed.mps import DMRG, MPO, MPS, compress_symmetric_mpo, dense_to_symmetric_mpo
from pyqed.mps.abelian_direct import AbelianSiteTensorData


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "results/channel_targeted_vector_excited_dmrg_n11_flux4_d128"
    / "n11_vector_excited_dmrg_states.pkl"
)
OUTPUT = HERE / "results/channel_targeted_vector_excited_dmrg_n11_flux5_d128"


def lift_factors(factors, maps):
    lifted = []
    for site, (factor, site_map) in enumerate(zip(factors, maps)):
        if not isinstance(factor, AbelianSiteTensorData):
            raise TypeError(f"site {site} is not a native Abelian tensor")
        physical_qns = tuple(site_map[state] for state in sorted(site_map))
        qns = (factor.qns[0], factor.qns[1], physical_qns)
        lifted.append(AbelianSiteTensorData(factor.data, qns, factor.dirs, copy=True))
    return lifted


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("rb") as handle:
        previous = pickle.load(handle)
    builder = AlternatingWilsonDVRMPO(
        npts=11,
        length=10.0,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=5,
    )
    maps, target, manager = builder.gauss_symmetry()
    raw = builder.build_mpo()
    hamiltonian = compress_symmetric_mpo(
        MPO(
            dense_to_symmetric_mpo(
                raw.factors,
                maps,
                native_site_storage=True,
            )
        )
    )
    lifted_states = [lift_factors(factors, maps) for factors in previous["states"]]
    warm_path = OUTPUT / "warm_start_flux4_states.pkl"
    with warm_path.open("wb") as handle:
        pickle.dump(
            {
                "energies": previous["energies"],
                "states": lifted_states,
                "strengths": previous["strengths"],
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    initial = MPS(
        lifted_states[0],
        labels=["lv", "rv", "p"],
        sites=hamiltonian.input_sites,
    )
    history = []

    def progress(**info):
        value = float(np.real(np.asarray(info.get("energy")).reshape(-1)[0]))
        history.append(
            {
                "sweep": int(info.get("sweep", -1)),
                "direction": str(info.get("direction")),
                "energy": value,
            }
        )
        print(
            f"[flux-5 lifted ground] half-sweep {len(history)} "
            f"({info.get('direction')}): {value:.12f}",
            flush=True,
        )

    checkpoint = OUTPUT / "ground_state_checkpoint_lifted.pkl"
    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=128,
        init_guess=initial,
        nsweeps=20,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        davidson_max_iter=300,
        noise=1.0e-6,
        performance="packed-compiled-fast",
        checkpoint_path=checkpoint,
        sweep_callback=progress,
    ).run()
    payload = {
        "description": "Exact zero-amplitude flux-4 to flux-5 MPS lift followed by DMRG",
        "source_flux_cutoff": 4,
        "flux_cutoff": 5,
        "bond_dim": 128,
        "initial_energy": float(previous["energies"][0]),
        "final_energy": float(solver.e_tot),
        "wall_seconds": float(perf_counter() - started),
        "converged": bool(solver.converged),
        "history": history,
    }
    data_path = OUTPUT / "lifted_ground_refinement.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(np.arange(1, len(history) + 1), [row["energy"] for row in history], "o-")
    ax.set(
        xlabel="half-sweep",
        ylabel=r"$E_0/g$",
        title=r"lifted flux-5 vacuum, $D=128$",
    )
    style(ax)
    figure_path = OUTPUT / "25_flux5_lifted_ground_refinement.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(checkpoint)
    print(warm_path)
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    main()
