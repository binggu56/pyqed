#!/usr/bin/env python3
"""Repair a metastable high-flux N=11 ground checkpoint."""

from __future__ import annotations

import argparse
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


HERE = Path(__file__).resolve().parent


def run(flux_cutoff):
    flux_cutoff = int(flux_cutoff)
    output = HERE / (
        f"results/channel_targeted_vector_excited_dmrg_n11_flux{flux_cutoff}_d64"
    )
    state_path = output / "n11_vector_excited_dmrg_states.pkl"
    with state_path.open("rb") as handle:
        previous = pickle.load(handle)

    builder = AlternatingWilsonDVRMPO(
        npts=11,
        length=10.0,
        coupling=1.0,
        mass=0.0,
        flux_cutoff=flux_cutoff,
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
    initial = MPS(
        previous["states"][0],
        labels=["lv", "rv", "p"],
        sites=hamiltonian.input_sites,
    )
    history = []

    def progress(**info):
        value = float(np.asarray(info.get("energy")).reshape(-1)[0])
        history.append(
            {
                "sweep": int(info.get("sweep", -1)),
                "direction": str(info.get("direction")),
                "energy": value,
            }
        )
        print(
            f"[ground repair flux={flux_cutoff}] half-sweep "
            f"{len(history)} ({info.get('direction')}): {value:.12f}",
            flush=True,
        )

    checkpoint = output / "ground_state_checkpoint_repaired.pkl"
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
        performance="symmetric",
        checkpoint_path=checkpoint,
        sweep_callback=progress,
    ).run()
    seconds = perf_counter() - started
    payload = {
        "description": "Single-state repair of metastable high-flux vacuum",
        "flux_cutoff": flux_cutoff,
        "bond_dim": 128,
        "half_sweeps": 20,
        "initial_energy": float(previous["energies"][0]),
        "final_energy": float(solver.e_tot),
        "wall_seconds": float(seconds),
        "converged": bool(solver.converged),
        "history": history,
    }
    data_path = output / "ground_state_repair.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(np.arange(1, len(history) + 1), [row["energy"] for row in history], "o-")
    ax.set(
        xlabel="half-sweep",
        ylabel=r"$E_0/g$",
        title=rf"ground repair, $\ell_{{\max}}={flux_cutoff}$",
    )
    style(ax)
    figure_path = output / f"23_flux{flux_cutoff}_ground_repair.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(checkpoint)
    print(data_path)
    print(figure_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-cutoff", type=int, required=True, choices=(3, 4))
    args = parser.parse_args()
    run(args.flux_cutoff)
