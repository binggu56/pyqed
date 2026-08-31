#!/usr/bin/env python3
"""Plot the 0.25 fs phenol GP TDVP2 state-rank convergence check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.namd.phenol_sa_casscf_5d_gp_control import _state_overlap
from pyqed.mps import MPS


ROOT = Path("dataset/phenol_5d_production/dynamics")
DEFAULT_CHECKPOINTS = {
    24: ROOT / "gp_ngp_full_overlap_r16_tdvp2_state24_1fs/gp_checkpoint.npz",
    32: ROOT / "gp_ngp_full_overlap_r16_tdvp2_state32_0p25fs/gp_checkpoint.npz",
    48: ROOT
    / "gp_ngp_full_overlap_r16_tdvp2_state48_0p25fs_normfixed/gp_checkpoint.npz",
}
DEFAULT_OUTPUT = ROOT / "gp_rank_convergence_0p25fs"


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        count = int(saved["factor_count"])
        state = MPS(
            [np.asarray(saved[f"factor_{site}"]) for site in range(count)]
        )
        closure = float(saved["closure"][-1])
        norm_defect = (
            float(saved["tdvp_norm_defect"][-1])
            if "tdvp_norm_defect" in saved.files
            else abs(closure)
        )
        return state, {
            "seconds": float(saved["seconds"]),
            "norm": float(saved["norms"][-1]),
            "cap_yield": float(saved["cap_yield"][-1]),
            "closure": closure,
            "tdvp_norm_defect": norm_defect,
            "bonds": [
                int(saved[f"factor_{site}"].shape[-1]) for site in range(count)
            ],
        }


def run(checkpoints, output):
    output.mkdir(parents=True, exist_ok=True)
    ranks = np.asarray(sorted(checkpoints), dtype=int)
    loaded = {rank: load_checkpoint(checkpoints[rank]) for rank in ranks}
    reference_rank = int(ranks[-1])
    reference = loaded[reference_rank][0]
    for rank in ranks:
        state, record = loaded[int(rank)]
        overlap = _state_overlap(state, reference)
        fidelity = abs(overlap) ** 2 / (
            state.norm_squared() * reference.norm_squared()
        )
        record["fidelity_to_rank_48"] = float(fidelity)
        record["infidelity_to_rank_48"] = float(max(0.0, 1.0 - fidelity))

    records = {int(rank): loaded[int(rank)][1] for rank in ranks}
    (output / "rank_convergence.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    figure, panels = plt.subplots(
        2, 2, figsize=(8.3, 6.2), constrained_layout=True
    )
    infidelity = np.asarray(
        [max(records[int(rank)]["infidelity_to_rank_48"], 1.0e-16) for rank in ranks]
    )
    panels[0, 0].semilogy(ranks, infidelity, "o-", color="#0072B2")
    panels[0, 0].set(
        xlabel="maximum state rank",
        ylabel="infidelity to rank 48",
        title="Wavefunction convergence",
        xticks=ranks,
    )
    panels[0, 1].plot(
        ranks,
        [records[int(rank)]["cap_yield"] for rank in ranks],
        "o-",
        color="#D55E00",
    )
    panels[0, 1].set(
        xlabel="maximum state rank",
        ylabel="CAP yield",
        title="Early absorbed probability",
        xticks=ranks,
    )
    panels[1, 0].semilogy(
        ranks,
        [records[int(rank)]["tdvp_norm_defect"] for rank in ranks],
        "o-",
        color="#009E73",
    )
    panels[1, 0].set(
        xlabel="maximum state rank",
        ylabel="cumulative removed norm",
        title="TDVP2 truncation diagnostic",
        xticks=ranks,
    )
    panels[1, 1].plot(
        ranks,
        [records[int(rank)]["seconds"] / 60.0 for rank in ranks],
        "o-",
        color="#CC79A7",
    )
    panels[1, 1].set(
        xlabel="maximum state rank",
        ylabel="wall time (min)",
        title="Five-step cost",
        xticks=ranks,
    )
    for label, panel in zip("abcd", panels.flat):
        panel.grid(alpha=0.2)
        panel.text(
            0.02, 0.96, label, transform=panel.transAxes,
            va="top", fontweight="bold",
        )
    figure.savefig(output / "phenol_gp_rank_convergence.png", dpi=350)
    figure.savefig(output / "phenol_gp_rank_convergence.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank24", type=Path, default=DEFAULT_CHECKPOINTS[24])
    parser.add_argument("--rank32", type=Path, default=DEFAULT_CHECKPOINTS[32])
    parser.add_argument("--rank48", type=Path, default=DEFAULT_CHECKPOINTS[48])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run({24: args.rank24, 32: args.rank32, 48: args.rank48}, args.output)


if __name__ == "__main__":
    main()
