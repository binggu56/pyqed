"""Compare fixed detached frames with ordinary conditional NARG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem import abelian
from pyqed.narg.qchem.su2_core import su2_irrep_tensor_roots


class _HalfFilledMol:
    nelec = (3, 3)
    spin = 0

    @staticmethod
    def energy_nuc():
        return 0.0


def hubbard_integrals(nsites=6, *, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def solve(h1e, eri, **options):
    solver = abelian.NARG(
        object(),
        mol=_HalfFilledMol(),
        n0=2,
        nstates=1,
        growth_sites=1,
        **options,
    )
    start = time.perf_counter()
    energy = float(solver.run(h1e=h1e, eri=eri)[0][0])
    return solver, energy, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/abelian_detached_hubbard"),
    )
    parser.add_argument("--frame-dims", type=int, nargs="+", default=(2, 4, 8, 16))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h1e, eri = hubbard_integrals()
    exact = float(
        su2_irrep_tensor_roots(
            h1e,
            eri,
            nelec=6,
            j2=0,
            nroots=1,
            m2=None,
        )[0][0]
    )
    records = []
    for frame_dim in args.frame_dims:
        chi = 8 * frame_dim
        plain_same, plain_same_energy, plain_same_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
        )
        plain_fair, plain_fair_energy, plain_fair_seconds = solve(
            h1e,
            eri,
            D=2 * frame_dim,
        )
        detached, detached_energy, detached_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
            chi=chi,
            dressing="detached_frames",
        )
        adaptive, adaptive_energy, adaptive_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
            chi=chi,
            dressing="detached_frames",
            frame_adapt_tol=0.1,
            frame_max_dim=chi,
            frame_expand_dim=max(1, frame_dim),
        )
        combined, combined_energy, combined_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
            chi=chi,
            dressing="detached+cc",
            frame_adapt_tol=0.1,
            frame_max_dim=chi,
            frame_expand_dim=max(1, frame_dim),
        )
        history = detached.detached_history
        adaptive_history = adaptive.detached_history
        combined_history = combined.detached_history
        cc_history = combined.dressing_history
        records.append(
            {
                "frame_D": frame_dim,
                "chi": chi,
                "plain_same_D_energy": plain_same_energy,
                "plain_same_D_error": plain_same_energy - exact,
                "plain_same_D_seconds": plain_same_seconds,
                "plain_comparable_rank_D": 2 * frame_dim,
                "plain_comparable_rank_energy": plain_fair_energy,
                "plain_comparable_rank_error": plain_fair_energy - exact,
                "plain_comparable_rank_seconds": plain_fair_seconds,
                "detached_energy": detached_energy,
                "detached_error": detached_energy - exact,
                "detached_seconds": detached_seconds,
                "maximum_frame_rank": max(item["frame_rank"] for item in history),
                "maximum_retained_rank": max(item["retained_dim"] for item in history),
                "maximum_frame_residual": max(
                    item["frame_residual_norm"] for item in history
                ),
                "adaptive_detached_energy": adaptive_energy,
                "adaptive_detached_error": adaptive_energy - exact,
                "adaptive_detached_seconds": adaptive_seconds,
                "adaptive_maximum_frame_rank": max(
                    item["frame_rank"] for item in adaptive_history
                ),
                "adaptive_maximum_retained_rank": max(
                    item["retained_dim"] for item in adaptive_history
                ),
                "adaptive_maximum_frame_residual": max(
                    item["frame_residual_norm"] for item in adaptive_history
                ),
                "detached_cc_energy": combined_energy,
                "detached_cc_error": combined_energy - exact,
                "detached_cc_seconds": combined_seconds,
                "detached_cc_maximum_frame_rank": max(
                    item["frame_rank"] for item in combined_history
                ),
                "detached_cc_maximum_response_rank": max(
                    item["response_rank"] for item in cc_history
                ),
                "detached_cc_maximum_response_residual": max(
                    item["maximum_response_residual"] for item in cc_history
                ),
                "detached_cc_iterative_fallbacks": sum(
                    item["iterative_fallbacks"] for item in cc_history
                ),
            }
        )

    payload = {
        "model": "open six-site half-filled Hubbard chain, t=0.7, U=2.0",
        "exact_energy": exact,
        "records": records,
    }
    (args.output_dir / "abelian_detached_hubbard.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    frame_dims = [item["frame_D"] for item in records]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].plot(
        frame_dims,
        [item["plain_same_D_error"] for item in records],
        marker="o",
        label="ordinary, same D",
    )
    axes[0].plot(
        frame_dims,
        [item["plain_comparable_rank_error"] for item in records],
        marker="o",
        label="ordinary, comparable rank",
    )
    axes[0].plot(
        frame_dims,
        [item["detached_error"] for item in records],
        marker="o",
        label="detached frame",
    )
    axes[0].plot(
        frame_dims,
        [item["adaptive_detached_error"] for item in records],
        marker="o",
        label="adaptive detached",
    )
    axes[0].plot(
        frame_dims,
        [item["detached_cc_error"] for item in records],
        marker="o",
        label="adaptive detached + CC",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="frame parameter D", ylabel="energy error (Hartree)")
    axes[0].legend(frameon=False)

    axes[1].plot(
        frame_dims,
        [item["plain_same_D_seconds"] for item in records],
        marker="o",
        label="ordinary, same D",
    )
    axes[1].plot(
        frame_dims,
        [item["plain_comparable_rank_seconds"] for item in records],
        marker="o",
        label="ordinary, comparable rank",
    )
    axes[1].plot(
        frame_dims,
        [item["detached_seconds"] for item in records],
        marker="o",
        label="detached frame",
    )
    axes[1].plot(
        frame_dims,
        [item["adaptive_detached_seconds"] for item in records],
        marker="o",
        label="adaptive detached",
    )
    axes[1].plot(
        frame_dims,
        [item["detached_cc_seconds"] for item in records],
        marker="o",
        label="adaptive detached + CC",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="frame parameter D", ylabel="wall time (s)")
    axes[1].legend(frameon=False)
    figure.savefig(args.output_dir / "abelian_detached_hubbard.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
