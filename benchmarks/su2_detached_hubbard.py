"""Benchmark reduced SU(2) detached frames on a Hubbard chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem.su2_chain import diagonalize_block, run_su2_narg_chain
from pyqed.narg.qchem.su2_core import su2_irrep_tensor_roots


def hubbard_integrals(nsites=6, *, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def solve(h1e, eri, *, D, **options):
    nsites = h1e.shape[0]
    start = time.perf_counter()
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {size: int(D) for size in range(2, nsites)},
        final_size=nsites,
        target_nelec=nsites,
        target_j2=0,
        backend="python",
        **options,
    )
    energy = float(
        diagonalize_block(
            chain.final,
            nelec=nsites,
            j2=0,
            nroots=1,
            backend="python",
        )[0][0]
    )
    return chain, energy, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/su2_detached_hubbard"),
    )
    parser.add_argument("--frame-dims", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument("--chi-factor", type=int, default=12)
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
        chi = int(args.chi_factor) * frame_dim
        plain, plain_energy, plain_seconds = solve(h1e, eri, D=frame_dim)
        plain_chi, plain_chi_energy, plain_chi_seconds = solve(h1e, eri, D=chi)
        fixed, fixed_energy, fixed_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
            dressing="detached_frames",
            chi=chi,
            frame_protect_dim=0,
        )
        protection_scan = []
        for protected in sorted({0, frame_dim // 2, frame_dim}):
            candidate = solve(
                h1e,
                eri,
                D=frame_dim,
                dressing="detached_frames",
                chi=chi,
                frame_adapt_tol=0.1,
                frame_max_dim=chi,
                frame_expand_dim=frame_dim,
                frame_protect_dim=protected,
            )
            protection_scan.append((*candidate, protected))
        adaptive, adaptive_energy, adaptive_selected_seconds, selected_protection = min(
            protection_scan,
            key=lambda item: item[1],
        )
        adaptive_seconds = sum(item[2] for item in protection_scan)
        combined, combined_energy, combined_seconds = solve(
            h1e,
            eri,
            D=frame_dim,
            dressing="detached+cc",
            chi=chi,
            frame_adapt_tol=0.1,
            frame_max_dim=chi,
            frame_expand_dim=frame_dim,
            frame_protect_dim=selected_protection,
            cc_level_shift=0.5,
        )
        fixed_history = fixed.timings["detached_by_size"].values()
        adaptive_history = adaptive.timings["detached_by_size"].values()
        cc_history = combined.timings["cc_by_size"].values()
        records.append(
            {
                "frame_D": frame_dim,
                "chi": chi,
                "ordinary_energy": plain_energy,
                "ordinary_error": plain_energy - exact,
                "ordinary_seconds": plain_seconds,
                "ordinary_same_chi_energy": plain_chi_energy,
                "ordinary_same_chi_error": plain_chi_energy - exact,
                "ordinary_same_chi_seconds": plain_chi_seconds,
                "fixed_detached_energy": fixed_energy,
                "fixed_detached_error": fixed_energy - exact,
                "fixed_detached_seconds": fixed_seconds,
                "fixed_maximum_frame_rank": max(
                    item["frame_rank"] for item in fixed_history
                ),
                "fixed_maximum_residual": max(
                    item["frame_residual_norm"]
                    for item in fixed.timings["detached_by_size"].values()
                ),
                "adaptive_detached_energy": adaptive_energy,
                "adaptive_detached_error": adaptive_energy - exact,
                "adaptive_detached_seconds": adaptive_seconds,
                "adaptive_selected_seconds": adaptive_selected_seconds,
                "adaptive_selected_protection": selected_protection,
                "adaptive_protection_scan": {
                    str(item[3]): item[1] for item in protection_scan
                },
                "adaptive_maximum_frame_rank": max(
                    item["frame_rank"] for item in adaptive_history
                ),
                "adaptive_maximum_residual": max(
                    item["frame_residual_norm"]
                    for item in adaptive.timings["detached_by_size"].values()
                ),
                "detached_cc_energy": combined_energy,
                "detached_cc_error": combined_energy - exact,
                "detached_cc_seconds": combined_seconds,
                "detached_cc_maximum_response_rank": max(
                    item["response_rank"] for item in cc_history
                ),
                "detached_cc_maximum_response_residual": max(
                    item["maximum_response_residual"]
                    for item in combined.timings["cc_by_size"].values()
                ),
                "detached_cc_iterative_fallbacks": sum(
                    item["iterative_fallbacks"]
                    for item in combined.timings["cc_by_size"].values()
                ),
            }
        )

    payload = {
        "model": "open six-site half-filled Hubbard chain, t=0.7, U=2.0",
        "exact_energy": exact,
        "detached_cc_level_shift": 0.5,
        "detached_chi_factor": int(args.chi_factor),
        "records": records,
    }
    (args.output_dir / "su2_detached_hubbard.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    frame_dims = [item["frame_D"] for item in records]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for key, label in (
        ("ordinary", "ordinary SU(2), same D"),
        ("fixed_detached", "fixed SU(2) detached"),
        ("adaptive_detached", "adaptive detached, protected-core scan"),
        ("detached_cc", "adaptive SU(2) detached + CC"),
        ("ordinary_same_chi", "ordinary SU(2), same chi"),
    ):
        axes[0].plot(
            frame_dims,
            [item[f"{key}_error"] for item in records],
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame_dims,
            [item[f"{key}_seconds"] for item in records],
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.legend(frameon=False)
    axes[0].set(xlabel="frame parameter D", ylabel="energy error (Hartree)")
    axes[1].set(xlabel="frame parameter D", ylabel="wall time (s)")
    figure.savefig(args.output_dir / "su2_detached_hubbard.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
