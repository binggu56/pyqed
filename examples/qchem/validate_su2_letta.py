"""Validate native SU(2)-LETTA dimers and plot their convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem import LETTA


def _run_case(name, h1e, eri, exact, *, seed, spin=0, algorithm="one_site"):
    started = time.perf_counter()
    state = LETTA.from_integrals(
        h1e,
        eri,
        symmetry="su2",
        nelec=2,
        spin=spin,
        graph=[(0, 1)],
        D=1,
        seed=seed,
    )
    initial = float(state.energy)
    state.run(
        nsweeps=1,
        algorithm=algorithm,
        tol=0.0,
        max_local_parameters=64,
    )
    trajectory = [initial]
    trajectory.extend(
        float(update["energy_after"])
        for update in state.history[0]["updates"]
    )
    result = {
        "name": name,
        "initial_energy": initial,
        "final_energy": float(state.energy),
        "exact_energy": float(exact),
        "spin": int(spin),
        "algorithm": str(algorithm),
        "absolute_error": float(abs(state.energy - exact)),
        "trajectory": trajectory,
        "parameters": int(state.nparameters),
        "storage_bytes": int(state.storage_nbytes),
        "elapsed_seconds": float(time.perf_counter() - started),
        "native_su2": bool(state.is_native_su2),
        "solvers": [
            str(update["solver"]) for update in state.history[0]["updates"]
        ],
    }
    state.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/su2_letta_validation"),
        help="Output stem for .json, .npz, and .png artifacts.",
    )
    args = parser.parse_args()

    one_body = np.array([[-1.0, -0.2], [-0.2, 0.5]])
    one_body_exact = 2.0 * np.linalg.eigvalsh(one_body)[0]

    hubbard_h1 = np.array([[0.0, -1.0], [-1.0, 0.0]])
    hubbard_eri = np.zeros((2, 2, 2, 2))
    hubbard_eri[0, 0, 0, 0] = 4.0
    hubbard_eri[1, 1, 1, 1] = 4.0
    hubbard_exact = 0.5 * (4.0 - np.sqrt(4.0**2 + 16.0))

    results = [
        _run_case("one-body dimer", one_body, None, one_body_exact, seed=4),
        _run_case(
            "Hubbard dimer",
            hubbard_h1,
            hubbard_eri,
            hubbard_exact,
            seed=2,
        ),
        _run_case(
            "Hubbard triplet",
            hubbard_h1,
            hubbard_eri,
            0.0,
            seed=3,
            spin=2,
            algorithm="two_site",
        ),
    ]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"cases": results}, indent=2) + "\n"
    )
    np.savez(
        output.with_suffix(".npz"),
        names=np.asarray([case["name"] for case in results]),
        final_energy=np.asarray([case["final_energy"] for case in results]),
        exact_energy=np.asarray([case["exact_energy"] for case in results]),
        absolute_error=np.asarray([case["absolute_error"] for case in results]),
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    for case in results:
        trajectory = np.asarray(case["trajectory"])
        axes[0].plot(
            np.arange(trajectory.size),
            trajectory,
            marker="o",
            label=case["name"],
        )
        axes[0].axhline(case["exact_energy"], color="0.55", lw=0.8, ls="--")
    axes[0].set_xlabel("Local update")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("One SU(2)-LETTA cycle")
    axes[0].legend(frameon=False)

    errors = np.maximum(
        np.asarray([case["absolute_error"] for case in results]),
        np.finfo(float).eps,
    )
    axes[1].bar(
        [case["name"] for case in results],
        errors,
        color=("#4477AA", "#CC6677", "#228833"),
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Absolute energy error")
    axes[1].set_title("Exact-reference agreement")
    axes[1].tick_params(axis="x", rotation=15)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)

    for case in results:
        print(
            f"{case['name']}: E={case['final_energy']:.15f} "
            f"exact={case['exact_energy']:.15f} "
            f"|dE|={case['absolute_error']:.3e}"
        )
    print(f"artifacts: {output}.json/.npz/.png")


if __name__ == "__main__":
    main()
