"""Runtime planning from a TDI pilot, not an eigensolver convergence forecast.

The hypothetical full-GW model assumes the measured near-gap chain cost and
adequate cache retention; the deeper-response probe can invalidate that
assumption. BSE budgets count trial vectors, using two real contraction
passes for complex full-BSE Davidson and scalar real passes for ARPACK.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    r = json.loads((root / "report.json").read_text())
    if r["status"] != "complete":
        raise ValueError("Complete the selected-orbital pilot first")
    baseline_path = root / "GMRES_baseline_report.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else r
    probe_path = root / "screening_thread_probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    deeper_path = root / "deeper_response_probe.json"
    deeper = json.loads(deeper_path.read_text()) if deeper_path.exists() else None
    n, o = r["nmo"], r["nocc"]
    v, targets = n - o, len(r["qp"]["orbitals"])
    dielectric = float(
        np.median(r["calls"]["dielectric"][1:] or r["calls"]["dielectric"])
    )
    rhs = (
        {str(row["blas_threads"]): row["median_seconds"] for row in probe["rows"]}
        if probe
        else {
            str(r["settings"].get("screening_blas_threads", 1)): float(
                np.mean(r["calls"]["screened_elements"])
            )
        }
    )
    # Cholesky/control allowance; full-target solves scale in equal-sized RHS batches.
    imaginary = {
        threads: 65 * (dielectric + seconds * n / targets + 2.0)
        for threads, seconds in rhs.items()
    }
    info = r["screening_info"]
    pairs = o * (o - 1) // 2 + v * (v - 1) // 2 + n
    per_chain = info["response_build_seconds"] / info["response_chains_created"]
    eval_count = (
        r.get("call_stats", {})
        .get("evaluate", {})
        .get("count", len(r["calls"]["evaluate"]))
    )
    evals = eval_count / targets
    reuse_rate = info["reused_response_seconds"] / max(1, info["reused_responses"])
    cold_model, reuse_model = pairs * per_chain, pairs * evals * reuse_rate
    cache_model = (
        max(o, v)
        * (info["iterations"] / info["response_chains_created"] + 1)
        * o
        * v
        * 8
    )
    budgets = {}
    for label, solver, width, factor in [
        ("TDA Davidson", "TDA", 8, 1),
        ("Full BSE Davidson", "Full BSE", 8, 2),
        ("Full BSE ARPACK", "Full BSE", 1, 1),
    ]:
        time = r["bse_action_timings"][solver + " batch " + str(width)]
        budgets[label] = {
            str(count): math.ceil(count / width) * factor * time
            for count in [100, 500, 2000]
        }
    model = dict(
        status="conditional planning model",
        geometry="TDI NCH3 FC",
        basis=r["settings"]["basis"],
        spatial_orbitals=n,
        occupied=o,
        virtual=v,
        transitions=o * v,
        auxiliaries=r["rank"],
        pilot_quadrature_points=r["settings"]["nw"],
        preparations_seconds={
            key: baseline["stages"][key]
            for key in ["RI build", "RHF", "AO to MO"]
            if key in baseline["stages"]
        },
        pilot_screening_seconds=r["stages"].get("imaginary screening"),
        pilot_QP_seconds=r["stages"]["selected contour GW"],
        max_QP_residual_Ha=max(map(abs, r["qp"]["residuals"])),
        imaginary_screening_798_targets_64_points_seconds=imaginary,
        hypothetical_full_GW_root_model=dict(
            pairs=pairs,
            mean_build_seconds_per_chain=per_chain,
            cold_seconds=cold_model,
            reuse_seconds=reuse_model,
            response_cache_GB_at_pilot_mean_chain_length=cache_model / 1e9,
            limitation="Assumes selected-state chain conditioning and sufficient cache; not a full-spectrum convergence guarantee",
        ),
        deeper_response_probe=deeper,
        BSE_operator_budgets_seconds=budgets,
        BSE_limitations="HF gaps outside the four selected QPs; vector counts are scenarios, not observed seven-root convergence; exclude solver orthogonalization/restarts",
        static_cache_seconds=dict(
            blas1=r["stages"]["BSE initialization and static caches"],
            blas8=probe.get("static_cache_blas8_seconds") if probe else None,
        ),
    )
    (root / "runtime_estimates.json").write_text(json.dumps(model, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = list(imaginary)
    axes[0].bar(labels, [imaginary[k] / 3600 for k in labels], color="#4978a8")
    axes[0].set(
        xlabel="BLAS threads for screening RHS solves",
        ylabel="Estimated screening wall time (h)",
        title="All 798 targets, nw=64; contractions use OpenMP=8",
    )
    for label, values in budgets.items():
        axes[1].plot(
            [int(k) for k in values],
            [value / 3600 for value in values.values()],
            "o-",
            label=label,
        )
    axes[1].set(
        xlabel="Trial vectors applied",
        ylabel="Operator-only wall time (h)",
        title="Full 142,817-transition space; assumed vector counts",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        "TDI / 6-31G: runtime budgets; full-spectrum GW convergence is unverified"
    )
    fig.savefig(root / "TDI_NCH3_FC_runtime_budgets.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    q = r["qp"]
    orbitals = q["orbitals"]
    axes[0].scatter(
        orbitals,
        (np.array(q["energies"]) - np.array(q["hf_energies"])) * 27.211386245988,
    )
    axes[0].set(ylabel="QP minus HF energy (eV)")
    axes[1].scatter(orbitals, q["weights"])
    axes[1].set(ylabel="QP weight")
    axes[2].scatter(orbitals, np.abs(q["residuals"]))
    axes[2].set(yscale="log", ylabel="Dyson residual (Ha)")
    for ax in axes:
        ax.set_xlabel("Spatial orbital (zero based)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "TDI contour GW pilot: nw=8 for timing; quadrature convergence not assessed"
    )
    fig.savefig(root / "TDI_NCH3_FC_QP_diagnostics.png", dpi=170)
    plt.close(fig)
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
