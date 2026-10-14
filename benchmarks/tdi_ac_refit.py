"""Compare Padé and two-pole fits of saved TDI self-energy; never run screening."""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed.gw.ac import refit, _twopole

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
source = a.input / "qp.npz"
metadata = json.loads((a.input / "report.json").read_text())
if (
    metadata["status"] != "complete"
    or metadata["completed_frequencies"] != metadata["nw"]
):
    raise ValueError("Require a complete saved imaginary-axis integration")
with np.load(source, allow_pickle=False) as saved:
    energy, sigma, omega, old_coeff, old_qp = [
        saved[k].copy() for k in ("hf", "sigma", "omega", "coefficients", "qp")
    ]
ids = np.asarray(metadata["qp_orbitals"], dtype=int)
ef = metadata["ac_info"]["fermi_level"]
if metadata["ac_info"]["method"] != "twopole" or not np.array_equal(
    ids, np.flatnonzero(np.isfinite(old_qp))
):
    raise ValueError(
        "Legacy static-shift recovery requires the original two-pole checkpoint"
    )
# The old checkpoint omitted Sigma_x-v_xc. Recover this constant from its
# converged QP equation; the same fixed shifts are used in every new fit.
static = (
    old_qp[ids]
    - energy[ids]
    - np.array(
        [_twopole(old_qp[p] - ef, old_coeff[:, j]).real for j, p in enumerate(ids)]
    )
)
report = dict(
    source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    screening_rerun=False,
    quadrature_points=metadata["nw"],
    orbitals=ids.tolist(),
    evaluation_points=sigma.shape[1],
    static_source="Recovered from saved two-pole QP equation; fixed for all fits",
    runs={},
)
arrays = dict(energy=energy, orbs=ids, static=static, sigma=sigma, omega=omega)
with threadpool_limits(limits=8, user_api="blas"):
    for method, points in [("twopole", None)] + [
        ("pade", n) for n in (12, 14, 16, 18, 20, 24)
    ]:
        key = method if points is None else f"pade_{points}"
        gw = SimpleNamespace(
            ac_sigma=sigma,
            ac_omega=omega,
            ac_info=dict(fermi_level=ef),
            nocc=2 * metadata["nocc"],
            orbital_indices=ids,
        )
        t = time.perf_counter()
        try:
            qp = refit(
                gw,
                energy,
                static=static,
                ac=method,
                **({} if points is None else dict(pade_npts=points)),
            )
            row = dict(
                status="complete",
                seconds=time.perf_counter() - t,
                qp_eV=(qp[ids] * 27.211386245988).tolist(),
                gap_eV=float(
                    (qp[metadata["nocc"]] - qp[metadata["nocc"] - 1]) * 27.211386245988
                ),
                weights=gw.qp_weights[ids].tolist(),
                residuals_Ha=gw.qp_residuals[ids].tolist(),
                info=gw.ac_info,
            )
            arrays[key] = qp
            arrays[key + "_coeff"] = gw.ac_coeff
            if gw.ac_nodes is not None:
                arrays[key + "_nodes"] = gw.ac_nodes
        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            row = dict(status="failed", seconds=time.perf_counter() - t, error=str(exc))
        report["runs"][key] = row
        print(
            key, json.dumps({k: v for k, v in row.items() if k != "info"}), flush=True
        )
np.savez(a.output / "fits.npz", **arrays)
if all(report["runs"][k]["status"] == "complete" for k in ("twopole", "pade_18")):
    report["two_pole_reproduction_max_Ha"] = float(
        np.max(abs(arrays["twopole"][ids] - old_qp[ids]))
    )
    report["pade18_minus_twopole_meV"] = (
        (arrays["pade_18"][ids] - arrays["twopole"][ids]) * 27211.386245988
    ).tolist()
(a.output / "comparison.json").write_text(json.dumps(report, indent=2))
if not all(report["runs"][k]["status"] == "complete" for k in ("twopole", "pade_18")):
    raise RuntimeError("Primary comparison failed; see comparison.json")
os.environ["MPLCONFIGDIR"] = "/private/tmp/pyqed-mpl"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for key, label, marker in [
    ("twopole", "Two-pole", "o"),
    ("pade_18", "Padé (18 points)", "x"),
]:
    axes[0, 0].plot(ids, report["runs"][key]["qp_eV"], marker + "-", label=label)
    axes[1, 0].semilogy(
        ids,
        np.asarray(report["runs"][key]["info"]["fit_max_errors"]) * 27211.386245988,
        marker + "-",
        label=label,
    )
axes[0, 0].set(xlabel="Spatial orbital (zero based)", ylabel="QP energy (eV)")
axes[0, 0].legend()
axes[0, 1].bar(ids, report["pade18_minus_twopole_meV"])
axes[0, 1].axhline(0, color="black", linewidth=0.7)
axes[0, 1].set(
    xlabel="Spatial orbital (zero based)", ylabel="Padé minus two-pole (meV)"
)
axes[1, 0].set(
    xlabel="Spatial orbital (zero based)",
    ylabel="Maximum imaginary-axis residual (meV)",
)
axes[1, 0].set_yticks([0.5, 1, 2, 5], labels=["0.5", "1", "2", "5"])
axes[1, 0].legend()
points = [
    n
    for n in (12, 14, 16, 18, 20, 24)
    if report["runs"][f"pade_{n}"]["status"] == "complete"
]
axes[1, 1].plot(
    points, [report["runs"][f"pade_{n}"]["gap_eV"] for n in points], "o-", label="Padé"
)
axes[1, 1].axhline(report["runs"]["twopole"]["gap_eV"], color="C1", label="Two-pole")
axes[1, 1].set(xlabel="Padé interpolation points", ylabel="HOMO–LUMO gap (eV)")
axes[1, 1].legend()
fig.suptitle(
    "TDI: identical saved 64-point self-energy, eight frontier orbitals\nNo screening rerun; imaginary-axis residuals do not bound real-axis errors"
)
fig.savefig(a.output / "TDI_Pade_vs_twopole.png", dpi=170)
