"""Time a 4 occupied / 4 virtual full-BSE window after the TDI AC pilot.

Uses the complete static RPA dielectric checkpoint from the contour pilot;
only the BSE excitation space is restricted. This is a timing/interop check,
not a converged excitation spectrum. Run after tdi_ac_benchmark.py finishes.
"""

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
import time
import numpy as np
from threadpoolctl import threadpool_limits
from benchmarks.tdi_window_convergence import input_data, restore_rhf
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ac-output", type=Path, required=True)
p.add_argument("--pade-refit", type=Path, help="Folder containing saved Padé fits.npz")
p.add_argument("--output", type=Path, help="Separate benchmark output folder")
a = p.parse_args()
output = a.output or a.ac_output
output.mkdir(parents=True, exist_ok=True)
report = json.loads((a.ac_output / "report.json").read_text())
if report["status"] != "complete":
    raise ValueError("Complete the AC benchmark first")
checkpoint = Path(report["checkpoint"])
source = checkpoint / "input_script_snapshot.py"
if hashlib.sha256(source.read_bytes()).hexdigest() != report["input_sha256"]:
    raise ValueError("Reference provenance mismatch")
xyz, cfg = input_data(source)
rows = [r.split() for r in xyz.strip().splitlines()[2:]]
mol = Molecule(
    atom=[(r[0], list(map(float, r[1:]))) for r in rows],
    basis=cfg["basis"],
    unit="angstrom",
    charge=0,
    spin=0,
)
with np.load(a.ac_output / "qp.npz") as saved:
    qp, energy = saved["qp"].copy(), saved["hf"].copy()
qp_source = a.ac_output / "qp.npz"
method = report["ac_info"]["method"]
if a.pade_refit:
    comparison = json.loads((a.pade_refit / "comparison.json").read_text())
    if comparison["runs"]["pade_18"]["status"] != "complete":
        raise ValueError("Padé refit is incomplete")
    if (
        comparison["source_sha256"]
        != hashlib.sha256(qp_source.read_bytes()).hexdigest()
    ):
        raise ValueError("Padé self-energy provenance mismatch")
    qp_source = a.pade_refit / "fits.npz"
    with np.load(qp_source) as saved:
        qp = saved["pade_18"].copy()
    method = "pade_18"
if not np.isfinite(qp[report["nocc"] - 4 : report["nocc"] + 4]).all():
    raise ValueError("All eight window QP energies must be available")
print(f"Restoring saved reference for {method} BSE", flush=True)
with threadpool_limits(limits=8, user_api="blas"):
    start = time.perf_counter()
    mf = restore_rhf(mol, checkpoint, 4000)
    np.testing.assert_array_equal(mf.mo_energy, energy)
    factors = np.load(checkpoint / "MO_RI_factors.npy", mmap_mode="r")
    gw = GW(mf, freq_int="ac", threads=8, mo_pair_factors=factors)
    gw.e_qp = qp
    cache = checkpoint / "screening_cache"
    signature = json.loads((cache / "reference.json").read_text())
    # The static (zero-frequency) factor is independent of quadrature size.
    expected_energy = hashlib.sha256(
        np.asarray(energy, dtype=float).tobytes()
    ).hexdigest()
    expected_coeff = hashlib.sha256(
        np.asarray(mf.mo_coeff, dtype=float).tobytes()
    ).hexdigest()
    if (
        signature["energy_sha256"] != expected_energy
        or signature["coefficient_sha256"] != expected_coeff
    ):
        raise ValueError("Static-screening reference mismatch")
    response = ContourDeformation(
        gw,
        energy,
        list(np.flatnonzero(np.isfinite(qp))),
        nw=8,
        prepare=False,
        transition_factors=np.load(
            cache / "occupied_virtual_factors.npy", mmap_mode="r"
        ),
    )
    response.static_screening = StaticScreening(
        factors, (np.load(cache / "cholesky_0.npy", mmap_mode="r"), True), 4000
    )
    gw._ac_screening = response
    restore_seconds = time.perf_counter() - start
    start = time.perf_counter()
    bse = BSE(gw, orbital_window=(4, 4), threads=8)
    init_seconds = time.perf_counter() - start
    assert bse._static_screening.cholesky is response.static_screening.cholesky
    print(
        f"Restored in {restore_seconds:.2f} s; solving BSE with 8 threads", flush=True
    )
with threadpool_limits(limits=1, user_api="blas"):
    start = time.perf_counter()
    bse.run(
        nroots=7,
        eigensolver="davidson",
        max_space=32,
        tol=1e-8,
        return_info=True,
        return_vectors=True,
    )
    solve_seconds = time.perf_counter() - start
out = dict(
    window=[4, 4],
    nroots=7,
    threads=8,
    restore_seconds=restore_seconds,
    initialization_seconds=init_seconds,
    solve_seconds=solve_seconds,
    energies_eV=(np.asarray(bse.e) * 27.211386245988).tolist(),
    info=bse.info,
    threading_info=getattr(bse, "threading_info", None),
    continuation=method,
    qp_source=str(qp_source),
    qp_source_sha256=hashlib.sha256(qp_source.read_bytes()).hexdigest(),
    screening_rerun=False,
    screening_reference="HF energies; saved full-space static RPA",
    integration_points=report["ac_info"]["nw"],
    scope="Restricted excitation-space timing; full static RPA screening; not window converged",
)
(output / "bse_window_report.json").write_text(
    json.dumps(
        out, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)
    )
)
if not bse.info["converged"]:
    raise RuntimeError("Restricted BSE did not converge; see bse_window_report.json")
np.savez(output / "bse_window.npz", energies=bse.e, xy=bse.xy)
os.environ["MPLCONFIGDIR"] = "/private/tmp/pyqed-mpl"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.plot(np.arange(1, 8), out["energies_eV"], "o-", label=method)
previous = a.ac_output / "bse_window_report.json"
if a.pade_refit and previous.exists() and previous.parent != output:
    baseline = json.loads(previous.read_text())
    ax.plot(np.arange(1, 8), baseline["energies_eV"], "x--", label="Saved two-pole")
ax.legend()
ax.set(
    xlabel="Excited state",
    ylabel="Full-BSE excitation energy (eV)",
    title="TDI AC→BSE timing pilot: 4 occupied + 4 virtual\nSaved HF-based RPA screening; window not converged",
)
fig.savefig(output / "TDI_AC_BSE_window.png", dpi=160)
print(json.dumps(out, default=str), flush=True)
