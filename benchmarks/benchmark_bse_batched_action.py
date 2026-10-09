"""Molecular BSE batching benchmark: water/cc-pVTZ, HF gaps, static TDH screening.

No GW quasiparticle correction is computed. This measures BSE actions/solves
and reports HF/screening setup separately, not total GW speedup.
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, lib
from pyqed.gw.bse import BSE, _bse_tda_diag, _bse_tda_matvec, _bse_tda_matmat
from pyqed.linalg import davidson, davidson_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if davidson_kernels.davidson_operator is None:
        raise RuntimeError(davidson_kernels.build_error)
    started = time.perf_counter()
    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvtz",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-11)
    if not mf.converged:
        raise RuntimeError("HF did not converge")
    mf.eri_factors = np.concatenate(
        [lib.unpack_tril(block) for block in mf.with_df.loop()]
    )
    hf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = BSE(mf, screening="TDH")
    model._ensure_screening()
    screening_seconds = time.perf_counter() - started
    diag = _bse_tda_diag(model)
    np.savez(
        args.output / "water_bse_factors.npz",
        mo_coeff=mf.mo_coeff,
        mo_energy=mf.mo_energy,
        pair_factors=model._pair_factors,
        screened_factors=model._M,
        screening_energies=model.e_rpa,
        nocc=model.nocc,
    )
    report = dict(
        molecule="water",
        basis="cc-pVTZ",
        density_fitting=True,
        orbital_gaps="HF (no GW QP correction)",
        screening="TDH",
        dimension=len(diag),
        hf_seconds=hf_seconds,
        screening_seconds=screening_seconds,
        action=[],
        solve=[],
    )
    vector = lambda x: _bse_tda_matvec(model, x)
    block = lambda x: _bse_tda_matmat(model, x)
    for width in (1, 8, 32):
        rng = np.random.default_rng(729 + width)
        x = rng.normal(size=(len(diag), width)) + 1j * rng.normal(
            size=(len(diag), width)
        )
        reference = np.column_stack([vector(v) for v in x.T])
        np.testing.assert_allclose(block(x), reference, rtol=1e-12, atol=1e-12)
        for label, action in [
            ("Single-vector", lambda x: np.column_stack([vector(v) for v in x.T])),
            ("Batched", block),
        ]:
            times = []
            for _ in range(3):
                started = time.perf_counter()
                action(x)
                times.append(time.perf_counter() - started)
            report["action"].append(
                dict(method=label, columns=width, seconds=float(np.median(times)))
            )
    reference = None
    for label, matmat in [("Single-vector", None), ("Batched", block)]:
        durations = []
        for repeat in range(3):
            started = time.perf_counter()
            energy, states, info = davidson(
                vector,
                8,
                diag=diag,
                matmat=matmat,
                backend="compiled",
                tolerance=1e-9,
                space=64,
                iterations=100,
            )
            durations.append(time.perf_counter() - started)
        residual = np.max(np.linalg.norm(block(states) - states * energy, axis=0))
        if reference is None:
            reference = energy
        np.testing.assert_allclose(energy, reference, atol=1e-9, rtol=0)
        if residual > 1e-9:
            raise AssertionError(f"Residual {residual} exceeds tolerance")
        report["solve"].append(
            dict(
                method=label,
                seconds=float(np.median(durations)),
                repetitions=durations,
                energies_hartree=energy.tolist(),
                max_residual=float(residual),
                max_difference=float(np.max(np.abs(energy - reference))),
                iterations=info["iterations"],
                matvecs=info["matvecs"],
                vector_callbacks=info["matvec_callback_calls"],
                block_callbacks=info["matmat_callback_calls"],
            )
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for label in ("Single-vector", "Batched"):
        rows = [r for r in report["action"] if r["method"] == label]
        axes[0].plot(
            [r["columns"] for r in rows],
            [r["seconds"] for r in rows],
            "o-",
            label=label,
        )
    axes[0].set(
        xlabel="Trial columns", ylabel="Action time (s)", title="Water/cc-pVTZ TDA-BSE"
    )
    axes[0].legend()
    bars = axes[1].bar(
        [r["method"] for r in report["solve"]], [r["seconds"] for r in report["solve"]]
    )
    for bar, row in zip(bars, report["solve"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['seconds']:.3f}",
            ha="center",
            va="bottom",
        )
    axes[1].set(ylabel="Solve time (s)", title="8 roots, median of 3 solves")
    fig.savefig(args.output / "water_bse_batching.png", dpi=180)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
