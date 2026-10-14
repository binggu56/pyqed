"""Time exact contour screening/residue actions against the original equations.

Synthetic real RI factors isolate these kernels; this is not a molecular
runtime prediction. Run with PYTHONPATH=. and one BLAS/OpenMP thread.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import scipy.linalg as la

from pyqed.gw.contour import ContourDeformation


class ReferenceContour(ContourDeformation):
    """Explicit validation reference for the original 256-transition kernels."""

    def dielectric(self, frequency, *, imaginary=False):
        z = frequency + 1j * self.eta
        weights = (-4*self.gaps/(frequency**2+self.gaps**2) if imaginary
                   else 4*self.gaps/(z*z-self.gaps**2))
        matrix = np.eye(len(self.factors), dtype=weights.dtype, order="F")
        for start in range(0, len(self.gaps), 256):
            s = slice(start, start+256)
            pairs = self.pairs[:, s]
            matrix -= (pairs*weights[s]) @ pairs.T
        return matrix

    def residue(self, frequency, orbital, intermediate):
        rhs = self.factors[:, intermediate, orbital]
        solution = la.solve(self.dielectric(frequency), rhs, assume_a="gen",
                            overwrite_a=True, check_finite=False)
        z = frequency + 1j*self.eta
        weights = -8*z*self.gaps/(z*z-self.gaps**2)**2
        derivative = 0j
        for start in range(0, len(self.gaps), 256):
            s = slice(start, start+256)
            projected = self.pairs[:, s].T @ solution
            derivative += np.sum(weights[s]*projected**2)
        return rhs @ (solution-rhs), derivative


def reference_and_current(rank, nmo):
    rng = np.random.default_rng(614)
    factors = rng.normal(scale=0.01, size=(rank, nmo, nmo))
    factors = (factors+factors.transpose(0, 2, 1))/2
    nocc = nmo//3
    energy = np.r_[np.linspace(-2, -0.5, nocc),
                   np.linspace(0.1, 2.0, nmo-nocc)]
    # Bypass quadrature preparation to isolate the hot kernels.
    objects = []
    for cls in (ReferenceContour, ContourDeformation):
        obj = object.__new__(cls)
        obj.factors = factors
        obj.gaps = (energy[nocc:]-energy[:nocc, None]).ravel()
        obj.pairs = factors[:, :nocc, nocc:].reshape(rank, -1)
        obj.eta, obj.block = 1e-6, 4096
        obj.residue_solver = "direct"
        obj.target_factors = None
        obj.screening_info = dict(solver="direct", solves=0, iterations=0,
                                 max_iterations=0, max_relative_residual=0., failed_solves=0)
        objects.append(obj)
    return objects


def measure(call, repeats):
    call()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter()-start)
    return statistics.median(durations)


def molecular_check(output):
    """Compare complete water/STO-3G QP and neutral energies at fixed controls."""
    import importlib
    from unittest.mock import patch
    from pyscf import df, gto, scf
    from pyqed.qchem.basis import PackedRIFactors
    from pyqed.gw.bse import BSE, TDA
    gw_module = importlib.import_module("pyqed.gw.gw")
    mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile, mf.conv_tol = None, 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Reference RHF did not converge")
    fitted = df.DF(mol).build()
    mol.eri_factors = PackedRIFactors(np.concatenate(list(fitted.loop())), mol.nao)
    results = []
    for cls in [ReferenceContour, ContourDeformation]:
        with patch("pyqed.gw.contour.ContourDeformation", cls):
            gw = gw_module.GW(mf, eta=1e-6, freq_int="contour").run(nw=64, tol=1e-9)
            results.append(dict(qp=gw.e_qp.copy(), weights=gw.qp_weights.copy(),
                                residuals=gw.qp_residuals.copy(),
                                tda=TDA(gw).run(nroots=3, tol=1e-8).e.copy(),
                                bse=BSE(gw).run(nroots=3, tol=1e-8,
                                                eigensolver="arpack").e.copy()))
    differences = {}
    for key in results[0]:
        np.testing.assert_allclose(results[0][key], results[1][key],
                                   atol=1e-10, rtol=1e-10)
        differences[key] = (results[1][key]-results[0][key]).tolist()
    (output/"water_energy_comparison.json").write_text(json.dumps(
        dict(system="water/STO-3G, RI", nw=64, eta=1e-6, qp_tol=1e-9,
             before={k: v.tolist() for k, v in results[0].items()},
             after={k: v.tolist() for k, v in results[1].items()},
             differences=differences), indent=2)+"\n")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), constrained_layout=True)
    for ax, key in zip(axes, ["qp", "tda", "bse"]):
        delta = differences[key]
        ax.plot(np.arange(len(delta)), np.array(delta)*1e12, "o")
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set(xlabel="Orbital" if key == "qp" else "Root", title=key.upper(),
               ylabel="After − before (10⁻¹² Hartree)")
    fig.suptitle("Water reference: unchanged quadrature and convergence settings")
    fig.savefig(output/"water_energy_agreement.png", dpi=160)
    plt.close(fig)
    print("WATER_MAX_DIFFERENCES", {k: max(abs(x) for x in v)
                                     for k, v in differences.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--molecular-check", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, nmo in [(128, 80), (256, 120), (512, 180)]:
        before, after = reference_and_current(rank, nmo)
        for action in ["dielectric", "residue"]:
            calls = [getattr(obj, action) for obj in [before, after]]
            def evaluate(call):
                return [call(frequency, 0, 1) if action == "residue"
                        else call(frequency) for frequency in [0.25, 1.3, 2.8]]
            expected, actual = [evaluate(call) for call in calls]
            error = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                        for a, b in zip(expected, actual))
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10)
            times = [measure(lambda: evaluate(call), args.repeats) for call in calls]
            row = dict(rank=rank, nmo=nmo, transitions=len(after.gaps),
                       action=action, before_s=times[0], after_s=times[1],
                       speedup=times[0]/times[1], max_absolute_difference=error)
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(scope="synthetic kernels, three frequencies per timed sample",
                  repeats=args.repeats, rows=rows)
    (args.output/"contour_kernel_timings.json").write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, action in zip(axes, ["dielectric", "residue"]):
        selected = [row for row in rows if row["action"] == action]
        x = np.arange(len(selected))
        ax.bar(x-0.18, [r["before_s"] for r in selected], width=0.36, label="Before")
        ax.bar(x+0.18, [r["after_s"] for r in selected], width=0.36, label="After")
        ax.set_xticks(x, [str(r["rank"]) for r in selected])
        ax.set(xlabel="Auxiliary rank", ylabel="Seconds / three frequencies",
               title="Screening matrix" if action == "dielectric" else "Complete residue")
        ax.legend(frameon=False)
    fig.suptitle("Contour GW kernels: unchanged equations, synthetic RI factors")
    fig.savefig(args.output/"contour_kernel_speedup.png", dpi=160)
    plt.close(fig)
    if args.molecular_check:
        molecular_check(args.output)


if __name__ == "__main__":
    main()
