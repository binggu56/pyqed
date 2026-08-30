"""One-parameter polaron disentangler benchmark for a two-mode LDR model.

The toy Hamiltonian is

    H = px^2/2 + py^2/2
        + 0.5 * omega_high^2 * x^2
        + 0.5 * omega_low^2 * y^2
        + coupling * x * y.

LDR treats the high-frequency ``x`` mode as the conditional, electronic-like
space and the low-frequency ``y`` mode as the slow nuclear grid.  For this
bilinear harmonic model the exact conditional high-mode basis is a displaced
oscillator.  The benchmark scans

    U_eta(y) = exp(-i * eta * y * p_x)

applied to the high-mode basis at y=0 and compares it with the exact
conditional LDR basis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pyqed.dvr.dvr_1d import SineDVR


@dataclass
class TwoModeModel:
    x: np.ndarray
    y: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    px: np.ndarray
    omega_high: float
    omega_low: float
    coupling: float
    x4: float
    y4: float
    x2y: float
    xy2: float
    x2y2: float


@dataclass
class ScanResult:
    etas: np.ndarray
    fidelity: np.ndarray
    weighted_fidelity: np.ndarray
    energy_rms: np.ndarray
    projected_energies: np.ndarray
    exact_ldr_energies: np.ndarray
    exact_dvr_energies: np.ndarray
    analytic_energies: np.ndarray
    expected_eta: float
    best_eta: float
    best_energy_eta: float
    optimized_energy_eta: float
    best_index: int
    best_energy_index: int
    bare_index: int
    optimized_energy_rms: float
    optimized_energy_energies: np.ndarray

    @property
    def bare_weighted_fidelity(self) -> float:
        return float(self.weighted_fidelity[self.bare_index])

    @property
    def best_weighted_fidelity(self) -> float:
        return float(self.weighted_fidelity[self.best_index])

    @property
    def bare_energy_rms(self) -> float:
        return float(self.energy_rms[self.bare_index])

    @property
    def best_energy_rms(self) -> float:
        return float(self.energy_rms[self.best_index])

    @property
    def minimum_energy_rms(self) -> float:
        return float(self.energy_rms[self.best_energy_index])


def build_model(
    *,
    omega_high: float = 4.0,
    omega_low: float = 1.0,
    coupling: float = 2.4,
    x4: float = 0.0,
    y4: float = 0.0,
    x2y: float = 0.0,
    xy2: float = 0.0,
    x2y2: float = 0.0,
    qmax: float = 7.0,
    npoints: int = 36,
) -> TwoModeModel:
    if omega_high <= 0.0 or omega_low <= 0.0:
        raise ValueError("Frequencies must be positive.")
    if abs(coupling) >= omega_high * omega_low:
        raise ValueError("The bilinear oscillator is unstable for |coupling| >= omega_high * omega_low.")
    dvr_x = SineDVR(-qmax, qmax, npoints, mass=1.0)
    dvr_y = SineDVR(-qmax, qmax, npoints, mass=1.0)
    px = dvr_x.momentum()
    px = 0.5 * (px + px.conj().T)
    return TwoModeModel(
        x=np.asarray(dvr_x.x, dtype=float),
        y=np.asarray(dvr_y.x, dtype=float),
        tx=np.asarray(dvr_x.t(), dtype=complex),
        ty=np.asarray(dvr_y.t(), dtype=complex),
        px=np.asarray(px, dtype=complex),
        omega_high=float(omega_high),
        omega_low=float(omega_low),
        coupling=float(coupling),
        x4=float(x4),
        y4=float(y4),
        x2y=float(x2y),
        xy2=float(xy2),
        x2y2=float(x2y2),
    )


def low_potential(model: TwoModeModel, y_value: np.ndarray | float) -> np.ndarray:
    y = np.asarray(y_value, dtype=float)
    return 0.5 * model.omega_low**2 * y**2 + model.y4 * y**4


def high_conditional_potential(model: TwoModeModel, y_value: float) -> np.ndarray:
    x = model.x
    y = float(y_value)
    return (
        0.5 * model.omega_high**2 * x**2
        + model.x4 * x**4
        + model.coupling * x * y
        + model.x2y * x**2 * y
        + model.xy2 * x * y**2
        + model.x2y2 * x**2 * y**2
    )


def is_pure_harmonic_bilinear(model: TwoModeModel) -> bool:
    return (
        model.x4 == 0.0
        and model.y4 == 0.0
        and model.x2y == 0.0
        and model.xy2 == 0.0
        and model.x2y2 == 0.0
    )


def high_hamiltonian(model: TwoModeModel, y_value: float) -> np.ndarray:
    return model.tx + np.diag(high_conditional_potential(model, y_value))


def exact_conditional_basis(model: TwoModeModel, nstates: int) -> tuple[np.ndarray, np.ndarray]:
    basis = np.empty((model.y.size, model.x.size, nstates), dtype=complex)
    energies = np.empty((model.y.size, nstates), dtype=float)
    for iy, y_value in enumerate(model.y):
        evals, evecs = la.eigh(high_hamiltonian(model, float(y_value)))
        basis[iy] = evecs[:, :nstates]
        energies[iy] = evals[:nstates]
    return basis, energies


def reference_high_basis(model: TwoModeModel, nstates: int) -> np.ndarray:
    evals, evecs = la.eigh(high_hamiltonian(model, 0.0))
    return evecs[:, :nstates]


def translated_basis(
    model: TwoModeModel,
    reference_basis: np.ndarray,
    eta: float,
    px_eigh: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    pvals, pvecs = px_eigh if px_eigh is not None else la.eigh(model.px)
    basis = np.empty((model.y.size, model.x.size, reference_basis.shape[1]), dtype=complex)
    for iy, y_value in enumerate(model.y):
        phase = np.exp(-1j * eta * y_value * pvals)
        shift = (pvecs * phase) @ pvecs.conj().T
        basis[iy] = shift @ reference_basis
    return basis


def slow_ground_weights(model: TwoModeModel) -> np.ndarray:
    hy = model.ty + np.diag(low_potential(model, model.y))
    evals, evecs = la.eigh(hy)
    weights = np.abs(evecs[:, 0]) ** 2
    return weights / np.sum(weights)


def subspace_fidelity(
    exact_basis: np.ndarray,
    trial_basis: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    ngrid, _, nstates = exact_basis.shape
    local = np.empty(ngrid, dtype=float)
    for iy in range(ngrid):
        overlap = exact_basis[iy].conj().T @ trial_basis[iy]
        local[iy] = float(np.linalg.norm(overlap, ord="fro") ** 2 / nstates)
    if weights is None:
        weights = np.full(ngrid, 1.0 / ngrid)
    return float(np.mean(local)), float(np.dot(weights, local))


def ldr_hamiltonian(model: TwoModeModel, basis: np.ndarray) -> np.ndarray:
    ny, _, nstates = basis.shape
    dim = ny * nstates
    ham = np.zeros((dim, dim), dtype=complex)

    for iy in range(ny):
        row = slice(iy * nstates, (iy + 1) * nstates)
        for jy in range(ny):
            col = slice(jy * nstates, (jy + 1) * nstates)
            overlap = basis[iy].conj().T @ basis[jy]
            ham[row, col] += model.ty[iy, jy] * overlap

    eye = np.eye(nstates, dtype=complex)
    for iy, y_value in enumerate(model.y):
        row = slice(iy * nstates, (iy + 1) * nstates)
        hx = high_hamiltonian(model, float(y_value))
        block = basis[iy].conj().T @ hx @ basis[iy]
        block += low_potential(model, y_value) * eye
        ham[row, row] += block

    return 0.5 * (ham + ham.conj().T)


def exact_dvr_energies(model: TwoModeModel, nroots: int) -> np.ndarray:
    nx = model.x.size
    ny = model.y.size
    x_grid, y_grid = np.meshgrid(model.x, model.y, indexing="xy")
    potential = (
        0.5 * model.omega_high**2 * x_grid**2
        + model.x4 * x_grid**4
        + low_potential(model, y_grid)
        + model.coupling * x_grid * y_grid
        + model.x2y * x_grid**2 * y_grid
        + model.xy2 * x_grid * y_grid**2
        + model.x2y2 * x_grid**2 * y_grid**2
    )
    ham = (
        sp.kron(sp.csr_matrix(model.ty), sp.eye(nx, format="csr"), format="csr")
        + sp.kron(sp.eye(ny, format="csr"), sp.csr_matrix(model.tx), format="csr")
        + sp.diags(potential.reshape(ny * nx), format="csr")
    )
    evals = spla.eigsh(ham, k=nroots, which="SA", return_eigenvectors=False, tol=1.0e-11)
    return np.sort(np.asarray(evals, dtype=float))


def analytic_energies(model: TwoModeModel, nroots: int) -> np.ndarray:
    if not is_pure_harmonic_bilinear(model):
        return np.full(nroots, np.nan, dtype=float)
    hessian = np.array(
        [
            [model.omega_high**2, model.coupling],
            [model.coupling, model.omega_low**2],
        ],
        dtype=float,
    )
    frequencies = np.sqrt(np.linalg.eigvalsh(hessian))
    levels = []
    cutoff = nroots + 4
    for n0 in range(cutoff):
        for n1 in range(cutoff):
            levels.append((n0 + 0.5) * frequencies[0] + (n1 + 0.5) * frequencies[1])
    return np.sort(np.asarray(levels, dtype=float))[:nroots]


def projected_energies_for_eta(
    model: TwoModeModel,
    reference_basis: np.ndarray,
    eta: float,
    nroots: int,
    px_eigh: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    basis = translated_basis(model, reference_basis, float(eta), px_eigh=px_eigh)
    ham = ldr_hamiltonian(model, basis)
    return np.linalg.eigvalsh(ham)[:nroots].real


def run_scan(
    *,
    omega_high: float = 4.0,
    omega_low: float = 1.0,
    coupling: float = 2.4,
    x4: float = 0.0,
    y4: float = 0.0,
    x2y: float = 0.0,
    xy2: float = 0.0,
    x2y2: float = 0.0,
    qmax: float = 7.0,
    npoints: int = 36,
    nstates: int = 4,
    nroots: int = 6,
    eta_min: float | None = None,
    eta_max: float | None = None,
    eta_points: int = 101,
    optimize_energy: bool = False,
    optimizer_xatol: float = 1.0e-4,
) -> ScanResult:
    model = build_model(
        omega_high=omega_high,
        omega_low=omega_low,
        coupling=coupling,
        x4=x4,
        y4=y4,
        x2y=x2y,
        xy2=xy2,
        x2y2=x2y2,
        qmax=qmax,
        npoints=npoints,
    )
    expected_eta = -model.coupling / model.omega_high**2
    if eta_min is None or eta_max is None:
        span = max(0.25, 2.5 * abs(expected_eta))
        if eta_min is None:
            eta_min = expected_eta - span
        if eta_max is None:
            eta_max = expected_eta + span
    etas = np.linspace(float(eta_min), float(eta_max), int(eta_points))

    exact_basis, _ = exact_conditional_basis(model, nstates)
    reference_basis = reference_high_basis(model, nstates)
    weights = slow_ground_weights(model)
    px_eigh = la.eigh(model.px)

    exact_ldr_ham = ldr_hamiltonian(model, exact_basis)
    exact_ldr_energies = np.linalg.eigvalsh(exact_ldr_ham)[:nroots].real
    exact_dvr = exact_dvr_energies(model, nroots)
    analytic = analytic_energies(model, nroots)

    fidelity = np.empty_like(etas)
    weighted_fidelity = np.empty_like(etas)
    energy_rms = np.empty_like(etas)
    projected = np.empty((etas.size, nroots), dtype=float)
    for ieta, eta in enumerate(etas):
        basis = translated_basis(model, reference_basis, float(eta), px_eigh=px_eigh)
        fidelity[ieta], weighted_fidelity[ieta] = subspace_fidelity(
            exact_basis,
            basis,
            weights=weights,
        )
        evals = projected_energies_for_eta(
            model,
            reference_basis,
            float(eta),
            nroots,
            px_eigh=px_eigh,
        )
        projected[ieta] = evals
        energy_rms[ieta] = float(np.sqrt(np.mean((evals - exact_dvr) ** 2)))

    best_index = int(np.argmax(weighted_fidelity))
    best_energy_index = int(np.argmin(energy_rms))
    bare_index = int(np.argmin(np.abs(etas)))

    optimized_energy_eta = float(etas[best_energy_index])
    optimized_energy_rms = float(energy_rms[best_energy_index])
    optimized_energy_energies = projected[best_energy_index].copy()
    if optimize_energy:
        def objective(eta: float) -> float:
            evals = projected_energies_for_eta(
                model,
                reference_basis,
                float(eta),
                nroots,
                px_eigh=px_eigh,
            )
            return float(np.sqrt(np.mean((evals - exact_dvr) ** 2)))

        bracket = opt.minimize_scalar(
            objective,
            bounds=(float(etas[0]), float(etas[-1])),
            method="bounded",
            options={"xatol": float(optimizer_xatol)},
        )
        if bracket.success:
            optimized_energy_eta = float(bracket.x)
            optimized_energy_rms = float(bracket.fun)
            optimized_energy_energies = projected_energies_for_eta(
                model,
                reference_basis,
                optimized_energy_eta,
                nroots,
                px_eigh=px_eigh,
            )

    return ScanResult(
        etas=etas,
        fidelity=fidelity,
        weighted_fidelity=weighted_fidelity,
        energy_rms=energy_rms,
        projected_energies=projected,
        exact_ldr_energies=exact_ldr_energies,
        exact_dvr_energies=exact_dvr,
        analytic_energies=analytic,
        expected_eta=float(expected_eta),
        best_eta=float(etas[best_index]),
        best_energy_eta=float(etas[best_energy_index]),
        optimized_energy_eta=optimized_energy_eta,
        best_index=best_index,
        best_energy_index=best_energy_index,
        bare_index=bare_index,
        optimized_energy_rms=optimized_energy_rms,
        optimized_energy_energies=optimized_energy_energies,
    )


def save_outputs(result: ScanResult, output_dir: Path, prefix: str = "ldr_polaron") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{prefix}.npz",
        etas=result.etas,
        fidelity=result.fidelity,
        weighted_fidelity=result.weighted_fidelity,
        energy_rms=result.energy_rms,
        projected_energies=result.projected_energies,
        exact_ldr_energies=result.exact_ldr_energies,
        exact_dvr_energies=result.exact_dvr_energies,
        analytic_energies=result.analytic_energies,
        expected_eta=result.expected_eta,
        best_eta=result.best_eta,
        best_energy_eta=result.best_energy_eta,
        optimized_energy_eta=result.optimized_energy_eta,
        best_index=result.best_index,
        best_energy_index=result.best_energy_index,
        bare_index=result.bare_index,
        optimized_energy_rms=result.optimized_energy_rms,
        optimized_energy_energies=result.optimized_energy_energies,
    )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(result.etas, result.weighted_fidelity, label="weighted fidelity")
    ax1.plot(result.etas, result.fidelity, "--", label="uniform fidelity")
    ax1.axvline(result.expected_eta, color="k", linestyle=":", label="analytic shift")
    ax1.axvline(result.best_eta, color="tab:red", linestyle="-.", label="best fidelity")
    ax1.axvline(result.best_energy_eta, color="tab:green", linestyle="--", label="best energy scan")
    ax1.axvline(result.optimized_energy_eta, color="tab:purple", linestyle="-", label="Brent energy")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel("subspace fidelity")
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_fidelity.png", dpi=180)
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(7.0, 4.2))
    ax2.semilogy(result.etas, result.energy_rms)
    ax2.axvline(result.expected_eta, color="k", linestyle=":", label="analytic shift")
    ax2.axvline(result.best_eta, color="tab:red", linestyle="-.", label="best fidelity")
    ax2.axvline(result.best_energy_eta, color="tab:green", linestyle="--", label="best energy scan")
    ax2.axvline(result.optimized_energy_eta, color="tab:purple", linestyle="-", label="Brent energy")
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel("RMS low-energy error")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_energy_error.png", dpi=180)
    plt.close(fig)


def format_summary(result: ScanResult) -> str:
    lines = [
        "One-parameter high/low LDR polaron-disentangler scan",
        f"  analytic eta      = {result.expected_eta:.8f}",
        f"  best fidelity eta = {result.best_eta:.8f}",
        f"  best energy scan  = {result.best_energy_eta:.8f}",
        f"  Brent energy eta  = {result.optimized_energy_eta:.8f}",
        f"  F(eta=0)          = {result.bare_weighted_fidelity:.8f}",
        f"  F(best)           = {result.best_weighted_fidelity:.8f}",
        f"  RMS E err eta=0   = {result.bare_energy_rms:.8e}",
        f"  RMS E err best-F  = {result.best_energy_rms:.8e}",
        f"  RMS E err scan-E  = {result.minimum_energy_rms:.8e}",
        f"  RMS E err Brent-E = {result.optimized_energy_rms:.8e}",
        "",
        "  level      analytic        exact-DVR        exact-LDR        eta=0           eta-best-F       eta-scan-E      eta-Brent-E",
    ]
    bare = result.projected_energies[result.bare_index]
    best = result.projected_energies[result.best_index]
    best_energy = result.projected_energies[result.best_energy_index]
    optimized = result.optimized_energy_energies
    for i, (ana, dvr, ldr, e0, eb, ebe, eopt) in enumerate(
        zip(
            result.analytic_energies,
            result.exact_dvr_energies,
            result.exact_ldr_energies,
            bare,
            best,
            best_energy,
            optimized,
        )
    ):
        lines.append(
            f"  {i:3d}  {ana:14.8f} {dvr:14.8f} {ldr:14.8f} "
            f"{e0:14.8f} {eb:14.8f} {ebe:14.8f} {eopt:14.8f}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omega-high", type=float, default=4.0)
    parser.add_argument("--omega-low", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=2.4)
    parser.add_argument("--x4", type=float, default=0.0)
    parser.add_argument("--y4", type=float, default=0.0)
    parser.add_argument("--x2y", type=float, default=0.0)
    parser.add_argument("--xy2", type=float, default=0.0)
    parser.add_argument("--x2y2", type=float, default=0.0)
    parser.add_argument("--qmax", type=float, default=7.0)
    parser.add_argument("--npoints", type=int, default=36)
    parser.add_argument("--nstates", type=int, default=4)
    parser.add_argument("--nroots", type=int, default=6)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=None)
    parser.add_argument("--eta-points", type=int, default=101)
    parser.add_argument("--optimize-energy", action="store_true")
    parser.add_argument("--optimizer-xatol", type=float, default=1.0e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/ldr_polaron_disentangler"))
    parser.add_argument("--prefix", default="ldr_polaron")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_scan(
        omega_high=args.omega_high,
        omega_low=args.omega_low,
        coupling=args.coupling,
        x4=args.x4,
        y4=args.y4,
        x2y=args.x2y,
        xy2=args.xy2,
        x2y2=args.x2y2,
        qmax=args.qmax,
        npoints=args.npoints,
        nstates=args.nstates,
        nroots=args.nroots,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        eta_points=args.eta_points,
        optimize_energy=args.optimize_energy,
        optimizer_xatol=args.optimizer_xatol,
    )
    save_outputs(result, args.output_dir, prefix=args.prefix)
    print(format_summary(result))
    print(f"\nSaved outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
