"""CASCI circular dichroism of methyl lactate with PCM.

This example uses the native PyQED path throughout:

    python examples/qchem/cd_pcm_methyl_lactate.py

The default calculation is production-oriented: RHF/CASCI(4e,4o)/6-31G with
10 roots and a converged static-PCM macroiteration limit.  For a cheap smoke
test, pass ``--quick``.  The plotted CD spectrum uses the LR-PCM subspace
correction as the main solvent spectrum.  Static PCM and determinant-space
LR-PCM are included as comparisons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.units import au2ev
from pyqed.qchem import CASCI, CD, Molecule, RHF

EV = au2ev

METHYL_LACTATE_ATOMS = (
    ("C", 0.000, 0.000, 0.000),
    ("H", 0.620, 0.620, 0.620),
    ("O", -0.950, 0.450, 0.850),
    ("H", -1.500, 1.000, 0.350),
    ("C", -0.500, -1.420, 0.200),
    ("H", -1.100, -1.680, -0.670),
    ("H", 0.350, -2.100, 0.270),
    ("H", -1.120, -1.550, 1.090),
    ("C", 1.180, 0.140, -0.980),
    ("O", 1.300, -0.180, -2.160),
    ("O", 2.120, 0.720, -0.250),
    ("C", 3.350, 0.910, -0.930),
    ("H", 3.250, 1.250, -1.960),
    ("H", 3.930, -0.010, -0.900),
    ("H", 3.890, 1.670, -0.370),
)


def atom_string():
    return "; ".join(f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, x, y, z in METHYL_LACTATE_ATOMS)


def build_molecule(basis):
    mol = Molecule(atom=atom_string(), unit="angstrom", basis=basis)
    mol.build(eri="s8")
    return mol


def run_cd_pcm(basis="6-31g", nstates=10, pcm_cycles=20, pcm_conv_tol=1e-7):
    mol = build_molecule(basis)
    mf = RHF(mol).run(verbose=0)

    gas_mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=nstates)
    pcm_kwargs = {"max_cycle": pcm_cycles, "conv_tol": pcm_conv_tol}
    pcm_mc = CASCI(mf, ncas=4, nelecas=4).PCM(**pcm_kwargs).run(nstates=nstates)
    det_lr_mc = CASCI(mf, ncas=4, nelecas=4).PCM(**pcm_kwargs).run(
        nstates=nstates,
        solvent_response="lr_pcm",
    )

    gas_cd = CD(gas_mc)
    pcm_cd = CD(pcm_mc)
    det_cd = CD(det_lr_mc)

    gas = gas_cd.run()
    pcm = pcm_cd.run(solvent_response="lr_pcm")
    det_lr = det_cd.run()

    return {
        "mol": mol,
        "pcm_cycles": pcm_cycles,
        "pcm_conv_tol": pcm_conv_tol,
        "gas_cd": gas_cd,
        "pcm_cd": pcm_cd,
        "det_cd": det_cd,
        "gas": gas,
        "pcm": pcm,
        "det_lr": det_lr,
    }


def broaden(x, energies, strengths, width=0.18):
    y = np.zeros_like(x)
    pref = width * np.sqrt(2.0 * np.pi)
    for energy, strength in zip(energies, strengths):
        y += strength * np.exp(-0.5 * ((x - energy) / width) ** 2) / pref
    return y


def plot_cd(results, output, width=0.18):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gas = results["gas"]
    pcm = results["pcm"]
    det_lr = results["det_lr"]

    gas_e = gas.excitation_energies * EV
    pcm_e = pcm.excitation_energies * EV
    lr_e = pcm.solvent_response_energies * EV
    det_e = det_lr.excitation_energies * EV

    xmin = max(0.0, min(gas_e.min(), pcm_e.min(), lr_e.min(), det_e.min()) - 0.9)
    xmax = max(gas_e.max(), pcm_e.max(), lr_e.max(), det_e.max()) + 0.7
    x = np.linspace(xmin, xmax, 1800)

    curves = {
        "gas": broaden(x, gas_e, gas.rotatory_strengths, width),
        "static": broaden(x, pcm_e, pcm.rotatory_strengths, width),
        "lr": broaden(x, lr_e, pcm.solvent_response_rotatory_strengths, width),
        "det": broaden(x, det_e, det_lr.rotatory_strengths, width),
    }
    scale = max(np.max(np.abs(y)) for y in curves.values()) or 1.0
    curves = {key: val / scale for key, val in curves.items()}

    fig = plt.figure(figsize=(10.4, 7.0), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.95, 0.72], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    stick_ax = fig.add_subplot(gs[1], sharex=ax)
    shift_ax = fig.add_subplot(gs[2], sharex=ax)
    colors = {"gas": "#486d92", "static": "#c96f2d", "lr": "#16836d", "det": "#8b6f9f"}

    for axis in (ax, stick_ax, shift_ax):
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e3e3e3", linewidth=0.75, alpha=0.9)

    ax.axhline(0.0, color="#666666", lw=0.9)
    ax.fill_between(x, 0.0, curves["lr"], color=colors["lr"], alpha=0.10, linewidth=0)
    ax.plot(x, curves["lr"], color=colors["lr"], lw=3.0, label="LR-PCM subspace")
    ax.plot(x, curves["static"], color=colors["static"], lw=2.0, ls=(0, (5, 3)), label="Static PCM")
    ax.plot(x, curves["gas"], color=colors["gas"], lw=1.9, alpha=0.85, label="Gas phase")
    ax.plot(x, curves["det"], color=colors["det"], lw=1.6, ls=":", alpha=0.75, label="LR-PCM determinant check")
    ax.set_ylabel("CD intensity (normalized)")
    ax.set_title("Methyl lactate CD in PCM", loc="left", pad=14, fontsize=16, fontweight="bold")
    ax.text(
        0.0,
        1.02,
        (
            f"CASCI(4e,4o)/{results['mol'].basis}, RHF reference, "
            f"PCM max_cycle={results['pcm_cycles']}, conv_tol={results['pcm_conv_tol']:.0e}"
        ),
        transform=ax.transAxes,
        fontsize=10.5,
        color="#555555",
        va="bottom",
    )
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=9.7, handlelength=3.1)

    stick_ax.axhline(0.0, color="#666666", lw=0.8)
    for energies, strengths, color, offset in [
        (gas_e, gas.rotatory_strengths, colors["gas"], -0.045),
        (pcm_e, pcm.rotatory_strengths, colors["static"], -0.015),
        (lr_e, pcm.solvent_response_rotatory_strengths, colors["lr"], 0.015),
        (det_e, det_lr.rotatory_strengths, colors["det"], 0.045),
    ]:
        stick_ax.vlines(energies + offset, 0.0, strengths, color=color, lw=1.75, alpha=0.85)
    stick_ax.set_ylabel("R sticks")
    stick_ax.tick_params(labelbottom=False)

    shift = lr_e - gas_e
    shift_ax.axhline(0.0, color="#666666", lw=0.85)
    shift_ax.bar(lr_e, shift, width=0.075, color=colors["lr"], alpha=0.82)
    shift_ax.set_ylabel("Shift (eV)")
    shift_ax.set_xlabel("Excitation energy (eV)")
    shift_ax.set_xlim(xmin, xmax)
    shift_ax.set_ylim(min(-0.03, shift.min() - 0.02), shift.max() + 0.04)

    fig.align_ylabels([ax, stick_ax, shift_ax])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(results, output):
    gas = results["gas"]
    pcm = results["pcm"]
    det_lr = results["det_lr"]
    mol = results["mol"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        fh.write("Methyl lactate CASCI/CD/PCM example\n")
        fh.write(f"basis {mol.basis}\n")
        fh.write(f"nao {mol.nao}\n")
        fh.write("build driver builtin eri s8\n")
        fh.write(
            "RHF/CASCI(4e,4o), native PCM integrals, "
            f"PCM max_cycle={results['pcm_cycles']}, conv_tol={results['pcm_conv_tol']:.3e}\n"
        )
        for label, result in [("gas", gas), ("pcm", pcm), ("det_lr", det_lr)]:
            fh.write(
                f"{label} excitation energies/eV: "
                + " ".join(f"{x:.8f}" for x in result.excitation_energies * EV)
                + "\n"
            )
            fh.write(
                f"{label} rotatory strengths: "
                + " ".join(f"{x:.8e}" for x in result.rotatory_strengths)
                + "\n"
            )
        fh.write(
            "pcm lr-subspace excitation energies/eV: "
            + " ".join(f"{x:.8f}" for x in pcm.solvent_response_energies * EV)
            + "\n"
        )
        fh.write(
            "pcm lr-subspace rotatory strengths: "
            + " ".join(f"{x:.8e}" for x in pcm.solvent_response_rotatory_strengths)
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--nstates", type=int, default=10)
    parser.add_argument("--pcm-cycles", type=int, default=20)
    parser.add_argument("--pcm-conv-tol", type=float, default=1e-7)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a cheap demo setting: --nstates 3 --pcm-cycles 2 unless explicitly overridden.",
    )
    parser.add_argument("--width", type=float, default=0.18, help="Gaussian broadening width in eV.")
    parser.add_argument(
        "--output",
        default="examples/qchem/methyl_lactate_cd_pcm_631g.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--summary",
        default="examples/qchem/methyl_lactate_cd_pcm_631g.txt",
        help="Output text summary path.",
    )
    args = parser.parse_args()

    if args.quick:
        if args.nstates == parser.get_default("nstates"):
            args.nstates = 3
        if args.pcm_cycles == parser.get_default("pcm_cycles"):
            args.pcm_cycles = 2

    results = run_cd_pcm(args.basis, args.nstates, args.pcm_cycles, args.pcm_conv_tol)
    plot_cd(results, args.output, width=args.width)
    write_summary(results, args.summary)

    pcm = results["pcm"]
    print(f"Wrote figure: {args.output}")
    print(f"Wrote summary: {args.summary}")
    print("LR-PCM subspace excitation energies/eV:")
    print(" ".join(f"{x:.4f}" for x in pcm.solvent_response_energies * EV))


if __name__ == "__main__":
    main()
