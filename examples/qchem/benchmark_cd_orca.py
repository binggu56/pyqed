"""Benchmark pyqed CD spectra against an ORCA ECD calculation.

This is a qualitative cross-code benchmark, not a strict numerical regression:
the pyqed side uses CASCI/STO-3G while the ORCA side defaults to
TDA-B3LYP/def2-SVP/CPCM(water).  The useful check is that both codes produce
finite chiral rotatory strengths and comparable solvent-aware spectra.

Examples:
    python examples/qchem/benchmark_cd_orca.py --run-orca
    python examples/qchem/benchmark_cd_orca.py --run-orca --orca-method noiter-casscf
    python examples/qchem/benchmark_cd_orca.py --run-orca --orca-method noiter-casscf --orca-solvent water

If an ORCA run has already been completed, pass the generated property file:
    python examples/qchem/benchmark_cd_orca.py \
        --orca-property /path/to/methyl_lactate_tddft_cpcm.property.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import CASCI, CD, Molecule, RHF

EV = 27.211386245988

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


def build_molecule():
    mol = Molecule(atom=atom_string(), unit="angstrom", basis="sto-3g")
    mol.build(driver="pyscf")
    return mol


def _singlet_state_weights(mc, tol=1e-6):
    weights = np.zeros(len(mc.e_tot))
    singlet_indices = [
        idx for idx in range(len(mc.e_tot))
        if abs(float(mc.spin_square(idx))) <= tol
    ]
    if not singlet_indices:
        raise RuntimeError("No singlet roots found for pyqed singlet-state-averaged PCM.")
    weights[singlet_indices] = 1.0 / len(singlet_indices)
    return weights, singlet_indices


def run_pyqed(nstates=10, pcm_cycles=2, pcm_average="none"):
    mol = build_molecule()
    mf = RHF(mol).run()

    gas_mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=nstates)
    pcm_kwargs = {"max_cycle": pcm_cycles}
    if pcm_average == "all":
        pcm_kwargs["state_average"] = True
        pcm_label = "SA-PCM"
    elif pcm_average == "singlet":
        state_weights, singlet_indices = _singlet_state_weights(gas_mc)
        pcm_kwargs["state_weights"] = state_weights
        pcm_label = f"singlet-SA-PCM[{len(singlet_indices)}]"
    elif pcm_average == "none":
        pcm_label = "PCM"
    else:
        raise ValueError("pcm_average must be 'none', 'all', or 'singlet'.")

    pcm_mc = CASCI(mf, ncas=4, nelecas=4).PCM(**pcm_kwargs).run(nstates=nstates)
    lr_pcm_mc = CASCI(mf, ncas=4, nelecas=4).PCM(**pcm_kwargs).run(
        nstates=nstates,
        solvent_response="lr_pcm",
    )

    gas = CD(gas_mc).run()
    pcm = CD(pcm_mc).run(solvent_response="lr_pcm")
    lr_pcm = CD(lr_pcm_mc).run()
    return {
        "pyqed gas CASCI": (gas.excitation_energies * EV, gas.rotatory_strengths),
        f"pyqed {pcm_label} static CASCI": (pcm.excitation_energies * EV, pcm.rotatory_strengths),
        f"pyqed {pcm_label} LR-subspace CASCI": (
            pcm.solvent_response_energies * EV,
            pcm.solvent_response_rotatory_strengths,
        ),
        f"pyqed {pcm_label} LR-determinant CASCI": (
            lr_pcm.excitation_energies * EV,
            lr_pcm.rotatory_strengths,
        ),
    }


def run_pyscf_tddft_pcm(nstates=10, method="tdhf", xc="b3lyp", basis="sto-3g", lebedev_order=3):
    """Run PySCF TDHF/TDDFT+PCM ECD sticks for the same methyl lactate geometry."""
    try:
        from pyscf import dft, gto, scf, solvent  # noqa: F401 - imports PCM hooks
    except Exception as exc:
        raise RuntimeError("PySCF with solvent support is required for TDHF/TDDFT+PCM/CD.") from exc

    pmol = gto.M(atom=atom_string(), unit="Angstrom", basis=basis, verbose=0)
    method_key = str(method).lower()
    if method_key in {"tdhf", "hf", "rpa"}:
        mf = scf.RHF(pmol).PCM()
        label_method = "TDHF"
    elif method_key in {"tddft", "td-dft", "dft"}:
        mf = dft.RKS(pmol).PCM()
        mf.xc = xc
        label_method = f"TDDFT({xc})"
    else:
        raise ValueError("method must be 'tdhf' or 'tddft'.")

    mf.with_solvent.lebedev_order = int(lebedev_order)
    mf.with_solvent.verbose = 0
    mf.run(verbose=0)

    td = mf.TDHF(equilibrium_solvation=False) if method_key in {"tdhf", "hf", "rpa"} else mf.TDDFT(equilibrium_solvation=False)
    td.nstates = int(nstates)
    td.kernel()

    # Align signs and magnetic-dipole convention with pyqed CD.
    electric = -td.transition_dipole()
    magnetic = 0.5 * td.transition_magnetic_dipole()
    rotatory = -np.einsum("nx,nx->n", electric, magnetic)
    return {
        f"PySCF {label_method}+PCM/{basis}": (np.asarray(td.e) * EV, rotatory),
    }


def write_orca_tddft_input(path, nroots=10, functional="B3LYP", basis="def2-SVP", solvent="water"):
    lines = [
        f"! {functional} {basis} TightSCF CPCM({solvent})",
        "",
        "%tddft",
        f"  nroots {nroots}",
        "  maxdim 50",
        "end",
        "",
        "* xyz 0 1",
    ]
    lines.extend(f"{sym:2s} {x:10.6f} {y:10.6f} {z:10.6f}" for sym, x, y, z in METHYL_LACTATE_ATOMS)
    lines.append("*")
    path.write_text("\n".join(lines) + "\n")


def _solvent_keyword(solvent):
    return f" CPCM({solvent})" if solvent else ""


def write_orca_rhf_input(path, basis="STO-3G", solvent=None):
    lines = [
        f"! RHF {basis} TightSCF{_solvent_keyword(solvent)}",
        "",
        "* xyz 0 1",
    ]
    lines.extend(f"{sym:2s} {x:10.6f} {y:10.6f} {z:10.6f}" for sym, x, y, z in METHYL_LACTATE_ATOMS)
    lines.append("*")
    path.write_text("\n".join(lines) + "\n")


def write_orca_noiter_casscf_input(path, moinp, nroots=10, basis="STO-3G", solvent=None):
    lines = [
        f"! RHF {basis} TightSCF Moread NoIter{_solvent_keyword(solvent)}",
        f'%moinp "{moinp.name}"',
        "",
        "%casscf",
        "  nel 4",
        "  norb 4",
        "  mult 1",
        f"  nroots {nroots}",
        "  DoCD true",
        "  DoDipoleLength true",
        "  DoDipoleVelocity true",
        "end",
        "",
        "* xyz 0 1",
    ]
    lines.extend(f"{sym:2s} {x:10.6f} {y:10.6f} {z:10.6f}" for sym, x, y, z in METHYL_LACTATE_ATOMS)
    lines.append("*")
    path.write_text("\n".join(lines) + "\n")


def find_orca(explicit=None):
    candidates = [
        explicit,
        shutil.which("orca"),
        "/Users/gugroup/Library/orca_6_1_1/orca",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError("Could not find ORCA. Pass --orca /path/to/orca.")


def _run_orca_input(exe, input_path):
    stdout_path = input_path.with_suffix(".out")
    with stdout_path.open("w") as handle:
        subprocess.run(
            [str(exe), input_path.name],
            cwd=input_path.parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return input_path.with_name(f"{input_path.stem}.property.txt")


def run_orca_tddft(outdir, nroots=10, orca=None):
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = outdir / "methyl_lactate_tddft_cpcm.inp"
    write_orca_tddft_input(input_path, nroots=nroots)
    return _run_orca_input(find_orca(orca), input_path)


def run_orca_noiter_casscf(outdir, nroots=10, orca=None, solvent=None):
    outdir.mkdir(parents=True, exist_ok=True)
    exe = find_orca(orca)

    suffix = f"_cpcm_{solvent}" if solvent else ""
    rhf_input = outdir / f"methyl_lactate_rhf_sto3g{suffix}.inp"
    write_orca_rhf_input(rhf_input, solvent=solvent)
    _run_orca_input(exe, rhf_input)

    casscf_input = outdir / f"methyl_lactate_noiter_casscf_cd{suffix}.inp"
    write_orca_noiter_casscf_input(
        casscf_input,
        rhf_input.with_suffix(".gbw"),
        nroots=nroots,
        solvent=solvent,
    )
    return _run_orca_input(exe, casscf_input)


def run_orca(outdir, nroots=10, orca=None, method="tddft", solvent=None):
    if method == "tddft":
        return run_orca_tddft(outdir, nroots=nroots, orca=orca)
    if method == "noiter-casscf":
        return run_orca_noiter_casscf(outdir, nroots=nroots, orca=orca, solvent=solvent)
    raise ValueError("method must be 'tddft' or 'noiter-casscf'.")


def _block_has_representation(block, representation):
    needle = f'"{representation.lower()}"'
    return any("Representation" in line and needle in line.lower() for line in block)


def parse_orca_ecd_property(path, representation="Velocity"):
    """Return ORCA ECD energies/eV and R/(1e40*cgs) from property.txt."""
    text = Path(path).read_text().splitlines()
    blocks = []
    i = 0
    while i < len(text):
        if not text[i].strip().endswith("_ECD_Spectrum"):
            i += 1
            continue
        j = i + 1
        while j < len(text) and text[j].strip() != "$End":
            j += 1
        blocks.append(text[i:j])
        i = j + 1

    for block in blocks:
        if not _block_has_representation(block, representation):
            continue
        ntrans = None
        for line in block:
            if "&NTrans" in line:
                ntrans = int(line.split()[-1])
                break
        if ntrans is None:
            raise ValueError(f"Could not find &NTrans in {path}.")

        start = next(i for i, line in enumerate(block) if "&ExcitationEnergies" in line)
        rows = []
        for line in block[start + 1 :]:
            parts = line.split()
            has_float_payload = len(parts) >= 5 and ("." in parts[1] or "e" in parts[1].lower())
            if parts and parts[0].isdigit() and has_float_payload:
                rows.append((float(parts[1]), float(parts[4])))
                if len(rows) == ntrans:
                    break
        if len(rows) != ntrans:
            raise ValueError(f"Found {len(rows)} ECD rows, expected {ntrans}.")
        return tuple(np.array(col, dtype=float) for col in zip(*rows))

    raise ValueError(f"No ECD spectrum block with Representation={representation!r}.")


def infer_orca_ecd_label(path, default_method, solvent=None):
    solvent_label = f" CPCM({solvent})" if solvent else ""
    text = Path(path).read_text()
    if "$CASSCF_ECD_Spectrum" in text:
        return f"ORCA NoIter CASSCF{solvent_label}"
    if "$CIS_ECD_Spectrum" in text:
        return "ORCA TDA-B3LYP CPCM" if default_method == "tddft" else "ORCA CIS/TDDFT"
    return "ORCA"


def print_table(datasets):
    labels = list(datasets)
    nrows = max(len(values[0]) for values in datasets.values())
    print("state " + " ".join(f"{label:>30s}" for label in labels))
    print("      " + " ".join(f"{'E/eV, R':>30s}" for _ in labels))
    for idx in range(nrows):
        cells = []
        for label in labels:
            energies, rotatory = datasets[label]
            if idx < len(energies):
                cells.append(f"{energies[idx]:9.4f}, {rotatory[idx]:10.4g}")
            else:
                cells.append("")
        print(f"{idx + 1:5d} " + " ".join(f"{cell:>30s}" for cell in cells))


def broaden_spectrum(energies, rotatory, x, width):
    signal = np.zeros_like(x, dtype=float)
    norm = 1.0 / (width * np.sqrt(2.0 * np.pi))
    for energy, strength in zip(energies, rotatory):
        signal += strength * norm * np.exp(-0.5 * ((x - energy) / width) ** 2)
    scale = np.max(np.abs(signal))
    if scale > 0.0:
        signal = signal / scale
    return signal


def plot_table(datasets, path, width=0.18):
    import matplotlib.pyplot as plt

    all_energies = np.concatenate([values[0] for values in datasets.values()])
    lower = max(0.0, float(np.min(all_energies)) - 0.8)
    upper = float(np.max(all_energies)) + 0.8
    x = np.linspace(lower, upper, 1600)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    for label, (energies, rotatory) in datasets.items():
        ax.plot(x, broaden_spectrum(energies, rotatory, x, width), label=label)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Normalized CD intensity")
    ax.set_title("Methyl lactate CD benchmark")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nstates", type=int, default=10)
    parser.add_argument("--pcm-cycles", type=int, default=20)
    parser.add_argument("--run-orca", action="store_true")
    parser.add_argument("--orca-method", choices=["tddft", "noiter-casscf"], default="tddft")
    parser.add_argument(
        "--orca-solvent",
        help="Add CPCM(solvent) to ORCA NoIter CASSCF benchmark, e.g. water.",
    )
    parser.add_argument("--orca")
    parser.add_argument("--orca-property", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/orca_cd_benchmark"))
    parser.add_argument("--representation", choices=["Length", "Velocity"], default="Velocity")
    parser.add_argument("--skip-pyqed", action="store_true")
    parser.add_argument(
        "--skip-tddft-pcm",
        action="store_true",
        help="Do not add the PySCF TDDFT+PCM/CD overlay to the comparison table/plot.",
    )
    parser.add_argument(
        "--tddft-method",
        choices=["tdhf", "tddft"],
        default="tdhf",
        help=(
            "Method for the solvent response overlay. Default TDHF matches the "
            "RHF/STO-3G CASCI reference more closely; use tddft for a B3LYP-style comparison."
        ),
    )
    parser.add_argument("--tddft-xc", default="b3lyp", help="XC functional when --tddft-method=tddft.")
    parser.add_argument("--tddft-basis", default="sto-3g", help="Basis for the TDHF/TDDFT+PCM overlay.")
    parser.add_argument(
        "--tddft-pcm-lebedev-order",
        type=int,
        default=3,
        help="Lebedev order for the PySCF TDDFT+PCM overlay. Small default keeps the plot fast.",
    )
    parser.add_argument(
        "--pyqed-pcm-state-average",
        action="store_true",
        help="Use equal-weight average over all pyqed determinant roots for the PCM density.",
    )
    parser.add_argument(
        "--pyqed-pcm-average",
        choices=["none", "all", "singlet"],
        default=None,
        help=(
            "PCM density averaging mode for pyqed. 'singlet' averages only roots "
            "with S^2 close to 0, useful for ORCA singlet-CASSCF comparisons."
        ),
    )
    parser.add_argument("--plot", type=Path, help="Write a normalized comparison plot.")
    parser.add_argument("--width", type=float, default=0.18, help="Gaussian broadening width in eV.")
    args = parser.parse_args()

    datasets = {}
    if not args.skip_pyqed:
        pcm_average = args.pyqed_pcm_average
        if pcm_average is None:
            pcm_average = "all" if args.pyqed_pcm_state_average else "none"
        datasets.update(
            run_pyqed(
                nstates=args.nstates,
                pcm_cycles=args.pcm_cycles,
                pcm_average=pcm_average,
            )
        )
    if not args.skip_tddft_pcm:
        datasets.update(
            run_pyscf_tddft_pcm(
                nstates=args.nstates,
                method=args.tddft_method,
                xc=args.tddft_xc,
                basis=args.tddft_basis,
                lebedev_order=args.tddft_pcm_lebedev_order,
            )
        )

    property_path = args.orca_property
    if args.run_orca:
        property_path = run_orca(
            args.outdir,
            nroots=args.nstates,
            orca=args.orca,
            method=args.orca_method,
            solvent=args.orca_solvent,
        )
    if property_path is not None:
        energies, rotatory = parse_orca_ecd_property(property_path, representation=args.representation)
        label = infer_orca_ecd_label(property_path, args.orca_method, solvent=args.orca_solvent)
        datasets[f"{label} {args.representation}"] = (energies, rotatory)

    print_table(datasets)
    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_table(datasets, args.plot, width=args.width)
        print(f"\nWrote normalized CD comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
