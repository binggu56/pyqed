"""Compare PyQED OM2 smoke calculations with published OM2 benchmark targets.

This script does not replace an executable MNDO benchmark.  It records
published aggregate OM2 MAEs and selected supporting-information molecule
targets next to small PyQED calculations, so development has a reproducible
reference page until Program MNDO is available locally.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem.semiempirical import (
    OM2,
    format_published_om2_benchmarks,
    format_published_om2_molecule_benchmarks,
    published_om2_molecule_benchmarks,
)


def main():
    print("Published aggregate OM2 benchmark targets")
    print(format_published_om2_benchmarks())
    print()

    print("Selected published molecule-level OM2 targets")
    g2_sample = published_om2_molecule_benchmarks("G2-CHNOF")
    print(format_published_om2_molecule_benchmarks(g2_sample))
    print()

    print("PyQED OM2 smoke calculations")
    h2 = OM2(atom="H 0 0 0; H 0 0 0.74", unit="angstrom").run()
    h2_mrci = h2.MRCI(nstates=2, full=True).run()
    print(f"H2 OM2 E_tot = {h2.e_tot:.12f} Eh")
    print("H2 MRCI roots =", " ".join(f"{e:.12f}" for e in h2_mrci.e), "Eh")
    print()

    print("Note")
    print("Published molecule values are heats of formation/interaction energies, not total energies.")
    print("Use this as a benchmark target registry until an MNDO executable is available.")


if __name__ == "__main__":
    main()
