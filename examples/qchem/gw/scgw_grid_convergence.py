"""Frequency-grid convergence check for the imaginary-axis scGW prototype."""

from pyqed.gw.scgw import frequency_convergence
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF


mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    unit="angstrom",
)
mol.build(eri="dense")
mf = RHF(mol).run(verbose=0)

rows = frequency_convergence(
    mf,
    nfreq_values=(7, 9, 11),
    wmax=10.0,
    method="scgw0",
    grid="tangent",
    density_nfreq=129,
    run_kwargs={
        "max_cycle": 3,
        "damping": 0.3,
    },
)

print("method grid nfreq scale e_tot delta_e_tot delta_qp_max grid_ok nelec")
for row in rows:
    print(
        row["method"],
        row["grid"],
        row["nfreq"],
        f"{row['wmax']:.1f}",
        f"{row['e_tot']:.12f}",
        "None" if row["delta_e_tot"] is None else f"{row['delta_e_tot']:.3e}",
        "None" if row["delta_qp_max"] is None else f"{row['delta_qp_max']:.3e}",
        row["grid_converged"],
        f"{row['nelec']:.10f}",
    )
