"""Minimal restartable native SU(2)-LETTA calculation."""

from pyqed import Molecule
from pyqed.qchem.letta import LETTA


mol = Molecule(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
mol.build()
mf = mol.RHF().run(verbose=0)

state = LETTA(
    mf,
    symmetry="su2",
    D=1,  # reduced multiplets per virtual sector
    seed=4,
)
state.run(
    nsweeps=4,
    algorithm="one_site",
    tol=1.0e-9,
    residual_tol=1.0e-8,
    truncation_tol=1.0e-7,
    consecutive_cycles=2,
    checkpoint="su2_letta.chk",
    verbose=1,
)

print("energy =", state.energy)
print("target =", state.target_sector)
print("native SU(2) =", state.is_native_su2)
print("convergence =", state.convergence_summary)

# Continue later without discarding the completed-cycle history:
# state = type(state).load_checkpoint("su2_letta.chk", workers=2)
# state.run(nsweeps=2, algorithm="one_site", reset_history=False)
