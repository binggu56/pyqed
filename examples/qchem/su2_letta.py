"""Minimal restartable native SU(2)-LETTA calculation."""

import numpy as np

from pyqed.narg.qchem import LETTA


h1e = np.array(
    [
        [-1.0, -0.2],
        [-0.2, 0.5],
    ]
)

state = LETTA.from_integrals(
    h1e,
    symmetry="su2",
    nelec=2,
    spin=0,  # doubled spin: spin=0 selects a singlet
    graph=[(0, 1)],
    D=1,  # reduced multiplets per virtual sector
    seed=4,
)
state.run(
    nsweeps=4,
    algorithm="two_site",
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
# state.run(nsweeps=2, algorithm="two_site", reset_history=False)
