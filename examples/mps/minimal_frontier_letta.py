"""Minimal U(1) frontier-LETTA calculation for a four-spin chain."""

from pyqed.lattice import SpinHalfSite
from pyqed.letta import FrontierLETTA
from pyqed.tn import Hamiltonian


sites = [SpinHalfSite()] * 4
H = Hamiltonian(sites)

# H = sum_i S_i . S_{i+1}
for i in range(len(sites) - 1):
    for operator in ("Sx", "Sy", "Sz"):
        H.add_product(1.0, (i, operator), (i + 1, operator))

# An edge (i, j), i < j, makes A^i carry the future physical index s_j.
graph = [(0, 2), (1, 3)]
state = FrontierLETTA(
    H,
    graph=graph,
    target_charge={"Sz": 0},
    D=4,
    adaptive_bond=True,
    seed=7,
)
state.run(nsweeps=4, enrich="amen", enrich_rank=2)

print(f"energy = {state.energy:.12f}")
print(f"ties = {state.graph}")

# Save a vector diagram when needed:
# state.draw("frontier_letta.pdf")
