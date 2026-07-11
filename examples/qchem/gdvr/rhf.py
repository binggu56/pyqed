import time

import numpy as np

from pyqed.qchem.gdvr.rhf import Molecule


stime = time.time()

Nz = 63  # number of grids
M = 1  # one transverse orbital
LZ = 4  # grid boundary [-Lz, Lz]

charges = np.array([1.0, 1.0, 1.0, 1.0], float)
coords = np.array([
    [0.0, 0.0, -2.0999999999999996],
    [0.0, 0.0, -0.7],
    [0.0, 0.0, 2.0999999999999996],
    [0.0, 0.0, 0.7],
], float)

mol = Molecule(charges, coords, nelec=4)
print(f"nelec = {mol.nelec}")
print(f"Enuc  = {mol.nuclear_repulsion_energy():.10f} Eh")

NEWTON_TOL = 1e-7
SWEEP_ITERATIONS = 10
TRUST_STEP = 1.0
NEWTON_RIDGE = 0.5
TRUST_RADIUS = 2
VERBOSE = True
DVR_METHOD = "sine"

print(f"\n==== Sweep Strategy: Nz={Nz}, M={M}, Lz={LZ} ====")
mf = mol.build(
    Lz=LZ,
    Nz=Nz,
    M=M,
    transverse_basis="631g",
    max_offset=None,
    auto_cut=False,
    verbose=VERBOSE,
    dvr_method=DVR_METHOD,
).RHF().run(
    conv=1e-6,
    max_iter=100,
    verbose=VERBOSE,
)

print(f"[SCF 0] E = {mf.e_tot:.12f} Eh  (iters={mf.info['iter']})")

mf.newton(
    tol=NEWTON_TOL,
    sweep_iterations=SWEEP_ITERATIONS,
    ridge=NEWTON_RIDGE,
    trust_step=TRUST_STEP,
    trust_radius=TRUST_RADIUS,
    scf_conv=1e-7,
    scf_max_iter=100,
    verbose=VERBOSE,
)

print(f"\nFinal Energy: {mf.e_tot:.12f} Eh")
print(f"Total time: {time.time() - stime:.2f}s")
