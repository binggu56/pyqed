import numpy as np
import matplotlib.pyplot as plt

from pyqed.qchem import Molecule, RTTDHF
from pyqed.qchem.rttdhf import gaussian_pulse

mol = Molecule(atom=[
    ["H", (0., 0., .917)],
    ["H", (0., 0., 0.)],
], basis="631g")

mol.build(aosym='s8')
mf = mol.RHF().run()

pulse = gaussian_pulse(
    amplitude=0.05,
    center=8.0,
    width=2.0,
    omega=0.7,
    polarization=(0, 0, 1),
)

rt = RTTDHF(mf, field=pulse).run(
    dt=0.02,
    nsteps=900,
    store_dm=True,
)

ana = mf.analyze()
frames = [0, 250, 400, 550, 750]

for it in frames:
    ana.plot_density(
        dm=rt.dms[it].real,
        nx=45,
        margin=3.0,
        title=f"electron density, t = {rt.times[it]:.1f} a.u.",
    )


### wavefunction overlap 

from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix


def occupied_orbitals_from_dm(rt, dm):
    # number of occupied spatial orbitals for closed-shell RT-TDHF
    nocc = int(round(rt.electron_count(dm) / 2))

    # AO density -> Lowdin orthogonal AO density
    p = rt.ao_to_orth(dm)
    p = 0.5 * (p + p.conj().T)

    # Natural orbitals in orthogonal AO basis
    occ, u = np.linalg.eigh(p)
    idx = np.argsort(occ)[::-1]
    u = u[:, idx]
    occ = occ[idx]

    # Back-transform to AO coefficients
    x, _ = rt._build_orthogonalizer()
    c_occ = x @ u[:, :nocc]

    return c_occ, occ


def overlap(rt1, it1, rt2, it2):
    dm1 = rt1.dms[it1]
    dm2 = rt2.dms[it2]

    c1, occ1 = occupied_orbitals_from_dm(rt1, dm1)
    c2, occ2 = occupied_orbitals_from_dm(rt2, dm2)

    # Cross-AO overlap between different geometries/bases
    s12 = _cross_ao_overlap_matrix(rt1.mol, rt2.mol)

    # Spatial occupied-orbital overlap
    m = c1.conj().T @ s12 @ c2

    # Closed-shell determinant overlap = alpha overlap * beta overlap
    ov = np.linalg.det(m) ** 2

    return ov
