import { links } from "../site-data";

const h2Rhf = `from pyqed.qchem import Molecule

mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    unit="angstrom",
    basis="sto-3g",
)
mol.build(driver="builtin", eri="auto")

mf = mol.RHF().run()
if not mf.converged:
    raise RuntimeError("The quickstart RHF calculation did not converge.")

print(f"RHF energy: {mf.e_tot:.12f} Eh")`;

const sineDvrOscillator = `import numpy as np
from pyqed.dvr import SineDVR

dvr = SineDVR(-8.0, 8.0, 80)
hamiltonian = dvr.t() + np.diag(0.5 * dvr.x**2)
energies = np.linalg.eigvalsh(hamiltonian)[:4]
print(np.array2string(energies, precision=8))`;

const heomSpinBoson = `import numpy as np
from pyqed import pauli
from pyqed.oqs import HEOMSolver

_, sx, _, sz = pauli()
H, rho0 = -0.5 * (sx + sz), np.diag([0.0, 1.0])
rho = HEOMSolver(H, c_ops=[sz], e_ops=[sz]).run(
    rho0, dt=0.02, nt=100, temperature=600,
    cutoff=5, reorganization=0.2, nado=5,
)
print(f"Final <sigma_z>: {rho[0, -1].real:.8f}")`;

const shinMetiu = `#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed import proton_mass as mp
from pyqed.models.ShinMetiu import ShinMetiu2
from pyqed.namd import Ehrenfest


OUT = Path("examples/namd/ehrenfest_histories.png")


def main():
    mol = ShinMetiu2()
    mol.build(domain=[[-10, 10]] * 2, npts=[31, 31])

    ed = Ehrenfest(ndim=mol.ndim, ntraj=1, nstates=mol.nstates, mass=[mp] * 2)
    ed.nac_driver = mol.nonadiabatic_coupling
    ed.sample(init_state=2, x0=[0.0, 1.3], ax=18.0)
    ed.run(dt=0.5, nt=400, nout=2)

    populations = np.real(np.diagonal(ed.rho_history, axis1=1, axis2=2))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for dim in range(ed.x_history.shape[1]):
        axes[0].plot(ed.times, ed.x_history[:, dim], label=f"x[{dim}]")
    axes[0].set_ylabel("Position (bohr)")
    axes[0].legend(loc="best")

    for state in range(populations.shape[1]):
        axes[1].plot(ed.times, populations[:, state], label=f"pop[{state}]")
    axes[1].set_ylabel("Population")
    axes[1].legend(loc="best")

    axes[2].plot(ed.times, ed.energy_history, label="Ehrenfest energy")
    axes[2].plot(ed.times, ed.norm_history, label="Electronic norm")
    axes[2].set_xlabel("Time (a.u.)")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()`;

export const examples = [
  {
    id: "h2-rhf",
    index: "01",
    track: "Electronic structure",
    title: "H₂ in one native RHF calculation",
    summary:
      "Build a molecule, select PyQED’s built-in integral path, and require a converged restricted Hartree–Fock result.",
    fileName: "examples/quickstart.py",
    code: h2Rhf,
    runCommand: "PYTHONPATH=. python examples/quickstart.py",
    prerequisites: "PyQED 0.2.0 core install",
    runtime: "Usually seconds on a laptop CPU",
    expected: "RHF energy: -1.116759310293 Eh",
    expectedNote: "STO-3G · 0.74 Å bond length",
    sourceHref: links.quickstartReleaseSource,
    guideHref: links.quickstart,
    guideLabel: "Quickstart guide",
  },
  {
    id: "sine-dvr-oscillator",
    index: "02",
    track: "Grid dynamics",
    title: "A harmonic oscillator with Sine DVR",
    summary:
      "Build the dense Sine DVR Hamiltonian in a few lines and recover the analytic oscillator ladder directly from its eigenvalues.",
    fileName: "sine_dvr_harmonic.py",
    code: sineDvrOscillator,
    runCommand:
      "PYTHONPATH=. python sine_dvr_harmonic.py",
    prerequisites: "PyQED 0.2.0 core install · NumPy",
    runtime: "Usually seconds on a laptop CPU",
    expected: "[0.5 1.5 2.5 3.5]",
    expectedNote: "First four harmonic-oscillator levels",
    sourceHref: links.sineDvrReleaseSource,
    guideHref: links.dvrGuide,
    guideLabel: "DVR guide",
  },
  {
    id: "heom-spin-boson",
    index: "03",
    track: "Open quantum systems",
    title: "Spin–boson dynamics with HEOM",
    summary:
      "Define a two-level Hamiltonian, couple it to a Drude bath, and propagate a non-Markovian population observable in one compact calculation.",
    fileName: "heom_spin_boson.py",
    code: heomSpinBoson,
    runCommand: "PYTHONPATH=. python heom_spin_boson.py",
    prerequisites: "PyQED 0.2.0 core install · NumPy · SciPy",
    runtime: "Usually seconds on a laptop CPU",
    expected: "Final <sigma_z>: -0.96907844",
    expectedNote: "100 steps · five-tier hierarchy",
    sourceHref: links.heomReleaseSource,
    guideHref: links.openDynamicsGuide,
    guideLabel: "Open-dynamics guide",
  },
  {
    id: "shin-metiu-ehrenfest",
    index: "04",
    track: "Nonadiabatic dynamics",
    title: "Shin–Metiu histories with Ehrenfest dynamics",
    summary:
      "Propagate one two-dimensional trajectory, retain position, population, energy, and norm histories, then render a diagnostic figure.",
    fileName: "examples/namd/ehrenfest_histories.py",
    code: shinMetiu,
    runCommand:
      "PYTHONPATH=. python examples/namd/ehrenfest_histories.py",
    prerequisites: "PyQED 0.2.0 core install · Matplotlib",
    runtime: "Allow about a minute · 400 steps on a 31×31 grid",
    expected: "Saved examples/namd/ehrenfest_histories.png",
    expectedNote: "Three panels: coordinates, populations, energy and norm",
    sourceHref: links.ehrenfestHistoriesReleaseSource,
    guideHref: `${links.tutorials}#nonadiabatic-and-geometric-dynamics`,
    guideLabel: "Dynamics learning path",
  },
] as const;
