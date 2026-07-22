import sys
from pathlib import Path

import numpy as np


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))


def _second_derivative_kinetic(n, dx, mass=1.0):
    kinetic = np.diag(np.full(n, 1.0 / (mass * dx * dx)))
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=1)
    kinetic += np.diag(np.full(n - 1, -0.5 / (mass * dx * dx)), k=-1)
    return kinetic


def _h2_rhf(r):
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf import RHF

    mol = Molecule(
        atom=f"H 0 0 {-0.5 * r}; H 0 0 {0.5 * r}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    return RHF(mol).run()


def test_rttdhf_frame_overlap_is_normalized_for_same_frame():
    _prefer_source_package()
    from pyqed.namd.tdldr.rttdhf import Frame, det_overlap

    frame = Frame(_h2_rhf(1.4))
    np.testing.assert_allclose(det_overlap(frame, frame), 1.0, atol=1.0e-10)


def test_rttdhf_frame_density_matches_existing_rttdhf_step():
    _prefer_source_package()
    from pyqed.namd.tdldr.rttdhf import Frame
    from pyqed.qchem import RTTDHF

    mf = _h2_rhf(1.4)
    frame = Frame(mf)
    rt = RTTDHF(mf)

    frame.step(time=0.0, dt=0.03)
    dm_rt = rt.step(mf.dm, time=0.0, dt=0.03)

    np.testing.assert_allclose(frame.density(), dm_rt, atol=1.0e-10)


def test_rttdhf_tdldr_runs_with_real_time_dependent_determinants():
    _prefer_source_package()
    from pyqed.namd.tdldr.rttdhf import Frame, Solver

    grid = np.array([1.3, 1.5])
    kinetic = _second_derivative_kinetic(grid.size, grid[1] - grid[0], mass=918.0)

    def field(time):
        return np.array([0.0, 0.0, 0.02 * np.sin(0.1 * time)])

    frames = [Frame(_h2_rhf(r), field=field) for r in grid]
    solver = Solver(grid, kinetic, frames)
    c0 = np.array([1.0, 0.0], dtype=complex)
    traj = solver.run(c0, dt=0.02, nsteps=3, store_hamiltonians=True)

    assert traj.coefficients.shape == (4, 2)
    assert traj.overlaps.shape == (4, 2, 2)
    assert traj.kinetic_hamiltonians.shape == (4, 2, 2)
    np.testing.assert_allclose(traj.norm, np.ones(4), atol=1.0e-12)
    np.testing.assert_allclose(traj.electron_counts, 2.0, atol=1.0e-10)
