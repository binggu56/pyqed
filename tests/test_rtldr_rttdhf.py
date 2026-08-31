import sys
from pathlib import Path

import numpy as np


class _TestNuclear:
    def __init__(self, points, kinetic):
        self.points = np.asarray(points)
        self._kinetic = np.asarray(kinetic)

    def kinetic(self):
        return self._kinetic


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
    mol.build()
    return RHF(mol).run()


def test_packed_jk_preserves_complex_time_dependent_density():
    from pyqed.qchem.basis import contract_jk_s8, pack_eri_s8

    rng = np.random.default_rng(7)
    factors = rng.normal(size=(4, 3, 3))
    factors = 0.5 * (factors + factors.swapaxes(1, 2))
    eri = np.einsum("pij,pkl->ijkl", factors, factors)
    trial = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    dm = 0.5 * (trial + trial.conj().T)

    vj, vk = contract_jk_s8(pack_eri_s8(eri), dm, 3)

    np.testing.assert_allclose(vj, np.einsum("lk,ijkl->ij", dm, eri), atol=1.0e-12)
    np.testing.assert_allclose(vk, np.einsum("lk,ilkj->ij", dm, eri), atol=1.0e-12)


def test_h4_collective_coordinates_move_only_inner_atoms():
    from examples.qchem.h4_rtldr_hhg import (
        NPOINTS,
        atomic_positions,
        collective_dvr,
    )

    reference = atomic_positions([0.0, 0.0])
    q = np.array([0.12, -0.08])
    displaced = atomic_positions(q)
    u2, u3 = displaced[1:3] - reference[1:3]

    np.testing.assert_allclose(displaced[[0, 3]], reference[[0, 3]])
    np.testing.assert_allclose((u2 + u3) / np.sqrt(2.0), q[0])
    np.testing.assert_allclose((u3 - u2) / np.sqrt(2.0), q[1])

    nuclear = collective_dvr()
    kinetic = nuclear.kinetic()
    assert nuclear.points.shape == (NPOINTS**2, 2)
    assert kinetic.shape == (NPOINTS**2, NPOINTS**2)
    np.testing.assert_allclose(kinetic.toarray(), kinetic.T.toarray(), atol=1.0e-12)


def test_rttdhf_frame_overlap_is_normalized_for_same_frame():
    _prefer_source_package()
    from pyqed.namd.rtldr import RTTDHFFrame, det_overlap

    frame = RTTDHFFrame(_h2_rhf(1.4))
    np.testing.assert_allclose(det_overlap(frame, frame), 1.0, atol=1.0e-10)


def test_rttdhf_frame_density_matches_existing_rttdhf_step():
    _prefer_source_package()
    from pyqed.namd.rtldr import RTTDHFFrame
    from pyqed.qchem import RTTDHF

    mf = _h2_rhf(1.4)
    frame = RTTDHFFrame(mf)
    rt = RTTDHF(mf)

    frame.propagate(time=0.0, dt=0.03)
    dm_rt = rt.step(mf.dm, time=0.0, dt=0.03)

    np.testing.assert_allclose(frame.density(), dm_rt, atol=1.0e-10)


def test_rttdhf_frame_overlap_carries_many_electron_action_phase():
    _prefer_source_package()
    from pyqed.namd.rtldr import RTTDHFFrame, det_overlap

    frame = RTTDHFFrame(_h2_rhf(1.4))
    before = frame.copy()
    energy = frame.phase_energy(time=0.0)
    dt = 1.0e-3

    frame.propagate(time=0.0, dt=dt)

    overlap = det_overlap(before, frame)
    np.testing.assert_allclose(
        overlap / abs(overlap),
        np.exp(-1j * energy * dt),
        atol=1.0e-7,
    )


def test_rttdhf_rtldr_runs_with_real_time_dependent_determinants():
    _prefer_source_package()
    import pyqed.namd as namd
    from pyqed.namd.rtldr import RTLDR, RTTDHFFrame

    grid = np.array([1.3, 1.5])
    kinetic = _second_derivative_kinetic(grid.size, grid[1] - grid[0], mass=918.0)

    def field(time):
        return np.array([0.0, 0.0, 0.02 * np.sin(0.1 * time)])

    frames = [RTTDHFFrame(_h2_rhf(r), field=field) for r in grid]
    points = np.column_stack((grid, np.zeros_like(grid)))
    solver = RTLDR(
        nuclear=_TestNuclear(points, kinetic),
        electronic=frames,
    )
    assert not hasattr(namd, "TDHFRTLDR")
    assert solver.ndim == 2
    c0, energy = solver.ground_state()
    traj = solver.run(c0, dt=0.02, nsteps=3, store_hamiltonians=True)

    assert np.isscalar(energy)
    assert traj.coefficients.shape == (4, 2)
    assert traj.overlaps.shape == (4, 2, 2)
    assert traj.kinetic_hamiltonians.shape == (4, 2, 2)
    assert traj.weighted_dipole.shape == (4, 3)
    assert traj.weighted_electron_count.shape == (4,)
    np.testing.assert_allclose(traj.norm, np.ones(4), atol=1.0e-12)
    np.testing.assert_allclose(traj.electron_counts, 2.0, atol=1.0e-10)
