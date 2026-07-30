import numpy as np

from pyqed.qchem.gdvr import Molecule, RTTDHF


def _eri_blocks(nz, m, value=0.0):
    block = np.full((m * m, m * m), value, dtype=float)
    return [[block.copy() for _ in range(nz)] for _ in range(nz)]


class _ToyGDVRMolecule:
    def __init__(self, eri_value=0.0, nelec=2):
        self.z = np.array([-1.0, 0.0, 1.0])
        self.shapes = {"Nz": 3, "M": 1, "size": 3}
        self.hcore = np.diag([0.0, 0.7, 1.4])
        self.eri_j = _eri_blocks(3, 1, value=0.0)
        self.eri_k = _eri_blocks(3, 1, value=eri_value)
        self.nelec = int(nelec)

    def nuclear_repulsion_energy(self):
        return 0.0


class _ToyGDVRRHF:
    def __init__(self, eri_value=0.0, nelec=2):
        self.mol = _ToyGDVRMolecule(eri_value=eri_value, nelec=nelec)
        self.mo_coeff = np.eye(3)
        self.mo_energy = np.diag(self.mol.hcore)
        self.mo_occ = np.zeros(3)
        self.mo_occ[: self.mol.nelec // 2] = 2.0
        self.dm = np.diag(self.mo_occ)

    def RTTDHF(self, *args, **kwargs):
        return RTTDHF(self, *args, **kwargs)


def test_gdvr_rttdhf_ground_state_is_stationary_without_field():
    mf = _ToyGDVRRHF()
    rt = RTTDHF(mf).run(dt=0.1, nsteps=5, store_dm=True)

    np.testing.assert_allclose(rt.dms[-1], rt.dms[0], atol=1e-12)
    np.testing.assert_allclose(rt.energies[-1], rt.energies[0], atol=1e-12)
    np.testing.assert_allclose(rt.electron_counts, 2.0, atol=1e-12)
    np.testing.assert_allclose(rt.electron_count(), 2.0, atol=1e-12)
    np.testing.assert_allclose(rt.fields, 0.0, atol=1e-12)
    np.testing.assert_allclose(rt.dipole_velocities, 0.0, atol=1e-12)
    np.testing.assert_allclose(rt.dipole_accelerations, 0.0, atol=1e-12)


def test_gdvr_rttdhf_kick_preserves_trace_and_hermiticity():
    mf = _ToyGDVRRHF()
    interaction = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    rt = mf.RTTDHF(interaction=interaction).run(
        dt=0.05,
        nsteps=4,
        store_dm=True,
        kick={"strength": 0.1},
    )

    np.testing.assert_allclose(rt.electron_count(), 2.0, atol=1e-12)
    np.testing.assert_allclose(rt.dms[-1], rt.dms[-1].conj().T, atol=1e-12)
    assert np.max(np.abs(rt.dipoles[:, 2])) > 0.0


def test_gdvr_rttdhf_preserves_complex_fock_response():
    mf = _ToyGDVRRHF(eri_value=1.0)
    rt = RTTDHF(mf)
    dm = mf.dm.astype(complex)
    dm[0, 1] = 0.2j
    dm[1, 0] = -0.2j

    fock = rt.field_free_fock(dm)

    assert np.iscomplexobj(fock)
    np.testing.assert_allclose(fock, fock.conj().T, atol=1e-12)


def test_gdvr_rttdhf_public_alias_and_scalar_field_uses_z_axis():
    mf = _ToyGDVRRHF()
    rt = RTTDHF(mf, field=lambda _time: 0.25).run(dt=0.0, nsteps=0)

    assert isinstance(rt, RTTDHF)
    np.testing.assert_allclose(rt.fields[0], [0.0, 0.0, 0.25])


def test_gdvr_molecule_cap_has_edge_profile():
    mol = Molecule([2.0], [[0.0, 0.0, 0.0]], nelec=2)
    mol.z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mol.shapes = {"Nz": 5, "M": 1, "size": 5}
    cap = mol.cap(width=1.5, strength=0.2)

    np.testing.assert_allclose(
        np.diag(cap),
        [0.2, 0.02222222222222222, 0.0, 0.02222222222222222, 0.2],
    )


def test_gdvr_rttdhf_cap_reduces_electron_count():
    mf = _ToyGDVRRHF()
    cap = np.diag([0.5, 0.0, 0.0])

    rt = RTTDHF(mf, cap=cap).run(dt=0.1, nsteps=3, store_dm=True)

    assert rt.electron_counts[-1] < rt.electron_counts[0]
    np.testing.assert_allclose(rt.dms[-1], rt.dms[-1].conj().T, atol=1e-12)


def test_gdvr_rttdhf_orbital_density_matches_reference_for_multiple_occupied_orbitals():
    mf = _ToyGDVRRHF(nelec=4)
    rt = RTTDHF(mf)

    orbitals, occupations = rt.occupied_orbitals(return_occupations=True)

    assert orbitals.shape == (3, 2)
    np.testing.assert_allclose(orbitals.conj().T @ orbitals, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(occupations, [2.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(rt.density_from_orbitals(orbitals), np.diag([1.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(rt.density_from_orbitals(orbitals, occupations), mf.dm, atol=1e-12)


def test_gdvr_rttdhf_orbital_run_stores_orthonormal_orbitals_and_occupations():
    mf = _ToyGDVRRHF(nelec=4)

    rt = RTTDHF(mf).run(
        dt=0.05,
        nsteps=2,
        method="orbital",
        store_dm=True,
        store_orbitals=True,
    )

    np.testing.assert_allclose(rt.orbital_occupations, [2.0, 2.0], atol=1e-12)
    for istep, orbitals in enumerate(rt.orbitals):
        np.testing.assert_allclose(orbitals.conj().T @ orbitals, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(
            rt.density_from_orbitals(orbitals, rt.orbital_occupations),
            rt.dms[istep],
            atol=1e-12,
        )


def test_gdvr_rttdhf_orbital_propagation_matches_density_propagation_without_cap():
    interaction = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    rt_density = RTTDHF(_ToyGDVRRHF(), interaction=interaction).run(
        dt=0.05,
        nsteps=3,
        store_dm=True,
        kick={"strength": 0.1},
        method="density",
    )
    rt_orbital = RTTDHF(_ToyGDVRRHF(), interaction=interaction).run(
        dt=0.05,
        nsteps=3,
        store_dm=True,
        kick={"strength": 0.1},
        method="orbital",
    )

    np.testing.assert_allclose(rt_orbital.dms, rt_density.dms, atol=1e-11)
    np.testing.assert_allclose(rt_orbital.dipoles, rt_density.dipoles, atol=1e-11)
    np.testing.assert_allclose(rt_orbital.electron_counts, rt_density.electron_counts, atol=1e-11)


def test_gdvr_rttdhf_orbital_cap_reduces_electron_count():
    mf = _ToyGDVRRHF()
    cap = np.diag([0.5, 0.0, 0.0])

    rt = RTTDHF(mf, cap=cap).run(dt=0.1, nsteps=3, method="orbital")

    assert rt.electron_counts[-1] < rt.electron_counts[0]
    assert rt.propagation_method == "orbital"
