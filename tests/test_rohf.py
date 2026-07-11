import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf import ROHF
from pyqed.qchem.mcscf.casci import CASCI


def _no2():
    mol = Molecule(
        atom="N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0",
        basis="sto-3g",
        charge=0,
        spin=1,
        unit="Angstrom",
    )
    mol.build()
    return mol


def test_native_rohf_runs_no2_doublet_and_matches_pyscf():
    mol = _no2()

    mf = mol.ROHF().run(conv_tol=1.0e-9, conv_tol_dm=1.0e-6, max_cycle=100, damping=0.25)

    assert isinstance(mf, ROHF)
    assert mf.converged
    assert mf.mo_coeff.shape == (mol.nao, mol.nao)
    assert mf.mo_occ.shape == (mol.nao,)
    np.testing.assert_allclose(mf.mo_occ.sum(), mol.nelec)
    np.testing.assert_allclose(mf.mo_occ[:11], 2.0)
    np.testing.assert_allclose(mf.mo_occ[11], 1.0)
    np.testing.assert_allclose(mf.mo_occ[12:], 0.0)
    np.testing.assert_allclose(np.einsum("ij,ji->", mol.overlap, mf.dm_alpha), 12.0, atol=1.0e-8)
    np.testing.assert_allclose(np.einsum("ij,ji->", mol.overlap, mf.dm_beta), 11.0, atol=1.0e-8)

    from pyscf import gto, scf

    pmol = gto.M(
        atom="N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0",
        basis="sto-3g",
        charge=0,
        spin=1,
        unit="Angstrom",
        verbose=0,
    )
    pmf = scf.ROHF(pmol).run(conv_tol=1.0e-10, max_cycle=100, verbose=0)
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1.0e-6)


def test_native_rohf_casci_no2_doublet_spin_square():
    mol = _no2()
    mf = mol.ROHF().run(conv_tol=1.0e-9, conv_tol_dm=1.0e-6, max_cycle=100, damping=0.25)

    mc = CASCI(mf, ncas=5, nelecas=(3, 2), spin=1).run(nstates=3, method="direct_ci")

    assert np.asarray(mc.e_tot).shape == (3,)
    for root in range(3):
        np.testing.assert_allclose(mc.spin_square(root), 0.75, atol=1.0e-8)


def test_native_rohf_as_scanner_reuses_spin_density():
    mol = _no2()
    mf = mol.ROHF().run(conv_tol=1.0e-8, conv_tol_dm=1.0e-6, max_cycle=100, damping=0.25)
    scanner = mf.as_scanner()

    geom = mol.atom_coords().copy()
    geom[1, 0] += 0.02
    energy = scanner(geom)

    assert np.isfinite(energy)
    assert isinstance(scanner.mf, ROHF)
    assert scanner.mol is mol
    assert scanner.dm_alpha.shape == (mol.nao, mol.nao)
    assert scanner.dm_beta.shape == (mol.nao, mol.nao)
    np.testing.assert_allclose(
        np.einsum("ij,ji->", mol.overlap, scanner.dm_alpha),
        12.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.einsum("ij,ji->", mol.overlap, scanner.dm_beta),
        11.0,
        atol=1.0e-7,
    )


def test_triatomic_scan_worker_accepts_native_rohf_casci():
    from pyqed.namd.triatomic import _normalize_triatomic_electronic_method
    from pyqed.namd.triatomic import _triatomic_scan_point_worker

    assert _normalize_triatomic_electronic_method("ROHF/CASCI") == "rohf-casci"
    xyz = np.array(
        [
            [1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.6, 1.0392304845, 0.0],
        ]
    )
    task = (
        (0, 0, 0),
        xyz,
        ("O", "N", "O"),
        "sto-3g",
        0,
        1,
        "Angstrom",
        5,
        (3, 2),
        3,
        "rohf/casci",
        {"scf_tol": 1.0e-9, "max_cycle": 100, "damping": 0.25, "verbose": 0},
    )

    idx, energies, mc = _triatomic_scan_point_worker(task)

    assert idx == (0, 0, 0)
    assert energies.shape == (3,)
    for root in range(3):
        np.testing.assert_allclose(mc.spin_square(root), 0.75, atol=1.0e-8)
