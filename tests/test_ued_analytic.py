import numpy as np

from pyqed.ued.ued import electron_density_ft


def test_analytic_density_ft_q_zero_recovers_electron_count():
    from pyqed.qchem.mol import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
        spin=0,
    )
    mol.build(driver="builtin", eri="dense", aosym="s1")
    mf = mol.RHF().run(tol=1e-10, conv_tol_dm=1e-8, max_cycle=80)

    dm = mf.make_rdm1()
    dm1 = dm[None, :, :]
    tdm1 = dm[None, None, :, :]
    s_vectors = np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.0]])

    ft_ii, ft_ij = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
        ao_ft_compiled=True,
    )

    assert ft_ii.shape == (1, 2)
    assert ft_ij.shape == (1, 1, 2)
    assert np.allclose(np.einsum("mn,nm->", mol.overlap, dm).real, mol.nelec)
    assert np.allclose(ft_ii[0, 0].real, mol.nelec)
    assert np.allclose(ft_ij[0, 0, 0].real, mol.nelec)


def test_ao_density_fourier_q_zero_matches_overlap_contraction():
    from pyqed.qchem.mol import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
        spin=0,
    )
    mol.build(driver="builtin", eri="dense", aosym="s1")

    mf = mol.RHF().run(tol=1e-10, conv_tol_dm=1e-8, max_cycle=80)
    dm = mf.make_rdm1()
    dm1 = dm[None, :, :]
    tdm1 = dm[None, None, :, :]
    s_vectors = np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.0]])

    ft_ii, ft_ij = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
    )
    ft_ii_compiled, ft_ij_compiled = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
        ao_ft_compiled=True,
    )
    ft_ii_direct, ft_ij_direct = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
        ao_ft_compiled=True,
        ao_ft_direct=True,
    )
    expected = np.einsum("mn,mn->", dm, mol.overlap)

    assert ft_ii.shape == (1, 2)
    assert ft_ij.shape == (1, 1, 2)
    assert np.allclose(ft_ii[0, 0], expected)
    assert np.allclose(ft_ij[0, 0, 0], expected)
    assert np.allclose(ft_ii_compiled, ft_ii)
    assert np.allclose(ft_ij_compiled, ft_ij)
    assert np.allclose(ft_ii_direct, ft_ii)
    assert np.allclose(ft_ij_direct, ft_ij)


def test_h2o_ao_density_ft_compiled_paths_agree():
    from pyqed.qchem.mol import Molecule

    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 -1.43233673 1.10715266; "
            "H 0 1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
        spin=0,
    )
    mol.build(driver="builtin", eri="dense", aosym="s1")

    mf = mol.RHF().run(tol=1e-10, conv_tol_dm=1e-8, max_cycle=80)
    dm = mf.make_rdm1()
    dm1 = dm[None, :, :]
    tdm1 = dm[None, None, :, :]
    s_vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.25, 0.35, 0.0],
        ]
    )

    ft_ii, ft_ij = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
    )
    ft_ii_compiled, ft_ij_compiled = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
        ao_ft_compiled=True,
    )
    ft_ii_direct, ft_ij_direct = electron_density_ft(
        dm1,
        tdm1,
        mol,
        s_vectors,
        backend="native",
        ao_ft_compiled=True,
        ao_ft_direct=True,
    )
    expected_q0 = mol.nelec

    assert mol.nao == 7
    assert ft_ii.shape == (1, len(s_vectors))
    assert ft_ij.shape == (1, 1, len(s_vectors))
    assert np.allclose(np.einsum("mn,nm->", mol.overlap, dm).real, expected_q0)
    assert np.allclose(ft_ii[0, 0].real, expected_q0)
    assert np.allclose(ft_ij[0, 0, 0].real, expected_q0)
    assert np.allclose(ft_ii_compiled, ft_ii)
    assert np.allclose(ft_ij_compiled, ft_ij)
    assert np.allclose(ft_ii_direct, ft_ii)
    assert np.allclose(ft_ij_direct, ft_ij)


def test_batched_ao_pair_ft_matrices_match_scalar_integrals():
    from pyqed.qchem.fourier import (
        ao_pair_ft_matrices,
        ao_pair_ft_matrices_compiled,
        gaussian_pair_ft,
        gaussian_pair_ft_batch,
    )
    from pyqed.qchem.mol import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
        spin=0,
    )
    mol.build(driver="builtin", eri="dense", aosym="s1")
    basis, transform = mol._cart_basis()
    assert transform is None

    gvecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [-0.2, 0.4, 0.1],
        ]
    )
    values = gaussian_pair_ft_batch(basis[0], basis[1], gvecs)
    expected = np.array([gaussian_pair_ft(basis[0], basis[1], g) for g in gvecs])
    matrices = ao_pair_ft_matrices(basis, gvecs, compiled=False)
    compiled_matrices = ao_pair_ft_matrices_compiled(basis, gvecs)

    assert np.allclose(values, expected)
    assert np.allclose(matrices[:, 0, 1], expected)
    assert np.allclose(matrices[:, 0, 1], matrices[:, 1, 0])
    assert np.allclose(compiled_matrices, matrices)


def test_ao_ft_plan_batch_matches_single_geometry_contracts():
    from pyqed.qchem.fourier import AOPairFTPlan
    from pyqed.qchem.mol import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
        spin=0,
    )
    mol.build(driver="builtin", eri="dense", aosym="s1")
    plan = AOPairFTPlan.from_molecule(mol)
    dm = np.eye(mol.nao, dtype=complex)
    dm1 = np.stack([dm, 0.5 * dm])
    tdm1 = np.zeros((2, 2, mol.nao, mol.nao), dtype=complex)
    tdm1[0, 0] = dm
    tdm1[1, 1] = 0.5 * dm
    tdm1[0, 1] = 0.25 * dm
    tdm1[1, 0] = 0.25 * dm
    gvecs = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.0]])

    origins_a = plan.origins
    origins_b = plan.origins_from_atom_coords(
        mol.atom_coords() + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    )
    ft_a = plan.contract(dm1, tdm1, gvecs, origins=origins_a, compiled=True)
    ft_b = plan.contract(dm1, tdm1, gvecs, origins=origins_b, compiled=True)
    ft_batch = plan.contract_batch(
        np.stack([dm1, dm1]),
        np.stack([tdm1, tdm1]),
        gvecs,
        np.stack([origins_a, origins_b]),
        compiled=True,
    )

    assert np.allclose(ft_batch[0][0], ft_a[0])
    assert np.allclose(ft_batch[0][1], ft_b[0])
    assert np.allclose(ft_batch[1][0], ft_a[1])
    assert np.allclose(ft_batch[1][1], ft_b[1])
