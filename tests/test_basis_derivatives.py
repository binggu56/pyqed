import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.basis_derivatives import (
    compact_eri_veff,
    compact_eri_veff_many,
    eri_derivative_veff_scalar,
    eri_derivatives,
    one_electron_derivatives,
    position_derivatives,
)


def _h2(z):
    mol = Molecule(
        atom=f"H 0 0 0; H 0 0 {z}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    return mol


def test_builtin_first_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 1.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    ds = one_electron_derivatives(mol, "overlap", order=1)
    dh = one_electron_derivatives(mol, "hcore", order=1)
    dg = eri_derivatives(mol, order=1)

    np.testing.assert_allclose(
        ds[1, 2],
        (mol_p.overlap - mol_m.overlap) / (2.0 * step),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        dh[1, 2],
        (mol_p.hcore - mol_m.hcore) / (2.0 * step),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        dg[1, 2],
        (mol_p.eri - mol_m.eri) / (2.0 * step),
        atol=1.0e-8,
    )


def test_builtin_second_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 2.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    d2s = one_electron_derivatives(mol, "overlap", order=2)
    d2h = one_electron_derivatives(mol, "hcore", order=2)
    d2g = eri_derivatives(mol, order=2)

    np.testing.assert_allclose(
        d2s[1, 2, 1, 2],
        (mol_p.overlap - 2.0 * mol.overlap + mol_m.overlap) / step**2,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        d2h[1, 2, 1, 2],
        (mol_p.hcore - 2.0 * mol.hcore + mol_m.hcore) / step**2,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        d2g[1, 2, 1, 2],
        (mol_p.eri - 2.0 * mol.eri + mol_m.eri) / step**2,
        atol=2.0e-7,
    )


def test_compact_eri_derivative_contractions_match_dense_h2():
    mol = _h2(1.4)
    rng = np.random.default_rng(3)
    dm = rng.normal(size=(mol.nao, mol.nao))
    dm = dm + dm.T

    dense1 = eri_derivatives(mol, order=1)
    compact1 = eri_derivatives(mol, order=1, compact=True)
    dense2 = eri_derivatives(mol, order=2)
    compact2 = eri_derivatives(mol, order=2, compact=True)

    def dense_veff(eri):
        vj = np.einsum("rs,pqrs->pq", dm, eri, optimize=True)
        vk = np.einsum("rs,prqs->pq", dm, eri, optimize=True)
        return vj - 0.5 * vk

    np.testing.assert_allclose(
        compact_eri_veff(compact1, dm, 5),
        dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff_many(compact1, dm)[5],
        dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff(compact2, dm, 5, 5),
        dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff_many(compact2, dm).reshape(6, 6, mol.nao, mol.nao)[5, 5],
        dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        eri_derivative_veff_scalar(mol, dm, dm, order=1)[5],
        np.einsum(
            "pq,pq->",
            dm,
            dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
            optimize=True,
        ),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        eri_derivative_veff_scalar(mol, dm, dm, order=2)[5, 5],
        np.einsum(
            "pq,pq->",
            dm,
            dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
            optimize=True,
        ),
        atol=1.0e-10,
    )


def test_builtin_position_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 1.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    dr = position_derivatives(mol, center=np.zeros(3))

    np.testing.assert_allclose(
        dr[1, 2],
        (mol_p.position_integral(center=np.zeros(3)) - mol_m.position_integral(center=np.zeros(3)))
        / (2.0 * step),
        atol=1.0e-8,
    )


def test_builtin_derivatives_follow_spherical_ao_transform_h2o():
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")

    ds = one_electron_derivatives(mol, "overlap", order=1)
    dh = one_electron_derivatives(mol, "hcore", order=1)
    dg = eri_derivatives(mol, order=1)

    assert ds.shape == (mol.natom, 3, mol.nao, mol.nao)
    assert dh.shape == (mol.natom, 3, mol.nao, mol.nao)
    assert dg.shape == (mol.natom, 3, mol.nao, mol.nao, mol.nao, mol.nao)
