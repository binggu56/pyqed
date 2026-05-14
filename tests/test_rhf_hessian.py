import numpy as np
import pytest

from pyqed.qchem import Molecule, RHF


def _total_dipole_origin_zero(method):
    mol = method.mol
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges(), dtype=float)
    nuclear = np.einsum("a,ax->x", charges, coords)
    r_ao = np.asarray(mol.position_integral(center=np.zeros(3)), dtype=float)
    electronic = -np.einsum("xij,ji->x", r_ao, method.make_rdm1(), optimize=True)
    return nuclear + electronic


def _rhf_at(coords):
    mol = Molecule(
        atom=[("H", tuple(coords[0])), ("H", tuple(coords[1]))],
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    return RHF(mol).run(tol=1.0e-11)


def test_native_rhf_hessian_matches_pyscf_h2():
    pytest.importorskip("pyscf")
    from pyscf import scf

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(tol=1.0e-11)

    hess = mf.Hessian().run()

    pmol = mol.topyscf()
    pmol.build(verbose=0)
    pmf = scf.RHF(pmol).run(conv_tol=1.0e-12, verbose=0)
    ref4 = pmf.Hessian().kernel()
    ref = ref4.transpose(0, 2, 1, 3).reshape(6, 6)

    np.testing.assert_allclose(hess, ref, atol=1.0e-8)


def test_native_rhf_analytic_dipole_derivative_matches_finite_difference_h2():
    coords = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    mf = _rhf_at(coords)
    hess = mf.Hessian()
    hess.run()
    analytic = hess.cartesian_dipole_derivatives()

    step = 1.0e-4
    finite_diff = np.zeros_like(analytic)
    for atom in range(coords.shape[0]):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_diff[atom, axis] = (
                _total_dipole_origin_zero(_rhf_at(plus))
                - _total_dipole_origin_zero(_rhf_at(minus))
            ) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite_diff, atol=1.0e-8)


def test_native_rhf_hessian_rejects_density_fit_reference():
    pytest.importorskip("pyscf")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(density_fit=True)

    with pytest.raises(NotImplementedError, match="builtin RHF reference"):
        mf.Hessian().run()
