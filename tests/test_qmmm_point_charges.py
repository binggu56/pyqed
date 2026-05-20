import numpy as np

from pyqed import Molecule
from pyqed.qchem import embed_point_charges
from pyqed.qchem.qmmm import (
    nuclear_point_charge_energy,
    point_charge_forces,
    point_charge_hcore,
    point_charge_hcore_derivatives,
)


def test_point_charge_hcore_and_nuclear_energy_are_finite():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    mol.build(driver="builtin")
    coords = np.array([[0.0, 0.0, 3.0]])
    charges = np.array([-0.2])

    hcore = point_charge_hcore(mol, coords, charges)
    hcore_deriv = point_charge_hcore_derivatives(mol, coords, charges)
    energy = nuclear_point_charge_energy(mol, coords, charges)

    assert hcore.shape == mol.hcore.shape
    assert hcore_deriv.shape == (mol.natom, 3, mol.nao, mol.nao)
    assert np.all(np.isfinite(hcore))
    assert np.all(np.isfinite(hcore_deriv))
    assert np.isfinite(energy)
    assert np.linalg.norm(hcore) > 0.0


def test_embed_point_charges_returns_energy_gradients_and_charge_forces():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    mol.build(driver="builtin")
    mf = embed_point_charges(
        mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
        fd_step=1e-3,
    )
    mf._finite_difference_qm_gradient = _fail_if_called

    energy, qm_grad, point_charge_forces = mf.energy_and_gradients()

    assert np.isfinite(energy)
    assert qm_grad.shape == (2, 3)
    assert point_charge_forces.shape == (1, 3)
    assert np.all(np.isfinite(qm_grad))
    assert np.all(np.isfinite(point_charge_forces))
    assert np.linalg.norm(point_charge_forces) > 0.0


def _fail_if_called(*args, **kwargs):
    raise AssertionError("QM gradients should use the native analytic RHF machinery.")


def test_embedded_qm_gradient_matches_energy_finite_difference():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    mol.build(driver="builtin")
    mf = embed_point_charges(
        mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )

    _, qm_grad, _ = mf.energy_and_gradients()

    step = 1e-4
    plus_mol = Molecule(atom=f"H 0 0 {step}; H 0 0 1.4", unit="b", basis="sto3g")
    plus_mol.build(driver="builtin")
    plus = embed_point_charges(
        plus_mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    ).kernel()
    minus_mol = Molecule(atom=f"H 0 0 {-step}; H 0 0 1.4", unit="b", basis="sto3g")
    minus_mol.build(driver="builtin")
    minus = embed_point_charges(
        minus_mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    ).kernel()
    finite_difference_grad = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(qm_grad[0, 2], finite_difference_grad, rtol=1e-5, atol=1e-7)


def test_point_charge_forces_match_energy_finite_difference():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="b", basis="sto3g")
    mol.build(driver="builtin")
    mf = embed_point_charges(
        mol.RHF(),
        coords=[[0.0, 0.0, 3.0]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    mf.run()
    analytic_force = point_charge_forces(
        mol,
        mf.make_rdm1(),
        [[0.0, 0.0, 3.0]],
        [-0.2],
    )[0, 2]

    step = 1e-4
    plus = embed_point_charges(
        mol.RHF(),
        coords=[[0.0, 0.0, 3.0 + step]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    ).kernel()
    minus = embed_point_charges(
        mol.RHF(),
        coords=[[0.0, 0.0, 3.0 - step]],
        charges=[-0.2],
        run_kwargs={"verbose": 0, "max_cycle": 100},
    ).kernel()
    finite_difference_force = -(plus - minus) / (2.0 * step)

    np.testing.assert_allclose(analytic_force, finite_difference_force, rtol=1e-5, atol=1e-7)
