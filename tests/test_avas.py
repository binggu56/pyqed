import numpy as np
import pytest

from pyqed.qchem import AVAS, CASSCF, Molecule, avas
from pyqed.qchem.mcscf.avas import (
    _reference_molecule,
    _target_ao_indices,
    _target_ao_transform,
)


def _native_reference(atom, basis="sto-3g", unit="angstrom", spin=0):
    mol = Molecule(atom=atom, unit=unit, basis=basis, spin=spin)
    mol.build(eri="dense")
    return mol.RHF().run()


def test_native_avas_selects_hydrogen_valence_space_for_casscf():
    mf = _native_reference("H 0 0 0; H 0 0 0.74")

    ncas, nelecas, mo = avas.run(mf, ["H 1s"])
    mc = CASSCF(mf, ncas=ncas, nelecas=nelecas).run(mo_coeff=mo)

    assert (ncas, nelecas) == (2, 2)
    assert mc.converged
    np.testing.assert_allclose(
        mo.conj().T @ mf.get_ovlp() @ mo,
        np.eye(mo.shape[1]),
        atol=1e-10,
    )


def test_native_avas_selects_nitrogen_2p_space():
    mf = _native_reference("N 0 0 0; N 0 0 1.1")
    selector = AVAS(mf, ["N 2p"], canonicalize=False)

    ncas, nelecas, mo = selector.run()

    # The N 2p projector also retains the strongly hybridized sigma pair.
    assert (ncas, nelecas) == (7, 8)
    assert selector.target_ao_indices.size == 6
    assert np.count_nonzero(selector.occ_weights >= selector.threshold) == 4
    assert np.count_nonzero(selector.vir_weights >= selector.threshold) == 3
    np.testing.assert_allclose(
        mo.conj().T @ mf.get_ovlp() @ mo,
        np.eye(mo.shape[1]),
        atol=1e-10,
    )


def test_native_avas_iao_path_is_orthonormal_and_retains_weights():
    mf = _native_reference("N 0 0 0; N 0 0 1.1")
    selector = AVAS(mf, "N 2p", with_iao=True)

    ncas, nelecas, mo = selector.run()

    assert ncas > 0
    assert 0 < nelecas <= mf.mol.nelec
    assert selector.occ_weights is not None
    assert selector.vir_weights is not None
    np.testing.assert_allclose(
        mo.conj().T @ mf.get_ovlp() @ mo,
        np.eye(mo.shape[1]),
        atol=1e-10,
    )


def test_native_avas_reports_unmatched_labels():
    mf = _native_reference("H 0 0 0; H 0 0 0.74")

    with pytest.raises(ValueError, match="No reference AOs match"):
        avas.run(mf, ["Fe 3d"])


def test_native_avas_open_shell_option_three_keeps_singly_occupied_orbitals():
    mol = Molecule(atom="O 0 0 0", unit="bohr", basis="sto-3g", spin=2)
    mol.build(eri="dense")
    mf = mol.ROHF().run()
    selector = AVAS(mf, ["O 2p"], openshell_option=3)

    ncas, nelecas, _ = selector.run()

    assert (ncas, nelecas) == (3, 4)
    assert np.count_nonzero(np.isclose(selector.occ_weights, 1.0)) == 2


def test_avas_supports_builtin_reference_natively():
    pytest.importorskip("builtin")
    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build()
    mf = mol.RHF().run()

    ncas, nelecas, mo = avas.run(mf, ["H 1s"])

    assert (ncas, nelecas) == (2, 2)
    np.testing.assert_allclose(
        mo.conj().T @ mf.get_ovlp() @ mo,
        np.eye(2),
        atol=1e-10,
    )


def test_native_avas_rejects_unconverged_reference_data():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
    mol.build(eri="dense")
    mf = mol.RHF()

    with pytest.raises(ValueError, match="Run the mean-field calculation"):
        avas.run(mf, ["H 1s"])


def test_minao_cartesian_d_shell_maps_to_five_spherical_functions():
    mol = Molecule(atom="Fe 0 0 0", unit="bohr", basis="sto-3g")

    reference = _reference_molecule(mol, "minao")
    labels = reference.ao_labels()
    target = _target_ao_indices(labels, ["Fe 3d"])
    transform = _target_ao_transform(labels, target)

    assert reference.cart
    assert target.size == 6
    assert transform.shape == (6, 5)
