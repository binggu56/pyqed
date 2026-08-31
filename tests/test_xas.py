import numpy as np
import pytest

from pyqed.qchem import CASCI, Molecule, RHF, TDA, XAS


def _h2_tda():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="s8")
    mf = RHF(mol).run()
    return TDA(mf).run(nstates=1)


def test_xas_from_native_tda_core_selected_h2():
    td = _h2_tda()

    xas = XAS(td, core_atoms="H").run()

    np.testing.assert_array_equal(xas.states, np.array([1]))
    assert xas.excitation_energies.shape == (1,)
    assert xas.transition_dipoles.shape == (1, 3)
    assert xas.oscillator_strengths.shape == (1,)
    assert xas.intensities.shape == (1,)
    assert xas.core_weights.shape == (1,)
    np.testing.assert_array_equal(xas.core_orbitals, np.array([0]))
    np.testing.assert_array_equal(xas.core_atom_indices, np.array([0, 1]))
    np.testing.assert_allclose(xas.core_weights, np.ones(1), atol=1e-12)
    np.testing.assert_allclose(xas.intensities, xas.oscillator_strengths, atol=1e-12)
    assert np.all(np.isfinite(xas.transition_dipoles))
    assert np.all(xas.oscillator_strengths >= 0.0)


def test_cvs_tda_xas_targets_water_oxygen_core_roots():
    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 1.43233673 1.10715266; "
            "H 0 -1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")
    mf = RHF(mol).run()

    cvs = XAS(TDA(mf), core_atoms="O").run(cvs=True, nstates=2)
    full_td = TDA(mf).run(nstates=10)
    selected = XAS(full_td, core_atoms="O", min_core_weight=0.9).run()

    np.testing.assert_array_equal(cvs.core_orbitals, np.array([0]))
    np.testing.assert_array_equal(cvs.core_atom_indices, np.array([0]))
    np.testing.assert_array_equal(cvs.states, np.array([1, 2]))
    np.testing.assert_allclose(cvs.core_weights, np.ones(2))
    np.testing.assert_allclose(cvs.excitation_energies, selected.excitation_energies[:2], atol=5e-4)
    np.testing.assert_allclose(cvs.oscillator_strengths, selected.oscillator_strengths[:2], rtol=0.06, atol=5e-4)
    assert np.all(cvs.excitation_energies > 5.0)
    assert np.all(cvs.oscillator_strengths >= 0.0)


def test_xas_can_target_core_atom_by_index_and_ranked_orbital():
    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 1.43233673 1.10715266; "
            "H 0 -1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")
    mf = RHF(mol).run()

    oxygen_by_symbol = XAS(TDA(mf), core_atoms="O").run(cvs=True, nstates=1)
    oxygen_by_index = XAS(TDA(mf), core_atoms=0).run(cvs=True, nstates=1)
    second_oxygen_orbital = XAS(TDA(mf), core_atoms=0, core_orbital_rank=1).run(
        cvs=True,
        nstates=1,
    )
    explicit_core_mo = XAS(TDA(mf), core_orbitals=[0]).run(cvs=True, nstates=1)

    np.testing.assert_array_equal(oxygen_by_symbol.core_atom_indices, np.array([0]))
    np.testing.assert_array_equal(oxygen_by_index.core_atom_indices, np.array([0]))
    np.testing.assert_array_equal(oxygen_by_symbol.core_orbitals, oxygen_by_index.core_orbitals)
    np.testing.assert_array_equal(explicit_core_mo.core_orbitals, np.array([0]))
    assert second_oxygen_orbital.core_orbitals[0] != oxygen_by_index.core_orbitals[0]
    assert second_oxygen_orbital.core_orbitals.size == 1


def test_xas_accepts_spectroscopic_core_string():
    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 1.43233673 1.10715266; "
            "H 0 -1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")
    mf = RHF(mol).run()

    from_constructor = XAS(TDA(mf), core="O 1s").run(cvs=True, nstates=1)
    from_run = XAS(TDA(mf)).run(cvs=True, nstates=1, core="O K")

    np.testing.assert_array_equal(from_constructor.core_atom_indices, np.array([0]))
    np.testing.assert_array_equal(from_constructor.core_orbitals, np.array([0]))
    np.testing.assert_array_equal(from_run.core_atom_indices, np.array([0]))
    np.testing.assert_array_equal(from_run.core_orbitals, np.array([0]))
    np.testing.assert_allclose(from_constructor.excitation_energies, from_run.excitation_energies)


def test_xas_core_string_rejects_unsupported_shell():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="s8")
    mf = RHF(mol).run()

    with pytest.raises(NotImplementedError, match="Core shell"):
        XAS(TDA(mf), core="H 2p").run(cvs=True, nstates=1)


def test_xas_spectrum_from_native_tda_h2():
    td = _h2_tda()
    xas = XAS(td, core_atoms=[0])

    grid, signal = xas.spectrum(width=0.2, units="ev")

    assert grid.shape == signal.shape
    assert grid.ndim == 1
    assert grid.size == 1000
    assert np.all(np.isfinite(signal))


def test_xas_from_sticks_and_lineshape_validation():
    xas = XAS.from_sticks(
        energies=[285.0, 287.0],
        oscillator_strengths=[0.1, 0.2],
        units="ev",
    )

    grid, signal = xas.spectrum(width=0.5, units="ev")

    assert grid.shape == signal.shape
    np.testing.assert_array_equal(xas.states, np.array([1, 2]))
    np.testing.assert_allclose(xas.oscillator_strengths, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="lineshape"):
        xas.spectrum(width=0.5, lineshape="triangle")


def test_xas_rejects_invalid_core_weight_threshold():
    td = _h2_tda()

    with pytest.raises(ValueError, match="min_core_weight"):
        XAS(td, min_core_weight=1.1).run()


def test_xas_from_full_active_casci_h2():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="s8")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    xas = XAS(mc, core_atoms="H").run()

    np.testing.assert_array_equal(xas.states, np.array([1]))
    assert xas.excitation_energies.shape == (1,)
    assert xas.transition_dipoles.shape == (1, 3)
    assert xas.oscillator_strengths.shape == (1,)
    assert xas.core_weights.shape == (1,)
    np.testing.assert_array_equal(xas.core_orbitals, np.array([0]))
    assert np.all(np.isfinite(xas.transition_dipoles))
    assert np.all(xas.core_weights >= 0.0)


def test_cvs_casci_xas_targets_core_hole_subspace_h2():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="s8")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method="ci")

    analyzer = XAS(mc, core_atoms="H")
    xas = analyzer.run(cvs=True, nstates=2)

    np.testing.assert_array_equal(xas.states, np.array([1, 2]))
    np.testing.assert_array_equal(xas.core_orbitals, np.array([0]))
    assert analyzer.cvs_determinant_indices.size > 0
    assert xas.excitation_energies.shape == (2,)
    assert xas.transition_dipoles.shape == (2, 3)
    np.testing.assert_allclose(xas.core_weights, np.ones(2))
    assert np.all(np.isfinite(xas.excitation_energies))
    assert np.all(np.isfinite(xas.oscillator_strengths))


def test_cvs_casci_xas_finds_water_oxygen_core_roots_without_many_valence_roots():
    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 1.43233673 1.10715266; "
            "H 0 -1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=mol.nao, nelecas=mol.nelec).run(nstates=10, method="ci")

    valence = XAS(mc, core_atoms="O", min_core_weight=0.05).run()
    cvs = XAS(mc, core_atoms="O").run(cvs=True, nstates=2)

    assert valence.states.size == 0
    np.testing.assert_array_equal(cvs.states, np.array([1, 2]))
    np.testing.assert_array_equal(cvs.core_orbitals, np.array([0]))
    np.testing.assert_allclose(cvs.core_weights, np.ones(2))
    assert np.all(cvs.excitation_energies > 5.0)
    assert np.all(cvs.oscillator_strengths >= 0.0)


def test_casci_xas_rejects_frozen_core_orbital():
    mol = Molecule(atom="Li 0 0 0; H 0 0 3.0", unit="bohr", basis="sto-3g")
    mol.build(eri="s8")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    with pytest.raises(ValueError, match="requires selected core orbitals to be active"):
        XAS(mc, core_atoms="Li").run()
