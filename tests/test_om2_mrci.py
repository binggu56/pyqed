import numpy as np
import pytest


def test_om2_imports_from_public_api():
    from pyqed.qchem import DEFAULT_OM2_PARAMETERS, OM2, OM2MRCIScanner, OM2ParameterError, OM2ParameterSet, SemiempiricalMRCI
    from pyqed.qchem.semiempirical import MRCI

    assert DEFAULT_OM2_PARAMETERS is not None
    assert OM2 is not None
    assert OM2MRCIScanner is not None
    assert OM2ParameterError is not None
    assert OM2ParameterSet is not None
    assert SemiempiricalMRCI is MRCI


def test_default_om2_parameters_include_full_published_rows():
    from pyqed.qchem.semiempirical import DEFAULT_OM2_PARAMETERS
    from pyqed.qchem.semiempirical.om2 import EV_TO_HARTREE

    carbon = DEFAULT_OM2_PARAMETERS.for_symbol("C")
    hydrogen = DEFAULT_OM2_PARAMETERS.for_symbol("H")

    np.testing.assert_allclose(carbon.beta_p_h, -4.04444703 * EV_TO_HARTREE)
    np.testing.assert_allclose(carbon.alpha_s_h, 0.09668329)
    np.testing.assert_allclose(carbon.f1, 0.49949211)
    np.testing.assert_allclose(carbon.g2, 0.99250289)
    np.testing.assert_allclose(carbon.ecp_faa, -305.68646337 * EV_TO_HARTREE)
    np.testing.assert_allclose(carbon.ecp_beta, -9.07185084 * EV_TO_HARTREE)
    np.testing.assert_allclose(carbon.ecp_alpha, 0.16985745)
    assert hydrogen.ecp_zeta is None


def test_published_om2_benchmark_targets_are_available():
    from pyqed.qchem.semiempirical import (
        PUBLISHED_OM2_G2_HEATS_OF_FORMATION_SAMPLE,
        PUBLISHED_OM2_GROUND_STATE_MAES,
        PUBLISHED_OM2_S22_SINGLE_POINT_INTERACTIONS,
        format_published_om2_benchmarks,
        format_published_om2_molecule_benchmarks,
        published_om2_benchmarks,
        published_om2_molecule_benchmarks,
    )

    records = published_om2_benchmarks()
    g2 = [rec for rec in records if rec.subset == "G2-CHNOF"][0]
    s22 = [rec for rec in records if rec.group == "S22" and rec.subset == "overall"][0]

    assert PUBLISHED_OM2_GROUND_STATE_MAES
    assert g2.mae == pytest.approx(3.37)
    assert g2.unit == "kcal/mol"
    assert s22.method == "OM2-D3"
    assert s22.mae == pytest.approx(0.91)
    table = format_published_om2_benchmarks(records[:2])
    assert "G2-CHNOF" in table
    assert "3.37 kcal/mol" in table

    g2_molecules = published_om2_molecule_benchmarks("G2-CHNOF")
    s22_molecules = published_om2_molecule_benchmarks("S22")
    water = [rec for rec in g2_molecules if rec.name == "water (H2O)"][0]
    carbon_monoxide = [rec for rec in g2_molecules if rec.name == "carbon monoxide (CO)"][0]
    water_dimer = [rec for rec in s22_molecules if rec.name == "water dimer"][0]
    all_molecules = published_om2_molecule_benchmarks("all")
    molecule_table = format_published_om2_molecule_benchmarks(g2_molecules[:2])

    assert PUBLISHED_OM2_G2_HEATS_OF_FORMATION_SAMPLE
    assert PUBLISHED_OM2_S22_SINGLE_POINT_INTERACTIONS
    assert water.reference == pytest.approx(-57.8)
    assert water.om2 == pytest.approx(-56.5)
    assert water.error == pytest.approx(1.3)
    assert water.doi == "10.1021/acs.jctc.5b01047"
    assert carbon_monoxide.om2 == pytest.approx(-20.3)
    assert water_dimer.error == pytest.approx(-1.98)
    assert len(all_molecules) == len(g2_molecules) + len(s22_molecules)
    assert "triplet methylene" in molecule_table


def test_om2_raises_clear_error_for_unsupported_default_element():
    from pyqed.qchem.semiempirical import OM2, OM2ParameterError

    with pytest.raises(OM2ParameterError, match="No OM2 parameters"):
        OM2(atom="Cl 0 0 0; H 0 0 1.3", unit="angstrom").run()


def test_mrci_diagonalizes_supplied_dense_hamiltonian():
    from pyqed.qchem.semiempirical import MRCI

    class ToyReference:
        h_ci = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.5, 0.2],
                [0.0, 0.2, 0.8],
            ]
        )

    driver = MRCI(ToyReference(), nstates=2).run()
    expected, _ = np.linalg.eigh(ToyReference.h_ci)

    np.testing.assert_allclose(driver.e, expected[:2])
    np.testing.assert_allclose(driver.e_tot, expected[:2])
    np.testing.assert_allclose(driver.e_elec, expected[:2])
    assert driver.ci.shape == (3, 2)
    assert driver.nstates == 2


def test_om2_uses_injected_reference_builder_for_development():
    from pyqed.qchem.semiempirical import OM2

    class ToyReference:
        e_tot = -1.0
        h_ci = np.diag([0.0, 0.4, 0.9])

    class ToyParameters:
        def build_reference(self, om2, **kwargs):
            assert om2.mol is not None
            return ToyReference()

    om2 = OM2(atom="H 0 0 0", parameters=ToyParameters()).run()
    mrci = om2.MRCI(nstates=2).run()

    np.testing.assert_allclose(mrci.e, [0.0, 0.4])
    assert om2.e_tot == -1.0


def test_om2_builds_valence_orbitals_from_atom_spec():
    from pyqed.qchem.semiempirical import OM2
    from pyqed.qchem.semiempirical.om2 import ANGSTROM_TO_BOHR

    om2 = OM2(atom="C 0 0 0; H 0 0 1.1", unit="angstrom").build()

    assert [orb.label for orb in om2.valence_orbitals()] == [
        "C1:2s",
        "C1:2px",
        "C1:2py",
        "C1:2pz",
        "H2:1s",
    ]
    assert om2.atom_symbols() == ("C", "H")
    assert om2.atom_coords().shape == (2, 3)
    np.testing.assert_allclose(
        np.linalg.norm(om2.atom_coords()[1] - om2.atom_coords()[0]),
        1.1 * ANGSTROM_TO_BOHR,
    )


def test_om2_parameter_set_builds_zero_order_hamiltonian_data():
    from pyqed.qchem.semiempirical import OM2, OM2ParameterError, OM2ParameterSet

    params = OM2ParameterSet(
        {
            "H": {"uss": -11.0, "beta_s": -1.0, "core_charge": 1, "gamma_ss": 1.0},
            "C": {
                "uss": -20.0,
                "upp": -10.0,
                "beta_s": -1.0,
                "beta_p": -0.5,
                "beta_pi": -0.4,
                "core_charge": 4,
                "gamma_ss": 1.0,
                "gamma_pp": 0.8,
            },
        }
    )
    om2 = OM2(atom="C 0 0 0; H 0 0 1.1", parameters=params)
    data = om2.build_hamiltonian_data()

    assert data.orbital_labels == ("C1:2s", "C1:2px", "C1:2py", "C1:2pz", "H2:1s")
    assert np.all(np.diag(data.hcore) < np.array([-20.0, -10.0, -10.0, -10.0, -11.0]))
    assert data.nelec == 5

    with pytest.raises(OM2ParameterError, match="closed-shell"):
        om2.run()


def test_default_om2_kernel_uses_xh_and_ecp_rows():
    from pyqed.qchem.semiempirical import OM2

    ch = OM2(atom="C 0 0 0; H 0 0 1.1", unit="angstrom")
    data = ch.build_hamiltonian_data()
    labels = data.orbital_labels
    idx = {label: i for i, label in enumerate(labels)}

    assert data.hcore[idx["C1:2px"], idx["H2:1s"]] == pytest.approx(0.0, abs=1e-12)
    assert data.hcore[idx["C1:2py"], idx["H2:1s"]] == pytest.approx(0.0, abs=1e-12)
    assert abs(data.hcore[idx["C1:2pz"], idx["H2:1s"]]) > 1e-6
    assert data.hcore[idx["H2:1s"], idx["H2:1s"]] < -12.64890000 / 27.211386245988


def test_om2_can_enable_three_center_orthogonalization_correction():
    from pyqed.qchem.semiempirical import OM2, OM2ParameterSet

    atom = "C 0 0 0; H 0 0 1.1; O 1.2 0 0"
    full = OM2(atom=atom, unit="angstrom", orthogonalization_correction=True).build_hamiltonian_data().hcore
    no_ortho_params = OM2ParameterSet(
        {
            symbol: {
                **params.__dict__,
                "f1": 0.0,
                "f2": 0.0,
                "g1": 0.0,
                "g2": 0.0,
            }
            for symbol, params in OM2(atom=atom, unit="angstrom").parameters.elements.items()
        }
    )
    no_ortho = OM2(atom=atom, unit="angstrom", parameters=no_ortho_params).build_hamiltonian_data().hcore

    assert np.linalg.norm(full - no_ortho) > 1e-8
    np.testing.assert_allclose(full, full.T, atol=1e-12)


def test_om2_default_reference_runs_closed_shell_h2():
    from pyqed.qchem.semiempirical import OM2

    om2 = OM2(atom="H 0 0 0; H 0 0 0.74", unit="angstrom").run()

    assert np.isfinite(om2.e_tot)
    assert om2.reference.converged
    assert om2.mo_coeff.shape == (2, 2)
    assert om2.hamiltonian_data.eri.shape == (2, 2, 2, 2)


def test_om2_can_disable_approximate_orthogonalization_correction():
    from pyqed.qchem.semiempirical import OM2

    atom = "C 0 0 0; O 1.128 0 0"
    full = OM2(atom=atom, unit="angstrom", orthogonalization_correction=True).run()
    plain = OM2(atom=atom, unit="angstrom").run()

    assert full.e_tot != pytest.approx(plain.e_tot)
    assert full.orthogonalization_correction is True
    assert plain.orthogonalization_correction is False


def test_om2_uses_distinct_one_center_gamma_parameters():
    from pyqed.qchem.semiempirical import OM2

    data = OM2(atom="C 0 0 0", unit="angstrom").build_hamiltonian_data()
    eri = data.eri

    assert eri[0, 0, 0, 0] != pytest.approx(eri[0, 0, 1, 1])
    assert eri[0, 0, 1, 1] != pytest.approx(eri[1, 1, 1, 1])
    assert eri[1, 1, 2, 2] != pytest.approx(eri[1, 1, 1, 1])


def test_om2_mrci_builds_selected_configurations_and_overlap():
    from pyqed.qchem.semiempirical import OM2

    om2 = OM2(atom="H 0 0 0; H 0 0 0.74", unit="angstrom").run()
    mrci1 = om2.MRCI(nstates=2, full=True).run()
    mrci2 = om2.MRCI(nstates=2, full=True).run()

    assert mrci1.e.shape == (2,)
    assert mrci1.ci.shape[1] == 2
    np.testing.assert_allclose(np.abs(mrci1.wavefunction_overlap(mrci2)), np.eye(2), atol=1e-10)


def _hf_determinant_ci_total_energy(reference, active_orbitals=None):
    from pyqed.qchem.semiempirical import MRCI

    driver = MRCI(reference, nstates=1, active_orbitals=active_orbitals, full=True)
    hamiltonian, _ref = driver._dense_hamiltonian()
    nocc = int(np.count_nonzero(np.asarray(reference.mo_occ) > 1.0e-8))
    hf_occupation = np.zeros_like(driver.determinants[0])
    hf_occupation[:, :nocc] = 1
    matches = np.where(np.all(driver.determinants == hf_occupation[None, :, :], axis=(1, 2)))[0]
    assert len(matches) == 1
    return float(hamiltonian[matches[0], matches[0]] + reference.energy_nuc())


def test_om2_active_space_mrci_contains_consistent_hf_determinant_energy():
    from pyqed.qchem.semiempirical import OM2

    om2 = OM2(atom="C 0 0 0; O 1.128 0 0", unit="angstrom").run()
    ref = om2.reference
    assert ref.converged
    nocc = int(np.count_nonzero(ref.mo_occ > 1.0e-8))
    active_orbitals = tuple(range(nocc - 2, nocc + 2))

    ci_hf_energy = _hf_determinant_ci_total_energy(ref, active_orbitals=active_orbitals)

    np.testing.assert_allclose(ci_hf_energy, ref.e_tot, atol=1.0e-8)


def test_om2_mrci_rejects_unconverged_reference():
    from pyqed.qchem.semiempirical import OM2

    # This distorted azomethane-like geometry converges poorly with the compact
    # native OM2 SCF damping.  MRCI must not use inconsistent damped-density
    # energies and final-orbital determinants from such a reference.
    atom = (
        "C -1.229657660272 1.296221544263 -0.000000000000; "
        "N -1.181078827891 0.000000000000 0.000000000000; "
        "N 1.181078827891 0.000000000000 0.000000000000; "
        "C 1.229657660272 -0.993012662503 0.833231140149; "
        "H -0.569040062390 1.453356420929 0.853323450060; "
        "H -2.240575399824 1.462061036334 0.338265630386; "
        "H -0.879357518011 1.850114581795 -0.873817438605; "
        "H 0.597812634428 -1.424037702542 0.067006464063; "
        "H 2.255226551094 -1.151245963874 0.540934028376; "
        "H 0.896375414973 -0.949631837023 1.869985151717"
    )
    om2 = OM2(atom=atom, unit="bohr").run(max_cycle=20)

    assert not om2.reference.converged
    with pytest.raises(RuntimeError, match="OM2 reference is not converged"):
        om2.MECI(nstates=3, ncas=4).run()


def test_om2_mrci_scanner_returns_result_object():
    from pyqed.qchem.semiempirical import OM2

    scanner = OM2(atom="H 0 0 0; H 0 0 0.74", unit="angstrom").as_scanner(nstates=1, full=True)
    result = scanner(atom="H 0 0 0; H 0 0 0.80")

    assert result.e.shape == (1,)
    assert result.ci.shape[1] == 1
