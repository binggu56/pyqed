import numpy as np

from pyqed.gw.gw import GW
from pyqed.gw.molgw_benchmark import (
    MOLGWSpectralData,
    compare_molgw_spectral_function,
    load_molgw_spectral_function,
)
from pyqed.gw.pes import PES, ao_plane_wave_dipoles, nuclear_center_of_mass, quasiparticle_weights
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.units import au2ev


def test_plane_wave_dipole_centered_s_gaussian_has_zero_k_zero_dipole():
    mol = Molecule(atom="H 0 0 0", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="dense")

    dipoles = ao_plane_wave_dipoles(mol, kvec=[0.0, 0.0, 0.0], origin=[0.0, 0.0, 0.0])

    assert dipoles.shape == (mol.nao, 3)
    np.testing.assert_allclose(dipoles, np.zeros_like(dipoles), atol=1e-12)


def test_default_pes_origin_is_nuclear_center_of_mass():
    mol = Molecule(atom="O 0 0 0; H 0 0 1.8", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="dense")

    center_of_mass = mol.center_of_mass()
    assert not np.allclose(center_of_mass, mol.nuc_charge_center())
    np.testing.assert_allclose(nuclear_center_of_mass(mol), center_of_mass)

    dipoles_default = ao_plane_wave_dipoles(mol, kvec=[0.1, 0.2, 0.3])
    dipoles_explicit = ao_plane_wave_dipoles(mol, kvec=[0.1, 0.2, 0.3], origin=center_of_mass)
    np.testing.assert_allclose(dipoles_default, dipoles_explicit)


def test_gw_photoelectron_spectrum_h2_smoke():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="angstrom")
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-3).run()

    analyzer = gw.pes(photon_energy=40.0, units="ev")
    assert isinstance(analyzer, PES)
    pes = analyzer.run(ndirections=6)

    assert pes.orbitals.shape == (1,)
    assert pes.binding_energies.shape == (1,)
    assert pes.kinetic_energies.shape == (1,)
    assert pes.intensities.shape == (1,)
    assert pes.binding_energies[0] > 0.0
    assert pes.kinetic_energies[0] > 0.0
    assert pes.intensities[0] >= 0.0
    assert pes.intensity_kind == "matrix_element"
    assert pes.intensity_units == "arb."
    np.testing.assert_allclose(pes.binding_energies[0], -gw.e_qp[0], atol=0.0)

    pes_cross = analyzer.run(ndirections=6, intensity="cross_section")
    assert pes_cross.intensity_kind == "cross_section"
    assert pes_cross.intensity_units == "a0^2"
    np.testing.assert_allclose(
        pes_cross.intensities,
        pes.intensities * pes_cross.cross_section_prefactors,
    )

    pes_normalized = analyzer.run(ndirections=6, intensity="normalized")
    assert pes_normalized.intensity_kind == "normalized"
    assert pes_normalized.intensity_units == "normalized"
    np.testing.assert_allclose(pes_normalized.intensities.max(), 1.0)

    direction = np.array([0.0, 0.0, 1.0])
    polarization = np.array([1.0, 0.0, 0.0])
    dipole = analyzer.transition_dipole(0, pes.kinetic_energies[0], direction)
    moment = analyzer.transition_moment(0, pes.kinetic_energies[0], direction, polarization)
    np.testing.assert_allclose(moment, np.dot(polarization, dipole))

    z = quasiparticle_weights(gw, orbitals=[0])
    assert z.shape == (1,)
    assert np.isfinite(z[0])
    assert z[0] > 0.0

    pes_z = gw.pes(photon_energy=40.0, units="ev").run(ndirections=4, qp_weight="gw")
    np.testing.assert_allclose(pes_z.qp_weights, z)

    pes_qp = gw.pes(photon_energy=40.0, units="ev").run(ndirections=4, dyson="qp")
    assert pes_qp.dyson_kind == "qp"
    np.testing.assert_allclose(pes_qp.qp_weights, z)
    np.testing.assert_allclose(pes_qp.intensities, pes_z.intensities)

    dyson = gw.dyson_orbital(orbital=0)
    assert dyson.converged
    np.testing.assert_allclose(dyson.energy, gw.e_qp[0], atol=1e-5)
    np.testing.assert_allclose(np.vdot(dyson.mo_coefficients, dyson.mo_coefficients).real, dyson.qp_weight)
    assert abs(dyson.mo_coefficients[0]) > 0.9 * np.sqrt(dyson.qp_weight)

    pes_matrix = gw.pes(photon_energy=40.0, units="ev").run(ndirections=4, dyson="matrix")
    assert pes_matrix.dyson_kind == "matrix"
    np.testing.assert_allclose(pes_matrix.qp_weights, [dyson.qp_weight], rtol=1e-6)
    np.testing.assert_allclose(pes_matrix.intensities, pes_qp.intensities, rtol=1e-5, atol=1e-10)

    x, signal = analyzer.spectrum(width=0.3)
    assert x.shape == signal.shape
    assert x.size == 1000
    assert np.all(np.isfinite(signal))

    eta0 = gw.eta
    binding_grid = np.linspace(10.0, 25.0, 41)
    spec = analyzer.spectral_function(
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
    )
    assert gw.eta == eta0
    assert spec.spectral_function.shape == (1, binding_grid.size)
    assert np.all(np.isfinite(spec.spectral_function))
    assert spec.spectral_function.max() > 0.0
    peak_binding = binding_grid[spec.spectral_function[0].argmax()]
    assert abs(peak_binding - (-gw.e_qp[0] * au2ev)) < 1.0
    spec_peaks = spec.peaks(source="spectral_function", threshold_rel=0.2)
    assert spec_peaks.source == "spectral_function"
    assert spec_peaks.units == "ev"
    assert spec_peaks.orbitals[0] == 0
    assert abs(spec_peaks.binding_energies[0] - (-gw.e_qp[0] * au2ev)) < 1.0

    satellite = analyzer.satellite_spectrum(
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
        ndirections=3,
    )
    assert satellite.signal.shape == binding_grid.shape
    assert satellite.spectral_function.shape == (1, binding_grid.size)
    assert np.all(np.isfinite(satellite.signal))
    assert satellite.signal.max() > 0.0
    assert satellite.averaging == "orientation_spectral_function"
    satellite_peaks = satellite.peaks(source="signal", threshold_rel=0.2)
    assert satellite_peaks.source == "signal"
    assert satellite_peaks.orbitals[0] == -1
    assert abs(satellite_peaks.binding_energies[0] - (-gw.e_qp[0] * au2ev)) < 1.0

    direct_spec = gw.spectral_function(
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
    )
    np.testing.assert_allclose(direct_spec.spectral_function, spec.spectral_function)

    direct_satellite = gw.satellite_spectrum(
        photon_energy=40.0,
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
        ndirections=3,
    )
    np.testing.assert_allclose(direct_satellite.signal, satellite.signal)

    spectral_pes = gw.spectral_pes(
        photon_energy=40.0,
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
        ndirections=3,
        approx="diagonal",
    )
    assert spectral_pes.approximation == "diagonal"
    assert spectral_pes.averaging == "orientation_spectral_diagonal"
    np.testing.assert_allclose(spectral_pes.signal, satellite.signal)

    spectral_pes_matrix = gw.spectral_pes(
        photon_energy=40.0,
        binding_grid=binding_grid,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
        ndirections=3,
        approx="matrix",
    )
    assert spectral_pes_matrix.approximation == "matrix"
    assert spectral_pes_matrix.spectral_matrix.shape == (1, 1, binding_grid.size)
    np.testing.assert_allclose(
        spectral_pes_matrix.spectral_function,
        spec.spectral_function,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spectral_pes_matrix.signal,
        spectral_pes.signal,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gw_arpes_fixed_and_transverse_polarization_h2():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="angstrom")
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-3).run()

    analyzer = gw.pes(photon_energy=40.0, units="ev")
    direction = np.array([0.0, 0.0, 1.0])
    polarization = np.array([1.0, 0.0, 0.0])

    fixed = analyzer.arpes(direction=direction, polarization=polarization)
    moment = analyzer.transition_moment(
        fixed.orbitals[0],
        fixed.kinetic_energies[0],
        direction,
        polarization,
    )
    np.testing.assert_allclose(fixed.intensities[0], abs(moment) ** 2)
    np.testing.assert_allclose(fixed.direction, direction)
    np.testing.assert_allclose(fixed.polarization, polarization)
    assert fixed.averaging == "angle_resolved_fixed_polarization"

    fixed_cross = analyzer.arpes(direction=direction, polarization=polarization, intensity="cross_section")
    assert fixed_cross.intensity_kind == "cross_section"
    assert fixed_cross.intensity_units == "a0^2/sr"
    np.testing.assert_allclose(
        fixed_cross.intensities,
        fixed.intensities * fixed_cross.cross_section_prefactors,
    )

    fixed_z = analyzer.arpes(direction=direction, polarization=polarization, qp_weight="gw")
    fixed_qp = analyzer.arpes(direction=direction, polarization=polarization, dyson="qp")
    assert fixed_qp.dyson_kind == "qp"
    np.testing.assert_allclose(fixed_qp.qp_weights, fixed_z.qp_weights)
    np.testing.assert_allclose(fixed_qp.intensities, fixed_z.intensities)

    fixed_matrix = analyzer.arpes(direction=direction, polarization=polarization, dyson="matrix")
    assert fixed_matrix.dyson_kind == "matrix"
    np.testing.assert_allclose(fixed_matrix.intensities, fixed_qp.intensities, rtol=1e-5, atol=1e-10)

    direct_matrix = gw.arpes(
        photon_energy=40.0,
        units="ev",
        direction=direction,
        polarization=polarization,
        dyson="matrix",
    )
    assert direct_matrix.dyson_kind == "matrix"
    np.testing.assert_allclose(direct_matrix.intensities, fixed_matrix.intensities)

    transverse = analyzer.arpes(direction=direction)
    dx, dy, _ = analyzer.transition_dipole(
        transverse.orbitals[0],
        transverse.kinetic_energies[0],
        direction,
    )
    expected = 0.5 * (abs(dx) ** 2 + abs(dy) ** 2)
    np.testing.assert_allclose(transverse.intensities[0], expected)
    np.testing.assert_allclose(transverse.direction, direction)
    assert transverse.polarization is None
    assert transverse.averaging == "angle_resolved_transverse_polarization"


def test_matrix_dyson_orbital_and_arpes_h2o_smoke():
    mol = Molecule(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-3).run()

    homo = mol.nelec // 2 - 1
    dyson = gw.dyson_orbital(orbital=homo)
    dyson_batch = gw.dyson_orbitals(orbitals=[homo])
    assert dyson.converged
    assert len(dyson_batch) == 1
    assert dyson.orbital == homo
    assert dyson.ao_coefficients.shape == (mol.nao,)
    assert dyson.orbital_space.shape == (len(mf.mo_energy),)
    assert dyson.qp_weight > 0.0
    np.testing.assert_allclose(dyson_batch[0].ao_coefficients, dyson.ao_coefficients)
    np.testing.assert_allclose(dyson.energy, gw.e_qp[homo], atol=1e-8)
    np.testing.assert_allclose(
        np.vdot(dyson.mo_coefficients, dyson.mo_coefficients).real,
        dyson.qp_weight,
    )

    direction = np.array([0.0, 0.0, 1.0])
    polarization = np.array([1.0, 0.0, 0.0])
    arpes = gw.arpes(
        photon_energy=60.0,
        units="ev",
        orbitals=[homo],
        direction=direction,
        polarization=polarization,
        dyson="matrix",
    )
    assert arpes.dyson_kind == "matrix"
    assert arpes.intensities.shape == (1,)
    assert arpes.intensities[0] >= 0.0
    np.testing.assert_allclose(arpes.qp_weights, [dyson.qp_weight], rtol=1e-6)

    pes = gw.pes(photon_energy=60.0, units="ev").run(
        orbitals=[homo],
        ndirections=3,
        dyson="matrix",
    )
    assert pes.dyson_kind == "matrix"
    assert pes.intensities.shape == (1,)
    assert pes.intensities[0] >= 0.0
    np.testing.assert_allclose(pes.qp_weights, [dyson.qp_weight], rtol=1e-6)

    binding_grid = np.linspace(7.0, 13.0, 15)
    spec_matrix = gw.spectral_matrix(
        binding_grid=binding_grid,
        units="ev",
        orbitals=[homo - 1, homo],
        eta=0.1 / au2ev,
    )
    assert spec_matrix.spectral_matrix.shape == (2, 2, binding_grid.size)
    for epos in range(binding_grid.size):
        np.testing.assert_allclose(
            spec_matrix.spectral_matrix[:, :, epos],
            spec_matrix.spectral_matrix[:, :, epos].T.conjugate(),
            atol=1e-12,
        )

    spectral_pes_matrix = gw.spectral_pes(
        photon_energy=60.0,
        binding_grid=binding_grid,
        units="ev",
        orbitals=[homo - 1, homo],
        direction=direction,
        polarization=polarization,
        eta=0.1 / au2ev,
        approx="matrix",
    )
    assert spectral_pes_matrix.approximation == "matrix"
    assert spectral_pes_matrix.averaging == "angle_resolved_spectral_matrix"
    assert spectral_pes_matrix.signal.shape == binding_grid.shape
    assert np.all(np.isfinite(spectral_pes_matrix.signal))
    assert spectral_pes_matrix.signal.max() > 0.0


def test_molgw_spectral_benchmark_table_loader_and_comparison(tmp_path):
    table = tmp_path / "molgw_spectrum.dat"
    table.write_text(
        "# energy_eV A_homo\n"
        "10.0 0.0\n"
        "11.0 1.0\n"
        "12.0 0.0\n",
        encoding="utf-8",
    )

    molgw = load_molgw_spectral_function(table, orbitals=[0], units="ev", axis="binding")
    assert molgw.spectral_function.shape == (1, 3)
    np.testing.assert_allclose(molgw.energy, [10.0, 11.0, 12.0])

    class Result:
        orbitals = np.array([0])
        binding_energies = np.array([10.0, 11.0, 12.0]) / au2ev
        spectral_function = np.array([[0.0, 2.0, 0.0]])

    bench = compare_molgw_spectral_function(Result(), molgw, normalize="area")
    assert bench.orbitals[0] == 0
    np.testing.assert_allclose(bench.rms, [0.0], atol=1e-14)
    np.testing.assert_allclose(bench.max_abs, [0.0], atol=1e-14)

    raw_table = tmp_path / "molgw_spectrum_raw.dat"
    raw_table.write_text(
        "# energy_eV A_homo_eV^-1\n"
        f"10.0 0.0\n11.0 {2.0 / au2ev:.16e}\n12.0 0.0\n",
        encoding="utf-8",
    )
    raw_molgw = load_molgw_spectral_function(raw_table, orbitals=[0], units="ev", axis="binding")
    raw_bench = compare_molgw_spectral_function(Result(), raw_molgw, normalize=None)
    np.testing.assert_allclose(raw_bench.rms, [0.0], atol=1e-14)


def test_gw_h2_spectral_function_matches_molgw_selfenergy_reference():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)
    gw = GW(mf, screening="TDH", eta=1e-3).run()

    # MOLGW 3.4 RI-HF/G0W0 reference from selfenergy_GW_state001.dat generated with:
    # basis='cc-pVDZ', auxil_basis='cc-pVDZ-RI', gaussian_type='cart',
    # postscf='G0W0', print_sigma='yes', eta=0.05 eV, selfenergy_state_range=0.
    molgw_rows = np.array(
        [
            [-20.20281898, 0.00096100],
            [-18.84224973, 0.00223686],
            [-17.48168048, 0.00996150],
            [-17.34562355, 0.01260538],
            [-17.07350970, 0.02239363],
            [-16.80139585, 0.05025644],
            [-16.52928200, 0.19746618],
            [-16.47485923, 0.30406639],
            [-16.44764784, 0.39215539],
            [-16.42043646, 0.52340272],
            [-16.39322507, 0.72968397],
            [-16.36601369, 1.07566223],
            [-16.33880230, 1.70012849],
            [-16.31159092, 2.88971667],
            [-16.28437953, 4.91988343],
            [-16.25716815, 6.25771495],
            [-16.22995676, 4.63611637],
            [-16.20274538, 2.69580029],
            [-16.17553399, 1.59857567],
            [-16.14832261, 1.02089601],
            [-16.12111122, 0.69789651],
            [-16.09389984, 0.50362643],
            [-16.06668845, 0.37911794],
            [-16.03947707, 0.29505564],
            [-15.98505430, 0.19266517],
            [-15.71294045, 0.04960033],
            [-15.44082660, 0.02218403],
            [-15.16871275, 0.01250868],
            [-14.76054197, 0.00662631],
            [-13.39997272, 0.00181991],
            [-12.03940347, 0.00083490],
        ]
    )
    molgw = MOLGWSpectralData(
        energy=molgw_rows[:, 0],
        spectral_function=molgw_rows[:, 1][None, :],
        orbitals=np.array([0]),
        units="ev",
        axis="omega",
    )
    spec = gw.spectral_function(
        omega_grid=molgw.energy,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
    )
    spec_matrix = gw.spectral_matrix(
        omega_grid=molgw.energy,
        units="ev",
        orbitals=[0],
        eta=0.05 / au2ev,
    )
    np.testing.assert_allclose(
        spec_matrix.spectral_function,
        spec.spectral_function,
        rtol=1e-10,
        atol=1e-12,
    )

    bench = compare_molgw_spectral_function(
        spec,
        molgw,
        source="spectral_function",
        units="ev",
        axis="omega",
        normalize=None,
    )
    assert bench.relative_rms[0] < 2.0e-3
    assert bench.max_abs[0] < 1.2e-2
    np.testing.assert_allclose(
        spec.spectral_function[0].max() / au2ev,
        molgw.spectral_function[0].max(),
        rtol=3.0e-4,
    )
