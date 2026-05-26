import numpy as np
import pytest

from pyqed.qchem import CASCI, CD, Molecule, RHF, TDA, TDDFT


def _h2_casci(nstates=2, driver='builtin'):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    if driver == 'builtin':
        mol.build(driver=driver, eri='s8')
    else:
        mol.build(driver=driver)
    mf = RHF(mol).run()
    return CASCI(mf, ncas=2, nelecas=2).run(nstates=nstates)


def _chiral_methyl_lactate_molecule():
    atom = (
        'C 0.000 0.000 0.000; '
        'H 0.620 0.620 0.620; '
        'O -0.950 0.450 0.850; '
        'H -1.500 1.000 0.350; '
        'C -0.500 -1.420 0.200; '
        'H -1.100 -1.680 -0.670; '
        'H 0.350 -2.100 0.270; '
        'H -1.120 -1.550 1.090; '
        'C 1.180 0.140 -0.980; '
        'O 1.300 -0.180 -2.160; '
        'O 2.120 0.720 -0.250; '
        'C 3.350 0.910 -0.930; '
        'H 3.250 1.250 -1.960; '
        'H 3.930 -0.010 -0.900; '
        'H 3.890 1.670 -0.370'
    )
    mol = Molecule(atom=atom, unit='angstrom', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')
    return mol


def test_builtin_angular_momentum_and_magnetic_dipole_integrals_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')

    center = mol.nuc_charge_center()
    position = mol.position_integral(center=center)
    moment = mol.moment_integral(center=center)
    rxgrad = mol.rxgrad_integral(center=center)
    angular = mol.angular_momentum_integral(center=center)
    magnetic = mol.magnetic_dipole_integral(center=center)
    magnetic_raw = mol.magnetic_dipole_integral(center=center, convention='raw')
    magnetic_operator = mol.magnetic_dipole_integral(center=center, convention='operator')

    assert position.shape == (3, mol.nao, mol.nao)
    assert rxgrad.shape == (3, mol.nao, mol.nao)
    assert angular.shape == (3, mol.nao, mol.nao)
    assert magnetic.shape == (3, mol.nao, mol.nao)
    np.testing.assert_allclose(moment, position, atol=1e-12)
    np.testing.assert_allclose(angular, -1j * rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic, -0.5 * rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic_raw, -rxgrad, atol=1e-12)
    np.testing.assert_allclose(magnetic_operator, 0.5j * rxgrad, atol=1e-12)


def test_cd_from_casci_backend_builtin_h2():
    mc = _h2_casci()

    result = CD(mc).run()

    assert result.ground == 0
    np.testing.assert_array_equal(result.states, np.array([1]))
    assert result.excitation_energies.shape == (1,)
    assert result.electric_dipoles.shape == (1, 3)
    assert result.magnetic_dipoles.shape == (1, 3)
    assert result.rotatory_strengths.shape == (1,)
    assert result.oscillator_strengths.shape == (1,)
    assert np.all(np.isfinite(result.excitation_energies))
    assert np.all(np.isfinite(result.electric_dipoles))
    assert np.all(np.isfinite(result.magnetic_dipoles))
    assert np.all(np.isfinite(result.rotatory_strengths))


def test_cd_accepts_scalar_target_state():
    mc = _h2_casci()

    result = CD(mc).run(states=1)

    np.testing.assert_array_equal(result.states, np.array([1]))
    assert result.electric_dipoles.shape == (1, 3)


def test_cd_rejects_empty_target_states():
    mc = _h2_casci()

    with pytest.raises(ValueError, match="at least one excited state"):
        CD(mc).run(states=[])


def test_cd_spectrum_from_casci_backend_builtin_h2():
    mc = _h2_casci()

    cd = CD(mc)
    x, signal = cd.spectrum(width=0.2, units='ev')

    assert x.shape == signal.shape
    assert x.ndim == 1
    assert x.size == 1000
    assert np.all(np.isfinite(signal))


def test_cd_spectrum_validates_lineshape_before_broadening():
    mc = _h2_casci()

    with pytest.raises(ValueError, match="lineshape"):
        CD(mc).spectrum(width=0.2, units='ev', lineshape='triangle')


def test_cd_from_tda_pcm_backend_h2():
    pytest.importorskip('pyscf.solvent')

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    # This TDDFT/PCM test deliberately uses PySCF-built orbitals.
    mol.build(driver='pyscf')
    mf = RHF(mol).run()

    gas_td = TDA(mf).run(nstates=1)
    pcm_td = TDA(mf).PCM().run(nstates=1)
    pcm_tddft = TDDFT(mf).PCM().run(nstates=1)

    gas_result = CD(gas_td).run()
    pcm_result = CD(pcm_td).run()
    pcm_tddft_result = CD(pcm_tddft).run()

    np.testing.assert_array_equal(gas_result.states, np.array([1]))
    np.testing.assert_array_equal(pcm_result.states, np.array([1]))
    assert gas_result.excitation_energies.shape == (1,)
    assert pcm_result.electric_dipoles.shape == (1, 3)
    assert pcm_result.magnetic_dipoles.shape == (1, 3)
    assert pcm_result.rotatory_strengths.shape == (1,)
    assert pcm_result.oscillator_strengths.shape == (1,)
    assert np.all(np.isfinite(pcm_result.rotatory_strengths))
    assert pcm_td.with_solvent.eps == pytest.approx(1.78)
    assert pcm_tddft.with_solvent.eps == pytest.approx(1.78)
    assert pcm_td.pcm_response_kernel.shape == (
        pcm_td.nocc,
        pcm_td.nvir,
        pcm_td.nocc,
        pcm_td.nvir,
    )
    assert pcm_tddft_result.electric_dipoles.shape == (1, 3)
    assert np.all(np.isfinite(pcm_tddft_result.rotatory_strengths))
    assert abs(pcm_result.excitation_energies[0] - gas_result.excitation_energies[0]) > 1e-8

    x, signal = CD(pcm_td).spectrum(width=0.2, units='ev')
    assert x.shape == signal.shape
    assert np.all(np.isfinite(signal))

    with pytest.raises(ValueError, match="attach PCM"):
        CD(pcm_td).run(solvent_response='lr_pcm')


def test_tddft_pcm_cd_matches_pyscf_h2o2():
    pyscf_gto = pytest.importorskip('pyscf.gto')
    pyscf_scf = pytest.importorskip('pyscf.scf')
    pytest.importorskip('pyscf.solvent')

    atom = (
        'O 0.000000 0.000000 0.000000; '
        'O 1.450000 0.000000 0.000000; '
        'H -0.450000 0.760000 0.000000; '
        'H 1.900000 0.760000 0.600000'
    )

    pmol = pyscf_gto.M(atom=atom, unit='Angstrom', basis='sto-3g', verbose=0)
    pmf = pyscf_scf.RHF(pmol).PCM()
    pmf.with_solvent.lebedev_order = 3
    pmf.with_solvent.verbose = 0
    pmf.run(verbose=0)

    ptd = pmf.TDHF(equilibrium_solvation=False)
    ptd.nstates = 3
    ptd.kernel()

    mol = Molecule(atom=atom, unit='angstrom', basis='sto-3g')
    # Keep PySCF AO ordering because the reference MO/TD data are injected below.
    mol.build(driver='pyscf')
    mf = RHF(mol)
    mf.mo_energy = np.array(pmf.mo_energy)
    mf.mo_coeff = np.array(pmf.mo_coeff)
    mf.mo_occ = np.array(pmf.mo_occ)
    mf.dm = np.array(pmf.make_rdm1())
    mf.hcore = np.array(mol.hcore)
    mf.vhf = np.array(pmf.get_veff(dm=pmf.make_rdm1()))
    mf.e_tot = float(pmf.e_tot)
    mf.e_nuc = float(pmol.energy_nuc())

    from pyqed.qchem.solvent.pcm import PCM

    pcm = PCM(mol)
    pcm.lebedev_order = 3
    pcm.verbose = 0
    td = TDDFT(mf).PCM(solvent_obj=pcm).run(nstates=3)

    cd = CD(td)
    result = cd.run()
    pyscf_rotatory = -np.einsum(
        'nx,nx->n',
        -ptd.transition_dipole(),
        0.5 * ptd.transition_magnetic_dipole(),
    )

    np.testing.assert_allclose(td.e, ptd.e, atol=1e-7)
    np.testing.assert_allclose(result.rotatory_strengths, pyscf_rotatory, atol=1e-7)
    np.testing.assert_allclose(cd.rotatory_strengths, result.rotatory_strengths, atol=0.0)
    assert cd.result is result


def test_chiral_cd_spectra_for_gas_phase_and_pcm_casci_backends(tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mol = _chiral_methyl_lactate_molecule()
    mf = RHF(mol).run()

    gas_mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=10)
    pcm_mc = CASCI(mf, ncas=4, nelecas=4).PCM(max_cycle=2).run(nstates=10)
    pcm_sa_mc = CASCI(mf, ncas=4, nelecas=4).PCM(
        max_cycle=2,
        state_average=True,
    ).run(nstates=10)
    pcm_det_mc = CASCI(mf, ncas=4, nelecas=4).PCM(max_cycle=2).run(
        nstates=10,
        solvent_response='lr_pcm',
    )

    gas_cd = CD(gas_mc)
    pcm_cd = CD(pcm_mc)
    pcm_sa_cd = CD(pcm_sa_mc)
    pcm_det_cd = CD(pcm_det_mc)
    gas_result = gas_cd.run()
    pcm_result = pcm_cd.run(solvent_response='lr_pcm')
    pcm_sa_result = pcm_sa_cd.run()
    pcm_det_result = pcm_det_cd.run()

    assert pcm_mc.with_solvent.max_cycle == 2
    assert pcm_mc.with_solvent.e is not None
    assert pcm_mc.with_solvent.v is not None
    assert pcm_sa_mc.with_solvent.state_average is True
    assert pcm_det_mc.lr_pcm_response_eps == pytest.approx(1.78)
    assert pcm_det_mc.lr_pcm_response_matrix.shape == (36, 36)
    assert pcm_det_mc.lr_pcm_raw_e_tot.shape == (10,)

    for result in (gas_result, pcm_result, pcm_sa_result, pcm_det_result):
        assert result.excitation_energies.shape == (9,)
        assert result.electric_dipoles.shape == (9, 3)
        assert result.magnetic_dipoles.shape == (9, 3)
        assert result.rotatory_strengths.shape == (9,)
        assert np.all(np.isfinite(result.excitation_energies))
        assert np.all(np.isfinite(result.electric_dipoles))
        assert np.all(np.isfinite(result.magnetic_dipoles))
        assert np.all(np.isfinite(result.rotatory_strengths))
        assert np.any(np.abs(result.rotatory_strengths) > 1e-6)

    assert pcm_result.solvent_response_model == 'lr_pcm'
    assert pcm_result.solvent_response_eps == pytest.approx(1.78)
    assert pcm_result.solvent_response_energies.shape == (9,)
    assert pcm_result.solvent_response_corrections.shape == (9,)
    assert pcm_result.solvent_response_matrix.shape == (9, 9)
    assert pcm_result.solvent_response_vectors.shape == (9, 9)
    assert pcm_result.solvent_response_electric_dipoles.shape == (9, 3)
    assert pcm_result.solvent_response_magnetic_dipoles.shape == (9, 3)
    assert pcm_result.solvent_response_rotatory_strengths.shape == (9,)
    assert pcm_result.solvent_response_oscillator_strengths.shape == (9,)
    assert np.all(np.isfinite(pcm_result.solvent_response_energies))
    assert np.all(np.isfinite(pcm_result.solvent_response_corrections))
    assert np.all(np.isfinite(pcm_result.solvent_response_matrix))
    assert np.all(np.isfinite(pcm_result.solvent_response_rotatory_strengths))
    np.testing.assert_allclose(
        pcm_result.solvent_response_matrix,
        pcm_result.solvent_response_matrix.T,
        atol=1e-12,
    )
    assert np.any(np.abs(pcm_result.solvent_response_corrections) > 1e-6)
    assert np.max(np.abs(pcm_result.excitation_energies - gas_result.excitation_energies)) > 1e-5
    assert np.max(np.abs(pcm_sa_result.excitation_energies - pcm_result.excitation_energies)) > 1e-6
    assert np.max(np.abs(pcm_det_result.excitation_energies - pcm_result.excitation_energies)) > 1e-5
    assert np.max(np.abs(pcm_det_result.rotatory_strengths - pcm_result.rotatory_strengths)) > 1e-5

    x = np.linspace(4.0, 16.0, 1000)
    _, gas_signal = gas_cd.spectrum(x=x, width=0.18, units='ev', result=gas_result)
    _, pcm_signal = pcm_cd.spectrum(x=x, width=0.18, units='ev', result=pcm_result)
    _, pcm_lr_signal = pcm_cd.spectrum(
        x=x,
        width=0.18,
        units='ev',
        result=pcm_result,
        energy_source='lr_pcm',
    )
    _, pcm_det_signal = pcm_det_cd.spectrum(x=x, width=0.18, units='ev', result=pcm_det_result)

    fig, ax = plt.subplots()
    ax.axhline(0.0, color='0.75', linewidth=0.8)
    ax.plot(x, gas_signal, label='gas phase')
    ax.plot(x, pcm_signal, '--', label='PCM raw')
    ax.plot(x, pcm_lr_signal, ':', label='PCM LR subspace')
    ax.plot(x, pcm_det_signal, '-.', label='PCM LR determinant')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('CD intensity (arb.)')
    ax.legend()
    fig.tight_layout()

    plot_path = tmp_path / 'methyl_lactate_cd_gas_pcm_lr_det.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    assert plot_path.stat().st_size > 0
