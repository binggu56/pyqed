import numpy as np
from pyscf import gto, scf, tdscf

from pyqed.qchem import Molecule, RTTDHF
from pyqed.qchem.hf import RHF
from pyqed.qchem.hf.rhf import get_jk as pyqed_get_jk


def test_rttdhf_ground_state_is_stationary_without_field():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    rt = RTTDHF(mf).run(dt=0.05, nsteps=5, store_dm=True)

    np.testing.assert_allclose(rt.dms[-1], rt.dms[0], atol=1e-8)
    np.testing.assert_allclose(rt.energies[-1], rt.energies[0], atol=1e-8)
    np.testing.assert_allclose(rt.electron_count(), mol.nelec, atol=1e-8)


def test_rttdhf_small_kick_shows_spectral_weight_near_pyscf_tdhf_root():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')
    mf = RHF(mol).run()

    pyscf_mol = gto.M(atom=atom, unit='Bohr', basis='sto-3g')
    pyscf_mf = scf.RHF(pyscf_mol).run()
    ref_root = tdscf.TDHF(pyscf_mf).kernel(nstates=1)[0][0]

    dt = 0.02
    nsteps = 1000

    rt_ref = RTTDHF(mf).run(dt=dt, nsteps=nsteps, store_dm=False)
    rt_kick = RTTDHF(mf).run(
        dt=dt,
        nsteps=nsteps,
        store_dm=False,
        kick={'strength': 1e-4, 'axis': 'z'},
    )

    baseline = rt_ref.dipoles[:, 2] - rt_ref.dipoles[0, 2]
    signal = rt_kick.dipoles[:, 2] - rt_kick.dipoles[0, 2]

    window = np.hanning(signal.size)
    freq = 2 * np.pi * np.fft.rfftfreq(signal.size, d=dt)
    spec0 = np.abs(np.fft.rfft(baseline * window))
    spec1 = np.abs(np.fft.rfft(signal * window))

    idx = np.argmin(np.abs(freq - ref_root))
    assert abs(freq[idx] - ref_root) < 0.05
    assert spec1[idx] > 1e-3
    assert spec1[idx] > 100.0 * spec0[idx] + 1e-4


def test_complex_jk_matches_pyscf_for_rt_coherences():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')
    RHF(mol).run()

    pyscf_mol = gto.M(atom=atom, unit='Bohr', basis='sto-3g')
    pyscf_mf = scf.RHF(pyscf_mol).run()

    nao = pyscf_mol.nao_nr()
    dm = np.zeros((nao, nao), dtype=complex)
    dm[0, 0] = 0.3
    dm[1, 1] = -0.1
    dm[0, 1] = 0.1 + 0.2j
    dm[1, 0] = 0.1 - 0.2j

    vj_ref, vk_ref = pyscf_mf.get_jk(dm=dm)
    vj, vk = pyqed_get_jk(mol, dm)

    np.testing.assert_allclose(vj, vj_ref, atol=1e-8)
    np.testing.assert_allclose(vk, vk_ref, atol=1e-8)
