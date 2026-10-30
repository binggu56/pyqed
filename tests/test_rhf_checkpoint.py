"""Opt-in RHF density checkpoints preserve inputs and validate restarts."""
import numpy as np
import pytest

h5py = pytest.importorskip('h5py')
from pyqed.qchem import Molecule, RHF
from pyqed.qchem.hf import checkpoint


def molecule(distance=.757):
    return Molecule(atom=f'O 0 0 0; H 0 {-distance} .587; H 0 {distance} .587',
                    unit='angstrom', basis='sto-3g').build(eri='s8')


def test_opt_in_checkpoint_and_density_restart(tmp_path):
    mol = molecule()
    mf = RHF(mol)
    assert mf.chkfile is None
    mf.chkfile = tmp_path/'rhf.chk'
    mf.run(max_cycle=1, stability_analysis=False)
    with h5py.File(mf.chkfile) as saved:
        assert saved.attrs['cycle'] == 1
        assert not saved.attrs['converged']
        np.testing.assert_array_equal(saved['dm'][()], mf.dm)
    resumed = RHF(mol)
    resumed.chkfile = mf.chkfile
    resumed.run(restart=True, tol=1e-10, stability_analysis=False)
    reference = RHF(mol).run(tol=1e-10, stability_analysis=False)
    assert resumed.converged
    assert resumed.e_tot == pytest.approx(reference.e_tot, abs=1e-9)
    before = mf.chkfile.read_bytes()
    changed = RHF(molecule(.8))
    changed.chkfile = mf.chkfile
    with pytest.raises(ValueError, match='does not match'):
        changed.run(restart=True)
    assert mf.chkfile.read_bytes() == before


def test_checkpoint_write_is_atomic(tmp_path, monkeypatch):
    mf = RHF(molecule())
    mf.chkfile = tmp_path/'rhf.chk'
    mf.run(stability_analysis=False)
    before = mf.chkfile.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('write failed')
    monkeypatch.setattr(h5py.Group, 'create_dataset', fail)
    with pytest.raises(OSError, match='write failed'):
        checkpoint.write(mf.chkfile, mf.mol, e_tot=mf.e_tot,
            mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy, mo_occ=mf.mo_occ, dm=mf.dm)
    assert mf.chkfile.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
