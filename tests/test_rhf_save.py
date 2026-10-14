import json
import zipfile
import numpy as np
import pytest
from pyqed.qchem import Molecule, RHF


def test_save_compressed_results(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 .74', unit='angstrom', basis='sto-3g')
    mol.build(eri='s8')
    mf = RHF(mol)
    path = tmp_path/'rhf.npz'
    with pytest.raises(RuntimeError, match='Run RHF'):
        mf.save(path)
    assert not path.exists()
    mf.run(tol=1e-10)
    assert mf.save(path) == path
    with np.load(path, allow_pickle=False) as saved:
        for key in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot', 'converged'):
            np.testing.assert_array_equal(saved[key], getattr(mf, key))
        np.testing.assert_array_equal(saved['atom_coords'], mol.atom_coords())
        np.testing.assert_array_equal(saved['atom_symbols'], mol.atom_symbols())
        metadata = json.loads(saved['metadata'].item())
        assert metadata['coordinate_unit'] == 'bohr'
        assert metadata['basis'] == 'sto-3g'
        assert metadata['charge'] == metadata['spin'] == 0
        assert all(saved[key].dtype.kind != 'O' for key in saved.files)
        assert not any('eri' in key for key in saved.files)
    with zipfile.ZipFile(path) as archive:
        assert all(entry.compress_type == zipfile.ZIP_DEFLATED for entry in archive.infolist())
