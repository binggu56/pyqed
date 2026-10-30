"""Atomic RHF density-restart checkpoints, without integral tensors or DIIS history."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def molecule_metadata(mol):
    return json.dumps(dict(symbols=mol.atom_symbols(), coords=mol.atom_coords().tolist(),
                           unit='bohr', basis=mol.basis, charge=mol.charge,
                           spin=mol.spin, cart=mol.cart), sort_keys=True,
                      default=lambda value: value.tolist())


def write(filename, mol, *, e_tot, mo_coeff, mo_energy, mo_occ, dm,
          converged=False, cycle=0):
    import h5py
    path = Path(filename)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        with h5py.File(temporary, 'w') as handle:
            handle.attrs.update(format='pyqed.rhf.checkpoint', version=1,
                                molecule=molecule_metadata(mol), converged=bool(converged),
                                cycle=int(cycle), e_tot=float(e_tot))
            for key, value in dict(mo_coeff=mo_coeff, mo_energy=mo_energy,
                                   mo_occ=mo_occ, dm=dm).items():
                handle.create_dataset(key, data=value, compression='gzip', shuffle=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_density(filename, mol):
    """Validate molecular identity and return the saved AO density."""
    import h5py
    with h5py.File(filename, 'r') as handle:
        if handle.attrs.get('format') != 'pyqed.rhf.checkpoint' or handle.attrs.get('version') != 1:
            raise ValueError('Unsupported RHF checkpoint format')
        if handle.attrs['molecule'] != molecule_metadata(mol):
            raise ValueError('RHF checkpoint molecule or basis does not match')
        dm = handle['dm'][()]
    if dm.shape != (mol.nao, mol.nao) or not np.all(np.isfinite(dm)):
        raise ValueError('Invalid density in RHF checkpoint')
    if not np.allclose(dm, dm.conj().T, atol=1e-10, rtol=0):
        raise ValueError('Checkpoint density is not Hermitian')
    if not np.isclose(np.einsum('ij,ji->', dm, mol.overlap).real, mol.nelec, atol=1e-6):
        raise ValueError('Checkpoint density has the wrong electron count')
    return dm
