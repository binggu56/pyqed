"""Selected CD curvature columns against small dense integral references."""
import numpy as np
import pytest
from pyqed.qchem import Molecule
from pyqed.qchem.eri_response import CoulombIntegrals
from pyqed.qchem.basis_derivatives import directional_eri_derivatives


@pytest.mark.parametrize('basis', ['sto-3g', 'cc-pvdz'])
def test_cd_directional_second_columns(basis):
    mol = Molecule(atom='H 0 0 0; F .1 .2 1.8', basis=basis, unit='bohr').build(
        eri='cd', workers=4, options={'eri_screen_tol': 0., 'low_rank_tol': 1e-10})
    modes = np.zeros((2,2,3))
    modes[0,1,2] = .4
    modes[1,0,0], modes[1,1,1] = .2, -.3
    # Dense four-index arrays are used only in this small reference test.
    reference = directional_eri_derivatives(mol, modes, order=2, backend="native")
    response = CoulombIntegrals(mol)
    actual = response._cd_columns(modes, order=2, workers=4)
    pairs = np.tril_indices(mol.nao)
    pivots = mol._cd_factor_data['pivots']
    expected = reference[:,:,pairs[0][pivots],pairs[1][pivots]][:,:,:,pairs[0],pairs[1]]
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=1e-9)
    np.testing.assert_allclose(actual, actual.swapaxes(0,1), atol=2e-10)
    if basis == 'sto-3g':
        independent = directional_eri_derivatives(mol, modes, order=2, backend='pyscf')
        np.testing.assert_allclose(reference, independent, atol=1e-8, rtol=1e-8)
    # cc-pVDZ native/PySCF general contractions differ within the same span.
    # Independently difference native first integrals in the actual AO frame.
    step = 2e-4
    displaced = []
    for sign in (1, -1):
        coords = mol.atom_coords()+sign*step*modes[1]
        shifted = Molecule(atom=list(zip(mol.atom_symbols(), coords)), basis=basis, unit='bohr').build(
            eri='cd', workers=4, options={'eri_screen_tol': 0., 'low_rank_tol': 1e-10})
        displaced.append(directional_eri_derivatives(shifted, modes[:1], order=1, backend='native')[0])
    np.testing.assert_allclose(reference[0,1], (displaced[0]-displaced[1])/(2*step), atol=2e-8, rtol=2e-6)
    mixed = response._cd_columns(modes, order=2, pair=1, workers=4)
    assert mixed.shape == (1,1,*actual.shape[2:])
    np.testing.assert_allclose(mixed[0,0], actual[0,1], atol=2e-10)
    reverse = response._cd_columns(modes, order=2, pair=2, workers=1)
    np.testing.assert_allclose(reverse, mixed, atol=2e-10)
    selected = response._cd_columns(modes, order=2, pairs=np.array([[0,0],[0,1],[1,1]]), workers=4)
    np.testing.assert_allclose(selected[:,0], actual[[0,0,1],[0,1,1]], atol=2e-10)
    weights = np.random.default_rng(21).normal(size=(2, *actual.shape[2:]))
    contracted = np.einsum('xyPq,kPq->xyk', actual, weights)
    for workers in (1, 4):
        scalar = response._cd_columns(modes, order=2, workers=workers, sensitivities=weights)
        np.testing.assert_allclose(scalar, contracted, atol=2e-9, rtol=1e-10)
        scalar_pairs = response._cd_columns(modes, order=2, workers=workers,
            sensitivities=weights, pairs=np.array([[1,0],[0,0],[1,1]]))
        np.testing.assert_allclose(scalar_pairs[:,0], contracted[[1,0,1],[0,0,1]],
                                   atol=2e-9, rtol=1e-10)
        scalar_pair = response._cd_columns(modes, order=2, workers=workers,
            sensitivities=weights, pair=1, early_contraction=False)
        np.testing.assert_allclose(scalar_pair[0,0], contracted[0,1], atol=2e-9, rtol=1e-10)
    with pytest.raises(ValueError):
        response._cd_columns(modes, order=2, sensitivities=weights, screen_tol=1e-8)
    with pytest.raises(ValueError):
        response._cd_columns(modes, order=1, pair=1)
    with pytest.raises(ValueError):
        response._cd_columns(modes, order=2, pair=4)
    single = response._cd_columns(modes, order=2, workers=1)
    np.testing.assert_allclose(single, actual, atol=2e-10)
    with pytest.raises(ValueError):
        response._cd_columns(modes, order=3)
