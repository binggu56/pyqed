"""Shell-local RI output compared with the Cartesian-transform reference."""
import numpy as np
import pytest
from pyqed.qchem import Molecule
from pyqed.qchem import basis as integrals


@pytest.mark.parametrize('screen', [0.0, 1e-4, 1e6])
def test_spherical_triplets_match_cartesian_reference(monkeypatch, screen):
    basis = {'H': [(l, [.7, .25], [.6, .4]) for l in range(4)]}
    aux = {'H': [(l, [.9, .3], [1., -.2]) for l in range(4)]}
    compute = integrals._compute_native_ri_pair_tensors_cpp
    checked = []

    def compare(signatures, aux_signatures, bounds, tol, primary, auxiliary, factor_options=None):
        cart = compute(signatures, aux_signatures, bounds, tol)
        independent = integrals._compute_native_ri_pair_tensors_cython(
            signatures, aux_signatures, bounds, tol)
        assert independent is not None
        np.testing.assert_allclose(cart[0], independent[0], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(cart[1], independent[1], atol=2e-12, rtol=2e-12)
        expected = integrals._transform_ri_tensors_to_spherical(
            cart[0], cart[1], primary, auxiliary)
        actual = compute(signatures, aux_signatures, bounds, tol, primary, auxiliary)
        assert actual is not None
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-12, rtol=2e-12)
        assert actual[2:] == cart[2:]
        checked.append(True)
        return compute(signatures, aux_signatures, bounds, tol, primary, auxiliary,
                       factor_options=factor_options)

    monkeypatch.setattr(integrals, '_compute_native_ri_pair_tensors_cpp', compare)
    mol = Molecule(atom='H 0 0 0; H .3 .5 1.4', unit='bohr', basis=basis)
    mol.build(eri='ri', auxbasis=aux, options={
        'ri_cache': False, 'ri_tensor_backend': 'cpp', 'ri_screen_tol': screen,
        'ri_storage': 'packed'})
    assert checked
    assert mol._builtin_build_info['ri']['kernel_info']['spherical_output']
    assert mol._builtin_build_info['ri']['kernel_info']['metric_seconds'] >= 0
    assert mol._builtin_build_info['ri']['kernel_info']['triplet_seconds'] >= 0


def test_spherical_build_does_not_transform_global_triplets(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Global Cartesian three-center tensor was built')
    monkeypatch.setattr(integrals, '_transform_ri_tensors_to_spherical', forbidden)
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='cc-pvdz')
    mol.build(eri='ri', options={'ri_cache': False, 'ri_tensor_backend': 'cpp'})
    assert mol._builtin_build_info['ri']['tensor_builder'] == 'shell-spherical-packed'


@pytest.mark.parametrize('nprim', [4, 5])
@pytest.mark.parametrize('screen', [0., 1e-2])
def test_batched_triplets_match_scalar_reference(nprim, screen):
    # 32/50 lanes exercise both a full batch and a partially filled final batch.
    exponents = tuple(np.linspace(.2, 1.1, nprim))
    weights = tuple(np.linspace(-.15, .4, nprim))
    signatures = tuple((tuple(shell), origin, exponents, weights)
                       for origin in [(0., 0., 0.), (.3, .5, 1.4)]
                       for shell in np.eye(3, dtype=int))
    auxiliary = tuple((tuple(shell), (.1, .2, .4), (.8, .2), (.6, -.1))
                      for shell in np.eye(3, dtype=int))
    bounds = np.arange(1, 37, dtype=float).reshape(6, 6)*1e-3
    actual = integrals._compute_native_ri_pair_tensors_cpp(
        signatures, auxiliary, bounds, screen)
    reference = integrals._compute_native_ri_pair_tensors_cython(
        signatures, auxiliary, bounds, screen)
    assert actual is not None and reference is not None
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)
@pytest.mark.parametrize('proportional', [True, False])
@pytest.mark.parametrize('zero_first', [True, False])
def test_contraction_component_weights(proportional, zero_first):
    signatures = []
    for origin in [(0., 0., 0.), (.3, .5, 1.4), (.8, .5, 1.4)]:
        for i, shell in enumerate(np.eye(3, dtype=int)):
            weights = [0. if zero_first else .7, -.2]
            weights = [(i+1)*x for x in weights]
            if not proportional and i == 1:
                weights[0] += .03
            signatures.append((tuple(shell), origin, (.9, .3), tuple(weights)))
    signatures = tuple(signatures)
    auxiliary = (((0, 0, 0), (.1, .2, .4), (.8, .2), (.6, -.1)),)
    bounds = np.ones((len(signatures), len(signatures)))
    actual = integrals._compute_native_ri_pair_tensors_cpp(signatures, auxiliary, bounds, 0.)
    # Arbitrary component weights can change the retained primitive list per AO.
    reference = (integrals._compute_aux_coulomb_metric(auxiliary),
                 integrals._compute_three_center_pair_tensor_from_signatures(signatures, auxiliary)[0])
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize('mask', range(8))
@pytest.mark.parametrize('nprim', [1, 4])
def test_triplet_zero_displacement_axes(mask, nprim):
    origin = np.array([.7, -.2, .3])
    offset = np.array([.3, -.5, 1.4])*[(mask >> axis) & 1 for axis in range(3)]
    signatures = tuple((tuple(shell), tuple(center), (.9, .6, .4, .2)[:nprim],
                        (.5, -.2, .1, .3)[:nprim])
                       for center in [origin, origin+offset]
                       for shell in np.eye(3, dtype=int))
    auxiliary = tuple(((x, y, 2-x-y), (.4, .6, -.1), (.8, .2), (.6, -.1))
                      for x in range(2, -1, -1) for y in range(2-x, -1, -1))
    bounds = np.ones((len(signatures), len(signatures)))
    actual = integrals._compute_native_ri_pair_tensors_cpp(signatures, auxiliary, bounds, 0.)
    reference = integrals._compute_native_ri_pair_tensors_cython(signatures, auxiliary, bounds, 0.)
    assert actual is not None and reference is not None
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)
