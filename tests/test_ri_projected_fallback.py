"""Automatic RI selection recovers from unsupported projected shell layouts."""
import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem import basis


REJECTION = 'Projected RI output requires supported complete Cartesian shells.'


def molecule():
    return Molecule(atom='H 0 0 -1; H 0 0 1', basis='sto-3g')


def test_automatic_projected_rejection_uses_packed_kernel(monkeypatch):
    options = {'ri_cache': False, 'workers': 1, 'ri_tensor_backend': 'cython'}
    reference = molecule().build(eri='ri', auxbasis='def2-universal-jkfit', options=options)

    def reject(*args, **kwargs):
        raise ValueError(REJECTION)

    monkeypatch.setattr(basis, '_compute_native_ri_pair_tensors_cpp', reject)
    actual = molecule().build(eri='ri', auxbasis='def2-universal-jkfit',
                              options={**options, 'ri_tensor_backend': 'auto'})
    assert actual._builtin_build_info['ri']['effective_tensor_backend'] == 'cython'
    np.testing.assert_allclose(actual.eri_factors, reference.eri_factors, atol=1e-12)


@pytest.mark.parametrize('backend,message', [('cpp', REJECTION), ('auto', 'whitening failure')])
def test_other_ri_failures_propagate(monkeypatch, backend, message):
    def reject(*args, **kwargs):
        raise ValueError(message)

    monkeypatch.setattr(basis, '_compute_native_ri_pair_tensors_cpp', reject)
    with pytest.raises(ValueError, match=message):
        molecule().build(eri='ri', auxbasis='def2-universal-jkfit', options={
            'ri_cache': False, 'workers': 1, 'ri_tensor_backend': backend})
