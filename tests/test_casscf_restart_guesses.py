import numpy as np
import pytest

from pyqed.qchem import Molecule, SecondOrderCASSCF
from pyqed.qchem.mcscf.direct_ci import _build_davidson_guess


@pytest.mark.parametrize('duplicate', [False, True])
def test_restart_guess_is_rank_complete_without_extra_vectors(duplicate):
    roots = np.eye(8)[:, :3]
    if duplicate:
        roots[:, 1] = roots[:, 0]
    actual = _build_davidson_guess(np.arange(8.), 3, roots)
    assert actual.shape == (8, 3)
    np.testing.assert_allclose(actual.T@actual, np.eye(3), atol=1e-12)
    assert _build_davidson_guess(np.arange(8.), 3).shape == (8, 6)
    assert _build_davidson_guess(np.arange(8.), 3, roots, min_vectors=5).shape == (8, 5)
