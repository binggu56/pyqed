"""Validate the non-projecting spatial-weight diagnostic used by benchmarks."""
from types import SimpleNamespace
import pickle
import numpy as np
import pytest


@pytest.mark.parametrize('weight', [0., .25, 1.])
def test_spatial_weights_resolve_known_irrep_mixture(monkeypatch, weight):
    from examples.qchem import benchmark_abelian_dmrg_scaling as benchmark
    from pyqed.qchem.symmetry import get_point_group
    from pyqed.mps.nonabelian.mps import MPS
    from pyqed.mps.nonabelian.states import build_reduced_product_spatial_mps
    from pyqed.mps.nonabelian.spatial_symmetry import sum_reduced_mps
    group = get_point_group('D2h')
    representations = {name: np.diag([1., group.character_table['B1u'][i]])
                       for i, name in enumerate(group.operation_names)}
    monkeypatch.setattr(benchmark, 'mo_symmetry_representations', lambda *args: representations)
    states = [MPS.from_tensors(build_reduced_product_spatial_mps(labels))
              for labels in [('single', 'empty'), ('empty', 'single')]]
    state = sum_reduced_mps(states, [np.sqrt(weight), np.sqrt(1-weight)])
    state = pickle.loads(pickle.dumps(state))
    before = [site.copy() for site in state.tensors]
    weights = benchmark.spatial_weights(state, SimpleNamespace(
        symmetry_info=SimpleNamespace(group=group)), np.eye(2))
    for label, value in weights.items():
        assert value == pytest.approx(weight if label == 'Ag' else
                                      1-weight if label == 'B1u' else 0., abs=1e-12)
    for old, new in zip(before, state.tensors):
        for key in old.data:
            np.testing.assert_array_equal(old.data[key], new.data[key])
