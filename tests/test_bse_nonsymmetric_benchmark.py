"""Validate the benchmark's positive-real projected-root policy."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def test_positive_root_selection_excludes_negative_and_complex_pairs():
    pytest.importorskip('pyscf')
    path=Path(__file__).resolve().parents[1]/'benchmarks/benchmark_bse_nonsymmetric.py'
    spec=importlib.util.spec_from_file_location('bse_nonsymmetric_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values=np.array([-3.,.5+1e-4j,2.,.2,1.])
    roots,vectors,indices=module.positive_roots(values,np.eye(5,dtype=complex),2,{})
    np.testing.assert_allclose(roots,[.2,1.,2.])
    np.testing.assert_allclose(vectors,np.eye(5)[:,[3,4,2]])
    with pytest.raises(RuntimeError,match='Not enough'):
        module.positive_roots(values,np.eye(5,dtype=complex),4,{})
