"""Public derivative calls own their temporary storage unless explicitly scoped."""
from pathlib import Path
import numpy as np
import pytest
from pyqed.gw import GW, BSE
from test_gw_api import mf


@pytest.fixture
def bse(mf):
    return BSE(GW(mf, freq_int='exact', eta=0.).run(),
               screening_energies='hf').run(nroots=1, low_rank=False)


@pytest.mark.parametrize('entry, options', [
    ('gradient', {}), ('gradient', {'method': 'forward'}),
    ('hessian', {}), ('hessian', {'method': 'polarization'}),
    ('derivatives', {'order': 1}), ('derivatives', {'order': 2}),
])
def test_standalone_cleanup(bse, monkeypatch, entry, options):
    from pyqed.gw.response_storage import ResponseStore
    original = ResponseStore.__init__
    paths = []
    def tracked(self):
        original(self)
        paths.append(Path(self.directory.name))
    monkeypatch.setattr(ResponseStore, '__init__', tracked)
    result = getattr(bse, entry)(state_ids=[0], modes=np.eye(6)[[5]], **options)
    for value in result if isinstance(result, tuple) else [result]:
        assert np.isfinite(value).all()
    assert bse._adjoint_workspace is None and bse._first_response_cache is None
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize('entry', ['gradient', 'hessian', 'derivatives'])
def test_failure_clears_active_scope(bse, monkeypatch, entry):
    from pyqed.gw import adjoint
    with bse.derivative_workspace():
        bse.gradient(state_ids=[0], modes=np.eye(6)[[5]])
        path = Path(bse._adjoint_workspace['store'].directory.name)
        assert path.exists()
        def fail(*args, **kwargs):
            raise RuntimeError('deliberate derivative failure')
        monkeypatch.setattr(adjoint, 'gradients' if entry == 'gradient' else 'hessians', fail)
        with pytest.raises(RuntimeError, match='deliberate'):
            getattr(bse, entry)(state_ids=[0], modes=np.eye(6)[[5]])
        assert bse._adjoint_workspace is None and bse._first_response_cache is None
        assert not path.exists()
