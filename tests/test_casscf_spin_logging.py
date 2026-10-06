import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import FirstOrderCASSCF, SecondOrderCASSCF
from pyqed.qchem.mcscf.direct_ci import CASCI


@pytest.mark.parametrize('driver', [FirstOrderCASSCF, SecondOrderCASSCF])
@pytest.mark.parametrize('verbose', [0, 4])
def test_internal_ci_logging_does_not_measure_spin(monkeypatch, capsys, driver, verbose):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()
    mf = mol.RHF().run()
    calls = []
    original = CASCI.spin_square
    def counted(mc, state_id=0):
        calls.append((mc, state_id))
        return original(mc, state_id)
    monkeypatch.setattr(CASCI, 'spin_square', counted)
    solver = driver(mf, ncas=2, nelecas=2, ms2=0, multiplicity=1, verbose=verbose)
    assert not hasattr(solver, 'spin_square_values')
    with pytest.raises(ValueError, match='converged'):
        solver.spin_square()
    solver.state_average([.5, .5]).run(nstates=2)
    assert solver.converged
    assert calls == [(solver.casci, 0), (solver.casci, 1)]
    np.testing.assert_allclose(solver.spin_square(), 0., atol=1e-10)
    output = capsys.readouterr().out
    assert 'CASCI Root' not in output
    assert ('CASSCF Root 0' in output) == (verbose > 0)
    np.testing.assert_allclose(solver.spin_square(0), solver.spin_square()[0])
    assert isinstance(solver.spin_square(1), float)
    values = solver.spin_square()
    values[:] = 42.
    np.testing.assert_allclose(solver.spin_square(), 0., atol=1e-10)
    assert len(calls) == 2
    solver.run(nstates=2)
    assert len(calls) == 4
    np.testing.assert_allclose(solver.spin_square(), 0., atol=1e-10)
