import numpy as np

from pyqed.qchem import COCAS, Molecule


def test_cocas_cd_checkpoint_exposes_paired_ci_and_orbitals():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="angstrom")
    mol.build(eri="cd", options={"low_rank_tol": 1e-10, "workers": 1})
    mf = mol.RHF().run(tol=1e-10)
    events = []
    solver = COCAS(mf, ncas=2, nelecas=2, max_cycles=2, use_cholesky=True)
    solver.run(nstates=1, macro_callback=events.append, raise_on_nonconvergence=False)
    assert solver.converged
    assert events
    event = events[-1]
    assert event["casci"].ci is not None
    np.testing.assert_allclose(event["energy"], solver.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(event["mo_coeff"], solver.mo_coeff, atol=1e-9, rtol=0)
    assert event["diagnostics"]["accepted"]
