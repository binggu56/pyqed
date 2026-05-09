import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.backends.nonabelian import _qchem_sweep_measure
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF
from pyqed.qchem.hf import RHF


def _h2_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    return RHF(mol).run()


def test_state_averaged_su2_dmrgscf_preserves_nonabelian_backend():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=1,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.dmrg.backend == "nonabelian"
    assert mc.dmrg_conv_tol == 1.0e-7
    assert mc.macro_converged is True
    assert mc.solver_converged == mc.dmrg.converged
    assert mc.converged == (mc.macro_converged and mc.solver_converged)
    assert mc.macro_iterations == 1
    assert len(mc.dmrg.states) == 2
    np.testing.assert_allclose(
        mc.e_tot,
        [-1.137275943783, -0.169291740911],
        atol=1.0e-8,
    )


def test_state_averaged_su2_dmrgscf_requires_final_inner_convergence():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    with pytest.raises(RuntimeError, match="active-space DMRG did not converge"):
        mc.run(
            nstates=2,
            weights=[0.5, 0.5],
            nsweeps=1,
            conv_tol=-1.0,
            mixer_zero_block_noise_scale=0.0,
        )

    assert mc.macro_converged is True
    assert mc.solver_converged is False
    assert mc.converged is False


def test_qchem_su2_sweep_measure_prefers_objective_residual():
    sweep_result = {
        "updates": [
            {"trunc_err": 0.0, "local_objective": {"metric": 2.5e-3}},
            {"trunc_err": 0.0, "local_objective": {"residual": 1.0e-2}},
        ]
    }

    assert _qchem_sweep_measure(sweep_result) == 1.0e-2
