import types

import numpy as np

from pyqed.models.heisenberg import Heisenberg
from pyqed.models.impurity.sbm import SBM
from pyqed.mps.tdmps import TDMPS


class _IdentityOp:
    def __matmul__(self, psi):
        return psi


def test_tdmps_run_uses_checkpoint_times_for_tail_interval():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    td = TDMPS(H, D=8)
    td.build_propagator = types.MethodType(lambda self, dt, order=2, scale=0: None, td)
    td.step = types.MethodType(lambda self, psi: psi, td)

    td.run(psi0, dt=0.1, steps=5, e_ops=[H], interval=2)

    np.testing.assert_allclose(td.times, np.array([0.2, 0.4, 0.5]))
    assert td.observables.shape == (3, 1)
    np.testing.assert_allclose(td.observables[:, 0], td.observables[0, 0])


def test_sbm_tddmrg_builds_hamiltonian_before_returning():
    model = SBM(Himp=None, alpha=0.1, delta=1.0, epsilon=0.0)
    model.nmodes = 1
    model.t0 = 0.0
    model.onsite = np.array([0.0])
    model.hopping = np.array([], dtype=float)

    td = model.TDDMRG(D=8, nb=4)

    assert model.H is not None
    assert td.H is model.H


def test_tdmps_dynamic_run_uses_split_propagation_without_full_rebuild():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    td = TDMPS(H, D=8, interaction_mpo=H, field=lambda t: 1.0)

    def _fail_build_propagator(self, dt, order=2, scale=0, time=0.0, field=None):
        raise AssertionError("full propagator rebuild should not be used in dynamic split mode")

    def _fake_static(self, dt, order=2, scale=0):
        self.U_static = _IdentityOp()
        self.U_static_half = _IdentityOp()
        return self.U_static, self.U_static_half

    def _fake_interaction(self, dt, time=0.0, field=None, order=2, scale=0):
        return _IdentityOp()

    td.build_propagator = types.MethodType(_fail_build_propagator, td)
    td.build_static_propagators = types.MethodType(_fake_static, td)
    td.build_interaction_propagator = types.MethodType(_fake_interaction, td)

    td.run(psi0, dt=0.1, steps=3, e_ops=[], interval=1, field=lambda t: 1.0)

    np.testing.assert_allclose(td.times, np.array([0.1, 0.2, 0.3]))
