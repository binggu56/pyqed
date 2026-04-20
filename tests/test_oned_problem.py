import numpy as np

from pyqed.mps.first_quantization import Chain


def test_oned_problem_onsite_terms_match_dense_reference():
    # H = Z_0 + Z_1 on two qubit-like sites.
    z = np.diag([1.0, -1.0])
    prob = Chain(nsites=2, local_dim=2, local_operator_mats={"Z": z})
    prob.add_uniform_onsite("Z", 1.0)

    h = prob.dense_hamiltonian()
    ref = np.diag([2.0, 0.0, 0.0, -2.0])
    np.testing.assert_allclose(h, ref)


def test_oned_problem_uniform_bond_builds_hopping():
    # H = - (|10><01| + |01><10|) using transitions.
    prob = Chain(nsites=2, local_dim=2)
    prob.add_uniform_bond("E1_0", "E0_1", coeff=-1.0, distance=1)
    prob.add_uniform_bond("E0_1", "E1_0", coeff=-1.0, distance=1)

    h = prob.dense_hamiltonian()
    ref = np.zeros((4, 4), dtype=complex)
    # basis ordering |00>, |01>, |10>, |11>
    ref[2, 1] = -1.0
    ref[1, 2] = -1.0
    np.testing.assert_allclose(h, ref)
