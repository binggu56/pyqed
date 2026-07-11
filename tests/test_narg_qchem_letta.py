import numpy as np

from pyqed.narg.qchem import LETTA


def _tfim_mpo(nsites, g=1.0):
    eye = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    if nsites == 1:
        return [(-g * x).reshape(1, 1, 2, 2)]

    w0 = np.zeros((1, 3, 2, 2))
    wm = np.zeros((3, 3, 2, 2))
    wl = np.zeros((3, 1, 2, 2))
    w0[0, 0] = -g * x
    w0[0, 1] = -z
    w0[0, 2] = eye
    wm[0, 0] = eye
    wm[1, 0] = z
    wm[2, 0] = -g * x
    wm[2, 1] = -z
    wm[2, 2] = eye
    wl[0, 0] = eye
    wl[1, 0] = z
    wl[2, 0] = -g * x
    return [w0] + [wm.copy() for _ in range(nsites - 2)] + [wl]


def test_qchem_letta_wraps_generic_letta_from_factorized_narg():
    dims = (2, 2, 2)
    rng = np.random.default_rng(31)
    t0 = rng.normal(size=(2, 2, 2))
    t1 = rng.normal(size=(4, 2, 2))
    coeff = rng.normal(size=(4, 1))
    mpo = _tfim_mpo(len(dims))

    letta = LETTA.from_narg([t0, t1], coeff, dims=dims, bond_dim=2, mpo=mpo)
    initial = letta.expect()
    psi = letta.state_vector()
    result = letta.run(nsweeps=1)

    expected = []
    for s0, s1, s2 in np.ndindex(*dims):
        amp = 0.0
        for a0 in range(t0.shape[1]):
            for a1 in range(t1.shape[1]):
                amp += t0[s0, a0, s1] * t1[s1 * t0.shape[1] + a0, a1, s2] * coeff[s2 * t1.shape[1] + a1, 0]
        expected.append(amp)
    expected = np.asarray(expected)
    expected /= np.linalg.norm(expected)

    assert type(letta).__name__ == "LETTA"
    assert len(letta.tensors) == len(dims)
    assert letta.tensors[-1].shape == (dims[-1], t1.shape[1])
    np.testing.assert_allclose(psi, expected, atol=1e-12)
    assert np.isfinite(initial)
    assert np.isfinite(result.energy)
    assert letta.dims == dims


def test_qchem_letta_appends_terminal_symmetry_mask():
    dims = (4, 4, 4)
    rng = np.random.default_rng(32)
    t0 = rng.normal(size=(4, 2, 4))
    t1 = rng.normal(size=(8, 3, 4))
    coeff = rng.normal(size=(4, 3, 1))
    mpo = [np.eye(dim).reshape(1, 1, dim, dim) for dim in dims]

    class FactorizedNARG:
        tensors = [t0, t1, coeff]
        bond_dim = 3
        tensor_qns = {
            "factors": [
                {
                    "row_qn": np.zeros((dims[0], 2), dtype=int),
                    "right_qn_by_next": np.zeros((dims[1], t0.shape[1], 2), dtype=int),
                },
                {
                    "row_qn": np.zeros((dims[1] * t0.shape[1], 2), dtype=int),
                    "right_qn_by_next": np.zeros((dims[2], t1.shape[1], 2), dtype=int),
                },
            ],
            "terminal_total_qn_by_site": np.zeros((dims[-1], t1.shape[1], 2), dtype=int),
            "target_qn": np.zeros(2, dtype=int),
        }

    letta = LETTA.from_narg(FactorizedNARG(), dims=dims, symmetry="abelian", mpo=mpo)

    assert len(letta.tensors) == len(dims)
    assert letta.tensors[-1].shape == (dims[-1], t1.shape[1])
    assert len(letta.engine.local_masks) == len(dims)
    assert letta.engine.local_masks[-1].shape == letta.tensors[-1].shape


def test_qchem_letta_requires_factorized_narg_input():
    class VectorOnlyNARG:
        energies = np.array([0.0])
        vectors = np.ones((4, 1))

    with np.testing.assert_raises(ValueError):
        LETTA.from_narg(VectorOnlyNARG(), dims=(2, 2))
