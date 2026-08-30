import numpy as np

from pyqed.letta import NNNLETTA


def _kron_all(ops):
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _tfim_mpo(nsites, g=0.7):
    eye = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    dense = np.zeros((2**nsites, 2**nsites))
    for i in range(nsites - 1):
        ops = [eye] * nsites
        ops[i] = z
        ops[i + 1] = z
        dense -= _kron_all(ops)
    for i in range(nsites):
        ops = [eye] * nsites
        ops[i] = x
        dense -= g * _kron_all(ops)

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
    return dense, [w0] + [wm.copy() for _ in range(nsites - 2)] + [wl]


def test_nnn_letta_state_vector_matches_manual_contraction():
    rng = np.random.default_rng(3)
    dims = (2, 2, 2, 2)
    tensors = [
        rng.normal(size=(1, 2, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 2, 1)),
    ]
    ansatz = NNNLETTA(dims, tensors=tensors)

    expected = []
    for config in np.ndindex(*dims):
        value = 0.0
        for alpha in range(3):
            value += tensors[0][0, config[0], config[1], config[2], alpha] * tensors[1][
                alpha, config[1], config[2], config[3], 0
            ]
        expected.append(value)
    expected = np.asarray(expected)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(ansatz.state_vector(), expected, atol=1.0e-12)


def test_nnn_letta_conditional_gauge_preserves_state_and_whitens_shared_pairs():
    ansatz = NNNLETTA((2, 2, 2, 2), bond_dim=2, seed=5)
    before = ansatz.state_vector()

    ansatz.canonicalize_conditional_bond(0, direction="lr", normalize=True)
    after = ansatz.state_vector()

    np.testing.assert_allclose(after, before / np.linalg.norm(before), atol=1.0e-12)
    left = ansatz.tensors[0]
    for s1 in range(2):
        for s2 in range(2):
            block = left[:, :, s1, s2, :].reshape(2, 2)
            np.testing.assert_allclose(block.conj().T @ block, np.eye(2), atol=1.0e-12)


def test_nnn_letta_adaptive_compression_preserves_state_and_tapers_bonds():
    ansatz = NNNLETTA((2,) * 6, bond_dim=8, seed=23)
    before = ansatz.state_vector()

    reports = ansatz.compress_conditional_bonds(direction="rl")

    np.testing.assert_allclose(ansatz.state_vector(), before, atol=2.0e-12)
    assert tuple(tensor.shape[-1] for tensor in ansatz.tensors[:-1]) == (2, 4, 2)
    assert all(report["relative_discarded_weight"] < 1.0e-24 for report in reports)


def test_nnn_letta_compression_tracks_physical_pair_dependent_ranks_with_masks():
    rng = np.random.default_rng(29)
    left = rng.normal(size=(1, 2, 2, 2, 2))
    right = rng.normal(size=(2, 2, 2, 2, 1))
    left[:, :, 0, 0, 1] = 0.0
    right[1, 0, 0, :, :] = 0.0
    ansatz = NNNLETTA((2, 2, 2, 2), tensors=[left, right])
    before = ansatz.state_vector()

    report = ansatz.compress_conditional_bond(0, direction="balanced")

    np.testing.assert_allclose(ansatz.state_vector(), before, atol=2.0e-12)
    assert report["sector_ranks"] == (1, 2, 2, 2)
    assert report["new_dim"] == 2
    assert ansatz.local_masks is not None
    assert not ansatz.local_masks[0][:, :, 0, 0, 1].any()
    assert not ansatz.local_masks[1][1, 0, 0, :, :].any()


def test_nnn_letta_from_mps_embeds_state_and_product_expectations():
    rng = np.random.default_rng(13)
    factors = [
        rng.normal(size=(1, 2, 2)),
        rng.normal(size=(2, 2, 3)),
        rng.normal(size=(3, 2, 2)),
        rng.normal(size=(2, 2, 1)),
    ]
    ansatz = NNNLETTA.from_mps(factors)

    expected = np.empty(16)
    for flat, config in enumerate(np.ndindex(2, 2, 2, 2)):
        value = np.ones(1)
        for site, tensor in enumerate(factors):
            value = value @ tensor[:, config[site], :]
        expected[flat] = value[0]
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(ansatz.state_vector(), expected, atol=1.0e-12)

    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    ops = [np.eye(2), z, np.eye(2), z]
    dense_op = _kron_all(ops)
    dense_value = np.vdot(expected, dense_op @ expected)
    np.testing.assert_allclose(ansatz.expectation_product_operator(ops), dense_value.real, atol=1.0e-12)


def test_nnn_letta_masked_conditional_gauge_preserves_state_and_support():
    rng = np.random.default_rng(17)
    dims = (2, 2, 2, 2, 2)
    local_qns = [0, 1]
    bond_qns = [[0], [0, 0, 1, 1], [0, 1, 1, 2], [2]]
    target = 2
    masks = []
    for tensor_index in range(3):
        shape = (
            len(bond_qns[tensor_index]),
            2,
            2,
            2,
            len(bond_qns[tensor_index + 1]),
        )
        mask = np.zeros(shape, dtype=bool)
        for left_index, q_left in enumerate(bond_qns[tensor_index]):
            for right_index, q_right in enumerate(bond_qns[tensor_index + 1]):
                if tensor_index < 2:
                    for s0, q0 in enumerate(local_qns):
                        if q_left + q0 == q_right:
                            mask[left_index, s0, :, :, right_index] = True
                else:
                    for s0, q0 in enumerate(local_qns):
                        for s1, q1 in enumerate(local_qns):
                            for s2, q2 in enumerate(local_qns):
                                if q_left + q0 + q1 + q2 == target:
                                    mask[left_index, s0, s1, s2, right_index] = True
        masks.append(mask)
    tensors = [rng.normal(size=mask.shape) * mask for mask in masks]
    ansatz = NNNLETTA(dims, tensors=tensors, local_masks=masks)
    before = ansatz.state_vector()

    ansatz.canonicalize_conditional_bond(0, direction="lr")
    ansatz.canonicalize_conditional_bond(1, direction="rl")

    np.testing.assert_allclose(ansatz.state_vector(), before, atol=1.0e-12)
    for tensor, mask in zip(ansatz.tensors, masks):
        assert np.max(np.abs(tensor[~mask])) == 0.0


def test_nnn_letta_mpo_sweep_lowers_energy():
    dims = (2, 2, 2, 2)
    _dense, mpo = _tfim_mpo(len(dims), g=0.7)
    ansatz = NNNLETTA(dims, bond_dim=2, seed=7)
    initial = ansatz.expectation_mpo(mpo)

    ansatz.run(mpo, nsweeps=2, tol=0.0, gauge="conditional")

    assert np.isfinite(ansatz.energy)
    assert ansatz.energy <= initial + 1.0e-10
    assert ansatz.history
    assert all(
        update["identity_metric"]
        for sweep in ansatz.history
        for update in sweep["updates"]
    )


def test_nnn_letta_identity_adaptive_sweep_matches_generalized_fixed_bonds():
    dims = (2, 2, 2, 2, 2)
    _dense, mpo = _tfim_mpo(len(dims), g=0.7)
    generalized = NNNLETTA(dims, bond_dim=2, seed=31)
    identity = NNNLETTA(dims, bond_dim=2, seed=31)

    generalized.run(
        mpo,
        nsweeps=2,
        tol=0.0,
        gauge="conditional",
        identity_metric=False,
        adapt_bonds=False,
    )
    identity.run(mpo, nsweeps=2, tol=0.0, gauge="conditional")

    np.testing.assert_allclose(identity.energy, generalized.energy, atol=2.0e-11)
    assert all(
        update["identity_metric"]
        for sweep in identity.history
        for update in sweep["updates"]
    )


def test_nnn_letta_matrix_free_mpo_sweep_lowers_energy():
    dims = (2, 2, 2, 2, 2)
    _dense, mpo = _tfim_mpo(len(dims), g=0.7)
    ansatz = NNNLETTA(dims, bond_dim=2, seed=11)
    initial = ansatz.expectation_mpo(mpo)

    ansatz.run(
        mpo,
        nsweeps=1,
        tol=0.0,
        gauge="conditional",
        local_solver="matrix_free",
        matrix_free_threshold=1,
    )

    assert np.isfinite(ansatz.energy)
    assert ansatz.energy <= initial + 1.0e-10
    assert ansatz.history[0]["updates"]


def test_nnn_letta_range2_environments_match_dense_projector():
    dims = (2, 2, 2, 2, 2)
    _dense, mpo = _tfim_mpo(len(dims), g=0.7)
    ansatz = NNNLETTA(dims, bond_dim=2, seed=9)

    psi = ansatz.state_vector()
    dense_energy = np.vdot(psi, ansatz.apply_mpo(mpo, psi)) / np.vdot(psi, psi)
    np.testing.assert_allclose(ansatz.expectation_mpo(mpo), dense_energy.real, atol=1.0e-12)

    left = ansatz._left_local_environments(mpo)
    right = ansatz._right_local_environments(mpo)
    for tensor_index in range(ansatz.nlocal_tensors):
        heff = ansatz._local_effective_from_environments(mpo, tensor_index, left, right)

        projector = ansatz.one_tensor_projector(tensor_index)
        h_projector = np.column_stack(
            [ansatz.apply_mpo(mpo, projector[:, col]) for col in range(projector.shape[1])]
        )
        dense_heff = projector.conj().T @ h_projector

        np.testing.assert_allclose(heff, dense_heff, atol=1.0e-12)
