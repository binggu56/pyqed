import numpy as np
import pytest
from itertools import combinations
from pyqed.qchem.mcscf import caspt2 as pt


@pytest.mark.parametrize("shift", [0.0, 0.25])
def test_cvs_contracted_energy_matches_explicit_two_electron_space(shift):
    n = 4
    ref = [(1 << 0) | (1 << (n + 1)), (1 << 1) | (1 << n)]
    ci = np.array([1.0, 1.0]) / np.sqrt(2)
    h = np.diag([-10.0, -1.0, -0.5, 1.0])
    h[1, 3] = h[3, 1] = 0.2
    eri = np.zeros((n, n, n, n))
    f = h.copy()
    f[1, 3] = f[3, 1] = 0
    # This additional channel removes the core electron and must be projected out.
    h[0, 3] = h[3, 0] = 0.7
    expected = []
    for builder in [pt._build_direct_signature_blocks, pt._build_direct_tensor_blocks]:
        blocks, _ = builder(ref, ci, 0, 3, n, cvs_core_orbitals=(0,))
        result = pt._solve_direct_signature_blocks(
            blocks,
            ref,
            ci,
            h,
            eri,
            f,
            -11.0,
            0,
            3,
            n,
            real_shift=0.0,
            imaginary_shift=shift,
            denominator_tol=1e-12,
            workers=1,
        )
        # Explicit CVS singlet |0,3>: coupling .2 and denominator -2.
        amplitude = -0.2 * 2 / (4 + shift**2)
        energy = 2 * 0.2 * amplitude + 2 * amplitude**2
        np.testing.assert_allclose(result["variational_energy"], energy, atol=1e-12)
        expected.append(result["variational_energy"])
        for block in blocks:
            for det in block["rows"]:
                assert pt._in_cvs_sector(det, (0,), n)
    np.testing.assert_allclose(*expected, atol=1e-12)


def test_core_hole_reference_does_not_require_cvs_in_first_order(monkeypatch):
    n = 4
    ref = [(1 << 0) | (1 << (n + 1)), (1 << 1) | (1 << n)]
    ci = np.ones(2) / np.sqrt(2)
    fock = np.diag([-10.0, -1.0, -0.5, 1.0])
    h = fock.copy()
    h[1, 3] = h[3, 1] = 0.2
    h[0, 3] = h[3, 0] = 0.7
    blocks, _ = pt._build_direct_tensor_blocks(ref, ci, 0, 3, n)
    # The core-electron excitation is part of ordinary perturbation theory,
    # even though it lies outside the one-hole reference sector.
    assert any(
        not pt._in_cvs_sector(row, (0,), n) for block in blocks for row in block["rows"]
    )

    def dense_is_forbidden(*args):
        raise AssertionError("Dense determinant-space Fock allocation")

    monkeypatch.setattr(pt, "_direct_local_fock_matrix", dense_is_forbidden)
    result = pt._solve_direct_signature_blocks(
        blocks,
        ref,
        ci,
        h,
        np.zeros((n,) * 4),
        fock,
        -11.0,
        0,
        3,
        n,
        real_shift=0.0,
        imaginary_shift=0.0,
        denominator_tol=1e-12,
        workers=1,
    )
    np.testing.assert_allclose(
        result["variational_energy"], -0.04 / 2 - 0.49 / 11, atol=1e-12
    )


def test_factored_sparse_fock_matches_dense_with_wide_determinants():
    ncore, ncas, nmo = 33, 3, 40
    inactive = ((1 << ncore) - 1) * (1 + (1 << nmo))
    rows = [
        inactive | (1 << (ncore + a)) | (1 << (nmo + ncore + b)) | (1 << virtual)
        for virtual in (36, 37)
        for a in range(ncas)
        for b in range(ncas)
    ]
    rng = np.random.default_rng(17)
    fock = rng.normal(size=(nmo, nmo))
    fock = (fock + fock.T) / 2
    dense = pt._direct_local_fock_matrix(rows, fock, ncore, ncas, nmo)
    sparse = pt._direct_local_fock_sparse(rows, fock, ncore, ncas, nmo)
    np.testing.assert_allclose(sparse.toarray(), dense, atol=1e-13)


def test_reference_subset_index_matches_brute_force_connections():
    ncore, ncas, nmo = 2, 4, 8
    inactive = 3 | (3 << nmo)
    active = [sum(1 << i for i in pair) for pair in combinations(range(2, 6), 2)]
    refs = [inactive | a | (b << nmo) for a in active for b in active]
    strings = [sum(1 << i for i in occ) for occ in combinations(range(nmo), 4)]
    rng = np.random.default_rng(12)
    rows = [
        int(strings[a] | (strings[b] << nmo))
        for a, b in rng.integers(0, len(strings), size=(200, 2))
    ]
    candidates = []
    groups = pt._direct_candidate_group_ids(
        rows, refs, ncore, ncas, nmo, {}, {}, candidates
    )
    for row, group in zip(rows, groups):
        expected = tuple(
            i for i, ref in enumerate(refs) if (ref & ~row).bit_count() <= 2
        )
        assert candidates[group] == expected


def test_coupled_blocks_use_blockwise_imaginary_shift():
    blocks = [
        {"class_id": 7, "rows": [1 << i], "basis": np.ones((1, 1))} for i in (1, 2)
    ]
    fock = np.diag([0.0, 2.0, 3.0])
    fock[1, 2] = fock[2, 1] = 0.2
    h = fock.copy()
    h[0, 1] = h[1, 0] = 0.4
    h[0, 2] = h[2, 0] = 0.6
    result = pt._solve_direct_signature_blocks(
        blocks,
        [1],
        np.ones(1),
        h,
        np.zeros((3,) * 4),
        fock,
        0.0,
        0,
        1,
        3,
        real_shift=0.0,
        imaginary_shift=0.25,
        denominator_tol=1e-12,
        workers=1,
    )
    denominator = -fock[1:, 1:]
    shifted = denominator + np.diag(0.25**2 / np.diag(denominator))
    rhs = np.array([0.4, 0.6])
    amplitude = np.linalg.solve(shifted, rhs)
    energy = 2 * rhs @ amplitude - amplitude @ denominator @ amplitude
    np.testing.assert_allclose(result["first_order_amplitudes"], amplitude, atol=1e-12)
    np.testing.assert_allclose(result["variational_energy"], energy, atol=1e-12)


def test_ras_fock_matches_partition_semicanonical_operator():
    from types import SimpleNamespace

    solver = object.__new__(pt.CASPT2)
    solver.mc = SimpleNamespace(ncore=1, ncas=4, ras_spaces=(2, 2, 0))
    rng = np.random.default_rng(53)
    fock = rng.normal(size=(6, 6))
    fock = (fock + fock.T) / 2
    rotation = np.eye(6)
    for s in (slice(1, 3), slice(3, 5)):
        _, rotation[s, s] = np.linalg.eigh(fock[s, s])
    canonical = rotation.T @ fock @ rotation
    canonical[1:5, 1:5] = np.diag(np.diag(canonical)[1:5])
    expected = rotation @ canonical @ rotation.T
    np.testing.assert_allclose(solver._ras_fock(fock), expected, atol=1e-13)
    # Plain CAS has no RAS-partition truncation of its generalized Fock.
    solver.mc.ras_spaces = (0, 4, 0)
    np.testing.assert_array_equal(solver._ras_fock(fock), fock)
