import numpy as np
import pytest
from scipy.linalg import eigh

from pyqed.qchem.pbc import Cell
from pyqed.qchem.pbc.scf import CPHF


def _h2_krhf(nk=1, jk_builder="reciprocal"):
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=nk,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder=jk_builder,
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    )
    if jk_builder == "gdf":
        mf.density_fit(
            auxbasis="sto-3g",
            reciprocal_kernel="full",
            recip_cut=2,
            pair_cut=0,
            pair_screen_tol=0.0,
            metric_tol=1.0e-12,
        )
    return mf.run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _finite_field_density(mf, perturbation, strength):
    dm_k = [
        np.array(mf.dm, copy=True)
        if mf.nkpts == 1
        else np.array(density, copy=True)
        for density in ([mf.dm] if mf.nkpts == 1 else mf.dm)
    ]
    h1_k = (
        [np.asarray(perturbation)]
        if mf.nkpts == 1
        else [np.asarray(block) for block in perturbation]
    )
    for _cycle in range(200):
        fock_k = [
            fock + float(strength) * h1
            for fock, h1 in zip(mf._build_fock_k(dm_k), h1_k)
        ]
        _energy, _coeff, _occupation, density_new = mf._solve_fock(
            fock_k,
            mf._overlap_k,
        )
        residual = max(
            np.linalg.norm(new - old)
            for new, old in zip(density_new, dm_k)
        )
        dm_k = [0.35 * old + 0.65 * new for old, new in zip(dm_k, density_new)]
        if residual < 1.0e-12:
            break
    else:
        raise RuntimeError("Finite-field reference SCF did not converge.")
    return dm_k


def _finite_q_field_density(
    mf,
    perturbations,
    q_index,
    strength,
    overlap_perturbations=None,
):
    nkpts = mf.nkpts
    nao = mf.cell.nao
    pair_by_k = {
        int(k_index): int(kq_index)
        for k_index, kq_index in mf.with_df.pair_keys(q_index)
    }
    self_opposite = all(
        pair_by_k[pair_by_k[k_index]] == k_index
        for k_index in range(nkpts)
    )
    diagonal = [np.array(density, copy=True) for density in mf.dm]
    off_diagonal = [
        np.zeros((nao, nao), dtype=np.complex128) for _kpoint in range(nkpts)
    ]
    overlap = np.zeros(
        (nkpts * nao, nkpts * nao), dtype=np.complex128
    )
    for k_index, block in enumerate(mf._overlap_k):
        rows = slice(k_index * nao, (k_index + 1) * nao)
        overlap[rows, rows] = block
    if overlap_perturbations is not None:
        for k_index, kq_index in pair_by_k.items():
            rows = slice(kq_index * nao, (kq_index + 1) * nao)
            columns = slice(k_index * nao, (k_index + 1) * nao)
            overlap[rows, columns] += (
                float(strength) * overlap_perturbations[k_index]
            )
            if not self_opposite:
                overlap[columns, rows] += (
                    float(strength)
                    * overlap_perturbations[k_index].conj().T
                )
        overlap = 0.5 * (overlap + overlap.conj().T)

    for _cycle in range(300):
        diagonal_fock = mf._build_fock_k(diagonal)
        vj_q, vk_q = mf.with_df.get_jk_response(off_diagonal, q_index)
        q_fock = []
        for k_index, kq_index in pair_by_k.items():
            block = (
                float(strength) * perturbations[k_index]
                + vj_q[k_index]
                - 0.5 * vk_q[k_index]
            )
            if mf.madelung is not None:
                block -= 0.5 * mf.madelung * (
                    mf._overlap_k[kq_index]
                    @ off_diagonal[k_index]
                    @ mf._overlap_k[k_index]
                )
            q_fock.append(block)

        fock = np.zeros_like(overlap)
        for k_index, block in enumerate(diagonal_fock):
            rows = slice(k_index * nao, (k_index + 1) * nao)
            fock[rows, rows] = block
        for k_index, kq_index in pair_by_k.items():
            rows = slice(kq_index * nao, (kq_index + 1) * nao)
            columns = slice(k_index * nao, (k_index + 1) * nao)
            fock[rows, columns] = q_fock[k_index]
            if not self_opposite:
                fock[columns, rows] = q_fock[k_index].conj().T
        fock = 0.5 * (fock + fock.conj().T)

        _energy, coefficients = eigh(fock, overlap)
        electron_pairs = mf.cell.nelectron * nkpts // 2
        occupied = coefficients[:, :electron_pairs]
        density = 2.0 * occupied @ occupied.conj().T
        diagonal_new = []
        off_diagonal_new = []
        for k_index, kq_index in pair_by_k.items():
            columns = slice(k_index * nao, (k_index + 1) * nao)
            rows_k = slice(k_index * nao, (k_index + 1) * nao)
            rows_kq = slice(kq_index * nao, (kq_index + 1) * nao)
            diagonal_new.append(density[rows_k, columns])
            off_diagonal_new.append(density[rows_kq, columns])
        residual = max(
            max(
                np.linalg.norm(new - old)
                for new, old in zip(diagonal_new, diagonal)
            ),
            max(
                np.linalg.norm(new - old)
                for new, old in zip(off_diagonal_new, off_diagonal)
            ),
        )
        diagonal = [
            0.35 * old + 0.65 * new
            for old, new in zip(diagonal, diagonal_new)
        ]
        off_diagonal = [
            0.35 * old + 0.65 * new
            for old, new in zip(off_diagonal, off_diagonal_new)
        ]
        if residual < 1.0e-12:
            return off_diagonal
    raise RuntimeError("Finite-q block SCF did not converge.")


def test_periodic_cphf_two_level_solution():
    coupling = 0.4
    h1 = np.asarray([[[0.0], [0.3]]], dtype=np.complex128)

    def fvind(mo1):
        response = [np.zeros_like(mo1[0])]
        response[0][:, 1, 0] = coupling * mo1[0][:, 1, 0]
        return response

    solver = CPHF(
        fvind,
        [np.asarray([0.0, 2.0])],
        [np.asarray([2.0, 0.0])],
        [h1],
        tol=1.0e-12,
    )
    mo1, _mo_e1 = solver.kernel()

    np.testing.assert_allclose(mo1[0][0, 1, 0], -0.3 / 2.4, atol=1.0e-12)
    assert solver.converged
    assert solver.residual_norm < 1.0e-12


def test_periodic_cphf_handles_real_linear_complex_response():
    coupling = 0.4
    h1 = np.asarray([[[0.0], [0.3 + 0.2j]]], dtype=np.complex128)

    def fvind(mo1):
        response = [np.zeros_like(mo1[0])]
        amplitude = mo1[0][:, 1, 0]
        response[0][:, 1, 0] = coupling * (amplitude + amplitude.conj())
        return response

    solver = CPHF(
        fvind,
        [np.asarray([0.0, 2.0])],
        [np.asarray([2.0, 0.0])],
        [h1],
        tol=1.0e-12,
    )
    mo1, _mo_e1 = solver.kernel()

    expected = -0.3 / (2.0 + 2.0 * coupling) - 0.1j
    np.testing.assert_allclose(mo1[0][0, 1, 0], expected, atol=1.0e-12)


@pytest.mark.parametrize("jk_builder", ["reciprocal", "gdf"])
def test_krhf_cphf_density_matches_finite_field(jk_builder):
    mf = _h2_krhf(jk_builder=jk_builder)
    assert mf.converged
    perturbation = np.asarray([[0.17, 0.11], [0.11, -0.08]])
    response = mf.response().kernel(perturbation, tol=1.0e-11)

    field = 2.0e-4
    plus = _finite_field_density(mf, perturbation, field)[0]
    minus = _finite_field_density(mf, perturbation, -field)[0]
    numerical = (plus - minus) / (2.0 * field)

    np.testing.assert_allclose(response.dm1[0], numerical, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(response.dm1[0], response.dm1[0].conj().T)
    assert response.converged
    assert response.residual_norm < 1.0e-10


def test_krhf_cphf_supports_multiple_kpoints_and_rejects_nonzero_q():
    mf = _h2_krhf(nk=(2, 1, 1))
    assert mf.converged
    perturbations = [
        np.asarray([[0.12, 0.03], [0.03, -0.04]])
        for _kpoint in range(mf.nkpts)
    ]
    response = mf.CPHF().kernel(perturbations, tol=1.0e-10)

    assert response.converged
    assert len(response.dm1) == mf.nkpts
    for density1, overlap in zip(response.dm1, mf._overlap_k):
        np.testing.assert_allclose(
            np.trace(density1[0] @ overlap),
            0.0,
            atol=2.0e-10,
        )

    with pytest.raises(NotImplementedError, match="requires jk_builder='gdf'"):
        mf.response().kernel(perturbations, qpoint=[0.1, 0.0, 0.0])


def test_gdf_zone_boundary_cphf_matches_finite_q_block_scf():
    mf = _h2_krhf(nk=(2, 1, 1), jk_builder="gdf")
    assert mf.converged
    q_index = 1
    qpoint = mf.with_df.qpts[q_index]
    perturbation = np.asarray(
        [[0.12, 0.03 + 0.01j], [0.03 - 0.01j, -0.04]],
        dtype=np.complex128,
    )
    perturbations = [perturbation, perturbation.conj().T]
    response = mf.response().kernel(
        perturbations,
        qpoint=qpoint,
        tol=1.0e-11,
    )

    field = 2.0e-4
    plus = _finite_q_field_density(mf, perturbations, q_index, field)
    minus = _finite_q_field_density(mf, perturbations, q_index, -field)
    numerical = [
        (plus_block - minus_block) / (2.0 * field)
        for plus_block, minus_block in zip(plus, minus)
    ]

    for analytic, reference in zip(response.dm1, numerical):
        np.testing.assert_allclose(analytic[0], reference, atol=3.0e-7, rtol=0.0)
    np.testing.assert_allclose(
        response.dm1[1][0],
        response.dm1[0][0].conj().T,
        atol=2.0e-12,
    )
    vj, vk = mf.with_df.get_jk_response(
        [block[0] for block in response.dm1],
        q_index,
    )
    np.testing.assert_allclose(vj[1], vj[0].conj().T, atol=2.0e-12)
    np.testing.assert_allclose(vk[1], vk[0].conj().T, atol=2.0e-12)
    assert response.q_index == q_index
    assert response.kq_indices == (1, 0)


def test_gdf_general_q_cphf_matches_finite_q_block_scf():
    mf = _h2_krhf(nk=(3, 1, 1), jk_builder="gdf")
    assert mf.converged
    q_index = 1
    qpoint = mf.with_df.qpts[q_index]
    perturbations = [
        np.asarray(
            [[0.12, 0.03 + 0.01j], [-0.02 + 0.04j, -0.04]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[-0.05 + 0.02j, 0.07], [0.01 - 0.03j, 0.09]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[0.03, -0.04 + 0.02j], [0.06 + 0.01j, -0.08 - 0.01j]],
            dtype=np.complex128,
        ),
    ]
    response = mf.response().kernel(
        perturbations,
        qpoint=qpoint,
        tol=1.0e-11,
    )

    field = 2.0e-4
    plus = _finite_q_field_density(mf, perturbations, q_index, field)
    minus = _finite_q_field_density(mf, perturbations, q_index, -field)
    numerical = [
        (plus_block - minus_block) / (2.0 * field)
        for plus_block, minus_block in zip(plus, minus)
    ]

    for analytic, reference in zip(response.dm1, numerical):
        np.testing.assert_allclose(analytic[0], reference, atol=5.0e-7, rtol=0.0)
    assert response.minus_q_index == 2
    assert response.kq_indices == (2, 0, 1)
    assert response.converged
    assert response.residual_norm < 1.0e-10


def test_gdf_general_q_cphf_moving_basis_matches_finite_q_block_scf():
    mf = _h2_krhf(nk=(3, 1, 1), jk_builder="gdf")
    assert mf.converged
    q_index = 1
    qpoint = mf.with_df.qpts[q_index]
    perturbations = [
        np.asarray(
            [[0.12, 0.03 + 0.01j], [-0.02 + 0.04j, -0.04]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[-0.05 + 0.02j, 0.07], [0.01 - 0.03j, 0.09]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[0.03, -0.04 + 0.02j], [0.06 + 0.01j, -0.08 - 0.01j]],
            dtype=np.complex128,
        ),
    ]
    overlap_perturbations = [
        np.asarray(
            [[0.013, -0.004 + 0.002j], [0.006 - 0.003j, -0.008]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[-0.005, 0.007 + 0.001j], [-0.002j, 0.004]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[0.003 + 0.001j, -0.006], [0.002 + 0.004j, -0.009]],
            dtype=np.complex128,
        ),
    ]
    response = mf.response().kernel(
        perturbations,
        s1=overlap_perturbations,
        qpoint=qpoint,
        tol=1.0e-11,
    )

    field = 1.0e-4
    plus = _finite_q_field_density(
        mf,
        perturbations,
        q_index,
        field,
        overlap_perturbations,
    )
    minus = _finite_q_field_density(
        mf,
        perturbations,
        q_index,
        -field,
        overlap_perturbations,
    )
    numerical = [
        (plus_block - minus_block) / (2.0 * field)
        for plus_block, minus_block in zip(plus, minus)
    ]

    for analytic, reference in zip(response.dm1, numerical):
        np.testing.assert_allclose(analytic[0], reference, atol=6.0e-7, rtol=0.0)
    assert response.converged
    assert response.residual_norm < 1.0e-10
