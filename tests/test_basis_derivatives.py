import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.basis_derivatives import (
    _directional_eri_derivative_scalar_cpp,
    _directional_eri_derivatives_cpp,
    _directional_one_electron_derivatives_cpp,
    compact_eri_veff,
    compact_eri_veff_many,
    directional_eri_derivatives,
    directional_one_electron_derivatives,
    eri_derivative_veff_scalar,
    eri_derivatives,
    one_electron_derivatives,
    one_index_eri_derivatives,
    one_index_one_electron_derivatives,
    position_derivatives,
)


def _h2(z):
    mol = Molecule(
        atom=f"H 0 0 0; H 0 0 {z}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    return mol


def test_builtin_first_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 1.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    ds = one_electron_derivatives(mol, "overlap", order=1)
    dh = one_electron_derivatives(mol, "hcore", order=1)
    dg = eri_derivatives(mol, order=1)

    np.testing.assert_allclose(
        ds[1, 2],
        (mol_p.overlap - mol_m.overlap) / (2.0 * step),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        dh[1, 2],
        (mol_p.hcore - mol_m.hcore) / (2.0 * step),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        dg[1, 2],
        (mol_p.eri - mol_m.eri) / (2.0 * step),
        atol=1.0e-8,
    )


def test_builtin_second_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 2.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    d2s = one_electron_derivatives(mol, "overlap", order=2)
    d2h = one_electron_derivatives(mol, "hcore", order=2)
    d2g = eri_derivatives(mol, order=2)

    np.testing.assert_allclose(
        d2s[1, 2, 1, 2],
        (mol_p.overlap - 2.0 * mol.overlap + mol_m.overlap) / step**2,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        d2h[1, 2, 1, 2],
        (mol_p.hcore - 2.0 * mol.hcore + mol_m.hcore) / step**2,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        d2g[1, 2, 1, 2],
        (mol_p.eri - 2.0 * mol.eri + mol_m.eri) / step**2,
        atol=2.0e-7,
    )


def test_directional_derivatives_match_cartesian_projection_h2():
    mol = _h2(1.4)
    rng = np.random.default_rng(12)
    directions = rng.normal(size=(2, mol.natom, 3))

    dh = one_electron_derivatives(mol, "hcore", order=1, backend="python")
    d2h = one_electron_derivatives(mol, "hcore", order=2, backend="python")
    dg = eri_derivatives(mol, order=1)
    d2g = eri_derivatives(mol, order=2)

    dh_projected = directional_one_electron_derivatives(
        mol, directions, "hcore", order=1, backend="native"
    )
    d2h_projected = directional_one_electron_derivatives(
        mol, directions, "hcore", order=2, backend="native"
    )
    dg_projected = directional_eri_derivatives(
        mol, directions, order=1, backend="native"
    )
    d2g_projected = directional_eri_derivatives(
        mol, directions, order=2, backend="native"
    )

    np.testing.assert_allclose(
        dh_projected,
        np.einsum("mAx,Axpq->mpq", directions, dh, optimize=True),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        d2h_projected,
        np.einsum("mAx,nBy,AxBypq->mnpq", directions, directions, d2h, optimize=True),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        dg_projected,
        np.einsum("mAx,Axpqrs->mpqrs", directions, dg, optimize=True),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        d2g_projected,
        np.einsum("mAx,nBy,AxBypqrs->mnpqrs", directions, directions, d2g, optimize=True),
        atol=1.0e-10,
    )


def test_pyscf_directional_derivatives_match_native_with_p_shells():
    pytest.importorskip("pyscf")
    mol = Molecule(
        atom=(
            "O 0 0 0; "
            "H 0 -1.43233673 1.10715266; "
            "H 0 1.43233673 1.10715266"
        ),
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    directions = np.random.default_rng(12).normal(size=(2, mol.natom, 3))

    for order in (1, 2):
        reference = directional_eri_derivatives(
            mol, directions, order=order, backend="python"
        )
        native = directional_eri_derivatives(
            mol, directions, order=order, backend="native"
        )
        compiled = directional_eri_derivatives(
            mol, directions, order=order, backend="pyscf"
        )
        automatic = directional_eri_derivatives(mol, directions, order=order)
        np.testing.assert_allclose(native, reference, atol=2.0e-10, rtol=1.0e-10)
        np.testing.assert_allclose(compiled, native, atol=2.0e-7, rtol=1.0e-7)
        np.testing.assert_allclose(automatic, compiled, atol=0.0, rtol=0.0)

    for order in (1, 2):
        native_hcore = directional_one_electron_derivatives(
            mol, directions, "hcore", order=order, backend="native"
        )
        compiled_hcore = directional_one_electron_derivatives(
            mol, directions, "hcore", order=order, backend="pyscf"
        )
        automatic_hcore = directional_one_electron_derivatives(
            mol, directions, "hcore", order=order
        )
        np.testing.assert_allclose(
            compiled_hcore, native_hcore, atol=1.0e-6, rtol=1.0e-7
        )
        np.testing.assert_allclose(automatic_hcore, native_hcore, atol=0.0, rtol=0.0)


def test_cpp_directional_one_electron_derivatives_match_python_reference():
    from pyqed.qchem.basis import _integrals_cpp

    if _integrals_cpp is None or not hasattr(
        _integrals_cpp, "compute_directional_one_electron_derivatives"
    ):
        pytest.skip("C++ directional one-electron derivative extension is unavailable")

    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    mol.builtin_parallel = True
    mol.builtin_parallel_min_nao = 0
    mol.builtin_eri_workers = 2
    directions = np.random.default_rng(23).normal(size=(2, mol.natom, 3))

    for kernel in ("overlap", "kinetic", "nuclear", "hcore"):
        for order in (1, 2):
            actual = _directional_one_electron_derivatives_cpp(
                mol, directions, kernel, order
            )
            reference = directional_one_electron_derivatives(
                mol, directions, kernel, order=order, backend="python"
            )
            np.testing.assert_allclose(actual, reference, atol=1.0e-10, rtol=1.0e-10)


def test_cpp_directional_eri_derivatives_match_python_reference():
    from pyqed.qchem.basis import _integrals_cpp

    if _integrals_cpp is None or not hasattr(
        _integrals_cpp, "compute_directional_eri_derivatives"
    ):
        pytest.skip("C++ directional ERI derivative extension is unavailable")

    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    mol.builtin_parallel = True
    mol.builtin_parallel_min_nao = 0
    mol.builtin_eri_workers = 2
    directions = np.random.default_rng(19).normal(size=(2, mol.natom, 3))

    for order in (1, 2):
        actual = _directional_eri_derivatives_cpp(mol, directions, order)
        reference = directional_eri_derivatives(
            mol, directions, order=order, backend="python"
        )
        np.testing.assert_allclose(actual, reference, atol=2.0e-10, rtol=1.0e-10)


def test_cpp_directional_eri_scalar_contraction_matches_dense_p_shells():
    from pyqed.qchem.basis import _integrals_cpp

    if _integrals_cpp is None or not hasattr(
        _integrals_cpp, "compute_directional_eri_derivative_scalar"
    ):
        pytest.skip("C++ derivative-contraction extension is unavailable")

    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    rng = np.random.default_rng(41)
    directions = rng.normal(size=(2, mol.natom, 3))
    dm_left = rng.normal(size=(mol.nao, mol.nao))
    dm_right = rng.normal(size=(mol.nao, mol.nao))

    for order in (1, 2):
        derivative = _directional_eri_derivatives_cpp(mol, directions, order)
        veff = np.einsum("...pqrs,rs->...pq", derivative, dm_right, optimize=True)
        veff -= 0.5 * np.einsum(
            "...prqs,rs->...pq", derivative, dm_right, optimize=True
        )
        reference = np.einsum("pq,...pq->...", dm_left, veff, optimize=True)
        actual = _directional_eri_derivative_scalar_cpp(
            mol,
            directions,
            dm_left,
            dm_right,
            order=order,
            workers=2,
        )
        np.testing.assert_allclose(actual, reference, atol=2.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("angular_momentum", (2, 3))
def test_cpp_directional_eri_derivatives_support_d_and_f_shells(
    angular_momentum,
):
    from pyqed.qchem.basis import _integrals_cpp, _shell

    if _integrals_cpp is None or not hasattr(
        _integrals_cpp, "compute_directional_eri_derivatives"
    ):
        pytest.skip("C++ directional ERI derivative extension is unavailable")

    shells = np.asarray(
        _shell(angular_momentum) + [(0, 0, 0)], dtype=np.int64
    )
    nao = len(shells)
    origins = np.zeros((nao, 3))
    origins[-1, 2] = 1.4
    exps = np.full((nao, 1), 0.8)
    exps[-1, 0] = 0.7
    weights = np.ones((nao, 1))
    nprim = np.ones(nao, dtype=np.int64)
    atom_ids = np.zeros(nao, dtype=np.int64)
    atom_ids[-1] = 1
    directions = np.array([[[0.1, -0.2, 0.3], [-0.15, 0.05, -0.25]]])
    pair_bounds = np.ones((nao, nao))

    def eri(displacement):
        moved = origins + displacement * directions[0, atom_ids]
        return _integrals_cpp.compute_dense_eri_cartesian(
            shells,
            moved,
            exps,
            weights,
            nprim,
            pair_bounds,
            0.0,
            6,
            2,
        )[0]

    first = _integrals_cpp.compute_directional_eri_derivatives(
        shells, origins, exps, weights, nprim, atom_ids, directions, 1, 2
    )[0]
    second = _integrals_cpp.compute_directional_eri_derivatives(
        shells, origins, exps, weights, nprim, atom_ids, directions, 2, 2
    )[0, 0]
    step_first = 2.0e-5
    step_second = 2.0e-4
    reference = eri(0.0)
    finite_first = (
        eri(step_first) - eri(-step_first)
    ) / (2.0 * step_first)
    finite_second = (
        eri(step_second) - 2.0 * reference + eri(-step_second)
    ) / step_second**2

    np.testing.assert_allclose(first, finite_first, atol=1.0e-9, rtol=1.0e-8)
    np.testing.assert_allclose(second, finite_second, atol=2.0e-7, rtol=1.0e-7)


def test_native_directional_eri_keeps_cross_shell_pair_components():
    pytest.importorskip("pyscf")
    mol = Molecule(
        atom="H 0 0 0; F 0 0 1.7",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")
    directions = np.zeros((1, mol.natom, 3))
    directions[0, 0, 2] = 1.0

    for order in (1, 2):
        native = directional_eri_derivatives(
            mol, directions, order=order, backend="native"
        )
        reference = directional_eri_derivatives(
            mol, directions, order=order, backend="pyscf"
        )
        np.testing.assert_allclose(native, reference, atol=1.0e-8, rtol=1.0e-8)

    # This component is lost if shell-pair and AO-pair canonical orderings are mixed.
    assert abs(native[0, 0, 0, 5, 3, 3]) > 1.0e-3


def test_compact_eri_derivative_contractions_match_dense_h2():
    mol = _h2(1.4)
    rng = np.random.default_rng(3)
    dm = rng.normal(size=(mol.nao, mol.nao))
    dm = dm + dm.T

    dense1 = eri_derivatives(mol, order=1)
    compact1 = eri_derivatives(mol, order=1, compact=True)
    dense2 = eri_derivatives(mol, order=2)
    compact2 = eri_derivatives(mol, order=2, compact=True)

    def dense_veff(eri):
        vj = np.einsum("rs,pqrs->pq", dm, eri, optimize=True)
        vk = np.einsum("rs,prqs->pq", dm, eri, optimize=True)
        return vj - 0.5 * vk

    np.testing.assert_allclose(
        compact_eri_veff(compact1, dm, 5),
        dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff_many(compact1, dm)[5],
        dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff(compact2, dm, 5, 5),
        dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        compact_eri_veff_many(compact2, dm).reshape(6, 6, mol.nao, mol.nao)[5, 5],
        dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        eri_derivative_veff_scalar(mol, dm, dm, order=1)[5],
        np.einsum(
            "pq,pq->",
            dm,
            dense_veff(dense1.reshape(-1, mol.nao, mol.nao, mol.nao, mol.nao)[5]),
            optimize=True,
        ),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        eri_derivative_veff_scalar(mol, dm, dm, order=2)[5, 5],
        np.einsum(
            "pq,pq->",
            dm,
            dense_veff(dense2.reshape(6, 6, mol.nao, mol.nao, mol.nao, mol.nao)[5, 5]),
            optimize=True,
        ),
        atol=1.0e-10,
    )


def test_one_index_eri_derivatives_reconstruct_total_derivative_h2():
    mol = _h2(1.4)
    total = eri_derivatives(mol, order=1)
    one = one_index_eri_derivatives(mol)

    rebuilt = (
        one
        + one.transpose(0, 1, 3, 2, 4, 5)
        + one.transpose(0, 1, 4, 5, 2, 3)
        + one.transpose(0, 1, 4, 5, 3, 2)
    )
    np.testing.assert_allclose(rebuilt, total, atol=1.0e-10)

    packed = one_index_eri_derivatives(mol, aosym="s2kl")
    pairs = [(p, q) for p in range(mol.nao) for q in range(p + 1)]
    for pair, (r, s) in enumerate(pairs):
        np.testing.assert_allclose(packed[..., pair], one[..., r, s], atol=1.0e-12)


def test_one_index_one_electron_derivatives_reconstruct_total_overlap_h2():
    mol = _h2(1.4)
    total = one_electron_derivatives(mol, "overlap", order=1, backend="python")
    ket = one_index_one_electron_derivatives(
        mol, "overlap", index="ket", backend="native"
    )
    bra = one_index_one_electron_derivatives(
        mol, "overlap", index="bra", backend="native"
    )
    ket_ref = one_index_one_electron_derivatives(
        mol, "overlap", index="ket", backend="python"
    )

    np.testing.assert_allclose(ket, ket_ref, atol=1.0e-12)
    np.testing.assert_allclose(bra, ket.transpose(0, 1, 3, 2), atol=1.0e-12)
    np.testing.assert_allclose(bra + ket, total, atol=1.0e-10)


def test_cpp_one_index_derivatives_match_python_with_p_shells():
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1")

    for kernel in ("overlap", "kinetic"):
        for index in ("bra", "ket"):
            for order in (1, 2):
                native = one_index_one_electron_derivatives(
                    mol, kernel, index=index, order=order, backend="native"
                )
                reference = one_index_one_electron_derivatives(
                    mol, kernel, index=index, order=order, backend="python"
                )
                np.testing.assert_allclose(
                    native, reference, atol=1.0e-10, rtol=1.0e-10
                )


def test_one_index_second_overlap_matches_asymmetric_finite_difference_h2():
    gto = pytest.importorskip("pyscf.gto")
    mol = _h2(1.4)
    step = 2.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)
    pmol = mol.topyscf()
    cross_p = gto.intor_cross("int1e_ovlp", pmol, mol_p.topyscf())
    cross_0 = gto.intor_cross("int1e_ovlp", pmol, pmol)
    cross_m = gto.intor_cross("int1e_ovlp", pmol, mol_m.topyscf())

    ket2 = one_index_one_electron_derivatives(
        mol,
        "overlap",
        index="ket",
        order=2,
        backend="native",
    )
    ket2_ref = one_index_one_electron_derivatives(
        mol, "overlap", index="ket", order=2, backend="python"
    )
    np.testing.assert_allclose(ket2, ket2_ref, atol=1.0e-11)
    np.testing.assert_allclose(
        ket2[1, 2, 1, 2],
        (cross_p - 2.0 * cross_0 + cross_m) / step**2,
        atol=2.0e-7,
    )


def test_one_index_eri_derivatives_match_pyscf_ip1_h2():
    pyscf = pytest.importorskip("pyscf")

    mol = _h2(1.4)
    pmol = mol.topyscf()
    pmol.build()
    ref = pmol.intor("int2e_ip1", comp=3, aosym="s1").reshape(
        3,
        mol.nao,
        mol.nao,
        mol.nao,
        mol.nao,
    )
    got = one_index_eri_derivatives(mol, convention="ip1").sum(axis=0)
    np.testing.assert_allclose(got, ref, atol=1.0e-10)


def test_builtin_position_derivatives_match_finite_difference_h2():
    mol = _h2(1.4)
    step = 1.0e-4
    mol_p = _h2(1.4 + step)
    mol_m = _h2(1.4 - step)

    dr = position_derivatives(mol, center=np.zeros(3))

    np.testing.assert_allclose(
        dr[1, 2],
        (mol_p.position_integral(center=np.zeros(3)) - mol_m.position_integral(center=np.zeros(3)))
        / (2.0 * step),
        atol=1.0e-8,
    )


def test_builtin_derivatives_follow_spherical_ao_transform_h2o():
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense")

    ds = one_electron_derivatives(mol, "overlap", order=1)
    dh = one_electron_derivatives(mol, "hcore", order=1)
    dg = eri_derivatives(mol, order=1)

    assert ds.shape == (mol.natom, 3, mol.nao, mol.nao)
    assert dh.shape == (mol.natom, 3, mol.nao, mol.nao)
    assert dg.shape == (mol.natom, 3, mol.nao, mol.nao, mol.nao, mol.nao)
