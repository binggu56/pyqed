from types import SimpleNamespace

import numpy as np

from pyqed.narg.qchem.su2 import NARG
from pyqed.narg.qchem.su2_overlap import (
    _graph_block_orbital_map,
    narg_reduced_mps_root_batch,
    narg_reduced_mps_states,
)
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI


class _DummyMol:
    nelec = (2, 2)
    spin = 0

    def energy_nuc(self):
        return 0.0


def _hubbard_solver(nsites=4, nstates=3, target_j2=0):
    h1e = np.zeros((nsites, nsites))
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -0.7
    for site in range(nsites):
        eri[site, site, site, site] = 2.0
    mol = _DummyMol()
    return NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=80,
        nstates=nstates,
        final_size=nsites,
        target_nelec=nsites,
        target_j2=target_j2,
        su2_backend="python",
    ).run()


def test_su2_narg_recursive_overlap_is_identity_without_determinant_expansion(
    monkeypatch,
):
    import pyqed.narg.qchem.su2_overlap as overlap_module

    solver = _hubbard_solver()
    transform_calls = []
    original_transform = overlap_module.apply_spatial_orbital_transform

    def counted_transform(*args, **kwargs):
        transform_calls.append(1)
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(
        overlap_module,
        "apply_spatial_orbital_transform",
        counted_transform,
    )
    overlap, info = solver.overlap(
        solver,
        mo_overlap=np.eye(4),
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    np.testing.assert_allclose(overlap, np.eye(3), atol=8.0e-15)
    assert info["backend"] == "recursive_su2"
    assert info["exact"] is True
    assert info["sector_preserving"] is True
    assert info["determinant_expansion"] is False
    assert info["component_expansion"] is False
    assert info["batched_roots"] is True
    assert info["root_batch_sizes"] == {"bra": 3, "ket": 3}
    assert info["orbital_transform_calls"] == 0
    assert info["overlap_contractions"] == 1
    assert len(transform_calls) == 0
    states = narg_reduced_mps_states(solver)
    assert len(states) == 3
    assert all(len(state.sites) == 4 for state in states)


def test_su2_narg_batched_overlap_preserves_requested_root_order():
    solver = _hubbard_solver()

    overlap = solver.overlap(
        solver,
        mo_overlap=np.eye(4),
        bra_state_ids=[2, 0],
        ket_state_ids=[0, 2],
        cutoff=0.0,
        max_bond=None,
    )

    np.testing.assert_allclose(overlap, [[0.0, 1.0], [1.0, 0.0]], atol=8.0e-15)


def test_su2_narg_batched_overlap_has_correct_triplet_reduced_normalization():
    solver = _hubbard_solver(target_j2=2)

    overlap = solver.overlap(
        solver,
        mo_overlap=np.eye(4),
        cutoff=0.0,
        max_bond=None,
    )

    np.testing.assert_allclose(overlap, np.eye(3), atol=8.0e-15)


def test_su2_narg_recursive_overlap_connects_rotated_four_orbital_chains():
    nsites = 4
    h1e = np.diag([-1.1, -0.35, 0.22, 0.81])
    h1e[np.arange(3), np.arange(1, 4)] = [-0.27, -0.19, -0.13]
    h1e = h1e + np.triu(h1e, 1).T
    eri = np.zeros((nsites,) * 4)
    eri[np.arange(nsites), np.arange(nsites), np.arange(nsites), np.arange(nsites)] = [
        1.8,
        1.5,
        1.2,
        0.9,
    ]
    orbital_rotation = np.linalg.qr(np.random.default_rng(4).normal(size=(nsites, nsites)))[0]
    rotated_h1e = orbital_rotation.T @ h1e @ orbital_rotation
    rotated_eri = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        eri,
        optimize=True,
    )
    mol = _DummyMol()
    options = dict(
        mol=mol,
        D=80,
        nstates=3,
        final_size=nsites,
        target_nelec=nsites,
        target_j2=0,
        su2_backend="python",
    )
    reference = NARG(
        SimpleNamespace(mol=mol), h1e=h1e, eri=eri, **options
    ).run()
    rotated = NARG(
        SimpleNamespace(mol=mol), h1e=rotated_h1e, eri=rotated_eri, **options
    ).run()

    overlap, info = reference.overlap(
        rotated,
        mo_overlap=orbital_rotation,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    np.testing.assert_allclose(reference.e_tot, rotated.e_tot, atol=5.0e-14)
    np.testing.assert_allclose(np.abs(overlap), np.eye(3), atol=8.0e-14)
    assert info["orbital_split"] in {"bra_only", "ket_only"}
    assert info["orbital_transform_calls"] == 1
    transformed_side = next(
        side
        for side in ("bra", "ket")
        if info["transforms"][side][0]["method"] != "identity_skip"
    )
    transform_info = info["transforms"][transformed_side][0]
    assert transform_info["orbital_factorization"] == "unitary_givens"
    assert transform_info["adjacent_gate_count"] == 6
    balanced, balanced_info = reference.overlap(
        rotated,
        mo_overlap=orbital_rotation,
        orbital_split="balanced",
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )
    np.testing.assert_allclose(overlap, balanced, atol=8.0e-14)
    assert balanced_info["orbital_transform_calls"] == 2


def test_su2_narg_overlap_graph_order_reduces_chain_cut_cost():
    solver = _hubbard_solver()
    overlap = np.eye(4)
    overlap[0, 3] = overlap[3, 0] = 0.25
    overlap[1, 2] = overlap[2, 1] = 0.20

    order, info = solver.overlap_orbital_order(
        solver,
        mo_overlap=overlap,
        return_info=True,
    )

    assert sorted(order) == list(range(4))
    assert info["cut_cost_after"] < info["cut_cost_before"]


def test_su2_narg_overlap_graph_threshold_reports_controlled_map_residual():
    orbital_map = np.array(
        [
            [1.0, 0.08, 2.0e-7, 0.0],
            [-0.03, 1.0, 0.0, -3.0e-7],
            [1.0e-7, 0.0, 1.0, -0.06],
            [0.0, 2.0e-7, 0.04, 1.0],
        ],
        dtype=complex,
    )

    approximated, blocks, residual = _graph_block_orbital_map(
        orbital_map,
        1.0e-6,
    )

    assert blocks == [(0, 1), (2, 3)]
    assert residual > 0.0
    assert residual <= np.linalg.norm(orbital_map - approximated)
    np.testing.assert_allclose(approximated[:2, 2:], 0.0)
    np.testing.assert_allclose(approximated[2:, :2], 0.0)

    solver = _hubbard_solver(nstates=1)
    exact = solver.overlap(
        solver,
        mo_overlap=orbital_map,
        orbital_split="ket_only",
        orbital_map_threshold=0.0,
        cutoff=0.0,
        max_bond=None,
    )
    thresholded, info = solver.overlap(
        solver,
        mo_overlap=orbital_map,
        orbital_split="ket_only",
        orbital_map_threshold=1.0e-6,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    assert info["exact"] is False
    assert info["orbital_map_residual"] > 0.0
    assert info["transforms"]["ket"][0]["orbital_block_count"] == 2
    np.testing.assert_allclose(thresholded, exact, atol=2.0e-6)


def _run_h2(distance):
    mol = Molecule(
        atom=f"H 0 0 0; H 0 0 {distance}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = RHF(mol).run()
    casci = CASCI(mf, ncas=2, nelecas=2, ncore=0, spin=0).run(
        nstates=4,
        method="direct_ci",
        use_cholesky=False,
    )
    narg = NARG(
        mf,
        ncas=2,
        nelecas=2,
        ncore=0,
        D=16,
        nstates=3,
        target_j2=0,
        su2_backend="python",
    ).run()
    return casci, narg


def test_su2_narg_cross_geometry_overlap_matches_exact_casci_singlets():
    cas_bra, narg_bra = _run_h2(1.4)
    cas_ket, narg_ket = _run_h2(1.5)

    actual, info = narg_bra.overlap(
        narg_ket,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )
    cas_overlap = cas_bra.overlap(cas_ket)
    singlets = np.array([0, 2, 3])
    expected = cas_overlap[np.ix_(singlets, singlets)]

    np.testing.assert_allclose(np.abs(actual), np.abs(expected), atol=2.0e-11)
    assert info["exact"] is True
    assert info["core_factor"] == 1.0
    assert info["mo_overlap"].shape == (2, 2)
    batch = narg_reduced_mps_root_batch(narg_bra)
    assert len(batch.sites[-1].qns[2]) == 3
    assert next(iter(batch.sites[-1].data.values())).shape[-1] == 3


def _run_lih(distance):
    mol = Molecule(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = RHF(mol).run()
    casci = CASCI(mf, ncas=2, nelecas=2, ncore=1, spin=0).run(
        nstates=4,
        method="direct_ci",
        use_cholesky=False,
    )
    narg = NARG(
        mf,
        ncas=2,
        nelecas=2,
        ncore=1,
        D=16,
        nstates=3,
        target_j2=0,
        su2_backend="python",
    ).run()
    return casci, narg


def test_su2_narg_cross_geometry_overlap_includes_frozen_core_schur_complement():
    cas_bra, narg_bra = _run_lih(3.0)
    cas_ket, narg_ket = _run_lih(3.1)

    actual, info = narg_bra.overlap(
        narg_ket,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )
    cas_overlap = cas_bra.overlap(cas_ket)
    singlets = np.array([0, 2, 3])
    expected = cas_overlap[np.ix_(singlets, singlets)]

    np.testing.assert_allclose(np.abs(actual), np.abs(expected), atol=3.0e-11)
    assert abs(info["core_factor"] - 1.0) > 1.0e-8
    assert info["mo_overlap"].shape == (3, 3)

    transported_driver, transport = NARG.from_parallel_transport(
        narg_bra,
        narg_ket.mf,
        transport_method="polar",
        ncas=2,
        nelecas=2,
        ncore=1,
        D=16,
        nstates=3,
        target_j2=0,
        su2_backend="python",
        return_info=True,
    )
    transported = transported_driver.run()
    transported_overlap = narg_bra.overlap(
        transported,
        cutoff=0.0,
        max_bond=None,
    )
    np.testing.assert_allclose(
        np.abs(transported_overlap),
        np.abs(expected),
        atol=3.0e-11,
    )
    assert (
        transport["active"]["offdiagonal_norm_after"]
        < transport["active"]["offdiagonal_norm_before"]
    )

    _matched, matched_info = narg_bra.parallel_transport_orbitals(
        narg_ket.mf,
        method="match",
        return_info=True,
    )
    rotation = matched_info["active_rotation"]
    assert np.all(np.count_nonzero(np.abs(rotation) > 1.0e-14, axis=0) == 1)
    assert np.all(np.count_nonzero(np.abs(rotation) > 1.0e-14, axis=1) == 1)
