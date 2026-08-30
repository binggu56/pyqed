import pytest
import numpy as np

from examples.mps.hubbard_2d_mps_vs_letta import (
    _bond_singlet_pair_correlation,
    _structure_factor_at,
    _structure_factor_grid,
    ed_phase_gaps,
    ed_ground_energy,
    hubbard_site_operators,
    hubbard_2d_dense_mpo,
    hubbard_2d_local_hamiltonian,
    lattice_distance_average,
    projected_mpo_spectrum,
    random_fixed_sector_abelian_mps,
    rung_site_qn_maps,
    site_qn_maps,
)
from pyqed.mps import MPO, dense_to_symmetric_mpo
from pyqed.mps.abelian_storage import SymmetryManager
from pyqed.mps.dmrg import DMRG, dmrg_matvec_options
from pyqed.letta import FrontierLETTA, TermwiseBlockMPOFrontier
from pyqed.letta.vmc import LocalHamiltonianActions


def test_analytical_hubbard_strings_match_existing_2x2_mpo():
    analytical, info = hubbard_2d_local_hamiltonian(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    factors, _mpo_info = hubbard_2d_dense_mpo(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )

    assert len(info["bonds"]) == 4
    np.testing.assert_allclose(
        analytical.to_dense(),
        MPO(factors).to_dense(),
        atol=2.0e-12,
    )

    actions = LocalHamiltonianActions(analytical)
    dense = analytical.to_dense()
    for configuration in (
        np.asarray((0, 1, 2, 3)),
        np.asarray((3, 0, 1, 2)),
        np.asarray((1, 2, 1, 2)),
    ):
        row = np.zeros(dense.shape[1], dtype=complex)
        for value, target in actions.configuration_actions(configuration):
            row[np.ravel_multi_index(tuple(target), analytical.dims)] += value
        np.testing.assert_allclose(
            row,
            dense[np.ravel_multi_index(tuple(configuration), analytical.dims)],
            atol=2.0e-12,
        )


def test_u1_hubbard_frontier_accepts_exact_termwise_backend():
    hamiltonian, info = hubbard_2d_local_hamiltonian(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    state = FrontierLETTA(
        hamiltonian,
        graph=info["bonds"],
        target_charge={"n": 4, "2sz": 0},
        D=2,
        seed=18,
        frontier_backend="termwise",
        chunk_size=3,
    )

    assert isinstance(state._hamiltonian_frontier, TermwiseBlockMPOFrontier)
    assert state._hamiltonian_frontier.nchunks == 7
    assert state._hamiltonian_frontier.chunk_size == 3
    assert state.hamiltonian_chunks == (3, 3, 3, 3, 3, 3, 2)
    assert state.stream_peak_frontier_elements > 0
    assert state.contraction_is_exact
    assert hamiltonian._materialized_terms is None
    np.testing.assert_allclose(
        state.energy,
        hamiltonian.expectation(state.state_vector()),
        atol=3.0e-12,
    )

    unchunked = FrontierLETTA(
        hamiltonian,
        graph=info["bonds"],
        target_charge={"n": 4, "2sz": 0},
        D=2,
        tensors=state.tensors,
        frontier_backend="termwise",
        chunk_size=1,
    )
    assert unchunked._hamiltonian_frontier.nchunks == hamiltonian.nproducts
    np.testing.assert_allclose(unchunked.energy, state.energy, atol=3.0e-12)
    probe = np.linspace(-0.3, 0.8, state.tensors[1].size)
    np.testing.assert_allclose(
        unchunked.hamiltonian_action(1, probe),
        state.hamiltonian_action(1, probe),
        atol=3.0e-12,
    )
    copied = state.copy()
    assert copied.chunk_size == 3
    assert copied.chunk_memory == 64
    assert copied._hamiltonian_frontier.nchunks == 7

    bounded = FrontierLETTA(
        hamiltonian,
        graph=info["bonds"],
        target_charge={"n": 4, "2sz": 0},
        D=2,
        tensors=state.tensors,
        frontier_backend="termwise",
        chunk_size=8,
        chunk_memory=16 / 2**20,
    )
    assert bounded.hamiltonian_chunks == (1,) * hamiltonian.nproducts
    np.testing.assert_allclose(bounded.energy, state.energy, atol=3.0e-12)


def test_termwise_chunk_size_must_be_positive_integer():
    hamiltonian, info = hubbard_2d_local_hamiltonian(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    with pytest.raises(TypeError, match="positive integer"):
        FrontierLETTA(
            hamiltonian,
            graph=info["bonds"],
            frontier_backend="termwise",
            chunk_size=True,
        )
    with pytest.raises(ValueError, match="positive integer"):
        FrontierLETTA(
            hamiltonian,
            graph=info["bonds"],
            frontier_backend="termwise",
            chunk_size=0,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        FrontierLETTA(
            hamiltonian,
            graph=info["bonds"],
            frontier_backend="termwise",
            chunk_memory=0,
        )


def test_termwise_active_windows_and_parallel_chunks_preserve_exact_energy():
    hamiltonian, info = hubbard_2d_local_hamiltonian(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    serial = FrontierLETTA(
        hamiltonian,
        graph=info["bonds"],
        target_charge={"n": 4, "2sz": 0},
        D=2,
        seed=23,
        frontier_backend="termwise",
        chunk_size=3,
        chunk_span=3,
    )
    parallel = FrontierLETTA(
        hamiltonian,
        graph=info["bonds"],
        target_charge={"n": 4, "2sz": 0},
        D=2,
        tensors=serial.tensors,
        frontier_backend="termwise",
        chunk_size=3,
        chunk_span=3,
        workers=2,
    )

    assert any(
        stop - start < len(hamiltonian.sites)
        for start, stop in serial.hamiltonian_windows
    )
    assert parallel.workers == 2
    assert parallel.chunk_span == 3
    np.testing.assert_allclose(parallel.energy, serial.energy, atol=3.0e-12)
    np.testing.assert_allclose(
        parallel.energy,
        hamiltonian.expectation(parallel.state_vector()),
        atol=3.0e-12,
    )
    probe = np.linspace(-0.4, 0.7, serial.tensors[1].size)
    np.testing.assert_allclose(
        parallel.hamiltonian_action(1, probe),
        serial.hamiltonian_action(1, probe),
        atol=3.0e-12,
    )


def test_termwise_parallel_and_span_options_are_validated():
    hamiltonian, info = hubbard_2d_local_hamiltonian(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    with pytest.raises(ValueError, match="chunk_span"):
        FrontierLETTA(
            hamiltonian,
            graph=info["bonds"],
            frontier_backend="termwise",
            chunk_span=0,
        )
    with pytest.raises(ValueError, match="workers"):
        FrontierLETTA(
            hamiltonian,
            graph=info["bonds"],
            frontier_backend="termwise",
            workers=0,
        )


def test_lattice_phase_diagnostics_resolve_staggered_peak_and_distance():
    coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
    stagger = np.asarray([1.0, -1.0, -1.0, 1.0])
    correlation = np.outer(stagger, stagger)

    assert _structure_factor_at(correlation, coords, np.pi, np.pi) == pytest.approx(4.0)
    assert _structure_factor_at(correlation, coords, 0.0, 0.0) == pytest.approx(0.0)
    _grid, peak = _structure_factor_grid(correlation, coords, lx=2, ly=2)
    assert peak == pytest.approx({"qx_over_pi": 1.0, "qy_over_pi": 1.0, "value": 4.0})
    assert lattice_distance_average(
        correlation * np.outer(stagger, stagger),
        coords,
        lx=2,
        ly=2,
    ) == pytest.approx([1.0, 1.0, 1.0])


def test_ed_phase_gaps_for_half_filled_2x2_hubbard():
    gaps = ed_phase_gaps(
        2,
        2,
        nup=2,
        ndown=2,
        hopping=1.0,
        hubbard_u=4.0,
        mu=0.0,
        periodic_x=False,
        periodic_y=False,
    )

    assert gaps["charge_gap"] == pytest.approx(2.70118105377, abs=1.0e-10)
    assert gaps["spin_gap"] == pytest.approx(0.296324631639, abs=1.0e-10)


def test_bond_pair_diagnostic_recognizes_normalized_two_site_singlet():
    wavefunction = np.zeros((4, 4), dtype=complex)
    wavefunction[1, 2] = 1.0 / np.sqrt(2.0)
    wavefunction[2, 1] = -1.0 / np.sqrt(2.0)
    left, singular_values, right = np.linalg.svd(wavefunction, full_matrices=False)
    root = np.sqrt(singular_values)
    factors = [
        (left * root[None, :])[None, :, :],
        (root[:, None] * right)[:, :, None],
    ]

    correlation = _bond_singlet_pair_correlation(
        factors,
        hubbard_site_operators(),
        [(0, 1)],
        [(0, None), (1, None)],
        [4, 4],
    )

    np.testing.assert_allclose(correlation, [[1.0]], atol=1.0e-12)


def test_abelian_dmrg_history_reports_post_truncation_energy_for_2d_hubbard():
    dense_mpo, _info = hubbard_2d_dense_mpo(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    qn_maps = site_qn_maps(4)
    opts = dmrg_matvec_options("packed-cpp-fast")
    symmetric_mpo = dense_to_symmetric_mpo(
        dense_mpo,
        qn_maps,
        native_site_storage=bool(opts.get("native_site_storage", False)),
    )
    sym_mgr = SymmetryManager(["charge", "sz"])
    target_qn = sym_mgr.get_target_qn(4, 0)
    initial = random_fixed_sector_abelian_mps(
        4,
        2,
        2,
        max_bond_dim=8,
        qn_maps=qn_maps,
        native_site_storage=bool(opts.get("native_site_storage", False)),
        seed=9,
    )

    dmrg = DMRG(
        MPO(symmetric_mpo),
        D=8,
        init_guess=initial,
        nsweeps=2,
        opt="2site",
        symmetry=True,
        target_qn=target_qn,
        sym_mgr=sym_mgr,
        site_qn_maps=qn_maps,
        not_conv_err=False,
        performance="packed-cpp-fast",
        abelian_matvec_options=opts,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=80,
        noise=1.0e-7,
    )
    dmrg.run()

    last = dmrg.sweep_history[-1]
    assert last["energy"] == pytest.approx(dmrg.energy, abs=1.0e-12)
    assert last["post_truncation_energy"] == pytest.approx(dmrg.energy, abs=1.0e-12)
    assert last["local_energy"] < dmrg.energy - 0.1


def test_rung_supersite_hubbard_mpo_matches_2x2_ed_spectrum():
    dense_mpo, info = hubbard_2d_dense_mpo(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
        site_grouping="rung",
    )
    projected = projected_mpo_spectrum(
        dense_mpo,
        nup=2,
        ndown=2,
        nroots=4,
        qn_maps=rung_site_qn_maps(2, 2),
    )
    ed, _info = ed_ground_energy(
        2,
        2,
        nup=2,
        ndown=2,
        hopping=1.0,
        hubbard_u=4.0,
        mu=0.0,
        periodic_x=False,
        periodic_y=False,
        nroots=4,
    )

    assert info["site_grouping"] == "rung"
    assert info["fused_blocks"] == [[0, 2], [1, 3]]
    np.testing.assert_allclose(projected, ed, atol=1.0e-12)


def test_column_supersite_hubbard_mpo_matches_2x3_ed_spectrum():
    dense_mpo, info = hubbard_2d_dense_mpo(
        2,
        3,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
        site_grouping="rung",
    )
    projected = projected_mpo_spectrum(
        dense_mpo,
        nup=3,
        ndown=3,
        nroots=4,
        qn_maps=rung_site_qn_maps(2, 3),
    )
    ed, _info = ed_ground_energy(
        2,
        3,
        nup=3,
        ndown=3,
        hopping=1.0,
        hubbard_u=4.0,
        mu=0.0,
        periodic_x=False,
        periodic_y=False,
        nroots=4,
    )

    assert info["site_grouping"] == "rung"
    assert info["fused_blocks"] == [[0, 2, 4], [1, 3, 5]]
    np.testing.assert_allclose(projected, ed, atol=1.0e-12)
