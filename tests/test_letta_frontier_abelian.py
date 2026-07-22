import numpy as np
import pytest

from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierAbelianLayout,
    LocalHamiltonian,
    LocalTerm,
    abelian_frontier_tied_letta_from_mps,
)


def _heisenberg_chain(nsites=4):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.diag([1.0, -1.0])
    exchange = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    return LocalHamiltonian(
        (2,) * nsites,
        tuple(LocalTerm((site, site + 1), exchange) for site in range(nsites - 1)),
    )


def _state(seed=5):
    hamiltonian = _heisenberg_chain()
    parents = ((1, 2), (2,), (3,), ())
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=2,
    )
    return AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parents,
        bond_dim=2,
        abelian_layout=layout,
        seed=seed,
    )


def _charge_resolved_mps(seed=23):
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=(1, 2, 3, 2, 1),
    )
    physical_sites = tuple((site,) for site in range(4))
    masks = layout.local_masks(physical_sites)
    rng = np.random.default_rng(seed)
    cores = []
    for mask in masks:
        native = np.where(mask, rng.normal(size=mask.shape), 0.0)
        cores.append(native.transpose(0, 2, 1))
    return layout, tuple(cores)


def _mps_vector(cores):
    dims = tuple(core.shape[1] for core in cores)
    values = []
    for configuration in np.ndindex(*dims):
        environment = np.ones(1)
        for core, physical in zip(cores, configuration):
            environment = environment @ core[:, physical, :]
        values.append(environment[0])
    vector = np.asarray(values)
    return vector / np.linalg.norm(vector)


def test_frontier_abelian_parent_legs_are_neutral_spectators():
    state = _state()
    mask = state.local_masks[0]

    # Site zero owns only its leading physical leg.  Changing either tied
    # parent therefore cannot alter charge compatibility.
    for left in range(mask.shape[0]):
        for right in range(mask.shape[1]):
            for physical in range(mask.shape[2]):
                assert np.all(
                    mask[left, right, physical]
                    == mask[left, right, physical, 0, 0]
                )

    assert state.nparameters == sum(
        allowed for allowed, _total in state.local_support_sizes()
    )
    assert state.nparameters < state.dense_nparameters


def test_frontier_abelian_state_has_only_target_charge_amplitudes():
    state = _state()
    vector = state.state_vector()
    for configuration, amplitude in zip(np.ndindex(*state.dims), vector):
        two_sz = sum(1 if local == 0 else -1 for local in configuration)
        if two_sz != 0:
            np.testing.assert_allclose(amplitude, 0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    "solver",
    ["direct", "whitened", "matrix_free", "block_sparse"],
)
def test_frontier_abelian_local_solvers_preserve_support_and_agree(solver):
    reference = _state(seed=7)
    direct = reference.copy()
    expected = direct.optimize_site(1, solver="direct")
    state = reference.copy()
    update = state.optimize_site(
        1,
        solver=solver,
        eig_tol=1.0e-11,
        maxiter=300,
    )

    assert update.accepted
    assert update.solver_converged
    np.testing.assert_allclose(update.energy, expected.energy, atol=2.0e-9)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_sector_gauge_preserves_state_and_support():
    state = _state(seed=11)
    vector = state.state_vector()
    energy = state.expectation()

    updates = state.canonicalize_frontier_gauge(weighting="uniform")

    assert updates
    assert all(update.applied for update in updates)
    assert all(update.message == "sector-balanced" for update in updates)
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-13)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-13)
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_frontier_abelian_rejects_sector_mixing_virtual_qr():
    state = _state()
    with pytest.raises(NotImplementedError, match="mix Abelian sectors"):
        state.canonicalize_virtual("left")


@pytest.mark.parametrize("direction", ["left", "right"])
def test_frontier_abelian_bond_expansion_updates_masks_without_changing_state(
    direction,
):
    state = _state(seed=19)
    vector = state.state_vector()
    energy = state.expectation()
    parameters = state.nparameters

    record = state.expand_bond(
        2,
        3,
        direction=direction,
        strategy="random",
        scale=1.0e-3,
        seed=3,
    )

    assert state.bond_dims == (1, 2, 3, 2, 1)
    assert state.abelian_layout.bond_dims == state.bond_dims
    assert record.old_dimension == 2
    assert record.new_dimension == 3
    assert state.nparameters > parameters
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-13)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-13)
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert tensor.shape == mask.shape
        np.testing.assert_array_equal(tensor[~mask], 0.0)

    copied = state.copy()
    assert copied.bond_dims == state.bond_dims
    assert copied.abelian_layout == state.abelian_layout
    np.testing.assert_allclose(copied.state_vector(), vector, atol=3.0e-13)


def test_frontier_abelian_layout_rejects_disconnected_explicit_sectors():
    local_qns = (((1,), (-1,)),) * 2
    with pytest.raises(ValueError, match="removes every entry"):
        layout = FrontierAbelianLayout(
            local_qns=local_qns,
            bond_qns=(((0,),), ((4,),), ((0,),)),
            target=(0,),
        )
        layout.local_masks(((0,), (1,)))


def test_charge_resolved_mps_lift_preserves_nonuniform_bonds_and_state():
    layout, cores = _charge_resolved_mps()
    hamiltonian = _heisenberg_chain()
    state = abelian_frontier_tied_letta_from_mps(
        hamiltonian,
        ((1, 2), (2,), (3,), ()),
        cores,
        local_qns=layout.local_qns,
        bond_qns=layout.bond_qns,
        target=layout.target,
        tie_noise=0.0,
    )

    assert state.bond_dims == layout.bond_dims
    np.testing.assert_allclose(
        state.state_vector(),
        _mps_vector(cores),
        atol=3.0e-13,
    )
    for tensor, mask in zip(state.tensors, state.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_charge_resolved_mps_lift_rejects_forbidden_core_entry():
    layout, cores = _charge_resolved_mps()
    cores = list(cores)
    native_mask = layout.local_masks(tuple((site,) for site in range(4)))[1]
    forbidden = np.argwhere(~native_mask)[0]
    core_coord = (forbidden[0], forbidden[2], forbidden[1])
    cores[1] = cores[1].copy()
    cores[1][core_coord] = 0.2

    with pytest.raises(ValueError, match="outside its Abelian charge support"):
        abelian_frontier_tied_letta_from_mps(
            _heisenberg_chain(),
            ((1,), (2,), (3,), ()),
            cores,
            abelian_layout=layout,
        )


def test_frontier_abelian_constructor_defaults_bonds_from_layout():
    layout = FrontierAbelianLayout.spin_half(
        4,
        target_two_sz=0,
        bond_dims=(1, 2, 3, 2, 1),
    )
    state = AbelianFrontierTiedLETTA(
        _heisenberg_chain(),
        (2,) * 4,
        ((1,), (2,), (3,), ()),
        abelian_layout=layout,
        seed=31,
    )
    assert state.bond_dims == layout.bond_dims

    with pytest.raises(ValueError, match="inconsistent with abelian_layout"):
        AbelianFrontierTiedLETTA(
            _heisenberg_chain(),
            (2,) * 4,
            ((1,), (2,), (3,), ()),
            abelian_layout=layout,
            bond_dim=4,
            seed=31,
        )
