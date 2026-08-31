import numpy as np
import pytest
from scipy.linalg import expm

try:
    import torch
except ModuleNotFoundError:
    torch = None

pytestmark = pytest.mark.skipif(
    torch is None,
    reason="requires optional torch backend",
)

from pyqed.dvr import DVR
from pyqed.ldr import CGLDR, ElectronicPartition, SeparableHamiltonian
from pyqed.ldr.coarse_grained import (
    CGLDRElectronicData,
    _cycle_step_counts,
    _dense_to_mpo,
    _diagonal_mpo,
)
from pyqed.mps.decompose import contract, decompose
from pyqed.mps.mps import MPS
from pyqed.mps.mps import _mpo_to_dense_operator


class _ToyElectronicPoint:
    def __init__(self, vectors, q):
        self.vectors = np.asarray(vectors)
        self.q = float(q)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.vectors, dtype=dtype, copy=copy)

    def vibronic_couplings(self, state_ids=None, modes=None):
        first_diabatic = np.array(
            [[0.2 * self.q, 0.2], [0.2, -0.2 * self.q]]
        )
        second_diabatic = np.array([[0.2, 0.0], [0.0, -0.2]])
        first = self.vectors.conj().T @ first_diabatic @ self.vectors
        second = self.vectors.conj().T @ second_diabatic @ self.vectors
        return first[..., None], second[..., None, None]


class _ToyAbInitioDriver:
    def __init__(self):
        self.calls = 0
        self.scanner_calls = 0

    def as_scanner(self, nstates=None):
        self.scanner_calls += 1
        return self

    def point(self, coordinates):
        self.calls += 1
        x, q = np.asarray(coordinates, dtype=float)
        hamiltonian = np.array(
            [[x + 0.1 * q**2, 0.2 * q], [0.2 * q, -x - 0.1 * q**2]]
        )
        energies, vectors = np.linalg.eigh(hamiltonian)
        return energies, _ToyElectronicPoint(vectors, q)

    @staticmethod
    def overlap(left, right):
        return np.asarray(left).conj().T @ np.asarray(right)


class _NoAnalyticDerivativeDriver(_ToyAbInitioDriver):
    def point(self, coordinates):
        self.calls += 1
        x, q = np.asarray(coordinates, dtype=float)
        hamiltonian = np.array([[x, 0.2 * q], [0.2 * q, -x]])
        return np.linalg.eigh(hamiltonian)


class _ThreeStateDriver:
    def __init__(self):
        self.requested_nstates = None

    def as_scanner(self, nstates=None):
        self.requested_nstates = nstates
        return self

    @staticmethod
    def point(coordinates):
        x = float(np.asarray(coordinates)[0])
        return np.array([x, x + 1.0, x + 2.0]), np.eye(3)

    @staticmethod
    def overlap(left, right):
        return np.asarray(left).conj().T @ np.asarray(right)


def _cgldr(
    domains,
    npts,
    *,
    nsampled,
    nexpanded,
    mass,
    center=None,
    to_geometry=None,
    **kwargs,
):
    sampled = tuple(f"R{axis}" for axis in range(nsampled))
    expanded = tuple(f"q{axis}" for axis in range(nexpanded))
    center = (
        tuple(center)
        if center is not None
        else (0.0,) * nexpanded
    )
    return CGLDR(
        DVR(
            domains=domains,
            npts=npts,
            mass=mass,
            names=sampled + expanded,
        ),
        ElectronicPartition(
            sampled=sampled,
            expanded=expanded,
            center=center,
        ),
        to_geometry=to_geometry,
        **kwargs,
    )


def _two_state_hamiltonian(coordinates):
    x = torch.as_tensor(coordinates[0], dtype=torch.float64)
    q = (
        torch.as_tensor(coordinates[1], dtype=torch.float64)
        if len(coordinates) > 1
        else torch.zeros((), dtype=torch.float64)
    )
    return torch.stack((
        torch.stack((x + 0.1 * q**2, 0.2 * q)),
        torch.stack((0.2 * q, -x - 0.1 * q**2)),
    ))


def _ab_initio_fixture(solver):
    grid_shape = tuple(solver.nx[:solver.dr])
    energies = np.empty((*grid_shape, solver.nstates))
    for index in np.ndindex(*grid_shape):
        coordinate = solver.x[0][index[0]]
        energies[index] = [coordinate, -coordinate]

    ngrid = int(np.prod(grid_shape))
    overlaps = np.zeros(
        (*grid_shape, solver.nstates, *grid_shape, solver.nstates),
        dtype=complex,
    )
    overlap_blocks = overlaps.reshape(
        ngrid, solver.nstates, ngrid, solver.nstates
    )
    for bra in range(ngrid):
        for ket in range(ngrid):
            overlap_blocks[bra, :, ket, :] = np.eye(solver.nstates)

    gradients = np.zeros(
        (*grid_shape, solver.dnr, solver.nstates, solver.nstates)
    )
    hessians = np.zeros(
        (
            *grid_shape,
            solver.dnr,
            solver.dnr,
            solver.nstates,
            solver.nstates,
        )
    )
    if solver.dnr:
        gradients[..., 0, 0, 1] = 0.2
        gradients[..., 0, 1, 0] = 0.2
        hessians[..., 0, 0, 0, 0] = 0.2
        hessians[..., 0, 0, 1, 1] = -0.2

    return CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        hamiltonian_gradients=gradients if solver.dnr else None,
        hamiltonian_hessians=hessians if solver.dnr else None,
        reactive_grids=tuple(np.asarray(x) for x in solver.x[:solver.dr]),
        metadata={"method": "test-CASCI", "energy_unit": "hartree"},
    )


def test_constructor_uses_public_api_and_computes_volume_element():
    dvr = DVR(
        domains=((-1.0, 1.0),),
        npts=(3,),
        mass=1.0,
        names=("R",),
    )
    partition = ElectronicPartition(sampled=("R",))
    solver = CGLDR(
        dvr,
        partition,
        state_ids=(0, 1),
    )

    assert solver.dvr is dvr
    assert solver.partition is partition
    assert np.isfinite(solver.dv)
    assert solver.dv > 0.0
    assert solver.tt_options["max_rank"] == 100
    assert solver.nsampled == 1
    assert solver.nexpanded == 0
    assert solver.state_ids == (0, 1)
    assert solver.sampled_names == ("R",)
    assert not hasattr(solver, "expansion_center")
    assert not hasattr(solver, "expansion_steps")


def test_state_ids_select_nonconsecutive_solver_roots():
    driver = _ThreeStateDriver()
    solver = CGLDR(
        DVR(
            domains=((-1.0, 1.0),),
            npts=(3,),
            mass=1.0,
            names=("Qs",),
        ),
        ElectronicPartition(sampled=("Qs",)),
        state_ids=(2, 0),
        solver=driver,
        to_geometry=lambda coordinates: (coordinates["Qs"],),
    )

    solver.prepare_electronic_data()

    assert driver.requested_nstates == 3
    np.testing.assert_allclose(
        solver.electronic_data.energies,
        np.column_stack((solver.x[0] + 2.0, solver.x[0])),
    )
    overlap_blocks = solver.electronic_data.overlaps.reshape(3, 2, 3, 2)
    np.testing.assert_allclose(overlap_blocks[0, :, 1, :], np.eye(2))


@pytest.mark.parametrize(
    "state_ids, error",
    [
        ((), "cannot be empty"),
        ((0, 0), "must be unique"),
        ((0, -1), "non-negative"),
    ],
)
def test_state_ids_are_validated(state_ids, error):
    with pytest.raises(ValueError, match=error):
        CGLDR(
            DVR(
                domains=((-1.0, 1.0),),
                npts=(3,),
                names=("Qs",),
            ),
            ElectronicPartition(sampled=("Qs",)),
            state_ids=state_ids,
        )


def test_electronic_partition_validates_named_dvr_axes():
    dvr = DVR(
        domains=((-1.0, 1.0), (-2.0, 2.0)),
        npts=(3, 4),
        names=("R", "q"),
    )

    with pytest.raises(ValueError, match="does not assign"):
        ElectronicPartition(sampled=("R",)).resolve(dvr)
    with pytest.raises(ValueError, match="absent"):
        ElectronicPartition(sampled=("R", "missing")).resolve(dvr)
    with pytest.raises(ValueError, match="one value"):
        ElectronicPartition(
            sampled=("R",),
            expanded=("q",),
            center=(),
        )


def test_partition_controls_internal_order_but_geometry_uses_dvr_names():
    driver = _ToyAbInitioDriver()
    dvr = DVR(
        domains=((-1.0, 1.0), (-2.0, 2.0)),
        npts=(3, 3),
        names=("q", "R"),
    )
    seen = []

    def to_geometry(coordinates):
        seen.append(dict(coordinates))
        return coordinates["R"], coordinates["q"]

    solver = CGLDR(
        dvr,
        ElectronicPartition(
            sampled=("R",),
            expanded=("q",),
            center=(0.0,),
        ),
        state_ids=(0, 1),
        solver=driver,
        to_geometry=to_geometry,
        expansion_modes=np.zeros((1, 1, 3)),
    )
    solver.prepare_electronic_data()

    assert solver.coordinate_names == ("R", "q")
    assert solver.reorder_indices == [1, 0]
    assert set(seen[0]) == {"R", "q"}
    np.testing.assert_allclose(solver.x[0], dvr.x[dvr.axis("R")])


def test_automatic_preparation_rejects_internal_expansion_coordinates():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=1.0,
        solver=_ToyAbInitioDriver(),
        to_geometry=lambda coordinates: tuple(coordinates.values()),
    )

    with pytest.raises(NotImplementedError, match="internal coordinates"):
        solver.prepare_electronic_data()


def test_automatic_preparation_requires_analytical_solver_derivatives():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=1.0,
        solver=_NoAnalyticDerivativeDriver(),
        to_geometry=lambda coordinates: tuple(coordinates.values()),
        expansion_modes=np.zeros((1, 1, 3)),
    )

    with pytest.raises(NotImplementedError, match="analytical"):
        solver.prepare_electronic_data()


@pytest.mark.parametrize(
    ("steps", "output_every", "expected"),
    [
        (0, 4, []),
        (3, 4, [3]),
        (8, 4, [4, 4]),
        (10, 4, [4, 4, 2]),
    ],
)
def test_cycle_step_counts(steps, output_every, expected):
    counts = _cycle_step_counts(steps, output_every)
    assert counts == expected
    assert sum(counts) == steps


def test_cycle_step_counts_rejects_invalid_values():
    with pytest.raises(ValueError, match="steps must be non-negative"):
        _cycle_step_counts(-1, 4)
    with pytest.raises(ValueError, match="output_every must be positive"):
        _cycle_step_counts(4, 0)


def test_dense_and_diagonal_mpo_adapters_match_dense_action():
    rng = np.random.default_rng(7)
    dense_state = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    state = MPS(decompose(dense_state, rank=16))

    identity = (
        np.eye(6)
        .reshape(2, 3, 2, 3)
        .transpose(0, 2, 1, 3)
        .reshape(4, 9)
    )
    identity_mpo = _dense_to_mpo(identity, [2, 3], max_rank=16)
    identity_result = identity_mpo.apply(state, max_bond=16)
    np.testing.assert_allclose(
        contract(identity_result.factors),
        dense_state,
        atol=1e-12,
    )

    diagonal = np.arange(1.0, 7.0).reshape(2, 3)
    diagonal_mpo = _diagonal_mpo(diagonal, max_rank=16)
    diagonal_result = diagonal_mpo.apply(state, max_bond=16)
    np.testing.assert_allclose(
        contract(diagonal_result.factors),
        diagonal * dense_state,
        atol=1e-12,
    )


def test_mps_compression_handles_highly_rectangular_bond_matrix():
    from pyqed.mps.decompose import compress

    left = np.ones((1, 2, 2))
    right = np.ones((2, 50_000, 1))

    factors = compress([left, right], chi_max=2, renormalize=False)

    assert factors[0].shape == (1, 2, 1)
    assert factors[1].shape == (1, 50_000, 1)


def test_current_mps_backend_builds_nonreactive_propagator():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        tt_options={"max_rank": 16},
    )
    solver.get_hamiltonian_matrices(_two_state_hamiltonian)
    solver.build_propagator(0.01)

    assert len(solver.coarse_hamiltonian_propagator.factors) == 3
    assert len(solver.half_kinetic_propagator.factors) == 3


def test_projected_kinetic_propagator_is_unitary_for_truncated_overlaps():
    solver = _cgldr(
        [(-1.0, 1.0)],
        [3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=0,
        mass=1.0,
        tt_options={"max_rank": 16},
    )
    data = _ab_initio_fixture(solver)
    overlap_blocks = data.overlaps.reshape(3, 2, 3, 2)
    for bra in range(3):
        for ket in range(3):
            overlap_blocks[bra, :, ket, :] = (
                0.7 ** abs(bra - ket)
            ) * np.eye(2)
    solver.set_electronic_data(data)
    solver.build_propagator(0.05)

    kinetic = _mpo_to_dense_operator(solver.kinetic_propagator)

    np.testing.assert_allclose(
        kinetic.conj().T @ kinetic,
        np.eye(6),
        atol=1.0e-11,
    )


def test_coarse_propagator_uses_palindromic_strang_order():
    solver = _cgldr(
        [(-1.0, 1.0)] * 3,
        [2, 2, 2],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=2,
        mass=1.0,
        center=[0.0, 0.0],
        tt_options={"max_rank": 64},
    )
    identity_qs = np.eye(2)
    zero = np.zeros((2, 2), dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]])
    sigma_z = np.diag([1.0, -1.0])

    def reactive_matrix(electronic):
        return torch.as_tensor(
            np.einsum("ab,ij->abij", electronic, identity_qs),
            dtype=torch.complex128,
        )

    solver.H_matrices = [
        reactive_matrix(zero),
        reactive_matrix(sigma_x),
        reactive_matrix(sigma_z),
        reactive_matrix(zero),
        reactive_matrix(zero),
        reactive_matrix(sigma_y),
    ]
    dt = 0.03
    propagator = solver._build_coarse_hamiltonian_propagator(dt)
    actual = _mpo_to_dense_operator(propagator)

    identity_electronic_qs = np.eye(4)
    qx = np.diag(solver.q_diff[0])
    qy = np.diag(solver.q_diff[1])
    hx = np.kron(np.kron(sigma_x, identity_qs), np.kron(qx, np.eye(2)))
    hy = np.kron(np.kron(sigma_z, identity_qs), np.kron(np.eye(2), qy))
    hxy = np.kron(np.kron(sigma_y, identity_qs), np.kron(qx, qy))
    ux = expm(-0.5j * dt * hx)
    uy = expm(-0.5j * dt * hy)
    uxy = expm(-1j * dt * hxy)
    expected = ux @ uy @ uxy @ uy @ ux

    np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    np.testing.assert_allclose(
        actual.conj().T @ actual,
        np.eye(identity_electronic_qs.shape[0] * 4),
        atol=1.0e-10,
    )


def test_run_uses_all_steps_and_save_data_false_writes_nothing(tmp_path):
    solver = _cgldr(
        [(-1.0, 1.0)],
        [3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=0,
        mass=1.0,
        tt_options={"max_rank": 16},
    )
    solver.output_folder = str(tmp_path)
    solver.get_hamiltonian_matrices(_two_state_hamiltonian)

    electronic = np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1)
    nuclear = np.ones(3, dtype=complex)
    nuclear /= np.linalg.norm(nuclear)
    initial_state = MPS([electronic, nuclear.reshape(1, 3, 1)])

    solver.run(
        initial_state,
        time_step=0.01,
        steps=3,
        output_every=2,
        save_data=False,
    )

    assert len(solver.states) == 3
    assert all(np.isfinite(state.norm_squared()) for state in solver.states)
    assert list(tmp_path.iterdir()) == []


def test_ab_initio_data_wires_energies_overlaps_and_derivatives():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        tt_options={"max_rank": 16},
    )
    data = _ab_initio_fixture(solver)

    solver.set_electronic_data(data)
    solver.build_propagator(0.01)

    np.testing.assert_allclose(
        solver.e0.numpy(),
        np.moveaxis(data.energies, -1, 0),
    )
    assert len(solver.H_matrices) == 3
    assert solver.A.shape == (2, 2, 3, 3)
    assert len(solver.coarse_hamiltonian_propagator.factors) == 3


def test_separable_hamiltonian_replaces_single_center_expansion():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        tt_options={"max_rank": 32},
    )
    reference = _ab_initio_fixture(solver)
    operators = np.zeros((3, 3, 2, 2), dtype=complex)
    operators[:, 0, 0, 0] = solver.x[0]
    operators[:, 0, 1, 1] = -solver.x[0]
    operators[:, 1, 0, 1] = 0.2
    operators[:, 1, 1, 0] = 0.2
    operators[:, 2, 0, 0] = 0.1
    operators[:, 2, 1, 1] = -0.1
    factors = np.stack((
        np.ones_like(solver.x[1]),
        solver.x[1],
        solver.x[1] ** 2,
    ))
    data = CGLDRElectronicData(
        energies=reference.energies,
        overlaps=reference.overlaps,
        separable_hamiltonian=SeparableHamiltonian(
            operators=operators,
            factors=(factors,),
        ),
        reactive_grids=reference.reactive_grids,
        expanded_grids=(np.asarray(solver.x[1]),),
    )

    solver.set_electronic_data(data)
    solver.build_propagator(0.02)

    np.testing.assert_allclose(solver.e0, 0.0)
    assert len(solver.H_matrices) == 3
    assert len(solver.coarse_hamiltonian_propagator.factors) == 3
    propagator = _mpo_to_dense_operator(
        solver.coarse_hamiltonian_propagator
    )
    field = data.separable_hamiltonian.evaluate()
    expected = np.zeros((2, 3, 3, 2, 3, 3), dtype=complex)
    for sampled in range(3):
        for expanded in range(3):
            expected[
                :,
                sampled,
                expanded,
                :,
                sampled,
                expanded,
            ] = expm(-0.02j * field[sampled, expanded])
    np.testing.assert_allclose(
        propagator.conj().T @ propagator,
        np.eye(propagator.shape[0]),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        propagator,
        expected.reshape(18, 18),
        atol=1.0e-10,
    )


def test_axial_cubic_hermite_reconstructs_two_secondary_modes():
    q1 = np.linspace(-1.0, 1.0, 5)
    q2 = np.linspace(-0.8, 0.8, 5)
    anchors = (q1[[0, 2, 4]], q2[[0, 2, 4]])
    identity = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    center = np.array([[0.3, 0.0], [0.0, -0.2]])

    def hamiltonian(first, second):
        return (
            center
            + (0.4 * first + 0.1 * first**2) * sigma_z
            + (-0.2 * second + 0.05 * second**2) * sigma_x
            + 0.07 * first * second * identity
        )

    values = (
        np.asarray([hamiltonian(value, 0.0) for value in anchors[0]]),
        np.asarray([hamiltonian(0.0, value) for value in anchors[1]]),
    )
    gradients = (
        np.asarray([
            (0.4 + 0.2 * value) * sigma_z
            for value in anchors[0]
        ]),
        np.asarray([
            (-0.2 + 0.1 * value) * sigma_x
            for value in anchors[1]
        ]),
    )
    mixed = np.zeros((2, 2, 2, 2))
    mixed[0, 1] = mixed[1, 0] = 0.07 * identity

    separable = SeparableHamiltonian.axial_cubic_hermite(
        (q1, q2),
        anchors,
        values,
        gradients,
        center_hamiltonian=center,
        mixed_hessians=mixed,
        extrapolation="error",
    )

    expected = np.empty((q1.size, q2.size, 2, 2))
    for i, first in enumerate(q1):
        for j, second in enumerate(q2):
            expected[i, j] = hamiltonian(first, second)
    np.testing.assert_allclose(separable.evaluate(), expected, atol=1.0e-13)
    assert separable.operators.shape == (14, 2, 2)
    assert len(separable.factors) == 2


def test_electronic_union_lowdin_compression_builds_correct_kinetic_metric():
    sampled_shape = (2, 2)
    physical_dimension = 2
    nraw = 4
    rotations = np.empty((*sampled_shape, physical_dimension, nraw))
    for index in np.ndindex(sampled_shape):
        angle = 0.2 * index[0] - 0.1 * index[1]
        physical = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        rotations[index] = np.concatenate(
            (physical, physical @ np.diag([1.2, 0.7])),
            axis=1,
        )

    ngrid = int(np.prod(sampled_shape))
    flat_rotations = rotations.reshape(ngrid, physical_dimension, nraw)
    raw_overlaps = np.einsum(
        "pda,qdb->paqb",
        flat_rotations.conj(),
        flat_rotations,
    ).reshape(*sampled_shape, nraw, *sampled_shape, nraw)

    q1 = np.linspace(-1.0, 1.0, 3)
    q2 = np.linspace(-0.5, 0.5, 3)
    physical_terms = np.empty((*sampled_shape, 3, 2, 2))
    for index in np.ndindex(sampled_shape):
        physical_terms[index + (0,)] = np.diag(
            [0.2 + index[0], -0.1 - index[1]]
        )
        physical_terms[index + (1,)] = np.array(
            [[0.0, 0.3], [0.3, 0.0]]
        )
        physical_terms[index + (2,)] = np.diag([0.1, -0.1])
    raw_operators = np.einsum(
        "...da,...tde,...eb->...tab",
        rotations.conj(),
        physical_terms,
        rotations,
        optimize=True,
    )
    raw_hamiltonian = SeparableHamiltonian(
        operators=raw_operators,
        factors=(
            np.stack((np.ones_like(q1), q1, np.ones_like(q1))),
            np.stack((np.ones_like(q2), np.ones_like(q2), q2)),
        ),
    )

    data = CGLDRElectronicData.from_electronic_union(
        overlaps=raw_overlaps,
        hamiltonian=raw_hamiltonian,
        rank=2,
        reactive_grids=(np.arange(2), np.arange(2)),
        expanded_grids=(q1, q2),
    )

    assert data.energies.shape == (2, 2, 2)
    assert data.basis_transforms.shape == (2, 2, 4, 2)
    assert data.metadata["raw_union_dimension"] == 4
    assert data.metadata["retained_union_dimension"] == 2
    transformed = data.overlaps.reshape(ngrid, 2, ngrid, 2)
    physical_frames = np.einsum(
        "...da,...ap->...dp",
        rotations,
        data.basis_transforms,
    ).reshape(ngrid, physical_dimension, 2)
    expected_overlaps = np.einsum(
        "pda,qdb->paqb",
        physical_frames.conj(),
        physical_frames,
    )
    np.testing.assert_allclose(transformed, expected_overlaps, atol=1.0e-12)
    for point in range(ngrid):
        np.testing.assert_allclose(
            transformed[point, :, point, :],
            np.eye(2),
            atol=1.0e-12,
        )

    solver = _cgldr(
        [(-1.0, 1.0)] * 4,
        [2, 2, 3, 3],
        state_ids=(0, 1),
        nsampled=2,
        nexpanded=2,
        mass=1.0,
        center=[0.0, 0.0],
        tt_options={"max_rank": 32},
    )
    data = CGLDRElectronicData(
        energies=data.energies,
        overlaps=data.overlaps,
        separable_hamiltonian=data.separable_hamiltonian,
        reactive_grids=tuple(np.asarray(x) for x in solver.x[:2]),
        expanded_grids=tuple(np.asarray(x) for x in solver.x[2:]),
        basis_transforms=data.basis_transforms,
        metric_eigenvalues=data.metric_eigenvalues,
        metadata=data.metadata,
    )
    solver.set_electronic_data(data)
    solver.build_propagator(0.01)
    kinetic = _mpo_to_dense_operator(solver.kinetic_propagator)
    np.testing.assert_allclose(
        kinetic.conj().T @ kinetic,
        np.eye(kinetic.shape[0]),
        atol=1.0e-10,
    )


def test_electronic_data_npz_round_trip(tmp_path):
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
    )
    data = _ab_initio_fixture(solver)
    filename = tmp_path / "cgldr_electronic_data.npz"
    data.to_npz(filename)

    loaded = CGLDRElectronicData.from_npz(filename)
    solver.load_electronic_data(filename)

    np.testing.assert_allclose(loaded.energies, data.energies)
    np.testing.assert_allclose(loaded.overlaps, data.overlaps)
    np.testing.assert_allclose(
        loaded.hamiltonian_gradients,
        data.hamiltonian_gradients,
    )
    assert loaded.metadata == data.metadata
    assert solver.electronic_data.metadata["method"] == "test-CASCI"


def test_separable_hamiltonian_npz_round_trip(tmp_path):
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
    )
    reference = _ab_initio_fixture(solver)
    operators = np.zeros((3, 2, 2, 2), dtype=complex)
    operators[:, 0, 0, 0] = 0.3
    operators[:, 0, 1, 1] = -0.2
    operators[:, 1, 0, 1] = 0.1
    operators[:, 1, 1, 0] = 0.1
    factors = np.stack((
        np.ones_like(solver.x[1]),
        solver.x[1],
    ))
    data = CGLDRElectronicData(
        energies=reference.energies,
        overlaps=reference.overlaps,
        separable_hamiltonian=SeparableHamiltonian(
            operators=operators,
            factors=(factors,),
        ),
        reactive_grids=reference.reactive_grids,
        expanded_grids=(np.asarray(solver.x[1]),),
    )
    filename = tmp_path / "field_data.npz"

    data.to_npz(filename)
    loaded = CGLDRElectronicData.from_npz(filename)
    solver.set_electronic_data(loaded)

    np.testing.assert_allclose(
        loaded.separable_hamiltonian.operators,
        operators,
    )
    np.testing.assert_allclose(
        loaded.separable_hamiltonian.factors[0],
        factors,
    )
    np.testing.assert_allclose(
        loaded.expanded_grids[0],
        solver.x[1],
    )


def test_separable_hamiltonian_rejects_wrong_expanded_grid():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
    )
    reference = _ab_initio_fixture(solver)
    data = CGLDRElectronicData(
        energies=reference.energies,
        overlaps=reference.overlaps,
        separable_hamiltonian=SeparableHamiltonian(
            operators=np.zeros((3, 1, 2, 2)),
            factors=(np.ones((1, 3)),),
        ),
        reactive_grids=reference.reactive_grids,
        expanded_grids=(np.linspace(-2.0, 2.0, 3),),
    )

    with pytest.raises(ValueError, match="expanded grid"):
        solver.set_electronic_data(data)


def test_existing_apes_overlap_archive_names_are_accepted(tmp_path):
    solver = _cgldr(
        [(-1.0, 1.0)],
        [3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=0,
        mass=1.0,
    )
    data = _ab_initio_fixture(solver)
    filename = tmp_path / "legacy_scan.npz"
    np.savez(
        filename,
        apes=data.energies,
        overlap_matrix=data.overlaps,
        grid_0=data.reactive_grids[0],
    )

    solver.load_electronic_data(filename)

    np.testing.assert_allclose(solver.apes, data.energies)


def test_electronic_data_rejects_nonidentity_self_overlap():
    solver = _cgldr(
        [(-1.0, 1.0)],
        [3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=0,
        mass=1.0,
    )
    data = _ab_initio_fixture(solver)
    overlaps = data.overlaps.copy()
    overlaps[0, :, 0, :] = 0.5 * np.eye(2)
    invalid = CGLDRElectronicData(
        energies=data.energies,
        overlaps=overlaps,
        reactive_grids=data.reactive_grids,
    )

    with pytest.raises(ValueError, match="self-overlap"):
        solver.set_electronic_data(invalid)


def test_displaced_adiabatic_scan_recovers_reference_basis_derivatives():
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
    )
    reference = _ab_initio_fixture(solver)
    displacements = np.array([[-0.02], [0.02]])
    grid_shape = tuple(solver.nx[:solver.dr])
    displaced_energies = np.empty((*grid_shape, 2, 2))
    displaced_overlaps = np.empty((*grid_shape, 2, 2, 2), dtype=complex)

    expected_gradient = reference.hamiltonian_gradients[..., 0, :, :]
    expected_hessian = reference.hamiltonian_hessians[..., 0, 0, :, :]
    for index in np.ndindex(*grid_shape):
        h0 = np.diag(reference.energies[index])
        for sample, displacement in enumerate(displacements[:, 0]):
            hamiltonian = (
                h0
                + expected_gradient[index] * displacement
                + 0.5 * expected_hessian[index] * displacement**2
            )
            values, vectors = np.linalg.eigh(hamiltonian)
            displaced_energies[index + (sample,)] = values
            displaced_overlaps[index + (sample,)] = vectors

    fitted = CGLDRElectronicData.from_displaced_adiabatic_data(
        energies=reference.energies,
        overlaps=reference.overlaps,
        displacements=displacements,
        displaced_energies=displaced_energies,
        reference_to_displaced_overlaps=displaced_overlaps,
        reactive_grids=reference.reactive_grids,
    )

    np.testing.assert_allclose(
        fitted.hamiltonian_gradients[..., 0, :, :],
        expected_gradient,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        fitted.hamiltonian_hessians[..., 0, 0, :, :],
        expected_hessian,
        atol=1e-10,
    )
    solver.set_electronic_data(fitted).build_propagator(0.01)
    assert fitted.metadata["displacement_fit_rank"] == 2

    electronic = np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1)
    reactive = np.ones(3, dtype=complex) / np.sqrt(3)
    coarse = np.ones(3, dtype=complex) / np.sqrt(3)
    initial_state = MPS([
        electronic,
        reactive.reshape(1, 3, 1),
        coarse.reshape(1, 3, 1),
    ])
    solver.run(
        initial_state,
        time_step=0.001,
        steps=2,
        output_every=1,
        save_data=False,
    )
    np.testing.assert_allclose(
        solver.states[-1].norm_squared(),
        1.0,
        atol=1e-10,
    )


def test_electronic_solver_generates_required_data_automatically():
    driver = _ToyAbInitioDriver()
    solver = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        solver=driver,
        to_geometry=lambda coordinates: tuple(coordinates.values()),
        expansion_modes=np.zeros((1, 1, 3)),
        tt_options={"max_rank": 16},
    )

    solver.build_propagator(0.01)

    assert driver.calls == 3
    assert driver.scanner_calls == 1
    assert solver.electronic_data.metadata[
        "derivative_source"
    ] == "analytic_vibronic_couplings"
    assert len(solver.H_matrices) == 3


def test_automatic_electronic_cache_avoids_recalculation(tmp_path):
    cache = tmp_path / "electronic_data.npz"
    driver = _ToyAbInitioDriver()
    first = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        solver=driver,
        to_geometry=lambda coordinates: tuple(coordinates.values()),
        expansion_modes=np.zeros((1, 1, 3)),
        electronic_cache=cache,
    )
    first.prepare_electronic_data()
    assert cache.exists()
    assert driver.calls == 3

    second = _cgldr(
        [(-1.0, 1.0), (-1.0, 1.0)],
        [3, 3],
        state_ids=(0, 1),
        nsampled=1,
        nexpanded=1,
        mass=[1.0, 1.0],
        center=[0.0],
        electronic_cache=cache,
    )
    second.build_propagator(0.01)

    np.testing.assert_allclose(second.apes, first.apes)
