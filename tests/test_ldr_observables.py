import numpy as np

from pyqed.dvr import DVR
from pyqed.ldr.observables import (
    nuclear_density_distance,
    nuclear_observables,
)


def test_cgldr_public_import_does_not_require_optional_torch_backend():
    from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition, SeparableHamiltonian

    assert CGLDR.__name__ == "CGLDR"
    assert CGLDRElectronicData.__name__ == "CGLDRElectronicData"
    assert ElectronicPartition.__name__ == "ElectronicPartition"
    assert SeparableHamiltonian.__name__ == "SeparableHamiltonian"


def test_separable_hamiltonian_polynomial_reconstructs_quartic_field():
    from pyqed.ldr import SeparableHamiltonian

    sampled = np.array([-0.4, 0.2])
    q = np.linspace(-0.3, 0.5, 5)
    center = 0.1
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    coefficients = np.zeros((sampled.size, 5, 2, 2), dtype=complex)
    for index, value in enumerate(sampled):
        coefficients[index, 0] = value * sigma_z
        coefficients[index, 1] = 0.2 * sigma_x
        coefficients[index, 2] = (0.1 + value) * sigma_z
        coefficients[index, 4] = -0.03 * sigma_x

    separable = SeparableHamiltonian.polynomial(
        q,
        coefficients,
        center=center,
    )

    expected = np.empty((sampled.size, q.size, 2, 2), dtype=complex)
    for i, value in enumerate(sampled):
        for j, coordinate in enumerate(q):
            dq = coordinate - center
            expected[i, j] = (
                value * sigma_z
                + 0.2 * dq * sigma_x
                + (0.1 + value) * dq**2 * sigma_z
                - 0.03 * dq**4 * sigma_x
            )

    np.testing.assert_allclose(separable.evaluate(), expected, atol=1.0e-14)
    np.testing.assert_allclose(
        separable.factors[0],
        np.vstack([(q - center) ** power for power in range(5)]),
    )


def test_cgldr_coordinate_expectations_from_recorded_mps_states():
    from pyqed.ldr import CGLDR
    from pyqed.mps.mps import MPS

    electronic = np.sqrt(np.array([0.3, 0.7]))
    first = np.sqrt(np.array([0.25, 0.75]))
    second = np.sqrt(np.array([0.2, 0.3, 0.5]))
    state = MPS([
        electronic.reshape(1, 2, 1),
        first.reshape(1, 2, 1),
        second.reshape(1, 3, 1),
    ])

    solver = object.__new__(CGLDR)
    solver.states = [state]
    solver.ndim = 2
    solver.x = (np.array([-1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
    solver.coordinate_names = ("R", "q")
    solver.initial_time = 0.0
    solver.time_step = 2.0
    solver.steps = 4

    result = solver.compute_coordinate_expectations(femtoseconds=False)

    np.testing.assert_allclose(result["means"], [[1.25, 2.3]])
    np.testing.assert_allclose(result["variances"], [[1.6875, 3.01]])
    np.testing.assert_allclose(result["times"], [0.0])
    assert result["names"] == ("R", "q")


def test_cgldr_numpy_backend_builds_precomputed_propagator_without_torch():
    from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition

    dvr = DVR(
        domains=((-1.0, 1.0),),
        npts=(3,),
        mass=1.0,
        names=("R",),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R",)),
        state_ids=(0, 1),
        tt_options={"max_rank": 16},
    )
    energies = np.column_stack((solver.x[0], -solver.x[0]))
    overlaps = np.zeros((3, 2, 3, 2), dtype=complex)
    for bra in range(3):
        for ket in range(3):
            overlaps[bra, :, ket, :] = np.eye(2)

    data = CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        reactive_grids=(solver.x[0],),
    )
    solver.set_electronic_data(data)
    solver.build_propagator(0.01)

    assert solver.backend == "numpy"
    assert isinstance(solver.e0, np.ndarray)
    assert solver.coarse_hamiltonian_propagator is not None
    assert solver.kinetic_propagator is not None


def test_cgldr_numpy_backend_builds_polynomial_secondary_propagator():
    from scipy.linalg import expm

    from pyqed.ldr import (
        CGLDR,
        CGLDRElectronicData,
        ElectronicPartition,
        SeparableHamiltonian,
    )
    from pyqed.mps.mps import _mpo_to_dense_operator

    dvr = DVR(
        domains=((-1.0, 1.0), (-0.5, 0.5)),
        npts=(2, 3),
        mass=(1.0, 1.0),
        names=("R", "q"),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R",), expanded=("q",), center=(0.0,)),
        state_ids=(0, 1),
        tt_options={"max_rank": 64},
    )
    overlaps = np.empty((2, 2, 2, 2), dtype=complex)
    for bra in range(2):
        for ket in range(2):
            overlaps[bra, :, ket, :] = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    coefficients = np.zeros((2, 4, 2, 2), dtype=complex)
    coefficients[:, 0] = solver.x[0][:, None, None] * sigma_z
    coefficients[:, 1] = 0.15 * sigma_x
    coefficients[:, 3] = -0.04 * sigma_z
    separable = SeparableHamiltonian.polynomial(
        solver.x[1],
        coefficients,
    )
    solver.set_electronic_data(
        CGLDRElectronicData(
            energies=np.zeros((2, 2)),
            overlaps=overlaps,
            separable_hamiltonian=separable,
            reactive_grids=(solver.x[0],),
            expanded_grids=(solver.x[1],),
        )
    )

    dt = 0.03
    solver.build_propagator(dt)
    actual = _mpo_to_dense_operator(solver.coarse_hamiltonian_propagator)
    field = separable.evaluate()
    expected = np.zeros((2, 2, 3, 2, 2, 3), dtype=complex)
    for r in range(2):
        for q in range(3):
            expected[:, r, q, :, r, q] = expm(-1j * dt * field[r, q])

    np.testing.assert_allclose(actual, expected.reshape(12, 12), atol=1.0e-12)


def test_cgldr_hamiltonian_mpo_matches_separable_dense_operator():
    from pyqed.ldr import (
        CGLDR,
        CGLDRElectronicData,
        ElectronicPartition,
        SeparableHamiltonian,
    )
    from pyqed.mps.mps import _mpo_to_dense_operator

    dvr = DVR(
        domains=((-1.0, 1.0), (-0.5, 0.5)),
        npts=(2, 3),
        mass=(1.0, 1.5),
        names=("R", "q"),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R",), expanded=("q",), center=(0.0,)),
        state_ids=(0, 1),
        tt_options={"max_rank": 64},
    )
    overlaps = np.empty((2, 2, 2, 2), dtype=complex)
    for bra in range(2):
        for ket in range(2):
            overlaps[bra, :, ket, :] = np.eye(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_z = np.diag([1.0, -1.0])
    operators = np.empty((2, 2, 2, 2), dtype=complex)
    operators[:, 0] = solver.x[0][:, None, None] * sigma_z
    operators[:, 1] = 0.17 * sigma_x
    factors = np.stack((
        np.ones_like(solver.x[1]),
        solver.x[1],
    ))
    separable = SeparableHamiltonian(
        operators=operators,
        factors=(factors,),
    )
    solver.set_electronic_data(CGLDRElectronicData(
        energies=np.zeros((2, 2)),
        overlaps=overlaps,
        separable_hamiltonian=separable,
        reactive_grids=(solver.x[0],),
        expanded_grids=(solver.x[1],),
    ))

    actual = _mpo_to_dense_operator(solver.build_hamiltonian())
    expected = np.kron(
        np.eye(2),
        np.kron(solver.axes[0].t(), np.eye(3))
        + np.kron(np.eye(2), solver.axes[1].t()),
    ).astype(complex)
    field = separable.evaluate()
    for sampled in range(2):
        for expanded in range(3):
            indices = [
                np.ravel_multi_index(
                    (state, sampled, expanded),
                    (2, 2, 3),
                )
                for state in range(2)
            ]
            expected[np.ix_(indices, indices)] += field[sampled, expanded]

    np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1.0e-12)


def test_cgldr_hybrid_tdvp_matches_exact_two_site_evolution():
    from scipy.linalg import expm

    from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition
    from pyqed.mps.decompose import contract
    from pyqed.mps.mps import MPS, _mpo_to_dense_operator

    dvr = DVR(
        domains=((-1.0, 1.0),),
        npts=(3,),
        mass=1.0,
        names=("R",),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R",)),
        state_ids=(0, 1),
        tt_options={"max_rank": 16},
    )
    energies = np.column_stack((0.2 * solver.x[0], -0.1 * solver.x[0]))
    overlaps = np.empty((3, 2, 3, 2), dtype=complex)
    for bra in range(3):
        for ket in range(3):
            overlaps[bra, :, ket, :] = np.eye(2)
    solver.set_electronic_data(CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        reactive_grids=(solver.x[0],),
    ))
    electronic = np.array([0.0, 1.0], dtype=complex).reshape(1, 2, 1)
    nuclear = np.array([0.2, 0.9, -0.3], dtype=complex)
    nuclear /= np.linalg.norm(nuclear)
    initial = MPS([electronic, nuclear.reshape(1, 3, 1)])

    dt = 0.04
    hamiltonian = _mpo_to_dense_operator(solver.build_hamiltonian())
    expected = expm(-2j * dt * hamiltonian) @ contract(initial.factors).reshape(-1)
    solver.run(
        initial,
        time_step=dt,
        steps=2,
        output_every=1,
        save_data=False,
        tdvp_options={"krylov_dim": 8, "krylov_tol": 1.0e-14},
        tdvp_warmup_steps=1,
    )
    actual = contract(solver.states[-1].factors).reshape(-1)

    np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    np.testing.assert_allclose(solver.states[-1].norm(), 1.0, atol=1.0e-12)
    assert max(solver.states[-1].bond_orders()) > 1
    assert solver.integrator_history.tolist() == ["tdvp2", "tdvp"]


def test_cgldr_accepts_overlap_projected_nuclear_kinetic_mpo_without_torch():
    from scipy.linalg import expm

    from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition
    from pyqed.mps.mpo import sop_to_mpo
    from pyqed.mps.mps import _mpo_to_dense_operator

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    nuclear_mpo = sop_to_mpo(
        (2, 2),
        [
            (0.4, (x, z)),
            (-0.2, (z, x)),
        ],
    )
    dvr = DVR(
        domains=((-1.0, 1.0), (-0.5, 0.5)),
        npts=(2, 2),
        mass=(1.0, 1.0),
        names=("R", "q"),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R",), expanded=("q",), center=(0.0,)),
        state_ids=(0, 1),
        tt_options={"max_rank": 64},
        nuclear_kinetic_mpo=nuclear_mpo,
        kinetic_exponential_options={"order": 12, "scale": 1},
    )
    energies = np.zeros((2, 2))
    overlaps = np.empty((2, 2, 2, 2), dtype=complex)
    for bra in range(2):
        for ket in range(2):
            overlaps[bra, :, ket, :] = (0.35 ** abs(bra - ket)) * np.eye(2)
    gradients = np.zeros((2, 1, 2, 2), dtype=complex)
    hessians = np.zeros((2, 1, 1, 2, 2), dtype=complex)
    solver.set_electronic_data(
        CGLDRElectronicData(
            energies=energies,
            overlaps=overlaps,
            hamiltonian_gradients=gradients,
            hamiltonian_hessians=hessians,
            reactive_grids=(solver.x[0],),
            expanded_grids=(solver.x[1],),
        )
    )

    nuclear_dense = _mpo_to_dense_operator(nuclear_mpo)
    expected = np.zeros((8, 8), dtype=complex)
    for bra_state in range(2):
        for bra_r in range(2):
            for bra_q in range(2):
                row = np.ravel_multi_index(
                    (bra_state, bra_r, bra_q),
                    (2, 2, 2),
                )
                nuclear_row = np.ravel_multi_index((bra_r, bra_q), (2, 2))
                for ket_state in range(2):
                    for ket_r in range(2):
                        for ket_q in range(2):
                            col = np.ravel_multi_index(
                                (ket_state, ket_r, ket_q),
                                (2, 2, 2),
                            )
                            nuclear_col = np.ravel_multi_index(
                                (ket_r, ket_q),
                                (2, 2),
                            )
                            expected[row, col] = (
                                overlaps[bra_r, bra_state, ket_r, ket_state]
                                * nuclear_dense[nuclear_row, nuclear_col]
                            )

    projected = _mpo_to_dense_operator(
        solver._build_projected_nuclear_kinetic_mpo()
    )
    np.testing.assert_allclose(projected, expected, atol=1.0e-12)

    dt = 0.07
    solver.build_propagator(dt)
    assert solver.projected_kinetic_dense is not None
    kinetic = _mpo_to_dense_operator(solver.kinetic_propagator)
    np.testing.assert_allclose(
        kinetic,
        expm(-1j * dt * expected),
        atol=1.0e-10,
    )


def test_cgldr_dense_projected_keo_orders_two_sampled_coordinates():
    from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition
    from pyqed.mps.mpo import sop_to_mpo
    from pyqed.mps.mps import _mpo_to_dense_operator

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    nuclear_mpo = sop_to_mpo(
        (2, 2, 2),
        [
            (0.3, (x, z, eye)),
            (0.2, (z, eye, x)),
            (-0.1, (eye, x, z)),
        ],
    )
    dvr = DVR(
        domains=((-1.0, 1.0), (-0.5, 0.5), (-0.25, 0.25)),
        npts=(2, 2, 2),
        mass=(1.0, 1.0, 1.0),
        names=("R", "S", "q"),
    )
    solver = CGLDR(
        dvr,
        ElectronicPartition(sampled=("R", "S"), expanded=("q",), center=(0.0,)),
        state_ids=(0, 1),
        tt_options={"max_rank": 64},
        nuclear_kinetic_mpo=nuclear_mpo,
    )
    energies = np.zeros((2, 2, 2))
    overlaps = np.empty((2, 2, 2, 2, 2, 2), dtype=complex)
    for bra_r in range(2):
        for bra_s in range(2):
            for ket_r in range(2):
                for ket_s in range(2):
                    distance = abs(bra_r - ket_r) + abs(bra_s - ket_s)
                    overlaps[bra_r, bra_s, :, ket_r, ket_s, :] = (
                        0.25 ** distance
                    ) * np.eye(2)
    gradients = np.zeros((2, 2, 1, 2, 2), dtype=complex)
    hessians = np.zeros((2, 2, 1, 1, 2, 2), dtype=complex)
    solver.set_electronic_data(
        CGLDRElectronicData(
            energies=energies,
            overlaps=overlaps,
            hamiltonian_gradients=gradients,
            hamiltonian_hessians=hessians,
            reactive_grids=solver.x[:2],
            expanded_grids=solver.x[2:],
        )
    )

    nuclear_dense = _mpo_to_dense_operator(nuclear_mpo)
    expected = np.zeros((16, 16), dtype=complex)
    for bra_state in range(2):
        for bra_r in range(2):
            for bra_s in range(2):
                for bra_q in range(2):
                    row = np.ravel_multi_index(
                        (bra_state, bra_r, bra_s, bra_q),
                        (2, 2, 2, 2),
                    )
                    nuclear_row = np.ravel_multi_index(
                        (bra_r, bra_s, bra_q),
                        (2, 2, 2),
                    )
                    for ket_state in range(2):
                        for ket_r in range(2):
                            for ket_s in range(2):
                                for ket_q in range(2):
                                    col = np.ravel_multi_index(
                                        (ket_state, ket_r, ket_s, ket_q),
                                        (2, 2, 2, 2),
                                    )
                                    nuclear_col = np.ravel_multi_index(
                                        (ket_r, ket_s, ket_q),
                                        (2, 2, 2),
                                    )
                                    expected[row, col] = (
                                        overlaps[
                                            bra_r,
                                            bra_s,
                                            bra_state,
                                            ket_r,
                                            ket_s,
                                            ket_state,
                                        ]
                                        * nuclear_dense[nuclear_row, nuclear_col]
                                    )

    projected = solver._build_dense_projected_nuclear_kinetic_operator()

    np.testing.assert_allclose(projected, expected, atol=1.0e-12)


def test_nuclear_observables_are_invariant_under_local_electronic_rotations():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(4, 3, 2, 2)) + 1j * rng.normal(
        size=(4, 3, 2, 2)
    )
    angles = np.linspace(-0.4, 0.5, 6).reshape(3, 2)
    rotations = np.empty((3, 2, 2, 2))
    rotations[..., 0, 0] = np.cos(angles)
    rotations[..., 0, 1] = -np.sin(angles)
    rotations[..., 1, 0] = np.sin(angles)
    rotations[..., 1, 1] = np.cos(angles)
    rotated = np.einsum("...ab,t...b->t...a", rotations, states)
    grids = (np.linspace(-1.0, 1.0, 3), np.linspace(-0.2, 0.2, 2))

    reference = nuclear_observables(states, grids, electronic_axis=-1)
    transformed = nuclear_observables(rotated, grids, electronic_axis=-1)

    for name in (
        "nuclear_density",
        "coordinate_means",
        "coordinate_second_moments",
        "coordinate_covariance",
        "coordinate_variances",
        "autocorrelation",
        "survival_probability",
        "norms",
    ):
        np.testing.assert_allclose(transformed[name], reference[name])

    np.testing.assert_allclose(
        reference["coordinate_variances"],
        np.diagonal(
            reference["coordinate_covariance"],
            axis1=1,
            axis2=2,
        ),
    )
    np.testing.assert_allclose(
        reference["survival_probability"],
        np.abs(reference["autocorrelation"]) ** 2,
    )
    np.testing.assert_allclose(reference["autocorrelation"][0], 1.0)


def test_nuclear_density_distance_has_expected_limits():
    left = np.asarray([[[0.5, 0.5]], [[1.0, 0.0]]])
    right = np.asarray([[[0.5, 0.5]], [[0.0, 1.0]]])

    distance = nuclear_density_distance(left, right)

    np.testing.assert_allclose(distance["total_variation"], [0.0, 1.0])
    np.testing.assert_allclose(distance["bhattacharyya"], [1.0, 0.0])
