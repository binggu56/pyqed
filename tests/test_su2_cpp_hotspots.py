import gc
from types import SimpleNamespace

import numpy as np
import pytest

def test_channel_resolved_local_basis_requires_fully_reduced_operator_core():
    from pyqed.mps.nonabelian.environment import (
        _uses_channel_resolved_local_basis,
    )

    two_site = SimpleNamespace(
        metadata={"contracted_channel_blocks_current": True}
    )
    canonical = SimpleNamespace(
        normal_complementary_plan=object(),
        normal_complementary_fully_reduced=False,
        fully_reduced_identity=False,
    )
    reduced = SimpleNamespace(
        normal_complementary_plan=object(),
        normal_complementary_fully_reduced=True,
        fully_reduced_identity=False,
    )
    assert not _uses_channel_resolved_local_basis(
        two_site,
        canonical,
        canonical,
        rank_coupled=True,
    )
    assert _uses_channel_resolved_local_basis(
        two_site,
        reduced,
        reduced,
        rank_coupled=True,
    )


from pyqed.mps import cpp_davidson
from pyqed.mps.nonabelian.coupling import (
    clebsch_gordan,
    left_or_right_fusion,
    ordered_two_m_values,
)
from pyqed.mps.su2 import SU2Irrep, SpinChargeSector


def _require_cpp_kernel(name):
    kernel = getattr(cpp_davidson, name, None)
    if kernel is None:
        pytest.skip(f"optional C++ kernel {name} is unavailable")
    return kernel


def _require_su2_cpp(name):
    try:
        from pyqed.mps.nonabelian import _su2_kernel
    except ImportError:
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    kernel = getattr(_su2_kernel, name, None)
    if kernel is None:
        pytest.skip(f"optional SU(2) C++ kernel {name} is unavailable")
    return kernel


def _packed_pool(arrays):
    arrays = tuple(np.ascontiguousarray(array, dtype=float) for array in arrays)
    offsets = np.cumsum(
        np.asarray([0, *(array.size for array in arrays)], dtype=np.int64)
    )
    shape_offsets = np.cumsum(
        np.asarray([0, *(array.ndim for array in arrays)], dtype=np.int64)
    )
    return (
        np.concatenate([array.reshape(-1) for array in arrays]),
        offsets,
        shape_offsets,
        np.asarray(
            [dim for array in arrays for dim in array.shape],
            dtype=np.int64,
        ),
    )


def test_cpp_real_array_pool_matches_numpy_packing():
    kernel = _require_su2_cpp("pack_real_array_pool")
    arrays = (
        np.arange(6, dtype=float).reshape(2, 3),
        np.arange(8, dtype=float).reshape(2, 2, 2),
        np.zeros((0, 2), dtype=float),
    )
    data, offsets, shape_offsets, shapes = kernel(arrays)

    np.testing.assert_array_equal(data, np.concatenate([array.reshape(-1) for array in arrays]))
    np.testing.assert_array_equal(offsets, np.asarray([0, 6, 14, 14], dtype=np.int64))
    np.testing.assert_array_equal(shape_offsets, np.asarray([0, 2, 5, 7], dtype=np.int64))
    np.testing.assert_array_equal(shapes, np.asarray([2, 3, 2, 2, 2, 0, 2], dtype=np.int64))


def test_cpp_block_table_runs_davidson_without_python_matvec_callbacks():
    table_cls = _require_cpp_kernel("BlockTable")
    matrix = np.asarray(
        [
            [-1.0, 0.25, 0.0],
            [0.25, 0.5, -0.2],
            [0.0, -0.2, 1.5],
        ],
        dtype=complex,
    )
    table = table_cls(
        (matrix,),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        3,
    )
    expected = np.linalg.eigvalsh(matrix)[0]
    for call in range(2):
        result = table.davidson(
            np.diag(matrix),
            np.ones(3, dtype=complex),
            1.0e-12,
            50,
            12,
            False,
        )
        assert result["accepted"]
        assert result["kind"] == "cpp_block_table_davidson"
        assert result["energy"] == pytest.approx(expected, abs=1.0e-11)
        assert bool(result["workspace_reused"]) is (call > 0)
    assert table.stats["davidson_calls"] == 2
    assert table.stats["davidson_workspace_reuses"] == 1


def test_su2_moving_environment_owns_local_table_and_davidson_workspace():
    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(20260724)
    diagonal_left = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    diagonal_left = 0.5 * (diagonal_left + diagonal_left.conj().T)
    diagonal_right = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    diagonal_right = 0.5 * (diagonal_right + diagonal_right.conj().T)
    coupling = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    blocks = (
        diagonal_left,
        diagonal_right,
        coupling,
        coupling.conj().T,
    )
    input_starts = np.asarray([0, 2, 2, 0], dtype=np.int64)
    output_starts = np.asarray([0, 2, 0, 2], dtype=np.int64)
    dense = np.block(
        [
            [diagonal_left, coupling],
            [coupling.conj().T, diagonal_right],
        ]
    )
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )

    assert owner.install_local_operator(
        "lr:0",
        blocks,
        input_starts,
        output_starts,
        4,
    ) is False
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    np.testing.assert_allclose(owner.local_matvec("lr:0", vector), dense @ vector)
    np.testing.assert_allclose(owner.local_diagonal("lr:0", 4), np.diag(dense))

    expected = np.linalg.eigvalsh(dense)[0]
    results = [
        owner.local_davidson(
            "lr:0",
            np.diag(dense),
            np.ones(4, dtype=complex),
            1.0e-12,
            50,
            12,
            False,
        )
        for _ in range(2)
    ]
    for call, result in enumerate(results):
        assert result["accepted"]
        assert result["kind"] == "cpp_su2_moving_environment_davidson"
        assert result["energy"] == pytest.approx(expected, abs=1.0e-11)
        assert bool(result["workspace_reused"]) is (call > 0)

    assert owner.install_local_operator(
        "lr:0",
        blocks,
        input_starts,
        output_starts,
        4,
    ) is True
    stats = owner.stats
    assert stats["local_operator_blocks"] == 4
    assert stats["local_topology_builds"] == 1
    assert stats["local_numeric_refreshes"] == 2
    assert stats["local_davidson_calls"] == 2
    assert stats["local_davidson_workspace_reuses"] == 1
    assert stats["borrowed_local_operator_bytes"] == dense.size * dense.itemsize
    assert stats["davidson_workspace_bytes"] > 0


def test_su2_moving_environment_owns_blockwise_svd_and_global_truncation():
    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    rng = np.random.default_rng(20260726)
    matrices = (
        rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)),
        rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6)),
    )
    result = owner.blockwise_svd(
        matrices,
        (1, 3),
        cutoff=1.0e-12,
        max_bond=4,
        max_bond_mode="reduced",
    )

    for matrix, left, singular, right in zip(
        matrices,
        result["left"],
        result["singular_values"],
        result["right"],
    ):
        np.testing.assert_allclose(
            left @ np.diag(singular) @ right,
            matrix,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            singular,
            np.linalg.svd(matrix, compute_uv=False),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert sum(len(indices) for indices in result["kept_indices"]) == 4
    assert result["kind"] == "cpp_su2_blockwise_svd"
    stats = owner.stats
    assert stats["block_svd_calls"] == 1
    assert stats["block_svd_blocks"] == 2
    assert stats["block_svd_workspace_growths"] > 0
    assert stats["block_svd_workspace_bytes"] > 0

    weighted = owner.blockwise_svd(
        (np.asarray([[0.79]]), np.asarray([[0.80]])),
        (1, 3),
        cutoff=0.0,
        max_bond=1,
        max_bond_mode="reduced",
    )
    assert [indices.tolist() for indices in weighted["kept_indices"]] == [[], [0]]
    state_budget = owner.blockwise_svd(
        (np.asarray([[0.79]]), np.asarray([[0.80]])),
        (1, 3),
        cutoff=0.0,
        max_bond=1,
        max_bond_mode="states",
    )
    assert [indices.tolist() for indices in state_budget["kept_indices"]] == [[0], []]
    per_sector = owner.blockwise_svd(
        (np.diag([3.0, 2.0]), np.diag([4.0, 1.0])),
        (1, 3),
        cutoff=0.0,
        max_bond=1,
        max_bond_mode="per_sector",
    )
    assert [indices.tolist() for indices in per_sector["kept_indices"]] == [[0], [0]]
    retained = owner.blockwise_svd(
        (np.asarray([[1.0]]), np.asarray([[0.0]])),
        (1, 3),
        cutoff=1.0e-10,
        max_bond=2,
        max_bond_mode="reduced",
        retain_sector_topology=True,
    )
    assert [indices.tolist() for indices in retained["kept_indices"]] == [[0], [0]]


def test_su2_two_site_split_uses_cpp_sweep_engine():
    from pyqed.mps.nonabelian.decompose import svd_two_site
    from pyqed.mps.nonabelian.tensor import NonabelianTensor

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    singlet = SpinChargeSector(0, SU2Irrep(0))
    doublet = SpinChargeSector(1, SU2Irrep(1))
    physical = SpinChargeSector(0, SU2Irrep(0))
    merged = NonabelianTensor(
        data={
            (singlet, physical, physical, physical): np.asarray([[[[0.79]]]]),
            (doublet, physical, physical, physical): np.asarray([[[[0.80]]]]),
        },
        qns=[[singlet, doublet], [physical], [physical], [physical]],
        dirs=[-1, 1, 1, 1],
        metadata={
            "contracted_channels": {
                (singlet, physical, physical, physical): (singlet,),
                (doublet, physical, physical, physical): (doublet,),
            },
        },
    )

    left, right, singular_values, truncation, kept = svd_two_site(
        merged,
        max_bond=1,
        max_bond_mode="reduced",
        absorb="right",
        sweep_engine=owner,
    )

    assert kept == 1
    assert doublet in singular_values
    assert singlet not in singular_values
    assert truncation == pytest.approx(
        1.0
        - doublet.irrep.dim * 0.80**2
        / (singlet.irrep.dim * 0.79**2 + doublet.irrep.dim * 0.80**2)
    )
    assert doublet in left.qns[2]
    assert doublet in right.qns[0]
    assert owner.stats["block_svd_calls"] == 1


def test_cpp_active_bond_merge_matches_reduced_tensor_channels():
    from pyqed.mps.nonabelian.contraction import (
        merge_mps_sites,
        merge_mps_sites_from_packed,
    )
    from pyqed.mps.nonabelian.tensor import NonabelianTensor
    from pyqed.mps.nonabelian.update import _expand_two_site_support

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    singlet = SpinChargeSector(0, SU2Irrep(0))
    doublet = SpinChargeSector(1, SU2Irrep(1))
    pair = SpinChargeSector(2, SU2Irrep(0))
    left = NonabelianTensor(
        {
            (singlet, singlet, singlet): np.arange(2.0).reshape(1, 1, 2) + 1.0,
            (singlet, doublet, doublet): np.arange(3.0).reshape(1, 1, 3) + 2.0,
        },
        [
            [singlet],
            [singlet, doublet],
            [singlet, singlet, doublet, doublet, doublet],
        ],
        [-1, 1, 1],
    )
    right = NonabelianTensor(
        {
            (singlet, singlet, singlet): np.arange(2.0).reshape(2, 1, 1) + 3.0,
            (doublet, doublet, pair): np.arange(6.0).reshape(3, 1, 2) + 4.0,
        },
        [
            [singlet, singlet, doublet, doublet, doublet],
            [singlet, doublet],
            [singlet, pair, pair],
        ],
        [-1, 1, 1],
    )
    reference = _expand_two_site_support(
        left,
        right,
        merge_mps_sites(left, right),
    )
    owner = owner_cls(np.eye(2), np.zeros((2,) * 4), 2)
    owner.install_mps((left, right))
    owner.begin_half_sweep("lr", 2)
    assert owner.claim_next_bond() == 0

    actual = merge_mps_sites_from_packed(
        left,
        right,
        owner.merge_active_bond(),
    )

    assert actual.data.keys() == reference.data.keys()
    for key in reference.data:
        np.testing.assert_allclose(
            actual.data[key],
            reference.data[key],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    expected_channels = reference.metadata["contracted_channel_blocks"]
    actual_channels = actual.metadata["contracted_channel_blocks"]
    assert actual_channels.keys() == expected_channels.keys()
    for key in expected_channels:
        np.testing.assert_allclose(
            actual_channels[key],
            expected_channels[key],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert actual.metadata["contracted_channels"] == reference.metadata[
        "contracted_channels"
    ]
    assert owner.stats["site_merge_calls"] == 1
    assert owner.stats["site_merge_blocks"] == 2
    assert owner.stats["site_merge_bytes"] > 0
    owner.abort_half_sweep()

    # Tensor copies retain metadata, so a marker copied from the previous
    # numerical state must never suppress the half-sweep refresh.
    changed_left = left.copy()
    changed_key = next(iter(changed_left.data))
    changed_left.data[changed_key] = 2.0 * changed_left.data[changed_key]
    changed_reference = _expand_two_site_support(
        changed_left,
        right,
        merge_mps_sites(changed_left, right),
    )
    owner.install_mps((changed_left, right))
    owner.begin_half_sweep("lr", 2)
    assert owner.claim_next_bond() == 0
    changed_actual = merge_mps_sites_from_packed(
        changed_left,
        right,
        owner.merge_active_bond(),
    )
    for key in changed_reference.data:
        np.testing.assert_allclose(
            changed_actual.data[key],
            changed_reference.data[key],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert owner.stats["site_merge_calls"] == 2
    owner.abort_half_sweep()


def test_su2_factor_kernel_block_action_matches_scalar_columns():
    from pyqed.mps.nonabelian.renormalized import FamilyCppFactorKernel

    rng = np.random.default_rng(9182)
    left_stack = rng.normal(size=(2, 2, 2, 2, 2, 2))
    right_stack = rng.normal(size=(2, 2, 2, 2, 2, 2))
    input_entry = SimpleNamespace(shape=(2, 2, 2, 2))
    term = SimpleNamespace(
        left_stack=left_stack,
        right_stack=right_stack,
        input_entry=input_entry,
        output_size=16,
        _use_direct_contraction=False,
    )
    kernel = FamilyCppFactorKernel.from_compiled_term(term)
    blocks = rng.normal(size=(2, 2, 2, 2, 4))

    actual = kernel.apply_blocks(blocks)
    expected = np.column_stack(
        [kernel.apply_block(blocks[..., idx]) for idx in range(blocks.shape[-1])]
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_cpp_su2_factorized_family_table_owns_action_and_davidson():
    table_cls = _require_cpp_kernel("SU2FactorizedFamilyTable")
    matrix = np.asarray(
        [
            [-1.0, 0.2, 0.0],
            [0.2, 0.4, -0.15],
            [0.0, -0.15, 1.2],
        ],
        dtype=complex,
    )
    transforms = (("dense", 0, np.eye(3, dtype=complex)),)
    entries = (
        (
            0,
            0,
            0,
            3,
            0,
            3,
            matrix,
            np.ones((1, 1), dtype=complex),
            (1, 3, 1, 1, 1, 1),
            (3, 1, 1, 1),
            (3, 1, 1, 1),
            3,
        ),
    )
    table = table_cls(transforms, entries, 3)
    vector = np.asarray([0.5, -0.25, 0.75], dtype=complex)

    np.testing.assert_allclose(table.matvec(vector), matrix @ vector)
    result = table.davidson(
        np.diag(matrix),
        np.ones(3, dtype=complex),
        1.0e-12,
        50,
        12,
        False,
    )

    assert result["accepted"]
    assert result["kind"] == "cpp_su2_factorized_family_davidson"
    assert result["energy"] == pytest.approx(np.linalg.eigvalsh(matrix)[0], abs=1.0e-11)
    assert table.stats["entries"] == 1
    assert table.stats["stored_factor_elements"] == 10
    assert table.stats["davidson_calls"] == 1


def test_cpp_su2_factorized_family_table_preserves_compact_transforms():
    table_cls = _require_cpp_kernel("SU2FactorizedFamilyTable")
    diagonal_values = np.asarray([2.0, -0.5], dtype=complex)
    kron_left = np.asarray([[1.0], [2.0]], dtype=complex)
    kron_right = np.asarray([[0.5], [-1.0]], dtype=complex)
    transforms = (
        (
            "diagonal",
            0,
            4,
            np.asarray([1, 3], dtype=np.int64),
            diagonal_values,
        ),
        ("kronecker", 2, kron_left, kron_right, 1, 1, 4),
    )

    def identity_entry(component, output_size):
        return (
            component,
            component,
            0,
            output_size,
            0,
            output_size,
            np.eye(output_size, dtype=complex),
            np.ones((1, 1), dtype=complex),
            (1, output_size, 1, 1, 1, 1),
            (output_size, 1, 1, 1),
            (output_size, 1, 1, 1),
            output_size,
        )

    table = table_cls(
        transforms,
        (identity_entry(0, 4), identity_entry(1, 4)),
        3,
    )
    vector = np.asarray([0.3, -0.7, 1.2], dtype=complex)
    expected_scale = np.asarray(
        [
            abs(diagonal_values[0]) ** 2,
            abs(diagonal_values[1]) ** 2,
            np.vdot(
                np.kron(kron_left[:, 0], kron_right[:, 0]),
                np.kron(kron_left[:, 0], kron_right[:, 0]),
            ).real,
        ]
    )

    np.testing.assert_allclose(table.matvec(vector), expected_scale * vector)


def test_cpp_real_array_nonzero_mask_matches_numpy():
    kernel = _require_su2_cpp("real_array_nonzero_mask")
    arrays = (
        np.zeros((2, 3), dtype=float),
        np.asarray([[0.0, -2.0], [0.0, 0.0]], dtype=float),
        np.asarray([1.0e-9], dtype=float),
    )

    np.testing.assert_array_equal(kernel(arrays), np.asarray([0, 1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(
        kernel(arrays, 1.0e-8),
        np.asarray([0, 1, 0], dtype=np.uint8),
    )


def test_cpp_entry_family_masks_match_packed_offsets():
    kernel = _require_su2_cpp("build_entry_family_masks")
    masks = kernel(
        np.asarray([0, 2, 3, 3, 5], dtype=np.int64),
        np.asarray([0, 2, 1, 2, 1], dtype=np.int64),
        np.asarray([1, 2, 4], dtype=np.uint64),
    )

    np.testing.assert_array_equal(
        masks,
        np.asarray([5, 2, 0, 6], dtype=np.uint64),
    )


@pytest.mark.parametrize("left_representation", [True, False])
def test_cpp_rank_coupled_real_pair_batch_matches_scalar_kernels(left_representation):
    batch = _require_su2_cpp("factorize_rank_coupled_real_pairs")
    packed_batch = _require_su2_cpp(
        "factorize_rank_coupled_real_pairs_packed"
    )
    left = _require_su2_cpp("factorize_rank_coupled_left_real")
    right = _require_su2_cpp("factorize_rank_coupled_right_real")
    rng = np.random.default_rng(31337)
    boundary_blocks = (
        rng.normal(size=(2, 2, 3)),
        rng.normal(size=(2, 1, 2)),
    )
    w_blocks = (
        rng.normal(size=(2, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 2)),
    )
    boundary_ids = np.asarray([0, 1, 0], dtype=np.int64)
    w_ids = np.asarray([0, 1, 0], dtype=np.int64)

    actual = batch(
        boundary_blocks,
        w_blocks,
        boundary_ids,
        w_ids,
        left_representation,
    )
    expected = [
        (
            left(boundary_blocks[boundary_idx], w_blocks[w_idx])
            if left_representation
            else right(w_blocks[w_idx], boundary_blocks[boundary_idx])
        )
        for boundary_idx, w_idx in zip(boundary_ids, w_ids)
    ]
    for actual_block, expected_block in zip(actual, expected):
        np.testing.assert_allclose(actual_block, expected_block)

    data, offsets, shape_offsets, shapes = packed_batch(
        boundary_blocks,
        w_blocks,
        boundary_ids,
        w_ids,
        left_representation,
    )
    for index, expected_block in enumerate(expected):
        shape = tuple(
            int(dim)
            for dim in shapes[
                int(shape_offsets[index]) : int(shape_offsets[index + 1])
            ]
        )
        actual_block = data[
            int(offsets[index]) : int(offsets[index + 1])
        ].reshape(shape)
        np.testing.assert_allclose(actual_block, expected_block)


def test_rank_coupled_block_reuses_persistent_packed_boundary():
    from pyqed.mps.nonabelian.environment import RightBlock

    block = RightBlock(
        {(1, 2): (np.ones((1, 2, 2), dtype=float),)},
        rank_coupled=True,
    )
    first = block.ensure_packed(side="right", bond=3)
    second = block.ensure_packed(side="right", bond=3)

    assert first is second
    assert block.packed_table is first


def test_cpp_real_boundary_handoff_accepts_numerically_real_complex_buffers():
    from pyqed.mps.nonabelian.environment import _real64_contiguous_or_none

    source = np.asarray([1.0 + 0.0j, -2.0 + 1.0e-16j])
    actual = _real64_contiguous_or_none(source)

    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, source.real)
    assert _real64_contiguous_or_none(
        np.asarray([1.0 + 1.0e-8j])
    ) is None


def test_cpp_contextual_metric_routes_match_reduced_recoupling_reference():
    from pyqed.mps.nonabelian.environment import (
        _component_basis_norm,
        _left_reduced_recoupling_coeff,
        _right_reduced_recoupling_coeff,
    )

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    left = np.asarray([[1.2, 0.1], [0.1, 0.8]], dtype=float)
    right = np.asarray([[0.9, -0.2], [-0.2, 1.4]], dtype=float)
    offsets = np.asarray([0, 4], dtype=np.int64)
    labels = np.asarray(
        [
            1, 2, 1, 2, 1, 2, 3,
            0,
            0, 1,
            0,
            0, 1,
            0,
            0, 3,
            1, 2, 2,
        ],
        dtype=np.int64,
    )
    owner.install_metric_boundary(
        "left",
        0,
        left.reshape(-1),
        offsets,
        labels,
        11,
        12,
    )
    owner.install_metric_boundary(
        "right",
        1,
        right.reshape(-1),
        offsets,
        labels,
        21,
        22,
    )
    result = owner.install_contextual_metric_routes(
        "reduced-metric",
        0,
        1,
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 1, 1, 2]], dtype=np.int64),
        np.asarray(
            [[0, 0, 1, 1, 1, 1, 1, 1, 2, 0]],
            dtype=np.int64,
        ),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        4,
        31,
    )
    assert result["metric_route_count"] == 1

    q_left = SpinChargeSector(0, SU2Irrep(0))
    q_physical = SpinChargeSector(1, SU2Irrep(1))
    q_middle = SpinChargeSector(1, SU2Irrep(1))
    q_right = SpinChargeSector(2, SU2Irrep(0))
    scalar = _left_reduced_recoupling_coeff(
        q_left,
        q_left,
        q_physical,
        q_physical,
        q_middle,
        q_middle,
        SU2Irrep(0),
        SU2Irrep(0),
        SU2Irrep(0),
        0,
        0,
        0,
        False,
    )
    scalar *= _right_reduced_recoupling_coeff(
        q_middle,
        q_middle,
        q_physical,
        q_physical,
        q_right,
        q_right,
        SU2Irrep(0),
        SU2Irrep(0),
        SU2Irrep(0),
        0,
        0,
        0,
        False,
    )
    scalar *= _component_basis_norm(
        q_middle,
        q_middle,
        SU2Irrep(0),
        0,
    )
    expected = scalar * np.kron(left, right.T)
    vector = np.asarray([0.2, -0.4, 0.7, 1.1])
    np.testing.assert_allclose(
        owner.factorized_metric_real_matvec("reduced-metric", vector),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factorized_metric_real_diagonal("reduced-metric", 4),
        np.diag(expected),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_sparse_rank_coupled_channels_pack_without_dense_channel_slots():
    from pyqed.mps.nonabelian.environment import RankCoupledChannelBlocks
    from pyqed.mps.nonabelian.su2_qchem_plan import (
        pack_rank_coupled_boundary_table_from_block_map,
    )

    blocks = RankCoupledChannelBlocks(
        {
            2: np.ones((1, 2, 2), dtype=float),
            7: np.full((1, 2, 2), 3.0, dtype=float),
        },
        n_channels=64,
    )
    packed = pack_rank_coupled_boundary_table_from_block_map(
        {(1, 2): blocks},
        side="right",
        bond=3,
    )

    assert len(blocks) == 64
    assert tuple(blocks.data) == (2, 7)
    np.testing.assert_array_equal(packed.channel_ids, np.asarray([2, 7]))
    assert packed.block_pool.n_arrays == 2


def test_large_boundary_channel_lookup_is_ephemeral(monkeypatch):
    from pyqed.mps.nonabelian.environment import RankCoupledChannelBlocks
    from pyqed.mps.nonabelian import su2_qchem_plan

    blocks = RankCoupledChannelBlocks(
        {2: np.ones((1, 2, 2), dtype=float)},
        n_channels=64,
    )
    packed = su2_qchem_plan.pack_rank_coupled_boundary_table_from_block_map(
        {(1, 2): blocks},
        side="right",
        bond=3,
    )
    monkeypatch.setattr(
        su2_qchem_plan,
        "_BOUNDARY_CHANNEL_LOOKUP_CACHE_MAX_BYTES",
        1,
    )

    lookup = packed.channel_index_lookup(7)

    assert lookup.shape == (1, 8)
    assert packed._channel_lookup_cache is None
    assert packed.stats["channel_lookup_cache_bytes"] == 0


def test_sparse_rank_coupled_channels_support_inplace_fallback_accumulation():
    from pyqed.mps.nonabelian.environment import RankCoupledChannelBlocks

    blocks = RankCoupledChannelBlocks(
        {3: np.ones((1, 2, 2), dtype=float)},
        n_channels=8,
    )
    blocks[3] += np.full((1, 2, 2), 2.0)

    np.testing.assert_array_equal(blocks[3], np.full((1, 2, 2), 3.0))
    assert tuple(blocks.data) == (3,)


def test_su2_system_builds_same_complementary_families_as_reference():
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    from pyqed.mps.nonabelian import _su2_kernel

    assert not hasattr(_su2_kernel, "NativeSU2DMRGEngine")
    from pyqed.qchem.dmrg.backends.reduced import (
        build_spatial_complementary_operator_families,
    )

    rng = np.random.default_rng(8675309)
    h1 = rng.normal(size=(3, 3))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(3, 3, 3, 3))
    eri[np.abs(eri) < 0.35] = 0.0
    engine = engine_cls(h1, eri, 2, cutoff=1.0e-10)
    expected = build_spatial_complementary_operator_families(h1, eri)
    actual = build_spatial_complementary_operator_families(
        h1,
        eri,
        moving_environment=engine,
    )

    assert engine.system_stats["backend"] == "cpp"
    assert engine.system_stats["stored_family_terms"] == actual.n_terms
    for name in expected.names:
        assert actual[name].rank == expected[name].rank
        assert actual[name].entries == expected[name].entries


def test_su2_system_builds_direct_normal_complementary_routes():
    """The C++ route plan follows the compact SU(2) NC channel layout."""

    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(20260725)
    n_sites = 5
    h1 = rng.normal(size=(n_sites, n_sites))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(n_sites,) * 4)
    owner = engine_cls(h1, eri, 4, cutoff=1.0e-14)

    total_routes = 0
    total_component_actions = 0
    for site in range(n_sites):
        plan = owner.normal_complementary_plan(site)
        assert plan["left_channels"] == 2 + 2 * n_sites + 6 * site * site
        assert plan["right_channels"] == (
            2 + 2 * n_sites + 6 * (site + 1) * (site + 1)
        )
        assert np.all(plan["source"] >= 0)
        assert np.all(plan["source"] < plan["left_channels"])
        assert np.all(plan["target"] >= 0)
        assert np.all(plan["target"] < plan["right_channels"])
        assert np.all(np.isfinite(plan["coefficient"]))
        total_routes += int(plan["source"].size)
        total_component_actions += int(plan["component_transition"].size)

        counts = plan["family_transition_counts"]
        assert set(counts) == {"S", "R", "A", "P", "B", "Q"}
        assert sum(counts.values()) <= int(plan["source"].size)

        left_qn = plan["left_channel_quantum_numbers"]
        right_qn = plan["right_channel_quantum_numbers"]
        assert left_qn.shape == (plan["left_channels"], 3)
        assert right_qn.shape == (plan["right_channels"], 3)
        action_transition = plan["component_transition"]
        assert np.all(action_transition >= 0)
        assert np.all(action_transition < plan["source"].size)
        assert np.all(np.isfinite(plan["component_coefficient"]))

        operator_two_j = np.asarray(
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
            dtype=np.int64,
        )
        operator_charge = np.asarray(
            [0, 0, 1, -1, 1, -1, 1, -1, 2, 2, -2, -2, 0, 0, 2, 2, -2, -2, 0, 0],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            right_qn[plan["target"], 0],
            left_qn[plan["source"], 0] + operator_charge[plan["operator"]],
        )
        for action in range(action_transition.size):
            transition = int(action_transition[action])
            source = int(plan["source"][transition])
            target = int(plan["target"][transition])
            operator_rank = int(
                operator_two_j[int(plan["operator"][transition])]
            )
            local_two_m = int(plan["component_local_two_m"][action])
            assert abs(local_two_m) <= operator_rank
            assert (operator_rank - local_two_m) % 2 == 0
            assert 0 <= int(plan["component_source"][action]) <= int(
                left_qn[source, 1]
            )
            assert 0 <= int(plan["component_target"][action]) <= int(
                right_qn[target, 1]
            )
            assert plan["component_coefficient"][action] != 0.0

    stats = owner.system_stats
    assert stats["normal_complementary_transition_count"] == total_routes
    assert (
        stats["normal_complementary_component_action_count"]
        == total_component_actions
    )
    assert stats["normal_complementary_memory_bytes"] > 0

    # At site one, the direct P and Q actions are the same screened
    # coefficients that block2's NC construction places between the normal
    # C/D channels and the complementary R/RD channels.
    site = 1
    plan = owner.normal_complementary_plan(site)
    source_d0 = 2 + site
    target_rd2 = 2 + (site + 1) + 2
    selector = (
        (plan["source"] == source_d0)
        & (plan["target"] == target_rd2)
        & (plan["operator"] == 8)  # PairCreate0
    )
    expected = -eri[site, 0, site, 2]
    assert np.any(selector)
    np.testing.assert_allclose(plan["coefficient"][selector], expected)

    # R/RD contains a one-electron endpoint and a fixed local triple
    # primitive.  Integral dependence belongs to the C++ transition arena,
    # not to a site-dependent Python operator table.
    target_rd2 = 2 + (site + 1) + 2
    target_r2 = 2 + n_sites + 2
    rd_create = (
        (plan["source"] == 1)
        & (plan["target"] == target_rd2)
        & (plan["operator"] == 2)
    )
    rd_triple = (
        (plan["source"] == 1)
        & (plan["target"] == target_rd2)
        & (plan["operator"] == 6)
    )
    r_destroy = (
        (plan["source"] == 1)
        & (plan["target"] == target_r2)
        & (plan["operator"] == 3)
    )
    r_triple = (
        (plan["source"] == 1)
        & (plan["target"] == target_r2)
        & (plan["operator"] == 7)
    )
    for selected in (rd_create, rd_triple, r_destroy, r_triple):
        assert np.count_nonzero(selected) == 1
    np.testing.assert_allclose(
        plan["coefficient"][rd_create],
        h1[site, 2] / np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        plan["coefficient"][rd_triple],
        2.0 * eri[site, 2, site, site],
    )
    np.testing.assert_allclose(
        plan["coefficient"][r_destroy],
        h1[2, site] / np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        plan["coefficient"][r_triple],
        2.0 * eri[2, site, site, site],
    )


def test_cpp_normal_complementary_core_action_matches_sparse_reference():
    """The NC transition arena executes without a Python MPO channel graph."""

    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(20260726)
    n_sites = 4
    h1 = rng.normal(size=(n_sites, n_sites))
    eri = rng.normal(size=(n_sites,) * 4)
    owner = engine_cls(h1, eri, 4, cutoff=1.0e-14)
    site = 2
    plan = owner.normal_complementary_plan(site)
    primitives = rng.normal(size=(20, 5))
    inputs = rng.normal(size=(int(plan["left_component_offsets"][-1]), 7))

    actual = owner.apply_normal_complementary_core(site, primitives, inputs)
    expected = np.zeros(
        (int(plan["right_component_offsets"][-1]), inputs.shape[1]),
        dtype=float,
    )
    for action in range(plan["component_transition"].size):
        transition = int(plan["component_transition"][action])
        source_channel = int(plan["source"][transition])
        target_channel = int(plan["target"][transition])
        source = (
            int(plan["left_component_offsets"][source_channel])
            + int(plan["component_source"][action])
        )
        target = (
            int(plan["right_component_offsets"][target_channel])
            + int(plan["component_target"][action])
        )
        primitive = primitives[
            int(plan["operator"][transition]),
            int(plan["component_local_two_m"][action]) + 2,
        ]
        expected[target] += (
            plan["component_coefficient"][action] * primitive * inputs[source]
        )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    vector = owner.apply_normal_complementary_core(site, primitives, inputs[:, 0])
    assert vector.ndim == 1
    np.testing.assert_allclose(vector, expected[:, 0], rtol=1.0e-13, atol=1.0e-13)


def test_cpp_normal_complementary_primitives_match_local_su2_algebra():
    """The system-owned primitive table is the projected local Fock algebra."""

    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    from pyqed import SpinHalfFermionOperators
    from pyqed.mps.nonabelian.environment import _physical_component_matrix

    h1 = np.diag([0.3, -0.2])
    eri = np.zeros((2,) * 4)
    eri[0, 0, 0, 0] = 0.7
    eri[1, 1, 1, 1] = -0.4
    ecore = 1.1
    engine = engine_cls(h1, eri, 2, ecore=ecore)
    fock = SpinHalfFermionOperators()
    create_standard = {
        1: np.asarray(fock["Cdu"], dtype=float),
        -1: np.asarray(fock["Cdd"], dtype=float),
    }
    create_dual = {
        -1: np.asarray(fock["Cdu"], dtype=float),
        1: -np.asarray(fock["Cdd"], dtype=float),
    }
    destroy = {
        -1: np.asarray(fock["Cu"], dtype=float),
        1: -np.asarray(fock["Cd"], dtype=float),
    }
    sectors = (
        SpinChargeSector(0, SU2Irrep(0)),
        SpinChargeSector(1, SU2Irrep(1)),
        SpinChargeSector(2, SU2Irrep(0)),
    )
    state_indices = ((0,), (1, 2), (3,))

    def couple(left, left_rank, right, right_rank, target_rank):
        result = {}
        for two_m in ordered_two_m_values(SU2Irrep(target_rank)):
            block = np.zeros((4, 4), dtype=float)
            for left_two_m, left_block in left.items():
                for right_two_m, right_block in right.items():
                    if left_two_m + right_two_m != two_m:
                        continue
                    block += clebsch_gordan(
                        SU2Irrep(left_rank),
                        SU2Irrep(right_rank),
                        SU2Irrep(target_rank),
                        left_two_m,
                        right_two_m,
                        two_m,
                    ) * (left_block @ right_block)
            result[two_m] = block
        return result

    pair_create = {
        rank: couple(create_dual, 1, create_dual, 1, rank)
        for rank in (0, 2)
    }
    pair_destroy = {
        rank: couple(destroy, 1, destroy, 1, rank)
        for rank in (0, 2)
    }
    hole = {
        rank: couple(create_standard, 1, destroy, 1, rank)
        for rank in (0, 2)
    }
    reduced_destroy = couple(hole[0], 0, destroy, 1, 1)
    reduced_create_standard = couple(
        create_standard,
        1,
        hole[0],
        0,
        1,
    )
    reduced_create = {
        -1: reduced_create_standard[1],
        1: -reduced_create_standard[-1],
    }

    def insert(table, operator_id, components, rank):
        rank_irrep = SU2Irrep(rank)
        for output_charge, output_sector in enumerate(sectors):
            rows = state_indices[output_charge]
            for input_charge, input_sector in enumerate(sectors):
                cols = state_indices[input_charge]
                for two_m, component in components.items():
                    block = component[np.ix_(rows, cols)]
                    basis = _physical_component_matrix(
                        output_sector,
                        input_sector,
                        rank_irrep,
                        two_m,
                        rank == 0,
                    )
                    norm = np.vdot(basis, basis)
                    if abs(norm) <= 1.0e-14:
                        assert np.linalg.norm(block) <= 1.0e-12
                        continue
                    factor = np.vdot(basis, block) / norm
                    np.testing.assert_allclose(
                        block,
                        factor * basis,
                        rtol=1.0e-13,
                        atol=1.0e-13,
                    )
                    table[
                        output_charge,
                        input_charge,
                        operator_id,
                        two_m + 2,
                    ] = np.real_if_close(factor)

    for site in range(2):
        expected = np.zeros((3, 3, 20, 5), dtype=float)
        insert(expected, 0, {0: np.eye(4)}, 0)
        core = ecore if site == 0 else 0.0
        local_h = np.diag(
            [
                core,
                core + h1[site, site],
                core + h1[site, site],
                core + 2.0 * h1[site, site] + eri[site, site, site, site],
            ]
        )
        insert(expected, 1, {0: local_h}, 0)
        insert(expected, 2, create_dual, 1)
        insert(expected, 3, destroy, 1)
        insert(expected, 4, {m: -op for m, op in create_dual.items()}, 1)
        insert(expected, 5, {m: -op for m, op in destroy.items()}, 1)
        insert(expected, 6, reduced_create, 1)
        insert(expected, 7, reduced_destroy, 1)
        insert(expected, 8, pair_create[0], 0)
        insert(expected, 9, pair_create[2], 2)
        insert(expected, 10, pair_destroy[0], 0)
        insert(expected, 11, pair_destroy[2], 2)
        insert(expected, 12, hole[0], 0)
        insert(expected, 13, hole[2], 2)
        expected[:, :, 14:20] = expected[:, :, 8:14]
        actual = engine.normal_complementary_primitives(site)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)

    assert (
        engine.system_stats["normal_complementary_primitive_bytes"]
        == 2 * 3 * 3 * 20 * 5 * np.dtype(float).itemsize
    )


def test_cpp_reduced_hamiltonian_build_bypasses_python_eri_builder(monkeypatch):
    _require_su2_cpp("SU2MovingEnvironment")
    from pyqed.qchem.dmrg.backends import reduced

    def fail_reference_builder(*_args, **_kwargs):
        raise AssertionError("production C++ build entered the Python ERI builder")

    monkeypatch.setattr(
        reduced.SpatialSpinFreeERIBuilder,
        "build",
        fail_reference_builder,
    )
    h1 = np.asarray([[0.2, 0.03], [0.03, -0.1]])
    eri = np.zeros((2, 2, 2, 2, 2, 2))
    eri[:, :, 0, 0, 0, 0] = 0.7
    eri[:, :, 1, 1, 1, 1] = 0.5
    hamiltonian = reduced.build_spatial_reduced_hamiltonian_mpo(
        [h1, h1],
        eri,
        n_elec=2,
    )

    assert hamiltonian.moving_environment is not None
    assert hamiltonian.info["two_body_builder"] == "SU2System[NC]"
    assert hamiltonian.info["reference_carrier"] is False
    assert hamiltonian.info["reference_carrier_source"] is None
    assert hamiltonian.info["normal_complementary_production"] is True
    assert hamiltonian.info["includes_core_energy"] is True
    assert hamiltonian.info["normal_complementary_routes"] == {
        "owner": "cpp_system",
        "layout": "su2_normal_complementary",
        "transition_count": (
            hamiltonian.moving_environment.system_stats[
                "normal_complementary_transition_count"
            ]
        ),
        "memory_bytes": (
            hamiltonian.moving_environment.system_stats[
                "normal_complementary_memory_bytes"
            ]
        ),
        "reference_carrier_required": False,
    }


def test_cpp_complementary_boundaries_share_family_storage():
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    from pyqed.mps.nonabelian.renormalized import (
        ComplementaryRenormalizedOperatorStack,
    )
    from pyqed.qchem.dmrg.backends.reduced import (
        build_spatial_complementary_operator_families,
    )

    h1 = np.eye(4)
    eri = np.arange(4**4, dtype=float).reshape((4,) * 4) / 1000.0
    engine = engine_cls(h1, eri, 4)
    families = build_spatial_complementary_operator_families(
        h1,
        eri,
        moving_environment=engine,
    )
    stack = ComplementaryRenormalizedOperatorStack(families=families)
    first = stack.put("left", 1)
    second = stack.put("left", 2, parent_key=first.key)

    source = families["P"].entries
    assert first.family_payloads["P"].entries is source
    assert second.family_payloads["P"].entries is source
    assert (
        first.family_payloads["P"].internal_terms
        + first.family_payloads["P"].cross_terms
        + first.family_payloads["P"].external_terms
        == len(source)
    )
    assert source.partition_counts("left", 1) == (
        first.family_payloads["P"].internal_terms,
        first.family_payloads["P"].cross_terms,
        first.family_payloads["P"].external_terms,
    )


def test_cpp_boundary_can_skip_redundant_python_family_operator_table():
    from pyqed.mps.nonabelian.renormalized import (
        ComplementaryRenormalizedOperatorEntry,
        RenormalizedBlockEntry,
    )

    boundary = RenormalizedBlockEntry(
        side="left",
        bond=1,
        block={},
    )
    complementary = ComplementaryRenormalizedOperatorEntry(
        side="left",
        bond=1,
        family_names=("S", "R", "A", "P", "B", "Q"),
        materialize_family_operator_table=False,
    )
    object.__setattr__(
        boundary,
        "complementary_operator_entry",
        complementary,
    )
    symbolic = object()

    assert boundary.put_symbolic_operator_table(symbolic) is symbolic
    assert boundary.symbolic_operator_table is symbolic
    assert complementary.family_operator_table is None


def test_cpp_su2_engine_reuses_boundary_topology_and_owns_sweep_stats():
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    h1 = np.eye(2)
    eri = np.zeros((2, 2, 2, 2))
    engine = engine_cls(h1, eri, 2)
    offsets = np.asarray([0, 4], dtype=np.int64)
    labels = np.asarray([0, 3, 1, 2, 2], dtype=np.int64)

    assert engine.install_boundary(
        "left",
        0,
        np.arange(4.0),
        offsets,
        labels,
        11,
        1,
    ) is False
    assert engine.install_boundary(
        "left",
        0,
        np.arange(4.0) + 1.0,
        offsets,
        labels,
        11,
        2,
    ) is True
    engine.begin_half_sweep("lr", 2)
    engine.record_bond(
        0,
        matvec_calls=5,
        davidson_iterations=3,
        kept_states=4,
        matvec_seconds=0.2,
        davidson_seconds=0.3,
        truncation_seconds=0.1,
    )
    engine.finish_half_sweep()

    stats = engine.stats
    assert stats["boundary_topology_builds"] == 1
    assert stats["boundary_numeric_refreshes"] == 2
    assert stats["boundary_reallocations"] == 1
    assert stats["borrowed_boundary_bytes"] == 4 * np.dtype(float).itemsize
    assert stats["half_sweeps"] == 1
    assert stats["bond_steps"] == 1
    assert stats["matvec_calls"] == 5
    assert stats["davidson_iterations"] == 3
    assert stats["memory_bytes"] >= engine.system_stats["memory_bytes"]
    assert engine.release_boundary("left", 0) is True
    assert engine.stats["boundary_count"] == 0
    assert engine.stats["borrowed_boundary_bytes"] == 0
    assert engine.release_boundary("left", 0) is False


@pytest.mark.parametrize("side", ("left", "right"))
def test_cpp_su2_engine_advances_owned_reduced_boundary(side):
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(20260725)
    engine = engine_cls(np.eye(2), np.zeros((2,) * 4), 2)

    def packed_labels(shape_offsets, shapes):
        topology = (
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray(shape_offsets, dtype=np.int64),
            np.asarray(shapes, dtype=np.int64),
        )
        return np.concatenate(
            (
                np.asarray([array.size for array in topology], dtype=np.int64),
                *topology,
            )
        )

    def pool(array):
        array = np.ascontiguousarray(array, dtype=float)
        return (
            array.reshape(-1),
            np.asarray([0, array.size], dtype=np.int64),
            np.asarray([0, array.ndim], dtype=np.int64),
            np.asarray(array.shape, dtype=np.int64),
        )

    if side == "left":
        parent = rng.normal(size=(2, 3, 4))
        bra = rng.normal(size=(3, 2, 5))
        ket = rng.normal(size=(4, 2, 6))
        mpo = rng.normal(size=(2, 3, 2, 2))
        output_shape = (3, 5, 6)
        expected = np.einsum(
            "xij,ipr,xypq,jqs->yrs",
            parent,
            bra,
            mpo,
            ket,
            optimize=False,
        )
        parent_bond, child_bond = 0, 1
    else:
        parent = rng.normal(size=(3, 5, 6))
        bra = rng.normal(size=(7, 2, 5))
        ket = rng.normal(size=(8, 2, 6))
        mpo = rng.normal(size=(4, 3, 2, 2))
        output_shape = (4, 7, 8)
        expected = np.einsum(
            "ipr,xypq,yrs,jqs->xij",
            bra,
            mpo,
            parent,
            ket,
            optimize=False,
        )
        parent_bond, child_bond = 2, 1

    parent_data, parent_offsets, parent_shape_offsets, parent_shapes = pool(parent)
    parent_labels = packed_labels(parent_shape_offsets, parent_shapes)
    engine.install_boundary(
        side,
        parent_bond,
        parent_data,
        parent_offsets,
        parent_labels,
        11,
        1,
    )
    bra_data, bra_offsets, bra_shape_offsets, bra_shapes = pool(bra)
    ket_data, ket_offsets, ket_shape_offsets, ket_shapes = pool(ket)
    mpo_data, mpo_offsets, mpo_shape_offsets, mpo_shapes = pool(mpo)
    output_offsets = np.asarray(
        [0, np.prod(output_shape)],
        dtype=np.int64,
    )
    output_shape_offsets = np.asarray([0, 3], dtype=np.int64)
    output_shapes = np.asarray(output_shape, dtype=np.int64)
    output_labels = packed_labels(output_shape_offsets, output_shapes)

    actual, same_topology = engine.advance_boundary(
        side,
        parent_bond,
        child_bond,
        np.asarray([[0, 0, 0, 0, 0]], dtype=np.int64),
        bra_data,
        bra_offsets,
        bra_shape_offsets,
        bra_shapes,
        ket_data,
        ket_offsets,
        ket_shape_offsets,
        ket_shapes,
        mpo_data,
        mpo_offsets,
        mpo_shape_offsets,
        mpo_shapes,
        output_offsets,
        output_shape_offsets,
        output_shapes,
        output_labels,
        22,
        2,
    )

    assert same_topology is False
    np.testing.assert_allclose(
        actual.reshape(output_shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    stats = engine.stats
    assert stats["boundary_update_calls"] == 1
    assert stats["boundary_update_routes"] == 1
    assert stats["owned_boundary_bytes"] == actual.nbytes
    assert stats["borrowed_boundary_bytes"] == parent.nbytes


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize(
    "labels",
    (
        (1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1),
        (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0),
        (1, 1, 2, 0, 1, 1, 1, 1, 2, 0, 2, 0, 0, 0),
        (1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0),
    ),
)
def test_cpp_reduced_environment_recoupling_matches_python_reference(
    side,
    labels,
):
    """NARG's shared CG core reproduces the full reduced boundary factor."""

    kernel = _require_su2_cpp("reduced_environment_recoupling")
    (
        physical_output_charge,
        physical_input_charge,
        boundary_bra_two_j,
        boundary_ket_two_j,
        physical_bra_two_j,
        physical_ket_two_j,
        next_bra_two_j,
        next_ket_two_j,
        left_channel_two_j,
        right_channel_two_j,
        operator_two_j,
        left_channel_two_m,
        right_channel_two_m,
        operator_two_m,
    ) = labels
    boundary_sectors = (
        SpinChargeSector(0, SU2Irrep(boundary_bra_two_j)),
        SpinChargeSector(0, SU2Irrep(boundary_ket_two_j)),
    )
    physical_sectors = (
        SpinChargeSector(
            physical_output_charge,
            SU2Irrep(physical_bra_two_j),
        ),
        SpinChargeSector(
            physical_input_charge,
            SU2Irrep(physical_ket_two_j),
        ),
    )
    next_sectors = (
        SpinChargeSector(0, SU2Irrep(next_bra_two_j)),
        SpinChargeSector(0, SU2Irrep(next_ket_two_j)),
    )
    sectors = (
        (*boundary_sectors, *physical_sectors, *next_sectors)
        if side == "left"
        else (*next_sectors, *physical_sectors, *boundary_sectors)
    )
    from pyqed.mps.nonabelian.environment import (
        _left_reduced_recoupling_coeff,
        _right_reduced_recoupling_coeff,
    )

    reference = (
        _left_reduced_recoupling_coeff
        if side == "left"
        else _right_reduced_recoupling_coeff
    )(
        *sectors,
        SU2Irrep(left_channel_two_j),
        SU2Irrep(right_channel_two_j),
        SU2Irrep(operator_two_j),
        left_channel_two_m,
        right_channel_two_m,
        operator_two_m,
    )
    actual = kernel(side, *labels)
    np.testing.assert_allclose(actual, reference, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("side", ("left", "right"))
def test_cpp_su2_engine_advances_direct_normal_complementary_boundary(side):
    """The boundary action consumes NC transitions without a Python MPO core."""

    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(20260726)
    engine = engine_cls(np.eye(2), np.zeros((2,) * 4), 2)
    plan = engine.normal_complementary_plan(0)
    selected = np.flatnonzero(
        (plan["source"] == 1)
        & (plan["operator"] == 2)
        & (
            plan["right_channel_quantum_numbers"][plan["target"], 1]
            == 1
        )
    )
    assert selected.size
    transition = int(selected[0])

    def packed_labels(shape_offsets, shapes):
        topology = (
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray(shape_offsets, dtype=np.int64),
            np.asarray(shapes, dtype=np.int64),
        )
        return np.concatenate(
            (
                np.asarray([array.size for array in topology], dtype=np.int64),
                *topology,
            )
        )

    def pool(array):
        array = np.ascontiguousarray(array, dtype=float)
        return (
            array.reshape(-1),
            np.asarray([0, array.size], dtype=np.int64),
            np.asarray([0, array.ndim], dtype=np.int64),
            np.asarray(array.shape, dtype=np.int64),
        )

    primitives = engine.normal_complementary_primitives(0)
    source_qn = plan["left_channel_quantum_numbers"][
        int(plan["source"][transition])
    ]
    target_qn = plan["right_channel_quantum_numbers"][
        int(plan["target"][transition])
    ]
    x_dim = int(source_qn[1]) + 1
    y_dim = int(target_qn[1]) + 1
    q_lb = SpinChargeSector(0, SU2Irrep(0))
    q_lk = SpinChargeSector(0, SU2Irrep(0))
    q_pb = SpinChargeSector(1, SU2Irrep(1))
    q_pk = SpinChargeSector(0, SU2Irrep(0))
    q_rb = SpinChargeSector(1, SU2Irrep(1))
    q_rk = SpinChargeSector(0, SU2Irrep(0))
    from pyqed.mps.nonabelian.environment import _physical_component_matrix

    left_irrep = SU2Irrep(int(source_qn[1]))
    right_irrep = SU2Irrep(int(target_qn[1]))
    operator_irrep = SU2Irrep(1)
    local_core = np.zeros((x_dim, y_dim, 2, 1), dtype=float)
    actions = np.flatnonzero(plan["component_transition"] == transition)
    for action in actions:
        source_component = int(plan["component_source"][action])
        target_component = int(plan["component_target"][action])
        local_two_m = int(plan["component_local_two_m"][action])
        physical = _physical_component_matrix(
            q_pb,
            q_pk,
            operator_irrep,
            local_two_m,
            False,
        )
        local_core[
            source_component,
            target_component,
            :,
            :,
        ] += (
            plan["component_coefficient"][action]
            * primitives[
                1,
                0,
                2,
                local_two_m + 2,
            ]
            * physical
        )

    if side == "left":
        parent = rng.normal(size=(x_dim, 3, 4))
        bra = rng.normal(size=(3, 2, 5))
        ket = rng.normal(size=(4, 1, 6))
        output_shape = (y_dim, 5, 6)
        expected = np.einsum(
            "xij,ipr,xypq,jqs->yrs",
            parent,
            bra,
            local_core,
            ket,
            optimize=False,
        )
        parent_bond, child_bond = 0, 1
    else:
        parent = rng.normal(size=(y_dim, 5, 6))
        bra = rng.normal(size=(7, 2, 5))
        ket = rng.normal(size=(8, 1, 6))
        output_shape = (x_dim, 7, 8)
        expected = np.einsum(
            "ipr,xypq,yrs,jqs->xij",
            bra,
            local_core,
            parent,
            ket,
            optimize=False,
        )
        parent_bond, child_bond = 1, 0

    parent_data, parent_offsets, parent_shape_offsets, parent_shapes = pool(parent)
    engine.install_boundary(
        side,
        parent_bond,
        parent_data,
        parent_offsets,
        packed_labels(parent_shape_offsets, parent_shapes),
        31,
        1,
    )
    bra_data, bra_offsets, bra_shape_offsets, bra_shapes = pool(bra)
    ket_data, ket_offsets, ket_shape_offsets, ket_shapes = pool(ket)
    output_offsets = np.asarray([0, np.prod(output_shape)], dtype=np.int64)
    output_shape_offsets = np.asarray([0, 3], dtype=np.int64)
    output_shapes = np.asarray(output_shape, dtype=np.int64)
    output_labels = packed_labels(output_shape_offsets, output_shapes)

    actual, same_topology = engine.advance_normal_complementary_boundary(
        side,
        parent_bond,
        child_bond,
        0,
        False,
        np.asarray(
            [[
                0,
                0,
                0,
                transition,
                1,
                0,
                0,
                q_lb.two_j,
                q_lk.two_j,
                q_pb.two_j,
                q_pk.two_j,
                q_rb.two_j,
                q_rk.two_j,
            ]],
            dtype=np.int64,
        ),
        bra_data,
        bra_offsets,
        bra_shape_offsets,
        bra_shapes,
        ket_data,
        ket_offsets,
        ket_shape_offsets,
        ket_shapes,
        output_offsets,
        output_shape_offsets,
        output_shapes,
        output_labels,
        32,
        2,
    )

    assert same_topology is False
    np.testing.assert_allclose(
        actual.reshape(output_shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    reused, same_topology = engine.advance_normal_complementary_boundary(
        side,
        parent_bond,
        child_bond,
        0,
        False,
        None,
        bra_data,
        bra_offsets,
        bra_shape_offsets,
        bra_shapes,
        ket_data,
        ket_offsets,
        ket_shape_offsets,
        ket_shapes,
        output_offsets,
        output_shape_offsets,
        output_shapes,
        output_labels,
        32,
        3,
        False,
        32,
    )
    assert same_topology is True
    np.testing.assert_allclose(
        reused.reshape(output_shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    from pyqed.mps.nonabelian.tensor import NonabelianTensor

    bra_key = (
        (q_lb, q_pb, q_rb)
        if side == "left"
        else (q_rb, q_pb, q_lb)
    )
    ket_key = (
        (q_lk, q_pk, q_rk)
        if side == "left"
        else (q_rk, q_pk, q_lk)
    )
    site_tensor = NonabelianTensor(
        data={bra_key: bra, ket_key: ket},
        qns=[
            list(dict.fromkeys((bra_key[0], ket_key[0]))),
            list(dict.fromkeys((bra_key[1], ket_key[1]))),
            list(dict.fromkeys((bra_key[2], ket_key[2]))),
        ],
        dirs=[-1, 1, 1],
        metadata={},
    )
    installed = engine.install_split_site(0, site_tensor)
    assert installed["blocks"] == 2
    direct, direct_same_topology = (
        engine.advance_normal_complementary_boundary_from_split_site(
            side,
            parent_bond,
            child_bond,
            0,
            False,
            np.asarray(
                [[
                    0,
                    0,
                    0,
                    transition,
                    1,
                    0,
                    0,
                    q_lb.two_j,
                    q_lk.two_j,
                    q_pb.two_j,
                    q_pk.two_j,
                    q_rb.two_j,
                    q_rk.two_j,
                ]],
                dtype=np.int64,
            ),
            (bra_key,),
            (ket_key,),
            site_tensor.metadata["_cpp_split_site"],
            output_offsets,
            output_shape_offsets,
            output_shapes,
            output_labels,
            32,
            3,
            route_topology_revision=17,
        )
    )
    assert direct_same_topology is True
    np.testing.assert_allclose(
        direct.reshape(output_shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert engine.stats["split_site_boundary_uses"] == 1
    assert engine.stats["split_site_count"] == 1
    replayed, replayed_same_topology = (
        engine.advance_normal_complementary_boundary_from_split_site(
            side,
            parent_bond,
            child_bond,
            0,
            False,
            np.empty((0, 13), dtype=np.int64),
            (bra_key,),
            (ket_key,),
            site_tensor.metadata["_cpp_split_site"],
            output_offsets,
            output_shape_offsets,
            output_shapes,
            output_labels,
            32,
            4,
            route_topology_revision=17,
        )
    )
    assert replayed_same_topology is True
    np.testing.assert_allclose(
        replayed.reshape(output_shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert engine.stats["cached_boundary_replays"] == 1
    assert engine.stats["split_site_boundary_uses"] == 2


@pytest.mark.parametrize("side", ("left", "right"))
def test_cpp_su2_engine_replays_metric_boundary_from_split_site(side):
    """The cached norm action consumes the current C++-owned split site."""

    from pyqed.mps.nonabelian.tensor import NonabelianTensor

    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    engine = engine_cls(np.eye(2), np.zeros((2,) * 4), 2)
    rng = np.random.default_rng(20260729)
    q0 = SpinChargeSector(0, SU2Irrep(0))
    key = (q0, q0, q0)
    site = rng.normal(size=(3, 2, 4))
    site_tensor = NonabelianTensor(
        data={key: site},
        qns=[[q0], [q0], [q0]],
        dirs=[-1, 1, 1],
        metadata={},
    )
    engine.install_split_site(0, site_tensor)

    def pool(array):
        array = np.ascontiguousarray(array, dtype=float)
        return (
            array.reshape(-1),
            np.asarray([0, array.size], dtype=np.int64),
            np.asarray([0, array.ndim], dtype=np.int64),
            np.asarray(array.shape, dtype=np.int64),
        )

    def packed_labels(shape_offsets, shapes):
        topology = (
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray(shape_offsets, dtype=np.int64),
            np.asarray(shapes, dtype=np.int64),
        )
        return np.concatenate(
            (
                np.asarray(
                    [array.size for array in topology],
                    dtype=np.int64,
                ),
                *topology,
            )
        )

    identity = np.eye(2, dtype=float).reshape(1, 1, 2, 2)
    mpo_data, mpo_offsets, mpo_shape_offsets, mpo_shapes = pool(identity)
    if side == "left":
        parent = rng.normal(size=(1, 3, 3))
        expected = np.einsum(
            "xij,ipr,xypq,jqs->yrs",
            parent,
            site,
            identity,
            site,
            optimize=False,
        )
        parent_bond, child_bond = 0, 1
    else:
        parent = rng.normal(size=(1, 4, 4))
        expected = np.einsum(
            "ipr,xypq,yrs,jqs->xij",
            site,
            identity,
            parent,
            site,
            optimize=False,
        )
        parent_bond, child_bond = 1, 0
    parent_data, parent_offsets, parent_shape_offsets, parent_shapes = pool(
        parent
    )
    engine.install_metric_boundary(
        side,
        parent_bond,
        parent_data,
        parent_offsets,
        packed_labels(parent_shape_offsets, parent_shapes),
        101,
        1,
    )
    output_offsets = np.asarray([0, expected.size], dtype=np.int64)
    output_shape_offsets = np.asarray([0, expected.ndim], dtype=np.int64)
    output_shapes = np.asarray(expected.shape, dtype=np.int64)
    output_labels = packed_labels(output_shape_offsets, output_shapes)
    marker = site_tensor.metadata["_cpp_split_site"]
    route_coefficient = 0.25 if side == "right" else 1.0
    expected = route_coefficient * expected

    actual, same_topology = (
        engine.advance_metric_boundary_from_split_site(
            side,
            parent_bond,
            child_bond,
            0,
            np.asarray([[0, 0, 0, 0, 0]], dtype=np.int64),
            (key,),
            (key,),
            marker,
            mpo_data,
            mpo_offsets,
            mpo_shape_offsets,
            mpo_shapes,
            output_offsets,
            output_shape_offsets,
            output_shapes,
            output_labels,
            102,
            2,
            route_topology_revision=17,
            route_coefficients=np.asarray([route_coefficient]),
        )
    )
    assert same_topology is False
    np.testing.assert_allclose(
        actual.reshape(expected.shape),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    replacement = np.ascontiguousarray(0.5 * site)
    site_tensor.data[key] = replacement
    engine.install_split_site(0, site_tensor)
    expected_replay = expected * 0.25
    replayed, replayed_same_topology = (
        engine.advance_metric_boundary_from_split_site(
            side,
            parent_bond,
            child_bond,
            0,
            np.empty((0, 5), dtype=np.int64),
            (key,),
            (key,),
            site_tensor.metadata["_cpp_split_site"],
            mpo_data,
            mpo_offsets,
            mpo_shape_offsets,
            mpo_shapes,
            output_offsets,
            output_shape_offsets,
            output_shapes,
            output_labels,
            102,
            3,
            route_topology_revision=17,
            route_coefficients=np.empty(0, dtype=float),
        )
    )
    assert replayed_same_topology is True
    np.testing.assert_allclose(
        replayed.reshape(expected.shape),
        expected_replay,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert engine.stats["cached_boundary_replays"] == 1
    assert engine.stats["split_site_boundary_uses"] == 2


def test_su2_system_revision_covers_all_immutable_inputs():
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    h1 = np.eye(2)
    eri = np.zeros((2, 2, 2, 2))
    reference = engine_cls(h1, eri, 2, ecore=0.0, orb_sym=[0, 0])
    identical = engine_cls(h1, eri, 2, ecore=0.0, orb_sym=[0, 0])
    changed_core = engine_cls(h1, eri, 2, ecore=0.25, orb_sym=[0, 0])
    changed_symmetry = engine_cls(h1, eri, 2, ecore=0.0, orb_sym=[0, 1])

    revision = reference.system_stats["revision"]
    assert identical.system_stats["revision"] == revision
    assert changed_core.system_stats["revision"] != revision
    assert changed_symmetry.system_stats["revision"] != revision


def test_cpp_su2_engine_owns_complete_bond_lifecycle_and_abort():
    engine_cls = _require_su2_cpp("SU2MovingEnvironment")
    engine = engine_cls(np.eye(3), np.zeros((3,) * 4), 2)

    engine.begin_half_sweep("lr", 3)
    assert engine.stats["lifecycle_phase"] == "ready"
    engine.begin_bond(0)
    assert engine.stats["active_bond"] == 0
    engine.mark_bond_solved()
    engine.mark_bond_split(kept_states=7, truncation_seconds=0.05)
    engine.mark_bond_advanced()
    engine.commit_bond(
        matvec_calls=11,
        davidson_iterations=4,
        matvec_seconds=0.2,
        davidson_seconds=0.3,
        energy=-1.25,
    )
    with pytest.raises(RuntimeError, match="every C\\+\\+ bond update"):
        engine.finish_half_sweep()
    engine.abort_half_sweep()

    stats = engine.stats
    assert stats["lifecycle_phase"] == "idle"
    assert stats["active_bond"] == -1
    assert stats["aborted_half_sweeps"] == 1
    assert stats["bond_prepares"] == 1
    assert stats["bond_solves"] == 1
    assert stats["bond_splits"] == 1
    assert stats["bond_advances"] == 1

    engine.begin_half_sweep("rl", 3)
    for bond in (1, 0):
        assert engine.claim_next_bond() == bond
        engine.mark_bond_solved()
        engine.mark_bond_split(kept_states=3, truncation_seconds=0.0)
        engine.mark_bond_advanced()
        engine.commit_bond()
    assert engine.claim_next_bond() == -1
    engine.finish_half_sweep()
    assert engine.stats["half_sweeps"] == 1
    assert np.isnan(engine.stats["last_half_sweep_energy"])

    engine.begin_half_sweep("lr", 2)
    assert engine.claim_next_bond() == 0
    engine.mark_bond_solved()
    engine.mark_bond_split(kept_states=3, truncation_seconds=0.0)
    engine.mark_bond_advanced()
    engine.commit_bond(energy=-2.5)
    engine.finish_half_sweep()
    assert engine.stats["last_half_sweep_energy"] == pytest.approx(-2.5)

    visited = []
    engine.begin_half_sweep("rl", 4)

    def execute_bond(bond):
        visited.append(int(bond))
        engine.mark_bond_solved()
        engine.mark_bond_split(kept_states=2)
        engine.mark_bond_advanced()
        engine.commit_bond(energy=-3.0 - 0.1 * bond)

    assert engine.execute_half_sweep(execute_bond) == 3
    assert visited == [2, 1, 0]
    engine.finish_half_sweep()
    stats = engine.stats
    assert stats["half_sweep_executor_calls"] == 1
    assert stats["half_sweep_executor_bonds"] == 3
    assert stats["half_sweep_python_bond_callbacks"] == 3
    assert stats["last_half_sweep_energy"] == pytest.approx(-3.0)


def test_nonabelian_moving_environment_uses_cpp_half_sweep_cursor():
    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    from pyqed.mps.nonabelian.sweep import MovingEnvironment

    owner = owner_cls(np.eye(6), np.zeros((6,) * 4), 6)
    connected = MovingEnvironment(
        [],
        mpo_factors=[],
        su2_moving_environment=owner,
    )
    assert connected.hamiltonian_stack.su2_moving_environment is owner

    moving = MovingEnvironment.__new__(MovingEnvironment)
    moving.su2_moving_environment = owner
    moving.cursor_owner = owner
    moving.cursor_calls = 0
    moving.cursor_steps = 0
    moving.cursor_failures = 0

    assert moving.sweep_bonds("lr", 6) == (0, 1, 2, 3, 4)
    assert moving.sweep_bonds("rl", 6) == (4, 3, 2, 1, 0)
    assert moving.cursor_calls == 2
    assert moving.cursor_steps == 10
    assert moving.cursor_failures == 0


def test_cpp_su2_parent_block_table_matches_numpy_and_reuses_storage():
    table_cls = _require_cpp_kernel("SU2ParentBlockTable")
    rng = np.random.default_rng(271828)
    batch_blocks = rng.normal(size=(4, 3, 2)) + 1j * rng.normal(size=(4, 3, 2))
    singleton_block = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    batch = SimpleNamespace(
        blocks=np.ascontiguousarray(batch_blocks),
        in_comps=np.asarray([0, 1, 0, 1], dtype=np.int64),
        out_comps=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    table = table_cls(
        (batch,),
        ((0, 2, np.ascontiguousarray(singleton_block)),),
    )
    parent_inputs = [
        np.ascontiguousarray(rng.normal(size=2) + 1j * rng.normal(size=2)),
        np.ascontiguousarray(rng.normal(size=2) + 1j * rng.normal(size=2)),
    ]

    expected = [
        np.zeros(3, dtype=complex),
        np.zeros(3, dtype=complex),
        np.zeros(2, dtype=complex),
    ]
    for block, in_comp, out_comp in zip(
        batch_blocks,
        batch.in_comps,
        batch.out_comps,
    ):
        expected[int(out_comp)] += block @ parent_inputs[int(in_comp)]
    expected[2] += singleton_block @ parent_inputs[0]

    for expected_apply_calls in (1, 2):
        actual = [np.zeros_like(output) for output in expected]
        assert table.apply(parent_inputs, actual)
        for actual_output, expected_output in zip(actual, expected):
            np.testing.assert_allclose(actual_output, expected_output)
        assert table.stats["apply_calls"] == expected_apply_calls

    assert table.stats["blocks"] == 5
    assert table.stats["stored_elements"] == batch_blocks.size + singleton_block.size


def test_su2_local_action_routes_parent_blocks_through_cpp_table():
    if getattr(cpp_davidson, "SU2ParentBlockTable", None) is None:
        pytest.skip("C++ SU(2) parent-block table is unavailable")
    from pyqed.mps.nonabelian.su2_kernel import SU2LocalAction

    class Basis:
        orthonormal_dim = 4
        n_components = 2
        component_transforms = (np.eye(2, dtype=complex),) * 2
        component_indices = (np.arange(2), np.arange(2))

        @staticmethod
        def _orth_slice(idx):
            return slice(2 * int(idx), 2 * int(idx) + 2)

    rng = np.random.default_rng(161803)
    parent_blocks = tuple(
        (
            in_comp,
            out_comp,
            np.ascontiguousarray(
                rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            ),
        )
        for in_comp, out_comp in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    action = SU2LocalAction.from_parent_blocks(
        Basis(),
        parent_blocks,
        backend="cpp",
    )
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    expected = np.zeros(4, dtype=complex)
    for in_comp, out_comp, block in parent_blocks:
        expected[2 * out_comp : 2 * out_comp + 2] += (
            block @ vector[2 * in_comp : 2 * in_comp + 2]
        )

    np.testing.assert_allclose(action.matvec(vector), expected)
    assert action.stats["cpp_parent_block_table"] is True
    assert action.stats["cpp_parent_block_table_stats"]["apply_calls"] == 1


def test_su2_parent_block_estimate_counts_unique_component_pairs():
    from pyqed.mps.nonabelian.renormalized import DirectOrthonormalFactorizedTable

    table = object.__new__(DirectOrthonormalFactorizedTable)
    object.__setattr__(
        table,
        "compiled_factorized_terms",
        SimpleNamespace(
            in_indices=np.asarray([0, 1, 0, 2], dtype=np.int64),
            out_indices=np.asarray([1, 0, 1, 2], dtype=np.int64),
        ),
    )
    object.__setattr__(table, "components", ((0, 1), (2,)))
    object.__setattr__(
        table,
        "component_basis",
        SimpleNamespace(
            component_indices=(
                np.arange(3, dtype=np.int64),
                np.arange(5, dtype=np.int64),
            ),
        ),
    )

    # Unique component routes are 0->0 and 1->1, not all four raw matches.
    assert table._estimate_qchem_component_parent_block_elements() == 3 * 3 + 5 * 5


def test_factor_match_topology_cache_is_compact_and_byte_bounded(monkeypatch):
    from pyqed.mps.nonabelian import su2_qchem_plan

    cache = su2_qchem_plan._FACTOR_MATCH_LAYOUT_CACHE
    cache.clear()
    monkeypatch.setattr(
        su2_qchem_plan,
        "_FACTOR_MATCH_LAYOUT_CACHE_BYTES",
        0,
    )
    monkeypatch.setattr(
        su2_qchem_plan,
        "_FACTOR_MATCH_LAYOUT_CACHE_MAX_BYTES",
        64,
    )
    first = tuple(np.arange(2, dtype=np.int64) for _ in range(4))
    second = tuple(np.arange(2, dtype=np.int64) + 2 for _ in range(4))
    oversized = tuple(np.arange(3, dtype=np.int64) for _ in range(4))

    fingerprint = su2_qchem_plan._array_tuple(np.arange(1000))
    assert fingerprint[0] == (1000,)
    assert len(fingerprint[1]) == 16

    su2_qchem_plan._factor_match_layout_put(cache, "first", first)
    assert tuple(cache) == ("first",)
    assert su2_qchem_plan._FACTOR_MATCH_LAYOUT_CACHE_BYTES == 64
    su2_qchem_plan._factor_match_layout_put(cache, "second", second)
    assert tuple(cache) == ("second",)
    assert su2_qchem_plan._FACTOR_MATCH_LAYOUT_CACHE_BYTES == 64
    su2_qchem_plan._factor_match_layout_put(cache, "oversized", oversized)
    assert tuple(cache) == ("second",)
    cache.clear()


def test_packed_qchem_streaming_matvec_avoids_dense_kernel_cache(monkeypatch):
    from pyqed.mps.nonabelian import su2_qchem_plan

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")

    class Entry:
        key = ("only",)
        shape = (2, 1, 1, 2)
        size = 4
        offset = 0
        slice = slice(0, 4)

    class Basis(tuple):
        size = 4

    class Pool:
        def __init__(self, arrays):
            self.arrays = tuple(arrays)
            self.data = np.concatenate([array.reshape(-1) for array in arrays])
            self.offsets = np.asarray(
                [0, *np.cumsum([array.size for array in arrays])],
                dtype=np.int64,
            )
            self.shape_offsets = np.asarray(
                [0, *np.cumsum([array.ndim for array in arrays])],
                dtype=np.int64,
            )
            self.shapes = np.asarray(
                [dim for array in arrays for dim in array.shape],
                dtype=np.int64,
            )

        def shape(self, index):
            return self.arrays[int(index)].shape

    class FactorTable:
        def __init__(self, arrays):
            self.arrays = tuple(arrays)
            self.factor_indices = np.arange(len(arrays), dtype=np.int64)
            self.factor_pool = Pool(arrays)

        def factor(self, index):
            return self.arrays[int(index)]

    rng = np.random.default_rng(8181)
    left = tuple(rng.normal(size=(2, 2, 1, 1, 1)) for _ in range(2))
    right = tuple(rng.normal(size=(1, 2, 2, 1, 1)) for _ in range(2))
    plan = SimpleNamespace(
        bond=0,
        left_factor_table=FactorTable(left),
        right_factor_table=FactorTable(right),
        _factorized_kernel=su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel,
    )
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    compiled = su2_qchem_plan.PackedSU2QChemCompiledTerms(
        basis=Basis((Entry(),)),
        plan=plan,
        in_indices=np.asarray([0, 0], dtype=np.int64),
        out_indices=np.asarray([0, 0], dtype=np.int64),
        left_indices=np.asarray([0, 1], dtype=np.int64),
        right_indices=np.asarray([0, 1], dtype=np.int64),
        su2_moving_environment=owner,
    )
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    expected_kernel = su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel(
        np.stack(left),
        np.stack(right),
        Entry(),
        Entry(),
    )
    monkeypatch.setattr(
        su2_qchem_plan,
        "_PACKED_QCHEM_ENTRY_KERNEL_CACHE_MAX_ELEMENTS",
        0,
    )

    np.testing.assert_allclose(
        compiled.apply_packed(vector, base_dtype=complex),
        expected_kernel @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        compiled.diagonal(),
        np.real(np.diag(expected_kernel)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    metric_routes = compiled.factorized_metric_routes()
    assert metric_routes is not None
    owner.install_factorized_metric(
        "reduced-qchem-metric",
        tuple((row[2], row[3], row[4], row[5]) for row in metric_routes),
        4,
        71,
        72,
    )
    np.testing.assert_allclose(
        owner.factorized_metric_matvec("reduced-qchem-metric", vector),
        expected_kernel @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert getattr(compiled, "_entry_kernel_items_cache", None) is None
    assert compiled.cpp_factor_route_calls == 1
    assert compiled.cpp_factor_diagonal_calls == 1
    stats = owner.stats
    assert stats["factor_route_count"] == 2
    assert stats["factor_route_topology_builds"] == 1
    assert stats["factor_route_numeric_refreshes"] == 1
    assert stats["factor_route_matvec_calls"] == 1
    assert stats["factor_route_diagonal_calls"] == 1
    assert stats["factor_route_scratch_growths"] == 1
    assert stats["borrowed_factor_pool_bytes"] == (
        plan.left_factor_table.factor_pool.data.nbytes
        + plan.right_factor_table.factor_pool.data.nbytes
    )

    transform, _ = np.linalg.qr(rng.normal(size=(4, 3)))
    projection_key = "lr:0:factor_projection:test"
    owner.install_factor_route_projection(
        projection_key,
        compiled._cpp_factor_route_key,
        (np.arange(4, dtype=np.int64),),
        (np.ascontiguousarray(transform),),
        (0,),
        4,
        3,
        101,
        202,
    )
    reduced_vector = rng.normal(size=3) + 1j * rng.normal(size=3)
    np.testing.assert_allclose(
        owner.factor_route_projected_matvec(projection_key, reduced_vector),
        transform.conj().T @ expected_kernel @ transform @ reduced_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert owner.stats["borrowed_factor_route_transform_bytes"] == transform.nbytes

    replacement_plan = SimpleNamespace(
        bond=1,
        left_factor_table=FactorTable(left[:1]),
        right_factor_table=FactorTable(right[:1]),
        _factorized_kernel=su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel,
    )
    su2_qchem_plan.PackedSU2QChemCompiledTerms(
        basis=Basis((Entry(),)),
        plan=replacement_plan,
        in_indices=np.asarray([0], dtype=np.int64),
        out_indices=np.asarray([0], dtype=np.int64),
        left_indices=np.asarray([0], dtype=np.int64),
        right_indices=np.asarray([0], dtype=np.int64),
        su2_moving_environment=owner,
    )
    assert owner.stats["factor_route_projection_components"] == 0
    assert owner.stats["borrowed_factor_route_transform_bytes"] == 0
    assert owner.stats["factor_route_projection_scratch_bytes"] == 0


def test_packed_qchem_raw_factors_stay_in_cpp_moving_environment():
    from pyqed.mps.nonabelian import su2_qchem_plan

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")

    class Entry:
        shape = (2, 2, 2, 2)
        size = 16
        offset = 0
        slice = slice(0, 16)

    class Basis(tuple):
        size = 16

    class Pool:
        def __init__(self, arrays):
            arrays = tuple(np.ascontiguousarray(value, dtype=float) for value in arrays)
            self.arrays = arrays
            if arrays:
                self.data, self.offsets, self.shape_offsets, self.shapes = (
                    _packed_pool(arrays)
                )
            else:
                self.data = np.empty(0, dtype=float)
                self.offsets = np.asarray([0], dtype=np.int64)
                self.shape_offsets = np.asarray([0], dtype=np.int64)
                self.shapes = np.empty(0, dtype=np.int64)

        @property
        def n_arrays(self):
            return len(self.arrays)

    class RawFactorTable:
        family_labels = ()

        def __init__(self, boundary, w_block):
            self.factor_indices = np.asarray([0], dtype=np.int64)
            self.factor_pool = Pool(())
            self.factor_boundary_pool = Pool((boundary,))
            self.factor_w_pool = Pool((w_block,))
            self.factor_boundary_array_ids = np.asarray([0], dtype=np.int64)
            self.factor_w_block_ids = np.asarray([0], dtype=np.int64)
            self.released = False

        def release_materialized_factors(self):
            self.released = True
            return True

    rng = np.random.default_rng(65537)
    left_boundary = rng.normal(size=(2, 2, 2))
    left_w = rng.normal(size=(2, 2, 2, 2))
    right_w = rng.normal(size=(2, 2, 2, 2))
    right_boundary = rng.normal(size=(2, 2, 2))
    left_table = RawFactorTable(left_boundary, left_w)
    right_table = RawFactorTable(right_boundary, right_w)
    plan = SimpleNamespace(
        bond=3,
        left_factor_table=left_table,
        right_factor_table=right_table,
        _factorized_kernel=su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel,
    )
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    compiled = su2_qchem_plan.PackedSU2QChemCompiledTerms(
        basis=Basis((Entry(),)),
        plan=plan,
        in_indices=np.asarray([0], dtype=np.int64),
        out_indices=np.asarray([0], dtype=np.int64),
        left_indices=np.asarray([0], dtype=np.int64),
        right_indices=np.asarray([0], dtype=np.int64),
        su2_moving_environment=owner,
    )
    left_factor = np.einsum(
        "xlk,xwab->lkwab",
        left_boundary,
        left_w,
        optimize=True,
    )
    right_factor = np.einsum(
        "wydc,yqr->wqrdc",
        right_w,
        right_boundary,
        optimize=True,
    )
    expected = su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel(
        left_factor[None, ...],
        right_factor[None, ...],
        Entry(),
        Entry(),
    )
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)

    np.testing.assert_allclose(
        compiled.apply_packed(vector, base_dtype=complex),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    real_vector = rng.normal(size=16)
    np.testing.assert_allclose(
        owner.factor_route_real_matvec(
            compiled._cpp_factor_route_key,
            real_vector,
        ),
        expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    owner.set_factor_routes_hermitianized(True)
    hermitian_expected = 0.5 * (expected + expected.T.conj())
    np.testing.assert_allclose(
        owner.factor_route_matvec(
            compiled._cpp_factor_route_key,
            vector,
        ),
        hermitian_expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factor_route_real_matvec(
            compiled._cpp_factor_route_key,
            real_vector,
        ),
        hermitian_expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    owner.set_factor_routes_hermitianized(False)
    np.testing.assert_allclose(
        compiled.diagonal(),
        np.real(np.diag(expected)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert left_table.released and right_table.released
    stats = owner.stats
    assert stats["raw_factor_routes"] is True
    assert stats["factor_route_count"] == 1
    assert stats["borrowed_raw_factor_source_bytes"] > 0
    assert stats["raw_factor_cache_misses"] == 2
    assert stats["resident_family_kernel_count"] == 1
    assert stats["resident_family_route_count"] == 1
    assert stats["resident_family_kernel_bytes"] <= 32_000_000
    assert (
        stats["resident_family_factor_pack_bytes"]
        <= stats["resident_family_factor_pack_budget_bytes"]
        == 4_000_000
    )
    assert stats["raw_factor_gemm_calls"] >= 4
    assert stats["raw_factor_cache_bytes"] == (
        left_factor.nbytes + right_factor.nbytes
    )
    assert stats["complementary_family_route_counts"] == {
        "S": 0,
        "R": 0,
        "A": 0,
        "P": 0,
        "B": 0,
        "Q": 0,
        "unlabeled": 1,
    }

    route_key = compiled._cpp_factor_route_key
    owner.begin_half_sweep("lr", 2)
    owner.record_bond(0)
    owner.finish_half_sweep()
    del compiled, plan, left_table, right_table
    gc.collect()
    np.testing.assert_allclose(
        owner.factor_route_matvec(route_key, vector),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_cpp_raw_factor_biclique_dense_pair_matches_reference():
    from pyqed.mps.nonabelian import su2_qchem_plan

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(104729)
    left_boundaries = tuple(rng.normal(size=(2, 2, 2)) for _ in range(2))
    left_ws = tuple(rng.normal(size=(2, 2, 2, 2)) for _ in range(2))
    right_boundaries = tuple(rng.normal(size=(2, 2, 2)) for _ in range(2))
    right_ws = tuple(rng.normal(size=(2, 2, 2, 2)) for _ in range(2))

    def raw_source(boundaries, w_blocks):
        b_data, b_offsets, b_shape_offsets, b_shapes = _packed_pool(boundaries)
        w_data, w_offsets, w_shape_offsets, w_shapes = _packed_pool(w_blocks)
        return (
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            b_offsets,
            b_shape_offsets,
            b_shapes,
            b_data,
            w_offsets,
            w_shape_offsets,
            w_shapes,
            w_data,
        )

    left_source = raw_source(left_boundaries, left_ws)
    right_source = raw_source(right_boundaries, right_ws)
    left_factors = tuple(
        np.einsum("xlk,xwab->lkwab", boundary, w_block, optimize=True)
        for boundary, w_block in zip(left_boundaries, left_ws)
    )
    right_factors = tuple(
        np.einsum("wydc,yqr->wqrdc", w_block, boundary, optimize=True)
        for boundary, w_block in zip(right_boundaries, right_ws)
    )
    route_left = np.asarray([0, 0, 1, 1], dtype=np.int64)
    route_right = np.asarray([0, 1, 0, 1], dtype=np.int64)

    class Entry:
        shape = (2, 2, 2, 2)
        size = 16
        offset = 0
        slice = slice(0, 16)

    expected = su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel(
        np.stack([left_factors[index] for index in route_left]),
        np.stack([right_factors[index] for index in route_right]),
        Entry(),
        Entry(),
    )
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    owner.install_raw_factor_routes(
        "raw-biclique",
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        route_left,
        route_right,
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 2, 2, 2]], dtype=np.int64),
        np.arange(2, dtype=np.int64),
        left_source,
        np.arange(2, dtype=np.int64),
        right_source,
        16,
        123,
        456,
    )
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)

    np.testing.assert_allclose(
        owner.factor_route_matvec("raw-biclique", vector),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factor_route_diagonal("raw-biclique", 16),
        np.real(np.diag(expected)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    owner.install_factorized_metric(
        "raw-biclique-metric",
        ((Entry(), Entry(), np.eye(2), np.eye(2)),),
        16,
        201,
        202,
    )
    from pyqed.mps.nonabelian.solver import _KroneckerBasisTransformBlock

    identity_projection = _KroneckerBasisTransformBlock(
        row_slice=slice(0, 16),
        orthonormal_indices=np.arange(16, dtype=np.int64),
        left_dim=4,
        selected_dim=2,
        local_dim=2,
        right_dim=2,
        local_transform=np.eye(2),
    )
    owner.install_indexed_factor_route_projection(
        "raw-biclique-projection",
        "raw-biclique",
        (identity_projection,),
        16,
        16,
        203,
        204,
    )
    real_vector = rng.normal(size=16)
    np.testing.assert_allclose(
        owner.factor_route_real_matvec("raw-biclique", real_vector),
        expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factor_route_projected_real_matvec(
            "raw-biclique-projection",
            real_vector,
        ),
        expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factorized_metric_real_matvec(
            "raw-biclique-metric",
            real_vector,
        ),
        real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    stats = owner.stats
    assert stats["fused_raw_route_group_count"] == 1
    assert stats["fused_raw_route_count"] == 4
    assert stats["dense_pair_kernel_count"] == 1
    assert stats["dense_pair_route_count"] == 4
    assert stats["dense_pair_kernel_elements"] == 16 * 16
    topology_builds = stats["factor_route_topology_builds"]
    numeric_refreshes = stats["factor_route_numeric_refreshes"]
    assert owner.install_raw_factor_routes(
        "raw-biclique",
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        route_left,
        route_right,
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 2, 2, 2]], dtype=np.int64),
        np.arange(2, dtype=np.int64),
        tuple(np.array(value, copy=True) for value in left_source),
        np.arange(2, dtype=np.int64),
        tuple(np.array(value, copy=True) for value in right_source),
        16,
        123,
        456,
    )
    assert owner.stats["factor_route_topology_builds"] == topology_builds
    assert owner.stats["factor_route_numeric_refreshes"] == numeric_refreshes
    np.testing.assert_allclose(
        owner.factor_route_matvec("raw-biclique", vector),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    updated_left_boundaries = (
        1.25 * left_boundaries[0],
        left_boundaries[1],
    )
    updated_left_source = raw_source(updated_left_boundaries, left_ws)
    updated_left_factors = tuple(
        np.einsum("xlk,xwab->lkwab", boundary, w_block, optimize=True)
        for boundary, w_block in zip(updated_left_boundaries, left_ws)
    )
    updated_expected = su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel(
        np.stack([updated_left_factors[index] for index in route_left]),
        np.stack([right_factors[index] for index in route_right]),
        Entry(),
        Entry(),
    )
    assert owner.install_raw_factor_routes(
        "raw-biclique",
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        route_left,
        route_right,
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 2, 2, 2]], dtype=np.int64),
        np.arange(2, dtype=np.int64),
        updated_left_source,
        np.arange(2, dtype=np.int64),
        right_source,
        16,
        123,
        457,
    )
    assert owner.stats["factor_route_topology_builds"] == topology_builds
    assert (
        owner.stats["factor_route_numeric_refreshes"]
        == numeric_refreshes + 1
    )
    np.testing.assert_allclose(
        owner.factor_route_matvec("raw-biclique", vector),
        updated_expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factor_route_projected_real_matvec(
            "raw-biclique-projection",
            real_vector,
        ),
        updated_expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_cpp_raw_factor_superchannel_owns_topology_and_matches_reference():
    from pyqed.mps.nonabelian import su2_qchem_plan

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    rng = np.random.default_rng(130363)
    left_boundary = rng.normal(size=(2, 2, 2))
    right_boundary = rng.normal(size=(2, 2, 2))
    left_ws = (
        rng.normal(size=(2, 1, 1, 1)),
        rng.normal(size=(2, 2, 1, 1)),
    )
    right_ws = (
        rng.normal(size=(1, 2, 1, 1)),
        rng.normal(size=(2, 2, 1, 1)),
    )

    def raw_source(boundary, w_blocks):
        boundary_pool = _packed_pool((boundary,))
        w_pool = _packed_pool(w_blocks)
        return (
            np.zeros(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            boundary_pool[1],
            boundary_pool[2],
            boundary_pool[3],
            boundary_pool[0],
            w_pool[1],
            w_pool[2],
            w_pool[3],
            w_pool[0],
        )

    left_source = raw_source(left_boundary, left_ws)
    right_source = raw_source(right_boundary, right_ws)
    left_factors = tuple(
        np.einsum("xlk,xwab->lkwab", left_boundary, block, optimize=True)
        for block in left_ws
    )
    right_factors = tuple(
        np.einsum("wydc,yqr->wqrdc", block, right_boundary, optimize=True)
        for block in right_ws
    )

    class Entry:
        shape = (2, 1, 1, 2)
        size = 4
        offset = 0
        slice = slice(0, 4)

    expected = sum(
        (
            su2_qchem_plan.SU2QChemSweepPlan._factorized_kernel(
                left[None, ...],
                right[None, ...],
                Entry(),
                Entry(),
            )
            for left, right in zip(left_factors, right_factors)
        ),
        np.zeros((4, 4), dtype=complex),
    )
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    owner.install_raw_factor_routes(
        "raw-superchannel",
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.arange(2, dtype=np.int64),
        np.arange(2, dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 1, 1, 2]], dtype=np.int64),
        np.arange(2, dtype=np.int64),
        left_source,
        np.arange(2, dtype=np.int64),
        right_source,
        4,
        301,
        302,
    )
    for source in (left_source, right_source):
        for index in (0, 1, 2, 3, 4, 6, 7, 8):
            source[index].fill(-1)

    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    real_vector = rng.normal(size=4)
    np.testing.assert_allclose(
        owner.factor_route_matvec("raw-superchannel", vector),
        expected @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        owner.factor_route_real_matvec("raw-superchannel", real_vector),
        expected @ real_vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    stats = owner.stats
    assert stats["resident_family_kernel_count"] == 1
    assert stats["resident_family_route_count"] == 2
    assert stats["raw_execution_group_count"] == 0
    assert stats["raw_execution_action_count"] == 0


def test_cpp_factor_route_projection_runs_davidson_without_python_callbacks():
    from pyqed.mps.nonabelian.renormalized import (
        MovingEnvironmentFactorRouteTable,
    )

    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )
    left = np.asarray(
        [[-1.0, 0.2], [0.2, 0.5]],
        dtype=float,
    ).reshape(2, 2, 1, 1, 1)
    right = np.asarray(
        [[1.0, 0.1], [0.1, 0.8]],
        dtype=float,
    ).reshape(1, 2, 2, 1, 1)
    owner.install_factor_routes(
        "route",
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([[2, 1, 1, 2]], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, left.size], dtype=np.int64),
        np.asarray([0, 5], dtype=np.int64),
        np.asarray(left.shape, dtype=np.int64),
        left.reshape(-1),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, right.size], dtype=np.int64),
        np.asarray([0, 5], dtype=np.int64),
        np.asarray(right.shape, dtype=np.int64),
        right.reshape(-1),
        4,
        11,
        12,
    )
    dense = np.column_stack(
        [
            owner.factor_route_matvec("route", np.eye(4, dtype=complex)[:, idx])
            for idx in range(4)
        ]
    )
    raw_result = owner.factor_route_davidson(
        "route",
        np.diag(dense),
        np.ones(4, dtype=complex),
        1.0e-11,
        50,
        12,
        False,
    )
    assert raw_result["accepted"]
    assert raw_result["energy"] == pytest.approx(
        np.linalg.eigvalsh(dense)[0],
        abs=1.0e-10,
    )
    assert owner.stats["factor_route_davidson_calls"] == 1

    rng = np.random.default_rng(12345)
    transform, _ = np.linalg.qr(rng.normal(size=(4, 3)))
    owner.install_factor_route_projection(
        "projection",
        "route",
        (np.arange(4, dtype=np.int64),),
        (np.ascontiguousarray(transform),),
        (0,),
        4,
        3,
        21,
        22,
    )
    reduced = transform.conj().T @ dense @ transform
    result = owner.factor_route_projected_davidson(
        "projection",
        np.diag(reduced),
        np.ones(3, dtype=complex),
        1.0e-11,
        50,
        12,
        False,
    )

    assert result["accepted"]
    assert result["kind"] == "cpp_su2_factor_route_davidson"
    assert result["energy"] == pytest.approx(
        np.linalg.eigvalsh(reduced)[0],
        abs=1.0e-10,
    )
    assert owner.stats["factor_route_projected_davidson_calls"] == 1
    table_stats = MovingEnvironmentFactorRouteTable(
        owner,
        "projection",
        3,
    ).stats
    assert table_stats["davidson_calls"] == 1
    assert table_stats["matvec_calls"] > 0


def test_cpp_factor_route_generalized_davidson_uses_factorized_metric(
    monkeypatch,
):
    monkeypatch.setenv("PYQED_SU2_USE_REAL_DAVIDSON", "1")
    owner_cls = _require_su2_cpp("SU2MovingEnvironment")
    owner = owner_cls(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )

    class Entry:
        shape = (2, 1, 1, 2)
        size = 4
        offset = 0
        slice = slice(0, 4)

    left = np.asarray(
        [[-1.0, 0.2], [0.2, 0.5]],
        dtype=float,
    ).reshape(2, 2, 1, 1, 1)
    right = np.asarray(
        [[1.0, 0.1], [0.1, 0.8]],
        dtype=float,
    ).reshape(1, 2, 2, 1, 1)
    owner.install_factor_routes(
        "generalized-route",
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([Entry.shape], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, left.size], dtype=np.int64),
        np.asarray([0, 5], dtype=np.int64),
        np.asarray(left.shape, dtype=np.int64),
        left.reshape(-1),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, right.size], dtype=np.int64),
        np.asarray([0, 5], dtype=np.int64),
        np.asarray(right.shape, dtype=np.int64),
        right.reshape(-1),
        4,
        31,
        32,
    )
    metric_left = np.asarray(
        [[1.3, 0.1], [0.1, 0.9]],
        dtype=float,
    )
    metric_right = np.asarray(
        [[1.1, -0.05], [-0.05, 1.4]],
        dtype=float,
    )
    owner.install_factorized_metric(
        "factorized-metric",
        ((Entry(), Entry(), metric_left, metric_right),),
        4,
        41,
        42,
    )
    identity = np.eye(4, dtype=complex)
    dense_h = np.column_stack(
        [
            owner.factor_route_matvec("generalized-route", identity[:, idx])
            for idx in range(4)
        ]
    )
    dense_n = np.column_stack(
        [
            owner.factorized_metric_matvec("factorized-metric", identity[:, idx])
            for idx in range(4)
        ]
    )
    expected_n = np.kron(metric_left, metric_right.T)
    np.testing.assert_allclose(
        dense_n,
        expected_n,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected = np.min(np.real(np.linalg.eigvals(np.linalg.solve(dense_n, dense_h))))
    result = owner.factor_route_generalized_davidson(
        "generalized-route",
        "factorized-metric",
        np.diag(dense_h),
        np.diag(dense_n),
        np.ones(4, dtype=complex),
        1.0e-12,
        1.0e-10,
        1.0e-12,
        100,
        12,
        False,
    )

    assert result["accepted"]
    assert result["kind"] == "cpp_su2_factor_route_generalized_davidson"
    assert result["energy"] == pytest.approx(expected, abs=1.0e-10)
    residual = (
        dense_h @ result["vector"]
        - result["energy"] * dense_n @ result["vector"]
    )
    assert np.linalg.norm(residual) <= 1.0e-9

    canonical = owner.prepare_canonical_reduced_projection(
        "factorized-metric",
        tolerance=1.0e-12,
    )
    assert canonical["compatible"]
    assert canonical["parent_dimension"] == 4
    assert canonical["orthonormal_dimension"] == 4
    projection_key = canonical["projection_key"]
    transform = np.column_stack(
        [
            owner.lift_factor_route_projection_vector(
                projection_key,
                np.eye(4, dtype=complex)[:, idx],
                4,
            )
            for idx in range(4)
        ]
    )
    np.testing.assert_allclose(
        transform.conj().T @ dense_n @ transform,
        np.eye(4),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    guess = np.asarray([1.0, -0.2, 0.3j, 0.5], dtype=complex)
    np.testing.assert_allclose(
        owner.canonical_reduced_projection_guess(
            projection_key,
            "factorized-metric",
            guess,
            4,
        ),
        transform.conj().T @ dense_n @ guess,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    canonical_h = transform.conj().T @ dense_h @ transform
    canonical_result = owner.factor_route_projected_davidson(
        projection_key,
        np.diag(canonical_h),
        np.ones(4, dtype=complex),
        1.0e-11,
        50,
        12,
        False,
    )
    assert canonical_result["accepted"]
    assert canonical_result["energy"] == pytest.approx(expected, abs=1.0e-10)
    canonical_parent = owner.lift_factor_route_projection_vector(
        projection_key,
        canonical_result["vector"],
        4,
    )
    canonical_residual = (
        dense_h @ canonical_parent
        - canonical_result["energy"] * dense_n @ canonical_parent
    )
    assert np.linalg.norm(canonical_residual) <= 1.0e-9
    owner.install_factorized_metric(
        "factorized-metric",
        ((Entry(), Entry(), metric_left, metric_right),),
        4,
        41,
        43,
    )
    cached_canonical = owner.prepare_canonical_reduced_projection(
        "factorized-metric",
        tolerance=1.0e-12,
    )
    assert cached_canonical["compatible"]
    assert cached_canonical["reused"]
    assert owner.stats["canonical_projection_builds"] == 1
    assert owner.stats["canonical_projection_reuses"] == 1
    assert owner.stats["canonical_projection_cache_entries"] == 1

    stats = owner.stats
    assert stats["factor_route_generalized_davidson_calls"] == 1
    assert stats["real_generalized_davidson_calls"] == 1
    assert stats["factorized_metric_matvec_calls"] > 4
    assert stats["factorized_metric_route_bytes"] > 0
    assert stats["factorized_metric_scratch_bytes"] > 0

    projection_blocks = (
        (
            slice(0, 2),
            np.asarray([0, 2], dtype=np.int64),
            np.eye(2),
        ),
        (
            slice(2, 4),
            np.asarray([1], dtype=np.int64),
            np.asarray([[1.0], [0.0]]),
        ),
    )
    transform = np.zeros((4, 3))
    for row_slice, indices, block in projection_blocks:
        transform[row_slice, indices] += block
    owner.install_indexed_factor_route_projection(
        "indexed-projection",
        "generalized-route",
        projection_blocks,
        4,
        3,
        51,
        52,
    )
    projected_h = transform.T @ dense_h @ transform
    projected_n = transform.T @ dense_n @ transform
    probe = np.asarray([0.2, -0.4, 0.7], dtype=complex)
    np.testing.assert_allclose(
        owner.factor_route_projected_matvec("indexed-projection", probe),
        projected_h @ probe,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    projected_expected = np.min(
        np.real(
            np.linalg.eigvals(
                np.linalg.solve(projected_n, projected_h)
            )
        )
    )
    projected_result = owner.factor_route_projected_generalized_davidson(
        "indexed-projection",
        "factorized-metric",
        np.diag(projected_h),
        np.diag(projected_n),
        np.ones(3, dtype=complex),
        1.0e-12,
        1.0e-10,
        1.0e-12,
        100,
        12,
        False,
    )
    assert projected_result["accepted"]
    assert projected_result["energy"] == pytest.approx(
        projected_expected,
        abs=1.0e-10,
    )
    projected_residual = (
        projected_h @ projected_result["vector"]
        - projected_result["energy"]
        * projected_n
        @ projected_result["vector"]
    )
    assert np.linalg.norm(projected_residual) <= 1.0e-9
    assert owner.stats["real_generalized_davidson_calls"] == 2

    complex_result = owner.factor_route_generalized_davidson(
        "generalized-route",
        "factorized-metric",
        np.diag(dense_h),
        np.diag(dense_n),
        np.asarray([1.0, 1.0j, -0.5, 0.25j]),
        1.0e-12,
        1.0e-10,
        1.0e-12,
        100,
        12,
        False,
    )
    assert complex_result["accepted"]
    assert complex_result["energy"] == pytest.approx(expected, abs=1.0e-10)
    assert owner.stats["factor_route_generalized_davidson_calls"] == 3
    assert owner.stats["real_generalized_davidson_calls"] == 2

    from pyqed.mps.nonabelian.solver import _KroneckerBasisTransformBlock

    compact_identity = _KroneckerBasisTransformBlock(
        row_slice=slice(0, 4),
        orthonormal_indices=np.arange(4, dtype=np.int64),
        left_dim=2,
        selected_dim=1,
        local_dim=1,
        right_dim=2,
        local_transform=np.ones((1, 1)),
    )
    owner.install_indexed_factor_route_projection(
        "kronecker-projection",
        "generalized-route",
        (compact_identity,),
        4,
        4,
        53,
        54,
    )
    np.testing.assert_allclose(
        owner.factor_route_projected_matvec(
            "kronecker-projection",
            np.asarray([0.2, -0.4, 0.7, 0.1], dtype=complex),
        ),
        dense_h @ np.asarray([0.2, -0.4, 0.7, 0.1], dtype=complex),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert owner.stats["borrowed_factor_route_transform_bytes"] == 8

    diagonal_left = np.diag([1.25, 0.8])
    diagonal_right = np.diag([0.7, 1.6])
    owner.install_factorized_metric(
        "diagonal-metric",
        ((Entry(), Entry(), diagonal_left, diagonal_right),),
        4,
        61,
        62,
    )
    diagonal_projection = owner.prepare_canonical_reduced_projection(
        "diagonal-metric",
        tolerance=1.0e-12,
    )
    assert diagonal_projection["compatible"]
    assert diagonal_projection["orthonormal_dimension"] == 4
    assert diagonal_projection["max_component_dimension"] == 1
    assert diagonal_projection["transform_elements"] == 4
    diagonal_transform = np.column_stack(
        [
            owner.lift_factor_route_projection_vector(
                diagonal_projection["projection_key"],
                np.eye(4, dtype=complex)[:, index],
                4,
            )
            for index in range(4)
        ]
    )
    diagonal_metric = np.kron(diagonal_left, diagonal_right.T)
    np.testing.assert_allclose(
        diagonal_transform.conj().T
        @ diagonal_metric
        @ diagonal_transform,
        np.eye(4),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert owner.stats["borrowed_factor_route_transform_bytes"] == 32


@pytest.mark.parametrize("role", ["left", "right"])
def test_cpp_component_parent_factor_group_matches_numpy_layout(role):
    kernel = _require_su2_cpp("pack_component_parent_factor_group")
    rng = np.random.default_rng(104729)
    shape = (2, 2, 1, 2, 2) if role == "left" else (1, 2, 2, 2, 2)
    arrays = tuple(rng.normal(size=shape) for _ in range(3))
    data, offsets, shape_offsets, shapes = _packed_pool(arrays)
    stack, (matrix, dims) = kernel(
        np.asarray([0, 2], dtype=np.int64),
        role,
        data,
        offsets,
        shape_offsets,
        shapes,
        np.asarray([0, 1, 2], dtype=np.int64),
    )

    expected_stack = np.stack([arrays[0], arrays[2]], axis=0)
    np.testing.assert_allclose(stack, expected_stack)
    if role == "left":
        tdim, ldim, kdim, wdim, adim, bdim = expected_stack.shape
        expected_matrix = expected_stack.transpose(1, 4, 2, 5, 0, 3).reshape(
            ldim * adim * kdim * bdim,
            tdim * wdim,
        )
    else:
        tdim, wdim, qdim, rdim, ddim, cdim = expected_stack.shape
        expected_matrix = expected_stack.transpose(0, 1, 4, 2, 5, 3).reshape(
            tdim * wdim,
            ddim * qdim * cdim * rdim,
        )
    np.testing.assert_allclose(matrix, expected_matrix)
    assert dims == expected_stack.shape


def test_cpp_component_parent_layout_matches_reduced_schedule():
    kernel = _require_su2_cpp("build_component_parent_block_layout")
    schedule = kernel(
        np.asarray([0, 4], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([4, 6], dtype=np.int64),
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([1, 1, 0], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 5, 10, 15], dtype=np.int64),
        np.asarray(
            [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2],
            dtype=np.int64,
        ),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 5, 10, 15], dtype=np.int64),
        np.asarray(
            [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2],
            dtype=np.int64,
        ),
    )

    assert schedule == (
        (0, 1, 0, 1, 0, 4, 0, 6, (0, 1), (0, 1)),
        (1, 0, 1, 0, 0, 6, 0, 4, (2,), (2,)),
    )


def test_su2_operator_engine_owns_and_reuses_tables(monkeypatch):
    from pyqed.mps.nonabelian import su2_qchem_plan

    calls = []
    packed_factor = object()

    def fake_factor_table(boundary, W, **kwargs):
        calls.append((boundary, W, kwargs))
        return packed_factor

    monkeypatch.setattr(
        su2_qchem_plan,
        "pack_rank_coupled_factor_table_from_boundary",
        fake_factor_table,
    )
    engine = su2_qchem_plan.SU2OperatorEngine()
    boundary = object()
    W = object()
    first = engine.factor_table(
        boundary,
        W,
        side="left",
        bond=2,
        representation="rank_coupled_left_factor_by_ket",
    )
    second = engine.factor_table(
        boundary,
        W,
        side="left",
        bond=2,
        representation="rank_coupled_left_factor_by_ket",
    )

    assert first is packed_factor
    assert second is packed_factor
    assert len(calls) == 1
    assert engine.stats["factor_hits"] == 1
    assert engine.stats["factor_puts"] == 1

    left_factor = object()
    right_factor = object()
    first_plan = engine.sweep_plan(
        bond=2,
        left_factor_table=left_factor,
        right_factor_table=right_factor,
    )
    second_plan = engine.sweep_plan(
        bond=2,
        left_factor_table=left_factor,
        right_factor_table=right_factor,
    )
    assert first_plan is second_plan
    assert engine.stats["plan_hits"] == 1
    assert engine.stats["plan_puts"] == 1


def test_su2_operator_engine_releases_stale_numeric_bond_tables(monkeypatch):
    from pyqed.mps.nonabelian import su2_qchem_plan

    tables = []

    def fake_factor_table(*args, **kwargs):
        value = float(len(tables) + 1)
        pool = SimpleNamespace(
            data=np.full(16, value),
            offsets=np.asarray([0, 16], dtype=np.int64),
            shape_offsets=np.asarray([0, 1], dtype=np.int64),
            shapes=np.asarray([16], dtype=np.int64),
        )
        table = SimpleNamespace(factor_pool=pool, revision=len(tables) + 1)
        tables.append(table)
        return table

    monkeypatch.setattr(
        su2_qchem_plan,
        "pack_rank_coupled_factor_table_from_boundary",
        fake_factor_table,
    )
    engine = su2_qchem_plan.SU2OperatorEngine(
        max_factor_tables=2,
        max_plans=1,
    )
    W = object()
    first = engine.factor_table(
        object(),
        W,
        side="left",
        bond=0,
        representation="rank_coupled_left_factor_by_ket",
    )
    second = engine.factor_table(
        object(),
        W,
        side="right",
        bond=0,
        representation="rank_coupled_right_factor_by_ket",
    )
    engine.sweep_plan(
        bond=0,
        left_factor_table=first,
        right_factor_table=second,
    )
    engine.factor_table(
        object(),
        W,
        side="left",
        bond=1,
        representation="rank_coupled_left_factor_by_ket",
    )

    assert engine.stats["factor_tables"] == 1
    assert engine.stats["plans"] == 0
    assert engine.stats["factor_clears"] == 1
    assert engine.stats["plan_clears"] == 1


def test_cpp_rank_coupled_factor_routes_pack_expected_rows_and_families():
    kernel = _require_cpp_kernel("pack_rank_coupled_factor_routes")
    result = kernel(
        np.asarray([10, 20], dtype=np.int64),
        np.asarray([0, 2, 3], dtype=np.int64),
        np.asarray([11, 12, 21], dtype=np.int64),
        np.asarray([[0, -1, 1], [2, 3, -1], [-1, 4, 5]], dtype=np.int64),
        np.asarray([100, 200], dtype=np.int64),
        np.asarray([0, 2, 3], dtype=np.int64),
        np.asarray([101, 102, 201], dtype=np.int64),
        np.asarray([0, 2, 1], dtype=np.int64),
        np.asarray([1, 0, 2], dtype=np.int64),
        np.asarray([7, 8, 9], dtype=np.int64),
        np.asarray([0, 1, 3, 3], dtype=np.int64),
        np.asarray([30, 31, 32], dtype=np.int64),
        True,
    )

    expected = {
        "key_boundary_ids": [10, 10, 20, 20],
        "key_physical_ids": [100, 200, 100, 200],
        "entry_offsets": [0, 3, 4, 5, 6],
        "out_boundary_ids": [11, 11, 12, 12, 21, 21],
        "out_physical_ids": [101, 102, 101, 201, 102, 201],
        "middle_ids": [1, 0, 1, 2, 0, 2],
        "family_offsets": [0, 2, 3, 5, 5, 6, 6],
        "family_ids": [31, 32, 30, 31, 32, 30],
        "factor_indices": [0, 1, 2, 3, 4, 5],
        "factor_boundary_array_ids": [0, 1, 2, 3, 5, 4],
        "factor_w_block_ids": [7, 8, 7, 9, 8, 9],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(result[name], np.asarray(values, dtype=np.int64))


@pytest.mark.parametrize("left_spin,right_spin,rank_spin", [(0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 2, 2)])
def test_cpp_rank_coupled_reduced_actions_match_python_cg_reference(
    left_spin,
    right_spin,
    rank_spin,
):
    kernel = _require_cpp_kernel("rank_coupled_reduced_actions")
    left = SU2Irrep(left_spin)
    right = SU2Irrep(right_spin)
    rank = SU2Irrep(rank_spin)
    operator = SimpleNamespace(rank_irrep=rank)
    term = SimpleNamespace(
        reduced_operator=operator,
        use_cg_coupling=True,
        visible_virtual_block=np.ones((1, 1), dtype=float),
    )
    compiled = kernel((term,), (left,), (right,))

    reference = []
    if right in left_or_right_fusion(left, rank):
        for row, two_m_left in enumerate(ordered_two_m_values(left)):
            for col, two_m_right in enumerate(ordered_two_m_values(right)):
                component = two_m_right - two_m_left
                coefficient = clebsch_gordan(
                    left,
                    rank,
                    right,
                    two_m_left,
                    component,
                    two_m_right,
                )
                if coefficient:
                    reference.append((row, col, component, coefficient))

    simplified = [(row, col, component, coefficient) for _, _, _, row, col, component, coefficient in compiled]
    assert len(simplified) == len(reference)
    for actual, expected in zip(simplified, reference):
        assert actual[:3] == expected[:3]
        assert not isinstance(actual[3], complex)
        assert actual[3] == pytest.approx(expected[3], abs=1.0e-14)
