import numpy as np
import pytest

from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.nonabelian.mps import MPS as ReducedMPS
from pyqed.mps.nonabelian.orbital_transform import (
    _adjacent_unitary_circuit,
    _orbital_circuit,
    _second_quantized_two_orbital_gate,
    apply_spatial_orbital_transform,
)
from pyqed.mps.nonabelian.states import (
    build_random_reduced_spatial_mps,
    spatial_target_sector,
)
from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRG
from pyqed.qchem.dmrg.dmrg import _fully_reduced_spatial_mps_to_component_mps
from pyqed.qchem.dmrg.overlap import su2_biorthogonal_overlap
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI


def _component_tensor(state):
    component = _fully_reduced_spatial_mps_to_component_mps(state)
    return np.asarray(tt_to_tensor(component.factors))


def test_compiled_orbital_channel_block_mix_matches_numpy():
    try:
        from pyqed.mps.nonabelian import _su2_kernel
    except ImportError:
        pytest.skip("optional SU(2) extension is not installed")
    mixer = getattr(_su2_kernel, "mix_orbital_channel_blocks", None)
    if mixer is None:
        pytest.skip("installed SU(2) extension predates the channel-mix kernel")

    rng = np.random.default_rng(91)
    coefficients = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    blocks = rng.normal(size=(4, 7, 6)) + 1j * rng.normal(size=(4, 7, 6))
    expected = np.tensordot(coefficients, blocks, axes=([1], [0]))

    np.testing.assert_allclose(mixer(coefficients, blocks), expected, atol=2.0e-14)


def test_compiled_orbital_channel_gate_matches_numpy():
    try:
        from pyqed.mps.nonabelian import _su2_kernel
    except ImportError:
        pytest.skip("optional SU(2) extension is not installed")
    kernel = getattr(_su2_kernel, "apply_orbital_channel_gate", None)
    if kernel is None:
        pytest.skip("installed SU(2) extension predates the orbital-gate kernel")

    rng = np.random.default_rng(93)
    gate = rng.normal(size=(4, 4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4, 4))
    projection = rng.normal(size=(5, 3, 4, 4, 4, 4))
    blocks = rng.normal(size=(3, 7, 6)) + 1j * rng.normal(size=(3, 7, 6))
    coefficients = np.einsum(
        "oixyab,xyab->oi",
        projection,
        gate,
        optimize=True,
    )
    expected = np.tensordot(coefficients, blocks, axes=([1], [0]))

    actual, actual_coefficients = kernel(
        gate,
        projection,
        blocks,
        0.0,
    )
    np.testing.assert_allclose(actual_coefficients, coefficients, atol=2.0e-13)
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)


def _apply_dense_orbital_circuit(tensor, circuit):
    tensor = np.asarray(tensor, dtype=complex)
    nsites = tensor.ndim
    for kind, *payload in circuit:
        if kind == "diagonal":
            for site, value in enumerate(payload[0]):
                local = np.array([1.0, value, value, value * value])
                shape = [1] * nsites
                shape[site] = 4
                tensor = tensor * local.reshape(shape)
            continue
        bond, gate = payload
        fock_gate = _second_quantized_two_orbital_gate(gate)
        tensor = np.tensordot(
            fock_gate,
            tensor,
            axes=([2, 3], [bond, bond + 1]),
        )
        tensor = np.moveaxis(tensor, [0, 1], [bond, bond + 1])
    return tensor


def test_reduced_su2_orbital_circuit_matches_component_reference():
    state = ReducedMPS.from_tensors(
        build_random_reduced_spatial_mps(
            4,
            target_sector=spatial_target_sector(4, 0),
            bond_multiplicity=2,
            seed=7,
        )
    )
    transform = np.array(
        [
            [1.10, 0.10, -0.03, 0.04],
            [-0.05, 0.93, 0.12, 0.00],
            [0.02, -0.07, 1.03, 0.08],
            [0.00, 0.03, -0.04, 0.96],
        ],
        dtype=complex,
    )
    expected = _apply_dense_orbital_circuit(
        _component_tensor(state),
        _orbital_circuit(np.linalg.inv(transform)),
    )

    transformed, info = apply_spatial_orbital_transform(
        state,
        transform,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    np.testing.assert_allclose(_component_tensor(transformed), expected, atol=2.0e-13)
    assert info["exact"] is True
    assert info["adjacent_gate_count"] > 0
    assert info["determinant_expansion"] is False
    assert all(
        site.metadata.get("physical_basis") == "fully_reduced_su2"
        for site in transformed.sites
    )

    compressed, compressed_info = apply_spatial_orbital_transform(
        state,
        transform,
        cutoff=0.0,
        max_bond=2,
        return_info=True,
    )
    assert compressed_info["exact"] is False
    assert compressed_info["max_bond"] == 2
    assert compressed_info["sum_gate_discarded_weight"] > 0.0
    assert compressed_info["truncated_gate_count"] > 0
    assert all(len(site.qns[2]) <= 2 for site in compressed.sites[:-1])
    assert not np.allclose(_component_tensor(compressed), expected, atol=2.0e-13)

    adaptive, adaptive_info = apply_spatial_orbital_transform(
        state,
        transform,
        cutoff=0.0,
        max_bond="adaptive",
        discarded_weight_budget=0.2,
        adaptive_max_bond=64,
        return_info=True,
    )
    assert adaptive_info["adaptive"] is True
    assert adaptive_info["requested_max_bond"] == "adaptive"
    assert adaptive_info["max_bond"] == 64
    assert adaptive_info["adaptive_budget_satisfied"] is True
    assert adaptive_info["sum_gate_discarded_weight"] <= 0.2 + 1.0e-14
    assert adaptive_info["truncated_gate_count"] > 0
    assert len(adaptive_info["gate_discarded_weight_budgets"]) == adaptive_info[
        "adjacent_gate_count"
    ]
    assert max(adaptive_info["gate_kept_reduced_bonds"]) <= 64
    assert all(len(site.qns[2]) <= 64 for site in adaptive.sites[:-1])

    _, ceiling_info = apply_spatial_orbital_transform(
        state,
        transform,
        cutoff=0.0,
        max_bond="adaptive",
        discarded_weight_budget=0.0,
        adaptive_max_bond=info["input_reduced_bond_dimension"],
        return_info=True,
    )
    assert ceiling_info["adaptive_budget_satisfied"] is False
    assert ceiling_info["peak_reduced_bond_dimension"] <= info[
        "input_reduced_bond_dimension"
    ]
    assert ceiling_info["sum_gate_discarded_weight"] > 0.0


def test_unitary_orbital_map_uses_one_exact_givens_sweep():
    state = ReducedMPS.from_tensors(
        build_random_reduced_spatial_mps(
            4,
            target_sector=spatial_target_sector(4, 0),
            bond_multiplicity=2,
            seed=17,
        )
    )
    rotation = np.linalg.qr(np.random.default_rng(23).normal(size=(4, 4)))[0]
    circuit = _orbital_circuit(rotation)
    left, singular, right_h = np.linalg.svd(rotation, full_matrices=False)
    svd_reference = (
        _adjacent_unitary_circuit(right_h)
        + [("diagonal", singular.astype(complex))]
        + _adjacent_unitary_circuit(left)
    )
    expected = _apply_dense_orbital_circuit(
        _component_tensor(state),
        svd_reference,
    )

    transformed, info = apply_spatial_orbital_transform(
        state,
        rotation,
        inverse=False,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    np.testing.assert_allclose(_component_tensor(transformed), expected, atol=3.0e-13)
    assert sum(step[0] == "gate" for step in circuit) == 6
    assert info["adjacent_gate_count"] == 6
    assert info["orbital_factorization"] == "unitary_givens"
    assert info["unitarity_residual"] < 1.0e-12


def test_block_diagonal_orbital_circuit_matches_full_factorization():
    state = ReducedMPS.from_tensors(
        build_random_reduced_spatial_mps(
            4,
            target_sector=spatial_target_sector(4, 0),
            bond_multiplicity=2,
            seed=11,
        )
    )
    orbital_map = np.array(
        [
            [1.08, 0.13, 0.0, 0.0],
            [-0.04, 0.95, 0.0, 0.0],
            [0.0, 0.0, 1.03, -0.09],
            [0.0, 0.0, 0.06, 0.91],
        ],
        dtype=complex,
    )

    full = apply_spatial_orbital_transform(
        state,
        orbital_map,
        inverse=False,
        cutoff=0.0,
        max_bond=None,
    )
    blocked, info = apply_spatial_orbital_transform(
        state,
        orbital_map,
        inverse=False,
        orbital_blocks=[(0, 1), (2, 3)],
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )

    np.testing.assert_allclose(
        _component_tensor(blocked),
        _component_tensor(full),
        atol=3.0e-13,
    )
    assert info["orbital_block_count"] == 2
    assert info["adjacent_gate_count"] == 4
    assert info["orbital_factorization"] == "svd_givens"


def _run_h2(atom):
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mf = RHF(mol).run()
    casci = CASCI(mf, ncas=2, nelecas=2).run(nstates=1)
    dmrg = DMRG(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )
    dmrg.run(
        nstates=1,
        nsweeps=3,
        mixer_zero_block_noise_scale=0.0,
        require_convergence=False,
    )
    return casci, dmrg


def test_cross_geometry_su2_overlap_is_exact_and_determinant_free(monkeypatch):
    cas_bra, dmrg_bra = _run_h2("H 0 0 0; H 0 0 1.4")
    cas_ket, dmrg_ket = _run_h2("H 0 0 0; H 0 0 1.5")

    import importlib

    overlap_module = importlib.import_module("pyqed.qchem.dmrg.overlap")

    def reject_determinants(*_args, **_kwargs):
        raise AssertionError("SU(2) overlap recovered determinant amplitudes")

    monkeypatch.setattr(
        overlap_module,
        "_dmrg_coefficients_in_cas_basis",
        reject_determinants,
    )
    actual, info = su2_biorthogonal_overlap(
        dmrg_bra,
        dmrg_ket,
        return_info=True,
    )

    np.testing.assert_allclose(actual, cas_bra.overlap(cas_ket), atol=1.0e-8)
    np.testing.assert_allclose(dmrg_bra.overlap(dmrg_bra), [[1.0]], atol=1.0e-10)
    assert info["backend"] == "su2"
    assert info["exact"] is False
    assert info["transforms"]["bra"][0]["cutoff"] == 1.0e-10
    assert info["transforms"]["bra"][0]["requested_max_bond"] == "auto"
    transform_info = info["transforms"]["bra"][0]
    input_bond = transform_info["input_reduced_bond_dimension"]
    assert transform_info["max_bond"] == max(
        input_bond,
        min(8192, max(256, 16 * input_bond)),
    )
    assert transform_info["sum_gate_discarded_weight"] >= 0.0
    assert info["sector_preserving"] is True
    assert info["determinant_expansion"] is False
    assert info["component_expansion"] is False
