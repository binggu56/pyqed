from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.units import amu_to_au, au2ev
from pyqed.pbc import PeriodicPhononMode
from pyqed.pbc.gw import (
    FULL_EWALD,
    ExcitonPhononChannel,
    ExcitonPhononContinuum,
    ExcitonPhononCoupling,
    PeriodicTDAElectronPhononDerivative,
    PeriodicBSEHaydockResult,
    PeriodicBSEOpticalResult,
    PeriodicTDAOperator,
    ProjectedTDAContinuum,
    TotalMomentumSector,
    DiagonalSelfEnergyCache,
    GammaPBCSCFAdapter,
    GDF,
    attach_pyscf_gdf_context,
    analytic_tda_electron_phonon_coupling,
    commensurate_tda_electron_phonon_coupling,
    gdf_mo_jk,
    gdf_orbital_pair_coupling,
    gdf_transition_factors,
    gdf_transition_metric,
    gamma_tda_electron_phonon_coupling,
    KBSE,
    KGW,
    KPointSCFAdapter,
    KPointTransitionSpace,
    KTDA,
    PYSCF_GDF,
    RECIPROCAL_EWALD_LR,
    SHORT_RANGE_EWALD,
    build_transition_space,
    dense_gamma_orbital_pair_coupling,
    dense_gamma_orbital_pair_metric,
    dense_gamma_transition_metric,
    diagonal_correlation_self_energy,
    diagonal_evgw,
    diagonal_finite_size_correction,
    diagonal_g0w0,
    direct_rpa,
    direct_tdh_matrices,
    full_ewald_orbital_pair_coupling,
    full_ewald_orbital_pair_metric,
    full_ewald_transition_metric,
    normalize_coulomb_component,
    periodic_bse,
    periodic_bse_absorption,
    periodic_bse_matrices,
    periodic_bse_spectrum,
    periodic_photoemission_spectrum,
    periodic_plane_wave_velocity_matrix_elements,
    periodic_spectral_function,
    periodic_tda,
    periodic_tda_operator,
    periodic_tda_spectrum,
    periodic_transition_velocity_matrix_elements,
    phonon_tda_electron_phonon_coupling,
    prebuild_gdf_q_ao_stores,
    pyscf_gdf_orbital_pair_coupling,
    pyscf_gdf_transition_factors,
    pyscf_gdf_transition_metric,
    reciprocal_orbital_pair_factors,
    reciprocal_transition_factors,
    screened_interaction_poles,
)
from pyqed.pbc.gw.self_energy import (
    _accumulate_pole_self_energy,
    _accumulate_pole_self_energy_python,
    _scaled_legendre_roots,
)
from pyqed.pbc.gw.integrals import (
    _gdf_gaussian_ft_batch,
    _gdf_image_keys,
    _gdf_is_reciprocal_zero,
    _gdf_metric_invsqrt,
    _pyscf_cell_from_reference,
)
from pyqed.qchem.basis import (
    ContractedGaussian,
    _basis_cy,
    _basis_path,
    _cart_shell_blocks,
    _pack_signatures_for_numba,
    make_contractions,
    parse_gbs,
    three_center_eri,
    two_center_coulomb,
)
from pyqed.qchem.fourier import (
    gaussian_basis_ft_batch,
    has_periodic_pair_ft_image_group_backend,
    has_periodic_pair_ft_many_backend,
    has_periodic_pair_ft_product_backend,
)
from pyqed.qchem.pbc import Cell
from pyqed.qchem.pbc.ewald import (
    _basis_fn_signature,
    short_range_three_center_eri,
    short_range_two_center_coulomb,
)
from pyqed.qchem.pbc.hf.ewald_rhf import _shifted_gaussian


def test_documented_pbc_h2_gw_bse_workflow_runs(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "pbc_h2_gw_bse.json"
    figure = tmp_path / "pbc_h2_gw_bse.pdf"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "pbc_h2_gw_bse.py"),
            "--nroots",
            "1",
            "--output",
            str(output),
            "--figure",
            str(figure),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(output.read_text())

    assert payload["krhf_converged"]
    assert payload["gw_converged"]
    assert payload["resolved_bse_roots"] == 1
    assert len(payload["tda_energy_Ha"]) == 1
    assert len(payload["bse_energy_Ha"]) == 1
    assert payload["gdf_factor_memory_bytes"] > 0
    assert payload["gw_cache_sizes"]["screened_interactions"] > 0
    assert figure.exists()
    assert figure.with_suffix(".png").exists()
    assert "Fundamental gap" in completed.stdout


def test_documented_pbc_lih_gw_pes_workflow_runs(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "pbc_lih_gw_pes.json"
    figure = tmp_path / "pbc_lih_gw_pes.pdf"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "pbc_lih_gw_pes.py"),
            "--precision",
            "1e-6",
            "--npoints",
            "81",
            "--output",
            str(output),
            "--figure",
            str(figure),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    payload = json.loads(output.read_text())
    data = np.load(output.with_suffix(".npz"))

    assert payload["system"] == "rocksalt LiH"
    assert payload["kmesh"] == [1, 1, 1]
    assert payload["krhf_converged"]
    assert payload["gw_converged"]
    assert payload["gdf_factor_memory_bytes"] > 0
    assert payload["gdf_stream_pair_batch_mb"] == 128.0
    assert payload["gdf_stream_pair_batch_size"] is None
    assert payload["gdf_build_timings"]["0"][
        "stream_pair_batch_pair_counts"
    ] == [1]
    assert payload["gw_cache_sizes"]["screened_interactions"] == 1
    assert len(payload["spectral_targets"]) > 0
    np.testing.assert_allclose(payload["binding_range_eV"], [0.0, 10.0])
    assert output.with_suffix(".csv").exists()
    assert figure.exists()
    assert figure.with_suffix(".png").exists()
    assert data["binding_energy_eV"].shape == (81,)
    assert data["detector_photoemission_signal"].shape == (81,)
    assert np.all(np.isfinite(data["detector_photoemission_signal"]))
    assert float(np.max(data["detector_photoemission_signal"])) > 0.0
    assert "PES peaks in 0-10 eV" in completed.stdout


def _assert_primitive_term_image_groups(primitive_terms, npair):
    pair_starts = np.asarray(primitive_terms["pair_term_starts"], dtype=np.int64)
    term_image = np.asarray(primitive_terms["term_image"], dtype=np.int64)
    group_starts = np.asarray(
        primitive_terms["pair_image_group_starts"],
        dtype=np.int64,
    )
    group_images = np.asarray(primitive_terms["image_group_image"], dtype=np.int64)
    group_term_start = np.asarray(
        primitive_terms["image_group_term_start"],
        dtype=np.int64,
    )
    group_term_stop = np.asarray(
        primitive_terms["image_group_term_stop"],
        dtype=np.int64,
    )
    group_product_start = np.asarray(
        primitive_terms["image_group_product_start"],
        dtype=np.int64,
    )
    group_product_stop = np.asarray(
        primitive_terms["image_group_product_stop"],
        dtype=np.int64,
    )
    product_term_start = np.asarray(
        primitive_terms["product_group_term_start"],
        dtype=np.int64,
    )
    product_term_stop = np.asarray(
        primitive_terms["product_group_term_stop"],
        dtype=np.int64,
    )
    product_factor_id = np.asarray(
        primitive_terms["product_group_factor_id"],
        dtype=np.int64,
    )
    product_factor_count = int(primitive_terms["product_group_factor_count"])

    assert pair_starts.shape == (npair + 1,)
    assert group_starts.shape == (npair + 1,)
    assert group_images.shape == group_term_start.shape == group_term_stop.shape
    assert group_product_start.shape == group_product_stop.shape == group_images.shape
    assert product_term_start.shape == product_term_stop.shape
    assert product_factor_id.shape == product_term_start.shape
    if len(product_factor_id):
        assert int(np.min(product_factor_id)) >= 0
        assert int(np.max(product_factor_id)) < product_factor_count
    else:
        assert product_factor_count == 0
    for pair_idx in range(npair):
        expected = []
        term = int(pair_starts[pair_idx])
        term_stop = int(pair_starts[pair_idx + 1])
        while term < term_stop:
            image = int(term_image[term])
            stop = term + 1
            while stop < term_stop and int(term_image[stop]) == image:
                stop += 1
            expected.append((image, term, stop))
            term = stop

        group_begin = int(group_starts[pair_idx])
        group_end = int(group_starts[pair_idx + 1])
        actual = [
            (
                int(group_images[group]),
                int(group_term_start[group]),
                int(group_term_stop[group]),
            )
            for group in range(group_begin, group_end)
        ]
        assert actual == expected
        for group in range(group_begin, group_end):
            p_begin = int(group_product_start[group])
            p_end = int(group_product_stop[group])
            assert 0 <= p_begin <= p_end <= len(product_term_start)
            assert (
                int(product_term_start[p_begin])
                if p_begin < p_end
                else int(group_term_start[group])
            ) == int(group_term_start[group])
            assert (
                int(product_term_stop[p_end - 1])
                if p_begin < p_end
                else int(group_term_stop[group])
            ) == int(group_term_stop[group])
            for product in range(p_begin, p_end):
                assert int(product_term_start[product]) < int(product_term_stop[product])
                assert int(group_term_start[group]) <= int(product_term_start[product])
                assert int(product_term_stop[product]) <= int(group_term_stop[group])
                if product + 1 < p_end:
                    assert int(product_term_stop[product]) == int(
                        product_term_start[product + 1]
                    )


@pytest.fixture(scope="module")
def gamma_h2_mf():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    return cell.KRHF(
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="ewald",
    ).run(max_cycle=80, conv_tol=1e-10, conv_tol_dm=1e-8)


@pytest.fixture()
def two_k_h2_reference():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="ewald",
    )
    mf.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.8, 0.7]),
    ]
    mf.mo_coeff = [np.eye(cell.nao), np.eye(cell.nao)]
    mf.mo_occ = [
        np.asarray([2.0, 0.0]),
        np.asarray([2.0, 0.0]),
    ]
    mf.dm = [
        np.diag([2.0, 0.0]),
        np.diag([2.0, 0.0]),
    ]
    return mf


@pytest.fixture()
def four_band_two_k_reference(two_k_h2_reference):
    mf = two_k_h2_reference
    coeff = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    mf.mo_energy = [
        np.asarray([-1.2, -0.9, 0.4, 0.8]),
        np.asarray([-1.1, -0.7, 0.5, 0.9]),
    ]
    mf.mo_coeff = [coeff.copy(), coeff.copy()]
    mf.mo_occ = [
        np.asarray([2.0, 2.0, 0.0, 0.0]),
        np.asarray([2.0, 2.0, 0.0, 0.0]),
    ]
    mf.dm = [
        np.diag([4.0, 0.0]).astype(np.complex128),
        np.diag([4.0, 0.0]).astype(np.complex128),
    ]
    return mf


@pytest.fixture(scope="module")
def real_two_k_h2_mf():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    return cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=1,
        jk_builder="ewald",
    ).run(max_cycle=60, conv_tol=1e-9, conv_tol_dm=1e-7)


def test_pbc_gw_exports_public_gamma_classes():
    assert KGW.__name__ == "KGW"
    assert KBSE.__name__ == "KBSE"
    assert KTDA.__name__ == "KTDA"


def test_coulomb_component_normalization_aliases():
    assert normalize_coulomb_component("full") == FULL_EWALD
    assert normalize_coulomb_component("full_ewald") == FULL_EWALD
    assert normalize_coulomb_component("reciprocal") == RECIPROCAL_EWALD_LR
    assert normalize_coulomb_component("lr") == RECIPROCAL_EWALD_LR
    assert normalize_coulomb_component("gdf") == GDF
    assert normalize_coulomb_component("density_fit") == GDF
    assert normalize_coulomb_component("pyscf_gdf") == PYSCF_GDF
    assert normalize_coulomb_component("sr", dense_gamma=True) == SHORT_RANGE_EWALD

    with pytest.raises(ValueError, match="coulomb_component"):
        normalize_coulomb_component("short_range_ewald")
    with pytest.raises(ValueError, match="coulomb_component"):
        normalize_coulomb_component("gdf", dense_gamma=True)


def test_coulomb_component_aliases_record_canonical_metadata(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")

    response = direct_tdh_matrices(space, q_index=0, coulomb_component="full")
    sigma = diagonal_g0w0(space, coulomb_component="full")
    tda = periodic_tda(
        space,
        q_index=0,
        coulomb_component="full",
        screened_exchange_scale=0.0,
        nroots=1,
    )

    assert response.coulomb_component == FULL_EWALD
    assert sigma.info["coulomb_component"] == FULL_EWALD
    assert tda.block.coulomb_component == FULL_EWALD
    assert tda.info["coulomb_component"] == FULL_EWALD


def test_pbc_gw_legacy_import_path_reexports_public_classes():
    from setuptools import find_packages

    from pyqed.gw import BSE as PackageBSE
    from pyqed.gw import GW as PackageGW
    from pyqed.gw import TDA as PackageTDA
    from pyqed.gw.bse import BSE as MolecularBSE
    from pyqed.gw.bse import TDA as MolecularTDA
    from pyqed.gw.gw import GW as MolecularGW
    from pyqed.gw.pbc import KBSE as LegacyKBSE
    from pyqed.gw.pbc import KGW as LegacyKGW
    from pyqed.gw.pbc import KTDA as LegacyKTDA
    from pyqed.gw.pbc.coulomb import FULL_EWALD as LegacyFullEwald
    from pyqed.gw.pbc.response import KPointTransitionSpace as LegacyTransitionSpace

    assert PackageGW is MolecularGW
    assert PackageBSE is MolecularBSE
    assert PackageTDA is MolecularTDA
    assert LegacyKGW is KGW
    assert LegacyKBSE is KBSE
    assert LegacyKTDA is KTDA
    assert LegacyTransitionSpace is KPointTransitionSpace
    assert LegacyFullEwald == FULL_EWALD
    packages = set(find_packages())
    assert "pyqed.gw" in packages
    assert "pyqed.pbc.gw" in packages
    assert "pyqed.gw.pbc" in packages


def test_gamma_pbc_scf_adapter_exposes_molecular_rhf_hooks(gamma_h2_mf):
    adapter = GammaPBCSCFAdapter(gamma_h2_mf)

    assert adapter.mol.nelectron == gamma_h2_mf.cell.nelectron
    assert adapter.mo_energy.shape == (gamma_h2_mf.cell.nao,)
    assert adapter.mo_coeff.shape == (gamma_h2_mf.cell.nao, gamma_h2_mf.cell.nao)
    assert adapter.eri.shape == (gamma_h2_mf.cell.nao,) * 4

    vj, vk = adapter.get_jk()
    np.testing.assert_allclose(adapter.get_k(), vk, atol=1e-12)
    np.testing.assert_allclose(adapter.get_j(), vj, atol=1e-12)
    np.testing.assert_allclose(adapter.get_veff(), vj - 0.5 * vk, atol=1e-12)
    assert adapter.get_eri_mo().shape == (gamma_h2_mf.cell.nao,) * 4


def test_gamma_pbc_adapter_rejects_true_kpoint_reference():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="ewald",
    )
    mf.mo_energy = [np.zeros(cell.nao), np.zeros(cell.nao)]
    mf.mo_coeff = [np.eye(cell.nao), np.eye(cell.nao)]
    mf.mo_occ = [np.asarray([2.0, 0.0]), np.asarray([0.0, 0.0])]
    mf.dm = [np.eye(cell.nao), np.zeros((cell.nao, cell.nao))]

    with pytest.raises(NotImplementedError, match="Gamma-only"):
        GammaPBCSCFAdapter(mf)


def test_gamma_centered_mesh_contains_gamma_and_is_closed():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 6.0, 7.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 3, 2), gamma_centered=True)
    gamma = np.flatnonzero(np.linalg.norm(kpts, axis=1) < 1.0e-12)
    np.testing.assert_array_equal(gamma, [0])
    shifted = cell.make_kpts((2, 3, 2))
    assert np.min(np.linalg.norm(shifted, axis=1)) > 1.0e-12

    mf = SimpleNamespace(
        cell=cell,
        kpts=kpts,
        mo_energy=[np.asarray([-0.5, 0.5]) for _ in kpts],
        mo_coeff=[np.eye(cell.nao) for _ in kpts],
        mo_occ=[np.asarray([2.0, 0.0]) for _ in kpts],
    )
    ref = KPointSCFAdapter(mf)
    qpts = ref.qpoint_mesh()
    assert len(qpts) == len(kpts)
    for kpt in kpts:
        for qpt in qpts:
            ref.find_kpoint_index(kpt - qpt)


def test_kpoint_scf_adapter_normalizes_multi_k_reference(two_k_h2_reference):
    ref = KPointSCFAdapter(two_k_h2_reference)

    assert ref.nkpts == 2
    assert ref.nband == 2
    assert ref.mo_energy.shape == (2, 2)
    assert ref.mo_coeff.shape == (2, 2, 2)
    assert ref.mo_occ.shape == (2, 2)
    np.testing.assert_array_equal(ref.occupied_bands(0), [0])
    np.testing.assert_array_equal(ref.virtual_bands(1), [1])

    qpts = ref.qpoint_mesh()
    assert qpts.shape == (2, 3)
    np.testing.assert_allclose(qpts[0], np.zeros(3), atol=1e-12)
    assert ref.find_kpoint_index(ref.kpts[0] + qpts[0]) == 0
    assert ref.find_kpoint_index(ref.kpts[0] + qpts[1]) == 1
    assert ref.normalize_k_index(1) == 1

    with pytest.raises(TypeError, match="k_index"):
        ref.occupied_bands(0.5)
    with pytest.raises(IndexError, match="k_index"):
        ref.occupied_bands(-1)
    with pytest.raises(IndexError, match="k_index"):
        ref.virtual_bands(ref.nkpts)
    with pytest.raises(ValueError, match="occupation_tol"):
        KPointSCFAdapter(two_k_h2_reference, occupation_tol=-1.0e-3)
    with pytest.raises(ValueError, match="occupation_tol"):
        KPointSCFAdapter(two_k_h2_reference, occupation_tol=1.0)


def test_kpoint_transition_space_builds_momentum_conserving_transitions(two_k_h2_reference):
    ref = KPointSCFAdapter(two_k_h2_reference)
    space = KPointTransitionSpace(ref, qpts="mesh")

    assert space.nqpts == 2
    np.testing.assert_array_equal(space.ntransitions_by_q, [2, 2])
    assert space.ntransitions == 4

    q0 = space.as_table(0)
    np.testing.assert_array_equal(q0["k"], [0, 1])
    np.testing.assert_array_equal(q0["kq"], [0, 1])
    np.testing.assert_allclose(q0["energy"], [1.5, 1.5])
    packed = space.transition_indices(0)
    np.testing.assert_array_equal(packed["q"], q0["q"])
    np.testing.assert_array_equal(packed["k"], q0["k"])
    np.testing.assert_array_equal(packed["kq"], q0["kq"])
    np.testing.assert_array_equal(packed["occ"], q0["occ"])
    np.testing.assert_array_equal(packed["vir"], q0["vir"])
    np.testing.assert_allclose(packed["energy"], q0["energy"])

    q1 = build_transition_space(ref, qpts=[space.qpts[1]]).as_table(0)
    np.testing.assert_array_equal(q1["k"], [0, 1])
    np.testing.assert_array_equal(q1["kq"], [1, 0])
    np.testing.assert_allclose(q1["energy"], [1.7, 1.3])


def test_periodic_q_index_requests_are_validated(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    with pytest.raises(IndexError, match="q_index"):
        space.transitions(-1)
    with pytest.raises(IndexError, match="q_index"):
        space.energies(space.nqpts)
    with pytest.raises(TypeError, match="q_index"):
        space.as_table(0.0)
    with pytest.raises(IndexError, match="q_index"):
        direct_tdh_matrices(space, q_index=-1)
    with pytest.raises(IndexError, match="q_index"):
        reciprocal_transition_factors(space, q_index=-1)
    with pytest.raises(IndexError, match="q_index"):
        full_ewald_transition_metric(space, q_index=-1)
    with pytest.raises(IndexError, match="q_index"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1,
            omega=0.5,
            q_indices=[-1],
        )
    with pytest.raises(IndexError, match="q_index"):
        periodic_tda(space, q_index=-1, direct_scale=1.0, nroots=1)
    with pytest.raises(IndexError, match="q_index"):
        periodic_tda_spectrum(
            space,
            q_indices=[0, -1],
            direct_scale=1.0,
            nroots=1,
            return_vectors=False,
        )


def test_periodic_orbital_pair_indices_are_validated(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    with pytest.raises(TypeError, match="k_index"):
        reciprocal_orbital_pair_factors(
            space,
            q_index=0,
            k_index=0.5,
            left_band=0,
            right_band=1,
        )
    with pytest.raises(TypeError, match="left_band"):
        reciprocal_orbital_pair_factors(
            space,
            q_index=0,
            k_index=0,
            left_band=0.5,
            right_band=1,
        )
    with pytest.raises(IndexError, match="right_band"):
        reciprocal_orbital_pair_factors(
            space,
            q_index=0,
            k_index=0,
            left_band=0,
            right_band=99,
        )
    with pytest.raises(TypeError, match="left_pair"):
        full_ewald_orbital_pair_metric(
            space,
            q_index=0,
            left_pair=(0, 0, 0.5, 1),
            right_pair=(0, 0, 0, 1),
        )
    with pytest.raises(ValueError, match="left_pair"):
        full_ewald_orbital_pair_metric(
            space,
            q_index=0,
            left_pair=(0, 0, 0),
            right_pair=(0, 0, 0, 1),
        )
    with pytest.raises(TypeError, match="k_index"):
        diagonal_correlation_self_energy(
            space,
            k_index=0.5,
            band_index=1,
            omega=0.5,
        )
    with pytest.raises(TypeError, match="band_index"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1.5,
            omega=0.5,
        )


def test_transition_space_can_limit_occ_and_virtual_band_windows(four_band_two_k_reference):
    ref = KPointSCFAdapter(four_band_two_k_reference)

    full = KPointTransitionSpace(ref, qpts="mesh")
    np.testing.assert_array_equal(full.ntransitions_by_q, [8, 8])

    limited = KPointTransitionSpace(
        ref,
        qpts="mesh",
        occ_bands=[1],
        vir_bands=[2],
    )
    np.testing.assert_array_equal(limited.ntransitions_by_q, [2, 2])
    q0 = limited.as_table(0)
    np.testing.assert_array_equal(q0["k"], [0, 1])
    np.testing.assert_array_equal(q0["kq"], [0, 1])
    np.testing.assert_array_equal(q0["occ"], [1, 1])
    np.testing.assert_array_equal(q0["vir"], [2, 2])
    assert limited.with_mo_energy(ref.mo_energy).occ_bands == (1,)

    per_k = KPointTransitionSpace(
        ref,
        qpts="mesh",
        occ_bands={0: [1], 1: [0]},
        vir_bands={0: [3], 1: [2]},
    )
    q1 = per_k.as_table(1)
    np.testing.assert_array_equal(q1["k"], [0, 1])
    np.testing.assert_array_equal(q1["kq"], [1, 0])
    np.testing.assert_array_equal(q1["occ"], [1, 0])
    np.testing.assert_array_equal(q1["vir"], [2, 3])

    with pytest.raises(ValueError, match="not occupied"):
        KPointTransitionSpace(ref, qpts="mesh", occ_bands=[2])
    with pytest.raises(ValueError, match="not virtual"):
        KPointTransitionSpace(ref, qpts="mesh", vir_bands=[1])
    with pytest.raises(TypeError, match="occ_bands"):
        KPointTransitionSpace(ref, qpts="mesh", occ_bands=[1.5])
    with pytest.raises(TypeError, match="vir_bands"):
        KPointTransitionSpace(ref, qpts="mesh", vir_bands=[2.5])
    with pytest.raises(TypeError, match="occ_bands"):
        KPointTransitionSpace(ref, qpts="mesh", occ_bands={0.5: [1]})
    with pytest.raises(IndexError, match="out-of-range band"):
        KPointTransitionSpace(ref, qpts="mesh", occ_bands=[99])


def test_transition_space_with_mo_energy_updates_gaps(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    updated_energy = np.asarray(two_k_h2_reference.mo_energy, dtype=float).copy()
    updated_energy[:, 1] += [0.2, 0.4]

    updated = space.with_mo_energy(updated_energy)

    np.testing.assert_allclose(space.energies(0), [1.5, 1.5])
    np.testing.assert_allclose(updated.energies(0), [1.7, 1.9])
    np.testing.assert_allclose(updated.energies(1), [2.1, 1.5])
    np.testing.assert_allclose(updated.transition_indices(0)["energy"], [1.7, 1.9])
    assert updated.reference is space.reference
    np.testing.assert_allclose(space.energies(0), [1.5, 1.5])


def test_multi_k_kgw_runs_diagonal_direct_rpa_g0w0(two_k_h2_reference):
    gw = KGW(two_k_h2_reference)
    space = gw.transition_space(qpts="mesh")

    assert gw.info["nkpts"] == 2
    assert space.nqpts == 2
    gw.run(direct_scale=1.0)

    assert gw.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert gw.info["converged"]
    assert gw.info["eta"] == gw.eta
    assert gw.info["direct_scale"] == 1.0
    assert gw.info["g2_tol"] == 1.0e-16
    assert gw.info["thresh"] == 1.0e-10
    assert gw.info["finite_size_correction"] is False
    np.testing.assert_array_equal(gw.info["q_indices"], [0, 1])
    assert gw.e_qp.shape == (2, 2)
    assert np.all(np.isfinite(gw.e_qp))


def test_gamma_periodic_kgw_does_not_require_dense_molecular_eri():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(1, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="reciprocal",
    )
    mf._validate()
    mf._periodic_setup()
    mf.mo_energy = np.asarray([-1.0, 0.5])
    mf.mo_coeff = np.eye(cell.nao, dtype=np.complex128)
    mf.mo_occ = np.asarray([2.0, 0.0])
    mf.dm = np.diag([2.0, 0.0]).astype(np.complex128)

    assert mf.eri is None
    gw = KGW(mf)
    gw.run(direct_scale=1.0, qp_bands=[1])

    assert gw.periodic_backend
    assert gw.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert gw.e_qp.shape == (1, 2)
    assert np.isfinite(gw.e_qp[0, 1])


def test_multi_k_kgw_can_target_qp_bands(two_k_h2_reference):
    gw = KGW(two_k_h2_reference)
    gw.run(direct_scale=1.0, qp_bands=[1])

    assert gw.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert gw.info["qp_bands"] == (1,)
    assert gw.info["nqp"] == 2
    assert gw.info["converged"]
    np.testing.assert_allclose(gw.e_qp[:, 0], np.asarray(two_k_h2_reference.mo_energy)[:, 0])
    assert np.all(np.isfinite(gw.e_qp[:, 1]))
    assert not np.allclose(gw.e_qp[:, 1], np.asarray(two_k_h2_reference.mo_energy)[:, 1])


def test_gamma_transition_reciprocal_factors_are_finite(gamma_h2_mf):
    gw = KGW(gamma_h2_mf)
    space = gw.transition_space(qpts="gamma")
    factors = space.reciprocal_factors(0)

    assert factors.q_index == 0
    assert factors.coulomb_component == "reciprocal_ewald_lr"
    assert factors.g2_tol == 1.0e-16
    assert factors.ntransitions == 1
    assert factors.ngvectors > 0
    assert factors.gvecs.shape == factors.gqvecs.shape
    assert factors.pair_density.shape == (1, factors.ngvectors)
    assert factors.weighted_pair_density.shape == factors.pair_density.shape
    assert np.all(np.isfinite(factors.coulomb_weights))
    assert np.all(factors.coulomb_weights > 0.0)
    assert np.all(np.isfinite(factors.pair_density))

    metric = factors.coulomb_metric()
    assert metric.shape == (1, 1)
    np.testing.assert_allclose(metric, metric.conj().T, atol=1e-12)
    assert metric[0, 0].real >= 0.0
    with pytest.raises(ValueError, match="g2_tol"):
        space.reciprocal_factors(0, g2_tol=-1.0)


def test_gamma_reciprocal_factor_metric_matches_dense_reciprocal_eri(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    reciprocal_metric = space.reciprocal_factors(0).coulomb_metric()

    dense_reciprocal = dense_gamma_transition_metric(
        space,
        q_index=0,
        component="reciprocal_ewald_lr",
    )
    dense_short = dense_gamma_transition_metric(
        space,
        q_index=0,
        component="short_range_ewald",
    )
    dense_background = dense_gamma_transition_metric(
        space,
        q_index=0,
        component="background",
    )
    dense_full = dense_gamma_transition_metric(
        space,
        q_index=0,
        component="full_ewald",
    )

    np.testing.assert_allclose(reciprocal_metric, dense_reciprocal, atol=1e-12)
    np.testing.assert_allclose(
        dense_full,
        dense_reciprocal + dense_short + dense_background,
        atol=1e-10,
    )
    assert dense_short[0, 0].real > 0.0
    assert dense_full[0, 0].real > reciprocal_metric[0, 0].real


def test_kgw_transition_factors_matches_standalone_helper(gamma_h2_mf):
    gw = KGW(gamma_h2_mf)
    space = gw.transition_space(qpts="gamma")
    from_space = reciprocal_transition_factors(space, 0)
    from_gw = gw.transition_factors(q_index=0, qpts="gamma")

    np.testing.assert_allclose(
        from_gw.weighted_pair_density,
        from_space.weighted_pair_density,
        atol=1e-12,
    )


def test_gamma_dense_orbital_pair_helpers_match_reciprocal_factors(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    transition = space.transitions(0)[0]
    transition_factors = space.reciprocal_factors(0)
    pair_factors = reciprocal_orbital_pair_factors(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
    )
    mismatched_pair_factors = reciprocal_orbital_pair_factors(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
        g2_tol=1.0e9,
    )

    assert transition_factors.g2_tol == 1.0e-16
    assert pair_factors.g2_tol == 1.0e-16
    dense_coupling = dense_gamma_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
        component="reciprocal_ewald_lr",
    )
    dense_metric = dense_gamma_orbital_pair_metric(
        space,
        q_index=0,
        left_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
        right_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
        component="reciprocal_ewald_lr",
    )

    np.testing.assert_allclose(
        dense_coupling,
        pair_factors.coulomb_coupling(transition_factors),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense_metric,
        pair_factors.weighted_pair_density @ pair_factors.weighted_pair_density.conj(),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="G bases"):
        mismatched_pair_factors.coulomb_coupling(transition_factors)
    with pytest.raises(ValueError, match="g2_tol"):
        reciprocal_orbital_pair_factors(
            space,
            q_index=0,
            k_index=transition.k_index,
            kq_index=transition.kq_index,
            left_band=transition.occ_band,
            right_band=transition.vir_band,
            g2_tol=-1.0,
        )


def test_full_ewald_pair_helpers_match_dense_gamma_reference(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    transition = space.transitions(0)[0]
    dense_metric = dense_gamma_transition_metric(space, q_index=0, component="full_ewald")
    full_metric = full_ewald_transition_metric(space, q_index=0)
    dense_pair_metric = dense_gamma_orbital_pair_metric(
        space,
        q_index=0,
        left_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
        right_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
        component="full_ewald",
    )
    full_pair_metric = full_ewald_orbital_pair_metric(
        space,
        q_index=0,
        left_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
        right_pair=(
            transition.k_index,
            transition.kq_index,
            transition.occ_band,
            transition.vir_band,
        ),
    )
    full_pair_coupling = full_ewald_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
    )

    np.testing.assert_allclose(full_metric, dense_metric, atol=1e-12)
    np.testing.assert_allclose(full_pair_metric, dense_pair_metric, atol=1e-12)
    np.testing.assert_allclose(full_pair_coupling, dense_metric[:, 0], atol=1e-12)


def test_gamma_direct_tdh_response_matrices_are_hermitian(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    response = direct_tdh_matrices(space, q_index=0)

    assert response.coulomb_component == "reciprocal_ewald_lr"
    assert response.direct_scale == 2.0
    assert response.g2_tol == 1.0e-16
    assert response.thresh is None
    assert response.A.shape == (1, 1)
    assert response.B.shape == (1, 1)
    np.testing.assert_allclose(response.A, response.A.conj().T, atol=1e-12)
    np.testing.assert_allclose(response.B, response.B.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        response.A - response.B,
        np.diag(response.transition_energy),
        atol=1e-12,
    )
    assert response.A[0, 0].real >= response.transition_energy[0]


def test_gamma_direct_tdh_can_use_dense_full_ewald_metric(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    reciprocal = direct_tdh_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="reciprocal_ewald_lr",
    )
    full = direct_tdh_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )
    dense_full = dense_gamma_transition_metric(space, q_index=0, component="full_ewald")

    assert full.coulomb_component == "full_ewald"
    np.testing.assert_allclose(full.B, dense_full, atol=1e-12)
    np.testing.assert_allclose(
        full.A - full.B,
        np.diag(full.transition_energy),
        atol=1e-12,
    )
    assert full.B[0, 0].real > reciprocal.B[0, 0].real

    full_rpa = direct_rpa(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )
    full_poles = space.screened_interaction(
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )
    assert full_rpa.coulomb_component == "full_ewald"
    assert full_poles.coulomb_component == "full_ewald"
    np.testing.assert_allclose(full_poles.bare_coulomb, dense_full, atol=1e-12)
    assert full_rpa.omega[0] > reciprocal.transition_energy[0]


def test_full_ewald_direct_tdh_supports_multi_k_reference(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    metric = full_ewald_transition_metric(space, q_index=0)

    response = direct_tdh_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    assert response.coulomb_component == "full_ewald"
    assert metric.shape == response.B.shape == (2, 2)
    np.testing.assert_allclose(
        response.B,
        metric / two_k_h2_reference.nkpts,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        response.A - response.B,
        np.diag(response.transition_energy),
        atol=1e-12,
    )
    assert np.all(np.isfinite(response.A))
    np.testing.assert_allclose(response.B, response.B.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        response.A - response.B,
        np.diag(response.transition_energy),
        atol=1e-12,
    )


def test_gdf_direct_tdh_uses_auxiliary_basis_factors(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    factors = gdf_transition_factors(space, q_index=0)
    metric = gdf_transition_metric(space, q_index=0)
    response = direct_tdh_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="gdf",
    )
    coupling = gdf_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=0,
        kq_index=0,
        left_band=1,
        right_band=1,
    )
    transition = space.transitions(0)[0]
    transition_coupling = gdf_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
    )

    assert factors.coulomb_component == GDF
    assert factors.factor_method == "periodic_auxiliary_gdf:range_separated"
    assert factors.auxbasis
    assert factors.aux_coord_type == "spherical"
    assert factors.naux_cart >= len(factors.metric_eigenvalues)
    assert factors.metric_rank == factors.naux
    assert response.coulomb_component == GDF
    assert factors.naux <= len(factors.metric_eigenvalues)
    assert metric.shape == response.B.shape == (2, 2)
    assert coupling.shape == (2,)
    np.testing.assert_allclose(
        response.B,
        metric / two_k_h2_reference.nkpts,
        atol=1e-12,
    )
    np.testing.assert_allclose(transition_coupling, metric[:, 0], atol=1e-12)
    np.testing.assert_allclose(response.B, response.B.conj().T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(metric).real >= -1e-10)


def test_gdf_mo_pair_block_applies_complex_metric_invsqrt_hermitianly():
    from pyqed.pbc.gw.integrals import (
        _gdf_metric_invsqrt,
        _gdf_mo_pair_block,
        _gdf_mo_pair_blocks_many,
    )

    metric = np.asarray(
        [
            [2.0, 0.4 + 0.3j],
            [0.4 - 0.3j, 1.5],
        ],
        dtype=np.complex128,
    )
    pair = np.asarray([0.7 + 0.2j, -0.1 + 0.5j], dtype=np.complex128)
    metric_invsqrt, _evals = _gdf_metric_invsqrt(metric, 1.0e-14, "test-aux")
    block = _gdf_mo_pair_block(
        pair[:, None, None],
        metric_invsqrt,
        np.ones((1, 1), dtype=np.complex128),
        np.ones((1, 1), dtype=np.complex128),
    )

    actual = np.vdot(block[:, 0, 0], block[:, 0, 0])
    expected = pair.conj() @ np.linalg.solve(metric, pair)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    batch = _gdf_mo_pair_blocks_many(
        np.stack([pair[:, None, None], 2.0 * pair[:, None, None]], axis=0),
        metric_invsqrt,
        np.ones((2, 1, 1), dtype=np.complex128),
        np.ones((2, 1, 1), dtype=np.complex128),
    )
    np.testing.assert_allclose(batch[0], block, atol=1.0e-12)
    np.testing.assert_allclose(batch[1], 2.0 * block, atol=1.0e-12)


def test_gdf_metric_invsqrt_keeps_numerically_real_auxiliary_gauge():
    metric = np.asarray(
        [
            [2.0, 0.4 + 1.0e-15j],
            [0.4 - 1.0e-15j, 1.5],
        ],
        dtype=np.complex128,
    )

    metric_invsqrt, _evals = _gdf_metric_invsqrt(
        metric, 1.0e-14, "test-aux"
    )

    assert np.count_nonzero(metric_invsqrt.imag) == 0
    np.testing.assert_allclose(
        metric_invsqrt @ metric_invsqrt.T,
        np.linalg.inv(metric.real),
        atol=1.0e-12,
    )


def test_gdf_q_ao_store_reuses_blocks_when_mo_cache_is_cleared(two_k_h2_reference):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")
    pair_count = len(list(_pair_keys_for_q(space, 0)))

    first = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    assert first.build_timings["q_ao_store_cache_misses"] == 1
    assert first.build_timings["q_ao_store_requested_pair_blocks"] == pair_count
    assert (
        first.build_timings["q_ao_store_existing_pair_blocks"]
        + first.build_timings["q_ao_store_missing_pair_blocks"]
        == pair_count
    )
    assert first.build_timings["q_ao_store_pair_blocks"] == pair_count
    if first.build_timings.get("q_ao_store_shared_aux_ft", False):
        assert first.build_timings["q_ao_store_existing_pair_blocks"] == pair_count
        assert first.build_timings["q_ao_store_missing_pair_blocks"] == 0
    assert first.build_timings["mo_transform_batches"] == 1
    assert first.build_timings["mo_transform_batch_pairs"] == pair_count
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 1

    space._gdf_factor_cache = {}
    _gdf_mf_cache(mf, "mo_pair_block").clear()
    second = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert second.build_timings["q_ao_store_cache_hits"] == 1
    assert second.build_timings["q_ao_store_missing_pair_blocks"] == 0
    assert second.build_timings.get("three_center_ao_cache_misses", 0) == 0
    assert second.build_timings["mo_pair_block_cache_misses"] == pair_count
    np.testing.assert_allclose(
        second.coulomb_metric(),
        first.coulomb_metric(),
        atol=1.0e-12,
    )


def test_prebuild_gdf_q_ao_stores_warms_factor_build(two_k_h2_reference):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")

    summaries = prebuild_gdf_q_ao_stores(space, q_indices=[0], g2_tol=1.0e-14)
    assert len(summaries) == 1
    assert summaries[0]["q_index"] == 0
    assert summaries[0]["pair_blocks"] == len(list(_pair_keys_for_q(space, 0)))
    assert summaries[0]["timings"]["q_ao_store_cache_misses"] == 1
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 1

    space._gdf_factor_cache = {}
    _gdf_mf_cache(mf, "mo_pair_block").clear()
    factors = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert factors.build_timings["q_ao_store_cache_hits"] == 1
    assert factors.build_timings["q_ao_store_missing_pair_blocks"] == 0
    assert factors.build_timings.get("three_center_ao_cache_misses", 0) == 0
    assert factors.ntransitions > 0


def test_prebuild_gdf_q_ao_stores_can_materialize_mo_blocks(two_k_h2_reference):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")
    pair_count = len(list(_pair_keys_for_q(space, 0)))

    summaries = prebuild_gdf_q_ao_stores(
        space,
        q_indices=[0],
        g2_tol=1.0e-14,
        materialize_cderi=True,
    )

    assert summaries[0]["timings"]["cderi_materialized_pair_blocks"] == pair_count
    assert len(_gdf_mf_cache(mf, "mo_pair_block")) == pair_count

    space._gdf_factor_cache = {}
    factors = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert factors.build_timings["q_ao_store_cache_hits"] == 1
    assert factors.build_timings["mo_pair_block_cache_hits"] == pair_count
    assert factors.build_timings.get("mo_pair_block_cache_misses", 0) == 0


def test_prebuild_gdf_q_ao_stores_auto_weighted_aux_screen_streams(two_k_h2_reference):
    if not has_periodic_pair_ft_many_backend():
        pytest.skip("periodic pair-FT many backend unavailable")

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_precision = 1.0e-4
    mf.gdf_reciprocal_kernel = "full"
    mf.gdf_pair_ft_screen_tol = 0.0
    if hasattr(mf, "gdf_g_block_size"):
        del mf.gdf_g_block_size
    space = KPointTransitionSpace(mf, qpts="mesh")

    summaries = prebuild_gdf_q_ao_stores(
        space,
        q_indices=[0],
        g2_tol=1.0e-14,
    )
    timings = summaries[0]["timings"]

    assert timings["g_block_size"] > 0
    assert timings["q_ao_store_shared_aux_ft"] is True
    assert timings["weighted_aux_screen_tol"] == pytest.approx(5.0e-7)
    assert timings["weighted_aux_screen_input_g_vectors"] > 0
    assert (
        timings["weighted_aux_screen_kept_g_vectors"]
        + timings["weighted_aux_screen_skipped_g_vectors"]
        == timings["weighted_aux_screen_input_g_vectors"]
    )


def test_prebuild_gdf_q_ao_stores_can_use_workers(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")

    summaries = prebuild_gdf_q_ao_stores(space, g2_tol=1.0e-14, workers=2)

    assert len(summaries) == len(space.qpts)
    assert {row["timings"]["prebuild_workers"] for row in summaries} == {2}
    assert all(row["timings"]["prebuild_parallel"] for row in summaries)
    if has_periodic_pair_ft_many_backend():
        assert all(
            row["timings"]["prebuild_pair_ft_plan_prewarmed"]
            for row in summaries
        )
        assert all(
            row["timings"]["prebuild_hermitian_q0_pair_ft_plan_prewarmed"]
            for row in summaries
        )


def test_prebuild_gdf_q_ao_stores_shares_short_range_across_q(
    two_k_h2_reference,
):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_pair_cut = 0
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_short_range_workers = 2
    mf.gdf_pair_ft_workers = 2
    mf._periodic_setup()
    space = KPointTransitionSpace(mf, qpts="mesh")

    summaries = prebuild_gdf_q_ao_stores(
        space,
        g2_tol=1.0e-14,
        workers=2,
    )
    timings = [row["timings"] for row in summaries]

    assert len(summaries) == 2
    assert all(row["multi_q_short_range_prebuild"] for row in timings)
    assert all(row["multi_q_short_range_qpoints"] == 2 for row in timings)
    assert all(row["multi_q_short_range_inner_workers"] == 2 for row in timings)
    assert all(row["multi_q_short_range_outer_workers"] == 1 for row in timings)
    assert all(row["prebuild_requested_workers"] == 2 for row in timings)
    assert all(row["prebuild_workers"] == 1 for row in timings)
    assert all(row["prebuild_parallel"] is False for row in timings)
    shared = [row["multi_q_short_range_shared_seconds"] for row in timings]
    assert sum(value > 0.0 for value in shared) == 1
    owner = timings[int(np.argmax(shared))]
    assert owner["multi_q_short_range_shared_timings"][
        "three_center_sr_bloch_batch_qpoints"
    ] == 2
    assert owner["multi_q_short_range_shared_timings"][
        "aux_metric_short_range_qpoints"
    ] == 2
    assert owner["multi_q_short_range_shared_timings"][
        "unconsumed_pair_blocks"
    ] == 0
    assert owner["multi_q_short_range_shared_timings"][
        "unconsumed_metric_blocks"
    ] == 0
    assert owner["multi_q_short_range_shared_timings"][
        "stream_pair_workspace_within_budget"
    ] is True
    assert all(row["aux_metric_sr_cache_consumes"] == 1 for row in timings)
    assert not _gdf_mf_cache(mf, "three_center_ao_short_range")
    assert not _gdf_mf_cache(mf, "aux_metric_short_range")
    for summary in summaries:
        assert summary["pair_blocks"] == len(
            list(_pair_keys_for_q(space, summary["q_index"]))
        )


def test_gdf_default_prebuild_workers_scales_on_many_core_hosts(monkeypatch):
    from pyqed.pbc.gw import integrals

    monkeypatch.setattr(integrals.os, "cpu_count", lambda: 12)

    assert integrals._gdf_default_prebuild_workers() == 6


def test_prebuild_gdf_q_ao_store_reuses_self_opposite_pair_blocks(
    two_k_h2_reference,
):
    from pyqed.pbc.gw.integrals import _gdf_opposite_q_index, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")

    q_index = 1
    assert _gdf_opposite_q_index(space, q_index) == q_index
    summaries = prebuild_gdf_q_ao_stores(space, q_indices=[q_index], g2_tol=1.0e-14)
    timings = summaries[0]["timings"]
    pair_count = len(list(_pair_keys_for_q(space, q_index)))

    assert timings.get("q_ao_store_opposite_q_reuses", 0) == 0
    assert timings["q_ao_store_cache_misses"] == 1
    assert timings["q_ao_store_requested_pair_blocks"] == pair_count
    assert timings["q_ao_store_existing_pair_blocks"] == pair_count
    assert timings["q_ao_store_self_opposite_pair_reuses"] == 1
    assert timings["q_ao_store_pair_blocks"] == pair_count


def test_gdf_self_opposite_pair_reuse_matches_ordered_pair_build(
    two_k_h2_reference,
):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache, _pair_keys_for_q

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(mf, qpts="mesh")
    q_index = 1
    pair_count = len(list(_pair_keys_for_q(space, q_index)))

    mf.gdf_self_opposite_pair_reuse = False
    ordered = gdf_transition_factors(space, q_index=q_index, g2_tol=1.0e-14)
    mf.gdf_self_opposite_pair_reuse = True
    reused = gdf_transition_factors(space, q_index=q_index, g2_tol=1.0e-14)

    assert ordered.build_timings["q_ao_store_source_pair_blocks"] == pair_count
    assert ordered.build_timings["q_ao_store_self_opposite_pair_reuses"] == 0
    assert reused.build_timings["q_ao_store_source_pair_blocks"] == pair_count // 2
    assert reused.build_timings["q_ao_store_self_opposite_pair_reuses"] == pair_count // 2
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 2
    assert ordered.pair_blocks.keys() == reused.pair_blocks.keys()
    for key in ordered.pair_blocks:
        np.testing.assert_allclose(
            reused.pair_blocks[key],
            ordered.pair_blocks[key],
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        reused.coulomb_metric(),
        ordered.coulomb_metric(),
        atol=1.0e-12,
    )


def test_prebuild_gdf_q_ao_stores_reuses_opposite_q_blocks():
    from pyqed.pbc.gw.integrals import _gdf_mf_cache

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(3, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.9, 0.6]),
        np.asarray([-0.8, 0.7]),
    ]
    mf.mo_coeff = [np.eye(cell.nao, dtype=np.complex128) for _ in range(3)]
    mf.mo_occ = [np.asarray([2.0, 0.0]) for _ in range(3)]
    mf.dm = [np.diag([2.0, 0.0]).astype(np.complex128) for _ in range(3)]
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 1
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0

    space = KPointTransitionSpace(mf, qpts="mesh")
    summaries = prebuild_gdf_q_ao_stores(space, g2_tol=1.0e-14)
    derived = [
        row for row in summaries
        if row["timings"].get("q_ao_store_opposite_q_reuses", 0)
    ]

    assert len(summaries) == len(space.qpts)
    assert len(derived) == 1
    assert {row["timings"]["prebuild_workers"] for row in summaries} == {3}
    target_q = derived[0]["q_index"]
    source_q = derived[0]["timings"]["q_ao_store_opposite_source_q_index"]
    stores = list(_gdf_mf_cache(mf, "q_ao_store").values())
    target_store = next(store for store in stores if store.q_index == target_q)
    source_store = next(store for store in stores if store.q_index == source_q)

    np.testing.assert_allclose(
        target_store.metric_invsqrt,
        source_store.metric_invsqrt.conj(),
        atol=1.0e-12,
    )
    for key, block in target_store.ao_blocks.items():
        source_key = (key[1], key[0])
        np.testing.assert_allclose(
            block,
            source_store.ao_blocks[source_key].conj().transpose(0, 2, 1),
            atol=1.0e-12,
        )


def test_gdf_opposite_q_conjugate_reuse_matches_direct_build(monkeypatch):
    import pyqed.pbc.gw.integrals as integrals

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(3, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="ewald",
    )
    mf.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.9, 0.6]),
        np.asarray([-0.8, 0.7]),
    ]
    mf.mo_coeff = [np.eye(cell.nao, dtype=np.complex128) for _ in range(3)]
    mf.mo_occ = [np.asarray([2.0, 0.0]) for _ in range(3)]
    mf.dm = [np.diag([2.0, 0.0]).astype(np.complex128) for _ in range(3)]

    space = KPointTransitionSpace(mf, qpts="mesh")
    q_index = next(
        index
        for index in range(len(space.qpts))
        if integrals._gdf_should_use_opposite_q(space, index) is not None
    )
    derived = integrals.gdf_transition_factors(space, q_index=q_index, g2_tol=1.0e-14)
    assert derived.factor_method.endswith(":opposite_q_conjugate")

    monkeypatch.setattr(integrals, "_gdf_should_use_opposite_q", lambda _space, _q: None)
    direct_space = KPointTransitionSpace(mf, qpts="mesh")
    direct = integrals.gdf_transition_factors(
        direct_space,
        q_index=q_index,
        g2_tol=1.0e-14,
    )

    np.testing.assert_allclose(
        derived.coulomb_metric(),
        direct.coulomb_metric(),
        atol=1.0e-12,
    )
    transition = space.transitions(q_index)[0]
    pair = (
        transition.k_index,
        transition.kq_index,
        transition.occ_band,
        transition.vir_band,
    )
    np.testing.assert_allclose(
        derived.orbital_pair_metric(pair, pair),
        direct.orbital_pair_metric(pair, pair),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        derived.orbital_pair_coupling(*pair),
        direct.orbital_pair_coupling(*pair),
        atol=1.0e-12,
    )


def test_gdf_accepts_pyscf_auxbasis_alias(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    factors = gdf_transition_factors(
        space,
        q_index=0,
        auxbasis="def2-svp-jkfit",
    )

    assert factors.auxbasis == "def2-sv(p)-jkfit"
    assert factors.aux_coord_type == "spherical"
    assert factors.naux_cart >= factors.naux


def test_gdf_cutoffs_can_be_decoupled_from_reference_cutoffs(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.pair_cut = 0
    mf.recip_cut = 2
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 5
    mf._periodic_setup()
    decoupled = gdf_transition_metric(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    del mf.gdf_pair_cut
    del mf.gdf_recip_cut
    mf.gdf_default_pair_cut = 2
    mf.gdf_default_recip_cut = 5
    mf.pair_cut = 2
    mf.recip_cut = 5
    mf._periodic_setup()
    direct = gdf_transition_metric(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    np.testing.assert_allclose(decoupled, direct, atol=1.0e-12)
    del mf.gdf_default_pair_cut
    del mf.gdf_default_recip_cut


def test_gdf_defaults_use_accuracy_controlled_range_separation(
    two_k_h2_reference,
):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.pair_cut = 0
    mf.recip_cut = 2
    for name in ("gdf_pair_cut", "gdf_recip_cut", "gdf_mesh", "gdf_precision"):
        if hasattr(mf, name):
            delattr(mf, name)

    ref = KGW(mf).transition_space(qpts="mesh").reference
    settings = integrals._gdf_backend_settings(ref)
    _recip_cut, pair_cut, mesh, _recip_key, kernel, omega, *_rest = settings
    assert pair_cut == "auto"
    assert mesh != "auto"
    assert all(int(value) > 1 for value in mesh)
    assert kernel == "range_separated"
    assert float(omega) > 0.0

    mf.gdf_reciprocal_kernel = "full"
    recip_cut, pair_cut = integrals._gdf_backend_cutoffs(ref)
    assert recip_cut == 15
    assert pair_cut == 3
    del mf.gdf_reciprocal_kernel

    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 5
    recip_cut, pair_cut = integrals._gdf_backend_cutoffs(ref)
    assert recip_cut == 5
    assert pair_cut == 2


def test_gdf_default_cutoffs_match_pyscf_lih_pair_metric():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto, scf

    atom = "Li 0 0 0; H 1.6 0 0"
    lattice = np.diag([7.0, 7.0, 7.0])
    auxbasis = "def2-svp-jkfit"
    pcell = gto.Cell()
    pcell.atom = atom
    pcell.a = lattice
    pcell.basis = "sto-3g"
    pcell.unit = "B"
    pcell.verbose = 0
    pcell.build()
    kpts = pcell.make_kpts((1, 1, 1))
    pmf = scf.KRHF(pcell, kpts=kpts, exxdiv="ewald").density_fit(
        auxbasis=auxbasis
    )
    pmf.conv_tol = 1.0e-10
    pmf.verbose = 0
    pmf.kernel()

    qcell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
    ).build()
    qmf = qcell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=0,
        pair_cut=3,
        recip_cut=7,
        jk_builder="ewald",
    )
    qmf.gdf_auxbasis = auxbasis
    qmf.gdf_pair_ft_screen_tol = 0.0
    qmf.gdf_g_block_size = 96
    qmf.mo_energy = [np.asarray(block).copy() for block in pmf.mo_energy]
    qmf.mo_coeff = [np.asarray(block).copy() for block in pmf.mo_coeff]
    qmf.mo_occ = [np.asarray(block).copy() for block in pmf.mo_occ]
    qmf._periodic_setup()
    space = KPointTransitionSpace(qmf)

    native = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    pyscf = pyscf_gdf_transition_factors(space, q_index=0)
    native_pairs = native.pair_blocks[(0, 0)].reshape(native.naux, -1).T
    pyscf_pairs = pyscf.pair_blocks[(0, 0)].reshape(pyscf.naux, -1).T
    native_metric = native_pairs @ native_pairs.conj().T
    pyscf_metric = pyscf_pairs @ pyscf_pairs.conj().T

    relative_error = np.linalg.norm(native_metric - pyscf_metric) / np.linalg.norm(
        pyscf_metric
    )
    assert relative_error < 1.0e-5
    assert native.factor_method == "periodic_auxiliary_gdf:range_separated"
    assert native.build_timings["short_range_screen_tol"] == 0.0
    assert native.build_timings["q_ao_store_shared_range_separated_passes"] == 1
    native_j, native_k = gdf_mo_jk(space, coulomb_component=GDF)
    pyscf_j, pyscf_k = gdf_mo_jk(space, coulomb_component=PYSCF_GDF)
    np.testing.assert_allclose(native_j, pyscf_j, rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(native_k, pyscf_k, rtol=1.0e-5, atol=1.0e-7)

    gw_options = {
        "direct_scale": 1.0,
        "linearized": True,
        "frequency_integration": "ac",
        "ac_nw": 16,
        "finite_size_correction": False,
        "qp_bands": [0, 1],
    }
    native_gw = diagonal_g0w0(space, coulomb_component=GDF, **gw_options)
    pyscf_gw = diagonal_g0w0(space, coulomb_component=PYSCF_GDF, **gw_options)
    np.testing.assert_allclose(native_gw.e_qp, pyscf_gw.e_qp, atol=1.0e-7)

    native_bse = periodic_bse_matrices(space, coulomb_component=GDF)
    pyscf_bse = periodic_bse_matrices(space, coulomb_component=PYSCF_GDF)
    np.testing.assert_allclose(native_bse.A, pyscf_bse.A, atol=2.0e-6)
    np.testing.assert_allclose(native_bse.B, pyscf_bse.B, atol=2.0e-6)


def test_gdf_automatic_policy_matches_pyscf_neon_pair_metric():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto, scf

    lattice = np.diag([8.0, 8.0, 8.0])
    auxbasis = "def2-svp-jkfit"
    pcell = gto.Cell()
    pcell.atom = "Ne 0 0 0"
    pcell.a = lattice
    pcell.basis = "sto-3g"
    pcell.unit = "B"
    pcell.verbose = 0
    pcell.build()
    kpts = pcell.make_kpts((1, 1, 1))
    pmf = scf.KRHF(pcell, kpts=kpts, exxdiv="ewald").density_fit(
        auxbasis=auxbasis
    )
    pmf.conv_tol = 1.0e-10
    pmf.verbose = 0
    pmf.kernel()

    qcell = Cell(
        atom="Ne 0 0 0",
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
    ).build()
    qmf = qcell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=0,
        pair_cut=3,
        recip_cut=7,
        jk_builder="ewald",
    )
    qmf.gdf_auxbasis = auxbasis
    qmf.gdf_pair_ft_screen_tol = 0.0
    qmf.gdf_g_block_size = 96
    qmf.mo_energy = [np.asarray(block).copy() for block in pmf.mo_energy]
    qmf.mo_coeff = [np.asarray(block).copy() for block in pmf.mo_coeff]
    qmf.mo_occ = [np.asarray(block).copy() for block in pmf.mo_occ]
    qmf._periodic_setup()
    space = KPointTransitionSpace(qmf)

    native = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    pyscf = pyscf_gdf_transition_factors(space, q_index=0)
    native_pairs = native.pair_blocks[(0, 0)].reshape(native.naux, -1).T
    pyscf_pairs = pyscf.pair_blocks[(0, 0)].reshape(pyscf.naux, -1).T
    native_metric = native_pairs @ native_pairs.conj().T
    pyscf_metric = pyscf_pairs @ pyscf_pairs.conj().T
    relative_error = np.linalg.norm(native_metric - pyscf_metric) / np.linalg.norm(
        pyscf_metric
    )

    assert relative_error < 1.0e-6
    assert native.metric_rank == pyscf.naux
    assert native.build_timings["q_ao_store_shared_range_separated_passes"] == 1


def test_gdf_mesh_can_replace_reciprocal_cutoff(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.pair_cut = 0
    mf.recip_cut = 2
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    if hasattr(mf, "gdf_mesh"):
        del mf.gdf_mesh
    mf._periodic_setup()
    shell_grid = gdf_transition_metric(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    del mf.gdf_recip_cut
    mf.gdf_mesh = (5, 5, 5)
    mf._periodic_setup()
    mesh_grid = gdf_transition_metric(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    np.testing.assert_allclose(mesh_grid, shell_grid, atol=1.0e-12)


def test_gdf_g_vector_streaming_matches_full_grid(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_g_block_size = 0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    full = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_g_block_size = 7
    mf.gdf_pair_ft_subblock_size = 2
    mf.gdf_defer_pair_ft_rows = True

    streamed = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert streamed.build_timings["g_block_size"] == 7
    assert streamed.build_timings["g_blocks"] > 1
    if (
        integrals.has_periodic_pair_ft_contract_backend()
        or integrals.has_periodic_pair_ft_many_backend()
    ):
        assert streamed.build_timings["q_ao_store_shared_aux_ft"] is True
        assert streamed.build_timings["q_ao_store_shared_aux_ft_blocks"] > 1
        assert streamed.build_timings["pair_ft_stream_g_blocks"] > 1
        if integrals.has_periodic_pair_ft_many_backend():
            assert streamed.build_timings["pair_ft_stream_backend"] == "sum_many"
            assert streamed.build_timings["pair_ft_many_direct_batch_returns"] > 0
            assert streamed.build_timings["pair_ft_stream_subblock_size"] == 2
            assert streamed.build_timings["pair_ft_stream_subblocks"] > 1
            assert streamed.build_timings["pair_ft_stream_screened_row_batches"] > 0
            assert streamed.build_timings["q_ao_store_hermitian_q0_pair_mask"] is True
            assert (
                streamed.build_timings["q_ao_store_hermitian_q0_computed_pairs"]
                < streamed.build_timings["q_ao_store_hermitian_q0_total_pairs"]
            )
        else:
            assert streamed.build_timings["pair_ft_stream_backend"] == "contract_many"
    np.testing.assert_allclose(
        streamed.coulomb_metric(),
        full.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        streamed.transition_vectors,
        full.transition_vectors,
        atol=1.0e-12,
    )


def test_gdf_pair_ft_subblock_size_auto_threshold():
    import pyqed.pbc.gw.integrals as integrals

    class DummyMF:
        pass

    mf = DummyMF()
    mf.gdf_pair_ft_subblock_max_mb = 1.0
    small = integrals._gdf_pair_ft_subblock_size(
        mf,
        ng=16,
        nkpts=2,
        nao=2,
    )
    large = integrals._gdf_pair_ft_subblock_size(
        mf,
        ng=10000,
        nkpts=8,
        nao=6,
    )

    assert small == 0
    assert 0 < large < 10000
    assert large % 8 == 0


def test_gdf_phase_blas_stream_backend_matches_sum_many(two_k_h2_reference):
    if not (
        has_periodic_pair_ft_many_backend()
        and has_periodic_pair_ft_image_group_backend()
    ):
        pytest.skip("phase-BLAS periodic pair-FT backend unavailable")

    mf = two_k_h2_reference
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_g_block_size = 7
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_pair_ft_stream_backend = "sum_many"
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    summed = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_pair_ft_stream_backend = "phase_blas"

    phased = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert phased.build_timings["pair_ft_stream_backend"] == "phase_blas"
    assert phased.build_timings["pair_ft_phase_blas"] is True
    np.testing.assert_allclose(
        phased.coulomb_metric(),
        summed.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        phased.transition_vectors,
        summed.transition_vectors,
        atol=1.0e-12,
    )


def test_gdf_auto_pair_cut_builds_shell_pair_image_plan(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = "auto"
    mf.gdf_pair_image_cut_max = 2
    mf.gdf_precision = 1.0e-4
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    timings = factors.build_timings
    assert timings["pair_cut"] == "auto"
    assert timings["pair_image_plan_auto"] is True
    assert timings["pair_image_plan_tol"] == pytest.approx(1.0e-6)
    assert timings["pair_image_plan_screen"] == "overlap"
    assert timings["pair_image_plan_max_cut"] <= 2
    assert timings["pair_image_plan_images"] > 0
    assert timings["pair_image_plan_kept_image_pairs"] > 0
    assert np.all(np.isfinite(factors.transition_vectors))


def test_gdf_shell_pair_image_mask_matches_cartesian_component_scan():
    import pyqed.pbc.gw.integrals as integrals

    cell = Cell(
        atom="Li 0 0 0; H 3.0 0 0",
        a=np.diag([8.0, 8.0, 8.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(nk=(1, 1, 1), eta=0.5, real_cut=1, pair_cut=1)
    mf._validate()
    mf._periodic_setup()
    basis = tuple(mf._basis)
    shell_blocks = tuple(integrals._cart_shell_blocks(basis))
    volume = abs(float(np.linalg.det(cell.lattice_vectors)))
    tolerance = 1.0e-8

    for shift in (np.zeros(3), np.asarray(cell.lattice_vectors[0]) * 2.0):
        expected = np.zeros((len(basis), len(basis)), dtype=np.bool_)
        for p0, p1, _lp in shell_blocks:
            for q0, q1, _lq in shell_blocks:
                keep = any(
                    integrals._gdf_pair_image_bound(
                        basis[p], basis[q], shift, volume, "overlap"
                    )
                    > tolerance
                    for p in range(p0, p1)
                    for q in range(q0, q1)
                )
                if keep:
                    expected[p0:p1, q0:q1] = True

        actual = integrals._gdf_shell_pair_keep_mask(
            basis,
            shell_blocks,
            shift,
            tolerance,
            volume,
            "overlap",
        )
        np.testing.assert_array_equal(actual, expected)


def test_gdf_auto_pair_cut_can_use_legacy_decay_screen(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = "auto"
    mf.gdf_pair_image_screen = "decay"
    mf.gdf_pair_image_cut_max = 1
    mf.gdf_precision = 1.0e-4
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    timings = factors.build_timings
    assert timings["pair_cut"] == "auto"
    assert timings["pair_image_plan_screen"] == "decay"
    assert timings["pair_image_plan_images"] > 0
    assert np.all(np.isfinite(factors.transition_vectors))


def test_gdf_auto_pair_cut_tolerance_factor_override(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_pair_cut = "auto"
    mf.gdf_pair_image_cut_max = 1
    mf.gdf_precision = 1.0e-4
    mf.gdf_pair_image_tol_factor = 0.25
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    plan = integrals._gdf_pair_image_plan(mf, "auto", 0.0)

    assert plan.auto is True
    assert plan.tolerance == pytest.approx(2.5e-5)


def test_gdf_auto_pair_cut_default_tolerance_tracks_precision(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-4

    assert integrals._gdf_pair_image_tolerance(mf, 0.0) == pytest.approx(1.0e-6)


def test_gdf_weighted_aux_screen_tolerance_tracks_precision(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-4

    assert integrals._gdf_weighted_aux_screen_tol(mf) == pytest.approx(5.0e-7)

    mf.gdf_weighted_aux_screen_tol_factor = 0.25
    assert integrals._gdf_weighted_aux_screen_tol(mf) == pytest.approx(2.5e-5)

    mf.gdf_weighted_aux_screen_tol = 0.0
    assert integrals._gdf_weighted_aux_screen_tol(mf) == 0.0


def test_gdf_pair_ft_coeff_tol_defaults_to_pair_screen(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference

    assert integrals._gdf_pair_ft_coeff_tol(mf, 1.0e-13) == pytest.approx(1.0e-13)

    mf.gdf_precision = 1.0e-4
    assert integrals._gdf_pair_ft_coeff_tol(mf, 1.0e-13) == pytest.approx(1.0e-7)

    mf.gdf_pair_ft_coeff_tol_factor = 1.0e-5
    assert integrals._gdf_pair_ft_coeff_tol(mf, 1.0e-13) == pytest.approx(1.0e-9)

    mf.gdf_pair_ft_coeff_tol = 2.0e-12
    assert integrals._gdf_pair_ft_coeff_tol(mf, 1.0e-13) == pytest.approx(2.0e-12)

    mf.gdf_pair_ft_coeff_tol = 0.0
    assert integrals._gdf_pair_ft_coeff_tol(mf, 1.0e-13) == 0.0


def test_gdf_pair_ft_factor_screen_tol_defaults_to_coeff_tol(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference

    assert integrals._gdf_pair_ft_factor_screen_tol(mf, 1.0e-7) == pytest.approx(
        1.0e-7
    )

    mf.gdf_pair_ft_factor_screen_tol_factor = 0.25
    assert integrals._gdf_pair_ft_factor_screen_tol(mf, 1.0e-7) == pytest.approx(
        2.5e-8
    )

    mf.gdf_pair_ft_factor_screen_tol = 0.0
    assert integrals._gdf_pair_ft_factor_screen_tol(mf, 1.0e-7) == 0.0


def test_gdf_pair_plan_tolerance_change_invalidates_factor_and_ao_store_cache(
    two_k_h2_reference,
):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_precision = 1.0e-4
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_pair_cut = "auto"
    mf.gdf_pair_image_cut_max = 1
    mf._periodic_setup()
    space = KPointTransitionSpace(mf, qpts="mesh")

    screened = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    mf.gdf_pair_ft_factor_screen_tol = 0.0
    factor_unscreened = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    mf.gdf_pair_image_tol = 0.0
    mf.gdf_pair_ft_coeff_tol = 0.0
    unscreened = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert screened is not factor_unscreened
    assert factor_unscreened is not unscreened
    assert screened is not unscreened
    assert len(space._gdf_factor_cache) == 3
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 3
    assert screened.build_timings["pair_image_plan_tol"] == pytest.approx(1.0e-6)
    assert screened.build_timings["pair_ft_coeff_tol"] == pytest.approx(1.0e-7)
    assert screened.build_timings["pair_ft_factor_screen_tol"] == pytest.approx(
        1.0e-7
    )
    assert factor_unscreened.build_timings["pair_image_plan_tol"] == pytest.approx(
        1.0e-6
    )
    assert factor_unscreened.build_timings["pair_ft_coeff_tol"] == pytest.approx(
        1.0e-7
    )
    assert factor_unscreened.build_timings["pair_ft_factor_screen_tol"] == 0.0
    assert unscreened.build_timings["pair_image_plan_tol"] == 0.0
    assert unscreened.build_timings["pair_ft_coeff_tol"] == 0.0
    assert unscreened.build_timings["pair_ft_factor_screen_tol"] == 0.0


def test_gdf_aux_transform_plan_matches_dense():
    import pyqed.pbc.gw.integrals as integrals
    from types import SimpleNamespace

    def fn(shell):
        return SimpleNamespace(
            origin=np.zeros(3),
            shell=tuple(shell),
            exps=np.array([1.0]),
            coefs=np.array([1.0]),
        )

    cart_basis = (
        fn((0, 0, 0)),
        fn((1, 0, 0)),
        fn((0, 1, 0)),
        fn((0, 0, 1)),
    )
    transform = np.zeros((4, 4), dtype=float)
    transform[0, 0] = 1.5
    transform[1:4, 1:4] = np.array(
        [
            [1.0, 0.25, 0.0],
            [-0.5, 0.75, 0.125],
            [0.0, -0.25, 1.25],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(12)
    cart_ft = (
        rng.normal(size=(9, 4)) + 1.0j * rng.normal(size=(9, 4))
    ).astype(np.complex128)
    aux = integrals._GDFAuxiliaryBasis(
        name="mock",
        coord_type="spherical",
        cart_basis=cart_basis,
        transform=transform,
    )
    mf = SimpleNamespace()
    timings = {}

    out = integrals._gdf_apply_aux_transform(cart_ft, aux, mf, timings=timings)

    np.testing.assert_allclose(out, cart_ft @ transform)
    assert timings["aux_ft_transform_backend"] in {"shell_block", "shell_block_cpp"}

    sparse_transform = np.zeros((4, 3), dtype=float)
    sparse_transform[0, 0] = 2.0
    sparse_transform[2, 1] = -1.0
    sparse_transform[3, 2] = 0.5
    sparse_aux = integrals._GDFAuxiliaryBasis(
        name="mock-sparse",
        coord_type="custom",
        cart_basis=cart_basis,
        transform=sparse_transform,
    )
    mf.gdf_aux_transform_sparse_density = 0.5
    sparse_timings = {}

    sparse_out = integrals._gdf_apply_aux_transform(
        cart_ft,
        sparse_aux,
        mf,
        timings=sparse_timings,
    )

    np.testing.assert_allclose(sparse_out, cart_ft @ sparse_transform)
    assert sparse_timings["aux_ft_transform_backend"] == "sparse_pattern"

    three_center = (
        rng.normal(size=(2, 4, 3, 3))
        + 1.0j * rng.normal(size=(2, 4, 3, 3))
    )
    transformed = integrals._gdf_apply_three_center_aux_transform(
        three_center,
        sparse_aux,
        mf,
    )
    reference = np.einsum(
        "aP,xamn->xPmn",
        sparse_transform,
        three_center,
        optimize=True,
    )
    np.testing.assert_allclose(transformed, reference)
    np.testing.assert_allclose(
        integrals._gdf_apply_three_center_aux_transform(
            three_center[0],
            sparse_aux,
            mf,
        ),
        reference[0],
    )


def test_gdf_auxiliary_shells_use_charge_normalization():
    import math

    import pyqed.pbc.gw.integrals as integrals

    basis = tuple(
        make_contractions(
            parse_gbs(_basis_path("def2-sv(p)-jkfit")),
            ["C"],
            np.zeros((1, 3)),
            coord_types="c",
        )
    )
    normalized = integrals._gdf_charge_normalized_auxiliary_basis(basis)
    target = math.sqrt(0.25 / math.pi)

    for start, _stop, angular_momentum in _cart_shell_blocks(normalized):
        fn = normalized[start]
        order = float(angular_momentum) + 1.5
        radial_moment = (
            math.gamma(order)
            / (2.0 * np.asarray(fn.exps, dtype=float) ** order)
        )
        radial_norm = 1.0 / np.sqrt(
            math.gamma(order)
            / (2.0 * (2.0 * np.asarray(fn.exps, dtype=float)) ** order)
        )
        multipole = np.dot(fn.coefs * radial_norm, radial_moment)
        assert multipole == pytest.approx(target, abs=2.0e-15)


def test_gdf_weighted_aux_metric_matches_einsum():
    import pyqed.pbc.gw.integrals as integrals

    rng = np.random.default_rng(23)
    aux_ft = (
        rng.normal(size=(17, 5)) + 1.0j * rng.normal(size=(17, 5))
    ).astype(np.complex128)
    weights = rng.random(17)
    timings = {}

    metric = integrals._gdf_weighted_aux_metric(aux_ft, weights, timings=timings)
    weighted_aux_conj = np.ascontiguousarray(weights[:, None] * aux_ft.conj())
    metric_from_conj = integrals._gdf_weighted_aux_metric_from_conj(
        aux_ft,
        weighted_aux_conj,
        timings=timings,
    )
    expected = np.einsum(
        "g,ga,gb->ab",
        weights,
        aux_ft.conj(),
        aux_ft,
        optimize=True,
    )

    np.testing.assert_allclose(metric, expected)
    np.testing.assert_allclose(metric_from_conj, expected)
    assert timings["aux_metric_contract_backend"] == "matmul"


def test_gdf_weighted_aux_pair_contracts_match_einsum():
    import pyqed.pbc.gw.integrals as integrals

    rng = np.random.default_rng(31)
    weighted_aux = (
        rng.normal(size=(11, 4)) + 1.0j * rng.normal(size=(11, 4))
    ).astype(np.complex128)
    pair_block = (
        rng.normal(size=(11, 3, 3)) + 1.0j * rng.normal(size=(11, 3, 3))
    ).astype(np.complex128)
    pair_batch = (
        rng.normal(size=(2, 11, 3, 3))
        + 1.0j * rng.normal(size=(2, 11, 3, 3))
    ).astype(np.complex128)

    block = integrals._gdf_contract_weighted_aux_pair_block(
        weighted_aux,
        pair_block,
    )
    batch = integrals._gdf_contract_weighted_aux_pair_batch(
        weighted_aux,
        pair_batch,
    )
    pair_mask = np.asarray(
        [
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    masked_batch = integrals._gdf_contract_weighted_aux_pair_batch(
        weighted_aux,
        pair_batch,
        pair_mask=pair_mask,
    )

    np.testing.assert_allclose(
        block,
        np.einsum("ga,gmn->amn", weighted_aux, pair_block, optimize=True),
    )
    np.testing.assert_allclose(
        batch,
        np.einsum("ga,xgmn->xamn", weighted_aux, pair_batch, optimize=True),
    )
    expected_masked = np.zeros_like(masked_batch)
    expected_masked[:, :, pair_mask] = batch[:, :, pair_mask]
    np.testing.assert_allclose(masked_batch, expected_masked)


def test_gdf_weighted_aux_screen_filters_rows():
    import pyqed.pbc.gw.integrals as integrals

    gqvecs = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    weighted_aux = np.ascontiguousarray(
        [
            [1.0e-8 + 0.0j, 2.0e-8 + 0.0j],
            [2.0e-6 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 3.0e-5 + 0.0j],
        ],
        dtype=np.complex128,
    )
    timings = {}

    filtered_g, filtered_aux = integrals._gdf_screen_weighted_aux_rows(
        gqvecs,
        weighted_aux,
        1.0e-6,
        timings=timings,
    )

    np.testing.assert_allclose(filtered_g, gqvecs[1:])
    np.testing.assert_allclose(filtered_aux, weighted_aux[1:])
    assert timings["weighted_aux_screen_input_g_vectors"] == 3
    assert timings["weighted_aux_screen_kept_g_vectors"] == 2
    assert timings["weighted_aux_screen_skipped_g_vectors"] == 1


def test_gdf_stream_backend_resolves_after_weighted_aux_screen(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals
    from pyqed.qchem.fourier import has_periodic_pair_ft_contract_backend

    mf = two_k_h2_reference
    mf.gdf_g_block_max_mb = 0.01
    backend, batch_mb = integrals._gdf_resolve_pair_ft_stream_backend(
        mf,
        "auto",
        ng=4,
        nkpts=1,
        nao=2,
    )
    if has_periodic_pair_ft_many_backend():
        assert backend == "sum_many"
    assert batch_mb < integrals._gdf_g_block_max_mb(mf)

    backend, batch_mb = integrals._gdf_resolve_pair_ft_stream_backend(
        mf,
        "auto",
        ng=100000,
        nkpts=1,
        nao=2,
    )
    if has_periodic_pair_ft_contract_backend():
        assert backend == "contract_many"
    assert batch_mb > integrals._gdf_g_block_max_mb(mf)


def test_gdf_auto_pair_cut_center_image_matches_fixed_zero(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    fixed = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_pair_cut = "auto"
    mf.gdf_pair_image_cut_max = 0
    mf.gdf_pair_image_tol = 0.0

    auto = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert auto.build_timings["pair_cut"] == "auto"
    assert auto.build_timings["pair_image_plan_auto"] is True
    assert auto.build_timings["pair_image_plan_max_cut"] == 0
    np.testing.assert_allclose(
        auto.coulomb_metric(),
        fixed.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        auto.transition_vectors,
        fixed.transition_vectors,
        atol=1.0e-12,
    )


def test_range_separated_g_vector_streaming_matches_full_grid(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_g_block_size = 0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    full = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_g_block_size = 7

    streamed = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert streamed.build_timings["g_block_size"] == 7
    assert streamed.build_timings["g_blocks"] > 1
    assert streamed.build_timings["q_ao_store_shared_range_separated_passes"] == 1
    assert streamed.build_timings["q_ao_store_shared_reciprocal_blocks_reused"] > 0
    assert (
        streamed.build_timings["pair_ft_stream_g_blocks"]
        == streamed.build_timings["g_blocks"]
    )
    np.testing.assert_allclose(
        streamed.coulomb_metric(),
        full.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        streamed.transition_vectors,
        full.transition_vectors,
        atol=1.0e-12,
    )


def test_range_separated_pair_partition_no_smooth_matches_default(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_g_block_size = 0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    default = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_rs_pair_partition = "smooth"
    mf.gdf_smooth_exponent_cutoff = -1.0

    partitioned = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert partitioned.build_timings["rs_engine"] == "shell_range_separated"
    assert partitioned.build_timings["rs_pair_partition"] == "smooth"
    assert partitioned.build_timings["rs_smooth_shells"] == 0
    assert partitioned.build_timings["rs_reciprocal_only_pairs"] == 0
    np.testing.assert_allclose(
        partitioned.coulomb_metric(),
        default.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        partitioned.transition_vectors,
        default.transition_vectors,
        atol=1.0e-12,
    )


def test_range_separated_pair_partition_shell_engine_metadata(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_rs_pair_partition = "smooth"
    mf.gdf_smooth_exponent_cutoff = 10.0
    mf.gdf_g_block_size = 0
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    timings = factors.build_timings
    nao = int(mf.cell.nao)
    assert timings["rs_engine"] == "shell_range_separated"
    assert timings["rs_pair_partition"] == "smooth"
    assert timings["rs_total_shells"] > 0
    assert timings["rs_smooth_shells"] == timings["rs_total_shells"]
    assert timings["rs_smooth_aos"] == nao
    assert timings["rs_reciprocal_only_pairs"] == nao * nao
    assert timings["rs_compact_pairs"] == 0
    assert timings["rs_exponent_stat"] == "max"
    assert np.all(np.isfinite(factors.transition_vectors))


def test_range_separated_shell_engine_emits_reciprocal_terms(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()
    ref = KGW(mf).transition_space(qpts="mesh").reference

    mf.gdf_rs_pair_partition = "smooth"
    mf.gdf_smooth_exponent_cutoff = -1.0
    engine = integrals._gdf_rs_shell_engine(
        ref,
        "range_separated",
        0.4,
        (5, 5, 5),
    )
    assert engine.partition_active is False
    assert engine.reciprocal_block_kernel("range_separated", 0.4) == (
        "range_separated",
        0.4,
    )
    gqvecs = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    weights = np.asarray([2.0, 3.0], dtype=float)
    terms = engine.reciprocal_terms(gqvecs, weights, 0.4)
    assert len(terms) == 1
    assert terms[0][0] == "default"
    np.testing.assert_allclose(terms[0][1], weights)
    assert terms[0][2] is None

    manual = np.zeros((mf.cell.nao, mf.cell.nao), dtype=bool)
    manual[0, 0] = True
    mf.gdf_reciprocal_only_pair_mask = manual
    engine = integrals._gdf_rs_shell_engine(
        ref,
        "range_separated",
        0.4,
        (5, 5, 5),
    )
    assert engine.partition_active is True
    assert engine.reciprocal_block_kernel("range_separated", 0.4) == (
        "full",
        None,
    )
    terms = engine.reciprocal_terms(gqvecs, weights, 0.4)
    labels = [label for label, _weights, _mask in terms]
    assert labels == ["compact_lr", "smooth_full"]
    compact_weights = terms[0][1]
    expected_lr = weights * np.exp(
        -np.einsum("gi,gi->g", gqvecs, gqvecs) / (4.0 * 0.4 * 0.4)
    )
    np.testing.assert_allclose(compact_weights, expected_lr)
    np.testing.assert_array_equal(terms[0][2], ~manual)
    np.testing.assert_allclose(terms[1][1], weights)
    np.testing.assert_array_equal(terms[1][2], manual)


def test_range_separated_pair_partition_streaming_matches_full_grid(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_rs_pair_partition = "all"
    mf.gdf_g_block_size = 0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    full = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    space._gdf_factor_cache = {}
    for name in list(vars(mf)):
        if name.startswith("_pbc_gdf_") and name.endswith("_cache"):
            getattr(mf, name).clear()
    mf.gdf_g_block_size = 7

    streamed = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    nao = int(mf.cell.nao)
    assert streamed.build_timings["rs_pair_partition"] == "all"
    assert streamed.build_timings["rs_reciprocal_only_pairs"] == nao * nao
    assert streamed.build_timings["three_center_short_range_pairs"] == 0
    assert streamed.build_timings["g_block_size"] == 7
    assert streamed.build_timings["g_blocks"] > 1
    np.testing.assert_allclose(
        streamed.coulomb_metric(),
        full.coulomb_metric(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        streamed.transition_vectors,
        full.transition_vectors,
        atol=1.0e-12,
    )


def test_range_separated_all_reciprocal_aux_matches_full_kernel(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_rs_aux_partition = "all"
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    partitioned = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    mf.gdf_reciprocal_kernel = "full"
    full = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert partitioned.build_timings["rs_aux_partition"] == "all"
    assert partitioned.build_timings["rs_aux_compact_functions"] == 0
    assert partitioned.build_timings["three_center_short_range_pairs"] == 0
    np.testing.assert_allclose(
        partitioned.coulomb_metric(),
        full.coulomb_metric(),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        partitioned.transition_vectors,
        full.transition_vectors,
        atol=1.0e-11,
    )


def test_range_separated_compact_aux_view_preserves_transform(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "def2-svp-jkfit"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_smooth_exponent_cutoff = 1.0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")
    aux = integrals._gdf_auxiliary_basis(
        space,
        integrals._gdf_auxbasis_name(space.reference),
        "cartesian",
    )
    engine = integrals._gdf_rs_aux_engine(
        space.reference,
        aux,
        "range_separated",
        0.4,
        (5, 5, 5),
    )
    compact = integrals._gdf_rs_compact_auxiliary_basis(aux, engine)

    assert 0 < compact.ncart < aux.ncart
    assert compact.naux == aux.naux
    np.testing.assert_allclose(
        compact.transform,
        aux.transform[engine.compact_cart_mask],
    )
    assert compact.cart_basis == tuple(
        fn
        for fn, keep in zip(aux.cart_basis, engine.compact_cart_mask)
        if keep
    )


def test_gdf_precision_can_derive_mesh(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_mesh = "auto"
    mf.gdf_precision = 1.0e-8
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    assert factors.build_timings["reciprocal_mode"] == "mesh"
    assert factors.build_timings["reciprocal_kernel"] == "range_separated"
    assert factors.build_timings["pair_cut"] == 2
    assert factors.build_timings["gdf_mesh_auto"] is True
    assert factors.build_timings["gdf_precision"] == 1.0e-8
    assert factors.build_timings["gdf_ke_cutoff"] > 0.0
    assert all(n > 0 for n in factors.build_timings["mesh"])
    assert np.all(np.isfinite(factors.coulomb_metric()))


def test_gdf_precision_defaults_to_fully_automatic_range_separation(
    two_k_h2_reference,
):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-8
    for name in (
        "gdf_pair_cut",
        "gdf_mesh",
        "gdf_omega",
        "gdf_reciprocal_kernel",
    ):
        if hasattr(mf, name):
            delattr(mf, name)
    mf._periodic_setup()
    ref = KGW(mf).transition_space(qpts="mesh").reference

    settings = integrals._gdf_backend_settings(ref)

    assert settings[1] == "auto"
    assert settings[4] == "range_separated"
    assert settings[5] > 0.0
    assert all(value > 1 for value in settings[2])


def test_gdf_auto_mesh_adds_one_native_reciprocal_safety_shell(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-8
    mf.gdf_mesh = "auto"
    mf._periodic_setup()
    ref = KGW(mf).transition_space(qpts="mesh").reference

    mf.gdf_mesh_safety_pad = 0
    unpadded = integrals._gdf_backend_settings(ref)
    mf.gdf_mesh_safety_pad = 2
    padded = integrals._gdf_backend_settings(ref)

    np.testing.assert_array_equal(np.asarray(padded[2]), np.asarray(unpadded[2]) + 2)
    assert unpadded[-1]["mesh_safety_pad"] == 0
    assert padded[-1]["mesh_safety_pad"] == 2


def test_gdf_auto_omega_resolves_from_precision_and_cutoff(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = "auto"
    mf.gdf_precision = 1.0e-8
    mf.gdf_recip_cut = 2
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    assert factors.build_timings["gdf_omega_auto"] is True
    assert factors.build_timings["gdf_omega"] >= 0.3
    assert factors.build_timings["gdf_precision"] == 1.0e-8
    assert factors.factor_method == "periodic_auxiliary_gdf:range_separated"
    assert np.all(np.isfinite(factors.coulomb_metric()))


def test_gdf_auto_block_size_honors_combined_workspace_budget(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_g_block_max_mb = 1.0
    block_size = integrals._gdf_auto_g_block_size(
        mf,
        mesh=(101, 101, 101),
        naux=256,
        nao_pair=400,
        nkpts=8,
    )
    bytes_per_g = 16 * (2 * 256 + 400 * 8)

    assert block_size >= 1
    assert block_size * bytes_per_g <= 0.75 * 1.0e6


def test_gdf_precision_derives_bounded_range_separated_work(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-4
    mf.gdf_mesh = "auto"
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = "auto"
    if hasattr(mf, "gdf_short_range_cut"):
        del mf.gdf_short_range_cut
    if hasattr(mf, "gdf_real_cut"):
        del mf.gdf_real_cut
    mf._periodic_setup()
    ref = KGW(mf).transition_space(qpts="mesh").reference

    settings = integrals._gdf_backend_settings(ref)
    mesh = settings[2]
    auto_info = settings[-1]
    short_range_cut = integrals._gdf_short_range_cut(ref)

    assert auto_info["omega_seed"] == pytest.approx(0.08)
    assert np.prod(mesh) < 1_000_000
    assert isinstance(short_range_cut, int)
    assert 0 <= short_range_cut <= 64
    assert integrals._gdf_short_range_screen_tol(ref) == 0.0
    expected_exp_cutoff = -np.log(1.0e-4) + 4.0 * np.log(10.0)
    assert integrals._gdf_short_range_primitive_exp_cutoff(ref) == pytest.approx(
        expected_exp_cutoff
    )

    mf.gdf_short_range_primitive_exp_cutoff = 0.0
    assert integrals._gdf_short_range_primitive_exp_cutoff(ref) == 0.0


def test_range_separated_factor_build_streams_image_tensors(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_short_range_workers = 2
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )
    timings = factors.build_timings

    assert timings["three_center_sr_component_streamed"] is True
    assert timings["three_center_sr_grouped_bloch"] is True
    assert (
        timings["three_center_short_range_compiled_calls"]
        < timings["three_center_sr_component_terms"]
    )
    assert timings["three_center_sr_max_inflight_tasks"] <= 4
    assert timings["three_center_sr_retained_image_tensor_bytes"] == 0
    assert timings["three_center_sr_result_cache_enabled"] is False
    assert timings["three_center_sr_peak_image_tensor_bytes"] == 0
    assert timings["three_center_sr_group_output_bytes"] > 0
    assert timings["three_center_sr_group_workspace_bytes_upper_bound"] > 0
    assert timings["three_center_sr_primitive_exp_cutoff"] > 0.0
    assert timings["three_center_sr_primitive_candidates"] > 0
    assert (
        0
        < timings["three_center_sr_primitive_skips"]
        < timings["three_center_sr_primitive_candidates"]
    )


def test_range_separated_primitive_screening_matches_unscreened_and_separates_cache(
    two_k_h2_reference,
):
    mf = two_k_h2_reference
    mf.gdf_precision = 1.0e-8
    mf.gdf_pair_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_short_range_primitive_exp_cutoff = 0.0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    unscreened = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    mf.gdf_short_range_primitive_exp_cutoff = "auto"
    screened = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)

    assert screened is not unscreened
    assert screened.build_timings["q_ao_store_cache_misses"] == 1
    assert screened.build_timings["three_center_sr_primitive_skips"] > 0
    assert screened.build_timings["three_center_ao_cache_misses"] > 0
    np.testing.assert_allclose(
        screened.coulomb_metric(),
        unscreened.coulomb_metric(),
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_gdf_metric_accepts_only_precision_sized_negative_modes():
    import pyqed.pbc.gw.integrals as integrals

    invsqrt, evals = integrals._gdf_metric_invsqrt(
        np.diag([-5.0e-5, 1.0]),
        1.0e-14,
        "test-aux",
        precision=1.0e-4,
    )
    assert invsqrt.shape == (2, 1)
    np.testing.assert_allclose(evals, [0.0, 1.0])

    with pytest.raises(np.linalg.LinAlgError, match="not positive semidefinite"):
        integrals._gdf_metric_invsqrt(
            np.diag([-2.0e-4, 1.0]),
            1.0e-14,
            "test-aux",
            precision=1.0e-4,
        )


def test_gdf_metric_relative_tol_drops_ill_conditioned_modes():
    import pyqed.pbc.gw.integrals as integrals

    invsqrt, evals = integrals._gdf_metric_invsqrt(
        np.diag([1.0e-12, 1.0, 2.0]),
        1.0e-14,
        "test-aux",
        relative_threshold=1.0e-10,
    )

    assert invsqrt.shape == (3, 2)
    np.testing.assert_allclose(evals, [1.0e-12, 1.0, 2.0])


def test_periodic_gdf_exposes_relative_metric_regularization(gamma_h2_mf):
    from pyqed.qchem.pbc.gdf import PeriodicGDF

    backend = PeriodicGDF(
        gamma_h2_mf,
        auxbasis="sto-3g",
        metric_relative_tol=1.0e-10,
    )

    assert backend.metric_relative_tol == 1.0e-10
    assert gamma_h2_mf.gdf_metric_relative_tol == 1.0e-10
    with pytest.raises(ValueError, match="metric_relative_tol"):
        PeriodicGDF(
            gamma_h2_mf,
            auxbasis="sto-3g",
            metric_relative_tol=1.0,
        )


def test_periodic_gdf_exposes_folded_workspace_budget(gamma_h2_mf):
    from pyqed.qchem.pbc.gdf import PeriodicGDF

    backend = PeriodicGDF(
        gamma_h2_mf,
        auxbasis="sto-3g",
        folded_batch_mb=64.0,
    )

    assert backend.folded_batch_mb == 64.0
    assert gamma_h2_mf.gdf_folded_batch_mb == 64.0
    with pytest.raises(ValueError, match="folded_batch_mb"):
        PeriodicGDF(
            gamma_h2_mf,
            auxbasis="sto-3g",
            folded_batch_mb=0.0,
        )


def test_gdf_aux_min_exponent_prunes_diffuse_primitives(two_k_h2_reference):
    import pyqed.pbc.gw.integrals as integrals

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "def2-svp-jkfit"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    full = integrals._gdf_auxiliary_basis(
        space,
        integrals._gdf_auxbasis_name(space.reference),
        "cartesian",
    )
    mf.gdf_aux_min_exponent = 0.3
    pruned = integrals._gdf_auxiliary_basis(
        space,
        integrals._gdf_auxbasis_name(space.reference),
        "cartesian",
    )

    assert pruned.ncart < full.ncart
    assert min(float(np.min(fn.exps)) for fn in pruned.cart_basis) >= 0.3


def test_periodic_gdf_exposes_aux_min_exponent(gamma_h2_mf):
    from pyqed.qchem.pbc.gdf import PeriodicGDF

    backend = PeriodicGDF(
        gamma_h2_mf,
        auxbasis="sto-3g",
        aux_min_exponent=0.1,
    )

    assert backend.aux_min_exponent == 0.1
    assert gamma_h2_mf.gdf_aux_min_exponent == 0.1
    with pytest.raises(ValueError, match="aux_min_exponent"):
        PeriodicGDF(
            gamma_h2_mf,
            auxbasis="sto-3g",
            aux_min_exponent=-0.1,
        )


def test_gdf_metric_tolerance_defaults_to_precision_aware_floor(
    two_k_h2_reference,
):
    from pyqed.pbc.gw.integrals import _gdf_metric_tol

    mf = two_k_h2_reference
    space = KPointTransitionSpace(mf)
    mf.gdf_precision = 1.0e-10
    if hasattr(mf, "gdf_metric_tol"):
        del mf.gdf_metric_tol

    assert _gdf_metric_tol(space.reference) == pytest.approx(1.0e-11)
    assert _gdf_metric_tol(space.reference, 3.0e-12) == pytest.approx(3.0e-12)

    mf.gdf_precision = 1.0e-15
    assert _gdf_metric_tol(space.reference) == pytest.approx(1.0e-14)


def test_short_range_auxiliary_integrals_recover_full_coulomb(gamma_h2_mf):
    basis = tuple(gamma_h2_mf._basis)
    a, b = basis[0], basis[1]

    np.testing.assert_allclose(
        short_range_two_center_coulomb(a, b, 0.0),
        two_center_coulomb(a, b),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        short_range_three_center_eri(a, b, a, 0.0),
        three_center_eri(a, b, a),
        atol=1.0e-12,
    )


def test_compiled_short_range_gdf_kernels_match_python(gamma_h2_mf):
    if (
        _basis_cy is None
        or not hasattr(_basis_cy, "compute_short_range_aux_metric")
        or not hasattr(_basis_cy, "compute_short_range_aux_metric_masked")
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor")
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor_masked")
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_pair_outer_masked",
        )
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_shell_blocked_masked",
        )
    ):
        pytest.skip("compiled short-range GDF kernels are not available")

    basis = tuple(gamma_h2_mf._basis)
    signatures = tuple(_basis_fn_signature(fn) for fn in basis)
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    shells = np.ascontiguousarray(shells, dtype=np.int64)
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    exps = np.ascontiguousarray(exps, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    nprim = np.ascontiguousarray(nprim, dtype=np.int64)

    eta = 0.47
    left_shift = np.asarray([0.2, -0.1, 0.0])
    right_shift = np.asarray([-0.3, 0.15, 0.25])
    left_origins = np.ascontiguousarray(origins + left_shift[None, :])
    right_origins = np.ascontiguousarray(origins + right_shift[None, :])
    left_basis = tuple(_shifted_gaussian(fn, left_shift) for fn in basis)
    right_basis = tuple(_shifted_gaussian(fn, right_shift) for fn in basis)

    metric = _basis_cy.compute_short_range_aux_metric(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        eta,
    )
    expected_metric = np.asarray(
        [
            [
                short_range_two_center_coulomb(left_basis[p], right_basis[q], eta)
                for q in range(len(basis))
            ]
            for p in range(len(basis))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(metric, expected_metric, atol=1.0e-11)
    pair_mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
    masked_metric = _basis_cy.compute_short_range_aux_metric_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        pair_mask,
        eta,
    )
    np.testing.assert_allclose(
        masked_metric,
        expected_metric * pair_mask,
        atol=1.0e-11,
    )

    tensor = _basis_cy.compute_short_range_three_center_tensor(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        eta,
    )
    expected_tensor = np.asarray(
        [
            [
                [
                    short_range_three_center_eri(
                        left_basis[p],
                        right_basis[q],
                        basis[a],
                        eta,
                    )
                    for q in range(len(basis))
                ]
                for p in range(len(basis))
            ]
            for a in range(len(basis))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(tensor, expected_tensor, atol=1.0e-11)
    aux_pair_mask = np.ones((len(basis), len(basis), len(basis)), dtype=np.uint8)
    aux_pair_mask[0, 0, 1] = 0
    aux_pair_mask[1, 1, 0] = 0
    masked_tensor = _basis_cy.compute_short_range_three_center_tensor_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        pair_mask,
        aux_pair_mask,
        eta,
    )
    np.testing.assert_allclose(
        masked_tensor,
        expected_tensor * pair_mask[None, :, :] * aux_pair_mask,
        atol=1.0e-11,
    )
    pair_outer_tensor = _basis_cy.compute_short_range_three_center_tensor_pair_outer_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        pair_mask,
        aux_pair_mask,
        eta,
    )
    np.testing.assert_allclose(
        pair_outer_tensor,
        expected_tensor * pair_mask[None, :, :] * aux_pair_mask,
        atol=1.0e-11,
    )
    blocks = _cart_shell_blocks(basis)
    shell_starts = np.ascontiguousarray(
        [start for start, _stop, _l in blocks],
        dtype=np.int64,
    )
    shell_stops = np.ascontiguousarray(
        [stop for _start, stop, _l in blocks],
        dtype=np.int64,
    )
    shell_blocked_tensor = _basis_cy.compute_short_range_three_center_tensor_shell_blocked_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        shell_starts,
        shell_stops,
        shell_starts,
        shell_stops,
        pair_mask,
        aux_pair_mask,
        eta,
    )
    np.testing.assert_allclose(
        shell_blocked_tensor,
        expected_tensor * pair_mask[None, :, :] * aux_pair_mask,
        atol=1.0e-11,
    )


def test_compiled_periodic_pair_ft_plan_matches_python_reference():
    import pyqed.qchem.fourier as fourier

    if not hasattr(_basis_cy, "compute_periodic_pair_ft_primitive_terms"):
        pytest.skip("compiled AO-pair Fourier plan builder is not available")

    cell = Cell(
        atom="Li 0 0 0; H 3.0 0 0",
        a=np.diag([8.0, 8.0, 8.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(nk=(1, 1, 1), eta=0.5, real_cut=1, pair_cut=1)
    mf._validate()
    mf._periodic_setup()
    basis = tuple(mf._basis)
    plan = fourier.AOBlockPairFTPlan(basis, basis)
    origins = np.ascontiguousarray([fn.origin for fn in basis], dtype=float)
    shifts = np.ascontiguousarray(
        [
            np.zeros(3),
            cell.lattice_vectors[0],
            -cell.lattice_vectors[1],
        ],
        dtype=float,
    )
    right_origins = np.ascontiguousarray(
        origins[None, :, :] + shifts[:, None, :]
    )
    image_pair_mask = np.ones((len(shifts), len(plan.pair_p)), dtype=np.bool_)
    image_pair_mask[2, ::3] = False
    reference = plan.periodic_primitive_terms(
        origins,
        right_origins,
        image_pair_mask=image_pair_mask,
        coeff_tol=1.0e-12,
        compiled=False,
    )
    compiled = plan.periodic_primitive_terms(
        origins,
        right_origins,
        image_pair_mask=image_pair_mask,
        coeff_tol=1.0e-12,
        compiled=True,
    )

    floating_keys = {"term_center"}
    identity_keys = {"product_group_factor_id"}
    metadata_keys = {"builder_backend"}
    for key in reference.keys() - floating_keys - identity_keys - metadata_keys:
        np.testing.assert_array_equal(compiled[key], reference[key])
    assert compiled["builder_backend"] == "compiled"
    assert reference["builder_backend"] == "python"
    np.testing.assert_allclose(
        compiled["term_center"],
        reference["term_center"],
        atol=2.0e-15,
        rtol=0.0,
    )
    compiled_factor_id = np.asarray(compiled["product_group_factor_id"])
    reference_factor_id = np.asarray(reference["product_group_factor_id"])
    np.testing.assert_array_equal(
        compiled_factor_id[:, None] == compiled_factor_id[None, :],
        reference_factor_id[:, None] == reference_factor_id[None, :],
    )

    gvecs = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.3, 0.1],
            [1.1, 0.4, -0.2],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    phases = np.ascontiguousarray(
        np.exp(1.0j * (np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, -0.3]]) @ shifts.T))
    )
    reference_ft = plan.periodic_sum_many(
        gvecs,
        origins,
        right_origins,
        phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=reference,
        threads=2,
    )
    compiled_ft = plan.periodic_sum_many(
        gvecs,
        origins,
        right_origins,
        phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=compiled,
        threads=2,
    )
    np.testing.assert_allclose(compiled_ft, reference_ft, atol=1.0e-14, rtol=0.0)

    factor_screen_tol = 1.0e-10
    screened_terms = dict(compiled)
    screened_terms["factor_screen_tol"] = factor_screen_tol
    screened_ft = plan.periodic_sum_many(
        gvecs,
        origins,
        right_origins,
        phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=screened_terms,
        threads=2,
    )
    screening_error = float(np.max(np.abs(screened_ft - compiled_ft)))
    assert screening_error > 0.0
    assert screening_error <= factor_screen_tol


def test_cpp_periodic_pair_ft_primitive_sum_matches_numba(monkeypatch, gamma_h2_mf):
    import pyqed.qchem.fourier as fourier

    if (
        fourier._gdf_cpp is None
        or not hasattr(fourier._gdf_cpp, "periodic_pair_ft_primitive_sum")
    ):
        pytest.skip("compiled C++ GDF pair-FT kernel is not available")
    if fourier._ao_block_pair_ft_primitive_sum_numba is None:
        pytest.skip("numba AO-pair Fourier reference kernel is not available")

    basis = tuple(gamma_h2_mf._basis)
    plan = fourier.AOBlockPairFTPlan(basis, basis)
    origins = np.ascontiguousarray(
        [np.asarray(fn.origin, dtype=float) for fn in basis],
        dtype=float,
    )
    shifts = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.1],
        ],
        dtype=float,
    )
    right_origins_batch = np.ascontiguousarray(origins[None, :, :] + shifts[:, None, :])
    image_pair_mask = np.ones((len(shifts), len(plan.pair_p)), dtype=np.bool_)
    image_pair_mask[1, 1::2] = False
    primitive_terms = plan.periodic_primitive_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    _assert_primitive_term_image_groups(primitive_terms, len(plan.pair_p))
    gvecs = np.ascontiguousarray(
        [
            [0.1, 0.2, -0.3],
            [0.7, -0.1, 0.4],
            [1.1, 0.3, 0.0],
        ],
        dtype=float,
    )
    kvec = np.asarray([0.2, -0.3, 0.1], dtype=float)
    phases = np.ascontiguousarray(np.exp(1.0j * (shifts @ kvec)))

    monkeypatch.setenv("PYQED_GDF_CPP", "1")
    cpp = plan.periodic_sum(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
    )
    monkeypatch.setattr(fourier, "_gdf_cpp", None)
    numba = plan.periodic_sum(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
    )

    np.testing.assert_allclose(cpp, numba, atol=1.0e-12)


def test_cpp_periodic_pair_ft_primitive_sum_many_matches_stacked_numba(
    monkeypatch,
    gamma_h2_mf,
):
    import pyqed.qchem.fourier as fourier

    if (
        fourier._gdf_cpp is None
        or not hasattr(fourier._gdf_cpp, "periodic_pair_ft_primitive_sum_many")
    ):
        pytest.skip("batched compiled C++ GDF pair-FT kernel is not available")
    if fourier._ao_block_pair_ft_primitive_sum_numba is None:
        pytest.skip("numba AO-pair Fourier reference kernel is not available")

    basis = tuple(gamma_h2_mf._basis)
    plan = fourier.AOBlockPairFTPlan(basis, basis)
    origins = np.ascontiguousarray(
        [np.asarray(fn.origin, dtype=float) for fn in basis],
        dtype=float,
    )
    shifts = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.1],
            [-0.2, 0.4, -0.3],
        ],
        dtype=float,
    )
    right_origins_batch = np.ascontiguousarray(origins[None, :, :] + shifts[:, None, :])
    image_pair_mask = np.ones((len(shifts), len(plan.pair_p)), dtype=np.bool_)
    image_pair_mask[1, 1::2] = False
    image_pair_mask[2, ::3] = False
    primitive_terms = plan.periodic_primitive_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    product_terms = plan.periodic_product_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    _assert_primitive_term_image_groups(primitive_terms, len(plan.pair_p))
    gvecs = np.ascontiguousarray(
        [
            [0.1, 0.2, -0.3],
            [0.7, -0.1, 0.4],
            [1.1, 0.3, 0.0],
        ],
        dtype=float,
    )
    kvecs = np.ascontiguousarray(
        [
            [0.2, -0.3, 0.1],
            [-0.4, 0.15, 0.25],
        ],
        dtype=float,
    )
    phases = np.ascontiguousarray(np.exp(1.0j * (kvecs @ shifts.T)))

    cpp = plan.periodic_sum_many(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
        threads=2,
    )
    product_cpp = None
    if has_periodic_pair_ft_product_backend():
        product_cpp = plan.periodic_sum_many(
            gvecs,
            left_origins=origins,
            right_origins_batch=right_origins_batch,
            phases=phases,
            image_pair_mask=image_pair_mask,
            primitive_terms=primitive_terms,
            product_terms=product_terms,
            compiled=True,
            threads=2,
        )
    phase_blas_cpp = None
    if has_periodic_pair_ft_image_group_backend():
        phase_blas_cpp = plan.periodic_sum_many_phase_blas(
            gvecs,
            left_origins=origins,
            right_origins_batch=right_origins_batch,
            phases=phases,
            image_pair_mask=image_pair_mask,
            primitive_terms=primitive_terms,
            compiled=True,
            threads=2,
        )
    weighted_aux = np.ascontiguousarray(
        [
            [0.7 + 0.1j, -0.2 + 0.3j],
            [0.4 - 0.5j, 0.1 + 0.2j],
            [-0.3 + 0.6j, 0.8 - 0.1j],
        ],
        dtype=np.complex128,
    )
    contract_cpp = None
    if hasattr(fourier._gdf_cpp, "periodic_pair_ft_primitive_contract_many"):
        contract_cpp = plan.periodic_contract_many(
            gvecs,
            weighted_aux,
            left_origins=origins,
            right_origins_batch=right_origins_batch,
            phases=phases,
            image_pair_mask=image_pair_mask,
            primitive_terms=primitive_terms,
            compiled=True,
            threads=8,
        )
    monkeypatch.setattr(fourier, "_gdf_cpp", None)
    numba = np.stack(
        [
            plan.periodic_sum(
                gvecs,
                left_origins=origins,
                right_origins_batch=right_origins_batch,
                phases=phase_row,
                image_pair_mask=image_pair_mask,
                primitive_terms=primitive_terms,
                compiled=True,
            )
            for phase_row in phases
        ],
        axis=0,
    )

    np.testing.assert_allclose(cpp, numba, atol=1.0e-12)
    if product_cpp is not None:
        np.testing.assert_allclose(product_cpp, numba, atol=1.0e-12)
    if phase_blas_cpp is not None:
        np.testing.assert_allclose(phase_blas_cpp, numba, atol=1.0e-12)
    if contract_cpp is not None:
        contract_ref = np.einsum("ga,xgmn->xamn", weighted_aux, numba, optimize=True)
        np.testing.assert_allclose(contract_cpp, contract_ref, atol=1.0e-12)


def test_cpp_gaussian_ft_batch_matches_python_reference():
    import pyqed.qchem.fourier as fourier

    if fourier._gdf_cpp is None or not hasattr(fourier._gdf_cpp, "gaussian_ft_batch"):
        pytest.skip("compiled C++ Gaussian FT kernel is not available")

    basis = (
        ContractedGaussian(
            origin=[0.0, 0.0, 0.0],
            shell=(0, 0, 0),
            exps=[0.7, 0.25],
            coefs=[0.9, 0.35],
        ),
        ContractedGaussian(
            origin=[0.35, -0.15, 0.25],
            shell=(1, 0, 0),
            exps=[0.6, 0.18],
            coefs=[0.8, 0.25],
        ),
        ContractedGaussian(
            origin=[-0.25, 0.2, -0.1],
            shell=(1, 1, 0),
            exps=[0.9, 0.3],
            coefs=[0.7, 0.45],
        ),
    )
    gvecs = np.ascontiguousarray(
        [
            [0.1, 0.2, -0.3],
            [0.7, -0.1, 0.4],
            [1.1, 0.3, 0.2],
            [-0.5, 0.8, -0.25],
        ],
        dtype=float,
    )
    signatures = tuple(_basis_fn_signature(fn) for fn in basis)
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)

    cpp = fourier.gaussian_ft_batch_compiled(
        gvecs,
        shells,
        origins,
        exps,
        weights,
        nprim,
        threads=2,
    )
    ref = np.vstack([_gdf_gaussian_ft_batch(fn, gvecs) for fn in basis]).T

    np.testing.assert_allclose(cpp, ref, atol=1.0e-12)


def test_native_gaussian_basis_ft_matches_pyscf_periodic_ft_ao():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto
    from pyscf.pbc.df import ft_ao

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    cartesian_basis, transform = cell.unit_molecule._cart_basis()
    momenta = np.asarray(
        [
            [0.2, -0.1, 0.3],
            [0.7, 0.4, -0.2],
            [1.1, -0.6, 0.5],
        ]
    )
    actual = gaussian_basis_ft_batch(
        cartesian_basis,
        momenta,
        transform=transform,
    )

    pyscf_cell = gto.Cell()
    pyscf_cell.atom = "H 0 0 0; H 1.4 0 0"
    pyscf_cell.a = np.diag([5.0, 5.0, 5.0])
    pyscf_cell.basis = "sto-3g"
    pyscf_cell.unit = "B"
    pyscf_cell.verbose = 0
    pyscf_cell.build()
    expected = ft_ao.ft_ao(pyscf_cell, momenta)

    np.testing.assert_allclose(actual, expected, atol=1.0e-7)


def test_cpp_periodic_pair_ft_grouped_p_terms_match_numba(monkeypatch):
    import pyqed.qchem.fourier as fourier

    if (
        fourier._gdf_cpp is None
        or not hasattr(fourier._gdf_cpp, "periodic_pair_ft_primitive_sum_many")
        or not hasattr(fourier._gdf_cpp, "periodic_pair_ft_primitive_contract_many")
        or not has_periodic_pair_ft_image_group_backend()
        or not has_periodic_pair_ft_product_backend()
    ):
        pytest.skip("batched compiled C++ GDF pair-FT kernels are not available")
    if fourier._ao_block_pair_ft_primitive_sum_numba is None:
        pytest.skip("numba AO-pair Fourier reference kernel is not available")

    basis = (
        ContractedGaussian(
            origin=[0.0, 0.0, 0.0],
            shell=(1, 0, 0),
            exps=[0.7, 0.25],
            coefs=[0.9, 0.35],
        ),
        ContractedGaussian(
            origin=[0.35, -0.15, 0.25],
            shell=(0, 1, 0),
            exps=[0.6, 0.18],
            coefs=[0.8, 0.25],
        ),
        ContractedGaussian(
            origin=[-0.25, 0.2, -0.1],
            shell=(0, 0, 0),
            exps=[0.9, 0.3],
            coefs=[0.7, 0.45],
        ),
    )
    plan = fourier.AOBlockPairFTPlan(basis, basis)
    origins = np.ascontiguousarray(
        [np.asarray(fn.origin, dtype=float) for fn in basis],
        dtype=float,
    )
    shifts = np.ascontiguousarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, -0.2, 0.1],
            [-0.3, 0.25, -0.35],
        ],
        dtype=float,
    )
    right_origins_batch = np.ascontiguousarray(origins[None, :, :] + shifts[:, None, :])
    image_pair_mask = np.ones((len(shifts), len(plan.pair_p)), dtype=np.bool_)
    image_pair_mask[1, 1::2] = False
    image_pair_mask[2, ::3] = False
    primitive_terms = plan.periodic_primitive_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    product_terms = plan.periodic_product_terms(
        origins,
        right_origins_batch,
        image_pair_mask=image_pair_mask,
    )
    _assert_primitive_term_image_groups(primitive_terms, len(plan.pair_p))
    assert np.any(primitive_terms["term_power"] != 0)

    gvecs = np.ascontiguousarray(
        [
            [0.1, 0.2, -0.3],
            [0.7, -0.1, 0.4],
            [1.1, 0.3, 0.2],
            [-0.5, 0.8, -0.25],
        ],
        dtype=float,
    )
    kvecs = np.ascontiguousarray(
        [
            [0.2, -0.3, 0.1],
            [-0.4, 0.15, 0.25],
        ],
        dtype=float,
    )
    phases = np.ascontiguousarray(np.exp(1.0j * (kvecs @ shifts.T)))
    weighted_aux = np.ascontiguousarray(
        [
            [0.7 + 0.1j, -0.2 + 0.3j],
            [0.4 - 0.5j, 0.1 + 0.2j],
            [-0.3 + 0.6j, 0.8 - 0.1j],
            [0.2 - 0.25j, -0.4 + 0.7j],
        ],
        dtype=np.complex128,
    )

    cpp = plan.periodic_sum_many(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
        threads=2,
    )
    product_cpp = plan.periodic_sum_many(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        product_terms=product_terms,
        compiled=True,
        threads=2,
    )
    phase_blas_cpp = plan.periodic_sum_many_phase_blas(
        gvecs,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
        threads=2,
    )
    contract_cpp = plan.periodic_contract_many(
        gvecs,
        weighted_aux,
        left_origins=origins,
        right_origins_batch=right_origins_batch,
        phases=phases,
        image_pair_mask=image_pair_mask,
        primitive_terms=primitive_terms,
        compiled=True,
        threads=12,
    )

    monkeypatch.setattr(fourier, "_gdf_cpp", None)
    ref = np.stack(
        [
            plan.periodic_sum(
                gvecs,
                left_origins=origins,
                right_origins_batch=right_origins_batch,
                phases=phase_row,
                image_pair_mask=image_pair_mask,
                primitive_terms=primitive_terms,
                compiled=True,
            )
            for phase_row in phases
        ],
        axis=0,
    )
    contract_ref = np.einsum("ga,xgmn->xamn", weighted_aux, ref, optimize=True)

    np.testing.assert_allclose(cpp, ref, atol=1.0e-12)
    np.testing.assert_allclose(product_cpp, ref, atol=1.0e-12)
    np.testing.assert_allclose(phase_blas_cpp, ref, atol=1.0e-12)
    np.testing.assert_allclose(contract_cpp, contract_ref, atol=1.0e-12)


def test_compiled_short_range_gdf_kernels_match_python_high_l():
    if (
        _basis_cy is None
        or not hasattr(_basis_cy, "compute_short_range_aux_metric")
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor")
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_pair_outer_masked",
        )
    ):
        pytest.skip("compiled short-range GDF kernels are not available")

    basis_dict = parse_gbs(_basis_path("def2-sv(p)-jkfit"))
    aux_basis = tuple(
        make_contractions(
            basis_dict,
            ["Li", "H"],
            np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            coord_types="c",
        )
    )
    indices = []
    for shell in ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)):
        indices.extend(
            idx for idx, fn in enumerate(aux_basis) if tuple(fn.shell) == shell
        )
    subset = tuple(aux_basis[idx] for idx in dict.fromkeys(indices))
    subset = subset[:4]

    signatures = tuple(_basis_fn_signature(fn) for fn in subset)
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    shells = np.ascontiguousarray(shells, dtype=np.int64)
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    exps = np.ascontiguousarray(exps, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    nprim = np.ascontiguousarray(nprim, dtype=np.int64)

    eta = 0.4770707159433863
    left_shift = np.asarray([0.3, -0.2, 0.1])
    right_shift = np.asarray([-0.1, 0.25, 0.4])
    left_origins = np.ascontiguousarray(origins + left_shift[None, :])
    right_origins = np.ascontiguousarray(origins + right_shift[None, :])
    left_basis = tuple(_shifted_gaussian(fn, left_shift) for fn in subset)
    right_basis = tuple(_shifted_gaussian(fn, right_shift) for fn in subset)

    metric = _basis_cy.compute_short_range_aux_metric(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        eta,
    )
    expected_metric = np.asarray(
        [
            [
                short_range_two_center_coulomb(left_basis[p], right_basis[q], eta)
                for q in range(len(subset))
            ]
            for p in range(len(subset))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(metric, expected_metric, atol=1.0e-11)

    tensor = _basis_cy.compute_short_range_three_center_tensor(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        eta,
    )
    expected_tensor = np.asarray(
        [
            [
                [
                    short_range_three_center_eri(
                        left_basis[p],
                        right_basis[q],
                        subset[a],
                        eta,
                    )
                    for q in range(len(subset))
                ]
                for p in range(len(subset))
            ]
            for a in range(len(subset))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(tensor, expected_tensor, atol=1.0e-11)
    pair_mask = np.ones((len(subset), len(subset)), dtype=np.uint8)
    aux_pair_mask = np.ones(
        (len(subset), len(subset), len(subset)),
        dtype=np.uint8,
    )
    pair_outer_tensor = _basis_cy.compute_short_range_three_center_tensor_pair_outer_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        pair_mask,
        aux_pair_mask,
        eta,
    )
    np.testing.assert_allclose(pair_outer_tensor, expected_tensor, atol=1.0e-11)


def test_compiled_short_range_gdf_shell_blocked_matches_dense_high_l():
    if (
        _basis_cy is None
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor")
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_shell_blocked_masked",
        )
    ):
        pytest.skip("compiled short-range GDF kernels are not available")

    basis_dict = parse_gbs(_basis_path("def2-sv(p)-jkfit"))
    aux_basis = tuple(
        make_contractions(
            basis_dict,
            ["Li", "H"],
            np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            coord_types="c",
        )
    )
    shell_blocks = _cart_shell_blocks(aux_basis)
    start, stop, _l = next(block for block in shell_blocks if block[2] == 2)
    subset = tuple(aux_basis[start:stop])
    signatures = tuple(_basis_fn_signature(fn) for fn in subset)
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    shells = np.ascontiguousarray(shells, dtype=np.int64)
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    exps = np.ascontiguousarray(exps, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    nprim = np.ascontiguousarray(nprim, dtype=np.int64)

    eta = 0.4770707159433863
    left_shift = np.asarray([0.3, -0.2, 0.1])
    right_shift = np.asarray([-0.1, 0.25, 0.4])
    left_origins = np.ascontiguousarray(origins + left_shift[None, :])
    right_origins = np.ascontiguousarray(origins + right_shift[None, :])
    dense = _basis_cy.compute_short_range_three_center_tensor(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        eta,
    )
    pair_mask = np.ones((len(subset), len(subset)), dtype=np.uint8)
    aux_pair_mask = np.ones(
        (len(subset), len(subset), len(subset)),
        dtype=np.uint8,
    )
    shell_starts = np.asarray([0], dtype=np.int64)
    shell_stops = np.asarray([len(subset)], dtype=np.int64)
    shell_blocked = _basis_cy.compute_short_range_three_center_tensor_shell_blocked_masked(
        shells,
        left_origins,
        right_origins,
        exps,
        weights,
        nprim,
        shells,
        origins,
        exps,
        weights,
        nprim,
        shell_starts,
        shell_stops,
        shell_starts,
        shell_stops,
        pair_mask,
        aux_pair_mask,
        eta,
    )

    np.testing.assert_allclose(shell_blocked, dense, atol=1.0e-10)


@pytest.mark.parametrize("kernel", ["long_range", "range_separated"])
def test_gdf_range_separated_kernels_require_omega(two_k_h2_reference, kernel):
    mf = two_k_h2_reference
    mf.gdf_reciprocal_kernel = kernel
    if hasattr(mf, "gdf_omega"):
        del mf.gdf_omega

    with pytest.raises(ValueError, match="gdf_omega"):
        gdf_transition_metric(
            KGW(mf).transition_space(qpts="mesh"),
            q_index=0,
        g2_tol=1.0e-14,
    )


def test_gdf_short_range_screening_requires_diagnostic_opt_in(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 0
    mf.gdf_short_range_screen_tol = 1.0e-12
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    with pytest.raises(ValueError, match="heuristic"):
        gdf_transition_metric(
            KGW(mf).transition_space(qpts="mesh"),
            q_index=0,
            g2_tol=1.0e-14,
        )


def test_gdf_long_range_reciprocal_kernel_is_cache_distinct(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    full_factors = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    full_metric = full_factors.coulomb_metric()

    mf.gdf_reciprocal_kernel = "long_range"
    mf.gdf_omega = 0.4
    long_range_factors = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    long_range_metric = long_range_factors.coulomb_metric()

    assert (
        long_range_factors.factor_method
        == "periodic_auxiliary_gdf:long_range_reciprocal"
    )
    assert long_range_factors.build_timings["reciprocal_kernel"] == "long_range"
    assert np.linalg.norm(long_range_metric - full_metric) > 1.0e-8

    mf.gdf_omega = 1.0e8
    large_omega_metric = gdf_transition_metric(
        space,
        q_index=0,
        g2_tol=1.0e-14,
    )

    np.testing.assert_allclose(large_omega_metric, full_metric, atol=1.0e-10)


def test_gdf_range_separated_adds_short_range_real_space_terms(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 0
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    mf.gdf_reciprocal_kernel = "long_range"
    long_range_factors = gdf_transition_factors(space, q_index=1, g2_tol=1.0e-14)
    long_range_metric = long_range_factors.coulomb_metric()

    mf.gdf_reciprocal_kernel = "range_separated"
    range_factors = gdf_transition_factors(space, q_index=1, g2_tol=1.0e-14)
    range_metric = range_factors.coulomb_metric()

    assert range_factors.factor_method == "periodic_auxiliary_gdf:range_separated"
    assert range_factors.build_timings["reciprocal_kernel"] == "range_separated"
    assert range_factors.build_timings["short_range_cut"] == 0
    assert np.linalg.norm(range_metric - long_range_metric) > 1.0e-8


def test_gdf_anisotropic_short_range_cut_uses_slab_image_box(two_k_h2_reference):
    cell = two_k_h2_reference.cell

    assert len(_gdf_image_keys(cell, 1)) == 27
    assert _gdf_image_keys(cell, (1, 0, 0)) == [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    assert len(_gdf_image_keys(cell, (2, 2, 0))) == 25


def test_gdf_range_separated_exact_short_range_uses_compiled_kernel(two_k_h2_reference):
    if (
        _basis_cy is None
        or not hasattr(_basis_cy, "compute_short_range_aux_metric")
        or not hasattr(_basis_cy, "compute_short_range_aux_metric_masked")
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor")
        or not hasattr(_basis_cy, "compute_short_range_three_center_tensor_masked")
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_pair_outer_masked",
        )
        or not hasattr(
            _basis_cy,
            "compute_short_range_three_center_tensor_shell_blocked_masked",
        )
    ):
        pytest.skip("compiled short-range GDF kernels are not available")

    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf.gdf_pair_ft_workers = 2
    mf.gdf_short_range_workers = 2
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    assert factors.build_timings["aux_metric_short_range_compiled_calls"] > 0
    assert factors.build_timings["three_center_short_range_compiled_calls"] > 0
    assert (
        factors.build_timings["three_center_short_range_shell_blocked_compiled_calls"]
        > 0
    )
    assert (
        factors.build_timings["three_center_short_range_image_pair_symmetry_reuses"]
        > 0
    )
    assert factors.build_timings["three_center_short_range_workers"] == 2
    assert factors.build_timings["three_center_sr_bloch_contiguous"] is True
    assert factors.build_timings["three_center_sr_shell_task_skips"] >= 0
    assert len(factors.build_timings["three_center_sr_worker_seconds"]) == 2
    assert factors.build_timings["pair_ft_workers"] == 2
    assert (
        factors.build_timings["three_center_short_range_compiled_calls"]
        < factors.build_timings["three_center_sr_component_terms"]
    )


def test_gdf_range_separated_q0_applies_g0_compensation(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 2
    mf.gdf_mesh = (5, 5, 5)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_omega = 0.4
    mf.gdf_short_range_cut = 1
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_aux_coord_type = "cartesian"
    mf._periodic_setup()

    factors = gdf_transition_factors(
        KGW(mf).transition_space(qpts="mesh"),
        q_index=0,
        g2_tol=1.0e-14,
    )

    assert factors.factor_method == "periodic_auxiliary_gdf:range_separated"
    assert factors.metric_rank > 0
    assert factors.build_timings["aux_metric_short_range_g0_corrections"] == 1
    assert factors.build_timings["three_center_short_range_g0_corrections"] >= 1


def test_gdf_reciprocal_zero_recognizes_lattice_equivalent_momenta():
    reference = SimpleNamespace(
        cartesian_to_scaled=lambda vector: np.asarray(vector, dtype=float)
    )

    assert _gdf_is_reciprocal_zero(reference, [0.0, 0.0, 0.0])
    assert _gdf_is_reciprocal_zero(reference, [1.0, -2.0, 3.0])
    assert _gdf_is_reciprocal_zero(reference, [1.0 + 5.0e-11, 0.0, 0.0])
    assert not _gdf_is_reciprocal_zero(reference, [0.5, 0.0, 0.0])


def test_gdf_matches_pyscf_gdf_transition_metrics(two_k_h2_reference):
    pytest.importorskip("pyscf")
    two_k_h2_reference.pair_cut = 2
    two_k_h2_reference.recip_cut = 5
    two_k_h2_reference._g_weight_cache = {}
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    for q_index in range(space.nqpts):
        native = gdf_transition_metric(space, q_index=q_index, g2_tol=1.0e-14)
        pyscf = pyscf_gdf_transition_metric(space, q_index=q_index)
        np.testing.assert_allclose(native, pyscf, rtol=1.0e-4, atol=1.0e-6)


def test_two_k_gdf_jk_matches_pyscf_gdf(two_k_h2_reference):
    pytest.importorskip("pyscf")
    mf = two_k_h2_reference
    mf.gdf_pair_cut = 2
    mf.gdf_recip_cut = 5
    mf.gdf_pair_ft_screen_tol = 0.0
    mf._periodic_setup()
    space = KGW(mf).transition_space(qpts="mesh")

    native_j, native_k = gdf_mo_jk(space, coulomb_component=GDF)
    pyscf_j, pyscf_k = gdf_mo_jk(space, coulomb_component=PYSCF_GDF)
    density_j, density_k = gdf_mo_jk(
        space,
        coulomb_component=GDF,
        dm=mf.dm,
    )

    np.testing.assert_allclose(native_j, pyscf_j, rtol=2.0e-4, atol=2.0e-6)
    np.testing.assert_allclose(native_k, pyscf_k, rtol=2.0e-4, atol=2.0e-6)
    np.testing.assert_allclose(density_j, native_j, atol=1.0e-12)
    np.testing.assert_allclose(density_k, native_k, atol=1.0e-12)


def test_three_k_gdf_matches_pyscf_at_nonzero_q():
    pytest.importorskip("pyscf")
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
    ).build()
    mf = cell.KRHF(kpts=cell.make_kpts((3, 1, 1)))
    mf.gdf_auxbasis = "def2-svp-jkfit"
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.mo_energy = [np.asarray([-1.0, 0.5]) for _ in range(3)]
    mf.mo_coeff = [np.eye(cell.nao) for _ in range(3)]
    mf.mo_occ = [np.asarray([2.0, 0.0]) for _ in range(3)]
    space = KPointTransitionSpace(mf, qpts="mesh")

    for q_index in range(space.nqpts):
        native = gdf_transition_factors(space, q_index=q_index, g2_tol=1.0e-14)
        pyscf = pyscf_gdf_transition_factors(space, q_index=q_index)
        if q_index == 0:
            assert native.build_timings["three_center_sr_grouped_bloch"] is True
            assert native.build_timings["short_range_image_domain"] == "radial"
        assert np.linalg.norm(space.qpts[q_index]) > 0.0 or q_index == 0
        for pair_key, native_block in native.pair_blocks.items():
            pyscf_block = pyscf.pair_blocks[pair_key]
            native_pairs = native_block.reshape(native.naux, -1).T
            pyscf_pairs = pyscf_block.reshape(pyscf.naux, -1).T
            np.testing.assert_allclose(
                native_pairs @ native_pairs.conj().T,
                pyscf_pairs @ pyscf_pairs.conj().T,
                rtol=2.0e-7,
                atol=2.0e-8,
            )


def test_polarized_ccpvdz_gdf_matches_representation_matched_pyscf():
    pytest.importorskip("pyscf")
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([7.0, 7.0, 7.0]),
        basis="cc-pvdz",
        unit="bohr",
        dimension=3,
    ).build()
    mf = cell.KRHF(kpts=cell.make_kpts((1, 1, 1)))
    mf.gdf_auxbasis = "cc-pvdz-jkfit"
    mf.mo_energy = [np.arange(cell.nao, dtype=float)]
    mf.mo_coeff = [np.eye(cell.nao)]
    mf.mo_occ = [np.r_[2.0, np.zeros(cell.nao - 1)]]
    space = KPointTransitionSpace(mf)

    native = gdf_transition_factors(space, q_index=0, g2_tol=1.0e-14)
    pyscf = pyscf_gdf_transition_factors(space, q_index=0)
    native_pairs = native.pair_blocks[(0, 0)].reshape(native.naux, -1).T
    pyscf_pairs = pyscf.pair_blocks[(0, 0)].reshape(pyscf.naux, -1).T
    native_metric = native_pairs @ native_pairs.conj().T
    pyscf_metric = pyscf_pairs @ pyscf_pairs.conj().T
    relative_error = np.linalg.norm(native_metric - pyscf_metric) / np.linalg.norm(
        pyscf_metric
    )
    assert relative_error < 5.0e-6

    native_j, native_k = gdf_mo_jk(space, coulomb_component=GDF)
    pyscf_j, pyscf_k = gdf_mo_jk(space, coulomb_component=PYSCF_GDF)
    np.testing.assert_allclose(native_j, pyscf_j, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(native_k, pyscf_k, rtol=1.0e-5, atol=1.0e-6)


def test_pyscf_gdf_mirror_preserves_cartesian_orbital_basis():
    pytest.importorskip("pyscf")
    from types import SimpleNamespace

    from pyscf.pbc import gto

    lattice = np.diag([8.0, 8.0, 8.0])
    expected = gto.Cell()
    expected.atom = "C 0 0 0"
    expected.a = lattice
    expected.basis = "def2-svp"
    expected.unit = "B"
    expected.cart = True
    expected.verbose = 0
    expected.build()
    source_cell = SimpleNamespace(
        _atom_symbols=["C"],
        _atom_coords=np.zeros((1, 3)),
        lattice_vectors=lattice,
        basis="def2-svp",
        charge=0,
        spin=0,
        dimension=3,
    )
    reference = SimpleNamespace(cell=source_cell, nao=int(expected.nao_nr()))

    mirrored = _pyscf_cell_from_reference(reference)

    assert mirrored.cart is True
    assert mirrored.nao_nr() == expected.nao_nr()


def test_pyscf_gdf_direct_tdh_uses_pyscf_density_fitting(two_k_h2_reference):
    pytest.importorskip("pyscf")
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    factors = pyscf_gdf_transition_factors(space, q_index=0)
    metric = pyscf_gdf_transition_metric(space, q_index=0)
    response = direct_tdh_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="pyscf_gdf",
    )
    poles = space.screened_interaction(
        q_index=0,
        direct_scale=1.0,
        coulomb_component="pyscf_gdf",
    )
    coupling = pyscf_gdf_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=0,
        kq_index=0,
        left_band=1,
        right_band=1,
    )

    assert factors.coulomb_component == PYSCF_GDF
    assert response.coulomb_component == PYSCF_GDF
    assert poles.coulomb_component == PYSCF_GDF
    assert metric.shape == response.B.shape == (2, 2)
    assert coupling.shape == (2,)
    np.testing.assert_allclose(metric, metric.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        response.B,
        metric / two_k_h2_reference.nkpts,
        atol=1e-12,
    )


def test_attach_pyscf_gdf_context_reuses_compatible_tensors(two_k_h2_reference):
    pytest.importorskip("pyscf")
    source_space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    expected = pyscf_gdf_transition_factors(source_space, q_index=0)
    context = source_space._pyscf_gdf_context
    target_space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")

    returned = attach_pyscf_gdf_context(target_space, context)
    actual = pyscf_gdf_transition_factors(target_space, q_index=0)

    assert returned is target_space
    assert target_space._pyscf_gdf_context is context
    np.testing.assert_allclose(actual.transition_vectors, expected.transition_vectors)


def test_ac_scaled_legendre_roots_match_infinite_interval_jacobian():
    roots, root_weights = np.polynomial.legendre.leggauss(12)
    freqs, weights = _scaled_legendre_roots(12)
    scale = 0.5

    np.testing.assert_allclose(freqs, scale * (1.0 + roots) / (1.0 - roots))
    np.testing.assert_allclose(weights, root_weights * 2.0 * scale / (1.0 - roots) ** 2)
    assert np.all(freqs > 0.0)
    assert np.all(weights > 0.0)


def test_pole_self_energy_accumulator_matches_python_reference():
    omega_eval = np.asarray([0.25, 0.5, 0.8])
    pole_omega = np.asarray([0.3, 1.1])
    weights = np.asarray(
        [
            [0.7, 0.2],
            [0.1, 0.6],
            [0.4, 0.5],
        ]
    )
    eps = np.asarray([-0.4, 0.15, 0.9])
    occupied = np.asarray([True, False, True])

    expected = _accumulate_pole_self_energy_python(
        omega_eval,
        pole_omega,
        weights,
        eps,
        occupied,
        1.0e-3,
    )
    actual = _accumulate_pole_self_energy(
        omega_eval,
        pole_omega,
        weights,
        eps,
        occupied,
        1.0e-3,
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-14)


def test_periodic_gw_ac_pyscf_gdf_matches_pyscf_kernel_same_reference():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto, gw, scf

    auxbasis = "def2-svp-jkfit"
    qcell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = qcell.make_kpts((2, 1, 1))

    pcell = gto.Cell()
    pcell.atom = "H 0 0 0; H 1.4 0 0"
    pcell.a = np.diag([5.0, 5.0, 5.0])
    pcell.basis = "sto-3g"
    pcell.unit = "B"
    pcell.charge = 0
    pcell.spin = 0
    pcell.verbose = 0
    pcell.build()

    pmf = scf.KRHF(pcell, kpts=kpts, exxdiv="ewald").density_fit(auxbasis=auxbasis)
    pmf.conv_tol = 1.0e-10
    pmf.max_cycle = 80
    pmf.verbose = 0
    pmf.kernel()

    qmf = qcell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=0,
        pair_cut=3,
        recip_cut=7,
        jk_builder="ewald",
    )
    qmf.gdf_auxbasis = auxbasis
    qmf.max_memory = 4000
    qmf.mo_energy = [np.asarray(block).copy() for block in pmf.mo_energy]
    qmf.mo_coeff = [np.asarray(block).copy() for block in pmf.mo_coeff]
    qmf.mo_occ = [np.asarray(block).copy() for block in pmf.mo_occ]

    result = diagonal_g0w0(
        KPointTransitionSpace(qmf),
        coulomb_component=PYSCF_GDF,
        direct_scale=1.0,
        linearized=True,
        frequency_integration="ac",
        ac_nw=24,
        finite_size_correction=True,
        finite_size_head_method="pyscf_gradient",
        qp_bands=[0, 1],
    )

    kgw = gw.KGW(pmf, freq_int="ac")
    kgw.linearized = True
    kgw.ac = "twopole"
    kgw.fc = True
    kgw.verbose = 0
    pyscf_qp = kgw.kernel(orbs=[0, 1], kptlist=[0, 1], nw=24)

    np.testing.assert_allclose(
        np.asarray(result.e_qp) * au2ev,
        np.asarray(pyscf_qp) * au2ev,
        atol=1.0e-8,
    )


def test_periodic_gw_ac_fixed_target_and_time_reversal_q_reduction(
    two_k_h2_reference,
    monkeypatch,
):
    from types import SimpleNamespace

    import pyqed.pbc.gw.self_energy as self_energy

    mf = two_k_h2_reference
    recip = 2.0 * np.pi * np.linalg.inv(mf.cell.lattice_vectors).T
    scaled = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [-1.0 / 3.0, 0.0, 0.0],
        ]
    )
    mf.kpts = scaled @ recip
    mf.nkpts = 3
    mf.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.9, 0.6]),
        np.asarray([-0.9, 0.6]),
    ]
    mf.mo_coeff = [np.eye(mf.cell.nao, dtype=np.complex128) for _ in range(3)]
    mf.mo_occ = [np.asarray([2.0, 0.0]) for _ in range(3)]
    mf.dm = [np.diag([2.0, 0.0]).astype(np.complex128) for _ in range(3)]
    space = KPointTransitionSpace(mf, qpts="mesh")

    def fake_factors(space, q_index, _component, _g2_tol):
        q_index = space.normalize_q_index(q_index)
        block = np.asarray([[[0.4, 0.2], [0.2, 0.3]]], dtype=np.complex128)
        pair_blocks = {}
        for k_index in range(space.reference.nkpts):
            kmq_index = space.reference.find_kpoint_index(
                space.reference.kpts[k_index] - space.qpts[q_index]
            )
            pair_blocks[(int(kmq_index), int(k_index))] = block.copy()
        transitions = space.transitions(q_index)
        factors = SimpleNamespace(
            transitions=transitions,
            transition_vectors=np.full((len(transitions), 1), 0.2 + 0.0j),
            pair_blocks=pair_blocks,
        )
        if not hasattr(space, "_gdf_factor_cache"):
            space._gdf_factor_cache = {}
        space._gdf_factor_cache[(q_index, "test")] = factors
        return q_index, factors

    monkeypatch.setattr(self_energy, "_factor_backend_for_ac", fake_factors)
    common = dict(
        direct_scale=1.0,
        coulomb_component=GDF,
        frequency_integration="ac",
        ac_nw=12,
        linearized=True,
    )
    full = diagonal_g0w0(space, qp_bands=[0, 1], **common)
    fixed = diagonal_g0w0(space, qp_bands={0: [0, 1]}, **common)
    prebuilt_factor = object()
    space._gdf_factor_cache[(0, "prebuilt")] = prebuilt_factor
    reduced = diagonal_g0w0(
        space,
        qp_bands={0: [0, 1]},
        q_reduction="time_reversal",
        **common,
    )

    np.testing.assert_allclose(fixed.e_qp[0], full.e_qp[0], atol=1.0e-12)
    np.testing.assert_allclose(reduced.e_qp[0], fixed.e_qp[0], atol=1.0e-12)
    assert space._gdf_factor_cache == {(0, "prebuilt"): prebuilt_factor}
    np.testing.assert_array_equal(fixed.info["target_k_indices"], [0])
    assert fixed.info["evaluated_kpoints"] == 1
    assert reduced.info["q_reduction"] == "time_reversal"
    assert len(reduced.info["q_evaluation_indices"]) == 2
    np.testing.assert_allclose(reduced.info["q_multiplicities"], [1.0, 2.0])

    with pytest.raises(ValueError, match="target k point"):
        diagonal_g0w0(
            space,
            qp_bands={1: [0]},
            q_reduction="time_reversal",
            **common,
        )
    with pytest.raises(NotImplementedError, match="frequency_integration='ac'"):
        diagonal_g0w0(space, q_reduction="time_reversal")


def test_periodic_spectral_function_has_correct_time_ordered_branches(
    two_k_h2_reference,
    monkeypatch,
):
    from types import SimpleNamespace

    import pyqed.pbc.gw.pes as pes

    mf = two_k_h2_reference
    space = KPointTransitionSpace(mf, qpts="mesh")

    def zero_self_energy(_space, k_index, band_index, omega, q_indices=None, **kwargs):
        del kwargs
        normalized_q = _space.normalize_q_indices(q_indices)
        return SimpleNamespace(
            k_index=int(k_index),
            band_index=int(band_index),
            sigma_c=np.zeros_like(np.asarray(omega), dtype=np.complex128),
            q_indices=normalized_q,
            coulomb_component=RECIPROCAL_EWALD_LR,
            finite_size_method=None,
        )

    monkeypatch.setattr(pes, "diagonal_correlation_self_energy", zero_self_energy)
    eta = 0.05
    occupied_binding = np.linspace(0.8, 1.2, 81)
    occupied = periodic_spectral_function(
        space,
        binding_grid=occupied_binding,
        units="au",
        bands={0: [0]},
        energy_reference=0.0,
        eta=eta,
        broadening=eta,
    )

    expected = eta / (
        np.pi * ((occupied.omega - space.reference.mo_energy[0, 0]) ** 2 + eta**2)
    )
    np.testing.assert_allclose(occupied.spectral_function[0], expected, atol=1.0e-12)
    np.testing.assert_allclose(occupied.signal, expected, atol=1.0e-12)
    assert occupied.targets.tolist() == [[0, 0]]
    assert occupied.binding_energies[np.argmax(occupied.signal)] == pytest.approx(1.0)
    peaks = occupied.peaks(source="spectral_function", units="au", max_peaks=1)
    assert peaks.targets.tolist() == [[0, 0]]
    assert peaks.binding_energies[0] == pytest.approx(1.0)

    virtual_omega = np.linspace(0.3, 0.7, 81)
    virtual = periodic_spectral_function(
        space,
        omega_grid=virtual_omega,
        units="au",
        bands={0: [1]},
        occupied_only=False,
        energy_reference=0.0,
        eta=eta,
        broadening=eta,
    )
    expected_virtual = eta / (
        np.pi * ((virtual.omega - space.reference.mo_energy[0, 1]) ** 2 + eta**2)
    )
    np.testing.assert_allclose(
        virtual.spectral_function[0],
        expected_virtual,
        atol=1.0e-12,
    )
    assert virtual.omega[np.argmax(virtual.signal)] == pytest.approx(0.5)

    parallel = periodic_plane_wave_velocity_matrix_elements(
        space.reference,
        np.asarray([[0.7, 0.0, 0.0]]),
        np.asarray([[0, 0]]),
        polarization=(1.0, 0.0, 0.0),
    )
    perpendicular = periodic_plane_wave_velocity_matrix_elements(
        space.reference,
        np.asarray([[0.7, 0.0, 0.0]]),
        np.asarray([[0, 0]]),
        polarization=(0.0, 1.0, 0.0),
    )
    isotropic = periodic_plane_wave_velocity_matrix_elements(
        space.reference,
        np.asarray([[0.7, 0.0, 0.0]]),
        np.asarray([[0, 0]]),
        polarization=None,
    )
    assert abs(parallel[0, 0]) > 0.0
    np.testing.assert_allclose(perpendicular, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.abs(isotropic) ** 2,
        np.abs(parallel) ** 2 / 3.0,
        atol=1.0e-14,
    )

    photoemission = periodic_photoemission_spectrum(
        space.reference,
        occupied,
        photon_energy=3.0,
        work_function=0.1,
        units="au",
        direction=(1.0, 0.0, 0.0),
        polarization=(1.0, 0.0, 0.0),
        surface_normal=(0.0, 0.0, 1.0),
        inner_potential=0.2,
        temperature=300.0,
        energy_resolution=0.02,
        binding_offset=0.0,
        momentum_broadening=0.2,
    )
    np.testing.assert_allclose(
        photoemission.kinetic_energies,
        3.0 - 0.1 - occupied.binding_energies,
        atol=1.0e-14,
    )
    assert photoemission.target_intensity.shape == occupied.spectral_function.shape
    assert photoemission.matrix_elements.shape == occupied.spectral_function.shape
    assert np.all(photoemission.signal >= 0.0)
    assert np.all(photoemission.momentum_weights >= 0.0)
    assert np.all(photoemission.momentum_weights <= 1.0)
    assert photoemission.signal.max() > 0.0
    assert photoemission.info["gauge"] == "velocity"


def test_kgw_periodic_spectral_function_reuses_exact_pole_cache(two_k_h2_reference):
    mf = two_k_h2_reference
    driver = KGW(mf, eta=0.05).run(
        backend="periodic",
        q_indices=[0],
        direct_scale=1.0,
        coulomb_component=RECIPROCAL_EWALD_LR,
        qp_bands={0: [0]},
    )
    run_space = driver._periodic_space
    cache_sizes_before = driver._periodic_cache.sizes()
    result = driver.spectral_function(
        binding_grid=np.linspace(0.5, 1.5, 31),
        units="au",
        bands={0: [0]},
        energy_reference=0.0,
        finite_size_correction=False,
    )

    assert driver.spectral_result is result
    assert driver._periodic_space is run_space
    assert result.spectral_function.shape == (1, 31)
    assert result.sigma_c.shape == result.green_function.shape == (1, 31)
    assert np.all(np.isfinite(result.spectral_function))
    assert np.all(result.spectral_function >= 0.0)
    assert result.signal.max() > 0.0
    assert result.info["q_indices"].tolist() == [0]
    assert result.info["cache_sizes"]["screened_interactions"] == 1
    assert result.info["cache_sizes"]["screened_interactions"] == (
        cache_sizes_before["screened_interactions"]
    )

    photoemission = driver.experimental_pes(
        spectral_result=result,
        photon_energy=3.0,
        work_function=0.1,
        units="au",
        direction=(1.0, 0.0, 0.0),
        polarization=(1.0, 0.0, 0.0),
        binding_offset=0.0,
    )
    assert driver.photoemission_result is photoemission
    assert photoemission.signal.shape == result.binding_energies.shape
    assert np.all(np.isfinite(photoemission.signal))


def test_gamma_direct_rpa_roots_are_finite(gamma_h2_mf):
    gw = KGW(gamma_h2_mf)
    from_space = direct_rpa(gw.transition_space(qpts="gamma"), q_index=0)
    from_gw = gw.rpa(q_index=0, qpts="gamma")

    assert from_space.omega.shape == (1,)
    assert from_space.vectors.shape == (1, 1)
    assert np.all(np.isfinite(from_space.omega))
    assert np.all(from_space.omega >= -1e-12)
    np.testing.assert_allclose(from_gw.omega, from_space.omega, atol=1e-12)


def test_two_k_direct_tdh_response_shapes(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    response = space.tdh_matrices(q_index=1, direct_scale=1.0)
    rpa = space.rpa(q_index=1, direct_scale=1.0)

    assert response.A.shape == (2, 2)
    assert response.B.shape == (2, 2)
    assert response.direct_scale == 1.0
    assert response.g2_tol == 1.0e-16
    assert response.thresh is None
    assert rpa.direct_scale == 1.0
    assert rpa.g2_tol == 1.0e-16
    assert rpa.thresh == 1.0e-10
    np.testing.assert_allclose(response.A, response.A.conj().T, atol=1e-12)
    np.testing.assert_allclose(response.B, response.B.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        response.A - response.B,
        np.diag(response.transition_energy),
        atol=1e-12,
    )
    assert rpa.omega.shape == (2,)
    assert np.all(np.isfinite(rpa.omega))


def test_gamma_screened_interaction_poles_match_rpa_block(gamma_h2_mf):
    gw = KGW(gamma_h2_mf)
    space = gw.transition_space(qpts="gamma")
    response = space.rpa(q_index=0)
    poles = screened_interaction_poles(space, q_index=0)
    from_gw = gw.screened_interaction(q_index=0, qpts="gamma")

    assert poles.q_index == 0
    assert poles.coulomb_component == "reciprocal_ewald_lr"
    assert poles.direct_scale == 2.0
    assert poles.g2_tol == 1.0e-16
    assert poles.thresh == 1.0e-10
    assert poles.ntransitions == 1
    assert poles.nmodes == 1
    np.testing.assert_allclose(poles.omega, response.omega, atol=1e-12)
    np.testing.assert_allclose(from_gw.coupling, poles.coupling, atol=1e-12)

    expected_abs = (
        abs(poles.kernel_coupling[0, 0])
        * np.sqrt(poles.transition_energy[0] / poles.omega[0])
    )
    np.testing.assert_allclose(abs(poles.coupling[0, 0]), expected_abs, atol=1e-12)

    residue = poles.mode_residue(0)
    np.testing.assert_allclose(residue, residue.conj().T, atol=1e-12)
    assert residue[0, 0].real >= 0.0
    assert poles.normalize_mode_index(0) == 0
    with pytest.raises(TypeError, match="mode"):
        poles.mode_residue(0.5)
    with pytest.raises(IndexError, match="mode"):
        poles.mode_residue(-1)
    with pytest.raises(IndexError, match="mode"):
        poles.mode_residue(poles.nmodes)


def test_orbital_pair_coupling_matches_transition_row(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    transition = space.transitions(0)[0]
    transition_factors = space.reciprocal_factors(0)
    pair_factors = reciprocal_orbital_pair_factors(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
    )
    poles = space.screened_interaction(q_index=0)

    bare_coupling = pair_factors.coulomb_coupling(transition_factors)
    np.testing.assert_allclose(
        bare_coupling,
        poles.bare_coulomb[:, 0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        poles.coupling_for_coulomb_vector(bare_coupling),
        poles.coupling[0],
        atol=1e-12,
    )


def test_two_k_screened_interaction_poles_are_hermitian(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    poles = space.screened_interaction(q_index=1, direct_scale=1.0)

    assert poles.coupling.shape == (2, 2)
    assert poles.kernel_coupling.shape == (2, 2)
    assert poles.direct_scale == 1.0
    assert poles.g2_tol == 1.0e-16
    assert poles.thresh == 1.0e-10
    assert np.all(np.isfinite(poles.omega))
    assert np.all(np.isfinite(poles.coupling))

    residue = poles.residue_metric()
    np.testing.assert_allclose(residue, residue.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(residue).real) >= -1e-10


def test_multi_k_direct_tdh_uses_kpoint_transition_weights(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    q_index = 1
    response = direct_tdh_matrices(space, q_index=q_index, direct_scale=1.0)
    metric = reciprocal_transition_factors(space, q_index).coulomb_metric()
    expected = metric / two_k_h2_reference.nkpts

    np.testing.assert_allclose(
        response.transition_weights,
        np.full(len(space.transitions(q_index)), 1.0 / two_k_h2_reference.nkpts),
    )
    np.testing.assert_allclose(
        response.A - np.diag(response.transition_energy),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(response.B, expected, atol=1.0e-12)


def test_multi_k_mode_coupling_uses_external_pair_kpoint_weight(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    q_index = 1
    transition = space.transitions(q_index)[0]
    transition_factors = reciprocal_transition_factors(space, q_index)
    pair_factors = reciprocal_orbital_pair_factors(
        space,
        q_index=q_index,
        k_index=transition.k_index,
        kq_index=transition.kq_index,
        left_band=transition.occ_band,
        right_band=transition.vir_band,
    )
    poles = screened_interaction_poles(space, q_index=q_index, direct_scale=1.0)

    bare_coupling = pair_factors.coulomb_coupling(transition_factors)
    expected = (
        np.sqrt(poles.transition_weights) * bare_coupling
    ).conj() @ poles.mode_projector

    np.testing.assert_allclose(
        poles.coupling_for_coulomb_vector(bare_coupling),
        expected,
        atol=1.0e-12,
    )


def test_gamma_diagonal_correlation_self_energy_is_finite(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, eta=1.0e-3)
    space = gw.transition_space(qpts="gamma")
    omega = gamma_h2_mf.mo_energy[0] + np.asarray([-0.1, 0.1])

    sigma = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=0,
        omega=omega,
        eta=gw.eta,
    )
    from_gw = gw.sigma_c(
        k_index=0,
        band_index=0,
        omega=float(omega[0]),
        qpts="gamma",
    )

    assert sigma.sigma_c.shape == (2,)
    assert sigma.q_contributions.shape == (1, 2)
    assert np.all(np.isfinite(sigma.sigma_c))
    np.testing.assert_allclose(
        np.sum(sigma.q_contributions, axis=0),
        sigma.sigma_c,
        atol=1e-12,
    )
    assert from_gw.sigma_c.shape == ()
    assert np.isfinite(from_gw.value())


def test_gamma_diagonal_self_energy_can_use_dense_full_ewald_couplings(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    omega = float(space.reference.mo_energy[0, 0])
    reciprocal = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=0,
        omega=omega,
        direct_scale=1.0,
        coulomb_component="reciprocal_ewald_lr",
        eta=1.0e-3,
    )
    full = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=0,
        omega=omega,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        eta=1.0e-3,
    )
    result = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    assert full.coulomb_component == "full_ewald"
    assert result.info["coulomb_component"] == "full_ewald"
    assert result.info["all_converged"]
    assert full.sigma_c.shape == ()
    assert np.isfinite(full.sigma_c)
    assert np.all(np.isfinite(result.e_qp))
    assert not np.allclose(full.sigma_c, reciprocal.sigma_c)


def test_two_k_diagonal_correlation_self_energy_accumulates_q_blocks(two_k_h2_reference):
    gw = KGW(two_k_h2_reference, eta=1.0e-3)
    space = gw.transition_space(qpts="mesh")
    omega = np.asarray([0.4, 0.6])

    sigma = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=omega,
        eta=gw.eta,
        direct_scale=1.0,
    )
    from_gw = gw.sigma_c(
        k_index=0,
        band_index=1,
        omega=omega,
        direct_scale=1.0,
    )

    np.testing.assert_array_equal(sigma.q_indices, [0, 1])
    assert sigma.eta == gw.eta
    assert sigma.direct_scale == 1.0
    assert sigma.g2_tol == 1.0e-16
    assert sigma.thresh == 1.0e-10
    assert sigma.average_q is True
    assert sigma.sigma_c.shape == (2,)
    assert sigma.q_contributions.shape == (2, 2)
    assert np.all(np.isfinite(sigma.sigma_c))
    np.testing.assert_allclose(
        np.sum(sigma.q_contributions, axis=0),
        sigma.sigma_c,
        atol=1e-12,
    )
    np.testing.assert_allclose(from_gw.sigma_c, sigma.sigma_c, atol=1e-12)


def test_diagonal_self_energy_cache_reuses_q_resolved_screening(two_k_h2_reference):
    space = KGW(two_k_h2_reference, eta=1.0e-3).transition_space(qpts="mesh")
    cache = DiagonalSelfEnergyCache()

    first = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=np.asarray([0.4, 0.6]),
        eta=1.0e-3,
        direct_scale=1.0,
        cache=cache,
    )
    sizes_after_first = cache.sizes()
    second = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=np.asarray([0.4, 0.6]),
        eta=1.0e-3,
        direct_scale=1.0,
        cache=cache,
    )

    assert sizes_after_first["screened_interactions"] == space.nqpts
    assert sizes_after_first["transition_factors"] == space.nqpts
    assert sizes_after_first["mode_couplings"] > 0
    assert cache.sizes() == sizes_after_first
    np.testing.assert_allclose(second.sigma_c, first.sigma_c, atol=1e-12)


def test_diagonal_g0w0_can_prebuild_screening_with_workers(two_k_h2_reference):
    space = KGW(two_k_h2_reference, eta=1.0e-3).transition_space(qpts="mesh")

    result = diagonal_g0w0(
        space,
        q_indices=[0, 1],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
        eta=1.0e-3,
        prebuild_screening=True,
        screening_workers=2,
    )

    assert len(result.info["screening_prebuild"]) == 2
    assert result.info["screening_prebuild_seconds"] >= 0.0
    assert result.info["cache_sizes"]["screened_interactions"] == 2
    assert {
        row["screening_workers"]
        for row in result.info["screening_prebuild"]
    } == {2}
    assert all(row["screening_parallel"] for row in result.info["screening_prebuild"])


def test_diagonal_g0w0_can_evaluate_target_bands_with_workers(two_k_h2_reference):
    space = KGW(two_k_h2_reference, eta=1.0e-3).transition_space(qpts="mesh")
    common = dict(
        q_indices=[0, 1],
        qp_bands=[0, 1],
        intermediate_bands=[0],
        direct_scale=1.0,
        eta=1.0e-3,
        prebuild_screening=True,
        screening_workers=2,
    )

    serial = diagonal_g0w0(space, target_workers=1, **common)
    parallel = diagonal_g0w0(space, target_workers=2, **common)

    assert parallel.info["target_workers"] == 2
    assert parallel.info["target_parallel"]
    assert parallel.info["cache_sizes"]["screened_interactions"] == 2
    np.testing.assert_allclose(parallel.e_qp, serial.e_qp, atol=1.0e-12)
    np.testing.assert_allclose(parallel.sigma_c, serial.sigma_c, atol=1.0e-12)


def test_kgw_periodic_can_prebuild_screening(two_k_h2_reference):
    gw = KGW(two_k_h2_reference, eta=1.0e-3).run(
        q_indices=[0, 1],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
        prebuild_screening=True,
        screening_workers=2,
    )

    assert gw.periodic_backend
    assert len(gw.info["screening_prebuild"]) == 2
    assert gw.info["cache_sizes"]["screened_interactions"] == 2


def test_kgw_periodic_can_use_target_workers(two_k_h2_reference):
    gw = KGW(two_k_h2_reference, eta=1.0e-3).run(
        q_indices=[0, 1],
        qp_bands=[0, 1],
        intermediate_bands=[0],
        direct_scale=1.0,
        prebuild_screening=True,
        screening_workers=2,
        target_workers=2,
    )

    assert gw.periodic_backend
    assert gw.info["target_workers"] == 2
    assert gw.info["target_parallel"]


def test_diagonal_self_energy_can_truncate_intermediate_band_sum(two_k_h2_reference):
    space = KGW(two_k_h2_reference, eta=1.0e-3).transition_space(qpts="mesh")

    full = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=np.asarray([0.4, 0.6]),
        eta=1.0e-3,
        direct_scale=1.0,
    )
    truncated = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=np.asarray([0.4, 0.6]),
        eta=1.0e-3,
        direct_scale=1.0,
        intermediate_bands=[0],
    )
    k_specific = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        q_indices=[0],
        eta=1.0e-3,
        direct_scale=1.0,
        intermediate_bands={0: [0]},
    )

    assert truncated.intermediate_bands == (0,)
    assert k_specific.intermediate_bands == {0: (0,)}
    assert truncated.sigma_c.shape == full.sigma_c.shape
    assert not np.allclose(truncated.sigma_c, full.sigma_c)
    assert np.isfinite(k_specific.sigma_c)

    with pytest.raises(IndexError, match="out-of-range band"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1,
            omega=0.4,
            intermediate_bands=[99],
        )
    with pytest.raises(TypeError, match="intermediate_bands"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1,
            omega=0.4,
            intermediate_bands=[0.5],
        )
    with pytest.raises(TypeError, match="intermediate_bands"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1,
            omega=0.4,
            intermediate_bands={0.5: [0]},
        )
    with pytest.raises(IndexError, match="out-of-range k"):
        diagonal_correlation_self_energy(
            space,
            k_index=0,
            band_index=1,
            omega=0.4,
            intermediate_bands={99: [0]},
        )


def test_two_k_diagonal_self_energy_can_use_full_ewald_couplings(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    sigma = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=np.asarray([0.4, 0.6]),
        direct_scale=1.0,
        coulomb_component="full_ewald",
        eta=1.0e-3,
    )
    result = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    assert sigma.coulomb_component == "full_ewald"
    assert result.info["coulomb_component"] == "full_ewald"
    assert sigma.sigma_c.shape == (2,)
    assert result.e_qp.shape == (2, 2)
    assert np.all(np.isfinite(sigma.sigma_c))
    assert np.all(np.isfinite(result.e_qp))
    assert result.info["all_converged"]


def test_two_k_diagonal_g0w0_result_tracks_on_shell_sigma(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    result = diagonal_g0w0(space, eta=1.0e-3, direct_scale=1.0)

    assert result.e_mf.shape == (2, 2)
    assert result.e_qp.shape == (2, 2)
    assert result.sigma_c.shape == (2, 2)
    assert result.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert result.info["eta"] == 1.0e-3
    assert result.info["direct_scale"] == 1.0
    assert result.info["g2_tol"] == 1.0e-16
    assert result.info["thresh"] == 1.0e-10
    assert result.info["finite_size_correction"] is False
    np.testing.assert_array_equal(result.info["q_indices"], [0, 1])
    assert np.all(result.converged)
    np.testing.assert_allclose(
        result.e_qp,
        result.e_mf + result.sigma_c.real,
        atol=1e-12,
    )


def test_two_k_diagonal_g0w0_linearized_update_matches_manual_z(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    step = 1.0e-5
    result = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        linearized=True,
        linearized_step=step,
        qp_bands={0: [1]},
    )
    eps = float(result.e_mf[0, 1])
    sigma = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=eps,
        eta=1.0e-3,
        direct_scale=1.0,
    ).value()
    sigma_shifted = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=eps + step,
        eta=1.0e-3,
        direct_scale=1.0,
    ).value()
    derivative = (sigma_shifted.real - sigma.real) / step
    expected = eps + sigma.real / (1.0 - derivative)

    assert result.info["linearized"] is True
    assert result.info["linearized_step"] == step
    assert result.info["solve_roots"] is False
    np.testing.assert_allclose(result.e_qp[0, 1], expected, atol=1.0e-12)


def test_two_k_diagonal_g0w0_can_target_qp_bands(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    full = diagonal_g0w0(space, eta=1.0e-3, direct_scale=1.0)
    cache = DiagonalSelfEnergyCache()

    targeted = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        qp_bands=[1],
        cache=cache,
    )
    np.testing.assert_array_equal(
        targeted.info["target_mask"],
        np.asarray([[False, True], [False, True]]),
    )
    assert targeted.info["qp_bands"] == (1,)
    assert targeted.info["target_bands"] == ((0, 1), (1, 1))
    assert targeted.info["nqp"] == 2
    assert targeted.info["cache_sizes"] == cache.sizes()
    assert targeted.info["cache_sizes"]["screened_interactions"] == space.nqpts
    assert targeted.info["all_converged"]
    np.testing.assert_allclose(targeted.e_qp[:, 1], full.e_qp[:, 1], atol=1e-12)
    np.testing.assert_allclose(targeted.e_qp[:, 0], targeted.e_mf[:, 0], atol=1e-12)
    assert np.all(targeted.converged[:, 1])
    assert not np.any(targeted.converged[:, 0])
    assert np.all(np.isfinite(targeted.sigma_c[:, 1]))
    assert np.all(np.isnan(targeted.sigma_c[:, 0].real))

    k_specific = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        qp_bands={1: [0]},
    )
    assert k_specific.info["qp_bands"] == {1: (0,)}
    assert k_specific.info["target_bands"] == ((1, 0),)
    np.testing.assert_array_equal(
        k_specific.info["target_mask"],
        np.asarray([[False, False], [True, False]]),
    )
    assert k_specific.converged[1, 0]
    assert not k_specific.converged[0, 0]

    with pytest.raises(IndexError, match="out-of-range band"):
        diagonal_g0w0(space, qp_bands=[99])
    with pytest.raises(TypeError, match="qp_bands"):
        diagonal_g0w0(space, qp_bands=[1.5])
    with pytest.raises(TypeError, match="qp_bands"):
        diagonal_g0w0(space, qp_bands={0.5: [1]})
    with pytest.raises(IndexError, match="out-of-range k"):
        diagonal_g0w0(space, qp_bands={99: [0]})


def test_two_k_diagonal_g0w0_can_truncate_intermediate_bands(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    full = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        qp_bands=[1],
    )
    truncated = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        qp_bands=[1],
        intermediate_bands=[0],
    )

    assert truncated.info["intermediate_bands"] == (0,)
    assert truncated.info["nqp"] == 2
    assert truncated.info["all_converged"]
    np.testing.assert_allclose(truncated.e_qp[:, 0], truncated.e_mf[:, 0], atol=1e-12)
    assert not np.allclose(truncated.e_qp[:, 1], full.e_qp[:, 1])

    gw = KGW(two_k_h2_reference, eta=1.0e-3).run(
        direct_scale=1.0,
        qp_bands=[1],
        intermediate_bands=[0],
    )
    assert gw.info["intermediate_bands"] == (0,)
    np.testing.assert_allclose(gw.e_qp, truncated.e_qp, atol=1e-12)


def test_two_k_diagonal_evgw_records_cycles(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    result = diagonal_evgw(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        max_cycle=2,
        conv_tol=0.0,
        damping=0.5,
        update_screening=False,
        solve_roots=False,
    )

    assert result.info["backend"] == "kpoint_diagonal_evgw_direct_rpa"
    assert result.info["method"] == "evgw"
    assert result.info["update_screening"] is False
    assert result.info["finite_size_correction"] is False
    assert result.info["cycles"] == 2
    assert result.info["all_converged"] is False
    assert len(result.history) == 2
    assert result.e_qp.shape == (2, 2)
    assert result.sigma_c.shape == (2, 2)
    assert np.all(result.converged)
    assert np.all(np.isfinite(result.e_qp))
    np.testing.assert_allclose(result.history[-1]["energy"], result.e_qp, atol=1e-12)


def test_periodic_gw_iteration_counts_are_validated(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    with pytest.raises(TypeError, match="maxiter"):
        diagonal_g0w0(
            space,
            direct_scale=1.0,
            solve_roots=True,
            maxiter=1.5,
            qp_bands=[1],
        )
    with pytest.raises(ValueError, match="linearized"):
        diagonal_g0w0(
            space,
            direct_scale=1.0,
            linearized=True,
            solve_roots=True,
        )
    with pytest.raises(ValueError, match="linearized_step"):
        diagonal_g0w0(
            space,
            direct_scale=1.0,
            linearized=True,
            linearized_step=0.0,
        )
    with pytest.raises(TypeError, match="max_cycle"):
        diagonal_evgw(
            space,
            direct_scale=1.0,
            max_cycle=1.5,
            solve_roots=False,
        )
    with pytest.raises(ValueError, match="max_cycle"):
        diagonal_evgw(
            space,
            direct_scale=1.0,
            max_cycle=0,
            solve_roots=False,
        )
    with pytest.raises(TypeError, match="maxiter"):
        KGW(two_k_h2_reference).evgw(
            direct_scale=1.0,
            max_cycle=1,
            solve_roots=True,
            maxiter=1.5,
            qp_bands=[1],
        )


def test_periodic_gw_finite_size_correction_adds_head_and_wing(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    bare = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
    )
    finite = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
        finite_size_correction=True,
    )
    direct = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
    )

    assert finite.finite_size_correction is True
    np.testing.assert_allclose(
        finite.finite_size_sigma,
        finite.finite_size_head + finite.finite_size_wing,
        atol=1e-12,
    )
    np.testing.assert_allclose(finite.finite_size_sigma, direct.sigma_c, atol=1e-12)
    np.testing.assert_allclose(finite.value(), bare.value() + direct.sigma_c, atol=1e-12)

    result = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        finite_size_correction=True,
    )
    assert result.info["finite_size_correction"] is True
    assert result.info["finite_size_sigma"].shape == result.e_mf.shape
    np.testing.assert_allclose(
        result.info["finite_size_sigma"],
        result.info["finite_size_head"] + result.info["finite_size_wing"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.e_qp,
        result.e_mf + result.sigma_c.real,
        atol=1e-12,
    )

    evgw = diagonal_evgw(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        max_cycle=1,
        solve_roots=False,
        finite_size_correction=True,
    )
    assert evgw.info["finite_size_correction"] is True
    assert evgw.info["finite_size_sigma"].shape == evgw.e_mf.shape

    gw = KGW(two_k_h2_reference, eta=1.0e-3).run(
        direct_scale=1.0,
        finite_size_correction=True,
    )
    assert gw.info["finite_size_correction"] is True
    np.testing.assert_allclose(gw.e_qp, result.e_qp, atol=1e-12)

    with pytest.raises(NotImplementedError, match="reciprocal_ewald_lr"):
        diagonal_g0w0(
            space,
            coulomb_component="full_ewald",
            finite_size_correction=True,
        )


def test_periodic_gw_pyscf_gdf_finite_size_uses_pyscf_convention(two_k_h2_reference):
    pytest.importorskip("pyscf")
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    bare = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="pyscf_gdf",
    )
    finite = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="pyscf_gdf",
        finite_size_correction=True,
    )
    direct = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        coulomb_component="pyscf_gdf",
    )

    assert finite.coulomb_component == PYSCF_GDF
    assert finite.finite_size_method.startswith("small_sphere_head_wing:pyscf_gdf")
    np.testing.assert_allclose(finite.finite_size_sigma, direct.sigma_c, atol=1e-12)
    np.testing.assert_allclose(finite.value(), bare.value() + direct.sigma_c, atol=1e-12)
    assert direct.sigma_c.real < 0.0

    result = diagonal_g0w0(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="pyscf_gdf",
        finite_size_correction=True,
        qp_bands=[1],
    )
    assert result.info["coulomb_component"] == PYSCF_GDF
    assert result.info["finite_size_correction"] is True
    assert result.info["finite_size_method"].startswith("small_sphere_head_wing:pyscf_gdf")
    assert result.info["finite_size_sigma"][0, 1].real < 0.0


def test_periodic_gw_builtin_gradient_finite_size_matches_pyscf_gradient(two_k_h2_reference):
    pytest.importorskip("pyscf")
    two_k_h2_reference.pair_cut = 3
    two_k_h2_reference._periodic_setup()
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    builtin = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        coulomb_component="pyscf_gdf",
        head_method="builtin_gradient",
    )
    pyscf = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        coulomb_component="pyscf_gdf",
        head_method="pyscf_gradient",
    )
    finite_q = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        coulomb_component="pyscf_gdf",
        head_method="finite_q",
    )

    assert builtin.method.endswith("builtin_gradient")
    np.testing.assert_allclose(builtin.sigma_c, pyscf.sigma_c, rtol=5.0e-4, atol=1.0e-8)
    assert abs(finite_q.sigma_c - pyscf.sigma_c) > 5.0 * abs(pyscf.sigma_c)


def test_periodic_gw_gdf_finite_size_uses_vector_convention(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    bare = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="gdf",
    )
    finite = diagonal_correlation_self_energy(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        eta=1.0e-3,
        direct_scale=1.0,
        coulomb_component="gdf",
        finite_size_correction=True,
    )
    direct = diagonal_finite_size_correction(
        space,
        k_index=0,
        band_index=1,
        omega=0.4,
        coulomb_component="gdf",
    )

    assert finite.coulomb_component == GDF
    assert finite.finite_size_method.startswith("small_sphere_head_wing:gdf")
    np.testing.assert_allclose(finite.finite_size_sigma, direct.sigma_c, atol=1e-12)
    np.testing.assert_allclose(finite.value(), bare.value() + direct.sigma_c, atol=1e-12)


def test_diagonal_g0w0_gdf_can_prebuild_q_ao_store(two_k_h2_reference):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KGW(mf).transition_space(qpts="mesh")

    result = diagonal_g0w0(
        space,
        coulomb_component="gdf",
        prebuild_gdf=True,
        q_indices=[0],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
        eta=1.0e-3,
    )

    assert result.info["coulomb_component"] == GDF
    assert len(result.info["gdf_prebuild"]) == 1
    assert result.info["gdf_prebuild"][0]["q_index"] == 0
    assert result.info["gdf_prebuild"][0]["timings"]["q_ao_store_cache_misses"] == 1
    assert result.info["gdf_prebuild_seconds"] >= 0.0
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 1


def test_diagonal_g0w0_prebuild_reuses_persistent_scf_gdf(two_k_h2_reference):
    mf = two_k_h2_reference.density_fit(
        auxbasis="def2-svp-jkfit",
        precision=1.0e-6,
        mesh="auto",
        omega="auto",
        pair_cut="auto",
        image_cut="auto",
        storage="memory",
    )
    try:
        mf.with_df.build(q_indices=[0], workers=1)
        space = KGW(mf).transition_space(qpts="mesh")
        q_index = space.find_qpoint_index(np.zeros(3))

        result = diagonal_g0w0(
            space,
            coulomb_component="gdf",
            prebuild_gdf=True,
            q_indices=[q_index],
            qp_bands=[1],
            intermediate_bands=[0],
            direct_scale=1.0,
            eta=1.0e-3,
        )

        assert result.info["gdf_prebuild_backend"] == "persistent_ao"
        assert len(result.info["gdf_prebuild"]) == 1
        timing = result.info["gdf_prebuild"][0]["timings"]
        assert timing["persistent_backend_reuse"] is True
        assert "q_ao_store_cache_misses" not in timing
        assert result.info["gdf_prebuild_seconds"] >= 0.0
        assert result.info["cache_sizes"]["transition_factors"] == 1
    finally:
        mf.with_df.close()


def test_gdf_g0w0_band_path_matches_mesh_and_reciprocal_periodicity():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        precision=1.0e-8,
        storage="memory",
        stream_pairs=True,
    )
    try:
        mf.with_df.build(workers=2)
        mf.run(max_cycle=40, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
        assert mf.converged
        gw = KGW(mf, eta=1.0e-3).g0w0(
            backend="periodic",
            coulomb_component="gdf",
            direct_scale=1.0,
            qp_bands=[0, 1],
            intermediate_bands=[0, 1],
            prebuild_screening=True,
        )
        cache_snapshot = {
            name: set(value)
            for name, value in vars(mf).items()
            if name.startswith("_pbc_gdf_") and isinstance(value, dict)
        }

        mesh = gw.band_structure(
            kpts=mf.kpts,
            qp_bands=[0, 1],
            intermediate_bands=[0, 1],
            reference="none",
            pair_workers=2,
        )
        np.testing.assert_allclose(mesh["mo_energy"], mf.mo_energy, atol=1.0e-12)
        np.testing.assert_allclose(mesh["qp_energy"], gw.e_qp, atol=1.0e-12)
        np.testing.assert_allclose(
            mesh["sigma_c"],
            gw.g0w0_result.sigma_c,
            atol=1.0e-12,
        )

        off_mesh = gw.band_structure(
            scaled_kpts=np.asarray([[0.125, 0.0, 0.0], [1.125, 0.0, 0.0]]),
            qp_bands=[0, 1],
            intermediate_bands=[0, 1],
            reference="none",
            pair_workers=2,
        )
        np.testing.assert_allclose(
            off_mesh["mo_energy"][0],
            off_mesh["mo_energy"][1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            off_mesh["qp_energy"][0],
            off_mesh["qp_energy"][1],
            atol=1.0e-12,
        )
        assert np.all(np.isfinite(off_mesh["qp_energy"]))
        assert off_mesh["info"]["closure_kpoints"] == 4
        assert off_mesh["info"]["requested_pairs"] == 4
        assert off_mesh["info"]["gdf_pair_build"]["qpoints"] == 2
        assert not off_mesh["interpolated"]
        assert {
            name: set(value)
            for name, value in vars(mf).items()
            if name.startswith("_pbc_gdf_") and isinstance(value, dict)
        } == cache_snapshot
        assert not mf.with_df._disk_maps
    finally:
        mf.with_df.close()


def test_diagonal_g0w0_gdf_prebuild_can_use_workers(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0
    space = KGW(mf).transition_space(qpts="mesh")

    result = diagonal_g0w0(
        space,
        coulomb_component="gdf",
        prebuild_gdf=True,
        prebuild_gdf_workers=2,
        q_indices=[0, 1],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
        eta=1.0e-3,
    )

    assert len(result.info["gdf_prebuild"]) == 2
    assert {
        row["timings"]["prebuild_workers"]
        for row in result.info["gdf_prebuild"]
    } == {2}
    assert all(
        row["timings"]["prebuild_parallel"]
        for row in result.info["gdf_prebuild"]
    )


def test_kgw_periodic_gdf_can_prebuild_q_ao_store(two_k_h2_reference):
    from pyqed.pbc.gw.integrals import _gdf_mf_cache

    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0

    gw = KGW(mf, eta=1.0e-3).run(
        coulomb_component="gdf",
        prebuild_gdf=True,
        q_indices=[0],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
    )

    assert gw.periodic_backend
    assert gw.info["coulomb_component"] == GDF
    assert len(gw.info["gdf_prebuild"]) == 1
    assert gw.info["gdf_prebuild"][0]["q_index"] == 0
    assert gw.info["gdf_prebuild"][0]["timings"]["q_ao_store_cache_misses"] == 1
    assert gw.info["gdf_prebuild_seconds"] >= 0.0
    assert len(_gdf_mf_cache(mf, "q_ao_store")) == 1


def test_kgw_periodic_evgw_gdf_can_prebuild_q_ao_store(two_k_h2_reference):
    mf = two_k_h2_reference
    mf.gdf_auxbasis = "sto-3g"
    mf.gdf_pair_cut = 0
    mf.gdf_recip_cut = 2
    mf.gdf_g_block_size = 2
    mf.gdf_pair_ft_screen_tol = 0.0

    gw = KGW(mf, eta=1.0e-3).evgw(
        coulomb_component="gdf",
        prebuild_gdf=True,
        q_indices=[0],
        qp_bands=[1],
        intermediate_bands=[0],
        direct_scale=1.0,
        max_cycle=1,
        solve_roots=False,
    )

    assert gw.periodic_backend
    assert gw.info["coulomb_component"] == GDF
    assert len(gw.info["gdf_prebuild"]) == 1
    assert gw.info["gdf_prebuild"][0]["q_index"] == 0


def test_two_k_diagonal_evgw_can_target_qp_bands(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    result = diagonal_evgw(
        space,
        eta=1.0e-3,
        direct_scale=1.0,
        max_cycle=1,
        conv_tol=0.0,
        update_screening=False,
        solve_roots=False,
        qp_bands=[1],
    )

    assert result.info["qp_bands"] == (1,)
    assert result.info["target_bands"] == ((0, 1), (1, 1))
    assert result.info["nqp"] == 2
    assert result.info["eta"] == 1.0e-3
    assert result.info["direct_scale"] == 1.0
    assert result.info["g2_tol"] == 1.0e-16
    assert result.info["thresh"] == 1.0e-10
    assert result.info["max_cycle"] == 1
    assert result.info["conv_tol"] == 0.0
    assert result.info["damping"] == 1.0
    np.testing.assert_array_equal(result.info["q_indices"], [0, 1])
    assert result.info["all_converged"] is False
    assert np.all(result.converged[:, 1])
    assert not np.any(result.converged[:, 0])
    np.testing.assert_allclose(result.e_qp[:, 0], result.e_mf[:, 0], atol=1e-12)
    assert np.all(np.isnan(result.sigma_c[:, 0].real))


def test_two_k_kgw_evgw_and_gnw0_drivers(two_k_h2_reference):
    evgw = KGW(two_k_h2_reference, eta=1.0e-3).evgw(
        direct_scale=1.0,
        max_cycle=1,
        conv_tol=0.0,
        damping=0.5,
        solve_roots=False,
    )
    gnw0 = KGW(two_k_h2_reference, eta=1.0e-3).gnw0(
        direct_scale=1.0,
        max_cycle=1,
        conv_tol=0.0,
        damping=0.5,
        solve_roots=False,
    )

    assert evgw.info["method"] == "evgw"
    assert evgw.method == "evgw"
    assert evgw.info["update_screening"] is True
    assert evgw.info["cycles"] == 1
    assert len(evgw.evgw_history) == 1
    assert gnw0.info["method"] == "gnw0"
    assert gnw0.method == "gnw0"
    assert gnw0.info["update_screening"] is False
    assert gnw0.info["cycles"] == 1
    assert evgw.e_qp.shape == gnw0.e_qp.shape == (2, 2)
    assert np.all(np.isfinite(evgw.e_qp))
    assert np.all(np.isfinite(gnw0.e_qp))

    tda = KTDA(evgw).run(q_index=0, direct_scale=1.0, nroots=1, return_vectors=False)
    assert tda.e.shape == (1,)
    assert np.all(np.isfinite(tda.e))


def test_kgw_reused_driver_clears_stale_periodic_results(two_k_h2_reference):
    gw = KGW(two_k_h2_reference, eta=1.0e-3).evgw(
        direct_scale=1.0,
        max_cycle=1,
        conv_tol=0.0,
        damping=0.5,
        solve_roots=False,
    )
    assert gw.evgw_result is not None
    assert gw.g0w0_result is None
    assert len(gw.evgw_history) == 1

    gw.g0w0(direct_scale=1.0)

    assert gw.method == "g0w0"
    assert gw.g0w0_result is not None
    assert gw.evgw_result is None
    assert gw.evgw_history == []


def test_two_k_periodic_bse_matrices_are_hermitian(two_k_h2_reference):
    gw = KGW(two_k_h2_reference).run(direct_scale=1.0)
    space = gw.transition_space(qpts="mesh")
    block = periodic_bse_matrices(
        space,
        q_index=0,
        qp_energy=gw.e_qp,
        direct_scale=1.25,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        g2_tol=1.0e-14,
        thresh=1.0e-9,
    )

    assert block.A.shape == (2, 2)
    assert block.B.shape == (2, 2)
    assert block.coulomb_component == "reciprocal_ewald_lr"
    assert block.direct_scale == 1.25
    assert block.exchange_scale == 0.25
    assert block.screened_exchange_scale == 0.75
    assert block.g2_tol == 1.0e-14
    assert block.thresh == 1.0e-9
    assert block.transition_table.shape == (2,)
    np.testing.assert_allclose(block.transition_weights, [0.5, 0.5])
    np.testing.assert_array_equal(block.transition_table["q"], [0, 0])
    np.testing.assert_allclose(block.transition_table["energy"], block.transition_energy)
    np.testing.assert_allclose(block.A, block.A.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.B, block.B.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        block.A - block.B,
        np.diag(block.transition_energy),
        atol=1e-12,
    )


def test_gamma_periodic_bse_direct_term_can_use_dense_full_ewald_metric(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    reciprocal = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    full = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    dense_full = dense_gamma_transition_metric(space, q_index=0, component="full_ewald")

    assert full.coulomb_component == "full_ewald"
    np.testing.assert_allclose(full.direct, dense_full, atol=1e-12)
    assert full.exchange[0, 0].real > 0.0
    np.testing.assert_allclose(
        full.screened_exchange,
        np.zeros_like(full.screened_exchange),
        atol=1e-12,
    )
    np.testing.assert_allclose(full.B, full.direct, atol=1e-12)
    np.testing.assert_allclose(
        full.A - full.B,
        np.diag(full.transition_energy),
        atol=1e-12,
    )
    assert full.B[0, 0].real > reciprocal.B[0, 0].real

    tda = periodic_tda(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=False,
    )
    bse = periodic_bse(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=False,
    )
    spectrum = periodic_tda_spectrum(
        space,
        q_indices=[0],
        direct_scale=1.0,
        coulomb_component="full_ewald",
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=False,
    )

    assert tda.block.coulomb_component == "full_ewald"
    assert bse.block.coulomb_component == "full_ewald"
    assert spectrum.info["coulomb_components"] == ("full_ewald",)
    assert tda.e.shape == bse.e.shape == (1,)
    assert np.all(np.isfinite(tda.e))
    assert np.all(np.isfinite(bse.e))


def test_gamma_periodic_bse_full_ewald_uses_dense_pair_couplings(gamma_h2_mf):
    space = KGW(gamma_h2_mf).transition_space(qpts="gamma")
    transition = space.transitions(0)[0]
    reciprocal = periodic_bse_matrices(space, q_index=0, direct_scale=1.0)
    full = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )
    expected_exchange = dense_gamma_orbital_pair_metric(
        space,
        q_index=0,
        left_pair=(
            transition.kq_index,
            transition.kq_index,
            transition.vir_band,
            transition.vir_band,
        ),
        right_pair=(
            transition.k_index,
            transition.k_index,
            transition.occ_band,
            transition.occ_band,
        ),
        component="full_ewald",
    )
    poles = space.screened_interaction(
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )
    occ_coupling = full_ewald_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=transition.k_index,
        kq_index=transition.k_index,
        left_band=transition.occ_band,
        right_band=transition.occ_band,
    )
    vir_coupling = full_ewald_orbital_pair_coupling(
        space,
        q_index=0,
        k_index=transition.kq_index,
        kq_index=transition.kq_index,
        left_band=transition.vir_band,
        right_band=transition.vir_band,
    )
    expected_screened_exchange = np.sum(
        poles.coupling_for_coulomb_vector(vir_coupling)
        * poles.coupling_for_coulomb_vector(occ_coupling).conj()
        / poles.omega
    )

    assert full.coulomb_component == "full_ewald"
    np.testing.assert_allclose(full.exchange[0, 0], expected_exchange, atol=1e-12)
    np.testing.assert_allclose(
        full.screened_exchange[0, 0],
        expected_screened_exchange.real,
        atol=1e-40,
    )
    assert not np.allclose(full.exchange, reciprocal.exchange)
    np.testing.assert_allclose(
        full.A - full.B,
        np.diag(full.transition_energy),
        atol=1e-12,
    )


def test_full_ewald_periodic_bse_supports_multi_k_reference(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    block = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    assert block.coulomb_component == "full_ewald"
    assert block.A.shape == block.B.shape == (2, 2)
    np.testing.assert_allclose(block.direct, block.direct.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.exchange, block.exchange.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.screened_exchange, block.screened_exchange.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.A, block.A.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.B, block.B.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        block.A - block.B,
        np.diag(block.transition_energy),
        atol=1e-12,
    )
    assert np.all(np.isfinite(block.A))


def test_gdf_periodic_bse_supports_multi_k_reference(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    block = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        coulomb_component="gdf",
    )
    gdf_metric = gdf_transition_metric(space, q_index=0)

    assert block.coulomb_component == GDF
    assert block.A.shape == block.B.shape == (2, 2)
    np.testing.assert_allclose(block.direct, 0.5 * gdf_metric, atol=1e-12)
    np.testing.assert_allclose(block.direct, block.direct.conj().T, atol=1e-12)
    np.testing.assert_allclose(block.exchange, block.exchange.conj().T, atol=1e-10)
    np.testing.assert_allclose(block.screened_exchange, block.screened_exchange.conj().T, atol=1e-10)
    np.testing.assert_allclose(block.A, block.A.conj().T, atol=1e-10)
    np.testing.assert_allclose(block.B, block.B.conj().T, atol=1e-10)
    assert np.all(np.isfinite(block.A))


def test_periodic_bse_table_tracks_supplied_qp_energy(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    qp_energy = np.asarray(two_k_h2_reference.mo_energy, dtype=float).copy()
    qp_energy[:, 1] += [0.2, 0.4]

    block = periodic_bse_matrices(
        space,
        q_index=1,
        qp_energy=qp_energy,
        direct_scale=1.0,
    )

    np.testing.assert_allclose(block.transition_energy, [2.1, 1.5])
    np.testing.assert_allclose(block.transition_table["energy"], block.transition_energy)
    np.testing.assert_allclose(space.as_table(1)["energy"], [1.7, 1.3])


def test_periodic_bse_screening_energy_matches_explicit_screening_space(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    screening_energy = np.asarray(two_k_h2_reference.mo_energy, dtype=float).copy()
    screening_energy[:, 1] += [0.2, 0.4]

    from_energy = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        screening_energy=screening_energy,
    )
    from_space = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        screening_space=space.with_mo_energy(screening_energy),
    )

    np.testing.assert_allclose(from_energy.A, from_space.A, atol=1e-12)
    np.testing.assert_allclose(from_energy.B, from_space.B, atol=1e-12)
    np.testing.assert_allclose(
        from_energy.screened_exchange,
        from_space.screened_exchange,
        atol=1e-12,
    )


def test_periodic_bse_inherits_pyscf_gdf_context_in_internal_screening_space(
    two_k_h2_reference,
):
    import pyqed.pbc.gw.bse as bse_module

    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")
    context = object()
    context_key = ("pyscf_gdf", "test-aux")
    space._pyscf_gdf_context = context
    space._pyscf_gdf_context_key = context_key

    inherited = bse_module._screening_space(space)
    shifted = bse_module._screening_space(
        space,
        screening_energy=np.asarray(two_k_h2_reference.mo_energy) + 0.1,
    )

    assert inherited._pyscf_gdf_context is context
    assert inherited._pyscf_gdf_context_key == context_key
    assert shifted._pyscf_gdf_context is context
    assert shifted._pyscf_gdf_context_key == context_key


def test_two_k_periodic_tda_and_bse_wrappers(two_k_h2_reference):
    gw = KGW(two_k_h2_reference).run(direct_scale=1.0)

    tda = KTDA(gw).run(q_index=0, direct_scale=1.0, nroots=2, return_vectors=True)
    bse = KBSE(gw).run(q_index=0, direct_scale=1.0, nroots=2, return_vectors=True)
    direct_tda = periodic_tda(
        gw.transition_space(qpts="mesh"),
        q_index=0,
        qp_energy=gw.e_qp,
        direct_scale=1.0,
        nroots=2,
    )
    direct_bse = periodic_bse(
        gw.transition_space(qpts="mesh"),
        q_index=0,
        qp_energy=gw.e_qp,
        direct_scale=1.0,
        nroots=2,
    )

    assert tda.info["backend"] == "kpoint_dense_bse"
    assert bse.info["backend"] == "kpoint_dense_bse"
    assert tda.info["uses_qp_energy"]
    assert bse.info["uses_qp_energy"]
    assert tda.info["direct_scale"] == 1.0
    assert bse.info["direct_scale"] == 1.0
    assert tda.info["exchange_scale"] == 1.0
    assert bse.info["exchange_scale"] == 1.0
    assert tda.info["screened_exchange_scale"] == 1.0
    assert bse.info["screened_exchange_scale"] == 1.0
    assert tda.info["g2_tol"] == 1.0e-16
    assert bse.info["g2_tol"] == 1.0e-16
    assert tda.info["thresh"] == 1.0e-10
    assert bse.info["thresh"] == 1.0e-10
    assert tda.info["matrix_symmetry_reuses"] == 1
    assert bse.info["matrix_symmetry_reuses"] == 1
    assert tda.bse_metric == "tda"
    assert bse.bse_metric == "full"
    assert tda.e.shape == (2,)
    assert bse.e.shape == (2,)
    assert tda.excitation_vectors.shape == (2, 2)
    assert bse.excitation_vectors.shape == (4, 2)
    assert bse.x.shape == (2, 2)
    assert bse.y.shape == (2, 2)
    assert np.all(np.isfinite(tda.e))
    assert np.all(np.isfinite(bse.e))
    np.testing.assert_allclose(tda.e, direct_tda.e, atol=1e-12)
    np.testing.assert_allclose(bse.e, direct_bse.e, atol=1e-12)
    np.testing.assert_allclose(bse.xy, direct_bse.vectors, atol=1e-12)
    metric_norm = np.sum(abs(bse.x) ** 2, axis=0) - np.sum(abs(bse.y) ** 2, axis=0)
    np.testing.assert_allclose(metric_norm, np.ones(2), atol=1e-8)


def test_periodic_bse_kernel_uses_symmetric_kpoint_quadrature(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="optical")
    metric = reciprocal_transition_factors(space, q_index=0).coulomb_metric()
    block = periodic_bse_matrices(
        space,
        q_index=0,
        direct_scale=1.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )

    np.testing.assert_allclose(block.transition_weights, [0.5, 0.5])
    np.testing.assert_allclose(block.direct, 0.5 * metric, atol=1.0e-12)
    np.testing.assert_allclose(
        block.A - np.diag(block.transition_energy),
        block.direct,
        atol=1.0e-12,
    )


def test_periodic_transition_velocity_uses_q0_vertical_transition_order(
    two_k_h2_reference,
):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="optical")
    velocity = periodic_transition_velocity_matrix_elements(space)

    assert velocity.shape == (2, 3)
    assert np.all(np.isfinite(velocity))
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1.0e-12)
    assert np.linalg.norm(velocity[:, 0]) > 0.0


def test_periodic_transition_velocity_matches_pyscf_grid(two_k_h2_reference):
    pytest.importorskip("pyscf")
    from pyqed.pbc.gw.finite_size import (
        _pyscf_gradient_head_transitions,
        cell_volume,
    )

    two_k_h2_reference.pair_cut = 3
    two_k_h2_reference._periodic_setup()
    space = KPointTransitionSpace(two_k_h2_reference, qpts="optical")
    velocity = periodic_transition_velocity_matrix_elements(space)
    pyscf_values = _pyscf_gradient_head_transitions(
        space,
        qvec=np.asarray([1.0, 0.0, 0.0]),
        energy_table=np.asarray(space.reference.mo_energy),
    )
    volume_factor = np.sqrt(cell_volume(space.reference))

    for row, transition in enumerate(space.transitions(0)):
        key = (transition.k_index, transition.occ_band, transition.vir_band)
        expected = velocity[row, 0] / transition.energy / volume_factor
        np.testing.assert_allclose(expected, pyscf_values[key], rtol=5.0e-4, atol=1.0e-8)


def test_pyscf_gradient_head_accepts_native_basis_dictionary(two_k_h2_reference):
    pytest.importorskip("pyscf")
    from pyqed.pbc.gw.finite_size import _pyscf_gradient_head_transitions

    two_k_h2_reference.cell.basis = {
        "H": [
            (
                0,
                np.asarray([3.42525091, 0.62391373, 0.16885540]),
                np.asarray([[0.15432897], [0.53532814], [0.44463454]]),
            )
        ]
    }
    space = KPointTransitionSpace(two_k_h2_reference, qpts="optical")
    values = _pyscf_gradient_head_transitions(
        space,
        qvec=np.asarray([1.0e-3, 0.0, 0.0]),
        energy_table=np.asarray(space.reference.mo_energy),
    )

    assert len(values) == len(space.transitions(0))
    assert all(np.isfinite(value) for value in values.values())


def test_periodic_tda_and_full_bse_build_polarized_optical_absorption(
    two_k_h2_reference,
):
    two_k_h2_reference.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.8, 1.0]),
    ]
    space = KPointTransitionSpace(two_k_h2_reference, qpts="optical")
    common = dict(
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=2,
        return_vectors=True,
    )
    tda = periodic_tda(space, **common)
    bse = periodic_bse(space, **common)
    velocity = np.asarray(
        [
            [0.3, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.complex128,
    )
    grid = np.linspace(0.0, 60.0, 1201)

    optical_x = periodic_bse_absorption(
        tda,
        energy_grid=grid,
        polarization="x",
        broadening=0.1,
        units="ev",
        transition_velocity=velocity,
    )
    optical_iso = tda.absorption(
        energy_grid=grid,
        broadening=0.1,
        units="ev",
        transition_velocity=velocity,
    )
    optical_full = bse.absorption(
        energy_grid=grid,
        polarization="x",
        broadening=0.1,
        units="ev",
        transition_velocity=velocity,
    )

    assert isinstance(optical_x, PeriodicBSEOpticalResult)
    np.testing.assert_allclose(
        optical_x.excitation_energies,
        np.asarray([1.5, 1.8]) * au2ev,
    )
    np.testing.assert_allclose(optical_x.oscillator_strengths, [0.12, 0.0])
    np.testing.assert_allclose(abs(optical_x.exciton_dipoles[0]), [0.2, 0.0, 0.0])
    np.testing.assert_allclose(
        optical_iso.oscillator_strengths,
        [0.04, 2.0 * 1.8 * (0.4 / 1.8) ** 2 / 3.0],
    )
    np.testing.assert_allclose(abs(optical_full.exciton_dipoles), abs(optical_x.exciton_dipoles))
    np.testing.assert_allclose(optical_full.oscillator_strengths, optical_x.oscillator_strengths)
    assert optical_x.dielectric_tensor_imag.shape == (3, 3, len(grid))
    assert optical_x.dielectric_imag.shape == grid.shape
    assert np.all(optical_x.dielectric_imag >= 0.0)
    assert optical_x.info["velocity_backend"] == "supplied"
    assert optical_x.info["kpoint_quadrature"] == "symmetric_sqrt_weights"


def test_periodic_bse_driver_exposes_optical_absorption(two_k_h2_reference):
    driver = KTDA(two_k_h2_reference).run(
        backend="periodic",
        qpts="optical",
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=True,
    )
    optical = driver.absorption(
        energy_grid=np.linspace(0.0, 60.0, 301),
        polarization=np.asarray([1.0, 1.0j, 0.0]),
        transition_velocity=np.asarray([[0.3, 0.2, 0.0], [0.0, 0.4, 0.0]]),
    )

    assert driver.optical_result is optical
    assert optical.polarization.shape == (3,)
    np.testing.assert_allclose(np.vdot(optical.polarization, optical.polarization), 1.0)


def test_periodic_bse_absorption_requires_q0_and_vectors(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    no_vectors = periodic_tda(
        space,
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=False,
    )
    with pytest.raises(ValueError, match="return_vectors=True"):
        no_vectors.absorption(
            transition_velocity=np.zeros((2, 3)),
        )

    finite_q = periodic_tda(
        space,
        q_index=1,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        nroots=1,
        return_vectors=True,
    )
    with pytest.raises(ValueError, match="q=0"):
        finite_q.absorption(
            transition_velocity=np.zeros((2, 3)),
        )


@pytest.mark.parametrize("component", ["reciprocal_ewald_lr", "gdf"])
@pytest.mark.parametrize("storage", ["transition_blocks", "factorized"])
def test_periodic_factorized_tda_operator_matches_dense_matrix(
    two_k_h2_reference,
    component,
    storage,
):
    if component == "gdf":
        two_k_h2_reference.gdf_auxbasis = "sto-3g"
        two_k_h2_reference.gdf_pair_cut = 0
        two_k_h2_reference.gdf_recip_cut = 2
        two_k_h2_reference.gdf_pair_ft_screen_tol = 0.0
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=1.25,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        coulomb_component=component,
        g2_tol=1.0e-12,
        thresh=1.0e-10,
    )
    block = periodic_bse_matrices(space, **common)
    operator = periodic_tda_operator(space, storage=storage, **common)
    vector = np.asarray([0.3 + 0.2j, -0.4 + 0.1j])

    assert isinstance(operator, PeriodicTDAOperator)
    assert not hasattr(operator, "A")
    np.testing.assert_allclose(operator.matvec(vector), block.A @ vector, atol=1.0e-10)
    np.testing.assert_allclose(
        operator.aslinearoperator().matvec(vector),
        block.A @ vector,
        atol=1.0e-10,
    )
    assert operator.info["nchannels"] == space.nqpts
    assert operator.info["storage"] == storage
    assert operator.info["operator_memory_bytes"] > 0
    if storage == "transition_blocks":
        assert operator.info["block_memory_bytes"] > 0
        assert operator.info["factor_memory_bytes"] == 0
    else:
        assert operator.info["factor_memory_bytes"] > 0
        assert operator.info["block_memory_bytes"] == 0


def test_periodic_tda_operator_direct_only_skips_transfer_channels(
    two_k_h2_reference,
):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=1.25,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        g2_tol=1.0e-12,
    )
    dense = periodic_bse_matrices(space, **common)
    operator = periodic_tda_operator(space, **common)
    vector = np.asarray([0.3 + 0.2j, -0.4 + 0.1j])

    np.testing.assert_allclose(operator.matvec(vector), dense.A @ vector, atol=1.0e-12)
    assert operator.info["nchannels"] == 0
    assert operator.info["factor_memory_bytes"] == 0
    assert operator.info["block_memory_bytes"] == 0


def test_periodic_tda_operator_supports_complex64_blocks(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=1.25,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        g2_tol=1.0e-12,
    )
    dense = periodic_bse_matrices(space, **common)
    operator = periodic_tda_operator(
        space,
        storage="transition_blocks",
        block_dtype="complex64",
        **common,
    )
    vector = np.asarray([0.3 + 0.2j, -0.4 + 0.1j])

    np.testing.assert_allclose(operator.matvec(vector), dense.A @ vector, atol=1.0e-7)
    assert all(group.matrices.dtype == np.complex64 for group in operator.block_groups)


def test_periodic_tda_operator_eigensolve_matches_dense_roots(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=1.25,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        g2_tol=1.0e-12,
    )
    dense = periodic_bse_matrices(space, **common)
    operator = periodic_tda_operator(space, **common)

    result = operator.eigensolve(nroots=1, tol=1.0e-12)

    np.testing.assert_allclose(result.e, np.linalg.eigvalsh(dense.A)[:1], atol=1.0e-11)
    assert result.vectors.shape == (2, 1)
    assert result.info["solver"] == "dense_small_fallback"
    assert result.info["residual_norms"][0] < 1.0e-11
    with pytest.raises(ValueError, match="nroots < dimension"):
        operator.eigensolve(nroots=operator.shape[0])


def test_periodic_tda_operator_matches_dense_finite_momentum(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    q_index = next(index for index in range(space.nqpts) if index != zero_index)
    common = dict(
        q_index=q_index,
        direct_scale=1.25,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        g2_tol=1.0e-12,
    )
    dense = periodic_bse_matrices(space, **common)
    operator = periodic_tda_operator(space, **common)
    vector = np.asarray([0.3 + 0.2j, -0.4 + 0.1j])

    np.testing.assert_allclose(operator.matvec(vector), dense.A @ vector, atol=1.0e-10)
    assert operator.q_index == q_index
    with pytest.raises(ValueError, match="q=0"):
        operator.absorption(transition_velocity=np.zeros((2, 3)))


def test_projected_tda_continuum_matches_explicit_q_basis():
    hamiltonian = np.asarray(
        [
            [-0.5, 0.12 - 0.04j, 0.03, 0.0],
            [0.12 + 0.04j, 0.1, -0.08j, 0.05],
            [0.03, 0.08j, 0.8, -0.09],
            [0.0, 0.05, -0.09, 1.4],
        ],
        dtype=np.complex128,
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    active = vectors[:, :1]
    coupling = np.asarray([[0.21, -0.07j, 0.13, 0.04 + 0.02j]])
    continuum = ProjectedTDAContinuum(
        hamiltonian,
        active,
        coupling,
        solver_tol=1.0e-13,
    )
    energy = 0.37
    eta = 0.025
    q_vectors = vectors[:, 1:]
    q_coupling = coupling @ q_vectors
    expected = (
        q_coupling / (energy + 1.0j * eta - energies[1:])[None, :]
    ) @ q_coupling.conj().T

    np.testing.assert_allclose(
        continuum.self_energy(energy, eta=eta),
        expected,
        atol=2.0e-12,
    )
    times = np.asarray([0.0, 0.3, 1.1])
    expected_memory = np.einsum(
        "ac,tc,bc->tab",
        q_coupling,
        np.exp(-1.0j * np.outer(times, energies[1:])),
        q_coupling.conj(),
        optimize=True,
    )
    np.testing.assert_allclose(
        continuum.memory_kernel(times),
        expected_memory,
        atol=3.0e-12,
    )
    sigma_operator = continuum.self_energy_operator(energy, eta=eta)
    np.testing.assert_allclose(
        sigma_operator.matvec(np.ones(1)),
        expected[:, 0],
        atol=2.0e-12,
    )
    np.testing.assert_allclose(continuum.excluded_hamiltonian, energies[:1, None])
    assert continuum.ncontinuum == 3
    assert continuum.removed_pole_coupling_norm > 0.0
    assert continuum.excluded_residual_norms[0] < 1.0e-12
    assert max(continuum.last_solve_info["residual_norms"]) < 1.0e-11
    assert np.linalg.eigvalsh(continuum.hybridization(energy, eta=eta))[0] >= 0.0


def test_projected_tda_continuum_is_active_phase_invariant():
    hamiltonian = np.diag([-0.4, 0.2, 0.9]).astype(np.complex128)
    active = np.asarray([[1.0], [0.0], [0.0]], dtype=np.complex128)
    coupling = np.asarray([[0.3, 0.12 - 0.02j, -0.08j]])
    reference = ProjectedTDAContinuum(hamiltonian, active, coupling)
    transformed = ProjectedTDAContinuum(
        hamiltonian,
        active * np.exp(0.73j),
        coupling,
    )

    np.testing.assert_allclose(
        transformed.self_energy(0.4, eta=0.03),
        reference.self_energy(0.4, eta=0.03),
        atol=2.0e-14,
    )


def test_projected_tda_continuum_allows_no_excluded_target_poles():
    hamiltonian = np.diag([0.2, 0.7, 1.1]).astype(np.complex128)
    coupling = np.asarray(
        [
            [0.12, -0.04j, 0.08],
            [0.03j, 0.09, -0.02],
        ]
    )
    continuum = ProjectedTDAContinuum(
        hamiltonian,
        np.zeros((3, 0)),
        coupling,
    )
    energy = 0.5
    eta = 0.04
    expected = (
        coupling / (energy + 1.0j * eta - np.diag(hamiltonian))[None, :]
    ) @ coupling.conj().T

    np.testing.assert_allclose(
        continuum.self_energy(energy, eta=eta),
        expected,
        atol=2.0e-13,
    )
    assert continuum.nactive == 2
    assert continuum.nexcluded == 0
    assert continuum.ncontinuum == 3


def test_total_momentum_sector_wraps_phonon_momentum(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    finite_index = next(index for index in range(space.nqpts) if index != zero_index)
    sector = TotalMomentumSector(space, total_q_index=zero_index)

    exciton_index = sector.exciton_q_index([finite_index], [1])
    assert sector.contains(exciton_index, [finite_index], [1])
    assert sector.exciton_q_index([finite_index], [2]) == zero_index
    assert not sector.contains(zero_index, [finite_index], [1])


def test_exciton_phonon_coupling_finite_difference_and_selection_rule(
    two_k_h2_reference,
):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    finite_index = next(index for index in range(space.nqpts) if index != zero_index)
    derivative = np.asarray(
        [
            [0.2, 0.04 - 0.03j],
            [-0.07j, -0.1],
        ]
    )
    reference = np.diag([0.4, 0.9]).astype(np.complex128)
    displacement = 0.015
    frequency = 0.25
    coupling = ExcitonPhononCoupling.from_finite_difference(
        reference + displacement * derivative,
        reference - displacement * derivative,
        displacement,
        frequency,
        phonon_q_index=finite_index,
        source_q_index=zero_index,
        target_q_index=finite_index,
        branch=0,
    ).validate_momentum(space)
    source = np.asarray([[1.0], [0.0]])
    target = np.eye(2)
    expected = derivative @ source / np.sqrt(2.0 * frequency)

    np.testing.assert_allclose(coupling.between(target, source), expected)
    np.testing.assert_allclose(coupling.active_to_target(source), expected.conj().T)
    channel = ExcitonPhononChannel.from_coupling(
        coupling,
        reference,
        source,
        occupation=0.2,
    )
    expected_sigma = (
        expected.conj().T
        / (0.6 + 0.02j - np.diag(reference))[None, :]
    ) @ expected
    np.testing.assert_allclose(
        channel.continuum.self_energy(0.6, eta=0.02),
        expected_sigma,
        atol=2.0e-13,
    )
    assert channel.phonon_q_index == finite_index
    assert channel.branch == 0
    assert channel.occupation == 0.2
    with pytest.raises(ValueError, match="momentum conservation"):
        ExcitonPhononCoupling(
            derivative,
            frequency,
            phonon_q_index=finite_index,
            source_q_index=zero_index,
            target_q_index=zero_index,
        ).validate_momentum(space)


def test_analytic_tda_electron_phonon_derivative_obeys_finite_q_rules(
    four_band_two_k_reference,
):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    phonon_index = next(
        index for index in range(space.nqpts) if index != zero_index
    )
    mo_couplings = (
        np.asarray(
            [
                [0.11, 0.02j, -0.03, 0.07],
                [0.04, -0.05, 0.09j, -0.02],
                [0.08j, 0.01, 0.13, -0.06],
                [-0.03, 0.05j, 0.04, -0.10],
            ],
            dtype=np.complex128,
        ),
        np.asarray(
            [
                [-0.04, 0.03, 0.05j, 0.02],
                [0.07j, 0.06, -0.01, 0.08],
                [0.02, -0.09j, 0.12, 0.04],
                [0.01j, -0.02, 0.03, -0.07],
            ],
            dtype=np.complex128,
        ),
    )
    source_table = space.as_table(zero_index)
    target_table = space.as_table(phonon_index)
    expected = np.zeros(
        (len(target_table), len(source_table)),
        dtype=np.complex128,
    )
    qvec = space.qpts[phonon_index]
    reference = space.reference
    for row, target in enumerate(target_table):
        for column, source in enumerate(source_table):
            if (
                target["k"] == source["k"]
                and target["occ"] == source["occ"]
            ):
                expected[row, column] += mo_couplings[int(source["kq"])][
                    int(target["vir"]),
                    int(source["vir"]),
                ]
            target_plus_q = reference.find_kpoint_index(
                reference.kpts[int(target["k"])] + qvec
            )
            if (
                target["kq"] == source["kq"]
                and target["vir"] == source["vir"]
                and target_plus_q == source["k"]
            ):
                expected[row, column] -= mo_couplings[int(target["k"])][
                    int(source["occ"]),
                    int(target["occ"]),
                ]

    kernel_derivative = np.arange(expected.size).reshape(expected.shape) * 1.0e-4j
    derivative = PeriodicTDAElectronPhononDerivative(
        space,
        zero_index,
        phonon_index,
        mo_couplings,
        kernel_derivative=kernel_derivative,
    )
    expected += kernel_derivative
    vector = np.linspace(0.2, 1.0, expected.shape[1]).astype(np.complex128)
    target_vector = np.linspace(-0.3, 0.4, expected.shape[0]).astype(
        np.complex128
    )

    np.testing.assert_allclose(derivative.toarray(), expected, atol=1.0e-14)
    np.testing.assert_allclose(derivative.matvec(vector), expected @ vector)
    np.testing.assert_allclose(
        derivative.rmatvec(target_vector),
        expected.conj().T @ target_vector,
    )
    assert derivative.target_q_index == phonon_index
    assert derivative.info["kernel_derivative_included"]


def test_static_transition_screening_response_matches_finite_difference():
    from pyqed.pbc.gw.electron_phonon import (
        _static_transition_screening_derivative,
    )

    rng = np.random.default_rng(14)
    target_factor = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    source_factor = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    target_matrix = target_factor @ target_factor.conj().T + np.eye(3)
    source_matrix = source_factor @ source_factor.conj().T + np.eye(2)
    matrix1 = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    left = rng.normal(size=3) + 1j * rng.normal(size=3)
    right = rng.normal(size=2) + 1j * rng.normal(size=2)
    left1 = rng.normal(size=2) + 1j * rng.normal(size=2)
    right1 = rng.normal(size=3) + 1j * rng.normal(size=3)
    scale = 1.7
    actual = _static_transition_screening_derivative(
        left,
        right,
        left1,
        right1,
        np.linalg.inv(target_matrix),
        np.linalg.inv(source_matrix),
        matrix1,
        scale,
    )

    matrix0 = np.block(
        [
            [target_matrix, np.zeros((3, 2))],
            [np.zeros((2, 3)), source_matrix],
        ]
    )
    matrix_derivative = np.block(
        [
            [np.zeros((3, 3)), matrix1],
            [np.zeros((2, 3)), np.zeros((2, 2))],
        ]
    )
    left0 = np.concatenate((left.conj(), np.zeros(2)))
    left_derivative = np.concatenate((np.zeros(3), left1))
    right0 = np.concatenate((np.zeros(3), right))
    right_derivative = np.concatenate((right1, np.zeros(2)))

    def value(displacement):
        return scale**2 * (
            left0 + displacement * left_derivative
        ) @ np.linalg.solve(
            matrix0 + displacement * matrix_derivative,
            right0 + displacement * right_derivative,
        )

    step = 1.0e-6
    np.testing.assert_allclose(
        actual,
        (value(step) - value(-step)) / (2.0 * step),
        atol=3.0e-8,
        rtol=2.0e-8,
    )


def test_analytic_gamma_electron_phonon_gap_derivative_matches_displacement(
    gamma_h2_mf,
):
    space = KPointTransitionSpace(gamma_h2_mf, qpts="gamma")
    operator = periodic_tda_operator(
        space,
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    mode = np.zeros((2, 3))
    mode[0, 0] = 1.0
    frequency = 0.2
    coupling = gamma_tda_electron_phonon_coupling(
        operator,
        mode,
        frequency,
        branch=0,
        cphf_tol=1.0e-11,
    )

    def displaced_gap(displacement):
        cell = Cell(
            atom=[("H", (displacement, 0.0, 0.0)), ("H", (1.4, 0.0, 0.0))],
            a=np.diag([5.0, 5.0, 5.0]),
            basis="sto-3g",
            unit="bohr",
            dimension=3,
            spin=0,
        ).build()
        mean_field = cell.KRHF(
            eta=0.5,
            real_cut=0,
            pair_cut=0,
            recip_cut=2,
            jk_builder="ewald",
        ).run(max_cycle=80, conv_tol=1.0e-11, conv_tol_dm=1.0e-9)
        assert mean_field.converged
        energies = np.asarray(mean_field.mo_energy, dtype=float).reshape(-1)
        return energies[1] - energies[0]

    step = 1.0e-4
    cartesian_derivative = (
        displaced_gap(step) - displaced_gap(-step)
    ) / (2.0 * step)
    from pyqed.units import amu_to_au

    mass = gamma_h2_mf.cell.unit_molecule.atom_mass_list()[0] * amu_to_au
    expected = cartesian_derivative / np.sqrt(mass)
    actual = coupling.derivative.matvec(np.ones(1))[0]

    np.testing.assert_allclose(actual.imag, 0.0, atol=2.0e-11)
    np.testing.assert_allclose(actual.real, expected, atol=5.0e-6, rtol=0.0)
    assert coupling.response.converged
    assert coupling.info["analytic_driver"] == "gamma_krhf_cphf"
    assert coupling.info["bse_screening_derivative"] == "frozen"
    assert not coupling.info["kernel_derivative_included"]


def test_analytic_tda_electron_phonon_coupling_applies_zero_point_scale(
    two_k_h2_reference,
):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    phonon_index = next(
        index for index in range(space.nqpts) if index != zero_index
    )
    fock_derivative = (
        np.asarray([[0.2, -0.03j], [0.04, -0.1]]),
        np.asarray([[-0.05, 0.02], [0.07j, 0.13]]),
    )
    frequency = 0.25
    coupling = analytic_tda_electron_phonon_coupling(
        space,
        zero_index,
        phonon_index,
        frequency,
        fock_derivative,
        branch=2,
    ).validate_momentum(space)
    source = np.ones((coupling.derivative.shape[1], 1))
    raw = coupling.derivative.matvec(source[:, 0])

    np.testing.assert_allclose(
        coupling.active_to_target(source),
        (raw / np.sqrt(2.0 * frequency))[None, :].conj(),
    )
    assert coupling.branch == 2
    assert coupling.info["approximation"] == "frozen_screening_one_body_fan"


def test_commensurate_tda_electron_phonon_driver_uses_folded_q_blocks(
    four_band_two_k_reference,
    monkeypatch,
):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    phonon_index = next(index for index in range(space.nqpts) if index != zero_index)
    operator = periodic_tda_operator(
        space,
        q_index=zero_index,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    fock = (
        np.asarray([[0.2, -0.03j], [0.04, -0.1]]),
        np.asarray([[-0.05, 0.02], [0.07j, 0.13]]),
    )
    overlap = tuple(np.zeros_like(block) for block in fock)

    class FakeQDerivative:
        fock_derivative = fock
        overlap_derivative = overlap
        response = object()
        mode_vector = np.asarray([[1.0, 0.0, 0.0]])
        cartesian_mode = mode_vector
        info = {
            "backend": "commensurate_twisted_supercell_gdf",
            "mesh": (2, 1, 1),
        }

    def fake_q_derivative(mean_field, qpoint, mode_vector, **kwargs):
        assert mean_field is four_band_two_k_reference
        np.testing.assert_allclose(qpoint, space.qpts[phonon_index])
        np.testing.assert_allclose(mode_vector, [[1.0, 0.0, 0.0]])
        assert kwargs["mesh"] == (2, 1, 1)
        return FakeQDerivative()

    monkeypatch.setattr(
        "pyqed.qchem.pbc.gdf_q_derivative",
        fake_q_derivative,
    )
    kernel1 = np.full(
        (
            len(space.transitions(phonon_index)),
            len(space.transitions(zero_index)),
        ),
        2.0e-4j,
        dtype=np.complex128,
    )
    monkeypatch.setattr(
        "pyqed.pbc.gw.electron_phonon."
        "commensurate_gdf_bare_tda_kernel_derivative",
        lambda source_operator, q_derivative: kernel1,
    )
    coupling = commensurate_tda_electron_phonon_coupling(
        operator,
        phonon_index,
        [[1.0, 0.0, 0.0]],
        0.25,
        kernel_derivative="bare_gdf",
        supercell_mesh=(2, 1, 1),
    )
    expected = analytic_tda_electron_phonon_coupling(
        space,
        zero_index,
        phonon_index,
        0.25,
        fock,
        overlap_derivative=overlap,
        kernel_derivative=kernel1,
    )

    np.testing.assert_allclose(
        coupling.electron_phonon_derivative.toarray(),
        expected.electron_phonon_derivative.toarray(),
    )
    assert coupling.info["analytic_driver"] == (
        "commensurate_twisted_supercell_gdf_cphf"
    )
    assert coupling.info["bse_screening_derivative"] == "frozen"
    assert coupling.info["bse_kernel_derivative"] == "frozen_orbital_bare_gdf"

    screened_response = object()
    screened_components = {
        "bare": 0.4 * kernel1,
        "screened": 0.6 * kernel1,
    }
    def fake_screened_derivative(source_operator, q_derivative):
        q_derivative.gdf_screened_interaction_derivative = screened_response
        q_derivative.gdf_screened_kernel_derivative_components = screened_components
        return kernel1

    monkeypatch.setattr(
        "pyqed.pbc.gw.electron_phonon."
        "commensurate_gdf_screened_tda_kernel_derivative",
        fake_screened_derivative,
    )
    screened = commensurate_tda_electron_phonon_coupling(
        operator,
        phonon_index,
        [[1.0, 0.0, 0.0]],
        0.25,
        kernel_derivative="screened_gdf",
        supercell_mesh=(2, 1, 1),
    )
    assert screened.gdf_screened_interaction_derivative is screened_response
    assert screened.gdf_kernel_derivative_components is screened_components
    assert screened.info["bse_screening_derivative"] == (
        "static_off_diagonal_direct_rpa_gdf"
    )
    assert screened.info["bse_kernel_derivative"] == (
        "bare_plus_static_screened_gdf"
    )


def test_native_phonon_mode_dispatches_to_finite_q_coupling(
    four_band_two_k_reference,
    monkeypatch,
):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    phonon_index = next(index for index in range(space.nqpts) if index != zero_index)
    operator = periodic_tda_operator(
        space,
        q_index=zero_index,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    scaled_qpoint = space.reference.cartesian_to_scaled(space.qpts[phonon_index])
    masses = (
        np.asarray(
            four_band_two_k_reference.cell.unit_molecule.atom_mass_list(),
            dtype=float,
        )
        * amu_to_au
    )
    eigenvector = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=float,
    )

    class FakePhonons:
        def mode(self, qpoint, branch):
            np.testing.assert_allclose(qpoint, scaled_qpoint)
            return PeriodicPhononMode(
                qpoint=qpoint,
                branch=branch,
                frequency=0.015,
                eigenvector=eigenvector,
                masses=masses,
                source="test_phonons",
            )

    class FakeCoupling:
        def __init__(self):
            self.info = {}

        def validate_momentum(self, candidate):
            assert candidate is space
            return self

    expected = FakeCoupling()

    def fake_coupling(
        candidate_operator,
        candidate_q_index,
        mode_vector,
        frequency,
        **kwargs,
    ):
        assert candidate_operator is operator
        assert candidate_q_index == phonon_index
        np.testing.assert_allclose(mode_vector, eigenvector / np.sqrt(2.0))
        assert frequency == pytest.approx(0.015)
        assert kwargs["branch"] == 4
        assert kwargs["kernel_derivative"] == "screened_gdf"
        assert kwargs["supercell_mesh"] == (2, 1, 1)
        return expected

    monkeypatch.setattr(
        "pyqed.pbc.gw.electron_phonon."
        "commensurate_tda_electron_phonon_coupling",
        fake_coupling,
    )
    actual = phonon_tda_electron_phonon_coupling(
        operator,
        FakePhonons(),
        phonon_index,
        branch=4,
        supercell_mesh=(2, 1, 1),
    )

    assert actual is expected
    assert actual.phonon_mode.branch == 4
    assert actual.info["phonon_source"] == "test_phonons"
    np.testing.assert_allclose(
        actual.info["phonon_qpoint_fractional"],
        scaled_qpoint,
    )


def test_native_phonon_mode_rejects_unstable_frequency(
    four_band_two_k_reference,
):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    operator = periodic_tda_operator(
        space,
        q_index=zero_index,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
    )
    masses = (
        np.asarray(
            four_band_two_k_reference.cell.unit_molecule.atom_mass_list(),
            dtype=float,
        )
        * amu_to_au
    )

    class UnstablePhonons:
        def mode(self, qpoint, branch):
            return PeriodicPhononMode(
                qpoint=qpoint,
                branch=branch,
                frequency=-0.002,
                eigenvector=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                masses=masses,
            )

    with pytest.raises(ValueError, match="unstable"):
        phonon_tda_electron_phonon_coupling(
            operator,
            UnstablePhonons(),
            zero_index,
            branch=0,
        )

def test_gamma_one_body_tda_electron_phonon_derivative_is_hermitian(
    four_band_two_k_reference,
):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    source_index = next(
        index for index in range(space.nqpts) if index != zero_index
    )
    first = np.asarray(
        [
            [0.11, 0.02j, -0.03, 0.07],
            [-0.02j, -0.05, 0.09j, -0.02],
            [-0.03, -0.09j, 0.13, -0.06],
            [0.07, -0.02, -0.06, -0.10],
        ],
        dtype=np.complex128,
    )
    second = np.asarray(
        [
            [-0.04, 0.03, 0.05j, 0.02],
            [0.03, 0.06, -0.01, 0.08],
            [-0.05j, -0.01, 0.12, 0.04],
            [0.02, 0.08, 0.04, -0.07],
        ],
        dtype=np.complex128,
    )
    derivative = PeriodicTDAElectronPhononDerivative(
        space,
        source_index,
        zero_index,
        (first, second),
    ).toarray()

    np.testing.assert_allclose(derivative, derivative.conj().T, atol=1.0e-14)


def test_exciton_phonon_continuum_sums_emission_and_absorption_channels():
    from pyqed.ldr import FeshbachEmbedding

    hamiltonian_1 = np.diag([0.4, 0.8]).astype(np.complex128)
    hamiltonian_2 = np.diag([0.6, 1.1]).astype(np.complex128)
    coupling_1 = np.asarray([[0.12, -0.05j], [0.03, 0.08]])
    coupling_2 = np.asarray([[0.04j, 0.09], [-0.07, 0.02j]])
    continuum_1 = ProjectedTDAContinuum(
        hamiltonian_1,
        np.zeros((2, 0)),
        coupling_1,
    )
    continuum_2 = ProjectedTDAContinuum(
        hamiltonian_2,
        np.zeros((2, 0)),
        coupling_2,
    )
    channels = (
        ExcitonPhononChannel(continuum_1, frequency=0.1, occupation=0.0),
        ExcitonPhononChannel(continuum_2, frequency=0.2, occupation=0.35),
    )
    continuum = ExcitonPhononContinuum(channels)
    energy = 0.7
    eta = 0.03
    expected = continuum_1.self_energy(energy - 0.1, eta=eta)
    expected += 1.35 * continuum_2.self_energy(energy - 0.2, eta=eta)
    expected += 0.35 * continuum_2.self_energy(energy + 0.2, eta=eta)

    np.testing.assert_allclose(
        continuum.self_energy(energy, eta=eta),
        expected,
        atol=2.0e-13,
    )
    times = np.asarray([0.0, 0.2, 0.7])
    expected_memory = (
        np.exp(-1.0j * 0.1 * times)[:, None, None]
        * continuum_1.memory_kernel(times)
    )
    expected_memory += (
        1.35 * np.exp(-1.0j * 0.2 * times)
        + 0.35 * np.exp(1.0j * 0.2 * times)
    )[:, None, None] * continuum_2.memory_kernel(times)
    np.testing.assert_allclose(
        continuum.memory_kernel(times),
        expected_memory,
        atol=3.0e-12,
    )
    assert np.min(np.linalg.eigvalsh(continuum.hybridization(energy, eta=eta))) > -1e-14
    active_hamiltonian = np.diag([0.25, 0.45])
    embedded = FeshbachEmbedding(active_hamiltonian, continuum)
    expected_green = np.linalg.inv(
        (energy + 1.0j * eta) * np.eye(2)
        - active_hamiltonian
        - expected
    )
    np.testing.assert_allclose(
        embedded.green_function(energy, eta=eta),
        expected_green,
        atol=2.0e-12,
    )
    convenience = continuum.run_spectrum(
        active_hamiltonian,
        np.linspace(0.2, 1.2, 9),
        eta=eta,
    )
    assert convenience is continuum.embedding
    assert convenience.success
    assert convenience.spectral_density.shape == (9,)
    augmented = continuum.augmented_hamiltonian(active_hamiltonian)
    identity = np.eye(augmented.shape[0], dtype=np.complex128)
    dense_augmented = np.column_stack(
        [augmented.matvec(identity[:, column]) for column in range(augmented.shape[0])]
    )
    complete_green = np.linalg.inv(
        (energy + 1.0j * eta) * identity - dense_augmented
    )
    np.testing.assert_allclose(
        complete_green[:2, :2],
        expected_green,
        atol=2.0e-12,
    )
    dynamics = continuum.run_dynamics(
        active_hamiltonian,
        np.asarray([1.0, 0.0]),
        np.linspace(0.0, 8.0, 41),
    )
    assert dynamics.success
    assert np.max(dynamics.continuum_population) > 1.0e-4
    np.testing.assert_allclose(dynamics.total_norm, 1.0, atol=2.0e-12)


def test_bose_occupation_uses_kelvin_and_exact_zero_temperature():
    from pyqed.pbc.gw import bose_occupation
    from pyqed.units import kelvin

    frequency = 0.008
    assert bose_occupation(frequency, 0.0) == 0.0
    np.testing.assert_allclose(
        bose_occupation(frequency, 300.0),
        1.0 / np.expm1(frequency / (300.0 * kelvin)),
        atol=1.0e-15,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        bose_occupation(frequency, -1.0)


def test_gdf_q_derivative_factor_cache_is_shared(monkeypatch):
    from pyqed.pbc.gw import electron_phonon as module

    reference = object()
    source_operator = SimpleNamespace(space=SimpleNamespace(reference=reference))
    q_derivative = SimpleNamespace()
    created = []

    class FakeFactors:
        def __init__(self, operator, derivative):
            created.append((operator, derivative))

    monkeypatch.setattr(module, "GDFQDerivativeFactors", FakeFactors)
    first = module.gdf_q_derivative_factors(source_operator, q_derivative)
    second = module.gdf_q_derivative_factors(source_operator, q_derivative)

    assert first is second
    assert created == [(source_operator, q_derivative)]


def test_periodic_haydock_absorption_matches_explicit_tda_roots(two_k_h2_reference):
    two_k_h2_reference.mo_energy = [
        np.asarray([-1.0, 0.5]),
        np.asarray([-0.8, 0.9]),
    ]
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=1.0,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
        g2_tol=1.0e-14,
        thresh=1.0e-12,
    )
    dense = periodic_tda(space, nroots=2, return_vectors=True, **common)
    operator = periodic_tda_operator(space, **common)
    velocity = np.asarray(
        [
            [0.3, 0.1, 0.0],
            [0.2, 0.4, 0.0],
        ],
        dtype=np.complex128,
    )
    grid = np.linspace(20.0, 60.0, 801)
    explicit = dense.absorption(
        energy_grid=grid,
        polarization="x",
        broadening=0.2,
        transition_velocity=velocity,
    )
    haydock = operator.absorption(
        energy_grid=grid,
        polarization="x",
        broadening=0.2,
        transition_velocity=velocity,
        niter=operator.shape[0],
    )

    assert isinstance(haydock, PeriodicBSEHaydockResult)
    np.testing.assert_allclose(
        haydock.dielectric_imag,
        explicit.dielectric_imag,
        atol=1.0e-12,
    )
    assert haydock.info["iterations"] == (2,)
    assert haydock.info["operator"]["backend"] == "kpoint_matrix_free_tda"


def test_periodic_ktda_haydock_driver_avoids_dense_result(two_k_h2_reference):
    velocity = np.asarray([[0.3, 0.1, 0.0], [0.2, 0.4, 0.0]])
    driver = KTDA(two_k_h2_reference)
    spectrum = driver.haydock(
        energy_grid=np.linspace(20.0, 60.0, 401),
        polarization="x",
        broadening=0.2,
        transition_velocity=velocity,
        niter=2,
        direct_scale=1.0,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
    )

    assert driver.optical_result is spectrum
    assert driver._periodic_operator is not None
    assert driver._periodic_result is None
    assert driver._periodic_spectrum is None
    assert driver.info["backend"] == "periodic_bse_haydock"
    assert spectrum.dielectric_imag.shape == (401,)
    assert spectrum.info["krylov_complete"] is True
    assert spectrum.info["converged"] is True


def test_periodic_ktda_matrix_free_eigensolve_driver(two_k_h2_reference):
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")
    common = dict(
        direct_scale=1.0,
        exchange_scale=0.25,
        screened_exchange_scale=0.75,
    )
    dense = periodic_tda(space, q_index=0, nroots=1, **common)

    driver = KTDA(two_k_h2_reference).eigensolve(
        nroots=1,
        tol=1.0e-12,
        **common,
    )

    np.testing.assert_allclose(driver.e, dense.e, atol=1.0e-11)
    assert driver._periodic_result is not None
    assert driver._periodic_operator is not None
    assert driver.x.shape == (2, 1)
    assert driver.info["backend"] == "kpoint_matrix_free_tda"


def test_periodic_bse_root_requests_are_validated(two_k_h2_reference):
    space = KGW(two_k_h2_reference).transition_space(qpts="mesh")

    empty_tda = periodic_tda(
        space,
        q_index=0,
        direct_scale=1.0,
        nroots=0,
        return_vectors=True,
    )
    empty_bse = periodic_bse(
        space,
        q_index=0,
        direct_scale=1.0,
        nroots=0,
        return_vectors=True,
    )
    assert empty_tda.e.shape == (0,)
    assert empty_tda.vectors.shape == (2, 0)
    assert empty_tda.info["nroots_requested"] == 0
    assert empty_tda.info["nroots_returned"] == 0
    assert empty_bse.e.shape == (0,)
    assert empty_bse.vectors.shape == (4, 0)
    assert empty_bse.info["nroots_requested"] == 0
    assert empty_bse.info["nroots_returned"] == 0

    with pytest.raises(ValueError, match="nroots"):
        periodic_tda(space, q_index=0, direct_scale=1.0, nroots=-1)
    with pytest.raises(ValueError, match="nroots"):
        periodic_bse(space, q_index=0, direct_scale=1.0, nroots=-1)
    with pytest.raises(TypeError, match="nroots"):
        periodic_tda(space, q_index=0, direct_scale=1.0, nroots=1.5)
    with pytest.raises(TypeError, match="nroots"):
        periodic_bse(space, q_index=0, direct_scale=1.0, nroots=1.5)
    with pytest.raises(RuntimeError, match="requested"):
        periodic_tda(space, q_index=0, direct_scale=1.0, nroots=99)
    with pytest.raises(RuntimeError, match="requested"):
        periodic_bse(space, q_index=0, direct_scale=1.0, nroots=99)

    spectrum = periodic_tda_spectrum(
        space,
        q_indices=[0],
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
    )
    assert spectrum.info["nroots_requested"] == (1,)
    assert spectrum.info["nroots_returned"] == (1,)


def test_periodic_bse_partial_roots_use_sparse_solver(four_band_two_k_reference):
    space = KPointTransitionSpace(four_band_two_k_reference, qpts="mesh")
    common = dict(
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        return_vectors=False,
    )

    dense_tda = periodic_tda(space, **common)
    sparse_tda = periodic_tda(space, nroots=2, **common)
    dense_bse = periodic_bse(space, **common)
    sparse_bse = periodic_bse(space, nroots=2, **common)

    assert sparse_tda.info["backend"] == "kpoint_sparse_bse"
    assert sparse_tda.info["solver"] == "sparse_tda"
    assert sparse_bse.info["backend"] == "kpoint_sparse_bse"
    assert sparse_bse.info["solver"] == "sparse_full_bse"
    np.testing.assert_allclose(sparse_tda.e, dense_tda.e[:2], atol=1.0e-10)
    np.testing.assert_allclose(sparse_bse.e, dense_bse.e[:2], atol=1.0e-10)


def test_periodic_bse_uses_requested_thresh_for_casida_roots(two_k_h2_reference):
    two_k_h2_reference.mo_energy = [
        np.asarray([-0.10, 0.10]),
        np.asarray([-0.10, 0.10]),
    ]
    space = KPointTransitionSpace(two_k_h2_reference, qpts="mesh")

    result = periodic_bse(
        space,
        q_index=0,
        direct_scale=0.0,
        exchange_scale=0.0,
        screened_exchange_scale=0.0,
        thresh=0.01,
        nroots=1,
        return_vectors=False,
    )
    np.testing.assert_allclose(result.e, [0.20], atol=1e-12)
    assert result.info["thresh"] == 0.01

    with pytest.raises(RuntimeError, match="positive roots"):
        periodic_bse(
            space,
            q_index=0,
            direct_scale=0.0,
            exchange_scale=0.0,
            screened_exchange_scale=0.0,
            thresh=0.10,
            nroots=1,
            return_vectors=False,
        )


def test_periodic_bse_wrappers_can_screen_from_qp_energy(two_k_h2_reference):
    gw = KGW(two_k_h2_reference, eta=1.0e-3).evgw(
        direct_scale=1.0,
        max_cycle=1,
        conv_tol=0.0,
        damping=0.5,
        solve_roots=False,
    )

    tda = KTDA(gw).run(
        q_index=0,
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        screening_from_qp=True,
    )
    bse = KBSE(gw).run(
        q_index=0,
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        screening_from_qp=True,
    )

    assert tda.info["uses_screening_energy"]
    assert bse.info["uses_screening_energy"]
    np.testing.assert_allclose(
        tda._periodic_result.block.transition_table["energy"],
        tda._periodic_result.block.transition_energy,
    )
    np.testing.assert_allclose(
        bse._periodic_result.block.transition_table["energy"],
        bse._periodic_result.block.transition_energy,
    )


def test_two_k_periodic_q_spectra(two_k_h2_reference):
    gw = KGW(two_k_h2_reference).run(direct_scale=1.0)
    space = gw.transition_space(qpts="mesh")

    direct_tda = periodic_tda_spectrum(
        space,
        qp_energy=gw.e_qp,
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
    )
    direct_bse = periodic_bse_spectrum(
        space,
        qp_energy=gw.e_qp,
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
    )
    tda_driver = KTDA(gw)
    bse_driver = KBSE(gw)
    wrapper_tda = tda_driver.q_spectrum(direct_scale=1.0, nroots=1, return_vectors=False)
    wrapper_bse = bse_driver.q_spectrum(direct_scale=1.0, nroots=1, return_vectors=False)
    qp_screened_subset = KTDA(gw).q_spectrum(
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        q_indices=[1],
        screening_from_qp=True,
    )
    mf_subset = KTDA(gw).q_spectrum(
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        q_indices=[1],
        use_qp=False,
    )

    assert direct_tda.nblocks == 2
    assert direct_bse.nblocks == 2
    np.testing.assert_array_equal(direct_tda.q_indices, [0, 1])
    np.testing.assert_array_equal(direct_tda.info["q_indices"], [0, 1])
    assert direct_tda.qpts.shape == (2, 3)
    assert direct_tda.info["uses_qp_energy"]
    assert direct_bse.info["uses_qp_energy"]
    assert direct_tda.info["coulomb_components"] == ("reciprocal_ewald_lr",)
    assert direct_tda.info["direct_scales"] == (1.0,)
    assert direct_bse.info["direct_scales"] == (1.0,)
    assert direct_tda.info["exchange_scales"] == (1.0,)
    assert direct_bse.info["exchange_scales"] == (1.0,)
    assert direct_tda.info["screened_exchange_scales"] == (1.0,)
    assert direct_bse.info["screened_exchange_scales"] == (1.0,)
    assert direct_tda.info["g2_tols"] == (1.0e-16,)
    assert direct_bse.info["g2_tols"] == (1.0e-16,)
    assert direct_tda.info["thresh_values"] == (1.0e-10,)
    assert direct_bse.info["thresh_values"] == (1.0e-10,)
    assert all(roots.shape == (1,) for roots in direct_tda.energies_by_q)
    assert all(roots.shape == (1,) for roots in direct_bse.energies_by_q)
    np.testing.assert_allclose(wrapper_tda.lowest_roots(), direct_tda.lowest_roots(), atol=1e-12)
    np.testing.assert_allclose(wrapper_bse.lowest_roots(), direct_bse.lowest_roots(), atol=1e-12)
    assert tda_driver.info == wrapper_tda.info
    assert bse_driver.info == wrapper_bse.info
    assert tda_driver.bse_metric == "tda"
    assert bse_driver.bse_metric == "full"
    assert tda_driver.excitation_energies == wrapper_tda.energies_by_q
    assert bse_driver.excitation_energies == wrapper_bse.energies_by_q
    assert tda_driver.excitation_vectors == (None, None)
    assert bse_driver.excitation_vectors == (None, None)
    assert qp_screened_subset.nblocks == 1
    np.testing.assert_array_equal(qp_screened_subset.q_indices, [1])
    assert qp_screened_subset.info["uses_qp_energy"]
    assert qp_screened_subset.info["uses_screening_energy"]
    assert qp_screened_subset.results[0].block.q_index == 1
    assert mf_subset.info["uses_qp_energy"] is False
    assert mf_subset.info["uses_screening_energy"] is False
    np.testing.assert_allclose(
        mf_subset.results[0].block.transition_energy,
        space.energies(1),
        atol=1e-12,
    )


def test_real_two_k_krhf_reference_runs_periodic_gw_bse(real_two_k_h2_mf):
    assert real_two_k_h2_mf.converged

    ref = KPointSCFAdapter(real_two_k_h2_mf)
    space = KPointTransitionSpace(ref, qpts="mesh")
    assert ref.nkpts == 2
    assert space.nqpts == 2
    np.testing.assert_array_equal(space.ntransitions_by_q, [2, 2])

    gw = KGW(real_two_k_h2_mf, eta=1e-3).run(direct_scale=1.0, g2_tol=1e-14)
    assert gw.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert gw.info["converged"]
    assert gw.e_qp.shape == ref.mo_energy.shape
    assert np.all(np.isfinite(gw.e_qp))

    tda = KTDA(gw).run(q_index=0, direct_scale=1.0, g2_tol=1e-14, nroots=2)
    bse = KBSE(gw).run(q_index=0, direct_scale=1.0, g2_tol=1e-14, nroots=2)
    assert tda.info["backend"] == "kpoint_dense_bse"
    assert bse.info["backend"] == "kpoint_dense_bse"
    assert tda.e.shape == (2,)
    assert bse.e.shape == (2,)
    assert np.all(np.isfinite(tda.e))
    assert np.all(np.isfinite(bse.e))
    assert bse.xy.shape == (4, 2)
    assert bse._periodic_result.block.transition_table.shape == (2,)


def test_gamma_pbc_kgw_smoke(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, screening="TDH", eta=1.0e-3).run()

    assert gw.info["pbc"]
    assert gw.info["backend"] == "gamma_molecular_bridge"
    assert gw.e_qp.shape == (gamma_h2_mf.cell.nao,)
    assert np.all(np.isfinite(gw.e_qp))


def test_explicit_molecular_backend_rejects_periodic_options(gamma_h2_mf, two_k_h2_reference):
    with pytest.raises(ValueError, match="periodic-only"):
        KGW(gamma_h2_mf).run(
            backend="molecular",
            direct_scale=1.0,
        )
    with pytest.raises(NotImplementedError, match="Multi-k"):
        KGW(two_k_h2_reference).run(backend="molecular")
    with pytest.raises(ValueError, match="periodic-only"):
        KTDA(gamma_h2_mf).run(
            backend="molecular",
            q_index=0,
            nroots=1,
        )
    with pytest.raises(ValueError, match="periodic-only"):
        KBSE(gamma_h2_mf).run(
            backend="molecular",
            screening_from_qp=True,
            nroots=1,
        )


def test_gamma_kgw_can_use_periodic_full_ewald_driver(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, eta=1.0e-3).g0w0(
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    assert gw.periodic_backend
    assert gw._gw is None
    assert gw.g0w0_result is not None
    assert gw.info["backend"] == "kpoint_diagonal_direct_rpa"
    assert gw.info["coulomb_component"] == "full_ewald"
    assert gw.info["converged"]
    assert gw.e_qp.shape == (1, gamma_h2_mf.cell.nao)
    assert np.all(np.isfinite(gw.e_qp))


def test_gamma_pbc_tda_and_bse_smoke(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, screening="TDH", eta=1.0e-3).run()

    tda = KTDA(gw).run(nroots=1, low_rank=False, return_vectors=True)
    bse = KBSE(gw).run(nroots=1, low_rank=False, return_vectors=True)

    assert tda.info["pbc"]
    assert bse.info["pbc"]
    assert tda.e.shape == (1,)
    assert bse.e.shape == (1,)
    assert np.all(np.isfinite(tda.e))
    assert np.all(np.isfinite(bse.e))


def test_bse_wrapper_reused_driver_clears_stale_route_state(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, screening="TDH", eta=1.0e-3).run()
    tda = KTDA(gw).run(
        q_index=0,
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
    )
    assert tda._solver is None
    assert tda._periodic_result is not None
    assert tda.excitation_vectors is None

    tda.run(
        backend="molecular",
        nroots=1,
        low_rank=False,
        return_vectors=True,
    )

    assert tda.info["backend"] == "gamma_molecular_bridge"
    assert tda._solver is not None
    assert tda._periodic_result is None
    assert tda._periodic_spectrum is None
    assert tda.excitation_vectors is tda._solver.excitation_vectors
    assert tda.e.shape == (1,)
    assert np.all(np.isfinite(tda.e))


def test_gamma_periodic_bse_wrapper_q_index_selects_periodic_backend(gamma_h2_mf):
    tda = KTDA(gamma_h2_mf).run(
        q_index=0,
        nroots=1,
        return_vectors=False,
        use_qp=False,
    )

    assert tda.info["backend"] == "kpoint_dense_bse"
    assert tda._periodic_result.block.q_index == 0
    assert tda.info["uses_qp_energy"] is False
    assert tda.e.shape == (1,)


def test_gamma_periodic_q_spectrum_defaults_to_periodic_backend(gamma_h2_mf):
    tda_driver = KTDA(gamma_h2_mf)
    bse_driver = KBSE(gamma_h2_mf)
    tda = tda_driver.q_spectrum(
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        use_qp=False,
    )
    bse = bse_driver.q_spectrum(
        direct_scale=1.0,
        nroots=1,
        return_vectors=False,
        use_qp=False,
    )

    assert tda.info["backend"] == "kpoint_dense_bse"
    assert bse.info["backend"] == "kpoint_dense_bse"
    assert tda.nblocks == bse.nblocks == 1
    np.testing.assert_array_equal(tda.q_indices, [0])
    np.testing.assert_array_equal(bse.q_indices, [0])
    assert tda.info["uses_qp_energy"] is False
    assert bse.info["uses_qp_energy"] is False
    assert tda_driver.info == tda.info
    assert bse_driver.info == bse.info
    assert tda_driver.excitation_energies == tda.energies_by_q
    assert bse_driver.excitation_energies == bse.energies_by_q
    assert tda.results[0].block.q_index == 0
    assert bse.results[0].block.q_index == 0
    assert np.all(np.isfinite(tda.lowest_roots()))
    assert np.all(np.isfinite(bse.lowest_roots()))

    with pytest.raises(NotImplementedError, match="q_spectrum"):
        KTDA(gamma_h2_mf).q_spectrum(
            backend="molecular",
            direct_scale=1.0,
            nroots=1,
            return_vectors=False,
            use_qp=False,
        )


def test_gamma_periodic_bse_wrappers_can_use_full_ewald_driver(gamma_h2_mf):
    gw = KGW(gamma_h2_mf, eta=1.0e-3).g0w0(
        direct_scale=1.0,
        coulomb_component="full_ewald",
    )

    tda = KTDA(gw).run(
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        nroots=1,
        return_vectors=False,
    )
    bse = KBSE(gw).run(
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        nroots=1,
        return_vectors=False,
    )
    direct_tda = KTDA(gamma_h2_mf).run(
        backend="periodic",
        q_index=0,
        direct_scale=1.0,
        coulomb_component="full_ewald",
        nroots=1,
        return_vectors=False,
        use_qp=False,
    )

    assert tda.info["backend"] == "kpoint_dense_bse"
    assert bse.info["backend"] == "kpoint_dense_bse"
    assert tda._periodic_result.block.coulomb_component == "full_ewald"
    assert bse._periodic_result.block.coulomb_component == "full_ewald"
    assert direct_tda._periodic_result.block.coulomb_component == "full_ewald"
    assert tda.info["uses_qp_energy"]
    assert direct_tda.info["uses_qp_energy"] is False
    assert tda.e.shape == bse.e.shape == direct_tda.e.shape == (1,)
    assert np.all(np.isfinite(tda.e))
    assert np.all(np.isfinite(bse.e))
    assert np.all(np.isfinite(direct_tda.e))
