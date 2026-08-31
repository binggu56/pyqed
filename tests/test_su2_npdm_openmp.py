import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG


def test_reduced_su2_npdm_openmp_matches_serial():
    atom = "; ".join(f"H 0 0 {1.6 * site}" for site in range(6))
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    solver = DMRG(
        mol.RHF().run(),
        ncas=6,
        nelecas=6,
        D=16,
        init_guess="cid",
        symmetry="su2",
        verbose=0,
    )
    solver.run(
        nsweeps=1,
        n_threads=1,
        require_convergence=False,
        mixer_zero_block_noise_scale=0.0,
    )

    environment = solver._su2_runtime.moving_environment
    if not environment.threading_info["available"]:
        pytest.skip("SU(2) extension was built without OpenMP")
    tensors = solver.dmrg.ground_state.tensors
    environment.set_num_threads(1)
    serial = environment.spatial_npdm(
        tensors,
        spin_rotation_reduction=True,
    )
    regions_before = environment.threading_info["parallel_regions"]
    environment.set_num_threads(2)
    parallel = environment.spatial_npdm(
        tensors,
        spin_rotation_reduction=True,
    )
    component_reference = environment.spatial_npdm(
        tensors,
        spin_rotation_reduction=True,
        component_reference=True,
    )

    np.testing.assert_array_equal(parallel["rdm1"], serial["rdm1"])
    np.testing.assert_array_equal(parallel["rdm2"], serial["rdm2"])
    np.testing.assert_allclose(
        parallel["rdm1"], component_reference["rdm1"], atol=1e-12
    )
    np.testing.assert_allclose(
        parallel["rdm2"], component_reference["rdm2"], atol=1e-12
    )
    assert np.trace(parallel["rdm1"]) == pytest.approx(6.0, abs=1e-11)
    assert np.einsum("pprr->", parallel["rdm2"]) == pytest.approx(
        30.0, abs=1e-10
    )
    assert environment.threading_info["parallel_regions"] > regions_before
