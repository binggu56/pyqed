import numpy as np

from pyqed.qchem.pbc import Cell, RHF


def test_pbc_cell_builds_native_1d_and_makes_kpts():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    assert cell.built
    assert cell.nao == 1
    assert cell.nelectron == 2
    assert cell.low_dim_ft_type == "inf_vacuum"
    np.testing.assert_allclose(
        cell.lattice_vectors,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
    )

    kpts = cell.make_kpts(3)
    assert kpts.shape == (3, 3)
    np.testing.assert_allclose(kpts[1], np.zeros(3), atol=1e-12)


def test_native_pbc_rhf_runs_gamma_and_kpoint_1d():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf_gamma = RHF(cell, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_gamma.e_tot)
    assert mf_gamma.nkpts == 1
    assert mf_gamma.dm.shape == (1, 1)
    assert np.isfinite(mf_gamma.e_nuc)

    kpts = cell.make_kpts(3)
    mf_k = RHF(cell, kpts=kpts, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_k.e_tot)
    assert mf_k.nkpts == 3
    assert len(mf_k.dm) == 3
    assert all(d.shape == (1, 1) for d in mf_k.dm)
    assert np.isfinite(mf_k.e_nuc)
