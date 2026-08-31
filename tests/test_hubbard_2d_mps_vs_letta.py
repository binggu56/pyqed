import pytest
import numpy as np

from examples.mps.hubbard_2d_mps_vs_letta import (
    ed_ground_energy,
    hubbard_2d_dense_mpo,
    projected_mpo_spectrum,
    random_fixed_sector_abelian_mps,
    rung_site_qn_maps,
    site_qn_maps,
)
from pyqed.mps import MPO, MPS, dense_to_symmetric_mpo
from pyqed.mps.abelian_storage import SymmetryManager
from pyqed.mps.dmrg import DMRG, dmrg_matvec_options


def test_abelian_dmrg_history_reports_post_truncation_energy_for_2d_hubbard():
    dense_mpo, _info = hubbard_2d_dense_mpo(
        2,
        2,
        hopping=1.0,
        hubbard_u=4.0,
        ordering="snake",
    )
    qn_maps = site_qn_maps(4)
    opts = dmrg_matvec_options("symmetric")
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

    hamiltonian = MPO(symmetric_mpo)
    dmrg = DMRG(
        hamiltonian,
        D=8,
        init_guess=MPS(
            initial,
            labels=["lv", "rv", "p"],
            sites=hamiltonian.input_sites,
        ),
        nsweeps=2,
        opt="2site",
        symmetry=True,
        target_qn=target_qn,
        sym_mgr=sym_mgr,
        site_qn_maps=qn_maps,
        not_conv_err=False,
        performance="symmetric",
        abelian_matvec_options=opts,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-9,
        davidson_max_iter=80,
        noise=1.0e-7,
    )
    dmrg.run()

    last = dmrg.sweep_history[-1]
    assert last["energy"] == pytest.approx(dmrg.e_tot, abs=1.0e-12)
    assert last["post_truncation_energy"] == pytest.approx(dmrg.e_tot, abs=1.0e-12)
    assert last["local_energy"] < dmrg.e_tot - 0.1


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
