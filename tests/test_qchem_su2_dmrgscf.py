import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.backends.nonabelian import _qchem_sweep_measure
from pyqed.qchem.dmrg.dmrg import QCDMRG
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF
from pyqed.qchem.mcscf.cocas import _fresh_macro_casci
from pyqed.qchem.hf import RHF


def _h2_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="dense",
        aosym="s1",
        options={"eri_backend": "cpp"},
    )
    return RHF(mol).run()


def _h2_factor_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(
        eri="factors",
        options={"eri_backend": "cpp", "low_rank_tol": 1.0e-12},
    )
    return RHF(mol).run()


def test_state_averaged_su2_dmrgscf_preserves_su2_solver():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=2,
        weights=[0.5, 0.5],
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.dmrg.backend == "su2"
    assert mc.dmrg_conv_tol == 1.0e-7
    assert mc.macro_converged is True
    assert mc.solver_converged == mc.dmrg.converged
    assert mc.converged == (mc.macro_converged and mc.solver_converged)
    assert mc.macro_iterations == 1
    assert len(mc.dmrg.states) == 2
    np.testing.assert_allclose(
        mc.e_tot,
        [-1.137275943783, -0.169291740911],
        atol=1.0e-8,
    )


def test_su2_dmrgscf_keeps_factorized_orbital_integrals(monkeypatch):
    mf = _h2_factor_rhf()

    def reject_dense_mo_eri(*_args, **_kwargs):
        raise AssertionError("factorized DMRGSCF must not assemble the full MO ERI")

    monkeypatch.setattr(mf, "get_eri_mo", reject_dense_mo_eri)
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=2,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.use_cholesky_integrals is True
    assert mc.integral_backend_override is None
    assert mc.integral_mode == "cholesky"
    assert mc.casci.integral_mode == "cholesky"
    assert mc.converged
    assert mc.casci.build_info["su2_runtime_reused"] is False
    assert mc.casci.build_info["final_su2_runtime_rebuilt"] is True
    np.testing.assert_allclose(mc.e_tot, -1.137275943783, atol=1.0e-8)


def test_abelian_spatial_dmrgscf_builds_rdms_from_mps_factors():
    mf = _h2_factor_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="sz",
        site="spatial",
        spatial_abelian_mpo="reference_grouped",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=2,
        symmetry="sz",
        compute_s2=False,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.converged
    np.testing.assert_allclose(mc.e_tot, -1.137275943783, atol=1.0e-8)
    solver = mc.casci
    solver.spatial_rdm2_algorithm = "npdm"
    reference_dm1, reference_dm2 = solver.make_rdm12(spatial=True)
    solver.spatial_rdm2_algorithm = "string"
    string_dm1, string_dm2 = solver.make_rdm12(spatial=True)
    np.testing.assert_allclose(string_dm1, reference_dm1, atol=1.0e-11)
    np.testing.assert_allclose(string_dm2, reference_dm2, atol=1.0e-11)
    assert solver.spatial_rdm_diagnostics["algorithm"] == "alpha_beta_strings"
    assert solver.spatial_rdm_diagnostics["determinants"] == 4
    assert solver.spatial_rdm_diagnostics["cache_hits"] >= 1
    assert solver.spatial_rdm_diagnostics["rdm2_seconds"] >= 0.0


def test_state_averaged_su2_dmrgscf_requires_final_inner_convergence():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    with pytest.raises(RuntimeError, match="active-space DMRG did not converge"):
        mc.run(
            nstates=2,
            weights=[0.5, 0.5],
            nsweeps=1,
            conv_tol=-1.0,
            mixer_zero_block_noise_scale=0.0,
        )

    assert mc.macro_converged is False
    assert mc.solver_converged is False
    assert mc.converged is False


def test_qchem_su2_sweep_measure_prefers_objective_residual():
    sweep_result = {
        "updates": [
            {"trunc_err": 0.0, "local_objective": {"metric": 2.5e-3}},
            {"trunc_err": 0.0, "local_objective": {"residual": 1.0e-2}},
        ]
    }

    assert _qchem_sweep_measure(sweep_result) == 1.0e-2


def test_su2_dmrgscf_separates_orbital_options_from_inner_sweeps():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=1,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    mc.run(
        nstates=1,
        nsweeps=2,
        optimizer="RCG",
        optimizer_max_steps=5,
        optimizer_max_step_norm=0.1,
        diis=False,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.converged


def test_su2_co_macro_solver_drops_topology_owned_route_cache():
    source = QCDMRG(
        _h2_rhf(),
        ncas=2,
        nelecas=2,
        D=8,
        symmetry="su2",
        init_guess="hf",
    )
    source._su2_runtime = object()

    reused = _fresh_macro_casci(source)
    trial = _fresh_macro_casci(source, rebuild_runtime=True)

    assert reused._su2_runtime is source._su2_runtime
    assert trial._su2_runtime is None
    assert trial.D == source.D


def test_su2_dmrgscf_nonredundant_orbital_driver_runs():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=2,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    mc.run(
        nstates=1,
        nsweeps=4,
        orbital_driver="nonredundant",
        optimizer="LBFGS",
        diis=False,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.converged
    assert mc.macro_converged
    assert mc.solver_converged
    assert mc.macro_diagnostics


def test_su2_dmrgscf_second_order_orbital_driver_runs():
    mf = _h2_rhf()
    mc = DMRGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        max_cycles=2,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )

    mc.run(
        nstates=1,
        nsweeps=4,
        orbital_driver="second_order",
        optimizer_max_step_norm=0.05,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.converged
    assert mc.macro_converged
    assert mc.solver_converged
    assert mc.macro_diagnostics


def test_su2_dmrgscf_h6_rebuilds_final_runtime_after_bond_topology_changes():
    atom = "; ".join(f"H 0 0 {1.8 * site}" for site in range(6))
    mol = Molecule(atom=atom, unit="bohr", basis="6-31g")
    mol.build(
        eri="factors",
        options={"eri_backend": "cpp", "low_rank_tol": 1.0e-12},
    )
    mc = DMRGSCF(
        RHF(mol).run(tol=1.0e-11),
        ncas=6,
        nelecas=6,
        D=32,
        max_cycles=8,
        macro_tol=1.0e-6,
        dmrg_conv_tol=1.0e-8,
        symmetry="su2",
        init_guess="hf",
        verbose=0,
    )
    mc.run(
        nstates=1,
        nsweeps=12,
        sweep_tol=1.0e-8,
        orb_grad_tol=1.0e-4,
        optimizer="RCG",
        optimizer_tol=1.0e-5,
        optimizer_max_steps=50,
        optimizer_max_step_norm=0.20,
        macro_trust_radius=0.20,
        warm_start_bonds=True,
        mixer_zero_block_noise_scale=0.0,
    )

    assert mc.converged
    assert 1 <= mc.macro_iterations <= 8
    assert all(row["step"] <= row["tr"] * (1.0 + 1.0e-8) for row in mc.macro_diagnostics)
    assert all(row["active_active_optimized"] for row in mc.macro_diagnostics)
    assert mc.casci.build_info["final_su2_runtime_rebuilt"] is True
    assert mc.dmrg.history[-1]["max_bond_mode"] == "reduced"
    np.testing.assert_allclose(mc.e_tot, -3.3126886264, atol=2.0e-9)
    dm1, dm2 = mc.casci.make_rdm12(spatial=True)
    np.testing.assert_allclose(np.trace(dm1), 6.0, atol=1.0e-11)
    np.testing.assert_allclose(np.einsum("pprr->", dm2), 30.0, atol=1.0e-10)
    assert mc.casci.spatial_rdm_diagnostics["magnetic_component_expansion"] is False
