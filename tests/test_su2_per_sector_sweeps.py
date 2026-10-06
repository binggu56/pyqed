"""Per-sector reduced sweeps against an explicit small ED reference."""

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import ED
from pyqed.qchem.dmrg.dmrg import QCDMRG


@pytest.mark.parametrize("core_shift", [0., -2000.])
@pytest.mark.parametrize("arithmetic", ["real", "complex"])
def test_per_sector_budget_uses_owned_sweeps_and_matches_ed(core_shift, arithmetic, monkeypatch):
    monkeypatch.setenv("PYQED_SU2_FORCE_COMPLEX_DAVIDSON", str(int(arithmetic == "complex")))
    norb = 4
    workspace_bytes = 1024 if arithmetic == "real" else 2048
    mol = Molecule(atom="; ".join(f"H 0 0 {1.6*i}" for i in range(norb)),
                   unit="bohr", basis="sto-3g")
    mol.build(eri="factors", options={"eri_backend": "cpp", "low_rank_tol": 1e-12})
    mf = mol.RHF().run(tol=1e-11)
    nuclear_energy = mol.energy_nuc()
    mol.energy_nuc = lambda: nuclear_energy + core_shift
    mf.e_tot += core_shift
    reference = ED(mf, ncas=norb, nelecas=norb, symmetry="su2", verbose=0).run()
    solver = QCDMRG(mf, ncas=norb, nelecas=norb, D=32, symmetry="su2", init_guess="hf", verbose=0)
    solver.run(nsweeps=12, conv_tol=1e-9, max_bond_mode="per_sector",
               su2_kernel_backend="cpp", mixer_zero_block_noise_scale=0., require_convergence=False,
               local_solver_kwargs={"workspace_budget_bytes": workspace_bytes})
    assert all(row["cpp_owned_half_sweep"] for row in solver.dmrg.history)
    assert all(row["max_bond_mode"] == "per_sector" for row in solver.dmrg.history)
    assert all(update["local_objective"]["estimated_basis_workspace_bytes"] <= workspace_bytes
               for row in solver.dmrg.history for update in row["updates"])
    np.testing.assert_allclose(solver.e_tot, reference.e_tot, rtol=0., atol=1e-8)
    state = solver.export_ground_state()
    assert all(dim <= 32 for b in range(len(state)-1) for dim in state.bond_basis(b).dims.values())
