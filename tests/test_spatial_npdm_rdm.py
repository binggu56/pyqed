from types import SimpleNamespace

import numpy as np

from pyqed.mps.mps import MPS
from pyqed.qchem.dmrg.dmrg import DMRG


def test_spatial_npdm_rdm2_matches_reference_paths():
    rng = np.random.default_rng(11)
    nsites = 4
    bonds = [1, 3, 4, 2, 1]
    tensors = [
        rng.normal(size=(bonds[i], 4, bonds[i + 1]))
        + 0.1j * rng.normal(size=(bonds[i], 4, bonds[i + 1]))
        for i in range(nsites)
    ]
    state = MPS(tensors, labels=["lv", "p", "rv"])

    dmrg = SimpleNamespace(
        ncas=nsites,
        ncore=0,
        dmrg=SimpleNamespace(ground_state=state),
    )
    dmrg._get_state_for_rdm = DMRG._get_state_for_rdm.__get__(dmrg)
    dmrg._make_spatial_site_rdm2 = DMRG._make_spatial_site_rdm2.__get__(dmrg)

    results = {}
    for algorithm in ("gram", "direct", "npdm"):
        dmrg.spatial_rdm2_algorithm = algorithm
        results[algorithm] = dmrg._make_spatial_site_rdm2(spatial=True)

    np.testing.assert_allclose(results["npdm"], results["gram"], atol=1e-12)
    np.testing.assert_allclose(results["npdm"], results["direct"], atol=1e-12)
