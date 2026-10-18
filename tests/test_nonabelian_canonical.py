"""Exact reduced gauge moves must not create null virtual directions."""
import numpy as np
import pytest
from pyqed.mps.nonabelian.canonical import mixed_canonicalize_sites, mixed_canonical_errors
from pyqed.mps.nonabelian.environment import contract_chain_expectation
from pyqed.mps.nonabelian.states import build_random_reduced_spatial_mps
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.su2 import SpinChargeSector, SU2Irrep


@pytest.mark.parametrize('center', [0, 2, 5])
@pytest.mark.parametrize('spatial', [False, True])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('charge,two_j', [(6,0), (6,2), (5,1)])
def test_exact_reduced_canonicalization_preserves_state_and_bond_spaces(center, spatial, dtype, charge, two_j):
    sites = build_random_reduced_spatial_mps(6, bond_multiplicity=2, seed=17,
        dtype=dtype, target_sector=SpinChargeSector(charge, SU2Irrep(two_j)),
        orbsym=[0,1,2,3,1,2] if spatial else None)
    if dtype is complex:
        rng = np.random.default_rng(18)
        for site in sites:
            for block in site.data.values():
                block += 1j*rng.normal(size=block.shape)
    before = [site.copy() for site in sites]
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, [None]*len(sites))
    norm = contract_chain_expectation(sites, identity)
    canonical = mixed_canonicalize_sites(sites, center, cutoff=0., max_bond=None)
    for old, new in zip(before, canonical):
        assert all(dim <= old.legs[2].dims.get(sector, 0)
                   for sector, dim in new.legs[2].dims.items())
    np.testing.assert_allclose(contract_chain_expectation(canonical, identity), norm, rtol=3e-12)
    np.testing.assert_allclose(contract_chain_expectation(canonical, identity, bra_sites=sites),
                               norm, rtol=3e-12)
    assert max(mixed_canonical_errors(canonical, center)) < 1e-12
    for old, new in zip(before, sites):
        for key in old.data:
            np.testing.assert_array_equal(old.data[key], new.data[key])
