import numpy as np

from pyqed.lattice import Block, SpinHalfFermionSite
from pyqed.mps.fermion import SpinHalfFermionChain
from pyqed.symmetry import Irrep, Leg


def _noninteracting_chain(nsites, nelec):
    h1e = np.diag(np.arange(nsites, dtype=float))
    eri = np.zeros((nsites,) * 4)
    return SpinHalfFermionChain(h1e, eri, nelec=nelec)


def test_spinful_fermion_site_carries_the_legacy_operator_aliases():
    site = SpinHalfFermionSite()

    np.testing.assert_array_equal(site.operator("Ntot"), site.operator("N"))
    np.testing.assert_array_equal(site.operator("NuNd"), site.operator("double"))
    np.testing.assert_array_equal(site.operator("JW"), site.operator("parity"))


def test_narg_reexports_the_shared_block_type():
    from pyqed.narg import Block as NARGBlock

    assert NARGBlock is Block


def test_exact_chain_owns_canonical_sites_and_an_irrep_sector_layout():
    chain = _noninteracting_chain(2, nelec=(1, 0)).run()

    assert all(isinstance(site, SpinHalfFermionSite) for site in chain.sites)
    assert chain.sites == (chain.site, chain.site)
    assert isinstance(chain.sector_space, Leg)
    assert isinstance(chain.block, Block)
    assert chain.block.qn is chain.sector_space
    assert chain.block.data["indices"] is chain.sector_indices
    assert chain.target_irrep == Irrep((1, 1))
    assert chain.sector_indices[chain.target_irrep].size == 2
    np.testing.assert_allclose(chain.e_tot, [0.0], atol=1.0e-12)


def test_exact_chain_number_only_sector_uses_the_same_irrep_interface():
    chain = _noninteracting_chain(2, nelec=1).run()

    assert chain.target_irrep == Irrep(1)
    assert chain.sector_indices[chain.target_irrep].size == 4
    np.testing.assert_allclose(chain.e_tot, [0.0], atol=1.0e-12)


def test_exact_chain_sector_layout_supports_non_power_of_two_site_counts():
    chain = _noninteracting_chain(3, nelec=None).run()

    assert tuple(irrep.charge for irrep in chain.sector_space.irreps) == tuple(range(7))
    assert tuple(chain.sector_space.dims.values()) == (1, 6, 15, 20, 15, 6, 1)
    assert sum(chain.sector_space.dims.values()) == 4**3
