import numpy as np
import pickle
import pytest

import pyqed.mps as mps_module
import pyqed.mps.nonabelian.mpo as mpo_module
import pyqed.symmetry as symmetry_module
from pyqed.lattice import BosonSite, CompositeSite, Site, SpinHalfSite
from pyqed.mps import MPS
from pyqed.mps.nonabelian.basis import SiteBasis
from pyqed.mps.nonabelian.mpo import SiteOperator
from pyqed.symmetry import Leg, IrrepTensor, u1_su2_leg


def test_site_has_one_canonical_namespace():
    assert not hasattr(mps_module, "Site")


def test_dense_mps_owns_canonical_physical_sites():
    sites = (SpinHalfSite(), BosonSite(2))
    tensors = (
        np.ones((1, 2, 2)),
        np.ones((2, 3, 1)),
    )
    state = MPS(tensors, sites=sites)

    assert state.sites == sites
    assert state.dims == [2, 3]
    assert state.copy().sites == sites
    assert state.to_order(("p", "lv", "rv")).sites == sites
    assert state.compress(2).sites == sites


def test_dense_mps_rejects_site_dimension_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        MPS((np.ones((1, 2, 1)),), sites=(BosonSite(2),))


def test_dense_mps_infers_anonymous_canonical_sites():
    state = MPS((np.ones((1, 2, 1)),))
    assert len(state.sites) == 1
    assert isinstance(state.sites[0], Site)
    assert state.sites[0].dim == 2


def test_leg_preserves_noncontiguous_product_basis_sectors():
    site = CompositeSite((SpinHalfSite(),) * 3)
    leg = site.leg
    basis = SiteBasis.from_leg(leg)
    diagonal = np.diag(np.arange(site.dim))
    packed = IrrepTensor.from_site_operator(site, diagonal).to_dense()

    assert tuple(basis.dims[sector] for sector in basis.sectors) == (1, 3, 3, 1)
    assert tuple(irrep.charge for irrep in leg.irreps) == (3, 1, -1, -3)
    np.testing.assert_array_equal(
        np.diag(packed),
        (0, 1, 2, 4, 3, 5, 6, 7),
    )


def test_mps_basis_and_mpo_leg_are_views_of_one_shared_leg():
    site = CompositeSite((SpinHalfSite(), SpinHalfSite()))
    shared_leg = site.leg
    basis = SiteBasis.from_leg(shared_leg)
    leg = basis.as_physical_leg()

    assert basis.as_physical_leg() == leg
    assert leg.total_dim == site.dim


def test_irrep_tensor_is_the_operator_source_for_the_mpo_view():
    site = CompositeSite((SpinHalfSite(), SpinHalfSite()))
    tensor = IrrepTensor.from_site_operator(site, "I")
    operator = SiteOperator.from_irrep_tensor(tensor)

    np.testing.assert_array_equal(operator.as_dense(), np.eye(site.dim))


def test_site_operator_requires_one_homogeneous_charge_transfer():
    site = SpinHalfSite()

    assert IrrepTensor.from_site_operator(site, "Sz").op.charge == 0
    assert IrrepTensor.from_site_operator(site, "Sp").op.charge == 2
    with pytest.raises(ValueError, match="multiple charge transfers"):
        IrrepTensor.from_site_operator(site, "Sx")


def test_mps_does_not_export_parallel_site_conversion_functions():
    for name in (
        "abelian_state_qns",
        "conserved_site_from_site",
        "grouped_state_indices",
        "physical_leg_from_site",
        "site_basis_from_site",
        "site_operator_from_site",
    ):
        assert not hasattr(mps_module, name)


def test_leg_is_not_redefined_in_narg():
    import pyqed.narg as narg

    assert not hasattr(narg, "Leg")


def test_removed_parallel_leg_classes_are_not_exported():
    assert not hasattr(mpo_module, "PhysicalLeg")
    assert not hasattr(symmetry_module, "IrrepSite")


def test_nonabelian_leg_distinguishes_reduced_and_full_dimensions():
    leg = u1_su2_leg(((0, 0, 1), (1, 1, 1), (2, 0, 1)))

    assert leg.reduced_dim == leg.dim == 3
    assert leg.full_dim == 4
    assert leg.dual().full_dim == 4
    assert pickle.loads(pickle.dumps(leg)) == leg


def test_legacy_conserved_site_module_is_removed():
    with pytest.raises(ModuleNotFoundError):
        __import__("pyqed.mps.abelian")
