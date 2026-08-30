import numpy as np
import pytest

from pyqed.lattice import (
    BosonSite,
    CompositeSite,
    Site,
    SpinHalfFermionSite,
    SpinHalfSite,
    SpinlessFermionSite,
)
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierAbelianLayout,
    FrontierTiedLETTA,
)
from pyqed.tn import LocalHamiltonian, LocalTerm
from pyqed.mps import MPO as PublicMPO
from pyqed.tn import MPO as DenseMPO
from pyqed.symmetry import Irrep, IrrepTensor, Leg, U1Symmetry
from pyqed.tn import MPO


def test_site_validates_and_freezes_local_data():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    site = Site(
        labels=("zero", "one"),
        operators={"X": x},
        charges=(0, 1),
        parities=(0, 1),
    )

    assert site.dim == site.d == 2
    assert site.labels == ("zero", "one")
    assert site.charges == ((0,), (1,))
    np.testing.assert_array_equal(site.operator("I"), np.eye(2))
    np.testing.assert_array_equal(site.operator("X"), x)
    with pytest.raises(ValueError):
        site.operator("X")[0, 0] = 2.0
    with pytest.raises(TypeError):
        site.operators["Z"] = np.eye(2)
    with pytest.raises(AttributeError):
        site._labels = ("changed",)


def test_site_accepts_canonical_leg_basis_and_statistics():
    leg = Leg(
        {Irrep(0): 1, Irrep(1): 1},
        symmetry=U1Symmetry("n"),
    )
    site = Site(
        leg=leg,
        basis=("empty", "occupied"),
        statistics="fermionic",
    )

    assert site.leg is leg
    assert site.basis == ("empty", "occupied")
    assert site.statistics == "fermionic"
    assert site.parities == (0, 1)
    number = IrrepTensor.from_site_operator(site, np.diag([0.0, 1.0]))
    assert number.bra is leg
    assert number.ket is leg


def test_standard_sites_have_basis_aligned_charges_and_operators():
    spin = SpinHalfSite()
    spinless = SpinlessFermionSite()
    spinful = SpinHalfFermionSite()
    boson = BosonSite(3)

    assert spin.charges == ((1,), (-1,))
    assert spin.charge_labels == ("2sz",)
    np.testing.assert_array_equal(spin.operator("Sz"), np.diag([0.5, -0.5]))
    assert spinless.charges == ((0,), (1,))
    assert spinless.parities == (0, 1)
    assert spinful.charges == ((0, 0), (1, 1), (1, -1), (2, 0))
    assert spinful.charge_labels == ("n", "2sz")
    assert spinful.parities == (0, 1, 1, 0)
    assert boson.dim == 4
    assert boson.charges == ((0,), (1,), (2,), (3,))


def test_composite_site_retains_factorization_and_fuses_charges():
    block = CompositeSite((SpinHalfSite(), SpinHalfSite()))

    assert block.dim == 4
    assert block.factor_dims == (2, 2)
    assert block.charges == ((2,), (0,), (0,), (-2,))
    assert block.flatten((1, 0)) == 2
    assert block.unflatten(2) == (1, 0)
    np.testing.assert_array_equal(
        block.operator_on(1, "Sz"),
        np.kron(np.eye(2), np.diag([0.5, -0.5])),
    )
    np.testing.assert_array_equal(
        block.product_operator(("Sx", "Sz")),
        np.kron(block.factors[0].operator("Sx"), block.factors[1].operator("Sz")),
    )


def test_composite_site_fuses_fermion_parity():
    block = CompositeSite((SpinlessFermionSite(), SpinlessFermionSite()))
    assert block.parities == (0, 1, 1, 0)


def test_hamiltonian_owns_sites_and_frontier_derives_dimensions():
    sites = (SpinHalfSite(), SpinHalfSite())
    exchange = sum(
        np.kron(sites[0].operator(name), sites[1].operator(name))
        for name in ("Sx", "Sy", "Sz")
    )
    hamiltonian = LocalHamiltonian(
        sites,
        (LocalTerm((0, 1), exchange),),
    )
    state = FrontierTiedLETTA(
        hamiltonian,
        ((1,), ()),
        bond_dim=1,
        seed=3,
    )

    assert hamiltonian.sites == sites
    assert hamiltonian.dims == (2, 2)
    assert state.sites == sites
    assert state.dims == (2, 2)
    assert np.isfinite(state.expectation())


def test_local_hamiltonian_builds_the_single_canonical_dense_mpo():
    sites = (SpinHalfSite(), SpinHalfSite())
    exchange = sum(
        np.kron(sites[0].operator(name), sites[1].operator(name))
        for name in ("Sx", "Sy", "Sz")
    )

    mpo = LocalHamiltonian(sites, (((0, 1), exchange),)).to_mpo()

    assert type(mpo) is MPO is PublicMPO is DenseMPO
    assert mpo.sites == sites
    assert mpo.factors is mpo.tensors


def test_abelian_layout_is_constructed_directly_from_sites():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = LocalHamiltonian(sites)
    layout = FrontierAbelianLayout.from_sites(
        sites,
        target=(0,),
        bond_dims=(1, 2, 2, 2, 1),
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        ((), (), (), ()),
        abelian_layout=layout,
        seed=4,
    )

    assert layout.local_qns == tuple(site.charges for site in sites)
    assert layout.dims == (2, 2, 2, 2)
    assert state.sites == sites


def test_integer_hamiltonian_dimensions_are_only_a_canonical_site_adapter():
    hamiltonian = LocalHamiltonian((2, 3))
    assert all(isinstance(site, Site) for site in hamiltonian.sites)
    assert hamiltonian.dims == (2, 3)
