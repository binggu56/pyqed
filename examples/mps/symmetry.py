"""Inspect the symmetry decomposition of a fused canonical fermion site."""

from pyqed.lattice import CompositeSite, SpinHalfFermionSite
from pyqed.symmetry import IrrepSite


physical_site = CompositeSite((SpinHalfFermionSite(),) * 3)
symmetry_space = IrrepSite.from_site(physical_site)

for irrep, dimension in symmetry_space.dims.items():
    print(f"sector {irrep.charge}: dimension {dimension}")
