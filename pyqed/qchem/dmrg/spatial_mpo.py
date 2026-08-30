"""Spatial-site carrier MPOs for block2-like Abelian DMRG.

The block2-table path applies the Hamiltonian through renormalized R/P family
MPO environments. In that mode the sweep still needs a well-formed spatial MPO
to initialize ordinary left/right environments and preserve the legacy DMRG
interface, but it does not need the expensive full spin-orbital carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.tn import MPO as TensorMPO


@dataclass(frozen=True)
class SpatialCarrierMPO:
    """A lightweight spatial carrier plus metadata."""

    factors: list[np.ndarray]
    info: dict

    @property
    def tensor_mpo(self):
        return TensorMPO(self.factors, homogeneous=False)


def build_spatial_block2_carrier_mpo(n_sites, *, local_dim=4):
    """Return a minimal identity-path carrier for spatial family environments."""

    n_sites = int(n_sites)
    local_dim = int(local_dim)
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")
    if local_dim <= 0:
        raise ValueError("local_dim must be positive.")
    try:
        from pyqed.mps import cpp_davidson as _cpp_davidson

        build = getattr(_cpp_davidson, "build_spatial_block2_carrier_mpo", None)
        if build is not None:
            native = build(n_sites, local_dim)
            return SpatialCarrierMPO(
                factors=list(native["factors"]),
                info=dict(native["info"]),
            )
    except Exception:
        pass
    ident = np.eye(local_dim, dtype=complex).reshape(1, 1, local_dim, local_dim)
    factors = [ident.copy() for _ in range(n_sites)]
    info = {
        "representation": "spatial_block2_table_carrier_mpo",
        "source": "spatial_identity_scaffold",
        "site": "spatial",
        "physical_dim": local_dim,
        "n_sites": n_sites,
        "mpo_max_bond": 1,
        "requires_complementary_family_mpos": True,
        "replaces_grouped_spin_orbital_carrier": True,
    }
    return SpatialCarrierMPO(factors=factors, info=info)
