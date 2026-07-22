"""Coulomb-component names used by periodic GW/BSE kernels."""

RECIPROCAL_EWALD_LR = "reciprocal_ewald_lr"
FULL_EWALD = "full_ewald"
GDF = "gdf"
PYSCF_GDF = "pyscf_gdf"
SHORT_RANGE_EWALD = "short_range_ewald"
COULOMB_BACKGROUND = "background"

SUPPORTED_PERIODIC_COULOMB_COMPONENTS = (
    RECIPROCAL_EWALD_LR,
    FULL_EWALD,
    GDF,
    PYSCF_GDF,
)

SUPPORTED_DENSE_GAMMA_COULOMB_COMPONENTS = (
    FULL_EWALD,
    RECIPROCAL_EWALD_LR,
    SHORT_RANGE_EWALD,
    COULOMB_BACKGROUND,
)

_PERIODIC_COMPONENT_ALIASES = {
    RECIPROCAL_EWALD_LR: RECIPROCAL_EWALD_LR,
    "reciprocal": RECIPROCAL_EWALD_LR,
    "long_range": RECIPROCAL_EWALD_LR,
    "lr": RECIPROCAL_EWALD_LR,
    FULL_EWALD: FULL_EWALD,
    "full": FULL_EWALD,
    GDF: GDF,
    "df": GDF,
    "density_fit": GDF,
    "density_fitting": GDF,
    PYSCF_GDF: PYSCF_GDF,
    "pyscf_df": PYSCF_GDF,
    "pyscf_density_fit": PYSCF_GDF,
    "pyscf_density_fitting": PYSCF_GDF,
}

_DENSE_GAMMA_COMPONENT_ALIASES = {
    RECIPROCAL_EWALD_LR: RECIPROCAL_EWALD_LR,
    "reciprocal": RECIPROCAL_EWALD_LR,
    "long_range": RECIPROCAL_EWALD_LR,
    "lr": RECIPROCAL_EWALD_LR,
    FULL_EWALD: FULL_EWALD,
    "full": FULL_EWALD,
    SHORT_RANGE_EWALD: SHORT_RANGE_EWALD,
    "short_range": SHORT_RANGE_EWALD,
    "sr": SHORT_RANGE_EWALD,
    COULOMB_BACKGROUND: COULOMB_BACKGROUND,
    "coulomb_background": COULOMB_BACKGROUND,
}


def normalize_coulomb_component(component, *, dense_gamma=False):
    """Return the canonical Coulomb-component name.

    The production periodic kernels currently support reciprocal Ewald
    long-range factors and dense full-Ewald small-cell diagnostics.  Dense
    Gamma validation helpers additionally expose the native short-range and
    neutralizing-background pieces.
    """

    key = str(component).lower()
    aliases = (
        _DENSE_GAMMA_COMPONENT_ALIASES
        if dense_gamma
        else _PERIODIC_COMPONENT_ALIASES
    )
    try:
        return aliases[key]
    except KeyError as exc:
        supported = (
            SUPPORTED_DENSE_GAMMA_COULOMB_COMPONENTS
            if dense_gamma
            else SUPPORTED_PERIODIC_COULOMB_COMPONENTS
        )
        joined = ", ".join(repr(name) for name in supported)
        raise ValueError(f"coulomb_component must be one of {joined}.") from exc


def is_full_ewald_component(component):
    """Return whether ``component`` selects dense full-Ewald kernels."""

    return normalize_coulomb_component(component) == FULL_EWALD


def is_gdf_component(component):
    """Return whether ``component`` selects dependency-free GDF kernels."""

    return normalize_coulomb_component(component) == GDF


def is_pyscf_gdf_component(component):
    """Return whether ``component`` selects PySCF GDF Coulomb kernels."""

    return normalize_coulomb_component(component) == PYSCF_GDF
