"""Native OM2/MRCI public API.

The module provides a runnable semiempirical OM2-style reference and MRCI
driver while keeping the exact OMx kernels isolated behind small functions.
The default parameters are the H/C/N/O/F OM2 values reported by Dral et al.,
J. Chem. Theory Comput. 12, 1082 (2016), Table 2.  The current integral model
uses a compact NDDO/orthogonalized-valence approximation; replacing it with the
full published OM2 orthogonalization/ECP kernels should not change the user
API.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

from pyqed.qchem.ci.fci import CI_H, SlaterCondon, givenΛgetB


EV_TO_HARTREE = 1.0 / 27.211386245988
ANGSTROM_TO_BOHR = 1.8897261246257702


class SemiempiricalMethodNotAvailable(NotImplementedError):
    """Raised when a requested semiempirical backend is not implemented."""


class OM2ParameterError(SemiempiricalMethodNotAvailable):
    """Raised when OM2 parameters or Hamiltonian builders are unavailable."""


@dataclass
class OM2HamiltonianData:
    """Semiempirical Hamiltonian data in an orthogonal valence AO basis."""

    hcore: np.ndarray
    eri: np.ndarray | None
    enuc: float
    nelec: int | tuple[int, int]
    orbital_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValenceOrbital:
    atom_index: int
    symbol: str
    shell: str
    axis: str | None = None

    @property
    def label(self):
        suffix = self.shell if self.axis is None else f"{self.shell}{self.axis}"
        return f"{self.symbol}{self.atom_index + 1}:{suffix}"


@dataclass(frozen=True)
class CIConfigurationData:
    active_binary: np.ndarray
    binary: np.ndarray
    active_orbitals: tuple[int, ...]
    frozen_occ: np.ndarray


@dataclass(frozen=True)
class OM2AtomicParameters:
    """Per-element OM2 parameter record.

    Energies are stored in Hartree unless a custom :class:`OM2ParameterSet`
    declares ``energy_unit="ev"``.
    """

    uss: float
    upp: float | None = None
    beta_s: float | None = None
    beta_p: float | None = None
    beta_pi: float | None = None
    beta_s_h: float | None = None
    beta_p_h: float | None = None
    alpha_s: float | None = None
    alpha_p: float | None = None
    alpha_pi: float | None = None
    alpha_s_h: float | None = None
    alpha_p_h: float | None = None
    zeta_s: float | None = None
    zeta_p: float | None = None
    core_charge: float | None = None
    gamma_ss: float | None = None
    gamma_sp: float | None = None
    gamma_pp: float | None = None
    gamma_pi: float | None = None
    f1: float | None = None
    f2: float | None = None
    g1: float | None = None
    g2: float | None = None
    ecp_zeta: float | None = None
    ecp_faa: float | None = None
    ecp_beta: float | None = None
    ecp_alpha: float | None = None


class OM2ParameterSet:
    """Validated OM2 parameter provider."""

    def __init__(self, elements=None, name="custom", energy_unit="hartree", citation=None):
        self.name = name
        self.energy_unit = str(energy_unit).lower()
        self.citation = citation
        self.elements = {}
        for symbol, params in (elements or {}).items():
            self.elements[_normalize_symbol(symbol)] = self._coerce_atomic_parameters(params)

    @classmethod
    def from_dict(cls, data, name="custom"):
        elements = data.get("elements", data)
        return cls(
            elements=elements,
            name=data.get("name", name) if isinstance(data, dict) else name,
            energy_unit=data.get("energy_unit", "hartree") if isinstance(data, dict) else "hartree",
            citation=data.get("citation") if isinstance(data, dict) else None,
        )

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data, name=data.get("name", path.stem))

    @staticmethod
    def _coerce_atomic_parameters(params):
        if isinstance(params, OM2AtomicParameters):
            return params
        return OM2AtomicParameters(**params)

    def for_symbol(self, symbol):
        symbol = _normalize_symbol(symbol)
        try:
            return self.elements[symbol]
        except KeyError as exc:
            raise OM2ParameterError(f"No OM2 parameters available for element {symbol!r}.") from exc

    def to_internal_units(self):
        if self.energy_unit in {"hartree", "eh", "au"}:
            return self
        if self.energy_unit not in {"ev", "electronvolt", "electronvolts"}:
            raise OM2ParameterError(f"Unsupported OM2 parameter energy unit {self.energy_unit!r}.")
        converted = {}
        energy_fields = {
            "uss",
            "upp",
            "beta_s",
            "beta_p",
            "beta_pi",
            "beta_s_h",
            "beta_p_h",
            "gamma_ss",
            "gamma_sp",
            "gamma_pp",
            "gamma_pi",
            "ecp_faa",
            "ecp_beta",
        }
        for symbol, params in self.elements.items():
            values = params.__dict__.copy()
            for field in energy_fields:
                if values.get(field) is not None:
                    values[field] = values[field] * EV_TO_HARTREE
            converted[symbol] = values
        return OM2ParameterSet(converted, name=self.name, energy_unit="hartree", citation=self.citation)


def _default_om2_parameters():
    # OM2 one-electron parameters from Dral et al., JCTC 2016, Table 2.
    elements_ev = {
        "H": dict(
            zeta_s=1.47386481,
            uss=-12.64890000,
            beta_s=-3.41998220,
            alpha_s=0.06607903,
            core_charge=1,
            gamma_ss=12.8480000,
            f1=0.29566861,
            f2=1.40190659,
            g1=0.65271563,
            g2=0.90843670,
        ),
        "C": dict(
            zeta_s=1.42036892,
            zeta_p=1.42036892,
            uss=-51.65550844,
            upp=-39.74369825,
            beta_s=-7.21406021,
            beta_p=-4.14394503,
            beta_pi=-5.97107657,
            alpha_s=0.09045297,
            alpha_p=0.05452192,
            alpha_pi=0.10204903,
            beta_s_h=-6.30164062,
            beta_p_h=-4.04444703,
            alpha_s_h=0.09668329,
            alpha_p_h=0.05283694,
            core_charge=4,
            gamma_ss=10.5900000,
            gamma_sp=9.5600000,
            gamma_pp=8.8600000,
            gamma_pi=7.8600000,
            f1=0.49949211,
            f2=0.72261226,
            g1=0.21284361,
            g2=0.99250289,
            ecp_zeta=5.16802668,
            ecp_faa=-305.68646337,
            ecp_beta=-9.07185084,
            ecp_alpha=0.16985745,
        ),
        "N": dict(
            zeta_s=1.33175233,
            zeta_p=1.33175233,
            uss=-74.37638240,
            upp=-57.60067613,
            beta_s=-10.84303446,
            beta_p=-7.62373736,
            beta_pi=-9.27936312,
            alpha_s=0.08974553,
            alpha_p=0.08759680,
            alpha_pi=0.13172314,
            beta_s_h=-9.49567107,
            beta_p_h=-8.51180846,
            alpha_s_h=0.11429048,
            alpha_p_h=0.10673732,
            core_charge=5,
            gamma_ss=12.2300000,
            gamma_sp=11.4700000,
            gamma_pp=11.0800000,
            gamma_pi=9.8400000,
            f1=0.64073384,
            f2=0.19580808,
            g1=0.13946233,
            g2=0.84373060,
            ecp_zeta=6.93980600,
            ecp_faa=-407.39202305,
            ecp_beta=-9.97910210,
            ecp_alpha=0.16173024,
        ),
        "O": dict(
            zeta_s=1.55214516,
            zeta_p=1.55214516,
            uss=-101.82723464,
            upp=-78.92823923,
            beta_s=-10.64436974,
            beta_p=-8.63610952,
            beta_pi=-9.21201190,
            alpha_s=0.13062089,
            alpha_p=0.09626876,
            alpha_pi=0.13071747,
            beta_s_h=-6.54238767,
            beta_p_h=-10.11307271,
            alpha_s_h=0.11112738,
            alpha_p_h=0.11891861,
            core_charge=6,
            gamma_ss=13.5900000,
            gamma_sp=12.6600000,
            gamma_pp=12.9800000,
            gamma_pi=11.5900000,
            f1=1.26450169,
            f2=1.14847352,
            g1=0.28309603,
            g2=0.78414131,
            ecp_zeta=7.58579774,
            ecp_faa=-514.45812327,
            ecp_beta=-14.16551053,
            ecp_alpha=0.34390559,
        ),
        "F": dict(
            zeta_s=1.45216726,
            zeta_p=1.45216726,
            uss=-120.62785370,
            upp=-107.27105397,
            beta_s=-6.25438426,
            beta_p=-13.93492471,
            beta_pi=-18.73205761,
            alpha_s=0.26624434,
            alpha_p=0.12261412,
            alpha_pi=0.21684388,
            beta_s_h=-6.25104378,
            beta_p_h=-13.94492971,
            alpha_s_h=0.44713918,
            alpha_p_h=0.15648906,
            core_charge=7,
            gamma_ss=15.4200000,
            gamma_sp=14.4800000,
            gamma_pp=14.5200000,
            gamma_pi=12.9800000,
            f1=2.11499396,
            f2=1.09156321,
            g1=0.31704089,
            g2=0.02140504,
            ecp_zeta=8.71226515,
            ecp_faa=-685.41988599,
            ecp_beta=-9.17960365,
            ecp_alpha=0.99971548,
        ),
    }
    return OM2ParameterSet(
        elements_ev,
        name="om2-dral-2016-h-c-n-o-f",
        energy_unit="ev",
        citation="Dral et al., J. Chem. Theory Comput. 12, 1082 (2016), Table 2.",
    ).to_internal_units()


@dataclass
class SemiempiricalMolecule:
    """Lightweight molecule container for valence semiempirical methods."""

    atom: Any
    charge: int = 0
    spin: int = 0
    unit: str = "angstrom"

    def atom_symbols(self):
        symbols, _ = _parse_atom_spec(self.atom)
        return symbols

    def atom_coords(self):
        _, coords = _parse_atom_spec(self.atom)
        return np.asarray(coords, dtype=float)

    @property
    def natom(self):
        return len(self.atom_symbols())


def _normalize_symbol(symbol):
    symbol = str(symbol).strip()
    if not symbol:
        raise ValueError("Empty element symbol.")
    return symbol[0].upper() + symbol[1:].lower()


DEFAULT_OM2_PARAMETERS = _default_om2_parameters()


def _parse_atom_spec(atom):
    if isinstance(atom, str):
        chunks = [chunk.strip() for chunk in atom.replace("\n", ";").split(";") if chunk.strip()]
        symbols = []
        coords = []
        for chunk in chunks:
            parts = chunk.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(f"Invalid atom specification chunk: {chunk!r}")
            symbols.append(_normalize_symbol(parts[0]))
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return tuple(symbols), np.asarray(coords, dtype=float)

    symbols = []
    coords = []
    for item in atom:
        if len(item) != 2:
            raise ValueError(f"Invalid atom entry: {item!r}")
        symbol, coord = item
        symbols.append(_normalize_symbol(symbol))
        coords.append(np.asarray(coord, dtype=float))
    return tuple(symbols), np.asarray(coords, dtype=float)


def _coords_to_bohr(coords, unit):
    unit = str(unit).lower()
    if unit in {"bohr", "b", "au", "a.u."}:
        return np.asarray(coords, dtype=float)
    if unit in {"angstrom", "ang", "a"}:
        return np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
    raise ValueError(f"Unsupported coordinate unit {unit!r}.")


def _valence_shells(symbol):
    symbol = _normalize_symbol(symbol)
    if symbol == "H":
        return [("1s", None)]
    if symbol in {"B", "C", "N", "O", "F"}:
        return [("2s", None), ("2p", "x"), ("2p", "y"), ("2p", "z")]
    if symbol in {"Si", "P", "S", "Cl"}:
        return [("3s", None), ("3p", "x"), ("3p", "y"), ("3p", "z")]
    raise OM2ParameterError(f"Valence orbital template for element {symbol!r} is not implemented.")


def _orbital_kind(orb):
    return "s" if orb.axis is None else "p"


def _orbital_axis_vector(orb):
    if orb.axis is None:
        return None
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    return axes[orb.axis]


def _gamma_for_orbitals(params_i, params_j, orb_i, orb_j, r_bohr):
    gi = _onsite_gamma(params_i, orb_i)
    gj = _onsite_gamma(params_j, orb_j)
    rho = 2.0 / max(gi + gj, 1e-12)
    return 1.0 / np.sqrt(r_bohr * r_bohr + rho * rho)


def _onsite_gamma(params, orb):
    if orb.axis is None:
        return params.gamma_ss or abs(params.uss) * 0.5
    return params.gamma_pp or params.gamma_ss or abs(params.upp or params.uss) * 0.5


def _one_center_coulomb(params, orb_i, orb_j):
    """Return the NDDO one-center Coulomb parameter for two AO densities."""
    kind_i = _orbital_kind(orb_i)
    kind_j = _orbital_kind(orb_j)
    if kind_i == "s" and kind_j == "s":
        return params.gamma_ss or abs(params.uss) * 0.5
    if kind_i != kind_j:
        return params.gamma_sp or params.gamma_ss or params.gamma_pp or abs(params.uss) * 0.5
    if orb_i.axis == orb_j.axis:
        return params.gamma_pp or params.gamma_ss or abs(params.upp or params.uss) * 0.5
    return params.gamma_pi or params.gamma_pp or params.gamma_ss or abs(params.upp or params.uss) * 0.5


def _pair_resonance(params_i, params_j, orb_i, orb_j, r_vec):
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        return 0.0
    rhat = r_vec / r
    kind_i = _orbital_kind(orb_i)
    kind_j = _orbital_kind(orb_j)
    if kind_i == "s" and kind_j == "s":
        beta_i = _xh_value(params_i, params_j, "beta_s_h", "beta_s")
        beta_j = _xh_value(params_j, params_i, "beta_s_h", "beta_s")
        alpha_i = _xh_value(params_i, params_j, "alpha_s_h", "alpha_s", default=0.1)
        alpha_j = _xh_value(params_j, params_i, "alpha_s_h", "alpha_s", default=0.1)
        alpha = 0.5 * (alpha_i + alpha_j)
        angular = 1.0
    elif kind_i == "s" or kind_j == "s":
        p_orb = orb_j if kind_i == "s" else orb_i
        p_params = params_j if kind_i == "s" else params_i
        s_params = params_i if kind_i == "s" else params_j
        axis = _orbital_axis_vector(p_orb)
        angular = float(np.dot(axis, rhat))
        if kind_j == "s":
            angular *= -1.0
        beta_i = p_params.beta_p_h if s_params.core_charge == 1 and p_params.beta_p_h is not None else p_params.beta_p
        beta_j = s_params.beta_s
        alpha_p = p_params.alpha_p_h if s_params.core_charge == 1 and p_params.alpha_p_h is not None else p_params.alpha_p
        alpha = 0.5 * ((alpha_p or 0.1) + (s_params.alpha_s or 0.1))
    else:
        ai = _orbital_axis_vector(orb_i)
        aj = _orbital_axis_vector(orb_j)
        ci = float(np.dot(ai, rhat))
        cj = float(np.dot(aj, rhat))
        sigma = ci * cj
        pi = float(np.dot(ai, aj)) - sigma
        beta_sigma = 0.5 * ((params_i.beta_p or 0.0) + (params_j.beta_p or 0.0))
        beta_pi = 0.5 * ((params_i.beta_pi or params_i.beta_p or 0.0) + (params_j.beta_pi or params_j.beta_p or 0.0))
        alpha = 0.5 * ((params_i.alpha_p or 0.1) + (params_j.alpha_p or 0.1))
        overlap = np.exp(-alpha * r) / np.sqrt(max(r, 1e-8))
        return (beta_sigma * sigma + beta_pi * pi) * overlap

    overlap = np.exp(-alpha * r) / np.sqrt(max(r, 1e-8))
    return 0.5 * (beta_i + beta_j) * angular * overlap


def _pair_overlap(params_i, params_j, orb_i, orb_j, r_vec):
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        return 1.0 if orb_i == orb_j else 0.0
    rhat = r_vec / r
    zi = params_i.zeta_s if orb_i.axis is None else params_i.zeta_p
    zj = params_j.zeta_s if orb_j.axis is None else params_j.zeta_p
    zeta = _mean_present((zi, zj), default=1.0)
    radial = np.exp(-zeta * r) * (1.0 + zeta * r + (zeta * r) ** 2 / 3.0)
    kind_i = _orbital_kind(orb_i)
    kind_j = _orbital_kind(orb_j)
    if kind_i == "s" and kind_j == "s":
        return radial
    if kind_i == "s" or kind_j == "s":
        p_orb = orb_j if kind_i == "s" else orb_i
        angular = float(np.dot(_orbital_axis_vector(p_orb), rhat))
        if kind_j == "s":
            angular *= -1.0
        return angular * radial
    ai = _orbital_axis_vector(orb_i)
    aj = _orbital_axis_vector(orb_j)
    sigma = float(np.dot(ai, rhat) * np.dot(aj, rhat))
    pi = float(np.dot(ai, aj) - sigma)
    return (sigma + 0.5 * pi) * radial


def _xh_value(params, partner_params, xh_field, default_field, default=0.0):
    if partner_params.core_charge == 1 and params.core_charge != 1:
        value = getattr(params, xh_field, None)
        if value is not None:
            return value
    value = getattr(params, default_field, None)
    return default if value is None else value


def _mean_present(values, default=0.0):
    present = [value for value in values if value is not None]
    if not present:
        return default
    return float(np.mean(present))


def _core_attraction(params_orb, params_core, orb, r_bohr):
    gamma = _core_gamma(params_orb, params_core, orb, r_bohr)
    return -(params_core.core_charge or 0.0) * gamma


def _core_gamma(params_orb, params_core, orb, r_bohr):
    gi = _onsite_gamma(params_orb, orb)
    gj = params_core.gamma_ss or abs(params_core.uss) * 0.5
    rho = 2.0 / max(gi + gj, 1e-12)
    return 1.0 / np.sqrt(r_bohr * r_bohr + rho * rho)


def _ecp_core_correction(params_core, r_bohr):
    """Deprecated raw ECP attraction term.

    The OM2 ECP-like parameters belong to the published OMx one-electron
    Hamiltonian/orthogonalization expressions.  Adding them directly as an
    extra core attraction grossly overbinds heavy-atom bonds, so the compact
    native kernel intentionally does not call this helper.
    """
    if params_core.ecp_zeta is None:
        return 0.0
    faa = params_core.ecp_faa or 0.0
    beta = params_core.ecp_beta or 0.0
    alpha = params_core.ecp_alpha or 0.0
    gaussian = faa * np.exp(-params_core.ecp_zeta * r_bohr * r_bohr)
    exponential = beta * np.exp(-alpha * r_bohr)
    return gaussian + exponential


def _local_core_diag(params_orb, params_core, orb, r_bohr):
    u = params_orb.uss if orb.axis is None else params_orb.upp
    return u + _core_attraction(params_orb, params_core, orb, r_bohr)


def _build_overlap_matrix(orbitals, params_by_orb, coords):
    nbf = len(orbitals)
    overlap = np.eye(nbf)
    for p, orb_p in enumerate(orbitals):
        for q, orb_q in enumerate(orbitals[:p]):
            if orb_p.atom_index == orb_q.atom_index:
                value = 0.0
            else:
                r_vec = coords[orb_q.atom_index] - coords[orb_p.atom_index]
                value = _pair_overlap(params_by_orb[p], params_by_orb[q], orb_p, orb_q, r_vec)
            overlap[p, q] = overlap[q, p] = value
    return overlap


def _om2_orthogonalization_correction(beta, overlap, local_core, orbitals, atom_params):
    """OM2 additive one-electron orthogonalization correction.

    This follows the OMx structure described in Dral et al. JCTC 2016:
    one-center block corrections use F1/F2 parameters of the partner atom,
    and two-center blocks receive the OM2 three-center G1/G2 correction from
    all remaining atoms.  The exact MNDO2005 implementation evaluates the same
    contractions with fully analytic OMx overlap/local-core integrals; this
    native implementation keeps the contractions explicit and isolated.
    """
    nbf = len(orbitals)
    correction = np.zeros_like(beta)
    atom_orbs = {}
    for idx, orb in enumerate(orbitals):
        atom_orbs.setdefault(orb.atom_index, []).append(idx)

    for p, orb_p in enumerate(orbitals):
        atom_p = orb_p.atom_index
        for q, orb_q in enumerate(orbitals):
            atom_q = orb_q.atom_index
            if atom_p == atom_q:
                for atom_b, idxs_b in atom_orbs.items():
                    if atom_b == atom_p:
                        continue
                    params_b = atom_params[atom_b]
                    f1 = params_b.f1 or 0.0
                    f2 = params_b.f2 or 0.0
                    for lam in idxs_b:
                        correction[p, q] -= 0.5 * f1 * (
                            overlap[p, lam] * beta[lam, q]
                            + beta[p, lam] * overlap[lam, q]
                        )
                        correction[p, q] += 0.25 * f2 * overlap[p, lam] * overlap[q, lam] * (
                            local_core[p, atom_b] + local_core[q, atom_b]
                        )
            else:
                for atom_c, idxs_c in atom_orbs.items():
                    if atom_c in {atom_p, atom_q}:
                        continue
                    params_c = atom_params[atom_c]
                    g1 = params_c.g1 or 0.0
                    g2 = params_c.g2 or 0.0
                    for nu in idxs_c:
                        correction[p, q] -= 0.5 * g1 * (
                            overlap[p, nu] * beta[nu, q]
                            + beta[p, nu] * overlap[nu, q]
                        )
                        correction[p, q] += 0.25 * g2 * overlap[p, nu] * overlap[q, nu] * (
                            local_core[p, atom_c] + local_core[q, atom_c]
                        )
    return 0.5 * (correction + correction.T)


def _nelec_tuple(nelec, spin):
    if isinstance(nelec, tuple):
        return tuple(int(x) for x in nelec)
    nalpha = (int(nelec) + int(spin)) // 2
    nbeta = int(nelec) - nalpha
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta != int(nelec):
        raise ValueError("Invalid electron count/spin combination.")
    return nalpha, nbeta


def _density_from_coeff(coeff, nocc):
    c_occ = coeff[:, :nocc]
    return 2.0 * c_occ @ c_occ.T


def _build_fock(hcore, eri, dm):
    j = np.einsum("pqrs,rs->pq", eri, dm, optimize=True)
    k = np.einsum("prqs,rs->pq", eri, dm, optimize=True)
    return hcore + j - 0.5 * k


def _electronic_energy(hcore, fock, dm):
    return 0.5 * float(np.einsum("pq,pq->", dm, hcore + fock, optimize=True))


def _safe_eigh(matrix):
    e, c = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(e)
    return e[order], c[:, order]


class OM2Reference:
    """Closed-shell semiempirical reference generated by :class:`OM2`."""

    def __init__(self, om2, hamiltonian_data):
        self.om2 = om2
        self.mol = om2.mol
        self.hamiltonian_data = hamiltonian_data
        self.hcore = np.asarray(hamiltonian_data.hcore, dtype=float)
        self.eri = np.asarray(hamiltonian_data.eri, dtype=float)
        self.enuc = float(hamiltonian_data.enuc)
        self.nelec = hamiltonian_data.nelec
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.dm = None
        self.e_elec = None
        self.e_tot = None
        self.converged = False
        self.niter = 0

    def get_hcore(self):
        return self.hcore

    def energy_nuc(self):
        return self.enuc

    def get_eri_mo(self, notation="chem"):
        if self.mo_coeff is None:
            raise ValueError("Run the OM2 reference before requesting MO ERIs.")
        c = self.mo_coeff
        eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.eri, c, c, c, c, optimize=True)
        if notation not in {"chem", "chemist"}:
            raise ValueError("Only chemist notation is implemented for OM2 MO ERIs.")
        return eri_mo

    def get_hcore_mo(self):
        if self.mo_coeff is None:
            raise ValueError("Run the OM2 reference before requesting MO hcore.")
        return self.mo_coeff.T @ self.hcore @ self.mo_coeff

    def run(self, max_cycle=100, conv_tol=1e-10, damping=0.25):
        nelec = int(self.nelec if not isinstance(self.nelec, tuple) else sum(self.nelec))
        if nelec % 2:
            raise OM2ParameterError("Native OM2 currently supports closed-shell references.")
        nocc = nelec // 2
        if nocc > self.hcore.shape[0]:
            raise ValueError("More occupied orbitals than valence basis functions.")

        eps, coeff = _safe_eigh(self.hcore)
        dm = _density_from_coeff(coeff, nocc)
        last_e = None
        fock = self.hcore.copy()
        for cycle in range(1, max_cycle + 1):
            fock = _build_fock(self.hcore, self.eri, dm)
            eps, coeff = _safe_eigh(fock)
            new_dm = _density_from_coeff(coeff, nocc)
            if cycle > 1 and damping:
                new_dm = (1.0 - damping) * new_dm + damping * dm
            e_elec = _electronic_energy(self.hcore, fock, new_dm)
            e_tot = e_elec + self.enuc
            if last_e is not None and abs(e_tot - last_e) < conv_tol and np.linalg.norm(new_dm - dm) < conv_tol ** 0.5:
                self.converged = True
                dm = new_dm
                last_e = e_tot
                break
            dm = new_dm
            last_e = e_tot

        self.niter = cycle
        self.mo_coeff = coeff
        self.mo_energy = eps
        self.mo_occ = np.zeros(self.hcore.shape[0])
        self.mo_occ[:nocc] = 2
        self.dm = dm
        self.e_elec = _electronic_energy(self.hcore, fock, dm)
        self.e_tot = self.e_elec + self.enuc
        return self

    def build_mrci_hamiltonian(self, driver):
        binary = driver.build_configurations(self)
        h1_spatial = self.get_hcore_mo()
        eri_spatial = self.get_eri_mo()
        eri_aa = eri_spatial - eri_spatial.swapaxes(1, 3)
        h1 = np.asarray([h1_spatial, h1_spatial])
        h2 = np.stack(
            (
                np.stack((eri_aa, eri_spatial)),
                np.stack((eri_spatial, eri_aa)),
            )
        )
        sc1, sc2 = SlaterCondon(binary)
        h_ci = CI_H(binary, h1, h2, sc1, sc2)
        driver.determinants = binary
        driver.determinant_labels = _determinant_labels(binary)
        return h_ci


class OM2:
    """Orthogonalization Model 2 semiempirical reference."""

    method = "OM2"

    def __init__(
        self,
        mol=None,
        atom=None,
        charge=0,
        spin=0,
        unit="angstrom",
        parameters=None,
        orthogonalization_correction=False,
        verbose=0,
    ):
        if mol is None and atom is None:
            raise ValueError("Provide either mol or atom.")
        self.mol = mol
        self.atom = atom
        self.charge = int(charge)
        self.spin = int(spin)
        self.unit = unit
        self.parameters = DEFAULT_OM2_PARAMETERS if parameters is None else parameters
        self.orthogonalization_correction = bool(orthogonalization_correction)
        if isinstance(self.parameters, OM2ParameterSet):
            self.parameters = self.parameters.to_internal_units()
        self.verbose = verbose

        self.e_tot = None
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.hamiltonian_data = None
        self.reference = None

    def build(self):
        """Prepare the OM2 molecule/reference object."""
        if self.mol is None:
            self.mol = SemiempiricalMolecule(
                atom=self.atom,
                charge=self.charge,
                spin=self.spin,
                unit=self.unit,
            )
        return self

    def atom_symbols(self):
        self.build()
        if hasattr(self.mol, "atom_symbols"):
            return tuple(_normalize_symbol(sym) for sym in self.mol.atom_symbols())
        symbols, _ = _parse_atom_spec(getattr(self.mol, "atom", self.atom))
        return symbols

    def atom_coords(self):
        self.build()
        if hasattr(self.mol, "atom_coords"):
            coords = np.asarray(self.mol.atom_coords(), dtype=float)
            return _coords_to_bohr(coords, getattr(self.mol, "unit", self.unit))
        _, coords = _parse_atom_spec(getattr(self.mol, "atom", self.atom))
        return _coords_to_bohr(coords, self.unit)

    def valence_orbitals(self):
        """Return the minimal valence orbital list used by OM2-type models."""
        orbitals = []
        for atom_index, symbol in enumerate(self.atom_symbols()):
            for shell, axis in _valence_shells(symbol):
                orbitals.append(ValenceOrbital(atom_index, symbol, shell, axis))
        return tuple(orbitals)

    def build_hamiltonian_data(self):
        """Build semiempirical Hamiltonian data from OM2 parameters."""
        self.build()
        if hasattr(self.parameters, "build_hamiltonian_data"):
            data = self.parameters.build_hamiltonian_data(self)
            self.hamiltonian_data = data
            return data
        if not isinstance(self.parameters, OM2ParameterSet):
            raise OM2ParameterError(
                "Unsupported OM2 parameter provider. Expected OM2ParameterSet "
                "or an object with build_hamiltonian_data(om2)."
            )

        orbitals = self.valence_orbitals()
        symbols = self.atom_symbols()
        coords = self.atom_coords()
        atom_params = tuple(self.parameters.for_symbol(symbol) for symbol in symbols)
        params_by_orb = tuple(self.parameters.for_symbol(orb.symbol) for orb in orbitals)
        nbf = len(orbitals)
        hcore = np.zeros((nbf, nbf), dtype=float)
        beta = np.zeros((nbf, nbf), dtype=float)
        local_core = np.zeros((nbf, len(symbols)), dtype=float)
        eri = np.zeros((nbf, nbf, nbf, nbf), dtype=float)

        for p, orb in enumerate(orbitals):
            params = params_by_orb[p]
            hcore[p, p] = params.uss if orb.axis is None else params.upp
            if hcore[p, p] is None:
                raise OM2ParameterError(f"Missing one-electron parameter for {orb.label}.")
            for atom_index, params_core in enumerate(atom_params):
                if atom_index == orb.atom_index:
                    local_core[p, atom_index] = hcore[p, p]
                else:
                    r = np.linalg.norm(coords[orb.atom_index] - coords[atom_index])
                    local_core[p, atom_index] = _local_core_diag(params, params_core, orb, r)

        for p, orb_p in enumerate(orbitals):
            params_p = params_by_orb[p]
            rp = coords[orb_p.atom_index]
            for atom_index, symbol in enumerate(symbols):
                if atom_index == orb_p.atom_index:
                    continue
                params_core = atom_params[atom_index]
                r = np.linalg.norm(rp - coords[atom_index])
                hcore[p, p] += _core_attraction(params_p, params_core, orb_p, r)

        for p, orb_p in enumerate(orbitals):
            params_p = params_by_orb[p]
            for q, orb_q in enumerate(orbitals[:p]):
                if orb_p.atom_index == orb_q.atom_index:
                    continue
                params_q = params_by_orb[q]
                r_vec = coords[orb_q.atom_index] - coords[orb_p.atom_index]
                beta[p, q] = beta[q, p] = _pair_resonance(params_p, params_q, orb_p, orb_q, r_vec)
                hcore[p, q] = hcore[q, p] = beta[p, q]

        overlap = _build_overlap_matrix(orbitals, params_by_orb, coords)
        if self.orthogonalization_correction:
            hcore += _om2_orthogonalization_correction(beta, overlap, local_core, orbitals, atom_params)

        for p, orb_p in enumerate(orbitals):
            params_p = params_by_orb[p]
            for q, orb_q in enumerate(orbitals):
                params_q = params_by_orb[q]
                if orb_p.atom_index == orb_q.atom_index:
                    gamma = _one_center_coulomb(params_p, orb_p, orb_q)
                else:
                    r = np.linalg.norm(coords[orb_p.atom_index] - coords[orb_q.atom_index])
                    gamma = _gamma_for_orbitals(params_p, params_q, orb_p, orb_q, r)
                eri[p, p, q, q] = gamma

        enuc = 0.0
        for i in range(len(symbols)):
            zi = atom_params[i].core_charge
            for j in range(i):
                zj = atom_params[j].core_charge
                r = np.linalg.norm(coords[i] - coords[j])
                gi = atom_params[i].gamma_ss or 0.5
                gj = atom_params[j].gamma_ss or 0.5
                rho = 2.0 / max(gi + gj, 1e-12)
                enuc += zi * zj / np.sqrt(r * r + rho * rho)

        data = OM2HamiltonianData(
            hcore=hcore,
            eri=eri,
            enuc=float(enuc),
            nelec=self._electron_count(),
            orbital_labels=tuple(orb.label for orb in orbitals),
        )
        self.hamiltonian_data = data
        return data

    def _electron_count(self):
        total = 0
        for symbol in self.atom_symbols():
            params = self.parameters.for_symbol(symbol)
            if params.core_charge is None:
                raise OM2ParameterError(f"Missing core charge for element {symbol!r}.")
            total += int(round(params.core_charge))
        return total - self.charge

    def run(self, **kwargs):
        """Run an OM2 reference calculation."""
        self.build()
        if self.parameters is not None and hasattr(self.parameters, "build_reference"):
            ref = self.parameters.build_reference(self, **kwargs)
        else:
            ref = OM2Reference(self, self.build_hamiltonian_data()).run(**kwargs)
        self.e_tot = getattr(ref, "e_tot", None)
        self.mo_coeff = getattr(ref, "mo_coeff", None)
        self.mo_occ = getattr(ref, "mo_occ", None)
        self.mo_energy = getattr(ref, "mo_energy", None)
        self.hamiltonian_data = getattr(ref, "hamiltonian_data", self.hamiltonian_data)
        self.reference = ref
        return self

    def MRCI(self, **kwargs):
        """Construct an MRCI driver on this OM2 reference."""
        return MRCI(self, **kwargs)

    def MECI(self, **kwargs):
        """Construct a MOPAC-style active-space CI driver on this OM2 reference."""
        return MECI(self, **kwargs)

    def as_scanner(self, **kwargs):
        """Return a geometry scanner that runs OM2/MRCI."""
        return OM2MRCIScanner(self, **kwargs)


def _determinant_labels(binary):
    labels = []
    for det in binary:
        alpha = tuple(np.flatnonzero(det[0]).astype(int))
        beta = tuple(np.flatnonzero(det[1]).astype(int))
        labels.append((alpha, beta))
    return tuple(labels)


def _selected_determinants(nmo, nalpha, nbeta, singles=True, doubles=True):
    ref_alpha = tuple(range(nalpha))
    ref_beta = tuple(range(nbeta))
    configs = {(ref_alpha, ref_beta)}

    def excite(occ, max_rank):
        occ = tuple(occ)
        vir = tuple(i for i in range(nmo) if i not in occ)
        out = {occ}
        ranks = [1]
        if max_rank >= 2:
            ranks.append(2)
        for rank in ranks:
            for holes in combinations(occ, rank):
                for parts in combinations(vir, rank):
                    new_occ = sorted((set(occ) - set(holes)) | set(parts))
                    out.add(tuple(new_occ))
        return out

    max_rank = 2 if doubles else 1 if singles else 0
    if max_rank == 0:
        return configs
    alpha_strings = excite(ref_alpha, max_rank)
    beta_strings = excite(ref_beta, max_rank)
    for alpha in alpha_strings:
        rank_a = len(set(alpha) - set(ref_alpha))
        for beta in beta_strings:
            rank_b = len(set(beta) - set(ref_beta))
            rank = rank_a + rank_b
            if rank == 0 or (rank == 1 and singles) or (rank == 2 and doubles):
                configs.add((tuple(alpha), tuple(beta)))
    return configs


def _binary_from_configurations(configs, nmo):
    configs = sorted(configs)
    alpha = np.asarray([cfg[0] for cfg in configs], dtype=np.int8)
    beta = np.asarray([cfg[1] for cfg in configs], dtype=np.int8)
    return givenΛgetB(alpha, beta, nmo)


def _validate_active_orbitals(active_orbitals, nmo):
    if active_orbitals is None:
        return None
    active = tuple(int(i) for i in active_orbitals)
    if not active:
        raise ValueError("active_orbitals must not be empty.")
    if len(set(active)) != len(active):
        raise ValueError("active_orbitals contains duplicate indices.")
    if min(active) < 0 or max(active) >= nmo:
        raise ValueError("active_orbitals contains an out-of-range MO index.")
    return active


def _embed_active_binary(active_binary, nmo, active_orbitals, frozen_occ):
    binary = np.zeros((len(active_binary), 2, nmo), dtype=np.int8)
    for spin in range(2):
        binary[:, spin, :] = frozen_occ[spin]
        binary[:, spin, active_orbitals] = active_binary[:, spin, :]
    return binary


def _determinant_overlap_matrix(left_binary, right_binary, mo_overlap):
    """Return Slater determinant overlaps from a one-particle MO overlap."""
    left_binary = np.asarray(left_binary, dtype=np.int8)
    right_binary = np.asarray(right_binary, dtype=np.int8)
    mo_overlap = np.asarray(mo_overlap, dtype=float)
    out = np.zeros((len(left_binary), len(right_binary)), dtype=float)

    for i, left in enumerate(left_binary):
        left_alpha = np.flatnonzero(left[0])
        left_beta = np.flatnonzero(left[1])
        for j, right in enumerate(right_binary):
            right_alpha = np.flatnonzero(right[0])
            right_beta = np.flatnonzero(right[1])
            if len(left_alpha) != len(right_alpha) or len(left_beta) != len(right_beta):
                continue
            s_alpha = mo_overlap[np.ix_(left_alpha, right_alpha)]
            s_beta = mo_overlap[np.ix_(left_beta, right_beta)]
            out[i, j] = np.linalg.det(s_alpha) * np.linalg.det(s_beta)

    return out


class MRCI:
    """Multireference CI driver for a semiempirical reference."""

    method = "MRCI"

    def __init__(
        self,
        reference,
        nstates=3,
        nref=None,
        singles=True,
        doubles=True,
        selection_threshold=0.0,
        spin=None,
        full=False,
        active_orbitals=None,
        verbose=0,
    ):
        self.reference = reference
        self.nstates = int(nstates)
        self.nref = nref
        self.singles = bool(singles)
        self.doubles = bool(doubles)
        self.selection_threshold = float(selection_threshold)
        self.spin = spin
        self.full = bool(full)
        self.active_orbitals = None if active_orbitals is None else tuple(int(i) for i in active_orbitals)
        self.verbose = verbose
        self.e = None
        self.e_elec = None
        self.ci = None
        self.determinants = None
        self.active_determinants = None
        self.determinant_labels = None

    @property
    def e_tot(self):
        return self.e

    def _ensure_reference(self):
        ref = self.reference
        if isinstance(ref, OM2):
            if getattr(ref, "reference", None) is None:
                ref.run()
            return getattr(ref, "reference", ref)
        return ref

    def build_configuration_data(self, reference=None):
        ref = self._ensure_reference() if reference is None else reference
        mo_occ = np.asarray(ref.mo_occ, dtype=float)
        nmo = mo_occ.size
        active_orbitals = _validate_active_orbitals(self.active_orbitals, nmo)
        if active_orbitals is None:
            active_orbitals = tuple(range(nmo))
            frozen_occ = np.zeros((2, nmo), dtype=np.int8)
            active_occ = mo_occ
        else:
            frozen_occ = np.zeros((2, nmo), dtype=np.int8)
            inactive = [i for i in range(nmo) if i not in active_orbitals]
            for i in inactive:
                if abs(mo_occ[i] - 2.0) < 1.0e-8:
                    frozen_occ[:, i] = 1
                elif abs(mo_occ[i]) > 1.0e-8:
                    raise ValueError("Active-space MECI/MRCI only supports closed-shell frozen orbitals.")
            active_occ = mo_occ[list(active_orbitals)]

        nelec = int(round(float(np.sum(active_occ))))
        nalpha, nbeta = _nelec_tuple(nelec, self.spin or 0)
        active_nmo = len(active_orbitals)
        if self.full:
            alpha = np.asarray(list(combinations(np.arange(active_nmo, dtype=np.int8), nalpha)), dtype=np.int8)
            beta = np.asarray(list(combinations(np.arange(active_nmo, dtype=np.int8), nbeta)), dtype=np.int8)
            active_binary = givenΛgetB(
                np.repeat(alpha, len(beta), axis=0),
                np.tile(beta, (len(alpha), 1)),
                active_nmo,
            )
        else:
            configs = _selected_determinants(active_nmo, nalpha, nbeta, singles=self.singles, doubles=self.doubles)
            active_binary = _binary_from_configurations(configs, active_nmo)
        binary = _embed_active_binary(active_binary, nmo, active_orbitals, frozen_occ)
        self.determinants = binary
        self.active_determinants = active_binary
        self.determinant_labels = _determinant_labels(binary)
        return CIConfigurationData(
            active_binary=active_binary,
            binary=binary,
            active_orbitals=active_orbitals,
            frozen_occ=frozen_occ,
        )

    def build_configurations(self, reference=None):
        return self.build_configuration_data(reference).binary

    def _dense_hamiltonian(self):
        ref = self._ensure_reference()
        if hasattr(ref, "build_mrci_hamiltonian"):
            h = ref.build_mrci_hamiltonian(self)
        elif hasattr(ref, "h_ci"):
            h = ref.h_ci
        else:
            raise SemiempiricalMethodNotAvailable(
                "MRCI needs a reference that can build a CI Hamiltonian. "
                "Implement reference.build_mrci_hamiltonian(driver) or provide reference.h_ci."
            )
        h = np.asarray(h, dtype=float)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("MRCI Hamiltonian must be a square matrix.")
        return 0.5 * (h + h.T), ref

    def run(self, nstates=None):
        if nstates is not None:
            self.nstates = int(nstates)
        h, ref = self._dense_hamiltonian()
        nroots = min(self.nstates, h.shape[0])
        if nroots == h.shape[0] or h.shape[0] <= max(4, nroots + 1):
            e, v = np.linalg.eigh(h)
            e = e[:nroots]
            v = v[:, :nroots]
        else:
            e, v = eigsh(h, k=nroots, which="SA")
            order = np.argsort(e)
            e = e[order]
            v = v[:, order]

        self.e_elec = np.asarray(e)
        enuc = float(ref.energy_nuc()) if hasattr(ref, "energy_nuc") else 0.0
        self.e = self.e_elec + enuc
        self.ci = np.asarray(v)
        self.nstates = int(len(self.e))
        return self

    def wavefunction_overlap(self, other):
        """Return CI-state overlap between two MRCI calculations.

        If the references expose ``get_mo_cross_overlap()``, determinant
        overlaps are built from the transported MO overlap.  This is the real
        semiempirical orbital-overlap path used by AM1/MECI.  References that
        do not provide cross-overlaps fall back to the legacy CI-vector
        pseudo-overlap.
        """
        if self.ci is None or other.ci is None:
            raise ValueError("Run both MRCI objects before computing overlaps.")
        if self.ci.shape[0] != other.ci.shape[0]:
            raise ValueError("MRCI determinant spaces have different dimensions.")
        ref = self._ensure_reference()
        other_ref = other._ensure_reference()
        if not hasattr(ref, "get_mo_cross_overlap"):
            return self.ci.T @ other.ci

        mo_overlap = ref.get_mo_cross_overlap(other_ref)
        det_overlap = _determinant_overlap_matrix(
            self.determinants,
            other.determinants,
            mo_overlap,
        )
        return self.ci.T @ det_overlap @ other.ci


class MECI(MRCI):
    """MOPAC-style multi-electron CI: full CI in a chosen active MO space."""

    method = "MECI"

    def __init__(self, reference, nstates=3, ncas=None, active_orbitals=None, **kwargs):
        if active_orbitals is not None and ncas is not None:
            raise ValueError("Specify either active_orbitals or ncas, not both.")
        if active_orbitals is None:
            if ncas is None:
                ncas = 2
            active_orbitals = _frontier_active_orbitals(reference, ncas)
        kwargs.setdefault("full", True)
        super().__init__(
            reference,
            nstates=nstates,
            active_orbitals=active_orbitals,
            **kwargs,
        )


def _frontier_active_orbitals(reference, ncas):
    if ncas is None:
        return None
    ref = reference
    if isinstance(ref, OM2):
        if getattr(ref, "reference", None) is None:
            ref.run()
        ref = getattr(ref, "reference", ref)
    elif getattr(ref, "mo_occ", None) is None and hasattr(ref, "run"):
        ref.run()
    mo_occ = np.asarray(ref.mo_occ, dtype=float)
    nmo = mo_occ.size
    nactive = int(ncas)
    if nactive <= 0 or nactive > nmo:
        raise ValueError("ncas must be between 1 and the number of MOs.")
    nocc = int(np.count_nonzero(mo_occ > 1.0e-8))
    start = max(0, nocc - (nactive + 1) // 2)
    start = min(start, nmo - nactive)
    return tuple(range(start, start + nactive))


class OM2MRCIScanner:
    """Geometry scanner for OM2/MRCI PES workflows."""

    def __init__(self, template, **mrci_kwargs):
        self.template = template
        self.mrci_kwargs = dict(mrci_kwargs)

    def __call__(self, atom=None, mol=None, **om2_kwargs):
        if isinstance(self.template, OM2):
            kwargs = dict(
                charge=self.template.charge,
                spin=self.template.spin,
                unit=self.template.unit,
                parameters=self.template.parameters,
                orthogonalization_correction=self.template.orthogonalization_correction,
                verbose=self.template.verbose,
            )
            kwargs.update(om2_kwargs)
            if atom is None and mol is None:
                atom = self.template.atom
                mol = self.template.mol if atom is None else None
            ref = OM2(mol=mol, atom=atom if mol is None else None, **kwargs).run()
        else:
            ref = self.template
        return MRCI(ref, **self.mrci_kwargs).run()
