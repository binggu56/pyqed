"""
Native strongly contracted NEVPT2 building blocks.

The public ``run`` method evaluates the eight standard strongly contracted
perturber classes.  The singly external ``Si`` and ``Sr`` classes use native
contracted 4-RDM intermediates when the optional C++ helper extension is
available, with an exact in-core 4-RDM fallback for validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import importlib

import numpy as np

from pyqed.qchem.mcscf.casci import (
    _annihilate_bit,
    _create_bit,
    _determinant_bits_from_binary,
    _get_veff_for_dm,
    _is_uhf_reference,
    _resolve_use_cholesky,
    transform_spatial_eri_to_mo,
)

_CASSCF_CPP_UNINITIALIZED = object()
_casscf_cpp = _CASSCF_CPP_UNINITIALIZED


def _cpp_attr(*names):
    global _casscf_cpp
    if _casscf_cpp is _CASSCF_CPP_UNINITIALIZED:
        try:
            _casscf_cpp = importlib.import_module("pyqed.qchem._casscf_cpp")
        except Exception:  # pragma: no cover - optional accelerator
            _casscf_cpp = None
    if _casscf_cpp is None:
        return None
    for name in names:
        attr = getattr(_casscf_cpp, name, None)
        if attr is not None:
            return attr
    return None


SC_PERTURBER_CLASSES = (
    "Sijrs",
    "Sijr",
    "Srsi",
    "Sij",
    "Srs",
    "Sir",
    "Si",
    "Sr",
)
SUPPORTED_SC_CLASSES = SC_PERTURBER_CLASSES
NUMERICAL_ZERO = 1.0e-14
DEFAULT_MAX_4RDM_NCAS = 6

SC_NEVPT2_THEORY = """
Strongly contracted NEVPT2 uses a CAS eigenstate |Psi0> as the zeroth-order
state and Dyall's Hamiltonian as H0.  The Dyall Hamiltonian keeps the exact
active-space Hamiltonian in the CAS block and uses diagonal orbital-energy
operators for inactive-core and external-virtual orbitals.  This choice makes
the method size-consistent and avoids the intruder-state level-shift machinery
usually needed by CASPT2.

The first-order interacting space is split into eight internally contracted
classes, conventionally labeled Sijrs, Sijr, Srsi, Sij, Srs, Sir, Si, and Sr.
In the strongly contracted variant each class is represented by one contracted
perturber vector

    |Phi_mu> = P_mu V |Psi0>

where P_mu projects onto the excitation class and V = H - H0.  The second-order
energy is then accumulated class by class through the contracted norm and
Dyall denominator,

    N_mu = <Phi_mu|Phi_mu>
    D_mu = <Phi_mu|H0 - E0|Phi_mu> / N_mu
    E2_mu = -N_mu / D_mu

or equivalently with the opposite sign absorbed into the denominator.  The
implemented code stores the numerator, norm, and denominator contractions for
each class.  Most SC classes require active 3-RDM-like intermediates; the
singly external Si/Sr classes additionally require 4-RDM contractions.  Sijrs
is the pure inactive-core to external-virtual pair class and only needs
core/virtual two-electron integrals and orbital-energy denominators.
""".strip()


@dataclass(frozen=True)
class NEVPT2Component:
    """A single strongly contracted NEVPT2 perturber-space contribution."""

    label: str
    norm: float
    energy: float


class NEVPT2:
    """
    Strongly contracted NEVPT2 driver for PyQED CASCI/CASSCF objects.

    Parameters
    ----------
    mc
        A converged PyQED CASCI/CASSCF-like object with restricted orbitals.
    root
        CASCI root index.
    classes
        Optional subset of SC-NEVPT2 perturber classes to evaluate.  The first
        native implementation supports all eight strongly contracted classes.
    allow_incomplete
        If ``False`` and unsupported classes are requested, ``run`` raises
        instead of returning a partial NEVPT2 correction.
    use_cholesky
        Forwarded to the MO-integral transformer when RI/Cholesky factors are
        available on the reference.
    max_4rdm_ncas
        Maximum active-orbital count for the exact in-core 4-RDM fallback used
        by the singly external ``Si`` and ``Sr`` classes when the C++ contracted
        helper is unavailable.
    """

    all_classes = SC_PERTURBER_CLASSES
    supported_classes = SUPPORTED_SC_CLASSES

    def __init__(
        self,
        mc,
        root: int = 0,
        classes=None,
        allow_incomplete: bool = False,
        use_cholesky=None,
        max_4rdm_ncas: int = DEFAULT_MAX_4RDM_NCAS,
        verbose: int = 0,
    ):
        self.mc = mc
        self.root = int(root)
        self.classes = tuple(classes) if classes is not None else self.all_classes
        self.allow_incomplete = bool(allow_incomplete)
        self.use_cholesky = use_cholesky
        self.max_4rdm_ncas = int(max_4rdm_ncas)
        self.verbose = verbose
        self.components: dict[str, NEVPT2Component] = {}
        self.e_corr: float | None = None
        self._integral_blocks_cache = None
        self._active_rdms123_cache = None
        self._active_rdm4_cache = None
        self._contracted_a16_terms_cache = None

    @staticmethod
    def theory():
        """Return a compact theory note for native strongly contracted NEVPT2."""
        return SC_NEVPT2_THEORY

    def run(self):
        """Evaluate the requested native SC-NEVPT2 perturber classes."""
        self._validate_reference()
        requested = tuple(str(label) for label in self.classes)
        unknown = sorted(set(requested) - set(self.all_classes))
        if unknown:
            raise ValueError(f"Unknown SC-NEVPT2 perturber classes: {unknown}")

        unsupported = tuple(label for label in requested if label not in self.supported_classes)
        if unsupported and not self.allow_incomplete:
            raise NotImplementedError(
                "Native full SC-NEVPT2 is not complete yet. "
                "Unsupported classes {} require active 3-RDM/intermediate support. "
                "Request classes=('Sijrs',) or set allow_incomplete=True for the "
                "currently implemented subset.".format(unsupported)
            )

        active_labels = tuple(label for label in requested if label in self.supported_classes)
        self.components = {}
        for label in active_labels:
            if label == "Sijrs":
                component = self._sijrs_component()
            elif label == "Sijr":
                component = self._sijr_component()
            elif label == "Srsi":
                component = self._srsi_component()
            elif label == "Sij":
                component = self._sij_component()
            elif label == "Srs":
                component = self._srs_component()
            elif label == "Sir":
                component = self._sir_component()
            elif label == "Si":
                component = self._si_component()
            elif label == "Sr":
                component = self._sr_component()
            else:  # pragma: no cover - guarded by supported_classes
                raise NotImplementedError(label)
            self.components[label] = component

        self.e_corr = float(sum(component.energy for component in self.components.values()))
        return self.e_corr

    def kernel(self):
        """Compatibility alias for :meth:`run`; new code should call ``run``."""
        return self.run()

    def _validate_reference(self):
        mc = self.mc
        if getattr(mc, "ci", None) is None:
            raise ValueError("Run CASCI/CASSCF before NEVPT2.")
        mo_coeff = getattr(mc, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("NEVPT2 requires molecular orbitals on the CASCI/CASSCF object.")
        if _is_uhf_reference(mo_coeff):
            raise NotImplementedError("Native SC-NEVPT2 currently supports restricted references only.")
        mo_energy = self._mo_energy()
        if mo_energy.ndim != 1:
            raise ValueError("NEVPT2 requires one-dimensional restricted MO energies.")
        if mo_energy.size < self._nocc:
            raise ValueError("MO energy array is shorter than the occupied orbital space.")

    @property
    def _ncore(self):
        return int(getattr(self.mc, "ncore"))

    @property
    def _ncas(self):
        return int(getattr(self.mc, "ncas"))

    @property
    def _nocc(self):
        return self._ncore + self._ncas

    def _mo_energy(self):
        mc = self.mc
        mo_energy = getattr(mc, "mo_energy", None)
        if mo_energy is None:
            mo_energy = getattr(getattr(mc, "mf", None), "mo_energy", None)
        if mo_energy is None:
            raise ValueError("NEVPT2 requires orbital energies; canonicalize or run RHF/CASSCF first.")
        return np.asarray(mo_energy, dtype=float)

    def _mo_coeff(self):
        return np.asarray(self.mc.mo_coeff, dtype=float)

    def _orbital_spaces(self):
        mo_coeff = self._mo_coeff()
        ncore = self._ncore
        nocc = self._nocc
        return mo_coeff[:, :ncore], mo_coeff[:, ncore:nocc], mo_coeff[:, nocc:]

    def _ci_vector(self):
        ci = getattr(self.mc, "ci", None)
        if ci is None:
            raise ValueError("Run CASCI/CASSCF before NEVPT2.")
        return np.asarray(ci[self.root])

    def _use_cholesky(self):
        if self.use_cholesky is not None:
            return self.use_cholesky
        return _resolve_use_cholesky(getattr(self.mc, "mf", None), None)

    def _eri_mo(self, mo_left, mo_right, mo_left_2, mo_right_2):
        return transform_spatial_eri_to_mo(
            self.mc.mf,
            mo_left,
            mo_right,
            mo_left_2,
            mo_right_2,
            use_cholesky=self._use_cholesky(),
        )

    def _h1eff_full_mo(self):
        mf = self.mc.mf
        mo_coeff = self._mo_coeff()
        hcore = np.asarray(mf.get_hcore(), dtype=float)
        mo_core, _mo_cas, _mo_virt = self._orbital_spaces()
        if mo_core.size == 0:
            corevhf = np.zeros_like(hcore)
        else:
            core_dm = np.dot(mo_core, mo_core.conj().T) * 2.0
            corevhf = np.asarray(_get_veff_for_dm(mf, core_dm), dtype=float)
        return reduce(np.dot, (mo_coeff.conj().T, hcore + corevhf, mo_coeff))

    def _integral_blocks(self):
        cached = self._integral_blocks_cache
        if cached is not None:
            return cached

        mo_coeff = self._mo_coeff()
        mo_core, mo_cas, mo_virt = self._orbital_spaces()
        ncore = self._ncore
        nocc = self._nocc
        nmo = mo_coeff.shape[1]

        ppaa = self._eri_mo(mo_coeff, mo_coeff, mo_cas, mo_cas)
        papa = self._eri_mo(mo_coeff, mo_cas, mo_coeff, mo_cas)
        pacv = self._eri_mo(mo_coeff, mo_cas, mo_core, mo_virt)
        cvcv = self._eri_mo(mo_core, mo_virt, mo_core, mo_virt)

        if ppaa.shape != (nmo, nmo, self._ncas, self._ncas):
            raise ValueError(f"Unexpected ppaa integral block shape {ppaa.shape}.")
        if papa.shape != (nmo, self._ncas, nmo, self._ncas):
            raise ValueError(f"Unexpected papa integral block shape {papa.shape}.")
        if pacv.shape != (nmo, self._ncas, ncore, mo_virt.shape[1]):
            raise ValueError(f"Unexpected pacv integral block shape {pacv.shape}.")

        blocks = {
            "h1eff": np.asarray(self._h1eff_full_mo(), dtype=float),
            "ppaa": np.asarray(ppaa, dtype=float),
            "papa": np.asarray(papa, dtype=float),
            "pacv": np.asarray(pacv, dtype=float),
            "cvcv": np.asarray(cvcv, dtype=float),
        }
        blocks["h1e"] = np.asarray(blocks["h1eff"][ncore:nocc, ncore:nocc], dtype=float)
        blocks["h2e"] = np.asarray(ppaa[ncore:nocc, ncore:nocc].transpose(0, 2, 1, 3), dtype=float)
        self._integral_blocks_cache = blocks
        return blocks

    def _active_rdms123(self):
        cached = self._active_rdms123_cache
        if cached is None:
            cached = _spin_free_rdms123(self._ci_vector(), self.mc.binary)
            self._active_rdms123_cache = cached
        return cached

    def _active_rdm4(self):
        cached = self._active_rdm4_cache
        if cached is None:
            ncas = self._ncas
            if self.max_4rdm_ncas >= 0 and ncas > self.max_4rdm_ncas:
                raise NotImplementedError(
                    "Native Si/Sr SC-NEVPT2 exact fallback currently uses an "
                    f"in-core 4-RDM and is capped at ncas <= {self.max_4rdm_ncas}; "
                    "increase max_4rdm_ncas for small validation jobs or use "
                    "classes that do not require the 4-RDM."
                )
            cached = _spin_free_rdm4(self._ci_vector(), self.mc.binary)
            self._active_rdm4_cache = cached
        return cached

    def _has_contracted_a16_a22(self):
        separate = (
            _cpp_attr("nevpt_a16_4rdm_terms") is not None
            and _cpp_attr("nevpt_a22_4rdm_terms") is not None
        )
        return separate or self._combine_contracted_a16_a22()

    def _has_a22_4rdm_energy_contract(self):
        return _cpp_attr("nevpt_a22_4rdm_energy") is not None

    def _combine_contracted_a16_a22(self):
        if _cpp_attr("nevpt_a16_a22_4rdm_terms") is None:
            return False
        labels = tuple(self.classes)
        return (
            "Si" in labels
            and "Sr" in labels
            and labels.index("Si") < labels.index("Sr")
            and self._ncore > 0
            and self._mo_energy().size > self._nocc
        )

    def _contracted_a22_and_cache_a16(self):
        a16_a22_terms = _cpp_attr("nevpt_a16_a22_4rdm_terms")
        if a16_a22_terms is None:
            raise RuntimeError("The C++ NEVPT2 contracted-term helper is not available.")

        h2e = np.ascontiguousarray(self._integral_blocks()["h2e"], dtype=np.float64)
        ci = np.ascontiguousarray(self._ci_vector(), dtype=np.float64)
        binary = np.ascontiguousarray(self.mc.binary, dtype=np.int8)
        a16_terms, a22_terms = a16_a22_terms(h2e, ci, binary)
        self._contracted_a16_terms_cache = a16_terms
        return a22_terms

    def _dms(self, include_rdm4=False):
        dm1, dm2, dm3 = self._active_rdms123()
        dms = {"1": dm1, "2": dm2, "3": dm3}
        if include_rdm4:
            dms["4"] = self._active_rdm4()
        return dms

    def _sijrs_component(self):
        """
        Core-core to external-virtual-virtual strongly contracted class.

        In spin-free restricted notation this is the same numerator structure as
        the closed-shell MP2 core/virtual block, restricted to inactive core and
        external virtual orbitals:

        ``sum_ijab (ia|jb) [2(ia|jb) - (ib|ja)] / (ei + ej - ea - eb)``.
        """
        ncore = self._ncore
        nocc = self._nocc
        mo_energy = self._mo_energy()
        nvirt = mo_energy.size - nocc
        if ncore == 0 or nvirt == 0:
            return NEVPT2Component("Sijrs", 0.0, 0.0)

        mo_core, _mo_cas, mo_virt = self._orbital_spaces()
        eri = np.asarray(
            self._eri_mo(mo_core, mo_virt, mo_core, mo_virt),
            dtype=float,
        )
        if eri.shape != (ncore, nvirt, ncore, nvirt):
            raise ValueError(
                "Unexpected (core, virtual, core, virtual) ERI shape "
                f"{eri.shape}; expected {(ncore, nvirt, ncore, nvirt)}."
            )

        theta = 2.0 * eri - eri.swapaxes(1, 3)
        numerator = eri * theta
        eps_core = mo_energy[:ncore]
        eps_virt = mo_energy[nocc:]
        denom = (
            eps_core[:, None, None, None]
            + eps_core[None, None, :, None]
            - eps_virt[None, :, None, None]
            - eps_virt[None, None, None, :]
        )
        if np.any(np.abs(denom) < 1.0e-12):
            raise ZeroDivisionError("Encountered near-zero Sijrs Dyall denominator.")
        norm = float(np.einsum("iajb,iajb->", eri, theta, optimize=True))
        energy = float(np.einsum("iajb,iajb->", numerator, 1.0 / denom, optimize=True))
        return NEVPT2Component("Sijrs", norm, energy)

    def _sijr_component(self):
        blocks = self._integral_blocks()
        dms = self._dms()
        ncore = self._ncore
        nocc = self._nocc
        if ncore == 0 or self._mo_energy().size == nocc:
            return NEVPT2Component("Sijr", 0.0, 0.0)

        h2e_v = blocks["pacv"][:ncore].transpose(3, 1, 2, 0)
        norm, energy = _sijr(blocks["h1e"], blocks["h2e"], h2e_v, dms, self._mo_energy(), ncore, nocc)
        return NEVPT2Component("Sijr", float(norm), float(energy))

    def _srsi_component(self):
        blocks = self._integral_blocks()
        dms = self._dms()
        ncore = self._ncore
        nocc = self._nocc
        nvirt = self._mo_energy().size - nocc
        if ncore == 0 or nvirt == 0:
            return NEVPT2Component("Srsi", 0.0, 0.0)

        h2e_v = blocks["pacv"][nocc:].transpose(3, 0, 2, 1)
        norm, energy = _srsi(blocks["h1e"], blocks["h2e"], h2e_v, dms, self._mo_energy(), ncore, nocc)
        return NEVPT2Component("Srsi", float(norm), float(energy))

    def _sij_component(self):
        blocks = self._integral_blocks()
        dms = self._dms()
        ncore = self._ncore
        if ncore == 0:
            return NEVPT2Component("Sij", 0.0, 0.0)

        h2e_v = blocks["papa"][:ncore, :, :ncore].transpose(1, 3, 0, 2)
        norm, energy = _sij(blocks["h1e"], blocks["h2e"], h2e_v, dms, self._mo_energy(), ncore)
        return NEVPT2Component("Sij", float(norm), float(energy))

    def _srs_component(self):
        blocks = self._integral_blocks()
        dms = self._dms()
        nocc = self._nocc
        nvirt = self._mo_energy().size - nocc
        if nvirt == 0:
            return NEVPT2Component("Srs", 0.0, 0.0)

        h2e_v = blocks["papa"][nocc:, :, nocc:].transpose(0, 2, 1, 3)
        norm, energy = _srs(blocks["h1e"], blocks["h2e"], h2e_v, dms, self._mo_energy(), nocc)
        return NEVPT2Component("Srs", float(norm), float(energy))

    def _sir_component(self):
        blocks = self._integral_blocks()
        dms = self._dms()
        ncore = self._ncore
        nocc = self._nocc
        nvirt = self._mo_energy().size - nocc
        if ncore == 0 or nvirt == 0:
            return NEVPT2Component("Sir", 0.0, 0.0)

        h2e_v1 = blocks["ppaa"][nocc:, :ncore].transpose(0, 2, 1, 3)
        h2e_v2 = blocks["papa"][nocc:, :, :ncore].transpose(0, 3, 1, 2)
        h1e_v = blocks["h1eff"][nocc:, :ncore]
        norm, energy = _sir(
            blocks["h1e"],
            blocks["h2e"],
            h2e_v1,
            h2e_v2,
            h1e_v,
            dms,
            self._mo_energy(),
            ncore,
            nocc,
        )
        return NEVPT2Component("Sir", float(norm), float(energy))

    def _si_component(self):
        blocks = self._integral_blocks()
        use_direct_a22 = self._has_a22_4rdm_energy_contract()
        use_contracted = use_direct_a22 or self._has_contracted_a16_a22()
        ncore = self._ncore
        nocc = self._nocc
        if ncore == 0:
            return NEVPT2Component("Si", 0.0, 0.0)

        dms = self._dms(include_rdm4=not use_contracted)
        contracted_a22 = (
            None
            if use_direct_a22
            else self._contracted_a22_and_cache_a16() if self._combine_contracted_a16_a22() else None
        )
        h2e_v = blocks["ppaa"][self._ncore:nocc, :ncore].transpose(0, 2, 1, 3)
        h1e_v = blocks["h1eff"][self._ncore:nocc, :ncore]
        norm, energy = _si(
            blocks["h1e"],
            blocks["h2e"],
            h2e_v,
            h1e_v,
            dms,
            self._ci_vector(),
            self.mc.binary,
            self._ncas,
            self._mo_energy(),
            ncore,
            contracted_a22=contracted_a22,
        )
        return NEVPT2Component("Si", float(norm), float(energy))

    def _sr_component(self):
        blocks = self._integral_blocks()
        use_contracted = self._has_contracted_a16_a22()
        nocc = self._nocc
        nvirt = self._mo_energy().size - nocc
        if nvirt == 0:
            return NEVPT2Component("Sr", 0.0, 0.0)

        dms = self._dms(include_rdm4=not use_contracted)
        contracted_a16 = self._contracted_a16_terms_cache if self._combine_contracted_a16_a22() else None
        h2e_v = blocks["ppaa"][nocc:, self._ncore:nocc].transpose(0, 2, 1, 3)
        h1e_v = blocks["h1eff"][nocc:, self._ncore:nocc] - np.einsum("mbbn->mn", h2e_v, optimize=True)
        norm, energy = _sr(
            blocks["h1e"],
            blocks["h2e"],
            h2e_v,
            h1e_v,
            dms,
            self._ci_vector(),
            self.mc.binary,
            self._mo_energy(),
            nocc,
            contracted_a16=contracted_a16,
        )
        return NEVPT2Component("Sr", float(norm), float(energy))


SCNEVPT2 = NEVPT2


def _apply_spin_free_excitation(vec, det_bits, det_index, ncas, p, q):
    vec = np.asarray(vec)
    out = np.zeros_like(vec, dtype=np.result_type(vec, float))
    for ket, bits0 in enumerate(det_bits):
        coeff = vec[ket]
        if coeff == 0:
            continue
        for spin in range(2):
            offset = spin * ncas
            bits1, phase1 = _annihilate_bit(bits0, offset + q)
            if phase1 == 0:
                continue
            bits2, phase2 = _create_bit(bits1, offset + p)
            if phase2 == 0:
                continue
            bra = det_index.get(bits2)
            if bra is not None:
                out[bra] += phase1 * phase2 * coeff
    return out


def _spin_free_e_vectors(ci, binary):
    ci = np.asarray(ci)
    binary = np.asarray(binary, dtype=np.int8)
    ncas = binary.shape[2]
    det_bits = _determinant_bits_from_binary(binary)
    det_index = {bits: idx for idx, bits in enumerate(det_bits)}
    vectors = np.empty((ncas, ncas, ci.size), dtype=np.result_type(ci, float))
    for p in range(ncas):
        for q in range(ncas):
            vectors[p, q] = _apply_spin_free_excitation(ci, det_bits, det_index, ncas, p, q)
    return vectors, det_bits, det_index


def _spin_free_rdms123(ci, binary):
    """
    Return PySCF/SC-NEVPT ordered spin-free active 1-, 2-, and 3-RDMs.

    ``dm2[p,q,r,s] = <E_pq E_rs>`` and
    ``dm3[p,q,r,s,t,u] = <E_pq E_rs E_tu>``.
    """
    rdm_builder = _cpp_attr("nevpt_spin_free_rdms123")
    if rdm_builder is not None:
        return tuple(
            np.real_if_close(x)
            for x in rdm_builder(
                np.ascontiguousarray(ci, dtype=np.float64),
                np.ascontiguousarray(binary, dtype=np.int8),
            )
        )
    return _spin_free_rdms123_python(ci, binary)


def _spin_free_rdms123_python(ci, binary):
    """
    Return PySCF/SC-NEVPT ordered spin-free active 1-, 2-, and 3-RDMs.

    ``dm2[p,q,r,s] = <E_pq E_rs>`` and
    ``dm3[p,q,r,s,t,u] = <E_pq E_rs E_tu>``.
    """
    ci = np.asarray(ci)
    ncas = np.asarray(binary).shape[2]
    evecs, det_bits, det_index = _spin_free_e_vectors(ci, binary)
    dm1 = np.einsum("i,pqi->pq", ci.conj(), evecs, optimize=True)
    dm2 = np.einsum("qpi,rsi->pqrs", evecs.conj(), evecs, optimize=True)
    dm3 = np.zeros((ncas,) * 6, dtype=np.result_type(ci, float))
    for t in range(ncas):
        for u in range(ncas):
            etu = evecs[t, u]
            for r in range(ncas):
                for s in range(ncas):
                    erstu = _apply_spin_free_excitation(etu, det_bits, det_index, ncas, r, s)
                    dm3[:, :, r, s, t, u] = np.einsum(
                        "qpi,i->pq",
                        evecs.conj(),
                        erstu,
                        optimize=True,
                    )
    return tuple(np.real_if_close(x) for x in (dm1, dm2, dm3))


def _spin_free_rdm4(ci, binary):
    """
    Return the raw spin-free active 4-RDM used by the in-core Si/Sr path.

    ``dm4[p,q,r,s,t,u,v,w] = <E_pq E_rs E_tu E_vw>``.
    """
    ci = np.asarray(ci)
    ncas = np.asarray(binary).shape[2]
    evecs, det_bits, det_index = _spin_free_e_vectors(ci, binary)
    dm4 = np.zeros((ncas,) * 8, dtype=np.result_type(ci, float))
    for v in range(ncas):
        for w in range(ncas):
            evw = evecs[v, w]
            for t in range(ncas):
                for u in range(ncas):
                    etuvw = _apply_spin_free_excitation(evw, det_bits, det_index, ncas, t, u)
                    for r in range(ncas):
                        for s in range(ncas):
                            erstuvw = _apply_spin_free_excitation(
                                etuvw,
                                det_bits,
                                det_index,
                                ncas,
                                r,
                                s,
                            )
                            dm4[:, :, r, s, t, u, v, w] = np.einsum(
                                "qpi,i->pq",
                                evecs.conj(),
                                erstuvw,
                                optimize=True,
                            )
    return np.real_if_close(dm4)


def _norm_to_energy(norm, h, diff):
    norm = np.asarray(norm)
    h = np.asarray(h)
    diff = np.asarray(diff)
    idx = np.abs(norm) > NUMERICAL_ZERO
    energy = -(norm[idx] / (diff[idx] + h[idx] / norm[idx])).sum()
    return float(np.real_if_close(norm.sum())), float(np.real_if_close(energy))


def _make_hdm1(dm1):
    delta = np.eye(dm1.shape[0])
    return 2.0 * delta - dm1.transpose(1, 0)


def _make_hdm2(dm1, dm2):
    delta = np.eye(dm2.shape[0])
    rm2 = np.einsum("ikjl->ijkl", dm2, optimize=True) - np.einsum(
        "jk,il->ijkl",
        delta,
        dm1,
        optimize=True,
    )
    return (
        np.einsum("klij->ijkl", rm2, optimize=True)
        + np.einsum("il,kj->ijkl", delta, dm1, optimize=True)
        + np.einsum("jk,li->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("ik,lj->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("jl,ki->ijkl", delta, dm1, optimize=True)
        - 2.0 * np.einsum("il,jk->ijkl", delta, delta, optimize=True)
        + 4.0 * np.einsum("ik,jl->ijkl", delta, delta, optimize=True)
    )


def _make_hdm3(dm1, dm2, dm3, hdm2):
    delta = np.eye(dm3.shape[0])
    return (
        -np.einsum("pb,qrac->pqrabc", delta, hdm2, optimize=True)
        - np.einsum("br,pqac->pqrabc", delta, hdm2, optimize=True)
        + 2.0 * np.einsum("bq,prac->pqrabc", delta, hdm2, optimize=True)
        + 2.0 * np.einsum("ap,bqcr->pqrabc", delta, dm2, optimize=True)
        - 4.0 * np.einsum("ap,cr,bq->pqrabc", delta, delta, dm1, optimize=True)
        + 2.0 * np.einsum("cr,bqap->pqrabc", delta, dm2, optimize=True)
        - np.einsum("bqapcr->pqrabc", dm3, optimize=True)
        + 2.0 * np.einsum("ar,pc,bq->pqrabc", delta, delta, dm1, optimize=True)
        - np.einsum("ar,bqcp->pqrabc", delta, dm2, optimize=True)
    )


def _make_a3(h1e, h2e, dm1, dm2, hdm1):
    delta = np.eye(dm2.shape[0])
    return (
        np.einsum("ia,ip->pa", h1e, hdm1, optimize=True)
        + 2.0 * np.einsum("ijka,pj,ik->pa", h2e, delta, dm1, optimize=True)
        - np.einsum("ijka,jpik->pa", h2e, dm2, optimize=True)
    )


def _make_k27(h1e, h2e, dm1, dm2):
    return (
        -np.einsum("ai,pi->pa", h1e, dm1, optimize=True)
        - np.einsum("iajk,pkij->pa", h2e, dm2, optimize=True)
        + np.einsum("iaji,pj->pa", h2e, dm1, optimize=True)
    )


def _make_a9(h1e, h2e, hdm2, hdm3):
    a9 = np.einsum("ib,pqai->pqab", h1e, hdm2, optimize=True)
    a9 += 2.0 * np.einsum("ijib,pqaj->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijjb,pqai->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijkb,pkqaij->pqab", h2e, hdm3, optimize=True)
    a9 += np.einsum("ia,pqib->pqab", h1e, hdm2, optimize=True)
    a9 -= np.einsum("ijja,pqib->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijba,pqji->pqab", h2e, hdm2, optimize=True)
    a9 += 2.0 * np.einsum("ijia,pqjb->pqab", h2e, hdm2, optimize=True)
    a9 -= np.einsum("ijka,pqkjbi->pqab", h2e, hdm3, optimize=True)
    return a9


def _make_a7(h1e, h2e, dm1, dm2, dm3):
    delta = np.eye(dm2.shape[0])
    rm2 = np.einsum("iljk->ijkl", dm2, optimize=True) - np.einsum(
        "ik,jl->ijkl",
        dm1,
        delta,
        optimize=True,
    )
    rm3 = (
        np.einsum("injmkl->ijklmn", dm3, optimize=True)
        - np.einsum("jn,imkl->ijklmn", delta, dm2, optimize=True)
        - np.einsum("km,ijln->ijklmn", delta, rm2, optimize=True)
        - np.einsum("kn,ijml->ijklmn", delta, rm2, optimize=True)
    )
    a7 = (
        -np.einsum("bi,pqia->pqab", h1e, rm2, optimize=True)
        - np.einsum("ai,pqbi->pqab", h1e, rm2, optimize=True)
        - np.einsum("kbij,pqkija->pqab", h2e, rm3, optimize=True)
        - np.einsum("kaij,pqkibj->pqab", h2e, rm3, optimize=True)
        - np.einsum("baij,pqij->pqab", h2e, rm2, optimize=True)
    )
    return rm2, a7


def _make_a12(h1e, h2e, dm2, dm3):
    return (
        np.einsum("ia,qpib->pqab", h1e, dm2, optimize=True)
        - np.einsum("bi,qpai->pqab", h1e, dm2, optimize=True)
        + np.einsum("ijka,qpjbik->pqab", h2e, dm3, optimize=True)
        - np.einsum("kbij,qpajki->pqab", h2e, dm3, optimize=True)
        - np.einsum("bjka,qpjk->pqab", h2e, dm2, optimize=True)
        + np.einsum("jbij,qpai->pqab", h2e, dm2, optimize=True)
    )


def _make_a13(h1e, h2e, dm1, dm2, dm3):
    delta = np.eye(dm3.shape[0])
    a13 = -np.einsum("ia,qbip->pqab", h1e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("pa,qb->pqab", h1e, dm1, optimize=True)
    a13 += np.einsum("bi,qiap->pqab", h1e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("pa,bi,qi->pqab", delta, h1e, dm1, optimize=True)
    a13 -= np.einsum("ijka,qbjpik->pqab", h2e, dm3, optimize=True)
    a13 += np.einsum("kbij,qjapki->pqab", h2e, dm3, optimize=True)
    a13 += np.einsum("blma,qmlp->pqab", h2e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("kpma,qbkm->pqab", h2e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("bpma,qm->pqab", h2e, dm1, optimize=True)
    a13 -= np.einsum("lbkl,qkap->pqab", h2e, dm2, optimize=True)
    a13 -= 2.0 * np.einsum("ap,mbkl,qlmk->pqab", delta, h2e, dm2, optimize=True)
    a13 += 2.0 * np.einsum("ap,lbkl,qk->pqab", delta, h2e, dm1, optimize=True)
    return a13


def _make_a16(h1e, h2e, dms, ci=None, binary=None, contracted_terms=None):
    dm3 = dms["3"]
    a16 = -np.einsum("ib,rpqiac->pqrabc", h1e, dm3, optimize=True)
    a16 += np.einsum("ia,rpqbic->pqrabc", h1e, dm3, optimize=True)
    a16 -= np.einsum("ci,rpqbai->pqrabc", h1e, dm3, optimize=True)
    if contracted_terms is not None:
        ca1, ac, ca2 = contracted_terms
        a16 -= ca1
        a16 += ac
        a16 -= ca2
    elif (
        _cpp_attr("nevpt_a16_4rdm_terms") is not None
        and ci is not None
        and binary is not None
    ):
        a16_terms = _cpp_attr("nevpt_a16_4rdm_terms")
        ca1, ac, ca2 = a16_terms(
            np.ascontiguousarray(h2e, dtype=np.float64),
            np.ascontiguousarray(ci, dtype=np.float64),
            np.ascontiguousarray(binary, dtype=np.int8),
        )
        a16 -= ca1
        a16 += ac
        a16 -= ca2
    else:
        dm4 = dms["4"]
        a16 -= np.einsum("kbij,rpqjkiac->pqrabc", h2e, dm4, optimize=True)
        a16 += np.einsum("ijka,rpqbjcik->pqrabc", h2e, dm4, optimize=True)
        a16 -= np.einsum("kcij,rpqbajki->pqrabc", h2e, dm4, optimize=True)
    a16 += np.einsum("jbij,rpqiac->pqrabc", h2e, dm3, optimize=True)
    a16 -= np.einsum("cjka,rpqbjk->pqrabc", h2e, dm3, optimize=True)
    a16 += np.einsum("jcij,rpqbai->pqrabc", h2e, dm3, optimize=True)
    return a16


def _make_a17(h1e, h2e, dm2, dm3):
    h1e = h1e - np.einsum("mjjn->mn", h2e, optimize=True)
    return (
        -np.einsum("pi,cabi->abcp", h1e, dm2, optimize=True)
        - np.einsum("kpij,cabjki->abcp", h2e, dm3, optimize=True)
    )


def _make_a19(h1e, h2e, dm1, dm2):
    h1e = h1e - np.einsum("mjjn->mn", h2e, optimize=True)
    return (
        -np.einsum("pi,ai->ap", h1e, dm1, optimize=True)
        - np.einsum("kpij,ajki->ap", h2e, dm2, optimize=True)
    )


def _make_a22(h1e, h2e, dms, ci=None, binary=None, contracted_terms=None, include_4rdm=True):
    dm2 = dms["2"]
    dm3 = dms["3"]
    a22 = -np.einsum("pb,kipjac->ijkabc", h1e, dm3, optimize=True)
    a22 -= np.einsum("pa,kibjpc->ijkabc", h1e, dm3, optimize=True)
    a22 += np.einsum("cp,kibjap->ijkabc", h1e, dm3, optimize=True)
    a22 += np.einsum("cqra,kibjqr->ijkabc", h2e, dm3, optimize=True)
    a22 -= np.einsum("qcpq,kibjap->ijkabc", h2e, dm3, optimize=True)
    if include_4rdm:
        if contracted_terms is not None:
            ac1, ac2, ca = contracted_terms
            a22 -= ac1
            a22 -= ac2
            a22 += ca
        elif (
            _cpp_attr("nevpt_a22_4rdm_terms") is not None
            and ci is not None
            and binary is not None
        ):
            a22_terms = _cpp_attr("nevpt_a22_4rdm_terms")
            ac1, ac2, ca = a22_terms(
                np.ascontiguousarray(h2e, dtype=np.float64),
                np.ascontiguousarray(ci, dtype=np.float64),
                np.ascontiguousarray(binary, dtype=np.int8),
            )
            a22 -= ac1
            a22 -= ac2
            a22 += ca
        else:
            dm4 = dms["4"]
            a22 -= np.einsum("pqrb,kiqjprac->ijkabc", h2e, dm4, optimize=True)
            a22 -= np.einsum("pqra,kibjqcpr->ijkabc", h2e, dm4, optimize=True)
            a22 += np.einsum("rcpq,kibjaqrp->ijkabc", h2e, dm4, optimize=True)
    a22 += 2.0 * np.einsum("jb,kiac->ijkabc", h1e, dm2, optimize=True)
    a22 += 2.0 * np.einsum("pjrb,kiprac->ijkabc", h2e, dm3, optimize=True)
    fdm2 = np.einsum("pa,kipc->ikac", h1e, dm2, optimize=True)
    fdm2 -= np.einsum("cp,kiap->ikac", h1e, dm2, optimize=True)
    fdm2 -= np.einsum("cqra,kiqr->ikac", h2e, dm2, optimize=True)
    fdm2 += np.einsum("qcpq,kiap->ikac", h2e, dm2, optimize=True)
    fdm2 += np.einsum("pqra,kiqcpr->ikac", h2e, dm3, optimize=True)
    fdm2 -= np.einsum("rcpq,kiaqrp->ikac", h2e, dm3, optimize=True)
    for i in range(h1e.shape[0]):
        a22[:, i, :, :, i, :] += fdm2 * 2.0
    return a22


def _make_a23(h1e, h2e, dm1, dm2, dm3):
    return (
        -np.einsum("ip,caib->abcp", h1e, dm2, optimize=True)
        - np.einsum("pijk,cajbik->abcp", h2e, dm3, optimize=True)
        + 2.0 * np.einsum("bp,ca->abcp", h1e, dm1, optimize=True)
        + 2.0 * np.einsum("pibk,caik->abcp", h2e, dm2, optimize=True)
    )


def _make_a25(h1e, h2e, dm1, dm2):
    return (
        -np.einsum("pi,ai->ap", h1e, dm1, optimize=True)
        - np.einsum("pijk,jaik->ap", h2e, dm2, optimize=True)
        + 2.0 * np.einsum("ap->pa", h1e, optimize=True)
        + 2.0 * np.einsum("piaj,ij->ap", h2e, dm1, optimize=True)
    )


def _sijr(h1e, h2e, h2e_v, dms, mo_energy, ncore, nocc):
    dm1 = dms["1"]
    dm2 = dms["2"]
    hdm1 = _make_hdm1(dm1)
    a3 = _make_a3(h1e, h2e, dm1, dm2, hdm1)
    diag = np.diag_indices(ncore)
    triu = np.triu_indices(ncore)
    norm = (
        2.0 * np.einsum("rpji,raji,pa->rji", h2e_v, h2e_v, hdm1, optimize=True)
        - np.einsum("rpji,raij,pa->rji", h2e_v, h2e_v, hdm1, optimize=True)
    )
    norm += norm.transpose(0, 2, 1)
    norm[:, diag[0], diag[1]] *= 0.5
    h = (
        2.0 * np.einsum("rpji,raji,pa->rji", h2e_v, h2e_v, a3, optimize=True)
        - np.einsum("rpji,raij,pa->rji", h2e_v, h2e_v, a3, optimize=True)
    )
    h += h.transpose(0, 2, 1)
    h[:, diag[0], diag[1]] *= 0.5
    diff = mo_energy[nocc:, None, None] - mo_energy[None, :ncore, None] - mo_energy[None, None, :ncore]
    return _norm_to_energy(norm[:, triu[0], triu[1]], h[:, triu[0], triu[1]], diff[:, triu[0], triu[1]])


def _srsi(h1e, h2e, h2e_v, dms, mo_energy, ncore, nocc):
    dm1 = dms["1"]
    dm2 = dms["2"]
    k27 = _make_k27(h1e, h2e, dm1, dm2)
    nvirt = h2e_v.shape[0]
    diag = np.diag_indices(nvirt)
    triu = np.triu_indices(nvirt)
    norm = (
        2.0 * np.einsum("rsip,rsia,pa->rsi", h2e_v, h2e_v, dm1, optimize=True)
        - np.einsum("rsip,sria,pa->rsi", h2e_v, h2e_v, dm1, optimize=True)
    )
    norm += norm.transpose(1, 0, 2)
    norm[diag] *= 0.5
    h = (
        2.0 * np.einsum("rsip,rsia,pa->rsi", h2e_v, h2e_v, k27, optimize=True)
        - np.einsum("rsip,sria,pa->rsi", h2e_v, h2e_v, k27, optimize=True)
    )
    h += h.transpose(1, 0, 2)
    h[diag] *= 0.5
    diff = mo_energy[nocc:, None, None] + mo_energy[None, nocc:, None] - mo_energy[None, None, :ncore]
    return _norm_to_energy(norm[triu], h[triu], diff[triu])


def _sij(h1e, h2e, h2e_v, dms, mo_energy, ncore):
    dm1 = dms["1"]
    dm2 = dms["2"]
    dm3 = dms["3"]
    hdm2 = _make_hdm2(dm1, dm2)
    hdm3 = _make_hdm3(dm1, dm2, dm3, hdm2)
    a9 = _make_a9(h1e, h2e, hdm2, hdm3)
    norm = 0.5 * np.einsum("qpij,baij,pqab->ij", h2e_v, h2e_v, hdm2, optimize=True)
    h = 0.5 * np.einsum("qpij,baij,pqab->ij", h2e_v, h2e_v, a9, optimize=True)
    diff = mo_energy[:ncore, None] + mo_energy[None, :ncore]
    return _norm_to_energy(norm, h, -diff)


def _srs(h1e, h2e, h2e_v, dms, mo_energy, nocc):
    dm1 = dms["1"]
    dm2 = dms["2"]
    dm3 = dms["3"]
    rm2, a7 = _make_a7(h1e, h2e, dm1, dm2, dm3)
    norm = 0.5 * np.einsum("rsqp,rsba,pqba->rs", h2e_v, h2e_v, rm2, optimize=True)
    h = 0.5 * np.einsum("rsqp,rsba,pqab->rs", h2e_v, h2e_v, a7, optimize=True)
    diff = mo_energy[nocc:, None] + mo_energy[None, nocc:]
    return _norm_to_energy(norm, h, diff)


def _sir(h1e, h2e, h2e_v1, h2e_v2, h1e_v, dms, mo_energy, ncore, nocc):
    dm1 = dms["1"]
    dm2 = dms["2"]
    dm3 = dms["3"]
    norm = (
        2.0 * np.einsum("rpiq,raib,qpab->ir", h2e_v1, h2e_v1, dm2, optimize=True)
        - np.einsum("rpiq,rabi,qpab->ir", h2e_v1, h2e_v2, dm2, optimize=True)
        - np.einsum("rpqi,raib,qpab->ir", h2e_v2, h2e_v1, dm2, optimize=True)
        + 2.0 * np.einsum("raqi,rabi,qb->ir", h2e_v2, h2e_v2, dm1, optimize=True)
        - np.einsum("rpqi,rabi,qbap->ir", h2e_v2, h2e_v2, dm2, optimize=True)
        + np.einsum("rpqi,raai,qp->ir", h2e_v2, h2e_v2, dm1, optimize=True)
        + 4.0 * np.einsum("rpiq,ri,qp->ir", h2e_v1, h1e_v, dm1, optimize=True)
        - 2.0 * np.einsum("rpqi,ri,qp->ir", h2e_v2, h1e_v, dm1, optimize=True)
        + 2.0 * np.einsum("ri,ri->ir", h1e_v, h1e_v, optimize=True)
    )
    a12 = _make_a12(h1e, h2e, dm2, dm3)
    a13 = _make_a13(h1e, h2e, dm1, dm2, dm3)
    h = (
        2.0 * np.einsum("rpiq,raib,pqab->ir", h2e_v1, h2e_v1, a12, optimize=True)
        - np.einsum("rpiq,rabi,pqab->ir", h2e_v1, h2e_v2, a12, optimize=True)
        - np.einsum("rpqi,raib,pqab->ir", h2e_v2, h2e_v1, a12, optimize=True)
        + np.einsum("rpqi,rabi,pqab->ir", h2e_v2, h2e_v2, a13, optimize=True)
    )
    diff = mo_energy[:ncore, None] - mo_energy[None, nocc:]
    return _norm_to_energy(norm, h, -diff)


def _si(
    h1e,
    h2e,
    h2e_v,
    h1e_v,
    dms,
    ci,
    binary,
    ncas,
    mo_energy,
    ncore,
    contracted_a22=None,
):
    dm1 = dms["1"]
    dm2 = dms["2"]
    dm3 = dms["3"]
    a22_4rdm_energy_contract = _cpp_attr("nevpt_a22_4rdm_energy")
    if a22_4rdm_energy_contract is not None and ci is not None and binary is not None:
        a22 = _make_a22(h1e, h2e, dms, include_4rdm=False)
        a22_energy = np.einsum("qpir,pqrabc,baic->i", h2e_v, a22, h2e_v, optimize=True)
        a22_energy += a22_4rdm_energy_contract(
            np.ascontiguousarray(h2e, dtype=np.float64),
            np.ascontiguousarray(h2e_v, dtype=np.float64),
            np.ascontiguousarray(ci, dtype=np.float64),
            np.ascontiguousarray(binary, dtype=np.int8),
        )
    else:
        a22 = _make_a22(h1e, h2e, dms, ci=ci, binary=binary, contracted_terms=contracted_a22)
        a22_energy = np.einsum("qpir,pqrabc,baic->i", h2e_v, a22, h2e_v, optimize=True)
    a23 = _make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = _make_a25(h1e, h2e, dm1, dm2)
    delta = np.eye(ncas)
    dm3_h = 2.0 * np.einsum("abef,cd->abcdef", dm2, delta, optimize=True) - dm3.transpose(0, 1, 3, 2, 4, 5)
    dm2_h = 2.0 * np.einsum("ab,cd->abcd", dm1, delta, optimize=True) - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2.0 * delta - dm1.transpose(1, 0)
    energy = (
        a22_energy
        + 2.0 * np.einsum("qpir,pqra,ai->i", h2e_v, a23, h1e_v, optimize=True)
        + np.einsum("pi,pa,ai->i", h1e_v, a25, h1e_v, optimize=True)
    )
    norm = (
        np.einsum("qpir,rpqbac,baic->i", h2e_v, dm3_h, h2e_v, optimize=True)
        + 2.0 * np.einsum("qpir,rpqa,ai->i", h2e_v, dm2_h, h1e_v, optimize=True)
        + np.einsum("pi,pa,ai->i", h1e_v, dm1_h, h1e_v, optimize=True)
    )
    return _norm_to_energy(norm, energy, -mo_energy[:ncore])


def _sr(h1e, h2e, h2e_v, h1e_v, dms, ci, binary, mo_energy, nocc, contracted_a16=None):
    dm1 = dms["1"]
    dm2 = dms["2"]
    dm3 = dms["3"]
    a16 = _make_a16(h1e, h2e, dms, ci=ci, binary=binary, contracted_terms=contracted_a16)
    a17 = _make_a17(h1e, h2e, dm2, dm3)
    a19 = _make_a19(h1e, h2e, dm1, dm2)
    energy = (
        np.einsum("ipqr,pqrabc,iabc->i", h2e_v, a16, h2e_v, optimize=True)
        + 2.0 * np.einsum("ipqr,pqra,ia->i", h2e_v, a17, h1e_v, optimize=True)
        + np.einsum("ip,pa,ia->i", h1e_v, a19, h1e_v, optimize=True)
    )
    norm = (
        np.einsum("ipqr,rpqbac,iabc->i", h2e_v, dm3, h2e_v, optimize=True)
        + 2.0 * np.einsum("ipqr,rpqa,ia->i", h2e_v, dm2, h1e_v, optimize=True)
        + np.einsum("ip,pa,ia->i", h1e_v, dm1, h1e_v, optimize=True)
    )
    return _norm_to_energy(norm, energy, mo_energy[nocc:])
