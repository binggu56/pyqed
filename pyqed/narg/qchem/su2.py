#!/usr/bin/env python3
"""SU(2)-adapted quantum-chemistry NARG backend."""

from __future__ import annotations

import numpy as np

from .active_space import CAS_OPTION_DEFAULTS, pop_active_space_options, prepare_active_space
from .su2_chain import diagonalize_block, run_su2_narg_chain


class NARG:
    """Object API for the direct-reduced SU(2) quantum-chemistry NARG driver."""

    DEFAULT_OPTIONS = {
        "D": 80,
        "D_by_size": None,
        "nstates": 6,
        "target_j2": None,
        "target_nelec": None,
        "final_size": None,
        "return_spin": False,
        **CAS_OPTION_DEFAULTS,
    }

    def __init__(self, mf, *, mol=None, h1e=None, eri=None, **options):
        self.mf = mf
        self.mol = mol if mol is not None else getattr(mf, "mol", None)
        self.h1e = h1e
        self.eri = eri
        self.options = dict(self.DEFAULT_OPTIONS)
        self.options.update(options)
        self.e_tot = None
        self.block = None
        self.spin_info = None
        self.result = None
        self.chain = None
        self.timings = None
        self.active_space = None
        self.ncas = None
        self.nelecas = None
        self.ncore = None
        self.mo_core = None
        self.mo_cas = None
        self.e_core = None

    def integrals(self):
        """Return MO one- and two-electron integrals for the wrapped mean field."""
        opts = dict(self.options)
        cas_options = pop_active_space_options(opts)
        h1e, eri, _, _ = prepare_active_space(
            self.mf,
            self.mol,
            h1e=self.h1e,
            eri=self.eri,
            **cas_options,
        )
        return h1e, eri

    def _set_active_space(self, active_space):
        self.active_space = active_space
        if active_space is None:
            self.ncas = self.nelecas = self.ncore = None
            self.mo_core = self.mo_cas = None
            self.e_core = None
            return
        self.ncas = active_space.ncas
        self.nelecas = active_space.nelecas
        self.ncore = active_space.ncore
        self.mo_core = active_space.mo_core
        self.mo_cas = active_space.mo_cas
        self.e_core = active_space.energy_core

    def _target_nelec(self, explicit=None):
        if explicit is not None:
            return int(explicit)
        if self.mol is not None and hasattr(self.mol, "nelec"):
            return int(np.sum(np.asarray(self.mol.nelec, dtype=int).reshape(-1)))
        return None

    def _target_j2(self, explicit=None):
        if explicit is not None:
            return int(explicit)
        if self.mol is not None:
            if hasattr(self.mol, "spin"):
                return int(self.mol.spin)
            if hasattr(self.mol, "nelec"):
                nelec = np.asarray(self.mol.nelec, dtype=int).reshape(-1)
                if nelec.size == 2:
                    return int(abs(nelec[0] - nelec[1]))
        return 0

    @staticmethod
    def _D_by_size(D, D_by_size, final_size):
        if D_by_size is not None:
            return {int(k): int(v) for k, v in dict(D_by_size).items()}
        D = int(D)
        out = {2: min(10, D)}
        for nsites in range(3, int(final_size)):
            out[nsites] = D
        return out

    def run(self, **options):
        """Run SU(2)-NARG and return ``(e_tot, block)`` by default."""
        opts = dict(self.options)
        opts.update(options)
        cas_options = pop_active_space_options(opts)
        h1e = opts.pop("h1e", None)
        eri = opts.pop("eri", None)

        active_mol = opts.pop("mol", None)
        if active_mol is not None:
            self.mol = active_mol
        if self.mol is None:
            self.mol = getattr(self.mf, "mol", None)

        h1e, eri, prepared_mol, active_space = prepare_active_space(
            self.mf,
            self.mol,
            h1e=h1e,
            eri=eri,
            **cas_options,
        )
        self.h1e = h1e
        self.eri = eri
        self.mol = prepared_mol
        self._set_active_space(active_space)

        final_size = opts.pop("final_size", None)
        final_size = h1e.shape[0] if final_size is None else int(final_size)
        target_nelec = self._target_nelec(opts.pop("target_nelec", None))
        if target_nelec is None:
            target_nelec = final_size
        target_j2 = self._target_j2(opts.pop("target_j2", None))
        nstates = int(opts.pop("nstates", 6))
        return_spin = bool(opts.pop("return_spin", False))
        D = opts.pop("D", self.DEFAULT_OPTIONS["D"])
        D_by_size = self._D_by_size(D, opts.pop("D_by_size", None), final_size)
        if opts:
            unknown = ", ".join(sorted(opts))
            raise TypeError(f"Unknown SU2-NARG options: {unknown}")

        self.chain = run_su2_narg_chain(
            h1e,
            eri,
            D_by_size,
            final_size=final_size,
            target_nelec=target_nelec,
        )
        roots, block = diagonalize_block(
            self.chain.final,
            nelec=target_nelec,
            j2=target_j2,
            nroots=nstates,
        )
        enuc = self.mol.energy_nuc() if self.mol is not None else 0.0
        self.e_tot = roots + enuc
        self.block = block
        self.timings = self.chain.timings
        self.spin_info = {
            "j2": target_j2,
            "spin": 0.5 * target_j2,
            "target_nelec": target_nelec,
        }
        if return_spin:
            self.result = (self.e_tot, self.block, self.spin_info)
        else:
            self.result = (self.e_tot, self.block)
        return self.result
