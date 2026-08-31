#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal libcint wrapper for qchem one-electron integrals.

This module intentionally exposes only the small subset of functionality
needed by pyqed's SOC helpers.
"""

from ctypes import POINTER, Structure, byref, c_double, c_int, cdll
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.special import factorial


def _find_libcint_backend():
    """
    Locate a local libcint-compatible shared library.
    """
    candidates = []

    try:
        import pyscf
    except ImportError:
        pyscf = None

    if pyscf is not None:
        root = Path(pyscf.__file__).resolve().parent
        candidates.extend([
            root / 'lib' / 'libcgto.dylib',
            root / 'lib' / 'libcgto.so',
            root / 'lib' / 'deps' / 'lib' / 'libcint.6.dylib',
            root / 'lib' / 'deps' / 'lib' / 'libcint.so',
        ])

    for path in candidates:
        if path.exists():
            return path

    raise OSError(
        "Could not locate a local libcint-compatible shared library. "
        "Install PySCF or provide a libcint backend."
    )


def ndptr(enable_null=False, **kwargs):
    """
    Wrapped ``numpy.ctypeslib.ndpointer`` that optionally accepts null pointers.
    """
    base = np.ctypeslib.ndpointer(**kwargs)

    if not enable_null:
        return base

    def from_param(cls, obj):
        return obj if obj is None else base.from_param(obj)

    return type(base.__name__, (base,), {"from_param": classmethod(from_param)})


class PairData(Structure):
    _fields_ = [
        ("rij", c_double * 3),
        ("eij", c_double),
        ("cceij", c_double),
    ]


class CINTOpt(Structure):
    _fields_ = [
        ("index_xyz_array", POINTER(POINTER(c_int))),
        ("non0ctr", POINTER(POINTER(c_int))),
        ("sortedidx", POINTER(POINTER(c_int))),
        ("nbas", c_int),
        ("log_max_coeff", POINTER(POINTER(c_double))),
        ("pairdata", POINTER(POINTER(PairData))),
    ]


class _LibCInt:
    def __init__(self):
        self._lib = cdll.LoadLibrary(str(_find_libcint_backend()))
        cfunc = self._lib.CINTdel_optimizer
        cfunc.argtypes = (POINTER(POINTER(CINTOpt)),)
        self._cache = {"CINTdel_optimizer": cfunc}

    def __getitem__(self, name):
        if name in self._cache:
            return self._cache[name]

        cfunc = getattr(self._lib, name)
        if name.endswith("_optimizer"):
            cfunc.argtypes = (
                POINTER(POINTER(CINTOpt)),
                ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                c_int,
                ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                c_int,
                ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
            )
        else:
            cfunc.argtypes = (
                ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS", "WRITEABLE")),
                ndptr(enable_null=True, dtype=c_int, ndim=1, flags=("C_CONTIGUOUS",)),
                ndptr(dtype=c_int, ndim=1, flags=("C_CONTIGUOUS",)),
                ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                c_int,
                ndptr(dtype=c_int, ndim=2, flags=("C_CONTIGUOUS",)),
                c_int,
                ndptr(dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
                POINTER(CINTOpt),
                ndptr(enable_null=True, dtype=c_double, ndim=1, flags=("C_CONTIGUOUS",)),
            )
            cfunc.restype = c_int

        self._cache[name] = cfunc
        return cfunc


LIBCINT = _LibCInt()


def normalized_coeffs(shell):
    """
    Normalize contraction coefficients in the libcint/PySCF convention.
    """
    angmom = _shell_angmom(shell)
    coeffs = _shell_coeff_matrix(shell)

    def gaussian_int(l, a):
        return 0.5 * factorial(0.5 * l - 0.5) * a ** (-0.5 * l - 0.5)

    def gto_norm(l, a):
        return 1.0 / np.sqrt(gaussian_int(2 * l + 2, 2 * a))

    cs = np.einsum("km,k->km", coeffs, gto_norm(angmom, shell.exps))
    es = gaussian_int(2 * angmom + 2, shell.exps[:, None] + shell.exps[None, :])
    ss = 1.0 / np.sqrt(np.einsum("km,kl,lm->m", cs, es, cs))
    return np.einsum("km,m->km", cs, ss)


def _shell_coeff_matrix(shell):
    coeffs = getattr(shell, "coeffs", None)
    if coeffs is None:
        coeffs = getattr(shell, "coefs", None)
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if coeffs.ndim != 2:
        raise ValueError("Shell contraction coefficients must be 1D or 2D.")
    return coeffs


def _shell_angmom(shell):
    angmom = getattr(shell, "angmom", None)
    if angmom is not None:
        return int(angmom)
    shell_tuple = getattr(shell, "shell", None)
    if shell_tuple is not None:
        return int(np.sum(shell_tuple))
    raise ValueError("Shell angular momentum could not be determined.")


def _shell_num_seg_cont(shell):
    num_seg = getattr(shell, "num_seg_cont", None)
    if num_seg is not None:
        return int(num_seg)
    return int(_shell_coeff_matrix(shell).shape[1])


def _shell_num_angmom(shell, coord_type):
    if coord_type == 'spherical':
        num_sph = getattr(shell, "num_sph", None)
        if num_sph is not None:
            return int(num_sph)
        angmom = _shell_angmom(shell)
        return 2 * angmom + 1

    num_cart = getattr(shell, "num_cart", None)
    if num_cart is not None:
        return int(num_cart)
    return 1


def _resolve_shell_icenter(shell, atom_coords, tol=1e-10):
    """
    Resolve the atom-center index for a shell.

    Native contracted Gaussians store their origin directly rather than an
    atom index, so recover the matching molecular center when needed.
    """
    icenter = getattr(shell, "icenter", None)
    if icenter is not None:
        return int(icenter)

    coord = getattr(shell, "coord", None)
    if coord is None:
        coord = getattr(shell, "origin", None)
    coord = np.asarray(coord, dtype=float)
    if coord.shape != (3,):
        raise ValueError(
            "Shell center could not be determined: missing icenter and invalid coord/origin."
        )

    matches = np.where(np.all(np.isclose(atom_coords, coord, atol=tol, rtol=0.0), axis=1))[0]
    if matches.size != 1:
        raise ValueError(
            "Shell center could not be determined uniquely from shell.coord/origin."
        )
    return int(matches[0])


class CBasis1e:
    """
    Minimal shell/basis buffer for one-electron libcint calls.
    """

    def __init__(self, basis, atom_symbols, atom_coords, coord_type='spherical'):
        coord_type = coord_type.lower()
        if coord_type not in ('spherical', 'cartesian'):
            raise ValueError("coord_type must be 'spherical' or 'cartesian'.")

        self.coord_type = coord_type
        self.basis = tuple(basis)
        self.atom_symbols = list(atom_symbols)
        self.atom_coords = np.asarray(atom_coords, dtype=float)

        if coord_type == 'spherical':
            suffix = '_sph'
        else:
            suffix = '_cart'
        self._suffix = suffix

        # Atom symbols are element labels in the molecule's native basis.
        from periodictable import elements
        atnums = [elements.isotope(symbol).number for symbol in self.atom_symbols]

        natm = len(atnums)
        nbas = 0
        nbfn = 0
        nenv = 20 + 4 * natm
        offs = []
        atom_ao_offsets = np.zeros(natm + 1, dtype=int)
        shell_icenters = []

        for shell in self.basis:
            icenter = _resolve_shell_icenter(shell, self.atom_coords)
            shell_icenters.append(icenter)
            shell_num_angmom = _shell_num_angmom(shell, coord_type)
            shell_num_seg = _shell_num_seg_cont(shell)
            shell_coeffs = _shell_coeff_matrix(shell)
            offs.extend([shell_num_angmom] * shell_num_seg)
            atom_ao_offsets[icenter + 1] += shell_num_angmom * shell_num_seg
            nbas += shell_num_seg
            nbfn += shell_num_angmom * shell_num_seg
            nenv += shell.exps.size + shell_coeffs.size

        self.natm = natm
        self.nbas = nbas
        self.nbfn = nbfn
        self._offs = np.asarray(offs, dtype=c_int)
        self._max_off = int(self._offs.max())
        self._ao_slices = np.cumsum(atom_ao_offsets)

        ienv = 20
        atm = np.zeros((natm, 6), dtype=c_int)
        bas = np.zeros((nbas, 8), dtype=c_int)
        env = np.zeros((nenv,), dtype=c_double)

        for atm_row, atnum, atcoord in zip(atm, atnums, self.atom_coords):
            atm_row[0] = atnum
            atm_row[1] = ienv
            env[ienv:ienv + 3] = atcoord
            ienv += 3
            atm_row[2] = 0
            atm_row[3] = ienv
            env[ienv] = 0.0
            ienv += 1
            atm_row[4:6] = 0

        ibas = 0
        for shell, icenter in zip(self.basis, shell_icenters):
            shell_coeffs = _shell_coeff_matrix(shell)
            nprim = shell_coeffs.shape[0]
            iexp = ienv
            ienv += shell.exps.size
            env[iexp:ienv] = shell.exps
            icoef = ienv
            ienv += shell_coeffs.size
            env[icoef:ienv] = normalized_coeffs(shell).reshape(-1, order="F")

            for iprim in range(icoef, icoef + shell_coeffs.size, nprim):
                bas[ibas, 0] = icenter
                bas[ibas, 1] = _shell_angmom(shell)
                bas[ibas, 2] = nprim
                bas[ibas, 3] = 1
                bas[ibas, 4] = 0
                bas[ibas, 5] = iexp
                bas[ibas, 6] = iprim
                bas[ibas, 7] = 0
                ibas += 1

        self.atm = atm
        self.bas = bas
        self.env = env

    @contextmanager
    def optimizer(self, opt_func):
        opt = POINTER(CINTOpt)()
        opt_func(byref(opt), self.atm, self.natm, self.bas, self.nbas, self.env)
        try:
            yield opt
        finally:
            LIBCINT["CINTdel_optimizer"](byref(opt))

    def ao_slice_by_atom(self, ia):
        """
        AO start/end indices for atom ``ia``.
        """
        return int(self._ao_slices[ia]), int(self._ao_slices[ia + 1])

    def int1e(self, func_name, components=tuple(), inv_origin=None, hermi=True):
        """
        Evaluate a one-electron integral.
        """
        func = LIBCINT[func_name + self._suffix]
        opt_func = LIBCINT[func_name + "_optimizer"]

        if len(components) == 0:
            shape_comp = (1,)
            squeeze = True
        else:
            shape_comp = tuple(components)
            squeeze = False

        prod_comp = int(np.prod(shape_comp))
        out = np.zeros((self.nbfn, self.nbfn) + shape_comp, dtype=np.float64, order="F")
        buf = np.zeros(prod_comp * self._max_off ** 2, dtype=np.float64)
        shls = np.zeros(2, dtype=np.int32)

        env = self.env.copy()
        if inv_origin is not None:
            env[4:7] = np.asarray(inv_origin, dtype=float)

        ipos = 0
        with self.optimizer(opt_func) as opt:
            for ishl in range(self.nbas):
                shls[0] = ishl
                p_off = int(self._offs[ishl])
                jpos = 0
                if hermi:
                    jshl_range = range(ishl + 1)
                else:
                    jshl_range = range(self.nbas)
                for jshl in jshl_range:
                    shls[1] = jshl
                    q_off = int(self._offs[jshl])
                    func(buf, None, shls, self.atm, self.natm, self.bas, self.nbas, env, opt, None)
                    buf_array = buf[:p_off * q_off * prod_comp].reshape(
                        p_off, q_off, *shape_comp, order="F"
                    )
                    out[ipos:ipos + p_off, jpos:jpos + q_off] = buf_array
                    if hermi and jshl != ishl:
                        out[jpos:jpos + q_off, ipos:ipos + p_off] = np.swapaxes(buf_array, 0, 1)
                    buf[:] = 0.0
                    jpos += q_off
                ipos += p_off

        if squeeze:
            out = out.squeeze(axis=-1)
        return out
