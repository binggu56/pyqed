# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:28:20 2022

Cartesian to internal coordinates transformation

Based on QCL https://github.com/ben-albrecht/qcl/blob/master/qcl/

@author: Bing Gu
"""

from __future__ import print_function
from __future__ import division

# import math
import os
import sys
import hashlib
import math
import numpy
from numpy import pi
from numpy.linalg import norm
import numpy as np

from pyqed import dag, au2angstrom
from pyqed.qchem.hf import RHF, ROHF, UHF
from periodictable import elements
try:
    from pyscf import dft, scf, gto, ao2mo
except ImportError:
    dft = None
    scf = None
    gto = None
    ao2mo = None


# import scipy.linalg as linalg
# from scipy.optimize import newton

# from pyscf.lib import logger
# import pyscf.ao2mo
# import pyscf
# from functools import reduce
from pyqed.qchem.basis import ContractedGaussian
from pyqed.qchem.basis import build_builtin, overlap



# try:
#     from cclib.parser.data import ccData
#     from cclib.parser.utils import PeriodicTable
# except ImportError:
#     print("Failed to load cclib!")
#     raise

# for _atm, _bas, _env
CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_ZETA   = 3
PTR_FRAC_CHARGE = 4
PTR_RADIUS = 5
ATM_SLOTS  = 6
ATOM_OF    = 0
ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
RADI_POWER = 3 # for ECP
KAPPA_OF   = 4
SO_TYPE_OF = 4 # for ECP
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8


_BUILTIN_OPTION_SPECS = (
    ("coord_type", "builtin_coord_type", "native_coord_type", str, "spherical"),
    ("parallel", "builtin_parallel", "native_parallel", bool, False),
    (
        "eri_workers", "builtin_eri_workers", "native_eri_workers",
        lambda v: None if v is None else int(v), None,
    ),
    ("parallel_min_nao", "builtin_parallel_min_nao", "native_parallel_min_nao", int, 12),
    ("eri_screen_tol", "builtin_eri_screen_tol", "native_eri_screen_tol", float, 1.0e-10),
    ("direct_scf_tol", "builtin_direct_scf_tol", "native_direct_scf_tol", float, 1.0e-13),
    ("eri_backend", "builtin_eri_backend", "native_eri_backend", str, "auto"),
    ("rys_cache_mib", "builtin_rys_cache_mib", "native_rys_cache_mib", int, 256),
    (
        "rys_spherical_cache_mib", "builtin_rys_spherical_cache_mib",
        "native_rys_spherical_cache_mib", int, 0,
    ),
    ("eri_representation", "builtin_eri_representation", "native_eri_representation", str, "auto"),
    ("aosym", "builtin_aosym", "native_aosym", lambda v: None if v is None else str(v), "s8"),
    ("auxbasis", "builtin_auxbasis", "native_auxbasis", lambda v: None if v is None else str(v), None),
    ("ri_metric_tol", "builtin_ri_metric_tol", "native_ri_metric_tol", float, 1e-10),
    ("ri_metric_solver", "builtin_ri_metric_solver", "native_ri_metric_solver", str, "auto"),
    ("ri_purpose", "builtin_ri_purpose", "native_ri_purpose", str, "jk"),
    ("ri_screen_tol", "builtin_ri_screen_tol", "native_ri_screen_tol", lambda v: None if v is None else float(v), None),
    ("ri_block_size", "builtin_ri_block_size", "native_ri_block_size", lambda v: None if v is None else int(v), None),
    ("ri_storage", "builtin_ri_storage", "native_ri_storage", str, "auto"),
    ("ri_tensor_backend", "builtin_ri_tensor_backend", "native_ri_tensor_backend", str, "auto"),
    ("ri_cache", "builtin_ri_cache", "native_ri_cache", bool, True),
    ("ri_cache_dir", "builtin_ri_cache_dir", "native_ri_cache_dir", lambda v: None if v is None else str(v), None),
    ("one_electron_cache", "builtin_one_electron_cache", "native_one_electron_cache", bool, True),
    ("auto_ri_min_nao", "builtin_auto_ri_min_nao", "native_auto_ri_min_nao", int, 24),
    ("low_rank_tol", "builtin_low_rank_tol", "native_low_rank_tol", float, 1e-10),
    ("low_rank_max_rank", "builtin_low_rank_max_rank", "native_low_rank_max_rank", lambda v: None if v is None else int(v), None),
    ("build_factors", "builtin_build_factors", "native_build_factors", bool, False),
)


def _pop_builtin_options(kwargs):
    """
    Collect builtin backend options from a namespaced dict plus legacy kwargs.

    Precedence is:
    1. explicit top-level builtin_* kwargs
    2. explicit top-level native_* kwargs
    3. builtin_options mapping
    4. native_options mapping
    5. defaults
    """
    raw_builtin = kwargs.pop("builtin_options", None)
    raw_native = kwargs.pop("native_options", None)
    builtin_options = {} if raw_builtin is None else dict(raw_builtin)
    native_options = {} if raw_native is None else dict(raw_native)

    options = {}
    for short_name, builtin_name, native_name, caster, default in _BUILTIN_OPTION_SPECS:
        value = default

        for source in (native_options, builtin_options):
            if short_name in source:
                value = source[short_name]
            elif builtin_name in source:
                value = source[builtin_name]
            elif native_name in source:
                value = source[native_name]

        if native_name in kwargs:
            value = kwargs.pop(native_name)
        if builtin_name in kwargs:
            value = kwargs.pop(builtin_name)

        options[short_name] = caster(value)

    return options


def _normalize_eri_representation(value):
    value = str(value).lower().replace("_", "-")
    aliases = {
        "factor": "factors",
        "factored": "factors",
        "factorized": "factors",
        "cholesky": "factors",
        "cd": "factors",
        "df": "ri",
        "density-fit": "ri",
        "density-fitting": "ri",
        "ri": "ri",
        "direct": "direct",
        "direct-scf": "direct",
        "direct_jk": "direct",
        "direct-jk": "direct",
        "auto": "auto",
        "automatic": "auto",
        "s4": "s4",
        "packed": "s4",
        "packed-dense": "s4",
        "s8": "s8",
        "packed8": "s8",
        "packed-8": "s8",
        "eightfold": "s8",
        "8fold": "s8",
        "dense-factors": "dense+factors",
        "dense-plus-factors": "dense+factors",
        "dense+factor": "dense+factors",
        "dense-ri": "dense+ri",
        "dense+df": "dense+ri",
        "dense-density-fit": "dense+ri",
        "dense-density-fitting": "dense+ri",
        "s4-factors": "s4+factors",
        "s4+factor": "s4+factors",
        "packed-factors": "s4+factors",
        "s8-factors": "s8+factors",
        "s8+factor": "s8+factors",
        "packed8-factors": "s8+factors",
    }
    return aliases.get(value, value)


def _normalize_aosym(value):
    if value is None:
        return "s1"
    value = str(value).lower().replace("_", "-")
    aliases = {
        "1": "s1",
        "s1": "s1",
        "dense": "s1",
        "full": "s1",
        "none": "s1",
        "4": "s4",
        "s4": "s4",
        "packed": "s4",
        "packed-dense": "s4",
        "8": "s8",
        "s8": "s8",
        "packed8": "s8",
        "packed-8": "s8",
        "eightfold": "s8",
        "8fold": "s8",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"s1", "s4", "s8"}:
        raise ValueError("builtin_aosym/native_aosym must be 's1', 's4', or 's8'.")
    return normalized


def _split_eri_representation_and_aosym(eri_representation, aosym="s1"):
    """
    Keep legacy eri='s8' style inputs working while making aosym the canonical
    home for AO permutation symmetry.
    """
    eri_representation = _normalize_eri_representation(eri_representation)
    aosym = _normalize_aosym(aosym)
    legacy = {
        "s4": ("dense", "s4"),
        "s8": ("dense", "s8"),
        "s4+factors": ("dense+factors", "s4"),
        "s8+factors": ("dense+factors", "s8"),
    }
    if eri_representation in legacy:
        eri_representation, legacy_aosym = legacy[eri_representation]
        if aosym != "s1" and aosym != legacy_aosym:
            raise ValueError("Conflicting ERI symmetry requested by eri and aosym.")
        aosym = legacy_aosym
    return eri_representation, aosym


def _normalize_builtin_options(options, strict=False):
    """
    Normalize a build-time builtin options mapping.
    """
    if options is None:
        return None
    if not hasattr(options, "items"):
        raise TypeError("build(options=...) must be a mapping.")

    raw_options = dict(options)
    raw_keys = set(raw_options)
    tmp = {"builtin_options": raw_options}
    normalized = _pop_builtin_options(tmp)
    if (
        raw_keys & {"eri_representation", "builtin_eri_representation", "native_eri_representation"}
        and not raw_keys & {"aosym", "builtin_aosym", "native_aosym"}
    ):
        normalized_representation = _normalize_eri_representation(normalized["eri_representation"])
        normalized["aosym"] = (
            "s8"
            if normalized_representation in {"dense", "dense+factors", "factors"}
            else "s1"
        )
    normalized["eri_representation"], normalized["aosym"] = _split_eri_representation_and_aosym(
        normalized["eri_representation"],
        normalized.get("aosym", "s1"),
    )
    if strict and tmp:
        unknown = ", ".join(sorted(tmp))
        raise ValueError(f"Unknown builtin build option(s): {unknown}")
    return normalized


_ANGULAR_LETTERS = "spdfghijklmno"


def _basis_shell_signature(basis_fn):
    return (
        tuple(int(x) for x in getattr(basis_fn, "shell")),
        tuple(float(x) for x in np.asarray(getattr(basis_fn, "origin"), dtype=float)),
        tuple(float(x) for x in getattr(basis_fn, "exps")),
    )


def _basis_contracted_shell_signature(basis_fn):
    shell = tuple(int(x) for x in getattr(basis_fn, "shell"))
    return (
        sum(shell),
        tuple(float(x) for x in np.asarray(getattr(basis_fn, "origin"), dtype=float)),
        tuple(float(x) for x in getattr(basis_fn, "exps")),
        tuple(round(float(x), 14) for x in getattr(basis_fn, "coefs")),
    )


def _cartesian_component_label(shell):
    lx, ly, lz = (int(x) for x in shell)
    l = lx + ly + lz
    if l == 0:
        return ""

    parts = []
    if lx:
        parts.append("x" if lx == 1 else f"x{lx}")
    if ly:
        parts.append("y" if ly == 1 else f"y{ly}")
    if lz:
        parts.append("z" if lz == 1 else f"z{lz}")
    return "".join(parts)


def _spherical_component_labels(l):
    if l == 0:
        return [""]
    if l == 1:
        return ["x", "y", "z"]
    if l == 2:
        return ["xy", "yz", "z2", "xz", "x2-y2"]
    if l == 3:
        return ["m-3", "m-2", "m-1", "m0", "m+1", "m+2", "m+3"]
    return [f"m{m:+d}" if m != 0 else "m0" for m in range(-l, l + 1)]


def _angular_shell_name(l, shell_count):
    if l >= len(_ANGULAR_LETTERS):
        raise ValueError(f"Angular momentum l={l} is not supported for AO labeling.")
    return f"{shell_count + l}{_ANGULAR_LETTERS[l]}"


def _match_atom_index_from_origin(atom_coords, origin, tol=1e-8):
    deltas = atom_coords - origin[None, :]
    idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
    if not np.allclose(atom_coords[idx], origin, atol=tol, rtol=0.0):
        raise ValueError(
            "Failed to match a basis-function center to an atom. "
            "AO labeling requires atom-centered Gaussian basis functions."
        )
    return idx
# pointer to env
PTR_EXPCUTOFF   = 0
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG   = 4
PTR_RINV_ZETA   = 7
PTR_RANGE_OMEGA = 8
PTR_F12_ZETA    = 9
PTR_GTG_ZETA    = 10
NGRIDS          = 11
PTR_GRIDS       = 12
AS_RINV_ORIG_ATOM = 17
AS_ECPBAS_OFFSET = 18
AS_NECPBAS      = 19
PTR_ENV_START   = 20
# parameters from libcint
NUC_POINT = 1
NUC_GAUSS = 2
# nucleus with fractional charges. It can be used to mimic MM particles
NUC_FRAC_CHARGE = 3
NUC_ECP = 4  # atoms with pseudo potential

def atomic_chain(natom, z, element='H', basis='631g', spin=0):

    # ds = np.linspace(-4, 4, natom)

    elements = [element, ] * natom

    R = np.zeros((natom, 3))
    R[:, 2] = z

    atom = build_atom_from_coords(elements, R)

    mol = Molecule(
        atom = atom,
        basis = basis,
        unit = 'b',
        spin = spin,
        )

    return mol

# DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)

def get_hcore_mo(mf):
    """
    calc the core Hamiltonian in MOs

    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if scf is not None and isinstance(mf, scf.rhf.RHF):
        mo_coeff = mf.mo_coeff
        return dag(mo_coeff) @ mf.get_hcore() @ mo_coeff

    elif scf is not None and isinstance(mf, scf.uhf.UHF):

        ha, hb = mf.get_hcore()
        Ca, Cb = mf.mo_coeff  # MOs for alpha and beta electrons

        return [dag(Ca) @ ha @ Ca, dag(Cb) @ hb @ Cb]

    else:
        raise ValueError('Input should be be mean-field object.')

def get_eri_mo(mf):
    """
    get the two-electron integrals as a numpy array of shape (N,N,N, N)
    where N is the number of orbitals


    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    mo_coeff : TYPE
        MOs. Not necesarrily the canonical orbitals. E.g. natural orbitals

    Returns
    -------
    eri : TYPE
        DESCRIPTION.

    """
    if scf is not None and ao2mo is not None and isinstance(mf, scf.rhf.RHF):
        Ca = mf.mo_coeff
        n = Ca.shape[-1]
        # eri = ao2mo.get_mo_eri(mol, mo_coeff)
        eri_aa = (ao2mo.general(mf._eri , (Ca, Ca, Ca, Ca),
                                compact=False)).reshape((n,n,n,n), order="C")
        return  eri_aa

    elif scf is not None and ao2mo is not None and isinstance(mf, scf.uhf.UHF):

        Ca, Cb = mf.mo_coeff
        n = Ca.shape[-1]

        eri_aa = (ao2mo.general( mf._eri , (Ca, Ca, Ca, Ca),
                                compact=False)).reshape((n,n,n,n), order="C")
        eri_aa -= eri_aa.swapaxes(1,3)

        eri_bb = (ao2mo.general( mf._eri , (Cb, Cb, Cb, Cb),
        compact=False)).reshape((n,n,n,n), order="C")
        eri_bb -= eri_bb.swapaxes(1,3)

        eri_ab = (ao2mo.general( mf._eri , (Ca, Ca, Cb, Cb),
        compact=False)).reshape((n,n,n,n), order="C")

        # eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        eri_ba = (ao2mo.general( mf._eri , (Cb, Cb, Ca, Ca),
        compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

        return H2

def build_atom_from_coords(atom_symbol_list, coords):
    """
    construct the atom data format (i.e. xyz format) used in pyscf from coordinates and atom symbols

    Parameters
    ----------
    atom_symbol_list : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.

    Returns
    -------
    atom : TYPE
        DESCRIPTION.

    """
    natm = len(atom_symbol_list)
    atom = []
    for n in range(natm):
        atom.append([atom_symbol_list[n],  coords[n, :].tolist()])

    return atom



def format_atom(atoms, unit='b', origin=0, axes=None):
    # '''Convert the input :attr:`Mole.atom` to the internal data format.
    # Including, changing the nuclear charge to atom symbol, converting the
    # coordinates to AU, rotate and shift the molecule.
    # If the :attr:`~Mole.atom` is a string, it takes ";" and "\\n"
    # for the mark to separate atoms;  "," and arbitrary length of blank space
    # to separate the individual terms for an atom.  Blank line will be ignored.

    # Args:
    #     atoms : list or str
    #         the same to :attr:`Mole.atom`

    # Kwargs:
    #     origin : ndarray
    #         new axis origin.
    #     axes : ndarray
    #         (new_x, new_y, new_z), new coordinates
    #     unit : str or number
    #         If unit is one of strings (B, b, Bohr, bohr, AU, au), the coordinates
    #         of the input atoms are the atomic unit;  If unit is one of strings
    #         (A, a, Angstrom, angstrom, Ang, ang), the coordinates are in the
    #         unit of angstrom;  If a number is given, the number are considered
    #         as the Bohr value (in angstrom), which should be around 0.53.
    #         Set unit=1 if wishing to preserve the unit of the coordinates.

    # Returns:
    #     "atoms" in the internal format. The internal format is
    #         | atom = [[atom1, (x, y, z)],
    #         |         [atom2, (x, y, z)],
    #         |         ...
    #         |         [atomN, (x, y, z)]]

    # Examples:

    # >>> gto.format_atom('9,0,0,0; h@1 0 0 1', origin=(1,1,1))
    # [['F', [-1.0, -1.0, -1.0]], ['H@1', [-1.0, -1.0, 0.0]]]
    # >>> gto.format_atom(['9,0,0,0', (1, (0, 0, 1))], origin=(1,1,1))
    # [['F', [-1.0, -1.0, -1.0]], ['H', [-1, -1, 0]]]
    # '''
    def str2atm(line):
        dat = line.split()
        try:
            coords = [float(x) for x in dat[1:4]]
        except:
            raise ValueError('Failed to parse geometry %s' % line)

            # else:
            #     coords = list(eval(','.join(dat[1:4])))
        if len(coords) != 3:
            raise ValueError('Coordinates error in %s' % line)

        return [dat[0], coords]

    if isinstance(atoms, str):
        # The input atoms points to a geometry file
        if os.path.isfile(atoms):
            try:
                atoms = readxyz(atoms)
            except ValueError:
                sys.stderr.write('\nFailed to parse geometry file  %s\n\n' % atoms)
                raise

        atoms = atoms.replace(';','\n').replace(',',' ').replace('\t',' ')
        fmt_atoms = []
        for dat in atoms.split('\n'):
            dat = dat.strip()
            if dat and dat[0] != '#':
                fmt_atoms.append(dat)


        if len(fmt_atoms[0].split()) < 4:
            # fmt_atoms = from_zmatrix('\n'.join(fmt_atoms))
            # TODO: add zmat supporter
            raise ValueError('Zmat not supported yet.')
        else:
            fmt_atoms = [str2atm(line) for line in fmt_atoms]

    else:
        fmt_atoms = []
        for atom in atoms:
            if isinstance(atom, str):
                if atom.lstrip()[0] != '#':
                    fmt_atoms.append(str2atm(atom.replace(',',' ')))
            else:
                if isinstance(atom[1], (int, float)):
                    fmt_atoms.append([atom[0], atom[1:4]])
                else:
                    fmt_atoms.append([atom[0], atom[1]])

    if axes is None:
        axes = np.eye(3)

    if is_au(unit):
        unit = 1
    else:
        unit = 1/au2angstrom

    c = numpy.array([a[1] for a in fmt_atoms], dtype=numpy.double)
    c = numpy.einsum('ix,kx->ki', axes * unit, c - origin)
    z = [a[0] for a in fmt_atoms]

    return list(map(list, zip(z, c.tolist())))
    # return list(zip(z, c.tolist()))



def is_au(unit):
    '''Return whether the unit is recognized as A.U. or not
    '''
    return unit.upper().startswith(('B', 'AU'))

# def fromfile(filename, format=None):
#     '''Read molecular geometry from a file
#     (in testing)

#     Supported formats:
#         | raw: Each line is  <symbol> <x> <y> <z>
#         | xyz: XYZ cartesian coordinates format
#         | zmat: Z-matrix format
#     '''
#     if format is None:  # Guess format based on filename
#         format = os.path.splitext(filename)[1][1:].lower()
#         if format not in ('xyz', 'zmat', 'sdf'):
#             format = 'raw'
#     with open(filename, 'r') as f:
#         return fromstring(f.read(), format)


def fromfile(filename, format=None):
        if format is None:  # Guess format based on filename
            format = os.path.splitext(filename)[1][1:].lower()
            if format in ('xyz'):
                return readxyz(filename)
        else:
            raise ValueError('Format {} not supported. Use XYZ'.format(format))


        # with open(filename, 'r') as f:
        #     return fromstring(f.read(), format)



    #     atoms = atoms.replace(';','\n').replace(',',' ').replace('\t',' ')
    #     fmt_atoms = []
    #     for dat in atoms.split('\n'):
    #         dat = dat.strip()
    #         if dat and dat[0] != '#':
    #             fmt_atoms.append(dat)

    #     if len(fmt_atoms[0].split()) < 4:
    #         fmt_atoms = from_zmatrix('\n'.join(fmt_atoms))
    #     else:
    #         fmt_atoms = [str2atm(line) for line in fmt_atoms]
    # else:
    #     fmt_atoms = []
    #     for atom in atoms:
    #         if isinstance(atom, str):
    #             if atom.lstrip()[0] != '#':
    #                 fmt_atoms.append(str2atm(atom.replace(',',' ')))
    #         else:
    #             if isinstance(atom[1], (int, float)):
    #                 fmt_atoms.append([_atom_symbol(atom[0]), atom[1:4]])
    #             else:
    #                 fmt_atoms.append([_atom_symbol(atom[0]), atom[1]])

    # if len(fmt_atoms) == 0:
    #     return []

    # if axes is None:
    #     axes = numpy.eye(3)

    # if isinstance(unit, str):
    #     if is_au(unit):
    #         unit = 1.
    #     else:
    #         unit = 1./param.BOHR
    # else:
    #     unit = 1./unit

    # c = numpy.array([a[1] for a in fmt_atoms], dtype=numpy.double)
    # c = numpy.einsum('ix,kx->ki', axes * unit, c - origin)
    # z = [a[0] for a in fmt_atoms]
    # return list(zip(z, c.tolist()))

    # return atom_symbols,




# class ccData_xyz(ccData):
#     """
#     ccData subclass for xyzfiles
#     TODO: Checks for previous steps before continuing,
#     i.e. check for dist_matrix before building conn_vector
#     Includes some hot new attributes and class methods
#     """

#     def __init__(self, attributes={}):
#         """Adding some new attributes for xyzfiles"""

#         self.newcoords = None
#         self.distancematrix = None

#         # Internal Coordinate Connectivity
#         self.connectivity = None
#         self.angleconnectivity = None
#         self.dihedralconnectivity = None

#         # Internal Coordinates
#         self.distances = None
#         self.angles = None
#         self.dihedrals = None

#         self._attrtypes['comment'] = str
#         self._attrlist.append('comment')
#         self._attrtypes['filename'] = str
#         self._attrlist.append('filename')
#         self._attrtypes['elements'] = list
#         self._attrlist.append('elements')

#         #self._attrtypes['distancematrix'] = np.ndarray
#         #self._attrlist.append('distancematrix')
#         #self._attrtypes['connectivity'] = list
#         #self._attrlist.append('connectivity')

#         super(ccData_xyz, self).__init__(attributes=attributes)

#         # Initialize new data types if attributes were parsed as an original ccdata_xyz
#         if not hasattr(self, 'elements'):
#             pt = PeriodicTable()
#             self.comment = '\n'
#             self.filename = ''
#             self.elements = []
#             for atomno in self.atomnos:
#                 self.elements.append(pt.element[atomno])

#     def _build_distance_matrix(self):
#         """Build distance matrix between all atoms
#            TODO: calculate distances only as needed for efficiency"""
#         coords = self.atomcoords[-1]
#         self.distancematrix = np.zeros((len(coords), len(coords)))
#         for i in range(len(coords)):
#             for j in [x for x in range(len(coords)) if x > i]:
#                 self.distancematrix[i][j] = norm(coords[i] - coords[j])
#                 self.distancematrix[j][i] = self.distancematrix[i][j]

#     def build_zmatrix(self):
#         """
#        'Z-Matrix Algorithm'
#         Build main components of zmatrix:
#         Connectivity vector
#         Distances between connected atoms (atom >= 1)
#         Angles between connected atoms (atom >= 2)
#         Dihedral angles between connected atoms (atom >= 3)
#         """
#         self._build_distance_matrix()

#         # self.connectivity[i] tells you the index of 2nd atom connected to atom i
#         self.connectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.angleconnectivity[i] tells you the index of
#         #    3rd atom connected to atom i and atom self.connectivity[i]
#         self.angleconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # self.dihedralconnectivity tells you the index of 4th atom connected to
#         #    atom i, atom self.connectivity[i], and atom self.angleconnectivity[i]
#         self.dihedralconnectivity = np.zeros(len(self.atomnos)).astype(int)

#         # Starts with r1
#         self.distances = np.zeros(len(self.atomnos))
#         # Starts with a2
#         self.angles = np.zeros(len(self.atomnos))
#         # Starts with d3
#         self.dihedrals = np.zeros(len(self.atomnos))

#         atoms = range(1, len(self.atomnos))
#         for atom in atoms:
#             # For current atom, find the nearest atom among previous atoms
#             distvector = self.distancematrix[atom][:atom]
#             distmin = np.array(distvector[np.nonzero(distvector)]).min()
#             nearestindices = np.where(distvector == distmin)[0]
#             nearestatom = nearestindices[0]

#             self.connectivity[atom] = nearestatom
#             self.distances[atom] = distmin

#             # Compute Angles
#             if atom >= 2:
#                 atms = [0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.connectivity[atms[1]]
#                 if atms[2] == atms[1]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[2] = idx
#                             break

#                 self.angleconnectivity[atom] = atms[2]

#                 self.angles[atom] = self._calc_angle(atms[0], atms[1], atms[2])

#             # Compute Dihedral Angles
#             if atom >= 3:
#                 atms = [0, 0, 0, 0]
#                 atms[0] = atom
#                 atms[1] = self.connectivity[atms[0]]
#                 atms[2] = self.angleconnectivity[atms[0]]
#                 atms[3] = self.angleconnectivity[atms[1]]
#                 if atms[3] in atms[:3]:
#                     for idx in range(1, len(self.connectivity[:atom])):
#                         if self.connectivity[idx] in atms and not idx in atms:
#                             atms[3] = idx
#                             break

#                 self.dihedrals[atom] =\
#                     self._calc_dihedral(atms[0], atms[1], atms[2], atms[3])
#                 if math.isnan(self.dihedrals[atom]):
#                     # TODO: Find explicit way to denote undefined dihedrals
#                     self.dihedrals[atom] = 0.0

#                 self.dihedralconnectivity[atom] = atms[3]

#     def _calc_angle(self, atom1, atom2, atom3):
#         """Calculate angle between 3 atoms"""
#         coords = self.atomcoords[-1]
#         vec1 = coords[atom2] - coords[atom1]
#         uvec1 = vec1 / norm(vec1)
#         vec2 = coords[atom2] - coords[atom3]
#         uvec2 = vec2 / norm(vec2)
#         return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

#     def _calc_dihedral(self, atom1, atom2, atom3, atom4):
#         """
#            Calculate dihedral angle between 4 atoms
#            For more information, see:
#                http://math.stackexchange.com/a/47084
#         """
#         coords = self.atomcoords[-1]
#         # Vectors between 4 atoms
#         b1 = coords[atom2] - coords[atom1]
#         b2 = coords[atom2] - coords[atom3]
#         b3 = coords[atom4] - coords[atom3]

#         # Normal vector of plane containing b1,b2
#         n1 = np.cross(b1, b2)
#         un1 = n1 / norm(n1)

#         # Normal vector of plane containing b1,b2
#         n2 = np.cross(b2, b3)
#         un2 = n2 / norm(n2)

#         # un1, ub2, and m1 form orthonormal frame
#         ub2 = b2 / norm(b2)
#         um1 = np.cross(un1, ub2)

#         # dot(ub2, n2) is always zero
#         x = np.dot(un1, un2)
#         y = np.dot(um1, un2)

#         dihedral = np.arctan2(y, x)*(180.0/pi)
#         if dihedral < 0:
#             dihedral = 360.0 + dihedral
#         return dihedral

#     def build_xyz(self):
#         """ Build xyz representation from z-matrix"""
#         coords = self.atomcoords[-1]
#         self.newcoords = np.zeros((len(coords), 3))
#         for i in range(len(coords)):
#             self.newcoords[i] = self._calc_position(i)
#         self.atomcoords[-1] = self.newcoords

#     def _calc_position(self, i):
#         """Calculate position of another atom based on internal coordinates"""

#         if i > 1:
#             j = self.connectivity[i]
#             k = self.angleconnectivity[i]
#             l = self.dihedralconnectivity[i]

#             # Prevent doubles
#             if k == l and i > 0:
#                 for idx in range(1, len(self.connectivity[:i])):
#                     if self.connectivity[idx] in [i, j, k] and not idx in [i, j, k]:
#                         l = idx
#                         break

#             avec = self.newcoords[j]
#             bvec = self.newcoords[k]

#             dst = self.distances[i]
#             ang = self.angles[i] * pi / 180.0

#             if i == 2:
#                 # Third atom will be in same plane as first two
#                 tor = 90.0 * pi / 180.0
#                 cvec = np.array([0, 1, 0])
#             else:
#                 # Fourth + atoms require dihedral (torsional) angle
#                 tor = self.dihedrals[i] * pi / 180.0
#                 cvec = self.newcoords[l]

#             v1 = avec - bvec
#             v2 = avec - cvec

#             n = np.cross(v1, v2)
#             nn = np.cross(v1, n)

#             n /= norm(n)
#             nn /= norm(nn)

#             n *= -sin(tor)
#             nn *= cos(tor)

#             v3 = n + nn
#             v3 /= norm(v3)
#             v3 *= dst * sin(ang)

#             v1 /= norm(v1)
#             v1 *= dst * cos(ang)

#             position = avec + v3 - v1

#         elif i == 1:
#             # Second atom dst away from origin along Z-axis
#             j = self.connectivity[i]
#             dst = self.distances[i]
#             position = np.array([self.newcoords[j][0] + dst, self.newcoords[j][1], self.newcoords[j][2]])

#         elif i == 0:
#             # First atom at the origin
#             position = np.array([0, 0, 0])

#         return position

#     @property
#     def splitatomnos(self):
#         """Returns tuple of atomnos from reactants joined by atoms 0 and 1"""
#         fragments = [[], []]

#         return fragments


#     def print_distance_matrix(self):
#         """Print distance matrix in formatted form"""

#         # Title
#         print("\nDistance Matrix")

#         # Row Indices
#         for i in range(len(self.distancematrix)):
#             print("%3d" % i, end="  ")

#         print("\n", end="")
#         idx = 0
#         for vector in self.distancematrix:

#             # Column indices
#             print(idx, end=" ")

#             # Actual Values
#             for element in vector:
#                 if not element == 0:
#                     print("%1.2f" % element, end=" ")
#                 else:
#                     print("%1s" % " ", end="    ")
#             print("\n", end="")
#             idx += 1

#     def print_xyz(self):
#         """Print Standard XYZ Format"""
#         if not self.newcoords.any():
#             self.build_xyz()

#         print(len(self.newcoords))

#         if self.comment:
#             print(self.comment, end='')
#         else:
#             print(self.filename, end='')

#         atomcoords = [x.tolist() for x in self.newcoords]
#         for i in range(len(atomcoords)):
#             atomcoords[i].insert(0, self.elements[i])

#         for atom in atomcoords:
#             print("  %s %10.5f %10.5f %10.5f" % tuple(atom))

#     def print_gzmat(self):
#         """Print Gaussian Z-Matrix Format
#         e.g.
#         0  3
#         C
#         O  1  r2
#         C  1  r3  2  a3
#         Si 3  r4  1  a4  2  d4
#         ...
#         Variables:
#         r2= 1.1963
#         r3= 1.3054
#         a3= 179.97
#         r4= 1.8426
#         a4= 120.10
#         d4=  96.84
#         ...
#         """
#         pt = PeriodicTable()

#         print('#', self.filename, "\n")
#         print(self.comment)

#         print(self.comment, end='')
#         for i in range(len(self.atomnos)):
#             idx = str(i+1)+" "
#             if i >= 3:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx,
#                       self.dihedralconnectivity[i]+1, " d"+idx.rstrip())
#             elif i == 2:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx,
#                       self.angleconnectivity[i]+1, " a"+idx.rstrip())
#             elif i == 1:
#                 print(pt.element[self.atomnos[i]], "",
#                       self.connectivity[i]+1, " r"+idx.rstrip())
#             elif i == 0:
#                 print(pt.element[self.atomnos[i]])

#         print("Variables:")

#         for i in range(1, len(self.atomnos)):
#             idx = str(i+1)+"="
#             if i >= 3:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#                 print("%s" % "d"+idx, "%6.2f" % self.dihedrals[i])
#             elif i == 2:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])
#                 print("%s" % "a"+idx, "%6.2f" % self.angles[i])
#             elif i == 1:
#                 print("%s" % "r"+idx, "%6.4f" % self.distances[i])

#     def print_zmat(self):
#         """Print Standard Z-Matrix Format"""
#         #TODO

#         """
#         0 1
#         O
#         O 1 1.5
#         H 1 1.0 2 120.0
#         H 2 1.0 1 120.0 3 180.0
#         """


# from pyqed import eig_asymm, is_positive_def, dag


def inertia_moment(mass, coords):
    """
    compute the inertia moment of a rigid body

    Parameters
    ----------
    mass : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.

    Returns
    -------
    im : TYPE
        DESCRIPTION.

    """
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    coords = coords - mass_center
    im = np.einsum('i,ij,ik->jk', mass, coords, coords)
    im = np.eye(3) * im.trace() - im
    return im


# class Molecule:
#     def __init__(self, geometry):
#         self.geometry = geometry

#         self.ge = None
#         self.ee = None
#         self.edip = None
#         self.mdip = None

#     # def atom_symbol(self):
#     #     return [mol.atom_symbol(i) for i in range(self.natoms)]
#     def rhf(self):
#         pass

#     def absorption(self, ttype='electron'):
#         # range, uv, ir, xray

#         pass

#     def photoelectron(self):
#         pass

#     def emission(self):
#         pass

# def build_molecular_integrals(mol):
#     """
#     electronic integrals in AO

#     Parameters
#     ----------
#     mol : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     atoms = mol.atom_symbols()
#     atcoords = mol.atom_coords()

#     if mol.basis in ['631g', '6-31g', '631G', '6-31G']:
#         # Obtain basis functions from the basis set files
#         basis_dict = parse_gbs("6-31g.1.gbs")
#         basis = make_contractions(basis_dict, atoms, atcoords, coord_types="c")


#     # compute overlap integrals in AO and MO basis
#     mol.overlap = overlap_integral(basis)


#     # olp_mo = overlap_integral(basis, transform=mo_coeffs.T)

#     # compute kinetic energy integrals in AO basis
#     k_int1e = kinetic_energy_integral(basis)
#     print("Shape kinetic energy integral: ", k_int1e.shape, "(#AO, #AO)")


#     # compute nuclear-electron attraction integrals in AO basis
#     atnums = np.array([1,1])
#     nuc_int1e = nuclear_electron_attraction_integral(
#             basis, atcoords, atnums)
#     print("Shape Nuclear-electron integral: ", nuc_int1e.shape, "(#AO, #AO)")

#     mol.hcore = k_int1e + nuc_int1e

#     #Compute e-e repulsion integral in MO basis, shape=(#MO, #MO, #MO, #MO)
#     int2e_mo = electron_repulsion_integral(basis, notation='chemist')
#     mol.eri = int2e_mo

def atom_mass_list(mol):
    '''
    A list of mass for all atoms in the molecule
    '''
    return np.array([elements.isotope(mol.atom_symbol(i)).mass \
                     for i in range(mol.natom)])



class Molecule:
    def __init__(self, atom, charge=0, spin=0, basis=None, unit='bohr', **kwargs):

        if isinstance(atom, str):
            # The input atom is a geometry file
            if os.path.isfile(atom):
                try:
                    self._atom = fromfile(atom)
                except ValueError:
                    sys.stderr.write('\nFailed to parse geometry file  %s\n\n' % atom)
                    raise
            else:
                self._atom = format_atom(atom)
        else:
            self._atom = format_atom(atom)

        self.natom = len(self._atom)

        # self.mol = mol
        # self.atom_coord = mol.atom_coord

        # TODO: add unit support.
        if unit.lower() in ['b', 'bohr']:
            for a in range(self.natom):
                self._atom[a][1] = list(np.array(self._atom[a][1]))

        elif unit.lower() in ['a', 'angstrom']:
            # raise ValueError('unit can only be Bohr.')
            for a in range(self.natom):
                self._atom[a][1] = list(np.array(self._atom[a][1])/au2angstrom)

        # self.mass = mol.atom_mass_list()

        self.spin = spin
        self.charge = charge

        # self.atom = atom

        self.distmat = None
        self.basis = basis

        self._nelec = None

        ######## DO NOT CHANGE ####

        self.e_nuc = None
        self.overlap = None
        self.hcore = None
        self.eri = None
        self.eri_s4 = None
        self.eri_s8 = None
        self.eri_factors = None
        self._builtin_direct_jk_data = None
        self.builtin_resolved_eri_representation = None
        self.builtin_resolved_aosym = None
        self.native_resolved_eri_representation = None
        self.native_resolved_aosym = None

        self.nao = None
        self.nmo = None
        self.unit = unit
        self.cart = False
        self._bas = None
        self._bas_cart = None
        self._ao_cart2sph = None
        builtin_options = _pop_builtin_options(kwargs)
        self._set_builtin_options(builtin_options)
        self._builtin_build_info = None

        self._native_build_info = self._builtin_build_info

        self.symmetry = False
        self.symmetry_info = None
        self.groupname = None
        self.irrep_names = None
        self.symm_orb = None
        self.ao_irrep_labels = None
        self.ao_irrep_ids = None
        self.symmetry_axis = None
        self.symmetry_tol = None


    @property
    def atom(self):
        return self._atom

    @atom.setter
    def atom(self, atm):
        self._atom = atm


    def atom_coord(self, a):
        return self._atom[a][1]

    def atom_coords(self):
        return np.array([self._atom[i][1] for i in range(self.natom)])

    def atom_symbol(self, i):
        return self._atom[i][0]

    def atom_symbols(self):
        return [self.atom_symbol(i) for i in range(self.natom)]

    def view(self, **kwargs):
        """Show this molecule in PyQED's interactive 3D viewer."""
        from pyqed.visualization import view

        return view(self, **kwargs)

    def _set_builtin_options(self, options):
        """
        Apply builtin backend options and keep legacy aliases in sync.
        """
        self.builtin_options = dict(options)
        self.builtin_options["eri_representation"], self.builtin_options["aosym"] = _split_eri_representation_and_aosym(
            self.builtin_options["eri_representation"],
            self.builtin_options.get("aosym", "s1"),
        )
        self.builtin_coord_type = self.builtin_options["coord_type"]
        self.builtin_parallel = self.builtin_options["parallel"]
        self.builtin_eri_workers = self.builtin_options["eri_workers"]
        self.builtin_parallel_min_nao = self.builtin_options["parallel_min_nao"]
        self.builtin_eri_screen_tol = self.builtin_options["eri_screen_tol"]
        self.builtin_direct_scf_tol = self.builtin_options["direct_scf_tol"]
        self.builtin_eri_backend = self.builtin_options["eri_backend"]
        self.builtin_rys_cache_mib = self.builtin_options["rys_cache_mib"]
        self.builtin_rys_spherical_cache_mib = self.builtin_options[
            "rys_spherical_cache_mib"
        ]
        self.builtin_eri_representation = self.builtin_options["eri_representation"]
        self.builtin_aosym = self.builtin_options["aosym"]
        self.builtin_auxbasis = self.builtin_options["auxbasis"]
        self.builtin_ri_metric_tol = self.builtin_options["ri_metric_tol"]
        self.builtin_ri_metric_solver = self.builtin_options["ri_metric_solver"]
        self.builtin_ri_purpose = self.builtin_options["ri_purpose"]
        self.builtin_ri_screen_tol = self.builtin_options["ri_screen_tol"]
        self.builtin_ri_block_size = self.builtin_options["ri_block_size"]
        self.builtin_ri_storage = self.builtin_options["ri_storage"]
        self.builtin_ri_tensor_backend = self.builtin_options["ri_tensor_backend"]
        self.builtin_ri_cache = self.builtin_options["ri_cache"]
        self.builtin_ri_cache_dir = self.builtin_options["ri_cache_dir"]
        self.builtin_one_electron_cache = self.builtin_options["one_electron_cache"]
        self.builtin_auto_ri_min_nao = self.builtin_options["auto_ri_min_nao"]
        self.builtin_low_rank_tol = self.builtin_options["low_rank_tol"]
        self.builtin_low_rank_max_rank = self.builtin_options["low_rank_max_rank"]
        self.builtin_build_factors = self.builtin_options["build_factors"]

        # Backward-compatible aliases for the older native_* API.
        self.native_options = self.builtin_options
        self.native_coord_type = self.builtin_coord_type
        self.native_parallel = self.builtin_parallel
        self.native_eri_workers = self.builtin_eri_workers
        self.native_parallel_min_nao = self.builtin_parallel_min_nao
        self.native_eri_screen_tol = self.builtin_eri_screen_tol
        self.native_direct_scf_tol = self.builtin_direct_scf_tol
        self.native_eri_backend = self.builtin_eri_backend
        self.native_rys_cache_mib = self.builtin_rys_cache_mib
        self.native_rys_spherical_cache_mib = self.builtin_rys_spherical_cache_mib
        self.native_eri_representation = self.builtin_eri_representation
        self.native_aosym = self.builtin_aosym
        self.native_auxbasis = self.builtin_auxbasis
        self.native_ri_metric_tol = self.builtin_ri_metric_tol
        self.native_ri_metric_solver = self.builtin_ri_metric_solver
        self.native_ri_purpose = self.builtin_ri_purpose
        self.native_ri_screen_tol = self.builtin_ri_screen_tol
        self.native_ri_block_size = self.builtin_ri_block_size
        self.native_ri_storage = self.builtin_ri_storage
        self.native_ri_tensor_backend = self.builtin_ri_tensor_backend
        self.native_ri_cache = self.builtin_ri_cache
        self.native_ri_cache_dir = self.builtin_ri_cache_dir
        self.native_one_electron_cache = self.builtin_one_electron_cache
        self.native_auto_ri_min_nao = self.builtin_auto_ri_min_nao
        self.native_low_rank_tol = self.builtin_low_rank_tol
        self.native_low_rank_max_rank = self.builtin_low_rank_max_rank
        self.native_build_factors = self.builtin_build_factors


    @property
    def nelec(self):
        if self._nelec is None:
            self._nelec = sum(self.atom_charges()) - self.charge

        return self._nelec

    def nuc_charge_center(self):

        charges = self.atom_charges()
        coords = self.atom_coords()

        return np.einsum('z,zx->x', charges, coords) / charges.sum()

    def build(
        self,
        options=None,
        eri=None,
        aosym=None,
        auxbasis=None,
        symmetry=None,
        symmetry_axis='z',
        symmetry_tol=1.0e-8,
    ):
        """
        build molecular integrals

        Parameters
        ----------
        options : dict, optional
            Native integral-build options. Use short keys such as
            ``eri_representation``, ``aosym``, ``low_rank_tol``,
            ``eri_screen_tol``, ``direct_scf_tol``, ``parallel``, ``eri_workers``,
            ``rys_cache_mib``, and ``rys_spherical_cache_mib``.
        eri : {'auto', 'dense', 's4', 's8', 'direct', 'factors', 'cd', 'ri'}, optional
            Short alias for the native ``eri_representation`` option.
        aosym : {'s1', 's4', 's8'}, optional
            AO ERI permutation symmetry for dense-like native storage. The
            default build uses ``eri='auto'``: compact exact ``s8``
            storage for small systems, native RI factors for medium systems
            when an auxiliary basis is available, and Cholesky/CD factors as a
            fallback. Explicit dense shortcuts such as ``eri='dense'`` default
            to ``aosym='s8'`` when no symmetry is supplied.
        auxbasis : str, optional
            Short alias for the native ``auxbasis`` option used by
            ``eri='ri'`` and ``eri='dense+ri'``.
        symmetry : bool or str, optional
            Build native Abelian point-group metadata after the AO integrals are
            available. ``True`` currently selects ``C2v``; strings such as
            ``'C2v'`` and ``'D2h'`` select a supported subgroup explicitly.
        symmetry_axis : {'z'}, optional
            Principal-axis convention for native symmetry. Only ``'z'`` is
            currently implemented.
        symmetry_tol : float, optional
            Geometry tolerance, in bohr, used when validating symmetry
            operations.

        Returns
        -------
        None.

        """
        if eri is not None:
            options = {} if options is None else dict(options)
            options["eri_representation"] = _normalize_eri_representation(eri)

        if aosym is not None:
            options = {} if options is None else dict(options)
            requested_aosym = _normalize_aosym(aosym)
            existing_aosym = _normalize_aosym(options.get("aosym", "s1"))
            if existing_aosym != "s1" and existing_aosym != requested_aosym:
                raise ValueError("Conflicting ERI symmetry requested by eri/options and aosym.")
            options["aosym"] = requested_aosym

        if auxbasis is not None:
            options = {} if options is None else dict(options)
            options["auxbasis"] = auxbasis

        if options is not None:
            self._set_builtin_options(_normalize_builtin_options(options, strict=True))

        self.eri_s4 = None
        self.eri_s8 = None
        self.eri_factors = None
        self._builtin_direct_jk_data = None
        self.builtin_resolved_eri_representation = None
        self.builtin_resolved_aosym = None
        self.native_resolved_eri_representation = None
        self.native_resolved_aosym = None
        self._builtin_build_info = None
        self._native_build_info = None
        self.symmetry = False
        self.symmetry_info = None
        self.groupname = None
        self.irrep_names = None
        self.symm_orb = None
        self.ao_irrep_labels = None
        self.ao_irrep_ids = None
        self.symmetry_axis = None
        self.symmetry_tol = None

        build_builtin(self)

        if symmetry:
            from pyqed.qchem.symmetry import build_molecular_symmetry

            info = build_molecular_symmetry(
                self,
                symmetry,
                axis=symmetry_axis,
                tol=float(symmetry_tol),
            )
            self.symmetry = True
            self.symmetry_info = info
            self.groupname = info.groupname
            self.irrep_names = info.irrep_names
            self.symm_orb = info.symm_orb
            self.ao_irrep_labels = info.ao_irrep_labels
            self.ao_irrep_ids = info.ao_irrep_ids
            self.symmetry_axis = symmetry_axis
            self.symmetry_tol = float(symmetry_tol)

    def geometry_signature(self, digits=12):
        """
        Hashable signature for the current geometry and integral build context.
        """
        coords = np.asarray(self.atom_coords(), dtype=float)
        rounded = np.round(coords, digits).reshape(-1)
        return (
            tuple(self.atom_symbols()),
            repr(self.basis),
            int(self.charge),
            int(self.spin),
            coords.shape,
            tuple(rounded.tolist()),
        )

    def geometry_hash(self, digits=12):
        """
        Short digest of the current geometry signature.
        """
        return hashlib.sha1(repr(self.geometry_signature(digits=digits)).encode('utf-8')).hexdigest()

    def _add_suffix(self, intor, cart=None):
        mol = self.topyscf()
        return mol._add_suffix(intor, cart) 
    
    def moment_integral(self, orders=None, center=np.array([0,0,0])):
        """Native Cartesian moments ``<(r-center)^orders>`` in the AO basis."""
        first_moments = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if orders is None:
            orders = first_moments
        orders = np.asarray(orders, dtype=int)
        center = np.asarray(center, dtype=float)
        if orders.ndim != 2 or orders.shape[1] != 3 or np.any(orders < 0):
            raise ValueError("orders must have shape (ncomponent, 3) and be nonnegative.")
        if center.shape != (3,):
            raise ValueError("center must be a length-3 Cartesian vector.")
        basis, transform = self._cart_basis()
        output = np.zeros((orders.shape[0], len(basis), len(basis)), dtype=float)
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                for component, powers in enumerate(orders):
                    value = 0.0
                    for ia, wa in enumerate(bra.prim_weights):
                        for ib, wb in enumerate(ket.prim_weights):
                            primitive = 0.0
                            for kx in range(int(powers[0]) + 1):
                                for ky in range(int(powers[1]) + 1):
                                    for kz in range(int(powers[2]) + 1):
                                        raised = (
                                            int(ket.shell[0]) + kx,
                                            int(ket.shell[1]) + ky,
                                            int(ket.shell[2]) + kz,
                                        )
                                        coefficient = (
                                            math.comb(int(powers[0]), kx)
                                            * math.comb(int(powers[1]), ky)
                                            * math.comb(int(powers[2]), kz)
                                            * (float(ket.origin[0]) - center[0]) ** (int(powers[0]) - kx)
                                            * (float(ket.origin[1]) - center[1]) ** (int(powers[1]) - ky)
                                            * (float(ket.origin[2]) - center[2]) ** (int(powers[2]) - kz)
                                        )
                                        primitive += coefficient * self._primitive_overlap(
                                            bra, ket, ia, ib, ket_shell=raised
                                        )
                            value += float(wa) * float(wb) * primitive
                    output[component, i, j] = value
        return self._apply_ao_transform(output, transform)

    def _cart_basis(self):
        basis = self._bas_cart if self._bas_cart is not None else self._bas
        if basis is None:
            raise ValueError("Build molecular integrals before requesting one-electron integrals.")
        if not all(isinstance(fn, ContractedGaussian) for fn in basis):
            raise RuntimeError("Native one-electron integrals require a ContractedGaussian basis.")
        return list(basis), getattr(self, "_ao_cart2sph", None)

    @staticmethod
    def _raise_axis(shell, axis, delta):
        shell = list(shell)
        shell[axis] += int(delta)
        if shell[axis] < 0:
            return None
        return tuple(shell)

    @staticmethod
    def _primitive_overlap(bra, ket, ia, ib, ket_shell=None):
        ket_shell = ket.shell if ket_shell is None else ket_shell
        return overlap(
            float(bra.exps[ia]),
            tuple(bra.shell),
            np.asarray(bra.origin, dtype=float),
            float(ket.exps[ib]),
            tuple(ket_shell),
            np.asarray(ket.origin, dtype=float),
        )

    def _moment_primitive_overlap(self, bra, ket, ia, ib, pos_axis, ket_shell, center):
        center_shift = float(ket.origin[pos_axis] - center[pos_axis])
        raised = self._raise_axis(ket_shell, pos_axis, 1)
        base = self._primitive_overlap(bra, ket, ia, ib, ket_shell=ket_shell)
        return self._primitive_overlap(bra, ket, ia, ib, ket_shell=raised) + center_shift * base

    def _position_integral_pair(self, bra, ket, pos_axis, center):
        value = 0.0
        for ia, wa in enumerate(bra.prim_weights):
            for ib, wb in enumerate(ket.prim_weights):
                value += wa * wb * self._moment_primitive_overlap(
                    bra, ket, ia, ib, pos_axis, tuple(ket.shell), center
                )
        return value

    def _rxgrad_integral_pair(self, bra, ket, pos_axis, deriv_axis, center):
        value = 0.0
        ket_shell = tuple(ket.shell)
        deriv_power = ket_shell[deriv_axis]
        for ia, wa in enumerate(bra.prim_weights):
            for ib, wb in enumerate(ket.prim_weights):
                beta = float(ket.exps[ib])
                term = 0.0
                lowered = self._raise_axis(ket_shell, deriv_axis, -1)
                if deriv_power > 0 and lowered is not None:
                    term += deriv_power * self._moment_primitive_overlap(
                        bra, ket, ia, ib, pos_axis, lowered, center
                    )
                raised = self._raise_axis(ket_shell, deriv_axis, 1)
                term -= 2.0 * beta * self._moment_primitive_overlap(
                    bra, ket, ia, ib, pos_axis, raised, center
                )
                value += wa * wb * term
        return value

    def _gradient_integral_pair(self, bra, ket, axis):
        value = 0.0
        ket_shell = tuple(ket.shell)
        power = ket_shell[axis]
        for ia, wa in enumerate(bra.prim_weights):
            for ib, wb in enumerate(ket.prim_weights):
                beta = float(ket.exps[ib])
                term = 0.0
                lowered = self._raise_axis(ket_shell, axis, -1)
                if power > 0 and lowered is not None:
                    term += power * self._primitive_overlap(
                        bra, ket, ia, ib, ket_shell=lowered
                    )
                raised = self._raise_axis(ket_shell, axis, 1)
                term -= 2.0 * beta * self._primitive_overlap(
                    bra, ket, ia, ib, ket_shell=raised
                )
                value += wa * wb * term
        return value

    @staticmethod
    def _apply_ao_transform(op, transform):
        if transform is None:
            return np.asarray(op)
        return np.einsum("pi,xpq,qj->xij", transform, op, transform, optimize=True)

    def position_integral(self, center=np.array([0,0,0])):
        """
        Native first-moment integrals ``<chi_mu | r - center | chi_nu>``.

        Returns
        -------
        np.ndarray
            Array of shape ``(3, nao, nao)`` in the molecule's AO basis.
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 Cartesian vector.")
        basis, transform = self._cart_basis()
        nao = len(basis)
        op = np.zeros((3, nao, nao), dtype=float)
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                for axis in range(3):
                    op[axis, i, j] = self._position_integral_pair(bra, ket, axis, center)
        return self._apply_ao_transform(op, transform)

    def rxgrad_integral(self, center=np.array([0,0,0])):
        """
        Native real angular generator integrals ``<chi_mu | (r-center) x grad | chi_nu>``.
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 Cartesian vector.")
        basis, transform = self._cart_basis()
        nao = len(basis)
        op = np.zeros((3, nao, nao), dtype=float)
        pairs = ((1, 2), (2, 0), (0, 1))
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                for component, (pos_a, deriv_b) in enumerate(pairs):
                    pos_b, deriv_a = deriv_b, pos_a
                    op[component, i, j] = (
                        self._rxgrad_integral_pair(bra, ket, pos_a, deriv_b, center)
                        - self._rxgrad_integral_pair(bra, ket, pos_b, deriv_a, center)
                    )
        return self._apply_ao_transform(op, transform)

    def angular_momentum_integral(self, center=np.array([0,0,0])):
        """
        Native angular-momentum integrals ``<chi_mu | -i (r-center) x grad | chi_nu>``.
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 Cartesian vector.")
        return -1j * self.rxgrad_integral(center=center)

    def magnetic_dipole_integral(self, center=np.array([0,0,0]), convention='cd'):
        """
        Native magnetic transition-dipole integrals.

        ``convention='cd'`` returns the standard real matrix
        ``-0.5 * (r-center) x grad`` used with real CI vectors for
        length-gauge rotatory strengths.  ``'operator'`` returns the physical
        one-electron magnetic-dipole operator
        ``-L/2 = 0.5j (r-center) x grad``.  ``'raw'``/``'pyscf'`` returns the
        historical unhalved real matrix ``-(r-center) x grad``.
        """
        rxgrad = self.rxgrad_integral(center=center)
        key = str(convention).lower()
        if key in {'cd', 'real', 'standard', 'orca'}:
            return -0.5 * rxgrad
        if key in {'raw', 'unhalved', 'pyscf'}:
            return -rxgrad
        if key in {'operator', 'physical'}:
            return 0.5j * rxgrad
        raise ValueError("convention must be 'cd', 'operator', or 'raw'.")

    def momentum_integral(self):
        """Native momentum integrals ``<chi_mu|-i grad|chi_nu>``."""
        basis, transform = self._cart_basis()
        operator = np.zeros((3, len(basis), len(basis)), dtype=complex)
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                for axis in range(3):
                    operator[axis, i, j] = -1j * self._gradient_integral_pair(
                        bra, ket, axis
                    )
        return self._apply_ao_transform(operator, transform)

    def ao_labels(self):
        """
        Native AO labels derived from the built basis metadata.

        Returns
        -------
        list[str]
            One label per AO in the current `mol.nao` ordering.
        """
        basis = self._bas
        if basis is None:
            raise ValueError("Build molecular integrals before requesting AO labels.")

        atom_coords = np.asarray(self.atom_coords(), dtype=float)
        atom_symbols = self.atom_symbols()
        labels = []

        if self.cart or self._ao_cart2sph is None or len(basis) == self.nao:
            shell_counts = {}
            shell_labels = {}
            for basis_fn in basis:
                shell = tuple(int(x) for x in basis_fn.shell)
                l = sum(shell)
                atom_idx = _match_atom_index_from_origin(
                    atom_coords, np.asarray(basis_fn.origin, dtype=float)
                )
                shell_key = (atom_idx, l, _basis_contracted_shell_signature(basis_fn))
                if shell_key not in shell_labels:
                    shell_counts[(atom_idx, l)] = shell_counts.get((atom_idx, l), 0) + 1
                    shell_labels[shell_key] = _angular_shell_name(l, shell_counts[(atom_idx, l)])
                shell_name = shell_labels[shell_key]
                component = _cartesian_component_label(shell)
                labels.append(f"{atom_idx} {atom_symbols[atom_idx]} {shell_name}{component}")
            if len(labels) != self.nao:
                raise ValueError("AO label count does not match mol.nao.")
            return labels

        shell_counts = {}
        shell_labels = {}
        start = 0
        nao_cart = len(basis)
        while start < nao_cart:
            basis_fn = basis[start]
            sig = _basis_contracted_shell_signature(basis_fn)
            shell = tuple(int(x) for x in basis_fn.shell)
            l = sum(shell)
            ncart = (l + 1) * (l + 2) // 2
            stop = start + ncart
            if stop > nao_cart or any(
                _basis_contracted_shell_signature(basis[k]) != sig
                for k in range(start, stop)
            ):
                raise ValueError("Invalid Cartesian shell ordering while building AO labels.")

            atom_idx = _match_atom_index_from_origin(
                atom_coords, np.asarray(basis_fn.origin, dtype=float)
            )
            shell_key = (atom_idx, l, _basis_contracted_shell_signature(basis_fn))
            if shell_key not in shell_labels:
                shell_counts[(atom_idx, l)] = shell_counts.get((atom_idx, l), 0) + 1
                shell_labels[shell_key] = _angular_shell_name(l, shell_counts[(atom_idx, l)])
            shell_name = shell_labels[shell_key]
            for component in _spherical_component_labels(l):
                labels.append(f"{atom_idx} {atom_symbols[atom_idx]} {shell_name}{component}")
            start = stop

        if len(labels) != self.nao:
            raise ValueError("AO label count does not match spherical mol.nao.")
        return labels


    def topyscf(self):
        """
        change to Pyscf Mol object

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if gto is None:
            raise ImportError("PySCF is not available in this environment.")
        atom = build_atom_from_coords(self.atom_symbols(), self.atom_coords())
        return gto.M(
            atom=atom,
            basis=self.basis,
            unit='bohr',
            charge=self.charge,
            spin=self.spin,
            cart=bool(getattr(self, "cart", False)),
        )

    @staticmethod
    def _canonical_ao_label(label):
        text = " ".join(str(label).replace("^", "").split())
        for angular in "fghi":
            text = text.replace(f"{angular}m+", f"{angular}+")
            text = text.replace(f"{angular}m-", f"{angular}-")
            text = text.replace(f"{angular}m0", f"{angular}0")
            text = text.replace(f"{angular}+0", f"{angular}0")
        return text

    def pyscf_ao_permutation(self, pyscf_mol=None):
        """Map native AO indices to the corresponding PySCF AO indices."""
        if self.nao is None:
            raise ValueError("Build the molecule before aligning PySCF AO order.")
        pmol = self.topyscf() if pyscf_mol is None else pyscf_mol
        native_labels = [self._canonical_ao_label(label) for label in self.ao_labels()]
        pyscf_labels = [self._canonical_ao_label(label) for label in pmol.ao_labels()]
        positions = {}
        for index, label in enumerate(pyscf_labels):
            positions.setdefault(label, []).append(index)
        try:
            permutation = np.asarray(
                [positions[label].pop(0) for label in native_labels],
                dtype=int,
            )
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "Cannot align native and PySCF AO labels for this molecule."
            ) from exc
        if len(permutation) != len(pyscf_labels) or any(positions.values()):
            raise ValueError("Native and PySCF AO label sets do not match.")
        return permutation

    def atom_mass_list(self):
        '''
        A list of mass for all atoms in the molecule
        '''

        return atom_mass_list(self)

    def atom_charge(self, atm_id):
        return elements.isotope(self.atom_symbol(atm_id)).number

    def atom_charges(self):
        return np.array([self.atom_charge(i) for i in range(self.natom)])

    def center_of_mass(self):
        '''
        return center of mass

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        mass = self.atom_mass_list()
        return np.einsum('i,ij->j', mass, self.atom_coords())/mass.sum()

    def inertia_moment(self):
        mass = self.mass
        coords = self.atom_coords()
        return inertia_moment(mass, coords)

    def molecular_frame(self):
        # transfrom to molecular frame

        R0 = self.center_of_mass()

        for i in range(self.natom):
            R = np.array(self._atom[i][1])
            R -= R0

            self._atom[i][1] = list(R)


        return self

    def set_geom(self, R):
        """
        update the molecular geometry (rebuild the AO integrals)

        Parameters
        ----------
        R : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # update coordinates
        for i in range(self.natom):
            self._atom[i][1] = list(np.asarray(R[i], dtype=float))

        # Invalidate AO integral data; low-rank history is preserved separately
        # and keyed by geometry/build settings.
        self.overlap = None
        self.hcore = None
        self.eri = None
        self.nao = None
        self.nmo = None
        self.cart = False
        self._bas = None
        self._bas_cart = None
        self._ao_cart2sph = None
        for attr in ('ao_moment', 'ao_dip', 'ao_magnetic_dip', '_atm', '_env'):
            if hasattr(self, attr):
                setattr(self, attr, None)

        # self.build()

        return self

    def eckart_frame(self, ref):
        """
        transform to the Eckart frame relative to a reference geometry

        Parameters
        ----------
        ref : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        atom_coords = eckart(ref.T, self.atom_coords().T, \
                             self.atom_mass_list())

        self.set_geom(atom_coords)
        return self

    def principle_axes(self):
        pass

    def _build_distance_matrix(self):
        """Build distance matrix between all atoms
           TODO: calculate distances only as needed for efficiency"""
        coords = self.atom_coords()
        natom = self.natom

        distancematrix = np.zeros((natom, natom))

        for i in range(natom):
            for j in range(i+1, natom):
                distancematrix[i, j] = np.linalg.norm(coords[:, i]-coords[:, j])
                distancematrix[j, i] = distancematrix[i, j]

        self.distmat =  distancematrix
        return distancematrix

    def _calc_angle(self, atom1, atom2, atom3):
        """
        Calculate angle in radians between 3 atoms

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vec1 = self.atom_coord(atom2) - self.atom_coord(atom1)
        uvec1 = vec1 / norm(vec1)
        vec2 = self.atom_coord(atom2) - self.atom_coord(atom3)
        uvec2 = vec2 / norm(vec2)
        return np.arccos(np.dot(uvec1, uvec2))*(180.0/pi)

    def _calc_dihedral(self, atom1, atom2, atom3, atom4):
        """

           Calculate dihedral angle (in radians) between 4 atoms
           For more information, see:
               http://math.stackexchange.com/a/47084

        Parameters
        ----------
        atom1 : TYPE
            DESCRIPTION.
        atom2 : TYPE
            DESCRIPTION.
        atom3 : TYPE
            DESCRIPTION.
        atom4 : TYPE
            DESCRIPTION.

        Returns
        -------
        dihedral : TYPE
            DESCRIPTION.

        """
        r1 = self.atom_coord(atom1)
        r2 = self.atom_coord(atom2)
        r3 = self.atom_coord(atom3)
        r4 = self.atom_coord(atom4)

        # Vectors between 4 atoms
        b1 = r2 - r1
        b2 = r2 - r3
        b3 = r4 - r3

        # Normal vector of plane containing b1,b2
        n1 = np.cross(b1, b2)
        un1 = n1 / norm(n1)

        # Normal vector of plane containing b1,b2
        n2 = np.cross(b2, b3)
        un2 = n2 / norm(n2)

        # un1, ub2, and m1 form orthonormal frame
        ub2 = b2 / norm(b2)
        um1 = np.cross(un1, ub2)

        # dot(ub2, n2) is always zero
        x = np.dot(un1, un2)
        y = np.dot(um1, un2)

        dihedral = np.arctan2(y, x)*(180.0/pi)
        if dihedral < 0:
            dihedral = 360.0 + dihedral
        return dihedral

    def zmat(self, rvar=False, avar=False, dvar=False):
        npart = self.natm

        if self.distmat is None:
            self._build_distance_matrix()

        distmat = self.distmat

        atomnames = self.atom_symbols()

        rlist = []
        alist = []
        dlist = []
        if npart > 0:
            # Write the first atom
            print(atomnames[0])

            if npart > 1:
                # and the second, with distance from first
                n = atomnames[1]
                rlist.append(distmat[0][1])
                if (rvar):
                    r = 'R1'
                else:
                    r = '{:>11.5f}'.format(rlist[0])
                print('{:<3s} {:>4d}  {:11s}'.format(n, 1, r))

                if npart > 2:
                    n = atomnames[2]

                    rlist.append(distmat[0][2])
                    if (rvar):
                        r = 'R2'
                    else:
                        r = '{:>11.5f}'.format(rlist[1])

                    alist.append(self._calc_angle(2, 0, 1))
                    if (avar):
                        t = 'A1'
                    else:
                        t = '{:>11.5f}'.format(alist[0])

                    print('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, 1, r, 2, t))

                    if npart > 3:
                        for i in range(3, npart):
                            n = atomnames[i]

                            rlist.append(distmat[i-3][i])
                            if (rvar):
                                r = 'R{:<4d}'.format(i)
                            else:
                                r = '{:>11.5f}'.format(rlist[i-1])

                            alist.append(self._calc_angle(i, i-3, i-2))
                            if (avar):
                                t = 'A{:<4d}'.format(i-1)
                            else:
                                t = '{:>11.5f}'.format(alist[i-2])

                            dlist.append(self._calc_dihedral(i, i-3, i-2, i-1))
                            if (dvar):
                                d = 'D{:<4d}'.format(i-2)
                            else:
                                d = '{:>11.5f}'.format(dlist[i-3])
                            print('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(n, i-2, r, i-1, t, i, d))
        if (rvar):
            print(" ")
            for i in range(npart-1):
                print('R{:<4d} = {:>11.5f}'.format(i+1, rlist[i]))
        if (avar):
            print(" ")
            for i in range(npart-2):
                print('A{:<4d} = {:>11.5f}'.format(i+1, alist[i]))
        if (dvar):
            print(" ")
            for i in range(npart-3):
                print('D{:<4d} = {:>11.5f}'.format(i+1, dlist[i]))

        return

    def jacobian(self, q):
        return

    def metric(self):
        pass

    def tofile(self,fname):
        pass

    def RHF(self, verbose=0):
        return RHF(self, verbose=verbose)

    def casci(
        self,
        ncas,
        nelecas,
        *,
        nstates=None,
        method="direct_ci",
        mf=None,
        scf_options=None,
        run_options=None,
        verbose=0,
    ):
        """Return a CASCI electronic driver configured for geometry scans."""

        if nstates is None:
            nstates = getattr(self, "nstates", 1)
        nstates = int(nstates)
        if nstates <= 0:
            raise ValueError("nstates must be positive")

        if mf is None:
            if self.nao is None:
                self.build()
            mf = self.RHF(verbose=verbose).run(**dict(scf_options or {}))

        from pyqed.qchem.mcscf.casci import CASCI

        electronic = CASCI(
            mf,
            ncas=int(ncas),
            nelecas=nelecas,
            verbose=verbose,
        )
        electronic.nstates = nstates
        electronic.scan_method = str(method)
        electronic.scan_run_kwargs = dict(run_options or {})
        return electronic

    def ROHF(self, verbose=0):
        return ROHF(self, verbose=verbose)

    def UHF(self):
        return UHF(self)

    def RKS(self):
        pass

    def UKS(self):
        pass

    def energy_nuc(self):
        return energy_nuc(self.atom_coords(), self.atom_charges())

def fakemol_for_charges(coords, expnt=1e16):
    if gto is None:
        raise ImportError("PySCF is not available in this environment.")
    return gto.fakemol_for_charges(coords=coords, expnt=expnt)

def intor_cross(intor, mol1, mol2, comp=None, grids=None):
    if gto is None:
        raise ImportError("PySCF is not available in this environment.")
    return gto.intor_cross(intor=intor, mol1=mol1, mol2=mol2, comp=comp, grids=grids)

def make_cintopt(atm, basis, env, intor):
    if gto is None:
        raise ImportError("PySCF is not available in this environment.")
    mol = gto.Mole()
    return gto.moleintor.make_cintopt(atm=mol._atm, bas=mol._bas, env=mol._env, intor=intor)

def energy_nuc(atcoords, atnums):
    # Compute Nucleus-Nucleus repulsion
    rab = np.triu(np.linalg.norm(atcoords[:, None]- atcoords, axis=-1))
    at_charges = np.triu(atnums[:, None] * atnums)[np.where(rab > 0)]
    nn_e = np.sum(at_charges / rab[rab > 0])
    return nn_e

def grad_nuc(mol, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates
    '''
    z = mol.atom_charges()
    r = mol.atom_coords()
    dr = r[:,None,:] - r
    dist = np.linalg.norm(dr, axis=2)
    diag_idx = np.diag_indices(z.size)
    dist[diag_idx] = 1e100
    rinv = 1./dist
    rinv[diag_idx] = 0.
    gs = np.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv**3)
    if atmlst is not None:
        gs = gs[atmlst]
    return gs


def readxyz(fname):
    """
    read XYZ file and parse it to `atom` format

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Returns
    -------
    atomic_symbols : TYPE
        DESCRIPTION.
    atomic_coordinates : TYPE
        DESCRIPTION.

    """
    with open(fname, 'r') as xyz_file:
        lines = xyz_file.readlines()[2:] # Skipping the first two lines

    atomic_symbols = []
    for line in lines:
        atomic_symbols.append(line.split()[0])

    atomic_coordinates = np.array([line.split()[1:4] for line in lines], dtype=np.float64)
    return build_atom_from_coords(atomic_symbols, atomic_coordinates)


# def dihedral(r1, r2, r3, r4):
#     """

#        Calculate dihedral angle (in radians) between 4 atoms
#        For more information, see:
#            http://math.stackexchange.com/a/47084

#     Parameters
#     ----------
#     r1 : 3D vector
#         position of the first atom
#     atom2 : TYPE
#         DESCRIPTION.
#     atom3 : TYPE
#         DESCRIPTION.
#     atom4 : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     dihedral : TYPE
#         DESCRIPTION.

#     """
#     import jax.numpy as jnp

#     # Vectors between 4 atoms
#     b1 = r2 - r1
#     b2 = r2 - r3
#     b3 = r4 - r3

#     # Normal vector of plane containing b1,b2
#     n1 = jnp.cross(b1, b2)
#     un1 = n1 / jnp.norm(n1)

#     # Normal vector of plane containing b1,b2
#     n2 = jnp.cross(b2, b3)
#     un2 = n2 / jnp.norm(n2)

#     # un1, ub2, and m1 form orthonormal frame
#     ub2 = b2 / jnp.norm(b2)
#     um1 = jnp.cross(un1, ub2)

#     # dot(ub2, n2) is always zero
#     x = jnp.dot(un1, un2)
#     y = jnp.dot(um1, un2)

#     dihedral = jnp.arctan2(y, x)*(180.0/pi)
#     if dihedral < 0:
#         dihedral = 360.0 + dihedral
#     return dihedral

def project_nac():
    pass

def metric():
    # metric tensor of curvilinear coordinates
    pass

def quasi_angular_momentum(mass, reference, changed):
    l = 0
    natom = reference.shape[-1]
    for k in range(natom):
        l += mass[k] * np.cross(reference[:,k], changed[:,k])
    return l

def eckart(reference, changed, mass, option=None):
    '''
    % Rotates 'changed' to satisfy both Eckart Conditions exactly with respect to 'reference'
    % Separate translational and rotational degrees of freedom from internal degrees of freedom
    %
    % reference: xyz coordinates as (3,NAtom)-matrix
    % changed: rotated xyz coordinates as (3,NAtom)-matrix
    % masses: 1D array of masses
    % option: shifts COM of the returned geometry to origin if it reads 'shiftCOM'
    %
    % xyz_rot: changed in orientation of reference as (3,NAtom)-matrix
    %
    %
    % Sorting of atoms has to be equal!

        Refs:
    % The procedure is following: Dymarsky, Kudin, J. Chem. Phys. 122, 124103 (2005) and
    % especially Coutsias, et al., J. Comput. Chem. 25, 1849 (2004).
    % According to Kudin, Dymarsky, J. Chem. Phys. 122, 224105 (2005) satisfying Eckart and
    % minimizing the RMSD is the same problem!
        '''
    assert reference.shape == changed.shape

    def com(mass, atom_coord):
        '''
        return center of mass

        Params
        ------
        mass: 1d array
            atomic mass
        atom_coord: 2darray
            cartesian coordinates [3, natom]

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return np.einsum('a,ja->j', mass, atom_coord)/mass.sum()


    # Imaginary coordinates are nonsense
    # if (isreal(reference) == 0) && (isreal(changed) == 0):
    #     raise ValueError('Imaginary coordinates in the XYZ-Structures!')

    natoms = len(mass)

# % shift origin to the center of mass
# % Eckart condition of translation (Eckart 1)
    com_ref = com(mass, reference)
    com_changed = com(mass, changed)

    for i in range(natoms):
        reference[:, i] -= com_ref
        changed[:, i] -= com_changed


    # if (abs(max(max(com_ref))) > 1e-4):
    #      raise Warning('Warning! Translational Eckart Condition for reference not satisfied!')



    # Quasi Angular Momentum
    # Eckart Condition of rotation (Eckart 2)
    # QAM = 0;
    # for k=1:NAtom
    #     QAM = QAM + masses(k)*cross(reference(:,k),changed(:,k));
    # end



    # Matrix A

    A = np.einsum('a, ia, ja -> ij', mass, changed, reference)

    F = np.zeros((4,4))

    F[0,0] = A[0,0] + A[1,1] + A[2,2]
    F[1,1] = A[0,0] - A[1,1] - A[2,2]
    F[2,2] = -A[0,0] + A[1,1] - A[2,2]
    F[3,3] = -A[0,0] - A[1,1] + A[2,2]

    F[1,0] = A[1,2] - A[2,1]
    F[0,1] = F[1,0]
    F[2,0] = A[2,0] - A[0,2]
    F[0,2] = F[2,0]
    F[3,0] = A[0,1] - A[1,0]
    F[0,3] = F[3,0]
    F[2,1] = A[0,1] + A[1,0]
    F[1,2] = F[2,1]
    F[3,1] = A[0,2] + A[2,0]
    F[1,3] = F[3,1]
    F[3,2] = A[1,2] + A[2,1]
    F[2,3] = F[3,2]


    # The maximum eigenvalue [and its corresponding eigenvector]
    # is the correct choice!!

    # [V,D] = eigh(F)
    # [D_, order] = sort(diag(D),'descend');
    # V = V(:,order);
    D_, V = np.linalg.eigh(F)
    idx = np.argsort(-D_)
    D_ = D_[idx]
    V = V[:,idx]

    # % [V,S,~] = svd(F);
    # % [~, order] = sort(diag(S),'descend');
    # % V = V(:,order);

    if (-D_[3] > D_[0]):
        q = V[:,3]
    else:
        q = V[:,0]


    U = np.zeros((3,3))

    U[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    U[1,1] = q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2
    U[2,2] = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2

    U[1,0] = 2 * ( q[1] * q[2] + q[0] * q[3])
    U[2,0] = 2 * ( q[1] * q[3] - q[0] * q[2])
    U[0,1] = 2 * ( q[1] * q[2] - q[0] * q[3])
    U[2,1] = 2 * ( q[2] * q[3] + q[0] * q[1])
    U[0,2] = 2 * ( q[1] * q[3] + q[0] * q[2])
    U[1,2] = 2 * ( q[2] * q[3] - q[0] * q[1])

    if (-D_[3] > D_[0]):
        U = -U


    # Transform 'changed' with T to satisfy Eckart 2
    xyz_rot = U @ changed;

    # # Explicit test of Eckart 2
    # QAM3 = 0;
    # for k=1:NAtom
    #     QAM3 = QAM3 + masses(k)*cross(reference(:,k),xyz_rot(:,k));
    # end

    # tmp = 0;
    # for i=1:1:NAtom
    #     tmp = tmp + (norm(xyz_rot(:,i) - reference(:,i)))^2;
    # end
    # RMSD = sqrt(tmp/NAtom);

    # if (nargin < 4)
    #     xyz_rot = xyz_rot + repmat(comref',1,NAtom);
    # else
    #     if ~(strcmp(option,'shiftCOM'))
    #         xyz_rot = xyz_rot + repmat(comref',1,NAtom);

    return xyz_rot


def scan_pes(method='dft'):
    from pyscf import dft, scf, gto, ao2mo

    x = np.arange(0.7, 4.01, .1)

    mol = gto.Mole()
    if method == 'hf':
        mf_scanner = scf.RHF(mol).as_scanner()
    elif method == 'dft':
        mf_scanner = dft.RKS(mol).set(xc='b3lyp').as_scanner()

    ehf1 = []
    for b in np.arange(0.7, 4.01, 0.1):
        mol = gto.M(verbose = 5,
                    output = 'out_hf-%2.1f' % b,
                    atom = [["F", (0., 0., 0.)],
                            ["H", (0., 0., b)],],
                    basis = 'cc-pvdz')
        ehf1.append(mf_scanner(mol))

    import matplotlib.pyplot as plt
    plt.plot(x, ehf1, '-o', label='HF,0.7->4.0')

def plot_mo_energy(mf):
    """
    plot the energy levels and occupations for a HF calculation

    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Refs
    https://www.andersle.no/posts/2022/mo/mo.html

    """
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 6))
    colors = matplotlib.cm.get_cmap("tab20")(np.linspace(0, 1, len(mf.mo_energy)))

    pos = []
    for i, (energy, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
        left = 3 * i
        right = 3 * i + 2.5
        length = right - left

        (line,) = ax.plot([left, right], [energy, energy], color=colors[i], lw=3)

        electron_x, electron_y = None, None
        if occ == 2:
            electron_x = [left + 0.25 * length, left + 0.75 * length]
            electron_y = [energy, energy]
        elif occ == 1:
            electron_x, electron_y = [left + 0.5], [energy]
        if electron_x and electron_y:
            ax.scatter(electron_x, electron_y, color=line.get_color())

        pos.append(left + 0.5 * length)

    ax.axhline(y=0, ls=":", color="k")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"#{i}" for i, _ in enumerate(pos)])
    ax.set(xlabel="MO number", ylabel="Energy / a.u.")
    sns.despine(fig=fig)


def intrinsic_orbitals(mf):
    """

    Get intrinsic bonding orbitals and localized intrinsic valence virtual orbitals (livvo):

    J. Chem. Theory Comput. 2013, 9, 11, 4834–4843

    Parameters
    ----------
    mf : TYPE
        DESCRIPTION.
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    orbitals : TYPE
        DESCRIPTION.

    Useage:

        To write all the canonical MOs

        ``write_all_coeffs(mol,
    orbitals["canonical"],
    prefix=f"{molecule_name}_cmo",
    dirname="cmo",
    margin=5)''

    """
    """Get molecular orbitals"""

    from pyscf import lo
    mol = mf.mol
    orbitals = {"canonical": mf.mo_coeff}

    orbocc = mf.mo_coeff[:, 0 : mol.nelec[0]]
    orbvirt = mf.mo_coeff[:, mol.nelec[0] :]

    ovlpS = mol.intor_symmetric("int1e_ovlp")

    iaos = lo.iao.iao(mol, orbocc)
    iaos = lo.orth.vec_lowdin(iaos, ovlpS)
    ibos = lo.ibo.ibo(mol, orbocc, locmethod="IBO")
    orbitals["ibo"] = ibos

    livvo = lo.vvo.livvo(mol, orbocc, orbvirt)
    orbitals["livvo"] = livvo
    return orbitals

# def intrinsic_orbitals(mf):
#     """

#     Get intrinsic atomic and bonding orbitals

#     J. Chem. Theory Comput. 2013, 9, 11, 4834–4843

#     Parameters
#     ----------
#     mf : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     iaos: IAO
#     ibos: IBO


#     """
#     from pyscf import lo

#     # Get intrinsic bonding orbitals and localized intrinsic valence virtual orbitals (livvo):
#     orbocc = mf.mo_coeff[:, 0 : mol.nelec[0]]

#     ovlpS = mol.intor_symmetric("int1e_ovlp")

#     iaos = lo.iao.iao(mol, orbocc)
#     iaos = lo.orth.vec_lowdin(iaos, ovlpS)
#     ibos = lo.ibo.ibo(mol, orbocc, locmethod="IBO")

#     return iaos, ibos


def find_homo_lumo(mf):
    lumo = float("inf")
    lumo_idx = None
    homo = -float("inf")
    homo_idx = None
    for i, (energy, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
        if occ > 0 and energy > homo:
            homo = energy
            homo_idx = i
        if occ == 0 and energy < lumo:
            lumo = energy
            lumo_idx = i

    return homo, homo_idx, lumo, lumo_idx


def view_mo(fname):
    """
    cube file

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    import py3Dmol

    data = None
    with open(fname, "r") as infile:
        data = infile.read()

    view = py3Dmol.view()
    view.addVolumetricData(
        data,
        "cube",
        {
            "isoval": 0.05,
            "smoothness": 5,
            "opacity": 0.8,
            "volformat": "cube",
            "color": "blue",
        },
    )
    view.addVolumetricData(
        data,
        "cube",
        {
            "isoval": -0.05,
            "smoothness": 5,
            "opacity": 0.8,
            "volformat": "cube",
            "color": "orange",
        },
    )
    view.addModel(data, "cube")
    view.setStyle({"stick": {}})
    view.zoomTo()
    view.show()


if __name__ == '__main__':
    # from pyscf import gto, tdscf, tools
    # from lime.units import au2fs, au2ev
    # Suppress scientific notation printouts and change default precision
    # np.set_printoptions(precision=4)
    # np.set_printoptions(suppress=True)

    # import proplot as plt
    from timeit import time

    # mol = gto.Mole()
    # mol.verbose = 3
    atom = [['F' , (0,      0., 0.)],
            ['Li', (0, 0., 2)]]
            # ['H', (1.5, 0, 0)]]
    #mol.basis = {'Ne': '6-31G'}
    mol = Molecule(atom)

    # d = mol._build_distance_matrix()
    # print(d)

    # This is from G2/97 i.e. MP2/6-31G*
    # mol.atom = [['H' , (0,      0., 0.)],
    #             ['H', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]



    start = time.time()

    mol.basis = '631g*'
    mol.build()

    print("time building AO integrals = ", time.time()-start)
    mol.RHF().run()



    # print(mol.atom_symbols())
    # print(mol.atom_mass_list())
    # print(mol.atom_charges())
    # print(mol.energy_nuc())

    # print(mol.eri.shape)

    # mf = scf.RHF(mol).run()

    # # plot_mo_energy(mf)

    # orbitals = intrinsic_orbitals(mf)


    # _, homo_idx, _, lumo_idx = find_homo_lumo(mf)
    # print(f"HOMO (index): {homo_idx}")
    # print(f"LUMO (index): {lumo_idx}")

    # tools.cubegen.orbital(
    # mol, "cmo_homo.cube", orbitals["canonical"][:, homo_idx], margin=5
    # )
    # tools.cubegen.orbital(
    #     mol, "cmo_lumo.cube", orbitals["canonical"][:, lumo_idx], margin=5
    # )
    # tools.cubegen.orbital(mol, "ibo_homo.cube", orbitals["ibo"][:, -1], margin=5)

    # tools.cubegen.orbital(
    #     mol, "livvo_lumo.cube", orbitals["livvo"][:, 0], margin=5
    # );



    # geometry2 = [['H' , (0.1,      0., 0.)],
    #             ['H', (1.3, 0., 0.)],
    #             ['H', (1.5, 0, 0)]]

    # mol2 = Molecule(atom=geometry2)



    # print(mol2.atom_coords().shape)
    # print(mol2.com())
    # mol2.molecular_frame()

    # mol2.eckart_frame(mol.atom_coords())

    # print(mol.natm)

    # scan_pes()
    # mole = Molecule(mol)
    # mol.zmat(rvar=True)
    # mf = scf.RHF(mol).run()

    # td = tdscf.TDRHF(mf)
    # td.kernel()
