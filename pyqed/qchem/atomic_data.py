#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 08:50:03 2025

atomic data from https://github.com/humeniuka/becke_multicenter_integration/blob/main/becke/AtomicData.py

"""

atom_names = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i", "xe","cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn",
"fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr",
"rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "uub", "uut", "uuq", "uup", "uuh"]
bohr_to_angs = 0.529177249
hartree_to_eV = 27.211396132
hartree_to_nm = 45.563352527    # lambda (in nm) = hartree_to_nm / (energy in Hartree)
hartree_to_wavenumbers = 219474.63 # E(in cm^-1) = E(in Hartree) * hartree_to_wavenumbers
hartree_to_kcalmol = 627.509469
autime2fs = 0.02418884326505
kBoltzmann = 3.1668114 * 1.0e-6 # in hartree/Kelvin
aumass2amu = 1.0/1822.888486192  # convert masses from atomic units to amu
ebohr_to_debye = 1.0/0.393430307 # 1 Debye = 0.393430307 e*a0
speed_of_light = 137.035999139   # speed of light in atomic units, inverse of fine structure constant
                                 # c = 1/alpha

# standard ambient temperature and pressure in a.u.
satp_temperature = 298.15  # Kelvin
satp_pressure = 3.3989296754502665e-09 # 100 kPa in Hartree/(bohr)^3
atm_pressure = 1.01325 * satp_pressure # 1 atmosphere in Hartree/(bohr)^3

avogadro_constant = 6.022140857e23

# conversion factor for going from  bohr^2 to megabarn (Mb)
# 1 Mb = 10^(-18) cm^2 = 10^(-18) * (10^-2)^2 m^2 = 10^(-22) m^2
# 1 bohr = 0.529 * 10^(-10) m
# 1 bohr^2 = (0.529)^2 10^(-20) m^2 = (0.529)^2 * 100 * Mb
bohr2_to_megabarn = bohr_to_angs**2 * 100.0

import re
def atomic_number(atom):
    try:
        # if atom is a number it represents a point charge
        atno = float(atom)
    except ValueError:
        # remove integers  meant to label inequivalent atoms of the same element, C12 -> C
        # and +/- signs
        elem = re.sub('-vanish|-appear', '', atom.lower())
        atno = atom_names.index(elem) + 1
    return atno

def element_name(Z):
    """translate atomic number Z into element name"""
    return atom_names[Z-1]

def nuclear_charge(Z):
    """
    positive charge of nuclei in units of -e as a function of atomic number
    """
    return Z

def atomlist2masses(atomlist):
    """
    assign masses to atoms. The mass is repeated for each coordinate
    """
    masses = []
    for (Zi, posi) in atomlist:
        mi = atom_masses[atom_names[Zi-1]]
        masses += [mi, mi, mi]
    return masses

# atomic radii taken from Cordero el.al.,"Covalent radii revisited", Dalton Trans., 2008, 2832-2838
# in bohr, in the paper the values are given in Angstrom with 2 decimals
covalent_radii = {k : v/bohr_to_angs for (k,v) in
                  {
    "h" : 0.31, "he": 0.28, "li": 1.28, "be": 0.96, "b" : 0.84,
    "c" : 0.76, "n" : 0.71, "o" : 0.66, "f" : 0.57, "ne": 0.58,
    "na": 1.66, "mg": 1.41, "al": 1.21, "si": 1.11, "p" : 1.07,
    "s" : 1.05, "cl": 1.02, "ar": 1.06, "k" : 2.03, "ca": 1.76,
    "sc": 1.70, "ti": 1.60, "v" : 1.53, "cr": 1.39, "mn": 1.39,
    "fe": 1.32, "co": 1.26, "ni": 1.24, "cu": 1.32, "zn": 1.22,
    "ga": 1.22, "ge": 1.20, "as": 1.19, "se": 1.20, "br": 1.20,
    "kr": 1.16, "rb": 2.20, "sr": 1.95, "y" : 1.90, "zr": 1.75,
    "nb": 1.64, "mo": 1.54, "tc": 1.47, "ru": 1.46, "rh": 1.42,
    "pd": 1.39, "ag": 1.45, "cd": 1.44, "in": 1.42, "sn": 1.39,
    "sb": 1.39, "te": 1.38, "i" : 1.39, "xe": 1.40, "cs": 2.44,
    "ba": 2.15, "la": 2.07, "ce": 2.04, "pr": 2.03, "nd": 2.01,
    "pm": 1.99, "sm": 1.98, "eu": 1.98, "gd": 1.96, "tb": 1.94,
    "dy": 1.92, "ho": 1.92, "er": 1.89, "tm": 1.90, "yb": 1.87,
    "lu": 1.87, "hf": 1.75, "ta": 1.70, "w" : 1.62, "re": 1.51,
    "os": 1.44, "ir": 1.41, "pt": 1.36, "au": 1.36, "hg": 1.32,
    "tl": 1.45, "pb": 1.46, "bi": 1.48, "po": 1.40, "at": 1.50,
    "rn": 1.50, "fr": 2.60, "ra": 2.21, "ac": 2.15, "th": 2.06,
    "pa": 2.00, "u" : 1.96, "np": 1.90, "pu": 1.87, "am": 1.80,
    "cm": 1.69
                  }.items()}

# Slater radii (converted to bohr) taken from table I of
# J. Slater, "Atomic Radii in Crystals", J.Chem.Phys. volume 41, number 10, 1964, pages 3199-3204
slater_radii = {k : v/bohr_to_angs for (k,v) in
                {
                    'h': 0.25,
# There is not radius for helium in Slater's article. Instead we take
# the radius for helium from Fig. 3 of
#   E. Clementi, D. Raimondi, W. Reinhard,
#   "Atomic Screening Constants from SCF Functions. II. Atoms with 37 to 86 Electrons",
#   J.Chem.Phys., 47, pages 1300-1307 (1967).
                    'he': 0.31,
                    'li': 1.45, 'be': 1.05, 'b': 0.85, 'c': 0.70, 'n': 0.65, 'o': 0.60, 'f': 0.50,
                    'na': 1.80, 'mg': 1.50, 'al': 1.25, 'si': 1.10, 'p': 1.00, 's': 1.00, 'cl': 1.00,
                    'k': 2.20, 'ca': 1.80, 'sc': 1.60, 'ti': 1.40, 'v': 1.35, 'cr': 1.40, 'mn': 1.40, 'fe': 1.40, 'co': 1.35, 'ni': 1.35, 'cu': 1.35, 'zn': 1.35, 'ga': 1.30, 'ge': 1.25, 'as': 1.15, 'se': 1.15, 'br': 1.15,
                    'rb': 2.35, 'sr': 2.00, 'y': 1.80, 'zr': 1.55, 'nb': 1.45, 'mo': 1.45, 'tc': 1.35, 'ru': 1.30, 'rh': 1.35, 'pd': 1.40, 'ag': 1.60, 'cd': 1.55, 'in': 1.55, 'sn': 1.45, 'sb': 1.45, 'te': 1.40, 'i': 1.40,
                    'cs': 2.60, 'ba': 2.15, 'la': 1.95, 'ce': 1.85, 'pr': 1.85, 'nd': 1.85, 'pm': 1.85, 'sm': 1.85, 'eu': 1.85, 'gd': 1.80, 'tb': 1.75, 'dy': 1.75, 'ho': 1.75, 'er': 1.75, 'tu': 1.75, 'yb': 1.75, 'lu': 1.75, 'hf': 1.55, 'ta': 1.45, 'w': 1.35, 're': 1.35, 'os': 1.30, 'ir': 1.35, 'pt': 1.35, 'au': 1.35, 'hg': 1.50, 'tl': 1.90, 'bi': 1.60, 'po': 1.90,
                    'ra': 2.15, 'ac': 1.95, 'th': 1.80, 'pa': 1.80, 'u': 1.85, 'np': 1.75, 'pu': 1.75, 'am': 1.75
                    }.items()}

# van der Waals radii taken from table 12 in
#     Mantina et. al.
#     "van der Waals Radii for the Whole Main Group"
#      J. Phys. Chem. A, Vol. 113, No. 19, 2009'
# and for the metals Ni,Cu, Zn, Ag, Au from table XIV in
#     Bondi
#     "van der Waals Volumes and Radii"
#     J. Phys. Chem., 1964, 68 (3), pp 441-451
#
# vdW radii (converted to bohr)
vdw_radii = {k: v/bohr_to_angs for (k,v) in
                 {'h': 1.10,                                                                     'he': 1.40,
                  'li': 1.81, 'be': 1.53, 'b': 1.92, 'c': 1.70, 'n': 1.55, 'o': 1.52, 'f': 1.47, 'ne': 1.54,
                  'na': 2.27, 'mg': 1.73, 'al':1.84, 'si':2.10, 'p': 1.80, 's': 1.80, 'cl':1.75, 'ar': 1.88,
                  'k': 2.75,  'ca': 2.31, 'ga':1.87, 'ge':2.11, 'as':1.85, 'se':1.90, 'br':1.83, 'kr': 2.02,
                  'rb':3.03,  'sr': 2.49, 'in':1.93, 'sn':2.17, 'sb':2.06, 'te':2.06, 'i': 1.98, 'xe': 2.16,
                  'cs':3.43,  'ba': 2.68, 'tl':1.96, 'pb':2.02, 'bi':2.07, 'po':1.97, 'at':2.02, 'rn': 2.20,
                  'fr':3.48,  'ra': 2.83,
                  'ni': 1.63, 'cu': 1.4, 'zn': 1.39, 'ag':1.72, 'au':1.66,
                 }.items()}

# ionization energies taken from NIST http://physics.nist.gov/PhysRefData/IonEnergy/tblNew.html in eV (with individual citations for each element)
ionization_energies_eV = {"h": 13.5984, "he": 24.5874, "li": 5.3917, "be": 9.3227, "b": 8.2980, "c": 11.2603, "n": 14.5341, "o": 13.6181, "f": 17.4228, "ne": 21.5645, "na": 5.1391, "mg": 7.6462, "al": 5.9858, "si": 8.1517, "p": 10.4867, "s": 10.3600, "cl": 12.9676, "ar": 15.7596, "k": 4.3407, "ca": 6.1132, "sc": 6.5615, "ti": 6.8281, "v": 6.7462, "cr": 6.7665, "mn": 7.4340, "fe": 7.9024, "co": 7.8810, "ni": 7.6399, "cu": 7.7264, "zn": 9.3942, "ga": 5.9993, "ge": 7.8994, "as": 9.7886, "se": 9.7524, "br": 11.8138, "kr": 13.9996, "rb": 4.1771, "sr": 5.6949, "y": 6.2173, "zr": 6.6339, "nb": 6.7589, "mo": 7.0924, "tc": 7.28, "ru": 7.3605, "rh": 7.4589, "pd": 8.3369, "ag": 7.5762, "cd": 8.9938, "in": 5.7864, "sn": 7.3439, "sb": 8.6084, "te": 9.0096, "i": 10.4513, "xe": 12.1298, "cs": 3.8939, "ba": 5.2117, "la": 5.5769, "ce": 5.5387, "pr": 5.473, "nd": 5.5250, "pm": 5.582, "sm": 5.6437, "eu": 5.6704, "gd": 6.1498, "tb": 5.8638, "dy": 5.9389, "ho": 6.0215, "er": 6.1077, "tm": 6.1843, "yb": 6.2542, "lu": 5.4259, "hf": 6.8251, "ta": 7.5496, "w": 7.8640, "re": 7.8335, "os": 8.4382, "ir": 8.9670, "pt": 8.9588, "au": 9.2255, "hg": 10.4375, "tl": 6.1082, "pb": 7.4167, "bi": 7.2855, "po": 8.414, "at": None, "rn": 10.7485, "fr": 4.0727, "ra": 5.2784, "ac": 5.3807, "th": 6.3067, "pa": 5.89, "u": 6.1939, "np": 6.2657, "pu": 6.0260, "am": 5.9738, "cm": 5.9914, "bk": 6.1979, "cf": 6.2817, "es": 6.3676, "fm": 6.50, "md": 6.58, "no": 6.65}

# electron affinities EA = Etot(A) - Etot(A-) in eV taken from "Chem. Rev. 2002, 102, 231-282", section "Experimental Photoelectron EA" Table 10
electron_affinities_eV = {"h": 0.75419, "li": 0.618, "b": 0.279723, "c": 1.262119, "o": 1.46111, "f": 3.401290, "na": 0.547930, "al": 0.43283, "si": 1.389521, "p": 0.7464, "s": 2.077103, "cl": 3.612724, \
                          "sc": 0.189, \
                          "ti": 0.080, \
                          "br": 3.363588, \
                          "ru": 1.04638, \
                          "ag": 1.30447, \
                          "i": 3.059038, \
                          "au": 2.30863  }

# static dipole polarizabilities of atoms (in atomic units) taken from
# "Table of experimental and calculated static dipole polarizabilities for the electronic ground states of the neutral elements" by Peter Schwerdtfeger
# Where available the experimental values are chosen otherwise the first theoretical value
static_polarizabilities_au = {"h": 4.5, "he": 1.38, "li": 164.0, "be": 37.76, "b": 20.5, "c": 11.0, "n": 7.6, "o": 6.04, "f": 3.76, "ne": 2.68, "na": 162.7, "mg": 71.7, "al": 46.0, "si": 37.6, "p": 24.7, "s": 19.6, "cl": 14.7, "ar": 11.07, "k": 290.6, "ca": 169.0, "sc": 120.0, "ti": 99.0, "v": 84.0, "cr": 78.0, "mn": 63.0, "fe": 57.0, "co": 51.0, "ni": 46.0, "cu": 53.4, "zn": 38.8, "ga": 54.9, "ge": 41.0, "as": 29.1, "se": 26.24, "br": 21.9, "kr": 17.08, "rb": 318.8, "sr": 186.0, "y": 153.0, "zr": 121.0, "nb": 106.0, "mo": 72.5, "tc": 77.0, "ru": 65.0, "rh": 58.0, "pd": 32.0, "ag": 52.2, "cd": 49.65, "in": 68.7, "sn": 42.4, "sb": 45.0, "te": 37.0, "i": 35.1, "xe": 27.8, "cs": 401.0, "ba": 268.0, "la": 210.0, "ce": 200.0, "pr": 190.0, "nd": 212.0, "pm": 203.0, "sm": 194.0, "eu": 187.0, "gd": 159.0, "tb": 172.0, "dy": 165.0, "ho": 159.0, "er": 153.0, "tm": 147.0,
"au": 35.1}

# occupation numbers of valence orbitals
# which are used to assign the correct occupation to orbitals loaded from hotbit .elm files
valence_occ_numbers = {"h": {"1s": 1}, "he": {"1s": 2}, \
"li": {"2s": 1}, "be": {"2s": 2}, "b": {"2s": 2, "2p": 1}, "c": {"2s": 2, "2p": 2}, "n": {"2s": 2, "2p": 3}, "o": {"2s": 2, "2p": 4}, "f": {"2s": 2, "2p": 5}, "ne": {"2s": 2, "2p": 6}, \
"na": {"3s": 1}, "mg": {"3s": 2}, "al": {"3s": 2, "3p": 1}, "si": {"3s": 2, "3p": 2}, "p": {"3s": 2, "3p": 3}, "s": {"3s": 2, "3p": 4}, "cl": {"3s": 2, "3p": 5}, "ar": {"3s": 2, "3p": 6}, \
    "sc": {"3d": 1, "4s": 2}, "ti": {"3d": 2, "4s": 2}, "zn": {"3d": 10, "4s": 2}, "br": {"4s": 2, "4p": 5}, \
"ru": {"4d": 7, "5s": 1}, \
"i": {"5s": 2, "5p": 5}, \
"au": {"4f": 14, "5d": 10, "6s": 1}\
}

# number of valence electrons
valence_electrons = dict([(at, sum(valence_occ_numbers[at].values())) for at in valence_occ_numbers.keys()])

# C6 parameters (in J nm^6 mol^-1) and van der Waals radii R0 (in Angstrom) for Grimme's semiempirical dispersion correction
# taken from "Semiempirical GGA-Type Density Functional Constructred with a Long-Range Dispersion Correction"
# Grimme,S. J. Comput. Chem. 2006 27 1787-1799
#
Grimme_C6 = {"h": 0.14, "he": 0.08, "li": 1.61, "be": 1.61, "b": 3.13, "c": 1.75, "n": 1.23, "o": 0.70, "f": 0.75, "ne": 0.63, "na": 5.71, "mg": 5.71, "al": 10.79, "si": 9.23, "p": 7.84, "s": 5.57, "cl": 5.07, "ar": 4.61, "k": 10.80, "ca": 10.80,
             "ti": 10.80,
             "zn": 10.80,   "br": 12.47,
             "ru": 24.67, "ag": 24.67,   "i": 31.50}
Grimme_R0 = {"h": 1.001, "he": 1.012, "li": 0.825, "be": 1.408, "c": 1.452, "n": 1.397, "o": 1.342, "f": 1.287, "ne": 1.243, "na": 1.144, "mg": 1.364, "al": 1.639, "si": 1.716, "p": 1.705, "s": 1.683, "cl": 1.639, "ar": 1.595, "k": 1.485, "ca": 1.474,
             "ti": 1.562,
             "zn": 1.562,   "br": 1.749,
             "ru": 1.639, "ag": 1.639,   "i": 1.892}