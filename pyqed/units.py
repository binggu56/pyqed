"""Physical constants and atomic-unit conversions used across PyQED.

Scalar physical constants are pinned to NIST/CODATA 2022. Atomic weights use
the CIAAW 2024 table; elements without a standard weight use the nominal mass
number printed in the IUPAC periodic table dated 4 May 2022. Secondary
conversions are derived from the primitives so aliases remain consistent.
"""

# https://physics.nist.gov/cuu/Constants/index.html
SPEED_OF_LIGHT_M_PER_S = 299792458.0
PLANCK_CONSTANT_J_HZ = 6.62607015e-34
ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
AVOGADRO_CONSTANT = 6.02214076e23
HARTREE_ENERGY_J = 4.3597447222060e-18
ATOMIC_TIME_SECOND = 2.4188843265864e-17
BOHR_RADIUS_METER = 5.29177210544e-11
ATOMIC_UNIT_ELECTRIC_FIELD_V_PER_M = 5.14220675112e11
ATOMIC_UNIT_MAGNETIC_FLUX_DENSITY_T = 2.35051757077e5
ATOMIC_UNIT_ELECTRIC_DIPOLE_C_M = 8.4783536198e-30
ELECTRON_MASS_U = 5.485799090441e-4
PROTON_ELECTRON_MASS_RATIO = 1836.152673426
FINE_STRUCTURE_CONSTANT = 7.2973525643e-3
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878188e-12
VACUUM_IMPEDANCE_OHM = 376.730313412

proton_mass = PROTON_ELECTRON_MASS_RATIO

au2fs = ATOMIC_TIME_SECOND * 1.0e15
au2as = ATOMIC_TIME_SECOND * 1.0e18

au2k = HARTREE_ENERGY_J / BOLTZMANN_CONSTANT_J_PER_K
au2ev = HARTREE_ENERGY_J / ELEMENTARY_CHARGE_C
au2kcalmol = HARTREE_ENERGY_J * AVOGADRO_CONSTANT / 4184.0
au2kjmol = HARTREE_ENERGY_J * AVOGADRO_CONSTANT / 1000.0
kcalmol2au = 1.0 / au2kcalmol
kjmol2au = 1.0 / au2kjmol

au2tesla = ATOMIC_UNIT_MAGNETIC_FLUX_DENSITY_T
tesla = 1.0 / au2tesla

au2kev = au2ev * 1.0e-3
au2mev = au2ev * 1.0e3

au2wn = au2wavenumber = HARTREE_ENERGY_J / (
    PLANCK_CONSTANT_J_HZ * SPEED_OF_LIGHT_M_PER_S * 100.0
)
wavenumber2hartree = wavenum2au = 1.0 / au2wavenumber
ev2wavenumber = ELEMENTARY_CHARGE_C / (
    PLANCK_CONSTANT_J_HZ * SPEED_OF_LIGHT_M_PER_S * 100.0
)

DEBYE_C_M = 1.0e-21 / SPEED_OF_LIGHT_M_PER_S
au2debye = ATOMIC_UNIT_ELECTRIC_DIPOLE_C_M / DEBYE_C_M
debye2au = 1.0 / au2debye
au2amu = ELECTRON_MASS_U
amu_to_au = amu2au = 1.0 / au2amu

BOHR_RADIUS_ANGSTROM = BOHR_RADIUS_METER * 1.0e10
au2angstrom = bohr2angstrom = BOHR_RADIUS_ANGSTROM
angstrom2au = 1.0 / au2angstrom
au2nm = bohr2nanometer = BOHR_RADIUS_METER * 1.0e9

ev2nm = electronvolt2nanometer = (
    PLANCK_CONSTANT_J_HZ * SPEED_OF_LIGHT_M_PER_S / ELEMENTARY_CHARGE_C * 1.0e9
)
hartree2nm = hartree2nanometer = (
    PLANCK_CONSTANT_J_HZ * SPEED_OF_LIGHT_M_PER_S / HARTREE_ENERGY_J * 1.0e9
)

fine_structure = alpha = FINE_STRUCTURE_CONSTANT
eps0 = epsilon_0 = VACUUM_PERMITTIVITY_F_PER_M
c0 = speed_of_light = SPEED_OF_LIGHT_M_PER_S
imp0 = VACUUM_IMPEDANCE_OHM

au2volt_per_meter = ATOMIC_UNIT_ELECTRIC_FIELD_V_PER_M
au2volt_per_angstrom = au2volt_per_meter * 1.0e-10

au2watt_per_meter_squared = (
    0.5 * speed_of_light * epsilon_0 * au2volt_per_meter**2
)
au2watt_per_centimeter_squared = au2watt_per_meter_squared * 1.0e-4
ghz2ev = PLANCK_CONSTANT_J_HZ / ELEMENTARY_CHARGE_C * 1.0e9
ghz2mev = ghz2ev * 1.0e3

# https://ciaaw.org/atomic-weights.htm
# https://iupac.org/what-we-do/periodic-table-of-elements/
atomic_mass = {
    "H": 1.008, "HE": 4.002602, "LI": 6.94, "BE": 9.0121831,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403162, "NE": 20.1797, "NA": 22.98976928, "MG": 24.305,
    "AL": 26.9815384, "SI": 28.085, "P": 30.973761998, "S": 32.06,
    "CL": 35.45, "AR": 39.95, "K": 39.0983, "CA": 40.078,
    "SC": 44.955907, "TI": 47.867, "V": 50.9415, "CR": 51.9961,
    "MN": 54.938043, "FE": 55.845, "CO": 58.933194, "NI": 58.6934,
    "CU": 63.546, "ZN": 65.38, "GA": 69.723, "GE": 72.63,
    "AS": 74.921595, "SE": 78.971, "BR": 79.904, "KR": 83.798,
    "RB": 85.4678, "SR": 87.62, "Y": 88.905838, "ZR": 91.222,
    "NB": 92.90637, "MO": 95.95, "TC": 97.0, "RU": 101.07,
    "RH": 102.90549, "PD": 106.42, "AG": 107.8682, "CD": 112.414,
    "IN": 114.818, "SN": 118.71, "SB": 121.76, "TE": 127.6,
    "I": 126.90447, "XE": 131.293, "CS": 132.90545196, "BA": 137.327,
    "LA": 138.90547, "CE": 140.116, "PR": 140.90766, "ND": 144.242,
    "PM": 145.0, "SM": 150.36, "EU": 151.964, "GD": 157.249,
    "TB": 158.925354, "DY": 162.5, "HO": 164.930329, "ER": 167.259,
    "TM": 168.934219, "YB": 173.045, "LU": 174.96669, "HF": 178.486,
    "TA": 180.94788, "W": 183.84, "RE": 186.207, "OS": 190.23,
    "IR": 192.217, "PT": 195.084, "AU": 196.96657, "HG": 200.592,
    "TL": 204.38, "PB": 207.2, "BI": 208.9804, "PO": 209.0,
    "AT": 210.0, "RN": 222.0, "FR": 223.0, "RA": 226.0,
    "AC": 227.0, "TH": 232.0377, "PA": 231.03588, "U": 238.02891,
    "NP": 237.0, "PU": 244.0, "AM": 243.0, "CM": 247.0,
    "BK": 247.0, "CF": 251.0, "ES": 252.0, "FM": 257.0,
    "MD": 258.0, "NO": 259.0, "LR": 262.0, "RF": 267.0,
    "DB": 268.0, "SG": 269.0, "BH": 270.0, "HS": 269.0,
    "MT": 277.0, "DS": 281.0, "RG": 282.0, "CN": 285.0,
    "NH": 286.0, "FL": 290.0, "MC": 290.0, "LV": 293.0,
    "TS": 294.0, "OG": 294.0,
}

electronvolt = eV = 1.0 / au2ev
wavenumber = 1.0 / au2wavenumber
kelvin = 1.0 / au2k
attosecond = 1.0 / au2as
femtosecond = fs = 1.0 / au2fs
eV_per_angstrom = au2angstrom / au2ev


class AtomicUnits:
    def __init__(self):
        self.ev = au2ev
