import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed import units
from pyqed.units import BOHR_RADIUS_ANGSTROM, au2angstrom, au2nm


def test_bohr_radius_uses_2022_nist_codata_value():
    assert BOHR_RADIUS_ANGSTROM == 0.529177210544
    assert au2angstrom == units.bohr2angstrom == BOHR_RADIUS_ANGSTROM
    assert au2nm == units.bohr2nanometer
    assert au2nm == pytest.approx(0.1 * BOHR_RADIUS_ANGSTROM)


def test_angstrom_geometry_conversion_uses_central_bohr_radius():
    distance_bohr = 1.4
    distance_angstrom = distance_bohr * BOHR_RADIUS_ANGSTROM
    mol = Molecule(
        atom=f"H 0 0 0; H 0 0 {distance_angstrom!r}",
        unit="angstrom",
    )

    np.testing.assert_allclose(
        mol.atom_coords(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, distance_bohr]],
        rtol=0.0,
        atol=1.0e-15,
    )


@pytest.mark.parametrize(
    ("name", "nist_2022"),
    [
        ("proton_mass", 1836.152673426),
        ("au2fs", 0.024188843265864),
        ("au2as", 24.188843265864),
        ("au2k", 315775.02480398706),
        ("au2ev", 27.211386245981167),
        ("au2kcalmol", 627.5094740628974),
        ("au2kjmol", 2625.499639479163),
        ("au2tesla", 235051.757077),
        ("au2wavenumber", 219474.63136314112),
        ("wavenumber2hartree", 4.556335252913159e-6),
        ("ev2wavenumber", 8065.543937349212),
        ("au2debye", 2.54174647147304),
        ("au2amu", 5.485799090441e-4),
        ("amu2au", 1822.8884862781415),
        ("au2nm", 0.0529177210544),
        ("ev2nm", 1239.8419843320025),
        ("hartree2nm", 45.56335252913159),
        ("fine_structure", 7.2973525643e-3),
        ("eps0", 8.8541878188e-12),
        ("speed_of_light", 299792458.0),
        ("imp0", 376.730313412),
        ("au2volt_per_meter", 5.14220675112e11),
        ("au2watt_per_centimeter_squared", 3.5094455277316284e16),
        ("ghz2ev", 4.135667696923859e-6),
        ("eV_per_angstrom", 0.019446903798300753),
    ],
)
def test_scalar_conversions_use_pinned_2022_codata(name, nist_2022):
    assert getattr(units, name) == pytest.approx(nist_2022, rel=2.0e-15)


def test_secondary_unit_conversions_are_derived_consistently():
    assert units.au2wn == units.au2wavenumber
    assert units.wavenum2au == 1.0 / units.au2wavenumber
    assert units.au2kev == units.au2ev * 1.0e-3
    assert units.au2mev == units.au2ev * 1.0e3
    assert units.electronvolt == units.eV == 1.0 / units.au2ev
    assert units.kelvin == 1.0 / units.au2k
    assert units.debye2au == 1.0 / units.au2debye
    assert units.angstrom2au == 1.0 / units.au2angstrom
    assert units.AtomicUnits().ev == units.au2ev


def test_atomic_masses_cover_current_symbols_and_fixed_entries():
    assert len(units.atomic_mass) == 118
    assert units.atomic_mass["KR"] == 83.798
    assert units.atomic_mass["RB"] == 85.4678
    assert units.atomic_mass["TE"] == 127.6
    assert units.atomic_mass["ZR"] == 91.222
    assert units.atomic_mass["GD"] == 157.249
    assert units.atomic_mass["LU"] == 174.96669
    assert units.atomic_mass["CF"] == 251.0
    assert units.atomic_mass["TC"] == 97.0
    assert units.atomic_mass["RF"] == 267.0
    assert units.atomic_mass["HS"] == 269.0
    assert units.atomic_mass["MT"] == 277.0
    assert units.atomic_mass["RG"] == 282.0
    assert units.atomic_mass["FL"] == 290.0
    assert units.atomic_mass["MC"] == 290.0
    assert "CT" not in units.atomic_mass

    from pyqed.qchem import atomic_data

    assert len(atomic_data.atom_names) == 118
    assert atomic_data.atom_names[-7:] == ["cn", "nh", "fl", "mc", "lv", "ts", "og"]
    assert atomic_data.atom_masses["cf"] == pytest.approx(
        units.atomic_mass["CF"] * units.amu2au
    )


def test_downstream_constant_aliases_share_the_central_values():
    from pyqed.mps.autompo import utils as autompo_units
    from pyqed.qchem import atomic_data

    assert atomic_data.hartree_to_eV == units.au2ev
    assert atomic_data.hartree_to_nm == units.hartree2nm
    assert atomic_data.hartree_to_wavenumbers == units.au2wavenumber
    assert atomic_data.avogadro_constant == units.AVOGADRO_CONSTANT
    assert autompo_units.ev2au == units.electronvolt
    assert autompo_units.cm2au == units.wavenumber2hartree
    assert autompo_units.angstrom2au == units.angstrom2au
