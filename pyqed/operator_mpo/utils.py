# -*- coding: utf-8 -*-
# Modified from renormalizer.utils.utils, renormalizer.utils.quantitiy and renormalizer.utils.constant

#### Utilities, including constant reference table, a Quantity Class that handles convertion from other units to a.u.
#### Also contain a cached_property class, similar to functools.cached_property, allowing calculating called value only the first time and call the value for the other times

#### Example use of Quantity Class:
# q1 = Quantity(1.0, "eV")
# q2 = Quantity(0.05, "a.u.")
# result = q1 + q2
# print(result.as_au())
## Will print out result in a.u.

from __future__ import division
import math
import logging
import numpy as np
from typing import List, Union
import scipy
from scipy.constants import physical_constants as c

# Constants table from scipy
##################### energy ##################
# 1 a.u. = au2ev eV
au2ev = c["Hartree energy in eV"][0]
ev2au = 1./au2ev

# 1 cm^-1 = cm2au a.u.
cm2au = (
    1.0e2
    * c["inverse meter-hertz relationship"][0]
    / c["hartree-hertz relationship"][0]
)
au2cm = 1./ cm2au

# 1 cm^-1 = cm2ev eV
cm2ev = cm2au * au2ev
ev2cm = 1./cm2ev

##################### time ##################
# 1 fs = fs2au a.u
fs2au = 1.0e-15 / c["atomic unit of time"][0]
au2fs = 1. / fs2au

ps2au = fs2au * 1000
au2ps = 1. / ps2au

################ temperature energy ###############
K2au = c["kelvin-hartree relationship"][0]
au2K = c["hartree-kelvin relationship"][0]

###################  mass #################
# atomic mass unit
amu2au = c["atomic mass constant"][0] / c["atomic unit of mass"][0]
angstrom2au = 1e-10 / c["atomic unit of length"][0]
au2amu = 1 / amu2au
au2angstrom = 1 / angstrom2au

################# wavelength energy #############
# nm to au
def nm2au(l):
    return 1.0e7 / l * cm2au

def au2nm(e):
    return 1.0e7 / (e / cm2au)

################### mobility ########################
# 1 cm^2/V s = mobility2au a.u.
# mobility2au = 23.505175500558234
mobility2au = au2ev * c["atomic unit of time"][0] / (c["atomic unit of length"][0] * 100) ** 2

################# dipole moment #################
debye2au = 0.393456
au2debye = 1 / debye2au

################ length ####################
au2m = c["atomic unit of length"][0]
m2au = 1/au2m


logger = logging.getLogger(__name__)

au_ratio_dict = {
    "meV": au2ev * 1e3,
    "eV": au2ev,
    "cm^{-1}": 1 / cm2au,
    "cm-1": 1 / cm2au,
    "K": au2K,
    "a.u.": 1,
    "au": 1,
    "fs": au2fs,
    "ps": au2ps
}

au_ratio_dict.update({k.lower(): v for k, v in au_ratio_dict.items()})

allowed_units = set(au_ratio_dict.keys())


def convert_to_au(num, unit):
    """Convert numeric value(s) to atomic units, handling both scalars and arrays."""
    if isinstance(num, (list, tuple, np.ndarray)):
        return np.array(num) / au_ratio_dict[unit]
    else:
        return num / au_ratio_dict[unit]


class Quantity:
    """
    Unified Quantity class using duck typing and property delegation.
    Maintains full backward compatibility while supporting both scalar and array values.
    """

    def __init__(self, value, unit="a.u."):
        if unit not in allowed_units:
            raise ValueError(f"Unit not in {allowed_units}, got {unit}.")

        # Store as numpy array for unified handling
        self._data = np.array(value, dtype=float, ndmin=1)  # Ensure at least 1D
        self.unit = unit

        # Temperature warning (only for scalar temperatures)
        if (
            self._data.size == 1
            and self._data[0] < 0.1
            and self._data[0] != 0
            and unit.lower() in ["k", "kelvin"]
        ):
            logger.warning("Temperature too low and might cause numerical errors")

    @property
    def value(self):
        """
        Scalar access - returns single value for 1-element arrays,
        raises error for multi-element arrays.
        """
        if self._data.size == 1:
            return float(self._data[0])
        else:
            raise AttributeError(
                f"Cannot access .value on multi-element Quantity. "
                f"Use .values or indexing. Shape: {self._data.shape}"
            )

    @property
    def values(self):
        """Array access - always returns numpy array."""
        return self._data

    @property
    def shape(self):
        """Shape property for numpy compatibility."""
        return self._data.shape

    @property
    def size(self):
        """Size property for numpy compatibility."""
        return self._data.size

    def as_au(self):
        """Convert to atomic units."""
        if self._data.size == 1:
            return convert_to_au(float(self._data[0]), self.unit)
        else:
            return convert_to_au(self._data, self.unit)

    def as_unit(self, unit):
        """Convert to specified unit."""
        au_value = self.as_au()
        new_value = au_value * au_ratio_dict[unit]
        return Quantity(new_value, unit)

    def to_beta(self):
        """Convert temperature to beta (1/kT)."""
        au_value = self.as_au()

        if self._data.size == 1:
            if au_value == 0:
                return math.inf
            return 1.0 / au_value
        else:
            # For arrays, return list of beta values
            result = []
            for val in au_value if isinstance(au_value, np.ndarray) else [au_value]:
                if val == 0:
                    result.append(math.inf)
                else:
                    result.append(1.0 / val)
            return result

    # Compatiblity with numpy methods
    __array_priority__ = 1000  # Use higher priority to override numpy

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        new_inputs = []
        for inp in inputs:
            if isinstance(inp, Quantity):
                new_inputs.append(inp.as_au())
            else:
                new_inputs.append(inp)

        result = ufunc(*new_inputs, **kwargs)

        return Quantity(result)

    def __array_function__(self, func, types, args, kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Quantity):
                new_args.append(arg.as_au())
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)

        return Quantity(result)

    # Arithmetic operations
    def __neg__(self):
        return Quantity(-self._data, self.unit)

    def __add__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Can only add Quantity objects")

        return Quantity(self.as_au() + other.as_au())

    def __sub__(self, other):
        if not isinstance(other, Quantity):
            raise TypeError("Can only subtract Quantity objects")

        return Quantity(self.as_au() - other.as_au())

    def __mul__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Multiplication between Quantity objects is not supported")

        return Quantity(self.as_au() * other)

    def __rmul__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Multiplication between Quantity objects is not supported")

        return Quantity(other * self.as_au())

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Division between Quantity objects is not supported")

        return Quantity(self.as_au() / other)

    def __eq__(self, other):
        if hasattr(other, "as_au"):
            return np.array_equal(self.as_au(), other.as_au())
        elif other == 0:
            return np.all(self._data == 0)
        else:
            raise TypeError(f"Quantity can only compare with Quantity or 0, not {type(other)}")

    def __ne__(self, other):
        return not self == other

    # Array-like operations
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        result = self._data[index]
        # Return scalar Quantity for single elements, array Quantity for slices
        if isinstance(result, np.ndarray) and result.ndim > 0:
            return Quantity(result, self.unit)
        else:
            return Quantity(float(result), self.unit)

    def __iter__(self):
        for value in self._data:
            yield Quantity(float(value), self.unit)

    # Numpy compatibility
    def __array__(self):
        """Allow numpy to convert this to array."""
        return self._data

    def __str__(self):
        if self._data.size == 1:
            return f"{float(self._data[0])} {self.unit}"
        else:
            return f"{list(self._data)} {self.unit}"

    def __repr__(self):
        if self._data.size == 1:
            return f"Quantity({float(self._data[0])}, '{self.unit}')"
        else:
            return f"Quantity({list(self._data)}, '{self.unit}')"

    # TODO: magic methods such as `__lt__` and so on


def parse_quantity_str(quantity_str: str):
    """
    Parse a string containing a quantity or list of quantities with units.

    Args:
        quantity_str: String in format "value unit" or "value1,value2,value3 unit"

    Returns:
        Quantity object

    Raises:
        ValueError: If input is invalid or cannot be parsed
    """
    if not quantity_str or not isinstance(quantity_str, str):
        raise ValueError("Input must be a non-empty string")

    cleaned_str = quantity_str.strip()

    if not cleaned_str:
        raise ValueError("Input string cannot be empty")

    # Try to find a known unit at the end of the string
    found_unit = None
    for unit in sorted(allowed_units, key=len, reverse=True):
        if cleaned_str.endswith(unit):
            unit_start = len(cleaned_str) - len(unit)
            if unit_start == 0:
                continue

            if unit_start > 0 and cleaned_str[unit_start-1].isspace():
                found_unit = unit
                value_str = cleaned_str[:unit_start].strip()
                break
            elif unit_start > 0 and not cleaned_str[unit_start-1].isspace():
                continue

    if found_unit:
        unit_str = found_unit
    else:
        # Fallback to last space method
        last_space_index = cleaned_str.rfind(' ')

        if last_space_index == -1:
            value_str = cleaned_str
            unit_str = ""
        else:
            value_str = cleaned_str[:last_space_index].strip()
            unit_str = cleaned_str[last_space_index:].strip()

            if unit_str and unit_str not in allowed_units:
                value_str = cleaned_str
                unit_str = ""

    # Parse values
    if ',' in value_str:
        try:
            values = [float(val.strip()) for val in value_str.split(',') if val.strip()]
            return Quantity(values, unit_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric values '{value_str}': {e}")
    else:
        try:
            value = float(value_str)
            return Quantity(value, unit_str)
        except ValueError as e:
            raise ValueError(f"Invalid numeric value '{value_str}': {e}")


# from https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


class cached_property():
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def calc_vn_entropy(p: Union[np.ndarray, List[float]]) -> float:
    # calculate Von Neumann entropy from density matrix eigenvalues (not singular values!)
    p = np.array(p)
    assert np.allclose(p[p<0], 0)
    p = p / p.sum()
    assert np.allclose(p.sum(), 1)
    p = p[0 < p]
    return - (p* np.log(p)).sum()


def calc_vn_entropy_dm(dm: np.ndarray) -> float:
    # calculate Von Neumann entropy from density matrix

    # reshape dm to square matrix
    dim = np.prod(dm.shape[:dm.ndim // 2])
    dm = dm.reshape((dim, dim))
    w, v = scipy.linalg.eigh(dm)
    return calc_vn_entropy(w)
