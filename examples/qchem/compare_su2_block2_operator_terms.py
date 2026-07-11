#!/usr/bin/env python3
"""Compare PyQED repeated-index SU(2) ERI coefficients with block2.

This diagnostic intentionally stops before DMRG.  block2's
``GeneralFCIDUMP.finalize()`` rewrites the raw SU(2) QC expression

    1/2 v[pqrs] a^dag_p a^dag_r a_s a_q

into its adjusted rank-coupled operator strings.  Those adjusted strings are
the most direct oracle for PyQED's fully reduced repeated-index Hamiltonian
builder.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyqed.mps.nonabelian.models import (  # noqa: E402
    _FULLY_REDUCED_EXCHANGE_RECOUPLING,
    _spinfree_exchange_recoupling_coefficients,
)


DEFAULT_PATTERNS = (
    (0, 1, 1, 2),
    (0, 1, 2, 1),
    (0, 1, 0, 2),
    (0, 1, 2, 0),
    (0, 0, 1, 2),
    (0, 1, 0, 1),
)


def block2_adjusted_terms(pattern):
    """Return block2 adjusted SU(2) terms for one unit ERI element."""

    nsites = max(pattern) + 1
    h2 = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    h2[pattern] = 1.0
    with tempfile.TemporaryDirectory(prefix="block2_su2_terms_") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.initialize_system(
            n_sites=nsites,
            n_elec=2,
            spin=0,
            orb_sym=[1] * nsites,
        )
        builder = driver.expr_builder()
        builder.add_sum_term(
            "((C+(C+D)0)1+D)0",
            h2,
            cutoff=1.0e-20,
            perm=[0, 2, 3, 1],
        )
        text = str(builder.finalize())

    terms = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"\s*TERM\s+(.*?)\s+::", line)
        if match is None:
            continue
        coeff = float(lines[index + 1].rsplit("=", 1)[1])
        terms.append((match.group(1), coeff))
    return terms


def pyqed_exchange_terms(pattern):
    """Return PyQED exchange coefficients relative to a unit ERI element."""

    if not (pattern[0] == pattern[3] or pattern[1] == pattern[2]):
        return None
    # SpatialSpinFreeERIBuilder applies the conventional 1/2 two-body factor
    # before the fully reduced exchange recoupling path.
    half_eri = 0.5
    terms = []
    for coeff, ranks in _spinfree_exchange_recoupling_coefficients(pattern):
        terms.append(
            (
                tuple(irrep.two_j for irrep in ranks),
                half_eri * _FULLY_REDUCED_EXCHANGE_RECOUPLING * coeff,
            )
        )
    return terms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        help="ERI patterns as comma-separated p,q,r,s entries. Defaults to representative repeated-index cases.",
    )
    args = parser.parse_args(argv)
    patterns = tuple(
        tuple(int(part) for part in item.split(","))
        for item in args.patterns
    ) or DEFAULT_PATTERNS

    for pattern in patterns:
        if len(pattern) != 4:
            raise SystemExit(f"Invalid pattern {pattern!r}; expected p,q,r,s.")
        print(f"\npattern {pattern}")
        print("  block2 adjusted SU2 terms:")
        for expr, coeff in block2_adjusted_terms(pattern):
            print(f"    {expr:32s} {coeff:+.16f}")
        pyqed_terms = pyqed_exchange_terms(pattern)
        if pyqed_terms is not None:
            print("  PyQED current exchange path:")
            for ranks, coeff in pyqed_terms:
                print(f"    ranks={ranks!s:18s} {coeff:+.16f}")
        else:
            print("  PyQED current exchange path: not an exchange repeated-index case")


if __name__ == "__main__":
    main()
