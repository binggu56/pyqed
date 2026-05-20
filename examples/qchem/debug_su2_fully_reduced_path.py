#!/usr/bin/env python3
"""Matrix-level diagnostics for the fully reduced SU(2) MPO path.

This script deliberately stops before DMRG.  It compares the native fully
reduced PyQED MPO contraction against the exact spin-orbital matrix projected
onto the same reduced path basis.  That makes SU(2) convention bugs visible as
small lists of reduced-basis matrix elements instead of slow sweep failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_reference_helpers():
    """Load the dense-reference helpers from the non-Abelian model tests."""

    helper_path = REPO_ROOT / "tests" / "test_nonabelian_models.py"
    spec = importlib.util.spec_from_file_location("_pyqed_su2_model_refs", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference helpers from {helper_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_state(index, path_spec):
    labels, bonds = path_spec
    bond_tags = tuple((int(bond.charge), int(bond.irrep.two_j)) for bond in bonds)
    return f"{index:3d} labels={labels} bonds={bond_tags}"


def _ratio_bucket(actual, expected, tol):
    if abs(expected) <= tol:
        return "spurious" if abs(actual) > tol else "zero"
    ratio = actual / expected
    if abs(ratio.imag) <= tol:
        ratio = float(np.real(ratio))
        nearest = round(ratio)
        if abs(ratio - nearest) <= 1.0e-10:
            return f"{nearest:g}x"
    return f"{ratio:.6g}"


def _compare_matrix(path_specs, basis_states, dense_vectors, mpo, expected_operator, *, tol, limit):
    mismatches = []
    buckets = Counter()
    max_abs = 0.0
    helpers = _load_reference_helpers()
    contract = helpers._contract_chain_transition

    for bra_index, bra_state in enumerate(basis_states):
        for ket_index, ket_state in enumerate(basis_states):
            expected = np.vdot(dense_vectors[bra_index], expected_operator @ dense_vectors[ket_index])
            actual = contract(bra_state, mpo, ket_state)
            delta = actual - expected
            max_abs = max(max_abs, float(abs(delta)))
            if abs(delta) > tol:
                buckets[_ratio_bucket(actual, expected, tol)] += 1
                if len(mismatches) < limit:
                    mismatches.append((bra_index, ket_index, actual, expected, delta))

    print(f"basis size: {len(path_specs)}")
    print(f"max |PyQED - exact|: {max_abs:.6e}")
    if buckets:
        print("mismatch buckets:", ", ".join(f"{key}={value}" for key, value in sorted(buckets.items())))
    else:
        print("mismatch buckets: none")
    for bra_index, ket_index, actual, expected, delta in mismatches:
        print()
        print("bra", _format_state(bra_index, path_specs[bra_index]))
        print("ket", _format_state(ket_index, path_specs[ket_index]))
        print(f"  PyQED={actual:+.16g} exact={expected:+.16g} delta={delta:+.6g}")
    return max_abs


def _one_body_channel_mpos(args, phys_leg):
    from pyqed.mps.nonabelian import AutoMPO
    from pyqed.mps.nonabelian.models import (
        _fully_reduced_double_transition_phase,
        _split_spatial_fermion_annihilation_channels,
    )
    from pyqed.mps.nonabelian.operators import (
        spatial_parity,
        time_reversed_reduced_operator,
    )
    from pyqed.mps.su2 import SU2Irrep

    parity = spatial_parity(phys_leg)
    double_phase = _fully_reduced_double_transition_phase(phys_leg, dtype=float)
    annihilate_empty_single, annihilate_single_double = (
        _split_spatial_fermion_annihilation_channels(phys_leg, dtype=float)
    )
    create_empty_single = annihilate_empty_single.adjoint()
    create_single_double = annihilate_single_double.adjoint()
    left_site = min(args.create, args.annihilate)
    right_site = max(args.create, args.annihilate)
    middle = {site: parity for site in range(left_site + 1, right_site)}

    if args.create < args.annihilate:
        channel_terms = (
            ("c01/a01", create_empty_single, annihilate_empty_single, -np.sqrt(2.0)),
            ("c01/a12", create_empty_single, annihilate_single_double, -np.sqrt(2.0)),
            ("c12/a01", create_single_double, annihilate_empty_single, -np.sqrt(2.0)),
            ("c12/a12", create_single_double, annihilate_single_double, 1.0 / np.sqrt(2.0)),
        )
        channel_mpos = []
        for label, creation, annihilation, default_coeff in channel_terms:
            channel_autompo = AutoMPO([phys_leg] * args.nsites)
            channel_autompo.add_reduced_string_product(
                (
                    args.create,
                    creation.left_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
                ),
                (args.annihilate, time_reversed_reduced_operator(annihilation)),
                intermediate_irreps=(SU2Irrep(1),),
                middle_operators=middle,
                coeff=1.0,
                family=("R", "__fully_reduced_one_body_split__"),
            )
            channel_mpos.append((label, default_coeff * args.value, channel_autompo.build()))
        return tuple(channel_mpos)

    channel_terms = (
        ("a01/c01", annihilate_empty_single, create_empty_single, np.sqrt(2.0)),
        ("a01/c12", annihilate_empty_single, create_single_double, np.sqrt(2.0)),
        ("a12/c01", annihilate_single_double, create_empty_single, np.sqrt(2.0)),
        ("a12/c12", annihilate_single_double, create_single_double, -1.0 / np.sqrt(2.0)),
    )
    channel_mpos = []
    for label, annihilation, creation, default_coeff in channel_terms:
        channel_autompo = AutoMPO([phys_leg] * args.nsites)
        channel_autompo.add_reduced_string_product(
            (
                args.annihilate,
                annihilation.right_multiply_sector_scalar(double_phase).right_multiply_sector_scalar(parity),
            ),
            (args.create, time_reversed_reduced_operator(creation)),
            intermediate_irreps=(SU2Irrep(1),),
            middle_operators=middle,
            coeff=1.0,
            family=("R", "__fully_reduced_one_body_split__"),
        )
        channel_mpos.append((label, default_coeff * args.value, channel_autompo.build()))
    return tuple(channel_mpos)


def _fit_one_body_channels(path_specs, basis_states, dense_vectors, args, phys_leg, expected_operator):
    helpers = _load_reference_helpers()
    contract = helpers._contract_chain_transition
    channel_mpos = _one_body_channel_mpos(args, phys_leg)

    columns = []
    labels = []
    default_coeffs = []
    for label, default_coeff, channel_mpo in channel_mpos:
        labels.append(label)
        default_coeffs.append(default_coeff)
        column = []
        for bra_state in basis_states:
            for ket_state in basis_states:
                column.append(contract(bra_state, channel_mpo, ket_state))
        columns.append(column)
    design = np.asarray(columns, dtype=complex).T

    target = []
    for bra_index, _bra_state in enumerate(basis_states):
        for ket_index, _ket_state in enumerate(basis_states):
            target.append(
                np.vdot(
                    dense_vectors[bra_index],
                    expected_operator @ dense_vectors[ket_index],
                )
            )
    target = np.asarray(target, dtype=complex)

    fitted, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    default_coeffs = np.asarray(default_coeffs, dtype=complex)
    fitted_error = design @ fitted - target
    default_error = design @ default_coeffs - target

    print("\nchannel span fit:")
    print(f"  rank: {rank} / {len(labels)}")
    print(f"  singular values: {', '.join(f'{value:.6g}' for value in singular_values)}")
    print(f"  default max error: {np.max(np.abs(default_error)):.6e}")
    print(f"  best-fit max error: {np.max(np.abs(fitted_error)):.6e}")
    print(f"  best-fit norm error: {np.linalg.norm(fitted_error):.6e}")
    for label, default_coeff, fit_coeff in zip(labels, default_coeffs, fitted):
        print(f"  {label:8s} default={default_coeff.real:+.12g} fit={fit_coeff.real:+.12g}")


def debug_one_body(args):
    from pyqed.mps.nonabelian import (
        AutoMPO,
        FullyReducedSpatialOrbitalSite,
        add_spatial_one_body_terms,
        physical_leg_from_spatial_orbital,
        spatial_target_sector,
    )

    helpers = _load_reference_helpers()
    target = spatial_target_sector(args.nelec, args.spin_twice)
    path_specs = helpers._reduced_spatial_path_specs(args.nsites, target)
    basis_states, dense_vectors = helpers._reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())

    h1e = np.zeros((args.nsites, args.nsites))
    h1e[args.create, args.annihilate] = args.value
    autompo = AutoMPO([phys_leg] * args.nsites)
    add_spatial_one_body_terms(autompo, h1e, cutoff=args.cutoff)
    mpo = autompo.build()
    expected = helpers._dense_spatial_one_body_hamiltonian(h1e)

    print(f"one-body h[{args.create},{args.annihilate}]={args.value:g}")
    max_abs = _compare_matrix(
        path_specs,
        basis_states,
        dense_vectors,
        mpo,
        expected,
        tol=args.tol,
        limit=args.limit,
    )
    if args.fit_channels:
        _fit_one_body_channels(
            path_specs,
            basis_states,
            dense_vectors,
            args,
            phys_leg,
            expected,
        )
    if args.channels:
        if args.create == args.annihilate:
            print("\nchannel breakdown: onsite term has no split channels")
        else:
            channel_mpos = [
                (label, default_coeff, channel_mpo)
                for label, default_coeff, channel_mpo in _one_body_channel_mpos(
                    args,
                    phys_leg,
                )
            ]
            print("\nchannel contributions for mismatches:")
            shown = 0
            contract = helpers._contract_chain_transition
            for bra_index, bra_state in enumerate(basis_states):
                for ket_index, ket_state in enumerate(basis_states):
                    expected_value = np.vdot(
                        dense_vectors[bra_index],
                        expected @ dense_vectors[ket_index],
                    )
                    actual_value = contract(bra_state, mpo, ket_state)
                    if abs(actual_value - expected_value) <= args.tol:
                        continue
                    print()
                    print("bra", _format_state(bra_index, path_specs[bra_index]))
                    print("ket", _format_state(ket_index, path_specs[ket_index]))
                    for label, default_coeff, channel_mpo in channel_mpos:
                        value = default_coeff * contract(bra_state, channel_mpo, ket_state)
                        if abs(value) > args.tol:
                            print(f"  {label}: {value:+.16g}")
                    shown += 1
                    if shown >= args.limit:
                        return max_abs
    return max_abs


def debug_exchange(args):
    from pyqed.mps.nonabelian import (
        AutoMPO,
        FullyReducedSpatialOrbitalSite,
        add_spatial_spinfree_eri_terms,
        physical_leg_from_spatial_orbital,
        spatial_target_sector,
    )

    helpers = _load_reference_helpers()
    target = spatial_target_sector(args.nelec, args.spin_twice)
    path_specs = helpers._reduced_spatial_path_specs(args.nsites, target)
    basis_states, dense_vectors = helpers._reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())

    pattern = tuple(int(part) for part in args.pattern.split(","))
    if len(pattern) != 4:
        raise SystemExit("ERI pattern must be p,q,r,s.")
    eri = np.zeros((args.nsites, args.nsites, args.nsites, args.nsites))
    eri[pattern] = args.value
    autompo = AutoMPO([phys_leg] * args.nsites)
    add_spatial_spinfree_eri_terms(autompo, eri, cutoff=args.cutoff)
    mpo = autompo.build()
    expected = helpers._dense_spatial_spinfree_eri_hamiltonian(eri)

    print(f"ERI pattern {pattern}={args.value:g}")
    return _compare_matrix(
        path_specs,
        basis_states,
        dense_vectors,
        mpo,
        expected,
        tol=args.tol,
        limit=args.limit,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--nsites", type=int, default=4)
    common.add_argument("--nelec", type=int, default=4)
    common.add_argument("--spin-twice", type=int, default=0)
    common.add_argument("--value", type=float, default=1.0)
    common.add_argument("--cutoff", type=float, default=1.0e-12)
    common.add_argument("--tol", type=float, default=1.0e-12)
    common.add_argument("--limit", type=int, default=12)
    common.add_argument("--fail-on-mismatch", action="store_true")

    one_body = subparsers.add_parser("one-body", parents=[common])
    one_body.add_argument("--create", type=int, default=0)
    one_body.add_argument("--annihilate", type=int, default=1)
    one_body.add_argument("--channels", action="store_true")
    one_body.add_argument("--fit-channels", action="store_true")
    one_body.set_defaults(func=debug_one_body)

    exchange = subparsers.add_parser("exchange", parents=[common])
    exchange.add_argument("--pattern", default="0,1,1,2")
    exchange.set_defaults(func=debug_exchange)

    args = parser.parse_args(argv)
    max_abs = args.func(args)
    if args.fail_on_mismatch and max_abs > args.tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
