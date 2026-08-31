#!/usr/bin/env python3
"""Validate native periodic GDF against representation-matched PySCF GDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import numpy as np

from pyqed.pbc.gw import (
    GDF,
    PYSCF_GDF,
    KPointTransitionSpace,
    attach_pyscf_gdf_context,
    diagonal_g0w0,
    gdf_mo_jk,
    gdf_transition_factors,
    periodic_bse_matrices,
    pyscf_gdf_transition_factors,
)
from pyqed.pbc.gw.integrals import (
    _gdf_normalize_auxbasis_name,
    _pyscf_builtin_basis_dict,
    _pyscf_cell_from_reference,
)
from pyqed.qchem.basis import _basis_path, parse_gbs
from pyqed.qchem.pbc import Cell
from pyqed.units import au2mev


HARTREE_TO_MEV = au2mev


@dataclass(frozen=True)
class ValidationCase:
    name: str
    atom: str
    lattice: np.ndarray
    basis: str
    auxbasis: str
    kmesh: tuple[int, int, int] = (1, 1, 1)
    basis_min_exponent: float | None = None
    gamma_centered: bool = False


def _cubic(length):
    return np.diag([float(length)] * 3)


def _fcc_primitive(length):
    half = 0.5 * float(length)
    return np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )


CASES = {
    "h2-3k": ValidationCase(
        "h2-3k",
        "H 0 0 0; H 1.4 0 0",
        _cubic(5.0),
        "sto-3g",
        "def2-svp-jkfit",
        (3, 1, 1),
    ),
    "lih": ValidationCase(
        "lih",
        "Li 0 0 0; H 1.6 0 0",
        _cubic(7.0),
        "sto-3g",
        "def2-svp-jkfit",
    ),
    "lih-rocksalt-2k": ValidationCase(
        "lih-rocksalt-2k",
        f"Li 0 0 0; H {7.72 / 2:.12f} {7.72 / 2:.12f} {7.72 / 2:.12f}",
        _fcc_primitive(7.72),
        "sto-3g",
        "def2-svp-jkfit",
        (2, 2, 2),
    ),
    "lih-rocksalt-2k-svp": ValidationCase(
        "lih-rocksalt-2k-svp",
        f"Li 0 0 0; H {7.72 / 2:.12f} {7.72 / 2:.12f} {7.72 / 2:.12f}",
        _fcc_primitive(7.72),
        "def2-svp",
        "def2-svp-jkfit",
        (2, 2, 2),
    ),
    "lih-rocksalt-2k-svp-solid": ValidationCase(
        "lih-rocksalt-2k-svp-solid",
        f"Li 0 0 0; H {7.72 / 2:.12f} {7.72 / 2:.12f} {7.72 / 2:.12f}",
        _fcc_primitive(7.72),
        "def2-svp",
        "def2-svp-jkfit",
        (2, 2, 2),
        basis_min_exponent=0.1,
    ),
    "h2-ccpvdz": ValidationCase(
        "h2-ccpvdz",
        "H 0 0 0; H 1.4 0 0",
        _cubic(7.0),
        "cc-pvdz",
        "cc-pvdz-jkfit",
    ),
    "he": ValidationCase(
        "he",
        "He 0 0 0",
        _cubic(7.0),
        "sto-3g",
        "def2-svp-jkfit",
    ),
    "diamond": ValidationCase(
        "diamond",
        f"C 0 0 0; C {6.74 / 4:.12f} {6.74 / 4:.12f} {6.74 / 4:.12f}",
        _fcc_primitive(6.74),
        "sto-3g",
        "def2-svp-jkfit",
    ),
    "bn": ValidationCase(
        "bn",
        f"B 0 0 0; N {6.83 / 4:.12f} {6.83 / 4:.12f} {6.83 / 4:.12f}",
        _fcc_primitive(6.83),
        "sto-3g",
        "def2-svp-jkfit",
    ),
}


def _case_basis(case):
    if case.basis_min_exponent is None:
        return case.basis
    threshold = float(case.basis_min_exponent)
    parsed = parse_gbs(_basis_path(case.basis))
    symbols = [part.split()[0] for part in case.atom.split(";")]
    basis = {}
    for symbol in dict.fromkeys(symbols):
        shells = []
        for angular_momentum, exponents, coefficients in parsed[symbol]:
            exponents = np.asarray(exponents, dtype=float)
            coefficients = np.asarray(coefficients, dtype=float)
            if coefficients.ndim == 1:
                coefficients = coefficients[:, None]
            keep = exponents >= threshold
            if np.any(keep):
                shells.append(
                    (
                        int(angular_momentum),
                        exponents[keep],
                        coefficients[keep],
                    )
                )
        if not shells:
            raise ValueError(
                f"Exponent floor {threshold} removed every {symbol} shell."
            )
        basis[symbol] = shells
    return basis


def _seed_native_reference(case):
    cell = Cell(
        atom=case.atom,
        a=case.lattice,
        basis=_case_basis(case),
        unit="bohr",
        dimension=3,
    ).build()
    kpts = cell.make_kpts(case.kmesh, gamma_centered=case.gamma_centered)
    mf = cell.KRHF(kpts=kpts)
    nocc = cell.nelectron // 2
    mf.mo_energy = [np.arange(cell.nao, dtype=float) for _ in kpts]
    mf.mo_coeff = [np.eye(cell.nao) for _ in kpts]
    mf.mo_occ = [
        np.r_[2.0 * np.ones(nocc), np.zeros(cell.nao - nocc)] for _ in kpts
    ]
    return cell, kpts, mf


def _pyscf_reference(
    case,
    precision,
    aux_min_exponent=None,
    metric_tol=1.0e-14,
    force_metric_eig=False,
):
    from pyscf.pbc import scf

    cell, kpts, seed = _seed_native_reference(case)
    seed.gdf_auxbasis = case.auxbasis
    reference = KPointTransitionSpace(seed, qpts="mesh").reference
    pyscf_cell = _pyscf_cell_from_reference(reference)
    pyscf_cell.precision = float(precision)
    pyscf_cell.build()
    auxbasis = _pyscf_builtin_basis_dict(
        _gdf_normalize_auxbasis_name(case.auxbasis),
        cell._atom_symbols,
    )
    mf = scf.KRHF(pyscf_cell, kpts=kpts, exxdiv="ewald")
    if force_metric_eig:
        from pyscf.pbc.df.df import GDF as PySCFGDF
        from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
        from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
        from pyscf.pbc.lib.kpts_helper import unique

        class EigenGDF(PySCFGDF):
            def _make_j3c(
                self,
                cell=None,
                auxcell=None,
                kptij_lst=None,
                cderi_file=None,
            ):
                cell = self.cell if cell is None else cell
                auxcell = self.auxcell if auxcell is None else auxcell
                cderi_file = (
                    self._cderi_to_save if cderi_file is None else cderi_file
                )
                kpts_union = (
                    self.kpts
                    if self.kpts_band is None
                    else unique(np.vstack([self.kpts, self.kpts_band]))[0]
                )
                if self._prefer_ccdf or cell.omega > 0:
                    builder = _CCGDFBuilder(cell, auxcell, kpts_union)
                    builder.eta = self.eta
                else:
                    builder = _RSGDFBuilder(cell, auxcell, kpts_union)
                builder.mesh = self.mesh
                builder.linear_dep_threshold = self.linear_dep_threshold
                builder.j2c_eig_always = True
                j_only = self._j_only or len(kpts_union) == 1
                builder.make_j3c(
                    cderi_file,
                    j_only=j_only,
                    dataname=self._dataname,
                    kptij_lst=kptij_lst,
                )

        mf.with_df = EigenGDF(pyscf_cell, kpts=kpts)
        mf.with_df.auxbasis = auxbasis
        mf.with_df.linear_dep_threshold = float(metric_tol)
    else:
        mf = mf.density_fit(auxbasis=auxbasis)
    if aux_min_exponent is not None:
        mf.with_df.exp_to_discard = float(aux_min_exponent)
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 80
    mf.verbose = 0
    started = time.perf_counter()
    mf.with_df.build()
    df_seconds = time.perf_counter() - started
    started = time.perf_counter()
    energy = mf.kernel()
    scf_after_df_seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError(f"PySCF KRHF did not converge for {case.name}.")
    return (
        cell,
        kpts,
        mf,
        float(energy),
        df_seconds,
        scf_after_df_seconds,
    )


def _native_space(
    case,
    cell,
    kpts,
    pyscf_mf,
    precision,
    short_range_cut=None,
    omega=None,
    primitive_exp_cutoff=None,
    self_opposite_pair_reuse=True,
    pair_image_tol_factor=None,
    pair_ft_coeff_tol_factor=None,
    pair_ft_factor_screen_tol=None,
    reciprocal_kernel="range_separated",
    rs_aux_partition="smooth",
    metric_relative_tol=None,
    aux_min_exponent=None,
    metric_tol=1.0e-14,
    reference_precision=1.0e-12,
):
    mf = cell.KRHF(kpts=kpts).density_fit(
        auxbasis=case.auxbasis,
        precision=float(precision),
        mesh="auto",
        omega="auto" if omega is None else float(omega),
        pair_cut="auto",
        stream_pairs=True,
        metric_tol=float(metric_tol),
    )
    mf.gdf_reciprocal_kernel = str(reciprocal_kernel)
    mf.gdf_rs_aux_partition = str(rs_aux_partition)
    if metric_relative_tol is not None:
        mf.gdf_metric_relative_tol = float(metric_relative_tol)
    if aux_min_exponent is not None:
        mf.gdf_aux_min_exponent = float(aux_min_exponent)
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_self_opposite_pair_reuse = bool(self_opposite_pair_reuse)
    if pair_image_tol_factor is not None:
        mf.gdf_pair_image_tol_factor = float(pair_image_tol_factor)
    if pair_ft_coeff_tol_factor is not None:
        mf.gdf_pair_ft_coeff_tol_factor = float(pair_ft_coeff_tol_factor)
    if pair_ft_factor_screen_tol is not None:
        mf.gdf_pair_ft_factor_screen_tol = float(pair_ft_factor_screen_tol)
    if short_range_cut is not None:
        mf.gdf_short_range_cut = int(short_range_cut)
    if primitive_exp_cutoff is not None:
        mf.gdf_short_range_primitive_exp_cutoff = primitive_exp_cutoff
    mf.mo_energy = [np.asarray(block).copy() for block in pyscf_mf.mo_energy]
    mf.mo_coeff = [np.asarray(block).copy() for block in pyscf_mf.mo_coeff]
    mf.mo_occ = [np.asarray(block).copy() for block in pyscf_mf.mo_occ]
    mf._periodic_setup()
    space = KPointTransitionSpace(mf, qpts="mesh")
    attach_pyscf_gdf_context(space, pyscf_mf)
    return space


def _pair_metric_diagnostics(native, pyscf):
    errors = []
    max_self_absolute_error = 0.0
    max_self_location = None
    native_rows = []
    pyscf_rows = []
    for pair_key, native_block in native.pair_blocks.items():
        pyscf_block = pyscf.pair_blocks[pair_key]
        left = native_block.reshape(native.naux, -1).T
        right = pyscf_block.reshape(pyscf.naux, -1).T
        native_rows.append(left)
        pyscf_rows.append(right)
        left_metric = left @ left.conj().T
        right_metric = right @ right.conj().T
        absolute = np.abs(left_metric - right_metric)
        flat_index = int(np.argmax(absolute))
        absolute_error = float(absolute.flat[flat_index])
        if absolute_error > max_self_absolute_error:
            matrix_index = np.unravel_index(flat_index, absolute.shape)
            max_self_absolute_error = absolute_error
            max_self_location = {
                "pair_key": [int(value) for value in pair_key],
                "matrix_index": [int(value) for value in matrix_index],
            }
        errors.append(
            np.linalg.norm(left_metric - right_metric)
            / max(np.linalg.norm(right_metric), np.finfo(float).tiny)
        )
    if not native_rows:
        return {
            "self_relative_error": 0.0,
            "global_relative_error": 0.0,
            "global_absolute_error": 0.0,
            "max_self_absolute_error": 0.0,
            "max_self_location": None,
        }

    left = np.concatenate(native_rows, axis=0)
    right = np.concatenate(pyscf_rows, axis=0)
    left_inner = left.conj().T @ left
    right_inner = right.conj().T @ right
    cross_inner = left.conj().T @ right
    left_norm2 = float(np.vdot(left_inner, left_inner).real)
    right_norm2 = float(np.vdot(right_inner, right_inner).real)
    cross_norm2 = float(np.vdot(cross_inner, cross_inner).real)
    difference_norm = np.sqrt(max(0.0, left_norm2 + right_norm2 - 2.0 * cross_norm2))
    return {
        "self_relative_error": float(max(errors, default=0.0)),
        "global_relative_error": float(
            difference_norm / max(np.sqrt(right_norm2), np.finfo(float).tiny)
        ),
        "global_absolute_error": float(difference_norm),
        "max_self_absolute_error": float(max_self_absolute_error),
        "max_self_location": max_self_location,
    }


def validate_case(
    case,
    precision=1.0e-8,
    reference_precision=1.0e-12,
    ac_nw=16,
    run_gw=True,
    run_bse=False,
    short_range_cut=None,
    omega=None,
    primitive_exp_cutoff=None,
    finite_size_correction=False,
    run_native_krhf=False,
    native_jk_builder="gdf",
    self_opposite_pair_reuse=True,
    pair_image_tol_factor=None,
    pair_ft_coeff_tol_factor=None,
    pair_ft_factor_screen_tol=None,
    reciprocal_kernel="range_separated",
    rs_aux_partition="smooth",
    metric_relative_tol=None,
    aux_min_exponent=None,
    metric_tol=1.0e-14,
    pyscf_metric_eig=False,
    bse_screened_exchange_scale=1.0,
):
    (
        cell,
        kpts,
        pyscf_mf,
        scf_energy,
        pyscf_df_seconds,
        pyscf_scf_after_df_seconds,
    ) = _pyscf_reference(
        case,
        reference_precision,
        aux_min_exponent=aux_min_exponent,
        metric_tol=metric_tol,
        force_metric_eig=pyscf_metric_eig,
    )
    space = _native_space(
        case,
        cell,
        kpts,
        pyscf_mf,
        precision,
        short_range_cut=short_range_cut,
        omega=omega,
        primitive_exp_cutoff=primitive_exp_cutoff,
        self_opposite_pair_reuse=self_opposite_pair_reuse,
        pair_image_tol_factor=pair_image_tol_factor,
        pair_ft_coeff_tol_factor=pair_ft_coeff_tol_factor,
        pair_ft_factor_screen_tol=pair_ft_factor_screen_tol,
        reciprocal_kernel=reciprocal_kernel,
        rs_aux_partition=rs_aux_partition,
        metric_relative_tol=metric_relative_tol,
        aux_min_exponent=aux_min_exponent,
        metric_tol=metric_tol,
    )
    q_rows = []
    factor_started = time.perf_counter()
    native_df = space.reference._pbc_mf.with_df
    native_df.build(workers=min(12, max(1, space.nqpts)))
    for q_index in range(space.nqpts):
        native = gdf_transition_factors(space, q_index=q_index, g2_tol=1.0e-14)
        pyscf = pyscf_gdf_transition_factors(space, q_index=q_index)
        pair_metric = _pair_metric_diagnostics(native, pyscf)
        eigenvalues = np.asarray(native.metric_eigenvalues, dtype=float)
        metric_threshold = float(
            native.build_timings.get(
                "metric_eigenvalue_threshold",
                native.factor_threshold,
            )
        )
        retained = eigenvalues[eigenvalues > metric_threshold]
        q_rows.append(
            {
                "q_index": int(q_index),
                "qvec": np.asarray(space.qpts[q_index]).tolist(),
                "pair_metric_relative_error": pair_metric["global_relative_error"],
                "pair_self_metric_relative_error": pair_metric["self_relative_error"],
                "pair_metric_absolute_error": pair_metric["global_absolute_error"],
                "pair_self_metric_max_absolute_error": pair_metric[
                    "max_self_absolute_error"
                ],
                "pair_self_metric_max_error_location": pair_metric[
                    "max_self_location"
                ],
                "native_rank": int(native.naux),
                "pyscf_rank": int(pyscf.naux),
                "metric_min_eigenvalue": float(eigenvalues[0]),
                "metric_max_eigenvalue": float(eigenvalues[-1]),
                "metric_min_retained_eigenvalue": float(retained[0]),
                "metric_eigenvalue_threshold": metric_threshold,
                "metric_condition_number": float(retained[-1] / retained[0]),
                "metric_whitening_condition_number": float(
                    np.sqrt(retained[-1] / retained[0])
                ),
                "metric_eigenvalues": eigenvalues.tolist(),
                "native_factor_bytes": int(
                    sum(np.asarray(block).nbytes for block in native.pair_blocks.values())
                ),
                "pyscf_factor_bytes": int(
                    sum(np.asarray(block).nbytes for block in pyscf.pair_blocks.values())
                ),
                "native_build_timings": dict(native.build_timings),
            }
        )
    factor_seconds = time.perf_counter() - factor_started

    native_j, native_k = gdf_mo_jk(space, coulomb_component=GDF)
    pyscf_j, pyscf_k = gdf_mo_jk(space, coulomb_component=PYSCF_GDF)
    row = {
        "case": case.name,
        "basis": case.basis,
        "basis_min_exponent": case.basis_min_exponent,
        "auxbasis": case.auxbasis,
        "aux_min_exponent": aux_min_exponent,
        "metric_tol": float(metric_tol),
        "pyscf_metric_eig": bool(pyscf_metric_eig),
        "kmesh": list(case.kmesh),
        "nao": int(cell.nao),
        "nkpts": int(len(kpts)),
        "pyscf_gdf_krhf_energy_Ha": scf_energy,
        "pyscf_reference_precision": float(reference_precision),
        "pyscf_gdf_build_seconds": pyscf_df_seconds,
        "pyscf_gdf_krhf_after_build_seconds": pyscf_scf_after_df_seconds,
        "pyscf_gdf_krhf_seconds": (
            pyscf_df_seconds + pyscf_scf_after_df_seconds
        ),
        "native_gdf_seconds": factor_seconds,
        "native_factor_bytes": int(
            sum(item["native_factor_bytes"] for item in q_rows)
        ),
        "pyscf_factor_bytes": int(
            sum(item["pyscf_factor_bytes"] for item in q_rows)
        ),
        "native_gdf_q_build_timings": {
            str(q_index): dict(timings)
            for q_index, timings in native_df.build_timings.items()
        },
        "native_gdf_multi_q_build_timings": [
            dict(timings) for timings in native_df.multi_q_build_timings
        ],
        "max_pair_metric_relative_error": max(
            item["pair_metric_relative_error"] for item in q_rows
        ),
        "max_abs_J_error_Ha": float(np.max(np.abs(native_j - pyscf_j))),
        "max_abs_K_error_Ha": float(np.max(np.abs(native_k - pyscf_k))),
        "max_abs_J_error_meV": float(
            HARTREE_TO_MEV * np.max(np.abs(native_j - pyscf_j))
        ),
        "max_abs_K_error_meV": float(
            HARTREE_TO_MEV * np.max(np.abs(native_k - pyscf_k))
        ),
        "q_blocks": q_rows,
    }

    if run_native_krhf:
        native_scf = cell.KRHF(
            kpts=kpts,
            eta=0.5,
            real_cut="auto",
            pair_cut="auto",
            recip_cut=5,
            jk_builder=native_jk_builder,
        )
        if native_jk_builder == "gdf":
            native_scf.density_fit(
                auxbasis=case.auxbasis,
                precision=float(precision),
                mesh="auto",
                omega="auto" if omega is None else float(omega),
                pair_cut="auto",
                stream_pairs=True,
                metric_tol=float(metric_tol),
            )
        else:
            native_scf.gdf_auxbasis = case.auxbasis
            native_scf.gdf_precision = float(precision)
            native_scf.gdf_mesh = "auto"
            native_scf.gdf_omega = "auto" if omega is None else float(omega)
            native_scf.gdf_pair_cut = "auto"
            native_scf.gdf_metric_tol = float(metric_tol)
        native_scf.gdf_reciprocal_kernel = str(reciprocal_kernel)
        native_scf.gdf_rs_aux_partition = str(rs_aux_partition)
        if metric_relative_tol is not None:
            native_scf.gdf_metric_relative_tol = float(metric_relative_tol)
        if aux_min_exponent is not None:
            native_scf.gdf_aux_min_exponent = float(aux_min_exponent)
        native_scf.gdf_self_opposite_pair_reuse = bool(
            self_opposite_pair_reuse
        )
        if pair_image_tol_factor is not None:
            native_scf.gdf_pair_image_tol_factor = float(pair_image_tol_factor)
        if pair_ft_coeff_tol_factor is not None:
            native_scf.gdf_pair_ft_coeff_tol_factor = float(
                pair_ft_coeff_tol_factor
            )
        if pair_ft_factor_screen_tol is not None:
            native_scf.gdf_pair_ft_factor_screen_tol = float(
                pair_ft_factor_screen_tol
            )
        started = time.perf_counter()
        if native_jk_builder == "gdf":
            native_scf.with_df.build(workers=min(12, max(1, len(kpts))))
        native_scf.run(max_cycle=80, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
        row["native_krhf"] = {
            "converged": bool(native_scf.converged),
            "jk_builder": str(native_jk_builder),
            "real_cut": int(native_scf.real_cut),
            "pair_cut": int(native_scf.pair_cut),
            "one_body_nuclear_cut": int(native_scf.one_body_nuclear_cut),
            "one_body_screen_tol": float(native_scf.one_body_screen_tol),
            "integral_build_timings": dict(native_scf.integral_build_timings),
            "energy_Ha": float(native_scf.e_tot),
            "energy_error_vs_pyscf_gdf_Ha": float(native_scf.e_tot - scf_energy),
            "seconds": float(time.perf_counter() - started),
            "iterations": int(native_scf.niter),
            "last_cycle": (
                None if not native_scf.scf_history else native_scf.scf_history[-1]
            ),
        }

    if run_gw:
        nocc = cell.nelectron // 2
        qp_bands = sorted({max(0, nocc - 1), min(cell.nao - 1, nocc)})
        options = dict(
            direct_scale=1.0,
            linearized=True,
            frequency_integration="ac",
            ac_nw=int(ac_nw),
            finite_size_correction=bool(finite_size_correction),
            qp_bands=qp_bands,
        )
        native_gw = diagonal_g0w0(space, coulomb_component=GDF, **options)
        pyscf_gw = diagonal_g0w0(space, coulomb_component=PYSCF_GDF, **options)
        qp_delta = np.asarray(native_gw.e_qp) - np.asarray(pyscf_gw.e_qp)
        selected = np.asarray(qp_delta)[:, qp_bands]
        row["gw"] = {
            "ac_nw": int(ac_nw),
            "finite_size_correction": bool(finite_size_correction),
            "qp_bands": qp_bands,
            "native_qp_Ha": np.asarray(native_gw.e_qp)[:, qp_bands].tolist(),
            "pyscf_factor_qp_Ha": np.asarray(pyscf_gw.e_qp)[:, qp_bands].tolist(),
            "max_abs_qp_error_Ha": float(np.max(np.abs(selected))),
            "max_abs_qp_error_meV": float(
                HARTREE_TO_MEV * np.max(np.abs(selected))
            ),
        }

    if run_bse:
        native_bse = periodic_bse_matrices(
            space,
            coulomb_component=GDF,
            screened_exchange_scale=float(bse_screened_exchange_scale),
        )
        pyscf_bse = periodic_bse_matrices(
            space,
            coulomb_component=PYSCF_GDF,
            screened_exchange_scale=float(bse_screened_exchange_scale),
        )
        native_tda = np.linalg.eigvalsh(native_bse.A)
        pyscf_tda = np.linalg.eigvalsh(pyscf_bse.A)
        exchange_delta = np.abs(native_bse.exchange - pyscf_bse.exchange)
        exchange_index = tuple(
            int(value)
            for value in np.unravel_index(
                int(np.argmax(exchange_delta)),
                exchange_delta.shape,
            )
        )
        left_transition = space.transitions(0)[exchange_index[0]]
        right_transition = space.transitions(0)[exchange_index[1]]
        exchange_q_index = space.find_qpoint_index(
            space.reference.kpts[right_transition.k_index]
            - space.reference.kpts[left_transition.k_index]
        )
        row["bse"] = {
            "screened_exchange_scale": float(bse_screened_exchange_scale),
            "max_abs_A_error_Ha": float(
                np.max(np.abs(native_bse.A - pyscf_bse.A))
            ),
            "max_abs_B_error_Ha": float(
                np.max(np.abs(native_bse.B - pyscf_bse.B))
            ),
            "max_abs_direct_error_Ha": float(
                np.max(np.abs(native_bse.direct - pyscf_bse.direct))
            ),
            "max_abs_exchange_error_Ha": float(
                np.max(exchange_delta)
            ),
            "max_exchange_error_location": {
                "matrix_index": list(exchange_index),
                "screen_q_index": int(exchange_q_index),
                "left_transition": {
                    "k_index": int(left_transition.k_index),
                    "kq_index": int(left_transition.kq_index),
                    "occ_band": int(left_transition.occ_band),
                    "vir_band": int(left_transition.vir_band),
                },
                "right_transition": {
                    "k_index": int(right_transition.k_index),
                    "kq_index": int(right_transition.kq_index),
                    "occ_band": int(right_transition.occ_band),
                    "vir_band": int(right_transition.vir_band),
                },
            },
            "max_abs_screened_exchange_error_Ha": float(
                np.max(
                    np.abs(
                        native_bse.screened_exchange
                        - pyscf_bse.screened_exchange
                    )
                )
            ),
            "max_abs_tda_eigenvalue_error_Ha": float(
                np.max(np.abs(native_tda - pyscf_tda))
            ),
            "max_abs_tda_eigenvalue_error_meV": float(
                HARTREE_TO_MEV * np.max(np.abs(native_tda - pyscf_tda))
            ),
        }
    return row


def _parse_float_list(value):
    return [float(item) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value):
    return [int(item) for item in str(value).split(",") if item.strip()]


def _parse_kmesh(value):
    parts = str(value).lower().replace("x", ",").split(",")
    mesh = tuple(int(item) for item in parts if item.strip())
    if len(mesh) != 3 or any(item <= 0 for item in mesh):
        raise argparse.ArgumentTypeError(
            "kmesh must contain three positive integers, for example 4,2,2"
        )
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "extended"), default="quick")
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--reference-precision", type=float, default=1.0e-12)
    parser.add_argument("--ac-nw", type=int, default=16)
    parser.add_argument("--short-range-cut", type=int)
    parser.add_argument("--omega", type=float)
    parser.add_argument("--primitive-exp-cutoff")
    parser.add_argument("--pair-image-tol-factor", type=float)
    parser.add_argument("--pair-ft-coeff-tol-factor", type=float)
    parser.add_argument("--pair-ft-factor-screen-tol", type=float)
    parser.add_argument(
        "--reciprocal-kernel",
        choices=("full", "range_separated"),
        default="range_separated",
    )
    parser.add_argument(
        "--rs-aux-partition",
        choices=("smooth", "off", "all"),
        default="smooth",
    )
    parser.add_argument("--metric-relative-tol", type=float)
    parser.add_argument("--metric-tol", type=float, default=1.0e-14)
    parser.add_argument("--metric-tol-ladder", type=_parse_float_list)
    parser.add_argument("--pyscf-metric-eig", action="store_true")
    parser.add_argument("--aux-min-exponent", type=float)
    parser.add_argument("--aux-min-exponent-ladder", type=_parse_float_list)
    parser.add_argument("--no-self-opposite-pair-reuse", action="store_true")
    parser.add_argument("--no-gw", action="store_true")
    parser.add_argument("--bse", action="store_true")
    parser.add_argument("--bse-screened-exchange-scale", type=float, default=1.0)
    parser.add_argument("--native-krhf", action="store_true")
    parser.add_argument(
        "--native-jk-builder",
        choices=("gdf", "reciprocal", "ewald"),
        default="gdf",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--precision-ladder", type=_parse_float_list)
    parser.add_argument("--frequency-ladder", type=_parse_int_list)
    parser.add_argument("--finite-size-ladder", action="store_true")
    parser.add_argument("--auxbasis", action="append")
    parser.add_argument("--kmesh-x", type=_parse_int_list)
    parser.add_argument(
        "--kmesh",
        action="append",
        type=_parse_kmesh,
        help="repeatable 3D mesh, for example --kmesh 2,2,2",
    )
    parser.add_argument(
        "--gamma-centered",
        action="store_true",
        help="use a Gamma-centered mesh instead of a half-shifted mesh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_validation.json"),
    )
    args = parser.parse_args()
    if args.kmesh_x and args.kmesh:
        parser.error("--kmesh-x and --kmesh are mutually exclusive")

    names = args.case
    if names is None:
        names = (
            ["h2-3k", "lih", "h2-ccpvdz"]
            if args.profile == "quick"
            else list(CASES)
        )
    requested = [CASES[name] for name in names]
    if args.gamma_centered:
        requested = [replace(case, gamma_centered=True) for case in requested]
    if args.auxbasis:
        requested = [
            replace(case, name=f"{case.name}:{aux}", auxbasis=aux)
            for case in requested
            for aux in args.auxbasis
        ]
    if args.kmesh_x:
        requested = [
            replace(case, name=f"{case.name}:{nk}x1x1", kmesh=(nk, 1, 1))
            for case in requested
            for nk in args.kmesh_x
        ]
    if args.kmesh:
        requested = [
            replace(
                case,
                name=f"{case.name}:{'x'.join(str(item) for item in mesh)}",
                kmesh=mesh,
            )
            for case in requested
            for mesh in args.kmesh
        ]

    studies = []
    precisions = args.precision_ladder or [args.precision]
    frequencies = args.frequency_ladder or [args.ac_nw]
    finite_size_options = [False, True] if args.finite_size_ladder else [False]
    aux_min_exponents = args.aux_min_exponent_ladder or [args.aux_min_exponent]
    metric_tols = args.metric_tol_ladder or [args.metric_tol]
    for case in requested:
        for precision in precisions:
            for ac_nw in frequencies:
                for finite_size in finite_size_options:
                    for aux_min_exponent in aux_min_exponents:
                        for metric_tol in metric_tols:
                            row = validate_case(
                                case,
                                precision=precision,
                                ac_nw=ac_nw,
                                run_gw=not args.no_gw,
                                run_bse=args.bse,
                                short_range_cut=args.short_range_cut,
                                omega=args.omega,
                                primitive_exp_cutoff=args.primitive_exp_cutoff,
                                finite_size_correction=finite_size,
                                run_native_krhf=args.native_krhf,
                                native_jk_builder=args.native_jk_builder,
                                self_opposite_pair_reuse=(
                                    not args.no_self_opposite_pair_reuse
                                ),
                                pair_image_tol_factor=args.pair_image_tol_factor,
                                pair_ft_coeff_tol_factor=(
                                    args.pair_ft_coeff_tol_factor
                                ),
                                pair_ft_factor_screen_tol=(
                                    args.pair_ft_factor_screen_tol
                                ),
                                reciprocal_kernel=args.reciprocal_kernel,
                                rs_aux_partition=args.rs_aux_partition,
                                metric_relative_tol=args.metric_relative_tol,
                                aux_min_exponent=aux_min_exponent,
                                metric_tol=metric_tol,
                                pyscf_metric_eig=args.pyscf_metric_eig,
                                bse_screened_exchange_scale=(
                                    args.bse_screened_exchange_scale
                                ),
                                reference_precision=args.reference_precision,
                            )
                            row["gdf_precision"] = float(precision)
                            row["requested_short_range_cut"] = args.short_range_cut
                            row["requested_omega"] = args.omega
                            row["requested_primitive_exp_cutoff"] = (
                                args.primitive_exp_cutoff
                            )
                            row["self_opposite_pair_reuse"] = bool(
                                not args.no_self_opposite_pair_reuse
                            )
                            row["requested_pair_image_tol_factor"] = (
                                args.pair_image_tol_factor
                            )
                            row["requested_pair_ft_coeff_tol_factor"] = (
                                args.pair_ft_coeff_tol_factor
                            )
                            row["requested_pair_ft_factor_screen_tol"] = (
                                args.pair_ft_factor_screen_tol
                            )
                            row["reciprocal_kernel"] = args.reciprocal_kernel
                            row["rs_aux_partition"] = args.rs_aux_partition
                            row["metric_relative_tol"] = args.metric_relative_tol
                            row["pyscf_metric_eig"] = bool(args.pyscf_metric_eig)
                            studies.append(row)
                            print(
                                f"{case.name}: aux_min_exp={aux_min_exponent} "
                                f"metric_tol={metric_tol:.1e} "
                                f"metric={row['max_pair_metric_relative_error']:.3e} "
                                f"J={row['max_abs_J_error_Ha']:.3e} "
                                f"K={row['max_abs_K_error_Ha']:.3e} Ha "
                                f"({row['max_abs_K_error_meV']:.6f} meV)",
                                flush=True,
                            )

    payload = {
        "profile": args.profile,
        "studies": studies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote {args.output}")

    if args.strict:
        failed = [
            row
            for row in studies
            if row["max_pair_metric_relative_error"] > 1.0e-5
            or row["max_abs_J_error_Ha"] > 1.0e-6
            or row["max_abs_K_error_Ha"] > 1.0e-6
            or row.get("gw", {}).get("max_abs_qp_error_Ha", 0.0) > 1.0e-5
            or row.get("bse", {}).get("max_abs_A_error_Ha", 0.0) > 1.0e-5
            or row.get("bse", {}).get("max_abs_B_error_Ha", 0.0) > 1.0e-5
            or (
                "native_krhf" in row
                and (
                    not row["native_krhf"]["converged"]
                    or abs(
                        row["native_krhf"][
                            "energy_error_vs_pyscf_gdf_Ha"
                        ]
                    )
                    > 1.0e-5
                )
            )
        ]
        if failed:
            raise SystemExit(
                "Strict GDF validation failed for: "
                + ", ".join(row["case"] for row in failed)
            )


if __name__ == "__main__":
    main()
