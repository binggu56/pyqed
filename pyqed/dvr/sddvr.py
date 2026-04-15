#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simultaneous-diagonalization DVR utilities.
"""

import numpy as np

from .asd import jacobi_angles


def _as_operator_stack(operators):
    mats = np.asarray(operators)
    if mats.ndim != 3:
        raise ValueError("operators must have shape (nops, n, n).")
    if mats.shape[1] != mats.shape[2]:
        raise ValueError("operators must be square matrices.")
    if not np.allclose(mats, np.swapaxes(mats, -1, -2)):
        raise ValueError("operators must be real symmetric matrices.")
    return np.array(mats, dtype=float, copy=True)


def simultaneous_diagonalize(operators, tol=1e-10, max_iter=1000, verbose=False):
    """
    Approximately simultaneously diagonalize a set of real symmetric operators.

    Parameters
    ----------
    operators : ndarray, shape (nops, n, n)
        Operators in a finite basis representation.
    tol : float, optional
        Gradient tolerance for the joint-diagonalization solver.
    max_iter : int, optional
        Maximum number of quasi-Newton iterations.
    verbose : bool, optional
        Whether to print optimizer progress.

    Returns
    -------
    transform : ndarray, shape (n, n)
        Orthogonal transform such that ``transform @ op @ transform.T`` is as
        diagonal as possible for all operators.
    diagonalized : ndarray, shape (nops, n, n)
        The transformed, quasi-diagonal operators.
    info : dict
        Monitoring information returned by the optimizer.
    """
    mats = _as_operator_stack(operators)
    work = mats.copy()
    transform, _, err = jacobi_angles(*work, sweeps=max_iter, eps=tol)
    diagonalized = np.stack(work, axis=0)
    info = {'error': err}
    return transform, diagonalized, info


class SDDVR:
    """
    Simultaneous-diagonalization DVR built from projected coordinate operators.

    Notes
    -----
    In a truncated multidimensional finite basis, projected coordinate
    operators may not commute exactly. SD-DVR replaces exact diagonalization of
    a single coordinate by an approximate joint diagonalization of several
    coordinate-like operators.
    """

    def __init__(self, operators, labels=None, tol=1e-10, max_iter=1000, verbose=False):
        self.operators = _as_operator_stack(operators)
        self.nops, self.nbasis, _ = self.operators.shape
        self.labels = list(labels) if labels is not None else [f"q{i}" for i in range(self.nops)]

        if len(self.labels) != self.nops:
            raise ValueError("labels must have the same length as the operator list.")

        self.transform, self.diagonalized, self.info = simultaneous_diagonalize(
            self.operators,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )
        self.grid = np.stack(
            [np.diag(self.diagonalized[k]).copy() for k in range(self.nops)],
            axis=1,
        )

    def fbr2dvr(self, operator):
        """
        Transform an operator from the original basis to the SD-DVR basis.
        """
        op = np.asarray(operator)
        if op.shape != (self.nbasis, self.nbasis):
            raise ValueError("operator shape does not match the SD-DVR basis size.")
        return self.transform @ op @ self.transform.T

    def dvr2fbr(self, operator):
        """
        Transform an operator from the SD-DVR basis back to the original basis.
        """
        op = np.asarray(operator)
        if op.shape != (self.nbasis, self.nbasis):
            raise ValueError("operator shape does not match the SD-DVR basis size.")
        return self.transform.T @ op @ self.transform

    def coordinate(self, index):
        """
        Quasi-diagonal coordinate operator in the SD-DVR basis.
        """
        return self.diagonalized[index].copy()

    def local_operator(self, func):
        """
        Build a diagonal local operator ``func(q_1, ..., q_n)`` on the SD-DVR grid.
        """
        values = np.asarray([func(*point) for point in self.grid], dtype=float)
        return np.diag(values)

    def diagonal_error(self):
        """
        Frobenius norm of the off-diagonal residual across all diagonalized operators.
        """
        err = 0.0
        for op in self.diagonalized:
            err += np.linalg.norm(op - np.diag(np.diag(op)))
        return err
