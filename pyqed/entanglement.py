#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:25:31 2023

@author: bing
"""
from numpy import sqrt, sort, real
from pyqed import sigmay, ket2dm, tensor

def concurrence(rho):
    """
    Calculate the concurrence entanglement measure for a two-qubit state.

    Parameters
    ----------
    state : qobj
        Ket, bra, or density matrix for a two-qubit state.

    Returns
    -------
    concur : float
        Concurrence

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)

    """
    if rho.isket and rho.dims != [[2, 2], [1, 1]]:
        raise Exception("Ket must be tensor product of two qubits.")

    elif rho.isbra and rho.dims != [[1, 1], [2, 2]]:
        raise Exception("Bra must be tensor product of two qubits.")

    elif rho.isoper and rho.dims != [[2, 2], [2, 2]]:
        raise Exception("Density matrix must be tensor product of two qubits.")

    if rho.isket or rho.isbra:
        rho = ket2dm(rho)

    sysy = tensor(sigmay(), sigmay())

    rho_tilde = (rho * sysy) * (rho.conj() * sysy)

    evals = rho_tilde.eigenenergies()

    # abs to avoid problems with sqrt for very small negative numbers
    evals = abs(sort(real(evals)))

    lsum = sqrt(evals[3]) - sqrt(evals[2]) - sqrt(evals[1]) - sqrt(evals[0])

    return max(0, lsum)