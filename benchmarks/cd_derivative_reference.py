"""Pre-batching CD response, retained only as a validation/benchmark reference."""
from itertools import product
import numpy as np
from pyqed.qchem import basis as gaussian
from pyqed.qchem.basis_derivatives import (
    _basis_and_transform, _basis_signature, _atom_ids_for_basis,
    _contracted_eri_from_signatures,
)
from pyqed.qchem.eri_response import _directional_signatures


def columns(mol, direction):
    data = mol._cd_factor_data
    basis, _ = _basis_and_transform(mol)
    signatures = tuple(_basis_signature(fn) for fn in basis)
    owners = _atom_ids_for_basis(basis, mol.atom_coords())
    dsigs = _directional_signatures(signatures, owners, direction)
    source = data['transform']
    source = np.eye(len(basis)) if source is None else source
    components = [np.flatnonzero(column) for column in source.T]
    rows, cols = np.tril_indices(data['nao'])
    result = np.zeros((len(data['pivots']), len(rows)))
    for k, pivot in enumerate(data['pivots']):
        c, d = rows[pivot], cols[pivot]
        for ij, (a, b) in enumerate(zip(rows, cols)):
            for indices in product(components[a], components[b], components[c], components[d]):
                value = 0.
                for center in range(4):
                    for dsig in dsigs[indices[center]]:
                        args = [signatures[i] for i in indices]
                        args[center] = dsig
                        value += _contracted_eri_from_signatures(*args)
                result[k, ij] += np.prod(source[indices, (a, b, c, d)]) * value
    return result


def factor_derivative(mol, derivative_columns):
    data = mol._cd_factor_data
    pair = gaussian._as_ri_pair_factors(data['factors'], data['nao'])
    result = np.zeros_like(pair)
    for k, pivot in enumerate(data['pivots']):
        column = derivative_columns[k].copy()
        if k:
            column -= result[:k].T @ pair[:k, pivot] + pair[:k].T @ result[:k, pivot]
        droot = .5 * column[pivot] / pair[k, pivot]
        result[k] = (column - pair[k] * droot) / pair[k, pivot]
    return result
