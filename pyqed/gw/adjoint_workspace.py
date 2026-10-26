"""Bounded, geometry-local storage shared by adjoint derivative calls."""
import numpy as np
from .response_storage import ResponseStore


class _Workspace(dict):
    def __deepcopy__(self, memo):
        # A copied BSE detaches on first access; do not copy large immutable
        # arrays or transfer ownership of the original object's disk files.
        return self


def workspace(bse, ids, directions, backend, settings):
    from .response import _response_key
    from .subspace_response import orbital_groups
    groups = orbital_groups(bse._scf.mo_energy, bse.nocc, settings[0])
    key = (_response_key(bse, np.zeros((1, 3*bse.mol.natom)), backend, groups),
           tuple(ids), tuple(settings))
    old = getattr(bse, '_adjoint_workspace', None)
    if old is not None and old['owner'] != id(bse):
        bse._adjoint_workspace = old = None
    requested = {direction.tobytes() for direction in directions}
    # Bound retained directions to a Cartesian basis-sized set. A new scan
    # replaces the workspace instead of accumulating old disk blocks forever.
    overflow = old is not None and len(set(old['rows']) | requested) > 3*bse.mol.natom
    if old is None or old['key'] != key or overflow:
        clear(bse)
        old = _Workspace(owner=id(bse), key=key, store=ResponseStore(), reference={}, rows={})
        bse._adjoint_workspace = old
    return old


def clear(bse):
    value = getattr(bse, '_adjoint_workspace', None)
    if value is not None:
        if value['owner'] == id(bse):
            value['store'].directory.cleanup()
        bse._adjoint_workspace = None
