"""Optional native kernels for LDR overlap matrix materialization."""

from __future__ import annotations


try:
    from . import _kernels_cpp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _kernels_cpp = None


def linked_overlap_dense(shape, links, *, nstates, average_paths=False):
    """Build the linked-overlap tensor with an optional compiled backend."""

    if _kernels_cpp is None or average_paths:
        return None

    try:
        from . import overlap as overlap_tools

        axes, indices, values = overlap_tools.pack(
            links,
            ndim=len(shape),
            nstates=int(nstates),
        )
        return _kernels_cpp.linked_overlap_dense(
            shape,
            axes,
            indices,
            values,
            int(nstates),
            average_paths=bool(average_paths),
        )
    except Exception:
        return None
