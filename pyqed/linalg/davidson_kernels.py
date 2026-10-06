"""Optional dense and matrix-free Davidson kernels, independent of MPS."""
from pathlib import Path
from ._extension import load_extension

_directory=Path(__file__).parent
_module,build_error=load_extension('pyqed.linalg._davidson',
    _directory/'davidson_module.cpp',
    [_directory/'davidson.hpp',_directory/'davidson_binding.hpp'])
davidson=getattr(_module,'davidson',None)
davidson_operator=getattr(_module,'davidson_operator',None)
