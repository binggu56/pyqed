"""In-use malloc ranges at quiescent, single-threaded benchmark checkpoints.

macOS 26 malloc_zone_statistics includes allocator-held storage that can grow
even when every application allocation is freed. Enumerate allocated pointer
ranges instead, using the SDK's malloc introspection API. Python's mmap-backed
small-object arenas and other non-malloc mappings are not included; record RSS
and Python allocation traces separately when diagnosing those.
"""
import ctypes
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory


_directory = _probe = None


def allocated_bytes():
    global _directory, _probe
    if platform.system() != 'Darwin':
        return None
    if _probe is None:
        _directory = TemporaryDirectory(prefix='pyqed-allocation-probe-')
        library = Path(_directory.name)/'allocation_probe.dylib'
        subprocess.run(['clang','-O2','-dynamiclib',
            str(Path(__file__).with_suffix('.c')),'-o',str(library)],
            check=True,capture_output=True)
        # Keep the GIL: the C recorder must not allocate or re-enter Python.
        _probe = ctypes.PyDLL(str(library)).allocated_ranges
        _probe.argtypes = [ctypes.POINTER(ctypes.c_size_t)]*2
        _probe.restype = ctypes.c_int
    size, count = ctypes.c_size_t(), ctypes.c_size_t()
    status = _probe(ctypes.byref(size),ctypes.byref(count))
    if status:
        raise RuntimeError(f'malloc in-use enumeration failed: {status}')
    return size.value
