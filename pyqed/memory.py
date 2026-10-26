"""Release unused numerical allocator pages without touching live arrays."""
import ctypes
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def _numeric_allocator_relief():
    """Return the platform allocator's free-page release hook, if available."""

    try:
        if sys.platform == "darwin":
            library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            hook = library.malloc_zone_pressure_relief
            hook.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            hook.restype = ctypes.c_size_t
            return lambda: hook(None, 0)
        if sys.platform.startswith("linux"):
            library = ctypes.CDLL(None)
            hook = library.malloc_trim
            hook.argtypes = [ctypes.c_size_t]
            hook.restype = ctypes.c_int
            return lambda: hook(0)
    except (AttributeError, OSError):
        pass
    return None


def release_free_numeric_pages():
    """Return freed large numerical arenas to the OS when supported."""

    hook = _numeric_allocator_relief()
    if hook is not None:
        hook()
