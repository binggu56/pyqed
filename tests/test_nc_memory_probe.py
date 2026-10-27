"""The stability meter must distinguish allocated buffers from allocator caches."""
import ctypes
import gc
import platform
import pytest
from examples.qchem._allocation_probe import allocated_bytes

pytestmark=pytest.mark.skipif(platform.system()!='Darwin',reason='macOS malloc introspection')


def functions():
    lib=ctypes.CDLL(None)
    malloc,free=lib.malloc,lib.free
    malloc.argtypes=[ctypes.c_size_t];malloc.restype=ctypes.c_void_p
    free.argtypes=[ctypes.c_void_p];free.restype=None
    allocated_bytes();gc.collect()
    return malloc,free


def test_memory_probe_counts_live_buffer_and_release():
    malloc,free=functions();before=allocated_bytes()
    size=4*1024**2;pointer=malloc(size)
    assert pointer
    try:
        assert allocated_bytes()-before>=size
    finally:
        free(pointer)
    assert allocated_bytes()-before<128*1024


def test_memory_probe_does_not_count_freed_allocation_churn():
    malloc,free=functions();before=allocated_bytes()
    for i in range(1000):
        pointer=malloc(1024**2+(i%100)*2032)
        assert pointer
        free(pointer)
    gc.collect()
    assert allocated_bytes()-before<128*1024
