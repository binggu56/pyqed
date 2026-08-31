from pathlib import Path

from pyqed.cache import file_signature, specs_equivalent


def test_file_signatures_are_portable_by_content(tmp_path):
    source = tmp_path / "source.bin"
    moved = tmp_path / "moved.bin"
    source.write_bytes(b"phenol-cache")
    moved.write_bytes(source.read_bytes())

    assert specs_equivalent(file_signature(source), file_signature(moved))


def test_file_signatures_reject_same_size_different_content(tmp_path):
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"GP")
    right.write_bytes(b"NO")

    assert not specs_equivalent(file_signature(left), file_signature(right))


def test_legacy_file_signature_requires_same_location_and_timestamp(tmp_path):
    path = tmp_path / "cache.bin"
    path.write_bytes(b"cache")
    current = file_signature(path)
    legacy = {key: current[key] for key in ("path", "size", "mtime_ns")}

    assert specs_equivalent(legacy, current)
    assert not specs_equivalent({**legacy, "path": str(tmp_path / "elsewhere")}, current)
