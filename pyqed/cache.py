"""Content-aware identities for portable numerical caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path


def file_signature(path, *, block_size=1024 * 1024):
    """Return a relocatable file identity plus useful provenance fields."""
    if path is None:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _is_file_signature(value):
    return isinstance(value, Mapping) and {"path", "size"} <= set(value)


def specs_equivalent(cached, expected):
    """Compare cache specs while allowing content-identical files to move.

    Legacy signatures without a digest remain valid only at the same path and
    modification time.  Once both signatures contain SHA-256 digests, path and
    timestamp are provenance rather than identity.
    """
    if _is_file_signature(cached) and _is_file_signature(expected):
        if int(cached["size"]) != int(expected["size"]):
            return False
        cached_digest = cached.get("sha256")
        expected_digest = expected.get("sha256")
        if cached_digest is not None and expected_digest is not None:
            return cached_digest == expected_digest
        return (
            cached.get("path") == expected.get("path")
            and cached.get("mtime_ns") == expected.get("mtime_ns")
        )
    if isinstance(cached, Mapping) and isinstance(expected, Mapping):
        return set(cached) == set(expected) and all(
            specs_equivalent(cached[key], expected[key]) for key in cached
        )
    if (
        isinstance(cached, Sequence)
        and isinstance(expected, Sequence)
        and not isinstance(cached, (str, bytes))
        and not isinstance(expected, (str, bytes))
    ):
        return len(cached) == len(expected) and all(
            specs_equivalent(left, right)
            for left, right in zip(cached, expected)
        )
    return cached == expected
