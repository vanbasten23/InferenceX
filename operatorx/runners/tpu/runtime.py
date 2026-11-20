"""TPU-specific runtime probes."""
from __future__ import annotations

from importlib import metadata


def collect() -> dict[str, str]:
    """TPU software versions for RunInfo.software; {} when libtpu isn't installed.

    Reads package metadata only, without importing jax: runtime.py calls every
    platform's collect(), GPU runs included.
    """
    out: dict[str, str] = {}
    for pkg in ("libtpu", "jax", "jaxlib", "tokamax"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out if "libtpu" in out else {}
