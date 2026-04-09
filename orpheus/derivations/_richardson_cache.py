"""Cache for Richardson-extrapolated reference eigenvalues.

Richardson extrapolation runs solvers at multiple mesh refinements,
taking ~minutes per case. This module caches the results in a JSON
file, keyed by case name with a hash of the inputs (XS, geometry,
solver settings) so the cache auto-invalidates when anything changes.

Usage in derivation modules::

    from ._richardson_cache import cached_richardson

    @cached_richardson("sn_slab_2eg_2rg", solver_params)
    def compute():
        # expensive Richardson extrapolation
        return k_ref, keffs  # float, list[float]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CACHE_FILE = Path(__file__).parent / "_richardson_cache.json"


def _hash_params(params: dict[str, Any]) -> str:
    """Deterministic hash of solver/geometry parameters."""
    raw = json.dumps(params, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    """Handle numpy arrays and other non-JSON types."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def get_cached(name: str, params: dict[str, Any]) -> float | None:
    """Return cached k_ref if the params hash matches, else None."""
    cache = _load_cache()
    entry = cache.get(name)
    if entry is None:
        return None
    if entry.get("hash") != _hash_params(params):
        return None
    return entry["k_ref"]


def store(name: str, params: dict[str, Any], k_ref: float,
          keffs: list[float]) -> None:
    """Store a Richardson result in the cache."""
    cache = _load_cache()
    cache[name] = {
        "hash": _hash_params(params),
        "k_ref": k_ref,
        "keffs": keffs,
    }
    _save_cache(cache)


def clear() -> None:
    """Remove the cache file."""
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
