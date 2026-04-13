"""Shared cross-section / mixture builders for tests.

**Stub in PR-1** — this module re-exports the canonical XS helpers from
``orpheus.derivations._xs_library`` so test files can ``from
tests._harness.xs import make_mixture, get_mixture, get_materials`` from
day one. In later PRs the ~200 LOC of duplicated ``_make_pure_absorber_1g``,
``_make_fission_only_2g``, etc. scattered across ``test_moc_verification.py``,
``test_cp_verification.py``, and ``test_sn_*.py`` will be lifted here.

Keeping the re-export surface stable now means per-module migration PRs
only need to change ``from orpheus.derivations import ...`` to
``from tests._harness.xs import ...`` — a search-and-replace.
"""

from __future__ import annotations

from orpheus.derivations._xs_library import (
    get_materials,
    get_mixture,
    get_xs,
    make_mixture,
    validate_all,
)

__all__ = [
    "get_materials",
    "get_mixture",
    "get_xs",
    "make_mixture",
    "validate_all",
]
