"""Shared mesh / geometry builders for tests.

**Stub in PR-1** — populated in later migration PRs with the
duplicated ``_ws_mesh``, ``_build_homogeneous_mesh``, and similar
helpers currently copy-pasted across ``test_moc_verification.py``,
``test_cp_verification.py``, ``test_sn_cylindrical.py``, and
``test_sn_spherical.py``.

The module exists now so import paths stabilize — migration PRs are
pure search-and-replace against ``from tests._harness.meshes import``.
"""

from __future__ import annotations

__all__: list[str] = []
