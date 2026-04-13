"""ORPHEUS V&V test harness: markers, registry, shared helpers, audit.

See ``docs/testing/architecture.rst`` for the full contributor guide.

Public API
----------
verify
    Class/function decorator that stamps pytest markers (``l0``/``l1``/
    ``l2``/``l3``, ``verifies``, ``catches``, ``slow``) in one call.
    Usage::

        from tests._harness import verify

        @verify.l0(equations=["transport-cartesian"], catches=["FM-07"])
        class TestSingleTrackAttenuation:
            ...

vv_cases
    Parametrize helper: auto-selects ``VerificationCase`` entries from
    the derivation registry matching a V&V level / method / geometry
    filter, and attaches the inherited level marker. Usage::

        from tests._harness import vv_cases

        @vv_cases(level="L1", method="cp", geometry="slab")
        def test_cp_slab_eigenvalue(case):
            ...

TestMetadata / TEST_REGISTRY
    In-memory V&V metadata populated by the conftest collection hook.
    One entry per pytest item keyed by ``item.nodeid``. Consumed by the
    audit CLI (``python -m tests._harness.audit``) and, via a Sphinx
    generator, by the ``docs/verification/`` matrix page.
"""

from __future__ import annotations

from tests._harness.registry import TEST_REGISTRY, TestMetadata
from tests._harness.verify import verify, vv_cases

__all__ = ["TEST_REGISTRY", "TestMetadata", "verify", "vv_cases"]
