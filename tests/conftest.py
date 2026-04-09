"""Shared fixtures for ORPHEUS verification tests."""

from __future__ import annotations

import pytest

from orpheus.derivations.reference_values import get as get_reference


@pytest.fixture
def ref():
    """Access analytical reference values by name.

    Usage in tests::

        def test_something(ref):
            case = ref("homo_1eg")
            assert abs(result.k_inf - case.k_inf) < 1e-10
    """
    return get_reference
