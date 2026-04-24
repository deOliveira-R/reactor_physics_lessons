"""Diagnostic: Lesson L131a consolidation — legacy single-surface ``compute_P_esc``
and ``compute_G_bc`` still use finite-N GL for multi-region slab-polar.

Created by numerics-investigator on 2026-04-23.

Issue #131 fixed the analogous bug in ``compute_P_esc_outer/_inner`` and
``compute_G_bc_outer/_inner`` (the per-face primitives driving the shipped
``white_f4`` closure). The audit for L131a discovered that the legacy
single-surface aggregate ``compute_P_esc`` and ``compute_G_bc`` retain the
finite-N GL fallthrough for ``len(radii) > 1``. Since the per-face sum
identity

    compute_P_esc = compute_P_esc_outer + compute_P_esc_inner

holds by construction (each face gets ½ E_2(τ_face)) and both per-face
primitives now return closed-form E_2, the aggregate ALSO has a closed
form for any piecewise-constant σ_t: the sum of the two face E_2 terms.

This diagnostic proves the bug is real at multi-region slab configs and
pins the expected post-fix accuracy.

If this test PASSES after the fix (≤1e-12 rel err), promote to
``tests/derivations/test_peierls_reference.py`` next to
``TestSlabPEscPerFace::test_sum_equals_legacy_compute_P_esc`` (extend the
parametrisation to multi-region). Without the fix it should FAIL with
~4e-3 rel err at N=24 (the Issue #131 signature).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    composite_gl_r,
    compute_G_bc,
    compute_G_bc_inner,
    compute_G_bc_outer,
    compute_P_esc,
    compute_P_esc_inner,
    compute_P_esc_outer,
)


# Two-region slab: |--σt=0.5--|--σt=2.0--|, L=2.0 total
@pytest.mark.parametrize(
    "radii, sig_t",
    [
        (np.array([1.0, 2.0]), np.array([0.5, 2.0])),
        (np.array([0.3, 1.0, 1.5]), np.array([1.5, 0.2, 3.0])),
    ],
    ids=["2region_thinThick", "3region_mixed"],
)
def test_multi_region_P_esc_matches_per_face_sum(radii, sig_t):
    """Multi-region slab ``compute_P_esc`` must equal sum of per-face
    closed-form E_2 primitives to machine precision.

    Before L131a fix: finite-N GL fallthrough gives ~4e-3 rel err at N=24.
    After fix: closed-form sum, ~1e-15 rel err.
    """
    r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
    P_tot = compute_P_esc(SLAB_POLAR_1D, r_nodes, radii, sig_t, n_angular=24)
    P_out = compute_P_esc_outer(SLAB_POLAR_1D, r_nodes, radii, sig_t)
    P_in = compute_P_esc_inner(SLAB_POLAR_1D, r_nodes, radii, sig_t)
    expected = P_out + P_in

    rel_err = np.max(np.abs(P_tot - expected) / np.maximum(np.abs(expected), 1e-30))
    assert rel_err < 1e-12, (
        f"Multi-region slab compute_P_esc deviates from per-face "
        f"closed-form sum: rel_err = {rel_err:.3e}. "
        f"This is the Issue #131 signature replicated in the single-"
        f"surface aggregate."
    )


@pytest.mark.parametrize(
    "radii, sig_t",
    [
        (np.array([1.0, 2.0]), np.array([0.5, 2.0])),
        (np.array([0.3, 1.0, 1.5]), np.array([1.5, 0.2, 3.0])),
    ],
    ids=["2region_thinThick", "3region_mixed"],
)
def test_multi_region_G_bc_matches_per_face_sum(radii, sig_t):
    """Multi-region slab ``compute_G_bc`` must equal sum of per-face
    closed-form 2E_2 primitives to machine precision."""
    r_nodes, _, _ = composite_gl_r(radii, 2, 4, dps=25)
    G_tot = compute_G_bc(SLAB_POLAR_1D, r_nodes, radii, sig_t, n_surf_quad=24)
    G_out = compute_G_bc_outer(SLAB_POLAR_1D, r_nodes, radii, sig_t)
    G_in = compute_G_bc_inner(SLAB_POLAR_1D, r_nodes, radii, sig_t)
    expected = G_out + G_in

    rel_err = np.max(np.abs(G_tot - expected) / np.maximum(np.abs(expected), 1e-30))
    assert rel_err < 1e-12, (
        f"Multi-region slab compute_G_bc deviates from per-face "
        f"closed-form sum: rel_err = {rel_err:.3e}. "
        f"This is the Issue #131 signature replicated in the single-"
        f"surface aggregate."
    )
