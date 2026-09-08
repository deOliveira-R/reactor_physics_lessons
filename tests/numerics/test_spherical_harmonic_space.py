r"""V&V test suite for the spherical-harmonic space / basis / projection algebra.

The five identities from §P1.5 of the moment-space + layering plan
become executable tests here. ALL tests in this file carry
``@pytest.mark.catches("ERR-039")`` — they pin the post-fix contract
that the (R, Π^T, Π^*) operators are separately typed with
mathematically distinct semantics.

The ``@pytest.mark.verifies(<label>)`` markers reference equation
labels that ship in the Sphinx ``docs/theory/foundations/spherical_harmonics.rst``
page under P1.6. The labels follow the ``sh-`` prefix established by
the test-architect's verification plan so that Phase 2's ``dual-`` /
``tensor-`` / ``sum-`` labels sit alongside in the same namespace.

Pillars (per ``vv-principles`` skill):

* ``test_basis_mass_matrix_against_lebedev`` — semi-analytical (the
  Lebedev rule integrates spherical harmonics exactly up to its
  declared degree; the resulting Gram matrix is the discrete
  semi-analytical reference for the continuous identity
  :math:`\langle Y_\ell^m, Y_{\ell'}^{m'}\rangle_{L^2(S^2)} =
  (4\pi/(2\ell+1)) \delta`).
* ``test_R_equals_2l_plus_1_times_S0`` / ``test_pi_R_is_4pi_identity``
  / ``test_H_equals_g_C_times_S0`` / ``test_T_carries_w_n`` —
  closed-form (the identities are algebraic, not numerical).
* ``test_*_codomain_*`` / ``test_*_roundtrip`` / ``test_*_equality_*``
  — type-system / construction-API checks (L0 software invariants).
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.basis import SphericalHarmonicBasis
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.quadrature import (
    lebedev_sphere,
    product_mu_phi,
)
from orpheus.numerics.spaces import SphericalHarmonicSpace


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def lebedev_L_pair():
    """Pair a Lebedev rule with an L that the rule integrates exactly.

    The Lebedev rule of order ``order`` integrates
    :math:`Y_\\ell^m Y_{\\ell'}^{m'}` exactly for
    :math:`\\ell + \\ell' \\le \\mathrm{order}`. For L=3, need
    order >= 6 — pick 13 for headroom.
    """
    measure = lebedev_sphere(13)
    L = 3
    return measure, L


def _mask_non_existent_m(c: np.ndarray, L: int) -> np.ndarray:
    """Zero out the |m| > l padding entries of a (L+1, 2L+1, ...) array.

    The :math:`(L+1, 2L+1)` storage shape leaves padded slots that the
    addition-theorem identity assumes to be zero; tests use this helper
    to construct "band-limited" inputs explicitly.
    """
    out = c.copy()
    for l_idx in range(L + 1):
        out[l_idx, 2 * l_idx + 1 :] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────
# B.1 — the five P1.5 identities (one test each)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("sh-space-metric")
def test_space_inner_product_weights_equal_4pi_over_2l_plus_1():
    r"""``SphericalHarmonicSpace.from_L(L)``'s head-axis measure is :math:`4\pi/(2\ell+1)` per degree.

    The Gram-matrix diagonal :math:`g_C` lives in exactly one place —
    on the space's single :class:`~orpheus.numerics.axis.HarmonicAxis` as its
    measure (CS4c step 6 item 6.2c-ii; the legacy ``inner_product_weights``
    slot stays ``None``) — and the padded ``(L+1, 2L+1)`` layout matches the
    :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
    storage convention (row :math:`\ell` carries
    :math:`4\pi/(2\ell+1)` in the :math:`2\ell+1` valid slots, zero in
    the :math:`|m|>\ell` padding).
    """
    L = 4
    space = SphericalHarmonicSpace.from_L(L)
    expected_per_ell = 4.0 * np.pi / (2.0 * np.arange(L + 1) + 1.0)
    assert space.inner_product_weights is None and space.metric is None
    assert space.axes is not None and len(space.axes) == 1
    weights = space.axes[0].weights
    assert weights is not None
    assert weights.shape == (L + 1, 2 * L + 1)
    for ell in range(L + 1):
        np.testing.assert_allclose(
            weights[ell, : 2 * ell + 1],
            expected_per_ell[ell],
            rtol=1e-15,
        )
        np.testing.assert_array_equal(weights[ell, 2 * ell + 1 :], 0.0)


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("real-sh-discrete-orthogonality")
def test_basis_mass_matrix_against_lebedev(lebedev_L_pair):
    r"""``SphericalHarmonicBasis.mass_matrix(lebedev_measure)`` ≈ the theoretical metric.

    Pins the SH convention against a structurally-independent
    discrete-orthogonality computation: the Lebedev rule of degree
    :math:`\ge 2L` integrates :math:`Y_\ell^m Y_{\ell'}^{m'}` exactly,
    giving the Gram matrix :math:`(4\pi/(2\ell+1))\delta_{\ell\ell'}
    \delta_{m m'}` on the diagonal of the ``(L+1, 2L+1, L+1, 2L+1)``
    4-tensor and zero (to FP roundoff) on the off-diagonal.
    """
    measure, L = lebedev_L_pair
    basis = SphericalHarmonicBasis(L=L)
    G = basis.mass_matrix(measure)  # (L+1, 2L+1, L+1, 2L+1)
    assert G.shape == (L + 1, 2 * L + 1, L + 1, 2 * L + 1)

    expected_per_ell = 4.0 * np.pi / (2.0 * np.arange(L + 1) + 1.0)
    # Diagonal: G[l, l+m, l, l+m] == 4π/(2l+1) for |m| <= l.
    for ell in range(L + 1):
        for m_off in range(2 * ell + 1):
            actual = G[ell, m_off, ell, m_off]
            np.testing.assert_allclose(
                actual, expected_per_ell[ell],
                rtol=1e-12,
                err_msg=f"ell={ell}, m_off={m_off}",
            )

    # Off-diagonal (ell != ell' or m != m'): ≈ 0 to quadrature
    # precision.
    for ell in range(L + 1):
        for ell_p in range(L + 1):
            for m_off in range(2 * ell + 1):
                for m_off_p in range(2 * ell_p + 1):
                    if (ell, m_off) == (ell_p, m_off_p):
                        continue
                    np.testing.assert_allclose(
                        G[ell, m_off, ell_p, m_off_p],
                        0.0,
                        atol=1e-12,
                    )


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("sh-addition-theorem-reconstruction")
def test_R_equals_2l_plus_1_times_S0(lebedev_L_pair):
    r"""``R.apply(c) == (2\ell+1) \cdot S_0(c)`` for random band-limited c.

    The addition-theorem reconstruction :math:`R` differs from the
    naked synthesis :math:`S_0` by the per-:math:`\ell` factor
    :math:`(2\ell+1)`. This pins the ERR-039 distinction at the
    operator-construction level: ``R`` is the frame's reconstruction face,
    which reads :math:`(2\ell+1)` live from
    :attr:`SphericalHarmonicBasis.addition_theorem_factor` (single canonical
    home for the literal).
    """
    measure, L = lebedev_L_pair
    basis = SphericalHarmonicBasis(L=L)
    Y = basis.evaluate(measure.nodes)
    R = GalerkinFrame(basis, measure).reconstruction

    rng = np.random.default_rng(seed=2026)
    c = _mask_non_existent_m(rng.standard_normal((L + 1, 2 * L + 1)), L)

    actual = R.apply(c)
    # (2ℓ+1) · S_0(c) per the addition-theorem formula.
    expected = basis.synthesize(c * basis.addition_theorem_factor[:, None], Y)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)

    # Structurally-independent cross-check on unit vectors: for
    # c = e_{ℓ₀, m_off₀} (single nonzero entry at one (ℓ, m) slot),
    # the einsum collapses to a single multiplication per ordinate
    # (no accumulation), so the (2ℓ+1) literal carried by R can be
    # read off bit-identically as the per-ordinate scalar factor on
    # the column ``Y[:, ℓ, m_off]``.  No FP-non-associativity room
    # — this is the structural independence per `lessons-L11`.
    for ell_0 in range(L + 1):
        for m_off_0 in range(2 * ell_0 + 1):
            e = np.zeros((L + 1, 2 * L + 1))
            e[ell_0, m_off_0] = 1.0
            R_e = R.apply(e)
            expected_e = (2.0 * ell_0 + 1.0) * Y[:, ell_0, m_off_0]
            np.testing.assert_array_equal(
                R_e, expected_e,
                err_msg=f"R(e_{{ell={ell_0}, m_off={m_off_0}}}) "
                        f"should equal (2ℓ+1) · Y[:, ℓ, m_off] bit-identically",
            )


@pytest.mark.l1
@pytest.mark.catches("ERR-039", "ERR-051")
@pytest.mark.verifies("pi-r-equals-4pi-i")
def test_pi_R_is_4pi_identity_on_band_limited(lebedev_L_pair):
    r""":math:`\Pi \cdot R = 4\pi \cdot I` on the band-limited coefficient space.

    The canonical pin of the Galerkin idempotency law: constructs the
    analysis (:math:`\Pi`) and reconstruction (:math:`R`) operators as the
    discrete :class:`~orpheus.numerics.frame.GalerkinFrame`'s faces and pins the
    genuine :math:`\Pi R = 4\pi I` identity (NOT the broken ``Π R == I`` that
    the retired :meth:`assert_galerkin_idempotency` was checking — see P1.6).
    """
    measure, L = lebedev_L_pair
    frame = GalerkinFrame(SphericalHarmonicBasis(L=L), measure)
    M = frame.analysis
    R = frame.reconstruction

    rng = np.random.default_rng(seed=42)
    c = _mask_non_existent_m(rng.standard_normal((L + 1, 2 * L + 1)), L)

    out = M.apply(R.apply(c))
    np.testing.assert_allclose(out, 4.0 * np.pi * c, rtol=1e-10, atol=1e-12)


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("hilbert-adjoint-equals-metric-times-S0")
def test_H_equals_parseval_metric_times_S0(lebedev_L_pair):
    r"""``M.H`` computed generically equals :math:`S_0(G^{-1} c)` — the PHYSICAL adjoint.

    The frame analysis face's ``.H`` property routes through the
    generic :class:`~orpheus.numerics.operator.AdjointOperator`
    wrapper, which composes :math:`(1/w_V) \cdot \Pi^\top(w_W \cdot c)`
    using the frame's ``measure_space`` quadrature weights as
    :math:`w_V` and — since F-0 (``frame_square_recarve.md``) — the
    PARSEVAL metric :math:`G^{-1}` (the inverse discrete Gram, dressed
    onto ``basis_space`` by the frame) as :math:`w_W`. The result is
    :math:`S_0(G^{-1}c) = R(c)/W` — the physical Hilbert adjoint of the
    analysis face for the carried covariant moments.

    ⛔ Pre-F-0 the codomain metric was the CONTINUUM Gram
    :math:`g_C = 4\pi/(2\ell+1)` and this gate pinned :math:`g_C\cdot
    S_0(c)` — the wrong side for covariant moments (`[M]` Parseval ratio
    118.7 vs 1.000, ``scratch/probe_f1_parseval.py``, 2026-08-24).

    This is the ERR-039 endpoint, one correction deeper: the metric, the
    transpose, and the Hilbert adjoint are SEPARATELY TYPED, their
    composition falls out of the generic machinery — and the metric the
    space carries is now the one Parseval certifies.
    """
    measure, L = lebedev_L_pair
    basis = SphericalHarmonicBasis(L=L)
    Y = basis.evaluate(measure.nodes)
    M = GalerkinFrame(basis, measure).analysis

    rng = np.random.default_rng(seed=99)
    c = _mask_non_existent_m(rng.standard_normal((L + 1, 2 * L + 1)), L)

    actual = M.H.apply(c)
    # Expected: S_0(G⁻¹ c) with the closed-form inverse Gram (2ℓ+1)/4π
    # (exact for the degree-exact Lebedev rule — independent of the
    # frame's own discrete contraction).
    g_inv_per_ell = (2.0 * np.arange(L + 1) + 1.0) / (4.0 * np.pi)
    c_scaled = c * g_inv_per_ell[:, None]
    expected = basis.synthesize(c_scaled, Y)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────
# B.2 — constructor / API surface
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.verifies("real-sh-discrete-orthogonality")
@pytest.mark.parametrize(
    "quadrature_factory",
    [
        pytest.param(lambda: lebedev_sphere(7), id="lebedev_7"),
        pytest.param(lambda: lebedev_sphere(13), id="lebedev_13"),
        pytest.param(
            lambda: product_mu_phi(6, 8)[0],
            id="product_mu_phi_6x8",
        ),
        pytest.param(
            lambda: product_mu_phi(8, 16)[0],
            id="product_mu_phi_8x16",
        ),
    ],
)
def test_mass_matrix_under_multiple_quadratures(quadrature_factory):
    r""":meth:`SphericalHarmonicBasis.mass_matrix` is exact across SH-degree-sufficient quadratures.

    For :math:`L=2` and a quadrature that integrates the
    :math:`Y_\ell^m Y_{\ell'}^{m'}` products up to total degree
    :math:`\ge 2L = 4`, the discrete Gram diagonal equals the
    theoretical :math:`4\pi/(2\ell+1)` to FP precision. Lebedev rules
    at order :math:`\ge 2L` and product (Gauss-Legendre × equispaced
    Chebyshev-equivalent) rules with sufficient
    :math:`(n_\mu, n_\phi)` both satisfy this.

    Level-symmetric :math:`S_N` rules are not in this parametrization
    for a historical reason that no longer holds. This paragraph read
    "at L=2, LS_8 has a 24% diagonal error and no LS order makes it
    exact" — ⛔ REFUTED 2026-08-23 (F-0): `[M]` at HEAD the LS4 and LS8
    discrete Gram diagonals match :math:`4\pi/(2\ell+1)` to ~2e-15 at
    L=2 (likely the old claim predates the #327 level-symmetric
    repair; the measurement is ``test_frame.py``'s
    ``test_parseval_dressing_installed_on_diagonal_frames``, which now
    dresses and pins LS4/LS8 at L∈{1,2}). This gate keeps its original
    scope — degree-sufficient continuum-exact rules.
    """
    L = 2
    basis = SphericalHarmonicBasis(L=L)
    measure = quadrature_factory()
    G = basis.mass_matrix(measure)
    expected_per_ell = 4.0 * np.pi / (2.0 * np.arange(L + 1) + 1.0)
    for ell in range(L + 1):
        for m_off in range(2 * ell + 1):
            actual = G[ell, m_off, ell, m_off]
            np.testing.assert_allclose(
                actual, expected_per_ell[ell],
                rtol=1e-12,
                err_msg=f"ell={ell}, m_off={m_off}",
            )


@pytest.mark.foundation
def test_moment_projection_codomain_is_spherical_harmonic_space():
    r"""The frame analysis face's ``codomain`` is a typed :class:`SphericalHarmonicSpace`.

    Type-level guarantee (software invariant — tagged ``foundation``
    per ``vv-principles`` §"V&V level taxonomy") that ``M.H``
    composition via the generic adjoint machinery finds the SH metric
    correctly. Equality is STRUCTURAL (CS4c step 6): the codomain is the
    frame's Parseval-dressed head, equal to another dressing of the same
    pairing and NOT to the basis's continuum mint of the same order.

    Also confirms the face's ``domain``/``codomain`` are cached (same
    object identity on repeat access — the frame caches its spaces) —
    the `coding-elegance` Pattern 3 fix for the Krylov-inner-loop
    allocation issue QA flagged in the Phase 1 review.
    """
    measure = lebedev_sphere(7)
    L = 2
    M = GalerkinFrame(SphericalHarmonicBasis(L=L), measure).analysis
    cod = M.codomain
    assert isinstance(cod, SphericalHarmonicSpace)
    assert cod.L == L
    assert cod.shape == (L + 1, 2 * L + 1)
    assert cod == GalerkinFrame(SphericalHarmonicBasis(L=L), measure).basis_space
    assert cod != SphericalHarmonicSpace.from_L(L)   # the continuum head is another space
    assert cod.name == SphericalHarmonicSpace.from_L(L).name

    # Caching: repeated access returns the SAME object (not just an
    # equal one).  Pins the @cached_property contract — the Krylov
    # inner loop's `AdjointOperator.apply` reads codomain + domain
    # per matvec; allocating fresh spaces per access would be
    # wasted work.
    assert M.codomain is M.codomain
    assert M.domain is M.domain


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("moment-projection-transpose-T", "hilbert-adjoint-equals-metric-times-S0")
def test_T_carries_w_n_and_H_carries_the_parseval_metric(lebedev_L_pair):
    r"""``M.apply_transpose`` carries :math:`w_n`; ``M.H`` carries :math:`G^{-1}`.

    Direct pin of the post-P1.4 contract, re-keyed at F-0: the two
    operators differ by the per-:math:`\ell` factor :math:`G^{-1}/w_n`
    (not a proper scalar — it lives in different axes), and ERR-039's
    original confusion no longer arises because the two are typed
    distinctly:

    .. code-block:: python

        M.apply_transpose(c)  → w_n · S_0(c)     # representation transpose
        M.H.apply(c)          → S_0(G⁻¹c) = R(c)/W   # Hilbert adjoint (Parseval)

    Both adjoint identities below are exact BY CONSTRUCTION of the
    sandwich; the load-bearing half is the second, whose coefficient-side
    pairing must use the DRESSED (Parseval) metric or the identity fails
    by :math:`(4\pi/(2\ell+1))^2` per ℓ.
    """
    measure, L = lebedev_L_pair
    M = GalerkinFrame(SphericalHarmonicBasis(L=L), measure).analysis

    rng = np.random.default_rng(seed=7)
    psi = rng.standard_normal(measure.n_points)
    c = _mask_non_existent_m(rng.standard_normal((L + 1, 2 * L + 1)), L)

    # ⟨Π ψ, c⟩  — Euclidean on coefficient space
    Mpsi = M.apply(psi)
    lhs_euclidean = float(np.sum(Mpsi * c))

    # ⟨ψ, Π^T c⟩_V_Euclidean — Π^T already carries w_n
    rhs_T = float(np.sum(psi * M.apply_transpose(c)))
    np.testing.assert_allclose(lhs_euclidean, rhs_T, rtol=1e-12, atol=1e-14)

    # ⟨Π ψ, c⟩_C  — coefficient inner product with the PARSEVAL metric G⁻¹
    g_inv_per_ell = (2.0 * np.arange(L + 1) + 1.0) / (4.0 * np.pi)
    c_in_C = c * g_inv_per_ell[:, None]
    lhs_metric = float(np.sum(Mpsi * c_in_C))

    # ⟨ψ, Π^* c⟩_V_W  — angular inner product with W metric
    H_c = M.H.apply(c)
    rhs_H = float(np.sum(measure.weights * psi * H_c))
    np.testing.assert_allclose(lhs_metric, rhs_H, rtol=1e-12, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────
# B.1r — the reconstruction face's transpose / adjoint (symmetric with
#        the analysis-face trio above; Frame carve Phase D)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
def test_R_transpose_equals_2l_plus_1_times_S0_transpose(lebedev_L_pair):
    r"""``R.apply_transpose(v) == (2\ell+1) \cdot S_0^\top(v)`` for random nodal v.

    The reconstruction representation transpose :math:`R^\top` differs from the
    naked analysis :math:`S_0^\top` by the SAME per-:math:`\ell` factor
    :math:`(2\ell+1)` that :meth:`reconstruct` carries — and is **measure-free**
    (no :math:`w_n`), symmetric with the forward. This is the transpose-side mirror
    of :func:`test_R_equals_2l_plus_1_times_S0`; it pins the ERR-039 literal on the
    transpose path too (a missing/extra :math:`(2\ell+1)`, or a baked-in :math:`w_n`,
    reddens here).
    """
    measure, L = lebedev_L_pair
    basis = SphericalHarmonicBasis(L=L)
    Y = basis.evaluate(measure.nodes)
    R = GalerkinFrame(basis, measure).reconstruction

    rng = np.random.default_rng(seed=2027)
    v = rng.standard_normal(measure.n_points)

    actual = R.apply_transpose(v)
    # (2ℓ+1) · S_0^T(v): the dual factor enters PER-TERM (folded into the Y table),
    # matching production's in-einsum factor placement — Σ_n (2ℓ+1)·Y·v — so the
    # reference is BIT-IDENTICAL to production (asserted exactly below), yet still
    # structurally independent (explicit broadcast × 2-operand einsum, NOT the
    # 3-operand fused production call).  Post-scaling the OUTPUT instead (f·Σ vs
    # Σ(f·)) drifts ~100 ULP by FP non-associativity — the summation then runs at
    # the ×f-larger magnitude — which is why the per-term fold is the right reference.
    Y_scaled = Y * basis.addition_theorem_factor[None, :, None]
    expected = np.einsum("nlm,n->lm", Y_scaled, v)
    np.testing.assert_array_equal(actual, expected)

    # Structurally-independent cross-check on a nodal unit vector v = e_{n0}:
    # (R^T e_{n0})_{ℓ,m} = (2ℓ+1) · Y[n0, ℓ, m] — a single multiplication per
    # (ℓ, m) slot, no accumulation, so the (2ℓ+1) literal is read off
    # bit-identically (no FP-non-associativity room — lessons-L11).
    for n0 in range(measure.n_points):
        e = np.zeros(measure.n_points)
        e[n0] = 1.0
        R_T_e = R.apply_transpose(e)
        expected_e = basis.addition_theorem_factor[:, None] * Y[n0, :, :]
        np.testing.assert_array_equal(
            R_T_e, expected_e,
            err_msg=f"R^T(e_{{n={n0}}}) should equal (2ℓ+1) · Y[n0, ℓ, m] "
                    f"bit-identically",
        )


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
def test_R_transpose_carries_d_ell_and_RH_carries_d_ell_squared(lebedev_L_pair):
    r"""``R.apply_transpose`` is the Euclidean transpose; ``R.H`` the W-Hilbert adjoint.

    The reconstruction-face mirror of
    :func:`test_T_carries_w_n_and_H_carries_the_parseval_metric`, pinning BOTH
    adjoint identities by their DEFINING inner-product law (independent of how
    ``R.H`` is built):

    .. code-block:: python

        R.apply_transpose(v) → (2ℓ+1) · S_0^T(v)      # Euclidean transpose
        R.H.apply(v)         → W · Σ_n w_n Y v = W·M(v)  # Hilbert adjoint (Parseval)

    via ``⟨R c, v⟩ = ⟨c, R^T v⟩`` (Euclidean, measure-free) and
    ``⟨R c, v⟩_W = ⟨c, R^* v⟩_{G⁻¹}`` — since F-0 the coefficient side carries
    the PARSEVAL metric :math:`G^{-1} = \mathrm{diag}((2\ell+1)/4\pi)` (the
    inverse discrete Gram the frame dresses onto ``basis_space``), under which
    the sandwich collapses per ℓ to the ONE scalar :math:`d_\ell G_\ell = 4\pi
    = W`. (Pre-F-0 the pairing used the continuum :math:`g_C` and ``R.H`` read
    :math:`(2\ell+1)^2/4\pi\cdot Y^{\mathsf T}W` — the wrong-side metric.)
    """
    measure, L = lebedev_L_pair
    R = GalerkinFrame(SphericalHarmonicBasis(L=L), measure).reconstruction

    rng = np.random.default_rng(seed=11)
    c = _mask_non_existent_m(rng.standard_normal((L + 1, 2 * L + 1)), L)
    v = rng.standard_normal(measure.n_points)

    Rc = R.apply(c)  # coefficients → nodal values

    # ⟨R c, v⟩  — Euclidean on the nodal side; R^T already carries (2ℓ+1), no w_n
    lhs_euclidean = float(np.sum(Rc * v))
    rhs_T = float(np.sum(c * R.apply_transpose(v)))
    np.testing.assert_allclose(lhs_euclidean, rhs_T, rtol=1e-12, atol=1e-14)

    # ⟨R c, v⟩_W  — nodal inner product with the W (quadrature-weight) metric
    lhs_metric = float(np.sum(measure.weights * Rc * v))
    # ⟨c, R^* v⟩_{G⁻¹}  — coefficient inner product with the PARSEVAL metric
    g_inv_per_ell = (2.0 * np.arange(L + 1) + 1.0) / (4.0 * np.pi)
    H_v = R.H.apply(v)
    rhs_H = float(np.sum((c * g_inv_per_ell[:, None]) * H_v))
    np.testing.assert_allclose(lhs_metric, rhs_H, rtol=1e-12, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────
# B.3 — equality-by-(name, shape) that keeps SH space composition robust
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_spherical_harmonic_space_equality_is_structural():
    r"""SphericalHarmonicSpace equality follows :class:`FunctionSpace`'s STRUCTURAL convention (CS4c step 6).

    Two ``SphericalHarmonicSpace.from_L(L)`` instances with the same
    :math:`L` produce equal objects — their head axes carry bit-identical
    measures (the same formula, the same arithmetic), so distinct
    ``ndarray`` allocations are one axis. A different order is a different
    space, and — the identity flip — a bare hand-named ``FunctionSpace``
    wearing the head's ``(name, shape)`` is NOT the head: an axis-built
    space is never equal to a name-built one.
    """
    a = SphericalHarmonicSpace.from_L(3)
    b = SphericalHarmonicSpace.from_L(3)
    c = SphericalHarmonicSpace.from_L(4)

    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, c}) == 2   # separation through the container

    from orpheus.numerics.space import FunctionSpace
    bare = FunctionSpace(name="spherical_harmonic_space", shape=(4, 7))
    assert a != bare and bare != a
