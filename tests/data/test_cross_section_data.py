"""L0 verification of the cross-section preprocessing pipeline.

Closes the three remaining orphan equations from issue #87 by
exercising :mod:`orpheus.data.macro_xs` against synthetic ``Isotope``
fixtures with hand-computable expected values:

* ``sigma-zero``      — :func:`orpheus.data.macro_xs.sigma_zeros.solve_sigma_zeros`
* ``xs-interp``       — :func:`orpheus.data.macro_xs.interpolation.interp_xs_field`
                        and :func:`~orpheus.data.macro_xs.interpolation.interp_sig_s`
* ``number-density``  — :func:`orpheus.data.macro_xs.recipes._number_density`

The fixture builder ``_make_iso`` constructs a minimal :class:`Isotope`
with ``NG = 421`` energy groups (the hard-coded library size) but
populates only the test groups with non-trivial values; every other
group carries a constant background that is large enough to keep
``solve_sigma_zeros`` numerically stable when iterating over the full
group grid.

These tests are L0 term verification of the cross-section preprocessing
chain: they verify each formula matches a hand calculation, with no
solver or eigenvalue involved. See :ref:`vv-foundation-tests` in
``docs/testing/architecture.rst`` for the L0 vs. foundation
distinction — these are physics-equation tests (the equations live
in ``docs/theory/homogeneous.rst`` and ``cross_section_data.rst``),
not software-invariant tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.interpolation import interp_sig_s, interp_xs_field
from orpheus.data.macro_xs.recipes import _AMU_TO_G, _number_density
from orpheus.data.macro_xs.sigma_zeros import solve_sigma_zeros
from orpheus.data.micro_xs.isotope import NG, Isotope

pytestmark = pytest.mark.l0


# ── Synthetic Isotope fixture ───────────────────────────────────────────


def _make_iso(
    name: str,
    sig0_points: list[float],
    sigT_baseline: float = 1.0,
    *,
    sigT_overrides: dict[tuple[int, int], float] | None = None,
    sigC_baseline: float = 0.0,
    sigF_baseline: float = 0.0,
    aw: float = 1.0,
) -> Isotope:
    """Build a minimal ``Isotope`` for L0 cross-section tests.

    Parameters
    ----------
    name
        Isotope label (cosmetic).
    sig0_points
        Background cross-section base points in **decreasing** order
        (the convention used by the real HELIOS library and assumed by
        :func:`solve_sigma_zeros._interp_sigT`).
    sigT_baseline
        Default total cross section applied to every (sig0_idx, group)
        cell. Must be > 0 to keep ``solve_sigma_zeros`` stable when
        iterating over groups that the test does not specifically
        exercise — division-by-zero / NaN propagation otherwise.
    sigT_overrides
        Optional ``{(sig0_idx, ig): value}`` map overriding the
        baseline at specific cells. Use this to set up the exact
        values the hand calculation expects.
    sigC_baseline, sigF_baseline
        Default capture and fission cross sections. Defaults to zero
        — irrelevant for sigma-zero / interp tests.
    aw
        Atomic weight in amu. Cosmetic for these tests.

    Returns
    -------
    Isotope
        Fully populated :class:`Isotope` with ``len(sig0_points)``
        sig0 base points and ``NG`` groups.
    """
    n_sig0 = len(sig0_points)
    sigT = np.full((n_sig0, NG), sigT_baseline, dtype=float)
    if sigT_overrides:
        for (s_idx, ig), value in sigT_overrides.items():
            sigT[s_idx, ig] = value

    return Isotope(
        name=name,
        aw=aw,
        temp=293.0,
        eg=np.linspace(2e7, 1e-5, NG + 1),
        sig0=np.array(sig0_points, dtype=float),
        sigC=np.full((n_sig0, NG), sigC_baseline),
        sigL=np.zeros((n_sig0, NG)),
        sigF=np.full((n_sig0, NG), sigF_baseline),
        sigT=sigT,
        nubar=np.zeros(NG),
        chi=np.zeros(NG),
        sigS=[
            [csr_matrix((NG, NG)) for _ in range(n_sig0)]
            for _ in range(3)  # 3 Legendre orders
        ],
    )


# ── sigma-zero ──────────────────────────────────────────────────────────


@pytest.mark.verifies("sigma-zero")
def test_sigma_zero_single_base_point_two_isotopes():
    """L0: with ``n_sig0 = 1`` the iteration converges in one step.

    For each group, ``_interp_sigT`` returns the sole sigT value
    regardless of input sig0, so the fixed-point iteration

        sig0_i = (escape + sum_{j != i} N_j * sigT_j) / N_i

    has a closed-form one-shot solution. Hand calculation for two
    isotopes A and B in group 17:

        N_A = 0.04, N_B = 0.02
        sigT_A[17] = 5.0, sigT_B[17] = 8.0
        escape_xs = 0

        sig0_A[17] = N_B * sigT_B[17] / N_A = 0.02 * 8.0 / 0.04 = 4.0
        sig0_B[17] = N_A * sigT_A[17] / N_B = 0.04 * 5.0 / 0.02 = 10.0
    """
    iso_a = _make_iso(
        "A",
        sig0_points=[1e10],
        sigT_baseline=1.0,
        sigT_overrides={(0, 17): 5.0},
    )
    iso_b = _make_iso(
        "B",
        sig0_points=[1e10],
        sigT_baseline=1.0,
        sigT_overrides={(0, 17): 8.0},
    )
    n_dens = np.array([0.04, 0.02])

    sig0 = solve_sigma_zeros([iso_a, iso_b], n_dens, escape_xs=0.0)

    assert sig0.shape == (2, NG)
    assert sig0[0, 17] == pytest.approx(4.0, rel=1e-12)
    assert sig0[1, 17] == pytest.approx(10.0, rel=1e-12)


@pytest.mark.verifies("sigma-zero")
def test_sigma_zero_with_escape_cross_section():
    """L0: a nonzero escape XS adds linearly to the background.

    For a single isotope (so the cross-isotope sum is zero), the
    iteration reduces to ``sig0 = escape / N``. With escape = 0.5 and
    N = 0.1, the converged sig0 is exactly 5.0.
    """
    iso = _make_iso("X", sig0_points=[1e10], sigT_baseline=1.0)
    n_dens = np.array([0.1])

    sig0 = solve_sigma_zeros([iso], n_dens, escape_xs=0.5)

    assert np.allclose(sig0[0], 5.0, rtol=1e-12)


@pytest.mark.verifies("sigma-zero")
def test_sigma_zero_two_base_points_log_linear_iteration():
    """L0: with ``n_sig0 > 1`` the converged sig0 satisfies the
    self-consistency condition.

    The interpolation table on log10(sig0) gives:

        sigT(sig0) = sigT[a] + (log10(sig0) - log10(sig0[a])) /
                     (log10(sig0[b]) - log10(sig0[a])) *
                     (sigT[b] - sigT[a])

    For two isotopes A and B with table

        sig0_points = [10^10, 10^0]    (decreasing)
        sigT_A[42]  = [1.0,   3.0]     (high sig0 → low sigT)
        sigT_B[42]  = [2.0,   2.0]     (constant)
        N_A = 0.05, N_B = 0.05
        escape = 0

    The fixed point requires sig0_A and sig0_B such that:

        sigT_A_eff = interp(sig0_A)
        sigT_B_eff = interp(sig0_B) = 2.0 (constant)
        sig0_A     = N_B * sigT_B_eff / N_A = 1 * 2.0 = 2.0
        sig0_B     = N_A * sigT_A_eff / N_B = 1 * sigT_A_eff

    With sig0_A = 2.0, log10(2.0) ≈ 0.30103. The interpolation
    coefficient is::

        f = (log10(sig0_A) - log10(sig0[1])) /
            (log10(sig0[0]) - log10(sig0[1]))
          = (0.30103 - 0) / (10 - 0) = 0.030103

    Note: the interpolation reverses sig0 to be increasing, so the
    bracketing pair is (sigT[1]=3.0, sigT[0]=1.0) and::

        sigT_A_eff = sigT[1] + f * (sigT[0] - sigT[1])
                   = 3.0 + 0.030103 * (1.0 - 3.0)
                   = 3.0 - 0.060206
                   ≈ 2.939794

    Then sig0_B = 1 * 2.939794. The test pins both to 1e-6 absolute
    tolerance against this hand calculation.
    """
    iso_a = _make_iso(
        "A",
        sig0_points=[1e10, 1e0],
        sigT_baseline=1.0,
        sigT_overrides={(0, 42): 1.0, (1, 42): 3.0},
    )
    iso_b = _make_iso(
        "B",
        sig0_points=[1e10, 1e0],
        sigT_baseline=1.0,
        sigT_overrides={(0, 42): 2.0, (1, 42): 2.0},
    )
    n_dens = np.array([0.05, 0.05])

    sig0 = solve_sigma_zeros([iso_a, iso_b], n_dens, escape_xs=0.0)

    expected_a = 2.0
    log_target = np.log10(expected_a)
    f = (log_target - 0.0) / (10.0 - 0.0)  # log10 fraction in reversed space
    expected_sigT_a_eff = 3.0 + f * (1.0 - 3.0)
    expected_b = expected_sigT_a_eff

    assert sig0[0, 42] == pytest.approx(expected_a, abs=1e-6)
    assert sig0[1, 42] == pytest.approx(expected_b, abs=1e-6)


# ── xs-interp ───────────────────────────────────────────────────────────


@pytest.mark.verifies("xs-interp")
def test_interp_xs_field_single_base_point_returns_field_unchanged():
    """L0: ``n_sig0 = 1`` is the no-interp short-circuit.

    The function must return a copy of ``field[0]`` regardless of
    the requested sig0 (no division by zero on the log10 of a single
    base point, no off-by-one).
    """
    iso = _make_iso("X", sig0_points=[1e10], sigT_baseline=2.0)
    field = iso.sigT  # shape (1, NG)
    sig0_target = np.full(NG, 1e5)

    result = interp_xs_field(field, iso, sig0_target)

    assert result.shape == (NG,)
    np.testing.assert_array_equal(result, field[0])
    # Must be a copy, not a view — the caller mutates the returned array.
    assert result is not field[0]


@pytest.mark.verifies("xs-interp")
def test_interp_xs_field_at_base_points_recovers_table_values():
    """L0: querying at exactly a base-point sig0 returns that row.

    With sig0_points = [1e10, 1e0] and target sig0 = 1e10 in every
    group, the interpolation must return the high-sig0 row (the
    first row of ``field`` because the table is in decreasing
    order). Symmetrically, target sig0 = 1e0 returns the second row.
    """
    iso = _make_iso(
        "X",
        sig0_points=[1e10, 1e0],
        sigT_baseline=1.0,
        sigT_overrides={(0, 100): 5.0, (1, 100): 9.0},
    )

    high = interp_xs_field(iso.sigT, iso, np.full(NG, 1e10))
    low = interp_xs_field(iso.sigT, iso, np.full(NG, 1e0))

    assert high[100] == pytest.approx(5.0)
    assert low[100] == pytest.approx(9.0)


@pytest.mark.verifies("xs-interp")
def test_interp_xs_field_log_linear_at_geometric_mean():
    """L0: query at the log10 midpoint returns the arithmetic
    average of the two endpoint values.

    For sig0_points = [1e10, 1e0] (log10 endpoints 10 and 0), the
    geometric midpoint is sig0 = 1e5 (log10 = 5). Linear interp on
    log10(sig0) at fraction 0.5 returns (sigT_low + sigT_high) / 2
    in every group.
    """
    iso = _make_iso(
        "X",
        sig0_points=[1e10, 1e0],
        sigT_baseline=1.0,
        sigT_overrides={(0, 200): 4.0, (1, 200): 10.0},
    )

    result = interp_xs_field(iso.sigT, iso, np.full(NG, 1e5))

    assert result[200] == pytest.approx((4.0 + 10.0) / 2.0, rel=1e-12)


@pytest.mark.verifies("xs-interp")
def test_interp_sig_s_log_linear_at_geometric_mean():
    """L0: scattering-matrix interpolation matches the scalar
    log-linear formula entry-by-entry.

    Builds a tiny 2-entry sparse matrix at each of two sig0 base
    points, queries at the log10 midpoint, and asserts each non-zero
    interpolates to the average of its two endpoint values.
    """
    n_sig0 = 2
    iso = _make_iso("X", sig0_points=[1e10, 1e0])

    # Override sigS[legendre=0] for a single-entry matrix at each base point.
    # Pattern: nonzero scatter from group 50 → group 51, value differs.
    rows = np.array([50])
    cols = np.array([51])
    vals_high = np.array([0.2])  # sig0 = 1e10
    vals_low = np.array([0.8])  # sig0 = 1e0
    iso.sigS[0][0] = csr_matrix((vals_high, (rows, cols)), shape=(NG, NG))
    iso.sigS[0][1] = csr_matrix((vals_low, (rows, cols)), shape=(NG, NG))

    # Note the convention: the from-group's sig0 drives the lookup
    # (see interp_sig_s body — `log_target[ifrom[i]]`). Set the target
    # sig0 row to 1e5 in group 50 (the from-group), arbitrary
    # elsewhere.
    sig0_target = np.full(NG, 1e10)
    sig0_target[50] = 1e5

    result = interp_sig_s(iso, legendre=0, sig0_row=sig0_target)

    assert result[50, 51] == pytest.approx((0.2 + 0.8) / 2.0, rel=1e-12)


@pytest.mark.verifies("xs-interp")
def test_interp_sig_s_single_base_point_returns_copy():
    """L0: ``n_sig0 = 1`` short-circuit returns the sole sparse matrix."""
    iso = _make_iso("X", sig0_points=[1e10])
    rows = np.array([10])
    cols = np.array([20])
    vals = np.array([0.7])
    iso.sigS[0][0] = csr_matrix((vals, (rows, cols)), shape=(NG, NG))

    result = interp_sig_s(iso, legendre=0, sig0_row=np.full(NG, 1e3))

    assert result[10, 20] == pytest.approx(0.7)
    # Must be a copy, not a view — downstream code mutates the matrix.
    assert result is not iso.sigS[0][0]


# ── number-density ──────────────────────────────────────────────────────


@pytest.mark.verifies("number-density")
def test_number_density_formula_against_hand_calc():
    """L0: ``N_i = rho_i / (m_u * A_i)`` per Eq. ``number-density``.

    The implementation in :func:`orpheus.data.macro_xs.recipes._number_density`
    converts ``g/cm^3`` to ``g/(barn*cm)`` via the 1e-24 cm^2/barn
    factor before dividing. Hand calculation for water:

        rho      = 1.0 g/cm^3
        A_water  = 18.0 amu
        m_u      = 1.660538e-24 g/amu
        N_water  = 1.0 * 1e-24 / (1.660538e-24 * 18.0)
                ≈ 0.033456e0  /(barn*cm)

    The atomic-mass-unit constant is imported directly from
    ``recipes`` to keep the test in lockstep with the implementation
    if the constant is ever refined to more decimal places.
    """
    rho = 1.0  # g/cm^3
    A = 18.0  # amu

    expected = rho * 1e-24 / (_AMU_TO_G * A)
    actual = _number_density(rho, A)

    assert actual == pytest.approx(expected, rel=1e-15)
    # Also check the order of magnitude against a published value
    # (water at standard density is ~0.0334 atoms/(barn*cm)).
    assert actual == pytest.approx(0.0334, rel=2e-3)


@pytest.mark.verifies("number-density")
def test_number_density_scales_linearly_with_density():
    """L0: the formula is linear in density at fixed atomic weight.

    Doubling rho doubles N, halving rho halves N — no hidden
    nonlinearity. This guards against accidental introduction of a
    log or exp in the density-conversion path.
    """
    A = 235.0  # uranium-235 atomic weight
    n1 = _number_density(10.0, A)
    n2 = _number_density(20.0, A)
    n_half = _number_density(5.0, A)

    assert n2 == pytest.approx(2.0 * n1, rel=1e-15)
    assert n_half == pytest.approx(0.5 * n1, rel=1e-15)


@pytest.mark.verifies("number-density")
def test_number_density_inversely_proportional_to_atomic_weight():
    """L0: doubling A halves N at fixed density.

    This guards against an accidental ``A^2`` or ``sqrt(A)``
    substitution in the denominator.
    """
    rho = 10.0
    n_a18 = _number_density(rho, 18.0)
    n_a36 = _number_density(rho, 36.0)

    assert n_a36 == pytest.approx(0.5 * n_a18, rel=1e-15)
