"""L0 parity tests for the unified flat-source CP builder.

Phase B.2a / B.4 gate: the unified :func:`~cp_geometry.build_cp_matrix`
reproduces the pre-refactor ``_slab_cp_matrix``, ``_cylinder_cp_matrix``,
and ``_sphere_cp_matrix`` outputs. As of B.4, the three facade
functions *delegate* to ``build_cp_matrix``, so these parity tests
become self-consistency checks (any call-site bug would surface
here before the end-to-end eigenvalue tests).

Kernel sources on each geometry:

- Slab: scipy ``expn(3, x)`` via :func:`_kernels.e3_vec`.
- Cylinder: Chebyshev interpolant of :math:`e^{\\tau}\\,\\mathrm{Ki}_3(\\tau)`
  built from :func:`_kernels.ki_n_mp` at 30 dps (~:math:`10^{-6}`
  absolute). Replaces the retired :class:`_kernels.BickleyTables`.
- Sphere: ``np.exp(-tau)``."""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations import cp_cylinder as _legacy_cyl
from orpheus.derivations import cp_slab as _legacy_slab
from orpheus.derivations import cp_sphere as _legacy_sph
from orpheus.derivations.cp_geometry import (
    CYLINDER_1D,
    SLAB,
    SPHERE_1D,
    build_cp_matrix,
)
from orpheus.derivations._xs_library import LAYOUTS, get_xs


# ═══════════════════════════════════════════════════════════════════════
# Helpers — materialise the same test inputs each facade uses
# ═══════════════════════════════════════════════════════════════════════

def _slab_inputs(ng_key: str, n_regions: int):
    layout = LAYOUTS[n_regions]
    t_arr = np.array(_legacy_slab._THICKNESSES[n_regions], dtype=float)
    sig_t_all = np.vstack([get_xs(r, ng_key)["sig_t"] for r in layout])
    return sig_t_all, t_arr


def _cyl_inputs(ng_key: str, n_regions: int):
    layout = LAYOUTS[n_regions]
    radii = np.array(_legacy_cyl._RADII[n_regions], dtype=float)
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = np.pi * (radii ** 2 - r_inner ** 2)
    sig_t_all = np.vstack([get_xs(r, ng_key)["sig_t"] for r in layout])
    return sig_t_all, radii, volumes, float(radii[-1])


def _sph_inputs(ng_key: str, n_regions: int):
    layout = LAYOUTS[n_regions]
    radii = np.array(_legacy_sph._RADII[n_regions], dtype=float)
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = (4.0 / 3.0) * np.pi * (radii ** 3 - r_inner ** 3)
    sig_t_all = np.vstack([get_xs(r, ng_key)["sig_t"] for r in layout])
    return sig_t_all, radii, volumes, float(radii[-1])


# ═══════════════════════════════════════════════════════════════════════
# Slab: bit-identity (same scipy kernel on both sides)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies(
    "cp-flat-source-double-integral",
    "cp-flat-source-derivation",
    "cp-unified-outer-integration",
    "dd-slab",
    "dc-slab",
    "e3-def",
)
@pytest.mark.parametrize("ng_key", ["1g", "2g", "4g"])
@pytest.mark.parametrize("n_regions", [1, 2, 4])
def test_unified_slab_matches_legacy(ng_key, n_regions):
    """Unified slab P_inf matches legacy ``_slab_cp_matrix`` to machine
    precision (same scipy ``expn(3, x)`` on both sides)."""
    sig_t_all, t_arr = _slab_inputs(ng_key, n_regions)
    R_cell = float(t_arr.sum())

    legacy = _legacy_slab._slab_cp_matrix(sig_t_all, t_arr)
    unified = build_cp_matrix(
        SLAB, sig_t_all, t_arr, t_arr, R_cell, n_quad_y=64,
    )

    np.testing.assert_allclose(
        unified, legacy, rtol=1e-12, atol=1e-14,
        err_msg=f"slab {ng_key} {n_regions}rg unified vs legacy drift",
    )


# ═══════════════════════════════════════════════════════════════════════
# Cylinder: facade-over-unified consistency (same code path on both sides)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies(
    "cp-flat-source-double-integral",
    "cp-flat-source-derivation",
    "cp-unified-outer-integration",
    "second-diff-cyl",
    "ki3-def",
)
@pytest.mark.parametrize("ng_key", ["1g", "2g", "4g"])
@pytest.mark.parametrize("n_regions", [1, 2, 4])
def test_unified_cylinder_matches_legacy(ng_key, n_regions):
    """Unified cylinder P_inf matches the cp_cylinder facade (which is
    itself a delegation to ``build_cp_matrix``). Bit-identity guards
    against drift in the facade wiring or accidental alternate call
    paths for the cylinder kernel."""
    sig_t_all, radii, volumes, R_cell = _cyl_inputs(ng_key, n_regions)

    legacy = _legacy_cyl._cylinder_cp_matrix(
        sig_t_all, radii, volumes, R_cell, n_quad_y=64,
    )
    unified = build_cp_matrix(
        CYLINDER_1D, sig_t_all, radii, volumes, R_cell, n_quad_y=64,
    )

    np.testing.assert_allclose(
        unified, legacy, rtol=1e-12, atol=1e-14,
        err_msg=f"cylinder {ng_key} {n_regions}rg unified vs legacy drift",
    )


# ═══════════════════════════════════════════════════════════════════════
# Sphere: bit-identity (same np.exp on both sides)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies(
    "cp-flat-source-double-integral",
    "cp-flat-source-derivation",
    "cp-unified-outer-integration",
    "second-diff-sph",
)
@pytest.mark.parametrize("ng_key", ["1g", "2g", "4g"])
@pytest.mark.parametrize("n_regions", [1, 2, 4])
def test_unified_sphere_matches_legacy(ng_key, n_regions):
    """Unified sphere P_inf matches legacy ``_sphere_cp_matrix`` to
    machine precision (same ``np.exp(-tau)`` on both sides)."""
    sig_t_all, radii, volumes, R_cell = _sph_inputs(ng_key, n_regions)

    legacy = _legacy_sph._sphere_cp_matrix(
        sig_t_all, radii, volumes, R_cell, n_quad_y=64,
    )
    unified = build_cp_matrix(
        SPHERE_1D, sig_t_all, radii, volumes, R_cell, n_quad_y=64,
    )

    np.testing.assert_allclose(
        unified, legacy, rtol=1e-12, atol=1e-14,
        err_msg=f"sphere {ng_key} {n_regions}rg unified vs legacy drift",
    )


# ═══════════════════════════════════════════════════════════════════════
# Δ² operator geometry-invariance sanity check
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("cp-second-difference-operator")
class TestSecondDifferenceOperator:
    """The Δ² operator is geometry-invariant: it depends on the kernel
    callable and three τ arguments only. Same inputs → same output
    regardless of which geometry 'owns' the kernel."""

    def test_identity_holds_for_all_three_kernels(self):
        from orpheus.derivations.cp_geometry import _second_difference

        gap = np.array([0.5, 1.0, 2.0])
        tau_i = np.array([0.3, 0.7, 1.1])
        tau_j = np.array([0.2, 0.4, 0.9])

        for geom in (SLAB, CYLINDER_1D, SPHERE_1D):
            direct = (geom.kernel_F3(gap)
                      - geom.kernel_F3(gap + tau_i)
                      - geom.kernel_F3(gap + tau_j)
                      + geom.kernel_F3(gap + tau_i + tau_j))
            via_op = _second_difference(
                geom.kernel_F3, gap, tau_i, tau_j,
            )
            np.testing.assert_allclose(
                via_op, direct, rtol=1e-14, atol=1e-14,
                err_msg=f"Δ² operator mismatch for {geom.kind}",
            )

    def test_degenerate_zero_tau_gives_zero(self):
        from orpheus.derivations.cp_geometry import _second_difference

        gap = np.array([0.5, 1.0, 2.0])
        zero = np.zeros_like(gap)
        for geom in (SLAB, CYLINDER_1D, SPHERE_1D):
            val = _second_difference(geom.kernel_F3, gap, zero, zero)
            np.testing.assert_allclose(val, 0.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Inner- and outer-integral antiderivative identities (§12)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("cp-inner-integral-antiderivative")
class TestInnerIntegralAntiderivative:
    r"""The inner (source-region) integration promotes the Level-1
    kernel to the Level-2 antiderivative:
    :math:`\int_a^b E_1\,\mathrm d\tau = E_2(a) - E_2(b)`,
    :math:`\int_a^b \mathrm{Ki}_1\,\mathrm d\tau = \mathrm{Ki}_2(a) - \mathrm{Ki}_2(b)`,
    :math:`\int_a^b e^{-\tau}\,\mathrm d\tau = e^{-a} - e^{-b}`.

    The differential form :math:`\frac{\mathrm d E_2}{\mathrm d\tau}
    = -E_1` (and its :math:`\mathrm{Ki}_n` analog) is pinned at
    :func:`tests.derivations.test_kernels.test_en_derivative_identity`
    and :func:`~.test_kin_derivative_identity`. This class pins the
    integrated form by endpoint evaluation — a second, independent
    check of the same identity."""

    @pytest.mark.parametrize("a,b", [(0.1, 1.0), (0.5, 3.0), (1.0, 5.0)])
    def test_e1_integral_to_e2(self, a, b):
        import mpmath
        with mpmath.workdps(30):
            left = float(mpmath.quad(
                lambda x: mpmath.expint(1, x), [a, b],
            ))
            right = float(mpmath.expint(2, a) - mpmath.expint(2, b))
        assert left == pytest.approx(right, abs=1e-12)

    @pytest.mark.parametrize("a,b", [(0.1, 1.0), (0.5, 3.0)])
    def test_ki1_integral_to_ki2(self, a, b):
        from orpheus.derivations._kernels import ki_n_mp
        import mpmath
        # Single mpmath.quad over the already-defined Ki_1 function —
        # each evaluation is one high-precision Ki_1 call, so the
        # whole test takes ~0.5s.
        left = float(mpmath.quad(
            lambda x: ki_n_mp(1, float(x), 20), [a, b],
        ))
        right = float(ki_n_mp(2, a, 30)) - float(ki_n_mp(2, b, 30))
        assert left == pytest.approx(right, abs=1e-8)

    @pytest.mark.parametrize("a,b", [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)])
    def test_exp_integral_to_exp(self, a, b):
        # ∫ exp(-x) dx = exp(-a) - exp(-b) (closed-form)
        import numpy as np
        left_analytical = np.exp(-a) - np.exp(-b)
        import mpmath
        with mpmath.workdps(30):
            left_quad = float(mpmath.quad(
                lambda x: mpmath.exp(-x), [a, b],
            ))
        assert left_quad == pytest.approx(left_analytical, abs=1e-14)


@pytest.mark.l0
@pytest.mark.verifies("cp-outer-integral-antiderivative")
class TestOuterIntegralAntiderivative:
    r"""The outer (target-region) integration promotes Level-2 to
    Level-3: :math:`\int_a^b E_2\,\mathrm d\tau = E_3(a) - E_3(b)`,
    :math:`\int_a^b \mathrm{Ki}_2\,\mathrm d\tau =
    \mathrm{Ki}_3(a) - \mathrm{Ki}_3(b)`, sphere unchanged."""

    @pytest.mark.parametrize("a,b", [(0.1, 1.0), (0.5, 3.0), (1.0, 5.0)])
    def test_e2_integral_to_e3(self, a, b):
        import mpmath
        with mpmath.workdps(30):
            left = float(mpmath.quad(
                lambda x: mpmath.expint(2, x), [a, b],
            ))
            right = float(mpmath.expint(3, a) - mpmath.expint(3, b))
        assert left == pytest.approx(right, abs=1e-12)

    @pytest.mark.parametrize("a,b", [(0.1, 1.0), (0.5, 3.0)])
    def test_ki2_integral_to_ki3(self, a, b):
        from orpheus.derivations._kernels import ki_n_mp
        import mpmath
        left = float(mpmath.quad(
            lambda x: ki_n_mp(2, float(x), 20), [a, b],
        ))
        right = float(ki_n_mp(3, a, 30)) - float(ki_n_mp(3, b, 30))
        assert left == pytest.approx(right, abs=1e-8)


# ═══════════════════════════════════════════════════════════════════════
# P_esc / white-BC closure (§15 cp-escape-from-p-cell)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("cp-escape-from-p-cell")
class TestEscapeFromPCell:
    r"""Row-sum escape identity: for each region :math:`i`,
    :math:`P_{{\rm out},i} = 1 - \sum_j P_{{\rm cell},ij}` before the
    white-BC closure, i.e. each row of :math:`P_{\rm cell}` plus its
    escape probability sums to 1 in the no-re-entry limit.

    In the full white-BC :math:`P_\infty`, the row sum equals 1
    exactly (neutron conservation: every neutron born in region
    :math:`i` eventually collides somewhere in the infinite lattice)."""

    def test_white_bc_row_sum_equals_one_cylinder(self):
        import numpy as np
        from orpheus.derivations.cp_geometry import CYLINDER_1D, build_cp_matrix

        # Simple 2-region, 1-group problem
        sig_t_all = np.array([[1.0], [0.5]])
        radii = np.array([0.5, 1.0])
        r_inner = np.array([0.0, 0.5])
        volumes = np.pi * (radii ** 2 - r_inner ** 2)
        P_inf = build_cp_matrix(
            CYLINDER_1D, sig_t_all, radii, volumes, 1.0, n_quad_y=64,
        )
        row_sums = P_inf[:, :, 0].sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="White-BC row-sum identity broken for cylinder",
        )

    def test_white_bc_row_sum_equals_one_sphere(self):
        import numpy as np
        from orpheus.derivations.cp_geometry import SPHERE_1D, build_cp_matrix

        sig_t_all = np.array([[1.0], [0.5]])
        radii = np.array([0.5, 1.0])
        r_inner = np.array([0.0, 0.5])
        volumes = (4.0 / 3.0) * np.pi * (radii ** 3 - r_inner ** 3)
        P_inf = build_cp_matrix(
            SPHERE_1D, sig_t_all, radii, volumes, 1.0, n_quad_y=64,
        )
        row_sums = P_inf[:, :, 0].sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="White-BC row-sum identity broken for sphere",
        )

    def test_white_bc_row_sum_equals_one_slab(self):
        import numpy as np
        from orpheus.derivations.cp_geometry import SLAB, build_cp_matrix

        sig_t_all = np.array([[1.0], [0.5]])
        t_arr = np.array([0.5, 0.5])
        P_inf = build_cp_matrix(
            SLAB, sig_t_all, t_arr, t_arr, 1.0,
        )
        row_sums = P_inf[:, :, 0].sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="White-BC row-sum identity broken for slab",
        )


# ═══════════════════════════════════════════════════════════════════════
# Cylinder Ki_3 Chebyshev interpolant accuracy (Phase B.4)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
class TestKi3ChebyshevInterpolant:
    """After Phase B.4, the cylinder kernel uses a Chebyshev interpolant
    of ``exp(tau) * Ki_3(tau)`` built from ``ki_n_mp(3, ·, 30)`` at
    module load — the legacy :class:`BickleyTables` retired. Accuracy
    target: ~1e-6 absolute on ``[0, 50]`` (vs legacy ~1e-3)."""

    def test_agrees_with_mpmath(self):
        from orpheus.derivations._kernels import ki_n_mp
        from orpheus.derivations.cp_geometry import _ki3_mp

        probes = np.linspace(0.01, 48.0, 40)
        cheb = _ki3_mp(probes)
        mp = np.array([float(ki_n_mp(3, float(t), 30)) for t in probes])
        err = np.abs(cheb - mp).max()
        assert err < 5e-6, (
            f"Ki_3 Chebyshev interpolant err {err:.3e} > 5e-6"
        )

    def test_ki3_at_zero_matches_pi_over_four(self):
        from orpheus.derivations.cp_geometry import _ki3_mp
        val = _ki3_mp(np.array([0.0]))[0]
        assert abs(val - np.pi / 4.0) < 5e-6

    def test_clamps_beyond_tau_max_to_zero(self):
        from orpheus.derivations.cp_geometry import _ki3_mp
        far = np.array([60.0, 100.0, 1e6])
        np.testing.assert_allclose(_ki3_mp(far), 0.0, atol=1e-20)
