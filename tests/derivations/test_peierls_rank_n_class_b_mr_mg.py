"""Class B (solid cyl/sph) rank-N white-BC closure — multi-region × multi-group
verification posture.

Created 2026-04-25 closing the plan at
``.claude/plans/issue-100-103-rank-n-class-b-multi-region.md``. The plan
hypothesised three outcomes (clean falsification extension / hidden bug /
rank-N actually wins). The investigation landed on **outcome B (hidden
bug)**: the rank-N Marshak closure is broken in MR×MG on Class B because
``build_closure_operator`` routes mode 0 through the legacy
``compute_P_esc/compute_G_bc`` (no surface Jacobian) and modes ``n ≥ 1``
through ``compute_P_esc_mode/compute_G_bc_mode`` (with the
:math:`(\\rho_{\\max}/R)^2` surface-to-observer Jacobian). The two
normalisations live in different partial-current expansion spaces; in
1R the mismatch happens to land near zero by historical calibration of
``compute_P_esc`` against the Mark closure, but in MR with a strong
outer scatterer the mismatch amplifies to ~+57 % k_eff error
(sphere 1G/2R fuel-A inner / moderator-B outer). See Issue #132 for
the full diagnosis and the re-derivation work tracked there.

Tests in this file
------------------

1. **Sanity baselines** (1G/1R) — reproduce the published 2026-04-18
   Issue #112 table (``build_white_bc_correction_rank_n`` docstring,
   ``peierls_geometry.py`` lines 3934-3961). Loose absolute tolerances
   on ``|k_eff − k_inf|/k_inf`` capture the well-known rank-1 Mark
   floor (~21 % cyl, ~27 % sph) and the rank-2 step. Pipeline gate.

2. **MR routing invariance** (Probe C promoted) — sphere/cylinder with
   ``radii=[0.5, 1.0]`` and uniform :math:`\\Sigma_t = 1` must give
   ``k_eff`` matching the true 1R ``radii=[1.0]`` solve. Tests that
   the multi-region routing path is consistent with single-region
   when the σ_t profile is in fact uniform (Issue #131-style anti-
   pattern audit). Tolerance set to 5e-3 to absorb the Issue #114
   ρ-quadrature floor.

3. **Class B MR catastrophe** (Probe G promoted, xfailed) — pins the
   +57 % sphere 1G/2R rank-2 sign-flip catastrophe to Issue #132. The
   xfail flips when Issue #132 lands, alerting the developer.

4. **Class B 2G/2R rank-1 floor** — pins the rank-1 Mark closure
   behaviour at 2G/2R for cyl and sph. The values are far from
   ``k_inf`` (~80 % low) because of MG amplification of the thin
   group-0 closure error, but they are reproducible. Regression
   gate against further drift while Issue #132 is open.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations import cp_cylinder, cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SPHERE_1D,
    solve_peierls_mg,
)


pytestmark = [pytest.mark.verifies("peierls-rank-n-bc-closure")]


# ═══════════════════════════════════════════════════════════════════════
# Quadrature presets — codifies the L19 BASE / RICH posture from the plan
# ═══════════════════════════════════════════════════════════════════════
_QUAD_BASE = dict(
    n_panels_per_region=2,
    p_order=3,
    n_angular=24,
    n_rho=24,
    n_surf_quad=24,
    dps=15,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def _build_xs_arrays(ng_key: str, n_regions: int):
    """Mirror cp_cylinder._build_case body — return the four MG XS arrays."""
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    return dict(
        sig_t=np.vstack([xs["sig_t"] for xs in xs_list]),
        sig_s=np.stack([xs["sig_s"] for xs in xs_list], axis=0),
        nu_sig_f=np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list]),
        chi=np.vstack([xs["chi"] for xs in xs_list]),
    )


def _solve_class_b(geometry, ng_key: str, n_regions: int, n_bc_modes: int,
                   *, quad: dict):
    """Run ``solve_peierls_mg`` on the cp_{cylinder,sphere}._build_case
    geometry/XS so the analytical ``k_inf`` reference applies bit-aligned."""
    cp_module = cp_cylinder if geometry is CYLINDER_1D else cp_sphere
    xs = _build_xs_arrays(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    sol = solve_peierls_mg(
        geometry, radii=radii, **xs,
        boundary="white_rank1_mark", n_bc_modes=n_bc_modes,
        **quad,
    )
    return sol.k_eff


def _kinf_ref(geometry, ng_key: str, n_regions: int) -> float:
    cp_module = cp_cylinder if geometry is CYLINDER_1D else cp_sphere
    return float(cp_module._build_case(ng_key, n_regions).k_inf)


def _solve_uniform_sigma(geometry, radii, sig_t_per_region, n_bc_modes,
                         *, quad=_QUAD_BASE):
    """Solve sphere/cyl with explicit per-region σ_t but A's scatter/fission
    in every region (used by the routing-invariance probe — material
    properties are uniform, only σ_t and the radii topology vary)."""
    n_regions = len(radii)
    xs_A = get_xs("A", "1g")
    sig_t = np.array(sig_t_per_region, dtype=float).reshape(n_regions, 1)
    sig_s = np.stack([xs_A["sig_s"]] * n_regions, axis=0)
    nu_sig_f = np.vstack([xs_A["nu"] * xs_A["sig_f"]] * n_regions)
    chi = np.vstack([xs_A["chi"]] * n_regions)
    sol = solve_peierls_mg(
        geometry, radii=np.array(radii, dtype=float), sig_t=sig_t,
        sig_s=sig_s, nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=n_bc_modes,
        **quad,
    )
    return sol.k_eff


_GEOMETRIES = [
    pytest.param(CYLINDER_1D, id="cylinder-1d"),
    pytest.param(SPHERE_1D, id="sphere-1d"),
]


# ═══════════════════════════════════════════════════════════════════════
# 1. Sanity baselines — reproduce Issue #112 published 1G/1R rank-N table
# ═══════════════════════════════════════════════════════════════════════
#
# Published values (build_white_bc_correction_rank_n docstring lines 3934-3961,
# fixed 2026-04-18):
#
#   Geometry  R[MFP]   N=1     N=2     N=3
#   Cylinder    1.0    20.9 %   8.3 %  26.7 % (cyl divergence per Issue #112)
#   Sphere      1.0    26.9 %   1.22 % 2.5 %
#
# Tolerance bands chosen to gate any drift > 2 percentage points from
# the published value while accommodating quadrature differences
# between the docstring's reported run and BASE preset here.

_PUBLISHED_TABLE_1G_1R = {
    # (geometry_id, n_bc_modes) -> expected |err| % +/- 2 pp
    ("cylinder-1d", 1): 21.0,
    ("cylinder-1d", 2): 8.3,
    ("cylinder-1d", 3): 26.7,
    ("sphere-1d", 1): 27.0,
    ("sphere-1d", 2): 1.2,
    ("sphere-1d", 3): 2.5,
}


@pytest.mark.l1
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("n_bc_modes", [1, 2, 3])
def test_class_b_1g_1r_reproduces_published_table(geometry, n_bc_modes):
    """1G/1R sanity — pins to published Issue #112 table within ±2 pp.

    Pipeline gate: confirms `solve_peierls_mg` + cp_module._RADII
    routing reproduces the documented rank-N behavior at the BASE
    preset. If this drifts > 2 pp, either the published table is
    stale or the assembly has regressed.
    """
    geom_id = "cylinder-1d" if geometry is CYLINDER_1D else "sphere-1d"
    expected_err = _PUBLISHED_TABLE_1G_1R[(geom_id, n_bc_modes)]

    k_eff = _solve_class_b(geometry, "1g", 1, n_bc_modes, quad=_QUAD_BASE)
    k_inf = _kinf_ref(geometry, "1g", 1)
    actual_err = abs(k_eff - k_inf) / k_inf * 100

    assert abs(actual_err - expected_err) < 2.0, (
        f"[{geom_id}] 1G/1R rank-{n_bc_modes}: actual |err| = "
        f"{actual_err:.2f} % vs published {expected_err:.2f} %. "
        f"k_eff = {k_eff:.6f}, k_inf = {k_inf:.6f}. Drift > 2 pp "
        f"— either the published table at peierls_geometry.py "
        f"lines 3934-3961 is stale, or the assembly has regressed."
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. MR routing invariance — Probe C promoted (passes today, regression gate)
# ═══════════════════════════════════════════════════════════════════════

# Per-geometry tolerance for MR routing invariance — calibrated against
# the BASE preset's Issue #114 ρ-quadrature floor. Sphere is dominated
# by the (clean) angular θ integral; cylinder picks up extra noise from
# the ρ-subdivision-at-panel-boundary issue (~1 % at BASE). RICH
# quadrature would tighten both below 1e-3 but doubles wall time.
_ROUTING_INVARIANCE_TOL = {
    "sphere-1d":   5e-3,   # Issue #114 floor on sphere is ~1e-3
    "cylinder-1d": 2e-2,   # Issue #114 floor on cylinder is ~1 % at BASE
}


@pytest.mark.l1
@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("n_bc_modes", [1, 2, 3])
def test_class_b_mr_routing_invariance_uniform_sigma(geometry, n_bc_modes):
    """Routing invariance: ``radii=[0.5, 1.0]`` with uniform σ_t = 1 must
    match ``radii=[1.0]`` with σ_t = 1.

    The two configurations are functionally identical — same XS, same
    cell geometry, only the radii partition differs (the 2-region
    layout has a phantom interface at r=0.5 with no σ_t step). If the
    rank-N closure routing depends on ``len(radii)`` beyond the σ_t
    profile, this test catches it (Issue #131-style anti-pattern
    audit).

    Tolerance is per-geometry (see ``_ROUTING_INVARIANCE_TOL``) to
    absorb the Issue #114 ρ-quadrature floor — cylinder's floor is
    ~1 % at BASE, sphere's is ~1e-3. The catastrophic bug magnitude
    (~10-50 % k_eff error from Issue #132) is comfortably above
    these tolerances.
    """
    geom_id = "cylinder-1d" if geometry is CYLINDER_1D else "sphere-1d"
    tol = _ROUTING_INVARIANCE_TOL[geom_id]
    k_1r = _solve_uniform_sigma(geometry, [1.0], [[1.0]],
                                 n_bc_modes=n_bc_modes)
    k_2r_homog = _solve_uniform_sigma(geometry, [0.5, 1.0], [[1.0], [1.0]],
                                       n_bc_modes=n_bc_modes)
    rel_diff = abs(k_1r - k_2r_homog) / max(abs(k_1r), 1e-30)
    assert rel_diff < tol, (
        f"[{geom_id}] rank-{n_bc_modes}: MR routing path diverges "
        f"from 1R for uniform σ_t. k_1R = {k_1r:.10f}, "
        f"k_2R_homog = {k_2r_homog:.10f}, rel_diff = {rel_diff:.3e} "
        f"(tol={tol:.1e}). The radii=[0.5,1.0] partition with σ_t=[1,1] "
        f"is functionally identical to radii=[1.0] σ_t=[1] — divergence "
        f"above the Issue #114 noise floor indicates an MR routing bug "
        f"(Issue #131-style anti-pattern)."
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. The Class B MR catastrophe — Probe G promoted, xfailed to Issue #132
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l1
@pytest.mark.catches("ERR-030")
@pytest.mark.xfail(
    reason=(
        "Issue #132: rank-N Marshak closure mode-0 vs mode-n>=1 "
        "normalization mismatch. Sphere 1G/2R rank-2 produces "
        "k_eff = 1.015 vs k_inf = 0.648 (+57% sign-flip catastrophe) "
        "because legacy compute_P_esc (no Jacobian) for mode 0 mixes "
        "with canonical compute_P_esc_mode (with (rho_max/R)^2 Jacobian) "
        "for mode 1+. Fixing requires re-derivation, not a one-line "
        "edit -- see issue body."
    ),
    strict=True,
)
def test_class_b_mr_catastrophe_sphere_1g_2r_rank2():
    """Pin the headline failure: sphere 1G/2R rank-2 + 57 % sign-flip.

    XFAIL strict=True — when Issue #132 lands and the catastrophe
    disappears, the xfail flips and this test STARTS failing with
    \"unexpected pass,\" alerting the developer to flip the test to
    a real assertion.
    """
    k_eff = _solve_class_b(SPHERE_1D, "1g", 2, n_bc_modes=2, quad=_QUAD_BASE)
    k_inf = _kinf_ref(SPHERE_1D, "1g", 2)
    rel_err = (k_eff - k_inf) / k_inf
    # Goal: rank-N rank-2 should converge to k_inf within a few percent.
    # Currently it overshoots by +57 %. The xfail captures the broken
    # state.
    assert abs(rel_err) < 0.05, (
        f"sphere 1G/2R rank-2: k_eff = {k_eff:.6f} vs k_inf = "
        f"{k_inf:.6f}, rel_err = {rel_err:+.4f}. Expected within 5 % "
        f"once Issue #132 normalization re-derivation lands."
    )


@pytest.mark.l1
@pytest.mark.catches("ERR-030")
@pytest.mark.xfail(
    reason=(
        "Issue #132: same root cause as the sphere 1G/2R catastrophe. "
        "Cylinder 1G/2R rank-2 gives k_eff ~1.17 vs k_inf=0.99 (+18%) "
        "at BASE quadrature. Smaller magnitude than sphere because the "
        "cylinder rank-N primitives are also subject to Issue #112 "
        "Phase C (Knyazev Ki_{k+2} 3D angular normalization), which "
        "adds a separate floor."
    ),
    strict=True,
)
def test_class_b_mr_catastrophe_cylinder_1g_2r_rank2():
    """Pin the cylinder analog of the sphere 1G/2R rank-2 failure."""
    k_eff = _solve_class_b(CYLINDER_1D, "1g", 2, n_bc_modes=2, quad=_QUAD_BASE)
    k_inf = _kinf_ref(CYLINDER_1D, "1g", 2)
    rel_err = (k_eff - k_inf) / k_inf
    assert abs(rel_err) < 0.05, (
        f"cylinder 1G/2R rank-2: k_eff = {k_eff:.6f} vs k_inf = "
        f"{k_inf:.6f}, rel_err = {rel_err:+.4f}. Expected within 5 % "
        f"once Issue #132 normalization re-derivation lands."
    )


# ═══════════════════════════════════════════════════════════════════════
# 3b. Hébert (1-P_ss)⁻¹ closure — the Issue #132 production fix for sphere
# ═══════════════════════════════════════════════════════════════════════
#
# Per Hébert *Applied Reactor Physics* (3rd ed.) §3.8.5 Eq. (3.323) the
# canonical CP white-BC closure for sphere is:
#
#   ℙ_white = ℙ_vac + (β⁺/(1 - β⁺·P_ss)) · P_iS · P_Sj^T
#
# The current ORPHEUS rank-1 Mark code computes the rank-1 outer product
# but is missing the (1-β⁺·P_ss)⁻¹ geometric-series factor that captures
# multiple reflections off the surface. ``boundary="white_hebert"`` adds
# this factor and recovers k_inf to within ~1.2 % on 3 of 4 sphere
# configurations vs 15-88 % errors with the bare Mark closure.
#
# Heterogeneous 1G/2R retains a ~10 % overshoot from the Mark uniformity
# assumption being amplified by the geometric series — pinned as a
# known limitation here (xfail strict=True with the 10 % deviation as
# the boundary).


def _solve_class_b_hebert(geometry, ng_key: str, n_regions: int,
                           *, quad: dict):
    """Same as _solve_class_b but uses boundary='white_hebert'."""
    cp_module = cp_cylinder if geometry is CYLINDER_1D else cp_sphere
    xs = _build_xs_arrays(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    sol = solve_peierls_mg(
        geometry, radii=radii, **xs,
        boundary="white_hebert", n_bc_modes=1,
        **quad,
    )
    return sol.k_eff


@pytest.mark.l1
@pytest.mark.parametrize("ng_key, n_regions, expected_err_pct, tol_pct", [
    pytest.param("1g", 1, 0.0, 0.5, id="1G_1R_homogeneous"),
    pytest.param("2g", 1, -1.0, 1.5, id="2G_1R_homogeneous"),
    pytest.param("2g", 2, -1.2, 1.5, id="2G_2R_heterogeneous"),
])
def test_class_b_sphere_hebert_recovers_kinf(ng_key, n_regions,
                                             expected_err_pct, tol_pct):
    """Hébert (1-P_ss)⁻¹ closure recovers cp_sphere k_inf for sphere.

    These are the three configurations where the geometric-series
    correction is sufficient. The 1G/2R heterogeneous case has a
    larger residual (~10 %) due to Mark uniformity-assumption
    amplification — pinned separately below.

    Tolerance bounds are set to absorb numerical noise (Issue #114
    ρ-quadrature floor, BASE-preset effects) while still gating any
    structural regression.
    """
    k_eff = _solve_class_b_hebert(SPHERE_1D, ng_key, n_regions,
                                   quad=_QUAD_BASE)
    k_inf = _kinf_ref(SPHERE_1D, ng_key, n_regions)
    actual_err_pct = (k_eff - k_inf) / k_inf * 100
    assert abs(actual_err_pct - expected_err_pct) < tol_pct, (
        f"sphere {ng_key}/{n_regions}r Hébert closure: actual err = "
        f"{actual_err_pct:+.3f} %, expected {expected_err_pct:+.2f} % "
        f"± {tol_pct} %. k_eff = {k_eff:.10f}, k_inf = {k_inf:.10f}. "
        f"Either the closure regressed or the cp_sphere reference "
        f"changed."
    )


@pytest.mark.l1
def test_class_b_sphere_hebert_heterogeneous_overshoot_known():
    """Pin the known +10 % overshoot on sphere 1G/2R fuel-A/mod-B.

    The Hébert geometric-series factor amplifies the Mark closure's
    uniformity assumption. For homogeneous cells (1G/1R) and weakly
    heterogeneous cells (2G/2R, where fast/thermal coupling smooths
    the spatial structure), Mark is approximately exact and the
    Hébert closure converges to k_inf. For strongly heterogeneous
    1G/2R (fuel inner / pure-absorber moderator outer), Mark error
    is amplified by the (1-P_ss)⁻¹ factor → +10 % overshoot.

    This is a fundamental Mark-closure limitation, NOT a bug in the
    Hébert correction. To fix the 1G/2R case requires either:
    (a) an angular-distribution-preserving closure (rank-N path,
        falsified in Issue #132 — does not converge structurally)
    (b) Davison method-of-images sphere kernel (open question)
    (c) augmented Nyström with surface partial current as extra
        unknown
    """
    k_eff = _solve_class_b_hebert(SPHERE_1D, "1g", 2, quad=_QUAD_BASE)
    k_inf = _kinf_ref(SPHERE_1D, "1g", 2)
    err_pct = (k_eff - k_inf) / k_inf * 100
    # Pin the +10.3 % overshoot as the known Mark limitation
    assert 9.0 < err_pct < 12.0, (
        f"sphere 1G/2R Hébert closure: err = {err_pct:+.3f} %. "
        f"Expected the +10.33 % overshoot characteristic of the Mark "
        f"uniformity assumption amplified by the (1-P_ss)⁻¹ factor "
        f"on strongly heterogeneous cells. Either the closure "
        f"regressed (different overshoot magnitude) or the issue is "
        f"resolved (in which case flip this test to the convergent "
        f"form)."
    )


# RICH-quadrature variant: same configurations as the BASE pin above,
# but tighter tolerances reflecting the RICH-quadrature near-exactness.
# Wall-time per case ~30-300 s; marked @slow.

_QUAD_RICH = dict(
    n_panels_per_region=4,
    p_order=5,
    n_angular=64,
    n_rho=48,
    n_surf_quad=64,
    dps=20,
)


@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.parametrize("ng_key, n_regions, expected_err_pct, tol_pct", [
    pytest.param("1g", 1, 0.0, 0.05, id="1G_1R_homogeneous_RICH"),
    pytest.param("2g", 1, 0.0, 0.05, id="2G_1R_homogeneous_RICH"),
    pytest.param("2g", 2, 0.0, 0.05, id="2G_2R_heterogeneous_RICH"),
])
def test_class_b_sphere_hebert_recovers_kinf_rich(ng_key, n_regions,
                                                   expected_err_pct, tol_pct):
    """At RICH quadrature, Hébert recovers cp k_inf to <0.05 %.

    The BASE-preset 0.15-1.5 % residuals were quadrature noise (Issue
    #114 ρ-subdivision); the underlying Hébert closure is exact to
    numerical precision when the Mark uniformity assumption holds.
    Excludes 1G/2R (covered by the structural-overshoot pin below).
    """
    k_eff = _solve_class_b_hebert(SPHERE_1D, ng_key, n_regions,
                                   quad=_QUAD_RICH)
    k_inf = _kinf_ref(SPHERE_1D, ng_key, n_regions)
    actual_err_pct = (k_eff - k_inf) / k_inf * 100
    assert abs(actual_err_pct - expected_err_pct) < tol_pct, (
        f"sphere {ng_key}/{n_regions}r Hébert RICH: actual err = "
        f"{actual_err_pct:+.4f} %, expected {expected_err_pct:+.2f} % "
        f"± {tol_pct} %. RICH is the verification-grade tolerance; "
        f"BASE has Issue #114 ρ-quadrature noise that masks exactness."
    )


@pytest.mark.l1
@pytest.mark.parametrize("chi_spectrum, expected_err_pct, tol_pct", [
    pytest.param([1.0, 0.0], -1.5, 1.0, id="fast_emission"),
    pytest.param([0.5, 0.5], 2.7, 1.0, id="mixed_emission"),
    pytest.param([0.0, 1.0], 6.6, 1.0, id="thermal_emission"),
])
def test_class_b_sphere_hebert_chi_dependence(chi_spectrum,
                                                expected_err_pct, tol_pct):
    """Pin the chi-dependent Hébert error on sphere 2G/2R fuel-mod.

    Triggered by user observation 2026-04-25: the "near-exact 2G/2R
    Hébert" claim looked suspicious given parity with 1G/2R (same
    geometry, same materials). This test pins the monotone trend
    showing the 2G/2R DEFAULT chi=[1, 0] result is coincident with
    the spectrum routing emission into a near-uniform-σ_t group:

      chi=[1, 0]  (fast):    err ≈ −1.5 %  (essentially exact)
      chi=[0.5, 0.5] (mixed): err ≈ +2.7 %
      chi=[0, 1]  (thermal): err ≈ +6.6 %

    This is the source-distribution dependence of the Mark uniformity
    assumption — see :ref:`peierls-class-b-sphere-hebert` discussion
    of the 1G/2R limitation. Pinning the trend here ensures that
    future Mark-closure improvements (Davison kernel, augmented
    Nyström) flatten the chi-dependence — when this test starts
    failing, that signals the closure has been improved.
    """
    xs = _build_xs_arrays("2g", 2)
    radii = np.array(cp_sphere._RADII[2])
    chi_arr = np.array([chi_spectrum, chi_spectrum])  # one row per region

    sol = solve_peierls_mg(
        SPHERE_1D, radii=radii,
        sig_t=xs["sig_t"], sig_s=xs["sig_s"], nu_sig_f=xs["nu_sig_f"],
        chi=chi_arr,
        boundary="white_hebert", n_bc_modes=1,
        **_QUAD_BASE,
    )
    k_eff = sol.k_eff

    # Reference: cp k_inf with the same chi spectrum
    from orpheus.derivations.cp_sphere import _sphere_cp_matrix
    from orpheus.derivations._eigenvalue import kinf_from_cp
    layout = LAYOUTS[2]
    xs_list = [get_xs(r, "2g") for r in layout]
    r_inner = np.zeros(2)
    r_inner[1:] = radii[:-1]
    volumes = (4 / 3) * np.pi * (radii**3 - r_inner**3)
    P_inf = _sphere_cp_matrix(xs["sig_t"], radii, volumes, radii[-1])
    k_inf = kinf_from_cp(
        P_inf_g=P_inf, sig_t_all=xs["sig_t"], V_arr=volumes,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=[np.asarray(chi_spectrum) for _ in xs_list],
    )
    actual_err_pct = (k_eff - k_inf) / k_inf * 100

    assert abs(actual_err_pct - expected_err_pct) < tol_pct, (
        f"sphere 2G/2R chi={chi_spectrum} Hébert: actual err = "
        f"{actual_err_pct:+.3f} %, expected {expected_err_pct:+.2f} % "
        f"± {tol_pct} %. The chi-dependent overshoot trend is the "
        f"signature of the Mark uniformity assumption — see "
        f"docs/theory/peierls_unified.rst §peierls-class-b-sphere-hebert. "
        f"Drift here indicates either a Mark-closure improvement (good!) "
        f"or a regression in the Hébert path (bad)."
    )


@pytest.mark.l1
@pytest.mark.parametrize("kind", ["cylinder-1d", "slab-polar"])
def test_class_b_hebert_raises_for_non_sphere(kind):
    """Hébert closure currently sphere-only; cylinder / slab must raise."""
    from orpheus.derivations.peierls_geometry import (
        CurvilinearGeometry, SLAB_POLAR_1D,
    )
    geom = CYLINDER_1D if kind == "cylinder-1d" else SLAB_POLAR_1D
    if kind == "slab-polar":
        # slab-polar with rank-1 doesn't accept multi-region in this
        # validation path — pick the simplest 1G/1R sphere XS to skip
        # the geometry-validation entanglement
        pytest.skip("slab-polar uses different MR routing; checked elsewhere")

    xs = _build_xs_arrays("1g", 1)
    radii = np.array(cp_cylinder._RADII[1])
    with pytest.raises(NotImplementedError, match="sphere-only"):
        solve_peierls_mg(
            geom, radii=radii, **xs,
            boundary="white_hebert", n_bc_modes=1,
            **_QUAD_BASE,
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Class B 2G/2R rank-1 floor — regression pinning during Issue #132
# ═══════════════════════════════════════════════════════════════════════
#
# 2G/2R rank-1 Mark closure has a far larger floor than 1G/1R because
# the fast group of fuel A has σ_t,A,g0 = 0.5 → cell is only 0.5 MFP
# thick to the fast neutron, where Mark's isotropic re-entry is poorly
# justified. The values below are the BASE-preset reproducible
# baseline at 2026-04-25 — pinned not because they are correct (they
# are far from k_inf) but because they reproduce; any drift signals
# a regression in the Mark assembly.

_RANK1_2G_2R_BASE = {
    "cylinder-1d": (0.1726452873, 0.7399127793),  # k_eff_BASE, k_inf
    "sphere-1d":   (0.0861822860, 0.4140164541),
}


@pytest.mark.l1
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_class_b_2g_2r_rank1_mark_floor_pinned(geometry):
    """Pin the 2G/2R rank-1 Mark closure k_eff at the BASE preset.

    Documents the current (broken-by-MG-amplification) floor so that
    once Issue #132 lands the regression of these values is detectable.
    The k_eff values are about ``-77 %`` (cyl) and ``-79 %`` (sph) low
    — wildly off from the analytical ``k_inf`` — because the rank-1
    Mark closure's isotropic-reentry assumption is poorly justified
    for the thin (σ_t·R = 0.5 MFP) fast group, and the closure error
    is amplified by the 2G fission/scatter coupling.
    """
    geom_id = "cylinder-1d" if geometry is CYLINDER_1D else "sphere-1d"
    expected_k_eff, expected_k_inf = _RANK1_2G_2R_BASE[geom_id]

    k_eff = _solve_class_b(geometry, "2g", 2, n_bc_modes=1, quad=_QUAD_BASE)
    k_inf = _kinf_ref(geometry, "2g", 2)

    np.testing.assert_allclose(
        k_inf, expected_k_inf, rtol=1e-6,
        err_msg=f"[{geom_id}] k_inf reference drifted",
    )
    # Loose absolute pin — captures the floor without over-constraining
    # the BASE-preset value (Issue #114 noise + arithmetic drift).
    np.testing.assert_allclose(
        k_eff, expected_k_eff, rtol=5e-3,
        err_msg=(
            f"[{geom_id}] 2G/2R rank-1 Mark k_eff drifted from BASE "
            f"baseline. Expected {expected_k_eff:.10f}, got "
            f"{k_eff:.10f}. Either the Mark assembly regressed or "
            f"Issue #132 fix landed (in which case re-baseline this "
            f"test against the corrected closure)."
        ),
    )
