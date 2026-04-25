"""Diagnostic A — derive G_bc(r_i, µ) for sphere and verify consistency.

Issue #132 augmented-Nyström investigation.

Setup
-----
The existing rank-1 Mark closure assembles K_bc = u v^T where the
surface "response" G_bc(r_i) is the scalar flux at r_i from a UNIFORM
ISOTROPIC inward partial current J^- on the surface (Lambert / Mark
assumption). For non-uniform eigenvectors this approximation breaks.

The augmented-Nyström hypothesis replaces this scalar by a vector of
M angular components J^+(µ_m) at Nyström nodes µ_m on the outgoing
hemisphere (µ ∈ (0,1]). Each J^+(µ_m) is treated as a separate
unknown and enforced by J^- = J^+ at each angle (white BC). This
requires deriving the per-angle kernel:

  G_bc(r_i, µ) = scalar flux at r_i per unit surface angular flux ψ^-
                 with support delta(µ_exit - µ) at the surface.

Sphere derivation (observer-centred)
------------------------------------
At observer r_i, the polar angle θ to the radius vector parametrises
ray directions. For a solid sphere of radius R the ray to the surface
has length

    rho_max(r_i, θ) = -r_i cos θ + sqrt(R² - r_i² sin² θ)

and exits at surface µ (cosine to outward normal) of

    µ_exit(r_i, θ) = (rho_max + r_i cos θ)/R = sqrt(1 - (r_i/R)² sin²θ)

This is INDEPENDENT of the sign of cos θ — the forward and back rays
land on antipodal surface points with the SAME µ_exit. So the map
θ → µ is two-to-one. Each branch is differentiable with

    dµ/dθ = -(r_i/R)² sin θ cos θ / µ_exit

Performing the change of variables in the observer integral

    phi(r_i) = 2π ∫_0^π sin θ exp(-τ(r_i, θ)) ψ^-(µ_exit) dθ

and recognising that θ ∈ [0, π/2] maps µ_exit = 1 → µ_min, and
θ ∈ [π/2, π] maps µ_exit = µ_min → 1, gives

    phi(r_i) = ∫_{µ_min}^1 K(r_i, µ) ψ^-(µ) dµ

with kernel

    K(r_i, µ) = 2π · [exp(-τ_+(µ)) + exp(-τ_-(µ))] · sin θ(µ) ·
                |dθ/dµ|

where τ_+ corresponds to the forward branch (cos θ > 0) and τ_- to
the back branch (cos θ < 0). For a SOLID HOMOGENEOUS sphere both τ
values are µ-only and equal:

    τ(r_i, µ) = Σ_t · (R · µ - r_i · cos θ),  with cos θ branched

Concretely, on the back branch cos θ < 0, the chord is longer:
    rho_back = +r_i |cos θ| + sqrt(R² - r_i² sin² θ) = R µ + r_i √(1-sin²θ-?)

Let's just compute numerically by walking θ ∈ [0, π].

Sanity gates
------------
1. ∫_0^1 K(r_i, µ) · (1/π) dµ = G_bc(r_i)  (Mark uniform isotropic
   re-emission with ψ^- = J^-/π = 1/π gives the existing scalar
   G_bc — this is the M=1 limit).
2. K(r_i, µ) ≥ 0 everywhere.
3. K(r_i, µ) → 0 for µ < µ_min(r_i) (rays don't reach there).

Promotion
---------
If the agreement holds to <1e-10 across r_i and configurations, this
is a candidate L1 regression for the per-angle kernel.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rho_max_solid_sphere(r_i: float, cos_theta: float, R: float) -> float:
    """Chord from r_i in direction θ to outer surface (solid sphere)."""
    discr = R * R - r_i * r_i * (1 - cos_theta * cos_theta)
    if discr < 0:
        return 0.0
    return -r_i * cos_theta + np.sqrt(discr)


def _mu_exit(r_i: float, cos_theta: float, R: float) -> float:
    """Surface µ_exit (cosine to outward normal at exit point)."""
    rho = _rho_max_solid_sphere(r_i, cos_theta, R)
    return (rho + r_i * cos_theta) / R


def _G_bc_uniform_observer(r_i: float, R: float, sig_t: float,
                           N: int = 256) -> float:
    """Existing G_bc: 2 ∫_0^π sin θ exp(-Σ_t · ρ_max) dθ.

    Reference for the Mark uniform-isotropic case. ψ^- = 1/π everywhere.
    """
    theta_pts, theta_wts = np.polynomial.legendre.leggauss(N)
    theta_pts_mapped = 0.5 * (theta_pts + 1) * np.pi
    theta_wts_mapped = 0.5 * np.pi * theta_wts

    total = 0.0
    for k in range(N):
        ct = np.cos(theta_pts_mapped[k])
        st = np.sin(theta_pts_mapped[k])
        rho = _rho_max_solid_sphere(r_i, ct, R)
        total += theta_wts_mapped[k] * st * np.exp(-sig_t * rho)
    return 2.0 * total


def K_per_angle_via_observer(r_i: float, mu: float, R: float,
                             sig_t: float) -> float:
    """K(r_i, µ): per-angle response kernel via observer-centred form.

    Build by direct change of variables θ → µ. For each µ ∈ (µ_min, 1]
    there are TWO θ values: forward (cos θ > 0) with µ_exit = µ on the
    "near" surface and back (cos θ < 0) with µ_exit = µ on the "far"
    surface. Sum both contributions.

    From µ = sqrt(1 - (r_i/R)² sin² θ):
       sin² θ = (R/r_i)² (1 - µ²)
       cos θ_+ = +sqrt(1 - sin²θ),  cos θ_- = -sqrt(1 - sin²θ)
       dµ/dθ = -(r_i/R)² sin θ cos θ / µ
       |dθ/dµ| = µ / ((r_i/R)² sin θ |cos θ|)

    The kernel after change of variables:
       K(µ) = 2π sin θ exp(-τ) · |dθ/dµ| (per branch)
            = 2π exp(-τ) · µ / ((r_i/R)² |cos θ|)

    For a SOLID HOMOGENEOUS sphere:
       ρ_+ = R µ - r_i |cos θ|   (forward, near surface)
       ρ_- = R µ + r_i |cos θ|   (back, far surface)
       τ_± = Σ_t · ρ_±

    At r_i = 0: forward and back collapse, µ_exit = 1 always. Special
    case the limit by routing through the existing observer integral.
    """
    if r_i == 0.0:
        # Degenerate: all rays land at µ_exit = 1. K is a delta. Don't
        # define a per-angle K; the augmented-Nyström is degenerate at
        # the centre. Caller must handle.
        return float("nan")
    R_over_ri = R / r_i
    sin2_theta = R_over_ri * R_over_ri * (1.0 - mu * mu)
    if sin2_theta > 1.0:
        return 0.0  # forbidden (µ < µ_min)
    abs_cos_theta = np.sqrt(max(1.0 - sin2_theta, 0.0))
    if abs_cos_theta == 0.0:
        return 0.0  # tangent grazing, measure-zero
    rho_fwd = R * mu - r_i * abs_cos_theta
    rho_bck = R * mu + r_i * abs_cos_theta
    pref = 2.0 * np.pi * mu / ((r_i / R) ** 2 * abs_cos_theta)
    return pref * (np.exp(-sig_t * rho_fwd) + np.exp(-sig_t * rho_bck))


def _mu_min(r_i: float, R: float) -> float:
    return np.sqrt(max(1.0 - (r_i / R) ** 2, 0.0))


def _gauss_legendre_on(a: float, b: float, n: int):
    pts, wts = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * pts + 0.5 * (b + a), 0.5 * (b - a) * wts


# ─────────────────────────────────────────────────────────────────────
# Sanity gate: ∫ K(r_i, µ) · (1/π) dµ over (µ_min, 1) = G_bc(r_i)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("r_i,R,sig_t", [
    (0.3, 1.0, 0.5),
    (0.5, 1.0, 0.5),
    (0.7, 1.0, 1.0),
    (0.9, 1.0, 2.0),
    (0.5, 1.0, 0.0),  # transparent limit
])
def test_K_uniform_collapse_matches_G_bc(r_i, R, sig_t):
    """ψ^- = 1/π Lambertian → K-integral = existing G_bc(r_i)."""
    G_bc_ref = _G_bc_uniform_observer(r_i, R, sig_t, N=512)

    # Integrate K(r_i, µ) · (1/π) over µ ∈ (µ_min(r_i), 1] via dense
    # Gauss-Legendre.
    mm = _mu_min(r_i, R)
    mu_pts, mu_wts = _gauss_legendre_on(mm + 1e-12, 1.0, 256)
    K_vals = np.array([K_per_angle_via_observer(r_i, mu, R, sig_t)
                       for mu in mu_pts])
    G_bc_via_K = float(np.sum(mu_wts * K_vals * (1.0 / np.pi)))

    rel_err = abs(G_bc_via_K - G_bc_ref) / max(abs(G_bc_ref), 1e-30)
    # NOTE: tolerance 5e-3, not machine-eps. The K(r_i, µ) kernel has an
    # inverse-square-root endpoint singularity at µ = µ_min(r_i); a
    # plain GL on µ ∈ (µ_min, 1) converges at O(1/sqrt(M)) — see
    # Diagnostic B for the explicit convergence-rate fit. This test
    # is the LOWER BOUND check (kernel form is at least correct);
    # the structural finding lives in Diagnostic B.
    assert rel_err < 5e-3, (
        f"K-integral collapse failed at r_i={r_i}, R={R}, σ_t={sig_t}: "
        f"G_bc_ref={G_bc_ref:.10e}, G_bc_via_K={G_bc_via_K:.10e}, "
        f"rel_err={rel_err:.2e}"
    )


@pytest.mark.parametrize("r_i,R,sig_t", [
    (0.3, 1.0, 0.5),
    (0.7, 1.0, 1.0),
    (0.9, 1.0, 2.0),
])
def test_K_nonnegative(r_i, R, sig_t):
    """K(r_i, µ) ≥ 0 for all µ in (µ_min, 1]."""
    mm = _mu_min(r_i, R)
    mu_pts, _ = _gauss_legendre_on(mm + 1e-9, 1.0, 64)
    for mu in mu_pts:
        K_val = K_per_angle_via_observer(r_i, mu, R, sig_t)
        assert K_val >= 0.0, (
            f"K({r_i}, {mu}) = {K_val} < 0 — derivation bug"
        )


def test_K_zero_outside_support():
    """K(r_i, µ) = 0 for µ < µ_min(r_i)."""
    r_i, R, sig_t = 0.5, 1.0, 0.5
    mm = _mu_min(r_i, R)
    # Try µ values strictly less than µ_min
    for mu in [0.0, 0.1 * mm, 0.5 * mm, mm - 1e-3]:
        K_val = K_per_angle_via_observer(r_i, mu, R, sig_t)
        assert K_val == 0.0, (
            f"K outside support: K({r_i}, {mu}) = {K_val} ≠ 0, "
            f"but µ_min={mm}"
        )


if __name__ == "__main__":
    print("=" * 78)
    print("Augmented-Nyström kernel derivation — sphere")
    print("=" * 78)
    print()
    print("  Sanity gate 1: K-integral collapse (M = 1 Lambertian)")
    print(f"  {'r_i':>6} {'R':>6} {'σ_t':>6} {'µ_min':>8} "
          f"{'G_bc_ref':>12} {'G_bc(K)':>12} {'rel_err':>10}")
    for r_i, R, sig_t in [(0.3, 1.0, 0.5), (0.5, 1.0, 0.5),
                          (0.7, 1.0, 1.0), (0.9, 1.0, 2.0),
                          (0.5, 1.0, 0.0)]:
        mm = _mu_min(r_i, R)
        G_bc_ref = _G_bc_uniform_observer(r_i, R, sig_t, N=512)
        mu_pts, mu_wts = _gauss_legendre_on(mm + 1e-12, 1.0, 256)
        K_vals = np.array([K_per_angle_via_observer(r_i, mu, R, sig_t)
                           for mu in mu_pts])
        G_bc_via_K = float(np.sum(mu_wts * K_vals * (1.0 / np.pi)))
        rel_err = abs(G_bc_via_K - G_bc_ref) / max(abs(G_bc_ref), 1e-30)
        print(f"  {r_i:>6.2f} {R:>6.2f} {sig_t:>6.2f} {mm:>8.4f} "
              f"{G_bc_ref:>12.6e} {G_bc_via_K:>12.6e} {rel_err:>10.2e}")
