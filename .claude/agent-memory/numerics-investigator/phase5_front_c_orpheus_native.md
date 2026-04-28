---
name: Phase 5 Front C — ORPHEUS-native continuous-µ FAILED at smoke test
description: Front C built µ-resolved F_out and G_in primitives consistent with Phase 4 P_esc_mode/G_bc_mode (1-30% projection error from grazing singularity, fixable). M1 K_bc assembly fails Probe D smoke test — k_eff oscillates wildly with Q because integrand has the SAME 1/µ singularity that diverges Phase 4 at high N. Both M1 native AND Sanchez Eq. (A6) inherit non-integrable diagonal at homogeneous sphere. Sanchez↔ORPHEUS Jacobian conversion is NOT a simple scalar (varies 1-73 with i,j). Verdict: FAIL at smoke test. Phase 5+ requires singularity treatment (Gauss-Jacobi, change-of-variables, or analytic subtraction) before any continuous-µ form is production-ready.
type: project
---

# Phase 5+ Front C — ORPHEUS-native continuous-µ specular sphere (2026-04-28)

## TL;DR

**Verdict: FAIL at smoke test.** k_eff via the ORPHEUS-native M1 form
oscillates wildly with µ-quadrature size Q (errors -45% → +33% across
Q ∈ {16, 32, 64, 128, 256}). The structural premise is sound but the
1/µ singularity at the diagonal blocks production wiring identically
to the Phase 5a Sanchez form.

## What worked

1. **µ-resolved primitives** `F_out_mu_sphere(r, µ)` and
   `G_in_mu_sphere(r, µ)` are derivable from the Phase 4
   `compute_P_esc_mode` / `compute_G_bc_mode` integrands by ω → µ
   change of variables.

2. **Polynomial-projection consistency PASSES** (after fixing K_esc
   bug — sphere is `e^{-τ}`, not `(1-e^{-τ})/τ`):
   - F_out projected against P̃_n(µ) ≈ Phase 4 P_esc_mode[n,j] to
     within 1-30% at Q=128 standard GL. Errors concentrated at
     endpoints (small r → narrow µ-support, large r → grazing
     singularity at u_min(r) = √(1-(r/R)²)).
   - G_in projected against P̃_n similar, ~10-30% off at endpoints.
   - Interior nodes (r ≈ R/2) match to <2%.

3. **Per-pair Sanchez form converges on most off-diagonals**:
   K[1,2], K[1,3] (where r_i < r_j) match to 6 figures by Q=32.
   K[2,1], K[3,0] (where r_i > r_j) match to 3-4 figures by Q=512.

## What failed

### Failure 1: M1 K_bc Q-divergence at sphere homogeneous

The M1 form `K_bc[i,j] = 2 ∫_0^1 G_in(r_i,µ)·F_out(r_j,µ)·T(µ) dµ`
with `T(µ) = 1/(1-e^{-σ·2Rµ})` has the SAME `1/µ` singularity as
the Phase 4 matrix-Galerkin `(I-TR)⁻¹` operator at µ→0. The integrand
diverges:

| Q   | k_eff_native | rel_to_kinf | rel_to_white_hebert |
|-----|--------------|-------------|---------------------|
| 16  | 0.169        | -18.9%      | -18.8%              |
| 32  | 0.239        | +14.8%      | +14.9%              |
| 64  | 0.149        | -28.4%      | -28.3%              |
| 128 | 0.277        | +32.8%      | +33.0%              |
| 256 | 0.132        | -36.5%      | -36.4%              |

Reference (white_hebert rank-1): k_eff = 0.208089 (-0.12% from k_inf).

The Q-oscillation pattern is exactly the matrix-Galerkin signature
expressed in continuous-µ form: standard GL nodes resolve grazing
modes (µ → 0) progressively as Q grows, and the `1/µ` singularity
keeps shifting the integral.

### Failure 2: Sanchez↔ORPHEUS Jacobian conversion is NOT a scalar

| K_orpheus[i,j] / K_sanchez[i,j] | min  | max  | std/median |
|---------------------------------|------|------|------------|
| Off-diagonal entries            | 1.0  | 73   | 0.99       |

Tested:
- Constant scalar `α·K_san`: rel_spread 99% — fails
- σ-scalar `σ·K_san`: ratio range 0.8 → 2.4 — fails
- Per-source `α·K_san/r_j²`: ratio range 1.5 → 8.1 — fails
- Volume-weight `(σ_t/divisor)·K_san·(rv·r_wts)`: ratio 0.29 → 0.79 — fails
- Holding j fixed: ratio K_p4(N=3)[i,j_fixed] / K_san[i,j_fixed]
  varies smoothly with i — suggests a more complex (non-separable)
  conversion.

Sanchez's `g_h(ρ' → ρ)` IS the Green's function for the optical-units
Peierls integral `Σ·φ(ρ) = ∫ g(ρ'→ρ)·q(ρ')·dρ'`, but ORPHEUS K_ij
discretises a different normalisation with explicit `rv·r_wts`. The
two are equivalent integral-equation kernels but the discrete K_ij
conversion involves the per-r' radial-volume Jacobian which is NOT
representable as a simple multiplicative factor.

### Failure 3: Phase 4 doesn't converge to Sanchez at high N

Phase 4 multibounce on sphere DIVERGES at N≥4 (per
`specular_mb_overshoot_root_cause.md`); the matrix-Galerkin form
in the polynomial basis cannot represent the `1/(1-e^{-σ2Rµ})`
multiplication operator (Trefethen-Embree, *Spectra and Pseudospectra*).

## Implementation sketch (working, but only consistent at Phase 4 N=∞)

```python
def F_out_mu_sphere(r_nodes, radii, sig_t, mu):
    """µ-resolved outgoing partial-current density at the surface
    from a unit volumetric source at each r_j.

    F_out(r, µ) = pref · u/(r²·|cos(ω)|) ·
                  [(R·µ - D)²·K_esc(τ_-) + (R·µ + D)²·K_esc(τ_+)]

    where D = √(r² - R²(1-µ²)) = r·|cos(ω)|. The two ± branches are
    the short and long chord segments from r to the surface, both
    yielding surface cosine µ. Sphere K_esc(τ) = e^{-τ}.

    Visibility: requires r ≥ R√(1-µ²), i.e., µ ≥ µ_min(r).
    """
    R = float(radii[-1])
    pref = 0.5  # SPHERE_1D.prefactor
    F = np.zeros((len(mu), len(r_nodes)))
    for q in range(len(mu)):
        mu_q = float(mu[q])
        for j in range(len(r_nodes)):
            r_j = float(r_nodes[j])
            h = R * np.sqrt(max(0, 1 - mu_q**2))
            if r_j**2 < h**2 - 1e-15:
                continue
            cos_om = np.sqrt(max(1 - (R/r_j)**2 * (1-mu_q**2), 0))
            if cos_om < 1e-15:
                continue
            sqrt_rh = np.sqrt(max(r_j**2 - h**2, 0))
            rho_minus = R*mu_q - sqrt_rh
            rho_plus  = R*mu_q + sqrt_rh
            tau_minus = sig_t[0] * rho_minus  # homogeneous
            tau_plus  = sig_t[0] * rho_plus
            K_minus = np.exp(-tau_minus)
            K_plus  = np.exp(-tau_plus)
            F[q, j] = pref * mu_q / (r_j**2 * cos_om) * (
                rho_plus**2 * K_plus + rho_minus**2 * K_minus
            )
    return F

# G_in_mu_sphere is structurally identical with K_esc → e^{-τ} (no ρ²
# Jacobian; instead a leading 2 from G_bc convention):
#   G_in(r, µ) = 2·R²·µ/(r²·|cos(ω)|) · (e^{-τ_+} + e^{-τ_-})

def compute_K_bc_specular_continuous_mu_sphere_native(
    geometry, r_nodes, r_wts, radii, sig_t, *, n_quad=64,
):
    R = float(radii[-1])
    sigma = float(sig_t[0])
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu_pts = 0.5 * (nodes + 1)
    mu_wts = 0.5 * wts

    tau_chord = _chord_tau_mu_sphere(radii, sig_t, mu_pts)
    T_mu = 1 / (1 - np.exp(-tau_chord))  # ← THE SINGULARITY: 1/µ at µ→0

    F_out = F_out_mu_sphere(r_nodes, radii, sig_t, mu_pts)
    G_in  = G_in_mu_sphere(r_nodes, radii, sig_t, mu_pts)

    rv = np.array([geometry.radial_volume_weight(rj) for rj in r_nodes])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(r_nodes[i], radii)]
        for i in range(len(r_nodes))
    ])
    divisor = geometry.rank1_surface_divisor(R)

    F_out_w = (rv * r_wts)[None, :] * F_out
    G_in_w  = (sig_t_n / divisor)[:, None] * G_in

    return 2 * np.einsum('iq,q,qj->ij', G_in_w, mu_wts * T_mu, F_out_w)
```

LoC for production wiring: ~150 LoC for sphere alone (multi-region
cyl/slab not addressed). But it is NOT production-ready due to the
singularity.

## Why the singularity blocks both forms equivalently

- Sanchez Eq. (A6) integrand has `1/µ_*` × `T(µ_-)` → at diagonal
  `ρ' = ρ`, `µ_* = µ`, leading factor goes as `T(µ_-)/µ ~ 1/µ²` as
  `µ → 0` (per V4 of `peierls_specular_continuous_mu.py`).
- ORPHEUS-native M1 form integrand has F_out · G_in · T(µ) → at the
  visibility-included µ → 0 limit, `F_out · G_in` is finite (chord
  to surface is finite) but `T(µ) ~ 1/(σ·2R·µ)` diverges as 1/µ.
  The diagonal entries (where µ→0 is in the integration support)
  inherit a 1/µ non-integrable singularity.

The "rev-by-cancellation" hope from the cross-domain V1 derivation
(`µ·T(µ) → 1/(2σR)` at µ→0) only saves the case where F_out·G_in
provides an extra `µ` factor — but they don't (the F_out integrand
has `µ/r²·cos(ω)` which → 0 only as `√(µ-µ_min)` near the visibility
cutoff, not as `µ`).

## Phase 5+ paths forward

The fundamental issue: any continuous-µ form on homogeneous specular
sphere has a non-integrable diagonal singularity at µ=0. Both
Sanchez and ORPHEUS-native forms inherit it. Production paths:

1. **Gauss-Jacobi µ-quadrature** with weight `1/(1-e^{-σ·2Rµ})` or
   `1/µ` absorbed. Custom quadrature; per-source visibility cutoff
   complicates the standard G-J recipe.

2. **Singularity subtraction**: split kernel into singular (closed-
   form) + smooth (numerically integrable) parts. Analog of
   ORPHEUS's existing E_1 subtraction in `build_volume_kernel`.
   Closed form for the singular part requires deriving cosh-like
   primitives à la Sanchez Eq. (A6), then converting.

3. **Multi-bounce truncation** (M2 from cross-domain): sum a finite
   number of bounces, each integral non-singular. Truncation bound
   `K_max ≈ 1/(σR) + log(1/tol)` bounces (cross-domain estimate).
   This is the **most promising production form** — each bounce
   integrand `µ · e^{-k·σ·2Rµ}` is well-behaved at µ→0.

4. **Phase 4 at low N (current shipped form)** with strict envelope
   N ∈ {1, 2, 3} — already in production via
   `closure="specular_multibounce"`. The Phase 4 form trades the
   diagonal singularity for high-N matrix-Galerkin divergence; at
   low N both pathologies are mild.

## Recommendation

Direction (4) is what's already shipped (`closure="specular_multibounce"`
N ≤ 3). The Phase 5 production form should target direction (3):
the bounce-resolved expansion. Each bounce has the form

```
K_bc^(k) = 2 ∫_0^1 G_in(r_i,µ)·F_out(r_j,µ)·µ·e^{-k·σ·2Rµ} dµ
```

The `µ·e^{-k·σ·2Rµ}` factor is bounded at µ=0 (the µ kills the 1/µ
divergence; this matches the V1 SymPy result that `µ·T(µ) → 1/(2σR)`
finite). Each bounce can be discretized with standard GL on [0,1].

**This is the M2 recipe from the cross-domain memo** — the direction
to dispatch next. Front C as built today does NOT bypass the
singularity; M2 does.

## Files shipped

- `derivations/diagnostics/diag_phase5_native_c01_orpheus_form.py` —
  µ-resolved F_out, G_in primitives + 5 probes A-E
- `derivations/diagnostics/diag_phase5_native_c02_kbc_magnitudes.py` —
  Sanchez vs Hebert magnitude comparison
- `derivations/diagnostics/diag_phase5_native_c03_consistency.py` —
  F_out / G_in projection against P̃_n recovers Phase 4 P_esc_mode /
  G_bc_mode (within Q=128 quadrature noise)
- `derivations/diagnostics/diag_phase5_native_c04_jacobian_audit.py` —
  finds K_esc=e^{-τ} bug; Phase 4 ω-integral matches u-integral
- `derivations/diagnostics/diag_phase5_native_c05_sigma_scalar.py` —
  Sanchez Q-convergence (off-diagonal converges, diagonal diverges
  linearly with Q); Sanchez↔ORPHEUS scalar conversion fails

## Lessons learned

- **Sphere K_esc is e^{-τ}, NOT (1-e^{-τ})/τ.** The latter is the
  slab/Bickley form. Confused me for an iteration. SymPy V4 already
  flagged this implicitly; explicit reading of
  `CurvilinearGeometry.escape_kernel_mp` is the fix.
- **`compute_P_esc` (legacy rank-1) and `compute_P_esc_mode(n=0)`
  are DIFFERENT** — the legacy form has no `(ρ_max/R)²` Jacobian.
  This is documented in `compute_P_esc_mode`'s docstring but easy to
  miss. The Phase 4 specular_multibounce uses `compute_P_esc_mode`-
  style integrand; `closure="white_hebert"` uses `compute_P_esc`.
- **The matrix-Galerkin divergence (Phase 4 high-N) and the
  continuous-µ diagonal singularity are the SAME pathology**
  (Trefethen-Embree's unbounded multiplication operator) expressed
  in different bases.
- **Front C's premise — "skip basis projection, integrate µ
  directly" — is structurally wrong** because the multiplication
  operator is the singular ingredient, not the basis. M1 inherits
  the singularity; M2 (bounce-resolved) is the right reformulation.

## Next dispatch

If the parent agent wants to attempt continuous-µ once more:

1. **Implement M2 (bounce-resolved) form** for sphere homogeneous.
   K_max truncation at K=20 should be sufficient for τ_R ≤ 5.
2. **Verify each bounce integrand bounded at µ→0** (V1 SymPy already
   proved `µ·T(µ) → 1/(2σR)`).
3. **Probe D smoke test**: k_eff vs white_hebert at thin τ_R = 2.5.
   Expected: ~-0.12% (rank-∞ Hebert ≡ rank-1 Hebert at this fixture
   because rank-1 dominates the eigenvector).
4. If M2 succeeds, ship as `closure="specular_continuous_mu_M2"`
   alongside the matrix form `closure="specular_multibounce"`.

Estimated complexity: ~200 LoC for M2 sphere homogeneous. Multi-
region sphere requires per-bounce tracking of which annuli the
chord crosses (similar to the matrix form's `_chord_tau_mu_sphere`).
