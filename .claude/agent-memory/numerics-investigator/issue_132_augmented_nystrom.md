---
name: Issue #132 augmented-Nyström FALSIFIED
description: User-proposed (N+M)×(N+M) block formulation with J^+(µ_m) as extra unknowns is mathematically equivalent to existing G·R·P Schur reduction; per-angle kernel K(r,µ) has r-dependent endpoint singularity; Marshak-basis convergence plateaus at +2.4% (Issue #100 floor)
type: project
---

Investigation 2026-04-25 of the user's proposal: replace the Hébert
Mark-uniformity assumption by an (N+M)×(N+M) block system where
J^+(µ_m) is M extra unknowns and J^-(µ) = J^+(µ) is enforced as a
constraint per-angle.

# Bottom line

**Falsified.** The augmented block system is **algebraically
equivalent** to the existing K_bc = G·R·P Schur reduction in
``BoundaryClosureOperator``. There is no NEW physics in the
augmented form vs the existing rank-M closure operator. The only
new question it asks is "what M-mode basis to use?" — and the two
answers tested both fail:

- **Naive Nyström-collocation on µ ∈ [0,1]** (the user's literal
  suggestion): K(r_i, µ) has an inverse-square-root endpoint
  singularity at µ_min(r_i) = √(1-(r_i/R)²) which depends on the
  observer. No FIXED µ-grid can resolve it. M=512 GL nodes still
  give 1-10% relative error against the analytical Lambertian
  collapse G_bc(r). Augmented eigensolve oscillates wildly
  (k_eff ranges from 0.34 to 25 across M=1..64 vs cp k_inf=1.5).

- **Marshak Legendre basis** (smooth, absorbs endpoint via the
  µ-orthogonal-polynomial weight): 1G/1R k_eff(M) plateaus at
  +2.4% relative to cp k_inf as M=3..8 — the SAME mode-0/mode-n≥1
  normalisation plateau that Issues #100 / #112 / #132 already
  documented for Class B MR. Rank-M Marshak DOES improve on rank-1
  (M=1: -27%, M=2: -1.1%, M=3+: +2.4%) but cannot beat the
  shipped Hébert (1-P_ss)⁻¹ closure (which is -1.5% on the
  matched chi=[1,0] 2G case). Do NOT ship as a replacement.

# Diagnostic scripts (kept under derivations/diagnostics/)

| Script | What it pins |
|---|---|
| ``diag_sphere_augmented_nystrom_a_kernel_derivation.py`` | K(r_i, µ) derivation via change of variables θ→µ; ½-uniform collapse to G_bc verified to ~1e-3 (limited by endpoint singularity) |
| ``diag_sphere_augmented_nystrom_b_naive_collocation.py`` | Naive Nyström convergence on µ ∈ [0,1]: M=512 still 1-10% error, log-log slope ~ -0.5 to -1.8 (sub-spectral) |
| ``diag_sphere_augmented_nystrom_c_block_equivalence.py`` | M=1: augmented k_eff = Schur k_eff to 6.66e-16. Confirms the block formulation IS the existing Schur reduction, just pre-eliminated. |
| ``diag_sphere_augmented_nystrom_d_solid_w_diagonal.py`` | Augmented system with proper diagonal W(µ_m) = exp(-2Σ_t R µ_m): k_eff catastrophic-divergent across M because per-angle G/P blocks unresolved at endpoint |
| ``diag_sphere_augmented_nystrom_e_marshak_basis.py`` | Marshak Legendre basis at M=1..8: plateaus at +2.4% rel err (the Issue #100 floor) |

# Derivations established

1. **Per-angle kernel** for solid sphere (observer-centred form):

   ```
   K(r_i, µ) = 2π · µ / ((r_i/R)² · |cos θ|) · [exp(-τ_+) + exp(-τ_-)]
   ```

   with cos θ = ±√(1 − (R/r_i)²(1-µ²)) (forward/back branches),
   τ_± = Σ_t · (R µ ∓ r_i |cos θ|), and µ ∈ [µ_min(r_i), 1] where
   µ_min(r_i) = √(1-(r_i/R)²). The kernel is integrable but has
   an inverse-square-root divergence at µ → µ_min.

2. **Surface-to-surface transmission** for solid homogeneous sphere
   in µ-basis is **diagonal**:

   ```
   W(µ_m → µ_n) = exp(-2 Σ_t R µ_m) · δ_{m,n}
   ```

   Because for a chord that starts and ends on the sphere surface,
   the entry-µ-to-inward-normal equals the exit-µ-to-outward-normal
   by spherical symmetry. (See diag_d for derivation.)

3. **Algebraic equivalence**:
   Augmented (N+M)×(N+M) block ⟺ K = K_vol + G·(I-W)⁻¹·P (Schur).

# Why the augmented direction cannot fix Issue #132

The Hébert Mark-uniformity overshoot for source-localized eigenvectors
(chi=[0,1] thermal: +6.6%, 1G/2R: +10.3%) requires CAPTURING the
non-uniform J^+(µ) on the surface. Both candidate bases fail:

- µ-Nyström: r-dependent endpoint singularity blocks fixed-grid
  representation
- Marshak Legendre: mode-0/mode-n≥1 normalisation plateau (Issue #100)
  caps convergence at ~2.4%

**The right next step is NOT this direction**. Likely candidates:

- (a) Source-spectrum-adapted Mark closure: replace the Mark
  uniformity assumption ψ^- = J^-/π with ψ^-(µ) ∝ a polynomial in
  µ whose coefficients are fit per-eigenvector (a Rayleigh-Ritz
  improvement on the closure, not on the discretization).
- (b) Per-cell Galerkin J^+(µ) representation that matches the
  CP-method's "isotropic-on-the-cell-volume but
  cosine-distributed-on-the-surface" assumption — this is the
  Sanchez 1976 multiple-collision K_∞ approach (already cited in
  Sphinx §peierls-class-b-sphere-hebert as "the rigorous
  alternative the Hébert series approximates").
- (c) Accept the Hébert closure as-is. The +6.6% on the
  thermal-emission limit is the documented limitation; it is
  competitive with classical CP at much lower computational cost,
  and the Sphinx page already says so.

# Recommendation

Close Issue #132 with WONTFIX-on-this-direction; document that:

1. The augmented (N+M) Nyström block formulation is algebraically
   equivalent to the shipped K_bc = G·R·P Schur reduction.
2. Per-angle Nyström-collocation on µ is structurally infeasible for
   sphere due to the r-dependent endpoint singularity.
3. Marshak Legendre at higher M hits the Issue #100 calibration plateau.
4. The acceptable closure for Class B is the shipped Hébert
   (1-P_ss)⁻¹; the chi-monotone overshoot is its documented limit.

If a stronger closure is needed, pursue direction (b) above
(Sanchez 1976 multiple-collision K_∞) — separate issue.

# Status of probe scripts

All five probes have a clear pass/fail criterion in their
test functions and are pytest-discoverable. Probe A and B are
candidates for permanent regression promotion (they encode the
**negative result** — proving the path doesn't work — and would
catch a future attempt to re-do this investigation).
