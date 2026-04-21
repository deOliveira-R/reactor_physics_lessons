---
name: Hébert 2009 Applied Reactor Physics Ch.3 — IC method extraction
description: Hébert's DP_N interface-current formulation — general abstract machinery in 2D Cartesian pincells only. No curvilinear (cylindrical/spherical) rank-N treatment. The c_in remapping problem is never addressed; 1D sphere/cylinder IC reduces to scalar. Confirms four-reference synthesis.
type: project
---

# Hébert 2009 — *Applied Reactor Physics* (2nd ed), Chapter 3 "The transport equation"

Source: `/workspaces/ORPHEUS/Hebert(2009)Chapter3.pdf`, 122 pages, Adobe Paper Capture scan. Chapter pagination 67-188 (internal page numbers as referenced below).

## §1 Chapter structure

- §3.1 Particle flux (pp. 67-70)
- §3.2 Derivation of transport equation (pp. 70-…)
- §3.3 Steady-state etc.
- §3.4-3.6 Multigroup, boundary conditions, adjoint
- §3.7 Spherical harmonics / P_N / SP_N (pp. ~88-107)
- **§3.8 The collision probability method** (pp. 107-129)
  - §3.8.1 The interface current method (pp. 109-111) — **central section for this query**
  - §3.8.2 Scattering-reduced matrices and power iteration (p. 112)
  - §3.8.3 Slab geometry (pp. 112-117)
  - §3.8.4 Cylindrical 1D geometry (pp. 117-123)
  - §3.8.5 Spherical 1D geometry (pp. 123-125)
  - §3.8.6 Unstructured 2D finite geometry (pp. 125-130)
- §3.9 Discrete ordinates (pp. 130-…)
- §3.10 Method of characteristics
- §3.11 Monte Carlo method
- Exercises: §3.1-3.12

## §2 The IC method as presented (§3.8.1, pp. 109-111)

**Angular-flux expansion.** At each point r_s of surface ∂V, the outgoing
angular flux is expanded in a *double P_N* (DP_N) series:

```
φ⁺(r_s, Ω)  =  (1/4π) · Σ_ρ φ_ρ⁺(r_s) · ψ_ρ(Ω, N⁺),   Ω·N⁺ > 0      (3.248)
```

with orthonormality (µ-weighted):

```
∫_{Ω·N>0} d²Ω (Ω·N) ψ_ν(Ω,N) ψ_ρ(Ω,N)  =  π δ_{νρ}                  (3.249)
```

**This is µ-weighted partial-current moments, identical to
Sanchez-McCormick 1982 §III.F.1 normalization.** Same basis, same
normalization.

**Worked DP₁ basis in 2D** (Eq. 3.250):

```
ψ_0(Ω,N)  =  1                          (DP-0 component)
ψ_1(Ω,N)  =  3√2 (Ω·N) − 2√2            (first DP₁ component)
ψ_2(Ω,N)  =  2 (Ω·N_⊥)                  (second DP₁ component)
```

Fig. 3.10 on p. 110 illustrates DP-0 + two DP₁ components between two
square (Cartesian) cells.

**Uniform-DP_N assumption.** "*Another usual approximation consists in
assuming that the expansion coefficients φ_ρ⁺(r_s) and φ_ρ⁻(r_s) are
uniform along each side of the cell.*" → uniform-DP_0 or uniform-DP_1
approximations. The ρ = (α, mode) index is split into surface (α) × mode
(ν); uniform ≡ zeroth moment along the surface.

**Response matrices** (Eqs. 3.252-3.255):

```
p_ij                    — CP, region-to-region                 (3.252)
p^(ρ)_{S_α j}           — surface-α mode-ρ into region j        (3.253)  [G_bc analog]
P^(ν)_{i S_β}           — region i out through surface β        (3.254)  [P_esc analog]
P^(ρν)_{S_α S_β}        — surface-α mode-ρ to surface-β mode-ν   (3.255)  [W transmission]
```

All four carry (Ω·N) µ-weights on surface integrals with the basis
functions ψ_ρ. **Same primitives as Sanchez-McCormick Eq. (166)
cluster** — this is literally the same machinery in different notation.

**Closure** (Eqs. 3.260-3.261):

```
φ_i      =  Σ_{β,ν} φ⁻_{ν,β} P^(ν)_{i S_β}  +  Σ_j Q_j p_ij              (3.260a)
φ⁺_{ρ,α} =  Σ_{β,ν} φ⁻_{ν,β} P^(ρν)_{S_α S_β}  +  Σ_j Q_j p^(ρ)_{S_α j}   (3.260b)

φ⁻_{ρ,β} =  Σ_α A^(ρ)_{αβ} φ⁺_{ρ,α}                                      (3.261)
```

Reciprocity (3.256-3.257) and conservation (3.258-3.259) stated but
proofs deferred to references. **Villarino-Stamm'ler normalization**
(Eqs. 3.347-3.352, p. 129) is applied to force conservation a
posteriori — a telling detail, since a correctly-conserving rank-N
closure should satisfy (3.258-3.259) by construction.

## §3 Where the method is actually applied

**CRITICAL FINDING — §3.8.3, §3.8.4, §3.8.5 are SCALAR CP ONLY:**

- **§3.8.3 slab** (pp. 112-117): derives p_ij in slab as scalar
  integrals over E_n functions (Eqs. 3.267-3.294). Boundary condition
  application is scalar: `β⁺` albedo acts on the single p_ij matrix
  (Eq. 3.323). **No DP_N expansion of boundary flux.**

- **§3.8.4 cylindrical 1D** (pp. 117-123): derives p_ij using Bickley
  functions Ki_n (Eqs. 3.295-3.317). Wigner-Seitz white BC (p. 122,
  Eqs. 3.318-3.323):

  ```
  P̃  =  P  +  (β⁺ / (1 − β⁺ P_SS)) · P_iS · p_Sj^T                    (3.323)
  ```

  This is **Stamm'ler's Eq. 34 scalar closure in modern notation.**
  Same identity ORPHEUS F.4 implements. No Legendre moments on the
  cylindrical surface. No DP-1 components. **Scalar only.**

- **§3.8.5 spherical 1D** (pp. 123-125): "Computing the tracking and
  collision probabilities in 1D spherical geometry is similar to the
  cylindrical geometry case." Eqs. 3.324-3.336 give scalar p_ij only.
  **No surface expansion. No inner-surface treatment. No c_in
  remapping discussion. No spherical IC.** The chapter ends 1D
  spherical here and moves to §3.8.6 unstructured 2D.

- **§3.8.6 unstructured 2D** (pp. 125-130): this is where the
  DP_N machinery from §3.8.1 is actually instantiated — but for
  **square pincells** (Fig. 3.19) and BASALA-like 2D Cartesian
  assemblies (Fig. 3.20). The response matrices T, P_vs, P_sv, P_ss
  (Eqs. 3.347, 3.353) are built over 2D-convex Cartesian surfaces. No
  curvilinear cell with a surface subdivided into concentric inner/
  outer faces — and **no 1D hollow-cylindrical or hollow-spherical
  worked example with DP_N**.

## §4 Does Hébert solve the c_in remapping problem?

**NO.** The chapter simply never encounters it:

1. The DP_N IC machinery (§3.8.1) is entirely abstract — it defines
   the primitives but instantiates them only for 2D Cartesian
   pincells.

2. 1D cylindrical and spherical cells are treated with **scalar
   CP + white BC** (§3.8.4-5). This is the textbook F.4 closure.

3. There is no hollow-cylinder or hollow-sphere worked example at any
   rank ≥ 1. There is no k_eff → k_inf convergence demonstration under
   N refinement for curvilinear cells.

4. The **inner surface** of a hollow region is never treated as a
   distinct subsurface with its own DP_N basis. In §3.8.4 the
   cylindrical cell has `r_i` and `r_{i+1}` bounding radii but the
   response matrix p_ij is scalar (pure integrated CP, no per-surface
   modes). Inner/outer surface decomposition with independent DP_N
   expansions on each is simply not done.

5. The `c_in = √(1 − (R/r_0)²(1 − µ²_emit))` outer→inner angle
   remapping that makes `P̃_n(c_in) ≠ P̃_n(µ_emit)` — the structural
   obstruction we diagnosed in our memory `rank_n_closure_four_references_synthesis.md`
   — is never discussed. There is no "angular projection operator,"
   no geometry-adapted basis, no convergence proof for curvilinear
   DP_N IC.

6. The only explicit DP_N example (Eq. 3.250) is for a 2D Cartesian
   interface where N⁺ and N_⊥ are well-defined fixed orthogonal
   directions. This is possible *because the surface is flat* —
   emission and arrival cosines coincide in the flat-interface
   geometry of two abutting square cells. The curvilinear
   complication we face simply doesn't arise in Hébert's example.

## §5 How does it compare to F.4 at N=1?

At N=1 scalar (DP-0 only, uniform) with **one** subsurface per cell and
white BC, Eq. 3.323 is **literally F.4's closure**:

```
P̃  =  P  +  (β / (1 − β P_SS)) · P_iS · p_Sj^T
```

with β = 1 for white BC. The geometric series
`1 + P_SS + P_SS² + …` appears explicitly as a rank-0 scalar operator.
This confirms (for the *cylindrical* case, which is the only 1D
curvilinear CP result actually given) that **Hébert's scalar closure
IS Stamm'ler Eq. 34 = ORPHEUS F.4**.

## §6 Implicit warnings about rank-N reliability

Page 129, p. 129 bottom: **"There is no guarantee that the collision,
escape and transmission probability matrices obtained that way are
conservative. Conservation relations can be forced using a consistent
normalization of the collision probability matrices. Many normalization
techniques are available, but the approach promoted by Villarino and
Stamm'ler in Ref. [30] is highly recommended."** (Eqs. 3.347-3.352.)

**This is a modern textbook admission that the DP_N IC primitives
do NOT satisfy conservation by construction** — which is exactly the
failure mode we diagnosed for Sanchez-McCormick §III.F.1 in the
hollow-sphere N=1, 2, 3, 4 conservation probe. Hébert's solution is
*a posteriori renormalization* (enforce `T s = g` by multiplicative
correction factors z_ℓ). This *is not* a c_in-aware closure; it's a
band-aid that forces the non-conserving transmission matrix back onto
the conservation manifold.

**Does Villarino-Stamm'ler fix the 1.42% plateau?** Not obviously.
It forces probability conservation at the level of integrated matrix
sums (3.258-3.259), but it does not recover the *mode-wise* identity
`W_oo[n,n] + W_io[n,n] = δ_{n,0}` that F.4's physics requires at
σ_t → 0. Could be worth a numerical experiment — see Recipe below.

## §7 Citations from §3.8.1 (Hébert's references)

- Ref. [25] for CP method: likely Carlvik or Roy.
- Ref. [26] for IC method: "Another important category of collision
  probability techniques is the interface current (IC) methods" —
  citation to the original IC literature. Probably Stamm'ler or
  Roy/Askew. Not verified from Ch.3 alone (bibliography is in a
  separate chapter).
- Ref. [30] Villarino-Stamm'ler: the normalization scheme (Eqs.
  3.347-3.352).

## §8 Bottom-line assessment

**Hébert 2009 is NOT the missing reference that solves the c_in remapping.**

- The abstract DP_N machinery is identical to Sanchez-McCormick 1982
  §III.F.1 (same basis, same µ-weight, same primitives, same closure
  form).
- It's only instantiated for **2D Cartesian** cells where the
  emission/arrival-cosine distinction is trivial.
- **All 1D curvilinear cases are scalar CP + Stamm'ler Eq. 34 white
  BC = ORPHEUS F.4**, no exception, no DP_N extension, no convergence
  proof, no worked k_eff → k_inf hollow-sphere example.
- Modern textbook admits conservation failure; solution is
  a posteriori Villarino-Stamm'ler renormalization, not a new closure.

**This is now the 5th independent reference** (Ligou 1982, Sanchez 2002,
Stamm'ler 1983 Ch.IV, Stacey 2007 Ch.9, Hébert 2009 Ch.3) that **either
uses scalar/DP-0 for curvilinear IC or instantiates DP_N only in flat-
interface geometry**. The Sanchez-McCormick 1982 §III.F.1 rank-N ladder
for curvilinear cells has **zero** cross-validation in the textbook
corpus.

## §9 Recommendation for Issue #119

**Branch C — close out with F.4 as production closure.**

Reasoning:
1. Five independent references converge on scalar/DP-0 for 1D
   curvilinear IC. None present a rank-N Legendre ladder with c_in
   remapping.
2. F.4 at N=1 scalar matches Hébert Eq. 3.323 bit-for-bit at the
   primitive level.
3. F.4's 0.077 % residual at σ_t·R = 5, r_0/R = 0.3 is the expected
   accuracy — consistent with the E_2 product-error floor Stacey
   flags on p. 329 (and the `[E_2(Σ)]^N ≠ E_2(NΣ)` error class).
4. Sanchez-McCormick 1982 §III.F.1 is a standalone theoretical
   construction that never crossed into any successor textbook,
   including Hébert's modern Montréal École Polytechnique graduate
   course reference.

**Optional future work (research tag, not for this issue):**
- Implement Villarino-Stamm'ler normalization (Hébert Eqs.
  3.347-3.352) on top of the existing Sanchez-McCormick rank-N
  primitives and measure whether the 1.42% plateau collapses to
  within F.4's 0.077%. If yes → Hébert's textbook does carry a
  load-bearing fix; if no → confirms rank-N is structurally broken
  by c_in without a geometry-adapted basis.
- Explore geometry-adapted basis: `{P̃_n(µ_emit)}` at outer,
  `{P̃_n(c_in(µ_emit))}` at inner — the user's "Direction C" in the
  next-session plan. This is genuinely novel work outside any of the
  five references.

## §10 Citation

A. Hébert (2009). *Applied Reactor Physics*. Presses Internationales
Polytechnique, Montréal. ISBN 978-2-553-01436-9. Chapter 3 "The
transport equation" (pp. 67-188). Referenced as 2nd edition in later
catalogs. 3rd edition (2020) is the current in-print version.

## §11 Hand-off notes

- The abstract DP_N IC framework of §3.8.1 is a clean pedagogical
  statement that matches Sanchez-McCormick 1982 §III.F.1 — useful to
  cite in ORPHEUS Sphinx docs as "this is the textbook IC method we
  attempted, see Hébert Eqs. 3.248-3.261 or Sanchez-McCormick Eqs.
  165-167."
- The lack of curvilinear worked examples + the Villarino-Stamm'ler
  renormalization recommendation are themselves evidence that
  curvilinear DP_N IC is numerically fragile in practice.
- For the ORPHEUS Sphinx theory page on Peierls rank-N closure:
  cite Hébert §3.8.4 Eq. 3.323 as the canonical scalar closure that
  F.4 implements. Cite §3.8.1 Eqs. 3.248-3.261 as the textbook
  statement of the DP_N extension that, per our conservation probe,
  structurally fails for curvilinear cells without further
  corrections.

## §12 Villarino-Stamm'ler renormalization recipe (Eqs. 3.347-3.352, p. 128-129)

The full a-posteriori conservation fix. Everything below is verbatim
from the PDF (image-OCR re-read at 300 dpi, not the garbled text
layer).

### §12.1 The six equations — verbatim

**Eq. (3.347)** — Symmetric T matrix of order (A + I), where A = #
surfaces, I = # volumes. Indices: α, β ∈ {1,…,A}; i, j ∈ {1,…,I}.
The ℓ,m indexing in 3.350-3.352 is the *combined* index on T (surface
block first, then volume block).

```latex
t_{\alpha,\beta} \;=\; \frac{S_\alpha}{4}\, P_{S_\alpha S_\beta},
\qquad
t_{\Lambda+i,\beta} \;=\; V_i\, P_{i S_\beta},
\qquad
t_{\Lambda+i,\Lambda+j} \;=\; V_i\, p_{ij}.
\tag{3.347}
```

(The printed text uses Λ in the subscript offsets; Λ ≡ A.  The
rank-0 versions: `P_{S_α S_β} = P^{(00)}_{S_α S_β}` and
`P_{i S_β} = P^{(0)}_{i S_β}`. T is built from **rank-0 (flat) CP
primitives only.** The V-S scheme is defined on the scalar-CP/DP-0
T; there is no rank-N generalization in Hébert.)

**Eq. (3.348)** — source and response vectors, both column vectors
of length A + I, same ordering as T:

```latex
\mathbf{s} \;=\; \operatorname{col}\!\Big\{(1;\, \alpha=1,\Lambda),\;(\Sigma_i;\, i=1,I)\Big\},
\qquad
\mathbf{g} \;=\; \operatorname{col}\!\Big\{(S_\alpha/4;\, \alpha=1,\Lambda),\;(V_i;\, i=1,I)\Big\}.
\tag{3.348}
```

So `s_ℓ = 1` for surface rows (ℓ = 1…A) and `s_ℓ = Σ_i` for volume
rows (ℓ = A+i); `g_ℓ = S_α/4` for surface rows and `g_ℓ = V_i` for
volume rows.

**Eq. (3.349)** — compact statement of reciprocity + conservation:

```latex
\mathbb{T}\,\mathbf{s} \;=\; \mathbf{g}.
\tag{3.349}
```

Symmetry of T = reciprocity (3.256-3.257).
`T s = g` row-by-row = conservation (3.258-3.259) with Σ_α S_α/4
and Σ_j V_j on the RHS.

**Eq. (3.350)** — additive symmetric correction: T is replaced by
`t̂_{ℓm} = (z_ℓ + z_m) t_{ℓm}`. The correction is **additive in the
two row/column factors** (Villarino-Stamm'ler) rather than
multiplicative (would be `z_ℓ z_m`, Bonalumi-style).

```latex
\hat{t}_{\ell m} \;=\; (z_\ell + z_m)\, t_{\ell m};
\qquad \ell,m = 1,\Lambda+I.
\tag{3.350}
```

**Eq. (3.351)** — substitute (3.350) into (3.349), solve for z:

```latex
z_\ell \sum_m t_{\ell m}\, s_m \;+\; \sum_m z_m\, t_{\ell m}\, s_m
\;=\; \sum_m \left[ \delta_{\ell m} \sum_k t_{\ell k}\, s_k \,+\, t_{\ell m}\, s_m \right] z_m
\;=\; g_\ell;
\qquad \ell = 1,\Lambda+I.
\tag{3.351}
```

**Eq. (3.352)** — Gauss-Seidel fixed-point iteration for z. The
index n is the **V-S inner iteration** (separate from the power
iteration n):

```latex
z_\ell^{(n+1)} \;=\;
\frac{g_\ell \,-\, \left[ \displaystyle\sum_{m<\ell} t_{\ell m}\, s_m\, z_m^{(n+1)} \;+\; \sum_{m>\ell} t_{\ell m}\, s_m\, z_m^{(n)} \right]}
     {t_{\ell\ell}\, s_\ell \,+\, \displaystyle\sum_m t_{\ell m}\, s_m},
\qquad \ell = 1,\Lambda+I.
\tag{3.352}
```

Initialize with `z_ℓ^{(0)} = 1/2 ∀ℓ` (so that `(z_ℓ + z_m) = 1` →
no correction at iteration 0, i.e., start from the raw T).
Livolant acceleration (Hébert Appendix C.1.3) is recommended for
speed-up but not required.

### §12.2 Eight questions answered

1. **Input matrices** — V-S operates on the assembled rank-0 T
   (3.347), which packs `p_ij` (vol-vol), `P_{iS_β}` (vol-surf),
   and `P_{S_α S_β}` (surf-surf) into one symmetric (A+I)×(A+I)
   matrix. V-S is **rank-0 only**; Hébert does not extend it to the
   mode-indexed `P^{(ν)}_{iS_β}`, `P^{(ρν)}_{S_α S_β}`.

2. **Conservation identity enforced** — `T s = g` (Eq. 3.349).
   Row-by-row this is: for a volume row (ℓ = A+i), `Σ_α P_{iS_α} ·
   1 + Σ_j p_{ij} · Σ_j = V_i / V_i = 1` (exactly Eq. 3.258); for
   a surface row (ℓ = α), `Σ_β P_{S_α S_β} · 1 + Σ_j p_{S_α j} ·
   Σ_j = (S_α/4)/(S_α/4) = 1` (Eq. 3.259). The 0th moment only.

3. **Correction factors z_ℓ** — one per row/column of T, so A+I
   total. Computed by **Gauss-Seidel fixed-point iteration** (Eq.
   3.352), converged to a tolerance on `‖T̂s − g‖`. One-shot direct
   solve is not given; iteration is the prescribed method.

4. **Correction application** — **additive and symmetric**:
   `t̂_{ℓm} = (z_ℓ + z_m) t_{ℓm}`. The "1/2" initial condition is
   the choice that makes `(z_ℓ + z_m) = 1` at startup. It is NOT
   `z^{1/2} T z^{1/2}` (multiplicative).

5. **Reciprocity preservation** — **YES**. `t̂_{ℓm} = (z_ℓ + z_m) t_{ℓm}
   = (z_m + z_ℓ) t_{mℓ} = t̂_{mℓ}` since T is symmetric and the `(z_ℓ
   + z_m)` scalar is symmetric in ℓ↔m. Hébert explicitly states
   "reciprocity … are satisfied if matrix T remains symmetric"
   (p. 129, above Eq. 3.349), and the additive form preserves
   symmetry by construction.

6. **Per-mode / per-surface / both** — V-S is **per-row on the
   combined surface+volume index ℓ**. Not per-mode (only rank-0
   primitives enter T in Hébert), not per-region-only, not
   per-surface-only — it's one factor per row of the *joint* T.

7. **Ref. [30] full citation** — E. A. Villarino and R. J. J.
   Stamm'ler (1992), *"HELIOS: Angularly Dependent Collision
   Probabilities,"* Nuclear Science and Engineering **112**(1),
   16-31.  DOI: 10.13182/NSE112-16. Not in Rodrigo's Zotero
   library; worth adding for Issue #119 provenance.

8. **Numerical results in Hébert** — **NONE** in Chapter 3. The
   BASALA 2D example on p. 126 (Fig. 3.20) illustrates the
   geometry but no table of V-S before/after error is provided.
   Hébert defers to Ref. [30] for validation. He recommends V-S
   but offers no quantitative demonstration.

### §12.3 Implementation cost estimate

**Small enough for a day, but unlikely to help the Sanchez rank-N
plateau. Decision: Branch C (close out).**

- **Code size**: 30-50 lines of Python.  The V-S iteration is
  ~20 lines; assembly of T, s, g from existing Sanchez primitives
  is ~15 lines; validation (check `‖T̂s − g‖ < tol`) is ~5 lines.
  Gauss-Seidel convergence is fast (Hébert implies ~10 iterations
  without Livolant acceleration).

- **Infrastructure**: zero new infrastructure. V-S takes as input
  the existing flat (rank-0) `P_ij`, `P_iS`, `P_SS` matrices and
  outputs renormalized versions. All ORPHEUS already has these at
  N = 0.

- **Why it probably won't close the 1.42% plateau on Sanchez
  rank-N primitives**:
  1. **V-S is defined on rank-0 only.** Hébert never extends
     3.350 to the mode-indexed `P^{(ρν)}_{S_α S_β}`. Applying V-S
     to the ν = ρ = 0 sub-block gives you DP-0 conservation,
     which ORPHEUS F.4 already satisfies exactly. It does **not
     give you rank-N conservation**.
  2. **The 1.42% plateau is a mode-coupling failure (`c_in`
     remapping), not a conservation violation.** V-S forces rank-0
     sums to unity; it does nothing to correct the fact that
     `W^{nn}_{oi} + W^{nn}_{oo} ≠ δ_{n,0}` for n > 0 in
     curvilinear cells. The error class is different.
  3. **Naive extension to per-mode** (one z_{(ρ,α)} per surface
     mode and one z_{(0,i)} per region, with identities
     `Σ_{β,ν} P^{(ρν)}_{S_α S_β} + Σ_j p^(ρ)_{S_α j} Σ_j = δ_{ρ,0}`)
     would require (A × (N+1) + I) correction factors per cell per
     group. Hébert does NOT write this down; no successor textbook
     does either. It's an unvalidated extension of V-S.

- **Recommended scope**: *if* you want to spend the half-day to
  confirm our hypothesis is correct, implement V-S at rank-0 on
  the current (converged-for-rank-0) matrices and verify the
  plateau is unchanged. This falsifies the "V-S rescue" hypothesis
  cheaply. If instead you accept the theoretical argument above,
  skip the implementation and close Issue #119 on Branch C with
  F.4 as production closure.

### §12.4 Suggested add to Zotero

Villarino, E. A. & Stamm'ler, R. J. J. (1992). *HELIOS: Angularly
Dependent Collision Probabilities*. Nuclear Science and Engineering
112(1), 16-31. DOI: 10.13182/NSE112-16. Worth having for Issue
#119 provenance and in case anyone returns to rank-N rescue
experiments. (Agent does not write to the library; user action.)
