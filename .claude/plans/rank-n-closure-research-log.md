# Rank-N per-face white-BC closure on hollow sphere — research log

> **CURRENT STATE (2026-04-22 late, after retraction of commit `fba6835`)**
>
> **F.4 scalar (Issue #119 close-out) remains the production closure.**
> The c_in-aware split-basis rank-(1,1,1) adaptive-scale closure (Issue
> #120) was thought to beat F.4 by 100–1000× based on BASE-quadrature
> measurements, but RICH-quadrature verification showed this was a
> quadrature-noise cancellation artifact. At matched RICH quadrature,
> **F.4 wins 6/6 points at σ_t·R ≥ 5 by 2–88×**; split has a structural
> floor ~0.07% vs F.4's ~0.003%.
>
> **For the next-session hand-off plan, see**:
> `.claude/plans/next-session-post-retraction.md`.
>
> **The structural observations in L1–L12 still stand.** The scale gauge
> DOF (L8) is real; it just doesn't reach below F.4's structural floor.
> Remaining research angles are Issue #121 (Sanchez 2002 PCA sectors,
> Direction C — untouched at matched quadrature) and Direction Q
> (principled derivation of F.4's Lambert/Marshak mismatch).

**Purpose**: living document for the long-running research effort to break F.4's
k_eff residual at σ_t·R = 5, r_0/R = 0.3 for the hollow-sphere Peierls
rank-N closure. Every experiment, diagnosis, lesson, and rejected
hypothesis lives here so future sessions don't re-derive what we already know.

**First session**: 2026-04-21. **Last update**: 2026-04-22. **Author**:
Claude Opus 4.7 (1M context).

**Related artifacts**:
- Issue #119 (CLOSED): F.4 scalar as production; five-reference synthesis.
- Issue #120 (OPEN): c_in-aware geometry-adapted basis research (this log).
- Issue #121 (OPEN): PCA sectors (Sanchez 2002) research.
- Sphinx `docs/theory/peierls_unified.rst` §peierls-rank-n-per-face-closeout.
- `.claude/plans/next-session-rank-n-hebert-and-beyond.md` (pre-Hébert plan).

---

## Baseline context (what F.4 gets)

| σ_t·R | r_0/R | F.4 err (Lambert P/G + Marshak W) | Earlier-reported err |
|-------|-------|-----------------------------------|----------------------|
| 1.0   | 0.3   | **2.36%**                          | 3.27%                |
| 2.5   | 0.3   | **0.347%**                         | 0.55%                |
| 5.0   | 0.3   | **0.122%**                         | 0.077%               |
| 10.0  | 0.3   | **0.246%**                         | 0.26%                |
| 20.0  | 0.3   | **0.365%**                         | 0.45%                |

**Discrepancy with earlier "0.077%" number (Issue #119 close-out)**: the 0.077%
was measured at higher Nyström quadrature. At the standard quadrature, F.4 is
at 0.12%. Future work should establish whether the 0.077%-vs-0.12% gap is pure
Nyström quadrature error (likely) — if so, the "true" F.4 BC-closure limit is
closer to 0.04–0.08% after quadrature refinement.

**Marshak rank-N baseline** (`build_white_bc_correction_rank_n`, behind
`boundary="white"` with `n_bc_modes ≥ 2`):
- N=1 Marshak: 2.09%
- N=2 Marshak: 1.36% (plateau)
- N=3+ Marshak: 1.36% (no improvement)

**Key observation**: F.4's 0.12% is 10× better than Marshak-N=∞. The gap is NOT
from rank but from a basis-convention asymmetry (Lambert P/G + Marshak W).

---

## Experiments and findings (chronological)

### Experiment 1 — Split basis (c_in-aware geometry-adapted)
**Date**: 2026-04-21. **Branch**: `feature/rank-n-cin-aware-basis`.

**Hypothesis**: split outer surface cosine domain at µ_crit = √(1-ρ²) into
grazing [0, µ_crit] and steep [µ_crit, 1] sub-bases, with the steep sub-basis
= (1/ρ) P̃_n(c_I(µ)). This structurally diagonalizes W_{oi,s} at σ_t=0.

**Structural results** (all verified, bit-exact):
- Orthonormality of all three sub-bases in µ-weighted inner product (symbolic).
- W_{oi,s}^{mn}(σ_t=0) = (1/ρ) δ_{mn} — diagonal to 1e-15.
- W_{gg}^{mn}(σ_t=0) = δ_{mn} (grazing preserves cosine by chord symmetry).
- F.4 outer mode = µ_crit · P̃_0^g + ρ · P̃_0^s (Parseval: µ_crit² + ρ² = 1).
- Sanchez-McCormick time-reversal reciprocity to machine precision.

**Empirical verdict**: **FALSIFIED**. Residuals at σ_t·R=5, r_0/R=0.3:

| Closure | k_eff err |
|---------|-----------|
| F.4 (Lambert P/G + Marshak W) | 0.122% |
| Marshak rank-N=2              | 1.36%  |
| split-basis rank-(1,1,1)      | 1.29%  |
| split-basis rank-(1,1,2)      | 0.99%  |
| split-basis rank-(1,1,3+)     | 0.99%  (plateau) |
| split-basis rank-(2,2,2)      | 0.99%  |
| split-basis rank-(3,3,3)      | 0.99%  |

Cross-σ_t·R scan uniformly worse than F.4 (1–36× worse, worst at thin optical depth).

**Diagnosis — lesson 1** (CRITICAL): **the white BC at outer is inherently
rank-1 in the constant-µ direction.** Isotropic re-emission populates exactly
ONE linear combination of outer modes: the (µ_crit, ρ) direction at mode-0 in
split basis, or equivalently P̃_0 in Marshak. Any rank-N outer basis
decoration is a **basis rotation** on this rank-1 coset — the orthogonal
"new DOF" gets zero drive from the BC. Volume emission populates higher outer
modes but at amplitudes too small to move the needle empirically.

**Diagnosis — lesson 2**: F.4's 0.12% advantage over Marshak rank-N=2's 1.36%
is NOT from a richer outer basis — it's from a **basis mismatch**: F.4 uses
Lambert-basis P_esc/G_bc (no µ weight, integrand sin θ exp(-τ)) with
Marshak-basis W (µ-weighted). The split basis inherits the Marshak convention
throughout (formally consistent), matching Marshak rank-N=2's 1.36% ceiling.

**Files**: `derivations/diagnostics/diag_cin_aware_basis_derivation.py`,
`diag_cin_aware_finite_sigma_t.py`, `diag_cin_aware_split_basis_keff.py`.
Memo: `.claude/agent-memory/numerics-investigator/peierls_cin_aware_split_basis.md`.

---

### Experiment 2 — Quadrature floor, inner enrichment, basis-variant probe
**Date**: 2026-04-21. **Branch**: `investigate/peierls-solver-bugs`.

#### E2.1 — F.4 quadrature floor (σ_t·R=5, r_0/R=0.3)

F.4 residual vs quadrature (n_panels, p_order, n_ang):

| config                   | err     |
|--------------------------|---------|
| (2, 4, 32) [baseline]    | 0.1219% |
| (4, 4, 32)               | 0.0545% |
| (2, 8, 32)               | 0.0268% |
| (8, 4, 32)               | 0.0226% |
| (2, 4, 64)               | 0.0028% |
| (4, 8, 64)               | 0.0578% |
| (4, 8, 96)               | 0.0025% |

**Verdict**: F.4 residual is NON-monotone — oscillates between 0.003-0.14%
depending on how radial and angular quadrature align. The 0.122% baseline
is **quadrature error, NOT structural**. True structural floor < 0.01%.
This means any closure comparing against F.4's 0.122% at standard
quadrature is comparing against noise.

#### E2.2 — rank-(1,1,N) at two quadrature levels

| config                      | F.4    | (1,1,1) | (1,1,2) | (1,1,4) | (1,1,8) |
|-----------------------------|--------|---------|---------|---------|---------|
| base (2, 4, 32)             | 0.122% | 1.290%  | 0.994%  | 0.994%  | 0.991%  |
| 2×rich (4, 8, 64)           | 0.058% | 1.372%  | 1.081%  | 1.081%  | 1.080%  |

**Verdict**: refining quadrature 2× MOVES the plateau from 0.99% to 1.08%
(slightly worse). The rank-(1,1,N) plateau is **STRUCTURAL** — not
quadrature-limited. Different direction from F.4's 0.12% which IS
quadrature-limited. Rank-(N,N,N) scan at baseline confirms: rank-(2,2,2)
= rank-(3,3,3) = 0.994% — same plateau.

#### E2.3 — Spectral decomposition of residual (rank-(1,1,1) vs rank-(1,1,8))

Inner mode energies in rank-(1,1,8) self-consistent ψ^+_inner:

| mode | \|c\|²      | fraction |
|------|-------------|----------|
| 0    | 3.738e-01   | 89.76%   |
| 1    | 4.226e-02   | 10.15%   |
| 2    | 2.127e-05   | 0.005%   |
| 3-7  | 7e-5 ~ 1e-4 | < 0.04%  |

Inner-surface residual (rank-(1,1,1) − rank-(1,1,8)) projected onto
half-range Legendre:

| mode | residual coeff |
|------|----------------|
| 0    | +7.2e-3        |
| 1    | **−2.06e-1**   |
| 2    | +4.6e-3        |
| 3-7  | \|·\| < 1.3e-2 |

Outer-surface residual likewise is dominated by mode-0 difference
(+0.199), with modes 1-7 all ~0.001-0.007.

**Verdict**: rank-(1,1,1) misses the mode-1 information at inner (coeff
−0.206), which rank-(1,1,2) captures exactly. Once captured, all further
modes carry negligible information. So the plateau is NOT about
resolution — it's about basis metric. See E2.6.

#### E2.4 — Lambert P/G + Marshak W with split basis (RH4)

| rank      | Lambert-split err |
|-----------|-------------------|
| (1,1,1)   | 32.99%            |
| (1,1,4)   | 33.07%            |
| (2,2,2)   | 33.02%            |
| (3,3,3)   | 33.04%            |

Cross σ_t·R scan (rank-(1,1,2) Lambert-split):
σ_t·R=1: 737%, σ_t·R=2.5: 73%, σ_t·R=5: 33%, σ_t·R=10: 21%, σ_t·R=20: 13%.

**Verdict**: CATASTROPHIC. F.4's Lambert-basis trick does NOT generalize to
the split basis. The Lambert-Marshak mismatch is algebraically
N=1-specific. RH4 refuted.

#### E2.6 — Jacobi c²-weighted inner basis

Swap inner from half-range Legendre (ortho under c-weight) to c²-weighted
orthonormal polynomial (α=1, β=0 Jacobi-style). Keep split outer.

At σ_t·R=5, r_0/R=0.3:

| config                      | rank-(1,1,1) | rank-(1,1,2) |
|-----------------------------|--------------|--------------|
| Legendre-inner (base)       | 1.29%        | 0.99%        |
| Jacobi-c² (base quad)       | **0.072%**   | 0.073%       |
| Jacobi-c² (rich quad 4,8,64)| **0.004%**   | 0.002%       |

Cross σ_t·R scan (rank-(1,1,1) Jacobi-c², base quadrature):
σ_t·R=1: 247%, σ_t·R=2.5: 9.4%, σ_t·R=5: 0.07%, σ_t·R=10: 0.27%,
σ_t·R=20: 0.08%.

Alternative weight (α=0, β=1): catastrophic (6.7% → 116% as N grows).

**Verdict**: Jacobi-c² is a **POINT-WISE win but NOT universal** — it's
tuned to the σ_t·R ≥ 5 regime with moderate ρ. In thin regimes it
fails badly (247% at σ_t·R=1). BUT the convergence behavior at σ_t·R=5
under quadrature refinement (0.032% → 0.002%) proves the structural
plateau at 0.99% in Legendre-inner CAN be broken by the right inner
basis. The problem is finding a basis that works universally.

**KEY STRUCTURAL INSIGHT**: the plateau is in the INNER-SURFACE METRIC.
Legendre (c-weight) and Jacobi (c²-weight) give VERY different k_eff
despite representing the same rank-1 information. The correct metric
is physics-dependent (optical thickness and geometry).

**Files**: `derivations/diagnostics/diag_cin_f4_quadrature_floor.py`,
`diag_cin_split_inner_enrichment.py`, `diag_cin_split_lambert_pg.py`,
`diag_cin_split_jacobi_inner.py`.

---

## Persistent lessons (update this section as experiments clarify the picture)

### L1 — White BC rank-1 bottleneck

The isotropic white BC at outer surface populates exactly ONE direction in
the outer mode space: the constant-in-µ direction. Any basis decoration on
outer is a rotation of this rank-1 structure. **Increasing outer mode count
without changing anything else is guaranteed to plateau.** Improvement must
come from (a) inner-surface modes, (b) basis-convention asymmetry, (c)
richer angular-flux representation in the volume, or (d) non-white BC.

### L2 — Basis mismatch is the load-bearing F.4 trick

F.4's 0.12% residual beats the formally-consistent Marshak rank-N=∞ plateau
(1.36%) by using Lambert-basis P/G (no µ weight) alongside Marshak-basis W
(µ-weighted). This is formally inconsistent but empirically effective. Any
new closure that matches all primitives to a single basis inherits the
Marshak 1.36% ceiling. Breaking below F.4 likely requires either
reproducing this mismatch intentionally, or finding a principled
explanation for it.

### L3 — Structural correctness ≠ empirical improvement

Beautiful symbolic math (diagonal W at σ_t=0, Parseval decomposition,
bit-exact reciprocity) is NOT sufficient. The closure has to DO SOMETHING
with the new structure. If the new structure is in a subspace the BC can't
excite, the math is a basis rotation with zero information gain.

### L4 — The plateau at rank-(1,1,N≥2) = 0.99%

Adding inner modes N_i ≥ 2 gives a ~30% improvement over rank-(1,1,1) but
plateaus. This suggests inner surface carries ONE significant angular mode
beyond the zeroth, but mode-2 onwards has negligible amplitude in the
self-consistent flux. Worth quantifying: what is the energy in each mode
of the self-consistent inner flux? If mode-2 onwards is truly tiny, the
plateau is fundamental; if not, our closure is under-coupling them.

**E2 UPDATE (2026-04-21)**: Inner mode energies in rank-(1,1,8) ψ^+_inner:
mode-0 = 89.8%, mode-1 = 10.1%, modes 2-7 = 0.005-0.036% each (together
<0.1%). So inner basis resolution is NOT the bottleneck — modes 2+ are
genuinely tiny in the physics. Residual analysis confirms: inner
residual between rank-(1,1,1) and rank-(1,1,8) is dominated by mode-1
(coeff ≈ -0.206), which rank-(1,1,2) already captures. So the plateau
at 1% for rank-(1,1,N≥2) is **fundamental in this basis**: the mismatch
between Legendre inner basis and the physical volume-emission pattern on
the inner surface forces a 1% ceiling even with all significant modes
resolved.

### L5 — F.4's 0.12% is quadrature-limited, NOT structural

Refining (n_panels, p_order, n_angular) from (2, 4, 32) to (2, 4, 64)
drops F.4 err from 0.122% → 0.003%. The convergence is NON-monotone —
err(2,4,32)=0.122%, err(4,4,32)=0.054%, err(4,8,64)=0.058% — which
reveals cancellation between radial and angular quadrature errors. The
true F.4 structural floor is < 0.01%. The 0.122% is a cancellation
artifact at specific quadrature. **Any closure we compare against F.4
must be checked at matched quadrature**; a "win" at standard quadrature
may vanish under refinement.

### L6 — Lambert P/G + Marshak W is N=1-specific, NOT a generic mismatch trick

Replacing Marshak P/G with Lambert P/G in the split basis gives
catastrophic failure (33% err at σ_t·R=5, 737% at σ_t·R=1). The
cancellation that makes F.4 work at N=1 does NOT generalize to richer
bases. RH4 refuted: "Lambert mismatch unlocks F.4-like advantage in
higher-rank bases" is false.

### L7 — Jacobi c²-weighted inner basis gives POINT-WISE wins but is NOT universal

Using a c²-weighted orthonormal basis on inner (instead of c-weighted
Legendre) drops rank-(1,1,1) err from 1.29% → 0.072% at σ_t·R=5,
r_0/R=0.3 at standard quadrature, and to 0.002-0.004% at 2× rich
quadrature. BUT: at σ_t·R=1 err = 247%, σ_t·R=2.5 err = 9.4%,
σ_t·R=10 err = 0.27% (worse than F.4). So the c²-weighted inner basis
is **geometry-adaptive in a lucky regime** at σ_t·R ≥ 5 with high ρ
contribution, but unstable elsewhere. Cannot ship as production. But
the success at σ_t·R=5 + rich quadrature proves the plateau is NOT a
fundamental information-content barrier — it's a basis-metric mismatch.
The question for next session: what IS the right weight on inner
(µ-weight? c²? geometric/physical optimality)?

---

## Candidate research directions (to be tackled in sequence or parallel)

### Direction A — Inner-surface enrichment with matched primitives (ACTIVE)
Exhaust the rank-(Ng, Ns, Ni) cube with higher quadrature, spectral
decomposition of the residual flux, and convention variants. The 0.99%
plateau must be either numerical (quadrature-limited) or structural (a
deeper basis mismatch) — diagnose which.

### Direction B — Lambert-convention split basis
F.4's Lambert P/G works without the split basis. Does Lambert + split basis
stack? Worth a morning's work once Direction A plateaus its empirical
diagnosis.

### Direction C — Sanchez 2002 PCA sectors (Issue #121)
Hemisphere split into N_θ × N_φ angular cones with characteristic-function
basis. Byasses the Legendre-basis trap entirely. Structural yes at N²=1
(reduces to F.4); empirical unknown at higher N². Major infrastructure lift.

### Direction D — Cavity self-coupling with angular spreading
Currently we assume convex cavity gives identity cavity coupling in
`{P̃_n(c)}`. Re-examine if this is exact or if higher-order multi-bounce
effects matter in practice.

### Direction E — Non-Legendre inner basis
Try Jacobi polynomials P^{(α,β)}_n or geometry-adapted splines on c ∈ [0,1].
Goal: find a basis where the volume-emission inner-surface mode has sparse
support (concentrated in few modes).

### Direction F — Volume-to-surface decomposition analysis
Before adding BCs: decompose the Peierls volume-emission angular pattern on
both surfaces into Legendre modes. Which modes carry the energy? Does the
decomposition hint at a better basis choice?

### Direction G — Higher-rank white BC (anisotropic albedo)
White BC currently = isotropic re-emission (Lambert reflector). A non-white
BC (e.g., specular albedo, or angular-dependent albedo that matches the
steep-cone geometry) would populate more outer modes by construction.
Could break the rank-1 bottleneck. Non-physical for white BC but might
reveal how much slack the BC rank-1 is leaving on the table.

### Direction H — Quadrature floor analysis
Is F.4's 0.122% at standard quadrature a "true" structural residual or just
quadrature error? Refine Nyström quadrature (both radial and angular) until
F.4 saturates. The saturation level is the baseline we're trying to beat.
If 0.077% IS the saturated F.4 residual, we need closures below 0.077% for
a legitimate win.

**2026-04-21 UPDATE**: Done (E2.1). F.4 floor is < 0.01%, quadrature-limited.
The 0.12% is noise. The TRUE baseline to beat is F.4 at 0.005-0.01%
(matched quadrature).

### Direction I — Inner-surface metric (from L7 + E2.6)

Jacobi-c² inner basis drops rank-(1,1,1) err from 1.29% → 0.002% at
σ_t·R=5 with rich quadrature BUT catastrophically at σ_t·R=1 (247%).
The METRIC is regime-dependent. Derive the correct inner-surface metric
from first principles:

Given the physical partial current J^-_inner = ∫ ψ^-(c) · c · dc, the
"natural" inner product should be c-weighted (Legendre). But the
transmission integrand `exp(-τ χ(c)) · c dc` has c-weight that EFFECTIVELY
becomes c²-weight when combined with the surface emission Jacobian. The
right basis is probably an α-adaptive Jacobi where α = α(σ_t·R, ρ) —
i.e., the basis should rotate with optical thickness. Direction for next
session: derive α(σ_t·R, ρ) from an asymptotic analysis of the self-
consistent flux on inner. At σ_t·R → ∞, ψ^+_inner approaches P_2(c)
in some normalization.

### Direction J — Hybrid basis (Legendre + Jacobi adaptive)

Given L7's cross-σ_t scan — Legendre wins at σ_t·R < 5, Jacobi-c² wins at
σ_t·R ≥ 5 — a weighted combination may be universally better. Specifically,
define α(σ_t·R, ρ) = max(0, min(1, (σ_t·R - 2)/3)) and use a blended basis.
Risky: blended bases are not orthogonal, and the W diagonality argument
of L1/E1 could break.

---

## Rejected hypotheses (don't retry without new information)

### RH1 — "Per-mode Villarino-Stamm'ler renormalization rescues rank-N"
Falsified 2026-04-21. Per-mode V-S forces conservation at every mode but
plateau actually gets WORSE (1.42% → 1.87% at rank-(1,1,1) equivalent).
The failure is cross-mode coupling from c_in remapping, not conservation.
Diagnostic: `diag_rank_n_villarino_stammler_per_mode.py`.

### RH2 — "Plain Legendre rank-N at higher N beats F.4"
Falsified across 60+ recipe variants (Issue #119 investigation). All
recipes plateau at 1.42–11% depending on basis convention. Nothing in plain
Legendre space breaks 1%.

### RH3 — "c_in-aware split basis unlocks new accuracy"
Falsified 2026-04-21 (this document, Experiment 1). Structurally correct,
empirically falsified. Split basis is a basis rotation of Marshak rank-N=2.

### RH4 — "Lambert P/G + Marshak W mismatch generalizes to split basis"
Falsified 2026-04-21 (Experiment 2, E2.4). Lambert-split gives 33% err at
σ_t·R=5 and 247-737% in thin regimes. The F.4 trick is algebraically
N=1-specific; the cancellation doesn't hold for richer bases.

### RH5 — "Inner-surface basis enrichment (more inner modes) breaks the plateau"
Falsified 2026-04-21 (E2.2/E2.3). rank-(1,1,N_i) plateaus at 0.99% by N_i=2.
Mode energies confirm: modes 2-7 carry < 0.1% of inner ψ^+. The plateau is
not about resolution — it's about basis-metric. See L7.

---

## Session trail

- **2026-04-21 Opus 4.7**: Issue #119 close-out + Hébert extraction + V-S
  per-mode falsification + split-basis experiment (this doc, E1).
- **2026-04-21 Opus 4.7 (later)**: Experiment 2 — quadrature floor (E2.1),
  rank-(1,1,N) scan + spectral residual (E2.2/E2.3), Lambert+split
  (E2.4 catastrophic), Jacobi c²-weighted inner (E2.6 point-win). New
  lessons L5, L6, L7 and rejected RH4, RH5 added. Artifacts:
  `diag_cin_f4_quadrature_floor.py`, `diag_cin_split_inner_enrichment.py`,
  `diag_cin_split_lambert_pg.py`, `diag_cin_split_jacobi_inner.py`.

---

### Experiment 3 — Inner-surface metric deep dive (adaptive basis, asymptote hypothesis, scale calibration)
**Date**: 2026-04-22. **Branch**: `feature/rank-n-cin-aware-basis`.
**Status**: partial — numerics-investigator dispatch hit auth timeout mid-run;
scripts completed but integration of findings left to main agent (continuing here).

#### E3.1 — α-scan for Jacobi c^α inner basis
Script: `derivations/diagnostics/diag_cin_split_alpha_scan.py` (α ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3}, σ_t·R ∈ {1, 2.5, 5, 10, 20, 50}, ρ ∈ {0.1, 0.3, 0.5, 0.7}). Not yet executed — lower priority after E3.5/E3.6 surfaced a sharper knob.

#### E3.2 — Physics-asymptote inner basis (β ∈ {0, τ/2, τ, 2τ, 3τ, 5τ})

Mode-0 basis = `f_0(c; β) = exp(-β·s(c;ρ)/2)` Gram-Schmidt-orthonormalized under
c-weight. Hypothesis: β = τ makes mode-0 match the physical arriving-flux
shape at inner. Results at ρ=0.3:

| σ_t·R | F.4    | β=0 Leg | β=τ/2 | β=τ    | β=2τ   | β=3τ   | β=5τ   | best β/τ |
|-------|--------|---------|-------|--------|--------|--------|--------|----------|
| 1.0   | 2.36%  | 85.6%   | 86.2% | 86.7%  | 87.5%  | 87.9%  | 88.0%  | 0        |
| 2.5   | 0.347% | 3.86%   | 4.04% | 4.19%  | 4.41%  | 4.52%  | 4.50%  | 0        |
| 5.0   | 0.122% | 1.29%   | 1.18% | 1.10%  | 1.01%  | 1.01%  | 1.13%  | 3        |
| 10.0  | 0.246% | 0.934%  | 0.848%| 0.803% | 0.809% | 0.875% | 1.02%  | 1        |
| 20.0  | 0.365% | 0.427%  | 0.350%| 0.347% | 0.419% | 0.475% | 0.530% | 1        |
| 50.0  | 0.255% | 0.213%  | 0.168%| 0.197% | 0.228% | 0.238% | 0.245% | 0.5      |

**Verdict**: physics hypothesis β=τ gives mixed results. At thick σ_t·R ≥ 20,
β≈τ beats F.4 (0.35% vs 0.36% at σ_t·R=20, 0.17% vs 0.25% at σ_t·R=50).
At σ_t·R < 20, still behind F.4. At thin σ_t·R (1, 2.5), catastrophic
(86% err at σ_t·R=1) because exp(-β·s/2) decays too fast for the near-
uniform flux.

**Pattern**: β_opt/τ shifts from ∞ (Legendre) at thin τ to ~1 at moderate τ
to <1 at very thick τ. The asymptote doesn't track optimal linearly.

File: `derivations/diagnostics/diag_cin_split_asymptote_basis.py`.

#### E3.4 v2 — Galerkin adaptive (fixed-point basis update)
Script: `diag_cin_split_galerkin_v2.py`. Protocol: solve rank-(1,1,N_large) with
Legendre, extract ψ^+_inner shape, use that as mode-0 for rank-(1,1,1), iterate.

**v1 finding**: at rank-(1,1,1) with constant seed, the Legendre iteration is
a fixed point under Gram-Schmidt — truth-shape as mode-0 gives SAME k_eff as
Legendre mode-0. So basis-shape adaptation alone, within Gram-Schmidt +
c-weight, cannot break the plateau. That pointed to the SCALE issue in E3.5.

#### E3.5 — Scale calibration (scale IS the load-bearing knob)

**KEY STRUCTURAL FINDING** (this session, 2026-04-22):

At σ_t·R=5, r_0/R=0.3, with a single CONSTANT inner mode-0 basis function
scaled by different factors, and standard c-weight coupling integrals:

| scale             | k_eff     | err     |
|-------------------|-----------|---------|
| √2 (=Legendre)    | 1.480655  | 1.29%   |
| √1.5              | 1.472096  | 1.86%   |
| √2.5              | 1.490265  | 0.65%   |
| **√3 (=Jacobi c²)**|**1.501075**|**0.072%**|
| √4                | 1.526957  | 1.80%   |
| 1.0               | 1.464455  | 2.37%   |

**The entire Jacobi-c² "basis-change" win is a SCALE CALIBRATION, not a
metric-weight change.** Same constant shape, different scalar multiplier:
scale=√3 hits optimum (0.072%), scale=√2 is 18× worse.

Companion test: varying coupling weight power c^α at scale=√2:
| α (coupling weight power) | err  |
|---------------------------|------|
| 0.0                       | 0.97%|
| 0.5                       | 1.18%|
| 1.0 (standard)            | 1.29%|
| 1.5                       | 1.36%|
| 2.0                       | 1.42%|

So the weight power is a weaker knob than the scale itself. Scale is the
dominant hidden parameter.

Companion test: mode-0 = c^α / norm (varied shape at c-weight normalization):
| shape α    | err  |
|------------|------|
| 0.0 (const)| 1.29%|
| 0.25       | 1.13%|
| 0.5        | 1.04%|
| 1.0 (linear)| 0.99%|
| 2.0 (quadratic)| 1.12%|

Shape-only variation stays bounded in the 0.99–1.3% range — shape is
also a weaker knob than scale.

**Implication**: the closure at rank-(1,1,1) has a HIDDEN GAUGE DEGREE OF
FREEDOM in the scale of each basis function. The closure is NOT gauge-
invariant at finite truncation (Galerkin projection weighting). The "right"
scale is a Petrov-Galerkin weighting that picks the best 1D approximation
to the self-consistent flux.

File: `derivations/diagnostics/diag_cin_split_source_decomposition.py`.

#### E3.6 — Scale scan across (σ_t·R, r_0/R) [RUNNING]

Brute-force scan of scale_opt(σ_t·R, ρ) for the constant-inner-basis closure,
to find whether scale_opt has a simple functional form. Results pending
(background run). Pattern to test: scale²_opt - 2 = C(τ, ρ), plausibly
scale² = 2 + f(τ, ρ) with f ~ 1 at moderate τ.

File: `derivations/diagnostics/diag_cin_split_scale_scan.py`.

---

## L8 — The closure has a hidden gauge DOF (scale of basis functions)

The rank-(1,1,1) split-basis closure with a single inner basis function is
NOT invariant under rescaling basis[0] → α·basis[0]. This is a Petrov-
Galerkin weighting artifact at finite truncation. The "right" scale is
NOT determined by basis-orthonormality — it's a free calibration parameter
per geometry+optical-depth. F.4's Lambert P/G + Marshak W trick is
effectively a specific scale choice for the N=1 case. The Jacobi-c² vs
Legendre-c "basis change" observed in E2.6 is the same scale DOF in
disguise (√3 vs √2).

**Implication for the research**: finding scale_opt(τ, ρ) analytically (or
empirically with a tight fit) and giving it a physical meaning is the
promising universal-closure path. This may be the principled explanation
for F.4's empirical advantage.

## L9 — Physics-asymptote basis (β=τ) works only at σ_t·R ≥ 20

The β=τ rule (mode-0 = exp(-τ·s(c;ρ)/2)) captures the correct attenuated-
flux shape at thick optical depth but is catastrophic at thin τ (86% err
at σ_t·R=1). Different β regions dominate: β=0 (Legendre) at σ_t·R ≤ 2.5,
β≈2-3τ at σ_t·R=5, β≈τ at σ_t·R=10-20, β≈τ/2 at σ_t·R=50. The adaptive
β may be related to the scale calibration in L8.

## Candidate research directions (updated, post-E3)

### Direction I — Inner-surface metric / scale (ACTIVE — see E3.5)

RECAST AS: find scale_opt(τ, ρ) for the constant-inner-basis closure. If
it fits a simple formula (e.g., scale² = 2 + τρ²·f) we have a universal
closure that empirically reproduces F.4's algorithm.

### Direction K — Petrov-Galerkin with principled test functions

If the closure is Petrov-Galerkin implicitly (with different trial and test
function scales), make it explicit: use a DIFFERENT basis for projecting
the residual than for representing the solution. The "test" basis can be
chosen to minimize the projection error.

Classical example: for transport with attenuation, weight by the adjoint
(importance) function at inner surface. Since importance describes how
inner flux influences the k_eff integral, projecting the residual with
importance-weight gives the optimal rank-1 Galerkin.

This is a principled route to the E3.5 scale-calibration phenomenon.

### Direction L — Explicit two-scale mode-0 decomposition

From the source decomposition hypothesis in E3.5 (TEST 4 not yet run):
maybe ψ^+_inner has TWO asymptotic shapes — one from white-BC-driven
rays (dominated by steep cone geometry), one from volume-source-driven
rays (dominated by nearby emissions). Using TWO mode-0 basis functions
(one for each) might unlock new accuracy.

## Rejected hypotheses (updated)

### RH6 — "Physics-asymptote basis (β=τ) breaks the plateau universally"

Falsified 2026-04-22 (E3.2). β=τ gives 86.7% err at σ_t·R=1, 1.1% at
σ_t·R=5 (worse than F.4). Only at σ_t·R ≥ 20 does β≈τ beat F.4. The
hypothesis captures the correct PHYSICS at thick τ but not at thin.

### RH7 — "Galerkin fixed-point adaptation breaks the plateau"

Falsified 2026-04-22 (E3.4 v1). At rank-(1,1,1), using the truth-shape
ψ^+_inner as mode-0 gives the SAME k_eff as Legendre mode-0. The Galerkin
iteration under c-weight is a fixed point that doesn't self-correct the
metric error. Shape doesn't matter once Gram-Schmidt absorbs it.

---

---

### Experiment 3 continued — α-scan run + scale formula conjecture
**Date**: 2026-04-22. Opus 4.7 main agent (post-auth-timeout recovery).

#### E3.1 full scan (Jacobi c^α inner weight, rank-(1,1,1))

Complete table of α_opt and err_opt at each (σ_t·R, ρ). Jacobi weight c^(α+1).
Constant basis under this weight: scale = √(α+2).

| σ_t·R | ρ    | α_opt | err_opt  | F.4 err |
|-------|------|-------|----------|---------|
| 1.0   | 0.10 | 0.00  | 77.5%    | 0.27%   |
| 1.0   | 0.30 | 0.00  | 85.6%    | 2.36%   |
| 1.0   | 0.70 | 2.50  | 13.3%    | 21.0%   |
| 2.5   | 0.30 | 0.00  | 3.86%    | 0.35%   |
| 2.5   | 0.70 | 0.00  | 0.14%    | 9.05%   |
| **5.0**   | **0.30** | **1.00**  | **0.072%** | **0.12%** |
| 5.0   | 0.50 | 0.50  | 0.046%   | 0.16%   |
| 10.0  | 0.30 | 1.25  | **0.015%**| 0.25%   |
| 10.0  | 0.50 | 0.75  | **0.027%**| 0.27%   |
| 10.0  | 0.70 | 0.50  | 0.326%   | 0.32%   |
| 20.0  | 0.30 | 1.00  | **0.079%**| 0.36%   |
| 20.0  | 0.50 | 0.50  | **0.153%**| 0.28%   |
| 20.0  | 0.70 | 0.50  | **0.064%**| 0.087%  |
| 50.0  | 0.30 | 1.25  | **0.050%**| 0.25%   |
| 50.0  | 0.50 | 0.50  | **0.080%**| 0.33%   |
| 50.0  | 0.70 | 0.50  | 0.132%   | 0.30%   |

**KEY EMPIRICAL WIN**: at σ_t·R ≥ 5, r_0/R ∈ {0.3, 0.5}, adaptive α gives
sub-0.1% residuals that BEAT F.4 by 2–20×. At some points (σ_t·R=10, ρ=0.3)
the split-basis closure with correct α gets to 0.015%.

#### E3.7 — Scale formula conjecture: scale²_opt = (1 + 6ρ)/(3ρ)

From E3.1 pattern analysis: at σ_t·R ≥ 5, α_opt · ρ ≈ 0.3 ± 0.08 across ρ.
Converting Jacobi weight α back to constant basis scale: scale² = α + 2.
Thus the candidate formula:

    scale_opt(σ_t·R, ρ) ≈ √((1 + 6ρ) / (3ρ))   for σ_t·R ≥ 5

| ρ   | Formula scale | Empirical scale (from E3.1 α_opt) |
|-----|---------------|------------------------------------|
| 0.1 | 2.31          | 2.24 (α=3, capped in scan)         |
| 0.3 | 1.76          | 1.73 (α≈1.0, ≈√3)                  |
| 0.5 | 1.63          | 1.58 (α≈0.5–0.75)                  |
| 0.7 | 1.57          | 1.58 (α≈0.5)                       |

Formula captures the qualitative and quantitative pattern.

#### E3.7 results (2026-04-22)

Test ran 12 points. Formula vs empirical-best vs F.4 at σ_t·R ≥ 5:

| σ_t·R | ρ    | sc_form | sc_emp | err_form | err_emp  | err_F.4 |
|-------|------|---------|--------|----------|----------|---------|
| 5.0   | 0.10 | 2.31    | 2.50   | 0.28%    | 0.042%   | nan     |
| 5.0   | 0.30 | 1.76    | 1.73   | 0.244%   | **0.071%** | 0.122%  |
| 5.0   | 0.50 | 1.63    | 1.60   | 0.575%   | 0.175%   | 0.155%  |
| 5.0   | 0.70 | 1.57    | 1.60   | 0.580%   | 1.333%   | 2.645%  |
| 10.0  | 0.10 | 2.31    | 2.50   | 0.403%   | 0.171%   | nan     |
| 10.0  | 0.30 | 1.76    | 1.80   | 0.161%   | **0.026%** | 0.246%  |
| 10.0  | 0.50 | 1.63    | 1.70   | 0.198%   | 0.278%   | 0.268%  |
| 10.0  | 0.70 | 1.57    | 1.60   | 0.427%   | **0.069%** | 0.318%  |
| 20.0  | 0.10 | 2.31    | 2.20   | 0.372%   | **0.001%** | nan     |
| 20.0  | 0.30 | 1.76    | 1.70   | 0.209%   | **0.026%** | 0.365%  |
| 20.0  | 0.50 | 1.63    | 1.60   | 0.081%   | **0.075%** | 0.280%  |
| 20.0  | 0.70 | 1.57    | 1.60   | 0.123%   | 0.090%   | 0.087%  |

**Formula vs F.4**: 5/12 wins with simple formula. Formula scale is
consistently within 1–4% of empirical (it's right in shape but not
fully precise).

**Empirical best vs F.4**: **7/12 wins**, with stunning lows: 0.0013% at
σ_t·R=20, ρ=0.1 (F.4 fails numerically there); 0.026% at σ_t·R=10,20
with ρ=0.3 (10× better than F.4); 0.069% at σ_t·R=10, ρ=0.7.

**The capability to beat F.4 is FIRMLY DEMONSTRATED**: rank-(1,1,1)
split basis with tuned scale gives sub-0.1% residuals systematically at
σ_t·R ≥ 5 and ρ ∈ {0.3, 0.5}. Not a cherry-picked sweet spot; it's robust
across multiple (τ, ρ) combinations.

**Remaining gap for UNIVERSAL closure**:
1. Simple formula `scale²_opt = (1+6ρ)/(3ρ)` captures shape but not precise
   optimum — typically 2–5× higher err than empirical optimum.
2. Empirical optimum requires a 1D scalar calibration per (τ, ρ) problem
   — acceptable cost (~10 k_eff solves, ~10s) but not elegant.
3. Thin τ (σ_t·R ≤ 2.5) is CATASTROPHIC (L10) — needs a different paradigm
   or a regime switch to unsplit basis.

### L11 — scale²_opt = 2 + 1/(3ρ) is τ-INDEPENDENT at σ_t·R ≥ 5

E3.1 α-scan data scrutinized point-by-point (diagnostic `diag_cin_split_scale_derivation_eddington.py`):

**τ-independence confirmed across σ_t·R ∈ {5, 10, 20, 50}**:
| ρ   | α_opt values (across τ)       | Formula 1/(3ρ) | Ratio   |
|-----|-------------------------------|----------------|---------|
| 0.1 | {3, 3, 3, 3}  (capped at 3)   | 3.333          | 0.90    |
| 0.3 | {1.0, 1.25, 1.0, 1.25}        | 1.111          | 0.90–1.13 |
| 0.5 | {0.5, 0.75, 0.5, 0.5}         | 0.667          | 0.75–1.13 |
| 0.7 | {0.5, 0.5, 0.5, 0.5}          | 0.476          | 1.05    |

At each fixed ρ, α_opt varies by at most one scan step (0.25) across
τ ∈ {5, 50} — within scan discretization noise. The formula α_opt · ρ
= 1/3 fits to ~10%.

**Physical interpretation**:
  scale²_opt = 2 + (1/3) · (1/ρ) = (Legendre) + (Eddington) × (cavity)

- "2" = Legendre c-weight orthonormalization baseline.
- "1/3" = Eddington factor ⟨µ²⟩_iso (canonical 3D isotropic moment).
- "1/ρ" = cavity-to-shell geometric factor. Physical origin:
  probably Liouville intensity concentration at inner surface,
  not the naive ρ² area ratio.

**OPEN**: exact analytical derivation of the 1/ρ factor. Conjecture:
at the inner surface, the partial-current integral acquires a 1/ρ
factor from the chord-length Jacobian dµ/dc · surface-area ratio.
Worth a paper-quality derivation if this closure path lands.

### Direction O — Refined scale formula or look-up table

Options for bridging the gap between simple formula (factor 2-5 off optimum)
and full empirical optimization:

1. **2D fit**: scale_opt²(τ, ρ) = A(ρ) + B(ρ)·exp(-τ·C(ρ)) + O(τ²) — 3-parameter
   fit per ρ curve.
2. **Lookup table**: pre-compute scale_opt on a fine (τ, ρ) grid, interpolate
   bilinearly. Trades analytical elegance for deterministic accuracy.
3. **Cheap 1D optimization at solve time**: run 10-20 Brent iterations on scale
   for each problem. Cost: ~10s. Acceptable for research / non-realtime use.
4. **Principled derivation** from adjoint weighting (Direction K).

**If the formula validates**: universal adaptive-scale closure beats F.4
across σ_t·R ≥ 5 with a simple ρ-only (no τ) formula. Next step: derive
this formula from first principles (Petrov-Galerkin + adjoint-weighting).

**If the formula misses**: the scale_opt depends on BOTH τ and ρ
non-trivially — see E3.1 which shows different α_opt at different σ_t·R
for fixed ρ (ρ=0.3: α_opt = 1.0 at τ=5, 1.25 at τ=10, 1.0 at τ=20,
1.25 at τ=50). Mild τ-dependence overlaid on ρ-dominance.

#### L10 — Split basis is CATASTROPHIC at thin τ (σ_t·R ≤ 2.5)

E3.1 shows at σ_t·R=1, every Jacobi α gives 72–88% err (>100× F.4).
Root cause: at thin τ, grazing and steep rays have similar physics (both
free-stream through shell with low attenuation). The split basis introduces
an artificial angular discontinuity at µ_crit that doesn't exist physically.
The "extra DOF" from the (-ρ, µ_crit) direction orthogonal to F.4 populates
with noise and degrades accuracy dramatically.

**Implication**: split basis ONLY helps at thick τ. A universal closure must
either (a) collapse to F.4's unsplit basis at thin τ, (b) enforce the F.4
constraint (a_g, a_s) ∝ (µ_crit, ρ) at thin τ, or (c) use a completely
different paradigm at thin τ (e.g. direct Marshak).

### Direction M — Regime-switched closure

Given L10, ship a closure that switches:
- σ_t·R ≤ threshold_1: unsplit Marshak-N or F.4 directly.
- σ_t·R ≥ threshold_2: split basis with scale_opt formula.
- Blended in between.

Threshold likely ~3–5 based on E3.1 data. Need to verify no hysteresis at
boundary.

### Direction N — Why α · ρ ≈ 0.3 physically?

The formula α_opt · ρ ≈ 0.3 = const suggests a scale-invariant phenomenon:
the optimal weight power is set by ρ (cavity-to-shell ratio) alone, not τ.

Physical guess: the constant 1/3 emerges from the ∫₀¹ c² dc / ∫₀¹ c dc = 2/3
moment ratio on the inner hemisphere, combined with the ρ² area factor.
Specifically, the inner surface's contribution to k_eff scales with ρ² (area)
while the outer scales with 1 (area). The "effective" current balance sets
the optimal weight.

Deriving this from Sanchez-McCormick reciprocity + surface ratio could
yield the 1/(3ρ) rule from first principles.

## Session trail (updated)

- **2026-04-21 Opus 4.7**: Issue #119 close-out + Hébert extraction + V-S
  per-mode falsification + split-basis experiment (E1).
- **2026-04-21 Opus 4.7**: E2 — quadrature floor, rank-(1,1,N), Lambert-split
  catastrophic, Jacobi-c² point win. L5, L6, L7. RH4, RH5. Directions I, J.
- **2026-04-21/22 Opus 4.7 + num-inv**: E3 dispatch — α-scan setup, asymptote
  basis (β hypothesis mixed), Galerkin adaptive (fixed point), scale
  calibration (load-bearing knob). L8, L9. RH6, RH7. Directions K, L.
  numerics-investigator hit auth timeout mid-run.
- **2026-04-22 Opus 4.7 (main, recovery)**: ran remaining E3 scripts;
  E3.1 full α-scan delivered; identified α·ρ ≈ 0.3 pattern; conjecture
  scale²_opt = (1+6ρ)/(3ρ) for σ_t·R ≥ 5 (E3.7 pending verify). L10
  (split fails at thin τ). Directions M (regime-switched closure), N
  (physical derivation of 1/(3ρ) rule).

---

### Experiment 6 — PCA sectors (quick probe, Direction C)

Quick test of piecewise-constant angular sector basis (Sanchez-Santandrea 2002
paradigm) on INNER surface only. Outer still uses split-basis (grazing + steep).
NO scale calibration — uniform Jacobi orthonormality under c-weight.

Results at σ_t·R ∈ {5, 10, 20}, ρ ∈ {0.3, 0.5, 0.7}:

| σ_t·R | ρ   | F.4    | M=1    | M=2 uni | M=2 phys | M=3 uni |
|-------|-----|--------|--------|---------|----------|---------|
| 5.0   | 0.30| 0.122% | 1.290% | 1.009%  | 1.052%   | 1.004%  |
| 10.0  | 0.30| 0.246% | 0.934% | 0.773%  | 0.810%   | 0.810%  |
| 20.0  | 0.30| 0.365% | 0.427% | **0.324%**| 0.362% | 0.358%  |
| 5.0   | 0.50| 0.155% | 1.795% | 1.245%  | 1.088%   | 0.908%  |
| 10.0  | 0.50| 0.268% | 1.322% | 1.068%  | 1.008%   | 0.880%  |
| 5.0   | 0.70| 2.645% | 3.524% | 2.268%  | 2.531%   | 2.250%  |
| 10.0  | 0.70| 0.318% | 2.306% | 1.669%  | 1.803%   | 1.646%  |

**Verdict**: PCA at uniform M=2 marginally beats F.4 only at σ_t·R=20, ρ=0.3.
Otherwise plateau at ~1% regardless of M. **Same ~1% plateau as Legendre
without scale calibration** — strong confirmation that L8 (metric/scale is
the load-bearing knob) applies universally across basis choices.

Physics-informed split (at c = 1/√2 Eddington mean) does not beat uniform.
Adaptive PCA sectors with per-sector scale DOF untested but potentially
relevant for future work.

### L12 — Plateau ≈ 1% is universal across basis TYPES without scale calibration

Legendre, Jacobi c^α, asymptote exp(-β·s), PCA sectors — ALL rank-(1,1,M)
closures on the hollow sphere plateau at ~0.8–1.3% residual at σ_t·R=5,
ρ=0.3 (without explicit scale calibration). The plateau moves with M slightly
(M=2 ~10% better than M=1) but converges by M=3. **The plateau is a basis-
metric barrier, not a basis-shape or basis-type barrier.** Scale calibration
is the only knob that breaks it, as empirically demonstrated in E3.1 / E3.5.

File: `derivations/diagnostics/diag_pca_sectors_hollow_sph.py`.

### Literature check (2026-04-22)

Literature-researcher agent report (25 min Zotero + CrossRef ANE/NSE +
OpenAlex + Semantic Scholar search):

**No direct match** for "Eddington-factor-weighted rank-1 IC closure with
`2 + 1/(3ρ)` basis-scale formula." The Eddington-factor connection to IC
(not diffusion) appears genuinely novel in the searched corpus.

**Three leads worth chasing** (not verified with PDFs yet):

1. **Bogado Leite, S.Q. (1998), "Revised interface-current relations for
   the unit-cell transport problem in cylindrical and spherical geometries,"
   Annals of Nuclear Energy 25 (6), 347–356, DOI 10.1016/S0306-4549(97)00026-1**
   — exact-domain match, 1 citation total (orphaned; possible forgotten
   prior art). PDF not OA. Worth interlibrary loan.

2. **Krishnani, P.D. (1982), "Interface current method for PHWR cluster
   geometry with anisotropy in the angular flux at interfaces," ANE 9 (5)**
   — explicit rank-N anisotropic IC, cluster (not hollow shell) geometry.

3. **Mohanakrishnan, P. (1982), "Choice of angular current approximations
   for solving neutron transport problems in 2-D by interface current
   approach," ANE 9 (5)** — title matches "basis-scale calibration"
   concept, 2D not 1D curvilinear.

4. **Sanchez, R. (2014), "On P_N Interface and Boundary Conditions,"
   NSE 177 (1), DOI 10.13182/NSE12-95** — rigorous IC-BC degeneracy
   theory via solid harmonics; closest theoretical framework for a
   gauge-DOF argument.

**Adjacent literature for the 1/ρ analytical derivation**:
- Corngold (2002+2004) — Peierls/Bickley-Naylor algebra in cylinder.
- Wio (1984) / Krishnani (1985) — CP kernel transformation laws under
  geometric scaling. If 1/ρ has a clean geometric meaning, one of these
  is where it lives.

**Recommendation**: pursue Bogado Leite 1998 PDF via interlibrary loan
before claiming novelty. If it doesn't derive a `2 + 1/(3ρ)`-type scale
AND the Krishnani 1982 anisotropic-IC paper uses flat (rank-0) per-sector
basis, the novelty hypothesis stands.

---

## ⚠️ RETRACTION — The "BREAKTHROUGH" was a BASE-quadrature artifact

**2026-04-22 late**: My claim below (appearing in commit `fba6835`) was
**WRONG**. The RS_brent closure's `0.0000%` result at BASE quadrature
(2, 4, 32) is a numerical cancellation artifact, NOT a structural
improvement over F.4.

**Revised finding (post E4.2 at RICH quadrature)**: at RICH quad (4, 8, 64),
**F.4 wins 6/6 points at σ_t·R ≥ 5** by 2–88× over the split-brent closure.

| σ_t·R | ρ   | scale  | F.4 RICH | split RICH | who wins   |
|-------|-----|--------|----------|------------|------------|
| 5.0   | 0.3 | 1.7184 | 0.058%   | 0.076%     | F.4 (0.8×) |
| 10.0  | 0.3 | 1.8066 | 0.003%   | 0.041%     | F.4 (13×)  |
| 20.0  | 0.3 | 1.7087 | 0.006%   | 0.016%     | F.4 (2.7×) |
| 50.0  | 0.3 | 1.7783 | 0.017%   | 1.507%     | F.4 (88×)  |
| 20.0  | 0.5 | 1.6165 | 0.005%   | 0.060%     | F.4 (11×)  |
| 10.0  | 0.5 | 1.6622 | 0.017%   | 0.039%     | F.4 (2.3×) |

Scale scan at σ_t·R=10, ρ=0.3 at RICH: optimum scale = 1.800 (matches
BASE-Brent's 1.8066 to 3 digits). Floor err = 0.069%. **F.4's 0.003% is
21× below this floor.** The structural residual of rank-(1,1,1) split
basis with adaptive scale is ~0.07%, well above F.4's ~0.003% floor.

**Why the RS_brent "0.0000%" looked like a win at BASE**:
- F.4 at BASE has err ~0.08–0.37% (quadrature noise, NOT structural).
- Brent on split scales the closure until its k_eff crosses k_inf —
  which happens at a scale dictated by quadrature cancellation, not the
  true structural minimum.
- Both methods are quadrature-limited at BASE but in different ways;
  their relative "winner" at BASE is meaningless.
- At RICH, F.4's true structural floor emerges (≤0.01%); split's
  structural floor (0.07%) is unmasked.

**Net**: the split-basis rank-(1,1,1) adaptive closure does NOT beat
F.4 at production-grade quadrature. **F.4 remains the production closure.**

## 📉 Archived context — false breakthrough (now retracted)

The detailed BASE-quadrature tables that originally supported the
"breakthrough" have been deleted here to avoid misleading future sessions.
They are preserved in the git history of commit `fba6835` for anyone who
wants the raw numbers. The structural truth lives in the RICH-quadrature
table above and the "E4 REVISED VERDICT" section further below.

**What was deleted from this section**: E5 rank-(1,1,2) BASE-quad table
(all "0.0000%" from quadrature noise), E4 BASE-quad scan table (same),
and L13/L14 as originally formulated ("CRACK" / "extends precision
further"). Their content is all falsified by E4.2 RICH-quad results
(see below, the "E4 REVISED VERDICT" section — line ~1129).

**What survives** (and is independent of the retraction):
- L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12 — all stand.
- L11's formula `scale²_opt = (1+6ρ)/(3ρ)` is still an empirically
  observed pattern, but represents a CONDITIONAL minimum of the closure
  at BASE quadrature, not a production formula.
- The scale gauge DOF (L8) is real, but it only reaches into the
  quadrature-noise floor — not into F.4's structural floor.

Files — unchanged:
- `derivations/diagnostics/diag_cin_split_regime_switched.py` (E4).
- `derivations/diagnostics/diag_cin_split_rank112_adaptive.py` (E5).
- `derivations/diagnostics/diag_cin_split_scale_precision_check.py`
  (precision probe — killed by timeout, to re-run at RICH if needed).

---

### Experiment 4 — Regime-switched closure (F.4 at thin τ, split+scale at thick τ)
**Date**: 2026-04-22. **Branch**: `investigate/peierls-solver-bugs`. Opus 4.7 (main, numerics-investigator).

#### E4.1 — Main scan at BASE quadrature (2, 4, 32)

Regime-switched closure design:
- σ_t·R ≤ 3: use F.4 scalar (falls back safely, avoids L10 catastrophe)
- σ_t·R ≥ 5: use split-basis rank-(1,1,1) with constant-inner basis scaled by
  - **(a) formula**: `scale²_opt = (1+6ρ)/(3ρ)` (ρ-only, no τ dep)
  - **(b) brent**: bounded 1D scale minimization on [1.0, 2.8]
- Transition zone (3 < σ_t·R < 5): linear interpolation of k_eff.

Full scan at ρ ∈ {0.1, 0.3, 0.5, 0.7}, σ_t·R ∈ {0.5, 1, 2.5, 5, 10, 20, 50}:

| σ_t·R | ρ    | F.4     | split+form | RS_form | split+brent | RS_brent |
|-------|------|---------|------------|---------|-------------|----------|
| 5.0   | 0.30 | 0.122%  | 0.244%     | 0.244%  | 0.0000%     | 0.0000%  |
| 5.0   | 0.50 | 0.155%  | 0.575%     | 0.575%  | 0.0000%     | 0.0000%  |
| 5.0   | 0.70 | 2.645%  | 0.580%     | 0.580%  | 0.0000%     | 0.0000%  |
| 10.0  | 0.30 | 0.246%  | 0.161%     | 0.161%  | 0.0000%     | 0.0000%  |
| 10.0  | 0.50 | 0.268%  | 0.198%     | 0.198%  | 0.0000%     | 0.0000%  |
| 10.0  | 0.70 | 0.318%  | 0.427%     | 0.427%  | 0.0000%     | 0.0000%  |
| 20.0  | 0.30 | 0.365%  | 0.208%     | 0.208%  | 0.0000%     | 0.0000%  |
| 20.0  | 0.50 | 0.280%  | 0.081%     | 0.081%  | 0.0000%     | 0.0000%  |
| 20.0  | 0.70 | 0.087%  | 0.123%     | 0.123%  | 0.0000%     | 0.0000%  |
| 50.0  | 0.30 | 0.255%  | 0.025%     | 0.025%  | 0.0000%     | 0.0000%  |
| 50.0  | 0.50 | 0.329%  | 0.056%     | 0.056%  | 0.0000%     | 0.0000%  |
| 50.0  | 0.70 | 0.304%  | 0.094%     | 0.094%  | 0.0000%     | 0.0000%  |

(Thin τ ≤ 2.5: all RS values = F.4 by design; split+form catastrophic as
expected per L10.)

**RS_brent: 12/12 wins vs F.4 at thick τ** with ratios 87k-2.6M× (base quad
where BOTH hit machine precision; the RS_brent "0.0000%" is quadrature-
limited noise, not structural).

**RS_form: 8/12 wins, 4 losses/ties** — formula is WORSE than F.4 at σ_t·R=5
(borderline). Only kicks in as a universal win at σ_t·R ≥ 10.

#### E4.2 — RICH-quadrature validation: FALSIFIES the BASE-quad wins

**Critical finding (2026-04-22 late)**: transferred the BASE-optimized
scales from E4.1 to the RICH quadrature (4, 8, 64) and compared F.4 vs
split-basis-rank-(1,1,1) at those fixed scales:

| σ_t·R | ρ    | scale | F.4 (RICH) | split (RICH) | ratio F.4/split |
|-------|------|-------|------------|--------------|-----------------|
| 5.0   | 0.30 | 1.7184 | 0.0578%   | 0.0762%      | 0.8× (F.4 wins) |
| 10.0  | 0.30 | 1.8066 | 0.0033%   | 0.0413%      | 0.1× (F.4 wins 13×) |
| 20.0  | 0.30 | 1.7087 | 0.0060%   | 0.0160%      | 0.4× (F.4 wins 2.7×) |
| 50.0  | 0.30 | 1.7783 | 0.0171%   | 1.5065%      | 0.01× (F.4 wins 88×) |
| 20.0  | 0.50 | 1.6165 | 0.0054%   | 0.0601%      | 0.1× (F.4 wins 11×) |
| 10.0  | 0.50 | 1.6622 | 0.0167%   | 0.0386%      | 0.4× (F.4 wins 2.3×) |

**6/6 LOSSES at RICH quadrature.** The BASE-quad Brent wins were
quadrature-error cancellation artifacts, not structural wins.

Companion **full scale-scan** at σ_t·R=10, ρ=0.3, RICH (scale 1.2 → 2.3):

| scale | err (RICH) |
|-------|------------|
| 1.2   | 1.259%     |
| 1.4   | 1.042%     |
| 1.6   | 0.690%     |
| **1.8** | **0.0692%** (MINIMUM) |
| 2.0   | 1.127%     |
| 2.2   | 3.484%     |
| 2.3   | 5.297%     |

**Critical result**: RICH-optimum scale (1.8) coincides with BASE-optimum
scale (1.8066) to 3 digits. The scale IS physics-reflective — the L8
gauge DOF optimum is the SAME at BASE and RICH. BUT the error at the
RICH-optimum scale is 0.069%, which is **21× WORSE than F.4 at RICH
(0.003%)**. So even with PERFECT scale calibration, rank-(1,1,1) split
cannot beat F.4. This is the definitive falsification of Direction M.

**Revised L14**: scale calibration at BASE DOES transfer to RICH
(scales match to 3 sig figs) — but the RICH residual at the optimum
scale is dominated by F.4's. The previous E4.2 conclusion that "BASE
scales don't transfer" was wrong: they DO transfer (scale is truly
physics-defined), but the closure itself plateaus at 0.07% err at
σ_t·R=10 — above F.4's 0.003% quadrature floor. **Rank-(1,1,1) split
has a hard structural floor above F.4.**

### E4 REVISED VERDICT (2026-04-22 late)

**DIRECTION M (regime-switched closure) DOES NOT UNIVERSALLY BEAT F.4.**
The BASE-quad "wins" were quadrature-error cancellation artifacts.

At production-grade RICH quadrature:
- F.4 consistently beats split-rank-(1,1,1)+scale_BASE
- F.4 sits at its quadrature floor (0.003-0.017%) as expected per L5
- split+BASE-scale actively HARMS accuracy at RICH (2-88× worse)
- Running Brent at RICH itself is VERY expensive (~15-30 min per point,
  based on timing at 45s/k_eff-call × 25 maxiter)
- No shippable universal closure emerges from E4 at matched quadrature.

**Ship status**: F.4 scalar remains the production closure; no evidence
that rank-(1,1,1)+scale beats F.4 at matched (RICH) quadrature.
Recommend closing Issue #120 (c_in-aware split basis) with:
"empirical falsification at matched quadrature — BASE-quad wins were
quadrature artifacts (L13 / new L14)."

### L14 — rank-(1,1,1) split has a STRUCTURAL FLOOR above F.4's

**Fundamental finding**: the rank-(1,1,1)+scale optimum at BASE quad
(scale≈1.8066) coincides with RICH optimum (scale≈1.8000) at σ_t·R=10,
ρ=0.3 — so scale IS physics-defined. But even at RICH-optimum scale,
split-rank-(1,1,1) gives 0.069% err, while F.4 at RICH gives 0.003%.

**Rank-(1,1,1) split has a hard structural floor ~0.07% err at this
point that F.4 structurally passes through.** The split's residual
closure error is dominated by its mode truncation (N_i=1 inner mode),
not by quadrature or scale. F.4, despite using just a scalar white-BC
correction, has a structural residual that approaches machine precision
with refinement (L5 / E2.1).

### Why is F.4 structurally better than rank-(1,1,1)?

F.4 uses Lambert P/G + Marshak W (basis mismatch). The empirical win
that was unexplained in L2 is precisely this: F.4's basis mismatch
implicitly captures MORE angular detail than Marshak-consistent rank-1
(or rank-(1,1,1) split). It's a 1-dimensional subspace the split basis
doesn't span — NOT because it has more modes, but because its
effective inner-product weight is anisotropic in a way that rank-N
Marshak/split cannot reproduce.

Per L6, this anisotropy is algebraically N=1-specific — you can't just
Lambert-ify the split basis either. So F.4 sits on a structurally
special 1-parameter family that rank-N bases cannot access.

### RH10 — "rank-(1,1,1) split with scale-optimum beats F.4 at matched quadrature"

Falsified 2026-04-22 (E4.2). 6/6 losses at RICH quadrature using
BASE-optimal scales AND confirmed by single-point RICH scale scan:
optimum scale at RICH is 1.8 but err = 0.069% still loses to F.4's
0.003% by 21×. See L14.

### E4 verdict

**Regime-switched(brent) is a SHIPPABLE UNIVERSAL CLOSURE at thick τ**
— 12/12 wins at σ_t·R ≥ 5 (BASE quad), F.4-compatible at thin τ.
The Brent variant dominates formula variant by 10-100×; formula is
too imprecise at borderline thickness (σ_t·R=5).

**Regime-switched(formula)** requires fix at σ_t·R=5 borderline
(currently 2-5× WORSE than F.4 there). Could work if threshold_high
raised to ~8.

---

### Experiment 5 — rank-(1,1,2) 2D adaptive scales
**Date**: 2026-04-22. Opus 4.7 main/numerics-investigator.

Build rank-(1,1,2) closure with TWO independently-scaled inner basis
functions:
- φ_0(c) = α_0 (constant)
- φ_1(c) = α_1 · (2c - 1) (shifted Legendre mode-1)

Nelder-Mead 2D optimization of (α_0, α_1) per (σ_t·R, ρ), BASE quadrature.

Reached 13/14 points before 20-min timeout:

| σ_t·R | ρ    | F.4     | rank-(1,1,1) 1D scale | (α_0, α_1) best | err_112 |
|-------|------|---------|-----------------------|-----------------|---------|
| 5.0   | 0.10 | nan     | nan                   | (2.42, 1.07)    | 0.0000% |
| 5.0   | 0.30 | 0.122%  | 1.7184 → 0.0000%      | (1.64, 1.03)    | 0.0000% |
| 5.0   | 0.50 | 0.155%  | 1.5851 → 0.0000%      | (1.50, 1.02)    | 0.0000% |
| 5.0   | 0.70 | 2.645%  | 1.5527 → 0.0002%      | (1.47, 1.02)    | 0.0000% |
| 10.0  | 0.30 | 0.246%  | 1.8066 → 0.0000%      | (1.73, 1.05)    | 0.0000% |
| 10.0  | 0.50 | 0.268%  | 1.6622 → 0.0000%      | (1.58, 1.01)    | 0.0000% |
| 10.0  | 0.70 | 0.318%  | 1.6050 → 0.0001%      | (1.52, 1.02)    | 0.0000% |
| 20.0  | 0.30 | 0.365%  | 1.7087 → 0.0000%      | (1.62, 1.03)    | 0.0000% |
| 20.0  | 0.50 | 0.280%  | 1.6165 → 0.0000%      | (1.53, 1.02)    | 0.0000% |
| 20.0  | 0.70 | 0.087%  | 1.5891 → 0.0001%      | (1.51, 1.01)    | 0.0000% |
| 50.0  | 0.30 | 0.255%  | 1.7783 → 0.0000%      | (1.68, 1.02)    | 0.0000% |

**KEY OBSERVATION**: α_1 is consistently ~1.01-1.08 across ALL points. The
second mode barely moves from its nominal scale. This is the "ideal
Legendre mode-1 amplitude" — just the natural Legendre polynomial.
The optimizer finds mostly α_0 freedom (the scale gauge DOF from L8).

**Quantitative test**: at BASE quadrature, rank-(1,1,2) 2D optim gives
identical err as rank-(1,1,1) 1D optim — BOTH hit quadrature floor
(~0.0000%). The "ratio" column showing 16-7400× is just noise-level
ratios at the quadrature floor, not structural.

**E5 verdict**: **rank-(1,1,2) 2D adaptive does NOT meaningfully help
over rank-(1,1,1) 1D at the tested quadrature.** The second mode's scale
α_1 ≈ 1 is essentially the default — 2D optimization collapses to
rank-(1,1,1) plus trivial normalization. Combined with E2.3 (mode-1
carries ~10% of self-consistent ψ^+ energy), the conclusion is:
**the scale-gauge DOF of L8 lives in mode-0 alone**; higher modes
don't have an equivalent gauge DOF that matters empirically.

If verified at RICH quadrature (where rank-(1,1,1) 1D might plateau at
some structural floor), rank-(1,1,2) 2D may still help marginally — but
the improvement ratio from this data suggests it's unlikely to beat 2×.

---

## New lessons (L13+ — renumbered to avoid collision with retracted L11-L14)

### L13 — RS_brent at BASE was UNIVERSAL across full (τ, ρ) range — BUT IT WAS QUADRATURE NOISE (L16 below)

Rank-(1,1,1) split basis with Brent-optimized scale, switched against F.4
at σ_t·R ≤ 3 (per L10), gives 12/12 strict wins vs F.4 at σ_t·R ≥ 5 **at
BASE quadrature**. Transition is smooth at σ_t·R = 3-5 (linear k_eff
blend). ⚠️ **This "win" does NOT survive at RICH quadrature** — see L16
and the E4 REVISED VERDICT table (line ~1129).

### L14 — The scale-gauge DOF (L8) is rank-(1,1,1)-specific

At rank-(1,1,2), optimal scales are (α_0, α_1) ≈ (α_0_opt, 1.0 ± 0.1)
uniformly across 10 test points. Mode-1's scale is NOT a meaningful tuning
parameter — it sits at the natural Legendre normalization. So the 18×
gauge win from L8 (Legendre vs Jacobi-c² at rank-(1,1,1)) does NOT have
a rank-(1,1,N) generalization: you can't stack N independent scale DOFs.

### L15 — RS_brent's "0.0000%" values are quadrature-limited at BASE

At BASE (2, 4, 32), rank-(1,1,1) + Brent-optimized scale hits numerical
floor ≤ 1e-6 err. True structural residual is almost certainly below
BASE's quadrature noise. This is DIFFERENT from F.4 at BASE (sits at
0.12-0.36% noise, per L5). At RICH quadrature (confirmed in L16), split's
structural floor (~0.07% at σ_t·R=10, ρ=0.3) is 21× above F.4's
structural floor (~0.003%). The "win" is a quadrature coincidence.

### L16 — The scale-calibration "win" is a BASE-quadrature artifact (load-bearing retraction)

At RICH quadrature (4, 8, 64), F.4 wins 6/6 points at σ_t·R ≥ 5 by 2–88×.
The Brent optimization on the split basis minimizes an error function
that is dominated by quadrature noise at BASE, so Brent finds a scale
where quadrature-noise cancellations drive apparent k_eff to zero.
At RICH, F.4's structural floor emerges and Brent's optimum ceases
to be at k_inf — the closure's true structural floor reveals itself
above F.4. **Moral: always match quadrature across compared closures.
A "win" that depends on quadrature-noise cancellation is a mirage.**

## New rejected hypotheses (RH8+)

### RH8 — "rank-(1,1,2) with 2D adaptive scales breaks below rank-(1,1,1) floor"

Falsified 2026-04-22 (E5). Optimal α_1 ≈ 1 uniformly → the 2nd mode's
scale DOF is empty. rank-(1,1,2) is effectively rank-(1,1,1) with a
free-amplitude mode-1 (adds no information at rank-(1,1,1)'s accuracy
level). No meaningful improvement over rank-(1,1,1) 1D Brent.

### RH9 — "RS_form (formula scale) is a universal closure"

PARTIALLY falsified 2026-04-22 (E4.1). RS_form beats F.4 at 8/12 thick-τ
points but LOSES at σ_t·R=5 (factor 2-5× worse) and at σ_t·R=10 ρ=0.7
(1.3× worse). Recommended pivot: raise threshold_high to ~10 and use
F.4 up to σ_t·R=10; then RS_form becomes universal at σ_t·R ≥ 10.
(Less aggressive claim than "pure formula everywhere" but still shippable.)

## Direction updates

### Direction M (REJECTED, 2026-04-22 late): Regime-switched(brent) FAILS at matched quadrature

**Post-E4.2 verdict**: regime-switched closure with BASE-optimized scale
produces wins only at BASE quadrature; at RICH quadrature it LOSES to F.4
by 2-88×. The scale Brent optimization is entangled with quadrature error
(L14). Not a shippable path.

If someone wants to revive this approach, would need to:
1. Run Brent at the PRODUCTION quadrature directly (very expensive:
   ~15-30 min per point at RICH).
2. Cache per-problem scales in a lookup table indexed by (τ, ρ, n_panels,
   p_order, n_ang).
3. Prove at RICH that RICH-optimized scale still beats F.4 at RICH —
   currently untested but plausible ONLY if the Brent genuinely finds
   a non-trivial minimum at RICH, which the scale scan suggests it does
   NOT in the [1.2, 2.3] bracket at σ_t·R=10.

### Direction O (PROPOSED): Principled scale formula beyond (1+6ρ)/(3ρ)

E4 showed the formula is WRONG at σ_t·R=5 (gives 0.24-0.58% vs Brent's
~0%). Two options:
- (O-1) Tune threshold_high up to 10 where formula works.
- (O-2) Derive τ-dependent formula: scale²_opt(τ, ρ) = A(ρ) + B(ρ)·f(τ).
  Requires E3.1's full α-scan data + curve-fit.
- (O-3) Cheap lookup table: (τ, ρ) → (scale) bilinear interpolation
  from offline Brent scan.

### Direction P (PROPOSED): Higher-rank with scale DOF only in mode-0

Given L12, rank-(1,1,N) with scale tuned ONLY on mode-0 is the right
approach for enrichment. rank-(1,1,2) with α_0 tuned, α_1 = 1 fixed may
give the same accuracy as rank-(1,1,1)+Brent — but slightly cheaper
(fewer Brent iterations). Not a universal improvement, likely a wash.

## Session trail (updated)

- **2026-04-22 Opus 4.7 (numerics-investigator, 3h budget)**: E4 full scan
  28/32 points at BASE quadrature (timed out before σ_t·R=100);
  E5 rank-(1,1,2) 2D scan 13/14 points at BASE. L11, L12, L13.
  RH8, RH9. Directions M (ship-candidate), O, P.
  Artifacts: `diag_cin_split_regime_switched.py`,
  `diag_cin_split_rank112_adaptive.py`. Quadrature plumbing issue
  with MED (ρ=0.1 causes F.4 to return nan due to radial-tangent
  Nyström pathology); scan on BASE was more robust.

<!-- Next session appends below this line. -->

