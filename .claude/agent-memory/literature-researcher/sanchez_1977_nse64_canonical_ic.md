---
name: Sanchez 1977 NSE 64 — canonical interface-current rank-N derivation (PDF read 2026-04-25)
description: Full extraction from Sanchez 1977 NSE 64(2):384-404. Rank-1/rank-3 IC formalism for 2D Cartesian (CARCINOMA) + 1D cylindrical sub-case (LEUKEMIA). Key finding: NO multiple-collision K_inf eigenvalue closure — paper is fixed-source only. Sanchez Galerkin solve = (I - APss)^-1 expansion, NOT a geometric-series-of-(1-Pss)^-1.
type: reference
---

# Sanchez 1977 NSE 64 — substantive content (read 2026-04-25)

## Headline citation

**R. Sanchez (October 1977)**, "Approximate Solutions of the
Two-Dimensional Integral Transport Equation by Collision Probability
Methods", *Nuclear Science and Engineering* **64**(2), 384-404.
DOI: [10.13182/NSE64-384](https://doi.org/10.13182/NSE64-384).
Submitted Jan 17 1977 from CEA-Saclay (SERMA). 21 pp.

## Scope (the paper is FIXED-SOURCE, NOT eigenvalue)

Both benchmark problems are "one-group fixed-source" with `S = 1
n/(cm³·s)` in moderator (Kavenoky 1974 benchmarks). Reference is
the CLUP-77 direct CP code. **There is no k_eff calculation,
no multiplication eigenvalue iteration, no `(1-P_ss)^-1` geometric
series anywhere in the paper.** Galerkin matrix solve is direct
Gaussian elimination on `X = (I - A·Pss)^-1`. This is the
multi-cell coupling matrix, NOT an eigenvalue closure.

## Section structure

- **BASIC EQUATIONS** (pp. 386-388): integral transport in cell
  decomposition; Galerkin projections; reciprocity; conservation.
- **DESCRIPTION OF THE TWO COMPUTER CODES** (pp. 388-394):
  - Angular representation Eqs. (12)-(14): rank-1 cosine + rank-3
    DP1 with two extra basis functions Eq. (13b,c).
  - LEUKEMIA (pp. 389-391): cylindrical cell + 3 P_ss models
    (Roth×4, P_ss-homogeneous, P_ss-heterogeneous) + 3
    cylindrization rules (Wigner / Wig-Ask / Askew).
  - CARCINOMA (pp. 391-392): true 2D, sectorial mesh.
  - Symmetries Eqs. (19)-(22), system reduction Eqs. (23)-(26).
- **NUMERICAL TESTS** (pp. 394-401): Tables III-IX, Figs. 3-4.
- **APPENDIX** (pp. 401-403): Bickley-Naylor reduction Eqs. (A.1)-
  (A.3) — the 2D and 1D cylindrical kernel forms.

## 1. The rank-N angular interface basis (Eqs. 12-13)

For each cell side α, the surface angular flux ψ_±(r_s, Ω) is
expanded as `ψ_±(r_s,Ω) = Σ_p J^p_{±,α} · ψ^p_α(Ω) · e_α(r_s)`
where `e_α(r_s)` is the characteristic function of side α (so the
expansion is **piecewise-uniform on each face**, not on the whole
cell — Roth's model is recovered by α = single face).

The three orthonormal basis functions (Eqs. 13a-c, p. 389):

```
ψ^0_α(Ω) = 1                                        (rank-0, cosine current)
ψ^1_α(Ω) = sqrt(2) · sin θ · sin φ                  (rank-1, sine moment)
ψ^2_α(Ω) = sqrt(2) · (3 sin θ cos φ - 8/(3π))       (rank-2, transverse cosine moment, recentered)
```

Here θ is the polar angle from the z-axis (translational symmetry
direction) and φ is the azimuth between the face normal n_± and
the projection of Ω onto the transverse plane. The "rank-3"
nomenclature in Sanchez means **three terms total** (rank-0 +
rank-1 + rank-2 of the standard P_N basis), NOT P_3. In ORPHEUS
notation this corresponds to a "double P_1" surface expansion on
each face.

Critically: **the Sanchez "rank-3" surface basis is just three
modes per face**, not a continuous quadrature. Per his Table IX,
LEUKEMIA(3,3) for Benchmark II has only 77 currents and 43 fluxes
total — small algebraic system. This is structurally identical to
the rank-M Schur path you (Rodrigo) already proved equivalent to
the augmented Nyström direction.

## 2. The Galerkin algebraic system (Eqs. 23-25)

System (23a-c) in matrix form:

```
φ   = P_·s · J_-  + P · F                  (23a)  in-cell flux
J_+ = P_ss · J_-  + P_s· · F               (23b)  outgoing currents
J_- = J_0 + A · J_+                         (23c)  geometric coupling
```

Substituting (23b) into (23c) and eliminating J_+ gives

```
J_- = J_0_bar + Pss_bar · P_s· · F
```

with `Pss_bar = X·A` and `X = (I - A·Pss)^-1`. This X is the
sole inverse in the paper. Final flux equation (24):

```
φ = φ_0 + P_bar · F,    where P_bar = P + P_·s · Pss_bar · P_s·
```

**X = (I - A·Pss)^-1 is NOT a geometric series in P_ss alone.**
A is the geometric coupling matrix (which surface of which cell
borders which surface of which other cell). Pss is per-cell
self-coupling. The product A·Pss is the inter-cell coupling
operator. The inverse is direct Gaussian elimination, not
iterative summation.

The iterative reformulation (Eqs. 26a-b) gives a different inverse
`Y = (I - P·s · L_s · G_0_..)^-1` for in-cell flux update, used in
a Jacobi-style cell-by-cell sweep. Sanchez says "application of
this iterative scheme will be the subject of a future paper" — so
no convergence claim or proof in this paper.

## 3. The 1D cylindrical sub-case — LEUKEMIA in detail

LEUKEMIA uses a **cylindrical model only for the in-cell collision
and escape probabilities**. It is NOT a 1D-radial transport
solver. The cell is partitioned by concentric circles; the
4-sided rectangular cell's boundary is mapped onto the circular
boundary by relative areas (sides 1-4 in counter-clockwise order,
each occupying a quarter of the cylinder circumference for square
cells).

Three cylindrization rules (Table I, p. 390):

| Model     | Σ*           | V*          | S*           |
|-----------|--------------|-------------|--------------|
| Wigner    | Σ            | V           | r·S (r<1)    |
| Askew     | r²·Σ         | V/r²        | S            |
| Wig-Ask   | uses Wigner CP/escape, real S in conservation |

with `r = √(πab)/(a+b) < 1`. **The Wig-Ask model is recommended;
Wigner can give large errors (up to +14% on water-gap cell of
Benchmark II — Table VI).**

The escape factorization Eq. (16) is rank-1 only:

```
P^kp_{i,Sj} = p_j · w^kp_j · V_i · Σ_i · P^k_{i,S}
```

with `w^00 = w^02 = 1` and other `w` values given by symmetry.
This is a **per-side weighting** of the rank-0 escape, meaning
the cylindrical sub-case forces the rank-3 angular structure to
be **derived from rank-1 escape moments** rather than computed
independently. This is the algebraic equivalence: rank-3 IC in
LEUKEMIA = rank-0 escape × per-side directional w-weights.

## 4. The three P_ss models (transmission probabilities)

Cylindrization gives Roth-style return effect (uncollided current
returning through the entering side). To kill the return effect,
LEUKEMIA replaces the cylindrical-cell P_ss with one of:

1. **Roth × 4** (Eq. 17): `P_ss,ij = (1 - δ_ij) · (S_j/(S-S_i)) · (P_ss)*`
   where `(P_ss)*` is total transmission from cylindrization.
   Per-side proportional sharing. Isotropic only (rank-0 surface
   flux). Eliminates Roth's return effect but keeps the pass-through
   effect.

2. **P_ss-homogeneous**: compute transmission on equivalent
   homogeneous rectangular cell preserving total `(P_ss)*`. Allows
   non-isotropic transmission (compatible with rank-3). Normalize
   `Pij_a` instead of `P_ss_a` — this is the closure detail that
   matters: re-normalizing P_ss gives worse results.

3. **P_ss-heterogeneous**: real heterogeneous cell with **dynamical
   cylindrization**: surface S in conservation/reciprocity is taken
   as a per-group parameter that makes Eq. (10b) automatically
   satisfied. Then Pij_a is normalized.

**Empirical finding from Tables III-VIII**: Wig-Ask + P_ss-
homogeneous + (1,3) (rank-1 flux, rank-3 surface) is the single
best combination on heterogeneous LWR cells. Errors generally
< 1% on Benchmark I, < 3% on Benchmark II.

## 5. Convergence and benchmark behavior

Comparing (1,1)-Roth (vanilla Roth model) vs (1,1)-Roth×4 on
Benchmark II:
- Wigner cylindrization: Cell I region 1 error 12.67% → 7.25%
- Wig-Ask cylindrization: 6.77% → 0.66%
- Askew cylindrization: 0.08% → -4.76% (Askew + Roth happens to
  cancel return effect on this geometry)

(1,3)-P_ss-heterogeneous on the same Cell I region 1:
- Wigner: 13.40%
- Wig-Ask: -1.16%
- Askew: -2.62%

**Bottom line**: rank-3 surface improves over rank-0/Roth by
factors of 5-10× on flux-gradient regions, with Wig-Ask
cylindrization. **No claim is made of convergence to k_inf because
the problem is fixed-source.** No paper sweep of N=1,3,5,7,...
exists; the choice was binary (1 or 3).

## 6. The 1D radial reduction (Appendix A)

Eqs. (A.2a-b) give the LEUKEMIA 1D cylindrical formulas:

```
P^00_ij = (2π / (V_i Σ_i Σ_j)) · ∫_0^p_< T_ij(R) dR + δ_ij · L_i
```

where:
- T_ii(R) = Ki_3(2t_i) + Ki_3(2t_{i-1}) + 2[Ki_3(t_i + t_{i-1}) -
  Ki_3(t_i - t_{i-1}) - Ki_3(0)]   (Eq. A.2a, self-collision)
- T_ij(R) = H'_i H_j [Ki_3(t_i + t_j) - Ki_3(|t_j - t_i|)]   (Eq. A.2a, off-diag)

with `t_i^2 = ρ_i² - R²`, `H_i f(t) = f(t_i) - f(t_{i-1})`. R is
the chord impact parameter ∈ [0, p_<] = min(p_i, p_j).

For the rank-3 LEUKEMIA flux (sin α and cos α moments),
Eq. (A.2b) gives:

```
P_ij^kl = (2/V_i) · ∫_0^p_< dR ∫_{t_{i-1}}^{t_i} dt ∫_{t_{j-1}}^{t_j} dt'
          [(R² - t·t')/(p·p')] · Ki_1(t + t')
```

**Note this uses Ki_1, NOT Ki_3.** This is the moment-weighted
form that picks up the `R²` Jacobian from the in-plane azimuthal
integration. The flat-source rank-0 P^00 form integrates to Ki_3
because both the source point and the field point are integrated
over their full annular cross-sections; the rank-1 cosine moment
does not give the same algebraic reduction.

Per Eq. (A.2b), Sanchez computes `\bar P_ij = P_ii^11 - P_ij^11`
(in which the Ki_1 coefficient vanishes at t = t') and then
derives `P_ij^11` from `\bar P_ij` plus `P_ij^00`. This handles the
log-divergence singularity of `dKi_1/dt` at t = t'.

## 7. Cross-check vs `phase4_cylinder_peierls.md`

**The phase4 memo's Ki_1 vs Ki_3 distinction is correct and
explicitly confirmed by Sanchez Eq. (A.2a vs A.2b)**:
- Region-average (rank-0) → Ki_3 kernel.
- Higher moment (rank-3 surface or in-cell sin/cos flux) → Ki_1
  kernel with explicit Jacobian.

**However**, the phase4 memo's "1/π prefactor" claim is for the
**point-wise Peierls equation for the scalar flux**, while
Sanchez's Eq. (A.2a) `2π / (V_i Σ_i Σ_j)` prefactor is for the
**region-to-region collision probability** (a different
normalization). Both are correct in their own context. The phase4
memo flagged this caveat itself; reading the paper confirms
Sanchez's CP integrals use Ki_3 with `2π / V_i Σ_i Σ_j`
prefactor.

## 8. Implementation feasibility for rank-N IC

**Per-face angular discretization**: yes, Sanchez Eq. (12) is
*explicitly* per-face: `ψ_±(r_s, Ω) = Σ_p J^p_{±,α} ψ^p_α(Ω) e_α(r_s)`.
Each face α has its own basis (the basis happens to have the same
*form* on every face, but the coefficients are independent).

**Algebraic equivalence to rank-M Schur**: the rank-3 LEUKEMIA
basis has dimension 3 per face × 4 faces = 12 surface unknowns
per cell (after symmetry, 6-15 in Table IX). This is structurally
**identical** to a per-face rank-M Schur block of the same
dimension. The "interface current method" is just a different
factorization of the same Galerkin system.

The CARCINOMA path (truly 2D in-cell with rank-3 surface) gives
better answers than LEUKEMIA but is not algebraically distinct in
the surface coupling — only in the in-cell flux representation.

## Answer to the user's algebraic-equivalence question

**Sanchez 1977 NSE 64 reduces algebraically to what is already in
ORPHEUS.** The rank-3 surface basis (3 modes per face) is
structurally identical to a rank-M Schur per-face block. The
in-cell CP machinery is the standard Ki_3 region-average form
(Eq. A.2a) that ORPHEUS already uses in `cp_cylinder.py`. The
"multi-collision K_inf" idea is not present in this paper at all
— Sanchez does fixed-source only with direct Gaussian elimination
on the (I - A·Pss)^-1 coupling matrix.

**The Mark uniformity overshoot on heterogeneous Class B cells
will not be fixed by implementing Sanchez 1977 rank-3 surfaces.**
The paper's own Tables VI-VII show that even with rank-3 surface
+ best cylindrization, fixed-source flux errors of -3% to +14%
persist on heterogeneous LWR cells (Benchmark II). This is the
structural ceiling of the "rank-N IC + cylindrical CP" family,
exactly as your prior memos predicted.

## Where the genuine novelty might live (NOT in this paper)

If a non-rank-1 surface response is needed, candidates are:

1. **Sanchez 2002 NSE 142** "double-PN approximation" — a higher-
   rank generalization of the surface basis, possibly containing
   Eq. (13)'s extension to rank ≥ 3.
2. **Bogado Leite 1998 ANE** — orphaned per
   `rank_n_ic_curvilinear_literature_leads.md`. Worth pulling.
3. **Sanchez & McCormick 1982 NSE 80 §III.F.1** — the abstract DP-N
   generalization that this 1977 paper instantiates. Per
   `sanchez_mccormick_rank_n_per_face.md` it has the µ-weight that
   ORPHEUS F.4 misses — but it's also the source you've already
   identified as not extending to curvilinear geometries.

## Recommendation

**Do not pursue Sanchez 1977 implementation.** It is the rank-3
surface IC reference, but rank-3 = 3 modes per face = algebraically
equivalent to the rank-M Schur path Rodrigo already falsified for
the Mark overshoot question. The Hébert (1-P_ss)^-1 closure ORPHEUS
ships is the **rank-0 collapse** of this same family, and Sanchez's
own Tables VI-VII demonstrate the rank-3 lift is bounded at
~3% × cell-averaged-flux for heterogeneous cells — not enough to
close a +50% overshoot.

The path forward on Issue #132 is either:
- Accept the Mark limit as the structural ceiling for the IC family
  and document it (most defensible given the four-reference
  synthesis in `rank_n_closure_four_references_synthesis.md`); OR
- Pull Sanchez 2002 NSE 142 / Bogado Leite 1998 ANE for a
  potentially distinct surface response (still speculative).
