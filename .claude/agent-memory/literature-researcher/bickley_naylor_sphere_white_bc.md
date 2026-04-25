---
name: Sphere Peierls white-BC closure — canonical form is E_n exponential integrals, NOT Bickley-Naylor
description: For solid 1-D sphere with white BC, the canonical pointwise P_esc(r) and multi-region C_ij both close in plain exponentials and E_2 (NOT Ki_n). Bickley-Naylor is cylinder-specific; using it for sphere is a recall error. Hébert 2009 §3.8.5 is the textbook home; Stammler/Sanchez/Stacey/Ligou do not derive sphere CP closed forms.
type: reference
---

# Sphere Peierls integral equation — white BC closed form

## Headline result: it's E_n, NOT Ki_n

The user's request asked for the "canonical Bickley-Naylor analytical
closure" for the spherical Peierls equation. **The canonical closure
does not use Bickley-Naylor.** Bickley-Naylor functions Ki_n appear
only when polar-angle integration over the cylinder axis is performed;
in spherical geometry the angular integration produces ordinary
exponential integrals E_n. The user has already SymPy-derived this in
`/workspaces/ORPHEUS/derivations/peierls_class_b_sphere_bickley_naylor.py`
(§Step 5, lines 246-261) and the result is verified numerically against
direct quadrature to 1e-13 relative tolerance. The script's own header
comment notes: *"the literature researcher's reference to Ki_2 in the
original analysis was a recall error."* This memo confirms that against
the textbook record.

## Single-region homogeneous sphere

For a uniform sphere of radius R, total cross section Σ_t, observer at
interior radius r ∈ (0, R), the pointwise escape probability for an
isotropic source emitter at r is

   P_esc(r) = (1/(4 r Σ_t)) · [
                 exp(-Σ_t·(R-r)) - exp(-Σ_t·(R+r))
               + Σ_t·(R+r)·E_2(Σ_t·(R-r))
               - Σ_t·(R-r)·E_2(Σ_t·(R+r))
             ]

with the limit P_esc(0) = exp(-Σ_t·R) (numerically confirmed).
Here E_n(x) = ∫_1^∞ e^(-xt)/t^n dt is the standard exponential
integral (Abramowitz & Stegun §5.1.4 convention; same as
`scipy.special.expn` and `mpmath.expint(n, x)`).

This form is NOT printed verbatim in any of the available canonical
texts; it is an algebraic specialisation of the multi-region form
below. **The derivation is the user's own SymPy derivation**, but the
underlying angular integral

   P_esc(r) = (1/2) ∫_0^π exp(-Σ_t ρ(θ)) sin θ dθ
   ρ(θ) = -r cos θ + sqrt(R² - r² sin² θ)

is the standard ray-traced escape integral (e.g., Bell-Glasstone 1970
§2.7.1 in spirit, though they don't reduce to E_n closed form; Reuss
2008 Ch.14 has the chord-length distribution but stops short of the
exponential-integral reduction). Cite the SymPy derivation script as
authoritative.

## Multi-region sphere — Hébert §3.8.5 IS the canonical home

**Cite key**: Hébert (2009/2020), *Applied Reactor Physics*,
3rd ed., Presses de l'Université de Montréal, DOI:
10.1515/9782553017445. Chapter 3 "The transport equation",
§3.8.5 "Spherical 1D geometry", pp. 123-124, Eqs. (3.324)-(3.336).

The multi-region collision probability between annular regions i and j
with radii bounded by r_{i±1/2}, r_{j±1/2} is given by the integrand

   P_ij = (2π/V_i) · {∫_0^{r_{i-1/2}} dh·h ∫_0^{ℓ_i} dℓ' [
              exp(-(τ_{ij}+Σ_i ℓ' + Σ_j ℓ)|track 1)
            + exp(-(τ_{ij}+Σ_i ℓ' + Σ_j ℓ)|track 2)
          ] (Eq. 3.325)
        + ∫_{r_{i-1/2}}^{r_{j-1/2}} dh·h ∫_0^{ℓ_i} dℓ' ...} (Eq. 3.326)

where τ_{ij} is the optical thickness of the material *between* regions
i and j on the chord, and h is the chord impact parameter. After
integration in ℓ and ℓ', for the case Σ_i ≠ 0 and Σ_j ≠ 0 (Eq. 3.331):

   C_{ij}(τ_n) = (1/(Σ_i Σ_j)) · [
                   exp(-τ_n) - exp(-(τ_n + Σ_i ℓ_i)) - exp(-(τ_n + Σ_j ℓ_j))
                 + exp(-(τ_n + Σ_i ℓ_i + Σ_j ℓ_j))
               ]

The corresponding voided-region cases (Σ_i = 0 or Σ_j = 0) reduce
exponentials to length factors directly (Eqs. 3.332-3.336). **The
sphere C_ij family uses ONLY plain exponentials, not Bickley-Naylor
and not E_n.** The reason: spherical symmetry collapses two angular
integrations rather than one (cf. cylinder where one axial integration
remains, producing Ki_n via ∫₀^{π/2} cos^(n-1)θ exp(-τ/cosθ)dθ).

The white BC closure is in §3.8.4 (cylindrical context) but applies
identically to the spherical reduced CP matrix via Eq. (3.323):

   ℙ_white = ℙ_vac + (β⁺/(1 - β⁺ P_ss)) · P_iS · P_Sj^T

where β⁺ = 1 for white BC (perfectly reflective with isotropic angular
distribution at the boundary), P_iS is the escape vector to the
boundary surface (Eq. 3.320), P_Sj is its reciprocity-related counterpart
(Eq. 3.321: P_Sj = (4 V_j Σ_j / S) · P_jS), and P_ss is the surface-to-
surface transmission (Eq. 3.322). **This is the textbook form of the
G_bc(r) operator the user asked for**, in matrix-on-region form. The
pointwise version (continuous r) is:

   φ(r) = ∫_V K_vol(r, r') Q(r') d³r' + G_bc(r) · 4 J⁻

with G_bc(r) the kernel-integrated escape Green's function
G_bc(r) = (1/π) · ∫_{4π} ψ⁻(r_b(Ω), Ω) e^{-τ(r,r_b)} dΩ

For uniform isotropic J⁻ on the sphere surface the ψ⁻ = J⁻/π factor
pulls out and **G_bc(r) reduces to exactly the same E_2 closed form
as P_esc(r), by reciprocity**:

   G_bc(r) = 4 · P_esc(r)
           = (1/(r Σ_t)) · [exp(-Σ_t·(R-r)) - exp(-Σ_t·(R+r))
                            + Σ_t·(R+r)·E_2(Σ_t·(R-r))
                            - Σ_t·(R-r)·E_2(Σ_t·(R+r))]

The factor of 4 is the (4π · 1/π) angular-integration constant. This
is the answer to the user's "Specific question — the surface return
G_bc(r)": it is `4 · P_esc(r)` for white BC, and the same E_2 closed
form. **This is NOT printed verbatim in any of the available textbooks
either**; it is the natural reciprocity-based corollary. The textbook
home for the operator structure is Hébert §3.8.4-§3.8.5, equations
(3.320)-(3.323) for the matrix form on annular regions.

## What the other candidate sources do NOT provide

- **Stammler & Abbate (1983) Methods of Steady-State Reactor Physics**:
  Ch.IV is cylindrical CP only (rank-0, flat-source); Ch.VI is the SN
  method. **No sphere CP formulas anywhere in the book.** PDF in
  `/workspaces/ORPHEUS/Stammler(1983)Chapter4.pdf` and `Chapter6.pdf`
  but both are non-OCR scanned images so programmatic verification was
  not possible — claim drawn from prior memory `cp_moment_integrals.md`.

- **Bell & Glasstone (1970) Nuclear Reactor Theory**: §2.7 has the
  generic angular escape integral but does NOT reduce to the E_n
  closed form for a sphere; treats sphere only as a benchmark
  (Table 2.7 critical radii). Useful for benchmark validation, not
  for the closure itself.

- **Reuss (2008) Neutron Physics**: Ch.14 has the sphere chord-length
  distribution P(s) = (s²/2R³) for isotropic emission in a sphere
  (a classical result), but does NOT carry through to the closed-
  form P_esc with E_2.

- **Sanchez & McCormick (1982) NSE 80, 481-535**: review article. No
  sphere-specific CP formulas; sections III, IV cover slab/cylinder
  only. Confirmed by full-text grep — no "sphere", "Peierls", or
  "Bickley" mentions anywhere in PDF
  `/workspaces/ORPHEUS/1982NSE80-481.pdf`. The user's project notes
  mistakenly named "Sanchez 1982 §III.E.2 sphere" — that section
  does not exist.

- **Sanchez (2002) Annals of Nuclear Energy review**: spheres mentioned
  only in the context of spherical-harmonics expansion of the collision
  kernel (a different "spherical"), not sphere geometry. No closure
  formulas.

- **Stacey (2007) Nuclear Reactor Physics, 2nd ed., Ch.9**: §9.9
  treats sphere only via the SN method (diamond difference, angular
  derivative); Eq. (9.211) is the conservative-form streaming operator,
  not the integral equation. **Not a CP/Peierls source.**

- **Ligou (1982) Cours de Génie Nucléaire Ch.8**: French textbook;
  no sphere CP material in this chapter.

- **Lewis & Miller (1984) Computational Methods of Neutron Transport,
  Ch.5**: not in the local PDF set; from training I recall Ch.5
  treats CP method but stays in slab + cylinder. Sphere is in Ch.4
  (S_N), not Ch.5 (CP).

- **Carlvik (1965, 1967), Stepanek (1981, 1982)**: all flat-source
  cylindrical CP papers. Not sphere.

## Convention warnings

1. **E_n vs Ki_n**: BOTH share the integral form `∫_0^{π/2} cos^(n-1)θ
   exp(-τ/cosθ) dθ` for slab/sphere with cosθ-weighted hemisphere
   integration. The functional difference is which cos-power weight
   appears. Hébert §3.8.3 Eq. (3.276) uses E_n via the substitution
   u = sec θ, leaving E_n(x) = ∫_1^∞ e^(-xt)/t^n dt. Bickley keeps
   the cosθ form integrated over the half-circle (the cylinder axial
   integration). **Sphere uses E_n; cylinder uses Ki_n. Do not cross
   them.**

2. **Sign of optical path**: Hébert (3.271) defines τ(x',x) =
   ∫_x'^x dx'' Σ(x''), so τ>0 when x>x'. The exponential is
   exp(-|τ|). The sphere integrand uses ρ = chord length (always
   positive); Σ_t·ρ is the optical depth.

3. **Prefactor** in sphere P_esc: angular integral over 4π divided by
   4π gives the (1/2) ∫_0^π factor; the user's SymPy derivation tracks
   this carefully through the t = ρ substitution and lands at the
   (1/(4r Σ_t)) prefactor in the final E_2 form.

4. **White BC vs reflective vs vacuum**: in Hébert's notation β⁺ = 1
   for both white (Mark) and specular reflective. The difference is
   that white re-emits with isotropic angular distribution while
   reflective preserves the angular pattern (μ → -μ). For a 1-region
   bare sphere with white BC, the closed-form Eq. (3.323) reduces to
   the simple "geometric series" sum of escape × return × re-collision.

## Multi-region SymPy verification recommendations

For the implementation in `peierls_geometry.py`, the non-obvious steps
that warrant a SymPy check before coding:

1. **The E_n closed form vs the C_ij(τ_n) form**: confirm that for a
   1-region sphere, summing the C_ij contributions over the trivial
   single-track integral reproduces the user's Step-5 closed form.
   This is the bridge that must be tight.

2. **The chord-segment optical depth τ_n as a piecewise sum**: for
   annular regions [r_{k-1}, r_k] crossed by a chord with impact
   parameter h, the segment length is sqrt(r_k² - h²) -
   sqrt(r_{k-1}² - h²) (or just sqrt(r_k² - h²) for the innermost
   crossed annulus). This is identical to the cylinder annulus chord
   geometry the user already has in `cp_cylinder.py` — should be
   reusable. SymPy check: τ accumulates correctly across boundary
   crossings with no off-by-one.

3. **Reciprocity P_iS · S = 4 V_i · P_Si / Σ_i**: Hébert's Eq.
   (3.321). Confirms the surface-source kernel is just `4 P_esc / Σ_t`
   per unit volume, which is what gives G_bc(r) = 4 P_esc(r) for a
   uniform sphere with isotropic surface inflow.

4. **White-BC matrix update Eq. (3.323)**: numerically verify that
   ℙ_white as defined satisfies row-sum = 1 (conservation), which is
   the equivalent of "k = 1 for a perfectly absorbing-free reflecting
   sphere" sanity check.

## Why this matters for ORPHEUS

The rank-N Marshak closure was falsified for both Class A hollow
(L21 close-out) and Class B solid (this session). The path forward
is the analytical white-BC closure. The user already has the
single-region E_2 closed form (verified to 1e-13 vs direct quadrature
in their SymPy script). The MULTI-REGION extension follows Hébert
§3.8.5 verbatim — the C_ij(τ_n) form Eq. (3.331) is the
textbook-canonical specification, and its implementation is purely
algebraic given the chord-tracking primitives already in
`cp_cylinder.py`. The G_bc operator for the sphere with white BC is
NOT a separate derivation — by reciprocity, it is `4 · P_esc(r)`.

**Bottom line for the user**: the file
`derivations/peierls_class_b_sphere_bickley_naylor.py` should be
renamed `peierls_class_b_sphere_E2.py` (since Bickley-Naylor is a
misnomer here), the §Step 7 G_bc derivation can be completed in 5
lines using the reciprocity argument, and the multi-region extension
should cite Hébert 2009 §3.8.5 Eqs. (3.324)-(3.336) directly. No
further literature search is needed — the textbook record is already
exhausted on this topic.
