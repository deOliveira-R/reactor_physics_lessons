---
name: Stamm'ler & Abbate 1983 Chapter VI — negative finding on interface currents
description: Ch.VI of Stamm'ler-Abbate (1983) "Methods of Steady-State Reactor Physics in Nuclear Design" is the SN method chapter, not interface currents. Records what IS in the chapter, what is NOT, and where the interface-current material actually lives in the same book.
type: reference
---

# Stamm'ler & Abbate (1983) Chapter VI — scope clarification

## Citation

Stamm'ler, R.J.J. and Abbate, M.J., *Methods of Steady-State Reactor
Physics in Nuclear Design*, Academic Press, 1983. Chapter VI:
"The Discrete Ordinates or Discrete S_N Method", pp. 191-230.

PDF on disk: `/workspaces/ORPHEUS/Stammler(1983)Chapter6.pdf` (40 pp.).

## Negative finding

**Chapter VI is the S_N method chapter.** It contains no interface-
current method, no multifunction collision probability, no rank-N
per-face surface-mode expansion, no Stepanek-style anisotropic
interface currents, and no Benoist homogenization. The entire 40
pages are the multigroup discrete-ordinates equations, quadrature
sets, finite-difference schemes, boundary conditions, and 2D iteration
strategy — all SN, not CP / interface-current.

The references section (p.230) confirms the scope: Arkuszewski,
Carlson-Lathrop, Chandrasekhar, Lathrop, Lathrop-Carlson, Lee, Mika,
Wick. These are SN-method references exclusively — no Sanchez,
Bonalumi, Pellaud, Stepanek, Benoist, or any interface-current
literature.

## What Chapter VI actually contains (reconstructed ToC)

| § | Pages | Content |
|---|---|---|
| 1 | 191-194 | Multigroup discrete-ordinates equations, (Ω̂·∇ + Σ)ϕ_m = S_m + (χ/k)F, linearly-anisotropic scattering, current-weighted Σ_{s1} |
| 2 | 195-200 | Spatial discretization in spherical symmetry; angular redistribution coefficients a_{m±1/2} = ½(1-µ²)∫4πrϕ dr/∫ϕ dr |
| 3 | 201-210 | Quadrature sets: level-symmetric, moment-matching conditions, Gauss-Legendre in 1D, P_L ↔ S_{L+1} equivalence in slab |
| 4 | 211-213 | Boundary conditions: specular reflection (Eq. 26), white reflection (Eq. 27-28), albedo (Eq. 29-30) — all implemented by reassigning ϕ_m at the boundary, not by mode expansion |
| 5 | 213-223 | Finite-difference schemes: weighted (Eqs. 32), diamond-difference (a=½), step (a=1), optimal a(τ) from Eq. (36) |
| 5.2 | 217-219 | Negative-flux fix-up |
| 6 | 220-228 | 2D iteration sweep strategy, FORTRAN listings, ray effect |
| Q | 228-229 | 12 exercises |
| Refs | 230 | SN-method references only |

## SN-specific white BC (NOT rank-N)

For the record — Eq. (27c) on p.212 gives Stamm'ler's SN white BC:

    ϕ_m(x_a) = j⁺(x_a) / [½ Σ_m w_m |µ_m|]     for µ_m > 0

i.e. the returning angular flux is **constant in µ** and set equal
to the outgoing partial current divided by ½Σw|µ|. This is a
rank-0 (scalar) closure, just expressed over the SN quadrature. It
does **not** generalize to rank-N in the sense we need for Issue
#119 — there's no per-mode basis, no transmission matrix, and no
reflection operator.

## Where the interface-current material actually lives in Stamm'ler-Abbate

Not in Chapter VI. The book is organized roughly:

- Ch. I-II: Boltzmann equation foundations
- Ch. III: Collision probability method (this is where CP primitives,
  Wigner-Seitz cells, and interface currents appear)
- Ch. IV: Integral transport, P_L theory in slab
- Ch. V: P_L spherical-harmonics in multi-D
- **Ch. VI: SN method** ← the PDF we have
- Ch. VII+: resonance treatment, thermalization, reactor calculation

For the rank-N interface-current formulation, the relevant chapter
in Stamm'ler-Abbate is **Chapter III (Collision Probability Method)**
— NOT Chapter VI. We do not have a PDF of Chapter III on disk.

## Follow-up: where to actually get the rank-N per-face formulation

Since Stamm'ler-Abbate Ch. VI is the wrong chapter, the canonical
rank-N per-face closure still lives in:

1. **Sanchez & McCormick (1982)** NSE 80 — see
   `sanchez_mccormick_rank_n_per_face.md` in this memory directory
   for the extraction we already have.
2. **Stamm'ler-Abbate Ch. III** — would need to acquire PDF
   separately. Bonalumi and Pellaud formulations are typically
   attributed here.
3. **Hébert (2009)** *Applied Reactor Physics* — Ch. 3 covers CP
   with interface currents, generally more modern notation.
4. **Stepanek (1982, 1984)** — the anisotropic-interface-current
   author cited in the prior Peierls investigation. Original papers
   in NSE / ANE.

## How to apply

- If asked for "Stamm'ler Ch. 6 interface currents" in the future:
  this citation is wrong. Redirect to Stamm'ler Ch. III (CP method)
  or to Sanchez-McCormick 1982.
- The SN white BC formula Eq. (27c) can be cited when a discrete-
  ordinates white BC needs a textbook reference, but it is not
  relevant to the rank-N hollow-sphere closure for Issue #119.
- Do not confuse "constant-in-µ returning flux" (Stamm'ler SN white
  BC) with the rank-0 Lambert closure in ORPHEUS Phase F.4 — they
  are superficially similar (both enforce an isotropic return) but
  live in different solver machineries (discretized-µ quadrature vs
  continuous integral transport).
