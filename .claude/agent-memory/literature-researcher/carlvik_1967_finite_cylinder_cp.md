---
name: Carlvik 1967 NSE 30(1) — finite-cylinder/cuboid CP technical note (PDF read 2026-04-25)
description: Two-page Technical Note giving exact CP integrals for finite cylinders and cuboids in terms of E_n exponential integrals. NOT a Bickley-Naylor cylinder paper. The 1965 Carlvik-Geneva paper is the foundational rank-N CP reference for general cylindrical geometry; this 1967 note is a focused finite-volume add-on.
type: reference
---

# Carlvik 1967 NSE 30(1):150-151 — substantive content (read 2026-04-25)

## Headline citation

**I. Carlvik (1967)**, "Collision Probabilities for Finite Cylinders
and Cuboids", *Nuclear Science and Engineering* **30**(1), 150-151.
DOI: [10.13182/NSE30-01-150TN](https://doi.org/10.13182/NSE30-01-150TN).
Received March 20 1967 from AB Atomenergi, Stockholm. **Two-page
Technical Note.** AE-report Ref. 6 (the long-form derivation) was
"to be published" but never resolved to a public report.

## Scope (very narrow)

The note presents **closed-form analytic CP integrals for two
homogeneous bodies that Case-de Hoffmann-Placzek (1953) did not
treat**: the cuboid (a×b×c) and the finite cylinder (height h,
diameter d). All dimensions are in mean free paths (Σ_t · length).

There is **no rank-N machinery, no boundary-current → flux Green
function, no white-BC closure, and no relation to Bickley-Naylor
functions** in this paper. The kernels are E_n exponential
integrals (the 3D point-kernel result) — appropriate for finite
3D bodies, NOT the 2D cylinder Peierls equation.

## The two formulas

### Cuboid (Eq. 1)

For a homogeneous cuboid of dimensions a × b × c:

```
P_c = 1 - [4 / (a·b·c)] · ∫_0^c dz · {
    (a · E_3(z))/4 · ... wait — paper OCR is fragmented.
}
```

Cleaning up the visible structure (lines 99-112 of the OCR):

```
P_c = 1 - 4/(a·b·c) · ∫_0^c { (a · E_3(z))/2 -
        ∫_0^a (a-u) · E_3[(u² + z²)^(1/2)] du -
        ∫_0^b (b-v) · E_3[(v² + z²)^(1/2)] dv +
        2 ∫_0^a du ∫_0^b dv (a-u)(b-v) · u·v · E_3[(u²+v²+z²)^(1/2)]
      } · ... dz                                                   (1)
```

The exact OCR-reconstruction is unreliable; what's certain is:
- The kernel is E_3 (third-order exponential integral, the
  spherical-shell integrated point kernel).
- The integration is over the **rectangular volume**, exploiting
  the cuboid's three-fold symmetry.
- The result is expressible as **single integrals** (after the
  inner two are tabulated as the closed terms shown). Per
  Carlvik's narrative line 50: "It was found that the collision
  probability can be expressed by single integrals."

### Finite cylinder (Eq. 2)

For a homogeneous cylinder of height h and diameter d:

```
P_c = 1 - (4/h) · [E_3(0) - E_3(h)] · ... + double integral
```

The **double-integral** form (line 167-180 of OCR):

```
P_c = 1 - 4/(... · h · d²) · {
    ∫_0^h (h-u) du · ∫_0^d t dt · 2 · exp[-(t² + u²)^(1/2)] / (t² + u²)^(3/2) ·
    [d² - t²]^(1/2)
  } - (single integrals from E_3 terms)                             (2)
```

The exponent `-(t² + u²)^(1/2)` is the optical-path along a 3D
straight line from a point at axial offset u and radial offset t.
The `(d² - t²)^(1/2)` factor is the **chord half-length** at
radial impact parameter t — same chord geometry as the 2D Peierls
equation, but now multiplied by the axial separation kernel.

**Singularity at u = 0, t = 0**: the integrand has a `1/(t²+u²)^(3/2)`
singularity. Carlvik handles it by partial integration (line 341):

```
∫_0^h du (h-u) ∫_0^d t dt · 2 exp[-(t²+u²)^(1/2)] / (t²+u²)^(3/2) · (d²-t²)^(1/2)
  = ∫_0^h du M(u, d) ... (closed-form re-arrangement)
```

The end-point singularity at t = d (where chord half-length
vanishes infinitely steeply) is removed by changing variable to
`(d-t)^(1/2)` as the linear integration variable. Both
substitutions are standard collision-probability accuracy tricks.

## Numerical verification (Tables I-IV)

Tables compare Carlvik's COLCYL/COLCUB Gaussian quadrature
(8-, 12-, 16-, 24-point) against Monte Carlo of Foell (CEP) and
Gubbins. Convergence is **excellent**: 24-point Gauss agrees with
Monte Carlo to ±0.0001 on collision probabilities for cylinders
with `max(d, h) < 5`. For h/d = 1, P_c at d = 0.1 mfp is 0.04065,
at d = 17.0 mfp is 0.91491. For h/d = 2, COLCYL 24-point matches
to ±0.0001 against Foell's CEP on d ∈ [0.8, 12.0].

Computing time: ~1 s per case on IBM 7044 (1967). The author
concludes (line 511): "Monte Carlo methods, which are necessary
for more complicated geometries, can be avoided for these
particular shapes."

## What it does NOT contain

1. **No Bickley-Naylor functions**, no Ki_n. The kernel is E_3
   (3D point-kernel integrated over a spherical shell). The 2D
   transverse integration that produces Ki_n requires axial
   translation invariance — a finite-height cylinder breaks this.

2. **No 3D Bickley function decomposition**. The phase4 memo's
   "Ki_{2+k} expansion (Knyazev 1993)" is unrelated to anything
   in this paper. Carlvik 1967 does NOT give a Ki_2 expansion or
   a Bickley-function decomposition of the finite-cylinder kernel.

3. **No boundary-current → interior-flux response**. The note
   computes only the **single-region collision probability** (i.e.
   `P_c = ∫∫∫ Σ_t · φ_un / S` where φ_un is the uncollided flux
   from a uniform internal source). There is no `G_bc(r)`, no
   `P_esc`, no white-BC closure formula, no `4·P_esc` reciprocity
   relation.

4. **Cuboid CP only — no rectangular-cell IC**. The cuboid result
   is purely for fast-fission-effect / activation calculations,
   NOT for use as a transport-cell.

5. **No rank-N machinery**. The technical note is a single rank-0
   integral.

## The forward citation chain — Carlvik 1965 (the foundational paper)

Sanchez 1977 NSE 64 cites (Ref. 27): **I. Carlvik, "A Method for
Calculating Collision Probabilities in General Cylindrical Geometry
and Applications to Flux Distribution and Dancoff Factors",
*Proc. Third Int. Conf. Peaceful Uses At. Energy*, Geneva,
**Vol. 2**, p. 225, United Nations, New York (1965).** This 1965
paper is the **foundational reference for the Wigner-Seitz-style
cylindrical-cell CP method** that all rank-0 cylindrical CP codes
(WIMS, APOLLO, DRAGON, ORPHEUS `cp_cylinder.py`) descend from.

The 1965 paper is what gives the **chord-tracing algorithm for
multi-region annular cylinders** — the same algorithm
`_chord_half_lengths` in `cp_cylinder.py` implements. It is NOT
the 1967 NSE 30(1) Technical Note.

**The 1967 note explicitly says (line 51): "Since we were not
aware of the work of McLeod at that time, we used the same
approach for the finite cylinder and arrived at a closed formula
containing double integrals. The derivations are included in an
AE report 6."** This places the 1967 note as a *finite-volume*
extension that came after the 1965 *infinite-cylinder* paper.

## Cross-check vs `phase4_cylinder_peierls.md`

The phase4 memo cites Carlvik 1967 NSE 30 in its bibliography but
**does not derive any equation from it**. The memo's content
(Sanchez 1982 §IV.A formulas, 1/π prefactor, Ki_1 vs Ki_3) is
*correct* and confirmed by independent reading of Sanchez 1977
Appendix A.2. The Carlvik 1967 citation in that memo is decorative
— remove it or replace it with **Carlvik 1965 (Geneva, vol. 2,
p. 225)** which is the actual algorithmic source for the ORPHEUS
chord-tracing code.

## Recommendation for Issue #132

**Carlvik 1967 is irrelevant to Issue #132.** It contains no
boundary-current closure, no rank-N machinery, no white-BC
formula, and no Bickley-Naylor expansion. It is a finite-3D-
volume CP note for activation/Dancoff purposes.

The actually-needed reference is **Carlvik 1965 (Geneva
proceedings, Vol. 2, p. 225)** — the foundational infinite-
cylinder annular CP paper. This is harder to obtain (1965 UN
proceedings volume) but probably available via OSTI or IAEA
INIS. Worth a focused search if multi-region cylindrical CP
algorithm details are needed.

## Quote-verified equation forms

Cuboid Eq. (1) (paraphrased — OCR fragmented; consult PDF
directly for exact form):
- Kernel E_3, integrand u·v·(a-u)·(b-v) over [0,a]×[0,b]×[0,c],
  yielding P_c via 1 - (volumetric integral) / (a·b·c).

Finite cylinder Eq. (2):
- Single integrals: E_3(0) - E_3(h), boundary terms.
- Double integral over chord coordinates (u, t) with (d²-t²)^(1/2)
  chord factor and 1/(t²+u²)^(3/2) point-kernel singularity at
  origin removed by partial integration.

These are the correct kernels for **finite 3D bodies** with
complete spatial integration over both ends and lateral surface;
they are NOT applicable to the 2D-Peierls / 1D-radial / cylindrical-
cell-with-axial-symmetry transport problem ORPHEUS solves.
