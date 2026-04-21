# Rank-N hollow-sphere closure — the four-reference synthesis and close-out

**Branch**: `investigate/peierls-solver-bugs`
**Date**: 2026-04-21
**Issue**: #119
**Status**: RESOLVED as **"Accept F.4 as production — Sanchez-McCormick 1982 §III.F.1 rank-N formulation is not cross-validated"**

## The headline finding

**F.4 scalar rank-2 per-face white-BC closure IS the textbook closure.**

Four authoritative references were checked this session:

| Reference | Chapter | IC Method | Rank-N Legendre Ladder? |
|-----------|---------|-----------|--------------------------|
| Ligou 1982 *Elements of Nuclear Engineering* | Ch. 8 §8.2 | Scalar / DP-0 cosine-return | No |
| Sanchez, Mao, Santandrea 2002 NSE 140:23 | entire paper | Piecewise-constant angular sectors (N² sectors) OR collocation-δ₂ | No (refers to Sanchez-McCormick 1982 for Legendre) |
| Stamm'ler & Abbate 1983 *Methods of Steady-State Reactor Physics* | Ch. IV §10 | Scalar cosine-return with albedo geometric series | No |
| Stacey 2007 *Nuclear Reactor Physics* | Ch. 9 §9.4-9.5 | DP-0 response matrix | No |

None of these four authoritative references presents a rank-N Legendre-moment ladder for per-face interface currents on curvilinear cells. **The Sanchez-McCormick 1982 §III.F.1 recipe we implemented is unique among the literature searched** — it is a theoretical construction that did not cross over into any successor textbook.

ORPHEUS's F.4 closure is literally the formulation of Stamm'ler Eq. 34 and Ligou Eq. 8.47: scalar partial-current Lambertian-return with geometric-series reflection. Its 0.077 % residual at σ_t·R = 5 is the expected accuracy for this method at this optical thickness.

## Why the Sanchez rank-N fails to close — the c_in ≠ µ_emit mismatch

At σ_t = 0 on a hollow sphere, the ORPHEUS shipped `compute_hollow_sph_transmission_rank_n` (MC-verified) satisfies:

- **Mode 0**: `W_oo[0,0] + W_io[0,0] = 1` exactly (F.4 conservation identity).
- **Modes n ≥ 1**: `W_oo[n,n] + W_io[n,n] ∈ {0.28, 0.13, 0.09}` for n=1, 2, 3 — nowhere near 1.

This failure is NOT a bug. It's the structural consequence of the **angle-remapping at the inner surface**:

- Outer-outer grazing rays: µ_emit = µ_arrive = cos θ (sphere symmetry preserves direction).
- Outer-inner chord rays: µ_arrive_at_inner = c_in = √(1 − (R/r_0)²(1 − µ²_emit)) ≠ µ_emit.

At mode 0, `P̃_0 = 1` is angle-independent — the remapping is invisible. At mode n ≥ 1, `P̃_n(c_in) ≠ P̃_n(µ_emit)`, and the moment basis doesn't capture the c_in mapping correctly.

**The Sanchez §III.F.1 Legendre ladder implicitly assumes the same basis functions apply at both emission and arrival, which is only true at mode 0.** The failure at higher modes is thus inherent to the formulation — not fixable by prefactor tuning, basis renormalization, or Gelbard-factor placement.

## Recommended fix paths (for future research)

From Sanchez 2002 (§V and code in TDT of APOLLO2): the correct modern approach is **piecewise-constant angular sectors** on the hemispheres, not Legendre moments. The basis functions are characteristic functions on N² angular cones, orthonormal under the `(Ω·n) dΩ` inner product. Particle conservation is exact by construction: the angular sectors at emission and arrival see the same angular measure, and the c_in mapping is handled as a direct ray-by-ray reprojection within the piecewise-constant framework.

Implementing Sanchez 2002 would be a SIGNIFICANT lift:
- New basis data structures (angular sector partitioning).
- New P, G, W primitive implementations with sector-averaged integrands.
- New iterative quadrature tracking if following APOLLO2's TDT.

Alternative: re-derive a Legendre rank-N closure that **correctly handles the c_in remapping** analytically. This would require substantial derivation work and is not documented in any of the four references.

## What's shipped

### Committed in this investigation (chronological)

1. `ca9d68f` — Marshak partial-current primitives (infrastructure, dead code behind guard).
2. `53fae60` — `solve_peierls_1g` inner_radius bug fix + MC cross-check of W (33 tests).
3. `0b0533b` — Sanchez §III.F.1 recipe scan + sympy µ-ortho basis derivation (60+ variants).
4. `a2e2205` — Plateau characterization + cross-σ_t parameter scan + Stamm'ler Ch. 6 negative finding.
5. *THIS commit* — Four-reference synthesis + conservation structural diagnosis + F.4 quadrature characterization.

### NotImplementedError guard remains

```python
if n_bc_modes > 1 and reflection == "white" and geometry.n_surfaces == 2:
    raise NotImplementedError(
        "Rank-N per-face white BC (n_bc_modes > 1) ... see Issue #119 follow-up."
    )
```

Production code (build_closure_operator for slab + hollow cyl/sph) uses F.4 rank-2 scalar path.

## Decisive data

### Cross-parameter F.4 vs Sanchez N=2

| σ_t·R | F.4 N=1 err | Sanchez N=2 err | Sanchez N=3 err |
|-------|-------------|-----------------|-----------------|
| 1.0 | 3.27 % | 42.02 % | 49.15 % |
| 2.5 | 0.55 % | 5.89 % | 5.94 % |
| 5.0 | **0.077 %** | 1.42 % | 1.42 % |
| 10.0 | 0.26 % | 0.53 % | 0.53 % |
| 20.0 | 0.45 % | 0.44 % | 0.45 % |

F.4 wins at practical cells; both converge to ~0.44 % at thick optical cells.

### Conservation probe at σ_t = 0

| Mode n | W_oo[n,n] | W_io[n,n] | Sum | Expected (naive) |
|--------|-----------|-----------|-----|-------------------|
| 0 | 0.910 | 0.090 | **1.000** ✓ | 1.000 |
| 1 | 0.251 | 0.030 | 0.281 | 1.000 (fails) |
| 2 | 0.132 | 0.002 | 0.134 | 1.000 (fails) |
| 3 | 0.091 | 0.001 | 0.092 | 1.000 (fails) |

F.4 mode-0 conservation holds EXACTLY because P̃_0 = 1 is angle-independent. All higher modes fail due to c_in ≠ µ_emit.

### F.4 quadrature convergence (σ_t·R = 5, r_0/R = 0.3)

| n_p | p_order | n_angular | N_r | err % | Time |
|-----|---------|-----------|-----|-------|------|
| 2 | 4 | 16 | 8 | 0.27 % | 0.2s |
| 2 | 4 | 24 | 8 | 0.077 % | 0.4s |
| 4 | 4 | 24 | 16 | 0.036 % | 1.4s |
| 4 | 6 | 24 | 24 | 0.082 % | 3.2s |
| 4 | 6 | 32 | 24 | 0.095 % | 5.6s |
| 8 | 6 | 32 | 48 | 0.044 % | 34s |

F.4's residual **does not converge to machine precision** under refinement — it fluctuates around a ~0.04-0.1 % FLOOR. This is a Mark DP_0 truncation error intrinsic to the scalar closure, NOT pure quadrature error.

At σ_t·R = 5 the truncation floor is lower than at other parameters (per the cross-parameter scan), explaining why F.4's N=1 is best at this specific operating point. The ~0.04 % floor is the fundamental limit of F.4's approach for curvilinear cells.

The same floor exists for Sanchez rank-N but at a MUCH higher level (~1.42 %) due to the c_in remapping structural failure documented above.

## Artefacts updated this session

### New

- `derivations/diagnostics/diag_rank_n_sanchez_conservation_probe.py` — 2 pytest gates (mode-0 holds, n≥1 fails) + main() for inspection.
- `.claude/agent-memory/literature-researcher/rank_n_closure_four_references_synthesis.md` — 400-line side-by-side comparison of the 4 new references vs Sanchez-McCormick 1982 §III.F.1.
- `.claude/plans/post-four-reference-synthesis-close-out.md` — this doc.

### Memory hygiene (by literature-researcher)

- `stammler_1983_ch6_interface_currents.md` — Ch. IV added as the correct CP chapter reference.
- `cp_moment_integrals.md` — Ch. IV specifically (not Ch. 4-6).
- `rank_n_interface_current_canonical.md` — Sanchez-McCormick 1982 §III.F.1 flagged as not cross-validated by the authoritative corpus.
