# Archived modules — `orpheus.derivations`

Code removed from the active `orpheus.derivations` package but preserved
verbatim because it may be useful for future work. Each archive entry
records *why* it was removed and *when* it should come back.

## Directory layout

```
derivations/archive/
├── README.md                        ← this file (the index)
├── peierls_moments.py               ← see "Moment-form Nyström"
├── peierls_slab_moments_assembly.py ← slab K assembly via moments
├── peierls_cylinder_polar_assembly.py ← cylinder-polar (φ-quad)
└── tests/
    ├── test_peierls_moments.py            ← L0 moment-recursion gates
    └── test_peierls_slab_moments.py       ← L1 slab moment-form K gates
```

## Archive entries

### Moment-form Nyström for production CP — see [GitHub Issue #117](https://github.com/deOliveira-R/ORPHEUS/issues/117)

**Files:**
- `peierls_moments.py` — closed-form polynomial moments for $E_n$,
  $\mathrm{Ki}_n$, $e^{-u}$.
- `peierls_slab_moments_assembly.py` — slab K assembly via the
  moment-form Nyström architecture (closed-form, exact, ~200 ms for
  N=24 at p=6, dps=30).
- `tests/test_peierls_moments.py` — 32 L0 gates of the closed-form
  moment recursions vs `mpmath.quad` (1e-13 / 1e-12 / 1e-15 tolerances
  per kernel family).
- `tests/test_peierls_slab_moments.py` — 9 L1 gates of the slab
  moment-form K matrix vs the legacy E_1 Nyström and the adaptive
  polar reference (1e-12 / 1e-10 elementwise).

**Why archived:** Issue #117 captures the full architecture and
literature derivations. The verification side (`peierls_geometry`) does
not need a fast K assembly — it can afford adaptive `mpmath.quad` per
element (`K_vol_element_adaptive`). The moment form is the *production*
path for a future higher-order discrete CP solver; this branch's CP
production module (`orpheus.cp`) uses flat-source CP and does not need
it.

**When to bring back:** When implementing higher-order spatial source
expansion in production CP, or when adding LS-MOC / quadratic-MOC to a
future MoC production solver. See Issue #117 for the trigger
conditions and the conditioning / Vandermonde caveats.

### Cylinder-polar assembly (explicit out-of-plane φ quadrature)

**Files:**
- `peierls_cylinder_polar_assembly.py` — the
  `_build_volume_kernel_cylinder_phi` body that computes
  $\mathrm{Ki}_1(\tau)$ via $\sum_{\varphi} \exp(-\tau/\cos\varphi)\,w_\varphi$
  with a 16-point GL on $[0, \pi/2]$.

**Why archived:** Mathematically equivalent to `cylinder-1d`
($\mathrm{Ki}_1$ evaluated via `ki_n_mp` / `ki_n_float` directly).
Verified element-wise to machine precision against `cylinder-1d` at
n_phi=32. The cylinder-polar route was a detour in the
"retire-Bickley" sub-thread of issue #116 — useful for the
exposition of the φ-decomposition but not a separate physics
construct.

**When to bring back:** If a future analysis needs to expose the
out-of-plane angular distribution explicitly (rather than as an
integrated Bickley value), e.g., for higher-order angular flux
moments at the cell surface, this is the assembly to start from.
Otherwise, `cylinder-1d` with closed-form `ki_n_mp` is the natural
form.
