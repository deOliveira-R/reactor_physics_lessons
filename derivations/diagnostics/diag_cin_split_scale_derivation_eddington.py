"""Analytical derivation attempt: scale²_opt = 2 + 1/(3ρ) from Eddington factor.

Hypothesis: the formula scale²_opt = (1 + 6ρ) / (3ρ) = 2 + 1/(3ρ) can be
derived from:
  1. The Legendre baseline (α=0 Jacobi, c-weight): scale² = 2.
  2. A geometric correction for the inner-surface solid-angle concentration.
  3. The Eddington factor 1/3 (from ⟨µ²⟩_iso in 3D).

This script:
  (a) Tabulates the empirical scale_opt at multiple (τ, ρ) from E3.1 data.
  (b) Numerically integrates candidate correction formulas to match.
  (c) Tests the variant: scale²_opt = 2 + (1/3) · geometric_factor(ρ),
      where geometric_factor = 1/ρ, ρ/something, etc.
  (d) Confirms (or refutes) the τ-independence at σ_t·R ≥ 5.

The goal: find a SIMPLE analytical formula that matches empirical.
"""
from __future__ import annotations

import math
import numpy as np


# Empirical data from E3.1 α-scan
# (σ_t·R, ρ, α_opt) — α_opt is Jacobi weight power such that weight = c^(α+1)
# At σ_t·R < 5 the pattern is qualitatively different — skip here.
EMPIRICAL = [
    # (σ_t·R, ρ, α_opt, err_opt)
    (5.0,  0.10, 3.00, 0.3827),  # α may be capped (scan max was 3)
    (5.0,  0.30, 1.00, 0.0716),
    (5.0,  0.50, 0.50, 0.0463),
    (5.0,  0.70, 0.50, 0.7932),
    (10.0, 0.10, 3.00, 0.4631),  # capped
    (10.0, 0.30, 1.25, 0.0153),
    (10.0, 0.50, 0.75, 0.0270),
    (10.0, 0.70, 0.50, 0.3259),
    (20.0, 0.10, 3.00, 0.0899),  # capped
    (20.0, 0.30, 1.00, 0.0787),
    (20.0, 0.50, 0.50, 0.1526),
    (20.0, 0.70, 0.50, 0.0636),
    (50.0, 0.10, 3.00, 0.0943),  # capped
    (50.0, 0.30, 1.25, 0.0500),
    (50.0, 0.50, 0.50, 0.0798),
    (50.0, 0.70, 0.50, 0.1324),
]


def scale_sq_from_alpha(alpha):
    """Jacobi weight α maps to basis scale via scale² = α + 2."""
    return alpha + 2.0


def main():
    print("=" * 80)
    print("Analytical derivation test: scale²_opt vs candidate formulas")
    print("=" * 80)

    print(f"\n{'σ_t·R':>6} {'ρ':>6} {'α_opt':>6} {'sc²_emp':>8} "
          f"{'α·ρ':>8} {'formulas: (1+6ρ)/(3ρ)':>22} "
          f"{'2+1/(3ρ)':>10} {'2+1/(3ρ²)':>10}")
    print("-" * 95)

    for row in EMPIRICAL:
        tau, rho, alpha, err = row
        sc_sq = scale_sq_from_alpha(alpha)
        alpha_rho = alpha * rho
        # Candidate 1: (1+6ρ)/(3ρ)   (same as 2 + 1/(3ρ))
        cand1 = (1 + 6*rho) / (3*rho)
        # Candidate 2: 2 + 1/(3ρ²)   (if it's 1/ρ² correction)
        cand2 = 2 + 1.0/(3*rho*rho)

        # Excess α·ρ over the conjectured 1/3:
        print(f"{tau:>6.1f} {rho:>6.2f} {alpha:>6.2f} {sc_sq:>8.3f} "
              f"{alpha_rho:>8.3f} {cand1:>22.3f} "
              f"{cand1:>10.3f} {cand2:>10.3f}")

    print("\n" + "=" * 80)
    print("Testing τ-independence: α_opt at fixed ρ across τ")
    print("=" * 80)
    for rho_target in [0.1, 0.3, 0.5, 0.7]:
        rows = [r for r in EMPIRICAL if r[1] == rho_target]
        print(f"\nρ={rho_target}:")
        for tau, rho, alpha, err in rows:
            pred = 1.0 / (3.0 * rho)
            print(f"  σ_t·R={tau:>5.1f}: α_emp={alpha:>4.2f}, α_pred=1/(3ρ)={pred:>6.3f}, "
                  f"ratio={alpha/pred:>5.3f}")

    print("\n" + "=" * 80)
    print("Core finding check:")
    print("=" * 80)
    print("Conjecture: scale²_opt = 2 + 1/(3ρ) [Legendre baseline + Eddington/cavity]")
    print("ρ     | scale_opt formula | √(2+1/(3ρ))")
    for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"{rho:.2f}  | {math.sqrt((1+6*rho)/(3*rho)):.4f}   | {math.sqrt(2 + 1/(3*rho)):.4f}")

    # Attempt: where does 1/(3ρ) come from?
    print("\n" + "=" * 80)
    print("Physical-insight test — possible origins of 1/(3ρ):")
    print("=" * 80)
    print("  • Eddington factor: ⟨µ²⟩_iso = 1/3 (3D isotropic flux)")
    print("  • 1/ρ could be:")
    print("    - inner surface flux concentration factor (R/r_0)² = 1/ρ²?")
    print("    - inner chord length factor (1-ρ) · something?")
    print("    - solid-angle ratio for the steep cone ≈ ρ² for small ρ?")
    print()
    print("Attempt: g(ρ) = 1/ρ comes from... the ratio of outer-to-inner")
    print("basis normalization under c-weight on the [0,1] half-hemisphere.")
    print("Integrating the steep cone's projection onto a c-weighted basis:")
    print()
    print("  ∫_{µ_crit}^{1} (1) · µ dµ = (1 - µ_crit²)/2 = ρ²/2")
    print("  vs total [0,1]: ∫_0^1 (1) · µ dµ = 1/2")
    print("  ratio = ρ². Hmm, ρ² not 1/ρ.")
    print()
    print("Alternative: inner surface angular-flux conservation")
    print("  ⟨c²⟩_inner · dA_inner = ⟨c²⟩_outer · dA_outer ·(some factor)")
    print("  (1/3) · 4π r_0² = (1/3) · 4π R² · (inner capture fraction)")
    print("  capture fraction = ρ². And (1/ρ) factor comes from flux")
    print("  inversion — flux at inner = 1/ρ² × flux at outer (Liouville).")
    print()
    print("So 1/(3ρ) = (1/3) · (1/ρ) — Eddington × geometric factor.")
    print("WHY 1/ρ not 1/ρ²? This is the load-bearing open question.")


if __name__ == "__main__":
    main()
