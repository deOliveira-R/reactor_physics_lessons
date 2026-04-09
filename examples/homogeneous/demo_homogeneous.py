#!/usr/bin/env python3
"""Run the homogeneous infinite reactor calculations and compare with MATLAB.

Reference MATLAB results:
    Aqueous reactor:  kInf = 1.03596
    PWR-like mixture: kInf = 1.01357
"""

from pathlib import Path

from orpheus.data.macro_xs.recipes import aqueous_uranium, pwr_like_mix
from orpheus.homogeneous.solver import solve_homogeneous_infinite
from plotting import plot_spectrum

OUTPUT = Path("results")


def run_aqueous():
    print("=" * 70)
    print("HOMOGENEOUS AQUEOUS REACTOR (H2O + U-235)")
    print("=" * 70)

    mix = aqueous_uranium()
    result = solve_homogeneous_infinite(mix)

    print(f"\n  k_inf = {result.k_inf:.5f}  (MATLAB: 1.03596)")
    print(f"  Total flux = {result.flux.sum():.5e}")
    print(f"  Sigma_prod = {result.sig_prod:.5e}")
    print(f"  Sigma_abs  = {result.sig_abs:.5e}")

    plot_spectrum(result, title="Aqueous Reactor", output_dir=OUTPUT, prefix="aqueous")
    return result


def run_pwr():
    print()
    print("=" * 70)
    print("HOMOGENEOUS PWR-LIKE MIXTURE (UO2 + Zry + H2O+B)")
    print("=" * 70)

    mix = pwr_like_mix()
    result = solve_homogeneous_infinite(mix)

    print(f"\n  k_inf = {result.k_inf:.5f}  (MATLAB: 1.01357)")
    print(f"  Total flux = {result.flux.sum():.5e}")
    print(f"  Sigma_prod = {result.sig_prod:.5e}")
    print(f"  Sigma_abs  = {result.sig_abs:.5e}")

    plot_spectrum(result, title="PWR-like Mixture", output_dir=OUTPUT, prefix="pwr")
    return result


def main():
    result_aq = run_aqueous()
    result_pwr = run_pwr()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Case':<25} {'Python k_inf':>14} {'MATLAB k_inf':>14} {'Match':>8}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*8}")
    for name, res, ref in [
        ("Aqueous reactor", result_aq, 1.03596),
        ("PWR-like mixture", result_pwr, 1.01357),
    ]:
        match = "YES" if abs(res.k_inf - ref) < 1e-4 else "NO"
        print(f"  {name:<25} {res.k_inf:>14.5f} {ref:>14.5f} {match:>8}")

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
