#!/usr/bin/env python3
"""Homogeneous infinite reactor eigenvalue solver.

Solves for the neutron spectrum and k-infinity in an infinite homogeneous
medium by iterating on the slowing-down / fission source equation:

    diag(SigT) * phi = chi * P/k + SigS0^T * phi + 2 * Sig2^T * phi

where P = (SigP + 2*colsum(Sig2)) . phi is the total production rate.

This script contains the full solver, plotting utilities, and a main()
that runs two demonstration cases:

  1. Aqueous reactor (H2O + U-235)
  2. PWR-like mixture (UO2 + Zry + H2O+B)

Reference MATLAB results:
    Aqueous reactor:  kInf = 1.03596
    PWR-like mixture: kInf = 1.01357

Dependencies:
    pip install numpy scipy matplotlib
    ORPHEUS data package (orpheus.data)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.recipes import aqueous_uranium, pwr_like_mix


# ===========================================================================
# Solver
# ===========================================================================

@dataclass
class HomogeneousResult:
    """Result of a homogeneous infinite reactor calculation."""

    k_inf: float
    flux: np.ndarray  # (NG,) — group fluxes normalised to 100 n/cm3/s production
    eg_mid: np.ndarray  # (NG,) — mid-group energies (eV)
    de: np.ndarray  # (NG,) — energy bin widths (eV)
    du: np.ndarray  # (NG,) — lethargy bin widths
    sig_prod: float  # one-group production XS (1/cm)
    sig_abs: float  # one-group absorption XS (1/cm)
    mixture: Mixture

    @property
    def flux_per_energy(self) -> np.ndarray:
        return self.flux / self.de

    @property
    def flux_per_lethargy(self) -> np.ndarray:
        return self.flux / self.du


def solve_homogeneous_infinite(
    mix: Mixture,
    n_iter: int = 5,
) -> HomogeneousResult:
    """Solve the eigenvalue problem for an infinite homogeneous reactor.

    Parameters
    ----------
    mix : Mixture
        Macroscopic cross sections for the homogeneous medium.
    n_iter : int
        Number of power iterations (5 is typically sufficient).

    Returns
    -------
    HomogeneousResult
    """
    ng = mix.ng
    sig2_colsum = np.array(mix.Sig2.sum(axis=1)).ravel()
    SigS0_T = mix.SigS[0].T.tocsr()
    Sig2_T = mix.Sig2.T.tocsr()

    # LHS matrix: diag(SigT) - SigS0^T - 2*Sig2^T
    A = diags(mix.SigT) - SigS0_T - 2.0 * Sig2_T

    phi = np.ones(ng)
    k_inf = 1.0

    for _ in range(n_iter):
        prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
        abs_rate = (mix.SigC + mix.SigF + mix.SigL + sig2_colsum) @ phi
        k_inf = prod_rate / abs_rate
        print(f"    k_inf = {k_inf:.5f}")

        # Solve: A * phi = chi * prod_rate / k_inf
        from scipy.sparse.linalg import spsolve
        rhs = mix.chi * prod_rate / k_inf
        phi = spsolve(A.tocsc(), rhs)

        # Normalise so total production = 100 n/cm3/s
        prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
        phi *= 100.0 / prod_rate

    prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
    abs_rate = (mix.SigC + mix.SigF + mix.SigL + sig2_colsum) @ phi
    total_flux = phi.sum()

    eg = mix.eg
    eg_mid = 0.5 * (eg[:ng] + eg[1 : ng + 1])
    de = eg[1 : ng + 1] - eg[:ng]
    du = np.log(eg[1 : ng + 1] / eg[:ng])

    return HomogeneousResult(
        k_inf=k_inf,
        flux=phi,
        eg_mid=eg_mid,
        de=de,
        du=du,
        sig_prod=prod_rate / total_flux,
        sig_abs=abs_rate / total_flux,
        mixture=mix,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def plot_spectrum(
    result: HomogeneousResult,
    title: str = "",
    output_dir: Path | str = ".",
    prefix: str = "spectrum",
) -> None:
    """Generate flux-per-energy and flux-per-lethargy plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flux per unit energy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(result.eg_mid, result.flux_per_energy)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"Neutron flux per unit energy (cm$^{-2}$s$^{-1}$eV$^{-1}$)")
    ax.set_title(f"{title} — Flux per unit energy" if title else "Flux per unit energy")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_flux_energy.pdf")
    plt.close(fig)

    # Flux per unit lethargy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(result.eg_mid, result.flux_per_lethargy)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"Neutron flux per unit lethargy (cm$^{-2}$s$^{-1}$)")
    ax.set_title(f"{title} — Flux per unit lethargy" if title else "Flux per unit lethargy")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_flux_lethargy.pdf")
    plt.close(fig)


# ===========================================================================
# Demo runs
# ===========================================================================

OUTPUT = Path("01_results")


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
