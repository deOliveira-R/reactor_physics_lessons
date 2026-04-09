"""Sigma-zero (background cross section) iterative solver.

Given a mixture of isotopes with tabulated total cross sections at various
sigma-zero base points, finds the self-consistent sigma-zero for every
isotope and energy group.
"""

from __future__ import annotations

import numpy as np

from orpheus.data.micro_xs.isotope import NG, Isotope


def solve_sigma_zeros(
    isotopes: list[Isotope],
    number_densities: np.ndarray,
    escape_xs: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Iteratively solve for background cross sections (sigma-zeros).

    Parameters
    ----------
    isotopes : list[Isotope]
        Isotopes in the mixture.
    number_densities : (n_iso,) array
        Atomic number densities in 1/(barn*cm).
    escape_xs : float
        Escape cross section S/(4V) in 1/cm.
    tol : float
        Convergence tolerance for the sigma-zero iterations.
    max_iter : int
        Maximum number of iterations per energy group.

    Returns
    -------
    sig0 : (n_iso, NG) array
        Converged sigma-zero values for each isotope and energy group.
    """
    n_iso = len(isotopes)
    sig0 = np.full((n_iso, NG), 1e10)
    sigT = np.zeros((n_iso, NG))

    for ig in range(NG):
        err = 1e10
        n_iter = 0

        while err > tol:
            # Interpolate total XS at current sigma-zeros
            for i, iso in enumerate(isotopes):
                sigT[i, ig] = _interp_sigT(iso, sig0[i, ig], ig)

            # Update sigma-zeros
            err = 0.0
            for i in range(n_iso):
                background = escape_xs + sum(
                    sigT[j, ig] * number_densities[j]
                    for j in range(n_iso) if j != i
                )
                new_sig0 = background / number_densities[i]
                err += (1.0 - new_sig0 / sig0[i, ig]) ** 2
                sig0[i, ig] = new_sig0

            err = np.sqrt(err)
            n_iter += 1
            if n_iter > max_iter:
                raise RuntimeError(
                    f"Sigma-zero iteration did not converge for group {ig}"
                )

    return sig0


def _interp_sigT(iso: Isotope, sig0_val: float, ig: int) -> float:
    """Interpolate total cross section at a given sigma-zero."""
    if iso.n_sig0 == 1:
        return iso.sigT[0, ig]
    log_sig0 = np.clip(np.log10(sig0_val), 0.0, 10.0)
    # sig0 is in decreasing order; np.interp requires increasing xp
    xp = np.log10(iso.sig0)[::-1]
    fp = iso.sigT[::-1, ig]
    return float(np.interp(log_sig0, xp, fp))
