"""Contamination analysis for curvilinear SN diamond difference.

Implements the asymptotic diffusion-limit analysis from
Bailey, Morel & Chang (2009) to evaluate the contamination factor β
for spherical and cylindrical geometries.

When β ≠ 0, the DD scheme satisfies a *contaminated* diffusion equation
near r = 0, producing the Morel–Montry flux dip.  The geometry-weighted
balance equation (ΔA/w factor + correct α recursion) should reduce β
toward zero.  Implementing Morel–Montry (M-M) angular weights
(Bailey Eq. 74) forces β = 0 exactly.

Usage::

    from derivations.sn_contamination import contamination_beta, morel_montry_weights
    beta = contamination_beta(quad, geometry="spherical")
    tau = morel_montry_weights(quad, geometry="spherical")
"""

from __future__ import annotations

import numpy as np


def _alpha_dome(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute α = cumsum(−w·μ), the non-negative dome coefficients.

    Parameters
    ----------
    mu : (M,) direction cosines (η for cylindrical, μ for spherical),
         sorted in increasing order.
    w : (M,) quadrature weights.

    Returns
    -------
    alpha : (M+1,) dome coefficients with α[0] = 0, α[M] ≈ 0.
    """
    M = len(mu)
    alpha = np.zeros(M + 1)
    for m in range(M):
        alpha[m + 1] = alpha[m] - w[m] * mu[m]
    return alpha


def _cell_edge_cosines(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute cell-edge direction cosines μ_{m+1/2}.

    Bailey et al. (2009) Eq. 52: μ_{m+1/2} = μ_{m-1/2} + w̃_m,
    with μ_{1/2} = −√(1 − ξ²) and μ_{M+1/2} = +√(1 − ξ²) for
    cylindrical, or μ_{1/2} = −1 and μ_{M+1/2} = +1 for spherical.

    For equally-weighted ordinates within a level, w̃_m = w_m.
    """
    M = len(mu)
    mu_edge = np.zeros(M + 1)
    mu_edge[0] = mu[0] - w[0] / 2  # approximate left edge
    for m in range(M):
        mu_edge[m + 1] = mu_edge[m] + w[m]
    return mu_edge


def contamination_beta(
    quad,
    geometry: str = "spherical",
) -> float | np.ndarray:
    """Compute the Bailey et al. contamination factor β.

    For **spherical** (Bailey Eq. 41)::

        β = ½ Σ_m μ_m [α_{m+1/2}·μ_{m+1/2} − α_{m-1/2}·μ_{m-1/2}]

    For **cylindrical** (Bailey Eq. 75), returns one β per μ-level::

        β_n = Σ_m η_m [α_{m+1/2,n}·η_{m+1/2,n} − α_{m-1/2,n}·η_{m-1/2,n}]

    β = 0 implies diffusion-limit consistency (no flux dip).

    Parameters
    ----------
    quad : quadrature object with mu_x, weights, and optionally level_indices.
    geometry : "spherical" or "cylindrical".

    Returns
    -------
    beta : float for spherical, ndarray of shape (n_levels,) for cylindrical.
    """
    if geometry == "spherical":
        mu = quad.mu_x
        w = quad.weights
        alpha = _alpha_dome(mu, w)
        mu_edge = _cell_edge_cosines(mu, w)

        beta = 0.0
        for m in range(len(mu)):
            beta += 0.5 * mu[m] * (
                alpha[m + 1] * mu_edge[m + 1] - alpha[m] * mu_edge[m]
            )
        return beta

    elif geometry == "cylindrical":
        betas = []
        for level_idx in quad.level_indices:
            eta = quad.mu_x[level_idx]
            w = quad.weights[level_idx]
            alpha = _alpha_dome(eta, w)
            eta_edge = _cell_edge_cosines(eta, w)

            beta = 0.0
            for m in range(len(eta)):
                beta += eta[m] * (
                    alpha[m + 1] * eta_edge[m + 1] - alpha[m] * eta_edge[m]
                )
            betas.append(beta)
        return np.array(betas)

    else:
        raise ValueError(f"Unknown geometry: {geometry!r}")


def morel_montry_weights(
    quad,
    geometry: str = "spherical",
) -> np.ndarray | list[np.ndarray]:
    """Compute Morel–Montry angular closure weights τ_m.

    Bailey et al. (2009) Eq. 74::

        τ_m = (μ_m − μ_{m-1/2}) / (μ_{m+1/2} − μ_{m-1/2})

    Standard DD has τ = 0.5; step has τ = 1.0.

    Parameters
    ----------
    quad : quadrature object.
    geometry : "spherical" or "cylindrical".

    Returns
    -------
    tau : (N,) for spherical, list of (M,) arrays for cylindrical.
    """
    if geometry == "spherical":
        mu = quad.mu_x
        w = quad.weights
        mu_edge = _cell_edge_cosines(mu, w)
        tau = np.empty(len(mu))
        for m in range(len(mu)):
            denom = mu_edge[m + 1] - mu_edge[m]
            tau[m] = (mu[m] - mu_edge[m]) / denom if abs(denom) > 1e-15 else 0.5
        return tau

    elif geometry == "cylindrical":
        tau_list = []
        for level_idx in quad.level_indices:
            eta = quad.mu_x[level_idx]
            w = quad.weights[level_idx]
            eta_edge = _cell_edge_cosines(eta, w)
            tau = np.empty(len(eta))
            for m in range(len(eta)):
                denom = eta_edge[m + 1] - eta_edge[m]
                tau[m] = (eta[m] - eta_edge[m]) / denom if abs(denom) > 1e-15 else 0.5
            tau_list.append(tau)
        return tau_list

    else:
        raise ValueError(f"Unknown geometry: {geometry!r}")


if __name__ == "__main__":
    from sn_quadrature import GaussLegendre1D, ProductQuadrature, LevelSymmetricSN

    print("=== Contamination Analysis (Bailey et al. 2009) ===\n")

    # Spherical
    for N in [4, 8, 16]:
        q = GaussLegendre1D.create(N)
        beta = contamination_beta(q, "spherical")
        tau = morel_montry_weights(q, "spherical")
        print(f"Spherical GL-{N}: β = {beta:.6e}, "
              f"τ range = [{tau.min():.4f}, {tau.max():.4f}] (DD = 0.5)")

    print()

    # Cylindrical
    for n_phi in [8, 16]:
        q = ProductQuadrature.create(n_mu=4, n_phi=n_phi)
        betas = contamination_beta(q, "cylindrical")
        taus = morel_montry_weights(q, "cylindrical")
        tau_all = np.concatenate(taus)
        print(f"Cylindrical Product(4×{n_phi}): "
              f"β_max = {np.abs(betas).max():.6e}, "
              f"τ range = [{tau_all.min():.4f}, {tau_all.max():.4f}]")

    print()

    # LS
    for order in [4, 6]:
        q = LevelSymmetricSN.create(order)
        betas = contamination_beta(q, "cylindrical")
        taus = morel_montry_weights(q, "cylindrical")
        tau_all = np.concatenate(taus)
        print(f"Cylindrical LS-S{order}: "
              f"β_max = {np.abs(betas).max():.6e}, "
              f"τ range = [{tau_all.min():.4f}, {tau_all.max():.4f}]")
