"""Semi-analytical spherical collision probability eigenvalues.

Derives k_inf for {1,2,4} energy groups × {1,2,4} regions using the
exponential kernel with y-weighted quadrature.  The CP matrix is
computed numerically; the eigenvalue problem is a finite matrix solve.

Spherical CP kernel
-------------------
For concentric spherical shells, the transmission kernel along a chord
at impact parameter y is simply ``exp(-τ)`` — full 3-D symmetry removes
any residual angular integration (unlike slab E₃ or cylinder Ki₃).

The reduced collision probability integral uses:

.. math::

    \\text{rcp}_{ij} = 2 \\int_0^{r_{\\text{cell}}}
        \\Delta_2[F](\\tau_i, \\tau_j, \\text{gap}) \\; y \\, dy

where :math:`F(\\tau) = e^{-\\tau}` and :math:`\\Delta_2[F]` is the
second-difference formula, and the extra factor of *y* in the
quadrature weight comes from the spherical area element ``2πy dy``.

White boundary condition
~~~~~~~~~~~~~~~~~~~~~~~~
The cell surface is :math:`S = 4\\pi r_{\\text{cell}}^2`, and the
escape-to-re-entry closure is the same as for slab / cylinder:

.. math::

    P_{\\text{in},i} = \\Sigma_{t,i} V_i P_{\\text{out},i} / S_{\\text{cell}}
"""

from __future__ import annotations

import numpy as np

from ._eigenvalue import kinf_from_cp
from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture


# ═══════════════════════════════════════════════════════════════════════
# Spherical CP matrix from exponential kernel
# ═══════════════════════════════════════════════════════════════════════

def _chord_half_lengths(radii: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    """Half-chord lengths l_k(y) for each spherical shell.  Shape (N, n_y).

    Same formula as cylindrical (chord through concentric annuli):
    for shell k with inner radius r_in and outer radius r_out,
    l_k(y) = sqrt(r_out² - y²) - sqrt(r_in² - y²)  for y < r_in,
    l_k(y) = sqrt(r_out² - y²)                      for r_in ≤ y < r_out.
    """
    N = len(radii)
    chords = np.zeros((N, len(y_pts)))
    r_inner = np.zeros(N)
    r_inner[1:] = radii[:-1]
    y2 = y_pts**2

    for k in range(N):
        r_out, r_in = radii[k], r_inner[k]
        outer = np.sqrt(np.maximum(r_out**2 - y2, 0.0))
        if r_in > 0:
            inner = np.sqrt(np.maximum(r_in**2 - y2, 0.0))
            mask = y_pts < r_in
            chords[k, mask] = outer[mask] - inner[mask]
        mask_p = (y_pts >= r_in) & (y_pts < r_out)
        chords[k, mask_p] = outer[mask_p]

    return chords


def _sphere_cp_matrix(
    sig_t_all: np.ndarray,
    radii: np.ndarray,
    volumes: np.ndarray,
    r_cell: float,
    n_quad_y: int = 64,
) -> np.ndarray:
    """Compute the infinite-lattice CP matrix for a spherical cell.

    Uses F(τ) = exp(-τ) as the kernel, with y-weighted Gauss-Legendre
    quadrature over the impact parameter.

    Returns P_inf : (N_reg, N_reg, ng).
    """
    N_reg = len(radii)
    ng = sig_t_all.shape[1]

    # Composite Gauss-Legendre quadrature over shell breakpoints
    gl_pts, gl_wts = np.polynomial.legendre.leggauss(n_quad_y)
    breakpoints = np.concatenate(([0.0], radii))
    y_all, w_all = [], []
    for seg in range(len(breakpoints) - 1):
        a, b = breakpoints[seg], breakpoints[seg + 1]
        y_all.append(0.5 * (b - a) * gl_pts + 0.5 * (b + a))
        w_all.append(0.5 * (b - a) * gl_wts)
    y_pts = np.concatenate(y_all)
    y_wts = np.concatenate(w_all)

    # Spherical weight: extra factor of y
    y_wts = y_wts * y_pts

    chords = _chord_half_lengths(radii, y_pts)
    n_y = len(y_pts)

    # Kernel F(τ) = exp(-τ), F(0) = 1
    def kernel(tau):
        return np.exp(-tau)
    kernel_zero = np.ones(n_y)

    P_inf_g = np.empty((N_reg, N_reg, ng))

    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau = sig_t_g[:, None] * chords

        bnd_pos = np.zeros((N_reg + 1, n_y))
        for k in range(N_reg):
            bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

        rcp = np.zeros((N_reg, N_reg))

        for i in range(N_reg):
            tau_i = tau[i, :]
            sti = sig_t_g[i]
            if sti == 0:
                continue

            # Self-same collision
            self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
                kernel_zero - kernel(tau_i)
            )
            rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

            for j in range(N_reg):
                tau_j = tau[j, :]
                if j > i:
                    gap_d = np.maximum(
                        bnd_pos[j, :] - bnd_pos[i + 1, :], 0.0)
                elif j < i:
                    gap_d = np.maximum(
                        bnd_pos[i, :] - bnd_pos[j + 1, :], 0.0)
                else:
                    gap_d = None

                if gap_d is not None:
                    dd = (kernel(gap_d)
                          - kernel(gap_d + tau_i)
                          - kernel(gap_d + tau_j)
                          + kernel(gap_d + tau_i + tau_j))
                else:
                    dd = np.zeros(n_y)

                gap_c = bnd_pos[i, :] + bnd_pos[j, :]
                dc = (kernel(gap_c)
                      - kernel(gap_c + tau_i)
                      - kernel(gap_c + tau_j)
                      + kernel(gap_c + tau_i + tau_j))

                rcp[i, j] += 2.0 * np.dot(y_wts, dd + dc)

        # Normalize: P_cell = rcp / (Σ_t · V)
        P_cell = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            if sig_t_g[i] * volumes[i] > 0:
                P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * volumes[i])

        # White boundary condition
        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        S_cell = 4.0 * np.pi * r_cell**2
        P_in = sig_t_g * volumes * P_out / S_cell
        P_inout = max(1.0 - P_in.sum(), 0.0)

        P_inf = P_cell.copy()
        if P_inout < 1.0:
            P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)
        P_inf_g[:, :, g] = P_inf

    return P_inf_g


# ═══════════════════════════════════════════════════════════════════════
# Spherical geometry parameters
# ═══════════════════════════════════════════════════════════════════════

# Radii for each region count (innermost first)
_RADII = {
    1: [1.0],
    2: [0.5, 1.0],
    4: [0.4, 0.45, 0.55, 1.0],
}

# Material IDs per region count (innermost = highest)
_MAT_IDS = {
    1: [2],
    2: [2, 0],
    4: [2, 3, 1, 0],
}


# ═══════════════════════════════════════════════════════════════════════
# Case generation
# ═══════════════════════════════════════════════════════════════════════

def _build_case(ng_key: str, n_regions: int) -> VerificationCase:
    """Build a spherical CP verification case."""
    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    radii = np.array(_RADII[n_regions])

    # Spherical shell volumes: V = 4π/3 (r_out³ - r_in³)
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = (4.0 / 3.0) * np.pi * (radii**3 - r_inner**3)

    r_cell = radii[-1]

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t_all = np.vstack([xs["sig_t"] for xs in xs_list])

    P_inf_g = _sphere_cp_matrix(sig_t_all, radii, volumes, r_cell)

    k_inf = kinf_from_cp(
        P_inf_g=P_inf_g,
        sig_t_all=sig_t_all,
        V_arr=volumes,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=[xs["chi"] for xs in xs_list],
    )

    mat_ids = _MAT_IDS[n_regions]
    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    geom_params_out = dict(
        radii=radii.tolist(),
        mat_ids=mat_ids,
    )

    name = f"cp_sph1D_{ng}eg_{n_regions}rg"
    dim = n_regions * ng

    latex = (
        rf"Spherical CP eigenvalue with {ng} groups, {n_regions} regions, "
        r"white boundary condition. "
        rf"The exp(-\tau)-based CP matrix yields a {dim}\times{dim} "
        r"eigenvalue problem."
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {k_inf:.10f}"
    )

    return VerificationCase(
        name=name,
        k_inf=k_inf,
        method="cp",
        geometry="sph1D",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=geom_params_out,
        latex=latex,
        description=f"{ng}G {n_regions}-region spherical CP (exp kernel, white BC)",
        tolerance="< 1e-5",
    )


def all_cases() -> list[VerificationCase]:
    """Return all spherical CP verification cases: {1,2,4}eg × {1,2,4}rg."""
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [1, 2, 4]:
            cases.append(_build_case(ng_key, n_regions))
    return cases
