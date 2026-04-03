"""Diamond-difference transport sweep for 1D and 2D Cartesian meshes.

Two paths, automatically dispatched:
- **1D cumprod**: for ny=1, mu_y=0 (Gauss-Legendre). Solves the
  recurrence via cumulative products — O(nc) numpy ops, ~ms.
- **2D wavefront**: for general 2D. Sweeps cells along anti-diagonals
  (i+j=const), vectorized within each diagonal.

Both paths produce the same result: the scalar flux φ(x,y,g)
integrated from the angular flux over all quadrature directions.
"""

from __future__ import annotations

import numpy as np

from sn_quadrature import AngularQuadrature


def transport_sweep(
    Q: np.ndarray,
    sig_t: np.ndarray,
    mesh_dx: np.ndarray,
    mesh_dy: np.ndarray,
    quad: AngularQuadrature,
    psi_bc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one full diamond-difference transport sweep.

    Parameters
    ----------
    Q : (nx, ny, ng) isotropic source density.
    sig_t : (nx, ny, ng) total macroscopic cross section.
    mesh_dx : (nx,) cell widths in x.
    mesh_dy : (ny,) cell widths in y.
    quad : angular quadrature providing directions and weights.
    psi_bc : mutable dict storing persistent boundary fluxes
        for reflective BCs between outer iterations.

    Returns
    -------
    angular_flux : (N, nx, ny, ng) angular flux per ordinate.
    scalar_flux : (nx, ny, ng) = Σ_n w_n ψ_n.
    """
    ny = len(mesh_dy)
    is_1d = (ny == 1 and np.all(np.abs(quad.mu_y) < 1e-15))

    if is_1d:
        return _sweep_1d_cumprod(Q, sig_t, mesh_dx, quad, psi_bc)
    else:
        return _sweep_2d_wavefront(Q, sig_t, mesh_dx, mesh_dy, quad, psi_bc)


# ═══════════════════════════════════════════════════════════════════════
# 1D cumprod path (fast, for GL quadrature on slab)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_1d_cumprod(
    Q: np.ndarray,
    sig_t: np.ndarray,
    dx: np.ndarray,
    quad: AngularQuadrature,
    psi_bc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """1D sweep using cumulative products for the DD recurrence."""
    nx = len(dx)
    ng = Q.shape[2]
    N = quad.N
    mu_x = quad.mu_x
    weights = quad.weights
    ref_x = quad.reflection_index("x")

    # Squeeze out the ny=1 dimension for 1D arrays
    Q_1d = Q[:, 0, :]        # (nx, ng)
    sig_t_1d = sig_t[:, 0, :]  # (nx, ng)

    # Precompute DD coefficients for positive directions
    n_half = N // 2
    mu_pos = np.abs(mu_x[N // 2:])  # positive half
    w_pos = weights[N // 2:]

    # stream_coeff[n,i,g] = 2μ / (2μ + dx[i]·Σ_t[i,g])
    # source_coeff[n,i,g] = 0.5·dx[i] / (2μ + dx[i]·Σ_t[i,g])
    denom = 2.0 * mu_pos[:, None, None] + dx[None, :, None] * sig_t_1d[None, :, :]
    stream_coeff = 2.0 * mu_pos[:, None, None] / denom
    source_coeff = 0.5 * dx[None, :, None] / denom

    # Initialize boundary fluxes
    if "bc_1d" not in psi_bc:
        psi_bc["bc_1d"] = {
            "left": np.zeros((n_half, ng)),
            "right": np.zeros((n_half, ng)),
        }
    bc = psi_bc["bc_1d"]

    angular_flux = np.zeros((N, nx, 1, ng))
    phi = np.zeros((nx, ng))
    bQ = source_coeff * Q_1d[None, :, :]

    for n in range(n_half):
        a = stream_coeff[n]  # (nx, ng)
        s = bQ[n]            # (nx, ng)

        # Forward sweep (positive direction)
        psi_fwd = _solve_recurrence(a, s, bc["left"][n])
        bc["right"][n, :] = _outgoing(a, s, bc["left"][n])
        phi += w_pos[n] * psi_fwd
        angular_flux[n_half + n, :, 0, :] = psi_fwd  # store cell-avg angular flux

        # Backward sweep (negative direction via reversal)
        psi_bwd = _solve_recurrence(a[::-1], s[::-1], bc["right"][n])
        bc["left"][n, :] = _outgoing(a[::-1], s[::-1], bc["right"][n])
        phi += w_pos[n] * psi_bwd[::-1]
        angular_flux[n_half - 1 - n, :, 0, :] = psi_bwd[::-1]

    return angular_flux, phi[:, None, :]  # restore ny=1 dim


def _solve_recurrence(
    a: np.ndarray, s: np.ndarray, psi0: np.ndarray,
) -> np.ndarray:
    """Solve DD recurrence via cumulative products. Returns cell-average flux."""
    nc = a.shape[0]
    cp = np.cumprod(a, axis=0)
    cs = np.cumsum(s / cp, axis=0)

    psi_in = np.empty_like(a)
    psi_in[0] = psi0
    if nc > 1:
        psi_in[1:] = cp[:-1] * (psi0[None, :] + cs[:-1])

    psi_out = a * psi_in + s
    return 0.5 * (psi_in + psi_out)


def _outgoing(
    a: np.ndarray, s: np.ndarray, psi0: np.ndarray,
) -> np.ndarray:
    """Outgoing flux at the end of a forward sweep."""
    cp = np.cumprod(a, axis=0)
    cs = np.cumsum(s / cp, axis=0)
    return cp[-1] * (psi0 + cs[-1])


# ═══════════════════════════════════════════════════════════════════════
# 2D wavefront path (vectorized along anti-diagonals)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_2d_wavefront(
    Q: np.ndarray,
    sig_t: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    quad: AngularQuadrature,
    psi_bc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """2D sweep using wavefront parallelism along anti-diagonals."""
    nx, ny, ng = Q.shape
    N = quad.N
    mu_x = quad.mu_x
    mu_y = quad.mu_y
    weights = quad.weights
    ref_x = quad.reflection_index("x")
    ref_y = quad.reflection_index("y")

    angular_flux = np.zeros((N, nx, ny, ng))
    scalar_flux = np.zeros((nx, ny, ng))

    # Persistent boundary flux arrays for reflective BCs
    if "bc_2d_x" not in psi_bc:
        psi_bc["bc_2d_x"] = np.zeros((N, nx + 1, ny, ng))
        psi_bc["bc_2d_y"] = np.zeros((N, nx, ny + 1, ng))

    psi_x = psi_bc["bc_2d_x"]  # (N, nx+1, ny, ng) face fluxes in x
    psi_y = psi_bc["bc_2d_y"]  # (N, nx, ny+1, ng) face fluxes in y

    weight_norm = 1.0 / weights.sum()

    for n in range(N):
        mx = mu_x[n]
        my = mu_y[n]
        w = weights[n]

        if abs(mx) < 1e-15 and abs(my) < 1e-15:
            continue

        abs_mx = abs(mx)
        abs_my = abs(my)

        # Sweep direction depends on sign of mu
        if mx >= 0:
            ix_range = range(nx)
            ix_in, ix_out = 0, 1   # psi_x indexing offset
        else:
            ix_range = range(nx - 1, -1, -1)
            ix_in, ix_out = 1, 0

        if my >= 0:
            iy_range = range(ny)
            iy_in, iy_out = 0, 1
        else:
            iy_range = range(ny - 1, -1, -1)
            iy_in, iy_out = 1, 0

        # Reflective BC: incoming boundary flux from reflected partner
        if mx >= 0:
            psi_x[n, 0, :, :] = psi_x[ref_x[n], 0, :, :]
        else:
            psi_x[n, nx, :, :] = psi_x[ref_x[n], nx, :, :]

        if my >= 0:
            psi_y[n, :, 0, :] = psi_y[ref_y[n], :, 0, :]
        else:
            psi_y[n, :, ny, :] = psi_y[ref_y[n], :, ny, :]

        # Wavefront sweep: cells on anti-diagonal k are independent
        # For the sweep order determined by (sign(mx), sign(my)),
        # we process diagonals in the appropriate order.
        # Convert to local indices where sweep goes (0,0) → (nx-1,ny-1)
        ix_list = list(ix_range)
        iy_list = list(iy_range)

        for k in range(nx + ny - 1):
            # Cells on this anti-diagonal in local sweep coordinates
            i_start = max(0, k - ny + 1)
            i_end = min(nx - 1, k)
            local_i = np.arange(i_start, i_end + 1)
            local_j = k - local_i

            # Map to actual grid indices
            ii = np.array([ix_list[li] for li in local_i])
            jj = np.array([iy_list[lj] for lj in local_j])

            # Gather incoming face fluxes
            psi_in_x = psi_x[n, ii + ix_in, jj, :]   # (n_diag, ng)
            psi_in_y = psi_y[n, ii, jj + iy_in, :]    # (n_diag, ng)

            # Diamond-difference equation
            dx_ii = dx[ii, None]  # (n_diag, 1) for broadcasting
            dy_jj = dy[jj, None]

            denom = sig_t[ii, jj, :] + 2.0 * abs_mx / dx_ii + 2.0 * abs_my / dy_jj
            Q_scaled = Q[ii, jj, :] * weight_norm

            psi_avg = (
                Q_scaled
                + 2.0 * abs_mx * psi_in_x / dx_ii
                + 2.0 * abs_my * psi_in_y / dy_jj
            ) / denom

            # Store outgoing face fluxes for next diagonal
            psi_x[n, ii + ix_out, jj, :] = 2.0 * psi_avg - psi_in_x
            psi_y[n, ii, jj + iy_out, :] = 2.0 * psi_avg - psi_in_y

            # Accumulate angular and scalar flux
            angular_flux[n, ii, jj, :] = psi_avg
            scalar_flux[ii, jj, :] += w * psi_avg

    return angular_flux, scalar_flux
