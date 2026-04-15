"""Diamond-difference transport sweep for 1D and 2D Cartesian meshes.

Two paths, automatically dispatched:

- **1D cumprod**: for ny=1, mu_y=0 (Gauss-Legendre). Solves the
  recurrence via cumulative products — O(nc) numpy ops, ~ms.
- **2D wavefront**: for general 2D. Sweeps cells along anti-diagonals
  (i+j=const), vectorized within each diagonal.

Both paths use the precomputed streaming stencil from :class:`SNMesh`
to avoid redundant per-ordinate per-cell divisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .quadrature import AngularQuadrature

if TYPE_CHECKING:
    from .geometry import SNMesh


def transport_sweep(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: SNMesh,
    psi_bc: dict,
    Q_aniso: np.ndarray | None = None,
    boundary_condition: str = "reflective",
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one full diamond-difference transport sweep.

    Parameters
    ----------
    Q : (nx, ny, ng) isotropic source density.
    sig_t : (nx, ny, ng) total macroscopic cross section.
    sn_mesh : SNMesh — augmented geometry with precomputed stencil.
    psi_bc : mutable dict storing persistent boundary fluxes
        for reflective BCs between outer iterations.
    Q_aniso : (N, nx, ny, ng) per-ordinate anisotropic source (P1+
        scattering). None for isotropic-only (P0).
    boundary_condition : "reflective" (default) or "vacuum".
        Vacuum BC zeros the incoming angular flux at every external
        face — needed for fixed-source verification (MMS) where the
        manufactured solution vanishes at the slab edges.

    Returns
    -------
    angular_flux : (N, nx, ny, ng) angular flux per ordinate.
    scalar_flux : (nx, ny, ng) = Σ_n w_n ψ_n.
    """
    if boundary_condition not in ("reflective", "vacuum"):
        raise ValueError(
            f"boundary_condition must be 'reflective' or 'vacuum', "
            f"got {boundary_condition!r}"
        )

    quad = sn_mesh.quad
    ny = sn_mesh.ny

    if sn_mesh.curvature == "spherical":
        if boundary_condition != "reflective":
            raise NotImplementedError(
                "vacuum BC not yet supported for spherical sweep"
            )
        return _sweep_1d_spherical(Q, sig_t, sn_mesh, psi_bc)

    if sn_mesh.curvature == "cylindrical":
        if boundary_condition != "reflective":
            raise NotImplementedError(
                "vacuum BC not yet supported for cylindrical sweep"
            )
        return _sweep_1d_cylindrical(Q, sig_t, sn_mesh, psi_bc)

    is_gl_1d = (ny == 1 and np.all(np.abs(quad.mu_y) < 1e-15)
                 and Q_aniso is None
                 and boundary_condition == "reflective")

    if is_gl_1d:
        return _sweep_1d_cumprod(Q, sig_t, sn_mesh, psi_bc)
    else:
        return _sweep_2d_wavefront(
            Q, sig_t, sn_mesh, psi_bc, Q_aniso,
            boundary_condition=boundary_condition,
        )


# ═══════════════════════════════════════════════════════════════════════
# 1D cumprod path (fast, for GL quadrature on slab)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_1d_cumprod(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: SNMesh,
    psi_bc: dict,
    Q_aniso: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D sweep using cumulative products for the DD recurrence.

    Uses the precomputed streaming stencil from SNMesh:
    streaming_x[n, i] = 2|μ_x[n]| / dx[i].
    """
    dx = sn_mesh.dx
    nx = len(dx)
    ng = Q.shape[2]
    quad = sn_mesh.quad
    N = quad.N
    weights = quad.weights
    ref_x = quad.reflection_index("x")

    # Squeeze out the ny=1 dimension for 1D arrays
    Q_1d = Q[:, 0, :]        # (nx, ng)
    sig_t_1d = sig_t[:, 0, :]  # (nx, ng)

    # Precompute DD coefficients for positive directions
    n_half = N // 2
    mu_pos = np.abs(quad.mu_x[N // 2:])  # positive half
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
# 1D spherical path (cell-by-cell with angular redistribution)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_1d_spherical(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: SNMesh,
    psi_bc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Spherical 1-D sweep with geometry-weighted angular redistribution.

    Processes ordinates sequentially from most negative :math:`\mu` to
    most positive, applying angular redistribution via the :math:`\alpha`
    coefficients at each cell.

    The balance equation includes a geometry factor
    :math:`\Delta A_i / w_n` on the redistribution term
    (Bailey et al. 2009), ensuring per-ordinate flat-flux consistency:

    .. math::

        \psi_{n,i} = \frac{S_i V_i + |\mu_n|(A_{\rm in}+A_{\rm out})
            \psi^s_{\rm in} + \frac{\Delta A_i}{w_n}
            (\alpha_{n+\frac12}+\alpha_{n-\frac12})\psi_{n-\frac12}}
            {2|\mu_n| A^s_{\rm out}
            + 2\frac{\Delta A_i}{w_n}\alpha_{n+\frac12} + \Sigma_t V_i}

    The :math:`\alpha` coefficients are computed as
    :math:`\alpha_{n+1/2} = \alpha_{n-1/2} - w_n \mu_n` and form a
    non-negative dome for μ-sorted ordinates.
    """
    nx = sn_mesh.nx
    ng = Q.shape[2]
    quad = sn_mesh.quad
    N = quad.N
    mu = quad.mu_x
    weights = quad.weights
    ref = quad.reflection_index("x")

    Q_1d = Q[:, 0, :]          # (nx, ng)
    sig_t_1d = sig_t[:, 0, :]  # (nx, ng)

    A = sn_mesh.face_areas     # (nx+1,) surface areas at cell faces
    V = sn_mesh.volumes[:, 0]  # (nx,) cell volumes
    alpha = sn_mesh.alpha_half  # (N+1,) non-negative dome
    dAw = sn_mesh.redist_dAw   # (nx, N) precomputed ΔA_i/w_n
    tau = sn_mesh.tau_mm       # (N,) Morel–Montry angular weights

    # Persistent boundary flux at the outer face (per ordinate)
    if "bc_sph" not in psi_bc:
        psi_bc["bc_sph"] = np.zeros((N, ng))
    bc_outer = psi_bc["bc_sph"]

    # Angular "face flux" between successive ordinates: ψ_{n-1/2, i}
    # Shape (nx, ng). Initialised to zero for the first ordinate (α_{1/2}=0).
    psi_angle = np.zeros((nx, ng))

    angular_flux = np.zeros((N, nx, 1, ng))
    scalar_flux = np.zeros((nx, ng))

    # Isotropic source → angular source density by dividing by sum(w)
    # Then multiply by cell volume for the balance equation
    weight_norm = 1.0 / weights.sum()
    QV = Q_1d * V[:, None] * weight_norm  # (nx, ng)

    for n in range(N):
        mu_n = mu[n]
        abs_mu = abs(mu_n)
        w_n = weights[n]
        alpha_in = alpha[n]       # α_{n-1/2} ≥ 0 (dome)
        alpha_out = alpha[n + 1]  # α_{n+1/2} ≥ 0 (dome)
        tau_n = tau[n]            # M-M angular closure weight
        c_out = alpha_out / tau_n               # denom coefficient
        c_in = (1.0 - tau_n) / tau_n * alpha_out + alpha_in  # numer coefficient

        if mu_n < 0:
            # Inward sweep: outer boundary → centre
            psi_spatial_in = bc_outer[ref[n]].copy()

            for i in range(nx - 1, -1, -1):
                A_in = A[i + 1]   # incoming face (outer)
                A_out = A[i]      # outgoing face (inner)
                dA_w = dAw[i, n]  # precomputed geometry factor

                denom = (2.0 * abs_mu * A_out
                         + dA_w * c_out
                         + sig_t_1d[i] * V[i])
                numer = (QV[i]
                         + abs_mu * (A_in + A_out) * psi_spatial_in
                         + dA_w * c_in * psi_angle[i])

                psi = numer / denom

                # WDD closures
                psi_spatial_out = 2.0 * psi - psi_spatial_in
                psi_angle[i] = (psi - (1.0 - tau_n) * psi_angle[i]) / tau_n

                angular_flux[n, i, 0, :] = psi
                scalar_flux[i] += w_n * psi

                psi_spatial_in = psi_spatial_out

        else:
            # Outward sweep: centre → outer boundary
            # At r=0, A[0] = 4π(0)² = 0, so no spatial incoming flux
            psi_spatial_in = np.zeros(ng)

            for i in range(nx):
                A_in = A[i]       # incoming face (inner)
                A_out = A[i + 1]  # outgoing face (outer)
                dA_w = dAw[i, n]

                denom = (2.0 * abs_mu * A_out
                         + dA_w * c_out
                         + sig_t_1d[i] * V[i])
                numer = (QV[i]
                         + abs_mu * (A_in + A_out) * psi_spatial_in
                         + dA_w * c_in * psi_angle[i])

                psi = numer / denom

                # WDD closures
                psi_spatial_out = 2.0 * psi - psi_spatial_in
                psi_angle[i] = (psi - (1.0 - tau_n) * psi_angle[i]) / tau_n

                angular_flux[n, i, 0, :] = psi
                scalar_flux[i] += w_n * psi

                psi_spatial_in = psi_spatial_out

            # Store outgoing flux at outer boundary for reflective BC
            bc_outer[n] = psi_spatial_out

    return angular_flux, scalar_flux[:, None, :]  # restore ny=1 dim


# ═══════════════════════════════════════════════════════════════════════
# 1D cylindrical path (per-level azimuthal redistribution)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_1d_cylindrical(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: SNMesh,
    psi_bc: dict,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Cylindrical 1-D sweep with geometry-weighted azimuthal redistribution.

    For each μ-level *p*, processes azimuthal ordinates sequentially
    from most-inward (:math:`\eta = -\sin\theta`) to most-outward
    (:math:`\eta = +\sin\theta`), applying the redistribution
    :math:`\alpha_{p,m+1/2}` which couples successive azimuthal
    directions on that level.

    The balance equation includes a geometry factor
    :math:`\Delta A_i / w_m` on the redistribution term
    (Bailey et al. 2009), ensuring per-ordinate flat-flux consistency:

    .. math::

        \psi_{m,i} = \frac{S_i V_i + |\eta_m|(A_{\rm in}+A_{\rm out})
            \psi^s_{\rm in} + \frac{\Delta A_i}{w_m}
            (\alpha_{m+\frac12}+\alpha_{m-\frac12})\psi_{m-\frac12}}
            {2|\eta_m| A^s_{\rm out}
            + 2\frac{\Delta A_i}{w_m}\alpha_{m+\frac12} + \Sigma_t V_i}

    The :math:`\alpha` coefficients are computed from the radial
    direction cosine :math:`\eta` (Bailey et al. Eq. 50) and form a
    non-negative dome, so the denominator is unconditionally positive.
    """
    nx = sn_mesh.nx
    ng = Q.shape[2]
    quad = sn_mesh.quad
    N = quad.N
    weights = quad.weights
    ref = quad.reflection_index("x")

    Q_1d = Q[:, 0, :]          # (nx, ng)
    sig_t_1d = sig_t[:, 0, :]  # (nx, ng)

    A = sn_mesh.face_areas     # (nx+1,) = 2πr at edges
    V = sn_mesh.volumes[:, 0]  # (nx,) cell volumes

    # Persistent boundary flux at the outer face (per ordinate)
    if "bc_cyl" not in psi_bc:
        psi_bc["bc_cyl"] = np.zeros((N, ng))
    bc_outer = psi_bc["bc_cyl"]

    angular_flux = np.zeros((N, nx, 1, ng))
    scalar_flux = np.zeros((nx, ng))

    # Isotropic source → angular source density
    weight_norm = 1.0 / weights.sum()
    QV = Q_1d * V[:, None] * weight_norm  # (nx, ng)

    # Process each μ-level independently
    for p, level_idx in enumerate(quad.level_indices):
        alpha = sn_mesh.alpha_per_level[p]  # (M+1,) non-negative dome
        dAw = sn_mesh.redist_dAw_per_level[p]  # (nx, M) precomputed
        tau_level = sn_mesh.tau_mm_per_level[p]  # (M,) M-M weights
        M = len(level_idx)

        # Azimuthal "face flux" between successive ordinates on this level.
        # Initialised to zero: α_{1/2} = 0 so the product α·ψ vanishes.
        psi_angle = np.zeros((nx, ng))

        for m_local in range(M):
            n = level_idx[m_local]  # global ordinate index
            eta_n = quad.mu_x[n]    # radial direction cosine
            abs_eta = abs(eta_n)
            w_n = weights[n]
            alpha_in = alpha[m_local]       # α_{m-1/2} ≥ 0
            alpha_out = alpha[m_local + 1]  # α_{m+1/2} ≥ 0
            tau_m = tau_level[m_local]       # M-M angular closure weight
            c_out = alpha_out / tau_m
            c_in = (1.0 - tau_m) / tau_m * alpha_out + alpha_in

            if eta_n < 0:
                # Inward sweep: outer → centre
                psi_spatial_in = bc_outer[ref[n]].copy()

                for i in range(nx - 1, -1, -1):
                    A_in = A[i + 1]
                    A_out = A[i]
                    dA_w = dAw[i, m_local]  # precomputed geometry factor

                    denom = (2.0 * abs_eta * A_out
                             + dA_w * c_out
                             + sig_t_1d[i] * V[i])
                    numer = (QV[i]
                             + abs_eta * (A_in + A_out) * psi_spatial_in
                             + dA_w * c_in * psi_angle[i])

                    psi = numer / denom

                    psi_spatial_out = 2.0 * psi - psi_spatial_in
                    psi_angle[i] = (psi - (1.0 - tau_m) * psi_angle[i]) / tau_m

                    angular_flux[n, i, 0, :] = psi
                    scalar_flux[i] += w_n * psi

                    psi_spatial_in = psi_spatial_out

            elif abs_eta < 1e-15:
                # Pure azimuthal ordinate (η≈0): no radial streaming.
                for i in range(nx):
                    dA_w = dAw[i, m_local]

                    denom = dA_w * c_out + sig_t_1d[i] * V[i]
                    numer = (QV[i]
                             + dA_w * c_in * psi_angle[i])

                    psi = numer / denom
                    psi_angle[i] = (psi - (1.0 - tau_m) * psi_angle[i]) / tau_m

                    angular_flux[n, i, 0, :] = psi
                    scalar_flux[i] += w_n * psi

            else:
                # Outward sweep: centre → outer
                psi_spatial_in = np.zeros(ng)

                for i in range(nx):
                    A_in = A[i]
                    A_out = A[i + 1]
                    dA_w = dAw[i, m_local]

                    denom = (2.0 * abs_eta * A_out
                             + dA_w * c_out
                             + sig_t_1d[i] * V[i])
                    numer = (QV[i]
                             + abs_eta * (A_in + A_out) * psi_spatial_in
                             + dA_w * c_in * psi_angle[i])

                    psi = numer / denom

                    psi_spatial_out = 2.0 * psi - psi_spatial_in
                    psi_angle[i] = (psi - (1.0 - tau_m) * psi_angle[i]) / tau_m

                    angular_flux[n, i, 0, :] = psi
                    scalar_flux[i] += w_n * psi

                    psi_spatial_in = psi_spatial_out

                # Store outgoing at outer boundary for reflective BC
                bc_outer[n] = psi_spatial_out

    return angular_flux, scalar_flux[:, None, :]  # restore ny=1 dim


# ═══════════════════════════════════════════════════════════════════════
# 2D wavefront path (vectorized along anti-diagonals)
# ═══════════════════════════════════════════════════════════════════════

def _sweep_2d_wavefront(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: SNMesh,
    psi_bc: dict,
    Q_aniso: np.ndarray | None = None,
    boundary_condition: str = "reflective",
) -> tuple[np.ndarray, np.ndarray]:
    """2D sweep using wavefront parallelism along anti-diagonals.

    Uses the precomputed streaming stencil from SNMesh:
    streaming_x[n, i] = 2|μ_x[n]| / dx[i],
    streaming_y[n, j] = 2|μ_y[n]| / dy[j].
    """
    dx = sn_mesh.dx
    dy = sn_mesh.dy
    nx, ny, ng = Q.shape
    quad = sn_mesh.quad
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

    # Precompute diagonal indices per sweep direction (4 directions).
    _diag_cache: dict[tuple[int, int], tuple] = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            ix_arr = np.arange(nx) if sx >= 0 else np.arange(nx - 1, -1, -1)
            iy_arr = np.arange(ny) if sy >= 0 else np.arange(ny - 1, -1, -1)
            diags = []
            for k in range(nx + ny - 1):
                i_start = max(0, k - ny + 1)
                i_end = min(nx - 1, k)
                local_i = np.arange(i_start, i_end + 1)
                local_j = k - local_i
                diags.append((ix_arr[local_i], iy_arr[local_j]))
            _diag_cache[(sx, sy)] = (
                0 if sx >= 0 else 1,   # ix_in
                1 if sx >= 0 else 0,   # ix_out
                0 if sy >= 0 else 1,   # iy_in
                1 if sy >= 0 else 0,   # iy_out
                diags,
            )

    # Precompute scaled source (avoids recomputing per diagonal)
    Q_scaled = Q * weight_norm
    has_aniso = Q_aniso is not None
    if has_aniso:
        Q_aniso_scaled = Q_aniso * weight_norm  # (N, nx, ny, ng)

    # Precomputed streaming stencil
    str_x = sn_mesh.streaming_x  # (N_ord, nx)
    str_y = sn_mesh.streaming_y  # (N_ord, ny)

    for n in range(N):
        mx = mu_x[n]
        my = mu_y[n]
        w = weights[n]

        # Per-ordinate source: isotropic + anisotropic (if present)
        Q_n = Q_scaled
        if has_aniso:
            Q_n = Q_scaled + Q_aniso_scaled[n]  # (nx, ny, ng)

        if abs(mx) < 1e-15 and abs(my) < 1e-15:
            # Pure z-directed ordinate: no streaming in x or y.
            psi_avg = Q_n / sig_t  # (nx, ny, ng)
            angular_flux[n, :, :, :] = psi_avg
            scalar_flux += w * psi_avg
            continue

        # Look up precomputed diagonal indices for this sweep direction
        key = (1 if mx >= 0 else -1, 1 if my >= 0 else -1)
        ix_in, ix_out, iy_in, iy_out, diags = _diag_cache[key]

        # Boundary condition — vacuum leaves incoming faces at zero
        # (no reflection copy), which equals zero since those entries
        # are never written except by this copy (see sweep.py docs).
        if boundary_condition == "reflective":
            if mx >= 0:
                psi_x[n, 0, :, :] = psi_x[ref_x[n], 0, :, :]
            else:
                psi_x[n, nx, :, :] = psi_x[ref_x[n], nx, :, :]

            if my >= 0:
                psi_y[n, :, 0, :] = psi_y[ref_y[n], :, 0, :]
            else:
                psi_y[n, :, ny, :] = psi_y[ref_y[n], :, ny, :]

        # Precomputed streaming for this ordinate
        str_x_n = str_x[n]  # (nx,)
        str_y_n = str_y[n]  # (ny,)

        for ii, jj in diags:
            # Gather incoming face fluxes
            psi_in_x = psi_x[n, ii + ix_in, jj, :]   # (n_diag, ng)
            psi_in_y = psi_y[n, ii, jj + iy_in, :]    # (n_diag, ng)

            # Precomputed streaming coefficients for these cells
            sx_ii = str_x_n[ii, None]  # (n_diag, 1)
            sy_jj = str_y_n[jj, None]  # (n_diag, 1)

            # Diamond-difference equation using precomputed stencil
            denom = sig_t[ii, jj, :] + sx_ii + sy_jj

            psi_avg = (
                Q_n[ii, jj, :]
                + sx_ii * psi_in_x
                + sy_jj * psi_in_y
            ) / denom

            # Store outgoing face fluxes for next diagonal
            psi_x[n, ii + ix_out, jj, :] = 2.0 * psi_avg - psi_in_x
            psi_y[n, ii, jj + iy_out, :] = 2.0 * psi_avg - psi_in_y

            # Accumulate angular and scalar flux
            angular_flux[n, ii, jj, :] = psi_avg
            scalar_flux[ii, jj, :] += w * psi_avg

    return angular_flux, scalar_flux
