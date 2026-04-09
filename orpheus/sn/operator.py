"""Direct transport operator for Krylov inner solves.

Provides the explicit operator T: ψ → T·ψ via finite differences, used
by the ``bicgstab`` inner solver path in :class:`SNSolver`.

Two geometries are supported:

* **Cartesian 2D** — ``T = μ_x ∂/∂x + μ_y ∂/∂y + Σ_t``
* **Spherical 1D** — ``T = μ (A ∂/∂r)/V + (α ∂/∂μ)/V + Σ_t``

The sweep-based solver (source iteration) inverts T implicitly via
diamond-difference sweeps.  This module forms T explicitly so that
scipy's Krylov solvers (BiCGSTAB, GMRES) can solve  T·ψ = b  directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .quadrature import AngularQuadrature


# ═══════════════════════════════════════════════════════════════════════
# Equation map: which (ordinate, cell) pairs are unknowns
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EquationMap:
    """Mapping between 1D solution vector and 4D angular flux."""

    n_eq: int               # number of angular unknowns (per group)
    n_unknowns: int         # n_eq * ng (total scalar unknowns)
    ordinate: np.ndarray    # (n_eq,) ordinate index for each unknown
    ix: np.ndarray          # (n_eq,) x-cell index
    iy: np.ndarray          # (n_eq,) y-cell index


def build_equation_map(
    nx: int, ny: int, quad: AngularQuadrature, ng: int,
) -> EquationMap:
    """Identify which (ordinate, cell) combos are unknowns.

    Filter: mu_z >= 0 (upper hemisphere), and NOT incoming at
    reflective boundaries (those are determined by reflection).
    """
    mu_x, mu_y = quad.mu_x, quad.mu_y
    # Need mu_z — for LebedevSphere it's available, for GL1D it's 0
    mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))

    ords, ixs, iys = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            for n in range(quad.N):
                if mu_z[n] < -1e-15:
                    continue
                if ix == 0 and mu_x[n] > 1e-15:
                    continue
                if ix == nx - 1 and mu_x[n] < -1e-15:
                    continue
                if iy == 0 and mu_y[n] > 1e-15:
                    continue
                if iy == ny - 1 and mu_y[n] < -1e-15:
                    continue
                ords.append(n)
                ixs.append(ix)
                iys.append(iy)

    n_eq = len(ords)
    return EquationMap(
        n_eq=n_eq,
        n_unknowns=n_eq * ng,
        ordinate=np.array(ords, dtype=int),
        ix=np.array(ixs, dtype=int),
        iy=np.array(iys, dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════════
# Solution ↔ angular flux conversion
# ═══════════════════════════════════════════════════════════════════════

def solution_to_angular_flux(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    nx: int, ny: int, ng: int,
) -> np.ndarray:
    """Convert 1D solution vector to 4D angular flux (ng, N, nx, ny).

    Applies z-reflection and reflective BCs to fill the full array
    from the reduced set of unknowns.
    """
    mu_x, mu_y = quad.mu_x, quad.mu_y
    mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))
    ref_x = quad.reflection_index("x")
    ref_y = quad.reflection_index("y")
    # z-reflection: need ref_z for Lebedev
    ref_z = getattr(quad, '_ref_z', np.arange(quad.N))

    fi = np.zeros((ng, quad.N, nx, ny))

    # Scatter solution into fi
    flux = solution.reshape(ng, eq_map.n_eq, order='F')
    for k in range(eq_map.n_eq):
        fi[:, eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]] = flux[:, k]

    # Z-reflection
    for n in range(quad.N):
        if mu_z[n] < -1e-15:
            fi[:, n, :, :] = fi[:, ref_z[n], :, :]

    # X reflective BCs
    for n in range(quad.N):
        if mu_x[n] > 1e-15:
            fi[:, n, 0, :] = fi[:, ref_x[n], 0, :]
        if mu_x[n] < -1e-15:
            fi[:, n, -1, :] = fi[:, ref_x[n], -1, :]

    # Y reflective BCs
    for n in range(quad.N):
        if mu_y[n] > 1e-15:
            fi[:, n, :, 0] = fi[:, ref_y[n], :, 0]
        if mu_y[n] < -1e-15:
            fi[:, n, :, -1] = fi[:, ref_y[n], :, -1]

    return fi


def angular_flux_to_scalar(
    fi: np.ndarray, quad: AngularQuadrature, nx: int, ny: int, ng: int,
) -> np.ndarray:
    """Integrate angular flux to scalar flux: φ = Σ w_n ψ_n."""
    sf = np.zeros((nx, ny, ng))
    for iy in range(ny):
        for ix in range(nx):
            sf[ix, iy, :] = np.sum(
                fi[:, :, ix, iy] * quad.weights[None, :], axis=1,
            )
    return sf


# ═══════════════════════════════════════════════════════════════════════
# Finite-difference gradients (diamond scheme with reflective BCs)
# ═══════════════════════════════════════════════════════════════════════

def _compute_gradients(
    fi: np.ndarray,
    n: int, ix: int, iy: int,
    quad: AngularQuadrature,
    nx: int, ny: int,
    dx: np.ndarray, dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diamond-scheme gradients with reflective BCs.

    Returns (dfi/dx, dfi/dy), each shape (ng,).
    """
    ref_x = quad.reflection_index("x")
    ref_y = quad.reflection_index("y")
    mu_x, mu_y = quad.mu_x, quad.mu_y

    # X gradient
    if mu_x[n] > 1e-15:
        if ix == 0:
            dfix = fi[:, ref_x[n], ix, iy] - fi[:, ref_x[n], ix + 1, iy]
        else:
            dfix = fi[:, n, ix, iy] - fi[:, n, ix - 1, iy]
    elif mu_x[n] < -1e-15:
        if ix == nx - 1:
            dfix = fi[:, ref_x[n], ix - 1, iy] - fi[:, ref_x[n], ix, iy]
        else:
            dfix = fi[:, n, ix + 1, iy] - fi[:, n, ix, iy]
    else:
        dfix = np.zeros(fi.shape[0])

    # Y gradient
    if mu_y[n] > 1e-15:
        if iy == 0:
            dfiy = fi[:, ref_y[n], ix, iy] - fi[:, ref_y[n], ix, iy + 1]
        else:
            dfiy = fi[:, n, ix, iy] - fi[:, n, ix, iy - 1]
    elif mu_y[n] < -1e-15:
        if iy == ny - 1:
            dfiy = fi[:, ref_y[n], ix, iy - 1] - fi[:, ref_y[n], ix, iy]
        else:
            dfiy = fi[:, n, ix, iy + 1] - fi[:, n, ix, iy]
    else:
        dfiy = np.zeros(fi.shape[0])

    return dfix / dx[ix], dfiy / dy[iy]


# ═══════════════════════════════════════════════════════════════════════
# Transport operator  T: ψ → μ·∇ψ + Σ_t·ψ
# ═══════════════════════════════════════════════════════════════════════

def transport_operator_matvec(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ny: int, ng: int,
    dx: np.ndarray, dy: np.ndarray,
) -> np.ndarray:
    """Apply the streaming + collision operator T·ψ.

    Parameters
    ----------
    solution : (n_unknowns,) flattened angular flux vector.
    sig_t : (nx, ny, ng) total cross section.

    Returns
    -------
    (n_unknowns,) result of T applied to the angular flux.
    """
    fi = solution_to_angular_flux(solution, eq_map, quad, nx, ny, ng)

    lhs = np.empty((ng, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n, ix, iy = eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]
        dfidx, dfidy = _compute_gradients(fi, n, ix, iy, quad, nx, ny, dx, dy)
        lhs[:, k] = (
            quad.mu_x[n] * dfidx
            + quad.mu_y[n] * dfidy
            + sig_t[ix, iy, :] * fi[:, n, ix, iy]
        )

    return lhs.ravel(order='F')


def build_transport_linear_operator(
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ny: int, ng: int,
    dx: np.ndarray, dy: np.ndarray,
) -> LinearOperator:
    """Build scipy LinearOperator for T = μ·∇ + Σ_t."""
    def matvec(x):
        return transport_operator_matvec(
            x, eq_map, quad, sig_t, nx, ny, ng, dx, dy,
        )

    n = eq_map.n_unknowns
    return LinearOperator((n, n), matvec=matvec, dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# RHS construction (fission + scattering + n2n, per ordinate)
# ═══════════════════════════════════════════════════════════════════════

def build_rhs(
    fission_source: np.ndarray,
    scalar_flux: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_s: dict[int, list[np.ndarray]],
    sig2: dict[int, np.ndarray],
    mat_map: np.ndarray,
    nx: int, ny: int, ng: int,
    scattering_order: int = 0,
    angular_flux: np.ndarray | None = None,
) -> np.ndarray:
    """Build the RHS source vector for T·ψ = b.

    All isotropic sources are divided by sum(weights) — the angular
    normalization for the discrete angular flux equation.

    For Pn scattering (scattering_order > 0), the scattering source
    is per-ordinate using Legendre moments of the angular flux:
        qS(n) = Σ_l (2l+1) · Σ_s^l^T @ [Σ_m fiL_lm · Y_lm(n)] / sum_w

    Parameters
    ----------
    fission_source : (nx, ny, ng) — already divided by sum(w) by the caller.
    scalar_flux : (nx, ny, ng) — current scalar flux for scattering.
    sig_s : dict[mat_id → list of (ng, ng)] Legendre scattering matrices.
    sig2 : dict[mat_id → (ng, ng)] (n,2n) matrices.
    scattering_order : Legendre order L (0 = P0 isotropic).
    angular_flux : (ng, N, nx, ny) angular flux for computing Legendre
        moments. Required if scattering_order > 0.

    Returns
    -------
    (n_unknowns,) RHS vector.
    """
    sum_w = float(quad.weights.sum())
    L = scattering_order
    mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))

    # Precompute Legendre moments if anisotropic scattering
    fiL = None
    Y = None
    if L > 0 and angular_flux is not None:
        Y = quad.spherical_harmonics(L)  # (N, L+1, 2L+1)
        w = quad.weights
        fiL = np.zeros((nx, ny, ng, L + 1, 2 * L + 1))
        for l in range(L + 1):
            for m in range(-l, l + 1):
                for n in range(quad.N):
                    fiL[:, :, :, l, l + m] += (
                        w[n] * angular_flux[:, n, :, :].T * Y[n, l, l + m]
                    )

    rhs = np.zeros((ng, eq_map.n_eq))
    eq_idx = 0
    for iy in range(ny):
        for ix in range(nx):
            mid = int(mat_map[ix, iy])
            phi_cell = scalar_flux[ix, iy, :]

            # Fission (already normalized by caller)
            qF = fission_source[ix, iy, :]

            # (n,2n) — isotropic
            q2 = 2.0 * (sig2[mid].T @ phi_cell) / sum_w

            for n in range(quad.N):
                if mu_z[n] < -1e-15:
                    continue
                if ix == 0 and quad.mu_x[n] > 1e-15:
                    continue
                if ix == nx - 1 and quad.mu_x[n] < -1e-15:
                    continue
                if iy == 0 and quad.mu_y[n] > 1e-15:
                    continue
                if iy == ny - 1 and quad.mu_y[n] < -1e-15:
                    continue

                # Scattering source (Pn expansion)
                qS = np.zeros(ng)
                for l in range(L + 1):
                    if l == 0:
                        # P0: isotropic, use scalar flux
                        qS += sig_s[mid][0].T @ phi_cell / sum_w
                    elif fiL is not None:
                        # P1+: anisotropic, use Legendre moments
                        SUM = np.zeros(ng)
                        for m in range(-l, l + 1):
                            SUM += fiL[ix, iy, :, l, l + m] * Y[n, l, l + m]
                        qS += (2 * l + 1) * (sig_s[mid][l].T @ SUM) / sum_w

                rhs[:, eq_idx] = qF + q2 + qS
                eq_idx += 1

    return rhs.ravel(order='F')


# ═══════════════════════════════════════════════════════════════════════
# Spherical 1D operator: T = μ(A∂/∂r)/V + (α∂/∂μ)/V + Σ_t
# ═══════════════════════════════════════════════════════════════════════

def build_equation_map_spherical(
    nx: int, quad: AngularQuadrature, ng: int,
) -> EquationMap:
    """Equation map for spherical 1D: all (ordinate, cell) pairs except
    incoming directions at the outer reflective boundary."""
    mu_x = quad.mu_x
    ords, ixs, iys = [], [], []
    for ix in range(nx):
        for n in range(quad.N):
            # Skip incoming at outer boundary (reflective BC)
            if ix == nx - 1 and mu_x[n] < -1e-15:
                continue
            ords.append(n)
            ixs.append(ix)
            iys.append(0)  # always iy=0 for 1D

    n_eq = len(ords)
    return EquationMap(
        n_eq=n_eq,
        n_unknowns=n_eq * ng,
        ordinate=np.array(ords, dtype=int),
        ix=np.array(ixs, dtype=int),
        iy=np.array(iys, dtype=int),
    )


def solution_to_angular_flux_spherical(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    nx: int, ng: int,
) -> np.ndarray:
    """Convert 1D solution vector to angular flux array (ng, N, nx, 1).

    Applies reflective BC at the outer boundary.
    """
    ref_x = quad.reflection_index("x")
    fi = np.zeros((ng, quad.N, nx, 1))

    flux = solution.reshape(ng, eq_map.n_eq, order='F')
    for k in range(eq_map.n_eq):
        fi[:, eq_map.ordinate[k], eq_map.ix[k], 0] = flux[:, k]

    # Reflective BC at outer boundary: incoming (μ<0) = reflected partner
    for n in range(quad.N):
        if quad.mu_x[n] < -1e-15:
            fi[:, n, -1, 0] = fi[:, ref_x[n], -1, 0]

    return fi


def transport_operator_matvec_spherical(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ng: int,
    face_areas: np.ndarray,
    volumes: np.ndarray,
    alpha_half: np.ndarray,
    redist_dAw: np.ndarray,
    tau_mm: np.ndarray,
) -> np.ndarray:
    r"""Apply the spherical transport operator T·ψ.

    .. math::

        (T\psi)_{n,i} = \frac{\mu_n}{V_i}
          \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
        + \frac{\Delta A_i}{w_n V_i}
          \bigl[\alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]
        + \Sigma_t \psi_{n,i}

    The :math:`\Delta A / w` geometry factor (``redist_dAw``, precomputed
    in :class:`SNMesh`) ensures per-ordinate flat-flux consistency
    (Bailey et al. 2009).

    Face fluxes are approximated by arithmetic averages of cell-centre values.
    """
    fi = solution_to_angular_flux_spherical(solution, eq_map, quad, nx, ng)
    ref_x = quad.reflection_index("x")
    A = face_areas       # (nx+1,)
    V = volumes[:, 0]    # (nx,)
    dAw = redist_dAw     # (nx, N) precomputed ΔA_i/w_n
    alpha = alpha_half   # (N+1,) non-negative dome
    N = quad.N
    mu = quad.mu_x

    lhs = np.empty((ng, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n = eq_map.ordinate[k]
        i = eq_map.ix[k]
        psi_ni = fi[:, n, i, 0]

        # ── Spatial streaming: μ (A ∂ψ/∂r) / V ──────────────────────
        if i < nx - 1:
            psi_right = 0.5 * (fi[:, n, i, 0] + fi[:, n, i + 1, 0])
        else:
            if mu[n] > 1e-15:
                psi_right = fi[:, n, i, 0]
            else:
                psi_right = fi[:, ref_x[n], i, 0]

        if i > 0:
            psi_left = 0.5 * (fi[:, n, i - 1, 0] + fi[:, n, i, 0])
        else:
            psi_left = 0.0

        streaming = mu[n] * (A[i + 1] * psi_right - A[i] * psi_left) / V[i]

        # ── Angular redistribution: (ΔA/w) (α ∂ψ/∂μ) / V ──────────
        # Angular face flux uses M-M weighted interpolation (τ).
        dA_w = dAw[i, n]  # precomputed geometry factor
        tau_n = tau_mm[n]

        if n < N - 1:
            psi_angle_right = tau_n * fi[:, n + 1, i, 0] + (1.0 - tau_n) * fi[:, n, i, 0]
        else:
            psi_angle_right = fi[:, n, i, 0]

        if n > 0:
            psi_angle_left = tau_mm[n - 1] * fi[:, n, i, 0] + (1.0 - tau_mm[n - 1]) * fi[:, n - 1, i, 0]
        else:
            psi_angle_left = fi[:, n, i, 0]

        redistribution = dA_w * (alpha[n + 1] * psi_angle_right
                                 - alpha[n] * psi_angle_left) / V[i]

        # ── Collision ────────────────────────────────────────────────
        collision = sig_t[i, 0, :] * psi_ni

        lhs[:, k] = streaming + redistribution + collision

    return lhs.ravel(order='F')


def build_transport_linear_operator_spherical(
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ng: int,
    face_areas: np.ndarray,
    volumes: np.ndarray,
    alpha_half: np.ndarray,
    redist_dAw: np.ndarray,
    tau_mm: np.ndarray,
) -> LinearOperator:
    """Build scipy LinearOperator for spherical T."""
    def matvec(x):
        return transport_operator_matvec_spherical(
            x, eq_map, quad, sig_t, nx, ng,
            face_areas, volumes, alpha_half, redist_dAw, tau_mm,
        )

    n = eq_map.n_unknowns
    return LinearOperator((n, n), matvec=matvec, dtype=float)


def build_rhs_spherical(
    fission_source: np.ndarray,
    scalar_flux: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_s: dict[int, list[np.ndarray]],
    sig2: dict[int, np.ndarray],
    mat_map: np.ndarray,
    nx: int, ng: int,
    scattering_order: int = 0,
    angular_flux: np.ndarray | None = None,
) -> np.ndarray:
    """Build the RHS source vector for spherical T·ψ = b.

    Same structure as Cartesian ``build_rhs`` but with spherical
    equation map (no y-direction, no z-reflection filtering).
    """
    sum_w = float(quad.weights.sum())
    L = scattering_order

    rhs = np.zeros((ng, eq_map.n_eq))
    eq_idx = 0
    for ix in range(nx):
        mid = int(mat_map[ix, 0])
        phi_cell = scalar_flux[ix, 0, :]

        qF = fission_source[ix, 0, :]
        q2 = 2.0 * (sig2[mid].T @ phi_cell) / sum_w

        for n in range(quad.N):
            if ix == nx - 1 and quad.mu_x[n] < -1e-15:
                continue

            qS = sig_s[mid][0].T @ phi_cell / sum_w
            rhs[:, eq_idx] = qF + q2 + qS
            eq_idx += 1

    return rhs.ravel(order='F')


# ═══════════════════════════════════════════════════════════════════════
# Cylindrical 1D operator: T = η(A∂/∂r)/V + (ΔA/w)(α∂/∂φ)/V + Σ_t
# ═══════════════════════════════════════════════════════════════════════

# Equation map and solution-to-flux reuse the spherical versions
# (both are 1D with reflective BC at the outer boundary).
build_equation_map_cylindrical = build_equation_map_spherical
solution_to_angular_flux_cylindrical = solution_to_angular_flux_spherical


def transport_operator_matvec_cylindrical(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ng: int,
    face_areas: np.ndarray,
    volumes: np.ndarray,
    alpha_per_level: list[np.ndarray],
    redist_dAw_per_level: list[np.ndarray],
    tau_mm_per_level: list[np.ndarray],
) -> np.ndarray:
    r"""Apply the cylindrical transport operator T·ψ.

    Per-level azimuthal redistribution with geometry-weighted
    :math:`\Delta A / w` factor and Morel–Montry angular closure.
    """
    fi = solution_to_angular_flux_cylindrical(solution, eq_map, quad, nx, ng)
    ref_x = quad.reflection_index("x")
    A = face_areas       # (nx+1,)
    V = volumes[:, 0]    # (nx,)
    N = quad.N
    mu = quad.mu_x

    # Build reverse map: global ordinate → (level, local index)
    ord_to_level = np.empty(N, dtype=int)
    ord_to_local = np.empty(N, dtype=int)
    for p, level_idx in enumerate(quad.level_indices):
        for m_local, n in enumerate(level_idx):
            ord_to_level[n] = p
            ord_to_local[n] = m_local

    lhs = np.empty((ng, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n = eq_map.ordinate[k]
        i = eq_map.ix[k]
        psi_ni = fi[:, n, i, 0]

        p = ord_to_level[n]
        m_local = ord_to_local[n]
        alpha = alpha_per_level[p]
        dAw = redist_dAw_per_level[p]
        tau_level = tau_mm_per_level[p]
        level_idx = quad.level_indices[p]
        M = len(level_idx)

        # ── Spatial streaming: η (A ∂ψ/∂r) / V ─────────────────────
        if i < nx - 1:
            psi_right = 0.5 * (fi[:, n, i, 0] + fi[:, n, i + 1, 0])
        else:
            if mu[n] > 1e-15:
                psi_right = fi[:, n, i, 0]
            else:
                psi_right = fi[:, ref_x[n], i, 0]

        if i > 0:
            psi_left = 0.5 * (fi[:, n, i - 1, 0] + fi[:, n, i, 0])
        else:
            psi_left = 0.0

        streaming = mu[n] * (A[i + 1] * psi_right - A[i] * psi_left) / V[i]

        # ── Angular redistribution: (ΔA/w)(α ∂ψ/∂φ) / V ───────────
        dA_w = dAw[i, m_local]
        tau_m = tau_level[m_local]

        if m_local < M - 1:
            n_next = level_idx[m_local + 1]
            psi_angle_right = tau_m * fi[:, n_next, i, 0] + (1.0 - tau_m) * fi[:, n, i, 0]
        else:
            psi_angle_right = fi[:, n, i, 0]

        if m_local > 0:
            n_prev = level_idx[m_local - 1]
            tau_prev = tau_level[m_local - 1]
            psi_angle_left = tau_prev * fi[:, n, i, 0] + (1.0 - tau_prev) * fi[:, n_prev, i, 0]
        else:
            psi_angle_left = fi[:, n, i, 0]

        redistribution = dA_w * (alpha[m_local + 1] * psi_angle_right
                                 - alpha[m_local] * psi_angle_left) / V[i]

        # ── Collision ────────────────────────────────────────────────
        collision = sig_t[i, 0, :] * psi_ni

        lhs[:, k] = streaming + redistribution + collision

    return lhs.ravel(order='F')


def build_transport_linear_operator_cylindrical(
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ng: int,
    face_areas: np.ndarray,
    volumes: np.ndarray,
    alpha_per_level: list[np.ndarray],
    redist_dAw_per_level: list[np.ndarray],
    tau_mm_per_level: list[np.ndarray],
) -> LinearOperator:
    """Build scipy LinearOperator for cylindrical T."""
    def matvec(x):
        return transport_operator_matvec_cylindrical(
            x, eq_map, quad, sig_t, nx, ng,
            face_areas, volumes,
            alpha_per_level, redist_dAw_per_level, tau_mm_per_level,
        )

    n = eq_map.n_unknowns
    return LinearOperator((n, n), matvec=matvec, dtype=float)


# RHS builder reuses the spherical version (same 1D isotropic structure).
build_rhs_cylindrical = build_rhs_spherical
