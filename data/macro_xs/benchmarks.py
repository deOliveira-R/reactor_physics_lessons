"""Synthetic cross-section benchmarks with known analytical solutions.

Each benchmark returns Mixture object(s) and the analytical k_inf,
enabling formal verification of transport solvers at arbitrary group
counts (1, 2, 4, …).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from .mixture import Mixture


def _make_mixture(
    sig_t: np.ndarray,
    sig_c: np.ndarray,
    sig_f: np.ndarray,
    nu: np.ndarray,
    chi: np.ndarray,
    sig_s: np.ndarray,
    eg: np.ndarray | None = None,
) -> Mixture:
    """Build a Mixture from N-group arrays.

    Parameters
    ----------
    sig_t : (ng,)  total macroscopic XS
    sig_c : (ng,)  capture (non-fission absorption)
    sig_f : (ng,)  fission XS
    nu    : (ng,)  average neutrons per fission
    chi   : (ng,)  fission spectrum (should sum to 1)
    sig_s : (ng, ng)  scattering matrix, sig_s[g'→g] = sig_s[g', g]
    eg    : (ng+1,) energy group boundaries in eV (optional, dummy if None)
    """
    ng = len(sig_t)
    if eg is None:
        eg = np.logspace(7, -3, ng + 1)  # 10 MeV to 1 meV

    SigS = [csr_matrix(sig_s)]
    Sig2 = csr_matrix((ng, ng))

    return Mixture(
        SigC=sig_c.copy(),
        SigL=np.zeros(ng),
        SigF=sig_f.copy(),
        SigP=(nu * sig_f).copy(),
        SigT=sig_t.copy(),
        SigS=SigS,
        Sig2=Sig2,
        chi=chi.copy(),
        eg=eg.copy(),
    )


# ═══════════════════════════════════════════════════════════════════════
# Homogeneous benchmarks (infinite medium)
# ═══════════════════════════════════════════════════════════════════════

def benchmark_1g_homogeneous() -> tuple[Mixture, float]:
    """1-group infinite medium.

    XS:  Σ_t = 1.0, Σ_s = 0.5, Σ_c = 0.2, Σ_f = 0.3, ν = 2.5, χ = 1.0
    Analytical:  k_inf = ν·Σ_f / Σ_a = 2.5 × 0.3 / 0.5 = 1.5
    """
    sig_t = np.array([1.0])
    sig_f = np.array([0.3])
    sig_c = np.array([0.2])       # absorption - fission = 0.5 - 0.3
    nu = np.array([2.5])
    chi = np.array([1.0])
    sig_s = np.array([[0.5]])     # within-group scattering

    k_inf = float(nu[0] * sig_f[0] / (sig_c[0] + sig_f[0]))
    return _make_mixture(sig_t, sig_c, sig_f, nu, chi, sig_s), k_inf


def benchmark_2g_homogeneous() -> tuple[Mixture, float]:
    """2-group infinite medium (fast + thermal, downscatter only).

    Group 1 (fast):    Σ_t=0.50  Σ_c=0.01  Σ_f=0.01  ν=2.5  χ=1.0
                       Σ_s(1→1)=0.38  Σ_s(1→2)=0.10
    Group 2 (thermal): Σ_t=1.00  Σ_c=0.02  Σ_f=0.08  ν=2.5  χ=0.0
                       Σ_s(2→2)=0.90

    The analytical k_inf is the dominant eigenvalue of inv(A)·F where
    A = diag(Σ_t) − Σ_s  and  F = χ ⊗ (ν·Σ_f).
    """
    sig_t = np.array([0.50, 1.00])
    sig_c = np.array([0.01, 0.02])
    sig_f = np.array([0.01, 0.08])
    nu = np.array([2.50, 2.50])
    chi = np.array([1.00, 0.00])
    sig_s = np.array([
        [0.38, 0.10],    # row 0: from fast  → fast, fast  → thermal
        [0.00, 0.90],    # row 1: from therm → fast, therm → thermal
    ])

    # Analytical eigenvalue: A·φ = (1/k)·F·φ  →  k = eigenvalue of inv(A)·F
    A = np.diag(sig_t) - sig_s.T  # transport − inscatter
    F = np.outer(chi, nu * sig_f)  # fission production
    M = np.linalg.solve(A, F)
    k_inf = float(np.max(np.real(np.linalg.eigvals(M))))

    return _make_mixture(sig_t, sig_c, sig_f, nu, chi, sig_s), k_inf


def benchmark_4g_homogeneous() -> tuple[Mixture, float]:
    """4-group infinite medium (fast → epi → thermal1 → thermal2).

    Downscatter cascade with fission only in thermal groups.
    """
    sig_c = np.array([0.01, 0.02, 0.03, 0.05])
    sig_f = np.array([0.005, 0.01, 0.05, 0.10])
    nu = np.array([2.80, 2.60, 2.50, 2.45])
    chi = np.array([0.60, 0.35, 0.05, 0.00])

    # Scattering: downscatter only (upper triangle in g'→g convention)
    sig_s = np.array([
        [0.28, 0.08, 0.02, 0.005],
        [0.00, 0.40, 0.12, 0.06],
        [0.00, 0.00, 0.55, 0.22],
        [0.00, 0.00, 0.00, 0.90],
    ])

    # Derive sig_t from consistency: sig_t = sig_c + sig_f + row_sum(sig_s)
    sig_t = sig_c + sig_f + sig_s.sum(axis=1)

    # Verify XS consistency: sig_t = sig_c + sig_f + sum(sig_s, axis=1)
    sig_a = sig_c + sig_f
    sig_s_total = sig_s.sum(axis=1)
    assert np.allclose(sig_t, sig_a + sig_s_total), \
        f"XS inconsistency: sig_t={sig_t} != sig_a+sig_s={sig_a+sig_s_total}"

    A = np.diag(sig_t) - sig_s.T
    F = np.outer(chi, nu * sig_f)
    M = np.linalg.solve(A, F)
    k_inf = float(np.max(np.real(np.linalg.eigvals(M))))

    return _make_mixture(sig_t, sig_c, sig_f, nu, chi, sig_s), k_inf


# ═══════════════════════════════════════════════════════════════════════
# Shared eigenvalue solver for heterogeneous CP benchmarks
# ═══════════════════════════════════════════════════════════════════════

def _kinf_from_cp(
    P_inf_g: np.ndarray,
    sig_t_all: np.ndarray,
    V_arr: np.ndarray,
    sig_s_mats: list[np.ndarray],
    nu_sig_f_mats: list[np.ndarray],
    chi_mats: list[np.ndarray],
) -> float:
    """Compute analytical k_inf from the CP matrix.

    Solves the generalised eigenvalue problem:
        (diag(Σ_t·V) − P^T·diag(V·Σ_s))·φ = (1/k)·P^T·diag(V·χ·νΣ_f)·φ

    Parameters
    ----------
    P_inf_g : (N_reg, N_reg, ng) — infinite-lattice CP matrix per group
    sig_t_all : (N_reg, ng)
    V_arr : (N_reg,) — region volumes (areas for cylinder, thicknesses for slab)
    sig_s_mats : list of (ng, ng) scattering matrices per region
    nu_sig_f_mats : list of (ng,) arrays per region
    chi_mats : list of (ng,) arrays per region
    """
    N_reg = P_inf_g.shape[0]
    ng = P_inf_g.shape[2]
    dim = N_reg * ng

    A_mat = np.zeros((dim, dim))
    B_mat = np.zeros((dim, dim))

    for i_reg in range(N_reg):
        for g in range(ng):
            row = i_reg * ng + g
            A_mat[row, row] = sig_t_all[i_reg, g] * V_arr[i_reg]

            for j_reg in range(N_reg):
                for gp in range(ng):
                    col = j_reg * ng + gp
                    # P(birth_j → collision_i, group g)
                    Pji = P_inf_g[j_reg, i_reg, g]
                    A_mat[row, col] -= Pji * V_arr[j_reg] * sig_s_mats[j_reg][gp, g]
                    B_mat[row, col] += Pji * V_arr[j_reg] * chi_mats[j_reg][g] \
                        * nu_sig_f_mats[j_reg][gp]

    M = np.linalg.solve(A_mat, B_mat)
    return float(np.max(np.real(np.linalg.eigvals(M))))


# ═══════════════════════════════════════════════════════════════════════
# Heterogeneous slab benchmarks
# ═══════════════════════════════════════════════════════════════════════

def benchmark_1g_slab() -> tuple[dict[int, Mixture], dict, float]:
    """1-group, 2-region slab (fuel + moderator).

    Fuel (mat 2): Σ_t=1.0  Σ_c=0.2  Σ_f=0.3  ν=2.5  Σ_s=0.5  t=0.5 cm
    Mod  (mat 0): Σ_t=2.0  Σ_c=0.1  Σ_s=1.9  no fission       t=0.5 cm

    Returns (materials_dict, geometry_params, k_inf_analytical).
    The k_inf is computed from the CP matrix via E_3 functions.
    """
    from scipy.special import expn

    # Fuel
    fuel = _make_mixture(
        sig_t=np.array([1.0]), sig_c=np.array([0.2]),
        sig_f=np.array([0.3]), nu=np.array([2.5]),
        chi=np.array([1.0]), sig_s=np.array([[0.5]]),
    )
    # Moderator (no fission, strong scattering)
    mod = _make_mixture(
        sig_t=np.array([2.0]), sig_c=np.array([0.1]),
        sig_f=np.array([0.0]), nu=np.array([0.0]),
        chi=np.array([1.0]), sig_s=np.array([[1.9]]),
    )

    t_fuel, t_mod = 0.5, 0.5
    materials = {2: fuel, 0: mod}
    geom_params = dict(n_fuel=1, n_clad=0, n_cool=1,
                       fuel_half=t_fuel, clad_thick=0.0, cool_thick=t_mod)

    # Compute analytical k_inf from the 1-group 2-region CP matrix
    sig_t_arr = np.array([1.0, 2.0])
    t_arr = np.array([t_fuel, t_mod])
    tau = sig_t_arr * t_arr  # optical thicknesses
    N = 2

    # Build CP matrix using the E_3 slab formula
    bnd_pos = np.array([0.0, tau[0], tau[0] + tau[1]])

    def e3(x):
        return float(expn(3, max(x, 0.0)))

    rcp = np.zeros((N, N))
    for i in range(N):
        sti = sig_t_arr[i]
        tau_i = tau[i]

        # Self-same
        rcp[i, i] += 0.5 * sti * (2 * t_arr[i] - (2.0 / sti) * (0.5 - e3(tau_i)))

        for j in range(N):
            tau_j = tau[j]
            if j > i:
                gap_d = bnd_pos[j] - bnd_pos[i + 1]
            elif j < i:
                gap_d = bnd_pos[i] - bnd_pos[j + 1]
            else:
                gap_d = None

            if gap_d is not None:
                gap_d = max(gap_d, 0.0)
                dd = e3(gap_d) - e3(gap_d + tau_i) - e3(gap_d + tau_j) + e3(gap_d + tau_i + tau_j)
            else:
                dd = 0.0

            gap_c = bnd_pos[i] + bnd_pos[j]
            dc = e3(gap_c) - e3(gap_c + tau_i) - e3(gap_c + tau_j) + e3(gap_c + tau_i + tau_j)
            rcp[i, j] += 0.5 * (dd + dc)

    P_cell = np.zeros((N, N))
    for i in range(N):
        P_cell[i, :] = rcp[i, :] / (sig_t_arr[i] * t_arr[i])

    # White BC correction
    P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
    P_in = sig_t_arr * t_arr * P_out
    P_inout = max(1.0 - P_in.sum(), 0.0)
    P_inf = P_cell + np.outer(P_out, P_in) / (1.0 - P_inout)

    k_inf = _kinf_from_cp(
        P_inf_g=P_inf[:, :, np.newaxis],
        sig_t_all=sig_t_arr[:, np.newaxis],
        V_arr=t_arr,
        sig_s_mats=[np.array([[0.5]]), np.array([[1.9]])],
        nu_sig_f_mats=[np.array([2.5 * 0.3]), np.array([0.0])],
        chi_mats=[np.array([1.0]), np.array([1.0])],
    )
    return materials, geom_params, k_inf


def benchmark_2g_slab() -> tuple[dict[int, Mixture], dict, float]:
    """2-group, 2-region slab (fuel + moderator).

    Fuel (mat 2):
        Group 1 (fast):    Σ_t=0.50  Σ_c=0.01  Σ_f=0.01  ν=2.5
                           Σ_s(1→1)=0.38  Σ_s(1→2)=0.10  χ=1.0
        Group 2 (thermal): Σ_t=1.00  Σ_c=0.02  Σ_f=0.08  ν=2.5
                           Σ_s(2→2)=0.90  χ=0.0

    Mod (mat 0):
        Group 1: Σ_t=0.60  Σ_c=0.02  Σ_s(1→1)=0.40  Σ_s(1→2)=0.18  χ=1.0
        Group 2: Σ_t=2.00  Σ_c=0.05  Σ_s(2→2)=1.95   χ=0.0

    Geometry: t_fuel = 0.5 cm, t_mod = 0.5 cm
    """
    from scipy.special import expn

    fuel = _make_mixture(
        sig_t=np.array([0.50, 1.00]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.50, 2.50]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
    )
    mod = _make_mixture(
        sig_t=np.array([0.60, 2.00]),
        sig_c=np.array([0.02, 0.05]),
        sig_f=np.array([0.00, 0.00]),
        nu=np.array([0.00, 0.00]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.40, 0.18], [0.00, 1.95]]),
    )

    t_fuel, t_mod = 0.5, 0.5
    materials = {2: fuel, 0: mod}
    geom_params = dict(n_fuel=1, n_clad=0, n_cool=1,
                       fuel_half=t_fuel, clad_thick=0.0, cool_thick=t_mod)

    # Analytical k_inf from the 4×4 CP eigenvalue problem
    # (2 regions × 2 groups)
    ng = 2
    N_reg = 2
    sig_t_all = np.array([[0.50, 1.00], [0.60, 2.00]])  # [region, group]
    t_arr = np.array([t_fuel, t_mod])

    def e3(x):
        return float(expn(3, max(x, 0.0)))

    # Build P_inf for each group
    P_inf_g = np.zeros((N_reg, N_reg, ng))
    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau_g = sig_t_g * t_arr
        bnd_pos = np.array([0.0, tau_g[0], tau_g[0] + tau_g[1]])

        rcp = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            sti = sig_t_g[i]
            tau_i = tau_g[i]
            rcp[i, i] += 0.5 * sti * (2 * t_arr[i] - (2.0 / sti) * (0.5 - e3(tau_i)))
            for j in range(N_reg):
                tau_j = tau_g[j]
                if j > i:
                    gap_d = bnd_pos[j] - bnd_pos[i + 1]
                elif j < i:
                    gap_d = bnd_pos[i] - bnd_pos[j + 1]
                else:
                    gap_d = None
                if gap_d is not None:
                    dd = e3(max(gap_d, 0)) - e3(max(gap_d, 0) + tau_i) \
                         - e3(max(gap_d, 0) + tau_j) + e3(max(gap_d, 0) + tau_i + tau_j)
                else:
                    dd = 0.0
                gap_c = bnd_pos[i] + bnd_pos[j]
                dc = e3(gap_c) - e3(gap_c + tau_i) - e3(gap_c + tau_j) + e3(gap_c + tau_i + tau_j)
                rcp[i, j] += 0.5 * (dd + dc)

        P_cell = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * t_arr[i])

        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        P_in = sig_t_g * t_arr * P_out
        P_inout = max(1.0 - P_in.sum(), 0.0)
        P_inf_g[:, :, g] = P_cell + np.outer(P_out, P_in) / (1.0 - P_inout)

    k_inf = _kinf_from_cp(
        P_inf_g=P_inf_g,
        sig_t_all=sig_t_all,
        V_arr=t_arr,
        sig_s_mats=[
            np.array([[0.38, 0.10], [0.00, 0.90]]),  # fuel
            np.array([[0.40, 0.18], [0.00, 1.95]]),   # mod
        ],
        nu_sig_f_mats=[
            np.array([2.5 * 0.01, 2.5 * 0.08]),  # fuel
            np.array([0.0, 0.0]),                   # mod
        ],
        chi_mats=[
            np.array([1.0, 0.0]),  # fuel
            np.array([1.0, 0.0]),  # mod
        ],
    )
    return materials, geom_params, k_inf


# ═══════════════════════════════════════════════════════════════════════
# Heterogeneous cylindrical (Wigner-Seitz) benchmarks
# ═══════════════════════════════════════════════════════════════════════

def _build_cylindrical_pinf(
    sig_t_all: np.ndarray,
    r_fuel: float,
    r_cell: float,
    n_quad_y: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P_inf for a 2-region cylindrical cell (1 fuel + 1 coolant).

    Uses the solver's own Ki₄-based CP computation.
    Returns (P_inf shape (2, 2, ng), volumes shape (2,)).
    """
    import sys
    from pathlib import Path
    _cp_dir = str(Path(__file__).resolve().parent.parent / "09.Collision.Probability")
    if _cp_dir not in sys.path:
        sys.path.insert(0, _cp_dir)

    from collision_probability import (
        CPGeometry, _build_ki_tables, _chord_half_lengths, _compute_cp_group,
    )

    ng = sig_t_all.shape[1]
    geom = CPGeometry.default_pwr(
        n_fuel=1, n_clad=0, n_cool=1,
        r_fuel=r_fuel, r_clad=r_fuel,
        pitch=r_cell * np.sqrt(np.pi),
    )

    ki_x, _, ki4_v = _build_ki_tables(20000, 50.0)

    gl_pts, gl_wts = np.polynomial.legendre.leggauss(n_quad_y)
    breakpoints = np.concatenate(([0.0], geom.radii))
    y_all, w_all = [], []
    for seg in range(len(breakpoints) - 1):
        a, b = breakpoints[seg], breakpoints[seg + 1]
        y_all.append(0.5 * (b - a) * gl_pts + 0.5 * (b + a))
        w_all.append(0.5 * (b - a) * gl_wts)
    y_pts = np.concatenate(y_all)
    y_wts = np.concatenate(w_all)
    chords = _chord_half_lengths(geom.radii, y_pts)

    P_inf_g = np.empty((2, 2, ng))
    for g in range(ng):
        P_inf_g[:, :, g] = _compute_cp_group(
            sig_t_all[:, g], geom, chords, y_pts, y_wts, ki_x, ki4_v,
        )

    return P_inf_g, geom.volumes


def benchmark_1g_cylinder() -> tuple[dict[int, Mixture], dict, float]:
    """1-group, 2-region Wigner-Seitz cylinder (fuel + moderator).

    Same XS as the 1-group slab benchmark:
    Fuel (mat 2): Σ_t=1.0  Σ_c=0.2  Σ_f=0.3  ν=2.5  Σ_s=0.5
    Mod  (mat 0): Σ_t=2.0  Σ_c=0.1  Σ_s=1.9  no fission

    Geometry: r_fuel = 0.5 cm, r_cell = 1.0 cm
    """
    fuel = _make_mixture(
        sig_t=np.array([1.0]), sig_c=np.array([0.2]),
        sig_f=np.array([0.3]), nu=np.array([2.5]),
        chi=np.array([1.0]), sig_s=np.array([[0.5]]),
    )
    mod = _make_mixture(
        sig_t=np.array([2.0]), sig_c=np.array([0.1]),
        sig_f=np.array([0.0]), nu=np.array([0.0]),
        chi=np.array([1.0]), sig_s=np.array([[1.9]]),
    )

    r_fuel, r_cell = 0.5, 1.0
    materials = {2: fuel, 0: mod}
    geom_params = dict(
        n_fuel=1, n_clad=0, n_cool=1,
        r_fuel=r_fuel, r_clad=r_fuel,
        pitch=r_cell * np.sqrt(np.pi),
    )

    sig_t_all = np.array([[1.0], [2.0]])
    P_inf_g, V_arr = _build_cylindrical_pinf(sig_t_all, r_fuel, r_cell)

    k_inf = _kinf_from_cp(
        P_inf_g=P_inf_g, sig_t_all=sig_t_all, V_arr=V_arr,
        sig_s_mats=[np.array([[0.5]]), np.array([[1.9]])],
        nu_sig_f_mats=[np.array([0.75]), np.array([0.0])],
        chi_mats=[np.array([1.0]), np.array([1.0])],
    )
    return materials, geom_params, k_inf


def benchmark_2g_cylinder() -> tuple[dict[int, Mixture], dict, float]:
    """2-group, 2-region Wigner-Seitz cylinder (fuel + moderator).

    Same XS as the 2-group slab benchmark.
    Geometry: r_fuel = 0.5 cm, r_cell = 1.0 cm
    """
    fuel = _make_mixture(
        sig_t=np.array([0.50, 1.00]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.50, 2.50]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
    )
    mod = _make_mixture(
        sig_t=np.array([0.60, 2.00]),
        sig_c=np.array([0.02, 0.05]),
        sig_f=np.array([0.00, 0.00]),
        nu=np.array([0.00, 0.00]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.40, 0.18], [0.00, 1.95]]),
    )

    r_fuel, r_cell = 0.5, 1.0
    materials = {2: fuel, 0: mod}
    geom_params = dict(
        n_fuel=1, n_clad=0, n_cool=1,
        r_fuel=r_fuel, r_clad=r_fuel,
        pitch=r_cell * np.sqrt(np.pi),
    )

    sig_t_all = np.array([[0.50, 1.00], [0.60, 2.00]])
    P_inf_g, V_arr = _build_cylindrical_pinf(sig_t_all, r_fuel, r_cell)

    k_inf = _kinf_from_cp(
        P_inf_g=P_inf_g, sig_t_all=sig_t_all, V_arr=V_arr,
        sig_s_mats=[
            np.array([[0.38, 0.10], [0.00, 0.90]]),
            np.array([[0.40, 0.18], [0.00, 1.95]]),
        ],
        nu_sig_f_mats=[
            np.array([0.025, 0.20]),
            np.array([0.0, 0.0]),
        ],
        chi_mats=[
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        ],
    )
    return materials, geom_params, k_inf
