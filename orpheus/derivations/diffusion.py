r"""Analytical and semi-analytical reference solutions for 1D diffusion.

Two problem families:

1. **Bare slab (1 region)** — 2-group vacuum-BC diffusion eigenvalue
   with the sine eigenfunction :math:`\phi_g(x) = c_g\sin(\pi x/L)`
   and the buckled matrix eigenvalue
   :math:`(\mathbf A_{\text{rem}} + D B^{2}\mathbf I)\phi = k^{-1}\mathbf F\phi`.
   Pure analytical (T1) — no quadrature, no iteration.
2. **Fuel + reflector (2 regions)** — 2-group vacuum-BC diffusion
   eigenvalue with interface matching. Solved semi-analytically
   via the transfer-matrix method: propagate the first-order ODE
   state :math:`\mathbf y = [\boldsymbol\phi; \mathbf J]` through
   each region by the matrix exponential :math:`\exp(\mathbf S\,t)`,
   compose across the interface (automatic continuity of
   :math:`\phi` and :math:`J`), and solve the transcendental
   vacuum-BC eigenvalue condition
   :math:`\det(\mathbf T_{\text{total}}[0{:}ng,\,ng{:}2ng]) = 0`
   via :func:`scipy.optimize.brentq`. Back-substitution gives the
   continuous :math:`\phi_g(x)` at arbitrary :math:`x`. T2
   semi-analytical.

Both families publish **two tiers** of reference during the
Phase-1.2 retrofit:

- **Legacy** ``derive_1rg`` / ``derive_2rg`` returning scalar
  :class:`~orpheus.derivations._types.VerificationCase` — kept
  for backward-compat with existing tests. ``derive_2rg`` still
  reads the Richardson cache; the cache is **not** deleted yet
  so the legacy ``test_spatial_convergence_reflected`` test
  keeps working until the Phase-1.2 consumer test (which uses
  the continuous reference) takes over.
- **Phase-0 continuous** ``derive_1rg_continuous`` /
  ``derive_2rg_continuous`` returning
  :class:`~orpheus.derivations.ContinuousReferenceSolution`
  with mesh-independent callable :math:`\phi_g(x)` — the
  honest reference this module always should have produced.

See :doc:`/theory/diffusion_1d` for the full mathematical
treatment with equation labels, and
:doc:`/verification/reference_solutions` for the campaign
philosophy.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)
from ._types import VerificationCase


# Default cross sections (from CORE1D.m)
_FUEL_XS = dict(
    transport=np.array([0.2181, 0.7850]),
    absorption=np.array([0.0096, 0.0959]),
    fission=np.array([0.0024, 0.0489]),
    production=np.array([0.0061, 0.1211]),
    chi=np.array([1.0, 0.0]),
    scattering=np.array([0.0160, 0.0]),
)

_REFL_XS = dict(
    transport=np.array([0.1887, 1.2360]),
    absorption=np.array([0.0004, 0.0197]),
    fission=np.array([0.0, 0.0]),
    production=np.array([0.0, 0.0]),
    chi=np.array([1.0, 0.0]),
    scattering=np.array([0.0447, 0.0]),
)


def _diffusion_coeffs(transport):
    """D = 1/(3*Sigma_tr)."""
    return 1.0 / (3.0 * transport)


def derive_1rg(fuel_height: float = 50.0) -> VerificationCase:
    r"""2-group bare slab: analytical buckling eigenvalue."""
    xs = _FUEL_XS
    D = _diffusion_coeffs(xs["transport"])
    B2 = (np.pi / fuel_height) ** 2

    A = np.diag(D * B2 + xs["absorption"] + xs["scattering"]) \
        - np.array([[0.0, 0.0], [xs["scattering"][0], 0.0]])
    F = np.outer(xs["chi"], xs["production"])
    M = np.linalg.solve(A, F)
    k_val = float(np.max(np.real(np.linalg.eigvals(M))))

    latex = (
        rf"Bare slab H = {fuel_height} cm, vacuum BCs. "
        rf":math:`B^2 = (\pi/H)^2 = {B2:.6e}`."
        "\n\n"
        r".. math::" "\n"
        rf"   k_{{\text{{eff}}}} = {k_val:.10f}"
    )

    return VerificationCase(
        name="dif_slab_2eg_1rg",
        k_inf=k_val,
        method="dif",
        geometry="slab",
        n_groups=2,
        n_regions=1,
        materials=_FUEL_XS,
        geom_params=dict(fuel_height=fuel_height),
        latex=latex,
        description=f"2-group diffusion bare slab (H={fuel_height} cm, vacuum BCs)",
        tolerance="O(h²)",
        vv_level="L1",
        # TODO: no labels yet — docs/theory/diffusion.rst does not exist
        # (issue #35). Populate with bare-slab-buckling / two-group-diffusion
        # labels once the theory page lands.
        equation_labels=(),
    )


def derive_2rg(
    fuel_height: float = 50.0,
    refl_height: float = 30.0,
) -> VerificationCase:
    r"""2-group fuel + reflector slab: Richardson-extrapolated reference.

    Geometry: [vacuum] fuel (0 to H_f) | reflector (H_f to H_f+H_r) [vacuum]

    The 2-group coupled system with interface matching has a complex
    transcendental equation. We use Richardson extrapolation from the
    diffusion solver at 4 mesh refinements (O(h²)) to obtain the reference.
    Results are cached to avoid recomputation on subsequent test runs.
    """
    from ._richardson_cache import get_cached, store

    H_f = fuel_height
    H_r = refl_height

    case_name = "dif_slab_2eg_2rg"
    dzs = [2.5, 1.25, 0.625, 0.3125]

    cache_params = dict(
        method="dif", fuel_height=H_f, refl_height=H_r, dzs=dzs,
        fuel_xs=_FUEL_XS, refl_xs=_REFL_XS,
    )

    k_val = get_cached(case_name, cache_params)
    if k_val is None:
        from orpheus.diffusion.solver import CoreGeometry, TwoGroupXS, solve_diffusion_1d

        fuel_xs = TwoGroupXS(**_FUEL_XS)
        refl_xs = TwoGroupXS(**_REFL_XS)

        keffs = []
        for dz in dzs:
            geom = CoreGeometry(
                bot_refl_height=0.0, fuel_height=H_f,
                top_refl_height=H_r, dz=dz,
            )
            result = solve_diffusion_1d(
                geom=geom, reflector_xs=refl_xs, fuel_xs=fuel_xs,
            )
            keffs.append(result.keff)

        # O(h²) Richardson extrapolation (ratio 2, two finest)
        k_val = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0
        store(case_name, cache_params, k_val, keffs)

    latex = (
        rf"Fuel + reflector slab: H_f = {H_f} cm, H_r = {H_r} cm. "
        r"Richardson-extrapolated from O(h²) mesh convergence."
        "\n\n"
        r".. math::" "\n"
        rf"   k_{{\text{{eff}}}} = {k_val:.10f}"
    )

    return VerificationCase(
        name=case_name,
        k_inf=k_val,
        method="dif",
        geometry="slab",
        n_groups=2,
        n_regions=2,
        materials=dict(fuel=_FUEL_XS, reflector=_REFL_XS),
        geom_params=dict(fuel_height=H_f, refl_height=H_r),
        latex=latex,
        description=(
            f"2-group diffusion fuel+reflector slab "
            f"(H_f={H_f}, H_r={H_r} cm, vacuum BCs)"
        ),
        tolerance="O(h²)",
        vv_level="L2",
        # TODO: no labels yet — docs/theory/diffusion.rst does not exist
        # (issue #35). Populate with fuel-reflector-matching / flux-continuity
        # / current-continuity labels once the theory page lands.
        equation_labels=(),
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical diffusion cases (bare slab only)."""
    return [derive_1rg()]


def solver_cases() -> list[VerificationCase]:
    """Return solver-computed diffusion cases (fuel+reflector Richardson).

    .. deprecated:: Phase 1.2
        The Richardson-extrapolated ``dif_slab_2eg_2rg`` reference is
        superseded by :func:`derive_2rg_continuous` which builds a
        genuine transcendental transfer-matrix reference with no
        self-crutch. This legacy path is kept alive only so the existing
        ``tests/diffusion/test_diffusion.py::test_spatial_convergence_reflected``
        test (which consumes the Richardson value) keeps working during
        the Phase-1.2 migration window. Phase 2 deletes it.
    """
    return [derive_2rg()]


# ═══════════════════════════════════════════════════════════════════════
# Phase-0 continuous references
# ═══════════════════════════════════════════════════════════════════════
#
# Helpers shared by the 1-region and 2-region continuous derivations.
# Both rest on the same first-order ODE system of the multigroup
# diffusion equation —
#
#     d/dx [phi; J] = S(k) · [phi; J],
#
# with S(k) = [[0, -D^-1], [-M(k), 0]] and M(k) the net removal
# matrix diag(absorption + out-scatter) − downscatter_coupling
# − (1/k)·chi ⊗ (nu·Sigma_f). See docs/theory/diffusion_1d.rst
# :eq:`diffusion-region-ode` and :eq:`diffusion-M-matrix` for the
# derivation, and :eq:`diffusion-mode-decomposition`,
# :eq:`diffusion-exponential-branch`, and
# :eq:`diffusion-trigonometric-branch` for how the ODE is solved
# region-by-region in the bounded real basis.


_NG = 2  # everything in this module is two-group


def _downscatter_matrix(xs_dict: dict) -> np.ndarray:
    """Build the 2x2 down-scattering coupling matrix from the 1D vector.

    ``xs_dict["scattering"]`` is the canonical diffusion convention:
    a ``(ng,)`` array where entry ``g`` is the total out-of-group
    scattering rate from group ``g``. For 2G with downscatter only,
    ``scattering = [Σ_{1→2}, 0]`` and the coupling matrix feeding
    in-scatter into the net-removal operator is ``[[0, 0], [Σ_{1→2}, 0]]``.
    """
    scat_out = np.asarray(xs_dict["scattering"], dtype=float)
    if scat_out.shape != (_NG,):
        raise ValueError(
            f"Expected 2-group scattering vector, got shape {scat_out.shape}"
        )
    return np.array([[0.0, 0.0], [scat_out[0], 0.0]])


def _net_removal_matrix(xs_dict: dict, k: float) -> np.ndarray:
    r"""Build the 2x2 net-removal matrix :math:`\mathbf M(k)`.

    .. math::

        \mathbf M(k) = \text{diag}(\Sigma_a + \Sigma_{s,\text{out}})
                       - \text{downscatter coupling}
                       - \frac{1}{k}\,\chi \otimes (\nu\Sigma_f)

    This matrix enters the diffusion ODE system as
    :math:`-D\,\phi'' + \mathbf M(k)\,\phi = 0`. The dependence on
    ``k`` is how the transcendental eigenvalue condition is
    assembled: search over ``k`` until the transfer matrix obeys
    the vacuum boundary conditions (see
    :func:`_solve_2region_vacuum_eigenvalue`).
    """
    absorption = np.asarray(xs_dict["absorption"], dtype=float)
    scat_out = np.asarray(xs_dict["scattering"], dtype=float)
    chi = np.asarray(xs_dict["chi"], dtype=float)
    production = np.asarray(xs_dict["production"], dtype=float)

    removal = np.diag(absorption + scat_out)
    downscatter = _downscatter_matrix(xs_dict)
    fission = np.outer(chi, production) / k
    return removal - downscatter - fission


def _diffusion_system_matrix(xs_dict: dict, k: float) -> np.ndarray:
    r"""Build the 4x4 first-order ODE system matrix :math:`\mathbf S(k)`.

    The multigroup diffusion equation
    :math:`-\mathbf D\,\boldsymbol\phi'' + \mathbf M(k)\,\boldsymbol\phi = 0`
    is rewritten in state form with
    :math:`\mathbf y = [\boldsymbol\phi; \mathbf J]`,
    :math:`\mathbf J = -\mathbf D\,\boldsymbol\phi'`, giving

    .. math::

        \frac{d\mathbf y}{dx} = \mathbf S(k)\,\mathbf y,
        \qquad
        \mathbf S(k) = \begin{pmatrix} \mathbf 0 & -\mathbf D^{-1} \\
                                        -\mathbf M(k) & \mathbf 0
                       \end{pmatrix}.

    The sign structure guarantees :math:`\mathbf J` and
    :math:`\boldsymbol\phi` are continuous across material
    interfaces automatically when the state vector is carried
    through a region-to-region composition.
    """
    D = _diffusion_coeffs(np.asarray(xs_dict["transport"], dtype=float))
    M = _net_removal_matrix(xs_dict, k)

    S = np.zeros((2 * _NG, 2 * _NG))
    S[:_NG, _NG:] = -np.diag(1.0 / D)
    S[_NG:, :_NG] = -M
    return S


def _region_spatial_modes(xs: dict, k: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Diagonalise the diffusion system matrix for one region.

    Solves the generalised eigenvalue problem

    .. math::

        \mathbf D^{-1}\,\mathbf M(k)\,\mathbf u_i \;=\; \mu_i\,\mathbf u_i,

    which governs the spatial decay/oscillation of the multigroup
    diffusion ODE :math:`-\mathbf D\,\boldsymbol\phi'' + \mathbf M(k)\,
    \boldsymbol\phi = 0`. Each eigenvalue :math:`\mu_i` gives a
    **pair** of spatial modes:

    - :math:`\mu_i > 0` (subcritical region or high-k_inf thermal
      suppression) — two **real exponentials** with
      :math:`\lambda_i = \sqrt{\mu_i}`: one growing from the left,
      one decaying from the left, both carrying the eigenvector
      :math:`\mathbf u_i` as the group amplitude vector.
    - :math:`\mu_i < 0` (supercritical/fissioning region) — two
      **real trigonometric modes** with
      :math:`\omega_i = \sqrt{-\mu_i}`: cosine and sine, also
      carrying :math:`\mathbf u_i`.

    Returns ``(mu_sq, u_mat)`` with ``mu_sq`` a length-``ng`` array
    of real or complex eigenvalues and ``u_mat`` the corresponding
    eigenvectors as columns of a ``(ng, ng)`` matrix. Callers
    inspect ``np.sign(mu_sq.real)`` to dispatch between the
    exponential and trigonometric branches.
    """
    D = _diffusion_coeffs(np.asarray(xs["transport"], dtype=float))
    M = _net_removal_matrix(xs, k)
    mu_sq, u_mat = np.linalg.eig(np.diag(1.0 / D) @ M)
    return mu_sq, u_mat


def _region_basis_matrices(
    xs: dict,
    k: float,
    region_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the (:math:`\phi`, :math:`J`) evaluation matrices for
    the four spatial basis modes of one region.

    For each region we choose a **real, bounded** basis of size 4
    (two modes per group for two groups):

    - If :math:`\mu_i^{2} > 0`: a pair of real exponentials
      anchored to opposite edges —
      :math:`e^{-\sqrt{\mu_i^{2}}\,(L_{\text{reg}} - (x - x_0))}`
      (anchored at the right edge) and
      :math:`e^{-\sqrt{\mu_i^{2}}\,(x - x_0)}`
      (anchored at the left edge). Both are bounded by 1 on
      :math:`[x_0, x_0 + L_{\text{reg}}]`.
    - If :math:`\mu_i^{2} < 0`: a real cosine/sine pair centred
      at the region midpoint with argument
      :math:`\sqrt{-\mu_i^{2}}\,(x - x_0 - L_{\text{reg}}/2)`.
      Both are bounded by 1.

    This real, bounded basis makes the assembled matching matrix
    well-conditioned — every entry is :math:`\mathcal O(1)` rather
    than spanning 30 decades like the naive transfer-matrix
    composition. The catastrophic cancellation in ``det`` that
    killed the first implementation attempt (observed
    :math:`\det \sim 10^{9}` with operand magnitudes :math:`\sim
    10^{26}`) is replaced by honest :math:`\mathcal O(1)` algebra.

    Returns ``(phi_left, J_left, phi_right, J_right)`` where each
    is a ``(ng, n_basis=2*ng)`` real matrix. Column ``j`` of
    ``phi_left`` is :math:`\boldsymbol\phi_j(x_0)` — the flux
    vector generated by the :math:`j`-th basis mode at the left
    edge of the region. ``phi_right`` is the same at the right
    edge, and ``J_*`` are the corresponding currents
    :math:`\mathbf J = -\mathbf D\,\boldsymbol\phi'`.

    With these four matrices, interface continuity at the
    fuel/reflector boundary becomes
    ``phi_right_fuel = phi_left_refl`` and
    ``J_right_fuel = J_left_refl``, while the vacuum BCs at the
    outer edges become ``phi_left_fuel @ c_fuel = 0`` and
    ``phi_right_refl @ c_refl = 0``. The full 8x8 matching
    matrix is assembled from these blocks.
    """
    D = _diffusion_coeffs(np.asarray(xs["transport"], dtype=float))
    D_diag = np.diag(D)
    mu_sq, u_mat = _region_spatial_modes(xs, k)

    # mu_sq is complex in general, but for physical cross sections
    # it is either real-positive (subcritical, exponential modes)
    # or real-negative (supercritical, trigonometric modes).
    # Numerical noise can introduce tiny imaginary parts that we
    # discard; a configuration with genuinely complex mu_sq would
    # indicate an unusual multigroup coupling structure we do not
    # target here.
    if np.abs(mu_sq.imag).max() > 1e-10 * np.abs(mu_sq.real).max():
        raise NotImplementedError(
            f"Region has genuinely complex mu_sq = {mu_sq}. "
            "The real-basis mode decomposition targets 2-group "
            "diffusion with downscatter, whose D^{-1} M is "
            "upper-triangular and has real spectrum."
        )
    mu_sq = mu_sq.real
    u_mat = u_mat.real

    L = region_length
    x_mid = L / 2

    # Build basis at x_left = 0 and x_right = L (relative to region origin)
    n_basis = 2 * _NG  # 2 modes per group
    phi_left = np.zeros((_NG, n_basis))
    phi_right = np.zeros((_NG, n_basis))
    J_left = np.zeros((_NG, n_basis))
    J_right = np.zeros((_NG, n_basis))

    for i in range(_NG):
        u_i = u_mat[:, i]
        mu_i = mu_sq[i]
        col_a = 2 * i        # first basis mode for eigenvalue i
        col_b = 2 * i + 1    # second basis mode for eigenvalue i

        if mu_i > 0:
            # Exponential branch
            lam = np.sqrt(mu_i)
            e_L = np.exp(-lam * L)
            # Mode a: anchored at right edge
            #   m_a(x) = exp(-lam * (L - x)) * u_i
            #   m_a'(x) = +lam * exp(-lam * (L - x)) * u_i
            #   at x=0: m_a = e_L * u_i,  m_a' = +lam * e_L * u_i
            #   at x=L: m_a = 1 * u_i,    m_a' = +lam * 1 * u_i
            phi_left[:, col_a] = e_L * u_i
            phi_right[:, col_a] = u_i
            J_left[:, col_a] = -D * (lam * e_L) * u_i
            J_right[:, col_a] = -D * lam * u_i

            # Mode b: anchored at left edge
            #   m_b(x) = exp(-lam * x) * u_i
            #   m_b'(x) = -lam * exp(-lam * x) * u_i
            phi_left[:, col_b] = u_i
            phi_right[:, col_b] = e_L * u_i
            J_left[:, col_b] = -D * (-lam) * u_i
            J_right[:, col_b] = -D * (-lam * e_L) * u_i
        else:
            # Trigonometric branch (mu_i <= 0)
            omega = np.sqrt(-mu_i) if mu_i < 0 else 0.0
            # Mode a: cosine centred at region midpoint
            #   m_a(x) = cos(omega * (x - L/2)) * u_i
            #   m_a'(x) = -omega * sin(omega * (x - L/2)) * u_i
            c_at_0 = np.cos(omega * (0 - x_mid))  # = cos(-omega*L/2) = cos(omega*L/2)
            c_at_L = np.cos(omega * (L - x_mid))  # = cos(omega*L/2)
            s_at_0 = np.sin(omega * (0 - x_mid))
            s_at_L = np.sin(omega * (L - x_mid))
            phi_left[:, col_a] = c_at_0 * u_i
            phi_right[:, col_a] = c_at_L * u_i
            J_left[:, col_a] = -D * (-omega * s_at_0) * u_i
            J_right[:, col_a] = -D * (-omega * s_at_L) * u_i

            # Mode b: sine centred at region midpoint
            #   m_b(x) = sin(omega * (x - L/2)) * u_i
            #   m_b'(x) = +omega * cos(omega * (x - L/2)) * u_i
            phi_left[:, col_b] = s_at_0 * u_i
            phi_right[:, col_b] = s_at_L * u_i
            J_left[:, col_b] = -D * (omega * c_at_0) * u_i
            J_right[:, col_b] = -D * (omega * c_at_L) * u_i

    return phi_left, J_left, phi_right, J_right


def _assemble_matching_matrix(
    xs_fuel: dict,
    xs_refl: dict,
    fuel_height: float,
    refl_height: float,
    k: float,
) -> np.ndarray:
    r"""Assemble the 8x8 matching matrix :math:`\mathbf C(k)`.

    The eigenvalue problem for the 2-region vacuum-BC slab has
    four boundary/matching constraints:

    1. :math:`\boldsymbol\phi_{\text{fuel}}(x = 0) = \mathbf 0`
       — 2 equations.
    2. :math:`\boldsymbol\phi_{\text{fuel}}(H_f) =
       \boldsymbol\phi_{\text{refl}}(H_f)` — 2 equations.
    3. :math:`\mathbf J_{\text{fuel}}(H_f) =
       \mathbf J_{\text{refl}}(H_f)` — 2 equations.
    4. :math:`\boldsymbol\phi_{\text{refl}}(H_f + H_r) = \mathbf 0`
       — 2 equations.

    Unknowns: 4 mode coefficients per region × 2 regions = 8.

    The 8 × 8 matrix :math:`\mathbf C(k)` acts on the coefficient
    vector :math:`[\mathbf c_{\text{fuel}};\,\mathbf c_{\text{refl}}]`,
    and the eigenvalue condition is
    :math:`\det(\mathbf C(k)) = 0`. Because every basis mode is
    bounded by 1 in its region, :math:`\mathbf C` has
    :math:`\mathcal O(1)` entries and :math:`\det(\mathbf C)` does
    not suffer from the catastrophic cancellation that killed the
    naive ``expm`` composition approach.
    """
    pL_f, JL_f, pR_f, JR_f = _region_basis_matrices(xs_fuel, k, fuel_height)
    pL_r, JL_r, pR_r, JR_r = _region_basis_matrices(xs_refl, k, refl_height)

    C = np.zeros((8, 8))
    # Rows 0-1: phi_fuel(0) = 0
    C[0:2, 0:4] = pL_f
    # Rows 2-3: phi_fuel(H_f) - phi_refl(H_f) = 0
    C[2:4, 0:4] = pR_f
    C[2:4, 4:8] = -pL_r
    # Rows 4-5: J_fuel(H_f) - J_refl(H_f) = 0
    C[4:6, 0:4] = JR_f
    C[4:6, 4:8] = -JL_r
    # Rows 6-7: phi_refl(H_f + H_r) = 0
    C[6:8, 4:8] = pR_r
    return C


def _physical_validation(
    xs_fuel: dict,
    xs_refl: dict,
    fuel_height: float,
    refl_height: float,
    k: float,
    tol: float = 1e-7,
) -> bool:
    r"""Return True if ``k`` is a genuine eigenvalue of the 2-region
    vacuum-BC slab.

    Rebuilds the matching matrix at ``k`` *slightly relaxed* so
    the SVD does not catch a discontinuity artefact, extracts
    the null vector, and explicitly evaluates:

    1. :math:`\boldsymbol\phi_{\text{fuel}}(0)` — must be ~0.
    2. :math:`\boldsymbol\phi_{\text{fuel}}(H_f) -
       \boldsymbol\phi_{\text{refl}}(H_f)` — interface continuity.
    3. :math:`\boldsymbol\phi_{\text{refl}}(H_f + H_r)` — must be ~0.

    If all three are below ``tol`` (relative to the peak flux),
    the root is physical. Otherwise it is a spurious sign change
    caused by the non-continuous eigenvalue ordering of
    :func:`numpy.linalg.eig` across critical :math:`k` values —
    at such crossovers ``det(C(k))`` flips sign by permutation
    rather than by crossing zero, and ``brentq`` returns a point
    where :math:`\mathbf C` is discontinuous rather than singular.
    """
    C = _assemble_matching_matrix(
        xs_fuel, xs_refl, fuel_height, refl_height, k,
    )
    try:
        nv = _extract_real_null_vector(C, tol=1e-3)
    except RuntimeError:
        return False

    c_f = nv[:4]
    c_r = nv[4:]

    phi_0 = _evaluate_modes_at(xs_fuel, k, fuel_height, 0.0, c_f)
    phi_if_f = _evaluate_modes_at(xs_fuel, k, fuel_height, fuel_height, c_f)
    phi_if_r = _evaluate_modes_at(xs_refl, k, refl_height, 0.0, c_r)
    phi_L = _evaluate_modes_at(xs_refl, k, refl_height, refl_height, c_r)

    # Scan the midpoint for a normalisation reference
    phi_mid_f = _evaluate_modes_at(
        xs_fuel, k, fuel_height, fuel_height / 2, c_f,
    )
    scale = max(
        np.linalg.norm(phi_mid_f),
        np.linalg.norm(phi_if_f),
        1e-30,
    )

    err_left = np.linalg.norm(phi_0) / scale
    err_interface = np.linalg.norm(phi_if_f - phi_if_r) / scale
    err_right = np.linalg.norm(phi_L) / scale
    return bool(
        err_left < tol
        and err_interface < tol
        and err_right < tol
    )


def _solve_2region_vacuum_eigenvalue(
    xs_fuel: dict,
    xs_refl: dict,
    fuel_height: float,
    refl_height: float,
    k_low: float = 0.1,
    k_high: float = 3.0,
    n_scan: int = 400,
    tol: float = 1e-14,
) -> float:
    r"""Find :math:`k_\text{eff}` for a 2-region vacuum-BC slab.

    Builds the 8 × 8 real matching matrix :math:`\mathbf C(k)` from
    :func:`_assemble_matching_matrix`, scans for sign changes in
    :math:`\det(\mathbf C(k))`, refines each via
    :func:`scipy.optimize.brentq`, and **physically validates**
    each candidate by reconstructing
    :math:`\boldsymbol\phi_{\text{fuel}}(0)`,
    :math:`\boldsymbol\phi_{\text{refl}}(H_f + H_r)`, and the
    fuel/reflector interface continuity of :math:`\boldsymbol\phi`
    from the null vector and confirming they are all ~0 (see
    :func:`_physical_validation`). The **physical fundamental
    mode** is the largest validated root.

    **Why both mathematical and physical validation.** The SVD
    smallest-singular-value check is necessary but not
    sufficient: :func:`numpy.linalg.eig` returns eigenvalues in
    an order that is not continuous across critical :math:`k`
    values where two eigenvalues cross. At such crossovers, the
    real-basis mode matrix is discontinuous, and
    :math:`\det(\mathbf C(k))` flips sign without passing
    through zero in a physically meaningful way — brentq will
    "converge" to a point where the matrix happens to be
    numerically singular by accident of the eigenvalue
    permutation, not because the BVP has a genuine solution.
    The physical validation rejects such points by checking
    whether the would-be null vector actually solves the
    boundary-value problem.

    **Why real-basis mode decomposition.** The earlier
    ``expm``-composition attempt was abandoned because the
    transfer matrix for an 80-cm diffusion slab has condition
    number :math:`\sim 10^{17}` and the resulting ``det`` of its
    upper-right block suffered catastrophic cancellation, finding
    spurious roots with :math:`\mathcal O(10^{-3})` null-vector
    residuals rather than machine-precision zeros. The
    mode-decomposition approach uses bounded basis functions
    (exponentials anchored to the nearer edge, or cos/sin
    centred at the region midpoint) so every matching-matrix
    entry is :math:`\mathcal O(1)` and there is no catastrophic
    cancellation to worry about.
    """
    def det_c(k: float) -> float:
        return float(np.linalg.det(_assemble_matching_matrix(
            xs_fuel, xs_refl, fuel_height, refl_height, k,
        )))

    ks = np.linspace(k_low, k_high, n_scan)
    dets = np.array([det_c(k) for k in ks])

    candidates = []
    for i in range(len(ks) - 1):
        if dets[i] * dets[i + 1] < 0:
            try:
                candidates.append(brentq(det_c, ks[i], ks[i + 1], xtol=tol))
            except Exception:
                pass

    validated = [
        k for k in candidates
        if _physical_validation(
            xs_fuel, xs_refl, fuel_height, refl_height, k,
        )
    ]

    if not validated:
        raise RuntimeError(
            f"No validated eigenvalue in [{k_low}, {k_high}]. "
            f"Found {len(candidates)} sign-change candidates but none "
            "passed physical validation (phi(0), phi(L), interface "
            "continuity all vanishing). The cross sections may lead "
            "to modes outside the real-basis assumption, or the scan "
            "range does not bracket the fundamental mode."
        )
    # Fundamental mode = largest validated k
    return max(validated)


def _extract_real_null_vector(
    C: np.ndarray, tol: float = 1e-7,
) -> np.ndarray:
    r"""Return the real 1-D null vector of ``C`` via SVD.

    ``C`` is the 8 × 8 real matching matrix. Its smallest singular
    value at the true eigenvalue is :math:`\mathcal O(\epsilon)`
    (machine precision in the conditioning of the real-basis modes).
    The corresponding right singular vector gives the coefficients
    in the 8-dimensional mode basis.
    """
    U, s, Vh = np.linalg.svd(C)
    if s[-1] > tol * s[0]:
        raise RuntimeError(
            f"Matching matrix is not singular: s[-1]/s[0] = "
            f"{s[-1] / s[0]:.2e} > {tol}"
        )
    return Vh[-1, :]


def _evaluate_modes_at(
    xs: dict,
    k: float,
    region_length: float,
    x_rel: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    r"""Evaluate :math:`\boldsymbol\phi(x)` for a region given the
    mode coefficients.

    ``x_rel`` is the position relative to the region's left edge
    (so ``0 <= x_rel <= region_length``). ``coeffs`` is the
    length-4 coefficient vector in the same basis order as
    :func:`_region_basis_matrices`. Returns the 2-vector
    :math:`\boldsymbol\phi(x) = \sum_j c_j\,m_j(x)\,\mathbf u_{j}`.
    """
    D = _diffusion_coeffs(np.asarray(xs["transport"], dtype=float))
    mu_sq, u_mat = _region_spatial_modes(xs, k)
    mu_sq = mu_sq.real
    u_mat = u_mat.real
    L = region_length
    x_mid = L / 2
    phi = np.zeros(_NG)

    for i in range(_NG):
        u_i = u_mat[:, i]
        mu_i = mu_sq[i]
        c_a = coeffs[2 * i]
        c_b = coeffs[2 * i + 1]
        if mu_i > 0:
            lam = np.sqrt(mu_i)
            m_a = np.exp(-lam * (L - x_rel))
            m_b = np.exp(-lam * x_rel)
        else:
            omega = np.sqrt(-mu_i) if mu_i < 0 else 0.0
            m_a = np.cos(omega * (x_rel - x_mid))
            m_b = np.sin(omega * (x_rel - x_mid))
        phi += (c_a * m_a + c_b * m_b) * u_i
    return phi


# ── Bare slab (T1 analytical) ─────────────────────────────────────

def _bare_slab_spectrum(xs_dict: dict, L: float) -> tuple[float, np.ndarray]:
    r"""Compute :math:`k_\infty` and the :math:`\ell^{2}`-normalised
    flux spectrum for a bare 2-group diffusion slab.

    The fundamental spatial mode is :math:`\sin(\pi x/L)`, so the
    PDE reduces to the 2x2 algebraic eigenvalue problem

    .. math::

        (\mathbf A_{\text{rem}} + DB^{2}\mathbf I)\,\boldsymbol\phi
          = \frac{1}{k}\,\mathbf F\,\boldsymbol\phi,

    where :math:`B^{2} = (\pi/L)^{2}`. Factoring and finding the
    dominant eigenvalue of :math:`(\mathbf A_{\text{rem}}+DB^{2})^{-1}\mathbf F`
    gives ``k`` and the right eigenvector gives the flux spectrum.
    """
    D = _diffusion_coeffs(np.asarray(xs_dict["transport"], dtype=float))
    absorption = np.asarray(xs_dict["absorption"], dtype=float)
    scat_out = np.asarray(xs_dict["scattering"], dtype=float)
    chi = np.asarray(xs_dict["chi"], dtype=float)
    production = np.asarray(xs_dict["production"], dtype=float)

    B2 = (np.pi / L) ** 2
    A = np.diag(D * B2 + absorption + scat_out) - _downscatter_matrix(xs_dict)
    F = np.outer(chi, production)
    M = np.linalg.solve(A, F)

    eigvals, eigvecs = np.linalg.eig(M)
    real_vals = np.real(eigvals)
    dominant = int(np.argmax(real_vals))
    k_val = float(real_vals[dominant])
    phi = np.real(eigvecs[:, dominant])
    if phi.sum() < 0:
        phi = -phi
    phi = np.where(np.abs(phi) < 1e-14, 0.0, phi)
    return k_val, phi / np.linalg.norm(phi)


def derive_1rg_continuous(
    fuel_height: float = 50.0,
) -> ContinuousReferenceSolution:
    r"""Phase-0 continuous reference for the 2-group bare slab diffusion problem.

    Returns a :class:`~orpheus.derivations.ContinuousReferenceSolution`
    with ``operator_form="diffusion"`` carrying:

    - ``k_eff`` — exact dominant eigenvalue of
      :math:`(\mathbf A + DB^{2})^{-1}\mathbf F`.
    - ``phi(x, g)`` — the multigroup eigenfunction
      :math:`c_g \sin(\pi x/L)` evaluated at any :math:`x`, with
      :math:`c_g` the :math:`\ell^{2}`-normalised spectrum vector.
    - Vacuum BC recorded in the ``ProblemSpec`` as
      ``{"left": "vacuum", "right": "vacuum"}``.

    See :eq:`bare-slab-eigenfunction`, :eq:`bare-slab-buckling`,
    and :eq:`bare-slab-critical-equation` on the theory page.
    """
    xs = _FUEL_XS
    L = float(fuel_height)
    k_val, phi_spectrum = _bare_slab_spectrum(xs, L)

    def phi(x: np.ndarray, g: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return phi_spectrum[g] * np.sin(np.pi * x / L)

    return ContinuousReferenceSolution(
        name="dif_slab_2eg_1rg",
        problem=ProblemSpec(
            materials={0: xs},
            geometry_type="slab",
            geometry_params={"length": L, "fuel_height": L},
            boundary_conditions={"left": "vacuum", "right": "vacuum"},
            external_source=None,
            is_eigenvalue=True,
            n_groups=_NG,
        ),
        operator_form="diffusion",
        phi=phi,
        provenance=Provenance(
            citation="Bell & Glasstone 1970, §7.4 Eq. (7.40)–(7.42)",
            derivation_notes=(
                "Bare 2-group slab with zero-flux vacuum BCs. Separation of "
                "variables gives phi_g(x) = c_g sin(pi x/L) for all groups, "
                "reducing the PDE to the buckled 2x2 matrix eigenvalue "
                "problem (A_rem + D·B^2) phi = (1/k) F phi with "
                "B^2 = (pi/L)^2. Dominant eigenvalue via numpy.linalg.eig; "
                "spectrum is the right eigenvector, l2-normalised and "
                "sign-adjusted to non-negative."
            ),
            sympy_expression=r"\phi_g(x) = c_g \sin(\pi x / L)",
            precision_digits=None,  # closed form
        ),
        k_eff=k_val,
        psi=None,
        equation_labels=(
            "diffusion-operator",
            "diffusion-coefficient",
            "bare-slab-buckling",
            "bare-slab-eigenfunction",
            "bare-slab-critical-equation",
        ),
        vv_level="L1",
        description=(
            f"2-group diffusion bare slab (H={L} cm, vacuum BCs). "
            f"Phase-1.2 continuous reference — sine eigenfunction."
        ),
        tolerance="< 1e-10 (k_eff), O(h²) flux shape",
    )


# ── 2-region fuel+reflector (T2 semi-analytical) ──────────────────

def derive_2rg_continuous(
    fuel_height: float = 50.0,
    refl_height: float = 30.0,
) -> ContinuousReferenceSolution:
    r"""Phase-0 continuous reference for the 2-group fuel+reflector slab.

    Replaces the Richardson-extrapolated :func:`derive_2rg` with a
    genuine transcendental reference built on the real-basis mode
    decomposition of the multigroup diffusion ODE in each region:

    1. In each region, diagonalise :math:`\mathbf D^{-1}\mathbf M(k)`
       to get eigenvalues :math:`\mu_i = \lambda_i^{2}` and
       eigenvectors :math:`\mathbf u_i`. Each eigenvalue gives a
       pair of bounded real basis modes (exp/exp when
       :math:`\mu_i > 0`, cos/sin when :math:`\mu_i < 0`).
    2. Assemble the 8 × 8 matching matrix :math:`\mathbf C(k)` from
       vacuum-BC, flux-continuity, and current-continuity
       constraints on the 8 mode coefficients (4 per region).
    3. Solve :math:`\det(\mathbf C(k)) = 0` via a coarse scan plus
       :func:`scipy.optimize.brentq`, then **physically validate**
       each candidate by reconstructing
       :math:`\boldsymbol\phi_{\text{fuel}}(0)`, the fuel/reflector
       interface continuity, and
       :math:`\boldsymbol\phi_{\text{refl}}(H_f + H_r)` from the
       null vector and checking they are all ~0. Spurious
       det-sign changes caused by the non-continuous eigenvalue
       ordering of :func:`numpy.linalg.eig` are rejected at this
       step.
    4. At the validated fundamental (largest) root, extract the
       null vector of :math:`\mathbf C(k_\text{fund})` via SVD
       and cache it as the mode-coefficient closure used by
       ``phi``. Per-``x`` evaluation reduces to a handful of real
       exponential / trigonometric evaluations in the appropriate
       region.

    Parameters
    ----------
    fuel_height, refl_height : float
        Thicknesses in cm. The default (50 + 30) matches the
        legacy :func:`derive_2rg` configuration, so the two
        references can be cross-checked against each other
        during the Phase-1.2 migration window.

    Notes
    -----
    See :eq:`diffusion-region-ode`, :eq:`diffusion-mode-decomposition`,
    :eq:`diffusion-matching-matrix`, :eq:`diffusion-transcendental`,
    and :eq:`diffusion-spurious-root-validation` on the theory
    page for the full algorithm. The ``Investigation history``
    section of that page records the two earlier approaches
    (``expm`` transfer-matrix composition and complex-eigenvalue
    mode decomposition) that were tried and abandoned, with the
    numerical evidence of their failure modes.
    """
    H_f = float(fuel_height)
    H_r = float(refl_height)
    L_total = H_f + H_r

    xs_fuel = _FUEL_XS
    xs_refl = _REFL_XS

    k_val = _solve_2region_vacuum_eigenvalue(xs_fuel, xs_refl, H_f, H_r)

    # Extract the 8-d mode coefficient vector from the null space of
    # the matching matrix. Split into fuel (first 4) and reflector
    # (last 4) coefficient blocks.
    C = _assemble_matching_matrix(xs_fuel, xs_refl, H_f, H_r, k_val)
    null_vec = _extract_real_null_vector(C)
    c_fuel = null_vec[:4]
    c_refl = null_vec[4:]

    # Normalise so the maximum |phi_g0| over a dense sweep is exactly 1.
    # The absolute amplitude of the eigenfunction is physically
    # meaningless (eigenvalue problem), so this gives a stable
    # reference scale for consumer-test assertion tolerances.
    _dense_x = np.linspace(0.0, L_total, 2001)
    _dense_phi = np.zeros((2001, _NG))
    for i, xi in enumerate(_dense_x):
        if xi <= H_f:
            _dense_phi[i] = _evaluate_modes_at(
                xs_fuel, k_val, H_f, xi, c_fuel,
            )
        else:
            _dense_phi[i] = _evaluate_modes_at(
                xs_refl, k_val, H_r, xi - H_f, c_refl,
            )

    # Sign-normalise so the fuel group-0 flux is positive at the peak
    _peak_idx = int(np.argmax(np.abs(_dense_phi[:, 0])))
    if _dense_phi[_peak_idx, 0] < 0:
        c_fuel = -c_fuel
        c_refl = -c_refl
        _dense_phi = -_dense_phi
    _max_abs = float(np.max(np.abs(_dense_phi)))
    if _max_abs == 0:
        raise RuntimeError(
            "Back-substituted flux is identically zero — "
            "null-vector extraction failed."
        )
    c_fuel = c_fuel / _max_abs
    c_refl = c_refl / _max_abs

    def phi(x: np.ndarray, g: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        original_shape = x.shape
        x_flat = x.ravel()
        if x_flat.size == 0:
            return np.zeros(original_shape)
        out = np.zeros(x_flat.size)
        for i, xi in enumerate(x_flat):
            if xi <= H_f:
                phi_vec = _evaluate_modes_at(
                    xs_fuel, k_val, H_f, xi, c_fuel,
                )
            else:
                phi_vec = _evaluate_modes_at(
                    xs_refl, k_val, H_r, xi - H_f, c_refl,
                )
            out[i] = phi_vec[g]
        return out.reshape(original_shape)

    return ContinuousReferenceSolution(
        name="dif_slab_2eg_2rg",
        problem=ProblemSpec(
            materials={0: xs_fuel, 1: xs_refl},
            geometry_type="slab",
            geometry_params={
                "fuel_height": H_f,
                "refl_height": H_r,
                "length": L_total,
            },
            boundary_conditions={"left": "vacuum", "right": "vacuum"},
            external_source=None,
            is_eigenvalue=True,
            n_groups=_NG,
        ),
        operator_form="diffusion",
        phi=phi,
        provenance=Provenance(
            citation="Bell & Glasstone 1970, §7.4; Duderstadt & Hamilton 1976, Ch. 7",
            derivation_notes=(
                "2-group fuel+reflector slab with zero-flux vacuum BCs. "
                "Each region's diffusion ODE -D·phi'' + M(k)·phi = 0 is "
                "solved in the region's OWN spatial-mode basis: eigenvalues "
                "mu = lambda^2 of D^(-1)·M give real exponentials "
                "exp(-|lambda|·(x-x0)) and exp(-|lambda|·(x1-x)) when "
                "mu > 0 (subcritical region), or real cos/sin pairs "
                "with argument sqrt(-mu)·(x-x_mid) when mu < 0 "
                "(supercritical region, i.e. fuel at k ~ 0.87). Each "
                "basis mode is bounded by 1 on its region, so the "
                "assembled 8x8 matching matrix C(k) has O(1) entries "
                "and det(C) is computed without catastrophic cancellation.\n\n"
                "The matching constraints are phi(0) = 0 (vacuum left), "
                "phi and J continuous at the fuel/reflector interface, "
                "and phi(L) = 0 (vacuum right) — 8 equations in 8 mode "
                "coefficients. det(C(k)) = 0 is the transcendental "
                "eigenvalue condition, bracketed by a coarse scan and "
                "refined via scipy.optimize.brentq to xtol=1e-14.\n\n"
                "Spurious sign changes in det(C(k)) arise from the "
                "non-continuous eigenvalue ordering of numpy.linalg.eig "
                "across critical k values where two eigenvalues cross. "
                "Every brentq candidate is physically validated: "
                "phi(0), phi(L), and the fuel/reflector interface "
                "continuity are explicitly reconstructed from the null "
                "vector; only candidates with residuals below 1e-7 are "
                "accepted. The fundamental mode is the largest validated "
                "root.\n\n"
                "Once k is found, the continuous phi(x) is evaluated at "
                "any point by pulling the 8-d mode coefficient vector "
                "from the null space of C(k_fund) via SVD, then "
                "substituting back into the region-local mode basis. "
                "All exponentials and trigonometric modes are bounded, "
                "so phi(x) evaluation is stable to machine precision: "
                "phi(0) and phi(L) are both ~1e-16, and interface "
                "continuity holds to ~1e-11. This replaces the "
                "Richardson-extrapolated reference from the legacy "
                "derive_2rg (which had ~1e-5 accuracy from its O(h^4) "
                "leading-order cancellation).\n\n"
                "The earlier attempt at a direct expm(S·t) transfer-"
                "matrix composition was abandoned because the 80-cm "
                "slab has condition number ~1e17 and the resulting "
                "det of the block suffered catastrophic cancellation — "
                "finding spurious 'roots' at which the null vector "
                "gave phi(L) ~ 5e-4 rather than machine precision. "
                "The real-basis mode decomposition here does not have "
                "that failure mode."
            ),
            sympy_expression=None,
            precision_digits=None,
        ),
        k_eff=k_val,
        psi=None,
        equation_labels=(
            "diffusion-operator",
            "diffusion-coefficient",
            "diffusion-region-ode",
            "diffusion-M-matrix",
            "diffusion-mode-decomposition",
            "diffusion-exponential-branch",
            "diffusion-trigonometric-branch",
            "diffusion-interface-matching",
            "diffusion-matching-matrix",
            "diffusion-transcendental",
            "diffusion-spurious-root-validation",
            "diffusion-back-substitution",
        ),
        vv_level="L1",
        description=(
            f"2-group diffusion fuel+reflector slab (H_f={H_f}, H_r={H_r} cm, "
            f"vacuum BCs). Phase-1.2 continuous reference — transcendental "
            f"transfer-matrix, replaces Richardson."
        ),
        tolerance="< 1e-10 (k_eff), O(h²) flux shape",
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Return all diffusion Phase-0 continuous reference solutions."""
    return [
        derive_1rg_continuous(),
        derive_2rg_continuous(),
    ]
