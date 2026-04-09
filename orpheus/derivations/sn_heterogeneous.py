r"""Semi-analytical eigenvalue for 1D multi-group multi-region slab.

Solves the transcendental eigenvalue problem for the 1D multi-group
diffusion equation on a slab with reflective boundary conditions using
the transfer matrix method.

For each region with width :math:`t` and material properties, the
spatial solution of :math:`-D\phi'' + M\phi = 0` (where :math:`M`
includes removal, scattering, and fission) has the transfer matrix:

.. math::

    \begin{pmatrix} \phi(t) \\ J(t) \end{pmatrix}
    = T(t)
    \begin{pmatrix} \phi(0) \\ J(0) \end{pmatrix}

The eigenvalue k is the value for which the full-slab transfer matrix
satisfies the reflective BC at both ends: :math:`J(0) = 0` and
:math:`J(L) = 0`.

The result is exact to the precision of the root-finder (typically
``< 1e-12``), independent of any spatial mesh.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import expm

from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_mixture
from .sn import _THICKNESSES, _MAT_IDS


def _diffusion_coeff(sig_t: np.ndarray) -> np.ndarray:
    """D_g = 1 / (3 Σ_{t,g})."""
    return 1.0 / (3.0 * sig_t)


def _build_system_matrix(mix, k: float) -> np.ndarray:
    """Build the 2ng × 2ng first-order ODE system matrix.

    The multi-group diffusion equation  -D φ'' + M φ = 0  becomes
    the first-order system:

        d/dx [φ]   [  0      I    ] [φ]
             [J] = [ D⁻¹M    0    ] [J]

    where J = -D dφ/dx is the current and M = Σ_t - Σ_s^T - (1/k)F.

    Returns the 2ng × 2ng matrix.
    """
    sig_t = np.array(mix.SigT)
    sig_s = mix.SigS[0].toarray()
    sig_p = np.array(mix.SigP)
    chi = mix.chi
    ng = len(sig_t)
    D = _diffusion_coeff(sig_t)

    # Net removal: M = Σ_t - Σ_s^T - (1/k)χ⊗(νΣ_f)
    M = np.diag(sig_t) - sig_s.T - np.outer(chi, sig_p) / k

    # First-order system: y = [φ; J], dy/dx = S y
    # where φ' = -D⁻¹ J  and  J' = -M φ
    # So: S = [[0, -D⁻¹], [-M, 0]]
    S = np.zeros((2 * ng, 2 * ng))
    S[:ng, ng:] = -np.diag(1.0 / D)
    S[ng:, :ng] = -M

    return S


def _transfer_matrix(mix, k: float, thickness: float) -> np.ndarray:
    """Compute the transfer matrix T(t) = expm(S * t) for a region.

    Maps [φ(0), J(0)] → [φ(t), J(t)].
    """
    S = _build_system_matrix(mix, k)
    return expm(S * thickness)


def _solve_slab_eigenvalue(
    materials: dict[int, object],
    thicknesses: list[float],
    mat_ids: list[int],
    k_low: float = 0.01,
    k_high: float = 10.0,
    tol: float = 1e-12,
) -> float:
    """Find keff for a multi-region slab with reflective BCs.

    Reflective BCs: J(0) = 0 and J(L) = 0.

    The condition is: the full transfer matrix T_total maps
    [φ(0), 0] → [φ(L), 0]. So the lower-right block of T_total
    applied to [φ(0), 0] must give J(L) = 0, meaning the
    (ng:, :ng) block of T_total must be singular when restricted
    to the φ subspace.

    Equivalently: det(T_total[ng:, :ng]) = 0.
    """
    n_regions = len(thicknesses)
    mix_first = materials[mat_ids[0]]
    ng = len(mix_first.SigT)

    def objective(k):
        # Compose transfer matrices: T_total = T_n * ... * T_2 * T_1
        T_total = np.eye(2 * ng)
        for i in range(n_regions):
            mix = materials[mat_ids[i]]
            T_i = _transfer_matrix(mix, k, thicknesses[i])
            T_total = T_i @ T_total

        # With J(0) = 0, the state is [φ(0); 0].
        # After propagation: [φ(L); J(L)] = T_total @ [φ(0); 0]
        # J(L) = T_total[ng:, :ng] @ φ(0) = 0  for non-trivial φ(0)
        # ⟹ det(T_total[ng:, :ng]) = 0
        return np.linalg.det(T_total[ng:, :ng])

    # Find bracket by scanning
    k_vals = np.linspace(k_low, k_high, 200)
    det_vals = [objective(k) for k in k_vals]

    # Find sign changes
    roots = []
    for i in range(len(det_vals) - 1):
        if det_vals[i] * det_vals[i + 1] < 0:
            k_root = brentq(objective, k_vals[i], k_vals[i + 1], xtol=tol)
            roots.append(k_root)

    if not roots:
        raise RuntimeError(
            f"No eigenvalue found in [{k_low}, {k_high}]. "
            f"Det range: [{min(det_vals):.2e}, {max(det_vals):.2e}]"
        )

    # Return the largest root (fundamental mode)
    return max(roots)


def derive_sn_heterogeneous(ng_key: str, n_regions: int) -> VerificationCase:
    """Derive eigenvalue for a heterogeneous slab via transcendental equation.

    Uses the 1D multi-group diffusion equation with transfer matrix
    method, interface matching, and reflective BCs. Solved to machine
    precision via brentq root-finder.
    """
    layout = LAYOUTS[n_regions]
    mat_ids = _MAT_IDS[n_regions]
    thicknesses = _THICKNESSES[n_regions]
    ng = int(ng_key[0])

    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    k_val = _solve_slab_eigenvalue(materials, thicknesses, mat_ids)

    latex = (
        rf"Semi-analytical {ng}G {n_regions}-region slab eigenvalue with "
        r"reflective BCs. Transfer matrix method for the multi-group "
        r"diffusion equation, solved via brentq to machine precision."
        "\n\n"
        r".. math::" "\n"
        rf"   k_{{\text{{eff}}}} = {k_val:.12f}"
    )

    return VerificationCase(
        name=f"sn_slab_{ng}eg_{n_regions}rg",
        k_inf=k_val,
        method="sn",
        geometry="slab",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=dict(thicknesses=thicknesses, mat_ids=mat_ids),
        latex=latex,
        description=(
            f"SN 1D slab, {ng}G {n_regions}-region — "
            f"diffusion transfer matrix (semi-analytical)"
        ),
        tolerance="< 1e-10",
    )


def all_cases() -> list[VerificationCase]:
    """Return all semi-analytical heterogeneous SN cases."""
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [2, 4]:
            cases.append(derive_sn_heterogeneous(ng_key, n_regions))
    return cases
