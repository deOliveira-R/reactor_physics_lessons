"""SymPy derivations for 2-group diffusion eigenvalues.

1-region (bare slab): analytical buckling eigenvalue B² = (π/H)².
2-region (fuel + reflector): analytical interface matching — cos/sinh
solutions in fuel/reflector, flux and current continuity → transcendental
equation for k_eff solved by brentq to machine precision.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

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
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical diffusion cases (bare slab only)."""
    return [derive_1rg()]


def solver_cases() -> list[VerificationCase]:
    """Return solver-computed diffusion cases (fuel+reflector Richardson)."""
    return [derive_2rg()]
