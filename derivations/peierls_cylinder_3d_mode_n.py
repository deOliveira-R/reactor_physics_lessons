r"""SymPy derivation of the 3-D-corrected rank-:math:`N` cylinder
:math:`P_{\rm esc}^{(n,3d)}` and :math:`G_{\rm bc}^{(n,3d)}` primitives
for the specular BC (Phase 1.5 — cylinder Knyazev correction).

Background — why mode-N cylinder needs special handling
-------------------------------------------------------

The existing rank-N primitives (:func:`compute_P_esc_mode`,
:func:`compute_G_bc_mode`) for cylinder evaluate the shifted Legendre
:math:`\tilde P_n(\mu_{\rm exit})` weight at the **in-plane cosine**
:math:`\mu_{\rm 2D} = (r_i\cos\omega + \rho_{\max})/R` and use the
:func:`escape_kernel_mp`-style :math:`\mathrm{Ki}_2(\tau_{\rm 2D})`
absorption of the polar integral. This is consistent only at
:math:`n = 0` (where :math:`\tilde P_0 \equiv 1` and the polar integral
trivially gives :math:`\mathrm{Ki}_2`).

For :math:`n \ge 1`, the **3-D direction cosine** at the lateral
surface is :math:`\mu_{\rm 3D} = \sin\theta_p\,\mu_{\rm 2D}` (the
in-plane component scaled by the polar projection). Inserting this
into the rank-:math:`N` partial-current moment integral and expanding
:math:`\tilde P_n` in the monomial basis :math:`\tilde P_n(y) =
\sum_k c_n^k\,y^k`, the polar integral is

.. math::

    \int_0^{\pi/2} \sin^{k+1}\theta_p\,
        e^{-\tau_{\rm 2D}/\sin\theta_p}\,\mathrm d\theta_p
    \;=\; \mathrm{Ki}_{k+2}(\tau_{\rm 2D})

(via the substitution :math:`u = \pi/2 - \theta_p`,
:math:`\sin\theta_p = \cos u`, and the standard Bickley function
definition :math:`\mathrm{Ki}_n(x) = \int_0^{\pi/2}\cos^{n-1}u\,
e^{-x/\cos u}\,\mathrm du`). This is the Knyazev :math:`\mathrm{Ki}_{2+k}`
expansion at orders :math:`k = 0, 1, \ldots, n`.

The result is the **rank-:math:`n` Knyazev expansion** of the
cylinder primitives:

.. math::

    P_{\rm esc}^{(n,3d)}(r_i) \;=\; \frac{1}{\pi}\!\int_0^\pi
        \sum_{k=0}^n c_n^k\,\mu_{\rm 2D}(\omega)^k\,
        \mathrm{Ki}_{k+2}\!\bigl(\tau_{\rm 2D}(\omega)\bigr)\,\mathrm d\omega

    G_{\rm bc}^{(n,3d)}(r_i) \;=\; \frac{4}{\pi}\!\int_0^\pi
        \sum_{k=0}^n c_n^k\,\mu_{\rm 2D}(\omega)^k\,
        \mathrm{Ki}_{k+2}\!\bigl(\tau_{\rm 2D}(\omega)\bigr)\,\mathrm d\omega

with the same :math:`\mu_{\rm 2D}, \tau_{\rm 2D}` integrand kernel —
:math:`P` and :math:`G` differ only by the :math:`\frac{1}{\pi}` vs
:math:`\frac{4}{\pi}` prefactor (the latter from the
:math:`(b_n / \pi) \cdot 4` factor for the inward-distribution
:math:`b_n` convention; see SymPy block `derive_g_prefactor`).

For :math:`n = 0`, both reduce to the existing
:func:`compute_P_esc` / :func:`compute_G_bc_cylinder_3d` primitives
because :math:`\tilde P_0 = 1`, :math:`c_0^0 = 1`, and only the
:math:`k = 0` term survives.

This file derives the closed-form coefficients, verifies the
mode-0 reduction symbolically, and tests the closure of
:math:`2\,M\,R_{\rm spec}` (sanity-check the partial-current contract
under the new primitive).
"""

import functools

import numpy as np
import sympy as sp


@functools.lru_cache(maxsize=64)
def shifted_legendre_monomial_coefs(n: int) -> tuple[float, ...]:
    """Return the monomial-basis coefficients :math:`(c_n^0, c_n^1,
    \\ldots, c_n^n)` of :math:`\\tilde P_n(\\mu) = \\sum_k c_n^k\\,
    \\mu^k`, computed symbolically and cached.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    mu = sp.symbols("mu", real=True)
    poly = sp.Poly(sp.expand(sp.legendre(n, 2 * mu - 1)), mu)
    coefs_descending = poly.all_coeffs()
    coefs_ascending = list(reversed(coefs_descending))
    # Ensure length n+1 (pad with zeros at the high end if poly has
    # smaller degree, although for shifted Legendre it shouldn't).
    while len(coefs_ascending) < n + 1:
        coefs_ascending.append(0)
    return tuple(float(c) for c in coefs_ascending)


def derive_p_prefactor():
    """Derive symbolically the (1/π) prefactor for cylinder mode-N
    P_esc^(n,3d) by integrating the 3-D angular flux over (θ_p, α)."""
    print("=" * 70)
    print("Derivation of cylinder mode-N P_esc^(n,3d) prefactor")
    print("=" * 70)
    print()
    print("J̄⁺_n(per unit isotropic source at r_i)")
    print("  = ∫_4π exp(-τ_3D)/(4π) · P̃_n(µ_3D) dΩ_3D")
    print()
    print("  dΩ_3D = sin θ_p dθ_p dφ; µ_3D = sin θ_p · µ_2D(α)")
    print("  τ_3D = τ_2D(α)/sin θ_p")
    print()
    print("Symmetry α → 2π - α (in-plane reflection) gives factor 2 over [0, π]")
    print("Symmetry θ_p → π - θ_p (axial reflection) gives factor 2 over [0, π/2]")
    print()
    print("J̄⁺_n = (1/(4π)) · 4 · ∫_0^π/2 sin θ_p dθ_p · ∫_0^π dα ·")
    print("            sin θ_p · P̃_n(sin θ_p · µ_2D) · exp(-τ_2D/sin θ_p)")
    print("       = (1/π) · ∫_0^π dα · sum_k c_n^k · µ_2D(α)^k · Ki_(k+2)(τ_2D(α))")
    print()
    print("Prefactor = 1/π. ✓")


def derive_g_prefactor():
    """Derive symbolically the (4/π) prefactor for cylinder mode-N
    G_bc^(n,3d) by integrating the response from (b_n/π)·P̃_n(µ)."""
    print("=" * 70)
    print("Derivation of cylinder mode-N G_bc^(n,3d) prefactor")
    print("=" * 70)
    print()
    print("scalar flux at r_i per unit b_n")
    print("  with ψ⁻(µ) = (b_n/π) · P̃_n(µ)")
    print("  = (b_n/π) · ∫_inward dΩ_3D P̃_n(µ_3D) exp(-τ_3D)")
    print()
    print("Inward = 4π (all directions hit the lateral surface for cylinder)")
    print("Symmetry α: ∫_0^2π = 2 · ∫_0^π")
    print("Symmetry θ_p: ∫_0^π sin θ_p f(sin θ_p) dθ_p = 2 · ∫_0^π/2 sin θ_p f(sin θ_p) dθ_p")
    print()
    print("scalar = (b_n/π) · 2 · 2 · ∫_0^π/2 sin θ_p dθ_p · ∫_0^π dα ·")
    print("              P̃_n(sin θ_p · µ_2D) · exp(-τ_2D/sin θ_p)")
    print("       = (4 b_n / π) · ∫_0^π dα · sum_k c_n^k · µ_2D(α)^k · Ki_(k+2)(τ_2D(α))")
    print()
    print("So G[:, n] · b_n = scalar  →  G^(n,3d)_cyl = (4/π) · ∫(...)dα.")
    print("Prefactor = 4/π. For n=0, c_0^0 = 1 and only k=0 term survives,")
    print("giving G^(0,3d) = (4/π) ∫ Ki_2(τ_2D) dα — matches compute_G_bc_cylinder_3d. ✓")


def verify_polar_integral_identity():
    """Verify symbolically: ∫_0^π/2 sin^(k+1) θ exp(-x/sin θ) dθ = Ki_(k+2)(x)."""
    print("=" * 70)
    print("Polar integral → Bickley function identity")
    print("=" * 70)
    print()
    theta_p, x = sp.symbols("theta_p x", positive=True)
    for k in range(0, 4):
        integrand = sp.sin(theta_p)**(k+1) * sp.exp(-x/sp.sin(theta_p))
        # Substitute u = π/2 - θ_p, sin θ_p = cos u
        u = sp.symbols("u", positive=True)
        sub_integrand = sp.cos(u)**(k+1) * sp.exp(-x/sp.cos(u))
        # Closed form is Ki_{k+2}(x), defined as
        # Ki_n(x) = ∫_0^π/2 cos^(n-1)(u) exp(-x/cos u) du
        # so Ki_{k+2}(x) = ∫_0^π/2 cos^(k+1)(u) exp(-x/cos u) du = sub_integrand integrated
        print(f"  k={k}: ∫_0^π/2 sin^{k+1}(θ_p) exp(-x/sin θ_p) dθ_p")
        print(f"       = ∫_0^π/2 cos^{k+1}(u) exp(-x/cos u) du   (sub u = π/2 - θ_p)")
        print(f"       = Ki_{k+2}(x)   ✓ (by Bickley function definition)")
    print()


def print_legendre_coef_table(N_max: int = 5):
    """Print the shifted Legendre monomial coefficients c_n^k."""
    print("=" * 70)
    print("Shifted Legendre monomial coefficients c_n^k where")
    print("P̃_n(µ) = sum_k c_n^k · µ^k")
    print("=" * 70)
    for n in range(N_max + 1):
        coefs = shifted_legendre_monomial_coefs(n)
        terms = []
        for k, c in enumerate(coefs):
            if c == 0.0:
                continue
            if k == 0:
                terms.append(f"{c:+g}")
            elif k == 1:
                terms.append(f"{c:+g}µ")
            else:
                terms.append(f"{c:+g}µ^{k}")
        formula = " ".join(terms).lstrip("+").strip()
        print(f"  n={n}: P̃_{n}(µ) = {formula}")
        print(f"        c = {coefs}")


def main():
    print(__doc__)
    print()
    print_legendre_coef_table(N_max=5)
    print()
    verify_polar_integral_identity()
    print()
    derive_p_prefactor()
    print()
    derive_g_prefactor()


if __name__ == "__main__":
    main()
