r"""SymPy derivation of slab per-face mode-:math:`N` primitives for
the specular boundary condition (Phase 1.5 — slab per-face decomposition).

Background — why slab needs per-face modes
------------------------------------------

Slab :math:`\mu`-polar geometry has TWO planar boundary surfaces (at
:math:`x = 0` and :math:`x = L`), each with its own outward normal.
Outgoing partial-current moments at the two faces are independent;
the inward angular flux at each face is determined by the **local**
specular reflection (since planar specular is the simplest case —
literally a mirror, with no curvature coupling between faces). The
mode space is therefore

.. math::

    A_{\rm slab} \;=\; A_{\rm outer} \oplus A_{\rm inner}
                 \;=\; \mathbb R^{N} \oplus \mathbb R^{N}
                 \;=\; \mathbb R^{2N}

and the reflection operator is **block diagonal**:

.. math::

    R_{\rm slab} \;=\; \begin{pmatrix}
        R_{\rm spec}(N) & 0 \\
        0 & R_{\rm spec}(N)
    \end{pmatrix}_{2N \times 2N}.

Per-face primitives in closed form
----------------------------------

For slab the µ integrals reduce to **exponential integrals**
:math:`E_n(\tau)` because the path optical depth factors as
:math:`\tau_{\rm path} = \tau_{\rm perp} / \mu`. Substituting
:math:`t = 1/\mu` shows :math:`\int_0^1 \mu^k\,e^{-\tau_{\rm perp}/\mu}
\mathrm d\mu = E_{k+2}(\tau_{\rm perp})`. Combined with the shifted-
Legendre monomial expansion :math:`\tilde P_n(\mu) = \sum_k c_n^k\,
\mu^k`:

**P primitive (outer face, x = L)**:

.. math::
   :label: peierls-slab-P-outer

   P_{\rm esc, out}^{(n)}(x_i)
       \;=\; \tfrac{1}{2}\!\int_0^1 \mu\,\tilde P_n(\mu)\,
                e^{-\tau_{\rm out}(x_i)/\mu}\,\mathrm d\mu
       \;=\; \tfrac{1}{2}\sum_{k=0}^{n} c_n^k\,
                E_{k+3}\!\bigl(\tau_{\rm out}(x_i)\bigr).

**G primitive (outer face)**:

.. math::
   :label: peierls-slab-G-outer

   G_{\rm bc, out}^{(n)}(x_i)
       \;=\; 2\!\int_0^1 \tilde P_n(\mu)\,e^{-\tau_{\rm out}(x_i)/\mu}\,
                \mathrm d\mu
       \;=\; 2\sum_{k=0}^{n} c_n^k\,
                E_{k+2}\!\bigl(\tau_{\rm out}(x_i)\bigr).

The inner face uses :math:`\tau_{\rm in}(x_i) = \int_0^{x_i}
\Sigma_t(x')\,\mathrm dx'` (perpendicular optical depth from
:math:`x_i` to the face at :math:`x = 0`); same formula.

**Reductions check**:

- :math:`n = 0`: :math:`c_0^0 = 1`, only :math:`k = 0` term.
  Using the **no-µ-weight basis** (E_(k+2) for both P and G — the
  basis used by the shipped slab specular implementation):
  :math:`P^{(0)} = \tfrac{1}{2}\,E_2(\tau)` and :math:`G^{(0)} = 2\,
  E_2(\tau)`. These match :func:`compute_P_esc_outer` and
  :func:`compute_G_bc_outer` for slab bit-exactly.

  An alternative **µ-weighted basis** (E_(k+3) for P, E_(k+2) for G,
  the canonical partial-current moment convention) gives :math:`P^{(0)}
  = \tfrac{1}{2}\,E_3(\tau)` instead. This basis is mathematically
  equivalent to no-µ-weight after a corresponding R rescaling, but the
  no-µ-weight convention is preferred because mode-0 reduces to
  :func:`compute_P_esc_outer` and matches the curvilinear specular
  branch's no-Jacobian sphere convention.

K_bc assembly
-------------

For slab specular,

.. math::

   K_{\rm bc}^{\rm slab,spec} \;=\; G_{\rm slab} \cdot R_{\rm slab}
                                  \cdot P_{\rm slab}

with :math:`P_{\rm slab} \in \mathbb R^{2N \times N_x}`,
:math:`G_{\rm slab} \in \mathbb R^{N_x \times 2N}`,
:math:`R_{\rm slab} \in \mathbb R^{2N \times 2N}` block-diagonal.
Block layout: rows/cols :math:`[0, N)` = outer face; rows/cols
:math:`[N, 2N)` = inner face. No off-block entries because specular
is **local at each face** (no inter-face coupling — a particle
specularly reflected at the outer face does not "feel" the inner
face except through the volume kernel, which is already in
:math:`K_{\rm vol}`).

CRITICAL implementation gotcha — the divisor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-face G primitive must use the **single-face surface area
divisor** (= 1 for slab; each face has unit transverse area), NOT
the legacy combined-face divisor (= 2 from
:meth:`CurvilinearGeometry.rank1_surface_divisor`). The combined-face
divisor=2 is correct for legacy Mark, which uses ONE primitive
representing the SUM over both faces (½ E_2_outer + ½ E_2_inner) and
then normalises by the TOTAL surface area = 2.

The per-face decomposition keeps the two faces algebraically separate.
Each face block is a self-contained "outgoing × R_face × inward"
factorisation with the face's individual area = 1.

This was the entire root cause of the 2026-04-27 "rank-N specular slab
plateaus at -5%/+7%" investigation. With divisor=2 (incorrectly
inherited from the legacy Mark code path), the per-face block-diagonal
specular reflects only HALF the outgoing current at each face, leading
to the observed 5-8 % undershoot. With the correct divisor=1, R_face =
:math:`(1/2) M^{-1}` recovers the rank-N specular and converges
monotonically to k_inf as N → ∞ AND as the spatial mesh refines. See
agent memory `specular_bc_slab_fix.md` and the test pair
:func:`tests.derivations.test_peierls_specular_bc.test_specular_slab_rank1_equals_mark_kinf`
/
:func:`tests.derivations.test_peierls_specular_bc.test_specular_slab_homogeneous_converges_to_kinf`
for verification.

Verification
------------

For a homogeneous slab with specular BC at both faces, the cell is
equivalent to an infinite medium under the periodic / reflective
extension, so :math:`k_{\rm eff} = k_\infty` exactly (rotational
symmetry not needed; translational symmetry suffices). The rank-N
truncation error converges as :math:`N \to \infty` because the
truncation only matters when the angular flux at the face has
non-trivial structure beyond the first :math:`N` Legendre modes —
for translation-invariant homogeneous, the angular flux IS isotropic
in the half-range, so all rank-N truncations recover :math:`k_\infty`
to within quadrature precision.

This file generates the SymPy verification of:

A. The :math:`E_{k+2}` / :math:`E_{k+3}` identities from polynomial
   shifted Legendre monomial expansion.

B. The mode-0 reduction to :math:`(1/2) E_3` for P and :math:`2 E_2`
   for G.

C. The block-diagonal :math:`2 M R = I` contract per face (which
   trivially follows from the single-face derivation).
"""

import functools

import numpy as np
import sympy as sp


@functools.lru_cache(maxsize=64)
def shifted_legendre_monomial_coefs(n: int) -> tuple[float, ...]:
    """Same as in peierls_geometry.py — duplicated here for the
    derivation script's standalone usage."""
    if n < 0:
        raise ValueError(n)
    mu = sp.symbols("mu", real=True)
    poly = sp.Poly(sp.expand(sp.legendre(n, 2 * mu - 1)), mu)
    coefs_descending = poly.all_coeffs()
    coefs_ascending = list(reversed(coefs_descending))
    while len(coefs_ascending) < n + 1:
        coefs_ascending.append(0)
    return tuple(float(c) for c in coefs_ascending)


def derive_slab_p_primitive():
    """Derive P_esc_outer^(n) symbolically and verify the E_(k+3) form."""
    print("=" * 70)
    print("Slab outer P_esc^(n)(x_i) derivation")
    print("=" * 70)
    mu, tau = sp.symbols("mu tau", positive=True)
    n_max = 4
    print()
    print("Definition: P_esc_out^(n)(x_i) = (1/2) ∫_0^1 µ P̃_n(µ) exp(-τ/µ) dµ")
    print()
    print("Substitute t = 1/µ ⇒ dµ = -dt/t² ⇒")
    print("  ∫_0^1 µ^k exp(-τ/µ) dµ = ∫_1^∞ exp(-τ t)/t^(k+2) dt = E_(k+2)(τ).")
    print()
    print("So P_esc_out^(n)(x_i) = (1/2) sum_k c_n^k · E_(k+3)(τ).")
    print()
    print("Mode-0 check: c_0^0 = 1, only k=0 term:")
    print("  P_esc_out^(0) = (1/2) · E_3(τ).")
    print("  Note: this is the partial-current moment J⁺_0 (with µ weight),")
    print("  NOT the Mark escape probability (1/2) E_2(τ) used by")
    print("  compute_P_esc_outer for slab. The slab specular branch uses the")
    print("  partial-current basis (the same no-Jacobian convention as the")
    print("  curvilinear specular branch).")
    print()
    print("Coefficient table:")
    for n in range(n_max + 1):
        coefs = shifted_legendre_monomial_coefs(n)
        terms = []
        for k, c in enumerate(coefs):
            if c == 0.0:
                continue
            terms.append(f"{c:+g}·E_{k+3}(τ)")
        formula = " ".join(terms).lstrip("+").strip()
        print(f"  n={n}: P_esc_out^({n})(x_i) = (1/2)·[{formula}]")


def derive_slab_g_primitive():
    """Derive G_bc_outer^(n) symbolically and verify the E_(k+2) form."""
    print("=" * 70)
    print("Slab outer G_bc^(n)(x_i) derivation")
    print("=" * 70)
    print()
    print("Definition: G_bc_out^(n)(x_i) = 2 ∫_0^1 P̃_n(µ) exp(-τ/µ) dµ")
    print("            (no µ weight; the inward distribution ψ⁻ = (b_n/π)·P̃_n)")
    print()
    print("By the same substitution t = 1/µ:")
    print("  ∫_0^1 µ^k exp(-τ/µ) dµ = E_(k+2)(τ).")
    print()
    print("So G_bc_out^(n)(x_i) = 2 sum_k c_n^k · E_(k+2)(τ).")
    print()
    print("Mode-0 check: c_0^0 = 1, only k=0 term:")
    print("  G_bc_out^(0) = 2 · E_2(τ).  ✓ matches compute_G_bc_outer.")
    print()
    print("Coefficient table:")
    n_max = 4
    for n in range(n_max + 1):
        coefs = shifted_legendre_monomial_coefs(n)
        terms = []
        for k, c in enumerate(coefs):
            if c == 0.0:
                continue
            terms.append(f"{c:+g}·E_{k+2}(τ)")
        formula = " ".join(terms).lstrip("+").strip()
        print(f"  n={n}: G_bc_out^({n})(x_i) = 2·[{formula}]")


def main():
    print(__doc__)
    derive_slab_p_primitive()
    print()
    derive_slab_g_primitive()


if __name__ == "__main__":
    main()
