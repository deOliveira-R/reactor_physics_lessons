"""SymPy derivation of the specular-BC reflection operator R_specular.

Derives the closed form of the rank-:math:`N` specular reflection matrix
in the Gelbard half-range shifted-Legendre basis used by the rank-:math:`N`
Peierls closure operator (`peierls_geometry.BoundaryClosureOperator`).

Result
------

.. math::

    R_{\\rm spec} \\;=\\; \\tfrac{1}{2}\\,M^{-1},
    \\qquad
    M_{nm} \\;=\\; \\int_0^1 \\mu\\,\\tilde P_n(\\mu)\\,\\tilde P_m(\\mu)\\,
    \\mathrm d\\mu

where :math:`\\tilde P_n(\\mu) = P_n(2\\mu - 1)`.

Background — what closure conventions live in P and G
-----------------------------------------------------

Per `compute_P_esc_mode` (peierls_geometry.py:2935): the rank-:math:`N`
escape primitive returns the **outgoing partial-current moment**

.. math::

    J^{+}_n(r_i) \\;=\\; 2\\pi\\int_0^1 \\mu\\,\\tilde P_n(\\mu)\\,
                        \\psi^{+}(r_b, \\mu)\\,\\mathrm d\\mu

where :math:`\\psi^{+}(r_b, \\mu)` is the outgoing angular flux at the
surface from a delta source at :math:`r_i` (uncollided). The
:math:`(\\rho_{\\max}/R)^2` Jacobian in the integrand is the surface-to-
observer change of variables :math:`\\mathrm dA_s/\\mathrm d\\Omega_{\\rm obs}
= d^2/|\\mu_s|`, with the :math:`|\\mu_s|` cancelling the
partial-current cosine weighting.

Per `compute_G_bc_mode` (peierls_geometry.py:3043): the rank-:math:`N`
response primitive consumes a coefficient :math:`b_n` representing the
inward angular flux :math:`\\psi^{-}(r_b, \\mu) = \\sum_n (b_n/\\pi)
\\tilde P_n(\\mu)` (the :math:`/\\pi` normalisation comes from the
isotropic mode-0 Mark convention :math:`\\psi^{-} = J^{-}/\\pi`). The
sphere case integrates :math:`2\\int_0^\\pi \\sin\\theta\\,\\tilde P_n(
\\mu_{\\rm exit})\\,e^{-\\tau}\\,\\mathrm d\\theta`.

Specular boundary condition
---------------------------

At a curvilinear surface with outward normal :math:`\\hat n_b`, specular
reflection maps the outgoing direction :math:`\\Omega_{\\rm out}` to the
inward direction :math:`\\Omega_{\\rm in} = \\Omega_{\\rm out} -
2(\\Omega_{\\rm out}\\cdot\\hat n_b)\\hat n_b`. In terms of the absolute
cosine :math:`\\mu = |\\Omega \\cdot \\hat n_b| \\in [0, 1]`, the inward
and outward distributions are equal at the same :math:`\\mu`:

.. math::

    \\psi^{-}(r_b, \\mu) \\;=\\; \\psi^{+}(r_b, \\mu).

This is the **exact** angular-flux preservation; nothing is averaged or
truncated to a single mode. The truncation to :math:`N` modes happens
because the rank-:math:`N` operator only resolves the first :math:`N`
basis components of :math:`\\psi^{+}`.

Derivation
----------

Expand the outgoing flux in the shifted-Legendre basis,

.. math::

    \\psi^{+}(r_b, \\mu) \\;=\\; \\sum_{m=0}^{N-1} e_m\\,\\tilde P_m(\\mu).

Insert into the partial-current moment definition,

.. math::

    J^{+}_n \\;=\\; 2\\pi \\int_0^1 \\mu\\,\\tilde P_n(\\mu)
                  \\Bigl(\\sum_m e_m \\tilde P_m(\\mu)\\Bigr)\\mathrm d\\mu
              \\;=\\; 2\\pi\\sum_m M_{nm}\\, e_m,

so :math:`J^{+} = 2\\pi\\,M\\,e`. The specular condition
:math:`\\psi^{-} = \\psi^{+}` says the inward expansion coefficients are
equal to :math:`e_m`. The G-input is :math:`b_n = \\pi\\,e_n` (because
:math:`\\psi^{-} = \\sum_n (b_n/\\pi)\\tilde P_n` includes the
:math:`1/\\pi` isotropic-Mark normalisation), so

.. math::

    b \\;=\\; \\pi\\,e \\;=\\; \\pi \\cdot \\frac{1}{2\\pi}\\,M^{-1}\\,J^{+}
       \\;=\\; \\tfrac{1}{2}\\,M^{-1}\\,J^{+}.

Therefore :math:`R_{\\rm spec} = \\tfrac{1}{2}\\,M^{-1}`.

Closed form for M
-----------------

Substituting :math:`x = 2\\mu - 1` and using Legendre orthogonality plus
the recurrence :math:`x\\,P_n(x) = \\frac{n+1}{2n+1}\\,P_{n+1}(x)
+ \\frac{n}{2n+1}\\,P_{n-1}(x)`:

.. math::

    M_{nm} = \\tfrac{1}{2}\\Biggl[
        \\frac{1}{2n+1}\\,\\delta_{nm}
        + \\frac{n+1}{(2n+1)(2n+3)}\\,\\delta_{m,n+1}
        + \\frac{n}{(2n+1)(2n-1)}\\,\\delta_{m,n-1}
    \\Biggr].

So :math:`M` is symmetric tridiagonal in the shifted-Legendre basis.
The diagonal is :math:`M_{nn} = 1/(2(2n+1))` and the off-diagonal is
:math:`M_{n,n\\pm 1}` per the formula above.

Sanity-check ladder
-------------------

Rank :math:`N = 1`: :math:`M = [[1/2]]`,
:math:`M^{-1} = [[2]]`, :math:`R_{\\rm spec} = [[1]]`. This matches
:func:`reflection_mark` and :func:`reflection_marshak` exactly — the
trivial truncation collapses to the isotropic mode-0 white BC because
the only resolvable angular shape is constant.

Rank :math:`N = 2`: :math:`M = [[1/2, 1/6], [1/6, 1/6]]`, det :math:`= 1/18`,
:math:`M^{-1} = [[3, -3], [-3, 9]]`,
:math:`R_{\\rm spec} = [[3/2, -3/2], [-3/2, 9/2]]`. This is **dense**
(off-diagonal :math:`-3/2`), unlike :func:`reflection_marshak` which
gives the diagonal :math:`[[1, 0], [0, 3]]`. Specular ≠ Marshak DP_N
white from rank 2 onward — the specular closure couples mode 0 to mode
1 because the partial-current weight :math:`\\mu` mixes adjacent
shifted-Legendre indices.

Verify J⁻ = J⁺ contract
-----------------------

With :math:`R_{\\rm spec} = \\tfrac{1}{2} M^{-1}`, feed in :math:`J^{+}`,
get :math:`b = \\tfrac{1}{2} M^{-1} J^{+}`. The inward partial-current
moment is :math:`J^{-}_m = 2\\pi \\int_0^1 \\mu \\tilde P_m \\psi^{-}\\,
\\mathrm d\\mu = 2\\sum_n b_n M_{mn} = 2 (M b)_m =
2 \\cdot \\tfrac{1}{2}(M M^{-1} J^{+})_m = J^{+}_m`. ✓

So :math:`R_{\\rm spec}` enforces the exact **rank-N partial-current
identity** :math:`J^{-}_m = J^{+}_m` for :math:`m = 0,\\ldots,N-1`. By
the moment-determinacy of an :math:`N`-truncated angular flux in this
basis, this is precisely the specular condition projected onto the
:math:`N`-mode subspace.
"""

import numpy as np
import sympy as sp


def build_M_symbolic(n_modes: int) -> sp.Matrix:
    """Symbolic M_nm = int_0^1 mu * P_tilde_n(mu) * P_tilde_m(mu) d mu.

    Computed by direct SymPy integration over the shifted Legendre basis.
    Used as the ground truth that the analytical tridiagonal closed form
    is verified against.
    """
    mu = sp.symbols("mu")
    # Shifted Legendre P̃_n(mu) = P_n(2*mu - 1).
    P_tilde = [sp.legendre(n, 2 * mu - 1) for n in range(n_modes)]
    M = sp.zeros(n_modes, n_modes)
    for n in range(n_modes):
        for m in range(n_modes):
            integrand = mu * P_tilde[n] * P_tilde[m]
            M[n, m] = sp.integrate(integrand, (mu, 0, 1))
    return M


def build_M_closed_form(n_modes: int) -> sp.Matrix:
    """Tridiagonal closed form of M from Legendre orthogonality + x*P_n
    recurrence:

        M_{nm} = (1/2) [ delta_{nm}/(2n+1)
                         + (n+1)/((2n+1)(2n+3)) * delta_{m,n+1}
                         + n/((2n+1)(2n-1)) * delta_{m,n-1} ]
    """
    M = sp.zeros(n_modes, n_modes)
    for n in range(n_modes):
        M[n, n] = sp.Rational(1, 2) * sp.Rational(1, 2 * n + 1)
        if n + 1 < n_modes:
            val = sp.Rational(n + 1, (2 * n + 1) * (2 * n + 3)) / 2
            M[n, n + 1] = val
            M[n + 1, n] = val
    return M


def build_R_specular_symbolic(n_modes: int) -> sp.Matrix:
    """R_specular = (1/2) M^{-1}, symbolic."""
    M = build_M_closed_form(n_modes)
    return sp.Rational(1, 2) * M.inv()


def main() -> None:
    print(__doc__)
    print("=" * 70)
    print("Symbolic verification (closed form vs direct integration)")
    print("=" * 70)
    for N in (1, 2, 3, 4, 5):
        M_sym = build_M_symbolic(N)
        M_cf = build_M_closed_form(N)
        diff = sp.simplify(M_sym - M_cf)
        ok = diff == sp.zeros(N, N)
        print(f"\n--- N = {N} ---")
        print(f"M (symbolic integration) =")
        sp.pprint(M_sym)
        print(f"M (closed form)         =")
        sp.pprint(M_cf)
        print(f"Match: {ok}")
        assert ok, f"Closed form does not match symbolic at N={N}"

        R = build_R_specular_symbolic(N)
        print(f"R_specular(N={N}) =")
        sp.pprint(R)

        # Verify the J^- = J^+ contract: 2 M R = I.
        I = sp.eye(N)
        identity_check = sp.simplify(2 * M_cf * R - I)
        contract_ok = identity_check == sp.zeros(N, N)
        print(f"2 M R = I (J^- = J^+ contract): {contract_ok}")
        assert contract_ok, f"Partial-current contract fails at N={N}"

    print("\n" + "=" * 70)
    print("Numerical R_specular ladder (rank 1 to 6)")
    print("=" * 70)
    for N in range(1, 7):
        R = build_R_specular_symbolic(N)
        R_np = np.array(R.tolist(), dtype=float)
        print(f"\nN = {N}:")
        print(R_np)


if __name__ == "__main__":
    main()
