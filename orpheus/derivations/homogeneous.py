"""SymPy derivations for infinite homogeneous medium eigenvalues.

Derives analytical k_inf for 1-group, 2-group, and 4-group systems
using symbolic matrix algebra, then evaluates numerically.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture

from ._eigenvalue import kinf_homogeneous
from ._types import VerificationCase


def _make_mixture(
    sig_t: np.ndarray,
    sig_c: np.ndarray,
    sig_f: np.ndarray,
    nu: np.ndarray,
    chi: np.ndarray,
    sig_s: np.ndarray,
) -> Mixture:
    """Build a Mixture from N-group arrays."""
    ng = len(sig_t)
    eg = np.logspace(7, -3, ng + 1)
    return Mixture(
        SigC=sig_c.copy(), SigL=np.zeros(ng),
        SigF=sig_f.copy(), SigP=(nu * sig_f).copy(),
        SigT=sig_t.copy(), SigS=[csr_matrix(sig_s)],
        Sig2=csr_matrix((ng, ng)), chi=chi.copy(), eg=eg.copy(),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1-group: k = nu * Sigma_f / Sigma_a  (fully symbolic)
# ═══════════════════════════════════════════════════════════════════════

def derive_1g() -> VerificationCase:
    r"""1-group infinite medium eigenvalue.

    .. math::
        k_\infty = \frac{\nu \Sigma_f}{\Sigma_a}
    """
    nu_s, Sig_f, Sig_a = sp.symbols(r'\nu \Sigma_f \Sigma_a', positive=True)

    k_expr = nu_s * Sig_f / Sig_a

    # Numeric XS
    xs = dict(sig_t=1.0, sig_c=0.2, sig_f=0.3, nu=2.5, sig_s_diag=0.5)
    sig_a_val = xs["sig_c"] + xs["sig_f"]

    k_val = kinf_homogeneous(
        sig_t=np.array([xs["sig_t"]]),
        sig_s=np.array([[xs["sig_s_diag"]]]),
        nu_sig_f=np.array([xs["nu"] * xs["sig_f"]]),
        chi=np.array([1.0]),
    )

    latex = (
        r"For a 1-group infinite homogeneous medium, the neutron balance is:"
        "\n\n"
        r".. math::" "\n"
        r"   \Sigma_a \phi = \frac{1}{k} \nu \Sigma_f \phi"
        "\n\n"
        r"which gives the analytical eigenvalue:"
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {sp.latex(k_expr)} = "
        rf"\frac{{{xs['nu']} \times {xs['sig_f']}}}"
        rf"{{{sig_a_val}}} = {k_val:.6f}"
    )

    mix = _make_mixture(
        sig_t=np.array([xs["sig_t"]]),
        sig_c=np.array([xs["sig_c"]]),
        sig_f=np.array([xs["sig_f"]]),
        nu=np.array([xs["nu"]]),
        chi=np.array([1.0]),
        sig_s=np.array([[xs["sig_s_diag"]]]),
    )

    return VerificationCase(
        name="homo_1eg",
        k_inf=k_val,
        method="homo",
        geometry="--",
        n_groups=1,
        n_regions=1,
        materials={0: mix},
        geom_params={},
        latex=latex,
        description="1-group infinite medium: k = nu*Sig_f / Sig_a = 1.5",
        tolerance="< 1e-12",
    )


# ═══════════════════════════════════════════════════════════════════════
# 2-group: solve characteristic polynomial of 2x2 inv(A)*F
# ═══════════════════════════════════════════════════════════════════════

def derive_2g() -> VerificationCase:
    r"""2-group infinite medium eigenvalue via characteristic polynomial.

    The eigenvalue problem is :math:`\mathbf{A}\phi = \frac{1}{k}\mathbf{F}\phi`
    where :math:`\mathbf{A} = \text{diag}(\Sigma_t) - \Sigma_s^T` and
    :math:`\mathbf{F} = \chi \otimes (\nu\Sigma_f)`.
    """
    # Numeric XS (same as benchmarks.py)
    sig_t = np.array([0.50, 1.00])
    sig_c = np.array([0.01, 0.02])
    sig_f = np.array([0.01, 0.08])
    nu = np.array([2.50, 2.50])
    chi = np.array([1.00, 0.00])
    sig_s = np.array([
        [0.38, 0.10],
        [0.00, 0.90],
    ])

    # Symbolic derivation
    A_sym = sp.Matrix([
        [sig_t[0] - sig_s[0, 0], -sig_s[1, 0]],
        [-sig_s[0, 1], sig_t[1] - sig_s[1, 1]],
    ])
    F_sym = sp.Matrix([
        [chi[0] * nu[0] * sig_f[0], chi[0] * nu[1] * sig_f[1]],
        [chi[1] * nu[0] * sig_f[0], chi[1] * nu[1] * sig_f[1]],
    ])

    k_val = kinf_homogeneous(
        sig_t=sig_t, sig_s=sig_s, nu_sig_f=nu * sig_f, chi=chi,
    )

    latex = (
        r"For 2-group infinite medium with downscatter only:"
        "\n\n"
        r".. math::" "\n"
        rf"   \mathbf{{A}} = {sp.latex(A_sym)}, \quad "
        rf"\mathbf{{F}} = {sp.latex(F_sym)}"
        "\n\n"
        r"The eigenvalue is the dominant root of "
        r":math:`\det(\mathbf{A}^{-1}\mathbf{F} - \lambda \mathbf{I}) = 0`:"
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {k_val:.10f}"
    )

    mix = _make_mixture(sig_t, sig_c, sig_f, nu, chi, sig_s)
    return VerificationCase(
        name="homo_2eg",
        k_inf=k_val,
        method="homo",
        geometry="--",
        n_groups=2,
        n_regions=1,
        materials={0: mix},
        geom_params={},
        latex=latex,
        description="2-group infinite medium (fast + thermal, downscatter only)",
        tolerance="< 1e-12",
    )


# ═══════════════════════════════════════════════════════════════════════
# 4-group: symbolic matrix structure, numeric eigenvalue
# ═══════════════════════════════════════════════════════════════════════

def derive_4g() -> VerificationCase:
    r"""4-group infinite medium eigenvalue.

    Symbolic 4x4 matrix with numeric XS substituted before solving.
    """
    sig_c = np.array([0.01, 0.02, 0.03, 0.05])
    sig_f = np.array([0.005, 0.01, 0.05, 0.10])
    nu = np.array([2.80, 2.60, 2.50, 2.45])
    chi = np.array([0.60, 0.35, 0.05, 0.00])
    sig_s = np.array([
        [0.28, 0.08, 0.02, 0.005],
        [0.00, 0.40, 0.12, 0.06],
        [0.00, 0.00, 0.55, 0.22],
        [0.00, 0.00, 0.00, 0.90],
    ])
    sig_t = sig_c + sig_f + sig_s.sum(axis=1)

    k_val = kinf_homogeneous(
        sig_t=sig_t, sig_s=sig_s, nu_sig_f=nu * sig_f, chi=chi,
    )

    latex = (
        r"For 4-group infinite medium (fast :math:`\to` epithermal "
        r":math:`\to` thermal1 :math:`\to` thermal2), "
        r"with downscatter cascade and fission in all groups:"
        "\n\n"
        r".. math::" "\n"
        rf"   \mathbf{{A}} = \text{{diag}}(\Sigma_t) - \Sigma_s^T \in "
        rf"\mathbb{{R}}^{{4 \times 4}}"
        "\n\n"
        r".. math::" "\n"
        rf"   \mathbf{{F}} = \chi \otimes (\nu\Sigma_f) \in "
        rf"\mathbb{{R}}^{{4 \times 4}}"
        "\n\n"
        r"The dominant eigenvalue of :math:`\mathbf{A}^{-1}\mathbf{F}`:"
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {k_val:.10f}"
    )

    mix = _make_mixture(sig_t, sig_c, sig_f, nu, chi, sig_s)
    return VerificationCase(
        name="homo_4eg",
        k_inf=k_val,
        method="homo",
        geometry="--",
        n_groups=4,
        n_regions=1,
        materials={0: mix},
        geom_params={},
        latex=latex,
        description="4-group infinite medium (downscatter cascade)",
        tolerance="< 1e-12",
    )


def all_cases() -> list[VerificationCase]:
    """Return all homogeneous verification cases."""
    return [derive_1g(), derive_2g(), derive_4g()]
