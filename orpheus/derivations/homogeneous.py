r"""Analytical reference solutions for the infinite homogeneous medium.

For an infinite homogeneous medium with reflective boundary
conditions the neutron transport equation is degenerate in space —
the flux is spatially flat, and :math:`k_\infty` is the dominant
eigenvalue of :math:`\mathbf{A}^{-1}\mathbf{F}` where

.. math::

    \mathbf{A} = \text{diag}(\Sigma_t) - (\Sigma_s + 2\Sigma_2)^{T},
    \qquad
    \mathbf{F} = \chi \otimes (\nu\Sigma_f).

This module publishes **both** tiers of reference for 1-group,
2-group, and 4-group test problems:

1. **Legacy** :class:`~orpheus.derivations._types.VerificationCase`
   — scalar :math:`k_\infty` only, consumed by the existing legacy
   tests.
2. **Phase-0 continuous**
   :class:`~orpheus.derivations.ContinuousReferenceSolution` —
   carries :math:`k_\infty` **and** the continuous
   :math:`\phi(x, g)` callable (spatially flat, equal to the
   :math:`\ell^{2}`-normalised dominant eigenvector of
   :math:`\mathbf{A}^{-1}\mathbf{F}`) so consumer tests can verify
   flux shape as well as the eigenvalue.

The homogeneous cases are the **simplest possible verification
reference** — no ansatz, no quadrature, no iteration. They land
first in the Phase-1 retrofit plan because they establish the
retrofit pattern (``derive_*`` returns a legacy case alongside
a continuous reference; ``continuous_cases()`` enumerates the
new-form list) without any new numerical machinery.

See :doc:`/verification/reference_solutions` for the contract and
the verification-campaign migration plan.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture

from ._eigenvalue import kinf_and_spectrum_homogeneous, kinf_homogeneous
from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)
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


def _build_continuous_homogeneous(
    name: str,
    description: str,
    materials: dict[int, Mixture],
    k_val: float,
    phi_spectrum: np.ndarray,
    provenance: Provenance,
    equation_labels: tuple[str, ...],
) -> ContinuousReferenceSolution:
    r"""Assemble a :class:`ContinuousReferenceSolution` for a homogeneous case.

    The spatial callable ``phi(x, g)`` returns the constant
    ``phi_spectrum[g]`` broadcast over the shape of ``x``. This is
    the honest-to-mathematics continuous solution for the infinite
    homogeneous medium: the flux is spatially flat, so the
    "reference at any point" is literally the same scalar.

    The problem is registered under ``operator_form="homogeneous"``
    — the one operator tag in the taxonomy that is degenerate in
    space and can be consumed by every solver as a sanity check on
    multigroup matrix algebra (which is exactly what the homogeneous
    tests do).
    """
    ng = len(phi_spectrum)

    def phi(x: np.ndarray, g: int = 0) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), phi_spectrum[g])

    return ContinuousReferenceSolution(
        name=name,
        problem=ProblemSpec(
            materials=materials,
            geometry_type="homogeneous",
            geometry_params={},
            boundary_conditions={},  # infinite medium — no boundary
            external_source=None,
            is_eigenvalue=True,
            n_groups=ng,
        ),
        operator_form="homogeneous",
        phi=phi,
        provenance=provenance,
        k_eff=k_val,
        psi=None,  # angular flux is isotropic by symmetry; not needed
        equation_labels=equation_labels,
        vv_level="L1",
        description=description,
        tolerance="< 1e-12",
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

    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
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
        vv_level="L1",
        equation_labels=("one-group-kinf", "inf-hom-balance"),
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

    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
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
        vv_level="L1",
        equation_labels=(
            "matrix-eigenvalue",
            "removal-matrix",
            "fission-matrix",
            "mg-balance",
            "two-group-A",
            "two-group-F",
            "two-group-Ainv",
            "two-group-M",
            "two-group-charpoly",
            "two-group-roots",
        ),
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

    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
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
        vv_level="L1",
        equation_labels=(
            "matrix-eigenvalue",
            "removal-matrix",
            "fission-matrix",
            "mg-balance",
        ),
    )


def all_cases() -> list[VerificationCase]:
    """Return all homogeneous legacy :class:`VerificationCase` objects."""
    return [derive_1g(), derive_2g(), derive_4g()]


# ═══════════════════════════════════════════════════════════════════════
# Phase-0 continuous references
# ═══════════════════════════════════════════════════════════════════════
#
# Each ``derive_Ng_continuous`` builds the same mathematical problem
# as the legacy ``derive_Ng`` above but returns a
# :class:`ContinuousReferenceSolution` carrying both ``k_eff`` and
# the flat spatial flux ``phi(x, g) = phi_spectrum[g]``. The
# spectrum is the :math:`\ell^{2}`-normalised dominant eigenvector
# of :math:`\mathbf{A}^{-1}\mathbf{F}` computed inside
# :func:`kinf_and_spectrum_homogeneous`.
#
# Intentional duplication with the legacy ``derive_Ng``: Phase 1.1
# keeps the legacy call paths untouched so nothing in
# ``tests/homogeneous/`` has to change. Phase 2 may fold them into
# the continuous derivations once every consumer has migrated.


def _derive_1g_inputs():
    """Shared cross sections for the 1-group homogeneous reference."""
    return dict(sig_t=1.0, sig_c=0.2, sig_f=0.3, nu=2.5, sig_s_diag=0.5)


def _derive_2g_inputs():
    """Shared cross sections for the 2-group homogeneous reference."""
    return dict(
        sig_t=np.array([0.50, 1.00]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.50, 2.50]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
    )


def _derive_4g_inputs():
    """Shared cross sections for the 4-group homogeneous reference."""
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
    return dict(
        sig_t=sig_t, sig_c=sig_c, sig_f=sig_f,
        nu=nu, chi=chi, sig_s=sig_s,
    )


def derive_1g_continuous() -> ContinuousReferenceSolution:
    r"""Phase-0 continuous reference for the 1-group homogeneous medium.

    The 1-group case is the single cleanest verification reference
    in the whole campaign: the eigenvalue is
    :math:`k_\infty = \nu\Sigma_f / \Sigma_a` in closed form, the
    flux is spatially flat and — with only one group — identically
    equal to 1 after :math:`\ell^{2}` normalisation. A consumer
    test that does not reproduce this to machine precision has a
    bug in its multigroup balance, not in the discretisation.
    """
    xs = _derive_1g_inputs()
    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
        sig_t=np.array([xs["sig_t"]]),
        sig_s=np.array([[xs["sig_s_diag"]]]),
        nu_sig_f=np.array([xs["nu"] * xs["sig_f"]]),
        chi=np.array([1.0]),
    )
    mix = _make_mixture(
        sig_t=np.array([xs["sig_t"]]),
        sig_c=np.array([xs["sig_c"]]),
        sig_f=np.array([xs["sig_f"]]),
        nu=np.array([xs["nu"]]),
        chi=np.array([1.0]),
        sig_s=np.array([[xs["sig_s_diag"]]]),
    )
    return _build_continuous_homogeneous(
        name="homo_1eg",
        description=(
            "1-group infinite medium — k_inf = nu·Sigma_f / Sigma_a. "
            "Phase-0 continuous reference."
        ),
        materials={0: mix},
        k_val=k_val,
        phi_spectrum=phi_spectrum,
        provenance=Provenance(
            citation="Bell & Glasstone 1970, §1.5 Eq. (1.58)",
            derivation_notes=(
                "One-group infinite homogeneous medium with reflective BC. "
                "The transport equation reduces to the scalar balance "
                "Sigma_a·phi = (1/k)·nu·Sigma_f·phi, giving the closed-form "
                "k_inf = nu·Sigma_f / Sigma_a. With a single group the "
                "spectrum is degenerate (phi = 1 after l2 normalisation)."
            ),
            sympy_expression=r"k_\infty = \nu \Sigma_f / \Sigma_a",
            precision_digits=None,  # closed form
        ),
        equation_labels=("one-group-kinf", "inf-hom-balance"),
    )


def derive_2g_continuous() -> ContinuousReferenceSolution:
    r"""Phase-0 continuous reference for the 2-group homogeneous medium.

    The flux spectrum is the non-trivial dominant eigenvector of
    the 2×2 matrix :math:`\mathbf{A}^{-1}\mathbf{F}`. This is the
    **first** reference where a consumer test has something to
    verify beyond :math:`k_{\text{eff}}` — namely the fast/thermal
    split of the converged multigroup flux.
    """
    xs = _derive_2g_inputs()
    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
        sig_t=xs["sig_t"], sig_s=xs["sig_s"],
        nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
    )
    mix = _make_mixture(
        xs["sig_t"], xs["sig_c"], xs["sig_f"],
        xs["nu"], xs["chi"], xs["sig_s"],
    )
    return _build_continuous_homogeneous(
        name="homo_2eg",
        description=(
            "2-group infinite medium (fast + thermal, downscatter only). "
            "Phase-0 continuous reference."
        ),
        materials={0: mix},
        k_val=k_val,
        phi_spectrum=phi_spectrum,
        provenance=Provenance(
            citation="Bell & Glasstone 1970, §7.4",
            derivation_notes=(
                "Two-group reflective infinite medium. The eigenvalue "
                "problem is A·phi = (1/k)·F·phi with A = diag(Sigma_t) - "
                "Sigma_s^T and F = chi ⊗ (nu·Sigma_f). Solved via "
                "characteristic polynomial (closed form for 2x2); "
                "the l2-normalised dominant eigenvector is the flux "
                "spectrum."
            ),
            sympy_expression=None,
            precision_digits=None,
        ),
        equation_labels=(
            "matrix-eigenvalue", "removal-matrix", "fission-matrix",
            "mg-balance", "two-group-A", "two-group-F", "two-group-Ainv",
            "two-group-M", "two-group-charpoly", "two-group-roots",
        ),
    )


def derive_4g_continuous() -> ContinuousReferenceSolution:
    r"""Phase-0 continuous reference for the 4-group homogeneous medium.

    Downscatter cascade with fission in all groups — the dominant
    eigenvector is the first genuinely non-trivial flux spectrum
    in the Phase-0 registry (1g degenerate, 2g has only two
    components). Used to detect bugs in multigroup chi-weighting
    that single out higher-order cascades.
    """
    xs = _derive_4g_inputs()
    k_val, phi_spectrum = kinf_and_spectrum_homogeneous(
        sig_t=xs["sig_t"], sig_s=xs["sig_s"],
        nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
    )
    mix = _make_mixture(
        xs["sig_t"], xs["sig_c"], xs["sig_f"],
        xs["nu"], xs["chi"], xs["sig_s"],
    )
    return _build_continuous_homogeneous(
        name="homo_4eg",
        description=(
            "4-group infinite medium (downscatter cascade, fission "
            "in all groups). Phase-0 continuous reference."
        ),
        materials={0: mix},
        k_val=k_val,
        phi_spectrum=phi_spectrum,
        provenance=Provenance(
            citation="Bell & Glasstone 1970, §7.4 (multigroup form)",
            derivation_notes=(
                "Four-group reflective infinite medium with downscatter "
                "cascade. Dense 4x4 A^{-1}·F solved via numpy.linalg.eig; "
                "the dominant eigenvector gives the non-trivial flux "
                "cascade shape."
            ),
            sympy_expression=None,
            precision_digits=None,
        ),
        equation_labels=(
            "matrix-eigenvalue", "removal-matrix", "fission-matrix",
            "mg-balance",
        ),
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Return all homogeneous Phase-0 continuous reference solutions."""
    return [
        derive_1g_continuous(),
        derive_2g_continuous(),
        derive_4g_continuous(),
    ]
