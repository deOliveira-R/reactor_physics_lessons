"""Verify the infinite-medium eigenvalue solver against SymPy analytical solutions."""

import pytest

from orpheus.derivations import get
from orpheus.homogeneous.solver import solve_homogeneous_infinite

# File-level verifies marker: every test in this file exercises the
# homogeneous eigenvalue chain end-to-end by asserting k_inf matches
# the SymPy-derived analytical reference to 1e-12. That tolerance is
# tight enough to pin every step of the derivation — if any of the
# labelled equations below were implemented incorrectly the k mismatch
# would be far larger than 1e-12, so a passing test is equation-level
# (L1) verification for every link in the chain.
#
# Declared explicitly here (rather than inherited from
# VerificationCase) so the Nexus AST pass picks it up via decorator
# parsing and writes TESTS edges.
#
# The 2G labels (two-group-*) and the power-iteration step labels
# (fission-source, fixed-source-solve, keff-update) are all exercised
# by the homo_2eg / homo_4eg cases — the analytical k_inf is derived
# symbolically via exactly those equations, so a solver k that matches
# to 1e-12 implies every link in the chain is correct. The absorption-
# xs label is the derived property used inside keff-update.
pytestmark = [pytest.mark.l1, pytest.mark.verifies(
    "one-group-kinf",
    "inf-hom-balance",
    "matrix-eigenvalue",
    "removal-matrix",
    "fission-matrix",
    "mg-balance",
    # B.1 additions (issue #87): the full 2G analytical chain and the
    # power-iteration step labels, all verified end-to-end by the
    # homo_2eg and homo_4eg parametric cases.
    "two-group-A",
    "two-group-F",
    "two-group-Ainv",
    "two-group-M",
    "two-group-charpoly",
    "two-group-roots",
    "fission-source",
    "fixed-source-solve",
    "keff-update",
    "absorption-xs",
)]


@pytest.mark.parametrize("case_name", [
    "homo_1eg",
    "homo_2eg",
    "homo_4eg",
])
def test_kinf_exact(case_name):
    """Eigenvalue must match analytical solution to machine precision."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    assert abs(result.k_inf - case.k_inf) < 1e-12, (
        f"k_inf mismatch: solver={result.k_inf:.10f} "
        f"analytical={case.k_inf:.10f}"
    )


@pytest.mark.verifies("normalisation")
def test_post_solve_production_rate_is_100():
    """L1: post-convergence flux is normalised to 100 n/cm^3/s production.

    After :func:`solve_homogeneous_infinite` converges, the flux is
    rescaled so the total production rate

    .. math::

       (\\Sigma_\\mathrm{p} + 2 \\cdot \\text{colsum}(\\Sigma_2))
       \\cdot \\boldsymbol{\\phi} = 100

    (see Eq. ``normalisation`` in docs/theory/homogeneous.rst). This
    test pins that invariant against the 2G and 4G cases — the 1G
    case is a degenerate one-scalar normalisation that a bug could
    accidentally satisfy.
    """
    import numpy as np

    for case_name in ("homo_2eg", "homo_4eg"):
        case = get(case_name)
        mix = next(iter(case.materials.values()))
        result = solve_homogeneous_infinite(mix)

        # Production = (SigP + 2 * colsum(Sig2)) @ phi
        sig_p = mix.SigP
        n2n_colsum = np.array(mix.Sig2.sum(axis=0)).ravel() if mix.Sig2 is not None else 0.0
        production = (sig_p + 2.0 * n2n_colsum) @ result.flux

        assert abs(production - 100.0) < 1e-9, (
            f"{case_name}: production rate = {production:.6e}, "
            f"expected 100.0 (normalisation constraint)"
        )
