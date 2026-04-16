"""Verify the 1D SN solver: homogeneous exact, spatial O(h²), angular spectral."""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.derivations._xs_library import get_mixture
from orpheus.geometry import homogeneous_1d, slab_fuel_moderator
from orpheus.sn.quadrature import GaussLegendre1D
from orpheus.sn.solver import solve_sn

pytestmark = pytest.mark.verifies(
    "transport-cartesian",
    "dd-cartesian-1d",
    "dd-solve",
    "dd-recurrence",
    "multigroup",
    "reflective-bc",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
)


# ─── Homogeneous infinite medium (SN with reflective BCs) ────────────

@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_1rg",
    "sn_slab_2eg_1rg",
    "sn_slab_4eg_1rg",
])
def test_homogeneous_exact(case_name):
    """SN 1D with reflective BCs on a homogeneous slab must match
    the analytical infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    materials = {0: mix}
    mesh = homogeneous_1d(20, 2.0, mat_id=0)
    quad = GaussLegendre1D.create(8)
    result = solve_sn(materials, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    assert abs(result.keff - case.k_inf) < 1e-8, (
        f"keff={result.keff:.8f} vs analytical={case.k_inf:.8f}"
    )


# ─── Heterogeneous: replaced by MMS continuous reference ─────────────
#
# Phase 2.1a of the verification campaign removed the legacy
# ``test_heterogeneous_convergence`` test that consumed
# ``sn_slab_Neg_Nrg`` (N > 1) references from
# ``orpheus.derivations.sn._derive_sn_heterogeneous``. Those
# references were Richardson-extrapolated from the SN solver
# itself (T3 circular self-verification) and have been deleted.
#
# The new heterogeneous SN spatial-operator verification lives in
# ``tests/sn/test_mms_heterogeneous.py`` and consumes the
# ``sn_mms_slab_2g_hetero`` Phase-0 ContinuousReferenceSolution
# from ``orpheus.derivations.sn_mms`` — the Method of Manufactured
# Solutions with smooth cross sections. See the heterogeneous MMS
# section of ``docs/theory/discrete_ordinates.rst`` for why.
#
# The eigenvalue-heterogeneous verification that the deleted test
# was nominally covering (but did not actually verify, because it
# compared the solver to its own extrapolant) will be restored in
# Phase 2.1b by a Case singular-eigenfunction reference.


# ─── Spatial convergence O(h²) ───────────────────────────────────────

def _convergence_order(values, spacings, reference):
    """Compute observed convergence order between successive refinements."""
    orders = []
    for i in range(1, len(values)):
        err_prev = abs(values[i - 1] - reference)
        err_curr = abs(values[i] - reference)
        if err_prev > 0 and err_curr > 0:
            orders.append(
                np.log(err_prev / err_curr)
                / np.log(spacings[i - 1] / spacings[i])
            )
    return orders


@pytest.mark.l1
def test_spatial_convergence():
    """Diamond-difference scheme must show O(h²) spatial convergence."""
    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}
    t_fuel, t_mod = 0.5, 0.5

    keffs = []
    dxs = []
    for n_per in [5, 10, 20, 40]:
        mesh = slab_fuel_moderator(
            n_fuel=n_per, n_mod=n_per, t_fuel=t_fuel, t_mod=t_mod,
        )
        quad = GaussLegendre1D.create(16)
        result = solve_sn(
            materials, mesh, quad,
            max_outer=300, max_inner=500, inner_tol=1e-10,
        )
        keffs.append(result.keff)
        dxs.append(t_fuel / n_per)

    # Richardson extrapolation reference
    k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0
    orders = _convergence_order(keffs, dxs, k_ref)

    assert orders[-1] > 1.7, (
        f"Expected O(h²) convergence, got order {orders[-1]:.2f}"
    )


# ─── L0 term verification of the DD cumprod recurrence (ERR-025) ────

@pytest.mark.l0
@pytest.mark.catches("ERR-025")
def test_sweep_1d_cumprod_recurrence_matches_symbolic_derivation():
    """Term-level verification that ``_sweep_1d_cumprod``'s face-flux
    recurrence coefficients match the symbolic derivation in
    :func:`orpheus.derivations.sn_balance.derive_cumprod_recurrence`.

    Rationale for this as an L0 test: the legacy code diverged silently
    from the derivation (ERR-025). Both a wrong coefficient formula and
    its correct replacement satisfy every homogeneous-eigenvalue test
    the SN module had, because the Rayleigh quotient
    :math:`k = \\nu\\Sigma_f\\phi / \\Sigma_a\\phi` is invariant under a
    uniform rescaling of :math:`\\phi`. An L0 test that directly probes
    the recurrence — not through eigenvalue machinery — is the minimal
    isolation of this failure mode.

    Strategy: substitute numerical values into the symbolic
    ``(a, b)`` expressions returned by ``derive_cumprod_recurrence()``,
    run ``_sweep_1d_cumprod`` on a 1-cell homogeneous slab with a
    controlled boundary inflow and a uniform source, and check that
    the returned cell-average angular flux matches the expected
    closed-form value

        .. math::

            \\psi_{\\text{cell}} = \\tfrac12 \\bigl(\\psi_{\\text{in}} +
            a\\psi_{\\text{in}} + b\\,Q / W\\bigr)

    up to 12 digits, for each positive ordinate independently. Any
    drift in either coefficient (sign flip, factor-of-two, missing
    :math:`1/W` normalization) is caught directly. This is a **white-box**
    check on the coefficient formula — it intentionally duplicates the
    derivation inside the test so that a future edit to the sweep code
    cannot silently drift without the symbolic source complaining.
    """
    import sympy as sp
    from orpheus.derivations.sn_balance import derive_cumprod_recurrence
    from orpheus.geometry import CoordSystem, Mesh1D
    from orpheus.sn.geometry import SNMesh
    from orpheus.sn.sweep import _sweep_1d_cumprod

    # Symbolic coefficients, captured silently.
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        a_sym, b_sym = derive_cumprod_recurrence()
    # a_sym = (2μ/Δx − Σt)/(2μ/Δx + Σt); b_sym = 2S/(2μ/Δx + Σt)
    # where S is the per-ordinate source (already divided by W).

    mu_sym, dx_sym, Sig_t_sym, S_sym = sp.symbols(
        "mu dx Sigma_t S", positive=True
    )

    # Minimal 1-cell, 1-group slab. N=4 gives us two independent positive
    # ordinates, so the test exercises the vectorised broadcast over
    # ordinates in addition to the scalar formula.
    ng = 1
    sig_t_val = 1.5  # arbitrary, non-trivial
    dx_val = 0.7     # arbitrary, non-trivial
    Q_val = 3.0      # arbitrary scalar-flux-units source

    quad = GaussLegendre1D.create(4)
    edges = np.array([0.0, dx_val])
    mesh = Mesh1D(
        edges=edges,
        mat_ids=np.zeros(1, dtype=int),
        coord=CoordSystem.CARTESIAN,
    )
    sn_mesh = SNMesh(mesh, quad)

    sig_t = np.full((1, 1, ng), sig_t_val)
    Q = np.full((1, 1, ng), Q_val)
    W = quad.weights.sum()
    n_half = quad.N // 2
    mu_pos = np.abs(quad.mu_x[n_half:])

    # Test-only controlled left-edge inflow: a distinct value per
    # positive ordinate so that any broadcasting error would show up.
    psi_in_left = np.array([[0.4], [0.9]])  # (n_half, ng)

    psi_bc = {"bc_1d": {
        "left": psi_in_left.copy(),
        "right": np.zeros((n_half, ng)),
    }}

    angular_flux, _ = _sweep_1d_cumprod(Q, sig_t, sn_mesh, psi_bc)

    # For each positive ordinate, compute the expected forward-sweep
    # cell-average from the symbolic derivation.
    for n in range(n_half):
        mu_val = float(mu_pos[n])
        a_num = float(a_sym.subs({mu_sym: mu_val, dx_sym: dx_val,
                                  Sig_t_sym: sig_t_val}))
        b_num = float(b_sym.subs({mu_sym: mu_val, dx_sym: dx_val,
                                  Sig_t_sym: sig_t_val,
                                  S_sym: Q_val / W}))
        psi_in = float(psi_in_left[n, 0])
        psi_out_expected = a_num * psi_in + b_num
        cell_avg_expected = 0.5 * (psi_in + psi_out_expected)

        # angular_flux shape is (N, nx=1, ny=1, ng=1). Positive
        # ordinate n lives at index n_half + n.
        cell_avg_code = float(angular_flux[n_half + n, 0, 0, 0])
        assert abs(cell_avg_code - cell_avg_expected) < 1e-12, (
            f"Ordinate n={n} (μ={mu_val:.6f}): "
            f"sweep gave {cell_avg_code:.10e}, "
            f"derivation gives {cell_avg_expected:.10e}, "
            f"Δ={cell_avg_code - cell_avg_expected:+.2e}. "
            "_sweep_1d_cumprod does not match "
            "sn_balance.derive_cumprod_recurrence."
        )


# ─── Heterogeneous absolute eigenvalue regression (ERR-025) ──────────

@pytest.mark.l1
@pytest.mark.catches("ERR-025")
def test_heterogeneous_absolute_keff():
    """2-region A+B reflective slab must match external references.

    Regression for the DD face-flux recurrence bug (ERR-025): the
    legacy ``_sweep_1d_cumprod`` used the wrong recurrence coefficients

        a = 2μ / (2μ + Δx·Σ_t)          (wrong)
        s = 0.5·Δx·Q / (2μ + Δx·Σ_t)    (wrong)

    instead of those derived in
    ``orpheus.derivations.sn_balance.derive_cumprod_recurrence``

        a = (2μ − Δx·Σ_t) / (2μ + Δx·Σ_t)
        b = 2·Δx·(Q/W) / (2μ + Δx·Σ_t)

    Both wrongs are off by a factor of two in opposite directions and
    cancelled for eigenvalue problems with a single material (a scale
    on φ leaves k = ν·Σ_f·φ / Σ_a·φ invariant). At material interfaces
    the cancellation broke because the factor depended on Σ_t(x),
    shifting k_eff by ~1.4e-2.

    Two independent references — a Case singular-eigenfunction
    expansion and the ORPHEUS CP slab solver (E₃ kernel) — agree on
    k ≈ 1.27461 for this config, against which the fixed DD
    recurrence now matches to 5e-5 at n_per=320.
    """
    fuel = get_mixture("A", "1g")  # Σ_t=1, Σ_s=0.5, νΣ_f=0.75 → k_inf=1.5
    mod = get_mixture("B", "1g")   # Σ_t=2, Σ_s=1.9, νΣ_f=0
    materials = {0: fuel, 1: mod}

    n_per = 320
    edges = np.linspace(0.0, 1.0, 2 * n_per + 1)
    mat_ids = np.array([0] * n_per + [1] * n_per)
    from orpheus.geometry import Mesh1D, CoordSystem
    mesh = Mesh1D(edges=edges, mat_ids=mat_ids, coord=CoordSystem.CARTESIAN)
    quad = GaussLegendre1D.create(8)

    result = solve_sn(
        materials, mesh, quad,
        max_outer=500, max_inner=500,
        keff_tol=1e-11, inner_tol=1e-11,
    )

    # Case singular-eigenfunction reference (16x16 matching matrix,
    # independently cross-checked against CP E₃ to 2e-4).
    k_ref = 1.27461604
    assert abs(result.keff - k_ref) < 5e-4, (
        f"keff={result.keff:.8f} vs Case reference={k_ref:.8f} "
        f"(Δ={result.keff - k_ref:+.2e})"
    )


# ─── Angular spectral convergence ────────────────────────────────────

@pytest.mark.l1
def test_angular_convergence():
    """Gauss-Legendre quadrature must show spectral convergence in angle."""
    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}

    keffs = []
    n_ords = [4, 8, 16, 32]
    for N in n_ords:
        mesh = slab_fuel_moderator(
            n_fuel=40, n_mod=40, t_fuel=0.5, t_mod=0.5,
        )
        quad = GaussLegendre1D.create(N)
        result = solve_sn(
            materials, mesh, quad,
            max_outer=300, max_inner=500, inner_tol=1e-10,
        )
        keffs.append(result.keff)

    k_ref = keffs[-1]
    orders = _convergence_order(keffs, [1 / N for N in n_ords], k_ref)
    assert len(orders) >= 2
    assert max(orders[:-1]) > 1.5, (
        f"Expected spectral convergence, got orders {orders}"
    )
