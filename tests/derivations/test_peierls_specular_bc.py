r"""End-to-end verification of the specular boundary condition.

Created 2026-04-27 as part of the specular-BC implementation
(plan: ``.claude/plans/specular-bc-method-of-images.md``, Phase 1).

The specular BC is :math:`\psi^{-}(r_b, \mu_{\rm in}) = \psi^{+}(r_b,
\mu_{\rm in})` — exact angular-flux preservation at the surface, with
no isotropic-uniformity averaging (unlike Mark / Marshak / Hébert
white BC closures, which all approximate the surface as having
isotropic re-entry).

In the rank-:math:`N` Marshak shifted-Legendre basis, this becomes
:math:`R_{\rm spec} = \tfrac{1}{2}\,M^{-1}` with :math:`M_{nm} =
\int_0^1 \mu\,\tilde P_n(\mu)\,\tilde P_m(\mu)\,\mathrm d\mu`,
symmetric tridiagonal in the basis (closed form derived in
``derivations/peierls_specular_bc.py`` and verified by SymPy at
:math:`N = 1,\ldots,5`). The construction satisfies the rank-:math:`N`
**partial-current identity** :math:`J^{-}_m = J^{+}_m` for all
:math:`m = 0,\ldots,N-1`.

This test file checks the **end-to-end pipeline** of the
``boundary="specular"`` closure through :func:`solve_peierls_1g`:

A. **Rank-1 reduction** — at :math:`N = 1` the dense :math:`R_{\rm
   spec}` collapses to :math:`[[1]]`, identical to Mark / Marshak.
   The K_bc matrix bit-equals the rank-1 Mark closure (because the
   no-Jacobian P primitive used inside the specular branch reduces
   exactly to ``compute_P_esc`` at :math:`n = 0`).

B. **Homogeneous convergence to k_inf (sphere)** — for a homogeneous
   sphere with white BC, the specular eigenvector is uniform
   (rotational symmetry forces the surface flux to be isotropic), so
   :math:`k_{\rm eff} \to k_\infty` as :math:`N \to \infty`. This
   test verifies the convergence ladder for :math:`N = 1, 2, 3, 4`
   and asserts the rank-4 result is within 0.5 % of :math:`k_\infty`
   under BASE quadrature.

C. **Cylinder caveat** — the cylinder ``compute_G_bc_mode`` primitive
   uses the surface-centred :math:`{\rm Ki}_1/d` form which has a
   documented 3-D-vs-2-D angular normalisation issue (Issue #112
   Phase C — the same Knyazev correction that fixes
   ``boundary="white_hebert"`` for cylinder). Specular at rank-1
   inherits the rank-1 Mark calibration limit; at higher ranks the
   off-diagonal R_spec coupling amplifies the Knyazev mismatch.
   Test C documents this as a known limitation rather than asserting
   convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations._eigenvalue import kinf_and_spectrum_homogeneous
from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SLAB_POLAR_1D,
    SPHERE_1D,
    solve_peierls_1g,
    solve_peierls_mg,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture: homogeneous fuel-A 1G XS, cell radius R = 5 cm.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def homogeneous_fuel_A_1G():
    xs = get_xs("A", "1g")
    sigt = xs["sig_t"]
    sigs = xs["sig_s"]
    nuf = xs["nu"] * xs["sig_f"]
    k_inf = float(nuf[0]) / (float(sigt[0]) - float(sigs[0, 0]))
    radii = np.array([5.0])
    return {
        "sig_t": sigt,
        "sig_s": sigs,
        "nu_sig_f": nuf,
        "k_inf": k_inf,
        "radii": radii,
    }


def _solve(geometry, fixture, *, n_bc_modes, boundary):
    return solve_peierls_1g(
        geometry,
        fixture["radii"],
        fixture["sig_t"],
        fixture["sig_s"],
        fixture["nu_sig_f"],
        boundary=boundary,
        n_bc_modes=n_bc_modes,
        p_order=4,
        n_panels_per_region=2,
        n_angular=24,
        n_rho=24,
        n_surf_quad=24,
        dps=20,
        tol=1e-10,
    )


# ──────────────────────────────────────────────────────────────────────
# A. Rank-1 reduction: specular at N=1 equals Mark
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_specular_rank1_sphere_equals_mark_kinf(homogeneous_fuel_A_1G):
    r"""At rank-1 specular for sphere has :math:`R = [[1]] = R_{\rm Mark}`
    AND uses the legacy :func:`compute_P_esc` mode-0 primitive (since
    :math:`\tilde P_0 = 1` and the no-Jacobian sphere P primitive
    reduces to :func:`compute_P_esc`), so the K_bc matrices are
    identical and :math:`k_{\rm eff}` matches bit-exactly.

    Cylinder rank-1 specular does NOT match rank-1 Mark anymore
    (since the Knyazev :math:`\mathrm{Ki}_{2+k}` correction was
    integrated for cylinder rank-N specular — see
    :func:`compute_G_bc_cylinder_3d_mode`). At :math:`n = 0` the
    cylinder primitive equals the 3-D-correct
    :func:`compute_G_bc_cylinder_3d` (which is what
    ``boundary="white_hebert"`` uses), and rank-1 specular k_eff
    closely matches rank-1 white_hebert, NOT rank-1 Mark.
    """
    sol_spec = _solve(
        SPHERE_1D, homogeneous_fuel_A_1G,
        n_bc_modes=1, boundary="specular",
    )
    sol_mark = _solve(
        SPHERE_1D, homogeneous_fuel_A_1G,
        n_bc_modes=1, boundary="white_rank1_mark",
    )
    np.testing.assert_allclose(
        sol_spec.k_eff, sol_mark.k_eff, rtol=1e-10, atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────
# B. Homogeneous sphere: specular converges to k_inf
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_specular_sphere_homogeneous_converges_to_kinf(
    homogeneous_fuel_A_1G,
):
    r"""For a homogeneous sphere with specular BC the eigenvector is
    radially uniform and the surface flux is isotropic; specular thus
    recovers :math:`k_\infty`. The convergence ladder is verified at
    :math:`N = 1, 2, 3, 4` with the rank-4 specular within 0.5 %
    of :math:`k_\infty` under BASE quadrature (smaller still under
    RICH).

    Numbers from the Phase-1 calibration sweep (BASE: p_order=4,
    n_quad=24, dps=20):

    - :math:`N = 1`: -0.82 % (rank-1 Mark calibration limit)
    - :math:`N = 2`: -0.66 %
    - :math:`N = 3`: -0.40 %
    - :math:`N = 4`: -0.07 %

    The ladder is monotonically improving and lands within 0.1 %
    of :math:`k_\infty` at rank-4 — the "sweet spot" for sphere
    specular. Higher :math:`N` is bounded above by the
    :math:`R_{\rm spec}` conditioning (entries grow polynomially
    with :math:`N`; at :math:`N = 8` the largest entry is ~30).
    """
    fix = homogeneous_fuel_A_1G
    k_inf = fix["k_inf"]

    errors = []
    for N in (1, 2, 3, 4):
        sol = _solve(SPHERE_1D, fix, n_bc_modes=N, boundary="specular")
        err = abs(sol.k_eff - k_inf) / k_inf
        errors.append(err)

    # Monotonically improving from N=1 to N=4.
    assert errors[3] < errors[2] < errors[1] < errors[0], (
        f"specular sphere ladder is not monotonically improving: "
        f"errors = {errors}"
    )
    # Rank-4 within 0.5 % under BASE quadrature.
    assert errors[3] < 5e-3, (
        f"specular sphere rank-4 error {errors[3]*100:.3f} % "
        f"exceeds 0.5 % tolerance"
    )


# ──────────────────────────────────────────────────────────────────────
# C. Cylinder caveat — high-rank divergence due to Issue #112 Phase C
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_specular_cylinder_homogeneous_converges_to_kinf(
    homogeneous_fuel_A_1G,
):
    r"""For a homogeneous cylinder with specular BC the eigenvector is
    radially uniform and the surface flux is isotropic; specular thus
    recovers :math:`k_\infty`. The Knyazev :math:`\mathrm{Ki}_{2+k}`
    expansion (:func:`compute_P_esc_cylinder_3d_mode`,
    :func:`compute_G_bc_cylinder_3d_mode`) makes the 3-D angular
    normalisation correct for all :math:`n \ge 0`, lifting the
    earlier rank-1-only limitation.

    Numbers from the calibration sweep (BASE: p_order=4, n_quad=24,
    dps=20, fuel A, R=5 cm, k_inf=1.5):

    - :math:`N = 1`: -0.31 % (Knyazev rank-1 ≡ white_hebert rank-1)
    - :math:`N = 2`: -0.28 %
    - :math:`N = 3`: -0.21 %
    - :math:`N = 4`: -0.11 %  ← within 0.5 % gate
    - :math:`N = 5`: -0.04 %
    - :math:`N = 6`: -0.02 %

    Monotonic improvement; rank-4 within 0.5 %, rank-6 within 0.05 %.
    """
    fix = homogeneous_fuel_A_1G
    k_inf = fix["k_inf"]

    errors = []
    for N in (1, 2, 3, 4):
        sol = _solve(CYLINDER_1D, fix, n_bc_modes=N, boundary="specular")
        err = abs(sol.k_eff - k_inf) / k_inf
        errors.append(err)

    # Monotonically improving from N=1 to N=4.
    assert errors[3] < errors[2] < errors[1] < errors[0], (
        f"specular cylinder ladder is not monotonically improving: "
        f"errors = {errors}"
    )
    # Rank-4 within 0.5 % under BASE quadrature.
    assert errors[3] < 5e-3, (
        f"specular cylinder rank-4 error {errors[3]*100:.3f} % "
        f"exceeds 0.5 % tolerance"
    )


# ──────────────────────────────────────────────────────────────────────
# D. Slab specular: per-face block-diag with single-face divisor (=1)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_specular_slab_rank1_equals_mark_kinf(homogeneous_fuel_A_1G):
    r"""At rank-1, slab specular per-face block-diag with single-face
    divisor produces the SAME dominant eigenvalue as Mark legacy.

    The two K_bc matrices differ ELEMENT-WISE — Mark legacy uses a
    combined-face primitive (½ E_2_outer + ½ E_2_inner) divided by
    the combined-face divisor (= 2), while specular per-face
    block-diag uses single-face primitives divided by the single-face
    divisor (= 1). Element-wise:

        K_bc_legacy = K_oo + K_oi + K_io + K_ii  (4 face combinations)
        K_bc_specular_blockdiag = K_oo + K_ii    (only same-face)

    However, the difference K_bc_legacy - K_bc_specular_blockdiag is
    anti-symmetric around L/2 and does NOT excite the dominant
    SYMMETRIC eigenmode for homogeneous slab. So the dominant
    eigenvalue (= k_eff) matches bit-exactly.

    See ``derivations/diagnostics/diag_slab_specular_08_*.py`` for the
    element-wise mathematical breakdown.
    """
    sol_spec = _solve(
        SLAB_POLAR_1D, homogeneous_fuel_A_1G,
        n_bc_modes=1, boundary="specular",
    )
    sol_mark = _solve(
        SLAB_POLAR_1D, homogeneous_fuel_A_1G,
        n_bc_modes=1, boundary="white_rank1_mark",
    )
    np.testing.assert_allclose(
        sol_spec.k_eff, sol_mark.k_eff, rtol=1e-10, atol=1e-12,
    )


@pytest.mark.foundation
def test_specular_slab_homogeneous_converges_to_kinf(
    homogeneous_fuel_A_1G,
):
    r"""For a homogeneous slab with specular BC at both faces the cell is
    equivalent to an infinite medium, so :math:`k_{\rm eff} \to k_\infty`
    as :math:`N \to \infty` and as the spatial mesh refines.

    The convergence ladder at BASE quadrature (p_order=4, n_panels=2,
    n_quad=24, dps=20):

    - :math:`N = 1`: ~ -0.19 % (matches Mark legacy at this quadrature)
    - :math:`N = 2`: ~ -0.18 %
    - :math:`N = 3`: ~ -0.16 %
    - :math:`N = 4`: ~ -0.15 %

    The improvements per rank are smaller than for sphere/cylinder
    because the slab Mark "rank-1 calibration limit" is already very
    close to k_inf for this homogeneous fuel-A fixture (slab Mark
    legacy reaches -0.19 % at BASE quadrature, vs sphere Mark at
    -0.82 %). Rank-N specular for slab improves ON TOP of Mark by
    capturing higher Legendre modes of ψ⁺ at each face.

    See ``derivations/diagnostics/diag_slab_specular_06_divisor_check.py``
    for the divisor-sensitivity sweep (divisor ∈ {0.5, 1.0, 2.0}) that
    isolated the per-face vs combined-face area normalisation as the
    root cause of the rank-N plateau.
    """
    fix = homogeneous_fuel_A_1G
    k_inf = fix["k_inf"]

    errors = []
    for N in (1, 2, 3, 4):
        sol = _solve(SLAB_POLAR_1D, fix, n_bc_modes=N, boundary="specular")
        err = abs(sol.k_eff - k_inf) / k_inf
        errors.append(err)

    # Monotonically improving from N=1 to N=4.
    assert errors[3] < errors[2] < errors[1] < errors[0], (
        f"specular slab ladder is not monotonically improving: "
        f"errors = {errors}"
    )
    # Rank-4 within 0.5 % under BASE quadrature (slab is more lenient
    # than sphere/cyl because rank-1 Mark for slab is already very
    # close to k_inf for the fuel-A fixture).
    assert errors[3] < 5e-3, (
        f"specular slab rank-4 error {errors[3]*100:.3f} % "
        f"exceeds 0.5 % tolerance"
    )


@pytest.mark.foundation
def test_slab_mark_decomposes_into_four_per_face_blocks():
    r"""Algebraic identity locking the relationship between the Mark
    legacy combined-face K_bc and the per-face block decomposition:

    .. math::

       K_{\rm bc}^{\rm Mark, legacy}
         \;=\; K_{oo} + K_{oi} + K_{io} + K_{ii}

    where each :math:`K_{ab} = \Sigma_t \cdot G_a \cdot P_b` with
    :math:`a, b \in \{outer, inner\}` and the per-face primitives use
    the **single-face** divisor (= 1 for slab), NOT the combined-face
    divisor (= 2 used by Mark legacy). The legacy :math:`K_{\rm bc}`
    matches the SUM with divisor=2 because the cross-face couplings
    K_oi and K_io are absorbed via Mark's combined ½ E_2_outer +
    ½ E_2_inner primitive normalised by the combined area.

    This identity proves the slab specular per-face block-diag closure
    (which keeps only :math:`K_{oo} + K_{ii}`) is the structurally
    correct restriction to "no cross-face coupling" — required for true
    specular reflection (a particle leaving outer face cannot magically
    appear at inner face except through the volume kernel).

    Promoted from a one-off diagnostic during the 2026-04-27 slab
    specular investigation (`specular_bc_slab_fix.md` agent memory).
    """
    import numpy as np
    from orpheus.derivations.peierls_geometry import (
        SLAB_POLAR_1D, _build_full_K_per_group, _slab_E_n,
        _slab_tau_to_inner_face, _slab_tau_to_outer_face,
        build_volume_kernel, composite_gl_r,
    )

    L_cell = 5.0
    sig_t_g = np.array([1.0])
    radii = np.array([L_cell])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=1, p_order=3, dps=15,
    )

    # Mark legacy K_bc via API.
    K_full = _build_full_K_per_group(
        SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "white_rank1_mark",
        n_angular=8, n_rho=8, n_surf_quad=8, n_bc_modes=1, dps=15,
    )
    K_vol = build_volume_kernel(
        SLAB_POLAR_1D, r_nodes, panels, radii, sig_t_g,
        n_angular=8, n_rho=8, dps=15,
    )
    K_bc_legacy = K_full - K_vol

    # Per-face mode-0 primitives (no-µ-weight basis).
    rv = np.array([
        SLAB_POLAR_1D.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sigt = float(sig_t_g[0])
    P_o = np.array([
        0.5 * _slab_E_n(2, _slab_tau_to_outer_face(float(x), radii, sig_t_g))
        for x in r_nodes
    ])
    P_i = np.array([
        0.5 * _slab_E_n(2, _slab_tau_to_inner_face(float(x), radii, sig_t_g))
        for x in r_nodes
    ])
    G_o = np.array([
        2.0 * _slab_E_n(2, _slab_tau_to_outer_face(float(x), radii, sig_t_g))
        for x in r_nodes
    ])
    G_i = np.array([
        2.0 * _slab_E_n(2, _slab_tau_to_inner_face(float(x), radii, sig_t_g))
        for x in r_nodes
    ])

    # Apply Mark's combined-face divisor = 2 to BOTH G primitives,
    # plus volume weighting on P. This is the convention that legacy
    # combines, so per-face blocks should re-sum to legacy with this
    # divisor.
    P_o_w = rv * r_wts * P_o
    P_i_w = rv * r_wts * P_i
    K_oo = sigt * np.outer(G_o, P_o_w) / 2.0
    K_oi = sigt * np.outer(G_o, P_i_w) / 2.0
    K_io = sigt * np.outer(G_i, P_o_w) / 2.0
    K_ii = sigt * np.outer(G_i, P_i_w) / 2.0
    K_sum = K_oo + K_oi + K_io + K_ii

    np.testing.assert_allclose(
        K_sum, K_bc_legacy, rtol=1e-12, atol=1e-12,
        err_msg=(
            "Mark legacy K_bc must decompose into K_oo + K_oi + K_io + K_ii "
            "in the per-face basis (with combined-face divisor = 2)"
        ),
    )

    # Cross-face contribution K_oi + K_io is non-trivial — proves that
    # specular block-diag (K_oo + K_ii only) DROPS real coupling that
    # the white BC (averaging) Mark closure includes.
    cross = K_oi + K_io
    assert np.max(np.abs(cross)) > 0.001, (
        f"Cross-face couplings K_oi + K_io should be > 0.001 in magnitude, "
        f"got max={np.max(np.abs(cross)):.4e}. If cross is zero, the "
        f"per-face decomposition has lost the structural distinction "
        f"between specular (no cross) and white (with cross)."
    )


# ──────────────────────────────────────────────────────────────────────
# E. Multi-energy / multi-region convergence (Phase 2)
# ──────────────────────────────────────────────────────────────────────
#
# Up to here we verified specular at 1G/1R for sphere/cyl/slab. The
# physically interesting cases are multi-energy (multi-group) and
# multi-region (heterogeneous). For specular BC with isotropic
# scattering on a HOMOGENEOUS cell, the eigenvector is uniform and
# the cell is infinite-medium equivalent regardless of the group
# count, so 2G/1R must give k_eff = k_inf_2G exactly (modulo
# truncation).
#
# Heterogeneous 1G/2R is the classical Class-B verification posture
# (cp_sphere k_inf is the Mark-closure reference; specular gives the
# angularly-exact pointwise eigenvalue, with no Mark uniformity
# assumption). The reference value differs from cp k_inf — that
# divergence quantifies the Mark closure error directly. The test
# here pins MONOTONIC CONVERGENCE of specular at increasing N (the
# closure converging to a stable value), not agreement with cp.
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def homogeneous_fuel_A_2G():
    """Fuel-A 2G XS, homogeneous single-region cell, R = 10 cm.

    Cell sized at the convergence sweet spot for the rank-N specular
    closure: fuel A 2G has σ_t = [0.5, 1.0]; at R = 10 the per-group
    optical thicknesses are τ_R = [5, 10] — beyond the documented
    thin-cell structural plateau (Mark error ~3-8 %) but not so
    thick that the spatial mesh becomes the bottleneck.

    The thin-cell plateau is the **single-bounce calibration limit**
    of the rank-N specular closure: at thin cells the geometric-
    series multi-bounce correction analogous to Hébert's
    :math:`(1-P_{ss})^{-1}` is needed. See agent memory
    ``specular_bc_thin_cell_plateau.md`` for the diagnostic cascade.
    """
    xs = get_xs("A", "2g")
    sig_t = np.atleast_2d(xs["sig_t"])  # (1 region, 2 groups)
    sig_s = xs["sig_s"][np.newaxis, :, :]  # (1, 2, 2)
    nu_sig_f = (xs["nu"] * xs["sig_f"])[np.newaxis, :]  # (1, 2)
    chi = xs["chi"][np.newaxis, :]  # (1, 2)
    k_inf = kinf_and_spectrum_homogeneous(
        xs["sig_t"], xs["sig_s"],
        xs["nu"] * xs["sig_f"], xs["chi"],
    )[0]
    radii = np.array([10.0])
    return {
        "sig_t": sig_t,
        "sig_s": sig_s,
        "nu_sig_f": nu_sig_f,
        "chi": chi,
        "k_inf": k_inf,
        "radii": radii,
    }


@pytest.fixture(scope="module")
def heterogeneous_AB_1G():
    """Heterogeneous 1G two-region cell: fuel A inner (r ∈ [0, 5]),
    moderator B outer (r ∈ [5, 10]). The OUTER region (moderator
    B with σ_t = 2.0 in 1G) gives boundary-region τ ≥ 10 — well
    outside the thin-cell single-bounce plateau. Test verifies
    monotonic convergence of specular k_eff (the value itself is a
    new specular reference, distinct from cp Mark k_inf which has
    the documented Mark uniformity overshoot).
    """
    xs_A = get_xs("A", "1g")
    xs_B = get_xs("B", "1g")
    sig_t = np.array([
        [float(xs_A["sig_t"][0])],
        [float(xs_B["sig_t"][0])],
    ])  # (2 regions, 1 group)
    sig_s = np.array([
        [[float(xs_A["sig_s"][0, 0])]],
        [[float(xs_B["sig_s"][0, 0])]],
    ])  # (2, 1, 1)
    nu_sig_f = np.array([
        [float((xs_A["nu"] * xs_A["sig_f"])[0])],
        [float((xs_B["nu"] * xs_B["sig_f"])[0])],
    ])  # (2, 1)
    chi = np.array([
        [float(xs_A["chi"][0])],
        [float(xs_B["chi"][0])],
    ])
    radii = np.array([5.0, 10.0])  # fuel inner radii=5, mod outer radii=10
    return {
        "sig_t": sig_t,
        "sig_s": sig_s,
        "nu_sig_f": nu_sig_f,
        "chi": chi,
        "radii": radii,
    }


@pytest.fixture(scope="module")
def heterogeneous_AB_2G():
    """Heterogeneous 2G two-region cell: fuel A inner (r ∈ [0, 5]),
    moderator B outer (r ∈ [5, 10]). The DEMONSTRATION case for
    multi-energy + multi-region specular: same geometry as the 1G
    fixture but with full 2G XS (downscatter A_2G[0,1]=0.10, no
    upscatter). Verification posture: monotonic convergence at
    increasing :math:`N` (specular value is the new pointwise
    reference, distinct from cp k_inf).
    """
    xs_A = get_xs("A", "2g")
    xs_B = get_xs("B", "2g")
    sig_t = np.stack([xs_A["sig_t"], xs_B["sig_t"]])  # (2 regions, 2 groups)
    sig_s = np.stack([xs_A["sig_s"], xs_B["sig_s"]])  # (2, 2, 2)
    nu_sig_f = np.stack([
        xs_A["nu"] * xs_A["sig_f"],
        xs_B["nu"] * xs_B["sig_f"],
    ])  # (2, 2)
    chi = np.stack([xs_A["chi"], xs_B["chi"]])  # (2, 2)
    radii = np.array([5.0, 10.0])
    return {
        "sig_t": sig_t,
        "sig_s": sig_s,
        "nu_sig_f": nu_sig_f,
        "chi": chi,
        "radii": radii,
    }


def _solve_mg(geometry, fixture, *, n_bc_modes, boundary):
    """MG driver for the multi-group / multi-region fixtures."""
    return solve_peierls_mg(
        geometry,
        fixture["radii"],
        fixture["sig_t"],
        fixture["sig_s"],
        fixture["nu_sig_f"],
        fixture["chi"],
        boundary=boundary,
        n_bc_modes=n_bc_modes,
        p_order=4,
        n_panels_per_region=2,
        n_angular=24,
        n_rho=24,
        n_surf_quad=24,
        dps=20,
        tol=1e-10,
    )


# ──────────────────────────────────────────────────────────────────────
# E.1 — 2G/1R (homogeneous multi-group): specular → k_inf_2G
# ──────────────────────────────────────────────────────────────────────
# For HOMOGENEOUS multi-group cells the eigenvector is still spatially
# uniform (no spatial structure beyond the group spectrum), so the
# cell is infinite-medium equivalent. Specular BC must recover
# k_inf_2G exactly (to within rank-N truncation + quadrature).


@pytest.mark.foundation
@pytest.mark.parametrize(
    "geometry,n_max,err_gate",
    [
        # sphere converges fastest under the rank-N specular closure
        pytest.param(SPHERE_1D, 4, 5e-3, id="sphere"),
        # cylinder needs N=6 to land within 0.6 % at this fuel-A 2G config
        pytest.param(CYLINDER_1D, 6, 6e-3, id="cylinder"),
        # slab plateaus at ~1 % in 2G/1R: per-face decomposition
        # interacts with multi-group spectrum sensitivity such that
        # the per-group Mark calibration error compounds. The test
        # gate accepts up to 1.5 % until the multi-bounce correction
        # is shipped (`specular_bc_thin_cell_plateau.md` follow-up).
        pytest.param(SLAB_POLAR_1D, 6, 1.5e-2, id="slab"),
    ],
)
def test_specular_2G_homogeneous_converges_to_kinf_2G(
    geometry, n_max, err_gate, homogeneous_fuel_A_2G,
):
    r"""For a homogeneous 2G cell the eigenvector is spatially
    uniform (modulated by the 2G fission/scattering matrix
    spectrum); the cell is infinite-medium equivalent under specular
    BC. The convergence ladder must satisfy:

    - Monotonic improvement from :math:`N = 1` to :math:`n_{\max}`
    - rank-:math:`n_{\max}` within ``err_gate`` of :math:`k_{\infty,2G}`

    Per-geometry :math:`n_{\max}` and ``err_gate`` reflect the
    documented per-geometry convergence rate of the rank-N specular
    closure for the 2G fuel-A homogeneous configuration. Sphere
    converges fastest; slab plateaus at ~1 % due to the per-face
    decomposition interacting with multi-group spectrum sensitivity
    in a way that surfaces the same single-bounce calibration
    weakness documented for thin-cell sphere/cyl
    (``specular_bc_thin_cell_plateau.md``).
    """
    fix = homogeneous_fuel_A_2G
    k_inf = fix["k_inf"]

    errors = []
    for N in range(1, n_max + 1):
        sol = _solve_mg(geometry, fix, n_bc_modes=N, boundary="specular")
        err = abs(sol.k_eff - k_inf) / k_inf
        errors.append(err)

    # Monotonic improvement (allow tiny numerical noise).
    monotone = all(errors[i + 1] <= errors[i] + 1e-6 for i in range(n_max - 1))
    assert monotone, (
        f"specular {geometry.kind} 2G/1R ladder not monotonically "
        f"improving: errors = {[f'{e*100:.4f}%' for e in errors]}"
    )
    assert errors[-1] < err_gate, (
        f"specular {geometry.kind} 2G/1R rank-{n_max} error "
        f"{errors[-1]*100:.4f} % exceeds {err_gate*100:.2f} % gate"
    )


# ──────────────────────────────────────────────────────────────────────
# E.2 — 1G/2R (heterogeneous single-group): monotonic convergence
# ──────────────────────────────────────────────────────────────────────
# For HETEROGENEOUS cells the surface flux is anisotropic and
# specular gives the angularly-exact pointwise eigenvalue. The
# reference value differs from cp_sphere/cp_cylinder k_inf (which
# uses Mark closure with the documented ~10 % overshoot on 1G/2R).
# This test pins MONOTONIC CONVERGENCE of specular at increasing N
# to a stable value — the value itself is the new specular reference,
# logged for documentation.


def _check_monotonic_and_settled(k_values, geom_label, label):
    """Helper: check k_values converge to a stable value.

    Two-stage gate: (a) when consecutive differences are above the
    quadrature noise floor (~1e-5 relative), require monotonic
    direction (no oscillation between corrected modes); (b) when
    differences are below the noise floor, accept the closure as
    converged regardless of sign. Always require the last step to
    be < 0.5 % relative AND the value to be physically sensible.
    """
    diffs = [k_values[i + 1] - k_values[i] for i in range(len(k_values) - 1)]
    rel_diffs = [d / k_values[i] for i, d in enumerate(diffs)]
    NOISE = 1e-5  # relative quadrature noise floor at BASE precision

    # Monotonic only required for diffs above noise.
    sig_diffs = [d for d, rd in zip(diffs, rel_diffs) if abs(rd) > NOISE]
    if sig_diffs:
        same_sign = (
            all(d >= -NOISE * abs(k_values[0]) for d in sig_diffs)
            or all(d <= NOISE * abs(k_values[0]) for d in sig_diffs)
        )
        assert same_sign, (
            f"specular {geom_label} {label} ladder oscillates with "
            f"significant amplitude: k = "
            f"{[f'{k:.6f}' for k in k_values]}; diffs = "
            f"{[f'{d:+.2e}' for d in diffs]}"
        )
    # Settled: last step < 0.5 % relative.
    settle_step = abs(k_values[-1] - k_values[-2]) / abs(k_values[-1])
    assert settle_step < 5e-3, (
        f"specular {geom_label} {label} last-step "
        f"{settle_step*100:.3f} % exceeds 0.5 % gate; closure has not "
        f"settled. k_values = {[f'{k:.6f}' for k in k_values]}"
    )
    # Physical sanity.
    assert 0.2 < k_values[-1] < 5.0, (
        f"specular {geom_label} {label} k_eff = {k_values[-1]:.6f} is "
        f"outside the physical sanity range [0.2, 5.0]"
    )
    return k_values[-1]


@pytest.mark.foundation
@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param(SPHERE_1D, id="sphere"),
        pytest.param(CYLINDER_1D, id="cylinder"),
        pytest.param(SLAB_POLAR_1D, id="slab"),
    ],
)
def test_specular_heterogeneous_1G2R_converges(
    geometry, heterogeneous_AB_1G,
):
    r"""For heterogeneous 1G/2R cells (fuel A inner + moderator B
    outer), specular k_eff converges monotonically as :math:`N` grows
    to a stable value (the new specular reference, distinct from cp
    Mark k_inf). This test pins:

    1. The convergence ladder is monotonic from :math:`N = 1` to
       :math:`N = 4` (no oscillation).
    2. The rank-3 → rank-4 step is < 0.5 % (closure has settled).
    3. The converged k_eff is finite and physically sensible (within
       a factor of 5 of unity — basic sanity).
    """
    fix = heterogeneous_AB_1G
    k_values = []
    for N in (1, 2, 3, 4):
        sol = _solve_mg(geometry, fix, n_bc_modes=N, boundary="specular")
        k_values.append(sol.k_eff)
    _check_monotonic_and_settled(k_values, geometry.kind, "1G/2R")


# ──────────────────────────────────────────────────────────────────────
# E.3 — 2G/2R demonstration (heterogeneous + multi-group)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param(SPHERE_1D, id="sphere"),
        pytest.param(CYLINDER_1D, id="cylinder"),
        pytest.param(SLAB_POLAR_1D, id="slab"),
    ],
)
def test_specular_heterogeneous_2G2R_converges(
    geometry, heterogeneous_AB_2G,
):
    r"""Demonstration test for the multi-energy + multi-region
    specular closure: 2G/2R fuel A inner + moderator B outer, same
    geometry as the 1G/2R fixture but with full 2G XS (downscatter
    A_2G[0,1]=0.10, no upscatter). Verifies monotonic convergence at
    increasing :math:`N` to a stable value (the new pointwise
    specular reference for 2G/2R Class B).
    """
    fix = heterogeneous_AB_2G
    k_values = []
    for N in (1, 2, 3, 4):
        sol = _solve_mg(geometry, fix, n_bc_modes=N, boundary="specular")
        k_values.append(sol.k_eff)
    _check_monotonic_and_settled(k_values, geometry.kind, "2G/2R")


# ──────────────────────────────────────────────────────────────────────
# F. Multi-bounce specular for thin sphere (Phase 3)
# ──────────────────────────────────────────────────────────────────────
# `closure="specular_multibounce"` is the rank-N analog of Hébert's
# (1 - P_ss)^{-1} multi-bounce correction layered on top of the
# rank-N specular closure. Lifts the documented thin-cell single-
# bounce plateau from ~-8 % (Mark calibration limit) to Hébert-
# quality accuracy at N ∈ {1, 2, 3}. At rank-1 reduces algebraically
# to Hébert white BC.
#
# Sphere-only in the current release; the cylinder Knyazev analog
# and the slab per-face block-T construction are Phase 4 follow-up.


@pytest.fixture(scope="module")
def thin_sphere_fuelA_like_1G():
    """1G XS sized to surface the thin-cell plateau on sphere R = 5
    cm: σ_t = 0.5, σ_s = 0.38, νσ_f = 0.025 (fuel-A-like fast-group
    proxy with weaker absorption). τ_R = 2.5 — well inside the
    documented single-bounce plateau region (τ_R ≲ 5)."""
    sig_t = np.array([0.5])
    sig_s = np.array([[0.38]])
    nu_sig_f = np.array([2.5 * 0.01])
    radii = np.array([5.0])
    k_inf = float(nu_sig_f[0]) / (float(sig_t[0]) - float(sig_s[0, 0]))
    return {
        "sig_t": sig_t,
        "sig_s": sig_s,
        "nu_sig_f": nu_sig_f,
        "radii": radii,
        "k_inf": k_inf,
    }


@pytest.mark.foundation
def test_specular_multibounce_rank1_equals_hebert(thin_sphere_fuelA_like_1G):
    r"""At rank-1, ``closure="specular_multibounce"`` reduces
    algebraically to Hébert's :math:`(1 - P_{ss})^{-1}` white BC. The
    construction
    :math:`K_{\rm bc}^{\rm spec,mb} = G\,R\,(I - T R)^{-1}\,P` collapses
    at :math:`N = 1` to :math:`G\,(1/(1-P_{ss}))\,P` (since :math:`R =
    [[1]]` and :math:`T_{00} = P_{ss}` exactly), bit-equal to
    ``closure="white_hebert"``.
    """
    fix = thin_sphere_fuelA_like_1G
    sol_mb = solve_peierls_1g(
        SPHERE_1D, fix["radii"], fix["sig_t"], fix["sig_s"],
        fix["nu_sig_f"], boundary="specular_multibounce", n_bc_modes=1,
        p_order=4, n_panels_per_region=2,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        tol=1e-10,
    )
    sol_heb = solve_peierls_1g(
        SPHERE_1D, fix["radii"], fix["sig_t"], fix["sig_s"],
        fix["nu_sig_f"], boundary="white_hebert", n_bc_modes=1,
        p_order=4, n_panels_per_region=2,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        tol=1e-10,
    )
    np.testing.assert_allclose(
        sol_mb.k_eff, sol_heb.k_eff, rtol=1e-8, atol=1e-10,
    )


@pytest.mark.foundation
def test_specular_multibounce_thin_sphere_lifts_plateau(
    thin_sphere_fuelA_like_1G,
):
    r"""For thin sphere (τ_R = 2.5) the bare ``specular`` closure
    plateaus at ~-5 % to -8 % from :math:`k_\infty`; the multi-bounce
    correction recovers Hébert-like accuracy at :math:`N \in \{1, 2,
    3\}`. Pins:

    1. Rank-1 multi-bounce within 0.5 % (algebraically Hébert).
    2. Rank-3 multi-bounce within 0.2 % (close to convergence).
    3. The improvement is large: bare specular at N=3 differs from
       multi-bounce at N=3 by > 5 %.
    """
    fix = thin_sphere_fuelA_like_1G
    k_inf = fix["k_inf"]

    sol_mb = {}
    for N in (1, 2, 3):
        sol = solve_peierls_1g(
            SPHERE_1D, fix["radii"], fix["sig_t"], fix["sig_s"],
            fix["nu_sig_f"],
            boundary="specular_multibounce", n_bc_modes=N,
            p_order=4, n_panels_per_region=2,
            n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
            tol=1e-10,
        )
        sol_mb[N] = sol.k_eff

    # 1. Rank-1 within 0.5 %.
    err1 = abs(sol_mb[1] - k_inf) / k_inf
    assert err1 < 5e-3, (
        f"specular_multibounce thin-sphere rank-1 error {err1*100:.4f} % "
        f"exceeds 0.5 % (Hébert-equivalent target). k_eff = {sol_mb[1]:.6f}, "
        f"k_inf = {k_inf:.6f}"
    )

    # 2. Rank-3 within 0.2 %.
    err3 = abs(sol_mb[3] - k_inf) / k_inf
    assert err3 < 2e-3, (
        f"specular_multibounce thin-sphere rank-3 error {err3*100:.4f} % "
        f"exceeds 0.2 % gate. k_eff = {sol_mb[3]:.6f}, k_inf = {k_inf:.6f}"
    )

    # 3. Significant improvement over bare specular at the same N.
    sol_bare = solve_peierls_1g(
        SPHERE_1D, fix["radii"], fix["sig_t"], fix["sig_s"],
        fix["nu_sig_f"], boundary="specular", n_bc_modes=3,
        p_order=4, n_panels_per_region=2,
        n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
        tol=1e-10,
    )
    improvement = abs(sol_bare.k_eff - sol_mb[3])
    assert improvement > 0.05 * k_inf, (
        f"multi-bounce should improve thin-sphere k_eff over bare specular "
        f"by > 5 % at rank-3, got {improvement / k_inf * 100:.3f} %. "
        f"bare = {sol_bare.k_eff:.6f}, multibounce = {sol_mb[3]:.6f}"
    )


def test_specular_multibounce_rejects_non_sphere():
    """Multi-bounce specular is sphere-only in this release; cylinder
    and slab raise NotImplementedError with a guidance message."""
    from orpheus.derivations.peierls_geometry import (
        _build_full_K_per_group,
        composite_gl_r,
    )
    radii = np.array([2.0])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region=1, p_order=2, dps=15,
    )
    sig_t_g = np.array([1.0])
    for geom in (CYLINDER_1D, SLAB_POLAR_1D):
        with pytest.raises(NotImplementedError, match="sphere-only"):
            _build_full_K_per_group(
                geom, r_nodes, r_wts, panels, radii, sig_t_g,
                "specular_multibounce",
                n_angular=4, n_rho=4, n_surf_quad=4, n_bc_modes=2, dps=15,
            )
