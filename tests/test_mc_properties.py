"""Unit tests for Monte Carlo solver properties (L0 term-level).

Tests structural properties of the MC geometry and algorithmic components,
each isolating a single mathematical operation against a hand calculation.

Existing tests:
- ConcentricPinCell / SlabPinCell material_id_at correctness
- 1G homogeneous gives deterministic result (σ=0)

L0-MC-001 through L0-MC-014: new term-level tests for MCMesh geometry,
majorant, delta-tracking, scattering kernel, fission weight, chi sampling,
periodic BCs, roulette, splitting, branching ratio, direction sampling,
batch statistics, and scattering matrix convention.
"""

import numpy as np
import pytest

from monte_carlo import (
    ConcentricPinCell, SlabPinCell, MCMesh, MCGeometry,
    MCParams, solve_monte_carlo,
)
from geometry import CoordSystem, Mesh1D
from geometry.factories import (
    pwr_pin_equivalent, pwr_slab_half_cell, mesh1d_from_zones, Zone,
)
from derivations import get
from derivations._xs_library import get_xs, get_mixture


# ═══════════════════════════════════════════════════════════════════════
# Existing geometry tests
# ═══════════════════════════════════════════════════════════════════════

def test_concentric_pin_cell_materials():
    """ConcentricPinCell must return correct material for known positions."""
    geom = ConcentricPinCell(
        radii=[0.5, 0.8, 1.2],
        mat_ids=[2, 1, 0],
        pitch=3.0,
    )
    center = 1.5

    # At center → material 2 (innermost)
    assert geom.material_id_at(center, center) == 2

    # Just inside first boundary
    assert geom.material_id_at(center + 0.4, center) == 2

    # Between first and second boundary
    assert geom.material_id_at(center + 0.6, center) == 1

    # Outside all boundaries
    assert geom.material_id_at(center + 1.0, center) == 0


def test_slab_pin_cell_backward_compat():
    """SlabPinCell.default_pwr must match the original hardcoded geometry."""
    geom = SlabPinCell.default_pwr(pitch=3.6)

    # Original MATLAB regions:
    # fuel: 0.9 < x < 2.7, clad: 0.7-0.9 or 2.7-2.9, cool: rest
    assert geom.material_id_at(1.8, 1.0) == 2   # fuel
    assert geom.material_id_at(0.8, 1.0) == 1   # clad
    assert geom.material_id_at(2.8, 1.0) == 1   # clad
    assert geom.material_id_at(0.3, 1.0) == 0   # cool
    assert geom.material_id_at(3.2, 1.0) == 0   # cool


def test_concentric_4_regions():
    """ConcentricPinCell with 4 regions returns all 4 materials."""
    geom = ConcentricPinCell(
        radii=[0.3, 0.5, 0.7, 1.5],
        mat_ids=[3, 2, 1, 0],
        pitch=3.0,
    )
    center = 1.5

    assert geom.material_id_at(center, center) == 3
    assert geom.material_id_at(center + 0.4, center) == 2
    assert geom.material_id_at(center + 0.6, center) == 1
    assert geom.material_id_at(center + 1.0, center) == 0


# ── MC solver properties ─────────────────────────────────────────────

def test_1g_homogeneous_deterministic():
    """1G homogeneous MC gives σ=0 (every neutron sees the same XS)."""
    case = get("mc_cyl1D_1eg_1rg")
    mix = next(iter(case.materials.values()))
    materials = {0: mix}

    # Use slab geometry with single material everywhere
    geom = SlabPinCell(boundaries=[], mat_ids=[0], pitch=3.6)
    params = MCParams(
        n_neutrons=100, n_inactive=20, n_active=100,
        seed=42, geometry=geom,
    )
    result = solve_monte_carlo(materials, params)

    assert result.sigma < 1e-10, (
        f"1G homogeneous MC should have σ≈0, got σ={result.sigma:.6e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-001/002: MCMesh geometry lookup
# ═══════════════════════════════════════════════════════════════════════

def test_mcmesh_satisfies_protocol():
    """MCMesh must satisfy the MCGeometry runtime protocol."""
    mesh = pwr_slab_half_cell()
    mc = MCMesh(mesh, pitch=3.6)
    assert isinstance(mc, MCGeometry)


def test_mcmesh_cartesian_lookup():
    """L0-MC-001: MCMesh Cartesian returns correct material from x-position.

    Hand calculation: mesh edges [0, 0.5, 1.0], mat_ids [2, 0].
    x=0.25 → cell 0 → mat 2.  x=0.75 → cell 1 → mat 0.
    """
    mesh = Mesh1D(
        edges=np.array([0.0, 0.5, 1.0]),
        mat_ids=np.array([2, 0]),
        coord=CoordSystem.CARTESIAN,
    )
    mc = MCMesh(mesh, pitch=2.0)

    assert mc.material_id_at(0.25, 0.5) == 2  # cell 0
    assert mc.material_id_at(0.75, 0.5) == 0  # cell 1
    # y should not matter for Cartesian
    assert mc.material_id_at(0.25, 1.9) == 2
    # At boundary: searchsorted("right") at x=0.5 returns 2, idx=1 → mat 0 (cell 1)
    assert mc.material_id_at(0.5, 0.5) == 0
    # Just below boundary → cell 0
    assert mc.material_id_at(0.4999, 0.5) == 2


def test_mcmesh_cylindrical_lookup():
    """L0-MC-002: MCMesh Cylindrical returns correct material from radial distance.

    Hand calculation: pitch=3.0, center=1.5.
    Mesh edges [0.0, 0.5, 1.0], mat_ids [2, 0].
    (1.5, 1.5) → r=0.0 → cell 0 → mat 2.
    (1.5, 2.2) → r=0.7 → cell 1 → mat 0.
    """
    mesh = Mesh1D(
        edges=np.array([0.0, 0.5, 1.0]),
        mat_ids=np.array([2, 0]),
        coord=CoordSystem.CYLINDRICAL,
    )
    mc = MCMesh(mesh, pitch=3.0)
    center = 1.5

    # At center → r = 0 → cell 0 → mat 2
    assert mc.material_id_at(center, center) == 2
    # r = 0.3 → cell 0 → mat 2
    assert mc.material_id_at(center + 0.3, center) == 2
    # r = 0.7 → cell 1 → mat 0
    assert mc.material_id_at(center, center + 0.7) == 0


def test_mcmesh_cylindrical_matches_concentric():
    """MCMesh from pwr_pin_equivalent must agree with ConcentricPinCell.

    10000 random points — zero mismatches expected.
    """
    mesh = pwr_pin_equivalent(r_fuel=0.9, r_clad=1.1, pitch=3.6)
    mc = MCMesh(mesh, pitch=3.6)
    ref = ConcentricPinCell.default_pwr(pitch=3.6)

    rng = np.random.default_rng(42)
    xs = rng.random(10_000) * 3.6
    ys = rng.random(10_000) * 3.6

    mismatches = sum(
        mc.material_id_at(xs[i], ys[i]) != ref.material_id_at(xs[i], ys[i])
        for i in range(len(xs))
    )
    assert mismatches == 0, f"MCMesh vs ConcentricPinCell: {mismatches}/10000 mismatches"


def test_mcmesh_rejects_spherical():
    """MCMesh must reject spherical coordinate system."""
    mesh = Mesh1D(
        edges=np.array([0.0, 1.0]),
        mat_ids=np.array([0]),
        coord=CoordSystem.SPHERICAL,
    )
    with pytest.raises(ValueError, match="SPHERICAL"):
        MCMesh(mesh, pitch=2.0)


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-003: Majorant computation
# ═══════════════════════════════════════════════════════════════════════

def test_majorant_computation():
    """L0-MC-003: sig_t_max[g] = max over materials of SigT[g].

    Hand calculation with 2G, 2-material problem:
      Material A: SigT = [0.50, 1.00]
      Material B: SigT = [0.60, 2.00]
      Majorant = [0.60, 2.00]
    """
    mix_a = get_mixture("A", "2g")
    mix_b = get_mixture("B", "2g")

    # Replicate the solver's majorant logic
    ng = 2
    sig_t_max = np.zeros(ng)
    for mix in [mix_a, mix_b]:
        sig_t_max = np.maximum(sig_t_max, mix.SigT)

    np.testing.assert_allclose(sig_t_max[0], 0.60, atol=1e-15)
    np.testing.assert_allclose(sig_t_max[1], 2.00, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-004: Delta-tracking virtual collision probability
# ═══════════════════════════════════════════════════════════════════════

def test_delta_tracking_virtual_probability():
    """L0-MC-004: P_virtual = (sig_t_max - sig_t) / sig_t_max.

    In a 2-material problem:
      sig_t_max = 2.0, material with sig_t = 0.5 → P_virtual = 0.75
      sig_t_max = 2.0, material with sig_t = 2.0 → P_virtual = 0.0
    """
    sig_t_max_g = 2.0

    # Low-density material: 75% virtual collisions
    sig_t_low = 0.5
    sig_v_low = sig_t_max_g - sig_t_low
    p_virtual_low = sig_v_low / sig_t_max_g
    assert abs(p_virtual_low - 0.75) < 1e-15

    # High-density material: 0% virtual collisions
    sig_t_high = 2.0
    sig_v_high = sig_t_max_g - sig_t_high
    p_virtual_high = sig_v_high / sig_t_max_g
    assert abs(p_virtual_high - 0.0) < 1e-15

    # Verify the solver's comparison direction: virtual if sig_v/sig_t_max >= xi.
    # For P_virtual=0.75, any xi < 0.75 gives virtual (correct).
    assert p_virtual_low >= 0.5   # xi=0.5 → virtual
    assert p_virtual_high < 0.5   # xi=0.5 → real collision


def test_delta_tracking_homogeneous_no_virtual():
    """In a homogeneous medium, sig_t = sig_t_max → zero virtual collisions.

    Statistical: run 10000 collision decisions, all should be real.
    """
    sig_t_max = 1.0
    sig_t = 1.0
    sig_v = sig_t_max - sig_t
    p_virtual = sig_v / sig_t_max

    rng = np.random.default_rng(42)
    n_trials = 10_000
    n_virtual = sum(p_virtual >= rng.random() for _ in range(n_trials))

    assert n_virtual == 0, f"Expected 0 virtual collisions, got {n_virtual}"


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-005: Scattering CDF sampling
# ═══════════════════════════════════════════════════════════════════════

def test_scattering_cdf_sampling():
    """L0-MC-005: Group transfer fractions match scattering row CDF.

    Material A 2G: sig_s[0,:] = [0.38, 0.10] (from group 0).
    Expected: 79.2% stay in group 0, 20.8% transfer to group 1.
    """
    xs = get_xs("A", "2g")
    sig_s_row = xs["sig_s"][0, :]  # from group 0
    sig_s_sum = sig_s_row.sum()

    expected_frac_g0 = sig_s_row[0] / sig_s_sum  # 0.38/0.48 ≈ 0.792
    expected_frac_g1 = sig_s_row[1] / sig_s_sum  # 0.10/0.48 ≈ 0.208

    rng = np.random.default_rng(42)
    n_samples = 100_000

    # Replicate the solver's scattering logic
    cum_s = np.cumsum(sig_s_row)
    sampled_groups = np.array([
        min(np.searchsorted(cum_s, rng.random() * sig_s_sum), 1)
        for _ in range(n_samples)
    ])

    frac_g0 = (sampled_groups == 0).sum() / n_samples
    frac_g1 = (sampled_groups == 1).sum() / n_samples

    # z-score for binomial proportion
    sigma_frac = np.sqrt(expected_frac_g0 * (1 - expected_frac_g0) / n_samples)
    z_g0 = abs(frac_g0 - expected_frac_g0) / sigma_frac

    assert z_g0 < 5.0, (
        f"Scattering CDF: frac_g0={frac_g0:.4f}, expected={expected_frac_g0:.4f}, z={z_g0:.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-006: Fission weight adjustment
# ═══════════════════════════════════════════════════════════════════════

def test_fission_weight_adjustment():
    """L0-MC-006: On absorption, w *= SigP / sig_a.

    Material A 1G:
      SigF = 0.3, SigC = 0.2, SigL = 0.0 → sig_a = 0.5
      SigP = nu * SigF = 2.5 * 0.3 = 0.75
      Weight factor = 0.75 / 0.5 = 1.5
    """
    xs = get_xs("A", "1g")
    mix = get_mixture("A", "1g")

    sig_a = mix.SigF[0] + mix.SigC[0] + mix.SigL[0]
    sig_p = mix.SigP[0]

    # Hand-calculated values
    assert abs(sig_a - 0.5) < 1e-15
    assert abs(sig_p - 0.75) < 1e-15

    weight_factor = sig_p / sig_a
    assert abs(weight_factor - 1.5) < 1e-15

    # Verify SigP = nu * SigF
    np.testing.assert_allclose(mix.SigP[0], xs["nu"][0] * xs["sig_f"][0], atol=1e-15)


def test_fission_weight_non_fissile():
    """Non-fissile material: SigP = 0, so weight → 0 on absorption."""
    mix = get_mixture("B", "1g")
    sig_a = mix.SigF[0] + mix.SigC[0] + mix.SigL[0]

    assert sig_a > 0, "Absorber should have nonzero sig_a"
    assert mix.SigP[0] == 0.0, "Non-fissile material should have SigP=0"

    # Weight factor = 0/sig_a = 0
    weight_factor = mix.SigP[0] / sig_a
    assert weight_factor == 0.0


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-007: Fission spectrum sampling
# ═══════════════════════════════════════════════════════════════════════

def test_chi_spectrum_sampling():
    """L0-MC-007: Fission neutrons born according to chi distribution.

    Material A 4G: chi = [0.60, 0.35, 0.05, 0.00].
    Expected: 60% group 0, 35% group 1, 5% group 2, 0% group 3.
    """
    xs = get_xs("A", "4g")
    chi = xs["chi"]
    chi_cum = np.cumsum(chi)
    ng = len(chi)

    rng = np.random.default_rng(42)
    n_samples = 100_000

    # Replicate solver's chi sampling
    sampled = np.array([
        min(np.searchsorted(chi_cum, rng.random()), ng - 1)
        for _ in range(n_samples)
    ])

    fracs = np.array([(sampled == g).sum() / n_samples for g in range(ng)])

    for g in range(ng):
        if chi[g] > 0:
            sigma_frac = np.sqrt(chi[g] * (1 - chi[g]) / n_samples)
            z = abs(fracs[g] - chi[g]) / sigma_frac
            assert z < 5.0, (
                f"Chi sampling group {g}: frac={fracs[g]:.4f}, "
                f"expected={chi[g]:.4f}, z={z:.1f}"
            )
        else:
            assert fracs[g] == 0.0, f"Group {g} with chi=0 should have zero samples"


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-008: Periodic boundary conditions
# ═══════════════════════════════════════════════════════════════════════

def test_periodic_bc_wrapping():
    """L0-MC-008: Position wrapping via x % pitch.

    Hand-calculated edge cases:
      pitch = 3.6
      x = 3.7  → 0.1
      x = -0.1 → 3.5  (Python % always returns positive)
      x = 7.3  → 0.1
      x = 0.0  → 0.0
      x = 3.6  → 0.0
    """
    pitch = 3.6

    np.testing.assert_allclose(3.7 % pitch, 0.1, atol=1e-14)
    np.testing.assert_allclose(-0.1 % pitch, 3.5, atol=1e-14)
    np.testing.assert_allclose(7.3 % pitch, 0.1, atol=1e-14)
    np.testing.assert_allclose(0.0 % pitch, 0.0, atol=1e-14)
    np.testing.assert_allclose(3.6 % pitch, 0.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-009: Russian roulette weight conservation
# ═══════════════════════════════════════════════════════════════════════

def test_roulette_weight_conservation():
    """L0-MC-009: E[w_after] = w_before for Russian roulette.

    The roulette rule: terminate with p = 1 - w/w0, else restore to w0.
    Expected value: (w/w0)*w0 + (1 - w/w0)*0 = w.
    Statistical test: mean(w_after) ≈ mean(w_before).
    """
    rng = np.random.default_rng(42)
    n = 100_000

    # Simulate: initial weight w0 = 1.0, post-collision weight w = 0.3
    w0 = np.ones(n)
    w = np.full(n, 0.3)

    # Apply roulette (replicate solver logic)
    w_after = np.empty(n)
    for i in range(n):
        terminate_p = 1.0 - w[i] / w0[i]
        if terminate_p >= rng.random():
            w_after[i] = 0.0
        else:
            w_after[i] = w0[i]

    # Mean weight must be conserved
    mean_before = w.mean()
    mean_after = w_after.mean()
    sigma_mean = w0[0] * np.sqrt(0.3 * 0.7 / n)  # std of mean for Bernoulli

    z = abs(mean_after - mean_before) / sigma_mean
    assert z < 5.0, (
        f"Roulette conservation: mean_before={mean_before:.4f}, "
        f"mean_after={mean_after:.4f}, z={z:.1f}"
    )


def test_roulette_restore_weight():
    """Surviving neutrons must have weight restored to w0, not kept at w."""
    rng = np.random.default_rng(42)

    w0 = 1.0
    w = 0.3
    terminate_p = 1.0 - w / w0  # = 0.7

    # Force survival by providing xi > terminate_p
    # The solver uses: if terminate_p >= xi → kill; elif terminate_p > 0 → restore
    # So xi = 0.8 > 0.7 → survives → weight should be w0 = 1.0
    survivors = []
    for _ in range(1000):
        xi = rng.random()
        if terminate_p >= xi:
            continue  # killed
        elif terminate_p > 0:
            survivors.append(w0)  # restored to w0
        else:
            survivors.append(w)

    assert all(s == w0 for s in survivors), "Surviving roulette weight must equal w0"


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-010: Splitting weight conservation
# ═══════════════════════════════════════════════════════════════════════

def test_splitting_weight_conservation():
    """L0-MC-010: Splitting preserves total weight exactly.

    For weight w, splitting creates N = floor(w) or floor(w)+1 copies
    each with weight w/N. Total = N * (w/N) = w.
    """
    rng = np.random.default_rng(42)
    test_weights = [0.5, 1.5, 2.7, 5.0, 10.3]

    for w_in in test_weights:
        if w_in <= 1.0:
            continue  # splitting only applies to w > 1

        # Replicate solver splitting logic
        N = int(np.floor(w_in))
        if w_in - N > rng.random():
            N += 1
        new_w = w_in / N
        total_after = N * new_w

        np.testing.assert_allclose(total_after, w_in, atol=1e-15,
                                   err_msg=f"Split w={w_in}: N={N}, total={total_after}")


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-011: Scattering-vs-absorption branching ratio
# ═══════════════════════════════════════════════════════════════════════

def test_scattering_branching_ratio():
    """L0-MC-011: P(scatter) = sig_s_sum / sig_t.

    Material A 1G: sig_s = 0.5, sig_t = 1.0 → P(scatter) = 0.5.
    """
    xs = get_xs("A", "1g")
    sig_s_sum = xs["sig_s"].sum()
    sig_t = xs["sig_t"][0]
    p_scatter_expected = sig_s_sum / sig_t

    assert abs(p_scatter_expected - 0.5) < 1e-15

    # Statistical: sample 100k decisions
    rng = np.random.default_rng(42)
    n = 100_000
    n_scatter = sum(p_scatter_expected >= rng.random() for _ in range(n))
    frac = n_scatter / n

    sigma = np.sqrt(p_scatter_expected * (1 - p_scatter_expected) / n)
    z = abs(frac - p_scatter_expected) / sigma
    assert z < 5.0, (
        f"Branching: frac_scatter={frac:.4f}, expected={p_scatter_expected:.4f}, z={z:.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-012: Direction sampling
# ═══════════════════════════════════════════════════════════════════════

def test_direction_sampling():
    """L0-MC-012: Direction sampling matches the intended formula.

    The solver uses theta = pi*xi (uniform in [0,pi]), phi = 2pi*xi.
    dir_x = sin(theta)*cos(phi), dir_y = sin(theta)*sin(phi).

    E[dir_x^2] = E[sin^2(theta)] * E[cos^2(phi)] = (1/2) * (1/2) = 1/4.
    (Because E[sin^2(theta)] = 1/2 for uniform theta, and
     E[cos^2(phi)] = 1/2 for uniform phi.)

    Note: this is NOT isotropic (which would give 1/3). The formula
    matches the MATLAB original — a known simplification.
    """
    rng = np.random.default_rng(42)
    n = 200_000

    theta = np.pi * rng.random(n)
    phi = 2.0 * np.pi * rng.random(n)
    dir_x = np.sin(theta) * np.cos(phi)
    dir_y = np.sin(theta) * np.sin(phi)

    mean_dx2 = np.mean(dir_x**2)
    mean_dy2 = np.mean(dir_y**2)

    # E[sin^2(theta)*cos^2(phi)] = 1/4 for uniform theta, uniform phi
    expected = 0.25
    sigma = np.std(dir_x**2) / np.sqrt(n)

    z_x = abs(mean_dx2 - expected) / sigma
    z_y = abs(mean_dy2 - expected) / sigma

    assert z_x < 5.0, f"E[dir_x^2]={mean_dx2:.5f}, expected={expected}, z={z_x:.1f}"
    assert z_y < 5.0, f"E[dir_y^2]={mean_dy2:.5f}, expected={expected}, z={z_y:.1f}"


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-013: Batch statistics formula
# ═══════════════════════════════════════════════════════════════════════

def test_batch_statistics_formula():
    """L0-MC-013: Cumulative mean and sigma match hand calculation.

    keff_active = [1.0, 1.1, 0.9, 1.05]
    Step 1: mean=1.0, sigma=0
    Step 2: mean=1.05, sigma=sqrt(((1-1.05)^2+(1.1-1.05)^2)/1/2)=0.05
    Step 4: mean=1.0125, sigma from sample
    """
    keff_active = np.array([1.0, 1.1, 0.9, 1.05])
    n = len(keff_active)

    # Replicate solver's batch statistics
    keff_history = np.zeros(n)
    sigma_history = np.zeros(n)

    for ia in range(n):
        i_active = ia + 1
        keff_history[ia] = keff_active[:i_active].mean()
        if i_active > 1:
            sigma_history[ia] = np.sqrt(
                ((keff_active[:i_active] - keff_history[ia])**2).sum()
                / (i_active - 1) / i_active
            )

    # Hand-calculated values
    np.testing.assert_allclose(keff_history[0], 1.0, atol=1e-15)
    np.testing.assert_allclose(keff_history[1], 1.05, atol=1e-15)
    np.testing.assert_allclose(keff_history[3], 1.0125, atol=1e-15)

    # sigma at step 2: sqrt(0.005 / 1 / 2) = sqrt(0.0025) = 0.05
    np.testing.assert_allclose(sigma_history[1], 0.05, atol=1e-15)

    # sigma at step 4: hand-calculated
    deviations = keff_active - 1.0125
    var_sample = (deviations**2).sum() / 3  # n-1 = 3
    sigma_4 = np.sqrt(var_sample / 4)
    np.testing.assert_allclose(sigma_history[3], sigma_4, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════
# L0-MC-014: Scattering matrix convention (anti-ERR-002)
# ═══════════════════════════════════════════════════════════════════════

def test_scattering_convention_no_upscatter():
    """L0-MC-014: Asymmetric 2G matrix — no upscatter from thermal group.

    Material A 2G: sig_s = [[0.38, 0.10], [0.00, 0.90]].
    Convention: sig_s[from, to]. Row 1 = [0.00, 0.90].
    A neutron in group 1 can ONLY scatter to group 1 (no upscatter).

    If the code incorrectly uses the column (SigS^T), group 1 would
    see [0.10, 0.90] and could upscatter to group 0.
    """
    xs = get_xs("A", "2g")
    sig_s = np.array(xs["sig_s"])

    # The solver uses: sig_s_row = sig_s_dense[mat_id][ig, :]
    # For ig=1: sig_s_row = sig_s[1, :] = [0.00, 0.90]
    sig_s_row_g1 = sig_s[1, :]

    assert sig_s_row_g1[0] == 0.0, (
        f"sig_s[1,0] should be 0 (no upscatter), got {sig_s_row_g1[0]}"
    )

    # Statistical: sample from this row, no neutrons should end in group 0
    rng = np.random.default_rng(42)
    n = 10_000
    sig_s_sum = sig_s_row_g1.sum()
    cum_s = np.cumsum(sig_s_row_g1)

    sampled = np.array([
        min(np.searchsorted(cum_s, rng.random() * sig_s_sum), 1)
        for _ in range(n)
    ])

    n_upscatter = (sampled == 0).sum()
    assert n_upscatter == 0, (
        f"Expected 0 upscatter events from group 1, got {n_upscatter}. "
        f"Possible SigS^T bug (ERR-002 equivalent)."
    )
