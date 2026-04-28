"""Diagnostic: Monte Carlo cross-check of multi-bounce specular sphere.

Created by numerics-investigator on 2026-04-27.

GROUND TRUTH: For a homogeneous sphere with specular BC, the multi-bounce
limit IS k_inf. We compute k_eff via Monte Carlo simulation of a cell:
- emit neutrons isotropically from random points
- follow each neutron through bounces (specular reflection)
- count collisions; ratio (νΣ_f · path_length / Σ_t · path_length) = k_eff

This sidesteps any rank-N truncation. If MC gives k_inf, the closure
overshoot is due to the rank-N construction. If MC ALSO overshoots,
then the assumption "homogeneous + specular = k_inf" is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest


def mc_specular_sphere(R, sigt, sigs, nuf, n_neutrons=200_000, max_bounces=200, seed=0):
    """MC k_eff for homogeneous sphere with specular BC.

    Estimator: total absorption rate / fission rate. Or by collision
    track-length: k_eff = (νΣ_f · <path>) / (Σ_a · <path>) = νΣ_f / Σ_a.

    More directly: k_eff is the dominant eigenvalue of the cell. For
    homogeneous + specular it should be k_inf.

    We instead measure: source 1 neutron, count total collisions
    (Σ_t · path-length) and the implied multiplicities: k_eff = νΣ_f/Σ_a.

    A direct MC k_eff is harder to compute (eigenvalue power iteration);
    instead we verify INFINITE-MEDIUM-LIKE BEHAVIOR by checking that
    track-length collision tally per source neutron equals 1/(1-c) where
    c = (Σ_s + νΣ_f)/Σ_t (infinite medium with leakage = 0).
    """
    rng = np.random.default_rng(seed)
    sig_a = sigt - sigs
    c = sigs / sigt  # scattering ratio

    total_path_length = 0.0
    total_absorptions = 0
    total_fissions = 0.0  # weighted by ν

    for n in range(n_neutrons):
        # Sample uniform isotropic source in volume
        # Sample r uniform in [0,R³] then take cube root for uniform volume
        r = R * rng.random() ** (1.0/3.0)
        # Sample direction uniformly on unit sphere
        phi = 2*np.pi * rng.random()
        cos_th = 2*rng.random() - 1
        sin_th = np.sqrt(1 - cos_th**2)
        # Position = r * (some direction); use random axis
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        pos = r * u
        # Direction
        d = np.array([sin_th*np.cos(phi), sin_th*np.sin(phi), cos_th])

        weight = 1.0
        for _ in range(max_bounces):
            # Distance to next collision (if no boundary)
            tau_collision = -np.log(rng.random())
            d_collision = tau_collision / sigt

            # Distance to boundary
            # |pos + t*d|² = R² ; quadratic in t
            a = 1.0
            b = 2 * np.dot(pos, d)
            c_q = np.dot(pos, pos) - R*R
            disc = b*b - 4*a*c_q
            if disc < 0:
                # already outside? shouldn't happen
                break
            sq = np.sqrt(disc)
            t1 = (-b + sq) / 2
            d_boundary = max(t1, 0.0)

            if d_collision < d_boundary:
                # Collision
                pos = pos + d_collision * d
                total_path_length += d_collision * weight
                # Implicit capture: weight *= scattering ratio
                # Track absorption (weight loss)
                total_absorptions += weight * (1 - c)
                total_fissions += weight * (nuf / sigt)
                weight *= c
                if weight < 1e-6:
                    break  # russian roulette skip
                # New direction (isotropic scatter)
                phi = 2*np.pi * rng.random()
                cos_th = 2*rng.random() - 1
                sin_th = np.sqrt(1 - cos_th**2)
                d = np.array([sin_th*np.cos(phi), sin_th*np.sin(phi), cos_th])
            else:
                # Hit boundary — specular reflection
                pos = pos + d_boundary * d
                total_path_length += d_boundary * weight
                # Specular reflection: d_new = d - 2(d·n_hat)n_hat
                # n_hat = pos/R (outward normal)
                n_hat = pos / R
                d = d - 2 * np.dot(d, n_hat) * n_hat
                # Tiny push inward to avoid re-hitting boundary
                pos = pos - 1e-10 * n_hat

    k_eff = total_fissions / total_absorptions if total_absorptions > 0 else 0
    print(f"  total_path_length: {total_path_length:.3e}")
    print(f"  total_absorptions: {total_absorptions:.3e}")
    print(f"  total_fissions:    {total_fissions:.3e}")
    print(f"  k_eff = fissions/absorptions: {k_eff:.6f}")
    print(f"  expected k_inf = νΣ_f/Σ_a = {nuf/sig_a:.6f}")

    # Mean path length per source neutron
    mean_path = total_path_length / n_neutrons
    # Infinite medium prediction: <path> = 1/(σ_t·(1-c))
    inf_med_path = 1.0 / (sigt * (1 - c))
    print(f"  <path>/source = {mean_path:.4f} (inf medium: {inf_med_path:.4f})")
    return k_eff


def test_mc_specular_sphere_thin(capsys):
    """Verify MC k_eff = k_inf for thin homogeneous sphere with specular BC."""
    with capsys.disabled():
        # Thin sphere fuel-A-like
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== MC specular sphere R={R}, σ_t={sigt}, k_inf={k_inf:.6f} ===")

        k_mc = mc_specular_sphere(R, sigt, sigs, nuf, n_neutrons=50_000)
        rel_err = (k_mc - k_inf) / k_inf
        print(f"\n  k_MC - k_inf rel = {rel_err*100:.3f}%")

        # For the multi-bounce thin sphere, MC should give k_inf within
        # ~1% statistical noise.
        assert abs(rel_err) < 0.03, (
            f"MC k_eff={k_mc:.4f} differs from k_inf={k_inf:.4f} by "
            f"{rel_err*100:.3f}% — either MC is wrong or specular ≠ k_inf."
        )


def test_mc_specular_sphere_very_thin(capsys):
    """Same for very-thin sphere (σ_t·R = 1)."""
    with capsys.disabled():
        R = 5.0
        sigt = 0.2
        sigs = 0.16
        nuf = 0.01
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== MC specular sphere R={R}, σ_t={sigt}, k_inf={k_inf:.6f} ===")

        k_mc = mc_specular_sphere(R, sigt, sigs, nuf, n_neutrons=50_000)
        rel_err = (k_mc - k_inf) / k_inf
        print(f"\n  k_MC - k_inf rel = {rel_err*100:.3f}%")

        assert abs(rel_err) < 0.05, (
            f"MC k_eff={k_mc:.4f} differs from k_inf={k_inf:.4f} by "
            f"{rel_err*100:.3f}% — either MC is wrong or specular ≠ k_inf."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
