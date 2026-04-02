#!/usr/bin/env python3
"""Formal verification of transport solvers using synthetic benchmarks.

Runs 1-group, 2-group, and 4-group problems with known analytical
solutions against the homogeneous and slab CP solvers.
"""

import sys
_root = str(__import__('pathlib').Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(__import__('pathlib').Path(_root) / '01.Homogeneous.Reactors'))
sys.path.insert(0, str(__import__('pathlib').Path(_root) / '02.Discrete.Ordinates'))

import numpy as np
from data.macro_xs.benchmarks import (
    benchmark_1g_homogeneous,
    benchmark_2g_homogeneous,
    benchmark_4g_homogeneous,
    benchmark_1g_slab,
    benchmark_2g_slab,
    benchmark_1g_cylinder,
    benchmark_2g_cylinder,
    _kinf_from_cp,
    _make_mixture,
)
from homogeneous import solve_homogeneous_infinite
from collision_probability_slab import SlabGeometry, solve_slab_cp, _compute_slab_cp_group
from collision_probability import CPGeometry, solve_collision_probability
from discrete_ordinates import (
    DOParams, PinCellGeometry, Quadrature, solve_discrete_ordinates,
)


def run_homogeneous_benchmarks():
    """Verify the infinite-medium eigenvalue solver."""
    print("=" * 65)
    print("HOMOGENEOUS INFINITE MEDIUM BENCHMARKS")
    print("=" * 65)
    print(f"{'Benchmark':<22s}  {'Groups':>6s}  {'Analytical':>10s}  "
          f"{'Solver':>10s}  {'Error':>10s}  {'OK':>3s}")
    print("-" * 65)

    for label, bench_fn in [
        ("1G homogeneous", benchmark_1g_homogeneous),
        ("2G homogeneous", benchmark_2g_homogeneous),
        ("4G homogeneous", benchmark_4g_homogeneous),
    ]:
        mix, k_analytical = bench_fn()
        result = solve_homogeneous_infinite(mix)
        err = abs(result.k_inf - k_analytical)
        ok = "✓" if err < 1e-4 else "✗"
        print(f"{label:<22s}  {mix.ng:>6d}  {k_analytical:10.6f}  "
              f"{result.k_inf:10.6f}  {err:10.2e}  {ok:>3s}")

    print()


def run_slab_benchmarks():
    """Verify the slab CP solver against analytical CP eigenvalues."""
    print("=" * 65)
    print("HETEROGENEOUS SLAB CP BENCHMARKS")
    print("=" * 65)
    print(f"{'Benchmark':<22s}  {'Groups':>6s}  {'Analytical':>10s}  "
          f"{'Solver':>10s}  {'Error':>10s}  {'OK':>3s}")
    print("-" * 65)

    for label, bench_fn in [
        ("1G 2-region slab", benchmark_1g_slab),
        ("2G 2-region slab", benchmark_2g_slab),
    ]:
        materials, geom_params, k_analytical = bench_fn()
        ng = materials[2].ng

        geom = SlabGeometry.default_pwr(**geom_params)
        result = solve_slab_cp(materials, geom,
                               keff_tol=1e-7, flux_tol=1e-6)

        err = abs(result.keff - k_analytical)
        ok = "✓" if err < 1e-3 else "✗"
        print(f"{label:<22s}  {ng:>6d}  {k_analytical:10.6f}  "
              f"{result.keff:10.6f}  {err:10.2e}  {ok:>3s}")

    print()


def run_cylinder_benchmarks():
    """Verify the cylindrical (Wigner-Seitz) CP solver."""
    print("=" * 65)
    print("HETEROGENEOUS CYLINDRICAL CP BENCHMARKS")
    print("=" * 65)
    print(f"{'Benchmark':<22s}  {'Groups':>6s}  {'Analytical':>10s}  "
          f"{'Solver':>10s}  {'Error':>10s}  {'OK':>3s}")
    print("-" * 65)

    for label, bench_fn in [
        ("1G 2-region cyl", benchmark_1g_cylinder),
        ("2G 2-region cyl", benchmark_2g_cylinder),
    ]:
        materials, geom_params, k_analytical = bench_fn()
        ng = materials[2].ng

        geom = CPGeometry.default_pwr(**geom_params)
        result = solve_collision_probability(
            materials, geom,
        )

        err = abs(result.keff - k_analytical)
        ok = "✓" if err < 1e-3 else "✗"
        print(f"{label:<22s}  {ng:>6d}  {k_analytical:10.6f}  "
              f"{result.keff:10.6f}  {err:10.2e}  {ok:>3s}")

    print()


def _slab_geom_for_do(n_fuel, n_mod, delta, ny=2):
    """Build an SN PinCellGeometry for a 2-region slab benchmark.

    Returns the geometry AND the effective material thicknesses (accounting
    for boundary half-volumes) so that the analytical CP reference can be
    computed for the exact same dimensions.
    """
    nx = n_fuel + n_mod
    vol = np.full((nx, ny), delta**2)
    vol[0, :] /= 2
    vol[-1, :] /= 2
    vol[:, 0] /= 2
    vol[:, -1] /= 2

    mat = np.zeros((nx, ny), dtype=int)
    mat[:n_fuel, :] = 2   # fuel
    mat[n_fuel:, :] = 0   # moderator

    geom = PinCellGeometry(nx=nx, ny=ny, delta=delta, mat_map=mat, volume=vol)

    # Effective thicknesses (sum of x-widths per material)
    x_widths = np.full(nx, delta)
    x_widths[0] /= 2
    x_widths[-1] /= 2
    t_fuel_eff = x_widths[:n_fuel].sum()
    t_mod_eff = x_widths[n_fuel:].sum()

    return geom, t_fuel_eff, t_mod_eff


def _analytical_slab_kinf(fuel_mix, mod_mix, t_fuel, t_mod):
    """Compute the analytical CP eigenvalue for a 2-region slab."""
    ng = fuel_mix.ng
    sig_t_all = np.array([
        fuel_mix.SigT,
        mod_mix.SigT,
    ])  # (2, ng)

    # Build P_inf for each group using the slab E_3 formula
    from scipy.special import expn

    def e3(x):
        return float(expn(3, max(x, 0.0)))

    t_arr = np.array([t_fuel, t_mod])
    P_inf_g = np.zeros((2, 2, ng))

    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau = sig_t_g * t_arr
        bnd_pos = np.array([0.0, tau[0], tau[0] + tau[1]])

        rcp = np.zeros((2, 2))
        for i in range(2):
            sti, tau_i = sig_t_g[i], tau[i]
            rcp[i, i] += 0.5 * sti * (2 * t_arr[i] - (2.0 / sti) * (0.5 - e3(tau_i)))
            for j in range(2):
                tau_j = tau[j]
                if j > i:
                    gap_d = bnd_pos[j] - bnd_pos[i + 1]
                elif j < i:
                    gap_d = bnd_pos[i] - bnd_pos[j + 1]
                else:
                    gap_d = None
                if gap_d is not None:
                    gap_d = max(gap_d, 0.0)
                    dd = e3(gap_d) - e3(gap_d + tau_i) - e3(gap_d + tau_j) \
                         + e3(gap_d + tau_i + tau_j)
                else:
                    dd = 0.0
                gap_c = bnd_pos[i] + bnd_pos[j]
                dc = e3(gap_c) - e3(gap_c + tau_i) - e3(gap_c + tau_j) \
                     + e3(gap_c + tau_i + tau_j)
                rcp[i, j] += 0.5 * (dd + dc)

        P_cell = np.zeros((2, 2))
        for i in range(2):
            P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * t_arr[i])
        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        P_in = sig_t_g * t_arr * P_out
        P_inout = max(1.0 - P_in.sum(), 0.0)
        P_inf_g[:, :, g] = P_cell + np.outer(P_out, P_in) / (1.0 - P_inout)

    # Build scattering / fission data for the eigenvalue solver
    sig_s_fuel = fuel_mix.SigS[0].toarray()
    sig_s_mod = mod_mix.SigS[0].toarray()
    nu_sigf_fuel = fuel_mix.SigP
    nu_sigf_mod = mod_mix.SigP

    return _kinf_from_cp(
        P_inf_g=P_inf_g, sig_t_all=sig_t_all, V_arr=t_arr,
        sig_s_mats=[sig_s_fuel, sig_s_mod],
        nu_sig_f_mats=[nu_sigf_fuel, nu_sigf_mod],
        chi_mats=[fuel_mix.chi, mod_mix.chi],
    )


def run_do_slab_benchmarks():
    """Verify the SN solver on slab benchmarks.

    Builds SN meshes at several resolutions and compares against the
    analytical CP eigenvalue for the EXACT same effective geometry.
    Shows mesh convergence.
    """
    print("=" * 65)
    print("DISCRETE ORDINATES SLAB BENCHMARKS")
    print("=" * 65)

    # --- 1-group benchmark ---
    fuel_1g = _make_mixture(
        sig_t=np.array([1.0]), sig_c=np.array([0.2]),
        sig_f=np.array([0.3]), nu=np.array([2.5]),
        chi=np.array([1.0]), sig_s=np.array([[0.5]]),
    )
    mod_1g = _make_mixture(
        sig_t=np.array([2.0]), sig_c=np.array([0.1]),
        sig_f=np.array([0.0]), nu=np.array([0.0]),
        chi=np.array([1.0]), sig_s=np.array([[1.9]]),
    )

    # --- 2-group benchmark ---
    fuel_2g = _make_mixture(
        sig_t=np.array([0.50, 1.00]),
        sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.01, 0.08]),
        nu=np.array([2.50, 2.50]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
    )
    mod_2g = _make_mixture(
        sig_t=np.array([0.60, 2.00]),
        sig_c=np.array([0.02, 0.05]),
        sig_f=np.array([0.00, 0.00]),
        nu=np.array([0.00, 0.00]),
        chi=np.array([1.00, 0.00]),
        sig_s=np.array([[0.40, 0.18], [0.00, 1.95]]),
    )

    for label, fuel, mod in [
        ("1G slab SN", fuel_1g, mod_1g),
        ("2G slab SN", fuel_2g, mod_2g),
    ]:
        ng = fuel.ng
        materials = {2: fuel, 0: mod}

        print(f"\n  {label} (mesh convergence):")
        print(f"  {'delta':>8s}  {'nx':>4s}  {'t_fuel':>7s}  {'t_mod':>7s}  "
              f"{'k_analytical':>12s}  {'k_SN':>12s}  {'Error':>10s}")
        print("  " + "-" * 70)

        for delta in [0.1, 0.05, 0.02, 0.01]:
            n_fuel = max(2, round(0.5 / delta))
            n_mod = max(2, round(0.5 / delta))

            geom, t_fuel, t_mod = _slab_geom_for_do(n_fuel, n_mod, delta)

            k_ref = _analytical_slab_kinf(fuel, mod, t_fuel, t_mod)

            result = solve_discrete_ordinates(
                materials, geom,
                params=DOParams(max_outer=300, bicgstab_tol=1e-6),
            )

            err = abs(result.keff - k_ref)
            print(f"  {delta:8.3f}  {geom.nx:4d}  {t_fuel:7.4f}  {t_mod:7.4f}  "
                  f"{k_ref:12.6f}  {result.keff:12.6f}  {err:10.2e}")

    print()


def main():
    print()
    run_homogeneous_benchmarks()
    run_slab_benchmarks()
    run_cylinder_benchmarks()
    run_do_slab_benchmarks()

    print("=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
