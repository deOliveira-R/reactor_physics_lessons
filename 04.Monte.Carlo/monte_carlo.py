"""Monte Carlo neutron transport solver for a PWR pin cell.

Simulates neutron histories in a 2D unit cell using Woodcock delta tracking,
analog absorption with fission weight adjustment, Russian roulette, and
splitting.  Returns keff with statistical uncertainty.

Port of MATLAB ``monteCarloPWR.m``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from data.macro_xs.mixture import Mixture
from geometry import CoordSystem, Mesh1D


# ---------------------------------------------------------------------------
# Geometry protocol — designed for delta-tracking (no distance-to-surface)
# ---------------------------------------------------------------------------

@runtime_checkable
class MCGeometry(Protocol):
    """Geometry interface for Monte Carlo delta-tracking.

    Implementations must provide:
    - ``material_id_at(x, y)`` — return the material ID at position (x, y)
    - ``pitch`` — unit cell side length for periodic boundary conditions

    The delta-tracking algorithm only needs the material at the collision
    point, not distance-to-surface. This makes the interface simple and
    extensible to future CSG implementations.
    """

    pitch: float

    def material_id_at(self, x: float, y: float) -> int:
        """Return the material ID at position (x, y)."""
        ...


@dataclass
class ConcentricPinCell:
    """Concentric cylindrical pin cell in a square lattice.

    Parameters
    ----------
    radii : outer radius of each annular region (innermost first).
    mat_ids : material ID for each annulus (same length as radii).
        Regions outside the outermost radius get the last material ID.
    pitch : side length of the square unit cell (cm).
    """

    radii: list[float]
    mat_ids: list[int]
    pitch: float

    def material_id_at(self, x: float, y: float) -> int:
        """Return material ID at (x, y) based on distance from pin center."""
        center = self.pitch / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        for k, r_k in enumerate(self.radii):
            if r <= r_k:
                return self.mat_ids[k]
        return self.mat_ids[-1]

    @classmethod
    def default_pwr(cls, pitch: float = 3.6) -> ConcentricPinCell:
        """Default PWR geometry matching the original MATLAB layout.

        Fuel:  r < 0.9 cm  (material 2)
        Clad:  0.9 ≤ r < 1.1 cm  (material 1)
        Cool:  r ≥ 1.1 cm  (material 0)
        """
        return cls(
            radii=[0.9, 1.1, pitch / np.sqrt(np.pi)],
            mat_ids=[2, 1, 0],
            pitch=pitch,
        )


class SlabPinCell:
    """1D slab geometry embedded in a 2D square cell (for MATLAB compatibility).

    Determines material based on x-coordinate only. Matches the original
    MATLAB monteCarloPWR.m hard-coded regions.

    Parameters
    ----------
    boundaries : list of x-coordinates separating regions.
    mat_ids : material ID for each region (len = len(boundaries) + 1).
        mat_ids[0] is for x < boundaries[0], mat_ids[-1] for x > boundaries[-1].
    pitch : unit cell side length (cm).
    """

    def __init__(
        self, boundaries: list[float], mat_ids: list[int], pitch: float,
    ):
        self.boundaries = boundaries
        self.mat_ids = mat_ids
        self.pitch = pitch

    def material_id_at(self, x: float, y: float) -> int:
        for k, bnd in enumerate(self.boundaries):
            if x < bnd:
                return self.mat_ids[k]
        return self.mat_ids[-1]

    @classmethod
    def default_pwr(cls, pitch: float = 3.6) -> SlabPinCell:
        """Original MATLAB slab geometry: cool|clad|fuel|clad|cool."""
        return cls(
            boundaries=[0.7, 0.9, 2.7, 2.9],
            mat_ids=[0, 1, 2, 1, 0],
            pitch=pitch,
        )


class MCMesh:
    """Augmented geometry for Monte Carlo delta-tracking.

    Wraps a :class:`~geometry.mesh.Mesh1D` and provides point-wise material
    lookup, following the same pattern as ``CPMesh`` and ``SNMesh``.

    Supported coordinate systems:

    * **Cartesian** — 1-D slab embedded in a 2-D square cell.
      Material determined by x-coordinate only.
    * **Cylindrical** — concentric annuli (Wigner-Seitz pin cell).
      Material determined by radial distance from cell centre.

    Parameters
    ----------
    mesh : Mesh1D
        Base geometry (from :mod:`geometry.factories`).
    pitch : float
        Unit cell side length (cm) for periodic boundary conditions.
    """

    def __init__(self, mesh: Mesh1D, pitch: float) -> None:
        if mesh.coord not in (CoordSystem.CARTESIAN, CoordSystem.CYLINDRICAL):
            raise ValueError(
                f"MCMesh supports CARTESIAN or CYLINDRICAL, got {mesh.coord}"
            )
        self.mesh = mesh
        self.pitch = pitch
        self._edges = mesh.edges
        self._mat_ids = mesh.mat_ids
        self._center = pitch / 2.0

    def material_id_at(self, x: float, y: float) -> int:
        """Return the material ID at position (x, y)."""
        if self.mesh.coord is CoordSystem.CARTESIAN:
            pos = x
        else:
            pos = np.sqrt((x - self._center)**2 + (y - self._center)**2)

        idx = np.searchsorted(self._edges, pos, side="right") - 1
        idx = max(0, min(idx, len(self._mat_ids) - 1))
        return int(self._mat_ids[idx])


@dataclass
class MCParams:
    """Monte Carlo simulation parameters."""

    n_neutrons: int = 100        # source neutrons per cycle
    n_inactive: int = 100        # inactive cycles (source convergence)
    n_active: int = 2000         # active cycles (tally accumulation)
    pitch: float = 3.6           # unit cell side length (cm)
    seed: int | None = None      # Rng seed (None = random)
    geometry: MCGeometry | None = None  # None → use default slab geometry


@dataclass
class MCResult:
    """Results of a Monte Carlo calculation."""

    keff: float               # estimated k-effective
    sigma: float              # standard deviation of keff
    keff_history: np.ndarray  # (n_active,) cumulative mean keff
    sigma_history: np.ndarray  # (n_active,) cumulative sigma
    flux_per_lethargy: np.ndarray  # (ng,) cell-averaged flux / du
    eg_mid: np.ndarray        # (ng,) mid-group energies
    elapsed_seconds: float


def solve_monte_carlo(
    materials: dict[int, Mixture],
    params: MCParams | None = None,
) -> MCResult:
    """Run Monte Carlo neutron transport simulation.

    Parameters
    ----------
    materials : dict mapping material ID (0=cool, 1=clad, 2=fuel) to Mixture.
    params : MCParams (default: 100 neutrons, 100 inactive + 2000 active).
    """
    t_start = time.perf_counter()

    if params is None:
        params = MCParams()

    rng = np.random.default_rng(params.seed)

    # Geometry: use provided or default slab layout
    geom = params.geometry
    if geom is None:
        geom = SlabPinCell.default_pwr(params.pitch)
    pitch = geom.pitch

    _any_mat = next(iter(materials.values()))
    ng = _any_mat.ng
    eg = _any_mat.eg
    eg_mid = 0.5 * (eg[:ng] + eg[1:ng + 1])
    du = np.log(eg[1:ng + 1] / eg[:ng])

    # Majorant: maximum total cross section across all materials and groups
    sig_t_max = np.zeros(ng)
    for mix in materials.values():
        sig_t_max = np.maximum(sig_t_max, mix.SigT)

    # Precompute dense scattering rows (from group ig to all groups)
    # sig_s_dense[mat_id][ig] = dense array of shape (ng,)
    sig_s_dense = {}
    for mat_id, mix in materials.items():
        rows = np.array(mix.SigS[0].todense())  # (ng, ng)
        sig_s_dense[mat_id] = rows

    # Cumulative fission spectrum for sampling (from any fissile material)
    chi_cum = np.cumsum(_any_mat.chi)

    # Scattering detector
    detect_s = np.zeros(ng)

    # Initialize neutron population
    max_n = params.n_neutrons * 4  # buffer for splitting
    x = rng.random(max_n) * pitch
    y = rng.random(max_n) * pitch
    weight = np.ones(max_n)
    i_group = np.array([np.searchsorted(chi_cum, rng.random()) for _ in range(max_n)], dtype=int)
    i_group = np.clip(i_group, 0, ng - 1)
    n_neutrons = params.n_neutrons

    keff_active = np.zeros(params.n_active)
    keff_history = np.zeros(params.n_active)
    sigma_history = np.zeros(params.n_active)

    total_cycles = params.n_inactive + params.n_active

    for i_cycle in range(1, total_cycles + 1):
        # Normalize weights to n_neutrons_born
        total_weight = weight[:n_neutrons].sum()
        weight[:n_neutrons] *= params.n_neutrons / total_weight
        weight0 = weight[:n_neutrons].copy()

        # Loop over neutrons
        for i_n in range(n_neutrons):
            ig = i_group[i_n]
            nx_, ny_ = x[i_n], y[i_n]
            w = weight[i_n]
            virtual_collision = False

            # Random walk until absorption
            while True:
                # Free path (Woodcock delta tracking)
                free_path = -np.log(rng.random()) / sig_t_max[ig]

                if not virtual_collision:
                    theta = np.pi * rng.random()
                    phi = 2.0 * np.pi * rng.random()
                    dir_x = np.sin(theta) * np.cos(phi)
                    dir_y = np.sin(theta) * np.sin(phi)

                nx_ += free_path * dir_x
                ny_ += free_path * dir_y

                # Periodic boundary conditions
                nx_ = nx_ % pitch
                ny_ = ny_ % pitch

                # Determine material from geometry
                mat_id = geom.material_id_at(nx_, ny_)
                mat = materials[mat_id]

                # Cross sections for this group
                sig_a = mat.SigF[ig] + mat.SigC[ig] + mat.SigL[ig]
                sig_p = mat.SigP[ig]

                sig_s_row = sig_s_dense[mat_id][ig, :]
                sig_s_sum = sig_s_row.sum()
                sig_t = sig_a + sig_s_sum
                sig_v = sig_t_max[ig] - sig_t

                # Virtual or real collision?
                if sig_v / sig_t_max[ig] >= rng.random():
                    virtual_collision = True
                else:
                    virtual_collision = False

                    if sig_s_sum / sig_t >= rng.random():
                        # Scattering
                        detect_s[ig] += w / sig_s_sum

                        # Sample outgoing energy group
                        cum_s = np.cumsum(sig_s_row)
                        ig = np.searchsorted(cum_s, rng.random() * sig_s_sum)
                        ig = min(ig, ng - 1)
                    else:
                        # Absorption -> convert to fission neutron
                        if sig_a > 0:
                            w *= sig_p / sig_a
                        else:
                            w = 0.0

                        # Sample new energy group from fission spectrum
                        ig = np.searchsorted(chi_cum, rng.random())
                        ig = min(ig, ng - 1)
                        break

            x[i_n] = nx_
            y[i_n] = ny_
            weight[i_n] = w
            i_group[i_n] = ig

        # Russian roulette
        for i_n in range(n_neutrons):
            if weight0[i_n] > 0:
                terminate_p = 1.0 - weight[i_n] / weight0[i_n]
            else:
                terminate_p = 1.0
            if terminate_p >= rng.random():
                weight[i_n] = 0.0
            elif terminate_p > 0:
                weight[i_n] = weight0[i_n]

        # Remove killed neutrons
        alive = weight[:n_neutrons] > 0
        n_alive = alive.sum()
        x[:n_alive] = x[:n_neutrons][alive]
        y[:n_alive] = y[:n_neutrons][alive]
        weight[:n_alive] = weight[:n_neutrons][alive]
        i_group[:n_alive] = i_group[:n_neutrons][alive]
        n_neutrons = n_alive

        # Split heavy neutrons
        n_new = 0
        for i_n in range(n_neutrons):
            if weight[i_n] > 1.0:
                N = int(np.floor(weight[i_n]))
                if weight[i_n] - N > rng.random():
                    N += 1
                new_w = weight[i_n] / N
                weight[i_n] = new_w
                for _ in range(N - 1):
                    idx = n_neutrons + n_new
                    if idx >= len(x):
                        # Grow arrays
                        grow = max(len(x), 100)
                        x = np.append(x, np.zeros(grow))
                        y = np.append(y, np.zeros(grow))
                        weight = np.append(weight, np.zeros(grow))
                        i_group = np.append(i_group, np.zeros(grow, dtype=int))
                    x[idx] = x[i_n]
                    y[idx] = y[i_n]
                    weight[idx] = new_w
                    i_group[idx] = i_group[i_n]
                    n_new += 1
        n_neutrons += n_new

        # keff for this cycle
        keff_cycle = weight[:n_neutrons].sum() / weight0.sum()

        i_active = i_cycle - params.n_inactive
        if i_active <= 0:
            if i_cycle % 20 == 0 or i_cycle <= 5:
                print(f"  Inactive {i_cycle:3d}/{params.n_inactive}  "
                      f"keff_cycle = {keff_cycle:.5f}  n = {n_neutrons}")
        else:
            ia = i_active - 1
            keff_active[ia] = keff_cycle
            keff_history[ia] = keff_active[:i_active].mean()
            if i_active > 1:
                sigma_history[ia] = np.sqrt(
                    ((keff_active[:i_active] - keff_history[ia])**2).sum()
                    / (i_active - 1) / i_active
                )
            if i_active % 200 == 0 or i_active <= 5:
                print(f"  Active {i_active:4d}/{params.n_active}  "
                      f"keff = {keff_history[ia]:.5f} +/- {sigma_history[ia]:.5f}  "
                      f"n = {n_neutrons}")

    elapsed = time.perf_counter() - t_start
    flux_du = detect_s / du

    print(f"  Elapsed: {elapsed:.1f}s")

    return MCResult(
        keff=keff_history[-1],
        sigma=sigma_history[-1],
        keff_history=keff_history,
        sigma_history=sigma_history,
        flux_per_lethargy=flux_du,
        eg_mid=eg_mid,
        elapsed_seconds=elapsed,
    )
