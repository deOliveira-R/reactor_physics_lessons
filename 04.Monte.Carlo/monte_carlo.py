"""Monte Carlo neutron transport solver for a PWR pin cell.

Simulates neutron histories in a 2D unit cell using Woodcock delta tracking,
analog absorption with fission weight adjustment, Russian roulette, and
splitting.  Returns keff with statistical uncertainty.

Architecture (MT-20260406-008):

- :class:`Particle` / :class:`Neutron` — per-particle state (extensible
  to gamma transport).
- :class:`NeutronBank` — population of neutrons with array-backed storage
  for performance, clean API for population control.
- Extracted functions: :func:`_precompute_xs`, :func:`_random_walk`,
  :func:`_russian_roulette`, :func:`_split_heavy`.
- :func:`solve_monte_carlo` — orchestrator (~60 lines).

Port of MATLAB ``monteCarloPWR.m``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from data.macro_xs.mixture import Mixture
from geometry import CoordSystem, Mesh1D


# ═══════════════════════════════════════════════════════════════════════
# Geometry protocol — designed for delta-tracking (no distance-to-surface)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Particle / Neutron — per-particle state
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    """Base class for transported particles.

    Designed for future extension to gamma transport or other particle
    types that share position and weight but differ in energy treatment.
    """

    x: float
    y: float
    weight: float
    alive: bool = True


@dataclass
class Neutron(Particle):
    """A neutron with discrete energy group."""

    group: int = 0


# ═══════════════════════════════════════════════════════════════════════
# NeutronBank — population management
# ═══════════════════════════════════════════════════════════════════════

class NeutronBank:
    """Array-backed population of neutrons.

    Stores parallel arrays for performance-critical inner loops while
    providing a clean population-control API (normalize, compact, grow).

    Parameters
    ----------
    x, y : ndarray — positions.
    weight : ndarray — statistical weights.
    group : ndarray — energy group indices.
    n : int — number of live neutrons (rest is buffer).
    """

    __slots__ = ("x", "y", "weight", "group", "n")

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        group: np.ndarray,
        n: int,
    ) -> None:
        self.x = x
        self.y = y
        self.weight = weight
        self.group = group
        self.n = n

    @classmethod
    def initialize(
        cls,
        n_neutrons: int,
        pitch: float,
        chi_cum: np.ndarray,
        ng: int,
        rng: np.random.Generator,
        buffer_factor: int = 4,
    ) -> NeutronBank:
        """Create an initial neutron population.

        Positions uniform in the cell, weights = 1, groups sampled from
        the fission spectrum.
        """
        max_n = n_neutrons * buffer_factor
        x = rng.random(max_n) * pitch
        y = rng.random(max_n) * pitch
        weight = np.ones(max_n)
        group = np.array(
            [np.searchsorted(chi_cum, rng.random()) for _ in range(max_n)],
            dtype=int,
        )
        group = np.clip(group, 0, ng - 1)
        return cls(x=x, y=y, weight=weight, group=group, n=n_neutrons)

    def normalize_weights(self, target: float) -> None:
        """Scale live weights so they sum to *target*."""
        total = self.weight[:self.n].sum()
        self.weight[:self.n] *= target / total

    def save_start_weights(self) -> np.ndarray:
        """Return a copy of live weights (for roulette and keff)."""
        return self.weight[:self.n].copy()

    def compact(self) -> None:
        """Remove dead neutrons (weight = 0) by compacting arrays."""
        alive = self.weight[:self.n] > 0
        n_alive = alive.sum()
        self.x[:n_alive] = self.x[:self.n][alive]
        self.y[:n_alive] = self.y[:self.n][alive]
        self.weight[:n_alive] = self.weight[:self.n][alive]
        self.group[:n_alive] = self.group[:self.n][alive]
        self.n = n_alive

    def grow(self, n_extra: int) -> None:
        """Extend arrays to accommodate *n_extra* more particles."""
        self.x = np.append(self.x, np.zeros(n_extra))
        self.y = np.append(self.y, np.zeros(n_extra))
        self.weight = np.append(self.weight, np.zeros(n_extra))
        self.group = np.append(self.group, np.zeros(n_extra, dtype=int))


# ═══════════════════════════════════════════════════════════════════════
# Cross-section preprocessing
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _PrecomputedXS:
    """Cached cross-section data that doesn't change between cycles."""

    sig_t_max: np.ndarray       # (ng,) majorant per group
    sig_s_dense: dict           # mat_id -> (ng, ng) dense scattering
    chi_cum: np.ndarray         # cumulative fission spectrum
    ng: int
    eg: np.ndarray              # energy group boundaries
    eg_mid: np.ndarray          # mid-group energies
    du: np.ndarray              # lethargy widths


def _precompute_xs(materials: dict[int, Mixture]) -> _PrecomputedXS:
    """Build majorant, dense scattering, and energy grid from materials."""
    _any_mat = next(iter(materials.values()))
    ng = _any_mat.ng
    eg = _any_mat.eg
    eg_mid = 0.5 * (eg[:ng] + eg[1:ng + 1])
    du = np.log(eg[1:ng + 1] / eg[:ng])

    sig_t_max = np.zeros(ng)
    for mix in materials.values():
        sig_t_max = np.maximum(sig_t_max, mix.SigT)

    sig_s_dense = {}
    for mat_id, mix in materials.items():
        sig_s_dense[mat_id] = np.array(mix.SigS[0].todense())

    chi_cum = np.cumsum(_any_mat.chi)

    return _PrecomputedXS(
        sig_t_max=sig_t_max,
        sig_s_dense=sig_s_dense,
        chi_cum=chi_cum,
        ng=ng,
        eg=eg,
        eg_mid=eg_mid,
        du=du,
    )


# ═══════════════════════════════════════════════════════════════════════
# Random walk — single neutron transport to absorption
# ═══════════════════════════════════════════════════════════════════════

def _random_walk(
    bank: NeutronBank,
    i_n: int,
    geom: MCGeometry,
    materials: dict[int, Mixture],
    xs: _PrecomputedXS,
    rng: np.random.Generator,
    tally: np.ndarray,
) -> None:
    """Transport neutron *i_n* until absorption (modifies bank in-place)."""
    ig = bank.group[i_n]
    nx_, ny_ = bank.x[i_n], bank.y[i_n]
    w = bank.weight[i_n]
    pitch = geom.pitch
    virtual_collision = False

    while True:
        free_path = -np.log(rng.random()) / xs.sig_t_max[ig]

        if not virtual_collision:
            theta = np.pi * rng.random()
            phi = 2.0 * np.pi * rng.random()
            dir_x = np.sin(theta) * np.cos(phi)
            dir_y = np.sin(theta) * np.sin(phi)

        nx_ += free_path * dir_x
        ny_ += free_path * dir_y
        nx_ = nx_ % pitch
        ny_ = ny_ % pitch

        mat_id = geom.material_id_at(nx_, ny_)
        mat = materials[mat_id]

        sig_a = mat.SigF[ig] + mat.SigC[ig] + mat.SigL[ig]
        sig_p = mat.SigP[ig]
        sig_s_row = xs.sig_s_dense[mat_id][ig, :]
        sig_s_sum = sig_s_row.sum()
        sig_t = sig_a + sig_s_sum
        sig_v = xs.sig_t_max[ig] - sig_t

        if sig_v / xs.sig_t_max[ig] >= rng.random():
            virtual_collision = True
        else:
            virtual_collision = False
            if sig_s_sum / sig_t >= rng.random():
                tally[ig] += w / sig_s_sum
                cum_s = np.cumsum(sig_s_row)
                ig = np.searchsorted(cum_s, rng.random() * sig_s_sum)
                ig = min(ig, xs.ng - 1)
            else:
                if sig_a > 0:
                    w *= sig_p / sig_a
                else:
                    w = 0.0
                ig = np.searchsorted(xs.chi_cum, rng.random())
                ig = min(ig, xs.ng - 1)
                break

    bank.x[i_n] = nx_
    bank.y[i_n] = ny_
    bank.weight[i_n] = w
    bank.group[i_n] = ig


# ═══════════════════════════════════════════════════════════════════════
# Population control
# ═══════════════════════════════════════════════════════════════════════

def _russian_roulette(
    bank: NeutronBank,
    weight0: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Kill low-weight neutrons or restore to start-of-cycle weight."""
    for i_n in range(bank.n):
        if weight0[i_n] > 0:
            terminate_p = 1.0 - bank.weight[i_n] / weight0[i_n]
        else:
            terminate_p = 1.0
        if terminate_p >= rng.random():
            bank.weight[i_n] = 0.0
        elif terminate_p > 0:
            bank.weight[i_n] = weight0[i_n]


def _split_heavy(bank: NeutronBank, rng: np.random.Generator) -> None:
    """Split neutrons with weight > 1 into multiple copies."""
    n_new = 0
    for i_n in range(bank.n):
        if bank.weight[i_n] > 1.0:
            N = int(np.floor(bank.weight[i_n]))
            if bank.weight[i_n] - N > rng.random():
                N += 1
            new_w = bank.weight[i_n] / N
            bank.weight[i_n] = new_w
            for _ in range(N - 1):
                idx = bank.n + n_new
                if idx >= len(bank.x):
                    bank.grow(max(len(bank.x), 100))
                bank.x[idx] = bank.x[i_n]
                bank.y[idx] = bank.y[i_n]
                bank.weight[idx] = new_w
                bank.group[idx] = bank.group[i_n]
                n_new += 1
    bank.n += n_new


# ═══════════════════════════════════════════════════════════════════════
# Parameters and results
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Solver orchestrator
# ═══════════════════════════════════════════════════════════════════════

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

    geom = params.geometry
    if geom is None:
        geom = SlabPinCell.default_pwr(params.pitch)
    pitch = geom.pitch

    # Precompute cross-section data
    xs = _precompute_xs(materials)

    # Scattering detector tally
    tally = np.zeros(xs.ng)

    # Initialize neutron population
    bank = NeutronBank.initialize(
        params.n_neutrons, pitch, xs.chi_cum, xs.ng, rng,
    )

    keff_active = np.zeros(params.n_active)
    keff_history = np.zeros(params.n_active)
    sigma_history = np.zeros(params.n_active)
    total_cycles = params.n_inactive + params.n_active

    # ── Power iteration cycles ────────────────────────────────────────
    for i_cycle in range(1, total_cycles + 1):
        bank.normalize_weights(params.n_neutrons)
        weight0 = bank.save_start_weights()

        # Transport all neutrons
        for i_n in range(bank.n):
            _random_walk(bank, i_n, geom, materials, xs, rng, tally)

        # Population control
        _russian_roulette(bank, weight0, rng)
        bank.compact()
        _split_heavy(bank, rng)

        # Cycle keff
        keff_cycle = bank.weight[:bank.n].sum() / weight0.sum()

        # Accumulate statistics
        i_active = i_cycle - params.n_inactive
        if i_active <= 0:
            if i_cycle % 20 == 0 or i_cycle <= 5:
                print(f"  Inactive {i_cycle:3d}/{params.n_inactive}  "
                      f"keff_cycle = {keff_cycle:.5f}  n = {bank.n}")
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
                      f"n = {bank.n}")

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return MCResult(
        keff=keff_history[-1],
        sigma=sigma_history[-1],
        keff_history=keff_history,
        sigma_history=sigma_history,
        flux_per_lethargy=tally / xs.du,
        eg_mid=xs.eg_mid,
        elapsed_seconds=elapsed,
    )
