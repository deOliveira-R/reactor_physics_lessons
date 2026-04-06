"""Augmented geometry for the Method of Characteristics.

``MOCMesh`` wraps a cylindrical :class:`~geometry.mesh.Mesh1D` and a
:class:`MOCQuadrature`, precomputing all ray-tracing data (tracks,
segments, reflective boundary links) for a pin-cell with concentric
annuli inside a square lattice cell.

**Inverse Wigner-Seitz**: the ``Mesh1D`` is built with
:func:`geometry.factories.pwr_pin_equivalent`, whose outer edge is the
Wigner-Seitz radius ``r_cell = pitch / sqrt(pi)``.  ``MOCMesh`` recovers
the pitch and reinterprets the outermost annular region as the square
border bounded by the cell walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import Mesh1D

from moc_quadrature import MOCQuadrature


# ── Data structures ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Segment:
    """One segment of a ray through a flat-source region."""

    region_id: int
    length: float


@dataclass(frozen=True)
class Track:
    """A single characteristic ray across the unit cell.

    Each track supports two traversal directions:

    * **Forward** — direction ``(cos phi, sin phi)`` with ``sin >= 0``.
      Enters at ``entry_point`` (``entry_surface``), exits at
      ``exit_point`` (``exit_surface``).
    * **Backward** — direction ``(-cos phi, -sin phi)``.
      Enters at ``exit_point``, exits at ``entry_point``.

    Reflective BC links describe where outgoing angular flux goes:

    * ``fwd_link`` / ``fwd_link_fwd``: when the forward sweep exits,
      its outgoing flux feeds into track ``fwd_link``.  If
      ``fwd_link_fwd`` is True the flux enters that track's forward
      entry; otherwise its backward entry.
    * ``bwd_link`` / ``bwd_link_fwd``: same for the backward sweep.
    """

    segments: tuple[Segment, ...]
    azi_index: int
    entry_point: tuple[float, float]
    exit_point: tuple[float, float]
    entry_surface: int   # 0=bottom, 1=right, 2=top, 3=left
    exit_surface: int
    # Forward sweep outgoing link
    fwd_link: int        # target track index
    fwd_link_fwd: bool   # True → feeds target's forward entry
    # Backward sweep outgoing link
    bwd_link: int
    bwd_link_fwd: bool


# ── Ray-geometry intersection helpers ───────────────────────────────

def _ray_circle_intersections(
    x0: float,
    y0: float,
    cos_phi: float,
    sin_phi: float,
    cx: float,
    cy: float,
    radius: float,
) -> list[float]:
    """Parameter values where ray intersects a circle.

    Returns 0 or 2 parameter values.
    """
    dx = x0 - cx
    dy = y0 - cy
    b = dx * cos_phi + dy * sin_phi
    c = dx * dx + dy * dy - radius * radius
    disc = b * b - c
    if disc < 0:
        return []
    sqrt_disc = np.sqrt(disc)
    return [-b - sqrt_disc, -b + sqrt_disc]


def _ray_box_intersections(
    x0: float,
    y0: float,
    cos_phi: float,
    sin_phi: float,
    pitch: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float], int, int]:
    """Entry and exit for a ray through [0, pitch]^2.

    Returns (s_entry, s_exit, entry_point, exit_point,
             entry_surface, exit_surface).
    """
    s_vals: list[tuple[float, int]] = []

    if abs(cos_phi) > 1e-15:
        s_left = -x0 / cos_phi
        s_right = (pitch - x0) / cos_phi
        y_left = y0 + s_left * sin_phi
        y_right = y0 + s_right * sin_phi
        if -1e-10 <= y_left <= pitch + 1e-10:
            s_vals.append((s_left, 3))
        if -1e-10 <= y_right <= pitch + 1e-10:
            s_vals.append((s_right, 1))

    if abs(sin_phi) > 1e-15:
        s_bottom = -y0 / sin_phi
        s_top = (pitch - y0) / sin_phi
        x_bottom = x0 + s_bottom * cos_phi
        x_top = x0 + s_top * cos_phi
        if -1e-10 <= x_bottom <= pitch + 1e-10:
            s_vals.append((s_bottom, 0))
        if -1e-10 <= x_top <= pitch + 1e-10:
            s_vals.append((s_top, 2))

    s_vals.sort(key=lambda t: t[0])

    filtered: list[tuple[float, int]] = [s_vals[0]]
    for sv in s_vals[1:]:
        if sv[0] - filtered[-1][0] > 1e-12:
            filtered.append(sv)
    s_vals = filtered

    s_entry, surf_entry = s_vals[0]
    s_exit, surf_exit = s_vals[1]
    entry_pt = (x0 + s_entry * cos_phi, y0 + s_entry * sin_phi)
    exit_pt = (x0 + s_exit * cos_phi, y0 + s_exit * sin_phi)
    return s_entry, s_exit, entry_pt, exit_pt, surf_entry, surf_exit


def _identify_region(
    x: float,
    y: float,
    cx: float,
    cy: float,
    radii: np.ndarray,
    n_regions: int,
) -> int:
    """Identify which FSR a point belongs to."""
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    for k in range(len(radii)):
        if r <= radii[k] + 1e-12:
            return k
    return n_regions - 1


def _trace_single_ray(
    x0: float,
    y0: float,
    cos_phi: float,
    sin_phi: float,
    pitch: float,
    cx: float,
    cy: float,
    radii: np.ndarray,
    n_regions: int,
) -> tuple[list[Segment], tuple[float, float], tuple[float, float], int, int]:
    """Trace one ray through the pin-cell geometry.

    Returns (segments, entry_point, exit_point, entry_surface, exit_surface).
    """
    s_entry, s_exit, entry_pt, exit_pt, surf_entry, surf_exit = (
        _ray_box_intersections(x0, y0, cos_phi, sin_phi, pitch)
    )

    crossings: list[float] = [s_entry, s_exit]
    for r_k in radii:
        hits = _ray_circle_intersections(x0, y0, cos_phi, sin_phi, cx, cy, r_k)
        for s in hits:
            if s_entry + 1e-12 < s < s_exit - 1e-12:
                crossings.append(s)
    crossings.sort()

    unique: list[float] = [crossings[0]]
    for s in crossings[1:]:
        if s - unique[-1] > 1e-12:
            unique.append(s)
    crossings = unique

    segments: list[Segment] = []
    for i in range(len(crossings) - 1):
        s_a, s_b = crossings[i], crossings[i + 1]
        length = s_b - s_a
        if length < 1e-14:
            continue
        s_mid = 0.5 * (s_a + s_b)
        mx = x0 + s_mid * cos_phi
        my = y0 + s_mid * sin_phi
        region = _identify_region(mx, my, cx, cy, radii, n_regions)
        segments.append(Segment(region_id=region, length=length))

    return segments, entry_pt, exit_pt, surf_entry, surf_exit


# ── Reflection helpers ──────────────────────────────────────────────

def _reflected_azi_index(phi: np.ndarray, azi_index: int) -> int:
    """Both vertical and horizontal reflections map phi -> pi - phi."""
    phi_refl = np.pi - phi[azi_index]
    if phi_refl < 0:
        phi_refl += np.pi
    if phi_refl >= np.pi:
        phi_refl -= np.pi
    return int(np.argmin(np.abs(phi - phi_refl)))


def _is_vertical(surface: int) -> bool:
    """Surfaces 1 (right) and 3 (left) are vertical walls."""
    return surface in (1, 3)


# ── MOCMesh ─────────────────────────────────────────────────────────

class MOCMesh:
    """Augmented geometry for the Method of Characteristics.

    Wraps a cylindrical :class:`~geometry.mesh.Mesh1D` (Wigner-Seitz pin
    cell) and an :class:`MOCQuadrature`, precomputing all ray-tracing
    data for a 2-D pin-cell transport calculation.

    The outermost ``Mesh1D`` region is reinterpreted as the square border
    via the inverse Wigner-Seitz transformation: ``pitch = edges[-1] * sqrt(pi)``.
    """

    def __init__(
        self,
        mesh: Mesh1D,
        quadrature: MOCQuadrature,
        ray_spacing: float = 0.05,
    ) -> None:
        self.mesh = mesh
        self.quad = quadrature
        self.ray_spacing = ray_spacing

        self.pitch: float = mesh.edges[-1] * np.sqrt(np.pi)
        self.n_regions: int = mesh.N
        self.radii: np.ndarray = np.asarray(mesh.edges[1:-1], dtype=float)
        self.region_mat_ids: np.ndarray = np.asarray(mesh.mat_ids, dtype=int)

        # Exact 2D areas
        areas = np.empty(self.n_regions)
        all_edges = mesh.edges
        for k in range(self.n_regions):
            r_inner = all_edges[k]
            if k < self.n_regions - 1:
                r_outer = all_edges[k + 1]
                areas[k] = np.pi * (r_outer**2 - r_inner**2)
            else:
                areas[k] = self.pitch**2 - np.pi * r_inner**2
        self.region_areas: np.ndarray = areas

        self._cx = self.pitch / 2.0
        self._cy = self.pitch / 2.0

        self.tracks: list[Track] = []
        self.tracks_per_azi: list[list[int]] = []
        self._effective_spacing: list[float] = []
        self._generate_tracks()
        self._link_tracks()

    def _generate_tracks(self) -> None:
        """Generate tracks for all azimuthal angles."""
        pitch = self.pitch
        ts = self.ray_spacing
        phi_arr = self.quad.phi

        for a_idx in range(self.quad.n_azi):
            cos_phi = np.cos(phi_arr[a_idx])
            sin_phi = np.sin(phi_arr[a_idx])

            corners = [(0, 0), (pitch, 0), (pitch, pitch), (0, pitch)]
            t_vals = [-c[0] * sin_phi + c[1] * cos_phi for c in corners]
            t_min, t_max = min(t_vals), max(t_vals)

            n_rays = max(1, int(np.ceil((t_max - t_min) / ts)))
            effective_ts = (t_max - t_min) / n_rays

            track_indices: list[int] = []
            for k in range(n_rays):
                t_k = t_min + (k + 0.5) * effective_ts
                x0 = -t_k * sin_phi
                y0 = t_k * cos_phi

                segments, entry_pt, exit_pt, entry_surf, exit_surf = (
                    _trace_single_ray(
                        x0, y0, cos_phi, sin_phi,
                        pitch, self._cx, self._cy,
                        self.radii, self.n_regions,
                    )
                )
                if not segments:
                    continue

                track_idx = len(self.tracks)
                self.tracks.append(Track(
                    segments=tuple(segments),
                    azi_index=a_idx,
                    entry_point=entry_pt,
                    exit_point=exit_pt,
                    entry_surface=entry_surf,
                    exit_surface=exit_surf,
                    fwd_link=-1, fwd_link_fwd=True,
                    bwd_link=-1, bwd_link_fwd=True,
                ))
                track_indices.append(track_idx)

            self.tracks_per_azi.append(track_indices)
            self._effective_spacing.append(
                (t_max - t_min) / max(1, len(track_indices))
            )

    def _link_tracks(self) -> None:
        """Set up reflective boundary condition links.

        Reflection rules for angles phi in [0, pi):

        +-----------+---------------+------------------+
        | Sweep dir | Exit surface  | Target direction |
        +-----------+---------------+------------------+
        | Forward   | vertical      | Forward          |
        | Forward   | horizontal    | Backward         |
        | Backward  | vertical      | Backward         |
        | Backward  | horizontal    | Forward          |
        +-----------+---------------+------------------+

        In all cases the reflected angle is pi - phi.
        """
        phi = self.quad.phi
        n_tracks = len(self.tracks)

        for i in range(n_tracks):
            track = self.tracks[i]

            # Forward sweep exits at exit_point through exit_surface
            fwd_refl_azi = _reflected_azi_index(phi, track.azi_index)
            fwd_target_is_fwd = _is_vertical(track.exit_surface)
            fwd_link = self._find_link(
                track.exit_point, fwd_refl_azi, fwd_target_is_fwd
            )

            # Backward sweep exits at entry_point through entry_surface
            bwd_refl_azi = _reflected_azi_index(phi, track.azi_index)
            bwd_target_is_fwd = not _is_vertical(track.entry_surface)
            bwd_link = self._find_link(
                track.entry_point, bwd_refl_azi, bwd_target_is_fwd
            )

            self.tracks[i] = Track(
                segments=track.segments,
                azi_index=track.azi_index,
                entry_point=track.entry_point,
                exit_point=track.exit_point,
                entry_surface=track.entry_surface,
                exit_surface=track.exit_surface,
                fwd_link=fwd_link,
                fwd_link_fwd=fwd_target_is_fwd,
                bwd_link=bwd_link,
                bwd_link_fwd=bwd_target_is_fwd,
            )

    def _find_link(
        self,
        exit_point: tuple[float, float],
        target_azi: int,
        target_is_fwd: bool,
    ) -> int:
        """Find the track at target_azi whose entry (fwd) or exit (bwd)
        point best matches the given exit_point."""
        ex, ey = exit_point
        best_idx = -1
        best_dist = float("inf")

        for j in self.tracks_per_azi[target_azi]:
            other = self.tracks[j]
            if target_is_fwd:
                ox, oy = other.entry_point
            else:
                ox, oy = other.exit_point
            d = (ox - ex) ** 2 + (oy - ey) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = j

        return best_idx

    def effective_spacing(self, azi_index: int) -> float:
        """Effective perpendicular ray spacing for azimuthal angle index."""
        return self._effective_spacing[azi_index]
