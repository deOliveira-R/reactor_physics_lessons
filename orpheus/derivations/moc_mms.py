r"""Method of Manufactured Solutions (MMS) for MOC spatial verification.

MMS verification of the 2-D flat-source Method of Characteristics
spatial operator.  Unlike the SN MMS (which injects a per-ordinate
source into ``solve_sn_fixed_source``), the MOC MMS requires a
**per-segment, per-angle** manufactured source because the MOC
solver's flat source is isotropic and the MMS streaming residual
is angle-dependent.

Ansatz
------

Smooth radial scalar flux on the square pin cell :math:`[0, P]^{2}`,
centred at :math:`(P/2, P/2)`:

.. math::
   :label: moc-mms-psi-ref

   \phi_{\text{ref}}(r)
   = 1 + A\,\cos\!\bigl(\pi r^{2} / R^{2}\bigr),
   \qquad r = \sqrt{(x - P/2)^{2} + (y - P/2)^{2}},
   \quad R = P/2

The radial form is chosen because any product of Cartesian cosines
centred at :math:`(P/2, P/2)` integrates to zero over annuli (by
symmetry of :math:`\sin\theta\cos\theta`), making the FSR averages
degenerate.  The :math:`r^{2}` argument ensures :math:`C^{\infty}`
smoothness at :math:`r = 0`.

Manufactured source
-------------------

With isotropic angular flux :math:`\psi_{\text{ref}} = \phi_{\text{ref}} / (4\pi)`:

.. math::
   :label: moc-mms-qext

   Q_{\text{mms}}(x, y, \varphi_a, \theta_p)
   = \frac{1}{4\pi}\Bigl[
       \sin\theta_p\,\bigl(
           \cos\varphi_a\;\partial_x\phi_{\text{ref}}
         + \sin\varphi_a\;\partial_y\phi_{\text{ref}}
       \bigr)
     + \Sigma_t\;\phi_{\text{ref}}
   \Bigr]

Partial derivatives of the radial ansatz:

.. math::

   \partial_x\phi = -\frac{2\pi A}{R^{2}}\,(x - P/2)\,
                      \sin\!\bigl(\pi r^{2}/R^{2}\bigr)

(analogously for :math:`\partial_y`).

The streaming residual is angle-dependent and vanishes when averaged
over :math:`4\pi`.  An isotropic per-FSR external source cannot carry
this information; per-segment injection is necessary.

FSR averages
------------

For annular FSRs from :math:`r_1` to :math:`r_2`:

.. math::

   \langle\phi_{\text{ref}}\rangle_i
   = 1 + \frac{A\,R^{2}}{\pi\,(r_2^{2} - r_1^{2})}
     \bigl[\sin(\pi r_2^{2}/R^{2}) - \sin(\pi r_1^{2}/R^{2})\bigr]

This is computed analytically — no numerical quadrature required
for the inner annuli.

Convergence
-----------

The flat-source approximation replaces the spatially-varying
:math:`Q_{\text{mms}}(s)` along each segment by its midpoint value.
The resulting scalar-flux error converges at
:math:`\mathcal{O}(h^{2})` in the FSR linear dimension (with
fixed track spacing and angular quadrature).

.. seealso::

   - :doc:`/theory/method_of_characteristics` — MMS verification section
   - :mod:`orpheus.derivations.sn_mms` — analogous SN MMS module
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.moc.geometry import MOCMesh
from orpheus.moc.quadrature import MOCQuadrature

from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)


@dataclass(frozen=True)
class MOCPinCellMMSCase:
    r"""Manufactured-solution case for 2-D flat-source MOC verification.

    The ansatz is a smooth radial scalar flux
    :math:`\phi(r) = 1 + A\cos(\pi r^{2}/R^{2})` on the square cell,
    with isotropic angular flux.

    Attributes
    ----------
    name : str
        Registry key.
    sigma_t : float
        Total cross section (1-group, homogeneous, pure absorber).
    pitch : float
        Square cell side length (cm).
    amplitude : float
        Cosine modulation amplitude :math:`A` (must be < 1).
    n_azi, n_polar : int
        Angular quadrature parameters (fixed across refinements).
    ray_spacing : float
        Perpendicular track spacing (cm), also fixed.
    tolerance : str
        Expected spatial convergence order.
    equation_labels : tuple[str, ...]
        Sphinx labels exercised by tests.
    """

    name: str
    sigma_t: float
    pitch: float
    amplitude: float
    n_azi: int
    n_polar: int
    ray_spacing: float
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "characteristic-ode",
        "bar-psi",
        "boyd-eq-45",
        "moc-mms-psi-ref",
        "moc-mms-qext",
    )

    @property
    def _R(self) -> float:
        return self.pitch / 2.0

    @property
    def _cx(self) -> float:
        return self.pitch / 2.0

    @property
    def _cy(self) -> float:
        return self.pitch / 2.0

    def _r2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x - self._cx) ** 2 + (y - self._cy) ** 2

    def phi_ref(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r""":math:`1 + A\cos(\pi r^{2}/R^{2})`."""
        return 1.0 + self.amplitude * np.cos(np.pi * self._r2(x, y) / self._R ** 2)

    def dphi_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        R2 = self._R ** 2
        return (
            -self.amplitude * 2.0 * np.pi / R2
            * (x - self._cx)
            * np.sin(np.pi * self._r2(x, y) / R2)
        )

    def dphi_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        R2 = self._R ** 2
        return (
            -self.amplitude * 2.0 * np.pi / R2
            * (y - self._cy)
            * np.sin(np.pi * self._r2(x, y) / R2)
        )

    def phi_ref_fsr_average(self, moc_mesh: MOCMesh) -> np.ndarray:
        r"""Volume-averaged :math:`\langle\phi\rangle_i` per FSR.

        For inner annuli the integral is analytical:

        .. math::

           \langle\phi\rangle_i = 1 +
             \frac{A\,R^{2}}{\pi\,(r_2^{2} - r_1^{2})}
             \bigl[\sin(\pi r_2^{2}/R^{2}) - \sin(\pi r_1^{2}/R^{2})\bigr]

        The outermost (square-border) FSR uses Cartesian quadrature.
        """
        nr = moc_mesh.n_regions
        edges = moc_mesh.mesh.edges
        R2 = self._R ** 2
        A = self.amplitude
        averages = np.empty(nr)

        for k in range(nr):
            r_in = edges[k]
            is_outermost = (k == nr - 1)

            if not is_outermost:
                r_out = edges[k + 1]
                dr2 = r_out ** 2 - r_in ** 2
                if dr2 < 1e-30:
                    averages[k] = self.phi_ref(
                        np.array([self._cx + r_in]),
                        np.array([self._cy]),
                    )[0]
                else:
                    sin_diff = (
                        np.sin(np.pi * r_out ** 2 / R2)
                        - np.sin(np.pi * r_in ** 2 / R2)
                    )
                    averages[k] = 1.0 + A * R2 / (np.pi * dr2) * sin_diff
            else:
                from numpy.polynomial.legendre import leggauss
                n_q = 32
                nodes, weights = leggauss(n_q)
                P = self.pitch
                x_n = 0.5 * P * (nodes + 1.0)
                y_n = 0.5 * P * (nodes + 1.0)
                wx = 0.5 * P * weights
                wy = 0.5 * P * weights
                r_inner = edges[k]

                integral = 0.0
                total_w = 0.0
                for ix in range(n_q):
                    for iy in range(n_q):
                        xp, yp = x_n[ix], y_n[iy]
                        if (xp - self._cx) ** 2 + (yp - self._cy) ** 2 > r_inner ** 2:
                            w = wx[ix] * wy[iy]
                            integral += w * self.phi_ref(
                                np.array([xp]), np.array([yp])
                            )[0]
                            total_w += w
                averages[k] = integral / total_w if total_w > 0 else 1.0

        return averages


def mms_sweep(case: MOCPinCellMMSCase, moc_mesh: MOCMesh) -> np.ndarray:
    r"""Per-characteristic MMS sweep returning scalar flux per FSR.

    Single-pass transport sweep with per-segment manufactured sources.
    Boundary fluxes are set to the reference angular flux (no
    reflective iteration).

    The scalar flux is reconstructed as:

    .. math::

       \phi_i = \langle\phi_{\text{ref}}\rangle_i
              + \frac{\delta\phi_i}{A_i\,\Sigma_t}

    The equilibrium term :math:`\langle\phi_{\text{ref}}\rangle_i` is
    computed analytically (the isotropic part of the MMS source
    averages to :math:`\Sigma_t\,\langle\phi_{\text{ref}}\rangle_i/(4\pi)`
    since the streaming residual integrates to zero over :math:`4\pi`).
    Only the transport correction :math:`\delta\phi_i` comes from the
    sweep — this is the term that carries the flat-source
    :math:`\mathcal{O}(h^{2})` error.
    """
    quad = moc_mesh.quad
    nr = moc_mesh.n_regions
    sig_t = case.sigma_t
    inv_4pi = 1.0 / (4.0 * np.pi)

    delta_phi = np.zeros(nr)

    for a_idx in range(quad.n_azi):
        cos_phi = np.cos(quad.phi[a_idx])
        sin_phi = np.sin(quad.phi[a_idx])
        ts = moc_mesh.effective_spacing(a_idx)
        omega_a = quad.omega_azi[a_idx]

        for t_idx in moc_mesh.tracks_per_azi[a_idx]:
            track = moc_mesh.tracks[t_idx]
            entry_x, entry_y = track.entry_point
            exit_x, exit_y = track.exit_point
            segments = track.segments
            n_seg = len(segments)

            seg_entry_offsets = np.empty(n_seg)
            seg_mid_offsets = np.empty(n_seg)
            s = 0.0
            for si, seg in enumerate(segments):
                seg_entry_offsets[si] = s
                seg_mid_offsets[si] = s + 0.5 * seg.length
                s += seg.length
            total_track_len = s

            for p_idx in range(quad.n_polar):
                sin_p = quad.sin_polar[p_idx]
                omega_p = quad.omega_polar[p_idx]
                weight = 4.0 * np.pi * omega_a * omega_p * ts * sin_p

                # --- Forward sweep ---
                for si, seg in enumerate(segments):
                    x_entry = entry_x + seg_entry_offsets[si] * cos_phi
                    y_entry = entry_y + seg_entry_offsets[si] * sin_phi
                    psi_in = case.phi_ref(
                        np.array([x_entry]), np.array([y_entry])
                    )[0] * inv_4pi

                    x_m = entry_x + seg_mid_offsets[si] * cos_phi
                    y_m = entry_y + seg_mid_offsets[si] * sin_phi
                    xm_a, ym_a = np.array([x_m]), np.array([y_m])

                    phi_val = case.phi_ref(xm_a, ym_a)[0]
                    streaming = (
                        cos_phi * case.dphi_dx(xm_a, ym_a)[0]
                        + sin_phi * case.dphi_dy(xm_a, ym_a)[0]
                    )
                    q_seg = inv_4pi * (phi_val + sin_p * streaming / sig_t)

                    tau = sig_t * seg.length / sin_p
                    if tau < 1e-10:
                        one_minus_exp = tau * (1.0 - 0.5 * tau)
                    else:
                        one_minus_exp = 1.0 - np.exp(-tau)

                    dpsi = (psi_in - q_seg) * one_minus_exp
                    delta_phi[seg.region_id] += weight * dpsi

                # --- Backward sweep ---
                cumul_bwd = 0.0
                for si in range(n_seg - 1, -1, -1):
                    seg = segments[si]
                    entry_offset_bwd = total_track_len - seg_entry_offsets[si] - seg.length
                    x_entry_b = exit_x - entry_offset_bwd * cos_phi
                    y_entry_b = exit_y - entry_offset_bwd * sin_phi
                    psi_in = case.phi_ref(
                        np.array([x_entry_b]), np.array([y_entry_b])
                    )[0] * inv_4pi

                    mid_offset = entry_offset_bwd + 0.5 * seg.length
                    x_m = exit_x - mid_offset * cos_phi
                    y_m = exit_y - mid_offset * sin_phi
                    xm_a, ym_a = np.array([x_m]), np.array([y_m])

                    phi_val = case.phi_ref(xm_a, ym_a)[0]
                    streaming = (
                        cos_phi * case.dphi_dx(xm_a, ym_a)[0]
                        + sin_phi * case.dphi_dy(xm_a, ym_a)[0]
                    )
                    q_seg_val = inv_4pi * (phi_val - sin_p * streaming / sig_t)

                    tau = sig_t * seg.length / sin_p
                    if tau < 1e-10:
                        one_minus_exp = tau * (1.0 - 0.5 * tau)
                    else:
                        one_minus_exp = 1.0 - np.exp(-tau)

                    dpsi = (psi_in - q_seg_val) * one_minus_exp
                    delta_phi[seg.region_id] += weight * dpsi

    phi_ref_avg = case.phi_ref_fsr_average(moc_mesh)
    areas = moc_mesh.region_areas
    phi = phi_ref_avg + delta_phi / (areas * sig_t)

    return phi


def build_moc_mms_case(
    sigma_t: float = 1.0,
    pitch: float = 2.0,
    amplitude: float = 0.3,
    n_azi: int = 32,
    n_polar: int = 3,
    ray_spacing: float = 0.02,
    name: str = "moc_mms_pincell",
) -> MOCPinCellMMSCase:
    r"""Build the canonical MOC MMS case.

    Default parameters give a 1-group pure absorber with
    :math:`\Sigma_t = 1\,\text{cm}^{-1}` on a 2 cm pitch pin cell.
    The amplitude :math:`A = 0.3` keeps :math:`\phi_{\text{ref}} > 0`
    everywhere.  Fine angular quadrature (32 azimuthal, 3 polar)
    and tight track spacing (0.02 cm) ensure that the volume-
    conservation property holds to :math:`\sim 0.1\%`.
    """
    return MOCPinCellMMSCase(
        name=name,
        sigma_t=sigma_t,
        pitch=pitch,
        amplitude=amplitude,
        n_azi=n_azi,
        n_polar=n_polar,
        ray_spacing=ray_spacing,
    )


def build_moc_mesh(case: MOCPinCellMMSCase, n_annuli: int) -> MOCMesh:
    """Build an MOCMesh with ``n_annuli + 1`` FSRs.

    The inner ``n_annuli`` regions are equal-area annuli inside
    the inscribed circle (radius P/2).  The outermost Mesh1D cell
    spans to the WS outer edge (the square-border FSR).
    """
    P = case.pitch
    ws_r = P / np.sqrt(np.pi)
    r_inscribed = P / 2.0
    area_inscribed = np.pi * r_inscribed ** 2
    area_per_annulus = area_inscribed / n_annuli
    radii = [np.sqrt(k * area_per_annulus / np.pi) for k in range(1, n_annuli + 1)]
    edges = np.array([0.0] + radii + [ws_r])
    mat_ids = np.zeros(n_annuli + 1, dtype=int)

    mesh = Mesh1D(edges=edges, mat_ids=mat_ids, coord=CoordSystem.CYLINDRICAL)
    quad = MOCQuadrature.create(n_azi=case.n_azi, n_polar=case.n_polar)
    return MOCMesh(mesh, quad, ray_spacing=case.ray_spacing)


# ── Phase-0 ContinuousReferenceSolution wrapper ───────────────────────

def _build_moc_mms_continuous_reference() -> ContinuousReferenceSolution:
    mms_case = build_moc_mms_case()

    def phi(x: np.ndarray, g: int = 0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return mms_case.phi_ref(x, np.full_like(x, mms_case._cy))

    return ContinuousReferenceSolution(
        name=mms_case.name,
        problem=ProblemSpec(
            materials={},
            geometry_type="pin-cell-2d",
            geometry_params={
                "pitch": mms_case.pitch,
                "mms_case": mms_case,
            },
            boundary_conditions={"all": "reflective"},
            external_source=None,
            is_eigenvalue=False,
            n_groups=1,
        ),
        operator_form="differential-moc",
        phi=phi,
        provenance=Provenance(
            citation=(
                "Salari & Knupp, SAND2000-1444 §6 (smooth MMS); "
                "Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals); "
                "Boyd et al. 2014, Ann. Nucl. Energy 68, 43-52 (MOC formulation)"
            ),
            derivation_notes=(
                "1-group pure-absorber MOC spatial-operator reference "
                "via the Method of Manufactured Solutions with "
                "per-characteristic source injection.  Ansatz "
                "φ_ref(r) = 1 + A cos(π r²/R²) with R = P/2, giving "
                "isotropic ψ_ref = φ_ref/(4π).  Manufactured source "
                "Q_mms = dψ_ref/ds + Σ_t ψ_ref is angle-dependent "
                "(streaming residual) and injected per segment.  "
                "Scalar flux reconstruction via generalised Boyd "
                "Eq. 45 with per-segment asymptotic fluxes.  "
                "Flat-source error converges O(h²)."
            ),
            sympy_expression=(
                r"\phi_{\text{ref}}(r) = 1 + A\cos(\pi r^2/R^2)"
            ),
            precision_digits=None,
        ),
        k_eff=None,
        psi=None,
        equation_labels=mms_case.equation_labels,
        vv_level="L1",
        description=(
            "1-group pure-absorber MOC MMS — per-characteristic "
            "manufactured source on 2-D pin cell.  Phase-2.2a "
            "continuous reference."
        ),
        tolerance="O(h^2)",
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Return the Phase-2.2a continuous references produced by this module."""
    return [_build_moc_mms_continuous_reference()]
