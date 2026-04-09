"""MOC transport solver satisfying the EigenvalueSolver protocol.

Solves the multi-group neutron transport equation via the Method of
Characteristics on a 2-D pin-cell geometry with reflective boundary
conditions.  The flat-source approximation is used within each FSR,
and the angular discretisation uses a product quadrature (azimuthal x
Tabuchi-Yamamoto polar).

Reference: Boyd et al. (2014) "The OpenMOC Method of Characteristics
Neutral Particle Transport Code", Ann. Nucl. Energy 68, 43-52.
"""

from __future__ import annotations

import numpy as np

from orpheus.data.macro_xs.mixture import Mixture

from .geometry import MOCMesh


class MOCSolver:
    """2-D Method of Characteristics eigenvalue solver.

    Satisfies the :class:`~numerics.eigenvalue.EigenvalueSolver` protocol.

    Flux representation: ``(n_regions, ng)`` — scalar flux per flat-source
    region and energy group.
    """

    def __init__(
        self,
        moc_mesh: MOCMesh,
        materials: dict[int, Mixture],
        keff_tol: float = 1e-6,
        flux_tol: float = 1e-5,
        n_inner_sweeps: int = 15,
    ) -> None:
        self.geom = moc_mesh
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        self.n_inner_sweeps = n_inner_sweeps

        nr = moc_mesh.n_regions
        _any_mat = next(iter(materials.values()))
        self.ng = _any_mat.ng

        # Per-region cross sections
        self.sig_t = np.empty((nr, self.ng))
        self.sig_a = np.empty((nr, self.ng))
        self.sig_p = np.empty((nr, self.ng))
        self.chi = np.empty((nr, self.ng))
        self.sig_s0: list[np.ndarray] = []
        self.sig2: list[np.ndarray] = []

        for k in range(nr):
            mat_id = int(moc_mesh.region_mat_ids[k])
            mix = materials[mat_id]
            self.sig_t[k, :] = mix.SigT
            self.sig_a[k, :] = mix.absorption_xs
            self.sig_p[k, :] = mix.SigP
            self.chi[k, :] = mix.chi
            self.sig_s0.append(mix.SigS[0].toarray())
            self.sig2.append(mix.Sig2.toarray())

        # Persistent boundary angular fluxes (survive between outer iters)
        n_tracks = len(moc_mesh.tracks)
        n_polar = moc_mesh.quad.n_polar
        ng = self.ng
        self._fwd_bflux = np.zeros((n_tracks, n_polar, ng))
        self._bwd_bflux = np.zeros((n_tracks, n_polar, ng))

    # ── EigenvalueSolver protocol ────────────────────────────────────

    def initial_flux_distribution(self) -> np.ndarray:
        return np.ones((self.geom.n_regions, self.ng))

    def compute_fission_source(
        self,
        flux_distribution: np.ndarray,
        keff: float,
    ) -> np.ndarray:
        """Fission source: chi * (SigP . phi) / keff."""
        nr = self.geom.n_regions
        fission_source = np.empty((nr, self.ng))
        for i in range(nr):
            production = self.sig_p[i, :] @ flux_distribution[i, :]
            fission_source[i, :] = self.chi[i, :] * production / keff
        return fission_source

    def solve_fixed_source(
        self,
        fission_source: np.ndarray,
        flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """MOC transport sweep with inner scattering iterations.

        Each inner sweep:
        1. Build isotropic source Q from fission + scattering + (n,2n)
        2. Sweep all tracks (forward + backward), accumulate delta_phi
        3. Update scalar flux from delta_phi (Boyd Eq. 45)
        """
        geom = self.geom
        quad = geom.quad
        nr = geom.n_regions
        ng = self.ng

        phi = flux_distribution.copy()

        for _inner in range(self.n_inner_sweeps):
            # 1. Build source: Q[i,g] = (1/4pi) * [fission + scatter + n2n]
            Q = np.empty((nr, ng))
            for i in range(nr):
                scatter = self.sig_s0[i].T @ phi[i, :]
                n2n = 2.0 * self.sig2[i].T @ phi[i, :]
                Q[i, :] = (fission_source[i, :] + scatter + n2n) / (4.0 * np.pi)

            # Q / sig_t (asymptotic angular flux per region)
            q_over_sigt = np.zeros((nr, ng))
            mask = self.sig_t > 0
            q_over_sigt[mask] = Q[mask] / self.sig_t[mask]

            # 2. Sweep all tracks, accumulate delta_phi
            delta_phi = np.zeros((nr, ng))

            for a_idx in range(quad.n_azi):
                ts = geom.effective_spacing(a_idx)
                omega_a = quad.omega_azi[a_idx]

                for t_idx in geom.tracks_per_azi[a_idx]:
                    track = geom.tracks[t_idx]

                    for p_idx in range(quad.n_polar):
                        sin_p = quad.sin_polar[p_idx]
                        omega_p = quad.omega_polar[p_idx]
                        weight = 4.0 * np.pi * omega_a * omega_p * ts * sin_p

                        # --- Forward sweep ---
                        psi = self._fwd_bflux[t_idx, p_idx, :].copy()
                        for seg in track.segments:
                            rid = seg.region_id
                            for g in range(ng):
                                if self.sig_t[rid, g] <= 0:
                                    continue
                                tau = self.sig_t[rid, g] * seg.length / sin_p
                                if tau < 1e-10:
                                    one_minus_exp = tau * (1.0 - 0.5 * tau)
                                else:
                                    one_minus_exp = 1.0 - np.exp(-tau)
                                dpsi = (psi[g] - q_over_sigt[rid, g]) * one_minus_exp
                                psi[g] -= dpsi
                                delta_phi[rid, g] += weight * dpsi

                        # Pass outgoing to linked track
                        fwd_t = track.fwd_link
                        if fwd_t >= 0:
                            if track.fwd_link_fwd:
                                self._fwd_bflux[fwd_t, p_idx, :] = psi
                            else:
                                self._bwd_bflux[fwd_t, p_idx, :] = psi

                        # --- Backward sweep ---
                        psi = self._bwd_bflux[t_idx, p_idx, :].copy()
                        for seg in reversed(track.segments):
                            rid = seg.region_id
                            for g in range(ng):
                                if self.sig_t[rid, g] <= 0:
                                    continue
                                tau = self.sig_t[rid, g] * seg.length / sin_p
                                if tau < 1e-10:
                                    one_minus_exp = tau * (1.0 - 0.5 * tau)
                                else:
                                    one_minus_exp = 1.0 - np.exp(-tau)
                                dpsi = (psi[g] - q_over_sigt[rid, g]) * one_minus_exp
                                psi[g] -= dpsi
                                delta_phi[rid, g] += weight * dpsi

                        bwd_t = track.bwd_link
                        if bwd_t >= 0:
                            if track.bwd_link_fwd:
                                self._fwd_bflux[bwd_t, p_idx, :] = psi
                            else:
                                self._bwd_bflux[bwd_t, p_idx, :] = psi

            # 3. Update scalar flux: Boyd Eq. 45
            #    phi_i = (4pi/sig_t) * [Q_i + delta_phi_i / A_i]
            #    where 4pi is already absorbed into the weight.
            #    So: phi_i = (4pi*Q_i + delta_phi_i/A_i) / sig_t_i
            phi_new = np.zeros((nr, ng))
            for i in range(nr):
                for g in range(ng):
                    if self.sig_t[i, g] > 0:
                        phi_new[i, g] = (
                            4.0 * np.pi * Q[i, g]
                            + delta_phi[i, g] / geom.region_areas[i]
                        ) / self.sig_t[i, g]
            phi = phi_new

        # Normalise flux (prevent overflow between outer iterations)
        total_prod = 0.0
        for i in range(nr):
            sig2_out = np.array(self.sig2[i].sum(axis=1)).ravel()
            total_prod += (
                (self.sig_p[i, :] + 2 * sig2_out) @ phi[i, :] * geom.region_areas[i]
            )
        if total_prod > 0:
            phi *= 1.0 / total_prod

        return phi

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        """k_eff = production / absorption (zero leakage, reflective BCs)."""
        nr = self.geom.n_regions
        p_rate = 0.0
        a_rate = 0.0
        for i in range(nr):
            A_i = self.geom.region_areas[i]
            phi_i = flux_distribution[i, :]
            sig2_out = np.array(self.sig2[i].sum(axis=1)).ravel()
            p_rate += (self.sig_p[i, :] + 2 * sig2_out) @ phi_i * A_i
            a_rate += self.sig_a[i, :] @ phi_i * A_i
        keff = p_rate / a_rate if a_rate > 0 else 1.0
        print(f"  keff = {keff:9.5f}")
        return keff

    def converged(
        self,
        keff: float,
        keff_old: float,
        flux_distribution: np.ndarray,
        flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        if iteration <= 2:
            return False
        dk = abs(keff - keff_old)
        dphi = np.linalg.norm(flux_distribution - flux_old) / max(
            np.linalg.norm(flux_distribution), 1e-30
        )
        return dk < self.keff_tol and dphi < self.flux_tol
