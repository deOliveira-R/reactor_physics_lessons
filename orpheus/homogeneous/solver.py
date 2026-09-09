"""Homogeneous infinite-medium reactor eigenvalue solver.

Solves for the neutron spectrum and k-infinity in an infinite homogeneous
medium.  All spatial and angular dependence integrates out; the transport
equation reduces to the pure energy-balance eigenvalue problem

    A φ = (1/k) F φ,
        A = diag(Σ_t) − Σ_s0ᵀ − 2·Σ₂ᵀ,
        F = χ ⊗ νΣ_f,

with k_inf = λ_max(A⁻¹F).

The eigenproblem is spelled in the operator algebra itself (taxonomy step
5b) over the problem's OWN hub, :class:`HomogeneousProblem` (CS4c coda,
ruling R-c1, 2026-09-08): the place the consumed objects live — the pose
space, the mixture-direct kernel and cross-section fields, and the bound
operators built on them — as cached, per-instance state the solver reads
off. The loss operator ``A = C − K_iso`` is the collision diagonal
C = diag(Σ_t) minus the model-shared isotropic energy operators
``IsotropicScattering`` (Σ_s0ᵀ) and ``IsotropicN2N`` (2·Σ₂ᵀ); streaming L
is identically zero in an infinite medium and is dropped — and the
multiplication operator is the composition
``K = MatrixInverseOperator(A) @ F`` (one eager LU factorization at
construction; the first production consumer of the dense direct inverse),
whose materialization feeds the shared Perron–Frobenius extraction
:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`.  The whole
infinite-medium spectrum runs through the SAME operator algebra the
meshed SN solver uses, not a bespoke matrix (#276).

(n,2n) convention: the (n,2n) reaction is a loss-side multiplicity-2
transfer.  It lives ONLY in A (as 2·Σ₂ᵀ), NOT in the fission production
F — the two emitted neutrons are redistributed by 2·Σ₂, they are not
produced with the fission spectrum χ.  Production is νΣ_f only.

.. seealso:: :ref:`theory-homogeneous` — Key Facts, eigenvalue equations, scattering convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis
from orpheus.numerics.eigenvalue import dominant_eigenpair
from orpheus.numerics.matrix_inverse_operator import MatrixInverseOperator
from orpheus.numerics.space import FunctionSpace
from orpheus.transport.fields.cross_section_field import CrossSectionField
from orpheus.transport.kernels import FissionKernel, TransferKernel
from orpheus.transport.material_field import FissionMaterialField, TransferMaterialField
from orpheus.transport.reaction_rate_functional import IntegratedReactionRate
from orpheus.transport.operators.isotropic_transfer import (
    IsotropicFission,
    IsotropicN2N,
    IsotropicScattering,
)
from orpheus.transport.operators.multiplication_operator import MultiplicationOperator

if TYPE_CHECKING:
    from orpheus.numerics.operator import OperatorProduct, OperatorSum


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HomogeneousResult:
    """Result of a homogeneous infinite reactor calculation.

    The energy-grid diagnostics (``representative_energy``, ``energy_widths``,
    ``lethargy_widths``) are ``None`` when the underlying
    :class:`~orpheus.data.macro_xs.mixture.Mixture` has no physical energy grid
    (synthetic / Sood-style XS, post-Phase-E). In that case ``flux_per_energy`` /
    ``flux_per_lethargy`` raise — the quantities are not defined without a grid.

    These three are the :class:`~orpheus.data.energy_grid.EnergyGrid` value
    object's own properties — ``representative_energy`` is the **geometric** group
    centre :math:`\\sqrt{E_{\\rm up}E_{\\rm lo}}` (the natural abscissa on the log
    energy axis, NOT the arithmetic midpoint) — read from ``mixture.energy_grid``
    rather than re-deriving the group geometry here.
    """

    k_inf: float
    flux: np.ndarray  # (NG,) — group fluxes normalised to 100 n/cm³/s production
    representative_energy: np.ndarray | None  # (NG,) — geometric group-centre energies (eV); None if no grid
    energy_widths: np.ndarray | None  # (NG,) — ΔE group widths (eV); None if no grid
    lethargy_widths: np.ndarray | None  # (NG,) — Δu lethargy widths; None if no grid
    sig_prod: float  # one-group production XS (1/cm)
    sig_abs: float  # one-group absorption XS (1/cm)
    mixture: Mixture

    @property
    def flux_per_energy(self) -> np.ndarray:
        if self.energy_widths is None:
            raise ValueError(
                "flux_per_energy is undefined for synthetic XS without an "
                "energy grid (mixture.eg is None). Build the Mixture from "
                "an Isotope library or pass an explicit eg= to make_mixture."
            )
        return self.flux / self.energy_widths

    @property
    def flux_per_lethargy(self) -> np.ndarray:
        if self.lethargy_widths is None:
            raise ValueError(
                "flux_per_lethargy is undefined for synthetic XS without "
                "an energy grid (mixture.eg is None). Build the Mixture "
                "from an Isotope library or pass an explicit eg= to "
                "make_mixture."
            )
        return self.flux / self.lethargy_widths


# ---------------------------------------------------------------------------
# Solver — k∞ = λ_max(A⁻¹F) over the transport operator algebra
# ---------------------------------------------------------------------------

def _pose_space(mix: Mixture) -> FunctionSpace:
    r"""The space the infinite-medium problem poses on: Energy ⊗ the quotient point.

    Minted from the MIXTURE — the problem's own physics — never read off
    a carrier (CS4a K2): the energy axis goes through the ONE energy-arm
    rule (:meth:`~orpheus.numerics.axis.EnergyAxis.from_materials`, the
    same rule ``MaterialMesh.bulk_space`` routes through, so the two
    spellings cannot diverge), and the spatial factor is the explicit
    quotient point with the COUNTING weight (``weights=None`` — the
    normalized per-unit-volume density convention; the counting-measure
    premise the rate pairing below rests on).

    Since the CS4c coda (R-c1) this is the ONE spelling of the pose: the
    hub's :attr:`HomogeneousProblem.space` calls it, and nothing on the
    homogeneous path builds a carrier any more (a genuine unit-cell
    ``Mesh1D`` carrier's ``bulk_space`` mints an ``==`` space — the
    identity-bridge gate keeps that reference honest).

    **CS4a-R rulings this space's consumers rest on.** The reaction
    rates in :func:`solve_homogeneous_infinite` read the typed
    integrated co-vector
    (:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
    — EE-1, landed CS4b S7; until then they were the tree's only raw
    two-ndarray pairing site, a recorded CS4a deliberate). The ⟨1,φ⟩
    total-flux leg IS this space's own pairing — the integration
    co-vector of the pose, not a reaction rate. And both condensed cross
    sections are spelled as SAME-PAIRING ratios ⟨Σ,φ⟩/⟨1,φ⟩, so they are
    measure-invariant whatever weight a future pose carries (XD-6) — the
    counting weight here is a convention this function states, not a
    contract those ratios depend on.
    """
    return FunctionSpace.of_axes(
        EnergyAxis.from_materials([mix]),
        # The one-cell counting axis is honestly generator-less (CS5):
        # the infinite-medium pose has no mesh, so no spatial measure
        # object exists to name — ``generator=None`` IS the record.
        Axis("spatial", (1,), kind=BasisKind.NODAL),
    )


@dataclass(frozen=True)
class HomogeneousProblem:
    r"""The infinite-medium problem — the HUB the homogeneous family's consumed objects live on.

    Ruled at the CS4c coda (R-c1, the user, 2026-09-08): *"The homogeneous
    problem needs a hub, just like the function SNMesh (future SNProblem)
    currently fulfills, to act as the place the consumed objects live (and
    a save state)."* This is that hub, minted in the solver module for now;
    the carve into a standalone ``HomogeneousProblem`` module with a thin
    Problem → Solution solver is the consumers campaign's, alongside
    ``SNMesh`` → ``SNProblem`` (plan §22.5).

    **What it determines, from its generating datum alone.** A
    :class:`~orpheus.data.macro_xs.mixture.Mixture` is the whole physics
    of an infinite homogeneous medium, so every consumed object is minted
    from it and from nothing else (O1's *honest pose*: no ``[0, 1]`` edges,
    no invented node, no coordinate system on the path — until the coda
    the cross sections came through a fabricated one-cell
    ``MaterialMesh.from_materials`` carrier whose edges, node and chart
    `[M]` nothing consumed):

    * the pose :attr:`space` — Energy ⊗ the quotient point
      (:func:`_pose_space`, the one spelling);
    * the material fields on the quotient point's one-cell :attr:`layout`
      — the scattering and (n,2n) transfer stacks
      (:attr:`scattering`, :attr:`n2n`) and the fission datum
      (:attr:`fission`), each the kernel tier's own mixture-direct mint;
    * the cross-section fields, BORN on the pose
      (:attr:`total_cross_section_field`,
      :attr:`absorption_cross_section_field`,
      :attr:`fission_production_field`) — never re-posed, so a field on a
      space nothing checks is unspellable here;
    * the bound operators on the pose — :attr:`collision`,
      :attr:`isotropic_transfer` (``IsoS + IsoN2N``), :attr:`loss`
      (``C − K_iso``), :attr:`production` (the fission dyad) and the
      multiplication operator :attr:`multiplication` (``A⁻¹F``, one eager
      LU at construction) — and the typed reaction-rate co-vectors
      :attr:`production_rate` / :attr:`absorption_rate`.

    **State.** Every consumed object is a ``cached_property`` — minted once
    per instance and then ``is``-identical on every read (the hub is the
    owner: identity is ``is`` WITHIN it), keyed by the mixture in the sense
    that two hubs over equal mixtures mint ``==`` objects (they are two
    owners; ``is`` across them is not a claim). Per-instance, never
    module-scope: a module memo would mask every decoy a gate installs on
    the pose (the verification plan's H-2).

    `[M]` bit-identical to the retired carrier route on the D5 population
    (8 of 8: ``k_inf``, the flux bytes, both rates) and on the operator
    tier (``A``/``F`` against the frozen pre-carve capture) — a re-source,
    not a re-baseline.
    """

    mixture: Mixture

    @property
    def ng(self) -> int:
        """The group count — the mixture's."""
        return self.mixture.ng

    @cached_property
    def space(self) -> FunctionSpace:
        r"""The pose, Energy ⊗ the quotient point — :func:`_pose_space`, the one spelling."""
        return _pose_space(self.mixture)

    @cached_property
    def layout(self) -> dict[int, tuple[np.ndarray, ...]]:
        r"""The quotient point's one-cell material layout: material ``0`` on cell ``0``."""
        return {0: (np.arange(1),)}

    # ── the material fields (the kernel tier's mixture-direct mints) ────
    @cached_property
    def scattering(self) -> TransferMaterialField:
        r"""The scattering channel's Legendre stack (yield 1) over the layout."""
        return TransferMaterialField({0: TransferKernel.scattering(self.mixture)}, self.layout)

    @cached_property
    def n2n(self) -> TransferMaterialField:
        r"""The (n,2n) channel's Legendre stack (yield 2) over the layout."""
        return TransferMaterialField({0: TransferKernel.n2n(self.mixture)}, self.layout)

    @cached_property
    def fission(self) -> FissionMaterialField:
        r"""The fission datum :math:`\chi \otimes \nu\Sigma_f` over the layout."""
        return FissionMaterialField({0: FissionKernel.from_mixture(self.mixture)}, self.layout)

    # ── the cross-section fields, born on the pose ─────────────────────
    def _field(self, values: np.ndarray) -> CrossSectionField:
        return CrossSectionField(values=np.asarray(values, dtype=float).reshape(self.ng, 1), space=self.space)

    @cached_property
    def total_cross_section_field(self) -> CrossSectionField:
        r""":math:`\Sigma_t` on the pose (1/cm)."""
        return self._field(self.mixture.SigT)

    @cached_property
    def absorption_cross_section_field(self) -> CrossSectionField:
        r""":math:`\Sigma_a` on the pose (1/cm)."""
        return self._field(self.mixture.absorption_xs)

    @cached_property
    def fission_production_field(self) -> CrossSectionField:
        r""":math:`\nu\Sigma_f` on the pose (1/cm) — production, the only fission channel in :math:`F`."""
        return self._field(self.mixture.SigP)

    # ── the bound operators on the pose ────────────────────────────────
    @cached_property
    def collision(self) -> MultiplicationOperator:
        r"""The collision diagonal :math:`C = M[\Sigma_t]` on the pose."""
        return MultiplicationOperator(
            coefficient=self.total_cross_section_field, domain=self.space, codomain=self.space,
        )

    @cached_property
    def isotropic_scattering(self) -> IsotropicScattering:
        r""":math:`\Sigma_{s0}^T` — the isotropic scattering transfer on the pose."""
        return IsotropicScattering(self.scattering.at_order(0), domain=self.space, codomain=self.space)

    @cached_property
    def isotropic_n2n(self) -> IsotropicN2N:
        r""":math:`2\Sigma_2^T` — the (n,2n) transfer on the pose (a loss-side multiplicity-2 channel)."""
        return IsotropicN2N(self.n2n.at_order(0), domain=self.space, codomain=self.space)

    @cached_property
    def isotropic_transfer(self) -> "OperatorSum":
        r""":math:`K_\mathrm{iso} = \Sigma_{s0}^T + 2\Sigma_2^T`."""
        return self.isotropic_scattering + self.isotropic_n2n

    @cached_property
    def loss(self) -> "OperatorSum":
        r"""The loss operator :math:`A = C - K_\mathrm{iso}` for an infinite medium.

        Streaming :math:`L` is identically zero in an infinite medium and
        dropped. Returned UN-materialized (an
        :class:`~orpheus.numerics.operator.OperatorSum`) — the consumer
        chooses the realization: :attr:`multiplication` hands it to
        :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`,
        whose constructor materializes it through the operator's own
        :meth:`~orpheus.numerics.operator.LinearOperator.as_matrix` (the
        shape derives from the threaded domain) and LU-factors it once.
        Every arm poses on :attr:`space`, so the ``OperatorSum`` guard
        VALIDATES the sum instead of skipping it.
        """
        return self.collision - self.isotropic_transfer

    @cached_property
    def production(self) -> IsotropicFission:
        r"""The fission production dyad :math:`F = \chi \otimes \nu\Sigma_f` on the pose."""
        return IsotropicFission(self.fission, domain=self.space, codomain=self.space)

    @cached_property
    def multiplication(self) -> "OperatorProduct":
        r"""The multiplication operator :math:`K = A^{-1} F`, spelled in the algebra.

        ``MatrixInverseOperator(loss) @ production`` — one eager LU
        factorization at construction, the realization the exactly-solvable
        0-D problem earns (the structure-keyed ``loss.inverse()`` would
        return the ITERATIVE splitting; constructing the matrix inverse
        explicitly IS the strategy choice) — composed with the fission dyad.
        """
        return MatrixInverseOperator(self.loss) @ self.production

    # ── the reaction-rate co-vectors ───────────────────────────────────
    @cached_property
    def production_rate(self) -> IntegratedReactionRate:
        r"""The typed integrated co-vector :math:`\langle\nu\Sigma_f, \cdot\rangle` (EE-1)."""
        return IntegratedReactionRate(self.fission_production_field)

    @cached_property
    def absorption_rate(self) -> IntegratedReactionRate:
        r"""The typed integrated co-vector :math:`\langle\Sigma_a, \cdot\rangle` (EE-1)."""
        return IntegratedReactionRate(self.absorption_cross_section_field)


def solve_homogeneous_infinite(mix: Mixture) -> HomogeneousResult:
    r"""Solve the infinite-medium eigenvalue problem for a homogeneous mixture.

    Spells the multiplication operator :math:`\mathbf{K} =
    \mathbf{A}^{-1}\mathbf{F}` in the operator algebra itself —
    ``K = MatrixInverseOperator(loss) @ production`` — from the loss
    operator :math:`\mathbf{A} = C - K_\mathrm{iso} =
    \operatorname{diag}(\Sigma_t) - \Sigma_{s0}^{T} - 2\Sigma_2^{T}`
    (model-shared transport operators on the problem's own hub,
    :class:`HomogeneousProblem`) and the
    fission production dyad :math:`\mathbf{F} = \chi \otimes \nu\Sigma_f`,
    then returns the dominant eigenpair of the materialized
    :math:`\mathbf{K}`: :math:`k_\infty = \lambda_{\max}` and the flux
    spectrum :math:`\varphi` (the corresponding right eigenvector),
    normalised so the fission production rate
    :math:`\nu\Sigma_f \cdot \varphi = 100` n/cm³/s.

    (n,2n) enters ONLY through :math:`\mathbf{A}` (as :math:`2\Sigma_2^T`),
    never the production :math:`\mathbf{F}` — see the module docstring.

    Parameters
    ----------
    mix : Mixture
        Macroscopic cross sections for the homogeneous medium.

    Returns
    -------
    HomogeneousResult
    """
    # The problem is the HUB (R-c1): the mixture-minted pose, the
    # mixture-direct fields and the bound operators live on it, minted
    # once per instance — the solver only reads.  Nothing is fabricated:
    # no carrier, no [0, 1] edges, no node, no chart (O1's honest pose).
    problem = HomogeneousProblem(mix)
    space = problem.space
    ng = problem.ng

    # k∞ and the flux spectrum φ are the EXACT dominant eigenpair of the
    # materialized K = A⁻¹F (the SAME operators the meshed SN solver uses;
    # #276), extracted by the shared Perron–Frobenius primitive
    # (:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`, the one home
    # of the complex-rejection + sign convention).  The 0-D infinite-medium
    # spectrum is exactly solvable, so the dense direct engine is the right
    # tool, not an iterative approximation.
    k_inf, phi = dominant_eigenpair(problem.multiplication.as_matrix())

    # The reaction rates are the typed integrated co-vectors ⟨Σx, ·⟩
    # (IntegratedReactionRate — EE-1, landed CS4b S7): production (νΣf)
    # and absorption (Σa), each the φ†=1 degenerate of the homogenization
    # PG bilinear ⟨φ†, M[Σx]φ⟩. The functional's measure authority is its
    # cross section's space, and the hub's fields are BORN on the pose
    # (CS4a K2 — the pose is the measure authority; G2.5's ×2-weighted-
    # pose mutation holds because the rates follow the pose), so the
    # pre-coda ``replace(..., space=space)`` re-poses are gone — there is
    # nothing to re-pose. Production is νΣf ONLY: the (n,2n) reaction is
    # a loss-side transfer folded into A as 2Σ₂ᵀ, never a production
    # channel.
    production_rate = problem.production_rate
    absorption_rate = problem.absorption_rate

    # Normalise the flux so the fission production rate νΣf·φ = 100 n/cm³/s.
    phi = phi * (100.0 / production_rate.evaluate(phi.reshape(ng, 1)))
    prod_rate = production_rate.evaluate(phi.reshape(ng, 1))
    abs_rate = absorption_rate.evaluate(phi.reshape(ng, 1))
    # The one-group condensed cross sections are INTENSIVE: σ̄x = ⟨Σx,φ⟩/⟨1,φ⟩.
    # The rate leg reads the XS field's space measure, the flux leg the
    # pose's own pairing — the same measure by the content-equality gate —
    # so a point-weight rescale moves BOTH legs together and the ratio is
    # measure-invariant by construction (CS4a-R ruling XD-6 — a quantity
    # documented as a cross section cannot scale with the point weight;
    # the XD-6 gate scales the weight ×2 and pins rate-moves/ratio-stays).
    # ⟨1,φ⟩ is the pose's integration co-vector, not a reaction rate —
    # it stays the space's own pairing. Bit-identical to the pre-review
    # ``float(phi.sum())`` on the counting point (D5 pins it).
    total_flux = space.inner_product(np.ones((ng, 1)), phi.reshape(ng, 1))

    if mix.eg is None:
        # Synthetic XS — no physical energy grid, so lethargy / per-energy
        # diagnostics are not defined.  k_inf and the flux spectrum still
        # carry meaningful information; only the per-energy plotting path
        # is unavailable.
        representative_energy = None
        energy_widths = None
        lethargy_widths = None
    else:
        # The group geometry is the EnergyGrid value object's own — the
        # GEOMETRIC group centre (the natural log-axis abscissa) and the
        # energy / lethargy widths — read off ``mixture.energy_grid``, NOT
        # re-derived here (single source of the group structure).
        eg = mix.energy_grid
        representative_energy = eg.representative_energy
        energy_widths = eg.energy_widths
        lethargy_widths = eg.lethargy_widths

    return HomogeneousResult(
        k_inf=k_inf,
        flux=phi,
        representative_energy=representative_energy,
        energy_widths=energy_widths,
        lethargy_widths=lethargy_widths,
        sig_prod=prod_rate / total_flux,
        sig_abs=abs_rate / total_flux,
        mixture=mix,
    )
