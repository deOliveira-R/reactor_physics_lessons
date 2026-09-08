r"""``AngularLift`` — an ENERGY binding lifted onto the angular composite by
the frame's :math:`\ell = 0` conjugation, acting through the body its ends
select.

Every collision gain of the within-group algebra

.. math::

    A \;=\; L + C - S - N_{2n} - B, \qquad A\,\psi = \tfrac{1}{k}\,F\,\psi

is the same shape: an energy operator :math:`E` on the scalar flux
(:math:`y\,\Sigma_{c,0}^{\mathsf T}` for a transfer channel, the dyad
:math:`|\chi\rangle\langle\nu\Sigma_f|` for fission), lifted onto the
per-ordinate composite by the frame's :math:`\ell = 0` conjugation
:math:`R_0\,E\,M_0 / W` — realised on the reaction-rate fast path
(:math:`\phi = \int\psi\,d\Omega`, then :math:`E\phi`, then the
producer-side :math:`/W` broadcast; no moment tensor on the hot loop) —
plus, for a transfer channel, the :math:`\ell \ge 1` redistribution
:math:`R\,\Lambda_{\ell\ge1}\,M / W`. This class is that lift, ONCE
(CS4c step 5, ruling R-1: ``{S, N₂ₙ} | {F}`` share the ℓ = 0 base and
differ by the datum, the energy binding it derives, and whether an
:math:`\ell \ge 1` part exists — never by a second spelling of the lift).

**Each binding acts through the body its ends select** (the step-5
outcome). The retained analysis face :math:`M \otimes I` has two ends —
the per-ordinate space it reads and the moment space it writes — and the
binding's DOMAIN interior is one of them:

* ``domain.interior == flux_analysis.domain`` — the **angular** end
  (:class:`AngularEnd`): the operand is the per-ordinate flux
  :class:`~orpheus.transport.fields.angular_flux.AngularFlux`;
  :math:`\phi` is its angular integral; a coefficient-space operator acts
  through ``frame.conjugate`` (:math:`R\,\Lambda\,M`); the transpose's
  cotangent is a per-ordinate source/sink;
* ``domain.interior == flux_analysis.codomain`` — the **moment** end
  (:class:`MomentEnd`): the operand is ALREADY :math:`M\psi`
  (:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
  — the 2-D Cartesian windowed iterate, which never materialises the
  per-ordinate flux between sweeps); :math:`\phi` is its :math:`\ell = 0`
  slot; a coefficient-space operator acts through ``frame.reconstruct_after``
  (:math:`R\,\Lambda`, :math:`M` skipped — re-projecting would
  double-project); the cotangent is a moment source/sink.

The selection happens ONCE, at construction; a third interior is refused,
and so is an operand of the right space but the wrong ROLE (space does
not determine role — a flux and a source/sink of one family share a
space). Until step 5 the moment operand was handed to the ANGULAR-bound
operator, which dispatched on the carrier's class per call (`[M]`
2026-09-04: 143 such feeds per windowed solve, on a bit-exact frozen
snapshot) — the shipped non-endomorphism the step-0 census measured. Now
the windowed driver binds the gains on the moment composite
(:meth:`on_moment_domain`) and every operand rides its operator's own
domain (:func:`~orpheus.transport.operators.lift.admit_composite`).

Why the scalar sub-space is the CODOMAIN's angular marginal (F-1 of the
step-5 verification plan): the moment composite's interior is a
tensor-product space with no axes, so it cannot name its own energy ⊗
spatial factor; the codomain is the angular composite for BOTH ends, and
its memoised ``retraction("angular").codomain`` is exactly the space
``AngularFlux.integrate_angular()`` returns — so the energy binding lives
on the space its operand rides, by identity, and there is ONE spelling of
the angular marginal on the operator tier.

Why the energy binding is bound at CONSTRUCTION (a declared field, not a
lazy cache): its plain-scalar admission on that axis-built sub-space is
the effective group-count guard — the composite ends carry no axes, so
the per-end energy admission is declaredly inert on them — and it is
where F-1 is checked. A lazy bind would defer both to the first apply.

The subclass contract is two members and one optional verb:

* :meth:`_bind_energy` — the datum's ENERGY binding on a given scalar
  space (the ℓ = 0 middle factor; bound once as :attr:`isotropic_energy`);
* :meth:`_frame_form` — the WHOLE action as one frame-conjugated product
  (the transpose's spelling: factor reversal, no arithmetic of its own);
* :meth:`_interior_action` — overridden by a subclass with an
  :math:`\ell \ge 1` part (the transfer core does; fission does not —
  :math:`\chi` carries no angle).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar, cast

import numpy as np

from orpheus.numerics.operator import BlockRole
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.spaces.full_field_space import FullFieldSpace
from orpheus.transport.fields._bases import (
    AngularField,
    BulkField,
    FieldRole,
    spatial_moments_per_axis_of,
)
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.fields.scalar_flux import ScalarFlux
from orpheus.transport.full_field import FullField
from orpheus.transport.operators.bound_operator import BoundOperator
from orpheus.transport.operators.lift import (
    CompositeBound,
    admit_composite,
    lift_bulk_action,
)
from orpheus.transport.source_sinks import (
    AngularSourceSink,
    HarmonicMomentSourceSink,
    ScalarSourceSink,
)

if TYPE_CHECKING:
    from orpheus.numerics.operator import LinearOperator, OperatorProduct
    from orpheus.transport.frames.harmonic_frame import (
        HarmonicAnalysisOperator,
        HarmonicFrame,
        HarmonicReconstructionOperator,
    )

__all__ = ["AngularEnd", "AngularLift", "MomentEnd"]

#: The ENERGY binding a lift derives from its datum (the ℓ = 0 middle factor).
EnergyT = TypeVar("EnergyT", bound=BoundOperator)


# ═══════════════════════════════════════════════════════════════════════
# The two ends of the analysis face — what the operand IS
# ═══════════════════════════════════════════════════════════════════════


class AngularEnd:
    r"""The domain's interior is the analysis face's DOMAIN: the operand is
    the per-ordinate flux :math:`\psi`."""

    #: The role leaf the body reads — admitted by class, once, at the verb.
    operand: ClassVar[type[BulkField]] = AngularFlux

    @staticmethod
    def scalar_flux(lift: "AngularLift", bulk: BulkField) -> ScalarFlux:
        # φ = ∫ψ dΩ — the ONE reduction body, on the space's memoised
        # retraction (bit-identical to the pre-step-5 arm).
        return cast(AngularFlux, bulk).integrate_angular()

    @staticmethod
    def conjugate(lift: "AngularLift", operator: "LinearOperator") -> "OperatorProduct":
        # R ∘ A ∘ M — the frame's production composition.
        return lift.frame.conjugate(operator)

    @staticmethod
    def cotangent(lift: "AngularLift", values: np.ndarray) -> BulkField:
        return AngularSourceSink(values=values, space=lift._domain_interior)


class MomentEnd:
    r"""The domain's interior is the analysis face's CODOMAIN: the operand
    is already :math:`M\psi` (the windowed moment iterate)."""

    operand: ClassVar[type[BulkField]] = HarmonicMomentFlux

    @staticmethod
    def scalar_flux(lift: "AngularLift", bulk: BulkField) -> ScalarFlux:
        # The ℓ=0 moment IS the scalar flux (Y_0^0 = 1) — the typed
        # accessor carries the convention; the target is the codomain's
        # scalar sub-space (a widened iterate cannot self-derive it).
        return cast(HarmonicMomentFlux, bulk).scalar_flux(
            space=lift._scalar_interior_space,
        )

    @staticmethod
    def conjugate(lift: "AngularLift", operator: "LinearOperator") -> "OperatorProduct":
        # R ∘ A — M already applied; conjugating would double-project.
        return cast("OperatorProduct", lift.frame.reconstruct_after(operator))

    @staticmethod
    def cotangent(lift: "AngularLift", values: np.ndarray) -> BulkField:
        domain_interior = lift._domain_interior
        return HarmonicMomentSourceSink(
            values=values,
            space=domain_interior,
            L=lift.frame.truncation_order,
            spatial_moments=spatial_moments_per_axis_of(domain_interior),
        )


# ═══════════════════════════════════════════════════════════════════════
# The lift
# ═══════════════════════════════════════════════════════════════════════


@dataclass(eq=False)
class AngularLift(CompositeBound, Generic[EnergyT]):
    r"""An energy binding lifted onto the angular composite — see the module
    docstring.

    Parameters
    ----------
    flux_analysis : HarmonicAnalysisOperator
        The minted FLUX analysis face :math:`M \otimes I`
        (``AngularFlux → HarmonicMomentFlux``) bound on the angular space
        this binding EMITS on (the codomain's interior). Its two ends are
        the two admissible domain interiors; its frame is the binding's
        (:attr:`frame` rides on it — provenance, zero extra state).
    source_reconstruction : HarmonicReconstructionOperator
        The minted SOURCE reconstruction face :math:`R \otimes I`
        (``HarmonicMomentSourceSink → AngularSourceSink``) landing on the
        same angular space, from the SAME frame (the two faces must meet
        in the middle — admitted) — the typed :math:`R` of the moment
        end's :math:`\ell \ge 1` route (the transfer core's).
    domain, codomain : FunctionSpace
        The two mandatory ends (kw-only, write-once —
        :class:`~orpheus.transport.operators.bound_operator.BoundOperator`):
        composite full-field spaces. The codomain's interior is the angular
        space the faces are bound on; the domain's interior is one of the
        analysis face's two ends and SELECTS the body.
    """

    flux_analysis: "HarmonicAnalysisOperator[AngularFlux, HarmonicMomentFlux]" = field(
        kw_only=True,
    )
    source_reconstruction: "HarmonicReconstructionOperator[HarmonicMomentSourceSink, AngularSourceSink]" = field(
        kw_only=True,
    )

    # A collision gain is volumetric — bulk only, no face-trace action; the
    # lift enters the composite by extension-by-zero on the trace. A
    # class-level default of the base's ``block_role`` instance attribute,
    # deliberately unannotated (a ClassVar annotation would override the
    # base's instance variable; a plain annotation would make it a field).
    block_role = BlockRole.BULK

    #: The selected end — DERIVED from the ends in :meth:`__post_init__`,
    #: never a ctor argument.
    _end: "type[AngularEnd] | type[MomentEnd]" = field(init=False, repr=False)
    #: The :math:`\ell = 0` ENERGY binding of this operator's own datum, on
    #: the emitted interior's scalar sub-space — the middle factor the fast
    #: path lifts, and what the scalar consumers read (the solver's
    #: ``K_iso = S.isotropic_energy + N2N.isotropic_energy`` at the ONE
    #: within-group construction site; the eigen-posing's ray seed). Bound
    #: at construction (module docstring: it IS the group-count admission);
    #: ``dataclasses.replace`` re-derives it exactly as it re-derives the end.
    isotropic_energy: EnergyT = field(init=False, repr=False)

    # ── the subclass contract ────────────────────────────────────────

    @abstractmethod
    def _bind_energy(self, scalar_space: FunctionSpace) -> EnergyT:
        """The datum's ENERGY binding, endomorphic on ``scalar_space``."""

    @abstractmethod
    def _frame_form(self) -> "OperatorProduct":
        r"""The whole action as ONE frame-conjugated product (pre-:math:`/W`)
        — the transpose's spelling."""

    # ── construction: admission, then the SELECTION ──────────────────

    def __post_init__(self) -> None:
        owner = type(self).__name__
        # Face admission against the angular space this binding EMITS on:
        # both faces are minted on the codomain's interior, and they must
        # MEET in the middle (one frame — the defect #426 step 2 repaired
        # was two mints of one recipe at two orders; the admission can see
        # exactly that).
        emitted = self._codomain_interior
        if self.flux_analysis.domain != emitted:
            raise TypeError(
                f"{owner}: the flux-analysis face is bound to a different "
                f"angular space than this binding's interior — mint the "
                f"faces from the SAME posed space (tier 2 does)."
            )
        if self.source_reconstruction.codomain != emitted:
            raise TypeError(
                f"{owner}: the source-reconstruction face lands on a "
                f"different angular space than this binding's interior — "
                f"mint the faces from the SAME posed space (tier 2 does)."
            )
        if self.source_reconstruction.domain != self.flux_analysis.codomain:
            raise TypeError(
                f"{owner}: the two faces do not meet — R's moment domain "
                f"{self.source_reconstruction.domain!r} is not M's codomain "
                f"{self.flux_analysis.codomain!r}; mint both from ONE frame."
            )
        # The selection: which END of the analysis face the domain's
        # interior is decides the body, once.
        consumed = self._domain_interior
        if consumed == self.flux_analysis.domain:
            self._end = AngularEnd
        elif consumed == self.flux_analysis.codomain:
            self._end = MomentEnd
        else:
            raise TypeError(
                f"{owner}: the domain's interior {consumed!r} is neither "
                f"end of the analysis face (angular "
                f"{self.flux_analysis.domain!r} / moment "
                f"{self.flux_analysis.codomain!r}) — a binding acts through "
                f"the body its ends select, and no body reads this space."
            )
        # The energy binding, bound NOW: its plain-scalar admission on the
        # axis-built scalar sub-space is the effective group-count guard
        # (the composite ends carry no axes), and F-1 — the emitted
        # interior names a scalar sub-space — is checked here, not at the
        # first apply.
        self.isotropic_energy = self._bind_energy(self._scalar_interior_space)

    # ── derived structure (single sources) ───────────────────────────

    @property
    def frame(self) -> "HarmonicFrame":
        r"""The HUB-interned frame the faces were minted from, riding on
        :attr:`flux_analysis` (provenance, zero extra state; the admission
        proves :attr:`source_reconstruction` meets it) — the conjugation
        properties read it to compose the frame forms."""
        return self.flux_analysis.frame

    @property
    def total_weight(self) -> float:
        r""":math:`W = \int_{S^2} d\Omega` — the binding measure's total
        angular weight (the producer-side :math:`/W`), read off the
        frame's MEASURE (operative data)."""
        return float(np.asarray(self.frame.measure.weights).sum())

    @property
    def _moment_space(self) -> FunctionSpace:
        r"""The bound frame's Parseval-dressed coefficient space — the
        endomorphic ends of the internally-minted moment factors.

        READ off the frame (``frame.basis_space``), never minted from an
        order: which family spans the moments is the quadrature's
        decision, and which METRIC it carries is the frame's (CS4c step 6
        item 6.2c-ii, ruling R-6.2c-1, 2026-09-08). Until then the ends
        bound the basis's CONTINUUM space (#429 Landing A) for a recorded
        reason that did not survive re-measurement: `[M]` the factor's
        Hilbert adjoint is its transpose on every PHYSICAL moment
        :math:`\varphi = M\psi` under BOTH metrics (33 of 33 rows), and the
        dressed end moves it only on arbitrary head draws off the range of
        analysis (5 of 33, three of them on DIAGONAL-Gram rules — not the
        "10 of 33 dense rows" the ruling recorded). With the ends on the
        frame's space the moment iterate, the operator ends and the
        frame's faces share ONE metric, and structural space identity
        admits them without a metric-blind seam."""
        return self.frame.basis_space

    @property
    def _scalar_interior_space(self) -> FunctionSpace:
        r"""The emitted interior's scalar ``(ng, *spatial)`` sub-space —
        the energy binding's ends and the moment end's scalar-flux target.

        The CODOMAIN's memoised angular marginal (F-1; the module
        docstring): the same object
        :meth:`~orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular`
        rides, so the ℓ = 0 emission's space and the energy binding's
        domain agree by IDENTITY, and the angular marginal has one
        spelling on this tier (the elegance review's S6 — a positional
        ``of_axes(*axes[1:])`` was a second one, 39× dearer per read)."""
        return self._codomain_interior.retraction("angular").codomain

    @property
    def is_adjointable(self) -> bool:
        # The Euclidean transpose is the frame form's factor reversal;
        # is_invertible inherits base False — an emission is not
        # invertible.
        return True

    # ── re-binding on the other end ──────────────────────────────────

    def on_moment_domain(self) -> Self:
        r"""This binding re-bound to CONSUME the moment representation —
        the same datum, the same faces, the same codomain; the domain's
        interior becomes the analysis face's codomain (:math:`M\psi`).

        The windowed SI driver's gain: the 2-D Cartesian iterate is held
        as harmonic moments, so the gains that read it are bound here
        (``S.on_moment_domain()``, ``N2N.on_moment_domain()``). A moment
        binding is not an endomorphism — its domain is the moment
        composite, its codomain the angular composite — and the
        ``OperatorSum`` guard never sees it: the windowed loop consumes
        the gains one by one. Built through :func:`dataclasses.replace`,
        so every admission re-runs and the selection lands on the moment
        end.
        """
        domain = self.domain
        if not isinstance(domain, FullFieldSpace) or domain.trace_space is None:
            raise TypeError(
                f"{type(self).__name__}.on_moment_domain: the bound domain "
                f"carries no trace block to pair with the moment interior."
            )
        return replace(
            self,
            domain=FullFieldSpace.from_blocks(
                self.flux_analysis.codomain, domain.trace_space,
            ),
        )

    # ── the action ───────────────────────────────────────────────────

    def apply(self, x: FullField, /) -> FullField:
        r"""The lifted emission :math:`T\psi` on the composite — the
        interior body selected at construction, the zero source/sink on
        the trace (a collision gain is volumetric). The operand is the
        end's FLUX leaf on the bound domain (both blocks); anything else is
        a typed refusal naming this operator."""
        psi = admit_composite(self, x, end="domain", carrier=self._end.operand)
        return lift_bulk_action(
            psi, self._interior_action, trace_role=FieldRole.SOURCE_SINK,
        )

    def _interior_action(self, bulk: BulkField) -> AngularSourceSink:
        r"""The bulk emission — the :math:`\ell = 0` lift alone here; a
        subclass with an :math:`\ell \ge 1` part overrides this to add it
        through :meth:`_combine`."""
        return self._combine(self._isotropic_source(bulk), None)

    def _isotropic_source(self, bulk: BulkField) -> ScalarSourceSink:
        r"""The :math:`\ell = 0` emission in iso scalar magnitude —
        :math:`E\,\phi` through the energy binding, riding the scalar
        flux's own space (the reaction-rate fast path: no moment tensor)."""
        phi = self._end.scalar_flux(self, bulk)
        return ScalarSourceSink(
            values=np.asarray(self.isotropic_energy.apply(phi.values)),
            space=phi.space,
        )

    def _combine(
        self, iso: ScalarSourceSink, aniso: AngularSourceSink | None,
    ) -> AngularSourceSink:
        r"""The producer-side combine :math:`(\text{iso}/W) + \text{aniso}`
        — the :math:`1/W` convention's ONE home (its normalisation chain:
        ``docs/theory/methods/sn/slab_multigroup.rst``). The zero
        accumulator of a purely isotropic lift rides the emitted
        interior; the containment dunder's cross-class arm returns the
        LARGER (angular) class (the #288 principled LSP exception)."""
        aniso_part = (
            aniso if aniso is not None
            else AngularSourceSink.zeros(self._codomain_interior)
        )
        return cast(
            AngularSourceSink, (iso / self.total_weight) + aniso_part,
        )

    def apply_transpose(self, x: FullField, /) -> FullField:
        r"""The Euclidean transpose :math:`T^{\mathsf T}\chi` on the
        composite — the frame form reversed by the
        :class:`~orpheus.numerics.operator.OperatorProduct` chain, then
        the producer :math:`/W`; the cotangent lands on the DOMAIN's
        interior in the end's own source/sink class; the trace is the
        implicit zero (a volumetric gain's transpose is volumetric). The
        operand is any ANGULAR-family cotangent on the codomain (the
        daggered flux or a source/sink — the transpose reads values only).

        This is the Euclidean transpose (L12), not the metric Hilbert
        adjoint ``.H`` (which composes the spaces' Riesz legs around it).
        """
        chi = admit_composite(self, x, end="codomain", carrier=AngularField)
        return lift_bulk_action(
            chi, self._interior_transpose, trace_role=FieldRole.SOURCE_SINK,
        )

    def _interior_transpose(self, bulk: BulkField) -> BulkField:
        values = (
            np.asarray(self._frame_form().apply_transpose(np.asarray(bulk.values)))
            / self.total_weight
        )
        return self._end.cotangent(self, values)
