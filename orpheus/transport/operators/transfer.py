r"""The transfer family's shared cores — the moment factor :math:`\Lambda` and the angular binding.

A **transfer channel** is a Legendre stack with a yield —
:class:`~orpheus.transport.kernels.TransferKernel`
:math:`(\{\Sigma_{c,\ell}\}, y_c)` — and the two collision-gain terms of
the within-group algebra

.. math::

    A \;=\; L + C - S - N_{2n} - B

are two instances of ONE object: the scattering gain :math:`S`
(:math:`y = 1`) and the first-class :math:`(n,2n)` gain :math:`N_{2n}`
(:math:`y = 2`), each the angular binding of its channel's field at the
solve's Legendre order. This module owns the bindings' shared arithmetic
(#426 step 2, ruled 2026-09-03 — the F2/F3 rulings in
``.claude/plans/n2n_anisotropy_kept.md`` §2b):

* :class:`LegendreMomentTransfer` — the per-ℓ block-diagonal moment
  factor :math:`\Lambda = y \sum_\ell \mathbf{P}_\ell \otimes
  \Sigma_{c,\ell}` on the frame's coefficient space;
* :class:`TransferOperator` — the angular binding :math:`T = R\,\Lambda\,M
  / W` on the posed composite: the shared :math:`\ell = 0` lift
  (:class:`~orpheus.transport.operators.angular_lift.AngularLift` — the
  scalar energy binding
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicTransfer`
  on the reaction-rate fast path, the producer-side combine, the
  transpose as factor reversal, and the SELECTION of the body from the
  binding's ends) plus this module's :math:`\ell \ge 1` redistribution.

**The kernel tier names the mathematical object; the operator tier names
the TERM.** The two terms of the algebra are thin role subclasses —
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` and
:class:`~orpheus.transport.operators.n2n.N2NOperator` — whose only
content is two class constants (which ``Mixture`` channel the tier-2
mint reads, and which P0 energy binding the lift derives) and the role
name; every verb lives HERE or on the lift base, once. An AST
gate (``tests/transport/test_transfer_roles.py``) asserts the roles define
nothing else, so the twin path the carve removed cannot regrow one
override at a time.

Until 2026-09-04 the :math:`(n,2n)` binding re-spelled this class's arms
with ``aniso = None`` and a frame minted at :math:`L = 0`: the operator
tier imposed a :math:`P_0` model on a channel whose tape stores the same
seven Legendre moments as elastic — `[M]` −414 Δk·10⁵ on a Be-reflected
fast slab (issue #426; ``docs/theory/methods/sn/adjoint.rst``
§sn-n2n-p0-truncation). The measured size is what retired the twin.

The theory lives in the book — one concept, one home:

* P0 in-scatter, the (n,2n) source, the Pℓ reconstruction, and the
  :math:`1/W` normalisation chain —
  ``docs/theory/methods/sn/slab_multigroup.rst §mg-inscatter-source``,
  §pn-scatter, §n2n-source, §pn-scatter-rlm.
* The no-prefactor :math:`Y_\ell^m` convention and the Funk–Hecke
  eigenbasis (why :math:`T = R\circ\Lambda\circ M`) —
  ``docs/theory/foundations/spherical_harmonics.rst
  §spherical-harmonics-eigenbasis``.
* The §5.6 integral-kernel reading and the apply-only capability
  surface — ``docs/theory/foundations/operator_algebra.rst
  §integral-kernel-category``, §capability-set-semantics.
* The Euclidean adjoint :math:`T^{T}` (forward fast-path vs adjoint
  frame-form) — ``docs/theory/methods/sn/adjoint.rst
  §sn-scattering-adjoint-source``, §sn-n2n-adjoint-source.

Capability surface
==================

``apply`` + ``apply_transpose`` (``is_adjointable=True``); **no**
``solve`` (``is_invertible=False``). A transfer gain is rank
:math:`O(N_{\text{cells}}\cdot N_{\text{groups}})` with no tractable
inverse — it is *applied*, never *inverted*, and the missing ``solve``
is structural method-absence (a composer refuses to build :math:`T^{-1}`
at construction time), not an advertising flag. The adjoint :math:`T^{T}`
rides the harmonic-frame :attr:`~TransferOperator.full_transfer_kernel`
(closes `#118 <https://github.com/deOliveira-R/ORPHEUS/issues/118>`_):
see :meth:`~TransferOperator.apply_transpose`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Self, cast, overload

import numpy as np

from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.basis.base import TruncatedBasis
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.transport.frames import (
    HarmonicAnalysisOperator,
    HarmonicFrame,
    HarmonicReconstructionOperator,
)
from orpheus.numerics.operator import (
    BlockRole,
    LinearOperator,
    OperatorProduct,
)

# Runtime imports of the field leaves — the moment factor's typed arms
# and the lift's per-end bodies construct them at call time. These
# modules form a leaf in the transport package dependency graph (they do
# not import the operators), so the imports are circular-import-safe.
from orpheus.transport.fields._bases import BulkField
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.source_sinks import (
    AngularSourceSink,
    HarmonicMomentSourceSink,
)
from orpheus.transport.kernels import TransferKernel
from orpheus.transport.material_field import TransferMaterialField
from orpheus.transport.operators.angular_lift import AngularEnd, AngularLift, MomentEnd
from orpheus.transport.operators.bound_operator import BoundOperator
from orpheus.transport.operators.isotropic_transfer import IsotropicTransfer
from orpheus.transport.operators.lift import interior_space_of

if TYPE_CHECKING:
    from orpheus.numerics.frame import FrameBase
    from orpheus.transport.mesh.material_xs_field import MaterialXSField


__all__ = ["LegendreMomentTransfer", "TransferOperator"]


def _moment_head_of(space: "FunctionSpace | None", owner: str) -> MomentHead:
    """Narrow a moment operator's end to the MomentHead surface, loudly."""
    if not isinstance(space, MomentHead):
        raise TypeError(
            f"{owner}: the moment ends must be an angular HEAD space "
            f"(SphericalHarmonicSpace or LegendreSpace); got "
            f"{type(space).__name__}."
        )
    return space


@dataclass(eq=False)
class LegendreMomentTransfer(BoundOperator):
    r"""Per-ℓ block-diagonal transfer :math:`\Lambda` on the moment space.

    The diagonal spectrum of the sum-of-tensor-products form
    :math:`\Lambda = y\sum_{\ell=0}^{L} \mathbf{P}_\ell \otimes
    \Sigma_{c,\ell}` (:math:`\mathbf{P}_\ell` selects the :math:`\ell`-th
    harmonic block, :math:`\Sigma_{c,\ell}` the per-material per-ℓ
    Legendre matmul on the group axis, :math:`y` the channel's yield) —
    see ``docs/theory/foundations/operator_algebra.rst
    §scattering-as-tensor-product-sum``.

    For an input moment field :math:`\phi_\ell^m(\vec r)` (the head's
    axes leading, then ``g``, then the spatial axes), the action is
    per-group-transfer per :math:`\ell` block,

    .. math::

        (\Lambda \phi)_\ell^m(\vec r)\bigg|_{g}
        \;=\; y \sum_{g'} \Sigma_{c,\ell}^{m(\vec r)}(g' \to g)\,
              \phi_\ell^m(\vec r)\bigg|_{g'},

    with :math:`m(\vec r)` the material id at cell :math:`\vec r` (the
    per-material structure IS the bound datum — see below).

    **The CS4c rebind (design record §14), generic in the channel (#426
    step 2):** the operator holds the representation-free datum — a
    :class:`~orpheus.transport.material_field.TransferMaterialField`
    already :meth:`at_order
    <orpheus.transport.material_field.TransferMaterialField.at_order>`
    the binding's order — plus its two mandatory ends (the
    :class:`~orpheus.transport.operators.bound_operator.BoundOperator`
    base; :math:`\Lambda` is endomorphic on the coefficient space of its
    order). ``L`` is DERIVED from the field (the order IS the field's —
    single source); the per-material dispatch and the yield live on the
    field's :meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source`
    verb, whose shape guard refuses a moments tensor at any other order.
    The :math:`(n,2n)` instance is this SAME class over the channel's
    field: until 2026-09-04 a twin (``N2NMomentOperator``) spelled its
    :math:`\ell = 0` block alone.

    ``skip_l0`` (default ``True``) skips the :math:`\ell = 0` block, which
    the project's P0 emission handles on a separate reaction-rate fast
    path. Set ``False`` for the full :math:`R\Lambda M\psi` composition on
    the LinearOperator surface — an ℓ-range selector inside the ONE datum,
    not a path switch.

    Capability set: ``{apply, apply_transpose}``; no efficient ``solve``
    (rank-deficient on the :math:`\ell = 0` block by design).
    :math:`\Lambda^{T}` (:meth:`apply_transpose`) is the per-ℓ group-axis
    transpose — the ONLY group-asymmetric factor of the kernel
    :math:`R\circ\Lambda\circ M`, so
    :math:`(R\circ\Lambda\circ M)^{T} = M^{T}\circ\Lambda^{T}\circ R^{T}`
    (``docs/theory/methods/sn/adjoint.rst §sn-scattering-adjoint-source``).

    Parameters
    ----------
    transfer : TransferMaterialField
        The per-material Legendre stacks (with their yield) over the mesh
        layout, at this binding's order.
    skip_l0 : bool, default ``True``
        Skip the :math:`\ell = 0` block (handled by the P0 fast path). Set
        ``False`` for the full :math:`R \Lambda M \psi` composition.
    domain, codomain : FunctionSpace
        The two mandatory ends (kw-only, write-once — the base) — the
        coefficient space of the field's order, both, for the shipped
        endomorphic binding.
    """

    transfer: "TransferMaterialField"
    skip_l0: bool = True

    @classmethod
    def on_basis(
        cls,
        transfer: "TransferMaterialField",
        basis: TruncatedBasis,
        *,
        skip_l0: bool = True,
    ) -> "LegendreMomentTransfer":
        r"""Tier-2 mint: bring a channel's field to the basis's order and
        bind the endomorphic ends on the BASIS's own coefficient space
        (``basis.space`` supplying both — the endomorphism sugar lives
        HERE, never on the exact ctor). The basis names the FAMILY (#429
        tracker 2.5: an integer cannot say which — full harmonics on a
        sphere rule, Legendre on a 1-D rule), so the ends are that basis's
        continuum-Gram space, never re-minted from ``L``. This is the
        frame-less algebra mint (the test tier's); PRODUCTION binds Λ on
        the frame's Parseval-dressed ``basis_space`` through the exact
        ctor (:meth:`TransferOperator._moment_transfer`), the ONE moment
        space the tree carries since CS4c step 6 item 6.2c-ii."""
        ends = basis.space
        return cls(
            transfer.at_order(basis.L),
            skip_l0=skip_l0,
            domain=ends,
            codomain=ends,
        )

    @classmethod
    def on_frame(
        cls,
        transfer: "TransferMaterialField",
        frame: "FrameBase",
        *,
        skip_l0: bool = True,
    ) -> "LegendreMomentTransfer":
        r"""Tier-2 mint on the FRAME's Parseval-dressed coefficient space —
        the ends production binds (CS4c step 6 item 6.2c-ii, ruling
        R-6.2c-1), so the factor composes with the frame's faces
        (``frame.conjugate(Λ)``) by construction. The order is the frame's
        bound family's; :meth:`on_basis` is the frame-less sibling on the
        basis's own continuum space."""
        basis = frame.basis
        if not isinstance(basis, TruncatedBasis):
            raise TypeError(
                f"LegendreMomentTransfer.on_frame: the frame's trial "
                f"{type(basis).__name__} carries no truncation order."
            )
        ends = frame.basis_space
        return cls(
            transfer.at_order(basis.L),
            skip_l0=skip_l0,
            domain=ends,
            codomain=ends,
        )

    @property
    def _head(self) -> MomentHead:
        r"""The angular HEAD the moment verbs read the layout from — this
        operator's own domain (the bound frame's coefficient space)."""
        return _moment_head_of(self.domain, type(self).__name__)

    @property
    def L(self) -> int:
        r"""The Legendre truncation :math:`L` — DERIVED from the bound
        field (the order is the field's; single source)."""
        return self.transfer.order

    @property
    def is_adjointable(self) -> bool:
        # Λ exposes its group-transpose Σ_{c,ℓ}^T (apply_transpose), so the
        # metric-aware .H is free. is_invertible
        # inherits base False — a per-ℓ source map is not invertible.
        return True

    @overload
    def apply(self, moments: "HarmonicMomentFlux") -> "HarmonicMomentSourceSink": ...

    @overload
    def apply(self, moments: np.ndarray) -> np.ndarray: ...

    def apply(
        self, moments: "np.ndarray | HarmonicMomentFlux",
    ) -> "np.ndarray | HarmonicMomentSourceSink":
        r"""Apply :math:`\Lambda` to a moment field — the **role-changing** edge.

        :math:`\Lambda` maps a flux moment to the emission **source**
        moment it produces (flux → source); the typed arm makes that role
        change explicit in the signature. (Why the role change lives on
        the operator and not on the frame:
        ``docs/theory/foundations/operator_algebra.rst
        §integral-kernel-category``.)

        Parameters
        ----------
        moments : np.ndarray or HarmonicMomentFlux
            Flux moment field with the head's axes leading, then the group
            axis, then the spatial axes. On the rectangular harmonic head
            the :math:`m`-axis is the addition-theorem-shifted index where
            slot ``l + m`` holds the :math:`(\ell, m)` entry; entries
            outside :math:`|m| \le \ell` are conventionally zero.

            Typed
            :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
            → typed
            :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
            (the emitted moment SOURCE) with matching ``L`` / ``mesh`` /
            spatial-moment width.  Bare ndarray → ndarray (the endomorphic
            moment-space view the :math:`R\circ\Lambda\circ M`
            kernel ``OperatorProduct`` composes on).

        Returns
        -------
        np.ndarray or HarmonicMomentSourceSink
            The emitted moment source, same shape as ``moments``.  The
            :math:`\ell = 0` block is zero when ``skip_l0`` is ``True``;
            otherwise the P0 contribution is included.  Typed in →
            typed source out; ndarray in → ndarray out.

        Notes
        -----
        Both arms route through the field's single per-material verb
        :meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source`;
        they differ only in the carrier wrap.
        """
        if isinstance(moments, HarmonicMomentFlux):
            out_values = self.transfer.moment_source(
                moments.values, skip_l0=self.skip_l0, head=self._head,
            )
            # flux moment → source moment: the explicit role change
            # (CS4b S4 — same space, new class; role is class identity).
            return HarmonicMomentSourceSink(
                values=out_values, space=moments.space, L=moments.L,
                spatial_moments=moments.spatial_moments,
            )
        return self.transfer.moment_source(
            moments, skip_l0=self.skip_l0, head=self._head,
        )

    def apply_transpose(
        self, moments: "np.ndarray | HarmonicMomentSourceSink",
    ) -> "np.ndarray | HarmonicMomentFlux":
        r"""Apply :math:`\Lambda^{T}` — the per-ℓ group-transpose (the role-REVERSING edge).

        The bare Euclidean transpose of :meth:`apply`: :math:`\Lambda^{T}` maps a
        source moment back into the flux-moment space it came from (source →
        flux), transposing the per-ℓ group-transfer
        :math:`\Sigma_{c,\ell}(g'\to g) \mapsto (g\to g')`.  Routes through the
        field's transpose verb
        :meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source_transpose`.

        This is the Euclidean transpose, **not** the metric Hilbert adjoint
        :math:`\Lambda^{\dagger} = G^{-1}\Lambda^{T}G` (the
        :attr:`~orpheus.numerics.operator.LinearOperator.H` wrapper's job). As the
        only group-asymmetric factor of the kernel :math:`R\circ\Lambda\circ M`,
        it is what lets the whole kernel transpose fall out for free — see
        ``docs/theory/methods/sn/adjoint.rst §sn-scattering-adjoint-source``.

        Typed
        :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
        → :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
        (the explicit role reversal); bare ndarray → ndarray (the endomorphic
        moment-space view the ``OperatorProduct.apply_transpose`` chain composes on).
        """
        if isinstance(moments, HarmonicMomentSourceSink):
            out_values = self.transfer.moment_source_transpose(
                moments.values, skip_l0=self.skip_l0, head=self._head,
            )
            return HarmonicMomentFlux(
                values=out_values, space=moments.space, L=moments.L,
                spatial_moments=moments.spatial_moments,
            )
        return self.transfer.moment_source_transpose(
            moments, skip_l0=self.skip_l0, head=self._head,
        )


@dataclass(eq=False)
class TransferOperator(AngularLift[IsotropicTransfer]):
    r"""The angular binding of a transfer channel: :math:`T = R\,\Lambda\,M/W` (P0 + Pℓ).

    **The CS4c binding (design record §14, step 5), generic in the channel
    (#426 step 2):** the exact ctor retains the representation-free datum,
    the minted faces, and the two ends — nothing richer —

    * :attr:`transfer` — the
      :class:`~orpheus.transport.material_field.TransferMaterialField`
      (per-material Legendre stacks with their yield, over the mesh
      layout), already at this binding's order (the order IS
      :attr:`legendre_order` — single source);
    * :attr:`flux_analysis` / :attr:`source_reconstruction` — the two
      typed faces minted from the HUB-interned
      :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`
      on the angular space this binding EMITS on (tier 2 mints them and
      forgets the frame; the :attr:`frame` accessor is PROVENANCE, riding
      on the faces);
    * the two mandatory ends (kw-only, write-once —
      :class:`~orpheus.transport.operators.bound_operator.BoundOperator`):
      the composite full-field spaces. The angular binding tier 2 mints is
      an endomorphism on the posed composite — the SAME instance
      ``L``/``C``/``B`` carry, so the within-group ``(L + C) − S − N₂ₙ − B``
      OperatorSum guard validates each gain arm natively. The windowed
      driver's gain is the sibling :meth:`on_moment_domain` returns — the
      same datum and faces, the domain's interior the moment composite —
      and the body it acts through is SELECTED by that end at construction
      (:class:`~orpheus.transport.operators.angular_lift.AngularLift`).

    **Two terms, one binding.** The scattering gain and the
    :math:`(n,2n)` gain are the role subclasses
    :class:`~orpheus.transport.operators.scattering.ScatteringOperator`
    and :class:`~orpheus.transport.operators.n2n.N2NOperator`, built on
    the SAME posed space at the SAME solve order through the same
    interned frame (their ``from_solver_data`` mints); the within-group
    algebra spells ``− S − N₂ₙ`` explicitly (§14.1) and bundles them as
    the solver sees fit. Every verb below is channel-agnostic: the yield
    enters the P0 fast path and the moment factor from the field's own
    datum, and the production accounting a :math:`y > 1` channel adds is
    arithmetic that vanishes for :math:`y = 1`.

    **The action, per end (selected once at construction).** The
    :math:`\ell = 0` half is the base's lift of :attr:`isotropic_energy`
    (this role's :attr:`isotropic_binding` of the datum's P0 head). The
    :math:`\ell \ge 1` half — absent when the binding
    :attr:`is_isotropic`, so an all-zero :math:`\Lambda_{\ell\ge1}` is
    never run — is the cached §5.6 :attr:`kernel` :math:`R\,\Lambda\,M`
    on the angular end, and the explicit typed grid path
    (:math:`\Lambda` then the minted source-reconstruction face) on the
    moment end; both end at the base's producer-side combine. The two
    routes share :math:`\Lambda` and the frame's :math:`R` and agree
    bit-for-bit (``tests/sn/operators/test_scattering_kernel_crosscheck.py``;
    the choice is legibility at the call site —
    ``docs/theory/foundations/operator_algebra.rst §integral-kernel-category``).

    Capability surface: ``{apply, apply_transpose}`` — no efficient
    ``solve``; the adjoint :math:`T^{T}` is free via the harmonic-frame
    :attr:`full_transfer_kernel` (see the base's ``apply_transpose``).
    """

    transfer: "TransferMaterialField"

    #: The P0 ENERGY binding this term lifts — the role subclass names its
    #: own (``IsotropicScattering`` for :math:`S`, ``IsotropicN2N`` for
    #: :math:`N_{2n}`); the core mints the shared
    #: :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicTransfer`.
    #: A ClassVar, not a field: it is the role's identity, not a datum.
    isotropic_binding: ClassVar[type[IsotropicTransfer]] = IsotropicTransfer
    #: The channel this term reads off the facade — the role's other fact
    #: (:meth:`TransferMaterialField.scattering
    #: <orpheus.transport.material_field.TransferMaterialField.scattering>`
    #: for :math:`S`, :meth:`~orpheus.transport.material_field.TransferMaterialField.n2n`
    #: for :math:`N_{2n}`). No default on the core: a role that forgets it
    #: fails at its first mint instead of silently reading scattering. A
    #: role is therefore two class constants and no code — the shape F3
    #: ruled, and what makes the AST role gate airtight.
    channel: ClassVar[Callable[["MaterialXSField"], "TransferMaterialField"]]

    #: The ℓ ≥ 1 body, selected ONCE in :meth:`__post_init__` — ``None`` when
    #: the datum carries no moment above ℓ = 0, else the route the selected
    #: END reads (keyed on the end CLASS: a new end is a KeyError at
    #: construction, never a silent fallback).
    _redistribution: "Callable[[BulkField], AngularSourceSink] | None" = field(
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._redistribution = (
            None if self.is_isotropic
            else {
                AngularEnd: self._redistribute_ordinates,
                MomentEnd: self._redistribute_moments,
            }[self._end]
        )

    # ── the lift's subclass contract ─────────────────────────────────

    def _bind_energy(self, scalar_space: FunctionSpace) -> IsotropicTransfer:
        # The role's own P0 binding of the datum's head — the field at
        # order 0 (the datum it reads and nothing richer; y travels in the
        # kernels).
        return type(self).isotropic_binding(
            self.transfer.at_order(0), domain=scalar_space, codomain=scalar_space,
        )

    def _frame_form(self) -> OperatorProduct:
        return self.full_transfer_kernel

    @classmethod
    def from_solver_data(
        cls,
        *,
        mat_xs: "MaterialXSField",
        scattering_order: int,
        space: "FunctionSpace",
    ) -> Self:
        r"""Tier-2 extract-and-mint (CS4c §14): the role's :attr:`channel`
        of the facade, bound on the posed composite at the solve's order
        through :meth:`from_field`. ONE body for both terms — the defect
        #426 step 2 repaired was two mint bodies of one recipe, one of
        which minted at :math:`L = 0` while the other minted at the solve's
        order; a role now carries no mint body to drift."""
        return cls.from_field(
            cls.channel(mat_xs), scattering_order=scattering_order, space=space,
        )

    @classmethod
    def from_field(
        cls,
        transfer: "TransferMaterialField",
        *,
        scattering_order: int,
        space: "FunctionSpace",
    ) -> Self:
        r"""Tier-2 mint (CS4c §14): bring a channel's field to the solve's
        order, mint the two faces from the HUB-interned frame, and bind
        the endomorphic composite ends from one ``space=``.
        :meth:`from_solver_data` extracts the role's channel and calls this.

        ``space`` is the composite
        :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.full_field_space`
        the solver threads — MANDATORY since the flip (the ends are
        write-once fields; the OperatorSum guard validates every build).
        The quadrature is reached through the space's angular axis (the
        CS5 generator channel inside
        :meth:`HarmonicFrame.for_space
        <orpheus.transport.frames.harmonic_frame.HarmonicFrame.for_space>`)
        — no ``quadrature=`` parameter survives, so a frame/space metric
        mismatch is unspellable. Both terms of the algebra mint at the
        SAME ``(rule, L)`` and therefore share ONE interned frame.
        """
        interior = interior_space_of(space, owner=f"{cls.__name__}.from_field")
        frame = HarmonicFrame.for_space(interior, scattering_order)
        return cls(
            transfer.at_order(scattering_order),
            flux_analysis=frame.flux_analysis_on(interior),
            source_reconstruction=frame.source_reconstruction_on(interior),
            domain=space,
            codomain=space,
        )

    @property
    def legendre_order(self) -> int:
        r"""The binding's Legendre order :math:`L` — DERIVED from the
        bound field (the order IS the field's; single source). ``0``
        means P0 only. It is the SOLVE's ``scattering_order`` for BOTH
        terms: the clamp reads the scattering stack alone (ruling O-1),
        and the :math:`(n,2n)` stack is brought to that order — so the
        (n,2n) binding's Legendre order is the elastic channel's clamp,
        which is why this property is not named after either channel."""
        return self.transfer.order

    @property
    def is_isotropic(self) -> bool:
        r"""``True`` iff this binding's :math:`\Lambda_{\ell\ge 1}` is the
        zero operator — order 0, or every moment above :math:`\ell = 0`
        exactly zero (an absent section, an ``NL = 1`` evaluation, a stack
        padded to the solve's order). The anisotropic body is then not
        selected at construction: the same statement ``legendre_order ==
        0`` used to make from the SHAPE, now made from the VALUES — the
        result is bit-identical (an all-zero :math:`\Lambda` reconstructs
        to exact zeros) and the :math:`R\Lambda M` product is not run."""
        return self.transfer.is_isotropic

    def _moment_transfer(self, *, skip_l0: bool) -> LegendreMomentTransfer:
        r"""Mint the moment-space :math:`\Lambda` factor on this binding's
        datum + moment ends — the ONE internal spelling (three consumers:
        the §5.6 kernel, the full conjugation, and the moment end's typed
        route)."""
        ends = self._moment_space
        return LegendreMomentTransfer(
            self.transfer, skip_l0=skip_l0, domain=ends, codomain=ends,
        )

    @cached_property
    def kernel(self) -> LinearOperator:
        r"""The §5.6 integral kernel — the :math:`\ell\ge 1` redistribution
        on THIS binding's domain.

        On the angular end :math:`R \circ \Lambda_{\ell\ge 1} \circ M` — the
        frame conjugation ``frame.conjugate(Λ)`` ``= OperatorProduct(R,
        OperatorProduct(Λ, M))``, so ``kernel.apply(ψ.values) = R(Λ(M ψ))``;
        on the moment end :math:`R \circ \Lambda_{\ell\ge 1}` (the operand
        is already :math:`M\psi`). The factors:

        * :math:`M` = ``frame.analysis`` (the :attr:`frame`'s
          :attr:`~orpheus.numerics.frame.FrameBase.analysis` face);
        * :math:`\Lambda_{\ell\ge 1}` = :class:`LegendreMomentTransfer`
          over this binding's field (``skip_l0=True``);
        * :math:`R` = ``frame.reconstruction`` (the :attr:`frame`'s
          :attr:`~orpheus.numerics.frame.FrameBase.reconstruction` face).

        This is the production :math:`\ell\ge 1` map on the angular end
        (:meth:`_redistribute_ordinates` is ``(1/W)·kernel``) and the moment
        end's 0-ULP crosscheck oracle (the typed route is production
        there). The producer-side :math:`1/W` lives OUTSIDE the kernel
        (at the ``apply`` boundary), so ``kernel.apply`` returns the
        source **pre**-:math:`1/W`. With it, :class:`TransferOperator`
        satisfies the
        :class:`~orpheus.transport.operators.integral_kernel_operator.IntegralKernelOperator`
        Protocol — the theory (why scattering IS a nonlocal integral
        kernel, why P0 is the local component) is in
        ``docs/theory/foundations/operator_algebra.rst §integral-kernel-category``.
        CACHED at first access (the kernel field is immutable).

        Raises
        ------
        ValueError
            If the binding :attr:`is_isotropic` — order 0, or every moment
            above :math:`\ell = 0` exactly zero — so :math:`R\Lambda M` would
            be the zero operator; the P0 emission is the LOCAL component
            handled by the lift's :math:`\ell = 0` half.
        """
        if self.is_isotropic:
            raise ValueError(
                f"{type(self).__name__}.kernel requires an anisotropic binding: "
                f"this one is order {self.legendre_order} with every moment "
                f"above l=0 exactly zero (order 0, an absent section, an NL=1 "
                f"evaluation, or a padded stack), so R∘Λ∘M is the zero "
                f"operator. The P0 emission is the LOCAL component, handled by "
                f"the lift's l=0 half (isotropic_energy)."
            )
        # Λ carries real spaces (== frame.basis_space), so the OperatorProduct
        # composability guard validates the composition natively — NO cast.
        return self._end.conjugate(self, self._moment_transfer(skip_l0=True))

    @cached_property
    def full_transfer_kernel(self) -> OperatorProduct:
        r"""The FULL emission kernel :math:`R\circ\Lambda_{\ell\ge 0}\circ M`
        (:math:`R\circ\Lambda_{\ell\ge 0}` on the moment end).

        The COMPLETE P0 + anisotropic emission as ONE frame-conjugated
        operator: the isotropic ℓ=0 transfer and the anisotropic ℓ≥1
        redistribution (one :class:`LegendreMomentTransfer`,
        ``skip_l0=False``) conjugated by the frame. The per-ordinate
        source is ``(1/W)·full_transfer_kernel.apply(ψ)``; its transpose
        ``(1/W)·full_transfer_kernel.apply_transpose(ψ*)`` is the adjoint
        :math:`T^{T}` (the lift base's ``apply_transpose`` reads it
        through :meth:`_frame_form`). Riding the same frame conjugation
        for iso and aniso is what lets the whole transpose fall out for
        free — ``docs/theory/methods/sn/adjoint.rst
        §sn-scattering-adjoint-source``. Under :math:`N_{2n}` this is the
        product whose reversal the §sn-n2n-adjoint-source equation states
        (its :math:`\ell = 0` block is the lift's reversal; the
        :math:`\ell \ge 1` blocks are the anisotropy's, since #426 step 2).

        Distinct from :attr:`kernel` (the §5.6 ℓ≥1 ANISOTROPIC
        subcomponent): this is the FULL ℓ≥0 emission. CACHED at first
        access (CS4c §14.7 — the satellite mint drops from once-per-apply
        to once-per-construction; the kernel field is immutable, so the
        cache cannot go stale).
        """
        return self._end.conjugate(self, self._moment_transfer(skip_l0=False))

    # ── the ℓ ≥ 1 half, per end ──────────────────────────────────────

    def _interior_action(self, bulk: BulkField) -> AngularSourceSink:
        redistribution = self._redistribution
        return self._combine(
            self._isotropic_source(bulk),
            None if redistribution is None else redistribution(bulk),
        )

    def _redistribute_ordinates(self, bulk: BulkField) -> AngularSourceSink:
        r"""The angular end's :math:`\ell \ge 1` emission:
        :math:`(1/W)\,R\,\Lambda_{\ell\ge1}\,M\,\psi` through the cached
        :attr:`kernel` — one composition, one reduction tree (the 0-ULP
        canary's spelling).

        Implements the Galerkin reconstruction :eq:`pn-scatter` from the
        angular-flux moments :eq:`flux-moments` as the literal operator
        composition :math:`Q^{\rm aniso}_n = \tfrac{1}{W}\,(R\,\Lambda\,M\,
        \psi)_n`.  The trailing :math:`1/W` is the producer-side
        per-ordinate projection (the source enters the sweep already in
        per-ordinate magnitude, so the sweep does NOT apply ``/W`` again);
        the full derivation — the M/Λ/R faces, the addition-theorem
        :math:`(2\ell+1)` factor, the :math:`1/W` normalisation chain — is
        in ``docs/theory/methods/sn/slab_multigroup.rst §pn-scatter-rlm`` and
        ``docs/theory/foundations/spherical_harmonics.rst``.  Reached
        through :meth:`apply` (which admits the operand) as the body the
        :class:`AngularEnd` selects; until #448 a public verb
        ``build_aniso_source`` wrapped it for the eigenvalue finalize's
        hand-rolled source — retired with that finalize (the operator-tier
        gates probe this route directly)."""
        return AngularSourceSink(
            values=self.kernel.apply(bulk.values) / self.total_weight,
            space=self._codomain_interior,
        )

    def _redistribute_moments(self, bulk: BulkField) -> AngularSourceSink:
        r"""The moment end: the explicit typed grid path — :math:`\Lambda`
        maps flux moments to source moments (the role-changing edge, in
        the signature), the minted source-reconstruction FACE synthesises
        the per-ordinate source on its bound angular codomain, then the
        producer-side :math:`1/W`. Numerically equals the kernel's
        ``reconstruct_after(Λ)`` reference."""
        emitted = self._moment_transfer(skip_l0=True).apply(
            cast(HarmonicMomentFlux, bulk),
        )
        return self.source_reconstruction.apply(emitted) / self.total_weight

    # ── Foldable / residual split ─────────────────────────────────────
    #
    # T = T_foldable + T_residual, additive at rtol=1e-14: T_foldable is the
    # P0 within-group self-transfer (diagonal y·Σ_{c,0}^{g→g}, foldable into
    # the removal cross-section σ_r = σ_t − y·Σ_{c,0}^{g→g}); T_residual
    # carries everything else (cross-group P0, all Pℓ≥1). Generic in the
    # yield (#426 step 2, ruling §4.4): the SI splitting reads S's; folding
    # the (n,2n) within-group block is a later, measured decision. Data API
    # only — no solver/sweep/iteration consumes these methods yet; the
    # intended consumer is a consistent DSA preconditioner (#2). Theory (why
    # each residual piece is unfoldable):
    # docs/theory/methods/sn/loss_representation.rst §loss-rep-removal-sigma.
    #
    # ⚠ LATENT CORRECTNESS TRAP (#215): do NOT wire the σ_r-SWEEP as the
    # within-group A_wg.inverse(). The σ_r-sweep inverts a DIAGONAL-in-angle
    # removal, but T_foldable = Σ_c0·P_iso is the ISOTROPIC-PROJECTION
    # self-transfer — the two coincide ONLY for isotropic flux, so the wiring
    # ships 46–56 % silent flux errors on anisotropic problems. Use consistent
    # DSA (#2) or Krylov (splitting-invariant, already production). Any
    # within-group accelerator MUST be gated on an ANISOTROPIC config — the
    # isotropic box cannot see this error. Full failure table:
    # docs/theory/methods/sn/slab_one_group.rst §si-sigma-r-fold-mismatch.

    def _sibling(
        self, field_: "TransferMaterialField",
    ) -> Self:
        r"""A sibling binding of ``field_`` on the SAME ends and of the SAME
        role — faces re-minted from the HUB at the sibling field's own
        order (the interned frame chain, so an order-0 sibling gets
        order-0 faces). An ANGULAR endomorphism's sibling: on a moment
        binding the re-minted faces' moment end no longer matches the
        domain's interior, and the lift's selection refuses loudly.
        """
        interior = self._codomain_interior
        frame = HarmonicFrame.for_space(interior, field_.order)
        return type(self)(
            field_,
            flux_analysis=frame.flux_analysis_on(interior),
            source_reconstruction=frame.source_reconstruction_on(interior),
            domain=self.domain,
            codomain=self.codomain,
        )

    def foldable_part(self) -> Self:
        r"""Return the P0 within-group self-transfer sibling of :math:`T`.

        Carries only the diagonal of each material's :math:`\Sigma_{c,0}`
        — the within-group self-transfer cross-section
        :math:`\Sigma_{c,0}^{g\to g}` per energy group (the yield travels
        in the sibling's kernels, so the sibling emits
        :math:`y\,\Sigma_{c,0}^{g\to g}`). All other channels (cross-group
        P0, every :math:`P_\ell \ge 1`) live in :meth:`residual_part`.

        Returns
        -------
        Self
            An order-0 sibling (of this operator's role) whose
            per-material kernel is the diagonal P0 head, on the same
            bound ends.
        """
        fold = TransferMaterialField(
            per_material={
                mid: TransferKernel(
                    moments=(np.diag(np.diag(k.p0)),),
                    multiplicity=k.multiplicity,
                )
                for mid, k in self.transfer.per_material.items()
            },
            cells_by_material=self.transfer.cells_by_material,
        )
        return self._sibling(fold)

    def residual_part(self) -> Self:
        r"""Return the non-foldable sibling of :math:`T`.

        Carries everything :meth:`foldable_part` does not: the
        off-diagonal of each material's :math:`\Sigma_{c,0}` (cross-group
        P0) and every :math:`P_\ell \ge 1` block verbatim.

        Notes
        -----
        Algebraic contract:
        ``T.apply(ψ) ≈ T.foldable_part().apply(ψ) +
        T.residual_part().apply(ψ)`` at ``rtol=1e-14``.
        """
        residual = TransferMaterialField(
            per_material={
                mid: TransferKernel(
                    moments=(
                        k.p0 - np.diag(np.diag(k.p0)),
                        *k.moments[1:],
                    ),
                    multiplicity=k.multiplicity,
                )
                for mid, k in self.transfer.per_material.items()
            },
            cells_by_material=self.transfer.cells_by_material,
        )
        return self._sibling(residual)

    def is_foldable_into_sigma_r(self) -> bool:
        r"""``True`` iff this operator is structurally the
        :meth:`foldable_part` of some parent :math:`T` — order 0 with
        every material's P0 diagonal.
        """
        if self.legendre_order != 0:
            return False
        return all(
            np.allclose(k.p0, np.diag(np.diag(k.p0)))
            for k in self.transfer.per_material.values()
        )

    def foldable_sigma(self) -> dict[int, np.ndarray]:
        r"""The per-material foldable cross-section
        :math:`(y\,\Sigma_{c,0}^{g\to g})_g` — ``{mid: (ng,)}``, fresh
        copies, read off the bound kernel field (the removal fold
        :math:`\Sigma_r = \Sigma_t - y\,\Sigma_{c,0}^{g\to g}`; the yield
        is 1 for scattering, so the shipped fold is unchanged)."""
        return {
            mid: k.multiplicity * np.diag(k.p0)
            for mid, k in self.transfer.per_material.items()
        }
