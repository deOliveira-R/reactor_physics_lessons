r"""The boundary-trace function space :math:`\Gamma = \partial\Omega \times S^2`.

A *trace space* is the :class:`FunctionSpace` that lives on the
boundary :math:`\partial\Omega` of the spatial domain, carrying the
angular degrees of freedom on each boundary face. It is the
domain/codomain space of every boundary operator (vacuum, albedo,
reflective, white, prescribed-inflow) and the storage space of the
boundary flux.

One space, two directional *selectors* — not three types
========================================================

Issues #205 / #201 (the View-G field-vocabulary refactor) collapse the
three previously-separate boundary-space notions into **one** concrete
:class:`AngularTraceSpace`:

* the per-face ``InflowTraceSpace`` / ``OutflowTraceSpace`` pair
  (Wave 2, ``transient-giggling-cake``), and
* the ad-hoc ``FunctionSpace("sn_boundary_flat")`` that
  :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux` built
  for its flat storage, and
* the dead ``boundary_trace_space()`` factory.

The unification rests on one observation: **inflow and outflow are
operations on a single space, not two spaces.** Whether an ordinate is
incoming or outgoing at a face is a *predicate* — :math:`\mathrm{sign}
(\Omega \cdot \hat n_f)` — evaluated against the same trace data, not a
property of the space's identity. So :class:`AngularTraceSpace` stores the
*signed* projection :math:`\Omega \cdot \hat n_f` once, per face, and
exposes :meth:`inflow_indices_for_face` / :meth:`outflow_indices_for_face`
as selectors over it. (These remain index *selectors*, not projection
*operators*; the :math:`|\Omega\cdot\hat n|`-weighted boundary inner
product they live in is now installed — see below.)

Whole-boundary storage + per-face access
=========================================

The space is the **whole** boundary: ``shape == (layout.total_size,)``
where ``layout`` is the :class:`~orpheus.numerics.face_layout.FaceLayout`
that packs every face into one flat buffer (the same descriptor
:class:`AngularBoundaryFlux` consumes). Per-face access is via the layout's
slot (``layout.faces[face].slice_view``) plus the per-face row of the
signed-projection table; the old per-face ``(N, ng)`` "space" is now a
*derived view*, not a class.

Inner product is the **partial-current metric**
:math:`G_s = |\Omega\cdot\hat n_f|\odot w_n` (the cosine-weighted angular
quadrature), installed at construction by :func:`_build_trace_metric_weights`.
This is the physically-correct boundary inner product under which the
``BoundaryOperator`` Hilbert adjoints (``B.H``) — reflective and white —
are correct (Wave O / O.2b, #208). The metric is group-independent (a
weight in angle, not energy). It is read ONLY by the adjoint path
(:class:`~orpheus.numerics.operator.AdjointOperator` and
:meth:`FunctionSpace.inner_product`); the forward sweep/matvec never reads
it, so installing it leaves every forward result bit-identical. (Before
O.2b this slot was Euclidean ``None``, matching the legacy
``sn_boundary_flat`` storage space.)

Geometric convention
====================

For each face with outward unit normal :math:`\hat n_f`, the signed
projection :math:`\Omega_n \cdot \hat n_f = \mathrm{sign}_f \cdot
\mu_{\text{axis}(f)}` classifies ordinate :math:`\Omega_n` as:

* **Inflow** iff :math:`\Omega_n \cdot \hat n_f < -\epsilon`
  (direction points INTO the domain),
* **Outflow** iff :math:`\Omega_n \cdot \hat n_f > +\epsilon`
  (direction points OUT of the domain),
* **Tangential** iff :math:`|\Omega_n \cdot \hat n_f| \leq \epsilon`
  (grazes the face; in NEITHER selector).

Principled tolerance (not a magic number)
-----------------------------------------

``TANGENTIAL_EPS = 4 * np.finfo(np.float64).eps`` (:math:`\approx
8.9\times10^{-16}`). It is a safety factor over the IEEE-754
dot-product round-off bound :math:`d \cdot u` (:math:`d \leq 3` spatial
dimensions, :math:`u = \epsilon_{\mathrm{mach}}/2` the unit round-off)
for the unit-vector projection :math:`\Omega\cdot\hat n = \langle \hat
n, \mu\rangle`. Empirically (``eps_probe.py``, Gauss-Legendre
``N=2..64`` + Lebedev orders ``3..53``): nominally-tangential cosines
are **exactly** ``0.0`` (quadrature symmetry — odd-N central node, all
off-axis 1-D components, Lebedev axis nodes), while the smallest
*genuine* cosine is :math:`2.44\times10^{-2}`. The gap
:math:`[0, 0.024]` spans ~14 orders, so this eps sits 4× above the
round-off floor and :math:`2.7\times10^{13}\times` below any genuine
projection — making the inflow/outflow masks **bit-identical** to both
the operator's former ``1e-15`` and the realizer's former ``1e-12``
(the band ``(eps, 1e-12)`` is empty). Since #325's node repoint every
shipped rule's tangential cosines are EXACTLY ``0.0`` (group-action
generation, no trig round-off), so the band ``(0, eps]`` is empty too
and the selectors classify identically to a bare sign test: the eps is
**demoted from classifier to provably-inert defensive guard**.
:func:`test_eps_sits_in_the_round_off_to_genuine_gap` (in
``tests/numerics/test_angular_trace_space.py``) pins the gap across
every shipped family so a future quadrature cannot silently reopen it.

Coord-system coverage
=====================

The face → outward-normal table is keyed on the **layout's** face
names (``SNMesh.boundary_face_layout``), which is the single source of
truth for "which faces exist":

* **Mesh1D slab** — two faces ``xmin`` / ``xmax``; outward normals
  :math:`\mp\hat x`.
* **Mesh1D curvilinear** (sphere / cylinder) — ONE face ``xmax`` (the
  outer radius :math:`r=R`). The geometric pole :math:`r=0` is a
  *regularity/symmetry condition* handled by the angular sweep's
  angular closure (``MorelMontryAngularSweep`` / Carlson seed), **not** a
  boundary face — there is no surface there and no inflow to impose.
  This is why the curvilinear layout has no inner face (it is NOT a
  ``left/right`` pair): a solid sphere has exactly one boundary.
* **Mesh2D Cartesian** — four faces ``xmin`` / ``xmax`` / ``ymin`` /
  ``ymax``; a 3-axis Cartesian mesh (C5.5, #225) all six. 2-D
  cylindrical ``(r, z)`` has no SN sweep and cannot become an SNMesh
  (refused at the axis conversion during construction — since C5.3 the
  trace itself is geometry-blind and never sees a mesh).

References
----------

* Lewis, E.E. & Miller, W.F. (1993). *Computational Methods of Neutron
  Transport*. American Nuclear Society. §3.7 (boundary trace operators
  in the discrete-ordinates setting), §6 (curvilinear angular
  redistribution / starting-direction closure at :math:`r=0`).
* ``.claude/plans/field_role_typing_view_g.md`` — A.2/A.3 AngularTraceSpace
  unification design (View-G, signed-:math:`\Omega\cdot\hat n`,
  principled eps, face-naming reconciliation).
* Issue #208 comment (2026-05-31) — the
  :math:`|\Omega\cdot\hat n|`-weighted partial-current boundary inner
  product for the Wave-O adjoint work (landed Phase 4 / O.2b).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.face_layout import (
    AXIS_NAMES,
    face_normal,
    face_streaming_normal,
)
from orpheus.numerics.space import FunctionSpace

if TYPE_CHECKING:
    from orpheus.numerics.face_layout import FaceLayout
    from orpheus.numerics.operator import TraceRestrictionOperator
    from orpheus.numerics.quadrature import Quadrature


__all__ = [
    "AngularFaceTraceSpace",
    "AngularTraceSpace",
    "TANGENTIAL_EPS",
    "TraceRole",
    "build_omega_dot_n",
]

#: Which directional class a per-face tier of the Γ ladder carries.
#: ``"full"`` is the whole ordinate slot Γ(f); ``"outflow"`` / ``"inflow"``
#: are the half-traces Γ₊(f) / Γ₋(f). The three do NOT partition: tangential
#: ordinates (``|Ω·n| <= TANGENTIAL_EPS``) belong to ``"full"`` alone, which
#: is why ``"not inflow"`` must never be spelled ``"outflow"``.
TraceRole = Literal["full", "outflow", "inflow"]


# Tangential tolerance for the unit-vector projection ``Ω · n``. A
# safety factor (×4) over the IEEE-754 dot-product round-off bound for a
# 3-component unit-vector inner product. Empirically bit-identical to
# the legacy ``1e-15`` (operator) and ``1e-12`` (realizer) tolerances —
# and since #325's node repoint a provably-inert defensive guard (every
# shipped rule's tangential cosines are exactly 0.0, so the band
# ``(0, eps]`` is empty); see the module docstring +
# :func:`test_eps_sits_in_the_round_off_to_genuine_gap`.
# Public: the ONE tangential threshold every trace-classification
# consumer shares (the selectors below; the reflective law's
# inflow→outflow invariant, ERR-045) — a second locally-minted epsilon
# would be a twin source of truth for "what counts as grazing".
TANGENTIAL_EPS: float = 4.0 * np.finfo(np.float64).eps


# The face → outward-normal parse is
# :func:`~orpheus.numerics.face_layout.face_normal`: it returns the
# ``(axis_index, sign)`` pair with ``axis_index`` selecting
# ``(mu_x, mu_y, mu_z)[axis_index]`` and ``sign`` the outward-normal sign (±1),
# so the signed projection is ``Ω · n = sign * mu[axis]``. One parse serves
# every supported mesh: 1-D meshes use ``xmin`` / ``xmax`` (radial axis is the
# x-axis; curvilinear has ``xmax`` only), 2-D Cartesian the four x/y faces,
# 3-D Cartesian all six.
#
# C5.3 (#225) derived the local ``_FACE_NORMALS`` table from
# :data:`~orpheus.numerics.face_layout.AXIS_NAMES`, killing a hand-listed
# 4-face transcription that silently lacked the z faces. Campaign phase
# **B3.4c** finished the job: the ``min``/``max`` ↔ sign half of the
# convention was still transcribed here AND at four other sites, so the table
# moved next to ``AXIS_NAMES`` as a two-way bijection and this module reads it.


def _quadrature_axis(quadrature: "Quadrature", axis: int) -> np.ndarray:
    r"""The ordinates' orbit-mean direction cosine along ``axis``.

    Used to build :math:`\Omega\cdot\hat n_f`, i.e. to decide inflow from
    outflow at a face — a **flux** question, so on an axis a 1-D rule has
    suppressed the answer is genuinely zero (nothing flows along it) rather
    than missing. That is exactly
    :meth:`~orpheus.numerics.quadrature.directional.Quadrature.mean_axis_cosine`,
    and delegating to it is the whole body.

    ⭐ **This function used to be a three-arm conditional over ``mu_x`` /
    ``mu_y`` / ``mu_z``, two arms of them written as
    ``getattr(quadrature, "mu_y", np.zeros_like(mu_x))``** — the
    ``coding-standards`` defaulted-``getattr`` idiom, which fails silently in
    the DEFAULT's direction: had ``mu_y`` ever been retired, every branch keyed
    on this value would have flipped with nothing raising. The ladder existed
    only because the accessor it called could not say whether its zero was an
    answer or an absence. Phase 0.2 gave that distinction a name, and the
    ladder collapsed with it — the conditional was a missing verb.
    """
    return np.asarray(quadrature.mean_axis_cosine(axis))


def build_omega_dot_n(
    quadrature: "Quadrature",
    faces: tuple[str, ...],
) -> NDArray:
    r"""Build the signed projection table :math:`\Omega \cdot \hat n_f`.

    Returns a ``(n_faces, n_ordinates)`` float array whose row ``f`` is
    :math:`\mathrm{sign}_f \cdot \mu_{\text{axis}(f)}` — the outward
    projection of every ordinate onto face ``f``'s normal. Inflow /
    outflow / tangential are derived from its sign on demand.

    C5.3 (#225): geometry-blind — every datum comes from the quadrature
    and the face NAMES (the axis-aligned outward normals are implied by
    the ``"{axis}{min|max}"`` convention). The former ``mesh``
    parameter was gate-only: its curvilinear-``Mesh2D`` refusal is
    unreachable (such a mesh cannot become an ``SNMesh`` — the axis
    conversion at construction refuses it), and the isinstance check
    carried no data.

    Public (#52, ERR-041): this is the SINGLE face-name → signed-
    projection primitive. The SN realizer's vacuum-orientation guard
    cross-checks hand-supplied ``inflow_indices`` against the row this
    function derives from the face name alone — a second, independently
    sourced encoding of the same orientation.
    """
    n_ord = int(quadrature.N)
    omega_dot_n = np.zeros((len(faces), n_ord), dtype=float)
    for f_idx, face in enumerate(faces):
        axis, sign = face_normal(face)
        # C5.5 (#225) fail-loud: a layout naming an axis-k FACE demands
        # GENUINE mu_k on the quadrature. Discriminate on VALUE, not
        # attribute presence — the per-axis cosines are properties that
        # zero-pad past the cubature's intrinsic dimensionality (e.g.
        # 1-D Gauss-Legendre carries mu_z == zeros(N), never an absent
        # attribute), so an attribute test can never fire. A boundary
        # face whose normal-axis cosines are ALL zero has Ω·n ≡ 0 for
        # every ordinate — a rank-mismatch (a z face on a quadrature
        # with no third cosine) that zero-padding would silently
        # misclassify as all-tangential (neither inflow nor outflow).
        mu_axis = _quadrature_axis(quadrature, axis)
        if not np.any(mu_axis):
            raise ValueError(
                f"Face {face!r} requires genuine "
                f"mu_{AXIS_NAMES[axis]} cosines, but every ordinate of "
                f"the quadrature has mu_{AXIS_NAMES[axis]} == 0 — a "
                f"rank-mismatch between the face layout and the "
                f"angular cubature."
            )
        omega_dot_n[f_idx] = sign * mu_axis
    return omega_dot_n


def _build_trace_metric_weights(
    omega_dot_n: NDArray,
    quad_weights: NDArray,
    layout: "FaceLayout[str]",
) -> NDArray:
    r"""Build the partial-current boundary metric :math:`G_s = |\Omega\cdot\hat n_f|\odot w_n`.

    The trace Hilbert metric is the **partial current** weight: pairing two
    boundary fields contracts angle against :math:`|\Omega\cdot\hat n_f|\,w_n`,
    i.e. the cosine-weighted angular quadrature (Lewis & Miller §3.7; the
    boundary inner product under which reflective/white BCs are self-adjoint).
    Wave O / O.2b (#208) — replaces the legacy Euclidean (``None``) metric.

    Returns the flat ``(layout.total_size,)`` diagonal-weight array that
    :meth:`FunctionSpace.inner_product` broadcasts against the trace state.
    The metric is **purely angular** — :math:`|\Omega\cdot\hat n_f|\,w_n`
    depends only on the ordinate (axis 0 of every face slot), not on energy
    group or on spatial position along the face. So for a face slot of shape
    ``(N, ng)`` (1-D) or ``(N, ng, n_face_cells)`` (2-D edge) the ``(N,)``
    cosine weight is broadcast across ALL trailing (group / spatial) axes.

    The row order of ``omega_dot_n`` matches ``tuple(layout.faces)`` (both
    derive from the same ordered layout), so ``enumerate(layout.faces)``
    aligns face slots with projection rows.
    """
    weights_flat = np.zeros((int(layout.total_size),), dtype=float)
    w_n = np.asarray(quad_weights, dtype=float)  # (N,)
    for f_idx, face_name in enumerate(layout.faces):
        slot = layout.faces[face_name]
        face_w = face_streaming_normal(omega_dot_n[f_idx], w_n)  # (N,) = |Ω·n_f| · w_n
        # Ordinate is axis 0 of the slot; reshape to (N, 1, 1, …) so the
        # per-ordinate cosine weight broadcasts across every trailing axis
        # (group, and — in 2-D — the cells along the boundary edge).
        face_w_axis0 = face_w.reshape((face_w.shape[0],) + (1,) * (len(slot.shape) - 1))
        flat_face = np.broadcast_to(face_w_axis0, slot.shape).reshape(-1)
        weights_flat[slot.offset : slot.offset + slot.flat_size] = flat_face
    return weights_flat


# ─────────────────────────────────────────────────────────────────────
# The trace space
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AngularTraceSpace(FunctionSpace):
    r"""The boundary-trace function space (View-G, role-agnostic).

    One concrete space for the whole boundary :math:`\Gamma`. Inflow and
    outflow are *selectors* over the signed projection
    :math:`\Omega\cdot\hat n`, not separate types. See the module
    docstring for the unification rationale and geometric convention.

    Parameters
    ----------
    name, shape, inner_product_weights
        Inherited from :class:`FunctionSpace`. ``name`` is ``"angular_trace"``
        and ``shape`` is the whole-boundary flat shape
        ``(layout.total_size,)``. ``inner_product_weights`` is the
        partial-current metric :math:`G_s = |\Omega\cdot\hat n_f|\odot w_n`
        (built by :func:`_build_trace_metric_weights`; see the module
        docstring) — NOT Euclidean.
    layout : FaceLayout
        The flat-buffer descriptor (which faces exist, per-face shapes,
        offsets). Carried as ``compare=False`` leaf-data so it does not
        pollute the ``(name, shape)`` identity — two trace spaces on
        meshes of the same total boundary size compare equal regardless
        of their face decomposition. Ordered iteration of
        ``layout.faces`` defines the row order of :attr:`omega_dot_n`.
    omega_dot_n : NDArray, shape ``(n_faces, n_ordinates)``
        The signed projection :math:`\Omega\cdot\hat n_f` per face. The
        single source of truth the inflow/outflow selectors AND the
        operator-side directional masks both read.
    directions : NDArray, shape ``(n_ordinates,)`` in 1-D, ``(n_ordinates, 3)`` otherwise
        The ordinate directions :math:`\Omega_n` themselves —
        :attr:`~orpheus.numerics.quadrature.Quadrature.nodes` verbatim.

        ⭐ **Why the FULL direction and not just** :attr:`omega_dot_n`. On an
        axis-aligned face the normal component is recoverable
        (:math:`\mu = \pm\,\Omega\cdot\hat n`), so a source varying only along
        the normal could read the projection — but a source varying
        *tangentially* (a beam at an angle to the face) cannot, and neither can
        one on a face whose normal is not an axis. Storing the projection alone
        would be a lossy reduction that happens to suffice for the ansätze
        shipped today, which is the shape of a decision that has to be undone
        later. `[M]` the array is dimension-generic as it stands: ``(8,)`` on
        ``gauss_legendre(8)``, ``(24, 3)`` on ``level_symmetric(4)``.

        ⚠ **Vocabulary crossing, deliberate.** The quadrature calls these
        ``nodes`` (measure vocabulary — a rule is a measure and its support is
        nodes). On a *trace* they are directions, which is the transport
        vocabulary this class already speaks (``omega_dot_n``). The rename is
        at the boundary between the two registers and happens exactly once,
        here; grep ``nodes`` in ``numerics/quadrature`` and ``directions`` in
        the trace/boundary tier.
    """

    # Required (an AngularTraceSpace cannot even size itself without its layout,
    # and the selectors read omega_dot_n unconditionally — a trace space
    # missing either is an illegal state); kw_only sidesteps the
    # inherited-defaults field-ordering rule. Construct via
    # :meth:`from_quadrature_and_layout`.
    layout: "FaceLayout[str]" = field(kw_only=True, repr=False, compare=False)
    omega_dot_n: NDArray = field(kw_only=True, repr=False, compare=False)
    directions: NDArray = field(kw_only=True, repr=False, compare=False)

    @property
    def partial_current_metric(self) -> NDArray:
        r"""The installed boundary metric :math:`G_s = |\Omega\cdot\hat n|\odot w` (flat).

        A trace space ALWAYS carries the partial-current metric — never
        the Euclidean ``None`` (the Wave-O decision;
        :meth:`from_quadrature_and_layout` installs it unconditionally) —
        so this named accessor narrows the base's ``Optional`` ONCE, with
        the construction guarantee spelled as a loud parse rather than a
        silent assumption at every consumer
        (``AngularBoundaryFlux.net_current`` reads THIS, not the generic
        ``inner_product_weights``).
        """
        weights = self.inner_product_weights
        if weights is None:
            raise TypeError(
                "AngularTraceSpace without its partial-current metric G_s — "
                "construct via from_quadrature_and_layout (the metric is "
                "not optional on a boundary trace)."
            )
        return weights

    @property
    def face_names(self) -> tuple[str, ...]:
        """Ordered face names (matching :attr:`omega_dot_n` row order)."""
        return tuple(self.layout.faces)

    @classmethod
    def from_quadrature_and_layout(
        cls,
        quadrature: "Quadrature",
        layout: "FaceLayout[str]",
    ) -> "AngularTraceSpace":
        r"""Build the trace space from a quadrature and a face layout.

        C5.3 (#225): geometry-blind — the former ``mesh`` parameter
        (``from_mesh_and_quadrature``) was gate-only and is retired;
        every datum comes from the quadrature and the layout's face
        names (axis-aligned outward normals implied by the
        ``"{axis}{min|max}"`` convention).

        Parameters
        ----------
        quadrature : Quadrature
            Angular quadrature exposing ``mu_x`` (always) and ``mu_y`` /
            ``mu_z`` (when applicable).
        layout : FaceLayout
            The boundary face layout (canonically
            :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` —
            the single source of truth for which faces exist and their
            flat packing). Its ordered faces drive the
            :attr:`omega_dot_n` rows; its ``total_size`` sets the space
            shape.

        Raises
        ------
        ValueError
            If ``layout`` names a face absent from the normal table.
        """
        faces = tuple(layout.faces)
        omega_dot_n = build_omega_dot_n(quadrature, faces)
        # Wave O / O.2b (#208): the partial-current boundary metric
        # G_s = |Ω·n_f| ⊙ w_n — the cosine-weighted angular quadrature under
        # which the BoundaryOperator Hilbert adjoints (B.H) are physically
        # correct.  Group-independent; built once at the producer (Pattern 7).
        inner_product_weights = _build_trace_metric_weights(
            omega_dot_n, quadrature.weights, layout,
        )
        # CS4b S3 (F2): the name carries a CONTENT digest, so the inherited
        # ``(name, shape)`` equality IS content equality — the same
        # mechanism as ``FunctionSpace.of_axes``'s derived name. Folded in:
        # the layout's structural identity (keys, offsets, sizes), the
        # signed projection table, and the quadrature's weights and
        # directions (which the trace metric |Ω·n̂|w is built from). Two
        # traces agreeing in all of these ARE the same space; a mesh whose
        # boundary content differs (layout, quadrature, face geometry)
        # mints an UNEQUAL one. Boundary LAWS are deliberately absent —
        # a law changes neither DOFs nor Gram (laws are operator data).
        import hashlib

        payload = b"".join((
            repr([
                (str(k), int(s.offset), int(s.flat_size))
                for k, s in layout.faces.items()
            ]).encode(),
            omega_dot_n.tobytes(),
            np.asarray(quadrature.weights, dtype=float).tobytes(),
            np.asarray(quadrature.nodes, dtype=float).tobytes(),
        ))
        digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
        return cls(
            name=f"angular_trace#{digest}",
            shape=(int(layout.total_size),),
            inner_product_weights=inner_product_weights,
            layout=layout,
            omega_dot_n=omega_dot_n,
            # The ordinate directions, kept verbatim. This factory is the ONLY
            # construction site of an AngularTraceSpace (`[M]` zero direct
            # ``AngularTraceSpace(...)`` calls in orpheus/ or tests/), which is
            # why a required field can be added here without touching a caller.
            directions=np.asarray(quadrature.nodes),
        )

    # ── Directional selectors ────────────────────────────────────────

    def _face_row(self, face: str) -> int:
        """Return the :attr:`omega_dot_n` row index for ``face``."""
        try:
            return self.face_names.index(face)
        except ValueError as exc:
            raise ValueError(
                f"Unknown face {face!r}; available: {self.face_names}"
            ) from exc

    def inflow_indices_for_face(self, face: str) -> np.ndarray:
        r"""Ordinate indices that are inflow at ``face``.

        Inflow iff :math:`\Omega\cdot\hat n_f < -\epsilon` (direction
        points into the domain). Tangential ordinates are excluded.
        """
        row = self.omega_dot_n[self._face_row(face)]
        return np.flatnonzero(row < -TANGENTIAL_EPS)

    def outflow_indices_for_face(self, face: str) -> np.ndarray:
        r"""Ordinate indices that are outflow at ``face``.

        Outflow iff :math:`\Omega\cdot\hat n_f > +\epsilon` (direction
        points out of the domain). Tangential ordinates are excluded.
        """
        row = self.omega_dot_n[self._face_row(face)]
        return np.flatnonzero(row > +TANGENTIAL_EPS)

    # ── The selectors, as OPERATORS: the trace maps γ± ────────────────

    @cached_property
    def _face_restrictions(
        self,
    ) -> "dict[tuple[str, str], TraceRestrictionOperator]":
        r"""``(role, face) -> γ``, built once per space.

        Built eagerly for every face because there are a handful of them and
        :meth:`inflow_restriction` sits on the **per-sweep** path — rebuilding
        one (with its uniqueness and sortedness validation) on every boundary
        application would move real work into the hot loop for no benefit,
        which is the shape of a performance regression that no correctness
        gate would catch.
        """
        from orpheus.numerics.operator import TraceRestrictionOperator

        n_ordinates = int(self.omega_dot_n.shape[1])
        out: dict[tuple[str, str], TraceRestrictionOperator] = {}
        for face in self.face_names:
            for role, indices in (
                ("inflow", self.inflow_indices_for_face(face)),
                ("outflow", self.outflow_indices_for_face(face)),
            ):
                # ``np.flatnonzero`` yields sorted, unique indices, so the
                # operator's guards are satisfied by construction here — they
                # exist for hand-built index sets.
                #
                # G6.3 step 1 (#330): BOUND to the Γ ladder, so γ± is
                # `Γ(f) → Γ±(f)` in the type system and not merely in the
                # prose. Two consequences: `.H` becomes the Hilbert adjoint
                # `G_V⁻¹γᵀG_W` (the restricted partial-current metric now
                # travels with the operator instead of being dropped), and a
                # composition against the wrong face or the wrong half is
                # REFUSED rather than silently accepted — `|Γ₊(f)| == |Γ₋(f)|`
                # on every shipped quadrature, so shape alone could never
                # catch it. The operator's own guard cross-checks these spaces
                # against `n_total` / `len(indices)`.
                out[(role, face)] = TraceRestrictionOperator(
                    indices, n_total=n_ordinates, axis=0,
                    domain=self.face_space(face),
                    codomain=(
                        self.inflow_space(face) if role == "inflow"
                        else self.outflow_space(face)
                    ),
                )
        return out

    def outflow_restriction(self, face: str) -> "TraceRestrictionOperator":
        r""":math:`\gamma_+` at ``face`` — the restriction onto :math:`\Gamma_+`.

        The **domain** of every boundary law: a law consumes the outflow trace
        and produces the inflow trace (:math:`\gamma_-\psi = R\,G\,\gamma_+\psi
        + q`), so this is the operator that hands it its argument.

        Its transpose is the scatter back into the full face slot. Note the
        pair is NOT a partition of that slot: the third class, **tangential**
        ordinates at :math:`|\Omega\cdot\hat n| \le \epsilon`, belongs to
        neither restriction — which is why "not inflow" must never be spelled
        as "outflow".
        """
        return self._face_restrictions[("outflow", self._checked_face(face))]

    def inflow_restriction(self, face: str) -> "TraceRestrictionOperator":
        r""":math:`\gamma_-` at ``face`` — the restriction onto :math:`\Gamma_-`.

        The **codomain** of every boundary law. Its transpose
        (:math:`\iota_-`) is what writes a law's image back into the full face
        slot, leaving the outflow and tangential rows untouched at zero.
        """
        return self._face_restrictions[("inflow", self._checked_face(face))]

    def _checked_face(self, face: str) -> str:
        """Raise the layout's own error for an unknown face name."""
        self._face_row(face)
        return face

    # ── The Γ ladder: the per-face tiers, AS SPACES (G6.1) ───────────

    @cached_property
    def _face_spaces(self) -> "dict[tuple[TraceRole, str], AngularFaceTraceSpace]":
        r"""``(role, face) -> Γ`` — the whole ladder, built once per space.

        Cached for the same reason as :attr:`_face_restrictions`: a
        half-trace space is per-``(face, quadrature)``, so constructing one
        per call would move allocation into the per-sweep path. Built on the
        trace space (which the mesh already caches), never at each
        realization.
        """
        out: dict[tuple[TraceRole, str], AngularFaceTraceSpace] = {}
        metric_flat = np.asarray(self.partial_current_metric)
        for face in self.face_names:
            slot = self.layout.faces[face]
            n_ordinates = int(slot.shape[0])
            # The metric restricted to THIS face, then reduced to its leading
            # (ordinate) axis: `[M]` it is constant across the trailing group /
            # codim-1 spatial axes by construction (`_build_trace_metric_weights`
            # broadcasts one per-ordinate vector across the slot), and
            # the resolved DiagonalMetric (`metric._broadcast_leading`)
            # re-expands a leading vector on application.
            # Storing the 1-D vector rather than the full slot keeps ONE
            # source of truth for the weight and matches the base's
            # broadcast convention.
            face_metric = slot.slice_view(metric_flat).reshape(n_ordinates, -1)[:, 0]
            tiers: tuple[tuple[TraceRole, np.ndarray], ...] = (
                ("full", np.arange(n_ordinates, dtype=np.intp)),
                ("outflow", self.outflow_indices_for_face(face)),
                ("inflow", self.inflow_indices_for_face(face)),
            )
            for role, indices in tiers:
                out[(role, face)] = AngularFaceTraceSpace(
                    name=_face_space_name(face, role),
                    shape=(len(indices), *slot.shape[1:]),
                    inner_product_weights=face_metric[indices],
                    face=face,
                    role=role,
                    ordinate_indices=np.asarray(indices, dtype=np.intp),
                    # The SAME restriction the metric gets one line above, on
                    # the SAME index set — which is the point: a tier's rows,
                    # its weights and its directions are three views of one
                    # selection, so they cannot drift out of correspondence.
                    # Indexing the LEADING axis alone is what makes this
                    # dimension-generic: `(N,) -> (m,)` in 1-D and
                    # `(N, 3) -> (m, 3)` in 3-D, both by the same expression.
                    directions=self.directions[indices],
                )
        return out

    def face_space(self, face: str) -> "AngularFaceTraceSpace":
        r""":math:`\Gamma(f)` — the whole ordinate slot at ``face``.

        The **middle tier** of the ladder, and NOT the direct sum
        :math:`\Gamma_+ \oplus \Gamma_-`: the tangential ordinates
        (:math:`|\Omega\cdot\hat n| \le \epsilon`) belong to neither half, and
        `[M]` there are 0 / 8 / 12 of them on ``gauss_legendre(4)`` /
        ``product(4,4)`` / ``lebedev(17)``. It is the DOMAIN of both trace
        maps :math:`\gamma_\pm`.
        """
        return self._face_spaces[("full", self._checked_face(face))]

    def outflow_space(self, face: str) -> "AngularFaceTraceSpace":
        r""":math:`\Gamma_+(f)` — the outflow half-trace at ``face``.

        The codomain of :math:`\gamma_+` and the **domain** of every boundary
        law (:math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`).
        """
        return self._face_spaces[("outflow", self._checked_face(face))]

    @cached_property
    def _current_spaces(self) -> "dict[str, FunctionSpace]":
        r"""``face -> S(f)`` — the angle-INTEGRATED per-face trace, built once.

        The tier the :math:`\Gamma` ladder was missing: not a subset of
        ordinates but the **collapse of the ordinate axis**. It is where a
        factored response's intermediate state lives — for the Lambertian, the
        cosine-weighted outgoing partial current
        :math:`J^+_g = \sum_{\Gamma_+} w\,|\Omega\cdot\hat n|\,\psi`, which
        today exists only as an anonymous local inside
        the welded ``AngularAverageOperator``'s apply, until G6.3 step 3b factored
        it and :class:`~orpheus.sn.boundary.angular.PartialCurrentOperator`
        gave the quantity its name.
        """
        out: dict[str, FunctionSpace] = {}
        for face in self.face_names:
            shape = tuple(self.layout.faces[face].shape[1:])
            out[face] = FunctionSpace(
                name=f"angular_trace[{face}:current]",
                shape=shape,
                # ⭐ EXPLICITLY unit, never ``None``. The angular measure
                # |Ω·n|·w is CONSUMED by the contraction that lands here, so
                # nothing further remains to weight — and the ladder's other
                # tiers carry no face area either (``AngularTraceSpace``'s own
                # metric is |Ω·n|·w alone), so adding one here would break that
                # convention and double-count against
                # ``ScalarTraceSpace``, which owns the area-weighted boundary
                # integral for angle-integrated quantities.
                #
                # Spelled as ones rather than left ``None`` because ``None``
                # encodes TWO states — "deliberately Euclidean" and "nobody
                # bound a metric" — and no gate can separate one value from
                # itself. The factored-adjoint theorem needs only
                # non-degeneracy here (the interior metric cancels from the
                # composite), so this records the INTENT, which is the part a
                # reader cannot recover.
                inner_product_weights=np.ones(shape, dtype=float),
            )
        return out

    def current_space(self, face: str) -> FunctionSpace:
        r""":math:`S(f)` — the angle-integrated trace at ``face``.

        Shape ``(ng, *face_spatial)``: the face slot with its **ordinate axis
        integrated out**, one value per group per boundary cell.

        This is the intermediate a factored response passes through
        (:math:`\Gamma_+(f) \to S(f) \to \Gamma_-(f)`), and the quantity is
        named rather than anonymous: the cosine-weighted outgoing **partial
        current**. Distinct from
        :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`,
        which carries the :math:`(J^+, J^-)` PAIR for the whole boundary under
        the face-AREA metric — diffusion's P1 Cauchy data, a different object
        with a different metric and a different scope.

        Its metric is deliberately unit; see :attr:`_current_spaces`. The
        factored-adjoint theorem makes any non-degenerate choice give the same
        composite adjoint, so the freedom is real — but the individual factors'
        adjoints DO depend on it, which is why it is chosen rather than defaulted.
        """
        return self._current_spaces[self._checked_face(face)]

    def inflow_space(self, face: str) -> "AngularFaceTraceSpace":
        r""":math:`\Gamma_-(f)` — the inflow half-trace at ``face``.

        The codomain of :math:`\gamma_-`, of the deck transformation :math:`G`,
        and of the constitutive response — i.e. of **every** realized boundary
        law, all of which are typed :math:`\Gamma_+ \to \Gamma_-` since campaign
        phase B3.2.

        ⛔ This said the response is an endomorphism *"BOTH ends of*
        :math:`R : \Gamma_- \to \Gamma_-` *— which is why* ``R``\ *, unlike*
        ``G``\ *, is an endomorphism and can be self-adjoint"* when it shipped
        with G6.1 on 2026-08-04, and that was wrong within the day. `[M]` **no
        realized response is an endomorphism of** :math:`\Gamma_-`:
        :math:`\Gamma_- \to \Gamma_-` is the *classifying* typing, while the
        realized response crosses (a constitutive surface is not a quotient, so
        no isometry provides the crossing and the physics does it). Nothing here
        is self-adjoint. See
        :class:`~orpheus.geometry.boundary._factors.BoundaryResponseKernel`.
        """
        return self._face_spaces[("inflow", self._checked_face(face))]


# ─────────────────────────────────────────────────────────────────────
# The per-face tiers of the Γ ladder
# ─────────────────────────────────────────────────────────────────────


def _face_space_name(face: str, role: "TraceRole") -> str:
    r"""The identity of a face tier: ``angular_trace[<face>:<role>]``.

    ⭐ **The face and the role are LOAD-BEARING, not decoration.**
    :meth:`FunctionSpace.__eq__` is ``(name, shape)`` and
    ``inner_product_weights`` is ``compare=False``, so the metric offers no
    secondary discrimination. `[M]` every shipped quadrature gives
    :math:`|\Gamma_+(x_{\min})| = |\Gamma_+(x_{\max})|` (2/2, 4/4, 49/49 on
    ``gauss_legendre(4)`` / ``product(4,4)`` / ``lebedev(17)``) over DIFFERENT
    ordinate indices — so a name omitting the face would make a space compare
    EQUAL to its opposite face's, and a cross-face composition would
    type-check while being wrong. That is the exact error class the binding
    exists to refuse, re-admitted by the mechanism meant to close it.
    """
    return f"angular_trace[{face}:{role}]"


@dataclass(frozen=True)
class AngularFaceTraceSpace(FunctionSpace):
    r"""One directional tier of the boundary trace at ONE face.

    The three tiers — :math:`\Gamma(f)` (the whole ordinate slot),
    :math:`\Gamma_+(f)` (outflow) and :math:`\Gamma_-(f)` (inflow) — are the
    SAME kind of object differing only in *which ordinates they carry*, so
    they are one class parameterised by :attr:`ordinate_indices`, not three.
    :attr:`role` is carried for the space's NAME (its identity) and for
    legibility; **nothing dispatches on it** — a predicate answering by role
    string would be the stringly-typed dispatch this design exists to avoid.

    Construct via :meth:`AngularTraceSpace.face_space` /
    :meth:`~AngularTraceSpace.outflow_space` /
    :meth:`~AngularTraceSpace.inflow_space`, never directly: the parent trace
    space owns the layout, the signed :math:`\Omega\cdot\hat n` and the
    partial-current metric, and is already cached per mesh.

    Parameters
    ----------
    face : str
        The face this tier lives on. Part of the space's IDENTITY via
        :func:`_face_space_name` — see that function for why omitting it
        silently re-admits cross-face composition.
    role : {"full", "outflow", "inflow"}
        Which directional class. Identity + legibility only.
    ordinate_indices : NDArray
        Indices into the face's **leading (ordinate) axis**. Sorted and
        unique. For ``role="full"`` this is ``arange(n_ordinates)``.

    Notes
    -----
    **Why the middle tier exists.** :math:`\Gamma(f)` is not recoverable as
    :math:`\Gamma_+ \oplus \Gamma_-`, because the tangential ordinates
    (:math:`|\Omega\cdot\hat n| \le \epsilon`) belong to neither half. `[M]`
    counts are 0 / 8 / 12 on ``gauss_legendre(4)`` / ``product(4,4)`` /
    ``lebedev(17)``. ⚠ Note the first: **``gauss_legendre(4)`` has no
    tangential ordinates at all**, so a gate written only on it is blind to
    this entire tier (``vv-principles`` Mode 7 — the fixture nulls the term it
    was meant to exercise). Gate on ``product(4,4)`` or ``lebedev(17)``.

    .. important::

       ⭐⭐ **The excess and the metric's degeneracy are ONE fact.** `[M]` on
       every shipped quadrature,

       .. math::

           \ker G_{\Gamma(f)} \;=\; \Gamma(f) \setminus
           (\Gamma_+ \sqcup \Gamma_-),

       because both sides are exactly
       :math:`|\Omega\cdot\hat n| \le \epsilon`. The rows the halves exclude
       are precisely the rows the metric annihilates.

       Two consequences, neither derivable from either statement alone:

       * ⭐ **In the QUOTIENT the full tier IS the direct sum of its halves** —
         :math:`\Gamma(f)/\ker G \cong \Gamma_+(f) \oplus \Gamma_-(f)`. As a
         *Hilbert* space, which is the only category an adjoint cares about, the
         decomposition holds; only the storage array carries the extra rows. So
         "the full trace is two half-spaces" is right in the right category and
         wrong as a statement about shape.
       * **Hence the full tier can never be a chain intermediate while the
         halves can.** The factored-adjoint theorem requires a non-degenerate
         intermediate metric, and the degeneracy *is* the excess.

       Contrast :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`,
       where the halves ARE an explicit axis (slot shape
       ``(2, ng, *face_spatial)``) and the partition is exact. That asymmetry is
       forced, not incidental: :math:`J^\pm` are two independently-defined
       *moments* with no ordinate-level pairing, whereas pairing individual
       ordinates across the hemispheres **is the specular deck transformation**
       — a boundary LAW. Encoding the angular halves as an axis would bake that
       law into a storage layout (the "a transformation hiding as a convention"
       hazard, issue **#328**), demand an even :math:`N` with no tangential
       ordinates — which ``lebedev(17)`` violates at 110 / 12 — and render the
       mirror unspellable as an operator, since it would be the identity on the
       storage.

    **The metric.** :attr:`inner_product_weights` is
    :math:`|\Omega\cdot\hat n_f|\odot w_n` restricted to
    :attr:`ordinate_indices` — a 1-D vector along the leading axis, which the
    resolved :class:`~orpheus.numerics.metric.DiagonalMetric`
    (:func:`~orpheus.numerics.metric._broadcast_leading`) re-expands across the
    trailing group / codim-1 spatial axes on application. It is **never**
    Euclidean: a half-trace pairing that dropped it would be the ERR-067
    family, and it is exactly what makes ``.H`` the Hilbert adjoint
    :math:`A^\dagger = G_V^{-1}A^{\mathsf T}G_W` rather than the bare
    transpose. On the two HALVES the metric is strictly positive (the
    tangential rows are excluded by construction); on the ``"full"`` tier it
    vanishes on the tangential rows, where the base's Moore-Penrose
    :meth:`~FunctionSpace.apply_inverse_metric` correctly returns zero.
    """

    # ``compare=False`` throughout: identity is ``(name, shape)`` (the base's
    # contract), and the name already encodes face + role, so these would be
    # redundant discriminators. It is also REQUIRED for ``ordinate_indices`` —
    # an array-valued compare field makes the dataclass ``__eq__`` raise on
    # the ambiguous element-wise truth value.
    face: str = field(kw_only=True, compare=False)
    role: "TraceRole" = field(kw_only=True, compare=False)
    ordinate_indices: NDArray = field(kw_only=True, repr=False, compare=False)
    #: The directions :math:`\Omega_n` of THIS tier's ordinates, in the tier's
    #: own row order — the parent's :attr:`AngularTraceSpace.directions`
    #: restricted to :attr:`ordinate_indices`.
    #:
    #: ⭐ **Row order is the contract.** Row ``i`` of anything living in this
    #: space is ordinate ``ordinate_indices[i]``, and ``directions[i]`` is that
    #: ordinate's direction. Before this field existed, a consumer needing the
    #: directions had to re-derive them by indexing the quadrature with
    #: ``inflow_indices_for_face(face)`` — correct, but re-derived at every
    #: consumer, and silently wrong if the two index sets ever diverged. A
    #: prescribed-inflow source is the first consumer that cannot be written
    #: at all without them (campaign P6): ``q(Ω)`` is a function of direction,
    #: so a source handed only a SHAPE can express nothing angular.
    directions: NDArray = field(kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Make the documented row-order contract REAL, not implied.

        ``ordinate_indices`` is documented "Sorted and unique", and
        :meth:`to_local` searchsorts it — an unsorted array would return
        silently wrong positions, not raise. So the invariant is guarded at
        construction (``vv-principles`` #14: a docstring that names a
        structure the body only implies must either assert it or weaken).
        The canonical builder (:attr:`AngularTraceSpace._face_spaces`)
        satisfies all three by construction; the guards exist for the
        hand-built case the class docstring already discourages.
        """
        # The base guards first (one-metric-source exclusivity + metric
        # admission — P7).
        super().__post_init__()
        idx = np.asarray(self.ordinate_indices)
        if idx.ndim != 1 or idx.size != int(self.shape[0]):
            raise ValueError(
                f"AngularFaceTraceSpace {self.name!r}: ordinate_indices must "
                f"be 1-D with one entry per leading-axis row; got shape "
                f"{idx.shape} against space shape {self.shape}."
            )
        if idx.size and (
            np.unique(idx).size != idx.size or not np.all(idx[1:] > idx[:-1])
        ):
            raise ValueError(
                f"AngularFaceTraceSpace {self.name!r}: ordinate_indices must "
                f"be sorted ascending and unique — row order IS the space's "
                f"contract, and to_local searchsorts this array. Got {idx!r}."
            )

    def to_local(self, global_rows: NDArray) -> NDArray:
        r"""Map face-slot ordinate indices to their row positions in THIS tier.

        For ``g`` a subset of :attr:`ordinate_indices`, returns ``p`` with
        ``ordinate_indices[p] == g`` — the local↔global index map, owned by
        the space because the embedding data is the space's
        (:attr:`ordinate_indices`; "row order is the contract"). Moved here
        from ``TraceRestrictionOperator.to_local`` at G6.5: the operator is
        the *arrow* :math:`\gamma_\pm`, but which global row sits where is a
        fact about the SUBSPACE, and every consumer of a narrowed trace needs
        it — the deck kernel to read a full-space permutation between two
        half-traces, a row-restricted emission to place a subset of
        :math:`\Gamma_-`.

        Owning it here keeps the classic slip out of call-site hands (it
        stays SPELLABLE — two committed gates pin it): the naive
        ``arange(g.size)`` is exactly correct when ``g`` happens to be a
        PREFIX of the index set — the 1-D slab case — and silently wrong in
        2-D, where a call site that hand-rolled the remap would be gated only
        by end-to-end solves.

        Raises
        ------
        ValueError
            If any requested row is not an ordinate of this tier — i.e. two
            different index sets were crossed (an inflow row asked of an
            outflow tier), the ERR-045 shape.
        """
        g = np.asarray(global_rows, dtype=np.intp)
        idx = np.asarray(self.ordinate_indices, dtype=np.intp)
        outside = g[~np.isin(g, idx)]
        if outside.size:
            raise ValueError(
                f"AngularFaceTraceSpace.to_local: row(s) "
                f"{np.unique(outside).tolist()} are not ordinates of "
                f"{self.name!r}, so they have no position in its row order. "
                f"This usually means two different index sets were crossed "
                f"(an inflow row asked of an outflow tier)."
            )
        return np.searchsorted(idx, g)

    def __repr__(self) -> str:
        return f"AngularFaceTraceSpace({self.name!r}, shape={self.shape})"
