r"""The realized boundary law ``B`` as a whole-trace BOUNDARY-block operator.

Wave O (Issue #208) extracts the boundary conditions from the streaming
operator ``L`` so that ``(L_full + C − S − F − B)\psi = q`` is the canonical
transport algebra. ``B`` is the **boundary-block** operator on the direct-sum
transport state ``V = V_bulk ⊕ V_boundary``: a 2×2 block matrix with only the
``A_ss`` (boundary → boundary) block non-zero. It maps the **outflow** trace to
the **inflow** trace via the per-face realized boundary laws (reflective /
vacuum / white / albedo / periodic), with **no bulk action**.

Block structure
===============

On ``V = V_bulk ⊕ V_boundary`` the four operator families are::

    C, S, F  →  [ A_bb  0 ]   (BULK   — bulk → bulk only)
                [ 0     0 ]

    L_full   →  [ A_bb  A_bs ] (FULL   — streaming couples bulk ↔ boundary)
                [ A_sb  0    ]

    B        →  [ 0     0   ] (BOUNDARY — boundary → boundary only, ``A_ss``)
                [ 0     A_ss]

The whole boundary block is the direct sum ``B = B_a + B_b`` of the two
per-system boundaries (a block-composed system's boundary is the direct sum of
its per-system boundary blocks over the composite biproduct):
:class:`SNBoundaryOperator` (``B_a``) is System A's **trace** ``A_ss`` leaf, and
:class:`RadialCharacteristicBoundaryOperator` (``B_b``) is System B's **ψ½
ray-corner** ``A_ss`` leaf (present only on a seed-carrying mesh). As a sibling
``−B`` of ``L`` in the :class:`~orpheus.numerics.operator.OperatorSum` algebra it
supplies the reflective coupling that ``L`` previously absorbed inside its sweep
(the ``inflow = bc.apply(outflow)`` re-apply); the outer Krylov / SI loop then
drives the boundary **consistency residual** ``ψ.inflow − B·ψ.outflow − q.inflow → 0``.

Construction
============

The per-face boundary laws already live on the
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` in the face-name-keyed ``bc`` dict
(each entry a :class:`~orpheus.geometry.boundary._bound_compat._BoundBoundaryOperator`
wrapping a realized law that carries :attr:`BlockRole.BOUNDARY`). The whole-trace
``B`` is the block composition over the mesh's true boundary faces: for each
face present in the trace it applies that face's law, reading the half-trace of
the face that law's GEOMETRY names. That is the installation face for every
constitutive law — a surface responds to what arrives at its own face — so
those laws occupy the DIAGONAL blocks. A quotient law is off-diagonal:
:class:`~orpheus.geometry.boundary.PeriodicBoundary` reads its partner
(``γ₋ψ|_f = γ₊ψ|_f'``), which is the whole of what makes it a torus
identification rather than a wall. See
:attr:`SNBoundaryOperator._face_domains`, which is that block index and is
certified a PERMUTATION of the faces (every face's outflow feeds exactly one
law).

⚠ Before campaign phase **B3.4c** this said "``B`` is block-diagonal over
faces — it never mixes faces", and the composition enforced it by feeding every
law its own face's ``γ₊`` unconditionally. That was not a property of ``B``; it
was periodic being silently wrong.

See :ref:`operator-algebra` and :ref:`bc-extraction` for the block-matrix
derivation and design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from orpheus.geometry.boundary import (
    PrescribedInflow,
    law_permutes_ordinates,
)
from orpheus.numerics.operator import (
    BlockRole,
    LinearOperator,
    MissingAdjoint,
    SystemRole,
    adjointable,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from orpheus.geometry.boundary import BoundaryTraceLaw
    from orpheus.numerics.space import FunctionSpace
    from orpheus.numerics.spaces.full_field_space import FullFieldSpace
    from orpheus.sn.loss_representation.sweep_schedule import SweepSchedule
    from orpheus.sn.mesh.augmented_mesh import SNMesh
    from orpheus.transport.fields._bases import (
        RadialCharacteristicBoundaryField,
    )
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.full_field import FullField
    from orpheus.transport.radial_characteristic_field import (
        RadialCharacteristicField,
    )
    from orpheus.transport.source_sinks import (
        AngularBoundarySourceSink,
        RadialCharacteristicBoundarySourceSink,
    )


__all__ = [
    "BoundarySplit",
    "RadialCharacteristicBoundaryOperator",
    "SNBoundaryOperator",
    "SNMaskedBoundaryOperator",
]

# Single source for BOTH
# :attr:`RadialCharacteristicBoundaryOperator.is_adjointable` (a ruled law's
# corner map is Euclidean-adjointable) AND
# :meth:`RadialCharacteristicBoundaryOperator._reflect_corner` (an unruled law
# is loud-deferred) — RULING P1's ray carrier.
def _has_ruled_corner_action(law: "BoundaryTraceLaw") -> bool:
    r"""Can this law's inflow at the off-quadrature :math:`\mu = \pm 1` ray be
    written down?

    The corner block is a **linear** operator acting on the ray alone, so a law
    qualifies on two counts and is loud-deferred otherwise (2.5d
    plan-of-record):

    * the law must not be the **prescribed-inflow family** — its inflow is a
      free parameter :math:`q`, not a function of the outflow ray, so a linear
      corner block structurally cannot carry it. This is a TYPE test on
      purpose: the disqualifying property belongs to the family whatever
      :math:`q` currently holds, and testing the source VALUE instead would
      quietly admit ``PrescribedInflow()`` at its default zero source;
    * then either :math:`R = 0` (nothing returns — the corner stays zero) or
      **the composite is a specular pairing** (a mirror pairs
      :math:`\mu = +1` with :math:`\mu = -1` exactly, off-quadrature included,
      which is why the swap is expressible without the quadrature).

    Everything else is genuinely unruled: a hemispheric average needs the
    :math:`|\Omega\cdot n|`-weighted outflow average at :math:`\mu = -1`; a
    spatial wrap needs the partner face's ray; an identity map pairs ordinate
    :math:`n` with itself, which is not a corner action at all.

    Why the pairing is asked of BOTH tiers (B3.4b)
    ----------------------------------------------

    Until B3.4b this read ``law.geometry_map.permutes_ordinates`` alone, which
    was complete while the *only* specular pairing lived in :math:`G`. The
    user's 2026-08-01 ruling put one in :math:`R` too — a polished wall's
    return is constitutive — so ``AlbedoBoundary(α, SpecularReturn(a))`` has
    ``G = SelfPairedDeck.identity()`` and would have been loud-deferred at the
    corner while
    ``ReflectiveBoundary(a, α)``, which it equals as a matrix, is ruled. That
    breaks the equivalence B3.4b asserts, in the one consumer that reads the
    factors rather than the realized operator.

    The pairing is therefore asked of each tier **in that tier's own
    vocabulary**, not through a shared Protocol member. Adding
    ``permutes_ordinates`` to :class:`BoundaryResponseKernel` would have been
    tidier to read and is exactly wrong: :class:`SpecularReemission` already
    carries ``is_adjointable``, so the extra member would make it satisfy
    :class:`BoundaryGeometryMap` structurally — collapsing the tier
    disjointness that ``tests/geometry/test_boundary_factors.py`` asserts
    precisely to stop a response from posing as a geometry. That test is the
    guard against the very conflation this campaign corrected, and a
    convenience member is not worth disarming it.

    Until campaign phase B2 this was ``kind in _RULED_CORNER_KINDS`` against
    the frozenset ``{"vacuum", "reflective"}`` — the same admission, hard-coded
    as tags because the pre-B2.0 shim discarded the law.

    .. note::

       This is a third realizer arm in disguise — *realize this law at the ray
       corner* — which is why it reads the same two factors
       :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` will read at
       phase B4. It is kept local to SN because the off-quadrature ray is a
       curvilinear-SN concept, not a geometry-package one.
    """
    if isinstance(law, PrescribedInflow):
        return False
    return (
        law_permutes_ordinates(law) or law.response_kernel.is_zero
    )


def _zero_bulk_source(mesh: "SNMesh"):
    r"""The zero-bulk ``A_ss`` carrier ``B_a`` emits on the System-A composite.

    Sized from the MESH (not ``zeros_like`` the input) so the carrier is correct
    for any bulk representation (full-angular
    :class:`~orpheus.transport.fields.angular_flux.AngularFlux` OR the Phase-5a
    windowed :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`);
    it carries the scheme's spatial-moment width (#240 D5b-S3) so it composes
    element-wise with the moment-carrying ``(L+C)ψ`` in the ``(L+C − S − B)ψ``
    matvec.

    Scope: **``B_a`` only** — one call site,
    :meth:`SNBoundaryOperator._apply_faces`. The sibling ``B_b``
    (:class:`RadialCharacteristicBoundaryOperator`) does NOT route through
    here: since the B.2b re-type it lives on System B's own carrier, whose
    interior member is a ``RadialCharacteristicInteriorSourceSink`` — a
    different type on a different space, so there is no shared zero-bulk
    concept for the two blocks to have a single source OF.
    """
    from orpheus.transport.source_sinks import AngularSourceSink

    return AngularSourceSink.zeros(mesh.angular_trial_space)


class SNBoundaryOperator(LinearOperator):
    r"""``B_a`` — System A's (trace) boundary law, the SN ``A_ss`` block.

    The boundary operator of the transport system (System A of the 2×2 coupled
    block operator — bulk⊕trace): block-structured over the mesh's true
    boundary faces (diagonal for every constitutive law, off-diagonal for a
    quotient — see :attr:`_face_domains`), ``B_a.apply(ψ)`` returns a
    :class:`~orpheus.transport.full_field.FullField` with **zero bulk**
    and, on each face, the composition ``ι₋ ∘ law ∘ γ₊`` — the realized law
    consumes that face's **outflow** half-trace and produces its **inflow**
    half-trace, and the scatter writes the image back into the face slot.

    Since campaign phase **B3.2** every outflow row of the output is zero
    **by typing**: ``B`` is the ``A_ss`` block ``V_outflow → V_inflow``, and
    the outflow rows — which carry no ``B`` term in the block matrix — are not
    in the law's codomain to be emitted on. Pre-B3.2 the law was a *full-face*
    operator and this class projected its image onto the inflow rows
    (``B_face = P_inflow ∘ law``), discarding the rest; the mismatch between
    that declared domain and the physics was the root defect the boundary
    review identified. See :meth:`_reflect_trace` for the transpose's
    mirror-image discipline. It
    composes as ``−B`` in ``(L_full + C − S − F − B)`` (acting on the same
    :class:`~orpheus.transport.full_field.FullField` carrier as ``L``/``C``/``S``/``F``).

    On a **seed-carrying** composite the whole boundary block is the direct sum
    ``B = B_a + B_b``, where ``B_b`` is the sibling
    :class:`RadialCharacteristicBoundaryOperator` — System B's (ψ½ ray-corner)
    boundary (RULING P1: a block-composed system's boundary is the direct sum of
    per-system boundary blocks over the composite biproduct; the off-diagonal
    structure is keyed to face physics — reflection is a per-system
    endomorphism ⇒ block-diagonal). "Direct sum" is meant literally, NOT as a
    ``+``: since the B.2d re-type ``B_a`` neither reads nor pads a ray block
    (its carrier's members are exactly ``interior`` and ``boundary``), and
    ``B_b`` lives on ``radial_characteristic_field_space``. The two are placed
    at the (A,A) and (B,B) slots of the coupled grid; spelling ``B_a + B_b``
    raises :class:`~orpheus.numerics.operator.IncompatibleOperatorComposition`
    (``OperatorSum`` requires equal domains — the shared space-agreement law
    :func:`~orpheus.numerics.operator._agreed_space` since G6.3 step 8.0) and
    always has, by design.

    The role is :attr:`BlockRole.BOUNDARY`; the domain and codomain are the
    mesh's composite carrier
    :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
    (``sn_mesh.full_field_space``) — the SAME space ``L``/``C``/``S``/``F``
    report, so the :class:`~orpheus.numerics.operator.OperatorSum` composition
    guard accepts ``(L + C - S - F - B)`` (Wave O / O.2b R5). ``B`` acts on the
    composite as the ``A_ss`` block (zero bulk; non-zero only on the trace
    block, where the cosine-weighted ``|Ω·n|·w`` partial-current metric lives).
    That block metric is what makes the Hilbert adjoint ``B.H`` the physically
    correct partial-current adjoint — the one channel by which the white-BC
    adjoint becomes available.

    Capabilities
    ------------

    ``apply`` always. ``apply_transpose`` is advertised iff EVERY per-face law
    advertises it. The discriminator is the law's FACTORS, not its class.

    ⭐ **Every shipped law now advertises one**, so the intersection is
    currently vacuous — it was not until **G6.3 step 3** (2026-08-04). The
    Lambertian was the lone holdout: its Euclidean and ``|Ω·n|·w``-weighted
    transposes differ, so it withheld the ambiguous one. Factoring its
    realization into a contraction and a broadcast removed the ambiguity —
    each link has ONE honest transpose — rather than choosing between the two
    readings. The intersection rule STAYS: it is the structure that keeps a
    future non-adjointable law from silently granting the composite a
    transpose it does not have.
    The intersection rule keeps ``apply_transpose`` honest: it is reachable
    only when every block can honour it.

    Since **B3.4c** the factor tier and the realized operator agree on that
    question, and a registry-wide gate holds them together
    (``tests/geometry/test_bc_universal_invariants.py``). They had drifted:
    ``SpatialWrap.is_adjointable`` declared ``False`` while the operator
    realizing periodic answered ``True``, so a consumer got opposite answers
    depending on which it asked — the declaration was reporting an unbuilt
    partner channel (#183) in a slot whose contract is a property of the map.
    B3.4c built the channel and the declaration became true. **Since P3
    (2026-08-05) the agreement holds for every shipped law with no
    exception.** The AFFINE
    :class:`~orpheus.geometry.boundary.PrescribedInflow` sat outside it until
    then — both its factors declare adjointable while its realized affine
    operator declined a transpose — and P3 closed the gap by realizing the
    law's LINEAR factor, the zero morphism, whose transpose is itself. The
    affine ``q`` is still not carried by the factor pair
    (:ref:`bc-affine-source-channel`); the difference is that the tier no
    longer pretends ``q`` is an operator.

    ⚠ **A capability widened here.** :attr:`is_adjointable` below is the
    per-face conjunction, so ONE non-adjointable law made the whole ``B``
    block non-adjointable. Pre-P3 a declared prescribed face did exactly
    that, putting ``B.apply_transpose`` and ``B.H`` out of reach on such a
    mesh; post-P3 it does not. The configuration is observable only in tests
    — #189 keeps prescribed inflow out of every production driver — which is
    why it went unnoticed.

    Since **B3.4b** an albedo face answers by its **re-emission closure**, not
    by its class: a specular closure realizes to the same scaled permutation
    reflective does, a diffuse one to the same Lambertian white uses, and the
    closure-free spelling never reaches here — the realizer refuses it. The
    predicate below already computed this correctly, since it reads each
    realized law's own ``is_adjointable``; it is the enumeration in prose that
    had to stop naming classes.

    ⚠ **Corrected 2026-08-06.** The diffuse arm above carried "(not
    adjointable)" and the specular arm "(adjointable)". `[M]` on a slab face,
    **all three are ``True``** — ``WhiteBoundary()``,
    ``AlbedoBoundary(0.7, IsotropicReturn(...))`` and
    ``AlbedoBoundary(0.7, SpecularReturn(...))`` — because B3.4b factored the
    Lambertian so the chain transposes leaf by leaf. Two lessons ride on this
    one line. (1) The paragraph is *itself* a correction pass — it says so in
    its own last sentence — and it still left a class-named falsehood in the
    text it was correcting; "stop naming classes" was applied to the subjects
    and not to the parenthetical verdicts. (2) The subject ("white") and the
    negation ("not adjointable") sit on DIFFERENT LINES, so the line-based grep
    that found and fixed the sibling claim on :774 could not see this one — a
    concept grep for a negated claim needs a multi-line window.

    Parameters
    ----------
    sn_mesh : SNMesh
        The augmented geometry — carries the per-face boundary laws
        (the face-name-keyed ``bc`` dict) and the unified trace space (same instance the
        composite carrier is bound to; the mesh-identity invariant of
        :class:`~orpheus.sn.operators.streaming.StreamingOperator` applies here too).
    """

    block_role = BlockRole.BOUNDARY

    def __init__(self, sn_mesh: "SNMesh") -> None:
        self.sn_mesh = sn_mesh

    @property
    def _face_laws(self) -> dict[str, LinearOperator]:
        """Map each true boundary face → its per-face realized law.

        Read from ``sn_mesh.bc`` for the faces the trace carries
        (slab ``xmin``/``xmax``; curvilinear ``xmax`` only; 2-D Cartesian
        all four) — the dict and the trace layout share their keys by
        construction (both derived from ``face_labels``, C4 / #220).
        Single source of truth — the laws are the same objects
        the sweep consumes, so ``B`` cannot drift from the realized BCs.
        """
        return {
            face: self.sn_mesh.bc[face]
            for face in self.sn_mesh.angular_trace.layout.faces
        }

    @property
    def _face_domains(self) -> dict[str, str]:
        r"""Each boundary face → the face whose :math:`\Gamma_+` its law consumes.

        **``B``'s block structure over faces, named.** Since B3.2 a realized law
        is typed :math:`\Gamma_+ \to \Gamma_-`; this says *whose* :math:`\Gamma_+`,
        so the pair ``(installation face, domain face)`` is the ``(row, column)``
        index of the block the law occupies. Every constitutive law is on the
        DIAGONAL (a surface responds to what arrives at its own face). Periodic
        is OFF-diagonal — ``γ₋ψ|_f = γ₊ψ|_{f'}`` — which is the whole content of
        being a quotient rather than a wall, and the reason **B3.4c** exists.

        Before B3.4c :meth:`_reflect_trace` fed every law its own face's
        :math:`\Gamma_+` unconditionally, so periodic returned a face's own
        outflow as its inflow. The defect was invisible to a shape check
        (``|Γ₊| == |Γ₋|`` everywhere) and to a single-face probe (with one draw
        shared by both faces the identity looks defensible, since periodicity
        DOES identify the faces) — it is observable only when the two faces
        carry different data, which is the real sweep's situation.

        The answer comes from the law's **geometry factor**, which is B3.0's
        ruling read one level up: :math:`G` carries the crossing, in ANGLE for a
        mirror and in SPACE for a wrap. A response kernel is constitutive and
        structurally cannot reach another face, so it is never asked. Spelled
        as the factor read rather than hidden behind a law-level helper,
        because the spelling IS the ruling.

        The map is certified a **permutation of the boundary faces**, which is
        the well-posedness statement for the whole block: every face's
        :math:`\Gamma_+` is consumed by exactly one law, so no outflow is read
        twice and none is dropped. Two ill-posed configurations it refuses,
        both silent before B3.4c:

        * **A half-declared periodic pair** (``xmin`` periodic, ``xmax``
          vacuum). A translation quotient is symmetric — a face cannot be glued
          to a partner that is not glued back — and here
          :math:`\Gamma_+(\texttt{xmax})` would feed two laws while
          :math:`\Gamma_+(\texttt{xmin})` fed none. It also breaks the
          transpose, whose whole-slot writes would then collide.
        * **A periodic face whose partner is not a boundary face at all** — a
          curvilinear mesh carries ``xmax`` alone, so a wrap installed there
          names a partner the trace has no slot for.
        """
        faces = self.sn_mesh.angular_trace.layout.faces
        domains = {
            face: self.sn_mesh.bc[face].law.geometry_map.domain_face(face)
            for face in faces
        }
        if sorted(domains.values()) != sorted(faces):
            unknown = sorted(set(domains.values()) - set(faces))
            detail = (
                f"{unknown} name no face of this mesh"
                if unknown
                else "some face's Γ₊ is consumed twice and another's not at all"
            )
            raise ValueError(
                f"SNBoundaryOperator: the per-face domain map {domains!r} is "
                f"not a permutation of this mesh's boundary faces "
                f"{sorted(faces)} — {detail}. Every face's outflow must feed "
                f"exactly one law. A periodic pair must be declared on BOTH "
                f"its faces: the translation quotient is symmetric, so a face "
                f"glued to a partner that is not glued back is not an "
                f"identification."
            )
        return domains

    @property
    def is_adjointable(self) -> bool:
        # B = ⊕ per-face laws; the composite adjoint exists iff EVERY face law
        # is adjointable. Since G6.3 step 3 every shipped law is (white was
        # the lone holdout until its realization was factored); an albedo face
        # still answers by its CLOSURE, per B3.4b — the mechanism survives the
        # answers becoming uniform.
        # Reading each REALIZED law's own predicate rather than its class is
        # what makes that automatic. The per-face
        # intersection rule, computed recursively like every composite
        # predicate. is_invertible inherits base False — NOT because the
        # reflection map is singular (a permutation is invertible), but
        # because ``B_face = ι₋ ∘ law ∘ γ₊`` is rank-deficient BY
        # CONSTRUCTION: ``γ₊`` discards the inflow and tangential rows on the
        # way in and ``ι₋`` leaves the outflow and tangential rows at zero on
        # the way out, so B maps a full face slot onto the inflow subspace and
        # cannot be inverted. (Pre-B3.2 the same rank deficiency arose from a
        # codomain projection applied to a full-face law; the narrowing moved
        # it from something the consumer imposed to something the types say.)
        laws = self._face_laws.values()
        return bool(laws) and all(law.is_adjointable for law in laws)

    @property
    def domain(self) -> "FunctionSpace":
        # The composite carrier (NOT the bare trace): ``B.apply`` consumes /
        # emits a full FullField (zero bulk + reflected trace), so the
        # advertised space must be the bulk ⊕ trace composite — matching the
        # ``L``/``C``/``S``/``F`` siblings for the OperatorSum composition
        # guard, and carrying the block-diagonal G-adjoint metric ``B.H``
        # reads. Wave O / O.2b R5.
        return self.sn_mesh.full_field_space

    @property
    def codomain(self) -> "FunctionSpace":
        return self.sn_mesh.full_field_space

    def _reflect_trace(
        self, boundary: "AngularBoundaryFlux", method: str,
        faces: "Iterable[str] | None" = None,
        rows: "Mapping[str, np.ndarray] | None" = None,
    ) -> "AngularBoundarySourceSink":
        r"""Core ``A_ss`` action on the trace ALONE — apply each face's law
        (``method`` ∈ {apply, apply_transpose}) to that face's slot, project onto
        the codomain row, and return a boundary-only
        :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`.

        ``B`` is the ``A_ss`` block ``V_outflow → V_inflow``, and **since
        campaign phase B3.2 the realized per-face law is typed that way too**:
        it consumes :math:`\Gamma_+` and produces :math:`\Gamma_-`. The face
        action is therefore the composition

        .. code-block:: text

            B_face = ι₋ ∘ law ∘ γ₊          (forward)
            B_faceᵀ = ι₊ ∘ lawᵀ ∘ γ₋        (Euclidean transpose)

        with ``γ±`` the trace restrictions
        (:class:`~orpheus.numerics.operator.TraceRestrictionOperator`, cached on
        the trace space) and ``ι± = γ±ᵀ`` their scatters. Nothing is computed
        and then discarded, and a non-zero outflow emission — which would
        corrupt the outflow-definition residual ``ψ.outflow − streamed``, a
        quantity carrying no ``B`` term at all — is **unrepresentable** rather
        than merely projected away.

        Since **G6.3 step 8** the ``law ∘ γ₊`` half is a genuine composition —
        ``face_action = law @ γ₊`` — not a sequence of ``.apply`` calls, so
        :class:`~orpheus.numerics.operator.OperatorProduct`'s composability
        check runs on every face of every call: ``law.domain`` must BE the
        half-trace ``γ₊`` emits. What that buys, and what it does not:

        * ⭐ **The transpose leg is derived, not written.** ``face_action``
          serves BOTH legs, because ``(law ∘ γ₊)ᵀ = γ₊ᵀ ∘ lawᵀ`` is
          :meth:`OperatorProduct.apply_transpose`. The ⚠ trap below therefore
          stopped being a thing to remember — see there.
        * The ``ι₋`` end stays an explicit ``γ₋`` verb rather than joining the
          product, because the ``rows`` branch does not emit through a plain
          scatter (it writes a SUBSET of Γ₋, placed by the half-trace SPACE's
          ``to_local`` — G6.5). Composing it would need a second,
          row-restricted operator per call to say something the whole-slot
          branch already says.
        * ⭐ **Periodic's check is LIVE — and it is the law the check was
          designed for** (G6.3 step 7). A boundary law is a ``@``-chain
          :math:`\Gamma_+ \to \Gamma_-` whose deck-transformation case is
          degenerate at LENGTH ONE (:ref:`bc-deck-length-one-chain`), and
          every law's link is a typed arrow between the two half-traces —
          periodic's an isomorphism between two DIFFERENT faces' spaces,
          :math:`\Gamma_+(f') \to \Gamma_-(f)`, derived from the wrap's
          MOTION by the same kernel that builds the mirror's. Periodic is
          the only off-diagonal block of ``B``, so composing ``law ∘ γ₊``
          here is precisely where a wrong-partner wiring (the B3.4c defect,
          98 % relative when live) now raises instead of computing a
          plausible wrong answer. (Until step 7 the link was a bare unbound
          :class:`~orpheus.numerics.operator.IdentityOperator` — an
          *endomorphism* :math:`V \to V` standing in for that isomorphism.
          The identity names no spaces — that is what "identity" means — so
          it could never be the arrow, and one ``None`` short-circuited
          this check on exactly the law it was designed for. Step 7
          replaced the link rather than annotating it.)

        Pre-B3.2 the law was a *full-face* operator and this method projected
        its image onto the inflow rows (``B_face = P_inflow ∘ law``), masking
        the input on the transpose leg. That slice-write was the root defect
        the boundary review identified: the law's declared domain was the whole
        face slot while its physics was ``outflow → inflow``, and every symptom
        in the review's §4 followed from the mismatch.

        ⚠ **The trap, and why it is now unspellable rather than remembered.**
        The transpose must scatter over :math:`\Gamma_+`, never over
        :math:`\Gamma_-`. Output-projecting ``lawᵀ`` onto the law's own
        codomain instead extracts its DIAGONAL block — for vacuum that spells a
        spurious ``+1`` where the forward is the ZERO map, and it was caught
        only by the A2a grid-reciprocity arm on the het-VACUUM sphere, because
        off-diagonal permutation laws are bit-identical under either spelling
        and every reflective-fixture gate stayed green over the wrong one.
        Step 8 removed the *choice*: the scatter is whatever the composed
        ``face_action`` transposes to, and ``γ₋`` is not one of its factors,
        so there is no index to get wrong here. ⚠ **Not the same as
        impossible** — a future edit could still hand-write
        ``γ₋ᵀ(lawᵀ(·))``, and `[M]` it would run silently, because
        :math:`|\Gamma_+| = |\Gamma_-|` makes the shapes agree. What changed
        is that re-opening the trap now takes *abandoning the composition*, a
        visible structural edit, rather than choosing the wrong one of two
        adjacent names. (Composing ``law @ γ₋`` instead — the other way to
        reach for it — does raise
        :class:`~orpheus.numerics.operator.IncompatibleOperatorComposition`.)
        The failure mode is kept in the record because it is what a future
        refactor must know not to re-open, and because a reader who sees only
        the fix cannot tell which of the two spellings is the right one.

        The metric-correct Hilbert adjoint ``B.H`` under ``|Ω·n|·w`` is
        separate; this Euclidean ``apply_transpose`` is the un-weighted shadow.
        ⚠ Note the composition does NOT change that: ``@`` is metric-neutral
        (:meth:`OperatorProduct.apply` is plain function composition), so
        routing through it leaves this leg Euclidean exactly as before —
        which is why every value here is bit-identical across step 8.

        This is the **single source of truth** for the boundary reflection: both
        the full-field :meth:`apply` (lifted onto a zero-bulk carrier) and the
        trace-only :meth:`reflect_into_inflow` (the bare-sweep inflow seed) route
        through it, so the two cannot drift (Cardinal Rule 2).
        """
        from orpheus.transport.source_sinks import AngularBoundarySourceSink

        # Single mesh source (mesh-identity invariant — see class docstring):
        # the output buffers, the trace selectors, and ``_face_laws`` ALL read
        # ``self.sn_mesh``, so a mismatched input trace cannot desync the
        # projection from the buffer geometry.
        mesh = self.sn_mesh
        trace = mesh.angular_trace
        out_boundary = AngularBoundarySourceSink.zeros(mesh.angular_trace)
        # ``faces=None`` reflects every boundary face (the whole-trace ``B``);
        # a face subset restricts the reflection to those faces — the Phase 3
        # Gauss-Seidel octant-group schedule reflects only the just-swept
        # group's reflective OUTGOING faces, leaving the rest of the inflow
        # trace untouched (zero in this returned sink).  The subset action is
        # the EXACT restriction of ``B``'s block ROWS — and note the reason,
        # because B3.4c falsified the one previously written here ("``B`` is
        # block-diagonal over faces, so no cross-face coupling is dropped").
        # Block-diagonality was SUFFICIENT for exactness but never NECESSARY:
        # ``faces`` filters which OUTPUT faces are emitted on, while the whole
        # input trace stays in scope, so an off-diagonal block (periodic) still
        # reads its partner's half-trace and the restriction is exact for a
        # quotient law too.
        #
        # ``rows`` (#226 step 2) restricts WITHIN a face: per face, only the
        # given ordinate rows of the codomain projection are emitted (a subset
        # of the inflow rows — the schedule-split ``B_lower``/``B_upper``
        # halves of :meth:`split`).  A face absent from ``rows`` emits nothing.
        # Row-granular restriction is exact for the same reason the face
        # restriction is: the projected action writes each target row
        # independently.
        face_laws = self._face_laws
        if faces is not None:
            unknown = set(faces) - set(face_laws)
            if unknown:
                raise ValueError(
                    f"_reflect_trace: face(s) {sorted(unknown)} are not "
                    f"boundary faces of this mesh; available faces: "
                    f"{sorted(face_laws)}."
                )
            face_laws = {f: face_laws[f] for f in faces}
        if rows is not None:
            face_laws = {f: law for f, law in face_laws.items() if f in rows}
        face_domains = self._face_domains
        for face, law in face_laws.items():
            # B3.2 — the law's DOMAIN is Γ₊ and its CODOMAIN is Γ₋, so the
            # face action is the composition ``ι₋ ∘ law ∘ γ₊`` spelled out.
            # Nothing is computed and then thrown away: the outflow rows the
            # pre-B3.2 slice-write discarded are simply not in the domain.
            #
            # B3.4c — and the ``γ₊`` is the DOMAIN face's, which is this face
            # for every law but periodic. The two names are distinct for the
            # same reason ``B`` is a block matrix rather than a diagonal one;
            # see :meth:`_face_domains`.
            domain_face = face_domains[face]
            face_in = boundary.face_view(domain_face)
            gamma_out = trace.outflow_restriction(domain_face)   # γ₊(domain)
            gamma_in = trace.inflow_restriction(face)            # γ₋(face)
            # ⭐ G6.3 step 8 — COMPOSE the pair, do not apply it in sequence.
            # ``law ∘ γ₊`` is one operator ``Γ(domain) → Γ₋(face)``, and
            # building it runs the composability check that a sequence of
            # ``.apply`` calls structurally cannot: ``law.domain`` must BE
            # ``γ₊(domain_face).codomain``, so feeding a law the wrong face's
            # half-trace — the B3.4c defect, invisible to a shape check
            # because ``|Γ₊| == |Γ₋| == |Γ|/2`` on every reachable face —
            # raises here instead of computing a plausible wrong answer.
            face_action = law @ gamma_out
            if method == "apply":
                image = face_action.apply(face_in)
                if rows is None:
                    out_boundary.face_view(face)[...] = (
                        gamma_in.apply_transpose(image)
                    )
                else:
                    # A row-restricted emission (the schedule's split halves):
                    # keep only the requested inflow rows of the image. The
                    # remap MUST go through the SPACE's ``to_local`` (G6.5 —
                    # Γ₋(f) owns its row order): the requested rows are a
                    # subset of Γ₋, and they are a PREFIX of it only in 1-D,
                    # so a hand-written ``arange`` is right on a slab and
                    # wrong in 2-D.
                    sel = rows[face]
                    out_boundary.face_view(face)[sel] = image[
                        trace.inflow_space(face).to_local(sel)
                    ]
            else:
                # The forward is ``ι₋ ∘ law ∘ γ₊``, so the Euclidean transpose
                # distributes as ``ι₊ ∘ lawᵀ ∘ γ₋`` — restrict the INPUT to
                # the forward's codomain (Γ₋), apply the law's transpose, and
                # scatter the result back over Γ₊. With ``rows`` the forward
                # additionally projects onto those inflow rows, which
                # transposes to masking the input by them FIRST.
                #
                # B3.4c — the two legs read and write DIFFERENT face slots when
                # the block is off-diagonal: the input is this face's Γ₋ and
                # the image scatters over the DOMAIN face's Γ₊, mirroring the
                # forward exactly. Whole-slot assignment stays safe because
                # ``_face_domains`` is certified a permutation of the faces, so
                # no two blocks scatter into one slot.
                transposed_in = boundary.face_view(face)
                #
                # The checked bridge licenses the raw verb (spec §39.1) —
                # unreachable-in-practice because :attr:`is_adjointable`
                # gates the composite eagerly, but the per-face raise keeps
                # the refusal loud if a caller bypasses the predicate.
                if not adjointable(law):
                    raise MissingAdjoint(
                        f"SNBoundaryOperator.apply_transpose: face {face!r} "
                        f"law {type(law).__name__} has no Euclidean "
                        f"transpose — reachable only when every face law is "
                        f"adjointable (see is_adjointable)."
                    )
                if rows is None:
                    restricted = gamma_in.apply(transposed_in)
                else:
                    sel = rows[face]
                    masked = np.zeros_like(transposed_in)
                    masked[sel] = transposed_in[sel]
                    restricted = gamma_in.apply(masked)
                # ⭐ step 8 — the SAME composed operator serves this leg,
                # because ``(law ∘ γ₊)ᵀ = γ₊ᵀ ∘ lawᵀ`` falls out of
                # :meth:`OperatorProduct.apply_transpose`. That is what makes
                # the ⚠ trap above STRUCTURAL rather than remembered: the
                # scatter is ``γ₊ᵀ`` *because γ₊ is the operator that was
                # composed*, so there is no longer a spelling of this line
                # that scatters over Γ₋. `[M]` bit-identical to the
                # hand-written ``γ₊ᵀ(lawᵀ(·))`` on every shipped law.
                out_boundary.face_view(domain_face)[...] = (
                    face_action.apply_transpose(restricted)
                )
        return out_boundary

    def _apply_faces(
        self, psi: "FullField", method: str,
        rows: "Mapping[str, np.ndarray] | None" = None,
    ) -> "FullField":
        r"""Lift the trace-only :meth:`_reflect_trace` onto the full
        :class:`~orpheus.transport.full_field.FullField` carrier with **zero
        bulk** — ``B_a``, the System-A (trace) boundary block on ``V = V_bulk ⊕
        V_boundary``.  #257 S8a: history-free (the matvec leaf is a base
        arrow ``FullField -> FullField``; the comonad lives on the driver).

        ``B_a`` touches only the trace; the ray-corner boundary is the sibling
        :class:`RadialCharacteristicBoundaryOperator` (``B_b``) — SYSTEM B's
        own boundary block, living at the coupled grid's (B,B) slot (RULING P1
        — a block-composed system's boundary is the direct sum of per-system
        boundary blocks; see the module docstring). B.2d: System B is its own
        composite, so ``B_a`` neither reads nor pads a ray block.
        """
        from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
        from orpheus.transport.full_field import FullField

        mesh = self.sn_mesh
        # The shared System-A matvec input parse (CS4c step 6 item 6.3 — the
        # R6 row of the monomorphic-leaves ledger: ONE body, the five
        # consumers L/LC × apply/transpose + this): a foreign carrier is a
        # typed TypeError naming this operator and the carrier it wanted, a
        # space-content mismatch a ValueError carrying the greppable
        # ``space-content invariant`` vocabulary. ``_apply_faces`` serves
        # BOTH ``apply`` and ``apply_transpose``; the context names the
        # caller rather than hard-coding ``apply``, which mis-attributed
        # every failure on the transpose path.
        FullField.require_member(
            psi, mesh=mesh, context=f"SNBoundaryOperator.{method}",
        )
        # Role parse at the composite boundary: ``B_a`` reads a FLUX trace
        # (``_reflect_trace`` applies the boundary law to outflow flux), but
        # the ``FullField.boundary`` slot erases the role (the F2-sibling
        # erasure — #289). A source-role
        # trace arriving here is a caller error worth raising loudly.
        trace = psi.boundary
        if not isinstance(trace, AngularBoundaryFlux):
            raise TypeError(
                f"SNBoundaryOperator: the input composite's boundary must "
                f"be an AngularBoundaryFlux trace; got {type(trace).__name__}."
            )
        return FullField(
            interior=_zero_bulk_source(mesh),
            boundary=self._reflect_trace(trace, method, rows=rows),
        )

    def apply(self, psi: "FullField") -> "FullField":
        r"""Forward action ``B_a·ψ`` — per-face boundary law on the trace, zero bulk."""
        return self._apply_faces(psi, "apply")

    def reflect_into_inflow(
        self, boundary: "AngularBoundaryFlux",
        faces: "Iterable[str] | None" = None,
    ) -> "AngularBoundarySourceSink":
        r"""Trace-only forward reflection ``B·ψ.outflow`` projected onto the
        inflow row — the ``A_ss`` action expressed on the boundary trace ALONE.

        Returns a boundary-only
        :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` whose
        **inflow** ordinate slots carry the per-face reflected outflow (``R·G``
        for reflective, the angular average for white, zero for vacuum) and whose
        outflow slots are zero. It is :meth:`apply` without the zero-bulk carrier
        — the trace-only entry for seeding ``ψ.inflow = B·ψ.outflow`` on a bare
        boundary buffer without fabricating a throwaway zero-bulk field just to
        reach the ``A_ss`` block.  No production driver seeds by hand any more
        (every within-group solve — and, since #448, the eigenvalue finalize's
        one reconstruction step — receives ``B·ψ.outflow`` as the ``B`` GAIN
        through ``rhs.boundary``); its consumers are the sweep-tier gates that
        drive bare sweeps in a loop, through :meth:`reflect_inflow_inplace`.

        ``faces`` (Phase 3 Gauss-Seidel): ``None`` (default) reflects every
        boundary face — the whole-trace Jacobi reflect.  A face subset restricts the
        reflection to those faces: the octant-group G-S schedule reflects only
        the just-swept group's reflective OUTGOING faces between octant-group
        sweeps, so a later group reads the fresh reflected inflow (the
        ``(L+C−B_lower)⁻¹`` forward substitution).  The subset restricts
        ``B``'s block ROWS and is exact because the whole input trace stays in
        scope — NOT because ``B`` is block-diagonal, which since B3.4c it is
        not (see :meth:`_reflect_trace`).
        """
        return self._reflect_trace(boundary, "apply", faces=faces)

    def reflect_inflow_inplace(
        self, boundary_flux: "AngularBoundaryFlux",
        faces: "Iterable[str] | None" = None,
    ) -> None:
        r"""In place: overwrite each face's inflow rows with the reflected
        outflow — ``ψ.inflow ← (B_a·ψ)|_{\rm inflow}``, face-restrictable.

        The MUTATING façade over :meth:`reflect_into_inflow` (single source —
        both route through :meth:`_reflect_trace`), carrying the sweep
        substrate's reflect signature
        (``Callable[[AngularBoundaryFlux, tuple[str, ...]], None]``).
        `[M]` #448: **no production caller** — the eigenvalue finalize's reflect
        of the converged trace (its last one) retired when the finalize became
        one step of the driven iteration, in which ``B`` is a gain; the
        octant-group Gauss-Seidel resolvent never routed here (it binds
        :meth:`SNMaskedBoundaryOperator.reflect_rows_inplace`, below).  It
        stays as the operator's own MUTATING verb for the sweep-tier gates
        that drive bare sweeps in a loop
        (``tests/sn/_test_helpers.py::reflect_outflow_into_inflow`` and the
        #448 trace gate) — the inflow-row selection is the operator's
        knowledge, not a test's.

        ⚠ NOT the reflect the reified ``M = (L+C−B_lower)`` supplies to
        :func:`~orpheus.sn.loss_representation._sweep_scheduled`. That one is
        :meth:`SNMaskedBoundaryOperator.reflect_rows_inplace` — ADDITIVE and
        restricted to ``B_lower``'s rows, because a forward-substitution row
        completes ``z_in = y_row + (Bz)_row`` on top of a seed. The two are
        deliberately not interchangeable: a whole-face ASSIGNMENT there drops
        ``y_row`` and stamps fresh values onto rows the splitting defines as
        lagged — the dissolved ``_GaussSeidelResolvent``'s overwrite defect
        (#226 §17 falsifier-3, round-trip O(1) at 2.667).

        Trace-only: the ψ½ ray corner is System B's boundary ``B_b``
        (:class:`RadialCharacteristicBoundaryOperator`), reflected by the
        coupled gain grid on a carrying mesh — one operator per system (RULING
        P1); its in-place ray sibling ``reflect_corner_inplace`` retired at
        #448 with its last caller (the pre-#448 finalize).
        """
        reflected = self.reflect_into_inflow(boundary_flux, faces=faces)
        trace = self.sn_mesh.angular_trace
        selected = (
            boundary_flux.layout.faces if faces is None else faces
        )
        for face in selected:
            inflow = trace.inflow_indices_for_face(face)
            boundary_flux.face_view(face)[inflow] = (
                reflected.face_view(face)[inflow]
            )

    def split(self, schedule: "SweepSchedule") -> "BoundarySplit":
        r"""Split ``B = B_lower + B_upper`` under ``schedule``'s octant order
        (#226 §17 W2 — the matrix splitting of the boundary G-S; a
        *splitting*, NOT a "regular splitting" — see #341 and
        :ref:`sn-boundary-gs-not-regular`).

        ``B_lower`` carries exactly the (face, inflow-row) couplings the
        scheduled sweep realizes IN-sweep (rows whose octant group is swept
        strictly after the face's reflect —
        :meth:`~orpheus.sn.loss_representation.sweep_schedule.SweepSchedule.lower_inflow_rows`);
        ``B_upper`` carries the complement (the cyclic back-edges plus every
        row of a never-reflected face — vacuum, white, albedo, periodic),
        lagged by the SI driver as an external gain.  The partition is exact:
        the specular map has no octant-diagonal, and the two row sets are
        complementary within each face's inflow by construction here.

        Returns a named pair so the two construction sites cannot be swapped
        silently: ``M = (L + C) - parts.lower`` and ``gains = (S, parts.upper)``.
        The Jacobi schedule yields an empty lower support (``B_lower = 0``,
        ``B_upper = B``) — the degenerate that recovers the plain lagged-``B``
        iteration.
        """
        lower_rows = schedule.lower_inflow_rows(self.sn_mesh)
        trace = self.sn_mesh.angular_trace
        upper_rows = {
            face: np.setdiff1d(
                trace.inflow_indices_for_face(face),
                lower_rows.get(face, np.empty(0, dtype=np.intp)),
            )
            # The same face set the per-face law iterates (single source —
            # the trace layout and ``bc`` share keys by construction). These
            # are the block ROWS; which COLUMN each reads is
            # :attr:`_face_domains`, and the split partitions rows.
            for face in self._face_laws
        }
        return BoundarySplit(
            lower=SNMaskedBoundaryOperator(self, lower_rows, schedule),
            upper=SNMaskedBoundaryOperator(self, upper_rows, schedule),
        )

    def apply_transpose(self, psi: "FullField") -> "FullField":
        r"""Euclidean transpose ``Bᵀ·ψ`` — per-face ``apply_transpose``, zero bulk.

        Reachable only when every per-face law is adjointable (see
        :attr:`is_adjointable`), which since G6.3 step 3 every SHIPPED law is —
        including white, whose realization was factored so that the Lambertian
        chain transposes leaf by leaf (B3.4b). `[M]` 2026-08-06: a slab with a
        ``WhiteBoundary()`` face reports ``is_adjointable = True`` and this
        method returns.

        ⚠ **Corrected 2026-08-06.** This docstring read "the white BC has no
        Euclidean transpose" — present-tense false since B3.4b, and in direct
        contradiction with :attr:`is_adjointable`'s own (correct, past-tense)
        note twelve properties above. Do not confuse it with
        :class:`RadialCharacteristicBoundaryOperator` (``B_b``), where white
        genuinely IS in the loud-deferred set: that predicate is about the
        sphere's **off-quadrature μ = ±1 ray corner**, a different action on a
        different carrier.

        What remains true, and is the reason both spellings exist: ``Bᵀ`` is
        the **Euclidean** transpose, while the physically-meaningful adjoint is
        ``B.H`` under the ``|Ω·n|·w`` trace metric (Wave O step O.2). For white
        the two differ; reciprocity gates must use ``.H``.
        """
        return self._apply_faces(psi, "apply_transpose")


class RadialCharacteristicBoundaryOperator(LinearOperator):
    r"""``B_b`` — System B's (ψ½ ray-corner) boundary law, the ray ``A_ss`` block.

    The boundary operator of the radial-characteristic system (System B of the
    2×2 coupled block operator — the ψ½ ray). A first-class sibling of
    :class:`SNBoundaryOperator` (``B_a``, System A's trace boundary), typed —
    since the B.2b re-type — on **System B's own carrier**: domain = codomain =
    ``sn_mesh.radial_characteristic_field_space``, acting
    ``RadialCharacteristicField → RadialCharacteristicField`` (reads
    the boundary member's FLUX corners, emits boundary-member SOURCE corners;
    the interior member is a zero source — "B_b touches the trace/bulk" is now
    UNSPELLABLE, Pattern 4). The system's boundary is the direct sum of
    per-system boundary blocks (RULING P1) — the two grid entries ``B_a`` at
    (A,A) and ``B_b`` at (B,B) of the within-group gain grid
    (:func:`orpheus.sn.coupled_system.build_within_group_system`).
    Unconstructable on a seedless mesh (System B does not exist there — the
    ctor guard mirrors ``A_BA``'s).

    The action is the ``(R, μ = ∓1)`` corner reflection that closes the ray's
    r = R boundary. The outer face law realizes on the trace carrier as an
    (ordinate ⊗ group) OPERATOR (``B_a``'s per-face law), but at the
    **off-quadrature** μ = ±1 ray it cannot act — so ``B_b`` applies the per-KIND
    specular FACT directly (RULING P1's "one law, two carriers": the
    carrier-indexed realizer that would produce both arms from one law is the
    named seam, built when DSA adds the 3rd carrier).

    Capabilities
    ------------

    ``apply`` always. ``apply_transpose`` iff the outer ray-face law is
    Euclidean-adjointable — reflective (involution) and vacuum are; white /
    albedo / periodic are the loud-deferred set (no ruled off-quadrature corner
    action). Per-leaf :attr:`is_adjointable` (NOT the whole-trace intersection —
    ``B_b`` has exactly one face, the outer radius).

    Adjoint metric (RULING P2)
    --------------------------

    ``B_b`` advertises the **Euclidean** :meth:`apply_transpose` only — no
    per-leaf ``.H``. The ray corner gauge is symmetric (``g₊ = g₋ = V(R)`` — both
    corners at r = R), so ``B_b.H = G_sd⁻¹ B_bᵀ G_sd = B_bᵀ``: the Euclidean
    transpose IS the ``G_sd = V_cell`` Hilbert adjoint. (``.H`` is realized ONCE
    at the composite via ``G⁺·apply_transpose·G``; a Euclidean block adjoint on
    System B is metric-correct here BECAUSE of the symmetric gauge — the
    ``G_sd``-reciprocity gate in ``test_psi_half_coupling.py`` pins the symmetry
    that keeps Mode-12 closed.)

    Parameters
    ----------
    sn_mesh : SNMesh
        The augmented geometry (seed-carrying — 1-D curvilinear). Carries the
        outer-face law ``sn_mesh.bc["xmax"]`` and the ray space
        (the split ψ½ spaces; the mesh-identity invariant of
        :class:`SNBoundaryOperator` applies here too).
    """

    block_role = BlockRole.BOUNDARY
    # B_b is System B's boundary — it acts within the ray system alone.
    # Since the B.2b re-type that is STRUCTURAL, not padding: the carrier
    # has no System-A bulk or trace slots to present-zero (see the class
    # docstring's Pattern-4 note).  Campaign step 4a.
    system_role = SystemRole.B

    def __init__(
        self,
        field_space: "FullFieldSpace | None",
        outer_law: "BoundaryTraceLaw",
    ) -> None:
        if field_space is None:
            raise ValueError(
                "RadialCharacteristicBoundaryOperator: the pose carries no "
                "ψ½ ray (radial_characteristic_field_space is None) — a "
                "seedless mesh (a Cartesian chart, or a curvilinear rule with "
                "no carrying level, R12a) has no System B, hence no "
                "ray-corner boundary block. B_b "
                "exists only on a seed-carrying mesh — the GL sphere, the "
                "σ_y-folded cylinder (Q5.6)."
            )
        #: System B's member composite — endomorphic domain/codomain.
        self._field_space = field_space
        #: The outer-radius face's realized boundary law (a carrying mesh is
        #: 1-D curvilinear: exactly ONE boundary face — ``xmax`` — carries
        #: it). Bound at construction (un-weld arc O-1).
        self._outer_law = outer_law

    @property
    def is_adjointable(self) -> bool:
        # Per-leaf (NOT B_a's whole-trace intersection): B_b has exactly one
        # face, the outer radius. Reflective (involution) + vacuum are
        # Euclidean-adjointable; white / albedo / periodic are the loud-deferred
        # set (:meth:`_reflect_corner` raises — no ruled off-quadrature corner
        # action → no transpose). is_invertible inherits base False. (The old
        # seedless defensive arm is dead under the ctor guard — retired.)
        #
        # ONE source of truth with ``_reflect_corner``'s guard: the transpose
        # exists exactly when the forward corner action does.
        return _has_ruled_corner_action(self._outer_law)

    @property
    def domain(self) -> "FunctionSpace":
        # System B's own member space (B.2b DP1; non-None by the ctor guard).
        # The B.2c CoupledOperator grid type-checks the (B, B) placement
        # against it: ``build_within_group_system`` composes ``A_BB = march −
        # B_b`` there.  There is no FullField-summed ``B = B_a + B_b`` and no
        # adapter to carry one — the transient ``_RayEmissionFullFieldGain``
        # was RETIRED at B.2d, the driver iterate is a ``CoupledField`` pair,
        # and nothing sums FullField-embedded ray gains anymore.
        return self._field_space

    @property
    def codomain(self) -> "FunctionSpace":
        return self._field_space

    def _reflect_corner(
        self, seed: "RadialCharacteristicBoundaryField", method: str,
    ) -> "RadialCharacteristicBoundarySourceSink":
        r"""The ``A_ss`` CORNER action on System B's boundary member (R13, 2.5d).

        The (R, μ = ∓1) corner pair closes the ray boundary on a seed-carrying
        mesh: the inward seed leg's r = R inflow is BC data, and for a
        specular-reflective outer face the reflected partner of the outward ray
        μ = +1 is EXACTLY the inward one μ = −1 (its own mirror — an
        off-quadrature ray, so the per-face law OPERATOR cannot act on it; the
        specular fact is applied directly). Forward:
        ``out.corner(level, −1) = ψ½.corner(level, +1)`` per carried level;
        Euclidean transpose: ``out.corner(level, +1) = χ̄.corner(level, −1)``.
        The opposite corners stay ZERO (``B_b`` touches only the inflow row /
        its transpose image — the exact ``_reflect_trace`` projection
        discipline); since the B.2b re-type the input IS the boundary member
        (the cells never enter — structural, not zeroed).

        Law dispatch — on the law's own **affine factors** (:math:`R`,
        :math:`G`, :math:`q`), NOT on the realized operator's composition tree
        (which is an (ordinate ⊗ group) operator over the QUADRATURE rows and
        structurally cannot act on the off-quadrature μ = ±1 ray):

        * :math:`G` permutes ordinates (a specular mirror) — the corner swap
          above; the mirror of μ = +1 is exactly μ = −1, so the pairing is
          exact off-quadrature.
        * otherwise :math:`R = 0` (vacuum) — no corner emission, the block
          stays all-zero.
        * anything else (white / albedo / periodic / a prescribed source) —
          **loud-deferred** (:class:`NotImplementedError`) per the 2.5d
          plan-of-record; see :func:`_has_ruled_corner_action` for which factor
          disqualifies each. (E.g. white re-emission at the off-quadrature ray
          needs the ``|Ω·n|``-weighted outflow average for μ = −1, not yet
          ruled.)

        Until campaign phase B2 this dispatched on the ``kind`` STRING the
        pre-B2.0 shim carried.

        .. warning::

           The swap is **unscaled** — it does not multiply by :math:`R`. That
           is exact for the α = 1 reflector every BC tag can declare
           (``_law_from_tag`` hard-codes ``albedo=1.0`` for reflective), and
           WRONG for a directly-constructed partially-reflecting law, which
           would re-emit its full outflow at the corner. The defect predates
           this phase — the tag set admitted every albedo too, since
           ``ReflectiveBoundary.key`` is ``"reflective"`` regardless — and B2
           preserved it deliberately rather than fold a physics fix into a
           repoint. It closes when B4 composes :math:`R \circ G` here.
        """
        from orpheus.transport.source_sinks import (
            RadialCharacteristicBoundarySourceSink,
        )

        law = self._outer_law
        if not _has_ruled_corner_action(law):  # single source with is_adjointable
            raise NotImplementedError(
                f"RadialCharacteristicBoundaryOperator: the outer-face law "
                f"{type(law).__name__} (G={type(law.geometry_map).__name__}, "
                f"R={law.response_kernel.amplitude}) has no ruled corner action "
                f"yet (white / albedo / periodic / a prescribed source at the "
                f"off-quadrature μ = ±1 ray — loud-deferred, 2.5d "
                f"plan-of-record)."
            )
        out = RadialCharacteristicBoundarySourceSink.zeros(seed.space)
        # R = 0 ⇒ zero corner emission (the all-zero ``out`` falls through);
        # G permutes ⇒ the specular swap (the mirror of μ = +1 is exactly μ = −1).
        if law_permutes_ordinates(law):
            for level in seed.levels:
                if method == "apply":
                    out.corner(level, -1)[...] = seed.corner(level, +1)
                else:  # apply_transpose — the Euclidean mirror image
                    out.corner(level, +1)[...] = seed.corner(level, -1)
        return out

    def _apply_faces(
        self, ray: "RadialCharacteristicField", method: str,
    ) -> "RadialCharacteristicField":
        r"""``B_b`` on System B's own carrier: the ray-corner action
        :meth:`_reflect_corner` on the boundary member, a zero-source interior.

        Since the B.2b re-type there is NO bulk/trace padding — the composite
        has no such slots (Pattern 4: "B_b touches the trace" is unspellable).
        The interior member is a zero SOURCE (``B_b`` writes only the corner);
        the production driver consumes this block natively at the (B,B) slot
        of the within-group gain grid (B.2d).
        """
        from orpheus.transport.fields.radial_characteristic_boundary_flux import (
            RadialCharacteristicBoundaryFlux,
        )
        from orpheus.transport.radial_characteristic_field import (
            RadialCharacteristicField,
        )
        from orpheus.transport.source_sinks import (
            RadialCharacteristicInteriorSourceSink,
        )

        # The shared System-B block-boundary parse (carrier class +
        # space-content — one parse body across A_BB / A_AB / B_b, B.2c).
        RadialCharacteristicField.require_member(
            ray,
            space=self._field_space,
            context=f"RadialCharacteristicBoundaryOperator.{method}",
        )
        # Role parse at the block boundary (the #289-F2 discipline, relocated
        # from the erased FullField slot to the boundary MEMBER — the composite
        # slots are role-erased): ``B_b`` reflects a FLUX corner. A source-role
        # member arriving here is a caller error worth raising loudly.
        if not isinstance(ray.boundary, RadialCharacteristicBoundaryFlux):
            raise TypeError(
                f"RadialCharacteristicBoundaryOperator: the input composite's "
                f"boundary must be a RadialCharacteristicBoundaryFlux corner; "
                f"got {type(ray.boundary).__name__}."
            )
        return RadialCharacteristicField(
            # The zero cells block of the emission rides the PARSED member's
            # own space — require_member above guarantees it content-equal
            # to this operator mesh's ψ½ interior space (and non-None).
            interior=RadialCharacteristicInteriorSourceSink.zeros(
                ray.interior.space,
            ),
            boundary=self._reflect_corner(ray.boundary, method),
        )

    def apply(self, ray: "RadialCharacteristicField") -> "RadialCharacteristicField":
        r"""Forward action ``B_b·ψ½`` — the ray-corner reflection on System B."""
        return self._apply_faces(ray, "apply")

    def apply_transpose(
        self, ray: "RadialCharacteristicField",
    ) -> "RadialCharacteristicField":
        r"""Euclidean transpose ``B_bᵀ·ψ½`` — the mirror-image corner swap.

        Reachable iff the outer ray-face law is adjointable (see
        :attr:`is_adjointable`). Euclidean = the ``G_sd = V_cell`` Hilbert adjoint
        because the corner gauge is symmetric (RULING P2).
        """
        return self._apply_faces(ray, "apply_transpose")


class SNMaskedBoundaryOperator(LinearOperator["FullField", "FullField"]):
    r"""One half of the schedule split ``B = B_lower + B_upper`` — the
    whole-trace :class:`SNBoundaryOperator` restricted to a per-face set of
    inflow ordinate ROWS (#226 §17 W2).

    The restriction composes a row projection with ``B``'s codomain
    projection: per face, only the given ordinate rows of the reflected
    inflow are emitted; every other slot (and the bulk, as for ``B``) is
    zero.  Which rows belong to which half is SCHEDULE-order semantics
    (:meth:`~orpheus.sn.loss_representation.sweep_schedule.SweepSchedule.lower_inflow_rows`),
    so the instance carries its :attr:`schedule` — the reified
    ``M = (L+C−B_lower)`` reads the walk order off its lower operand rather
    than pairing a foreign schedule with a mismatched mask.  Construct via
    :meth:`SNBoundaryOperator.split` (the named pair keeps lower/upper from
    swapping silently); the exactness of the partition is pinned by the
    W2-split gate.

    A masked half is NOT invertible and does not advertise a transpose
    (``B_lowerᵀ`` masks input rows, not output rows — mint it when the
    adjoint-inverse carve #280 produces a consumer), so it is apply-only
    and the two-axis contract holds by the base defaults.
    """

    block_role = BlockRole.BOUNDARY

    def __init__(
        self,
        inner: "SNBoundaryOperator",
        rows: "Mapping[str, np.ndarray]",
        schedule: "SweepSchedule",
    ) -> None:
        #: The whole-trace boundary law this is a row restriction of.
        self.inner = inner
        #: Per-face inflow ordinate rows this half emits (global ordinate
        #: indices into each face's ``(N, …)`` trace slot).
        self.rows = rows
        #: The octant-order schedule the row split was derived from.
        self.schedule = schedule

    @property
    def sn_mesh(self) -> "SNMesh":
        return self.inner.sn_mesh

    @property
    def domain(self) -> "FunctionSpace":
        return self.inner.domain

    @property
    def codomain(self) -> "FunctionSpace":
        return self.inner.codomain

    def apply(self, psi: "FullField") -> "FullField":
        r"""``B_half·ψ`` — the per-face law projected onto :attr:`rows`, zero bulk."""
        return self.inner._apply_faces(psi, "apply", rows=self.rows)

    def reflect_rows_inplace(
        self, boundary_flux: "AngularBoundaryFlux", faces: "Iterable[str]",
    ) -> None:
        r"""In place, ADDITIVE, on :attr:`rows` only:
        ``bf[f][rows] += (B·bf)[f][rows]`` for each given face.

        The inter-group row update of the reified forward substitution
        (#226 §17 W2): solving :math:`M z = y` on a strictly-lower inflow
        row reads :math:`z_{\rm in} = y_{\rm row} + (B z)_{\rm row}` — the
        buffer already holds the seed :math:`y_{\rm row}`, so ACCUMULATING the
        fresh reflection completes the inhomogeneous row exactly.  This is what
        makes ``M.inverse()`` exact for arbitrary data on the INFLOW rows (the
        source subspace ``{y : y.outflow-rows = 0}``), not merely on
        production's zero-lower-inflow-row subspace; restricting to
        :attr:`rows` leaves the upper (lagged) rows carrying the seed the
        splitting :math:`\psi_{k+1} = M^{-1}(q + B_{\rm upper}\psi_k)` says
        they carry.  ⚠ Additive, NOT whole-face overwrite: the dissolved
        resolvent's OVERWRITE dropped :math:`y_{\rm row}` — benign in
        production (zero on a reflective face) but O(1)-wrong as an inverse,
        and it stamped fresh values onto rows the iterate defines as lagged.

        Contrast :meth:`SNBoundaryOperator.reflect_inflow_inplace` — the
        whole-face ASSIGNMENT ``ψ.inflow ← B·ψ.outflow`` between BARE sweeps,
        the right semantics where the inflow is wholly recomputed each sweep
        rather than a solved unknown of a linear row (the sweep-tier gates;
        no production driver since #448).
        """
        selected = {
            face: self.rows[face]
            for face in faces
            if face in self.rows and np.asarray(self.rows[face]).size
        }
        if not selected:
            return
        reflected = self.inner._reflect_trace(
            boundary_flux, "apply", faces=tuple(selected), rows=selected,
        )
        for face, rows in selected.items():
            boundary_flux.face_view(face)[rows] += (
                reflected.face_view(face)[rows]
            )

    def __repr__(self) -> str:
        n_rows = sum(int(np.asarray(r).size) for r in self.rows.values())
        return (
            f"SNMaskedBoundaryOperator({self.inner!r}, "
            f"rows={n_rows} over {len(self.rows)} faces, "
            f"schedule={self.schedule.kind!r})"
        )


class BoundarySplit(NamedTuple):
    """The named ``B = B_lower + B_upper`` pair from :meth:`SNBoundaryOperator.split`."""

    lower: SNMaskedBoundaryOperator
    upper: SNMaskedBoundaryOperator
