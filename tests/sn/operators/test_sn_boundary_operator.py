r"""Wave O (Issue #208) step O.4a.2 Commit 1 — the whole-trace boundary
operator ``B`` (:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`).

``B`` is the ``A_ss`` block of the SN algebra ``(L_full + C − S − F − B)``: a
BOUNDARY-block leaf on the :class:`TimedFullField` carrier that applies each
true boundary face's realized law to that face's trace slot, with zero bulk
action. These foundation tests pin the assembly BEFORE anything consumes ``B``
(the ``−B`` wiring + the bare-``L_full`` flip is O.4a.2 Commit 2):

* the role / domain / predicate contract;
* zero bulk action;
* **per-face wiring** — ``B`` applies the RIGHT face's law to the RIGHT slot,
  emitting on the **inflow row only** (``B`` is the ``A_ss`` block
  ``V_outflow → V_inflow``; the discriminating case uses asymmetric BCs so a
  face↔face swap is caught);
* **block-diagonal over faces** — a single-face perturbation stays on that face;
* the ``is_adjointable`` face conjunction (True iff every face law honours the
  transpose; see the stub negative, which supplies an apply-only law because no
  SHIPPED law drops the predicate any more — corrected 2026-08-06, see below).

⚠ **Correction (2026-08-06).** This header read "white would drop it" from
``d7e13164`` (2026-06-03) until B3.4b landed the Lambertian's re-emission
closure. `[M]` a ``WhiteBoundary()`` face now reports ``is_adjointable = True``
and ``B.apply_transpose`` returns, so the illustration was present-tense false
and would have told a reader that a working configuration was unsupported. The
CLAIM the conjunction makes is unchanged and still gated — only its example was
stale, which is why the negative below uses an explicit stub rather than naming
a shipped law it can no longer name.

Campaign phase B3.2 — the re-posed C-1 gates (RG-1 … RG-5)
==========================================================

B3.2 narrowed the realized SN boundary law from ``full-face → full-face`` to
:math:`\Gamma_+ \to \Gamma_-`, so ``B``'s per-face action is the composition
``ι₋ ∘ law ∘ γ₊`` and nothing is computed and then discarded. Constraint
**C-1** of the boundary-machinery review requires the gates that asserted the
OLD (discard) contract to be **re-posed to state the new one**, never deleted
and never weakened. What each one became, and what still reddens it:

============  ========================================================
gate          the narrowed contract it states
============  ========================================================
**RG-1**      ``B``'s inflow rows ARE ``law(γ₊ψ)`` — the law was handed
              its honest domain (replaces the leg whose reference fed
              the law a full face)
**RG-2**      ``B`` emits NOTHING outside :math:`\Gamma_-`. Widened
              from the old ``got[outflow]`` leg: the face slot is a
              THREE-way split ``I ⊔ O ⊔ T`` and the old leg left the
              **tangential** rows unguarded (measured: 4 of 8 on the
              ``cyl_reflective`` fixture)
**RG-2b**     …and it emits SOMETHING on :math:`\Gamma_-`, so RG-2 is
              not satisfied by the zero operator
**RG-3**      the realized law's SHAPE contract, plus the leg that
              distinguishes a narrowed law from an endomorphism
**RG-4**      ``Bᵀ`` emits nothing outside :math:`\Gamma_+`
**RG-5**      the schedule split partitions ``B`` on a **2-D** mesh —
              the only regime that activates B3.2's new
              ``sel → position-within-Γ₋`` remap
============  ========================================================

The bug the old leg was written for — *the law's outflow image leaking into
the output* — is now **unspellable** (the law has no outflow image; its
codomain IS :math:`\Gamma_-`). That is strictly better than a red, and it is
why the re-posed legs pin the WRITE TARGET and the DOMAIN instead. The
mutations that redden each leg are named in its docstring.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from orpheus.geometry import (
    BC, Mesh1D, Mesh2D, Region, RegionMesh, StructuredGeometry,
)
from orpheus.geometry.boundary import PeriodicBoundary
from orpheus.numerics.operator import (
    BlockRole,
    BoundaryOperator,
    BulkOperator,
    FullOperator,
    IncompatibleOperatorComposition,
    LinearOperator,
    MissingAdjoint,
)
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.operators.boundary import SNBoundaryOperator
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.source_sinks import AngularBoundarySourceSink
from orpheus.transport.timed_full_field import TimedFullField
from tests.sn._test_helpers import placeholder_materials

pytestmark = [pytest.mark.foundation]


def _sn(geometry: str, bcs: tuple, nx: int = 4, ng: int = 1) -> SNMesh:
    geom = StructuredGeometry(
        geometry=geometry,
        regions=(Region(mat_id=0, outer_thickness_cm=2.0),),
        bcs=bcs,
    )
    mesh = Mesh1D.from_geometry(geom, region_meshes=(RegionMesh(n_cells=nx),))
    # Cylinder's angular redistribution needs a level-structured quadrature;
    # slab / sphere accept the 1-D Gauss–Legendre set.
    quad = (
        Quadrature.folded_product(n_mu=2, n_phi=4)
        if geometry == "CYL"
        else Quadrature.gauss_legendre(n_ordinates=4)
    )
    return SNMesh(mesh, quad, placeholder_materials(ng=ng))


def _random_state(sn: SNMesh, seed: int = 7) -> TimedFullField:
    rng = np.random.default_rng(seed)
    z = TimedFullField.zeros(interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn.full_field_space)
    return replace(
        z,
        interior=replace(z.interior, values=rng.uniform(0.5, 2.0, size=z.interior.values.shape)),
        boundary=replace(
            z.boundary, values=rng.uniform(0.5, 2.0, size=z.boundary.values.shape),
        ),
    )


# Geometry × BC cases reachable through SNMesh (1-D faces support only
# reflective / vacuum). Slab uses ASYMMETRIC BCs so the per-face wiring test
# discriminates a face↔face swap.
_CASES = {
    "slab_vacuum_reflective": ("SLB", (BC.vacuum, BC.reflective)),
    "slab_reflective_reflective": ("SLB", (BC.reflective, BC.reflective)),
    "sphere_reflective": ("SPH", (BC.reflective,)),
    # MANDATORY (B3.2 / RG-2): the ONLY fixture here carrying TANGENTIAL
    # ordinates — ``product(2, 4)`` puts 4 of its 8 ordinates at
    # ``|Ω·n| ≤ ε`` on every face, so it is the sole discriminator between
    # "outside the outflow rows" and "outside Γ₋". Do not swap it for a
    # Gauss–Legendre set to make the fixture cheaper.
    "cyl_reflective": ("CYL", (BC.reflective,)),
}


def _sn_2d(nx: int = 4, ny: int = 4, ng: int = 2) -> SNMesh:
    r"""A 2-D reflective Cartesian ``SNMesh`` — RG-5's mandatory fixture.

    B3.2 introduced a NEW piece of index arithmetic on the schedule-split
    path: the requested rows are a subset of :math:`\Gamma_-`, so they must be
    remapped to POSITIONS within it (``Γ₋(f).to_local(sel)`` — the half-trace
    SPACE's remap since G6.5). In 1-D the
    schedule hands each face entirely to one half, so the requested rows ARE a
    prefix of the inflow set and the naive ``arange(sel.size)`` is *exactly
    correct* — the whole 1-D suite is blind. In 2-D the lower-half rows
    interleave. Hence a 2-D mesh, and hence ``ng=2`` and a non-square-free
    trailing shape so an axis mix-up cannot hide either.
    """
    mesh = Mesh2D(
        edges_x=np.linspace(0.0, 1.0, nx + 1),
        edges_y=np.linspace(0.0, 1.0, ny + 1),
        mat_map=np.zeros((ny, nx), dtype=int),
        bc_xmin=BC.reflective, bc_xmax=BC.reflective,
        bc_ymin=BC.reflective, bc_ymax=BC.reflective,
    )
    return SNMesh(mesh, Quadrature.level_symmetric(4), placeholder_materials(ng=ng))


def _gather(face_slot: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``γ_S`` by hand — plain fancy indexing, no ORPHEUS operator.

    The re-posed gates must not take their reference from the very primitive
    the production path composes (``TraceRestrictionOperator``), or the gate
    and the code share an upstream and the cross-check is procedural rather
    than structural. Two lines of numpy is the independent spelling.
    """
    return face_slot[rows]


def _scatter(rows: np.ndarray, values: np.ndarray, n_face: int) -> np.ndarray:
    """``ι_S`` by hand — zeros, then write the selected rows."""
    out = np.zeros((n_face,) + values.shape[1:], dtype=values.dtype)
    out[rows] = values
    return out


class TestContract:
    def test_block_role_is_boundary_and_exclusive(self) -> None:
        B = SNBoundaryOperator(_sn("SLB", (BC.vacuum, BC.reflective)))
        assert B.block_role is BlockRole.BOUNDARY
        assert isinstance(B, BoundaryOperator)
        assert not isinstance(B, BulkOperator)
        assert not isinstance(B, FullOperator)

    def test_domain_and_codomain_are_the_full_field_space(self) -> None:
        # Wave O / O.2b R5: ``B`` is an endomorphism on the composite
        # carrier (bulk ⊕ trace) — ``B.apply`` consumes / emits a full
        # ``TimedFullField`` (zero bulk + reflected trace), so it advertises
        # ``sn.full_field_space``, the SAME space L/C/S/F report (so the
        # OperatorSum guard accepts ``L + C - S - F - B``). The trace metric
        # ``B.H`` reads lives on the composite's trace block.
        sn = _sn("SLB", (BC.vacuum, BC.reflective))
        B = SNBoundaryOperator(sn)
        assert B.domain is sn.full_field_space
        assert B.codomain is sn.full_field_space
        # the composite trace block IS the mesh trace space (block identity)
        assert B.domain.trace_space is sn.angular_trace


class TestApply:
    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_apply_is_zero_bulk(self, case_id: str) -> None:
        """``B`` is ``A_ss`` only — no bulk action."""
        sn = _sn(*_CASES[case_id])
        out = SNBoundaryOperator(sn).apply(_random_state(sn))
        assert not out.interior.values.any()
        # B.5.2: B.apply emits B·ψ.outflow — the operator output is Aψ (a
        # source/sink), NOT a residual.  Its boundary is the source/sink role
        # leaf (mirrors the bulk's AngularSourceSink); the residual only arises
        # from from_balance, never straight off the operator output.
        assert isinstance(out.boundary, AngularBoundarySourceSink)

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_apply_per_face_equals_law_of_the_outflow_half_trace(
        self, case_id: str,
    ) -> None:
        r"""**RG-1** — per-face wiring: ``B``'s inflow rows ARE ``law(γ₊ψ)``.

        ``B`` is the ``A_ss`` block :math:`V_{\rm outflow} \to V_{\rm inflow}`.
        Since B3.2 the realized law is typed :math:`\Gamma_+ \to \Gamma_-`, so
        the wiring claim is sharper than it was: ``B`` must hand the law
        **exactly the outflow half-trace** and write the law's whole image on
        the inflow rows. There is no slicing left to check — the discard is
        gone.

        RE-POSED FROM (C-1): the pre-B3.2 leg asserted
        ``got[inflow] == bc.apply(WHOLE face slot)[inflow]``. That reference is
        the retired representation: it feeds a full face to a
        :math:`\Gamma_+`-domain operator. Under B3.2 it does not merely fail —
        it does not RUN (``IndexError``, or a silently truncated gather), which
        is the domain-narrowing signature: the old gate stops executing rather
        than going tautological.

        Reddens on (each measured, §2 of ``scratch/b3_2_migration.md``):
        handing the law the WRONG half (``γ₋`` for ``γ₊``); handing it the FULL
        face; writing the image to the outflow rows; a face↔face swap (the
        asymmetric-BC slab param is what makes that observable — vv failure
        mode #5).
        """
        sn = _sn(*_CASES[case_id])
        psi = _random_state(sn)
        out = SNBoundaryOperator(sn).apply(psi)
        for face in sn.angular_trace.layout.faces:
            bc = sn.bc[face]
            inflow = sn.angular_trace.inflow_indices_for_face(face)
            outflow = sn.angular_trace.outflow_indices_for_face(face)
            got = out.boundary.face_view(face)
            gamma_plus = _gather(psi.boundary.face_view(face), outflow)
            np.testing.assert_array_equal(
                got[inflow], bc.apply(gamma_plus),
                err_msg=(
                    f"{case_id} face {face!r}: B's inflow rows are not "
                    f"law(γ₊ψ) — the law was handed the wrong domain, or its "
                    f"image was written to the wrong rows."
                ),
            )

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_apply_emits_nothing_outside_the_inflow_trace(
        self, case_id: str,
    ) -> None:
        r"""**RG-2** — ``B``'s codomain IS :math:`\Gamma_-`, stated as a
        write-target contract.

        RE-POSED FROM (C-1): the pre-B3.2 leg was
        ``assert not got[outflow].any()``. It survives B3.2 syntactically and
        can still red — but it carries a **measured hole**: the face slot is a
        THREE-way partition :math:`I_f \sqcup O_f \sqcup T_f`, and the old leg
        left the tangential rows completely unguarded. On the
        ``cyl_reflective`` fixture 4 of 8 ordinates are tangential, and the
        pre-B3.2 code's law image on them was non-zero (1.846) and silently
        discarded. So the leg is not weakened here — it is **widened**, from
        ``outflow`` to ``complement(inflow)``, which costs nothing and closes
        the hole.

        ``pytest.fail`` rather than a bare ``assert``: the canonical invocation
        is ``python -O`` and this predicate must survive a future move into a
        non-collected helper module (vv Mode 8).

        Reddens on: writing the image to the outflow rows; passing ``γ₊``
        through to the output's outflow rows; ANY leak onto the tangential rows
        (the leg the old spelling could not see).
        """
        sn = _sn(*_CASES[case_id])
        psi = _random_state(sn)
        out = SNBoundaryOperator(sn).apply(psi)
        for face in sn.angular_trace.layout.faces:
            got = out.boundary.face_view(face)
            inflow = sn.angular_trace.inflow_indices_for_face(face)
            off_codomain = np.setdiff1d(np.arange(got.shape[0]), inflow)
            peak = float(np.abs(got[off_codomain]).max()) if off_codomain.size else 0.0
            if peak != 0.0:
                pytest.fail(
                    f"{case_id} face {face!r}: B emitted {peak:.3e} OUTSIDE "
                    f"its codomain Γ₋ (rows {off_codomain.tolist()}) — the "
                    f"A_ss block is not Γ₊ → Γ₋. Note the face is a THREE-way "
                    f"split: inflow={inflow.tolist()}, and the complement "
                    f"carries both outflow AND tangential ordinates."
                )

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_apply_actually_emits_on_the_inflow_trace(
        self, case_id: str,
    ) -> None:
        r"""**RG-2b** — the non-vacuity control RG-2 cannot ship without.

        RG-2 (``nothing outside Γ₋``) is satisfied perfectly by the ZERO
        operator, so on its own it is a gate that a total regression would
        pass. This leg pins that ``B`` puts something on :math:`\Gamma_-`.

        Summed over faces, not per-face: a **vacuum** face's law IS the zero
        map ``Γ₊ → Γ₋`` (that is its entire content since B3.2), so
        ``slab_vacuum_reflective``'s xmin legitimately contributes nothing.
        Every fixture here carries at least one reflective face.

        Reddens on: writing the law's image to the outflow rows instead (Γ₋
        goes empty) — i.e. exactly the mutation RG-1 catches, caught here from
        the opposite side.
        """
        sn = _sn(*_CASES[case_id])
        psi = _random_state(sn)
        out = SNBoundaryOperator(sn).apply(psi)
        live = sum(
            int(np.count_nonzero(
                out.boundary.face_view(face)[
                    sn.angular_trace.inflow_indices_for_face(face)
                ]
            ))
            for face in sn.angular_trace.layout.faces
        )
        if live == 0:
            pytest.fail(
                f"{case_id}: B emitted NOTHING anywhere on Γ₋ — RG-2 would be "
                f"vacuously satisfied by the zero operator, so this fixture "
                f"proves nothing about the codomain."
            )

    def test_block_diagonal_no_face_mixing(self) -> None:
        """A perturbation on ONE face's input slot affects ONLY that face's
        output (``B`` is block-diagonal over faces — it never mixes faces)."""
        sn = _sn("SLB", (BC.reflective, BC.reflective))
        B = SNBoundaryOperator(sn)
        psi = _random_state(sn, seed=1)
        other = _random_state(sn, seed=2)
        # psi3 = psi with ONLY the xmin slot replaced (xmax slot identical to psi).
        b3 = replace(psi.boundary, values=psi.boundary.values.copy())
        psi3 = replace(psi, boundary=b3)
        psi3.boundary.face_view("xmin")[:] = other.boundary.face_view("xmin")

        out = B.apply(psi)
        out3 = B.apply(psi3)
        # xmax output unchanged — it depends only on the (identical) xmax input.
        np.testing.assert_array_equal(
            out.boundary.face_view("xmax"),
            out3.boundary.face_view("xmax"),
        )
        # Sanity: the xmin perturbation actually changed the xmin output (else
        # the block-diagonal claim would be vacuous).
        assert not np.array_equal(
            out.boundary.face_view("xmin"),
            out3.boundary.face_view("xmin"),
        )


class TestApplyTransposeCapability:
    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_adjointable_when_all_faces_support(
        self, case_id: str,
    ) -> None:
        """Reflective / vacuum faces are all adjointable, so ``B`` is."""
        sn = _sn(*_CASES[case_id])
        B = SNBoundaryOperator(sn)
        assert B.is_adjointable
        # The Euclidean transpose of the row-projected forward ``B_face =
        # P_inflow ∘ law`` is ``B_faceᵀ = lawᵀ ∘ P_inflow`` — the per-face law
        # transpose applied to the INFLOW-masked input, full image written.
        # (B.2d d3 rewire: the previous spelling asserted ``got[outflow] ==
        # lawᵀ(ψ_face)[outflow]``, which on a VACUUM face pinned the law
        # object's spurious identity-on-outflow diagonal — the masked-regime
        # snapshot of vv anti-pattern #12.  The honest vacuum transpose is
        # ZERO; see psi_half ``test_b_a_vacuum_transpose_is_the_honest_zero``.)
        psi = _random_state(sn)
        out = B.apply_transpose(psi)
        for face in sn.angular_trace.layout.faces:
            bc = sn.bc[face]
            inflow = sn.angular_trace.inflow_indices_for_face(face)
            outflow = sn.angular_trace.outflow_indices_for_face(face)
            got = out.boundary.face_view(face)
            face_in = psi.boundary.face_view(face)
            # B3.2: the forward is ``ι₋ ∘ law ∘ γ₊``, so the Euclidean
            # transpose distributes as ``ι₊ ∘ lawᵀ ∘ γ₋`` — RESTRICT the input
            # to the forward's codomain, transpose the law, SCATTER over Γ₊.
            # (Pre-B3.2 the reference built a full-face ``masked`` array and
            # took the law's full transpose image; that is the retired
            # representation — it feeds a Γ₋-domain operator a full face.)
            expected = _scatter(
                outflow,
                bc.apply_transpose(_gather(face_in, inflow)),
                got.shape[0],
            )
            np.testing.assert_array_equal(
                got, expected,
                err_msg=(
                    f"{case_id} face {face!r}: Bᵀ ≠ ι₊ ∘ lawᵀ ∘ γ₋ — the "
                    f"Euclidean transpose of the narrowed A_ss block."
                ),
            )

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_transpose_emits_nothing_outside_the_outflow_trace(
        self, case_id: str,
    ) -> None:
        r"""**RG-4** — ``Bᵀ``'s codomain is :math:`\Gamma_+`, and nothing else.

        The mirror of RG-2 on the transpose. It is the ONLY gate that pins the
        narrowing of the transpose's write target: pre-B3.2 ``apply_transpose``
        wrote the law's image over the WHOLE face slot, and on every reachable
        law that wide write is **value-invisible** (the extra rows were zero
        anyway), so no value gate anywhere can see it. Without this leg, "the
        transpose's codomain narrowed" is an unverifiable claim.

        Its own non-vacuity control is the sibling above: RG-4 alone would be
        satisfied by ``Bᵀ ≡ 0``, and
        :meth:`test_adjointable_when_all_faces_support` is the equality that
        rules that out (every reflective face's transpose image is non-zero).
        Neither leg ships without the other.

        Reddens on: the transpose writing onto the inflow (or tangential) rows.
        """
        sn = _sn(*_CASES[case_id])
        B = SNBoundaryOperator(sn)
        out = B.apply_transpose(_random_state(sn))
        for face in sn.angular_trace.layout.faces:
            got = out.boundary.face_view(face)
            outflow = sn.angular_trace.outflow_indices_for_face(face)
            off = np.setdiff1d(np.arange(got.shape[0]), outflow)
            peak = float(np.abs(got[off]).max()) if off.size else 0.0
            if peak != 0.0:
                pytest.fail(
                    f"{case_id} face {face!r}: Bᵀ emitted {peak:.3e} OUTSIDE "
                    f"its codomain Γ₊ (rows {off.tolist()}) — the transpose of "
                    f"a Γ₊ → Γ₋ block lands on Γ₊ alone."
                )

    def test_adjointability_drops_when_a_face_lacks_it(self) -> None:
        """The predicate is a face CONJUNCTION — if any face law cannot
        transpose, ``B`` must NOT be adjointable (vv L11 negative; prevents a
        silent wrong/raising adjoint in a Krylov adjoint solve).

        ⚠ The stub is not a stand-in for a shipped law. This docstring named
        "e.g. the white BC" until 2026-08-06; `[M]` white reports
        ``is_adjointable = True`` since B3.4b closed the Lambertian's
        re-emission, and **every** shipped law now honours the transpose (the
        P5 rows in ``test_g_adjoint_reciprocity`` exercise a declared
        prescribed + white slab). So the conjunction's False arm is reachable
        only through a duck-typed law like the one below — which is exactly
        why the negative must construct one instead of picking a law off the
        shelf, and why it stays even though nothing shipped trips it today.
        """

        sn = _sn("SLB", (BC.vacuum, BC.reflective))
        _n_inflow = sn.angular_trace.inflow_indices_for_face("xmin").size

        class _NoTransposeLaw(LinearOperator):
            # Honest per-axis predicates (the caps frozenset retired with
            # carve P4): apply-only — neither axis available.
            #
            # ⚠ Subclasses :class:`LinearOperator` since **G6.3 step 8**, and
            # that is a real contract narrowing, not test scaffolding. Before
            # step 8 ``_reflect_trace`` only ever called ``law.apply``, so a
            # bare duck-type with an ``apply`` was a sufficient surrogate; the
            # face action is now the COMPOSITION ``law @ γ₊``, and a thing that
            # is not a morphism cannot be composed (`[M]` the bare stub raised
            # ``TypeError: unsupported operand type(s) for @``). Production is
            # unaffected — every ``_face_laws`` entry is realizer output and
            # therefore already an operator — so the honest fix is to make the
            # surrogate honour the contract it is standing in for, NOT to add
            # a guard for a case the type system now covers.
            # S4-amendment: the base DEMANDS an answer from every subclass; this
            # double is a deliberately-unbound probe, so it DECLARES the unbound
            # state instead of inheriting a silent default (which no longer exists).
            domain = None
            codomain = None
            is_adjointable = False
            is_invertible = False

            def apply(self, x):  # noqa: D401 - stub
                # B3.2 (C-1, the "fourth search"): a duck-typed surrogate is a
                # law, so it owes the law's CONTRACT — Γ₊ → Γ₋. The pre-B3.2
                # stub was ``return x``, a full-face identity; left alone it
                # would be a latent landmine, green only because the transpose
                # path raises before this body is reached. Grep-invisible to
                # the retirement audit, hence pinned here explicitly.
                return np.zeros((_n_inflow,) + np.asarray(x).shape[1:])

        class _BWithStubFace(SNBoundaryOperator):
            @property
            def _face_laws(self):
                laws = dict(super()._face_laws)
                laws[next(iter(laws))] = _NoTransposeLaw()
                return laws

        B = _BWithStubFace(sn)
        assert not B.is_adjointable
        # B.2d d3 bite-test: a caller bypassing the predicate hits the loud
        # per-face refusal, never a silent wrong transpose (the raise at
        # ``_reflect_trace``'s guarded ``adjointable(law)`` narrowing).
        with pytest.raises(MissingAdjoint, match="no Euclidean"):
            B.apply_transpose(_random_state(sn))


class TestTheFaceActionIsCOMPOSED:
    r"""⭐ **G6.3 step 8** — the face action is ``law @ γ₊``, so the
    composability check runs on the PRODUCTION path.

    Before step 8 :meth:`_reflect_trace` spelled the action as a sequence of
    ``.apply`` calls, which cannot check anything: the binding G6.3 put on the
    law and on ``γ±`` was honest metadata that nothing consulted (`[M]` 4941
    bindings across the suite, zero failures). Composing is what turns it into
    enforcement.

    The defect class it now refuses is **B3.4c's**: a law handed the wrong
    face's :math:`\Gamma_+`. That is invisible to a shape check —
    :math:`|\Gamma_+| = |\Gamma_-| = |\Gamma|/2` on every reachable face — and
    was MEASURED at 98 % relative error when it was live, so "it would look
    obviously wrong" is not a defence.

    ⚠ **Exactly ONE row here is a catcher for step 8**, and the flip-proof
    says so: reverting :meth:`_reflect_trace` to the sequential ``.apply``
    spelling reddens ``test_a_wrong_face_domain_map_now_RAISES_on_apply`` and
    **nothing else** (`[M]` 1 of 8). The other rows gate the *algebra* the
    catcher rests on — that the composition types correctly, that it is
    arithmetic-neutral, that the wrong restriction is refused, and how far the
    guarantee currently reaches. That is the honest split, not a coverage gap:
    asking "by what mechanism would THIS row see a change in
    ``_reflect_trace``?" answers "it wouldn't" for seven of them, and a row
    that cannot see the property must not be counted as guarding it
    (:ref:`verification-anti-patterns`, #18).
    """

    @staticmethod
    def _slab():
        return _sn("SLB", (BC.vacuum, BC.reflective))

    def test_a_wrong_face_domain_map_now_RAISES_on_apply(self) -> None:
        """⭐ The mutation, run as a gate — B3.4c re-injected at its source.

        Swapping ``_face_domains`` on a NON-periodic mesh is exactly the
        pre-B3.4c wiring, and it is the one mutation that this step's
        machinery exists to catch. Pre-step-8 it computed a plausible number;
        now the composition refuses to be formed at all.
        """
        sn = self._slab()

        class _BWithSwappedDomains(SNBoundaryOperator):
            @property
            def _face_domains(self):
                return {"xmin": "xmax", "xmax": "xmin"}

        with pytest.raises(IncompatibleOperatorComposition, match="A.domain"):
            _BWithSwappedDomains(sn).apply(_random_state(sn))

    def test_the_UNSWAPPED_map_composes(self) -> None:
        """The positive control (vv anti-#11): the same path, right wiring."""
        sn = self._slab()
        assert SNBoundaryOperator(sn)._face_domains == {
            "xmin": "xmin", "xmax": "xmax",
        }
        SNBoundaryOperator(sn).apply(_random_state(sn))  # MUST NOT raise

    def test_the_transpose_leg_reuses_the_SAME_composed_operator(self) -> None:
        r"""⭐ Why the ⚠ scatter trap stopped being a thing to remember.

        The forward is ``ι₋ ∘ law ∘ γ₊`` and its Euclidean transpose is
        ``ι₊ ∘ lawᵀ ∘ γ₋`` — the scatter must be over :math:`\Gamma_+`, and
        output-projecting onto :math:`\Gamma_-` instead extracts the DIAGONAL
        block (for vacuum, a spurious ``+1`` where the forward is ZERO). Both
        spellings are bit-identical for off-diagonal permutation laws, which
        is why every reflective fixture stayed green over the wrong one.

        Since step 8 one composed ``face_action`` serves both legs, and
        ``(law ∘ γ₊)ᵀ = γ₊ᵀ ∘ lawᵀ`` is the product's own transpose law, so
        there is no index left to choose wrongly. ⚠ **This row does not claim
        the trap is impossible** — a hand-written ``γ₋ᵀ(lawᵀ(·))`` would still
        run silently (the shapes agree). It claims the narrower, true thing:
        reaching for the wrong restriction *through the composition* is
        refused, so re-opening the trap now requires abandoning the composed
        form, which is a visible structural edit rather than a one-word slip.
        """
        sn = self._slab()
        trace = sn.angular_trace
        law = sn.bc["xmin"]
        # The right one composes...
        assert (law @ trace.outflow_restriction("xmin")).codomain is (
            trace.inflow_space("xmin")
        )
        # ...and the wrong one cannot be formed.
        with pytest.raises(IncompatibleOperatorComposition, match="A.domain"):
            _ = law @ trace.inflow_restriction("xmin")

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_composing_changed_no_ARITHMETIC(self, case_id) -> None:
        """`[M]` both legs bit-identical to the pre-step-8 sequential spelling.

        The reference is written out here as the three separate ``.apply``
        calls the method used before, so the two sides are genuinely different
        expressions of the same algebra rather than one calling the other.
        A one-time equivalence claim: step 8 is a re-spelling, and any drift
        would mean ``@`` is not function composition.
        """
        sn = _sn(*_CASES[case_id])
        trace = sn.angular_trace
        rng = np.random.default_rng(11)
        for face, law in ((f, sn.bc[f]) for f in trace.layout.faces):
            gamma_out = trace.outflow_restriction(face)
            gamma_in = trace.inflow_restriction(face)
            composed = law @ gamma_out

            face_in = rng.standard_normal(trace.layout.faces[face].shape)
            np.testing.assert_array_equal(
                composed.apply(face_in), law.apply(gamma_out.apply(face_in)),
            )
            y = rng.standard_normal(gamma_in.codomain.shape)
            np.testing.assert_array_equal(
                composed.apply_transpose(y),
                gamma_out.apply_transpose(law.apply_transpose(y)),
            )

    def test_PERIODIC_the_law_the_check_was_designed_for_is_policed(
        self,
    ) -> None:
        r"""⭐ The composability check is LIVE on the off-diagonal block —
        step 7's prize, and the row that was this suite's transitional pin.

        Periodic is the ONLY law whose domain face differs from its
        installation face — the only off-diagonal block of ``B`` — which
        makes it exactly the law the composed face action was designed to
        police, and (until G6.3 step 7) the only one it could not: the link
        was a bare unbound ``IdentityOperator() & IdentityOperator()``, and
        one ``None`` short-circuited the check. An
        :class:`~orpheus.numerics.operator.IdentityOperator` is an
        *endomorphism* :math:`V \to V`; a torus wrap needs an *isomorphism
        between two different spaces*, :math:`\Gamma_+(f') \to \Gamma_-(f)`.
        Step 7 REPLACED the link with that arrow, derived from the wrap's
        MOTION (this row asserted the opposite until then — it was the
        strict pin that reddened when the step landed, with a failure
        message naming its own re-scope).

        Two legs, and the second is the catcher: the RIGHT face's
        restriction composes into ``Γ(xmax) → Γ₋(xmin)``; the WRONG face's —
        feeding the law its OWN face's outflow, the exact pre-B3.4c defect
        (98 % relative when live, invisible to every shape check because
        ``|Γ₊| == |Γ₋|``) — now cannot even be FORMED.
        """
        sn = TestPeriodicReadsThePartnerFace._periodic_slab()
        law = sn.bc["xmin"]
        trace = sn.angular_trace
        assert law.domain == trace.outflow_space("xmax")
        assert law.codomain == trace.inflow_space("xmin")
        composed = law @ trace.outflow_restriction("xmax")
        assert composed.codomain == trace.inflow_space("xmin")
        with pytest.raises(IncompatibleOperatorComposition):
            law @ trace.outflow_restriction("xmin")


class TestNarrowedLawDomain:
    r"""**RG-3 / RG-3b** — the realized law's own type, at the mesh-wired face.

    RG-1/RG-2/RG-4 gate the COMPOSITE ``B``. These gate the LAW object ``B``
    composes, which is where B3.2's narrowing actually lives — and where a
    regression to a full-face realization would be caught even if the
    composite's restriction accidentally papered over it.
    """

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_realized_law_maps_gamma_plus_to_gamma_minus(
        self, case_id: str,
    ) -> None:
        r"""**RG-3** — ``law : (|Γ₊|, …) → (|Γ₋|, …)``, and it is NOT an
        endomorphism of the full face.

        The second leg is load-bearing and is NOT redundant with the first.
        **[M]** ``|Γ₊| == |Γ₋|`` on every quadrature × face pair in the tree
        (gauss_legendre 4/5/8, product 2×4/3×4/4×8, lebedev 9/17,
        level_symmetric 4/6 — all faces), so a shape assertion ALONE cannot
        distinguish ``Γ₊ → Γ₋`` from ``Γ₊ → Γ₊``: the error class sits inside
        the measured functional's invariance group (vv Mode 12). The
        discriminator that escapes it: feed the law the FULL face and check it
        does not emit ``N`` rows. A narrowed law structurally cannot (its
        output length is ``|Γ₋| < N``); an un-narrowed endomorphism always
        does. That is exactly the leg that catches the still-full-face laws in
        ``test_boundary_law_domain.py``.
        """
        sn = _sn(*_CASES[case_id])
        for face in sn.angular_trace.layout.faces:
            law = sn.bc[face]
            inflow = sn.angular_trace.inflow_indices_for_face(face)
            outflow = sn.angular_trace.outflow_indices_for_face(face)
            slot = _random_state(sn).boundary.face_view(face)
            n_face = slot.shape[0]
            if inflow.size == n_face:
                pytest.fail(
                    f"{case_id} face {face!r}: |Γ₋| == N, so the "
                    f"'does not emit N rows' leg below is vacuous on this "
                    f"fixture."
                )
            image = law.apply(_gather(slot, outflow))
            if image.shape != (inflow.size,) + slot.shape[1:]:
                pytest.fail(
                    f"{case_id} face {face!r}: the realized law maps "
                    f"{(outflow.size,) + slot.shape[1:]} -> {image.shape}; "
                    f"the narrowed contract is Γ₊ -> Γ₋, i.e. "
                    f"{(inflow.size,) + slot.shape[1:]}."
                )
            full_image_rows = np.asarray(law.apply(slot)).shape[0]
            if full_image_rows == n_face:
                pytest.fail(
                    f"{case_id} face {face!r}: the realized law emits "
                    f"{n_face} rows for a full-face input — it is an "
                    f"ENDOMORPHISM of the whole face slot, not a Γ₊ -> Γ₋ "
                    f"map. (|Γ₊| == |Γ₋| here, so the shape leg above cannot "
                    f"see this — vv Mode 12.)"
                )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "B3.2 GAP — the narrowed law does not validate its domain. "
            "TraceRestrictionOperator carries the shape guard the crosswalk "
            "§9 designed, but the operator the realizer EMITS does not: "
            "reflective is a bare PermutationOperator on the reduced axis "
            "(np.take silently truncates a full-face input and returns wrong "
            "values), and vacuum's ZeroOperator space hook ignores its input "
            "length entirely. MEASURED: both return |Γ₋| rows with NO raise. "
            "The composite never spells it (it always composes γ₊ first), so "
            "no value moves today — but 'hand the law the full face' stays a "
            "silent wrong answer until the guard lands. Delete this marker "
            "when it does. "
            "OWNER = campaign phase B4 (RE-ATTRIBUTED 2026-08-14). The "
            "crosswalk parked this in B3.4 as 'the one place the guard "
            "belongs — the phase that restructures around R o G'. B3.4a/b/c "
            "LANDED and did NOT restructure around R o G; that is B4, which "
            "owns the generic factor-reading dispatch this guard belongs in. "
            "So the phase name was wrong, not the reasoning."
        ),
    )
    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_realized_law_refuses_a_full_face_input(self, case_id: str) -> None:
        r"""**RG-3b** — the negative test for the domain guard **B4** owes.

        This is the gate that makes "hand the law the full face" UNSPELLABLE
        rather than merely wrong. It is a deliberate strict xfail: the marker
        set IS the todo list, so the guard cannot land silently and this row
        cannot rot into a green that means nothing.

        ⛔ **Re-attributed from B3.4 to B4 on 2026-08-14.** The crosswalk
        placed it in B3.4 because that was to be "the phase that restructures
        around :math:`R \circ G`". B3.4a/b/c landed and did **not** do that —
        the generic factor-reading dispatch is B4's, and a domain guard added
        anywhere else would be a per-law arm of exactly the ``isinstance``
        ladder B4 exists to collapse. The reasoning was right; only the phase
        name was wrong.

        Structured so that EXACTLY ONE statement can fail and it is the
        documented one (vv Mode 8, class 4): the fixture work is done first and
        a fixture problem raises OUTSIDE the ``pytest.raises`` block, where it
        would surface as a real error rather than a misattributed xfail.
        """
        sn = _sn(*_CASES[case_id])
        face = next(iter(sn.angular_trace.layout.faces))
        law = sn.bc[face]
        slot = _random_state(sn).boundary.face_view(face)
        outflow = sn.angular_trace.outflow_indices_for_face(face)
        if outflow.size == slot.shape[0]:
            pytest.fail(
                f"{case_id} face {face!r}: |Γ₊| == N, so a full-face input is "
                f"not a distinguishable wrong shape and this gate is vacuous."
            )
        with pytest.raises((ValueError, IndexError)):
            law.apply(slot)


class TestScheduleSplitPartition2D:
    r"""**RG-5** — the split partitions ``B`` on a **2-D** mesh.

    B3.2's row-restricted emission carries index arithmetic that did not exist
    before it: the requested rows ``sel`` index the FACE, but the law's image
    is indexed by POSITION WITHIN :math:`\Gamma_-`, so the write needs
    ``Γ₋(f).to_local(sel)`` (the half-trace SPACE's remap since G6.5).

    **[M]** The plausible transcription ``arange(sel.size)`` is *exactly
    correct in 1-D* — the schedule hands each face entirely to one half, so
    ``sel`` IS a prefix of the inflow set — and wrong in 2-D, where the
    lower-half rows interleave (measured on a 2-D reflective mesh:
    ``face ymin: inflow=[2,3,6,7,10,11,…]`` but ``lower rows=[6,7,14,15,22,23]``,
    not a prefix). The boundary suite's existing split gate
    (``test_psi_half_coupling.py::test_split_masked_halves_are_trace_only``) is
    built on a SPHERE and stays green under that mutation; the only catchers
    are end-to-end 2-D solves in ``tests/sn/solve/``. This is the mechanism
    gate that closes the §0.6 config blindness.
    """

    def test_split_halves_partition_the_whole_trace_on_a_2d_mesh(self) -> None:
        """``B_lower + B_upper == B`` bit-identically, 2-D reflective."""
        from orpheus.sn.loss_representation.sweep_schedule import (
            SweepSchedule,
            reflective_faces,
        )

        sn = _sn_2d()
        B = SNBoundaryOperator(sn)
        parts = B.split(SweepSchedule.gauss_seidel(
            sn.ndim, sn.quad.octants, reflective_faces(sn)))
        psi = _random_state(sn, seed=17)
        whole = B.apply(psi).boundary.values
        total = (
            parts.lower.apply(psi).boundary.values
            + parts.upper.apply(psi).boundary.values
        )
        np.testing.assert_array_equal(
            total, whole,
            err_msg=(
                "B_lower + B_upper != B on a 2-D mesh — the narrowed "
                "sel -> position-within-Γ₋ remap does not partition the "
                "whole-trace reflection. (A 1-D fixture is BLIND to this: "
                "there the requested rows are a prefix of the inflow set and "
                "arange(sel.size) is accidentally correct.)"
            ),
        )

    def test_the_2d_lower_rows_are_not_a_prefix_of_the_inflow_set(self) -> None:
        r"""The ACTIVATION guard for the gate above — without it, RG-5 could
        silently degrade into a 1-D-equivalent fixture and stop testing the
        remap at all (a decayed gate, vv Mode 8 class 7).

        Pins the structural property that makes the 2-D mesh discriminating:
        on at least one face, the schedule's lower-half rows are NOT a prefix
        of :math:`\Gamma_-`, so ``arange`` and ``to_local`` genuinely differ.
        """
        from orpheus.sn.loss_representation.sweep_schedule import (
            SweepSchedule,
            reflective_faces,
        )

        sn = _sn_2d()
        lower_rows = SweepSchedule.gauss_seidel(
            sn.ndim, sn.quad.octants, reflective_faces(sn)).lower_inflow_rows(sn)
        discriminating = []
        for face, sel in lower_rows.items():
            inflow = sn.angular_trace.inflow_indices_for_face(face)
            sel = np.asarray(sel)
            if sel.size and not np.array_equal(
                np.searchsorted(inflow, sel), np.arange(sel.size)
            ):
                discriminating.append(face)
        if not discriminating:
            pytest.fail(
                "No face of the 2-D fixture has lower-half rows that differ "
                "from a prefix of Γ₋ — RG-5 has decayed into a 1-D-equivalent "
                "configuration and no longer exercises the to_local remap. "
                f"lower_rows={ {f: np.asarray(s).tolist() for f, s in lower_rows.items()} }"
            )


class TestFaceRestrictedReflect:
    """The face-restricted reflect — since CS4c step 6 item 6.5 spelled on the
    ONE live verb, ``SNMaskedBoundaryOperator.reflect_rows_inplace`` through
    the Jacobi split's ``upper`` half (every inflow row of every face): the
    octant-group Gauss-Seidel schedule reflects ONLY the just-swept group's
    reflective outgoing faces between octant-group sweeps (the
    ``(L+C−B_lower)⁻¹`` forward substitution), and a face subset must be the
    EXACT restriction of the whole-trace reflect.

    The restriction is exact because the whole INPUT trace stays in scope —
    NOT because ``B`` is block-diagonal over faces (since B3.4c it is not: a
    quotient law reads its partner's half-trace). Until item 6.5 this class
    read the retired trace-only ``reflect_into_inflow(faces=...)``; its
    ``faces=None ≡ all faces`` claim died with that verb (the live verb has
    no default), and its unknown-face refusal MOVED into the live verb.
    """

    @staticmethod
    def _full_inflow_mask(sn: SNMesh):
        from orpheus.sn.loss_representation.sweep_schedule import SweepSchedule

        return SNBoundaryOperator(sn).split(
            SweepSchedule.jacobi(sn.ndim, sn.quad.octants),
        ).upper

    @staticmethod
    def _inflow_zeroed(sn: SNMesh, boundary: AngularBoundaryFlux) -> AngularBoundaryFlux:
        """A copy whose INFLOW rows are zero and whose OUTFLOW rows (the
        reflect's input) are the original's — the additive reflect then reads
        as the assignment."""
        out = boundary.copy()
        trace = sn.angular_trace
        for face in out.layout.faces:
            out.face_view(face)[trace.inflow_indices_for_face(face)] = 0.0
        return out

    def test_subset_reflects_only_selected_faces(self) -> None:
        """``faces=("xmax",)`` emits reflected inflow on xmax and leaves the
        unselected xmin face's inflow rows untouched (zero)."""
        sn = _sn("SLB", (BC.reflective, BC.reflective))
        boundary = _random_state(sn).boundary
        assert isinstance(boundary, AngularBoundaryFlux)
        mask = self._full_inflow_mask(sn)
        trace = sn.angular_trace
        only_xmax = self._inflow_zeroed(sn, boundary)
        mask.reflect_rows_inplace(only_xmax, ("xmax",))
        full = self._inflow_zeroed(sn, boundary)
        mask.reflect_rows_inplace(full, tuple(full.layout.faces))
        xmax_inflow = trace.inflow_indices_for_face("xmax")
        xmin_inflow = trace.inflow_indices_for_face("xmin")
        # Selected face: bit-identical to the whole-trace reflect (the exact
        # restriction — each target row is written independently from the
        # same input trace).
        np.testing.assert_array_equal(
            only_xmax.face_view("xmax")[xmax_inflow], full.face_view("xmax")[xmax_inflow],
        )
        # Unselected face: its inflow rows stay at the zero they were set to.
        assert not only_xmax.face_view("xmin")[xmin_inflow].any(), (
            "reflect_rows_inplace(faces=('xmax',)) emitted on the unselected "
            "xmin face — the restriction is not clean."
        )
        # Sanity: the selected face actually carries non-zero reflected inflow
        # (else the restriction claim would be vacuous).
        assert only_xmax.face_view("xmax")[xmax_inflow].any()

    def test_subset_partitions_the_whole_trace_reflect(self) -> None:
        """Single-face reflects sum EXACTLY (on the inflow rows) to the
        whole-trace reflect — the per-face restrictions are a clean partition
        (vv L11: catches a face↔face coupling leak that the
        reflect-only-selected test alone would miss)."""
        sn = _sn("SLB", (BC.reflective, BC.reflective))
        boundary = _random_state(sn).boundary
        assert isinstance(boundary, AngularBoundaryFlux)
        mask = self._full_inflow_mask(sn)
        trace = sn.angular_trace
        full = self._inflow_zeroed(sn, boundary)
        mask.reflect_rows_inplace(full, tuple(full.layout.faces))
        xmin_only = self._inflow_zeroed(sn, boundary)
        mask.reflect_rows_inplace(xmin_only, ("xmin",))
        xmax_only = self._inflow_zeroed(sn, boundary)
        mask.reflect_rows_inplace(xmax_only, ("xmax",))
        for face in full.layout.faces:
            rows = trace.inflow_indices_for_face(face)
            np.testing.assert_array_equal(
                full.face_view(face)[rows],
                xmin_only.face_view(face)[rows] + xmax_only.face_view(face)[rows],
            )

    def test_unknown_face_raises(self) -> None:
        """A face not on the mesh boundary is a caller error — raise, do not
        silently skip (illegal states unrepresentable). The refusal moved
        into the live verb at CS4c step 6 item 6.5: until then
        ``reflect_rows_inplace`` filtered the face away silently while the
        retired trace-only verb raised."""
        sn = _sn("SLB", (BC.reflective, BC.reflective))
        boundary = _random_state(sn).boundary
        assert isinstance(boundary, AngularBoundaryFlux)
        with pytest.raises(ValueError, match="boundary faces"):
            self._full_inflow_mask(sn).reflect_rows_inplace(boundary, ("bogus_face",))


# ─────────────────────────────────────────────────────────────────────
# B3.4c — the partner-face channel
# ─────────────────────────────────────────────────────────────────────


class TestPeriodicReadsThePartnerFace:
    r"""``B`` is block-STRUCTURED, and a quotient law sits off the diagonal.

    The claim only exists at this level: the realized periodic operator is an
    identity, so nothing about it is wrong in isolation — the defect B3.4c
    fixed was in WHICH half-trace the composition hands it. Before B3.4c
    :meth:`_reflect_trace` fed every law its own face's :math:`\Gamma_+`, so a
    periodic face returned its own outflow as its inflow (MEASURED 98 %
    relative against the partner-face reference).

    Periodic is not in ``SNMesh.BOUNDARY_OPERATOR_REGISTRY`` (#189), so
    ``BC("periodic")`` refuses at parse and the law is installed through the
    method's own ``realize_boundary_law`` hook — the same production arm the
    tag path would reach, one step later.
    """

    @staticmethod
    def _periodic_slab(nx: int = 4, ng: int = 1) -> SNMesh:
        sn = _sn("SLB", (BC.vacuum, BC.vacuum), nx=nx, ng=ng)
        law = PeriodicBoundary(axis="x")
        for face in ("xmin", "xmax"):
            sn.bc[face] = sn.realize_boundary_law(law, face)
        return sn

    def test_the_domain_map_is_the_off_diagonal_swap(self) -> None:
        """``_face_domains`` IS the block index — and it is the face-level
        trace digraph the SCC criterion (#324) needs."""
        B = SNBoundaryOperator(self._periodic_slab())
        assert B._face_domains == {"xmin": "xmax", "xmax": "xmin"}

    def test_each_face_receives_its_PARTNERS_outflow(self) -> None:
        r"""The requirement: :math:`\gamma_-\psi|_f = \gamma_+\psi|_{f'}`.

        The two faces carry INDEPENDENT random data — with a shared draw the
        rows would coincide and a per-face endomorphism would look correct,
        which is the whole reason the defect survived to B3.4c. The negative
        leg (``!= own outflow``) is asserted too, so a fixture where the two
        happen to agree cannot pass this vacuously.
        """
        sn = self._periodic_slab()
        B = SNBoundaryOperator(sn)
        trace = sn.angular_trace
        psi = _random_state(sn, seed=11)
        out = B.apply(psi)
        for face, partner in (("xmin", "xmax"), ("xmax", "xmin")):
            got = out.boundary.face_view(face)[
                trace.inflow_indices_for_face(face)
            ]
            partner_out = psi.boundary.face_view(partner)[
                trace.outflow_indices_for_face(partner)
            ]
            own_out = psi.boundary.face_view(face)[
                trace.outflow_indices_for_face(face)
            ]
            np.testing.assert_array_equal(got, partner_out)
            assert not np.array_equal(got, own_out), (
                f"{face}: the partner's outflow equals this face's own, so "
                f"the fixture cannot discriminate the pre-B3.4c defect."
            )

    def test_the_transpose_returns_along_the_same_channel(self) -> None:
        r"""Euclidean reciprocity :math:`\langle Bx, y\rangle
        = \langle x, B^\top y\rangle` on independently-seeded ``x`` and ``y``.

        ⚠ This is NOT sufficient on its own and is not claimed to be: if the
        forward reads the wrong face and the transpose scatters to the same
        wrong face, reciprocity STILL holds (`[M]` — reverting the composition
        leaves this green at 1.15e-16 relative). It pins that the two legs
        agree; the leg above pins that they agree on the RIGHT face.
        """
        sn = self._periodic_slab()
        B = SNBoundaryOperator(sn)
        x = _random_state(sn, seed=11)
        y = _random_state(sn, seed=29)
        lhs = float(np.sum(B.apply(x).boundary.values * y.boundary.values))
        rhs = float(
            np.sum(x.boundary.values * B.apply_transpose(y).boundary.values)
        )
        assert lhs == rhs, f"<Bx,y> = {lhs!r} but <x,Bty> = {rhs!r}"

    def test_a_half_declared_periodic_pair_is_refused(self) -> None:
        """A face glued to a partner that is not glued back is not an
        identification — and its whole-slot transpose writes would collide.

        This is the well-posedness statement ``_face_domains`` certifies: every
        face's :math:`\\Gamma_+` must feed exactly one law. Here ``xmax`` would
        feed two (its own vacuum law and ``xmin``'s wrap) while ``xmin``'s fed
        none.
        """
        sn = _sn("SLB", (BC.vacuum, BC.vacuum))
        sn.bc["xmin"] = sn.realize_boundary_law(
            PeriodicBoundary(axis="x"), "xmin",
        )
        with pytest.raises(ValueError, match="not a permutation"):
            SNBoundaryOperator(sn)._face_domains

    def test_the_output_never_aliases_the_input(self) -> None:
        """The realized periodic body is now a bare ``IdentityOperator``, which
        returns its argument BY REFERENCE — so the safe-aliasing contract has
        to be earned by the composition rather than by the leaf.

        It is: the trace restriction is fancy indexing, which copies, and the
        image is scattered into a freshly-zeroed sink. Pinned because the leaf
        that used to guarantee it (``PeriodicWrapOperator``, whose body was
        ``x.copy()``) retired at B3.4c, and a future composition that passed a
        view straight through would reintroduce the hazard silently.
        """
        sn = self._periodic_slab()
        psi = _random_state(sn, seed=5)
        before = psi.boundary.values.copy()
        out = SNBoundaryOperator(sn).apply(psi)
        out.boundary.values[...] = 1e9
        np.testing.assert_array_equal(psi.boundary.values, before)
