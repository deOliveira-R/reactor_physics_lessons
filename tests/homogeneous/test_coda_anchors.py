r"""CS4c **coda** — the PRE-CARVE anchors (test-architect, 2026-09-08).

Landed on the UNMODIFIED tree, before the coda's first production edit, so
that every row below is a measurement of the tree the carve starts from.
The coda (plan ``.claude/plans/cs4c_binding_design.md`` §26) has three
commits: **C1** the homogeneous problem gets a hub and supplies its own
data (the fabricated one-cell carrier stops being constructed on the
path); **C2** the fabricated carrier path retires; **C3** the record.

Four claim kinds live here, and they have DIFFERENT fates — read the class
docstrings before touching a row:

* :class:`TestTheOperatorAssemblyAgainstRawMixtureData` — **REFERENCE**
  (structurally independent: the expected value is built from the raw
  :class:`~orpheus.data.macro_xs.mixture.Mixture` arrays, never from the
  carrier, the facade, or an operator). MUST STAY GREEN through C1 and C2.
* :class:`TestTheOperatorAssemblyIsByteStableAcrossTheCoda` — **RECORD**
  against a frozen pre-carve capture (``_fixtures/coda_precarve_operators.json``,
  produced by this module's ``capture`` entry point on the unmodified tree).
  MUST STAY GREEN through C1 and C2; the capture is NEVER regenerated.
* :class:`TestTodaysFabricatedCarrier` — **RECORD of a state the coda
  DELETES**. Designed to red, and the commit that reds each row is named in
  the row's own docstring. This is ``plan-authoring`` §6c's red-before
  made executable, not a landmine.
* :class:`TestNoMaterialMeshIsBuiltOnTheHomogeneousPath` and
  :class:`TestTheHubHasNotLandedYet` — the RULED post-carve gates, shipped
  as ``xfail(strict=True)`` so their XPASS is a FAILURE and the landing
  commit MUST delete the marker (a self-retiring todo list).

**What this module deliberately does NOT duplicate.**
``test_byte_stability.py`` (D5) is ALREADY the frozen pre-carve record of
the end-to-end answer — ``_fixtures/cs1_prewiring.json``, captured at
``24a991ba`` (PRE-wiring) and never regenerated. It is the coda's
bit-identity wall for ``k_inf``/``flux``/``sig_prod``/``sig_abs`` and MUST
NOT be re-captured. This module pins the tier D5 cannot localize: the
assembled **operator matrices** :math:`A` and :math:`F`, on all eight D5
cases (the tree pins :math:`A` on ONE case today, against the facade's own
cached views — not an independent reference).

**Population** — the eight D5 producing mixtures, imported from
``test_byte_stability`` so there is ONE list (coding-elegance Pattern 2;
the in-tree precedent is ``test_operator_spaces.py``'s ``_all_d5_mixtures``).

**Activation, stated (`vv` anti-#20 / anti-#25).** The eight cases do NOT
all exercise the same terms — the census is asserted by
:func:`test_the_activation_table_is_what_the_reference_rows_rest_on` so a
future fixture edit that silently nulls a term reds:

======================  =========  ====================================
term                    cases      the row that would go vacuous without it
======================  =========  ====================================
:math:`2\Sigma_2^{T}`   1 of 8     the loss reference's (n,2n) leg
group-coupled           6 of 8     the loss reference's off-diagonal leg
:math:`\Sigma_{s0}`     (1g nulls it)
``eg``-bearing energy   1 of 8     the ``from_grid`` energy arm
:math:`P_1` present     3 of 8     the ``at_order(0)`` truncation
``ng > 1``              6 of 8     the fission dyad's DIRECTION
                                   (:math:`\chi\otimes\nu\Sigma_f` vs its
                                   transpose is invisible at ``ng = 1``)
======================  =========  ====================================
"""

from __future__ import annotations

import base64
import json
import pathlib

import numpy as np
import pytest

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.homogeneous.solver import (
    _assemble_loss_operator,
    _pose_space,
    solve_homogeneous_infinite,
)
from orpheus.transport.mesh.material_mesh import MaterialMesh
from orpheus.transport.operators.isotropic_transfer import IsotropicFission

pytestmark = pytest.mark.foundation

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "coda_precarve_operators.json"


def _require(condition: bool, message: str) -> None:
    """A ``-O``-firing assertion (NOT a bare ``assert``; `vv` Mode 8)."""
    if not condition:
        pytest.fail(message)


def _d5_cases() -> dict[str, Mixture]:
    """The eight D5 producing mixtures — ONE list, D5's own (Pattern 2)."""
    from tests.homogeneous.test_byte_stability import _mixture_cases

    return {k: v for k, v in _mixture_cases().items() if isinstance(v, Mixture)}


#: Parametrize by LABEL, never by a value a production call produces: a
#: ``parametrize`` argument list is evaluated at COLLECTION, so a battery
#: arm that makes the mixture registry raise would kill collection and be
#: read off the summary as ``FAILED=0`` (`vv` Mode 8, third pipeline
#: class). The literal is itself gated below against ``_d5_cases()``, so
#: it cannot silently drift from D5's population.
_CASE_IDS = (
    "homo_1eg",
    "homo_2eg",
    "homo_2eg_n2n",
    "homo_2eg_with_eg",
    "homo_4eg",
    "mixture_A_1g",
    "mixture_A_2g",
    "mixture_A_4g",
)


# ── The ONE production-mirroring construction (the C1 migration point) ──
#
# ⚠ §6b: C1 re-points THIS FUNCTION BODY (and nothing else in this module)
# at the hub — ``HomogeneousProblem(mix).loss`` / ``.production``. The rows
# below are written against its RETURN, so the migration is one body.


def _production_operators(mix: Mixture):
    """``(A, F)`` as the production path assembles them today."""
    space = _pose_space(mix)
    mat_xs = MaterialMesh.from_materials({0: mix}).material_xs_field()
    loss = _assemble_loss_operator(mat_xs, space)
    production = IsotropicFission.from_material_xs(mat_xs, space=space)
    return np.asarray(loss.as_matrix()), np.asarray(production.as_matrix())


# ── The structurally-independent reference: RAW Mixture arrays only ──


def _raw_loss_reference(mix: Mixture) -> np.ndarray:
    r""":math:`A = \operatorname{diag}(\Sigma_t) - (\Sigma_{s0} + 2\Sigma_2)^{T}`.

    Built from the mixture's OWN arrays — never from the carrier, the
    ``MaterialXSField`` facade, its caches, or any operator. The
    multiplicity 2 is spelled here as the literal the module docstring of
    ``orpheus/homogeneous/solver.py`` states, so a kernel-side
    multiplicity regression reds against a hand-written constant.
    """
    sig_s0 = np.asarray(mix.SigS[0].todense())
    sig_2 = np.asarray(mix.Sig2[0].todense())
    return np.diag(mix.SigT) - (sig_s0 + 2.0 * sig_2).T


def _raw_fission_reference(mix: Mixture) -> np.ndarray:
    r""":math:`F = \chi \otimes \nu\Sigma_f` — the outer product, direction pinned."""
    return np.outer(np.asarray(mix.chi), np.asarray(mix.SigP))


class TestTheOperatorAssemblyAgainstRawMixtureData:
    """Claim kind **REFERENCE** — the expected value is raw ``Mixture`` data.

    Structural independence: neither side of these comparisons reads the
    fabricated carrier, the ``MaterialXSField`` facade, or its dense
    caches; the reference is `numpy` over the mixture's own arrays. That
    is what makes them survive C1 (the mint changes) and C2 (the carrier
    dies) unchanged, and what makes them a genuine anchor rather than a
    twin of the thing under test.

    Today the tree pins :math:`A` on ONE case
    (``test_homogeneous.py::test_assemble_loss_operator_matches_fused_oracle``,
    whose own docstring says its reference *"shares ``mat_xs`` data with
    the fused form (so it is NOT structurally independent)"*) and pins
    :math:`F` at the matrix tier nowhere on this path.
    """

    @pytest.mark.parametrize("case", _CASE_IDS)
    def test_loss_matrix_equals_the_raw_mixture_reference(self, case: str) -> None:
        r"""``A = diag(Σ_t) − (Σ_s0 + 2Σ₂)ᵀ`` from the mixture alone, 8 cases."""
        mix = _d5_cases()[case]
        got, _ = _production_operators(mix)
        want = _raw_loss_reference(mix)
        _require(
            got.shape == want.shape,
            f"{case}: A has shape {got.shape}, the reference {want.shape}",
        )
        np.testing.assert_allclose(
            got, want, rtol=0.0, atol=1e-14,
            err_msg=f"{case}: the assembled loss matrix left the raw-mixture reference",
        )

    @pytest.mark.parametrize("case", _CASE_IDS)
    def test_fission_matrix_equals_the_raw_mixture_dyad(self, case: str) -> None:
        r"""``F = χ ⊗ νΣ_f`` from the mixture alone — DIRECTION pinned, 8 cases.

        The transpose ``νΣ_f ⊗ χ`` is a different matrix wherever
        ``ng > 1`` (6 of the 8 cases; at ``ng = 1`` both are the same
        1×1 scalar and the row carries no direction information — stated,
        not silently counted, per `vv` anti-#20).
        """
        mix = _d5_cases()[case]
        _, got = _production_operators(mix)
        want = _raw_fission_reference(mix)
        np.testing.assert_allclose(
            got, want, rtol=0.0, atol=1e-14,
            err_msg=f"{case}: the fission dyad left χ ⊗ νΣ_f",
        )

    def test_the_activation_table_is_what_the_reference_rows_rest_on(self) -> None:
        """The census the two reference rows' coverage claim rests on.

        A fixture edit that silently nulls the (n,2n) channel, the group
        coupling, the ``eg`` grid or the ``P_1`` block would leave the
        reference rows GREEN and vacuous for that term (`vv` anti-#25 —
        the term that is nulled is exactly where a defect survives). This
        row makes the denominator falsifiable.
        """
        cases = _d5_cases()
        _require(
            tuple(sorted(cases)) == _CASE_IDS,
            f"the parametrize LABEL list has drifted from D5's population: "
            f"{tuple(sorted(cases))} != {_CASE_IDS}",
        )
        n2n = [
            n for n, m in cases.items()
            if np.count_nonzero(np.asarray(m.Sig2[0].todense()))
        ]
        coupled = [
            n for n, m in cases.items()
            if np.count_nonzero(
                np.asarray(m.SigS[0].todense())
                - np.diag(np.diag(np.asarray(m.SigS[0].todense())))
            )
        ]
        gridded = [n for n, m in cases.items() if m.eg is not None]
        p1 = [n for n, m in cases.items() if len(m.SigS) > 1]
        multigroup = [n for n, m in cases.items() if m.ng > 1]
        _require(
            n2n == ["homo_2eg_n2n"],
            f"the 2Σ₂ᵀ loss term is activated by {sorted(n2n)}, not by "
            f"['homo_2eg_n2n'] alone — the (n,2n) coverage denominator moved",
        )
        _require(len(coupled) == 6, f"group coupling: {len(coupled)} of 8, expected 6")
        _require(len(gridded) == 1, f"eg-bearing: {len(gridded)} of 8, expected 1")
        _require(len(p1) == 3, f"P1-bearing: {len(p1)} of 8, expected 3")
        _require(
            len(multigroup) == 6,
            f"ng > 1 (the fission-dyad DIRECTION witness): {len(multigroup)} of 8, "
            f"expected 6",
        )


class TestTheOperatorAssemblyIsByteStableAcrossTheCoda:
    """Claim kind **RECORD** — a frozen PRE-CARVE capture on disk.

    The oracle is ``_fixtures/coda_precarve_operators.json``, produced by
    this module's ``capture`` entry point on the unmodified tree (the
    generator and the gate share ONE payload helper —
    :func:`_production_operators` — so a generator computing the payload
    one way and a gate recomputing it another way is unspellable;
    coding-elegance Pattern 2, D5's own discipline).

    ⛔ **Never regenerate.** Its whole value is that it predates every line
    the coda touches (`vv` §bit-identity, criterion "the anchor must not be
    re-baselined BY the carve that it gates"). Bytes, not a hash, so a
    mismatch localizes.

    Pairs with the REFERENCE class above per the two-anchor template: a
    frozen record says *something moved*, the raw-mixture reference says
    *which side is right*.
    """

    @pytest.mark.parametrize("case", _CASE_IDS)
    def test_the_assembled_matrices_are_byte_stable(self, case: str) -> None:
        """``A`` and ``F`` are bit-identical to the pre-carve capture."""
        recorded = json.loads(_FIXTURE.read_text())[case]
        A, F = _production_operators(_d5_cases()[case])
        for name, live in (("A", A), ("F", F)):
            want = np.frombuffer(
                base64.b64decode(recorded[f"{name}_b64"]), dtype=float
            ).reshape(tuple(int(n) for n in recorded[f"{name}_shape"]))
            if not np.array_equal(live, want):
                moved = np.flatnonzero(want.ravel() != live.ravel())
                pytest.fail(
                    f"{case}.{name} moved at flat indices {moved.tolist()}: "
                    f"{want.ravel()[moved]} -> {live.ravel()[moved]} — the coda "
                    f"is a bit-identical re-source (`[M]` 8 of 8 pre-carve)"
                )


class TestTodaysFabricatedCarrier:
    """Claim kind **RECORD of a state the coda DELETES** — designed to red.

    O1's tell (``kernel_and_medium_objectives.md:72``) is *"no ``[0,1]``
    edges, no invented node, no coordinate system on the path"*. Both
    halves are asserted here so the carve's red is ATTRIBUTABLE: the
    CONSTRUCTION half dies at C2 (the factory retires), the
    NOT-CONSUMED half goes vacuous at C1 (nothing fabricated is built,
    so an alien carrier can no longer be injected — ``plan-authoring``
    §10's acceptance-artifact widening, declared rather than discovered).
    """

    def test_the_carrier_the_solver_builds_today_is_fabricated(self) -> None:
        r"""RED AT **C2** (``MaterialMesh.from_materials`` retires).

        Every datum here is invented by the factory, not by the physics:
        a unit cell on ``[0, 1]``, a node at ``0.5``, a CARTESIAN chart,
        ``mesh is None``. Delete this row in the C2 commit — its subject
        is gone, and O1's tell is then a grep obligation, not a gate.
        """
        from orpheus.geometry import CoordSystem

        carrier = MaterialMesh.from_materials({0: _d5_cases()["homo_2eg"]})
        _require(
            bool(np.array_equal(carrier.axes[0].edges, np.array([0.0, 1.0]))),
            f"the fabricated edges moved: {carrier.axes[0].edges}",
        )
        _require(carrier.mesh is None, "the carrier is no longer mesh-less")
        _require(
            carrier.coord is CoordSystem.CARTESIAN,
            f"the invented chart moved: {carrier.coord}",
        )
        _require(
            bool(np.array_equal(carrier.volume_measure.nodes.ravel(), [0.5])),
            f"the invented node moved: {carrier.volume_measure.nodes.ravel()}",
        )
        _require(
            bool(np.array_equal(carrier.volumes, [1.0])),
            f"the fabricated volume moved: {carrier.volumes}",
        )

    @pytest.mark.parametrize("case", _CASE_IDS)
    def test_no_fabricated_spatial_datum_reaches_the_answer(
        self, case: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r"""VACUOUS AT **C1** (nothing fabricated is constructed any more).

        The strongest available statement of O1's second half, and
        strictly stronger than G2.4 (``test_operator_spaces.py``), which
        mutates the carrier's ``volumes`` alone: here the WHOLE fabricated
        carrier is replaced by an alien one — edges ``[10, 13]`` (volume
        3), node ``11.5``, a SPHERICAL chart, and consequently a
        **different** ``bulk_space`` — and the four reported numerics are
        bit-identical. `[M]` 8 of 8 on the unmodified tree.

        ⚠ It is ALSO the pre-carve evidence for a hazard the carve closes:
        ``MultiplicationOperator`` never compares its coefficient's space
        to its ends (`[M]` ``grep -c 'coefficient\.space'`` = 0 in
        ``multiplication_operator.py``), which is WHY a coefficient minted
        on a wrong space passes today. After C1 the σ_t field is born on
        the pose and the disagreement is unspellable on this path.

        Delete this row in the C1 commit (its injection point is gone);
        the surviving claim is the construction spy below.
        """
        from orpheus.geometry import CoordSystem
        from orpheus.transport.mesh.axis import AxisMesh

        mix = _d5_cases()[case]
        baseline = solve_homogeneous_infinite(mix)

        def alien(materials):
            obj = MaterialMesh.__new__(MaterialMesh)
            obj._init_data(
                axes=(AxisMesh(edges=np.array([10.0, 13.0])),),
                mesh=None, mat_map=None, materials=materials,
            )
            obj.coord = CoordSystem.SPHERICAL
            return obj

        # Precondition, asserted: the alien really is a different carrier
        # (a decoy that does not discriminate is a dud — `vv` #17). Stated
        # RELATIVE to the fabricated carrier, never against a literal, so
        # a global measure mutation cannot red this row for an instrument
        # reason (`[M]` battery arm VOLUMES_X2).
        probe = alien({0: mix})
        fabricated = MaterialMesh.from_materials({0: mix})
        _require(
            not np.array_equal(probe.volumes, fabricated.volumes)
            and probe.coord is not fabricated.coord
            and probe.coord is CoordSystem.SPHERICAL
            and probe.bulk_space != fabricated.bulk_space,
            "the decoy carrier does not differ from the fabricated one",
        )

        monkeypatch.setattr(MaterialMesh, "from_materials", staticmethod(alien))
        alienated = solve_homogeneous_infinite(mix)

        _require(
            alienated.k_inf == baseline.k_inf
            and bool(np.array_equal(alienated.flux, baseline.flux))
            and alienated.sig_prod == baseline.sig_prod
            and alienated.sig_abs == baseline.sig_abs,
            f"{case}: an ALIEN fabricated carrier moved the answer — some "
            f"fabricated spatial datum IS consumed on the homogeneous path "
            f"(k {baseline.k_inf!r} -> {alienated.k_inf!r})",
        )


class TestNoMaterialMeshIsBuiltOnTheHomogeneousPath:
    r"""G2.4's ruled RE-POSE — a CONSTRUCTION SPY, with its positive control.

    The user's coda ruling (plan §26): *"G2.4 is re-posed as a construction
    spy — no ``MaterialMesh`` is constructed on the homogeneous path — with
    a positive control"*. G2.4 itself goes INERT at C1 (its ``volumes ×2``
    mutation has nothing to mutate once no carrier is built), which is the
    ``plan-authoring`` §10 shape; this class is what replaces it.

    The spy wraps ``MaterialMesh._init_data`` — the ONE data-construction
    body EVERY surface funnels into (``from_materials``,
    ``MaterialMesh.__init__``, ``SNMesh._init_core``,
    ``DiffusionMesh._init_core``). That handle is chosen so it SURVIVES C2:
    a spy on ``from_materials`` would lose its subject with the factory and
    read a confident zero for the wrong reason.
    """

    @staticmethod
    def _count(monkeypatch: pytest.MonkeyPatch) -> list[int]:
        calls: list[int] = []
        raw = MaterialMesh._init_data

        def spy(self, **kwargs):
            calls.append(1)
            return raw(self, **kwargs)

        monkeypatch.setattr(MaterialMesh, "_init_data", spy)
        return calls

    def test_the_construction_spy_sees_a_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSITIVE CONTROL — green before AND after the coda.

        Without it, a zero from the ruled gate below is ambiguous between
        *"nothing was constructed"* and *"the spy is not installed"*
        (`vv` anti-#17: the harness lies before the code does, and it lies
        in the safe-looking direction).
        """
        from orpheus.geometry import Mesh1D

        calls = self._count(monkeypatch)
        MaterialMesh(
            Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0])),
            {0: _d5_cases()["homo_2eg"]},
        )
        _require(
            len(calls) == 1,
            f"the construction spy saw {len(calls)} constructions, expected 1 "
            f"— the handle MaterialMesh._init_data is no longer the ONE "
            f"construction body, and every count this class reports is void",
        )

    def test_todays_solve_constructs_exactly_one_material_mesh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED AT **C1** — the RECORD half, and the spy's own attribution leg.

        `[M]` on the unmodified tree ``solve_homogeneous_infinite`` builds
        exactly ONE ``MaterialMesh`` (``solver.py:238``). Delete this row in
        the C1 commit, in the same edit that deletes the ``xfail`` marker
        below. Keeping BOTH rows is what makes a failure attributable: if
        the spy itself breaks, THIS row reds loudly instead of the ruled row
        silently swallowing it inside its ``xfail``.
        """
        calls = self._count(monkeypatch)
        solve_homogeneous_infinite(_d5_cases()["homo_2eg"])
        _require(
            len(calls) == 1,
            f"the homogeneous solve constructed {len(calls)} MaterialMesh "
            f"objects, not 1 — if this is C1 landing, delete this row and the "
            f"xfail marker on its sibling in the same commit",
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "coda C1 has not landed: orpheus/homogeneous/solver.py:238 still "
            "calls MaterialMesh.from_materials({0: mix}).material_xs_field(). "
            "When C1 lands this row XPASSes — a FAILURE under strict — and the "
            "marker must be deleted in the landing commit."
        ),
    )
    def test_no_material_mesh_is_constructed_on_the_homogeneous_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RULED gate (plan §26, G2.4's re-pose). Exactly one statement
        below can fail, and it is the documented one."""
        calls = self._count(monkeypatch)
        solve_homogeneous_infinite(_d5_cases()["homo_2eg"])
        _require(
            len(calls) == 0,
            f"the homogeneous path constructed {len(calls)} MaterialMesh "
            f"objects — O1's honest pose fabricates nothing",
        )


class TestTheHubHasNotLandedYet:
    """R-c1's todo marker — the hub's EXISTENCE, nothing more.

    The user's ruling (plan §26 R-c1): the coda mints a
    ``HomogeneousProblem``-shaped hub in ``orpheus/homogeneous/`` that owns
    the pose space and the mixture-direct kernel / cross-section fields as
    cached, keyed state, the solver reading them off it.

    ⚠ SCOPE, stated so nobody reads more into a green: this row asserts the
    hub EXISTS and is constructible from a ``Mixture``. Its identity
    contract (``is`` within one instance for the pose space and each field;
    equal mixtures minting ``==`` state) is NOT asserted here — an
    ``xfail`` hides every failure, so a contract assertion added to this
    body would go silent the day the hub lands wrong. Those rows are
    specified in ``scratch/_step6/test_architect_verification_plan_coda.md``
    §3 and land IN the C1 commit as ordinary green gates.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "coda C1 has not landed: orpheus.homogeneous exposes no "
            "HomogeneousProblem hub (R-c1). When C1 lands this row XPASSes — "
            "a FAILURE under strict — and the marker must be deleted."
        ),
    )
    def test_the_homogeneous_problem_hub_exists(self) -> None:
        import orpheus.homogeneous.solver as solver_module

        hub = getattr(solver_module, "HomogeneousProblem", None)
        _require(
            hub is not None,
            "orpheus.homogeneous.solver exposes no HomogeneousProblem "
            "(R-c1's hub); the coda's C1 has not landed",
        )
        assert hub is not None
        problem = hub(_d5_cases()["homo_2eg"])
        _require(
            problem.space == _pose_space(_d5_cases()["homo_2eg"]),
            "the hub's pose space is not the mixture-minted Energy ⊗ point",
        )


if __name__ == "__main__":  # pragma: no cover - capture entry point
    import sys

    # ``_d5_cases`` imports D5's own list through the ``tests`` package, so
    # the repo root must be importable when this file is run as a script.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        _FIXTURE.parent.mkdir(exist_ok=True)
        payload: dict[str, dict[str, object]] = {}
        for name, mixture in _d5_cases().items():
            A, F = _production_operators(mixture)
            entry: dict[str, object] = {}
            for label, arr in (("A", A), ("F", F)):
                contiguous = np.ascontiguousarray(arr, dtype=float)
                entry[f"{label}_b64"] = base64.b64encode(
                    contiguous.tobytes()
                ).decode("ascii")
                entry[f"{label}_shape"] = list(contiguous.shape)
            payload[name] = entry
        _FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"captured {len(payload)} cases -> {_FIXTURE}")
    else:
        print("usage: python tests/homogeneous/test_coda_anchors.py capture")
