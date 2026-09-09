r"""The homogeneous problem's HUB — :class:`~orpheus.homogeneous.solver.HomogeneousProblem`
(CS4c coda C1, ruling R-c1, 2026-09-08; the verification plan's §3.1 gates H1–H4, H6).

*"The homogeneous problem needs a hub, just like the function SNMesh (future
SNProblem) currently fulfills, to act as the place the consumed objects live
(and a save state)."* — the user's ruling. Four claims no value gate can
make, each gated here:

* **H1** — the state is CACHED: every consumed object is ``is``-identical
  on repeated reads of ONE hub (the row that forces ``cached_property``;
  a plain property mints a fresh object per access — a latent false red
  this campaign has met before);
* **H2** — the state is KEYED, not interned: two hubs over equal mixtures
  mint ``==`` objects with equal hashes, hubs over distinct mixtures mint
  unequal ones — and ``is`` ACROSS hubs is deliberately NOT asserted (two
  owners are two objects);
* **H3** ⭐ — every consumed field is BORN on the pose, and every bound
  operator's ends ARE the pose (``is``): the hazard the coda closes —
  `[M]` ``MultiplicationOperator`` never compares its coefficient's space
  to its ends (0 ``coefficient.space`` reads), which is why a field minted
  on a fabricated carrier's space passed for years — is unspellable here;
* **H4** ⭐⭐ — the solver READS the hub (a Mode-11 ROUTE gate): a DECOY
  installed on the hub's consumed surface moves the answer, with an
  activation leg (a hub over the decoy really differs) and an anti-dud leg.
* **H6** is a grep obligation, not a gate: the pose is minted ONCE —
  ``grep -c 'FunctionSpace.of_axes' orpheus/homogeneous/solver.py`` reads 1.

Foundation mark: software invariants of the hub; the values are D5's and
the anchors' (bit-identical to the retired carrier route, 8 of 8).
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.homogeneous.solver import (
    HomogeneousProblem,
    _pose_space,
    solve_homogeneous_infinite,
)
from orpheus.numerics.space import FunctionSpace
from orpheus.transport.fields.cross_section_field import CrossSectionField

pytestmark = pytest.mark.foundation


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


_CONSUMED = (
    "space", "layout", "scattering", "n2n", "fission",
    "total_cross_section_field", "absorption_cross_section_field", "fission_production_field",
    "collision", "isotropic_scattering", "isotropic_n2n", "isotropic_transfer",
    "loss", "production", "multiplication", "production_rate", "absorption_rate",
)


@pytest.mark.parametrize("groups", ["1g", "2g", "4g"])
def test_H1_every_consumed_object_is_cached_within_one_hub(groups: str) -> None:
    problem = HomogeneousProblem(get_mixture("A", groups))
    for name in _CONSUMED:
        first, second = getattr(problem, name), getattr(problem, name)
        _require(first is second, f"{name}: two reads of ONE hub minted two objects — the state is not cached")


def test_H2_hubs_over_equal_mixtures_mint_equal_state_and_distinct_mixtures_do_not() -> None:
    a, b = HomogeneousProblem(get_mixture("A", "2g")), HomogeneousProblem(get_mixture("A", "2g"))
    _require(a is not b, "precondition: two hubs are two objects")
    _require(a.space == b.space and hash(a.space) == hash(b.space), "equal mixtures must pose one space")
    _require(a.total_cross_section_field.space == b.total_cross_section_field.space, "fields on equal poses")
    np.testing.assert_array_equal(a.loss.as_matrix(), b.loss.as_matrix())
    np.testing.assert_array_equal(a.production.as_matrix(), b.production.as_matrix())
    # ⚠ NOT asserted, deliberately: ``a.space is b.space`` — two owners.
    c = HomogeneousProblem(get_mixture("A", "4g"))
    _require(a.space != c.space, "distinct mixtures must pose distinct spaces")
    _require(a.loss.as_matrix().shape != c.loss.as_matrix().shape, "distinct mixtures, distinct operators")


@pytest.mark.parametrize("groups", ["1g", "2g", "4g"])
def test_H3_every_field_is_born_on_the_pose_and_every_operator_ends_on_it(groups: str) -> None:
    problem = HomogeneousProblem(get_mixture("A", groups))
    pose = problem.space
    _require(pose == _pose_space(problem.mixture), "the hub's space is THE pose (one spelling)")
    for name in ("total_cross_section_field", "absorption_cross_section_field", "fission_production_field"):
        field = getattr(problem, name)
        _require(isinstance(field, CrossSectionField), f"{name} is a CrossSectionField")
        _require(field.space is pose, f"{name} was not born on the pose (space is not the hub's)")
        _require(field.values.shape == pose.shape, f"{name} shape {field.values.shape} != pose {pose.shape}")
    for name in ("collision", "isotropic_scattering", "isotropic_n2n", "isotropic_transfer", "loss", "production", "multiplication"):
        op = getattr(problem, name)
        _require(op.domain is pose and op.codomain is pose, f"{name}'s ends are not the hub's pose (by identity)")
    # the values are the mixture's, verbatim
    mix = problem.mixture
    np.testing.assert_array_equal(problem.total_cross_section_field.values.ravel(), np.asarray(mix.SigT, dtype=float))
    np.testing.assert_array_equal(problem.fission_production_field.values.ravel(), np.asarray(mix.SigP, dtype=float))
    np.testing.assert_array_equal(problem.absorption_cross_section_field.values.ravel(), np.asarray(mix.absorption_xs, dtype=float))


def test_H4_the_solver_reads_the_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoy on the hub's consumed surface (Σ_t × 1.5) MOVES k_inf — so the
    solve reads the hub and not a private re-derivation. Activation leg: a
    hub over the decoy really differs; anti-dud leg: the decoy differs from
    the honest field. The decoy is installed on the CLASS (a fresh hub is
    minted inside the solve), and the property it replaces is the one the
    collision operator consumes."""
    mix = get_mixture("A", "2g")
    baseline = solve_homogeneous_infinite(mix)
    honest = HomogeneousProblem(mix).total_cross_section_field

    def decoy(self: HomogeneousProblem) -> CrossSectionField:
        return CrossSectionField(values=np.asarray(self.mixture.SigT, dtype=float).reshape(self.ng, 1) * 1.5, space=self.space)

    # anti-dud: the decoy is not the honest field
    _require(not np.array_equal(decoy(HomogeneousProblem(mix)).values, honest.values), "the decoy must differ")
    monkeypatch.setattr(HomogeneousProblem, "total_cross_section_field", property(decoy))
    # activation: a hub over the decoy carries it
    _require(np.allclose(HomogeneousProblem(mix).total_cross_section_field.values, honest.values * 1.5), "activation: the hub must serve the decoy")
    moved = solve_homogeneous_infinite(mix)
    _require(abs(moved.k_inf - baseline.k_inf) > 1e-6 * abs(baseline.k_inf), (
        f"k_inf did not move under a ×1.5 collision decoy on the hub ({moved.k_inf} vs {baseline.k_inf}) "
        f"— the solver does not read the hub"))


def test_H5_the_hub_constructs_no_material_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ruled construction spy's twin, on the HUB directly (the solve-level
    row lives in ``test_coda_anchors.py``): touching every consumed object
    constructs no ``MaterialMesh``."""
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    calls: list[int] = []
    raw = MaterialMesh._init_data

    def spy(self, *args, **kwargs):
        calls.append(1)
        return raw(self, *args, **kwargs)

    monkeypatch.setattr(MaterialMesh, "_init_data", spy)
    problem = HomogeneousProblem(get_mixture("A", "2g"))
    for name in _CONSUMED:
        getattr(problem, name)
    problem.multiplication.as_matrix()
    _require(len(calls) == 0, f"the hub constructed {len(calls)} MaterialMesh objects — O1's honest pose fabricates nothing")


def test_the_hub_is_exported_and_is_a_function_space_owner() -> None:
    import orpheus.homogeneous as pkg

    _require("HomogeneousProblem" in pkg.__all__, "the hub is the package's surface")
    problem = pkg.HomogeneousProblem(get_mixture("A", "1g"))
    _require(isinstance(problem.space, FunctionSpace) and problem.ng == 1, "a 1-group hub poses a (1, 1) space")
