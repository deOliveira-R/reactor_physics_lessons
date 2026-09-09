r"""Carrier-admission gates: ``mesh is None`` has ONE meaning (S7 G7.1–G7.3, re-posed at the CS4c coda).

History. CS4b S7 (2026-08-24) un-welded a sentinel that carried TWO
meanings — the mesh-less infinite-medium 1-cell carrier
(``MaterialMesh.from_materials``) and the d≥3 axis-native carrier — with
typed refusals (G7.1 the SN promotion refusal as a real ``raise`` under the
canonical ``-O``; G7.2 one ``areas`` message per arm; G7.3 the two states
discriminated by ``ndim``). The CS4c coda (C1 ``5caad3d6``, C2 this commit)
retired the fabricated carrier and its factory — the infinite-medium
problem poses on ``HomogeneousProblem.space`` and builds no carrier — so
the arms that served it (the SN promotion refusal, the ``areas`` "no faces
at all" arm, the diffusion bounded-geometry refusal) had NO reachable
input left and retired with it (memo H-1: `[M]` no producer of
``mesh is None and ndim == 1`` exists — ``SNMesh.from_axes`` synthesizes
a legacy adapter at d≤2 and leaves ``mesh = None`` only at d≥3;
``MaterialMesh.__init__`` always carries a mesh).

What is gated now:

G7.1 — a real carrier promotes (the positive leg survives; its negative
leg's subject is gone).
G7.2 — the two surviving ``areas`` arms each name their own case, pinned
by their shortest distinctive fragments; collapsing them reddens a row.
G7.3 — the SINGLETON law that replaced the discrimination law: every
d≤2 constructor carries a mesh, and ``mesh is None`` ⟹ ``ndim ≥ 3``.
This is the THEOREM the retirement rests on (``coding-standards``: what
kept the duplicate arms distinguishable is now the sole guarantor, and
it owes a witness) — a positive control (the d=3 carrier constructs with
``mesh is None``) and the d=1 / d=2 axis-native constructions that
synthesize their adapter.
R1 — the retirement is UNSPELLABLE: no class in the carrier hierarchy
has a ``from_materials``; the homonym ``EnergyAxis.from_materials`` (the
energy-arm rule, gated at ``test_kernels.py``) SURVIVES.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.diffusion import DiffusionMesh
from orpheus.geometry import BC, CoordSystem, Mesh1D, Mesh2D
from orpheus.numerics.axis import EnergyAxis
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.mesh.axis import AxisMesh
from orpheus.transport.mesh.material_mesh import MaterialMesh
from tests.sn._test_helpers import placeholder_materials

pytestmark = [pytest.mark.foundation]


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _legacy_1d() -> MaterialMesh:
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, 6), mat_ids=np.zeros(5, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    return MaterialMesh(mesh, placeholder_materials(ng=2))


def _legacy_2d() -> MaterialMesh:
    mesh2d = Mesh2D(
        edges_x=np.linspace(0.0, 1.0, 3),
        edges_y=np.linspace(0.0, 2.0, 4),
        mat_map=np.zeros((2, 3), dtype=int),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    return MaterialMesh(mesh2d, placeholder_materials(ng=2))


def _axes(*extents_and_cells: tuple[float, int]) -> tuple[AxisMesh, ...]:
    return tuple(
        AxisMesh(edges=np.linspace(0.0, ext, n + 1)) for ext, n in extents_and_cells
    )


def _sn_from_axes(axes: tuple[AxisMesh, ...]) -> SNMesh:
    quadrature = (
        Quadrature.gauss_legendre(4) if len(axes) == 1
        else Quadrature.level_symmetric(sn_order=4)
    )
    return SNMesh.from_axes(axes, quadrature, placeholder_materials(ng=2))


def _d3_sn() -> SNMesh:
    return _sn_from_axes(_axes((1.0, 2), (2.0, 3), (3.0, 2)))


class TestG71PromotionRefusal:
    def test_a_real_carrier_promotes(self):
        """Positive leg (vv #11): the legacy 1-D carrier promotes to a
        solvable SN phase space. (Its negative leg — the mesh-less 1-cell
        carrier refused with a typed ``ValueError`` — retired with the
        carrier at the CS4c coda: nothing can be built for it to refuse.)"""
        sn = SNMesh.from_material_mesh(
            _legacy_1d(), Quadrature.gauss_legendre(4),
        )
        if sn.ng != 2 or sn.spatial_shape != (5,):
            pytest.fail("promotion must preserve the carrier's data block")


class TestG72AreasNamesItsOwnCase:
    def test_axis_native_arm_names_the_carrier(self):
        """The d=3 axis-native SN carrier (mesh is None, ndim=3) — the
        pre-S7 message blamed '2-D meshes' here, falsely."""
        with pytest.raises(AttributeError, match="axis-native"):
            _d3_sn().areas

    def test_legacy_2d_arm_keeps_its_true_message(self):
        with pytest.raises(AttributeError, match="not defined for 2-D"):
            _legacy_2d().areas

    def test_the_two_surviving_arms_differ(self):
        """Collapsing the two arms back to one message reddens this."""
        messages = []
        for carrier in (_d3_sn(), _legacy_2d()):
            with pytest.raises(AttributeError) as exc:
                carrier.areas
            messages.append(str(exc.value))
        _require(messages[0] != messages[1], "the two areas arms must name different cases")


class TestG73MeshNoneHasOneMeaning:
    r"""The singleton law: ``mesh is None`` ⟹ ``ndim ≥ 3`` — a theorem of the
    constructors, and the sole guarantor that the retired arms had no input.

    Positive control: the d=3 axis-native carrier really is mesh-less
    (``from_axes`` leaves ``mesh = None`` only at d≥3). Then every d≤2
    construction path carries a mesh: the legacy 1-D and 2-D constructors
    by their signature, and ``from_axes`` at d=1 / d=2 by SYNTHESIS of the
    legacy adapter (``legacy_mesh_from_axes``). Mutating the synthesis to
    leave ``mesh = None`` at d≤2 reddens this class and nothing else in the
    tree pins it — which is why it is here."""

    def test_positive_control_the_d3_carrier_is_mesh_less(self):
        d3 = _d3_sn()
        _require(d3.mesh is None and d3.ndim == 3, "the d=3 axis-native carrier must spell mesh is None")

    @pytest.mark.parametrize("build", ["legacy_1d", "legacy_2d", "axes_1d", "axes_2d"])
    def test_every_d_le_2_construction_carries_a_mesh(self, build: str):
        carrier = {
            "legacy_1d": _legacy_1d,
            "legacy_2d": _legacy_2d,
            "axes_1d": lambda: _sn_from_axes(_axes((1.0, 4))),
            "axes_2d": lambda: _sn_from_axes(_axes((1.0, 2), (2.0, 3))),
        }[build]()
        _require(carrier.ndim <= 2, "precondition: a d≤2 carrier")
        _require(
            carrier.mesh is not None,
            f"{build}: a d={carrier.ndim} carrier spelled mesh is None — the singleton law broke",
        )
        expected = Mesh1D if carrier.ndim == 1 else Mesh2D
        _require(
            isinstance(carrier.mesh, expected),
            f"{build}: the adapter is a {type(carrier.mesh).__name__}, not a {expected.__name__}",
        )

    def test_a_one_d_carrier_has_faces(self):
        """The retired 'no faces at all' arm's complement: every 1-D carrier
        answers ``areas`` (a Mesh1D concept), including a one-cell one."""
        one_cell = MaterialMesh(
            Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.zeros(1, dtype=int)),
            placeholder_materials(ng=2),
        )
        _require(one_cell.areas.shape == (2,), "a one-cell 1-D carrier has two faces")


class TestR1TheFactoryIsUnspellable:
    def test_no_carrier_class_has_a_from_materials(self):
        """The retirement is structural: the name is absent from every class in
        the carrier hierarchy (a subclass could not have inherited it either).
        Re-adding the factory anywhere in the hierarchy reddens this."""
        for cls in (MaterialMesh, SNMesh, DiffusionMesh):
            _require(
                not hasattr(cls, "from_materials"),
                f"{cls.__name__}.from_materials exists — the fabricated carrier's factory is back",
            )

    def test_the_energy_axis_homonym_survives(self):
        """``EnergyAxis.from_materials`` shares only the NAME (the energy-arm
        rule hoisted at CS4a K1; gated at ``test_kernels.py``) — a name-keyed
        sweep must not have taken it."""
        axis = EnergyAxis.from_materials(placeholder_materials(ng=2).values())
        _require(axis.shape == (2,), "EnergyAxis.from_materials must still mint the energy axis")

    def test_nothing_in_the_tree_can_build_a_mesh_less_one_d_carrier(self):
        """The retired arms' precondition, stated as the singleton law's
        corollary on the one surface that admits an axis tuple: ``from_axes``
        at d=1 carries a mesh."""
        sn = _sn_from_axes(_axes((1.0, 1)))
        _require(sn.ndim == 1 and sn.mesh is not None, "a d=1 from_axes carrier must carry its adapter")
