r"""DiffusionMesh construction laws — the diffusion phase space (#290 P7a).

Intrinsic-property gates for the method-mesh type (project standard:
every math-bearing type ships a test of its DEFINING laws). The
defining laws of :class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh`
are its CONSTRUCTION invariants — what makes an instance a solvable
diffusion phase space rather than bare data:

* **realized boundary laws at construction** — ``mesh.bc`` maps each
  boundary face to the realized albedo operator :math:`J^- =
  \mathcal{A} J^+`, with the ruling-3 tag semantics (vacuum = Marshak
  :math:`\mathcal{A}=0`; zero-flux = the Dirichlet idealization
  :math:`\mathcal{A}=-1`) and the reflective infinite-lattice default
  on undeclared faces;
* **law coverage ≡ face coverage** — ``bc`` and the trace derive from
  the ONE ``face_labels`` inventory (the structural invariant that
  retired the P4 boundary operator's coverage validation);
* **the composite carrier identity** — ``full_field_space`` is built
  over THIS mesh's cached ``scalar_trace`` (operator/field identity
  guards ride on it);
* **promotion** — ``from_material_mesh`` re-derives the data block
  bit-identically with NO extra parameters (BCs live on the axes);
* **admission refusals** — multi-D, unsupported BC tag, parameter-less
  albedo (the mesh-less refusal retired at the CS4c coda with the
  fabricated carrier it refused): each fires AT CONSTRUCTION, so "a
  diffusion phase space with unresolved/unrealizable BCs" is
  unrepresentable (the refusals moved here from the P5 solver at P7a).

Foundation tier — software invariants on the phase-space type; the
downstream physics (stencil, k, MMS) is gated in ``test_operators.py``
/ ``test_solver.py`` / ``test_mms.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.diffusion import DiffusionMesh
from orpheus.geometry import BC, CoordSystem, Mesh1D, Mesh2D
from orpheus.transport.mesh.material_mesh import MaterialMesh

pytestmark = [pytest.mark.foundation]

_MATS = {0: get_mixture("A", "2g")}


def _mesh1d(bc_left: BC | None = None, bc_right: BC | None = None) -> Mesh1D:
    return Mesh1D(
        np.linspace(0.0, 10.0, 5), np.zeros(4, dtype=int),
        bc_left=bc_left, bc_right=bc_right,
    )


_PROBE = np.array([1.5, 2.5])   # a J⁺ face slot, (ng,)


class TestRealizedBoundaryLaws:
    def test_bc_realized_per_face_with_ruling3_semantics(self):
        """The tag → 𝒜 table THROUGH the mesh path: vacuum J⁻=0,
        zero-flux J⁻=−J⁺ — realized operators, not stored tags."""
        dm = DiffusionMesh(
            _mesh1d(BC("vacuum"), BC("zero_flux")), _MATS,
        )
        np.testing.assert_array_equal(
            np.asarray(dm.bc["xmin"].apply(_PROBE)), 0.0,
        )
        np.testing.assert_array_equal(
            np.asarray(dm.bc["xmax"].apply(_PROBE)), -_PROBE,
        )

    def test_reflective_and_albedo_actions(self):
        dm = DiffusionMesh(
            _mesh1d(BC("reflective"), BC("albedo", {"albedo": 0.3})), _MATS,
        )
        np.testing.assert_array_equal(
            np.asarray(dm.bc["xmin"].apply(_PROBE)), _PROBE,
        )
        np.testing.assert_allclose(
            np.asarray(dm.bc["xmax"].apply(_PROBE)), 0.3 * _PROBE,
            rtol=1e-15,
        )

    def test_undeclared_faces_default_to_reflective(self):
        """The infinite-lattice convention (the SN default, mirrored):
        an axis with no BC declaration realizes 𝒜 = 1."""
        dm = DiffusionMesh(_mesh1d(), _MATS)
        for face in ("xmin", "xmax"):
            np.testing.assert_array_equal(
                np.asarray(dm.bc[face].apply(_PROBE)), _PROBE,
            )

    def test_law_coverage_equals_face_coverage(self):
        """bc and the trace derive from the ONE face_labels inventory —
        the structural invariant that made the P4 boundary operator's
        coverage validation unrepresentable-state ceremony."""
        dm = DiffusionMesh(_mesh1d(BC("vacuum"), BC("reflective")), _MATS)
        assert set(dm.bc) == set(dm.scalar_trace.face_names)

    def test_sphere_pole_is_not_a_bc_face(self):
        """Curvilinear: the r=0 pole is a regularity condition, not a
        face — bc and trace both carry only the outer face."""
        sphere = Mesh1D(
            np.linspace(0.0, 5.0, 4), np.zeros(3, dtype=int),
            coord=CoordSystem.SPHERICAL,
        )
        dm = DiffusionMesh(sphere, _MATS)
        assert dm.scalar_trace.face_names == ("xmax",)
        assert set(dm.bc) == {"xmax"}


class TestSpaces:
    def test_composite_carrier_rides_the_cached_trace(self):
        dm = DiffusionMesh(_mesh1d(), _MATS)
        ffs = dm.full_field_space
        assert ffs.trace_space is dm.scalar_trace
        assert ffs is dm.full_field_space          # cached — one identity
        # CS4b S4: the interior IS the carrier's cached axis-built mint
        # (the Q2 three-family unification's diffusion leg) — and its
        # metric ACTION is cell volumes broadcast over the group axis
        # (asserted on the action, not the storage: axis-built spaces
        # carry per-axis weights, not one dense array).
        assert ffs.interior_space is dm.bulk_space
        x = np.arange(1.0, 1.0 + float(np.prod(ffs.interior_space.shape))).reshape(
            ffs.interior_space.shape
        )
        np.testing.assert_array_equal(
            ffs.interior_space.apply_metric(x),
            x * np.asarray(dm.volumes)[None, :],
        )


class TestPromotion:
    def test_from_material_mesh_is_a_bit_identical_data_join(self):
        """The promotion re-derives the data block from the carrier's
        own axes/mesh/mat_map/materials — no extra parameters, values
        bit-identical, BCs read from the axes' tags."""
        mm = MaterialMesh(_mesh1d(BC("vacuum"), BC("reflective")), _MATS)
        dm = DiffusionMesh.from_material_mesh(mm)
        assert isinstance(dm, MaterialMesh)
        assert dm.materials is mm.materials
        assert dm.mesh is mm.mesh
        assert dm.axes == mm.axes
        np.testing.assert_array_equal(dm.mat_map, mm.mat_map)
        np.testing.assert_array_equal(dm.volumes, mm.volumes)
        assert dm.ng == mm.ng
        # The tags realized: vacuum on xmin (J⁻ = 0).
        np.testing.assert_array_equal(
            np.asarray(dm.bc["xmin"].apply(_PROBE)), 0.0,
        )


class TestAdmissionRefusals:
    def test_multi_d_mesh_is_refused_at_construction(self):
        mesh2d = Mesh2D(
            edges_x=np.array([0.0, 1.0, 2.0]),
            edges_y=np.array([0.0, 1.0]),
            mat_map=np.zeros((2, 1), dtype=int),
        )
        with pytest.raises(ValueError, match="1-D"):
            DiffusionMesh(mesh2d, _MATS)

    def test_multi_d_promotion_is_refused(self):
        mesh2d = Mesh2D(
            edges_x=np.array([0.0, 1.0, 2.0]),
            edges_y=np.array([0.0, 1.0]),
            mat_map=np.zeros((2, 1), dtype=int),
        )
        with pytest.raises(ValueError, match="1-D"):
            DiffusionMesh.from_material_mesh(MaterialMesh(mesh2d, _MATS))

    def test_unsupported_bc_kind_is_refused_with_the_supported_list(self):
        # "white" is DELIBERATELY absent: at P1 it coincides with
        # reflective (the P3 realizer's coincidence note).
        with pytest.raises(ValueError, match="'white'.*Supported.*albedo"):
            DiffusionMesh(_mesh1d(BC("white"), BC("vacuum")), _MATS)

    def test_albedo_without_parameter_is_refused(self):
        with pytest.raises(ValueError, match="albedo.*parameter"):
            DiffusionMesh(_mesh1d(BC("albedo"), BC("vacuum")), _MATS)
