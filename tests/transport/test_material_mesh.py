r"""MaterialMesh — the method-agnostic mesh + materials data carrier.

:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` is the
"mesh + materials" middle type between geometry ``Mesh1D`` (material
*ids*, no cross sections) and ``SNMesh`` (mesh + materials + quadrature +
sweep machinery).  These tests pin its **intrinsic data contract** (the
``coding-elegance`` standard: a type ships a test of its defining
invariants) and the **data/behavior split** with ``SNMesh(MaterialMesh)``:

* the carrier holds mesh + materials and exposes the method-agnostic data
  accessors (``ng`` / ``volumes`` / ``volume_measure`` / ``ndim`` /
  ``spatial_shape`` / ``mat_map`` / ``material_xs_field``);
* ``ng`` consistency is enforced at construction;
* ``SNMesh`` **is-a** ``MaterialMesh`` (Liskov) — every carrier accessor
  works on an ``SNMesh``, and its data block is bit-identical to a
  standalone ``MaterialMesh`` built from the same inputs;
* ``SNMesh.from_material_mesh`` promotes a carrier to a solvable phase
  space (the homogenization re-solve path).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.mesh import (
    InconsistentMaterialsError,
    MaterialMesh,
    MaterialXSField,
)
from orpheus.sn.mesh.augmented_mesh import SNMesh

pytestmark = pytest.mark.foundation


def _mix(sig_t, *, ng):
    """A minimal balanced-free synthetic Mixture with a given ng."""
    sig_t = np.asarray(sig_t, dtype=float)
    return Mixture(
        SigC=0.5 * sig_t, SigL=np.zeros(ng), SigF=np.zeros(ng),
        SigP=np.zeros(ng), SigT=sig_t,
        SigS=[csr_matrix(np.diag(0.5 * sig_t))], Sig2=[csr_matrix((ng, ng))],
        chi=np.zeros(ng), eg=None,
    )


def _two_material_mesh(ng=2):
    mats = {0: _mix([1.0, 1.5][:ng], ng=ng), 1: _mix([1.2, 1.8][:ng], ng=ng)}
    mesh = Mesh1D(
        edges=np.linspace(0.0, 5.0, 6),
        mat_ids=np.array([0, 0, 1, 1, 0]),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    return mesh, mats


# ── Carrier data contract ─────────────────────────────────────────────

def test_material_mesh_holds_mesh_and_materials():
    mesh, mats = _two_material_mesh()
    mm = MaterialMesh(mesh, mats)
    # Un-weld arc (R20/R21): the attribute is the parsed stage-1
    # DECLARATION, not an alias of the caller's dict — same entries by
    # identity, read-only mapping surface (the old aliasing let callers
    # mutate the mesh's materials after construction).
    from orpheus.data.materials import Materials
    assert isinstance(mm.materials, Materials)
    assert all(mm.materials[k] is mats[k] for k in mats)
    assert mm.ng == 2
    assert mm.ndim == 1
    assert mm.spatial_shape == (5,)
    np.testing.assert_array_equal(mm.mat_map, mesh.mat_ids)
    np.testing.assert_array_equal(mm.volumes, mesh.volumes)


def test_material_mesh_volume_measure_matches_mesh():
    mesh, mats = _two_material_mesh()
    mm = MaterialMesh(mesh, mats)
    mu = mm.volume_measure
    np.testing.assert_array_equal(mu.weights, mesh.volume_measure.weights)


def test_material_mesh_builds_xs_field():
    mesh, mats = _two_material_mesh()
    mm = MaterialMesh(mesh, mats)
    field = mm.material_xs_field()
    assert isinstance(field, MaterialXSField)
    # per-cell SigT view follows the mat_map (cell 2 is material 1).
    sig_t = field.total_cross_section  # (ng, nx)
    assert sig_t.shape == (2, 5)
    np.testing.assert_allclose(sig_t[:, 2], mats[1].SigT)
    np.testing.assert_allclose(sig_t[:, 0], mats[0].SigT)


# ── ng-consistency invariant (the defining law) ───────────────────────

def test_inconsistent_ng_raises_at_construction():
    mesh = Mesh1D(
        edges=np.linspace(0.0, 2.0, 3), mat_ids=np.array([0, 1]),
        coord=CoordSystem.CARTESIAN,
    )
    mats = {0: _mix([1.0, 1.5], ng=2), 1: _mix([1.0], ng=1)}
    with pytest.raises(InconsistentMaterialsError, match="uniform ng"):
        MaterialMesh(mesh, mats)


def test_missing_material_id_raises():
    mesh = Mesh1D(
        edges=np.linspace(0.0, 2.0, 3), mat_ids=np.array([0, 7]),
        coord=CoordSystem.CARTESIAN,
    )
    with pytest.raises(ValueError, match="references material ids"):
        MaterialMesh(mesh, {0: _mix([1.0, 1.0], ng=2)})


# ── Liskov: SNMesh IS-A MaterialMesh ──────────────────────────────────

def test_snmesh_is_a_material_mesh():
    assert issubclass(SNMesh, MaterialMesh)


def test_snmesh_data_block_bit_identical_to_standalone_carrier():
    """An SNMesh's inherited data block matches a standalone MaterialMesh
    built from the same mesh + materials (the split is bit-identical)."""
    mesh, mats = _two_material_mesh()
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    snm = SNMesh(mesh, quad, mats)
    mm = MaterialMesh(mesh, mats)
    assert snm.ng == mm.ng
    assert snm.ndim == mm.ndim
    assert snm.spatial_shape == mm.spatial_shape
    np.testing.assert_array_equal(snm.mat_map, mm.mat_map)
    np.testing.assert_array_equal(snm.volumes, mm.volumes)
    np.testing.assert_array_equal(
        snm.volume_measure.weights, mm.volume_measure.weights,
    )
    # Every carrier accessor is callable on the SNMesh (substitutability).
    assert isinstance(snm.material_xs_field(), MaterialXSField)


# ── from_material_mesh promotion (the data/behavior join) ─────────────

def test_from_material_mesh_round_trips_to_solvable_snmesh():
    mesh, mats = _two_material_mesh()
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    mm = MaterialMesh(mesh, mats)
    snm = SNMesh.from_material_mesh(mm, quad)
    assert isinstance(snm, SNMesh)
    # carries the carrier's data verbatim …
    np.testing.assert_array_equal(snm.mat_map, mm.mat_map)
    np.testing.assert_array_equal(snm.volumes, mm.volumes)
    assert snm.materials is mm.materials
    # … plus the SN method layer (quadrature + streaming stencil).
    assert snm.quad is quad
    assert snm.reduced is not None


# ── Materials declaration vs assignment ──────────────────────────────
# Until the CS4c coda (2026-09-08) this section pinned the defining
# invariants of ``MaterialMesh.from_materials`` — the mesh-less one-cell
# carrier of the infinite homogeneous medium. The factory retired with the
# coda (the homogeneous problem poses on its own space and builds no
# carrier); the ONE live guard those rows exercised survives here with its
# witness on a genuine one-cell carrier.


def test_a_materials_declaration_must_cover_every_referenced_id():
    """A cell referencing material id 0 with a declaration lacking key 0
    fails loud at construction (parse, don't validate downstream) — the
    ``Materials.restrict`` guard, witnessed on a genuine one-cell carrier."""
    mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.zeros(1, dtype=int))
    with pytest.raises(ValueError, match="references material ids"):
        MaterialMesh(mesh, {3: _mix([1.0, 1.0], ng=2)})
