"""Generate the 2-D octant-sweep equivalence snapshots.

Run::

    python -m tests.sn.regression._generate_2d_octant_snapshots
    python -m tests.sn.regression._generate_2d_octant_snapshots --case 03_l7_trap_mixedBC_2g_het_LS4
    python -m tests.sn.regression._generate_2d_octant_snapshots --list

Each snapshot writes ``snapshots/2d_octant_equivalence_<case_id>.npz``
containing (Wave O #208 O.4b Phase E — bare-sweep schema):

* ``angular_flux`` — ``(N, ng, nx, ny)`` float64
* ``scalar_flux`` — ``(ng, nx, ny)`` float64
* ``face_xmin`` — ``(N, ng, ny)`` float64 (post-sweep boundary face view)
* ``face_xmax`` — ``(N, ng, ny)`` float64
* ``face_ymin`` — ``(N, ng, nx)`` float64
* ``face_ymax`` — ``(N, ng, nx)`` float64
* ``case_id`` — np.array(case.case_id)
* ``case_description`` — np.array(case.description)
* ``failure_mode`` — np.array(case.failure_mode)
* ``generator_commit`` — short SHA

Snapshot grounding (Wave O #208 O.4b Phase E migration)
=======================================================

This script generates from the CURRENT (bare) ``_sweep_jacobi``
with the external ``reflect_outflow_into_inflow`` injected before each
sweep — exactly the production iteration shape, and exactly what the
companion test at
:file:`tests/sn/sweep/cartesian_2d/test_2d_octant_sweep_equivalence.py`
runs.  Both the generator and the test drive the sweep through the SAME
two helpers — :func:`combine_source` and :func:`run_sweeps`, imported
below from the test module — so generator and test CANNOT drift
(coding-elegance Pattern 2, single source of truth).

* Vacuum cases (01/04/06) regenerate bit-identical (within nulp=64) to
  the previously-committed snapshots: the reflect inject is a provable
  no-op for ``B = 0``, and the bare sweep ≡ the legacy sweep for zero
  inflow.  A divergence on regeneration would mean the bare sweep
  changed the vacuum path — a bug to investigate, NOT to commit.
* Reflective cases (02/03/05) regenerate to NEW values: the legacy
  intra-sweep (Gauss-Seidel) reflection was replaced by the inter-sweep
  (Jacobi-like) external reflect, so the per-sweep values change (same
  converged fixed point, slower rate).  These NEW values are the
  migrated baselines, grounded by case 7 (the structurally-independent
  closed-form reflective anchor in the test module).

Schema migration: the legacy schema stored the full interior-edge
``psi_x_post`` ``(N, ng, nx+1, ny)`` / ``psi_y_post`` ``(N, ng, nx,
ny+1)`` arrays (the legacy AngularBoundaryFlux ``xmin_xmax_buf`` /
``ymin_ymax_buf`` fields).  Those fields no longer exist: the L2
AngularBoundaryFlux persists ONLY the four boundary face slices; the interior
edges are EPHEMERAL inside ``_sweep_jacobi``.  The test only ever
compared the boundary slices, so this script now stores exactly the
four persisted face views.

Parity with :mod:`tests.sn.regression._generate_snapshots`:

* Same ``snapshots/`` directory.
* Same ``--case`` / ``--list`` CLI.
* Same ``generator_commit`` metadata.

Drift protocol — when this generator's output legitimately changes
(e.g. an upstream Mesh2D refactor changes the canonical edges/center
spacing): (1) audit why the new output is correct, (2) re-run the
generator, (3) commit both new snapshots AND the audit narrative in
the same commit.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from tests.sn.sweep.cartesian_2d.test_2d_octant_sweep_equivalence import (
    CASES,
    SNAPSHOT_DIR,
    OctantEquivalenceCase,
    _snapshot_path,
    run_sweeps,
)


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def generate_one(
    case: OctantEquivalenceCase, *, sha: str | None = None,
) -> Path:
    """Run the case under the CURRENT bare sweep + external reflect and write .npz.

    Drives the sweep through :func:`run_sweeps` — the SAME helper the
    companion test uses — so the generator and the test are guaranteed
    to produce identical outputs (coding-elegance Pattern 2).  After
    ``case.n_sweeps`` reflect-then-sweep iterations, the post-sweep
    boundary face state lives in ``inputs.boundary_flux``; we snapshot
    the four persisted face views (the cross-iteration stateful link)
    alongside the final angular / scalar flux.
    """
    inputs = case.builder()
    angular_flux, scalar_flux = run_sweeps(inputs, case.n_sweeps)

    bf = inputs.boundary_flux
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = _snapshot_path(case.case_id)

    payload = dict(
        angular_flux=np.asarray(angular_flux, dtype=np.float64),
        scalar_flux=np.asarray(scalar_flux, dtype=np.float64),
        face_xmin=np.asarray(bf.face_view("xmin"), dtype=np.float64),
        face_xmax=np.asarray(bf.face_view("xmax"), dtype=np.float64),
        face_ymin=np.asarray(bf.face_view("ymin"), dtype=np.float64),
        face_ymax=np.asarray(bf.face_view("ymax"), dtype=np.float64),
        case_id=np.array(case.case_id),
        case_description=np.array(case.description),
        failure_mode=np.array(case.failure_mode),
        generator_commit=np.array(sha or _git_short_sha()),
    )
    np.savez_compressed(out, **payload)
    return out


def generate_all(case_ids: list[str] | None = None) -> list[Path]:
    sha = _git_short_sha()
    written = []
    for case in CASES:
        if case_ids and case.case_id not in case_ids:
            continue
        path = generate_one(case, sha=sha)
        written.append(path)
        print(f"wrote  {path.relative_to(Path.cwd())}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2-D octant-sweep equivalence snapshots from the "
            "CURRENT bare sweep + external reflect."
        ),
    )
    parser.add_argument(
        "--case", action="append", default=None,
        help="Restrict to a specific case_id (repeatable).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available cases and exit.",
    )
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"  {case.case_id:50s}  {case.description}")
        return
    written = generate_all(case_ids=args.case)
    print(f"generated {len(written)} snapshot(s) in {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
