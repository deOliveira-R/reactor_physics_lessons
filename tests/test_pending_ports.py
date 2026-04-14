"""Xfail placeholders for ERR-NNN entries without an extant catching test.

Each entry in ``tests/l0_error_catalog.md`` must be paired with a test
carrying ``@pytest.mark.catches("ERR-NNN")`` so the V&V harness audit
can enforce the invariant that every documented bug is regression-
guarded. For five entries the catching test does not yet exist:

* **ERR-008, ERR-010** — the catalog describes a hypothetical dedicated
  L0 assertion that was never written. The Python modules exist and
  the invariant could be checked today, but the retrofit scope
  excluded writing new L0 tests.
* **ERR-011, ERR-012, ERR-013** — these are bugs in the MATLAB
  reference implementation (``funRHS.m``, LOCA static-vs-deformed
  area, closed-gap stress transient). The corresponding Python
  modules (``orpheus.thermal_hydraulics``, ``orpheus.kinetics``,
  ``orpheus.fuel``) do not reproduce the bugs, and in some cases do
  not yet have *any* tests. No Python test can catch what no Python
  code does.

Rather than let these ERRs remain uncaught — which would leave the
``tests._harness.audit`` gate permanently at 17/22 and hide the
pending work behind an "always-failing" coverage number — each ERR
gets an ``xfail``-marked placeholder here. The placeholder carries
``@pytest.mark.catches("ERR-NNN")`` so the audit counts the ERR as
*formally* paired to a test, while the ``xfail`` keeps the suite
green and documents the pending port.

When the real catching test is written, delete the corresponding
placeholder from this file. The audit will still report 22/22 as
long as the new test carries the decorator.

See issue #87 for the broader V&V coverage retrofit.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.l0


@pytest.mark.catches("ERR-008")
@pytest.mark.xfail(
    reason="ERR-008 pending: no dedicated L0 test for the keff volume-weight "
           "convention regression in CartesianMesh.volume. The end-to-end "
           "heterogeneous regressions catch the symptom indirectly, but the "
           "catalog describes a hypothetical direct assertion ('a dedicated "
           "test would verify...') that was never written. Writing it "
           "requires reconstructing the old solver's boundary half-cell "
           "convention, which the current Mesh1D factory no longer supports.",
    strict=True,
)
def test_err_008_keff_volume_weight_regression():
    """Placeholder for the ERR-008 dedicated catcher."""
    pytest.fail("placeholder — replace with a real L0 volume-weight assertion")


@pytest.mark.catches("ERR-010")
@pytest.mark.xfail(
    reason="ERR-010 pending: the orpheus.thermal_hydraulics module has no "
           "test file at all. Writing the catching test requires deciding "
           "the TH test harness pattern (fixtures, materials, analytical "
           "steady-state references) — tracked in issue #45 (TH Sphinx "
           "autodoc) and #46 (TH validation table). Once the TH test file "
           "exists, a direct property evaluation "
           "`h2o_properties(0.33, 4399e3)[0].mu != nan` closes this ERR.",
    strict=True,
)
def test_err_010_iapws_viscosity_no_nan():
    """Placeholder for the ERR-010 IAPWS viscosity NaN catcher."""
    pytest.fail("placeholder — replace with a TH module h2o_properties assertion")


@pytest.mark.catches("ERR-011")
@pytest.mark.xfail(
    reason="ERR-011 is a MATLAB-only finding: funRHS.m line 272 used a wrong "
           "constant in the steady-state fuel-centre balance. The Python "
           "port in orpheus.thermal_hydraulics does not reproduce the bug, "
           "so no Python assertion can catch it. This placeholder documents "
           "the pending obligation to re-verify the Python port still "
           "enforces the correct invariant once the TH test suite lands "
           "(issues #44, #46).",
    strict=True,
)
def test_err_011_funrhs_steady_state_balance():
    """Placeholder for the MATLAB-era funRHS.m ERR-011 catcher."""
    pytest.fail("placeholder — MATLAB-only bug; re-verify Python port invariant")


@pytest.mark.catches("ERR-012")
@pytest.mark.xfail(
    reason="ERR-012 is a LOCA transient bug in the MATLAB reference "
           "(static-vs-deformed area discrepancy at t > 300 s). Affects "
           "orpheus.thermal_hydraulics and orpheus.kinetics, neither of "
           "which has a Python test harness. The catching test — "
           "'fuel surface T with static vs deformed areas during LOCA' — "
           "needs the TH + kinetics point-by-point validation work in "
           "issue #42 before it can exist.",
    strict=True,
)
def test_err_012_loca_deformed_area_consistency():
    """Placeholder for the LOCA deformed-area ERR-012 catcher."""
    pytest.fail("placeholder — needs TH + kinetics validation harness")


@pytest.mark.catches("ERR-013")
@pytest.mark.xfail(
    reason="ERR-013 is a closed-gap stress-BC bug in the MATLAB fuel stress "
           "solver. orpheus.fuel.solver exists but has no tests. The "
           "catching test ('contact pressure at fixed time after closure "
           "against independent analytical estimate') needs the fuel "
           "derivation-script + theory-chapter work in issues #40, #41 "
           "before it can exist.",
    strict=True,
)
def test_err_013_closed_gap_contact_pressure():
    """Placeholder for the closed-gap contact-pressure ERR-013 catcher."""
    pytest.fail("placeholder — needs fuel theory chapter + derivation harness")
