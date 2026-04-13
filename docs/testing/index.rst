Testing & V&V
=============

ORPHEUS enforces a four-level verification & validation ladder: **L0**
term verification (hand calc vs code), **L1** equation verification
(analytical / MMS with asserted convergence order), **L2** integration
(multi-group heterogeneous, cross-method), and **L3** validation
(experimental comparison). Every test declares the rung it lives on via
``pytest.mark.l0`` / ``l1`` / ``l2`` / ``l3`` and, for equation-level
tests, the Sphinx equation labels it exercises via
``pytest.mark.verifies``.

This chapter documents the test-harness architecture, the tagging
convention, and the audit workflow.

.. toctree::
   :maxdepth: 2

   architecture
