.. _theory-index:

======================
Theory and Derivations
======================

The ORPHEUS theory corpus. It is **not** a summary of the code — it is the
knowledge base the code is written *from*: full derivations, the design
rationale (not just *what* but *why*), the conventions, the gotchas, and what
verifies each equation.

**Why read this before the literature.** A textbook can tell you what the
diamond closure is. It cannot tell you *this project's conventions*, *which
test pins this equation*, *what breaks when the* :math:`\alpha`-*dome goes
negative*, or *why the projection frame is Petrov-Galerkin and not Galerkin*.
Those are the product; the standard derivations are connective tissue. One
fact about the canon makes the point concrete: grepping all 122 pages of
Hébert Ch. 3 and all 80 pages of Stacey Ch. 9 for
``verification | benchmark | manufactured solution`` returns **zero hits in
both** — and S\ :sub:`N` is a method where *wrong code converges*.


The parts
=========

The tree mirrors the code's dependency layering (``data → geometry →
numerics → transport → methods/<m>``), so knowing where the code lives tells
you where its theory lives.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Part
     - What it documents
     - Read it when
   * - :ref:`theory-conventions`
     - **Read first.** Symbol, normalization and indexing conventions, and
       the crosswalk to the literature's mutually-contradictory ones.
     - Importing any equation from a paper or textbook; debugging an answer
       that is wrong by a constant factor.
   * - :ref:`theory-foundations`
     - The math every method shares: the operator algebra
       :math:`A = L + C - S - N_{2n} - B`, frames and projection, the
       boundary law,
       cross-section data, measures, geometry, and the infinite-medium
       baseline.
     - Touching any solver. The operator algebra is the spine — the
       single highest-value page in the corpus.
   * - :ref:`theory-transport-methods`
     - The **production solvers** you run for analysis: S\ :sub:`N`,
       collision probability, method of characteristics, Monte Carlo, and
       the diffusion (P1) limit.
     - Modifying or extending a production solver.
   * - :ref:`theory-reference-solvers`
     - The **continuous reference solvers** that supply the analytical and
       semi-analytical truth values the verification suite consumes — plus
       reserved slots for methods queued for implementation.
     - Writing a new reference, debugging a verification gap, or extending a
       truth set.
   * - :ref:`theory-verification`
     - The **V&V machinery and evidence**: the L0..L3 ladder and evidence
       taxonomy, the test-harness tagging contract, the cross-method (L4)
       protocol, the reference-solution contract, and the auto-generated
       per-equation verification matrix.
     - Designing or tagging a test, auditing coverage, or asking "what
       pins this equation?"

The production/reference split is **load-bearing**, not cosmetic. The two
serve different masters:

- **Production solvers are tuned for scale.** They must be fast on realistic
  problems: real mesh refinement, multi-region pin cells, full multi-group
  cross-section data. Their error budget is a discretisation error that
  decreases under refinement.
- **Reference solvers are tuned for accuracy.** They must be
  arbitrary-precision evaluable on the **specialised** problems that admit
  analytical or semi-analytical solutions. Their error budget is a
  numerical-quadrature floor, typically :math:`10^{-8}` or better.

Mixing the two confuses readers and corrupts the V&V vocabulary. This page is
the canonical entry point that prevents the confusion.

.. toctree::
   :maxdepth: 2
   :caption: The corpus

   conventions/index
   foundations/index
   methods/index
   references/index
   verification/index


Cross-cutting
=============

The shared vocabulary.

.. toctree::
   :maxdepth: 2
   :caption: Cross-cutting

   glossary


Multiphysics
============

Support solvers outside the neutron-transport core. Their long-term home is
undecided — they may be extracted from this repository — so they are
deliberately carried unrestructured rather than filed into a part.

.. toctree::
   :maxdepth: 1
   :caption: Multiphysics

   thermal_hydraulics
   fuel_behaviour
   reactor_kinetics
