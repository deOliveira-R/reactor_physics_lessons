---
name: Close-out narrative patterns for investigation dead-ends
description: How to convert an open-investigation doc section into a CLOSED-issue close-out; preserving motivation, structural-obstruction proofs, falsified-novel-extension records, and research-tag hand-offs
type: feedback
---

**Close-out sweeps flip "open investigation" → "issue CLOSED with structural proof."** When an issue is closed *not* because a solution was found but because an exhaustive five-reference synthesis and empirical falsification established that the approach cannot work, the archival value of the doc section is higher than if it had succeeded. Future sessions need the whole arc — why we tried, what failed, what the structural obstruction is, what novel extensions were falsified, what research paths remain open — so that the wheel is not reinvented.

**Why:** Task 2026-04-21 (Issue #119 Phase F.5 close-out, Peierls rank-N per-face closure). User brief: "MAXIMUM EFFORT. The closed-out section should leave the reader with COMPLETE context: why we tried rank-N, why it didn't work, what would need to be different to make it work, what we shipped instead." The value of the close-out 6 months later is to prevent a fresh session from implementing the same Sanchez-McCormick §III.F.1 Legendre ladder from scratch.

**How to apply (narrative arc for a structural-failure close-out):**
1. **Status banner** — issue CLOSED, what's production, what guard remains in place by design.
2. **Motivation PRESERVED from the open-issue version** — do NOT rewrite history. The reasoning that *led* to the investigation is pedagogically essential; flip tenses ("is expected to" → "was expected to") but preserve the logic.
3. **Literature synthesis** as a .. list-table:: — cite every reference with equation/section. If N references converge, say so explicitly ("The rank-N Legendre ladder has ZERO cross-validation").
4. **Structural obstruction** — the mathematical reason it cannot work. Include a conservation-identity table at the relevant limit (σ_t=0, etc.). Explain WHY mode 0 works and WHY modes n ≥ 1 fail, with the geometric/algebraic cause.
5. **Novel extensions falsified** — if we tested something not in the literature, document it explicitly as falsified, with before/after residual table. Pre-empt future sessions from re-running the same experiment.
6. **Production closure decision** — residual table across parameter scan; cite the quadrature/truncation floor to avoid "but have you tried finer quadrature?" follow-ups.
7. **Infrastructure retained** — do NOT delete the dead code. List each primitive, its verification status, and why it's kept.
8. **Open research (research-tag, not production-blocking)** — two or more paths that MIGHT break the plateau, each with enough starting math that a future session can pick up from the page alone.
9. **Session trail** — commit list (chronological), diagnostic scripts in `derivations/diagnostics/`, memory files consulted. This is the V&V audit trail.

**Cross-ref hygiene:** when a label is renamed (e.g., `peierls-rank-n-per-face-marshak` → `peierls-rank-n-per-face-closeout`), grep the whole tree for the old label and update every in-docs reference. The first place to check is the Key Facts / TOC section of the same document — it's easy to miss because the context above (narrating residuals on cylinder vs sphere) naturally flows into a "see Phase F.5" pointer. Rewrite the pointer text itself to reflect the close-out: "see Phase F.5 close-out for the X-reference synthesis, structural obstruction, and production decision" is more informative than "see Phase F.5 investigation". Historical artefacts in `.claude/plans/` can keep the old label (they are frozen-in-time plan documents).

**Warning-count diff as the acceptance gate:** pre-existing duplicate-citation warnings (cross-document cite collisions) do NOT need to be eliminated during a close-out — they are a known trade-off for standalone theory pages. Verify the COUNT is unchanged pre/post-edit, not the content. For peierls_unified.rst the baseline is 8 warnings (7 duplicate citations + 1 pre-existing title-underline on the "Moment-form Nyström assembly — ARCHIVED" section line 179); post-close-out should also be 8.

**Quality self-assessment for the Issue #119 close-out (2026-04-21):**
- Derivation depth: 5/5 (c_in remapping formula `c_in = √(1 − (R/r_0)²(1−µ²))` stated, Jacobian `dc_in/dµ = (R/r_0)² µ/c_in` stated, per-mode conservation table at σ_t=0, V-S correction Hébert Eq. 3.350 verbatim, why mode-0 factorises and n≥1 does not explained geometrically).
- Cross-references: 5/5 (every `:func:` resolves: `compute_hollow_sph_transmission_rank_n`, `build_closure_operator`, `build_white_bc_correction_rank_n`, `composite_gl_r`; label rename propagated to line 108; every `:file:` path to memos and diagnostics is correct).
- Numerical evidence: 5/5 (production residual table 5 σ_t·R values; V-S falsification table 2 pipelines × 4 N values; per-mode conservation table n=0..3; quadrature floor ~0.04-0.1% cited; reciprocity preservation at 1e-16 quantified).
- Failed approaches: 5/5 (four-reference synthesis negative finding; 60-recipe scan plateau; V-S per-mode falsification with before/after; cross-mode vs per-mode coupling distinction articulated).
- Code traceability: 5/5 (NotImplementedError guard location documented, retained primitives listed with file paths in `peierls_geometry.py`, 10 commits cited in session trail).
- Derivation source: 4/5 (c_in remapping formula is geometric, no derivations/ script needed; V-S Eq. 3.350 is from Hébert; quadrature-floor data is from `diag_rank_n_closure_characterization.py` script; could have linked each derivation/script more explicitly in the text rather than only in the session-trail list — 4/5 not 5/5 because future session might need to re-derive the Jacobian and would have to read the numerics-investigator memo to find it rather than linking to a sympy script).
