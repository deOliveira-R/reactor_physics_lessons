# ORPHEUS — Open Reactor Physics Educational University System

## Knowledge Sources

This project has two complementary knowledge systems. Use **both** before making changes.

### Sphinx Documentation (`docs/`)
The `docs/` directory contains theory, derivations, and equations that explain the physics behind each solver. These documents cross-reference the code via `:func:`, `:class:`, and `:mod:` directives.

- **Before implementing or modifying a solver**: read the relevant theory page in `docs/theory/` to understand the physics, equations, and design decisions.
- **Before implementing a new module**: check if a theory page exists. If not, write one as part of the implementation — document the equations and derivations alongside the code.
- **After modifying equations or algorithms**: update the corresponding Sphinx page. Build with `python -m sphinx -b html docs docs/_build/html` and verify zero warnings.
- **Documentation tasks**: use the **archivist** agent (`.claude/agents/archivist/`) for all Sphinx work. It knows the project conventions, quality standards, and derivation source-of-truth rules.

### GitNexus Knowledge Graph
GitNexus indexes the code structure (1000+ nodes, 2400+ edges). It knows what calls what, but NOT the physics. Use it for:
- **Impact analysis** before editing any symbol
- **Code navigation** (callers, callees, execution flows)
- **Refactoring safety** (rename, detect changes)

### The Two Together
- **GitNexus** tells you WHAT the code does and what depends on it
- **Sphinx docs** tell you WHY the code does it and the equations it implements
- Before any non-trivial change: check both. GitNexus for blast radius, docs for physics correctness.

---

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architecture)
- Define BOTH execution + verification steps
- If something breaks → STOP and re-plan
- Write detailed specs to remove ambiguity

### 2. Specialized Agent Fleet

Five custom agents in `.claude/agents/` — use them instead of generic subagents for their domains. Each has a `lessons.md` that compounds across sessions.

| Agent | Invoke when | Key rule |
|-------|-------------|----------|
| **archivist** | Writing/reviewing Sphinx docs | Sphinx-as-brain: full derivations, not summaries. Demands derivation scripts before archiving. |
| **qa** | Reviewing code, validating claims | 6 AI failure modes checklist. Demands multi-group + heterogeneous tests. |
| **numerics-investigator** | Solver gives wrong answers | 7-step diagnostic cascade. Writes scripts in `derivations/diagnostics/`. Promotes to tests. |
| **literature-researcher** | Need equations from papers | Source priority by topic. Maps notation to ORPHEUS. Never references export-controlled codes. |
| **test-architect** | Planning verification BEFORE implementation | Analytical solution catalog. Failure mode coverage matrix. 1-group is degenerate. |

**Dispatch rules:**
- Documentation tasks → **archivist** (not generic subagent)
- "Is this correct?" → **qa**
- "Why is this broken?" → **numerics-investigator**
- "What does Bailey Eq. 50 say?" → **literature-researcher**
- "What tests do we need for feature X?" → **test-architect**
- Routine code search / exploration → built-in Explore agent (lightweight)

**After every specialized agent invocation**: the main agent must review the output with full session context before committing. Sub-agents lack conversation history.

### 3. Improvement Tracking and Self-Improvement

- Every module has an `IMPROVEMENTS.md` with tracked items (`XX-YYYYMMDD-NNN`)
- TODOs exist in exactly TWO places: the tracker AND the code location (with matching tracking number)
- After ANY bug or improvement discovery → add to IMPROVEMENTS.md immediately
- Items flow: OPEN → IMPL → DONE (DONE requires Sphinx documentation)
- At session start: read IMPROVEMENTS.md for context
- At session end: verify no orphan TODOs exist outside the tracker

### 4. Verification Before Done
- Never mark done without proof
- Run tests, check logs, simulate real usage
- Compare expected vs actual behavior
- Ask: "Would a senior engineer approve this?"

### 5. Demand Elegance
- Ask: "Is there a simpler / cleaner way?"
- Avoid hacky or temporary fixes
- Optimize for long-term maintainability
- Skip overengineering for small fixes

### 6. Autonomous Bug Fixing
- Bugs → fix immediately (no hand-holding)
- Trace logs, errors, failing tests
- Find root cause, not symptoms
- Fix CI failures proactively

### 7. Skills = System Layer
- Skills are NOT just markdown files
- They include code, scripts, data, workflows
- Use skills for:
  - verification
  - automation
  - data analysis
  - scaffolding
- Skills = reusable intelligence

### 8. File System = Context Engine
- Use folders for:
  - references/
  - scripts/
  - templates/
- Enable progressive disclosure
- Structure improves reasoning quality

### 9. Avoid Over-Constraining AI
- Don't force rigid steps
- Provide context, not micromanagement
- Let AI adapt to the problem
- Flexibility > strict instructions

## Task Management
1. Plan first → write tasks with checklist
2. Verify before execution
3. Track progress continuously
4. Explain changes at each step
5. Document results clearly
6. Capture lessons after completion

## Core Principles
- Simplicity First → minimal, clean solutions
- Systems > Prompts
- Verification > Generation
- Iteration > Perfection
- No Lazy Fixes → solve root cause

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ReactorPhysics_MATLAB** (1531 symbols, 4434 relationships, 122 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/ReactorPhysics_MATLAB/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ReactorPhysics_MATLAB/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ReactorPhysics_MATLAB/clusters` | All functional areas |
| `gitnexus://repo/ReactorPhysics_MATLAB/processes` | All execution flows |
| `gitnexus://repo/ReactorPhysics_MATLAB/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
