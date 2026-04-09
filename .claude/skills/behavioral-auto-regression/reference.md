# Behavioral Auto-Regression: Full Procedure and Rationale

## The Problem

Claude Code agents have a system prompt that contains tool selection
directives. The Grep tool description includes `ALWAYS use Grep for
search tasks` — one of the strongest obligation markers. When an agent
encounters a code exploration question ("what calls X?", "what depends
on Y?"), it categorizes this as a "search task" and the Grep ALWAYS
directive fires before any AGENT.md override is evaluated.

This means even well-intentioned AGENT.md instructions like "prefer
Nexus over Grep" are ineffective — the agent has already committed to
Grep before it reads the preference.

## Root Cause: Categorization, Not Priority

The issue is not that the agent ignores AGENT.md. It is that the agent
categorizes code exploration as "search" before it evaluates the
AGENT.md override. The Grep directive fires on the category match, not
on a priority comparison.

## The Fix: Reclassification

Instead of competing with the Grep ALWAYS directive (prohibition vs
obligation — a conflict the agent resolves unpredictably), we
**reclassify** the task so the Grep directive never fires:

> These are **exploration tasks, NOT search tasks.**

Once the agent categorizes the task as "exploration" instead of "search",
the Grep ALWAYS directive doesn't activate. The AGENT.md dispatch table
then steers the agent to the correct Nexus tool with zero conflict.

## Vocabulary Hierarchy

Keywords from strongest to weakest, as they appear in the Claude Code
system prompt:

| Weight | Keywords | Effect |
|--------|----------|--------|
| Hardest | `CRITICAL`, `IMPORTANT` | Flags for elevated attention |
| Hard | `MUST`, `NEVER` | Obligation / prohibition |
| Medium | `ALWAYS`, `Reserve...exclusively` | Default behavior / scope restriction |
| Soft | `Prefer`, `Avoid` | Tiebreaker when no stronger rule applies |
| Redirect | `instead`, `rather than` | "Not X, do Y" — gives replacement path |

The two patterns that work best together:
1. **Prohibition + redirect:** "NEVER do X — instead do Y"
2. **Reclassification:** "These are X tasks, NOT Y tasks"

## The Override Block Template

Every AGENT.md that uses Nexus skills MUST contain this block, adapted
for its specific domain. Place it immediately after the agent's opening
description.

```markdown
## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

[Domain-specific description] are **exploration tasks, NOT search tasks.**
The Nexus MCP tools are the ONLY permitted tools for [domain]. Nexus
[explains why it's superior]; Grep only matches text and CANNOT [what
Grep can't do].

NEVER use Grep for [domain]. RESERVE Grep exclusively for literal text
search where Nexus cannot answer the question. Before using Grep, you
MUST state why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| [question 1] | `mcp__nexus__[tool]` |
| [question 2] | `mcp__nexus__[tool]` |
| "Find literal string 'foo'" | Grep (this is the ONLY valid Grep use) |
```

### Why Each Element Matters

- **`CRITICAL`** — matches the weight tier of the strongest system prompt
  directives
- **"OVERRIDE all default tool preferences"** — directly echoes the
  CLAUDE.md override marker
- **"exploration tasks, NOT search tasks"** — THE KEY: reclassifies
  so the Grep ALWAYS directive never fires
- **`MUST`/`NEVER`/`ONLY permitted`** — hard obligations matching
  system prompt vocabulary
- **Dispatch table** — concrete pattern matching; when the agent sees
  "what calls X?", the table fires before default categorization
- **Justification gate** — "MUST state why Nexus is insufficient"
  creates friction even if the agent reflexively reaches for Grep

## System Prompt Biases to Counter

| Bias | Source | Counter |
|------|--------|---------|
| `ALWAYS use Grep for search tasks` | Grep tool description | Reclassify as "not search" |
| "use the appropriate dedicated tool" (lists Grep) | Bash tool description | Name Nexus as "the dedicated tool" |
| MCP tools have zero promotional directives | System prompt | Explicit MUST use in dispatch table |
| Deferred MCP tools need ToolSearch | System architecture | Skills pre-load tool awareness |
| Built-in Explore agent suggested | Agent tool description | Deny built-in Explore in CLAUDE.md |

## Validation Procedure

### The Iterative Loop

For each agent:

1. **Launch** with a real task. Prompt includes:
   "Report every tool you used and why you chose it over alternatives."

2. **Count Grep calls.** Any Grep call for a question Nexus can answer
   is a regression.

3. **If Grep > 0:** Resume the same agent. Ask:
   - "Why did you use Grep for X instead of Nexus?"
   - "What would the Nexus query be?"
   - "Do you wish Nexus had a convenience function for this?"

4. **If a convenience function is identified:** Implement it in
   sphinxcontrib-nexus (query.py + server.py + cli.py).

5. **Launch a FRESH agent** with the same task. Compare Grep count.

6. **Repeat** until only legitimate literal text searches use Grep.

### Validated Results (2026-04-09)

| Agent | Task | Grep | Nexus | Result |
|-------|------|------|-------|--------|
| Explorer | Understand SN module | 0 | 9 | Pass |
| QA | Verification coverage audit | 0 | 8+ | Pass |
| Test-architect | Design fuel module tests | 0 | 6 | Pass |
| Archivist | CP theory page audit | 0 | 8 | Pass |
| Numerics-investigator | Diagnose cylindrical SN bug | 0 | 12 | Pass |

### Convenience Functions Identified by Loop

| Function | Requested by | What it does |
|----------|-------------|-------------|
| `callers(node_id, transitive)` | Explorer | Deduplicated caller list |
| `callees(node_id, transitive)` | Explorer | Deduplicated callee list |
| `verification_audit()` | QA | Single-call V&V report |

## Architecture: Where Override Blocks Live

```
CLAUDE.md
  └─ CRITICAL: Tool Selection Override (project-level, all agents)

.claude/agents/<agent>/AGENT.md
  └─ CRITICAL: Tool Selection Override (agent-specific dispatch table)

.claude/skills/<nexus-skill>/SKILL.md
  └─ IMPORTANT: one-liner reinforcement (avoids duplication with AGENT.md)

.claude/skills/<nexus-skill>/scripts/tool-override-block.md
  └─ The specific block to copy into AGENT.md for this skill
```

The comprehensive override lives in AGENT.md (one place per agent).
Skills have a lightweight reinforcement. The scripts/ directory contains
the canonical block to copy when creating a new agent.

## When This Procedure Should Run

- After adding a new agent that uses Nexus skills
- After a system prompt update that may change tool selection directives
- After observing an agent using Grep for exploration questions
- Periodically as a regression check (quarterly or after major updates)
