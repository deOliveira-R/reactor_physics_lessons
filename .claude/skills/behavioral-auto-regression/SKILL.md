---
name: behavioral-auto-regression
description: "Proactively use when agent behaviour regresses to using Grep instead of Nexus for code exploration. Diagnoses the root cause (system prompt bias), extracts the vocabulary needed to override it, and produces corrected AGENT.md blocks. Examples: \"Agent is using Grep instead of Nexus\", \"Tool selection is wrong\", \"Bias steering isn't working\""
---

# Behavioral Auto-Regression

Diagnose and fix agent tool-selection regression. When agents default to
Grep for code exploration instead of Nexus MCP tools, the root cause is
a mismatch between the agent's system prompt vocabulary and the AGENT.md
override vocabulary.

## When to Use

- An agent used Grep for a code exploration question (callers, dependents,
  architecture, equations) instead of Nexus
- A new agent was added without the tool override block
- The system prompt changed and existing overrides stopped working

## What to Do

### 1. Identify the regression

Check the agent's tool usage report. Any Grep call for a question that
Nexus can answer (callers, dependents, coverage, equations) is a
regression.

### 2. Probe the agent's system prompt vocabulary

Launch a fresh agent with the probe prompt in
[scripts/probe-prompt.md](scripts/probe-prompt.md). This extracts:
- Which keywords carry the most weight (CRITICAL, MUST, NEVER, etc.)
- Which directives steer toward Grep (usually `ALWAYS use Grep for search tasks`)
- What phrasing would effectively override those directives

### 3. Apply the reclassification fix

The key mechanism is **reclassification**: define code exploration as
"NOT a search task" so the Grep `ALWAYS` directive never fires. This is
more effective than prohibition alone.

Every AGENT.md that uses Nexus skills MUST contain a CRITICAL Tool
Selection Override block. The correct block for each skill is in:
`<nexus-skill>/scripts/tool-override-block.md`

The block must:
1. Use `CRITICAL` + `OVERRIDE` — strongest vocabulary tier
2. Reclassify: "These are **exploration tasks, NOT search tasks**"
3. Use `MUST`/`NEVER`/`ONLY permitted` — hard obligations
4. Include a dispatch table mapping questions to specific Nexus tools
5. Add a justification gate: "Before using Grep, you MUST state why
   Nexus is insufficient"

### 4. Validate the fix

Launch the agent with a real task. Prompt must include:
"Report every tool you used and why."

If Grep count = 0, the fix works. If not, resume the agent and ask
why it chose Grep — this identifies missing convenience functions or
unclear tool signatures (Phase 2 candidates).

See [reference.md](reference.md) for the full procedure, rationale,
and validation results.
