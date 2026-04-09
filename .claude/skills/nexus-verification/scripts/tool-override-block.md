# Tool Override Block for nexus-verification

Copy this block into the AGENT.md of any agent that uses the
nexus-verification skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Verification assessment, test coverage mapping, and documentation drift
detection are **exploration tasks, NOT search tasks.** The Nexus MCP
tools are the ONLY permitted tools for V&V assessment. Nexus traces
equation -> code -> test chains; Grep cannot determine verification
status or dependency chains.

NEVER use Grep for verification assessment. Before using Grep, you
MUST state why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "Full V&V audit" | `mcp__nexus__verification_audit` (single call) |
| "What's verified?" | `mcp__nexus__verification_coverage` |
| "Which equations have no tests?" | `mcp__nexus__verification_coverage({status_filter: "implemented"})` |
| "Which docs are stale?" | `mcp__nexus__staleness` |
| "What tests cover X?" | `mcp__nexus__callers` (on the function, filter tests.*) |
| "What tests to re-run?" | `mcp__nexus__retest` |
| "Find literal string 'foo'" | Grep (ONLY valid Grep use) |
