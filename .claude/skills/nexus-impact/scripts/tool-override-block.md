# Tool Override Block for nexus-impact

Copy this block into the AGENT.md of any agent that uses the
nexus-impact skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Impact analysis, dependency tracing, and blast radius assessment are
**exploration tasks, NOT search tasks.** The Nexus MCP tools are the
ONLY permitted tools for these tasks. Nexus traverses the call graph
and dependency graph; Grep only matches text and CANNOT find transitive
dependencies.

NEVER use Grep for impact analysis. Before using Grep, you MUST state
why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "What depends on X?" | `mcp__nexus__impact` (direction=upstream) |
| "What does X depend on?" | `mcp__nexus__impact` (direction=downstream) |
| "What calls X?" | `mcp__nexus__callers` |
| "What tests need re-running?" | `mcp__nexus__retest` |
| "What did my changes affect?" | `mcp__nexus__detect_changes` |
| "Is it safe to change X?" | `mcp__nexus__impact` then `mcp__nexus__retest` |
| "Find literal string 'foo'" | Grep (ONLY valid Grep use) |
