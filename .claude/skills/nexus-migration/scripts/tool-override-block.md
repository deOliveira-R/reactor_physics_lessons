# Tool Override Block for nexus-migration

Copy this block into the AGENT.md of any agent that uses the
nexus-migration skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Dependency analysis and migration planning are **exploration tasks, NOT
search tasks.** The Nexus MCP tools are the ONLY permitted tools for
migration planning. Nexus traces the full dependency graph; Grep only
finds import statements and CANNOT determine transitive dependencies.

NEVER use Grep for dependency analysis. Before using Grep, you MUST
state why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "Plan migration from X to Y" | `mcp__nexus__migration_plan` |
| "Find all uses of package X" | `mcp__nexus__query` (node_types=external) + `mcp__nexus__impact` |
| "What depends on numpy.ndarray?" | `mcp__nexus__impact` (upstream, edge_types=type_uses) |
| "What tests to run?" | `mcp__nexus__retest` |
| "Find literal string 'foo'" | Grep (ONLY valid Grep use) |
