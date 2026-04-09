# Tool Override Block for nexus-refactoring

Copy this block into the AGENT.md of any agent that uses the
nexus-refactoring skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Rename analysis, dependency mapping for refactoring, and blast radius
assessment are **exploration tasks, NOT search tasks.** The Nexus MCP
tools are the ONLY permitted tools for safe refactoring. Nexus finds
all references via the call graph (high confidence); Grep finds text
matches that may be comments, strings, or unrelated symbols.

NEVER use Grep to find references for renaming. Before using Grep, you
MUST state why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "What depends on X?" | `mcp__nexus__impact` (upstream) |
| "Preview a rename" | `mcp__nexus__rename` (dry_run=true) |
| "Apply a rename" | `mcp__nexus__rename` (dry_run=false) |
| "What changed?" | `mcp__nexus__detect_changes` |
| "What tests to run?" | `mcp__nexus__retest` |
| "Find literal string 'foo'" | Grep (ONLY valid Grep use) |
