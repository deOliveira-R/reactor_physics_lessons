# Tool Override Block for nexus-debugging

Copy this block into the AGENT.md of any agent that uses the
nexus-debugging skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Bug tracing, error diagnosis, and equation verification are
**exploration tasks, NOT search tasks.** The Nexus MCP tools are the
ONLY permitted tools for tracing from failing tests to equations.
Nexus follows the call graph to find which equations are on the
failure path; Grep cannot trace call chains or find equation links.

NEVER use Grep for debugging exploration. Before using Grep, you MUST
state why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "Why is this test failing?" | `mcp__nexus__trace_error` |
| "Which equation might be wrong?" | `mcp__nexus__trace_error` then `mcp__nexus__provenance_chain` |
| "What does this function call?" | `mcp__nexus__context` or `mcp__nexus__callees` |
| "What calls this function?" | `mcp__nexus__callers` |
| "Where does this value come from?" | `mcp__nexus__impact` (downstream) |
| "Find error message 'foo'" | Grep (ONLY valid Grep use) |
