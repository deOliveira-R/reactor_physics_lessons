# Tool Override Block for nexus-exploring

Copy this block into the AGENT.md of any agent that uses the
nexus-exploring skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

The following rules OVERRIDE all default tool preferences. Follow them
exactly as written.

Understanding code architecture, tracing callers/callees, finding
dependents, and exploring module structure are **exploration tasks, NOT
search tasks.** The Nexus MCP tools are the ONLY permitted tools for
code exploration. Nexus understands code relationships — callers,
dependents, equations, theory connections. Grep only matches text.

NEVER use Grep for exploration. RESERVE Grep exclusively for literal
text search (error messages, magic constants, config values) where
Nexus cannot answer the question. Before using Grep, you MUST state
why Nexus is insufficient for this specific query.

| Question | MUST use |
|----------|----------|
| "How does X work?" | `mcp__nexus__context` + `mcp__nexus__neighbors` |
| "What calls X?" | `mcp__nexus__callers` (or transitive=true for full chain) |
| "What does X call?" | `mcp__nexus__callees` |
| "How do A and B connect?" | `mcp__nexus__shortest_path` |
| "What's the math behind X?" | `mcp__nexus__provenance_chain` |
| "Show me the main components" | `mcp__nexus__god_nodes` + `mcp__nexus__communities` |
| "Find symbol named X" | `mcp__nexus__query` |
| "Find literal string 'foo'" | Grep (this is the ONLY valid Grep use) |
