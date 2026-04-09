# Tool Override Block for nexus-guide

Copy this block into the AGENT.md of any agent that uses the
nexus-guide skill. Place it immediately after the agent's
opening description, before any workflow content.

---

## CRITICAL: Tool Selection Override

For any task involving code understanding, debugging, impact analysis,
or verification: the Nexus MCP tools OVERRIDE all default tool
preferences. These are **exploration tasks, NOT search tasks.** Match
your task to a nexus skill and follow its workflow.

NEVER use Grep for code exploration. RESERVE Grep exclusively for
literal text search where Nexus cannot answer the question. Before
using Grep, you MUST state why Nexus is insufficient.
