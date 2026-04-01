# OPAL — Open Platform for Analytical Thermalhydraulics

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architecture)
- Define BOTH execution + verification steps
- If something breaks → STOP and re-plan
- Write detailed specs to remove ambiguity

### 2. Subagent Strategy
- Use subagents aggressively for complex problems
- Split tasks: research, execution, analysis
- One task per agent for clarity
- Parallelize thinking, not just execution

### 3. Self-Improvement Loop
- After ANY mistake → log it in gotchas.md
- Convert mistakes into rules
- Review past lessons before starting
- Iterate until error rate drops

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