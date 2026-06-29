# Agent Instructions

## General Response Style
- Be concise and direct.
- Explain assumptions clearly.
- Do not over-engineer the solution.
- Prefer simple, idiomatic solutions over clever ones.

## Planning Mode
- Read the project plan and existing files first.
- Ask clarifying questions only when missing information blocks implementation.
- Follow the tech stack and architecture defined in `plan.md`. If the plan does not specify something, ask before assuming.
- It is fine to suggest a different stack or approach if there is a strong reason, but flag it explicitly and wait for approval.
- Break work into phases and identify dependencies.
- Before presenting a final plan, review it for:
  - architecture consistency
  - data flow
  - security/privacy risks
  - testing strategy
  - deployment complexity

## Change / Edit Mode
- Inspect the current codebase before editing.
- Follow the existing project structure and style.
- Match the libraries, patterns, and conventions already in use. Do not introduce a new dependency without a clear reason.
- Split large changes into independent chunks.
- Use sub-agents when available and useful; otherwise implement sequentially.
- After each significant change, summarize:
  - changed files
  - reason for change
  - how to test
- If a change would be faster with a different tool/framework than what the plan specifies, mention it as a suggestion rather than silently switching.

## Testing & Quality Checks
- Run available project checks after non-trivial changes. For tiny edits (typo, single-line tweak), skip.
- Prefer scripts from package.json, pyproject.toml, Makefile, README, or docs.
- Typical checks:
  - lint
  - type check
  - unit tests
  - build
  - smoke test
- If no testing tools exist, ask whether to add basic tests or skip testing.

## Database Changes
- Do not change database schema casually.
- If the project uses Drizzle ORM:
  - run drizzle generate after schema changes
  - run drizzle migrate after generation
  - never run drizzle push unless explicitly approved
- If another ORM is used, follow that ORM's migration workflow.

## UI Design
- If DESIGN.md exists, always follow it.
- If DESIGN.md does not exist, use a simple, clean, responsive design.
- For UI work, include:
  - loading state
  - empty state
  - error state
  - mobile responsiveness

## Security
- Never hardcode API keys or secrets.
- Use environment variables.
- Keep .env out of git.
- Maintain .env.example.
- Do not log sensitive user data, API keys, tokens, or PII.

## RAG / AI Rules
- Separate product retrieval, policy retrieval, and answer generation.
- Return sources/citations when answering from retrieved documents.
- If retrieval confidence is low, say so instead of hallucinating.
- Keep prompts versioned and documented.
- Log useful debugging metadata without exposing private data.

## Documentation
- Update README when setup, scripts, API routes, or usage changes.
- Keep docs practical and runnable.