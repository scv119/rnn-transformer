# AGENT.md — Git Commit Mistake Prevention

Use this checklist before every commit/push.

## Goal
Prevent repeated git mistakes by forcing a short post-commit review and a running mistake log.

## Pre-commit checklist (mandatory)
1. Run `git status --short` and verify only intended files are staged.
2. Stage explicitly (avoid broad `git add .` unless requested).
3. Run `git diff --staged` and verify:
   - no secrets/tokens
   - no accidental large/generated artifacts
   - no unrelated changes
4. Ensure commit message is specific and scoped.
5. If logs/PIDs/tmp files changed, confirm whether they should be committed.

## Post-commit summary (mandatory)
After each commit, append a brief entry to `memory/git-mistakes.md` with:
- Commit hash
- What changed
- Any mistake caught (or `none`)
- What guardrail/check prevented (or should prevent) it next time

Template:

```md
## <YYYY-MM-DD HH:mm TZ> — <short hash>
- Scope: <what this commit did>
- Mistake: <none | description>
- Prevention: <checklist item or new rule>
```

## If a mistake is discovered after commit
1. Document it in `memory/git-mistakes.md`.
2. Fix immediately with one of:
   - amend commit
   - follow-up fix commit
   - revert
3. Add/adjust a concrete rule in this file so the same error is less likely.

## Common mistakes to watch
- Committing unrelated working-tree changes.
- Forgetting to update docs after command/output path changes.
- Accidentally committing live logs, pid files, temp files.
- Ambiguous commit messages.
- Pushing without checking staged diff.

## Behavior rule
When asked to commit, prefer small, atomic commits and clearly report:
- commit hash
- files changed
- whether any non-target dirty files were left uncommitted.
