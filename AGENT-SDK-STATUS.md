# Agent SDK Python Fork — _find_cli() Change Status

**Timestamp:** 2026-05-20 — CST

---

## Fork Location
```
C:\Users\larry\ClaudeNotes\shared\projects\agent-sdk-python-frozen\
```

---

## Git Status
- **Branch:** main
- **Ahead of upstream:** 2 commits
- **Uncommitted changes:** None (only 2 untracked CHANGELOG.md files — build artifacts)
- **Status:** Clean

---

## Recent Commits
```
811dd2b fix(transport): prefer system CLI over bundled binary ✓
5a533b2 fix(transport): remove SDK entrypoint tagging to prevent billing classification
7837c92 chore: bump bundled CLI version to 2.1.145
5459309 chore: bump bundled CLI version to 2.1.144
c352a50 chore: bump bundled CLI version to 2.1.143
```

---

## _find_cli() Change Analysis

### File Location
`src/claude_agent_sdk/_internal/transport/subprocess_cli.py` — lines 80–97

### Change Present: YES

**System CLI priority change IS IMPLEMENTED AND COMMITTED**

### Code (lines 80–98)
```python
def _find_cli(self) -> str:
    """Find Claude Code CLI binary."""
    # Prefer system-installed CLI (has user's hooks, plugins, settings)
    if cli := shutil.which("claude"):
        return cli

    locations = [
        Path.home() / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".local/bin/claude",
        Path.home() / "node_modules/.bin/claude",
        Path.home() / ".yarn/bin/claude",
        Path.home() / ".claude/local/claude",
    ]

    for path in locations:
        if path.exists() and path.is_file():
            return str(path)
```

### Key Points
- ✓ **shutil.which("claude")** is the first check (line 83)
- ✓ **Prefers system-installed CLI** over bundled paths
- ✓ Fallback locations are checked only if system CLI not found
- ✓ Comment clarifies rationale: "has user's hooks, plugins, settings"

---

## Commit Status
- **Change committed:** YES — commit `811dd2b`
- **Uncommitted diffs:** NONE
- **Ready for use:** YES

---

## Next Step
**Ready to deploy.** No further changes needed; the feature is committed and clean.
