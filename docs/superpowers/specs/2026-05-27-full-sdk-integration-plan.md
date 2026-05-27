# Full SDK Integration Plan — AutonomousRunner + Agent SDK
# Created: 2026-05-27 05:45 CST

## Goal
Wire the AutonomousRunner and LiteLLMHTTPTransport into the full Agent SDK feature set. Enable multi-agent, mid-task model switching, session persistence, hooks, streaming, and MCP tools. Then use the runner itself (with free models) to build the remaining features — bootstrapping.

## Current State
- LiteLLMHTTPTransport: built, tested, talks to router-proxy (:8016)
- AutonomousRunner: built, tested, 4-tier stall diagnosis, checkpoint/resume
- Built-in tools: Read, Write, Edit, Glob, Bash, Grep
- Smoke test: PASSED (2 turns, 1 tool call, 7.5s)

## Phase 1: SDK Client Integration (foundation)
**Goal:** Make `ClaudeSDKClient` work with LiteLLM transport as a drop-in.

### Files to modify:
- `src/claude_agent_sdk/client.py` — add `transport` parameter to constructor, if `transport="litellm"`, create LiteLLMHTTPTransport instead of SubprocessCLITransport
- `src/claude_agent_sdk/types.py` — add `LiteLLMOptions` dataclass (base_url, model, api_key, max_turns, checkpoint_dir)
- `src/claude_agent_sdk/__init__.py` — export AutonomousRunner, LiteLLMHTTPTransport, RunResult

### New file:
- `src/claude_agent_sdk/litellm_config.py` — configuration helper that reads from env vars or config file:
  - `LITELLM_BASE_URL` (default: http://127.0.0.1:8016)
  - `LITELLM_API_KEY` (default: from env)
  - `LITELLM_DEFAULT_MODEL` (default: qwen/qwen3-max)

### Test: 
```python
client = ClaudeSDKClient(transport="litellm", model="qwen/qwen3-max")
result = await client.query("echo hello world using bash")
```

---

## Phase 2: Mid-Task Agent Switching
**Goal:** Switch models during a running task without losing context.

### How it works:
- Runner gets a `switch_model(new_model: str)` method
- Internally: saves checkpoint, updates transport model, continues loop
- No new transport needed — just change `self._transport._model`

### Files to modify:
- `src/claude_agent_sdk/runner.py` — add `switch_model()` method, add `model_history` list to RunResult
- `src/claude_agent_sdk/_internal/transport/litellm_http.py` — add `set_model(model: str)` method

### New feature: Auto-escalation
- If stall handler hits tier 3 (repair prompt), optionally auto-switch to a smarter model before bailing
- Config: `escalation_model: str | None` — if set, tier 3 switches to this model instead of bailing at tier 4
- Example: start with `nvidia/free-model`, escalate to `claude-sonnet-4-6` on stall

### Test:
```python
runner = AutonomousRunner(model="qwen/qwen3-max", escalation_model="claude-sonnet-4-6")
result = await runner.run(task="Complex refactor task")
# If qwen stalls, auto-escalates to claude
print(result.model_history)  # ["qwen/qwen3-max", "claude-sonnet-4-6"]
```

---

## Phase 3: Multi-Agent Support
**Goal:** Runner can spawn sub-runners for parallel work. Agents can hand off tasks.

### Architecture:
```
MainRunner (orchestrator)
├── SubRunner A (model: qwen, task: "build backend")
├── SubRunner B (model: nvidia, task: "build frontend")  
└── SubRunner C (model: qwen, task: "write tests")
```

### New files:
- `src/claude_agent_sdk/multi_agent.py` — MultiAgentOrchestrator class
  - `add_agent(name, model, task, cwd)` — register a sub-agent
  - `run_parallel()` — run all agents concurrently via asyncio.gather
  - `run_sequential()` — run agents in order, passing context forward
  - `handoff(from_agent, to_agent, context)` — transfer conversation state

### New built-in tool for runner:
- `SpawnAgent` tool — model can call this to create sub-runners during execution
  - Input: `{"task": "...", "model": "...", "cwd": "..."}`
  - Runner creates a sub-runner, runs it, returns result as tool output
  - Sub-runner inherits parent's checkpoint_dir (subdirectory)

### Shared state:
- Agents share a workspace directory (filesystem is the coordination layer)
- Parent can read sub-agent checkpoints to see progress
- `MultiAgentResult` aggregates all sub-results

### Test:
```python
orchestrator = MultiAgentOrchestrator()
orchestrator.add_agent("backend", model="qwen/qwen3-max", task="Build FastAPI backend")
orchestrator.add_agent("frontend", model="nvidia/free", task="Build React frontend")
results = await orchestrator.run_parallel(cwd="/project")
```

---

## Phase 4: Hook System Integration
**Goal:** Runner calls SDK's hook system before/after tool execution.

### Hook points:
- `pre_tool_use(tool_name, params)` → can modify params or block
- `post_tool_use(tool_name, params, result)` → can modify result
- `pre_turn(turn_number, messages)` → can inject context
- `post_turn(turn_number, response)` → can log or modify
- `on_stall(tier, consecutive_stalls)` → custom stall handling
- `on_complete(result)` → post-processing

### Files to modify:
- `src/claude_agent_sdk/runner.py` — add hook dispatch points in the agentic loop
- New: `src/claude_agent_sdk/hooks.py` — HookRegistry class, decorator-based registration

### Test:
```python
runner = AutonomousRunner(model="qwen/qwen3-max")

@runner.hook("pre_tool_use")
async def log_tools(tool_name, params):
    print(f"About to call: {tool_name}")
    return params  # return modified params or None to block

result = await runner.run(task="Build something")
```

---

## Phase 5: Streaming SSE Output
**Goal:** See model output in real-time, not after each turn completes.

### Files to modify:
- `src/claude_agent_sdk/_internal/transport/litellm_http.py` — add streaming support to `do_turn()`:
  - Set `stream=True` in API call
  - Parse SSE chunks, yield partial text + tool calls
  - Emit events via callback
- `src/claude_agent_sdk/runner.py` — add `on_stream` callback parameter

### Test:
```python
async def on_token(text):
    print(text, end="", flush=True)

runner = AutonomousRunner(model="qwen/qwen3-max", on_stream=on_token)
result = await runner.run(task="Explain this code")
```

---

## Phase 6: MCP Tool Bridge  
**Goal:** Runner can use MCP servers (same ones Claude Code uses).

### New file:
- `src/claude_agent_sdk/tools/mcp_bridge.py` — MCP client that:
  - Connects to MCP servers via stdio or SSE
  - Discovers available tools
  - Converts MCP tool schemas to OpenAI function-calling format
  - Dispatches tool calls to MCP servers
  - Returns results

### Integration:
- Runner's tool registry merges built-in tools + MCP-discovered tools
- Config: `mcp_servers: dict[str, MCPConfig]` in runner constructor

---

## Phase 7: Session Stores
**Goal:** Persistent sessions via Redis/Postgres/S3 (match existing SDK stores).

### Files to modify:
- `src/claude_agent_sdk/runner.py` — add `session_store` parameter
- Reuse existing SDK session store implementations (they already exist in examples/)
- Checkpoint data writes to session store instead of (or in addition to) filesystem

---

## Phase 8: Permission Model
**Goal:** Control what tools agents can use, with approval callbacks.

### New file:
- `src/claude_agent_sdk/permissions.py` — PermissionPolicy class
  - `allow(tool_names)` — whitelist
  - `deny(tool_names)` — blacklist  
  - `require_approval(tool_names, callback)` — ask before executing
  - `sandbox(cwd_restrict, network_restrict)` — containment

---

## Build Order & Bootstrapping Strategy

Each phase is built by the runner itself using free models:

| Phase | Priority | Model | Est. Complexity |
|-------|----------|-------|-----------------|
| 1. SDK Client Integration | P0 | qwen/qwen3-max | Low — config + wiring |
| 2. Mid-Task Agent Switching | P0 | qwen/qwen3-max | Low — 2 methods |
| 3. Multi-Agent Support | P1 | qwen/qwen3-max | Medium — new orchestrator class |
| 4. Hook System | P1 | qwen/qwen3-max | Medium — decorator pattern |
| 5. Streaming SSE | P2 | qwen/qwen-turbo | Medium — SSE parsing |
| 6. MCP Tool Bridge | P2 | qwen/qwen3-max | High — MCP protocol |
| 7. Session Stores | P3 | qwen/qwen-turbo | Low — reuse existing |
| 8. Permission Model | P3 | qwen/qwen-turbo | Low — policy class |

**Bootstrapping:** After Phase 1, the runner can build Phase 2. After Phase 2+3, multi-agent can parallelize Phase 4-8.

## Acceptance Criteria
- All phases pass smoke tests
- Mid-task model switch works (start qwen, switch to claude, result correct)
- Multi-agent parallel run completes both sub-tasks
- Hook fires on every tool call
- Streaming shows tokens in real-time
- Existing SDK code (bridge_sdk.py) works with `transport="litellm"` config change
