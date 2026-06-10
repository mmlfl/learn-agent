# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the agent (interactive REPL)
uv run python main.py

# Generate daily tech articles
uv run python daily_news.py
```

## Architecture

This is a from-scratch Python agent framework built on the OpenAI-compatible chat completions API (targeting Alibaba DashScope for Qwen models). It has no test suite — verification is manual via the REPL.

**Agent loop** (`main.py`): The core loop sends user messages to the LLM, processes tool calls, and feeds results back. The `SYSTEM` prompt instructs the model to plan with `todo_write`, delegate complex subtasks via `spawn_task`, and work one subtask at a time. Tools are defined as JSON schemas in `tools.py`.

**Tool system** (`tools.py`): Seven tools — `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `todo_write`, `spawn_task`. Handlers are plain functions keyed by name in `TOOL_HANDLERS`. The subagent (`spawn_task`) spawns a fresh LLM conversation with a reduced tool set (`SUB_TOOLS`) and a 30-round cap; it returns only its conclusion to the parent agent.

**Security** (`rules.py` + `hooks.py`): Three-tier gate on tool execution. Tier 1 (`DENY_LIST`) hard-blocks dangerous commands like `rm -rf /`. Tier 2 checks `PERMISSION_RULES` (workspace-escape for file writes, destructive bash patterns). Tier 3 prompts the user for approval. These are wired as `PreToolUse` hooks.

**Hooks** (`hooks.py`): Event-driven hook manager (`HookManager`) with `on()` decorator registration and `trigger()` dispatch. Events: `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop`. Callbacks return `None` to pass, or a string to block. Built-in hooks cover permission checks, large-output warnings, and session statistics.

**UI** (`ui.py`): Rich-based terminal rendering. `AgentLoopDisplay` renders the main agent's rounds (thinking spinner, model response, tool calls, final answer). `SubAgentDisplay` tracks subagent execution with a collapsible tree of tool calls. `Theme` class defines unified color/style constants. Global singletons (`agent_display`, `subagent_display`, `console`) are imported by other modules.

**Config** (`config.py`): Just `WORKDIR = Path.cwd()` — the working directory for path sandboxing.

**daily_news.py**: Standalone script that uses a separate LLM call to generate AI/Agent tech articles (news roundups or deep-dive explainers), saved to `docs/`.

## Key details

- **Model**: Configured via `.env` (`MODEL_NAME`, `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`). Defaults to `qwen3.7-max`.
- **Path sandboxing**: `safe_path()` in `tools.py` enforces all file operations stay within `WORKDIR`. Tools reject paths that escape via `..` traversal.
- **Todo files**: Session task lists are persisted as JSON to `todos/todos_{timestamp}.json`. The directory is git-ignored.
- **Subagent tools**: Subagents get a reduced tool set — no `todo_write` or `spawn_task` (to prevent infinite recursion).
- **Encoding handling**: `read_file` tries UTF-8 first, falls back to GBK (relevant for Chinese Windows environments).
