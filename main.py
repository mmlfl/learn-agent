import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import tools
from hooks import hooks, ToolUseEvent, ToolResultEvent
from tools import TOOLS, TOOL_HANDLERS
from config import WORKDIR
from ui import agent_display, subagent_display, console

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "qwen-vl-plus")

SYSTEM = f"""You are a coding agent at {os.getcwd()} on Windows. You have access to tools and subagents.

## Task Workflow

When the user gives you a complex task:

1. **Plan at the goal level** — call todo_write to break the task into 2-5 meaningful subtasks. Each subtask should be a complete goal (e.g. "understand all source files"), NOT a single operation (e.g. "read main.py"). Avoid micro-tasking.

2. **Execute each subtask** — for the current subtask, decide:
   - If it requires multiple operations (reading several files, running commands then analyzing) → use `spawn_task` to delegate to a subagent
   - If it's a single operation (read one known file, run one command) → do it yourself

3. **Mark progress** — when a subtask is complete, use todo_write to mark it completed and move the next one to in_progress

4. **Repeat** until all subtasks are done, then deliver the final answer.

Always work one subtask at a time."""


# ── Agent 循环 ──────────────────────────────────────────────
def agent_loop(messages: list):
    """
    完整的 Agent 循环：
    用户消息 → 模型决策 → 工具调用 → 结果返回 → 最终回复
    """
    round_num = 1

    while True:
        # ── 轮次标题 ──
        agent_display.show_round_header(round_num)

        # ── 1. 调用模型（带 spinner） ──
        t0 = time.time()
        with agent_display.model_thinking():
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=4000,
            )
        elapsed = time.time() - t0
        choice = response.choices[0]

        # ── 2. 显示模型回复 ──
        agent_display.show_model_response(
            finish_reason=choice.finish_reason,
            content=choice.message.content,
            elapsed=elapsed,
        )

        # ── 3. 加入对话历史 ──
        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls if choice.message.tool_calls else None
        })

        # ── 4. 如果没有工具调用，结束 ──
        if choice.finish_reason != "tool_calls":
            agent_display.show_final_answer(choice.message.content or "")
            return

        # ── 5. 显示工具调用清单 ──
        agent_display.show_tool_calls(choice.message.tool_calls)

        # ── 6. 执行工具 ──
        for tc in choice.message.tool_calls:
            args = tc.function.arguments
            try:
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                pass
            name = tc.function.name

            # Hook: 执行前校验
            hook_result = hooks.trigger("PreToolUse", ToolUseEvent(
                tool_name=name,
                tool_args=args,
                tool_call_id=tc.id,
            ))
            if hook_result is not None:
                agent_display.show_tool_result(
                    name, args, f"[阻止] {hook_result}",
                    success=False,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"[Blocked] {hook_result}",
                })
                continue

            # 执行工具
            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                output = f"Error: Unknown tool '{name}'"
                agent_display.show_tool_result(name, args, output, success=False)
            else:
                output = handler(**args)
                agent_display.show_tool_result(name, args, output,
                                               success=not output.startswith("Error"))

            # Hook: 执行后
            hooks.trigger("PostToolUse", ToolResultEvent(
                tool_name=name,
                tool_args=args,
                tool_call_id=tc.id,
                output=output,
            ))

            # 加入对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

        round_num += 1
        agent_display.show_round_end()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent_display.show_welcome(MODEL)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools.TASK_FILE = WORKDIR / "todos" / f"todos_{session_ts}.json"

    messages = [
        {"role": "system", "content": SYSTEM},
    ]

    while True:
        try:
            query = console.input(f"\n[yellow bold]👤 你:[/] ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            agent_display.show_goodbye()
            break

        messages.append({"role": "user", "content": query})
        agent_loop(messages)
