import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import tools
from compact import compact_layer1, compact_layer2, tool_result_budget, reactive_compact
from hooks import hooks, ToolUseEvent, ToolResultEvent
from memory import build_memory_system, extract_memories, load_memories, consolidate_memories
from skill_loader import SKILL_SYSTEM
from tools import TOOLS, TOOL_HANDLERS
from config import WORKDIR
from ui import agent_display, subagent_display, console
from logger import logger

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
MAX_REACTIVE_RETRIES = 1

def agent_loop(messages: list):
    """
    完整的 Agent 循环：
    用户消息 → 模型决策 → 工具调用 → 结果返回 → 最终回复
    """
    round_num = 1
    reactive_retries = 0

    while True:
        #第二层压缩
        messages = compact_layer2(messages)
        # ── 轮次标题 ──
        agent_display.show_round_header(round_num)

        # ── 1. 调用模型（带 spinner） ──
        t0 = time.time()
        try:
            with agent_display.model_thinking():
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_completion_tokens=4000,
                )
        except Exception as e:
            err_msg = str(e).lower()
            if ("too_long" in err_msg or "too many tokens" in err_msg
                    or "maximum context" in err_msg):
                if reactive_retries < MAX_REACTIVE_RETRIES:
                    messages[:] = reactive_compact(messages)
                    reactive_retries += 1
                    continue
                raise RuntimeError(
                    f"应急压缩重试 {MAX_REACTIVE_RETRIES} 次后仍失败，"
                    f"请使用 /clear 或新会话重试"
                ) from e
            raise
        elapsed = time.time() - t0
        reactive_retries = 0  # 调用成功，重置计数
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
            # 第一层压缩
            messages = compact_layer1(messages)
            #一轮结束 看看能否提取记忆
            extract_memories(messages)
            #如果记忆过多,执行修剪,重复合并等等
            consolidate_memories()
            agent_display.show_final_answer(choice.message.content or "")
            return

        # ── 5. 显示工具调用清单 ──
        agent_display.show_tool_calls(choice.message.tool_calls)

        # ── 6. 执行工具 ──
        for tc in choice.message.tool_calls:
            args = tc.function.arguments
            name = tc.function.name
            try:
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "you pass the wrong arguments to this tool,please try again to delivery"
                               "the correct arguments",
                    "name": name
                })
                continue

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
                    "name": name
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
            output = tool_result_budget(output, tc.id, name=name, args=args)
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
                "name": name
            })

        round_num += 1
        agent_display.show_round_end()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent_display.show_welcome(MODEL)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    todo_file = WORKDIR / f"todos_{session_ts}.json"
    tools.TASK_FILE = todo_file

    memory_system = build_memory_system()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": SKILL_SYSTEM},
        {"role": "system", "content": memory_system},
    ]

    while True:
        memory_system = build_memory_system()
        messages[2] = {"role": "system", "content": memory_system}
        try:
            query = console.input(f"\n[yellow bold]👤 你:[/] ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            agent_display.show_goodbye()
            if todo_file.exists():
                todo_file.unlink()
            break

        logger.log_user_message(query)
        messages.append({"role": "user", "content": query})
        #执行前,先进行相关记忆注入
        load_memories(messages)

        agent_loop(messages)

        if todo_file.exists():
            todo_file.unlink()
