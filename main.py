import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import tools
from compact import compact_layer1, compact_layer2, tool_result_budget, reactive_compact
from config import WORKDIR
from hooks import hooks, ToolUseEvent, ToolResultEvent
from logger import logger
from memory import build_memory_system, extract_memories, load_memories, consolidate_memories
from prompt import get_static_system_messages
from tools import TOOLS, TOOL_HANDLERS
from ui import agent_display, console

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME", "qwen-vl-plus")


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

    #每次对话开启 创建任务规划文件
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    todo_file = WORKDIR / f"todos_{session_ts}.json"
    tools.TASK_FILE = todo_file

    messages = get_static_system_messages()
    MEMORY_INDEX = len(messages)
    messages.append({})

    while True:
        #获取动态记忆,由于对话的执行,用户可能会存储更新记忆
        memory_system = build_memory_system()
        messages[MEMORY_INDEX] = {"role": "system", "content": memory_system}
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

        #一轮对话结束 删除可能存在的任务规划文件
        if todo_file.exists():
            todo_file.unlink()
