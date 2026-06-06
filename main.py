import json
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import tools
from hooks import hooks, ToolUseEvent, ToolResultEvent
from tools import TOOLS, TOOL_HANDLERS, WORKDIR

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
MODEL = os.getenv("MODEL_ID","qwen-plus-2025-07-28")

SYSTEM = f"""You are a coding agent at {os.getcwd()} on windows system.

 ## Task Workflow
 When the user gives you a complex task:
 1. FIRST, call todo_write to break it down into subtasks (pending status)
 2. Call todo_read to see what to work on next
 3. Execute that task using the available tools (bash, read_file, write_file, etc.)
 4. When done, call todo_write to mark the task completed and the next one in_progress
 5. Call todo_read again to get the next task
 6. Repeat until todo_read says "All tasks completed!"

 Always work one task at a time. Use todo_read to stay on track."""


# ── ANSI 颜色定义 ──────────────────────────────────────────
C = "\033[36m"      # 青色 - 标题/边框
G = "\033[32m"      # 绿色 - 成功
Y = "\033[33m"      # 黄色 - 警告
R = "\033[91m"      # 亮红 - 错误
B = "\033[34m"      # 蓝色 - 工具名
M = "\033[35m"      # 紫色 - 高亮
D = "\033[90m"      # 灰色 - 次要信息
Z = "\033[0m"       # 重置
BOLD = "\033[1m"    # 加粗

def _box_top(text: str, width: int = 60) -> str:
    return f"{C}╔{'═' * (width - 2)}╗\n║ {BOLD}{text}{Z}{C} {' ' * (width - len(text) - 5)}║\n╚{'═' * (width - 2)}╝{Z}"

def _sep(label: str = "") -> str:
    if label:
        return f"{D}── {label} {'─' * (52 - len(label))}{Z}"
    return f"{D}{'─' * 58}{Z}"


# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    """
    完整的 Agent 循环：
    用户消息 → 模型决策 → 工具调用 → 结果返回 → 最终回复
    """
    round_num = 1

    while True:
        # ═══════════ 轮次标题 ═══════════
        print(f"\n{_box_top(f'🔄  Round {round_num}')}")

        # 1. 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=4000,
        )
        choice = response.choices[0]

        # 2. 打印模型返回信息
        print(f"\n  {_sep('Model Response')}")
        reason_color = G if choice.finish_reason == "stop" else Y
        print(f"  {D}finish_reason:{Z} {reason_color}{choice.finish_reason}{Z}")
        content = choice.message.content or f'{D}(空){Z}'
        print(f"  {D}content:{Z}      {content}")

        # 3. 把模型的回复加入对话历史
        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls if choice.message.tool_calls else None
        })

        # 4. 打印工具调用信息
        if choice.message.tool_calls:
            print(f"\n  {_sep('Tool Calls')}")
            for i, tc in enumerate(choice.message.tool_calls, 1):
                args_obj = json.loads(tc.function.arguments)
                arg_str = json.dumps(args_obj, ensure_ascii=False)
                print(f"  {BOLD}{M}{i}.{Z} {B}{tc.function.name}{Z}")
                print(f"     {D}args:{Z} {arg_str}")
                print(f"     {D}id:{Z}   {D}{tc.id}{Z}")

        # 5. 如果模型没有调用工具，结束循环
        if choice.finish_reason != "tool_calls":
            print(f"\n  {G}✅ 模型完成回复，退出循环{Z}")
            return

        # 6. 执行工具调用
        print(f"\n  {_sep('Executing')}")
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            name = tc.function.name

            # ── 工具执行前校验 ──
            hook_result = hooks.trigger("PreToolUse", ToolUseEvent(
                tool_name=name,
                tool_args=args,
                tool_call_id=tc.id,
            ))
            if hook_result is not None:
                # 权限被拒绝，阻止工具执行
                print(f"  {R}🚫 {name} 被阻止: {hook_result}{Z}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"[Blocked] {hook_result}",
                })
                continue  # 跳过此工具，处理下一个

            # ── 执行工具 ──
            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                output = f"Error: Unknown tool '{name}'"
                print(f"  {R}✗{Z} {name} → {D}未知工具{Z}")
            else:
                output = handler(**args)
                # 打印执行结果
                cmd_preview = args.get('command', args.get('path', json.dumps(args, ensure_ascii=False)))
                status = G + "✓" + Z if not output.startswith("Error") else R + "✗" + Z
                print(f"  {status} {B}{name}{Z} → {D}{cmd_preview[:60]}{Z}")

            # 显示结果预览
            preview = output[:150].replace('\n', ' ')
            if len(output) > 150:
                preview += f"{D} ... ({len(output)} chars total){Z}"
            print(f"     {D}↳{Z} {preview}")

            # 工具执行后钩子
            hooks.trigger("PostToolUse", ToolResultEvent(
                tool_name=name,
                tool_args=args,
                tool_call_id=tc.id,
                output=output,
            ))

            # 把工具执行结果加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

        round_num += 1
        print(f"\n  {_sep()}")
        print(f"  {D}继续下一轮，将工具结果返回模型...{Z}")


# 在 main 函数中调用
if __name__ == "__main__":
    print(f"{C}╔{'═' * 58}╗{Z}")
    print(f"{C}║{Z}  {BOLD}🤖 Agent Loop — {M}{MODEL}{Z}                   {C}║{Z}")
    print(f"{C}║{Z}  {D}输入问题回车发送，输入 q 退出{Z}              {C}║{Z}")
    print(f"{C}╚{'═' * 58}╝{Z}\n")

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tools.TASK_FILE = WORKDIR / "todos" / f"todos_{session_ts}.json"

    messages = [
        {"role": "system", "content": SYSTEM},
    ]

    while True:
        try:
            query = input(f"\n{Y}👤 你:{Z} ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            print(f"  {D}👋 Goodbye{Z}")
            break

        # 添加用户消息
        messages.append({"role": "user", "content": query})

        # 运行 Agent 循环
        agent_loop(messages)

        # 打印最终回复
        print(f"\n{_box_top('🤖  Final Answer')}")
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                print(f"  {msg['content']}")
                break
        print(f"{C}{'═' * 60}{Z}\n")