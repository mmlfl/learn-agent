import json
import os
from pathlib import Path

from openai import OpenAI

from config import TOOL_RESULTS_DIR, SYSTEM_MESSAGES_LEN
from ui import agent_display

LAYER1_LEN = 30
LAYER2_MAX_TOKEN = 100000
PERSIST_THRESHOLD = 10_000

COMPACT_CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


KEEP_TOOL_RESULTS = 2

def compact_layer1(messages: list):
    """第一层压缩：保留最后 KEEP_TOOL_RESULTS 条工具结果，其余替换为占位符。

    不碰 system 头部、user 消息、assistant 消息，只压缩旧的 tool 消息。
    一轮结束后调用一次，不在每次工具调用后重复执行。
    """
    # 找出所有 tool 消息的位置
    tool_positions = [i for i, msg in enumerate(messages) if msg["role"] == "tool"]

    if len(tool_positions) <= KEEP_TOOL_RESULTS:
        return messages  # 不够多，不用压缩

    # 最后 KEEP_TOOL_RESULTS 条保留，之前的全部压缩
    cutoff = tool_positions[-(KEEP_TOOL_RESULTS)]  # 倒数第 N 条的位置
    to_compress = [p for p in tool_positions if p < cutoff]

    if not to_compress:
        return messages

    compacted = 0
    for idx in to_compress:
        name = messages[idx].get("name", "unknown")
        messages[idx]["content"] = f"[Earlier tool result compacted] used {name}"
        compacted += 1

    if compacted:
        agent_display.show_compact("1", f"将 {compacted} 条旧 tool 结果替换为占位符")

    return messages


def compact_layer2(messages: list):
    if len(str(messages)) < LAYER2_MAX_TOKEN:
        return messages

    before_chars = len(str(messages))
    system_headers = messages[0:SYSTEM_MESSAGES_LEN]
    conversation = json.dumps(messages[SYSTEM_MESSAGES_LEN:], ensure_ascii=False, default=str)

    COMPACT_PROMPT = (
        "You are a conversation compressor. Summarize the agent conversation below "
        "so work can resume without losing context.\n\n"
        "## Must Preserve\n"
        "1. **Current goal** — what the user asked for, and which subtask is in progress\n"
        "2. **Key findings** — important discoveries, decisions made, errors encountered\n"
        "3. **Files touched** — which files were read, written, or edited, and WHY\n"
        "4. **Remaining work** — what still needs to be done\n"
        "5. **User constraints** — any preferences, restrictions, or special requests\n\n"
        "## Rules\n"
        "- Be specific: use actual file paths, function names, line numbers\n"
        "- Keep it under 500 words\n"
        "- Do NOT call any tools — output plain text only\n\n"
        "## Conversation\n"
        f"{conversation}"
    )

    result = COMPACT_CLIENT.chat.completions.create(
        model=os.getenv("COMPACT_MODEL_NAME"),
        messages=[{"role": "user", "content": COMPACT_PROMPT}],
        extra_body={"enable_thinking": False},
    )

    system_headers.append({
        "role": "user",
        "content": f"[Context Compressed]\n\n{result.choices[0].message.content}"
    })

    after_chars = len(str(system_headers))
    agent_display.show_compact(
        "2", f"LLM 全量摘要 — {before_chars:,} → {after_chars:,} chars"
    )
    return system_headers


def reactive_compact(messages: list) -> list:
    """应急压缩：API 返回 prompt_too_long 时触发，比 L2 更激进。

    保留 system_headers + LLM 摘要 + 最后 5 条消息（保留当前对话骨架）。
    """
    before_chars = len(str(messages))
    system_headers = messages[0:SYSTEM_MESSAGES_LEN]
    tail = messages[-5:]
    conversation = json.dumps(messages[SYSTEM_MESSAGES_LEN:-5], ensure_ascii=False, default=str)

    prompt = (
        "Emergency summary. The conversation is too long for the context window.\n"
        "Summarize the conversation below, preserving ONLY what is essential:\n"
        "1. Current goal and which subtask is in progress\n"
        "2. Key findings and decisions\n"
        "3. Files modified and why\n"
        "4. Remaining work\n"
        "Be extremely concise — under 200 words.\n"
        "Do NOT call any tools — output plain text only.\n\n"
        f"{conversation}"
    )

    result = COMPACT_CLIENT.chat.completions.create(
        model=os.getenv("COMPACT_MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )

    compressed = [{
        "role": "user",
        "content": f"[Emergency Compressed]\n\n{result.choices[0].message.content}"
    }]

    after_chars = len(str(system_headers + compressed + tail))
    agent_display.show_compact(
        "!", f"应急压缩 — {before_chars:,} → {after_chars:,} chars"
    )
    return system_headers + compressed + tail


def tool_result_budget(output: str, tool_call_id: str,
                       name: str = "", args: dict = None) -> str:
    if len(output) < PERSIST_THRESHOLD:
        return output

    # 生成可读文件名: {tool_name}_{arg_hint}_{id前8位}.txt
    arg_hint = ""
    if args:
        # 优先取 path/pattern/file 等参数的文件名部分
        for key in ("path", "pattern", "file"):
            val = args.get(key, "")
            if val:
                arg_hint = "_" + Path(val).name
                break
        # 如果都没有，取 command 的前 20 字符
        if not arg_hint:
            cmd = args.get("command", "")
            if cmd:
                safe = cmd[:20].replace(" ", "_").replace("/", "_").replace("\\", "_")
                arg_hint = "_" + safe

    filename = f"{name}{arg_hint}_{tool_call_id[:8]}.txt"
    path = TOOL_RESULTS_DIR / filename
    path.write_text(output, encoding="utf-8")

    agent_display.show_compact(
        "3", f"{name} → {len(output):,} chars, 落盘至 {path.name}"
    )

    return (
        f"<tool-output-truncated>\n"
        f"Size: {len(output):,} chars (showing first 500). "
        f"Full output saved to {path}\n"
        f"IMPORTANT: Do NOT read this file in full — it will be truncated again.\n"
        f"To find specific content, use bash: grep, head -n N, or tail -n N.\n"
        f"Preview:\n{output[:500]}\n"
        f"</tool-output-truncated>\n"
    )