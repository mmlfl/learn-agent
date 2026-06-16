import json
import os
import re

from openai import OpenAI

from config import MEMORY_DIR, MEMORY_INDEX

MEMORY_CLIENT = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)
CONSOLIDATE_THRESHOLD = 10


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """写入一个记忆文件,带yaml frontMatter"""
    slug = name.lower().replace(" ", "-").replace("/", "-")
    file_name = f"{slug}.md"
    file_path = MEMORY_DIR / file_name
    file_path.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    _rebuild_index()
    return file_path


def _rebuild_index():
    """从所有记忆文件重建 MEMORY.md 索引"""
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        meta, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def read_memory_index() -> str:
    """读取 MEMORY.md 索引（每轮注入 SYSTEM）"""
    if not MEMORY_INDEX.exists():
        return ""
    return MEMORY_INDEX.read_text(encoding="utf-8").strip() or ""


def read_memory_file(filename: str) -> str | None:
    """读取单个记忆文件的完整内容"""
    path = MEMORY_DIR / filename
    return path.read_text(encoding="utf-8") if path.exists() else None


def list_memory_files() -> list[dict]:
    """列出所有记忆文件及其元数据"""
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result


def extract_memories(messages: list):
    """从最近对话中提取新记忆,每轮结束后运行"""
    # 取最近3条用户消息（跳过工具调用和 AI 回复，只提取用户表达的偏好）
    dialogue_parts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str) and content.strip():
                dialogue_parts.append(content)
            if len(dialogue_parts) >= 3:
                break
    dialogue = "\n".join(reversed(dialogue_parts))
    # 列出已有记忆，避免重复
    existing = list_memory_files()
    existing_desc = (
        "\n".join(f"- {m['name']}: {m['description']}" for m in existing)
        if existing else "(none)"
    )

    prompt = (
        "Extract ONLY facts the USER explicitly stated — preferences, constraints, or background.\n"
        "IGNORE anything the assistant discovered through tool calls (file contents, env vars,\n"
        "system info, project structure). If the user didn't say it, don't extract it.\n"
        "Return a JSON array. Each item must have these fields:\n"
        "- name: kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (personal preference), 'feedback' (how to work),\n"
        "  'project' (project fact/context), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown. End with **Why:** and **How to apply:** lines.\n\n"
        "Examples:\n"
        '[{"name": "user-preference-tabs",'
        ' "type": "user",'
        ' "description": "User prefers tabs for indentation",'
        ' "body": "User prefers using tabs, not spaces, for indentation.\\n\\n'
        '**Why:** Consistency with existing codebase conventions.\\n'
        '**How to apply:** Always use tabs when writing or editing files."},\n'
        ' {"name": "feedback-no-mock-database",'
        ' "type": "feedback",'
        ' "description": "Do not mock the database in tests",'
        ' "body": "The user instructed not to mock the database layer in tests.\\n\\n'
        '**Why:** Mocking hides real bugs; prefer test containers or fixtures.\\n'
        '**How to apply:** Use real DB test helpers instead of unittest.mock."}]\n\n'
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )
    try:
        response = MEMORY_CLIENT.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "system", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            if mem.get("description") and mem.get("body"):
                write_memory_file(mem["name"], mem.get("type", "user"),
                                  mem["description"], mem["body"])
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass


def consolidate_memories():
    """记忆文件达到阈值时触发去重合并"""
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return
    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )
    try:
        response = MEMORY_CLIENT.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "system", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()
        count = 0
        for mem in items:
            if mem.get("description") and mem.get("body"):
                write_memory_file(mem["name"], mem.get("type", "user"),
                                  mem["description"], mem["body"])
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    files = list_memory_files()
    if not files:
        return []
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # ... 处理多模态 ...
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]
    # 构建目录（只传 name + description，不传 body）
    catalog = "\n".join(
        f"{i}: {f['name']} — {f['description']}" for i, f in enumerate(files)
    )
    # 独立的 side-query，不跟主对话混合
    prompt = (
        "Select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )
    try:
        response = MEMORY_CLIENT.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "system", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not match:
            return []
        indices = json.loads(match.group())
        result = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(files):
                result.append(files[idx]["filename"])
                if len(result) >= max_items:
                    break
        return result
    except Exception:
        return []


def load_memories(messages: list) -> str:
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""

    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    messages.append({"role": "user","content": "\n\n".join(parts)})
    return "\n\n".join(parts)


def build_memory_system():
    catalog = read_memory_index()
    return (
        f"memories available:\n{catalog}\n"
        "Use load_memory to get full details when needed."
    )

