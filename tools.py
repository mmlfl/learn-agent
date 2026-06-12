import json
import os
import subprocess
from pathlib import Path
from openai import OpenAI

from hooks import hooks, ToolUseEvent
from config import WORKDIR
from skill_loader import SKILL_REGISTRY
from ui import subagent_display, console
TASK_FILE = Path()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "bash command to run",
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of bytes to read (default 10000)",
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "New text to insert",
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g. /tmp/*.log)",
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "Create and manage a task list. Use this tool to add, update, or mark tasks as complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The list of tasks to create or update.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "任务唯一标识"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "任务标题"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "任务状态"
                                }
                            },
                            "required": ["id", "title", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_task",
            "description": ("Launch a subagent to handle complex, multi-step subtasks independently"
                           "(e.g. reading multiple files and summarizing). The subagent has fresh context and returns only its conclusion."
                            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "a task description let sub agent know what is this task"
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "if a skill is suitable for this task, load the skill and return it ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "the skill's name which you need"
                    }
                },
                "required": ["name"]
            }
        }
    },
]


# ── Tool execution ────────────────────────────────────────
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(**kwargs) -> str:
    command = kwargs.get("command")
    if not command:
        return "Error: Missing 'command' parameter"

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, encoding='utf-8', errors='replace',
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(**kwargs) -> str:
    path = kwargs.get("path")
    if not path:
        return "Error: Missing 'path' parameter"

    limit = kwargs.get("limit", 10000)

    try:
        # 🔧 指定 UTF-8 编码读取文件
        content = safe_path(path).read_text(encoding='utf-8')
        lines = content.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        return content
    except UnicodeDecodeError:
        # 🔧 如果 UTF-8 失败，尝试 GBK
        try:
            content = safe_path(path).read_text(encoding='utf-8')
            lines = content.splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
                return "\n".join(lines)
            return content
        except Exception as e:
            return f"Error: Cannot read file (encoding issue): {e}"
    except Exception as e:
        return f"Error: {e}"


def run_write(**kwargs) -> str:
    path = kwargs.get("path")
    if not path:
        return "Error: Missing 'path' parameter"

    content = kwargs.get("content")
    if content is None:
        return "Error: Missing 'content' parameter"

    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(**kwargs) -> str:
    path = kwargs.get("path")
    if not path:
        return "Error: Missing 'path' parameter"

    old_text = kwargs.get("old_text")
    if old_text is None:
        return "Error: Missing 'old_text' parameter"

    new_text = kwargs.get("new_text")
    if new_text is None:
        return "Error: Missing 'new_text' parameter"

    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: '{old_text}' not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1),encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(**kwargs) -> str:
    pattern = kwargs.get("pattern")
    if not pattern:
        return "Error: Missing 'pattern' parameter"

    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def run_todo_write(todos: list) -> str:
    # 1.读已有任务 (如果文件已存在)
    existing = {}
    if TASK_FILE and TASK_FILE.exists():
        existing = {t["id"]: t for t in json.loads(TASK_FILE.read_text(encoding="utf-8"))}
    # 2.增量合并
    for t in todos:
        existing[t["id"]] = t  # 相同id更新,新id新增
    merged = list(existing.values())
    # 3. 写回
    if TASK_FILE:
        TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
        TASK_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    # 4. 返回格式化的任务清单给模型看
    icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
    lines = []
    for t in merged:
        lines.append(f"  {icons.get(t['status'], '?')} [{t['status']}] {t['id']}: {t['title']}")
    return "Tasks:\n" + "\n".join(lines) if lines else "No tasks."


SUB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "bash command to run",
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of bytes to read (default 10000)",
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "New text to insert",
                    }
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g. /tmp/*.log)",
                    }
                },
                "required": ["pattern"]
            }
        }
    },
]
SUB_TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


def extract_content(messages: list[dict]) -> str:
    result = []
    for message in messages:
        if message.get("role") == "tool":
            result.append(message["content"])
    return "\n".join(result)


def run_spawn_task(**kwargs) -> str | None:
    """创建子 agent 处理主 agent 发过来的任务，最后将总结返回。"""
    SYSTEM_PROMPT = """
        you are a subagent,you just to complete this task and return the result to master agent
    """
    description = kwargs.get("description", "no input")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    client = OpenAI(
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # ── 开始追踪子 agent ──
    tracker = subagent_display.start(description)

    round_time = 0
    while round_time < 30:
        round_time += 1

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen-vl-plus"),
            messages=messages,
            tools=SUB_TOOLS,
            tool_choice="auto",
            max_tokens=4096
        )
        choice = response.choices[0]

        # 子 agent 返回最终结果
        if choice.finish_reason != "tool_calls":
            subagent_display.finish(tracker, choice.message.content)
            return choice.message.content

        # 加入对话历史
        messages.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": choice.message.tool_calls if choice.message.tool_calls else None
        })

        # 记录本轮工具调用
        round_logs: list[dict] = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = tc.function.arguments
                try:
                    if isinstance(args, str):
                        args = json.loads(args)
                except Exception:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"[Blocked] {args} this is wrong,please try this again",
                        "name": name
                    })

                hook_result = hooks.trigger("PreToolUse", ToolUseEvent(
                    tool_name=name,
                    tool_args=args,
                    tool_call_id=tc.id
                ))
                if hook_result is not None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"[Blocked] {hook_result}",
                        "name": name
                    })
                    round_logs.append({"name": name, "args": args, "output": f"[Blocked] {hook_result}"})
                    continue

                handler = SUB_TOOL_HANDLERS.get(name)
                if handler is None:
                    output = f"Error: Unknown tool '{name}'"
                else:
                    output = handler(**args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"{output}",
                    "name": name
                })
                round_logs.append({
                    "name": name,
                    "args": args,
                    "args_raw": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments, ensure_ascii=False),
                })

        # 将本轮工具调用记录到 tracker
        if round_logs:
            tracker.add_round(round_logs)

    # 超过 30 轮
    console.print("  [yellow]⚠ 子 agent 思考轮数超过了 30 轮[/]")
    result = extract_content(messages)
    subagent_display.finish(tracker, result)
    return result

def run_load_skill(**kwargs) -> str | None:
    name = kwargs.get("name","no input")
    if name == "no input":
        return "Wrong arguments,please try again"
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return "No such skill"
    if skill.get("enabled", True) == False:
        return "Skill disabled"
    return skill.get("content")

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "spawn_task": run_spawn_task,"load_skill":run_load_skill,
}
