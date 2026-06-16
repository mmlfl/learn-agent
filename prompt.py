from config import PROMPT_SECTIONS, WORKDIR
from skill_loader import SYSTEM_SKILL

SYSTEM_IDENTITY = "You are a coding agent."

SYSTEM_WORKFLOW = """
 You have access to tools and subagents.

## Task Workflow

When the user gives you a complex task:

1. **Plan at the goal level** — call todo_write to break the task into 2-5 meaningful subtasks. Each subtask should be a complete goal (e.g. "understand all source files"), NOT a single operation (e.g. "read main.py"). Avoid micro-tasking.

2. **Execute each subtask** — for the current subtask, decide:
   - If it requires multiple operations (reading several files, running commands then analyzing) → use `spawn_task` to delegate to a subagent
   - If it's a single operation (read one known file, run one command) → do it yourself

3. **Mark progress** — when a subtask is complete, use todo_write to mark it completed and move the next one to in_progress

4. **Repeat** until all subtasks are done, then deliver the final answer.

Always work one subtask at a time.
"""

PROMPT_SECTIONS["identity"] = SYSTEM_IDENTITY
PROMPT_SECTIONS["workflow"] = SYSTEM_WORKFLOW
PROMPT_SECTIONS["workspace"] = f"Working directory: {WORKDIR}"
PROMPT_SECTIONS["skill"] = SYSTEM_SKILL


def get_static_system_messages() -> list[dict]:
    """返回静态 system 消息，只在启动时调用一次"""
    messages = []
    for k, v in PROMPT_SECTIONS.items():
        messages.append({"role": "system", "content": v})
    return messages

