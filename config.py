from pathlib import Path

WORKDIR = Path.cwd()

SKILLS_DIR = WORKDIR / 'skills'

MEMORY_DIR = WORKDIR / 'memory'
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / 'MEMORY.md'

TOOL_RESULTS_DIR = WORKDIR / 'tools_budget_content'
TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_MESSAGES_LEN = 3