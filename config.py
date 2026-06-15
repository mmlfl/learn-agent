from pathlib import Path

WORKDIR = Path.cwd()

SKILLS_DIR = WORKDIR / 'skills'

MEMORY_DIR = WORKDIR / 'memory'
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / 'MEMORY.md'