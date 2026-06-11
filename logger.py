"""
Session logger — writes Markdown-formatted log for easy reading.

Open logs/session_xxx.md in VS Code (Ctrl+Shift+V) for rendered view.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

WORKDIR = Path.cwd()


class SessionLogger:

    def __init__(self):
        self._file = None
        self._path: Optional[Path] = None
        self._start_time: float = 0.0

    # ── open / close ──

    def open(self, model: str) -> Path:
        log_dir = WORKDIR / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"session_{ts}.md"
        self._file = open(self._path, "w", encoding="utf-8")
        self._start_time = time.time()

        self._w(f"# Agent Session Log")
        self._w("")
        self._w(f"| | |")
        self._w(f"|---|---|")
        self._w(f"| **Date** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        self._w(f"| **Model** | `{model}` |")
        self._w(f"| **Dir** | `{WORKDIR}` |")
        self._w("")
        self._w("---")
        self._w("")
        return self._path

    def close(self) -> None:
        if self._file:
            elapsed = time.time() - self._start_time
            m, s = divmod(int(elapsed), 60)
            self._w("")
            self._w("---")
            self._w("")
            self._w(f"*Session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · duration {m}m{s}s*")
            self._file.close()
            self._file = None

    @property
    def file_path(self) -> Optional[Path]:
        return self._path

    # ── internal ──

    def _w(self, line: str) -> None:
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def _json_block(self, raw: str) -> str:
        """Try to parse and pretty-print JSON; fall back to raw."""
        import json as _json
        try:
            obj = _json.loads(raw)
            return _json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return raw

    # ── session events ──

    def log_user_message(self, query: str) -> None:
        self._w("## User")
        self._w("")
        self._w(f"> {query}")
        self._w("")
        self._w("---")
        self._w("")

    def log_round_header(self, n: int) -> None:
        self._w(f"### Round {n}")
        self._w("")

    def log_model_response(self, finish_reason: str, elapsed: float) -> None:
        emoji = "stop" if finish_reason == "stop" else "tools"
        if finish_reason == "stop":
            self._w(f"**LLM** `STOP` *{elapsed:.1f}s*")
        else:
            self._w(f"**LLM** `TOOLS` *{elapsed:.1f}s*")
        self._w("")

    def log_tool_call(self, name: str, raw_args: str) -> None:
        self._w(f"#### `{name}`")
        self._w("")
        if raw_args:
            body = self._json_block(raw_args)
            if len(body) > 8000:
                body = body[:8000] + f"\n\n... truncated ({len(body)} chars total)"
            self._w("```json")
            self._w(body)
            self._w("```")
        else:
            self._w("*(no args)*")
        self._w("")

    def log_tool_result(self, name: str, success: bool) -> None:
        tag = "OK" if success else "FAIL"
        emoji = ":white_check_mark:" if success else ":x:"
        self._w(f"> {emoji} **{tag}**")
        self._w("")

    def log_final_answer(self, content: str) -> None:
        self._w("---")
        self._w("")
        self._w("## Final Answer")
        self._w("")
        self._w(content)
        self._w("")

    def log_subagent_start(self, description: str) -> None:
        self._w("> 🐳 **SubAgent**")
        self._w(f"> {description}")
        self._w("")

    def log_subagent_tool(self, name: str, raw_args: str) -> None:
        self._w(f"* `{name}`")
        if raw_args:
            body = self._json_block(raw_args)
            self._w(f"  ```json")
            for line in body.splitlines():
                self._w(f"  {line}")
            self._w(f"  ```")
        self._w("")

    def log_subagent_finish(self, rounds: int, tools: int, elapsed: float) -> None:
        self._w(f"> :white_check_mark: **SubAgent done** · {rounds}r · {tools} tools · {elapsed:.1f}s")
        self._w("")


logger = SessionLogger()
