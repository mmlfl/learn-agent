"""
UI module — pretty terminal output + session logging.

Terminal:  visual structure with Rule/Panel, color-coded tools, symbols
Log file:  raw tool calls with args (logger.py)
"""

import json
import time
from typing import Optional

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich import box
from rich.status import Status
from rich.rule import Rule

from logger import logger

console = Console()


# ============================================================
# Style constants
# ============================================================

# Tool -> (label, color, symbol)
TOOL = {
    "bash":       ("bash",       "blue",         ">_"),
    "read_file":  ("read_file",  "green",        "R "),
    "write_file": ("write_file", "green",        "W "),
    "edit_file":  ("edit_file",  "green",        "E "),
    "glob":       ("glob",       "bright_cyan",  "* "),
    "grep":       ("grep",       "bright_cyan",  "grep"),
    "todo_write": ("todo",       "magenta",      "T "),
    "spawn_task": ("spawn",      "cyan",         "S "),
}


# ============================================================
# Helpers
# ============================================================

def _arg_summary(args: dict) -> str:
    for key in ("command", "path", "pattern", "description"):
        if key in args:
            val = str(args[key])
            return val[:55] + "..." if len(val) > 55 else val
    if "todos" in args:
        items = args["todos"]
        if isinstance(items, list):
            c = {"pending": 0, "in_progress": 0, "completed": 0}
            for t in items:
                s = t.get("status", "")
                if s in c: c[s] += 1
            parts = []
            if c["in_progress"]: parts.append(f"{c['in_progress']} active")
            if c["pending"]: parts.append(f"{c['pending']} pending")
            if c["completed"]: parts.append(f"{c['completed']} done")
            return ", ".join(parts) if parts else "0"
    return json.dumps(args, ensure_ascii=False)[:40]


def _time_fmt(s: float) -> str:
    if s < 1:   return f"{s*1000:.0f}ms"
    if s < 60:  return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec}s"


# ============================================================
# AgentLoopDisplay
# ============================================================

class AgentLoopDisplay:

    def __init__(self):
        self._t0: float = 0.0
        self._prev: float = 0.0
        self._round: int = 0

    # ── welcome ──
    def show_welcome(self, model: str) -> None:
        log_path = logger.open(model)
        console.print()
        console.print(Panel(
            f"[bold cyan]Agent Loop[/]  |  [magenta bold]{model}[/]  |  "
            f"[dim]q / exit / Ctrl+C[/]\n"
            f"[dim]log: {log_path}[/]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print()

    # ── round header ──
    def show_round_header(self, n: int) -> None:
        self._t0 = time.time()
        self._round = n
        logger.log_round_header(n)
        if self._prev > 0:
            console.print(Rule(
                f"[bold]Round {n}[/]  [dim]({_time_fmt(self._prev)})[/]",
                style="cyan", align="center",
            ))
        else:
            console.print(Rule(f"[bold]Round {n}[/]", style="cyan", align="center"))

    # ── thinking ──
    def model_thinking(self) -> Status:
        return console.status(
            f"[yellow]Round {self._round}  thinking...[/]",
            spinner="dots",
        )

    # ── model response ──
    def show_model_response(self, finish_reason: str, content: Optional[str],
                            elapsed: float) -> None:
        logger.log_model_response(finish_reason, elapsed)
        is_stop = finish_reason == "stop"
        color = "green" if is_stop else "yellow"
        label = "STOP" if is_stop else "TOOLS"
        console.print(f"  [bold {color}]{label}[/] [dim]{_time_fmt(elapsed)}[/]")

    # ── tool calls (before execution) ──
    def show_tool_calls(self, tool_calls: list) -> None:
        if not tool_calls:
            return
        for tc in tool_calls:
            name = tc.function.name
            raw = tc.function.arguments
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            logger.log_tool_call(name, raw)
            # Terminal: symbol + colored name + dim args
            label, color, sym = TOOL.get(name, (name, "white", "? "))
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                args = {}
            arg = _arg_summary(args)
            console.print(
                Text(f"  {sym} ", style="dim") +
                Text(label, style=f"bold {color}") +
                Text(f"  {arg}", style="dim")
            )

    # ── tool result (after execution) ──
    def show_tool_result(self, name: str, args: dict, output: str,
                         label: str = "", success: bool = True) -> None:
        logger.log_tool_result(name, success)
        tag = "OK" if success else "FAIL"
        sc = "green" if success else "red"
        _, color, sym = TOOL.get(name, (name, "white", "? "))
        arg = _arg_summary(args)

        console.print(
            Text(f"  [{tag}] ", style=f"bold {sc}") +
            Text(sym, style="dim") +
            Text(name, style=color) +
            Text(f"  {arg}", style="dim") +
            (Text(f"  {label}", style="yellow") if label else Text(""))
        )

    # ── round end ──
    def show_round_end(self) -> None:
        self._prev = time.time() - self._t0 if self._t0 else 0

    # ── final answer ──
    def show_final_answer(self, content: str) -> None:
        logger.log_final_answer(content)
        console.print()
        console.print(Rule("Done", style="cyan", align="center"))
        console.print(Panel(content, border_style="cyan", box=box.ROUNDED, padding=(0, 2)))
        console.print()

    # ── goodbye ──
    def show_goodbye(self) -> None:
        logger.close()
        console.print(f"\n  [dim]log -> {logger.file_path}[/]\n")


# ============================================================
# SubAgentTracker
# ============================================================

class SubAgentTracker:
    def __init__(self, description: str):
        self.description = description
        self.rounds = 0
        self.tool_count = 0
        self.tool_logs: list[dict] = []
        self._start = time.time()

    def add_round(self, tool_calls: list[dict]) -> None:
        self.rounds += 1
        for tc in tool_calls:
            self.tool_count += 1
            self.tool_logs.append({
                "round": self.rounds,
                "name": tc.get("name", "?"),
                "args": tc.get("args", {}),
                "args_raw": tc.get("args_raw", ""),
            })

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


# ============================================================
# SubAgentDisplay
# ============================================================

class SubAgentDisplay:

    def start(self, description: str) -> SubAgentTracker:
        tracker = SubAgentTracker(description)
        logger.log_subagent_start(description)
        desc_short = description[:55] + "..." if len(description) > 55 else description
        console.print(
            Text("  [sub] ", style="dim") +
            Text(desc_short, style="magenta")
        )
        return tracker

    def finish(self, tracker: SubAgentTracker, result: Optional[str]) -> None:
        for log in tracker.tool_logs:
            raw = log.get("args_raw", "")
            if not raw:
                raw = json.dumps(log["args"], ensure_ascii=False)
            logger.log_subagent_tool(log["name"], raw)

        logger.log_subagent_finish(tracker.rounds, tracker.tool_count,
                                   tracker.elapsed)

        ok = result is not None
        tag = "OK" if ok else "FAIL"
        sc = "green" if ok else "red"

        console.print(
            Text(f"  [{tag}] sub ", style=f"bold {sc}") +
            Text(f"{tracker.rounds}r/{tracker.tool_count}t", style="dim") +
            Text(f"  {_time_fmt(tracker.elapsed)}", style="dim")
        )


# ============================================================
# Global singletons
# ============================================================

agent_display = AgentLoopDisplay()
subagent_display = SubAgentDisplay()
