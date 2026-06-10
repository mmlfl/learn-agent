"""
UI module — Rich-based terminal rendering for the agent loop.

Design:
  - Rich Panels with color-coded borders per tool type
  - Show full args and output content (not metrics-only)
  - Todo output rendered as a formatted tree
  - Clean round separation with Rule lines
"""

import json
import os
import textwrap
import time
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.style import Style
from rich import box
from rich.status import Status
from rich.rule import Rule
from rich.columns import Columns

console = Console()


# ============================================================
# Theme — color palette & tool metadata
# ============================================================

class T:
    """Theme shortcuts."""
    CYAN    = "cyan"
    GREEN   = "green"
    RED     = "red"
    YELLOW  = "yellow"
    BLUE    = "blue"
    MAGENTA = "magenta"
    BRIGHT_CYAN = "bright_cyan"
    DIM     = "bright_black"
    BOLD    = "bold"
    WHITE   = "white"

# (label_str, color_str, border_color_str)
TOOL_META = {
    "bash":       ("bash",       T.BLUE,    "blue"),
    "read_file":  ("read",       T.GREEN,   "green"),
    "write_file": ("write",      T.GREEN,   "green"),
    "edit_file":  ("edit",       T.GREEN,   "green"),
    "glob":       ("glob",       T.BRIGHT_CYAN, "bright_cyan"),
    "grep":       ("grep",       T.BRIGHT_CYAN, "bright_cyan"),
    "todo_write": ("todo",       T.MAGENTA, "magenta"),
    "spawn_task": ("spawn",      T.CYAN,    "cyan"),
}


# ============================================================
# Helpers
# ============================================================

def _trunc(text: str, n: int) -> str:
    """Truncate to n chars, collapse newlines."""
    s = text.replace("\n", " ").replace("\r", "")
    if len(s) > n:
        return s[:n-3] + "..."
    return s


def _safe(text: str) -> str:
    """Strip Unicode replacement chars from decode errors."""
    return text.replace('�', '')


def _arg_str(args: dict) -> str:
    """Human-readable one-liner for tool arguments."""
    if "command" in args:
        return args["command"]
    if "path" in args:
        return args["path"]
    if "pattern" in args:
        return args["pattern"]
    if "description" in args:
        return args["description"]
    if "todos" in args:
        items = args["todos"]
        if not isinstance(items, list):
            return "?"
        c = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in items:
            s = t.get("status", "")
            if s in c: c[s] += 1
        parts = []
        if c["completed"]: parts.append(f"{c['completed']} done")
        if c["in_progress"]: parts.append(f"{c['in_progress']} active")
        if c["pending"]: parts.append(f"{c['pending']} pending")
        return ", ".join(parts) if parts else "0"
    return json.dumps(args, ensure_ascii=False)


def _time_fmt(s: float) -> str:
    if s < 1:   return f"{s*1000:.0f}ms"
    if s < 60:  return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec}s"


def _escape_markup(text: str) -> str:
    """Escape [ so Rich doesn't parse it as markup. ] is safe on its own."""
    return text.replace("[", "\\[")


def _format_output(output: str, max_lines: int = 20) -> str:
    """Prepare tool output for display: sanitize, limit lines, wrap long lines."""
    clean = _safe(output)
    lines = clean.splitlines()
    # Limit total lines
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... ({len(clean.splitlines()) - max_lines} more lines, {len(output)} chars total)")
    # Wrap long lines
    wrapped = []
    tw = _term_width() - 12  # leave room for panel padding
    for ln in lines:
        if len(ln) > tw:
            wrapped.extend(textwrap.wrap(ln, width=tw) or [ln])
        else:
            wrapped.append(ln)
    return _escape_markup("\n".join(wrapped))


def _term_width() -> int:
    try:
        return max(60, min(os.get_terminal_size().columns, 140))
    except OSError:
        return 100


# ============================================================
# Todo tree rendering
# ============================================================

def _render_todo_inline(output: str) -> None:
    """Parse todo_write output and print a compact status tree."""
    clean = _safe(output)
    icons = {"pending": " [ ]", "in_progress": " [>]", "completed": " [x]"}
    colors = {"pending": T.DIM, "in_progress": T.YELLOW, "completed": T.GREEN}
    any_found = False
    for ln in clean.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("Tasks") or ln.startswith("No"):
            continue
        for status in ("completed", "in_progress", "pending"):
            needle = f"[{status}]"
            if needle in ln:
                idx = ln.index(needle)
                tail = ln[idx + len(needle):].strip()
                if tail and tail[0].isdigit() and ':' in tail:
                    tail = tail.split(':', 1)[-1].strip()
                console.print(
                    Text(f"    {icons[status]} ", style=colors[status]) +
                    Text(_escape_markup(tail), style=colors[status])
                )
                any_found = True
                break
    if not any_found:
        for ln in clean.splitlines()[:10]:
            console.print(f"    [dim]{_escape_markup(_trunc(ln, 80))}[/]")


# ============================================================
# AgentLoopDisplay
# ============================================================

class AgentLoopDisplay:

    def __init__(self):
        self._t0: float = 0.0
        self._prev: float = 0.0

    # ── welcome ──
    def show_welcome(self, model: str) -> None:
        console.print()
        console.print(Panel(
            f"[bold cyan]Agent Loop[/]  |  [magenta]{model}[/]\n"
            f"[dim]plan > delegate > execute   |   q / exit / Ctrl+C to quit[/]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print()

    # ── round header ──
    def show_round_header(self, n: int) -> None:
        self._t0 = time.time()
        console.print()
        tag = f"Round {n}"
        if self._prev > 0:
            tag += f"  [dim](last {_time_fmt(self._prev)})[/]"
        console.print(Rule(tag, style="cyan", align="center"))

    # ── thinking spinner ──
    def model_thinking(self) -> Status:
        return console.status("[yellow]Thinking...[/]", spinner="dots")

    # ── model response ──
    def show_model_response(self, finish_reason: str, content: Optional[str],
                            elapsed: float) -> None:
        if finish_reason == "stop":
            reason, rcolor = "stop", T.GREEN
        else:
            reason, rcolor = "tool_calls", T.YELLOW

        console.print(Text(f"LLM {reason}  {_time_fmt(elapsed)}",
                          style=f"bold {rcolor}"))

        if content and content.strip():
            lines = [l for l in content.splitlines() if l.strip()]
            preview = "\n".join(lines[:4])
            if len(lines) > 4:
                preview += f"\n[dim]... ({len(lines)} lines)[/]"
            console.print(Panel(preview, title="Thought", title_align="left",
                                border_style=T.DIM, box=box.SIMPLE, padding=(0, 1)))

    # ── tool calls list ──
    def show_tool_calls(self, tool_calls: list) -> None:
        if not tool_calls:
            return
        console.print()  # spacing
        for i, tc in enumerate(tool_calls, 1):
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception:
                args = {}
            label, color, _ = TOOL_META.get(name, (name, T.DIM, T.DIM))
            arg = _arg_str(args)
            console.print(
                Text(f"[{i}]  ", style=T.DIM) +
                Text(label, style=f"bold {color}") +
                Text(f"  {_trunc(arg, 70)}", style=T.DIM)
            )

    # ── single tool result ──
    def show_tool_result(self, name: str, args: dict, output: str,
                         label: str = "", success: bool = True) -> None:
        tool_label, tool_color, border_color = TOOL_META.get(
            name, (name, T.DIM, T.DIM))
        arg = _arg_str(args)
        status_tag = "OK" if success else "FAIL"
        status_color = T.GREEN if success else T.RED

        # Header line
        hdr = Text()
        hdr.append(f"[{status_tag}] ", style=f"bold {status_color}")
        hdr.append(f"{tool_label}  ", style=f"bold {tool_color}")
        hdr.append(arg, style=T.DIM)
        if label:
            hdr.append(f"  [{label}]", style=T.YELLOW)
        console.print(hdr)

        # Output body — show in colored panel (except todo: tree is better)
        if output and name != "todo_write":
            body = _format_output(output)
            if body.strip():
                bcol = T.RED if not success else border_color
                console.print(Panel(
                    body,
                    border_style=bcol,
                    box=box.SIMPLE,
                    padding=(0, 1),
                ))

        # Special: render todo tree after the panel
        if name == "todo_write" and success:
            _render_todo_inline(output)

    # ── round end ──
    def show_round_end(self) -> None:
        self._prev = time.time() - self._t0 if self._t0 else 0

    # ── final answer ──
    def show_final_answer(self, content: str) -> None:
        console.print()
        console.print(Rule("Done", style="cyan", align="center"))
        console.print(Panel(content, border_style="cyan", box=box.ROUNDED, padding=(0, 2)))
        console.print()

    # ── goodbye ──
    def show_goodbye(self) -> None:
        console.print("\n  [dim]Bye.[/]\n")


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
                "output": tc.get("output", ""),
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
        console.print()
        console.print(Panel(
            f"[bold magenta]SubAgent[/]  [dim]|[/]  {description}",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        return tracker

    def finish(self, tracker: SubAgentTracker, result: Optional[str]) -> None:
        ok = result is not None
        tag = "DONE" if ok else "WARN"
        tcolor = T.GREEN if ok else T.RED

        console.print()
        summary = Text()
        summary.append(f"[{tag}] ", style=f"bold {tcolor}")
        summary.append(f"{tracker.rounds} rounds", style=T.BOLD)
        summary.append(f"  {tracker.tool_count} tools", style=T.DIM)
        summary.append(f"  {_time_fmt(tracker.elapsed)}", style=T.DIM)
        console.print(summary)

        # Tool log tree
        if tracker.tool_logs:
            tree = Tree("[dim]calls[/]", guide_style=T.DIM)
            cur_round = 0
            rnode = None
            for log in tracker.tool_logs:
                if log["round"] != cur_round:
                    cur_round = log["round"]
                    rnode = tree.add(f"[bold]R{cur_round}[/]")
                tl, tc, _ = TOOL_META.get(log["name"], (log["name"], T.DIM, T.DIM))
                a = _trunc(_arg_str(log["args"]), 60)
                out = _safe(log["output"])
                m = f"{len(out.splitlines())}L {len(out)}C"
                if rnode is not None:
                    rnode.add(
                        Text(tl, style=tc) +
                        Text(f"  {a}  ", style=T.DIM) +
                        Text(f"-> {m}", style=T.DIM)
                    )
            console.print(tree)

        if result:
            console.print(Panel(
                result[:300],
                title="Return", title_align="left",
                border_style=T.DIM,
                box=box.SIMPLE,
                padding=(0, 1),
            ))
        console.print()


# ============================================================
# Global singletons
# ============================================================

agent_display = AgentLoopDisplay()
subagent_display = SubAgentDisplay()
