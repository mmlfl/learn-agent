"""
UI 模块 — 基于 Rich 的终端界面渲染。

职责：
  - AgentLoopDisplay: 主 agent 循环展示（轮次、模型回复、工具调用、最终回复）
  - SubAgentDisplay:   子 agent 展示（摘要行 + 可折叠工具调用树）
  - Theme:             统一的颜色/样式常量

用法：
  from ui import agent_display, subagent_display, console

  # 主 agent
  agent_display.show_welcome("qwen-plus")
  agent_display.show_round_header(1)
  with agent_display.model_thinking():
      response = client.chat.completions.create(...)
  agent_display.show_model_response("tool_calls", content, 1.2)
  ...

  # 子 agent
  tracker = subagent_display.start("探索项目结构")
  # ... 执行工具调用，用 tracker.add_round(...) 记录 ...
  subagent_display.finish(tracker, result_text)
"""

import time
import json
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.style import Style
from rich import box
from rich.status import Status

# ── 全局 Console 单例 ────────────────────────────────────
console = Console()

# ═══════════════════════════════════════════════════════════
# Theme — 统一颜色/样式
# ═══════════════════════════════════════════════════════════

class Theme:
    TITLE     = Style(color="cyan", bold=True)
    SUCCESS   = Style(color="green")
    TOOL      = Style(color="blue")
    DIM       = Style(color="bright_black")
    SUBAGENT  = Style(color="magenta")
    ERROR     = Style(color="red")
    MODEL     = Style(color="yellow")
    USER      = Style(color="yellow", bold=True)
    HIGHLIGHT = Style(color="magenta", bold=True)
    BOLD      = Style(bold=True)
    RULE      = Style(color="bright_black")


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _truncate(text: str, max_len: int = 80) -> str:
    """截断文本，替换换行为空格"""
    text = text.replace("\n", " ").replace("\r", "")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _tool_arg_preview(args: dict) -> str:
    """从工具参数中提取可读的简短描述"""
    if "command" in args:
        return _truncate(args["command"], 60)
    if "path" in args:
        return _truncate(args["path"], 60)
    if "description" in args:
        return _truncate(args["description"], 60)
    if "todos" in args:
        return f"{len(args['todos'])} tasks"
    return _truncate(json.dumps(args, ensure_ascii=False), 60)


def _output_preview(output: str, max_len: int = 200) -> str:
    """工具输出预览"""
    if not output:
        return "(空)"
    preview = output[:max_len].replace("\n", " ")
    if len(output) > max_len:
        preview += f"  ... ({len(output)} chars)"
    return preview


# ═══════════════════════════════════════════════════════════
# AgentLoopDisplay — 主 agent 循环渲染
# ═══════════════════════════════════════════════════════════

class AgentLoopDisplay:
    """主 Agent 循环的终端渲染器。"""

    def __init__(self):
        self._round_start: float = 0.0

    # ── 欢迎横幅 ────────────────────────────────────────
    def show_welcome(self, model: str) -> None:
        console.print()
        console.print(Panel(
            f"[bold cyan]🤖 Agent Loop[/]  —  [magenta]{model}[/]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print(f"  [dim]输入问题回车发送，输入 q 退出[/]")
        console.print()

    # ── 轮次标题 ────────────────────────────────────────
    def show_round_header(self, n: int) -> None:
        self._round_start = time.time()
        console.print()
        # 用一条水平线 + 标题 代替原来的粗框
        text = Text()
        text.append("─" * 56, style=Theme.RULE)
        text.append(f"\n🔄 Round {n}", style=Theme.TITLE)
        console.print(text)

    # ── 模型思考 spinner ───────────────────────────────
    def model_thinking(self, label: str = "模型思考中") -> Status:
        """返回一个 Rich Status context manager，包裹 API 调用。

        用法:
            with agent_display.model_thinking():
                response = client.chat.completions.create(...)
        """
        return console.status(
            f"[yellow]⏳ {label}...[/]",
            spinner="dots",
        )

    # ── 模型回复 ────────────────────────────────────────
    def show_model_response(self, finish_reason: str, content: Optional[str],
                            elapsed: float, agent_label: str = "模型回复") -> None:
        """显示模型返回的 finish_reason、文本内容、耗时。"""
        reason_style = Theme.SUCCESS if finish_reason == "stop" else Theme.MODEL
        line = Text()
        line.append(f"💬 {agent_label}", style=Theme.BOLD)
        line.append("  ·  ", style=Theme.DIM)
        line.append(finish_reason, style=reason_style)
        line.append(f"  ·  {elapsed:.1f}s", style=Theme.DIM)
        console.print(line)

        if content:
            console.print(Panel(
                content[:600],
                title="Response",
                title_align="left",
                border_style=Theme.DIM,
                padding=(0, 1),
            ))

    # ── 工具调用清单 ────────────────────────────────────
    def show_tool_calls(self, tool_calls: list) -> None:
        """显示本轮所有工具调用的清单（一行一个）。"""
        if not tool_calls:
            return
        console.print(Text("🔧 工具调用", style=Theme.BOLD))

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except Exception:
                args = {}
            arg_str = _tool_arg_preview(args)
            if name == "spawn_task":
                console.print(f"  🐳 [magenta]spawn_task[/] → [dim]{arg_str}[/]")
            else:
                console.print(f"  · [blue]{name}[/][dim]({arg_str})[/]")

    # ── 单个工具结果 ────────────────────────────────────
    def show_tool_result(self, name: str, args: dict, output: str,
                         label: str = "", success: bool = True) -> None:
        """显示单个工具的执行结果（一行摘要）。"""
        icon = "✓" if success else "✗"
        style = Theme.SUCCESS if success else Theme.ERROR
        arg_str = _tool_arg_preview(args)

        line = Text()
        if label:
            line.append(f"  {label} {icon} ", style=style)
        else:
            line.append(f"  {icon} ", style=style)
        line.append(name, style=Theme.TOOL)
        line.append(f" → [dim]{arg_str}[/]")

        preview = _output_preview(output)
        line.append(f"\n    ↳ [dim]{preview}[/]")
        console.print(line)

    # ── 轮次结束 ────────────────────────────────────────
    def show_round_end(self) -> None:
        elapsed = time.time() - self._round_start if self._round_start else 0
        console.print(f"  [dim]继续下一轮 ({elapsed:.1f}s)...[/]")

    # ── 最终回复 ────────────────────────────────────────
    def show_final_answer(self, content: str) -> None:
        console.print()
        console.print(Panel(
            content,
            title="🤖 Final Answer",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print("─" * 56, style=Theme.RULE)
        console.print()

    # ── 退出 ────────────────────────────────────────────
    def show_goodbye(self) -> None:
        console.print("  [dim]👋 Goodbye[/]")


# ═══════════════════════════════════════════════════════════
# SubAgentTracker — 子 agent 执行过程的数据记录
# ═══════════════════════════════════════════════════════════

class SubAgentTracker:
    """记录子 agent 的执行过程。"""

    def __init__(self, description: str):
        self.description = description
        self.rounds: int = 0
        self.tool_count: int = 0
        self.tool_logs: list[dict] = []   # [{round, name, args, output}]
        self._start: float = time.time()
        self.status: str = "running"       # running | done | error

    def add_round(self, tool_calls: list[dict]) -> None:
        """记录一轮工具调用。tool_calls 格式: [{name, args, output}, ...]"""
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


# ═══════════════════════════════════════════════════════════
# SubAgentDisplay — 子 agent 渲染
# ═══════════════════════════════════════════════════════════

class SubAgentDisplay:
    """子 Agent 的终端渲染器（摘要 + 可折叠日志）。"""

    def start(self, description: str) -> SubAgentTracker:
        """开始追踪一个子 agent，打印启动行。"""
        tracker = SubAgentTracker(description)
        console.print(f"  🐳 [magenta]子Agent: {description}[/] [dim]⏳ 启动...[/]")
        return tracker

    def finish(self, tracker: SubAgentTracker, result: Optional[str]) -> None:
        """子 agent 执行完毕，打印摘要 + 工具调用树 + 返回值预览。"""
        elapsed = tracker.elapsed
        ok = result is not None

        # ── 摘要行 ──
        summary = Text()
        summary.append(f"     {'✅' if ok else '⚠️'} 完成", style=Theme.SUCCESS if ok else Theme.ERROR)
        summary.append(f"  ·  {tracker.rounds} 轮", style=Theme.DIM)
        summary.append(f"  ·  {tracker.tool_count} 次工具调用", style=Theme.DIM)
        summary.append(f"  ·  {elapsed:.1f}s", style=Theme.DIM)
        console.print(summary)

        # ── 可折叠工具调用树 ──
        if tracker.tool_logs:
            tree = Tree(f"[dim]📋 详细日志 ({tracker.rounds} 轮)[/]", guide_style=Theme.DIM)
            current_round = 0
            round_node = None
            for log in tracker.tool_logs:
                if log["round"] != current_round:
                    current_round = log["round"]
                    round_node = tree.add(f"[dim]R{current_round}[/]", guide_style=Theme.DIM)
                arg_str = _tool_arg_preview(log["args"])
                if round_node is not None:
                    round_node.add(f"[blue]{log['name']}[/] [dim]({arg_str})[/]")
            console.print(tree)

        # ── 返回值预览 ──
        if result:
            preview = _truncate(result, 300)
            console.print(f"     [dim]📤 返回:[/] {preview}")


# ═══════════════════════════════════════════════════════════
# 全局单例（其他模块 import 这些）
# ═══════════════════════════════════════════════════════════

agent_display = AgentLoopDisplay()
subagent_display = SubAgentDisplay()
