"""
钩子系统 — 事件对象 + 装饰器注册。

每个事件类型对应一个 dataclass，回调签名固定为 (event) -> str | None：
  - 返回 None  → 继续后续 hook / 放行
  - 返回字符串 → 停止事件链（字符串即原因，用于权限拒绝等场景）
"""
import traceback
from dataclasses import dataclass
from typing import Callable, Any

from rules import check_deny_list, check_rules, ask_user
from config import WORKDIR
from ui import console

# ═══════════════════════════════════════════════════════════
# 事件类型
# ═══════════════════════════════════════════════════════════

@dataclass
class UserPromptEvent:
    """用户提交了新的对话消息"""
    query: str

@dataclass
class ToolUseEvent:
    """模型请求调用一个工具（PreToolUse 阶段）"""
    tool_name: str
    tool_args: dict
    tool_call_id: str

@dataclass
class ToolResultEvent:
    """工具执行完成（PostToolUse 阶段）"""
    tool_name: str
    tool_args: dict
    tool_call_id: str
    output: str

@dataclass
class StopEvent:
    """Agent 循环即将结束"""
    messages: list


# ═══════════════════════════════════════════════════════════
# Hook 管理器
# ═══════════════════════════════════════════════════════════

class HookManager:
    """事件驱动的钩子管理器。

    用法:
        hooks = HookManager()

        @hooks.on("PreToolUse")
        def check(e: ToolUseEvent) -> str | None:
            if dangerous(e): return "blocked"
            return None
    """

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {}

    # ── 注册 ──────────────────────────────────────────

    def on(self, event: str):
        """装饰器：将函数注册为某个事件的回调。

        @hooks.on("PreToolUse")
        def my_hook(e: ToolUseEvent) -> str | None:
            ...
        """
        def decorator(fn: Callable):
            self._hooks.setdefault(event, []).append(fn)
            return fn
        return decorator

    # ── 触发 ──────────────────────────────────────────

    def trigger(self, event: str, event_obj) -> str | None:
        """触发一个事件，依次调用所有注册的回调。

        - 每个回调接收事件对象
        - 回调抛出异常 → 打印错误、跳过此回调、继续链
        - 回调返回非 None → 立即停止链并将该值返回给调用者
        - 全部返回 None → 返回 None
        """
        for cb in self._hooks.get(event, []):
            try:
                result = cb(event_obj)
            except TypeError as e:
                print(f"\n❌ \033[91m[HOOK ERROR] {event} → {cb.__name__}() 参数错误\033[0m")
                print(f"   \033[90m{type(e).__name__}: {e}\033[0m")
                continue
            except Exception as e:
                print(f"\n❌ \033[91m[HOOK ERROR] {event} → {cb.__name__}() 执行异常\033[0m")
                print(f"   \033[90m{type(e).__name__}: {e}\033[0m")
                for line in traceback.format_exc().strip().split('\n')[-3:]:
                    print(f"   \033[90m{line}\033[0m")
                continue

            if result is not None:
                return result
        return None


# ═══════════════════════════════════════════════════════════
# 全局实例（其他模块 import 这个单例）
# ═══════════════════════════════════════════════════════════

hooks = HookManager()


# ═══════════════════════════════════════════════════════════
# 内置 Hook 注册
# ═══════════════════════════════════════════════════════════

# ── UserPromptSubmit ──────────────────────────────────

@hooks.on("UserPromptSubmit")
def context_inject(e: UserPromptEvent) -> str | None:
    """每次用户输入时打印当前工作目录"""
    console.print(f"  [dim][HOOK] UserPromptSubmit: working in {WORKDIR}[/]")
    return None


# ── PreToolUse ────────────────────────────────────────

@hooks.on("PreToolUse")
def permission_check(e: ToolUseEvent) -> str | None:
    """权限检查：返回 None=放行，返回字符串=拒绝原因"""
    # 闸门 1：硬拒绝列表
    if e.tool_name == "bash":
        reason = check_deny_list(e.tool_args.get("command", ""))
        if reason:
            console.print(f"  [red]⛔ {reason}[/]")
            return reason

    # 闸门 2 + 3：规则匹配 → 用户审批
    reason = check_rules(e.tool_name, e.tool_args)
    if reason:
        decision = ask_user(e.tool_name, e.tool_args, reason)
        if decision == "deny":
            return f"User denied: {reason}"

    return None


# ── PostToolUse ───────────────────────────────────────

@hooks.on("PostToolUse")
def large_output_warning(e: ToolResultEvent) -> str | None:
    """大输出提醒"""
    if len(str(e.output)) > 100000:
        console.print(f"  [yellow][HOOK] ⚠ {e.tool_name} 返回了 {len(str(e.output))} 字符的大输出[/]")
    return None


# ── Stop ─────────────────────────────────────────────

@hooks.on("Stop")
def session_summary(e: StopEvent) -> str | None:
    """会话结束时打印统计"""
    tool_count = sum(
        1 for m in e.messages
        if m.get("role") == "tool"
    )
    console.print(f"  [dim][HOOK] Stop: session used {tool_count} tool calls[/]")
    return None
