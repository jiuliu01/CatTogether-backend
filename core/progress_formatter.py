"""Turn raw agent/tool events into short, safe, user-visible progress lines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FormattedProgress:
    kind: str
    title: str
    detail: str = ""


_PHASE_LABELS = {
    "entry": "开始处理任务",
    "classifying": "正在识别任务",
    "planning": "正在规划任务",
    "researching": "正在调研分析",
    "coding": "正在编码实现",
    "reviewing": "正在复核结果",
    "revising": "正在修正问题",
    "summarizing": "正在整理最终结果",
    "delegating": "正在邀请同事协作",
}


def _clean(value: Any, limit: int = 180) -> str:
    from core.output_safety import sanitize_agent_text

    text = " ".join(str(value or "").replace("`", "'").split())
    text = sanitize_agent_text(text).text
    return text[:limit]


def _tool_name(value: Any) -> str:
    name = _clean(value, 100).lower()
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1]


def format_progress(kind: str, data: dict) -> FormattedProgress | None:
    """Format one callback payload without exposing raw tool results."""
    if kind == "stage":
        phase = _clean(data.get("phase"), 40) or "entry"
        return FormattedProgress("stage", _PHASE_LABELS.get(phase, f"正在进行 {phase}"))

    if kind == "text":
        text = _clean(data.get("delta"), 500)
        return FormattedProgress("action", text) if text else None

    if kind == "tool_call":
        tool = _tool_name(data.get("tool"))
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        path = _clean(args.get("file_path") or args.get("path") or args.get("pattern"), 140)
        if tool == "read":
            return FormattedProgress("action", f"正在读取 {path or '项目文件'}")
        if tool == "glob":
            return FormattedProgress("action", f"正在扫描 {path or '项目目录'}")
        if tool in {"grep", "search"}:
            query = _clean(args.get("pattern") or args.get("query"), 100)
            return FormattedProgress("action", f"正在检索 {query or '相关代码'}")
        if tool in {"edit", "write"}:
            return FormattedProgress("action", f"正在修改 {path or '项目文件'}")
        if tool == "delegate":
            target = _clean(args.get("target"), 60) or "同事"
            task = _clean(args.get("task"), 140)
            return FormattedProgress("delegate", f"正在邀请 {target} 协作", task)
        if tool == "report_progress":
            # The backend callback produces the canonical checkpoint event.
            return None
        if tool.startswith("wiki_"):
            labels = {
                "wiki_write": "正在创建 Wiki 文档",
                "wiki_append": "正在更新 Wiki 文档",
                "wiki_read": "正在读取 Wiki 文档",
                "wiki_list": "正在查看 Wiki 目录",
            }
            return FormattedProgress("artifact", labels.get(tool, "正在处理 Wiki 文档"))
        return FormattedProgress("action", f"正在使用 {tool or '工具'}")

    if kind == "tool_result":
        # Raw results may contain source, credentials, or huge payloads.
        return None

    if kind in {
        "checkpoint", "artifact", "delegate", "delegate_requested",
        "delegate_started", "delegate_completed", "delegate_failed",
        "artifact_created", "artifact_failed", "stalled", "error",
    }:
        title = _clean(data.get("title") or data.get("message"), 220)
        detail = _clean(data.get("detail") or data.get("url"), 300)
        return FormattedProgress(kind, title or "任务状态已更新", detail)

    if kind == "heartbeat":
        elapsed = max(int(data.get("elapsed_seconds") or 0), 0)
        return FormattedProgress("heartbeat", f"仍在处理中，已运行 {elapsed} 秒")

    return None
