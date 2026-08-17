"""Build the interactive Feishu card JSON for a run's progress + result.

Uses the Feishu card schema 2.0 format — collapsible_panel is a v2 component
and is NOT supported under the legacy template format. One card carries the
whole run: final summary on top (always visible), each agent/stage as a
collapsed panel, and a JSON-2.0-compatible footer at the bottom.

Validate with the Feishu card builder tool if the protocol changes again.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# phase -> (emoji, label) shown in panel titles. Kept here (not imported from
# progress.py) to avoid a circular import: progress.py imports CardBuilder.
STAGE_LABELS: dict[str, tuple[str, str]] = {
    "entry": ("🐱", "接单中"),
    "delegating": ("🤝", "委派中"),
    "classifying": ("🤔", "识别任务"),
    "planning": ("📋", "规划任务"),
    "researching": ("🔍", "调研分析"),
    "coding": ("💻", "编码实现"),
    "reviewing": ("✅", "代码复核"),
    "revising": ("🔧", "修正问题"),
    "summarizing": ("📝", "汇总结果"),
}

_PANEL_SNAPSHOT_LIMIT = 6_000
_CARD_SUMMARY_BYTES = 16_000
_CARD_PROCESS_BYTES = 4_000
_CARD_STAGE_LIMIT = 8


def _truncate_utf8(text: str, max_bytes: int, suffix: str = "") -> str:
    value = text or ""
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    suffix_bytes = suffix.encode("utf-8")
    budget = max(max_bytes - len(suffix_bytes), 0)
    encoded = value.encode("utf-8")[:budget]
    while encoded:
        try:
            return encoded.decode("utf-8") + suffix
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return suffix[:max_bytes]


@dataclass
class StageBlock:
    agent_name: str
    phase: str
    label: str
    buffer: list[str] = field(default_factory=list)
    closed: bool = False

    def append(self, delta: str) -> None:
        if delta:
            self.buffer.append(delta)

    @property
    def text(self) -> str:
        return "".join(self.buffer).strip()


class CardBuilder:
    def __init__(self, run_id: str, *, timeline_limit: int = 30) -> None:
        self.run_id = run_id
        self.timeline_limit = max(int(timeline_limit), 1)
        self.stages: list[StageBlock] = []
        self._current: StageBlock | None = None
        self.summary: str = ""
        self.status: str = "进行中"
        self.entry_agent_name: str = ""  # shown in the card header title
        self.latest_status: str = "正在启动任务…"
        self.elapsed_seconds: int = 0
        self.completed_actions: list[str] = []
        self.artifacts: list[dict] = []

    def set_entry_agent(self, name: str) -> None:
        self.entry_agent_name = name or ""

    def begin_stage(self, agent_name: str, phase: str) -> StageBlock:
        if (
            self._current is not None
            and not self._current.closed
            and self._current.agent_name == (agent_name or "")
            and self._current.phase == (phase or "")
        ):
            return self._current
        if self._current is not None:
            self._current.closed = True
            self._remember_completed(self._current)
        emoji, label = STAGE_LABELS.get(phase, ("•", phase or "进行中"))
        block = StageBlock(agent_name=agent_name or "", phase=phase or "", label=f"{emoji} {label}")
        self.stages.append(block)
        self._current = block
        return block

    def ensure_stage(self, agent_name: str, phase: str) -> StageBlock:
        if (
            self._current is None
            or self._current.agent_name != (agent_name or "")
            or self._current.phase != (phase or "")
        ):
            return self.begin_stage(agent_name, phase)
        return self._current

    def append_text(self, delta: str) -> None:
        if self._current is None:
            self.begin_stage("", "entry")
        self._current.append(delta)
        self._trim_timeline()

    def record_progress(
        self,
        title: str,
        *,
        detail: str = "",
        agent_name: str = "",
        phase: str = "entry",
        timestamp: str = "",
    ) -> None:
        self.ensure_stage(agent_name, phase)
        self.latest_status = title or self.latest_status
        prefix = f"{timestamp}  " if timestamp else ""
        line = f"{prefix}{title}"
        if detail:
            line += f"\n  {detail}"
        self._current.append(("\n" if self._current.text else "") + line)
        self._trim_timeline()

    def _trim_timeline(self) -> None:
        """Keep recent card items; the complete history lives in Run JSONL."""
        total = sum(len(stage.buffer) for stage in self.stages)
        while total > self.timeline_limit:
            for stage in self.stages:
                if stage.buffer:
                    stage.buffer.pop(0)
                    total -= 1
                    break
            else:
                break

    def set_elapsed(self, seconds: int) -> None:
        self.elapsed_seconds = max(int(seconds), 0)

    def complete_current_stage(self) -> None:
        if self._current is not None:
            self._current.closed = True
            self._remember_completed(self._current)

    def _remember_completed(self, stage: StageBlock) -> None:
        action = f"{stage.agent_name or 'Agent'}：{stage.label.replace('中', '')}"
        if action not in self.completed_actions:
            self.completed_actions.append(action)

    def set_summary(self, text: str) -> None:
        self.summary = (text or "").strip()
        self.status = "已完成"

    def set_final_message(self, text: str) -> None:
        """Set only the user-facing terminal answer."""
        self.set_summary(text)

    def set_artifacts(self, artifacts: list[dict] | None) -> None:
        """Attach created documents/files without replacing the final answer."""
        self.artifacts = [item for item in (artifacts or []) if isinstance(item, dict)]

    def set_short_result(self, full_text: str) -> None:
        """Backward-compatible alias for the user-facing final message."""
        self.summary = (full_text or "").strip()
        self.status = "已完成"

    def result_content(self) -> str:
        """Visible result text, including an explicit reply-role label."""
        content = self.summary or "任务已完成。"
        artifact_lines: list[str] = []
        for artifact in self.artifacts:
            title = str(artifact.get("title") or "交付物")
            url = str(artifact.get("url") or "")
            status = str(artifact.get("status") or "created")
            if status == "failed":
                artifact_lines.append(f"- ⚠️ {title}：创建失败")
            elif url:
                artifact_lines.append(f"- 📄 [{title}]({url})")
            else:
                artifact_lines.append(f"- 📄 {title}")
        if artifact_lines:
            content = f"{content}\n\n**交付物：**\n" + "\n".join(artifact_lines)
        if self.entry_agent_name:
            content = f"**回复角色：{self.entry_agent_name}**\n\n{content}"
        return content

    def process_snapshot(self) -> list[dict[str, str]]:
        """Serializable work-log snapshot for Run diagnostics and recovery."""
        return [
            {
                "agent_name": stage.agent_name,
                "phase": stage.phase,
                "label": stage.label,
                "text": stage.text[:_PANEL_SNAPSHOT_LIMIT],
            }
            for stage in self.stages
        ]

    def build(self) -> dict:
        # --- header ---
        status_emoji = "✅" if self.status == "已完成" else "⏳"
        header_title = f"{status_emoji} 任务{self.status}"
        if self.entry_agent_name:
            header_title = f"[{self.entry_agent_name}] {header_title}"

        # --- body elements ---
        elements: list[dict] = []

        completed = ""
        if self.completed_actions:
            completed = "\n\n**做了什么：**\n" + "\n".join(
                f"✓ {item}" for item in self.completed_actions[-5:]
            )

        # Top: live status while running; final answer/artifact when complete.
        if self.summary:
            body = _truncate_utf8(
                self.result_content() + completed,
                _CARD_SUMMARY_BYTES,
                "\n\n…（内容较长，卡片已截断；完整结果保存在 Run 中）",
            )
            elements.append({"tag": "markdown", "content": body})
        else:
            elapsed = f"{self.elapsed_seconds} 秒" if self.elapsed_seconds else "刚刚开始"
            body = f"**最新进展：** {self.latest_status}\n\n**已运行：** {elapsed}{completed}"
            elements.append({"tag": "markdown", "content": body})

        # Middle: one collapsed panel per stage.
        if self.stages:
            elements.append(self._divider())
            visible_stages = self.stages[-_CARD_STAGE_LIMIT:]
            panel_budget = max(_CARD_PROCESS_BYTES // len(visible_stages), 300)
            for stage in visible_stages:
                panel_text = stage.text or "正在等待 Agent 输出；工具动作和心跳会继续更新。"
                panel_text = _truncate_utf8(
                    panel_text,
                    panel_budget,
                    "\n…（已截断，完整见 Run 事件）",
                )
                elements.append({
                    "tag": "collapsible_panel",
                    "expanded": False,
                    "header": {
                        "title": {"tag": "plain_text",
                                  "content": self._panel_title(stage)},
                    },
                    "elements": [
                        {"tag": "markdown", "content": panel_text},
                    ],
                })

        # Bottom: JSON 2.0-compatible footer (legacy `note` is unsupported).
        elements.append(self._divider())
        elements.append({
            "tag": "div",
            "text": {
                "tag": "plain_text",
                "content": self._note_text(),
                "text_size": "notation",
                "text_color": "grey",
            },
        })

        # Interactive-message content is the raw card JSON. JSON 2.0 is selected
        # by the root-level schema field; template envelopes are for template
        # cards and make raw cards fail validation.
        return {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {
                "title": {"tag": "plain_text", "content": header_title},
                "template": "blue" if self.status != "已完成" else "green",
            },
            "body": {
                "elements": elements,
            },
        }

    @staticmethod
    def _panel_title(stage: StageBlock) -> str:
        prefix = f"[{stage.agent_name}] " if stage.agent_name else ""
        return f"{prefix}{stage.label}（工作过程）"

    @staticmethod
    def _divider() -> dict:
        # JSON 2.0 does not support several legacy display components (for
        # example note). A minimal div is accepted by the v2 schema and keeps
        # the card robust across Feishu client versions.
        return {
            "tag": "div",
            "text": {"tag": "plain_text", "content": "────────"},
        }

    def _note_text(self) -> str:
        role = f"回复角色：{self.entry_agent_name} · " if self.entry_agent_name else ""
        return f"{role}Run ID: {self.run_id} · 点击 ▸ 展开查看工作过程"
