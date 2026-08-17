"""Resolve how a task should be delivered without deciding Agent execution.

The safe default is chat. Wiki capabilities are granted only when the user
explicitly asks to read, create, or modify a document.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from config import settings


OutputMode = Literal["chat", "wiki", "both"]


@dataclass(frozen=True)
class OutputIntent:
    mode: OutputMode = "chat"
    wiki_capabilities: tuple[str, ...] = ()
    reason: str = "default_chat"

    @property
    def artifact_requested(self) -> bool:
        return self.mode in ("wiki", "both") and bool(
            {"wiki_write", "wiki_append"} & set(self.wiki_capabilities)
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["wiki_capabilities"] = list(self.wiki_capabilities)
        return data


_DOC_NOUN = r"(?:飞书文档|知识库|wiki|文档|报告|方案)"
_CREATE = r"(?:写成|写一份|生成|创建|整理成|输出(?:一份)?|制作)"
_READ = r"(?:读一下|读取|查看|看看|列出|有哪些|打开)"
_UPDATE = r"(?:追加|加一节|补充|更新|修改|改一下|写入)"


def resolve_output_intent(content: str) -> OutputIntent:
    """Return chat/wiki/both plus the minimum Wiki tool capabilities."""
    if not settings.output_intent_routing:
        return OutputIntent()

    text = " ".join((content or "").strip().split())
    lowered = text.lower()
    if not re.search(_DOC_NOUN, lowered, re.IGNORECASE):
        return OutputIntent()

    if re.search(rf"{_READ}.{{0,20}}{_DOC_NOUN}|{_DOC_NOUN}.{{0,20}}{_READ}", lowered, re.IGNORECASE):
        caps = ("wiki_read", "wiki_list")
        return OutputIntent("chat", caps, "explicit_wiki_read")

    if re.search(rf"{_UPDATE}.{{0,24}}{_DOC_NOUN}|{_DOC_NOUN}.{{0,24}}{_UPDATE}", lowered, re.IGNORECASE):
        caps = ("wiki_read", "wiki_list", "wiki_append")
        return OutputIntent("wiki", caps, "explicit_wiki_update")

    create_requested = bool(
        re.search(rf"{_CREATE}.{{0,32}}{_DOC_NOUN}|{_DOC_NOUN}.{{0,32}}{_CREATE}", lowered, re.IGNORECASE)
        or re.search(r"(?:完整|不少于\s*\d+\s*字).{0,16}(?:分析)?报告", lowered, re.IGNORECASE)
    )
    if create_requested:
        mode: OutputMode = "wiki" if re.search(r"只(?:写|放|更新).{0,12}(?:文档|wiki)", lowered, re.IGNORECASE) else "both"
        return OutputIntent(mode, ("wiki_write",), "explicit_document_create")

    # Mentioning a document alone is not permission to create or modify it.
    return OutputIntent()


def compact_artifact_final_message(
    message: str,
    *,
    artifact_requested: bool,
    artifacts: list[dict],
) -> tuple[str, bool]:
    """Prevent a complete document body from being duplicated into chat."""
    value = (message or "").strip()
    limit = max(int(settings.artifact_final_message_limit), 100)
    if artifact_requested and artifacts and len(value) > limit:
        return "文档已按要求生成或更新，完整内容请查看下方交付物。", True
    return value, False
