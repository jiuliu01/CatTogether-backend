"""Parse Feishu admin commands for agent lifecycle operations.

Recognized commands (admin-only; enforcement is in the event handler):
  生成 agent <id> (role=<role>) (name=<name>) (from=<template>)
  制定 <agent_id>：<key>=<value> ...
  消除 <agent_id>
  把 <src_chat> 的 <agent_id> 移植到 <dst_chat> (mode=copy|move)
  列出 agent

Returns a parsed AdminCommand or None when the message is not an admin command.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AdminCommand:
    op: str  # generate | specify | eliminate | transplant | list
    agent_id: str | None = None
    name: str | None = None
    role: str | None = None
    from_template: str | None = None
    fields: dict | None = None
    src_chat: str | None = None
    dst_chat: str | None = None
    mode: str = "copy"


_KV_RE = re.compile(r"(\w+)\s*[=：:]\s*(\S+)")


def parse_admin_command(content: str) -> AdminCommand | None:
    text = content.strip()

    # 生成 agent <id> (role=...) (name=...) (from=...)
    m = re.match(r"生成\s+agent\s+(\S+)(.*)", text)
    if m:
        agent_id = m.group(1)
        rest = m.group(2)
        kvs = dict(_KV_RE.findall(rest))
        return AdminCommand(
            op="generate",
            agent_id=agent_id,
            name=kvs.get("name"),
            role=kvs.get("role"),
            from_template=kvs.get("from"),
        )

    # 消除 <agent_id>
    m = re.match(r"消除\s+(\S+)", text)
    if m:
        return AdminCommand(op="eliminate", agent_id=m.group(1))

    # 制定 <agent_id>：<key>=<value> ...
    m = re.match(r"制定\s+(\S+)\s*[：:](.*)", text)
    if m:
        kvs = dict(_KV_RE.findall(m.group(2)))
        return AdminCommand(op="specify", agent_id=m.group(1), fields=kvs)

    # 把 <src> 的 <agent> 移植到 <dst> (mode=copy|move)
    m = re.match(r"把\s+(\S+)\s+的\s+(\S+)\s+移植到\s+(\S+)(.*)", text)
    if m:
        kvs = dict(_KV_RE.findall(m.group(4)))
        return AdminCommand(
            op="transplant",
            src_chat=m.group(1),
            agent_id=m.group(2),
            dst_chat=m.group(3),
            mode=kvs.get("mode", "copy"),
        )

    # 列出 agent
    if re.match(r"列出\s+agent", text):
        return AdminCommand(op="list")

    return None
