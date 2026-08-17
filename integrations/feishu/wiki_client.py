"""Feishu Wiki client: create spaces/nodes and read/write document content.

Used both by the backend (auto-archive long results) and by the agent-facing
wiki MCP tool (user says "整理成文档"/"给文档加一节"/"读一下我们的设计文档").

Token caching is shared with FeishuSender (tenant_access_token). No DB: the
space_id is cached on the Project.
"""
from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote

import httpx

from config import settings
from integrations.feishu.sender import FeishuAPIError, feishu_sender
from core.project_store import project_store

logger = logging.getLogger(__name__)

_WIKI_BASE = "/open-apis/wiki/v2"
_DOCX_BASE = "/open-apis/docx/v1"


class FeishuWikiClient:
    def __init__(self) -> None:
        self._token_lock = None  # reuse feishu_sender's token
        self._wiki_base_url_cache: str | None = None

    async def _token(self) -> str:
        return await feishu_sender._access_token()

    async def _request(self, method: str, path: str, json_body: dict | None = None) -> dict:
        token = await self._token()
        url = f"{settings.feishu_api_base_url}{path}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.request(
                method, url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                json=json_body,
            )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if data.get("code") != 0:
            raise FeishuAPIError(
                data.get("msg")
                or f"Feishu error {data.get('code')} (HTTP {resp.status_code})"
            )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise FeishuAPIError(f"Feishu HTTP {resp.status_code}") from exc
        return data.get("data", {})

    # --- space ---
    async def ensure_space(self, project_id: str, name: str) -> str:
        """Resolve the project's writable Wiki space and cache its space_id.

        Feishu's create-space endpoint does not support tenant_access_token,
        which is the only identity a bot process has. A space must therefore be
        created once by a user and the app added as a member/administrator.
        """
        project = await project_store.get(project_id)
        if project and project.wiki_space_id:
            return project.wiki_space_id

        space_id = settings.feishu_wiki_space_id
        if not space_id:
            # tenant_access_token can list spaces only after the app has been
            # added to them. Prefer an exact project-name match; a sole
            # accessible space is also unambiguous.
            data = await self._request("GET", f"{_WIKI_BASE}/spaces?page_size=50")
            spaces = [
                item for item in data.get("items", [])
                if isinstance(item, dict) and item.get("space_id")
            ]
            matched = next((item for item in spaces if item.get("name") == name), None)
            if matched:
                space_id = str(matched["space_id"])
            elif len(spaces) == 1:
                space_id = str(spaces[0]["space_id"])

        if not space_id:
            raise FeishuAPIError(
                "未找到可写的飞书知识空间。请先创建知识空间、把机器人应用添加为"
                "知识空间管理员，并配置 FEISHU_WIKI_SPACE_ID。"
            )
        await project_store.set_wiki_space(project_id, space_id)
        return space_id

    # --- node ---
    async def create_doc_node(self, space_id: str, title: str, markdown: str) -> tuple[str, str]:
        """Create a Wiki docx node, write content. Return (node_token, url)."""
        # Resolve the browser URL before creating anything, so missing tenant
        # domain configuration cannot leave an orphan node behind.
        await self._wiki_base_url()
        data = await self._request("POST", f"{_WIKI_BASE}/spaces/{space_id}/nodes", {
            "obj_type": "docx",
            "node_type": "origin",
            "title": title,
        })
        node_token = data.get("node", {}).get("node_token") or data.get("node_token")
        obj_token = data.get("node", {}).get("obj_token") or data.get("obj_token")
        if not node_token or not obj_token:
            raise FeishuAPIError("wiki node creation returned no tokens")
        # Write content into the document's root block.
        await self._write_blocks(obj_token, markdown)
        url = f"{await self._wiki_base_url()}/{node_token}"
        return node_token, url

    async def append_content(self, node_token: str, markdown: str) -> None:
        """Append content to an existing Wiki node."""
        obj_token = await self._get_obj_token(node_token)
        doc = await self._request("GET", f"{_DOCX_BASE}/documents/{obj_token}")
        root_block_id = doc.get("document", {}).get("document_id") or obj_token
        await self._write_blocks(obj_token, markdown, parent_block_id=root_block_id)

    async def node_url(self, node_token: str) -> str:
        """Return the browser URL for an existing Wiki node."""
        return f"{await self._wiki_base_url()}/{node_token}"

    async def read_node(self, node_token: str) -> str:
        """Read a Wiki node's content back as markdown."""
        obj_token = await self._get_obj_token(node_token)
        data = await self._request("GET", f"{_DOCX_BASE}/documents/{obj_token}/blocks",
                                    json_body=None)
        blocks = data.get("items", [])
        return blocks_to_markdown(blocks)

    async def list_nodes(self, space_id: str) -> list[dict]:
        """List nodes in a space."""
        data = await self._request("GET", f"{_WIKI_BASE}/spaces/{space_id}/nodes")
        items = data.get("items", [])
        base_url = await self._wiki_base_url()
        return [
            {
                "node_token": it.get("node", {}).get("node_token") or it.get("node_token"),
                "title": it.get("node", {}).get("title") or it.get("title", ""),
                "url": (
                    f"{base_url}/"
                    f"{it.get('node', {}).get('node_token') or it.get('node_token', '')}"
                ),
            }
            for it in items
        ]

    # --- internals ---
    async def _get_obj_token(self, node_token: str) -> str:
        encoded = quote(node_token, safe="")
        data = await self._request(
            "GET", f"{_WIKI_BASE}/spaces/get_node?token={encoded}"
        )
        obj_token = data.get("node", {}).get("obj_token")
        if not obj_token:
            raise FeishuAPIError("get_node returned no obj_token")
        return obj_token

    async def _write_blocks(self, obj_token: str, markdown: str, parent_block_id: str | None = None) -> None:
        if parent_block_id is None:
            parent_block_id = obj_token  # document root
        blocks = markdown_to_blocks(markdown)
        if not blocks:
            return
        # Feishu accepts at most 50 children per request. Long agent reports
        # routinely exceed that, so append in bounded chunks.
        path = (
            f"{_DOCX_BASE}/documents/{obj_token}/blocks/"
            f"{parent_block_id}/children"
        )
        for offset in range(0, len(blocks), 50):
            await self._request("POST", path, {
                "children": blocks[offset:offset + 50],
                "index": -1,
            })
            if offset + 50 < len(blocks):
                # Document edits are limited to 3 requests/second.
                await asyncio.sleep(0.35)

    async def _wiki_base_url(self) -> str:
        if self._wiki_base_url_cache:
            return self._wiki_base_url_cache
        base = (settings.feishu_wiki_base_url or "").strip().rstrip("/")
        if not base:
            try:
                data = await self._request(
                    "GET", "/open-apis/tenant/v2/tenant/query"
                )
                domain = str(data.get("tenant", {}).get("domain") or "").strip()
                if domain:
                    base = f"https://{domain}"
            except Exception:
                logger.debug("failed to resolve Feishu tenant domain", exc_info=True)
        if not base:
            raise FeishuAPIError(
                "无法获取飞书企业域名。请配置 FEISHU_WIKI_BASE_URL（例如 "
                "https://example.feishu.cn/wiki），或为应用开通"
                " tenant:tenant.domain:read 权限。"
            )
        if not base.endswith("/wiki"):
            base = f"{base}/wiki"
        self._wiki_base_url_cache = base
        return base


# --- markdown <-> blocks conversion ---

def markdown_to_blocks(markdown: str) -> list[dict]:
    """Convert markdown text to Feishu docx block children.

    Supports: headings (##/###), fenced code blocks, bullet lists, paragraphs.
    Unsupported constructs fall back to plain text paragraphs.
    """
    blocks: list[dict] = []
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append({
                "block_type": 14,  # code block
                "code": {
                    "style": {"language": 1},
                    "elements": [{"text_run": {"content": "\n".join(code_lines)}}],
                },
            })
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            blocks.append({
                "block_type": {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}[level],
                "heading" + str(level): {
                    "elements": [{"text_run": {"content": m.group(2)}}],
                },
            })
            i += 1
            continue

        # bullet list
        m = re.match(r"^\s*[-*]\s+(.*)", line)
        if m:
            blocks.append({
                "block_type": 12,  # bullet
                "bullet": {
                    "elements": [{"text_run": {"content": m.group(1)}}],
                },
            })
            i += 1
            continue

        # paragraph (skip empty lines)
        if line.strip():
            blocks.append({
                "block_type": 2,  # text/paragraph
                "text": {
                    "elements": [{"text_run": {"content": line}}],
                },
            })
        i += 1
    return blocks


def blocks_to_markdown(blocks: list[dict]) -> str:
    """Convert Feishu docx blocks back to a rough markdown string."""
    parts: list[str] = []
    for b in blocks:
        bt = b.get("block_type", 0)
        if bt == 2:  # text
            text = _extract_text(b.get("text", {}))
            if text:
                parts.append(text)
        elif 3 <= bt <= 8:  # heading 1-6
            level = bt - 2
            key = f"heading{level}"
            text = _extract_text(b.get(key, {}))
            parts.append(f"{'#' * level} {text}")
        elif bt == 12:  # bullet
            text = _extract_text(b.get("bullet", {}))
            parts.append(f"- {text}")
        elif bt == 14:  # code
            text = _extract_text(b.get("code", {}))
            parts.append(f"```\n{text}\n```")
    return "\n".join(parts)


def _extract_text(container: dict) -> str:
    elements = container.get("elements", [])
    return "".join(
        e.get("text_run", {}).get("content", "")
        for e in elements if isinstance(e, dict)
    )


feishu_wiki_client = FeishuWikiClient()
