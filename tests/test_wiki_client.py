import asyncio

from integrations.feishu.wiki_client import markdown_to_blocks, blocks_to_markdown


def test_markdown_heading_to_block():
    blocks = markdown_to_blocks("## 标题\n正文")
    # heading2 (block_type 4) + paragraph (block_type 2)
    assert blocks[0]["block_type"] == 4
    assert blocks[1]["block_type"] == 2
    assert blocks[0]["heading2"]["elements"][0]["text_run"]["content"] == "标题"


def test_markdown_code_block():
    blocks = markdown_to_blocks("```\nprint('hi')\n```")
    assert blocks[0]["block_type"] == 14
    assert "print('hi')" in blocks[0]["code"]["elements"][0]["text_run"]["content"]


def test_markdown_bullet_list():
    blocks = markdown_to_blocks("- 项一\n- 项二")
    assert all(b["block_type"] == 12 for b in blocks)
    assert blocks[0]["bullet"]["elements"][0]["text_run"]["content"] == "项一"


def test_blocks_to_markdown_roundtrip():
    md = "## 标题\n\n正文段\n\n- 列表项"
    blocks = markdown_to_blocks(md)
    out = blocks_to_markdown(blocks)
    assert "## 标题" in out
    assert "正文段" in out
    assert "- 列表项" in out


def test_empty_markdown():
    assert markdown_to_blocks("") == []


def test_ensure_space_uses_configured_space_and_caches_on_project(monkeypatch):
    """A bot cannot create spaces; it uses the configured existing space."""
    from integrations.feishu import wiki_client as wc
    from core.project_store import project_store
    from models.project import Project

    state = {"cached": None, "requests": 0}

    async def fake_request(self, method, path, json_body=None):
        state["requests"] += 1
        raise AssertionError("configured space should not call the create/list API")

    async def fake_get(project_id):
        return Project(
            id=project_id, name="p", workspace_dir="/tmp",
            wiki_space_id=state["cached"],
        )

    async def fake_set_wiki(project_id, space_id):
        state["cached"] = space_id

    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)
    monkeypatch.setattr(project_store, "get", fake_get)
    monkeypatch.setattr(project_store, "set_wiki_space", fake_set_wiki)
    monkeypatch.setattr(wc.settings, "feishu_wiki_space_id", "space_test_123")

    async def run():
        client = wc.FeishuWikiClient()
        sid1 = await client.ensure_space("proj1", "测试项目")
        # second call should hit cache (no new POST)
        sid2 = await client.ensure_space("proj1", "测试项目")
        return sid1, sid2

    sid1, sid2 = asyncio.run(run())
    assert sid1 == "space_test_123"
    assert sid2 == "space_test_123"
    assert state["requests"] == 0


def test_ensure_space_auto_selects_single_accessible_space(monkeypatch):
    from integrations.feishu import wiki_client as wc
    from core.project_store import project_store
    from models.project import Project

    cached = {"space_id": None}

    async def fake_get(project_id):
        return Project(
            id=project_id,
            name="p",
            workspace_dir="/tmp",
            wiki_space_id=cached["space_id"],
        )

    async def fake_set_wiki(project_id, space_id):
        cached["space_id"] = space_id

    async def fake_request(self, method, path, json_body=None):
        assert method == "GET"
        return {"items": [{"space_id": "space_only", "name": "知识库"}]}

    monkeypatch.setattr(wc.settings, "feishu_wiki_space_id", None)
    monkeypatch.setattr(project_store, "get", fake_get)
    monkeypatch.setattr(project_store, "set_wiki_space", fake_set_wiki)
    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)

    sid = asyncio.run(wc.FeishuWikiClient().ensure_space("proj1", "项目名"))

    assert sid == "space_only"
    assert cached["space_id"] == "space_only"


def test_create_doc_node_returns_url(monkeypatch):
    from integrations.feishu import wiki_client as wc

    async def fake_request(self, method, path, json_body=None):
        if path.endswith("/nodes") and method == "POST":
            return {"node": {"node_token": "nt_abc", "obj_token": "ot_abc"}}
        # write_blocks call
        return {}

    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)
    monkeypatch.setattr(
        wc.settings, "feishu_wiki_base_url", "https://example.feishu.cn/wiki"
    )

    async def run():
        client = wc.FeishuWikiClient()
        node_token, url = await client.create_doc_node("space1", "标题", "## 内容")
        return node_token, url

    nt, url = asyncio.run(run())
    assert nt == "nt_abc"
    assert url == "https://example.feishu.cn/wiki/nt_abc"


def test_wiki_base_url_can_come_from_tenant_domain(monkeypatch):
    from integrations.feishu import wiki_client as wc

    async def fake_request(self, method, path, json_body=None):
        assert path == "/open-apis/tenant/v2/tenant/query"
        return {"tenant": {"domain": "tenant.feishu.cn"}}

    monkeypatch.setattr(wc.settings, "feishu_wiki_base_url", None)
    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)

    base = asyncio.run(wc.FeishuWikiClient()._wiki_base_url())

    assert base == "https://tenant.feishu.cn/wiki"


def test_get_obj_token_makes_one_request_with_query(monkeypatch):
    from integrations.feishu import wiki_client as wc

    calls = []

    async def fake_request(self, method, path, json_body=None):
        calls.append((method, path))
        return {"node": {"obj_token": "doc_abc"}}

    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)

    token = asyncio.run(wc.FeishuWikiClient()._get_obj_token("wiki token"))

    assert token == "doc_abc"
    assert calls == [
        ("GET", "/open-apis/wiki/v2/spaces/get_node?token=wiki%20token")
    ]


def test_write_blocks_chunks_at_feishu_limit(monkeypatch):
    from integrations.feishu import wiki_client as wc

    requests = []

    async def fake_request(self, method, path, json_body=None):
        requests.append(json_body)
        return {}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(wc.FeishuWikiClient, "_request", fake_request)
    monkeypatch.setattr(wc.asyncio, "sleep", no_sleep)
    markdown = "\n".join(f"第 {index} 行" for index in range(51))

    asyncio.run(wc.FeishuWikiClient()._write_blocks("doc", markdown))

    assert [len(body["children"]) for body in requests] == [50, 1]
    assert all(body["index"] == -1 for body in requests)
