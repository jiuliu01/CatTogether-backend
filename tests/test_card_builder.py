import json

from integrations.feishu.card_builder import CardBuilder, StageBlock
from integrations.feishu.sender import FeishuSender


def _body_elements(card: dict) -> list[dict]:
    return card["body"]["elements"]


def test_card_uses_schema_2():
    """Interactive-message content is raw card JSON with root schema 2.0."""
    card = CardBuilder("run1").build()
    assert card["schema"] == "2.0"
    assert "body" in card
    assert "header" in card
    assert "type" not in card
    tags = [element["tag"] for element in card["body"]["elements"]]
    assert "note" not in tags
    assert "hr" not in tags


def test_large_chinese_card_stays_below_feishu_payload_limit():
    cb = CardBuilder("run-size")
    cb.set_final_message("最终答案" * 2000)
    for index in range(10):
        cb.begin_stage(f"角色{index}", "researching")
        cb.append_text("阶段过程" * 1000)

    card = cb.build()
    FeishuSender._validate_card_payload(card)
    assert len(json.dumps(card, ensure_ascii=False).encode("utf-8")) <= 28_000


def test_header_carries_entry_agent_name():
    cb = CardBuilder("run2")
    cb.set_entry_agent("橘长")
    cb.set_summary("done")
    card = cb.build()
    title = card["header"]["title"]["content"]
    assert "橘长" in title
    assert "已完成" in title
    top = _body_elements(card)[0]["content"]
    assert "回复角色：橘长" in top


def test_empty_card_has_summary_and_note():
    card = CardBuilder("run1").build()
    els = _body_elements(card)
    tags = [e["tag"] for e in els]
    assert tags[0] == "markdown"  # "正在处理…"
    assert "collapsible_panel" not in tags
    assert tags[-1] == "div"
    assert "run1" in els[-1]["text"]["content"]


def test_summary_on_top_always_visible():
    cb = CardBuilder("run2")
    cb.set_entry_agent("橘长")
    cb.set_summary("backend 目录结构如下：\n```\nmain.py\n```")
    card = cb.build()
    top = _body_elements(card)[0]
    assert top["tag"] == "markdown"
    assert "backend 目录结构" in top["content"]


def test_each_stage_is_a_collapsed_panel():
    cb = CardBuilder("run3")
    cb.set_entry_agent("橘长")
    cb.begin_stage("橘长", "entry")
    cb.append_text("我来直接查看 backend 目录结构。")
    cb.begin_stage("暹罗", "coding")
    cb.append_text("已修改 api/rest.py。")
    cb.set_summary("已完成 /hello 接口。")
    card = cb.build()
    els = _body_elements(card)

    panels = [e for e in els if e["tag"] == "collapsible_panel"]
    assert len(panels) == 2
    assert all(p["expanded"] is False for p in panels)
    titles = [p["header"]["title"]["content"] for p in panels]
    assert any("橘长" in t and "接单中" in t for t in titles)
    assert any("暹罗" in t and "编码实现" in t for t in titles)
    assert "backend 目录结构" in panels[0]["elements"][0]["content"]
    assert "api/rest.py" in panels[1]["elements"][0]["content"]


def test_text_before_any_stage_opens_default_panel():
    cb = CardBuilder("run4")
    cb.append_text("早早出现的输出")
    card = cb.build()
    panels = [e for e in _body_elements(card) if e["tag"] == "collapsible_panel"]
    assert len(panels) == 1
    assert "早早出现的输出" in panels[0]["elements"][0]["content"]


def test_long_panel_text_is_truncated():
    cb = CardBuilder("run5")
    cb.begin_stage("橘长", "entry")
    cb.append_text("x" * 30000)
    card = cb.build()
    panel = [e for e in _body_elements(card) if e["tag"] == "collapsible_panel"][0]
    content = panel["elements"][0]["content"]
    assert "已截断" in content
    assert len(content) < 30000


def test_note_always_last_with_run_id():
    cb = CardBuilder("run6")
    cb.set_entry_agent("橘长")
    cb.begin_stage("橘长", "entry")
    cb.set_summary("done")
    card = cb.build()
    els = _body_elements(card)
    assert els[-1]["tag"] == "div"
    assert "run6" in els[-1]["text"]["content"]
    assert "回复角色：橘长" in els[-1]["text"]["content"]


def test_process_snapshot_is_serializable():
    cb = CardBuilder("run-process")
    cb.begin_stage("橘长", "entry")
    cb.append_text("先检查代码。")

    assert cb.process_snapshot() == [{
        "agent_name": "橘长",
        "phase": "entry",
        "label": "🐱 接单中",
        "text": "先检查代码。",
    }]


def test_card_timeline_keeps_recent_items_only():
    cb = CardBuilder("run-limit", timeline_limit=2)
    cb.begin_stage("橘长", "entry")
    cb.record_progress("第一条", agent_name="橘长")
    cb.record_progress("第二条", agent_name="橘长")
    cb.record_progress("第三条", agent_name="橘长")

    text = cb.process_snapshot()[0]["text"]
    assert "第一条" not in text
    assert "第二条" in text and "第三条" in text


def test_short_result_full_text_inline():
    cb = CardBuilder("run8")
    cb.set_entry_agent("橘长")
    cb.set_short_result("短结果，直接显示。")
    card = cb.build()
    top = _body_elements(card)[0]
    assert "短结果，直接显示。" in top["content"]
    assert "完整文档" not in top["content"]


def test_final_message_and_artifact_are_shown_separately():
    cb = CardBuilder("run-artifact")
    cb.set_entry_agent("橘长")
    cb.set_final_message("核心结论在聊天里。")
    cb.set_artifacts([{
        "artifact_id": "node1",
        "type": "feishu_wiki",
        "title": "完整分析报告",
        "url": "https://tenant.feishu.cn/wiki/node1",
        "status": "created",
    }])

    top = _body_elements(cb.build())[0]["content"]
    assert "核心结论在聊天里" in top
    assert "交付物" in top
    assert "完整分析报告" in top
    assert "https://tenant.feishu.cn/wiki/node1" in top
