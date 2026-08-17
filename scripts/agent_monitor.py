#!/usr/bin/env python3
"""Agent run observability panel — a standalone, read-only local web viewer.

It reads the durable artifacts the backend already writes (no backend import,
no FastAPI, no Feishu connection) and serves a single-page dashboard:

  - run list (left): status / task / entry agent / time
  - event timeline (middle): RunProgressEvent stream + structured tool calls
  - memory + conversation (right): memory writes correlated by run_id, plus
    the session_messages thread for the selected run

Usage::

    python backend/scripts/agent_monitor.py [--port 8088] [--data-dir backend/data]

The backend is unaware of this script; start/stop it freely. A background
poller refreshes in-memory state every 1.5s by watching file mtimes.

When the backend runs with ``CT_AGENT_TOOL_TRACE=1``, per-run structured tool
calls are also read from ``data/feishu/run_tools/<run_id>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Data layer — pure stdlib, read-only
# ---------------------------------------------------------------------------


class DataStore:
    """Reads runs / events / tools / memory / sessions from disk + SQLite."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.runs_dir = data_dir / "feishu" / "runs"
        self.events_dir = data_dir / "feishu" / "run_events"
        self.tools_dir = data_dir / "feishu" / "run_tools"
        self.db_path = data_dir / "memory.db"
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}          # run_id -> summary
        self._events: dict[str, list[dict]] = {}  # run_id -> events
        self._tools: dict[str, list[dict]] = {}   # run_id -> tool events
        self._memory: list[dict] = []             # recent memory writes
        self._facts: dict[str, dict] = {}         # fact_id -> fact row
        self._mtimes: dict[str, float] = {}

    # --- helpers ---
    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        out: list[dict] = []
        if not path.exists():
            return out
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        return out

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _changed(self, key: str, path: Path) -> bool:
        m = self._mtime(path)
        if m != self._mtimes.get(key):
            self._mtimes[key] = m
            return True
        return False

    # --- runs ---
    def _load_run(self, path: Path) -> dict | None:
        raw = self._read_json(path)
        if not isinstance(raw, dict) or not raw.get("id"):
            return None
        outputs = raw.get("outputs") or {}
        final = (
            outputs.get("final_message")
            or outputs.get("summary")
            or (outputs.get("run_result") or {}).get("final_message")
            or ""
        )
        delegations = raw.get("delegations") or []
        return {
            "id": raw["id"],
            "status": raw.get("status", ""),
            "original_task": (raw.get("original_task") or "")[:120],
            "entry_agent_id": raw.get("entry_agent_id", ""),
            "entry_agent_name": (outputs.get("reply_agent_name") or ""),
            "workspace_dir": raw.get("workspace_dir", ""),
            "project_id": raw.get("project_id", ""),
            "thread_id": raw.get("thread_id", ""),
            "channel_id": raw.get("channel_id", ""),
            "created_at": raw.get("created_at", ""),
            "updated_at": raw.get("updated_at", ""),
            "error": raw.get("error", ""),
            "final_message": (final or "")[:2000],
            "delegation_count": len(delegations),
            "delegations": [
                {
                    "id": d.get("id", ""),
                    "target_agent_id": d.get("target_agent_id", ""),
                    "status": d.get("status", ""),
                    "access": d.get("access", ""),
                    "task": (d.get("task") or "")[:120],
                }
                for d in delegations
            ],
        }

    def _refresh_runs(self) -> None:
        if not self.runs_dir.exists():
            return
        for path in self.runs_dir.glob("*.json"):
            if not self._changed(f"run:{path.name}", path):
                continue
            run = self._load_run(path)
            if run:
                self._runs[run["id"]] = run

    # --- events / tools (per run, loaded on demand) ---
    def _load_events(self, run_id: str) -> list[dict]:
        path = self.events_dir / f"{run_id}.jsonl"
        return self._read_jsonl(path)

    def _load_tools(self, run_id: str) -> list[dict]:
        path = self.tools_dir / f"{run_id}.jsonl"
        return self._read_jsonl(path)

    # --- memory (SQLite) ---
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _refresh_memory(self) -> None:
        if not self._changed("memory.db", self.db_path):
            return
        try:
            conn = self._connect()
        except sqlite3.Error:
            return
        try:
            rows = conn.execute(
                "SELECT event_id, event_type, actor, payload, status, "
                "fact_ids, created_at, error FROM events "
                "ORDER BY created_at DESC LIMIT 300"
            ).fetchall()
            self._memory = [dict(r) for r in rows]
            # Eagerly load facts referenced by recent events so the panel can
            # show what was actually written without a second round-trip.
            fact_ids: set[str] = set()
            for r in self._memory:
                try:
                    for fid in json.loads(r.get("fact_ids") or "[]"):
                        if isinstance(fid, str):
                            fact_ids.add(fid)
                except ValueError:
                    pass
            self._facts = {}
            if fact_ids:
                placeholders = ",".join("?" for _ in fact_ids)
                frows = conn.execute(
                    f"SELECT id, domain, kind, scope_id, text, status, "
                    f"created_at FROM facts WHERE id IN ({placeholders})",
                    tuple(fact_ids),
                ).fetchall()
                for fr in frows:
                    self._facts[fr["id"]] = dict(fr)
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    def _session_messages(self, thread_id: str) -> list[dict]:
        if not thread_id or not self.db_path.exists():
            return []
        try:
            conn = self._connect()
        except sqlite3.Error:
            return []
        try:
            rows = conn.execute(
                "SELECT role, agent_id, content, created_at FROM session_messages "
                "WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    # --- public API ---
    def poll(self) -> None:
        """Refresh changed slices. Called by the poller thread."""
        with self._lock:
            self._refresh_runs()
            self._refresh_memory()

    def run_summaries(self) -> list[dict]:
        with self._lock:
            runs = list(self._runs.values())
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return runs

    def run_detail(self, run_id: str) -> dict | None:
        with self._lock:
            run = self._runs.get(run_id)
        if not run:
            return None
        events = self._load_events(run_id)
        tools = self._load_tools(run_id)
        # Correlate memory writes for this run.
        run_memory: list[dict] = []
        with self._lock:
            for r in self._memory:
                payload = _safe_json(r.get("payload"))
                if (payload or {}).get("run_id") == run_id:
                    run_memory.append(self._decorate_memory(r))
        messages = self._session_messages(run.get("thread_id", ""))
        return {
            "run": run,
            "events": events,
            "tools": tools,
            "memory": run_memory,
            "messages": messages,
        }

    def _decorate_memory(self, row: dict) -> dict:
        payload = _safe_json(row.get("payload")) or {}
        actor = _safe_json(row.get("actor")) or {}
        try:
            fact_ids = json.loads(row.get("fact_ids") or "[]")
        except ValueError:
            fact_ids = []
        facts = [self._facts[fid] for fid in fact_ids if fid in self._facts]
        return {
            "event_id": row.get("event_id", ""),
            "event_type": row.get("event_type", ""),
            "status": row.get("status", ""),
            "error": row.get("error", ""),
            "created_at": row.get("created_at", ""),
            "why": {
                "run_id": payload.get("run_id"),
                "agent_id": payload.get("agent_id"),
                "agent_role": payload.get("agent_role"),
                "source_type": payload.get("source_type"),
                "succeeded": payload.get("succeeded"),
                "mutation_paths": payload.get("mutation_paths") or [],
                "allowed_domains": payload.get("allowed_domains") or [],
            },
            "attempted": {
                "domain": payload.get("domain"),
                "kind": payload.get("kind"),
                "text": (payload.get("text") or "")[:500],
            },
            "actor": actor,
            "facts": [
                {
                    "id": f.get("id", ""),
                    "domain": f.get("domain", ""),
                    "kind": f.get("kind", ""),
                    "scope_id": f.get("scope_id", ""),
                    "text": (f.get("text") or "")[:500],
                    "status": f.get("status", ""),
                }
                for f in facts
            ],
        }


def _safe_json(text) -> dict | None:
    if not text:
        return None
    if isinstance(text, dict):
        return text
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _json_response(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


PANEL_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CatTogether Agent Monitor</title>
<style>
  :root { color-scheme: light dark; }
  body { margin:0; font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
         font-size:13px; background:#0f1115; color:#d8dee9; }
  header { padding:8px 14px; background:#1a1d24; border-bottom:1px solid #2a2f3a;
           display:flex; gap:12px; align-items:center; }
  header h1 { font-size:14px; margin:0; font-weight:600; }
  header .meta { color:#7a8290; font-size:12px; }
  header .filters { margin-left:auto; display:flex; gap:8px; }
  header input, header select { background:#0f1115; color:#d8dee9; border:1px solid #2a2f3a;
                                 border-radius:4px; padding:3px 6px; font-size:12px; }
  .layout { display:grid; grid-template-columns:300px 1fr 1fr; height:calc(100vh - 42px); }
  .col { overflow-y:auto; border-right:1px solid #2a2f3a; }
  .col:last-child { border-right:none; }
  .col h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em;
            color:#7a8290; margin:0; padding:10px 12px 6px; position:sticky; top:0;
            background:#0f1115; }
  .run-item { padding:8px 12px; border-bottom:1px solid #1e222b; cursor:pointer; }
  .run-item:hover { background:#161a22; }
  .run-item.active { background:#1e2a3a; border-left:3px solid #5e81ac; }
  .run-item .task { color:#e5e9f0; margin-bottom:3px; }
  .run-item .sub { color:#7a8290; font-size:11px; display:flex; gap:8px; }
  .ws-group { border-bottom:1px solid #2a2f3a; }
  .ws-head { padding:8px 12px; background:#161a22; cursor:pointer;
             display:flex; align-items:center; gap:6px; user-select:none;
             position:sticky; top:0; z-index:2; }
  .ws-head:hover { background:#1c2230; }
  .ws-head .arrow { color:#7a8290; font-size:10px; width:10px; }
  .ws-head .ws-name { color:#88c0d0; font-weight:600; font-size:12px;
                      overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
                      flex:1; }
  .ws-head .ws-count { color:#7a8290; font-size:11px; }
  .ws-body { }
  .ws-body.collapsed { display:none; }
  .badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px;
           font-weight:600; text-transform:uppercase; }
  .b-completed { background:#2e5e3a; color:#a3d9a3; }
  .b-failed { background:#5e2e2e; color:#d9a3a3; }
  .b-running, .b-classifying, .b-planning, .b-coding, .b-researching,
  .b-waiting_children, .b-recovering { background:#3a3a2e; color:#d9c9a3; }
  .b-queued { background:#2a2f3a; color:#9aa3b0; }
  .ev { padding:6px 12px; border-bottom:1px solid #1e222b; }
  .ev .head { display:flex; gap:8px; align-items:baseline; }
  .ev .ts { color:#7a8290; font-size:11px; font-variant-numeric:tabular-nums; }
  .ev .who { color:#88c0d0; font-weight:600; }
  .ev .phase { color:#b48ead; font-size:11px; }
  .ev .kind { color:#ebcb8b; font-size:11px; }
  .ev .title { color:#e5e9f0; }
  .ev .detail { color:#9aa3b0; margin-top:3px; padding-left:12px;
                border-left:2px solid #2a2f3a; white-space:pre-wrap; }
  .ev.tool { background:#161a22; }
  .ev.tool .tool { color:#bf88c0; font-weight:600; }
  .ev.tool .args { color:#9aa3b0; font-size:11px; }
  .ev.tool .result { color:#8fa1b3; margin-top:3px; padding-left:12px;
                     border-left:2px solid #2a2f3a; white-space:pre-wrap;
                     max-height:120px; overflow-y:auto; }
  .mem-card { margin:8px 12px; padding:8px; background:#161a22;
              border-radius:5px; border:1px solid #2a2f3a; }
  .mem-card .top { display:flex; gap:8px; align-items:center; margin-bottom:5px; }
  .mem-card .etype { color:#88c0d0; font-weight:600; font-size:12px; }
  .mem-card .status { font-size:10px; padding:1px 5px; border-radius:3px; }
  .mem-card .why { font-size:11px; color:#9aa3b0; margin-bottom:5px; }
  .mem-card .why span { color:#d08770; }
  .mem-card .fact { padding:4px 6px; margin-top:4px; background:#0f1115;
                    border-radius:3px; font-size:12px; }
  .mem-card .fact .k { color:#b48ead; font-size:10px; }
  .mem-card .fact .t { color:#e5e9f0; }
  .msg { padding:6px 12px; }
  .msg.user .bubble { background:#1e3a5e; color:#e5e9f0; }
  .msg.agent .bubble { background:#2a2f3a; color:#e5e9f0; }
  .bubble { display:inline-block; max-width:90%; padding:6px 10px;
            border-radius:8px; white-space:pre-wrap; word-break:break-word; }
  .msg .who { font-size:10px; color:#7a8290; margin-bottom:2px; }
  .empty { color:#7a8290; padding:20px; text-align:center; }
  a { color:#88c0d0; }
</style>
</head>
<body>
<header>
  <h1>🐱 CatTogether Agent Monitor</h1>
  <span class="meta" id="meta">loading…</span>
  <div class="filters">
    <input id="f-agent" placeholder="agent 过滤" />
    <select id="f-status">
      <option value="">所有状态</option>
      <option>completed</option><option>failed</option><option>running</option>
      <option>queued</option><option>waiting_children</option>
    </select>
    <input id="f-task" placeholder="任务关键词" />
  </div>
</header>
<div class="layout">
  <div class="col" id="runs"></div>
  <div class="col" id="timeline"><h2>事件时间线</h2><div class="empty">选择左侧的 run</div></div>
  <div class="col" id="right"><h2>记忆写入 / 对话</h2><div class="empty">选择左侧的 run</div></div>
</div>
<script>
let activeRun = null, collapsedGroups = new Set();
const $ = s => document.querySelector(s);
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function badge(s){const c=(s||"").toLowerCase();return '<span class="badge b-'+c+'">'+esc(s)+'</span>';}
function fmtTime(s){try{return new Date(s).toLocaleTimeString();}catch(e){return s;}}
function fmtDate(s){try{return new Date(s).toLocaleString();}catch(e){return s;}}
function lastSeg(ws){
  if(!ws) return '未绑定';
  const parts = ws.split(/[\\\\/]/).filter(Boolean);
  return parts[parts.length-1] || ws;
}

async function loadRuns(){
  const r = await fetch('/api/runs'); const runs = await r.json();
  const fa=$('#f-agent').value.toLowerCase(), fs=$('#f-status').value,
        ft=$('#f-task').value.toLowerCase();
  const filtered = runs.filter(x=>{
    if(fa && !(x.entry_agent_id||"").toLowerCase().includes(fa) &&
       !(x.entry_agent_name||"").toLowerCase().includes(fa)) return false;
    if(fs && x.status!==fs) return false;
    if(ft && !(x.original_task||"").toLowerCase().includes(ft)) return false;
    return true;
  });
  $('#meta').textContent = filtered.length+' / '+runs.length+' runs · '+new Date().toLocaleTimeString();
  // Group by workspace_dir
  const groups = {};
  const order = [];
  for(const x of filtered){
    const key = x.workspace_dir || '';
    if(!groups[key]){ groups[key]=[]; order.push(key); }
    groups[key].push(x);
  }
  let html = '';
  for(const key of order){
    const items = groups[key];
    const collapsed = collapsedGroups.has(key) ? 'collapsed' : '';
    const arrow = collapsed ? '▶' : '▼';
    const title = key ? esc(lastSeg(key)) : '未绑定 / 隔离';
    const full = key ? ' title="'+esc(key)+'"' : '';
    html += '<div class="ws-group">'+
      '<div class="ws-head" onclick="toggleGroup(this, '+JSON.stringify(key)+')">'+
      '<span class="arrow">'+arrow+'</span>'+
      '<span class="ws-name"'+full+'>'+esc(title)+'</span>'+
      '<span class="ws-count">'+items.length+'</span>'+
      '</div><div class="ws-body '+collapsed+'">';
    for(const x of items){
      const cls = x.id===activeRun?'active':'';
      html += '<div class="run-item '+cls+'" onclick="selectRun(\\''+x.id+'\\')">'+
        '<div class="task">'+esc(x.original_task||'(无任务)')+'</div>'+
        '<div class="sub">'+badge(x.status)+'<span>'+(x.entry_agent_name||x.entry_agent_id||'')+'</span><span>'+fmtTime(x.created_at)+'</span></div>'+
      '</div>';
    }
    html += '</div></div>';
  }
  if(!filtered.length) html = '<div class="empty">无匹配 run</div>';
  $('#runs').innerHTML = html;
  if(!activeRun && filtered.length) selectRun(filtered[0].id);
}

function toggleGroup(headEl, key){
  const body = headEl.nextElementSibling;
  const arrow = headEl.querySelector('.arrow');
  if(collapsedGroups.has(key)){
    collapsedGroups.delete(key);
    body.classList.remove('collapsed');
    arrow.textContent = '▼';
  } else {
    collapsedGroups.add(key);
    body.classList.add('collapsed');
    arrow.textContent = '▶';
  }
}

async function selectRun(id){
  activeRun = id;
  document.querySelectorAll('.run-item').forEach(e=>e.classList.toggle('active', e.getAttribute('onclick').includes(id)));
  const r = await fetch('/api/runs/'+id); if(!r.ok) return;
  const d = await r.json();
  renderTimeline(d); renderRight(d);
}

function renderTimeline(d){
  const events = d.events||[], tools = d.tools||[];
  // Merge by approximate time; tools carry their own ts.
  const items = events.map(e=>({type:'ev', t:e.created_at, e}));
  for(const t of tools){ items.push({type:'tool', t:t.ts, e:t}); }
  items.sort((a,b)=>(a.t||"").localeCompare(b.t||""));
  if(!items.length){ $('#timeline').innerHTML='<h2>事件时间线</h2><div class="empty">无事件</div>'; return; }
  $('#timeline').innerHTML='<h2>事件时间线 ('+items.length+')</h2>'+items.map(it=>{
    if(it.type==='tool'){
      const t=it.e, args=Object.entries(t.args||{}).map(([k,v])=>k+'='+esc(v)).join(' · ');
      return '<div class="ev tool"><div class="head"><span class="ts">'+fmtTime(t.ts)+'</span>'+
        '<span class="who">'+esc(t.agent_name||t.agent_id||'')+'</span>'+
        '<span class="phase">'+esc(t.phase)+'</span>'+
        '<span class="tool">'+esc(t.tool)+'</span></div>'+
        (args?'<div class="args">'+esc(args)+'</div>':'')+
        (t.result?'<div class="result">'+esc(t.result)+'</div>':'')+'</div>';
    }
    const e=it.e;
    return '<div class="ev"><div class="head"><span class="ts">'+fmtTime(e.created_at)+'</span>'+
      '<span class="who">'+esc(e.agent_name||e.agent_id||'')+'</span>'+
      '<span class="phase">'+esc(e.phase)+'</span>'+
      '<span class="kind">'+esc(e.kind)+'</span>'+
      '<span class="title">'+esc(e.title)+'</span></div>'+
      (e.detail?'<div class="detail">'+esc(e.detail)+'</div>':'')+'</div>';
  }).join('');
}

function renderRight(d){
  const mem = d.memory||[], msgs = d.messages||[];
  let html = '<h2>记忆写入 ('+mem.length+')</h2>';
  if(!mem.length) html += '<div class="empty">本 run 无记忆写入</div>';
  for(const m of mem){
    const w=m.why||{}, facts=m.facts||[];
    const factHtml = facts.map(f=>'<div class="fact"><span class="k">'+esc(f.domain)+' · '+esc(f.kind)+'</span><div class="t">'+esc(f.text)+'</div></div>').join('');
    html += '<div class="mem-card"><div class="top">'+
      '<span class="etype">'+esc(m.event_type)+'</span>'+
      '<span class="status">'+esc(m.status)+'</span></div>'+
      '<div class="why"><span>agent:</span> '+esc(w.agent_id||'-')+' · <span>source:</span> '+esc(w.source_type||'-')+' · <span>succeeded:</span> '+esc(w.succeeded)+'<br>'+
      (w.mutation_paths&&w.mutation_paths.length?'<span>改动:</span> '+esc(w.mutation_paths.join(', ')):'')+'</div>'+
      (m.attempted&&m.attempted.text?'<div class="why">拟写: '+esc(m.attempted.text)+'</div>':'')+
      factHtml+
      (m.error?'<div class="why" style="color:#d9a3a3">错误: '+esc(m.error)+'</div>':'')+
      '</div>';
  }
  html += '<h2>对话 ('+msgs.length+')</h2>';
  if(!msgs.length) html += '<div class="empty">无对话消息</div>';
  for(const m of msgs){
    html += '<div class="msg '+m.role+'"><div class="who">'+esc(m.role)+(m.agent_id?' · '+esc(m.agent_id):'')+' · '+fmtTime(m.created_at)+'</div>'+
      '<div class="bubble">'+esc(m.content)+'</div></div>';
  }
  $('#right').innerHTML = html;
}

$('#f-agent').oninput=loadRuns; $('#f-status').onchange=loadRuns; $('#f-task').oninput=loadRuns;
loadRuns();
setInterval(loadRuns, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store: DataStore = None  # type: ignore[assignment]

    def log_message(self, *args) -> None:  # silence default stderr logging
        pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            _html_response(self, PANEL_HTML)
            return
        if path == "/api/runs":
            _json_response(self, self.store.run_summaries())
            return
        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            detail = self.store.run_detail(run_id)
            if detail is None:
                _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            _json_response(self, detail)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def poll_loop(store: DataStore, interval: float, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            store.poll()
        except Exception:
            pass
        stop.wait(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="CatTogether agent monitor")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--interval", type=float, default=1.5)
    args = ap.parse_args()

    store = DataStore(args.data_dir)
    store.poll()  # initial load
    Handler.store = store

    stop = threading.Event()
    poller = threading.Thread(
        target=poll_loop, args=(store, args.interval, stop), daemon=True
    )
    poller.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"CatTogether agent monitor → http://127.0.0.1:{args.port}")
    print(f"data dir: {args.data_dir}")
    print(f"runs loaded: {len(store.run_summaries())}")
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
