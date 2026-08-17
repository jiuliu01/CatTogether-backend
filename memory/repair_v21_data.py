"""Idempotent repair for the first Memory 2.1 migration/runtime rollout.

Repairs four issues without keeping a second copy of the data:

* project-agent History/Skill rows accidentally scoped to literal ``projects``;
* real Feishu project/user rows left under tenant ``default``;
* existing Agent specs missing the new ``memory_search`` capability;
* empty scope_registry after migration.

Run ``python -m memory.repair_v21_data`` for a report and add ``--execute`` to
apply it.  The database changes are committed in one transaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import settings


_BACKEND = Path(__file__).resolve().parent.parent
_PROJECT_ID = re.compile(r"(?i)(?:data[\\/]projects[\\/]|移植到\s+)([0-9a-f]{32})")


@dataclass(frozen=True)
class ProjectInfo:
    project_id: str
    tenant_id: str
    created_at: datetime
    is_feishu: bool


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _projects() -> dict[str, ProjectInfo]:
    result: dict[str, ProjectInfo] = {}
    for path in (_BACKEND / "data" / "projects").glob("*/project.json"):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        project_id = str(data.get("id") or path.parent.name)
        created_raw = str(data.get("created_at") or "1970-01-01T00:00:00+00:00")
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except ValueError:
            created_at = datetime.fromisoformat("1970-01-01T00:00:00+00:00")
        result[project_id] = ProjectInfo(
            project_id=project_id,
            tenant_id=str(data.get("tenant_key") or "default"),
            created_at=created_at,
            is_feishu=bool(data.get("feishu_chat_id")),
        )
    return result


def _bindings() -> list[dict]:
    data = _read_json(_BACKEND / "data" / "feishu" / "bindings.json")
    return data if isinstance(data, list) else []


def _user_scopes() -> dict[str, str]:
    """Map canonical user scopes to their Feishu tenant."""
    result: dict[str, str] = {}
    for path in (_BACKEND / "data" / "feishu" / "runs").glob("*.json"):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        tenant = str(data.get("tenant_key") or "")
        open_id = str(data.get("sender_open_id") or "")
        if not tenant or not open_id:
            continue
        external_id = f"feishu:{tenant}:{open_id}"
        scope = "u_" + hashlib.sha256(external_id.encode("utf-8")).hexdigest()
        result[scope] = tenant
    return result


def _external_users() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in (_BACKEND / "data" / "feishu" / "runs").glob("*.json"):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        tenant = str(data.get("tenant_key") or "")
        open_id = str(data.get("sender_open_id") or "")
        if tenant and open_id:
            result[f"feishu:{tenant}:{open_id}"] = tenant
    return result


def _infer_history_projects(
    conn: sqlite3.Connection,
    projects: dict[str, ProjectInfo],
) -> dict[str, str]:
    rows = conn.execute(
        "SELECT message_id, agent_id, content, created_at FROM session_messages "
        "WHERE channel_id='projects'"
    ).fetchall()
    direct: dict[str, str] = {}
    by_time: dict[str, str] = {}
    for row in rows:
        matches = _PROJECT_ID.findall(row[2] or "")
        if matches:
            target = matches[-1]
            direct[row[0]] = target
            by_time[row[3]] = target

    real_projects = sorted(
        (item for item in projects.values() if item.is_feishu),
        key=lambda item: item.created_at,
    )
    for row in rows:
        message_id, agent_id, _content, created_raw = row
        if message_id in direct:
            continue
        if created_raw in by_time:
            direct[message_id] = by_time[created_raw]
            continue
        # Human project histories have no embedded project ID.  Match them to
        # the latest real Feishu project that already existed at that time.
        if agent_id == "coordinator" and real_projects:
            created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
            candidates = [item for item in real_projects if item.created_at <= created]
            if candidates:
                direct[message_id] = candidates[-1].project_id
                continue
        direct[message_id] = "legacy-unassigned"
    return direct


def _move_fact_scope_tenant(
    conn: sqlite3.Connection,
    domain: str,
    scope_id: str,
    tenant_id: str,
) -> int:
    fact_ids = [
        row[0] for row in conn.execute(
            "SELECT id FROM facts WHERE tenant_id='default' AND domain=? AND scope_id=?",
            (domain, scope_id),
        )
    ]
    if not fact_ids or tenant_id == "default":
        return 0
    placeholders = ",".join("?" for _ in fact_ids)
    params = (tenant_id, *fact_ids)
    conn.execute(
        f"""INSERT OR IGNORE INTO entities
            (entity_id, tenant_id, canonical_name, type, aliases, first_seen_at,
             last_seen_at, merged_into_id)
            SELECT entity_id, ?, canonical_name, type, aliases, first_seen_at,
                   last_seen_at, merged_into_id
            FROM entities WHERE tenant_id='default' AND entity_id IN (
              SELECT entity_id FROM fact_entity_mentions
              WHERE tenant_id='default' AND fact_id IN ({placeholders})
            )""",
        params,
    )
    conn.execute(
        f"""INSERT OR IGNORE INTO entity_aliases (entity_id, alias, tenant_id)
            SELECT entity_id, alias, ? FROM entity_aliases
            WHERE tenant_id='default' AND entity_id IN (
              SELECT entity_id FROM fact_entity_mentions
              WHERE tenant_id='default' AND fact_id IN ({placeholders})
            )""",
        params,
    )
    for table in ("fact_entity_mentions", "memory_audit_log", "transaction_outbox"):
        key = {
            "fact_entity_mentions": "fact_id",
            "fact_relations": "src_fact_id",
            "memory_audit_log": "fact_id",
            "transaction_outbox": "fact_id",
        }[table]
        conn.execute(
            f"UPDATE {table} SET tenant_id=? WHERE tenant_id='default' "
            f"AND {key} IN ({placeholders})",
            params,
        )
    conn.execute(
        f"UPDATE fact_relations SET tenant_id=? WHERE tenant_id='default' "
        f"AND (src_fact_id IN ({placeholders}) OR dst_fact_id IN ({placeholders}))",
        (tenant_id, *fact_ids, *fact_ids),
    )
    conn.execute(
        f"UPDATE facts SET tenant_id=? WHERE tenant_id='default' AND id IN ({placeholders})",
        params,
    )
    return len(fact_ids)


def _upgrade_agent_specs(execute: bool, project_ids: set[str] | None = None) -> int:
    changed = 0
    for path in (_BACKEND / "data" / "projects").glob("*/agents/*.json"):
        if project_ids is not None and path.parent.parent.name not in project_ids:
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        memory = data.get("memory") if isinstance(data.get("memory"), dict) else {}
        if memory.get("enabled", True) is False:
            continue
        capabilities = list(data.get("mcp_capabilities") or [])
        if "memory_search" in capabilities:
            continue
        capabilities.append("memory_search")
        data["mcp_capabilities"] = capabilities
        changed += 1
        if execute:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def repair(execute: bool = False) -> dict[str, int]:
    projects = _projects()
    active_project_ids = {
        project_id for project_id, info in projects.items() if info.is_feishu
    }
    bindings = _bindings()
    user_scopes = _user_scopes()
    external_users = _external_users()
    conn = sqlite3.connect(str(settings.memory_sqlite_path))
    conn.execute("PRAGMA busy_timeout=5000")
    mapping = _infer_history_projects(conn, projects)
    stats = {
        "history_rows": len(mapping),
        "skill_rows": conn.execute(
            "SELECT count(*) FROM skill_registry WHERE scope_id='projects'"
        ).fetchone()[0],
        "agent_specs": _upgrade_agent_specs(False, active_project_ids),
        "facts": 0,
    }
    for project_id, info in projects.items():
        if info.tenant_id != "default":
            stats["facts"] += conn.execute(
                "SELECT count(*) FROM facts WHERE tenant_id='default' AND domain='project' AND scope_id=?",
                (project_id,),
            ).fetchone()[0]
    for scope_id, tenant in user_scopes.items():
        if tenant != "default":
            stats["facts"] += conn.execute(
                "SELECT count(*) FROM facts WHERE tenant_id='default' AND domain='user' AND scope_id=?",
                (scope_id,),
            ).fetchone()[0]
    if not execute:
        conn.close()
        return stats

    conn.execute("BEGIN IMMEDIATE")
    try:
        stats["facts"] = 0
        for message_id, project_id in mapping.items():
            info = projects.get(project_id)
            tenant = info.tenant_id if info else "default"
            conn.execute(
                "UPDATE session_messages SET channel_id=?, tenant_id=? WHERE message_id=?",
                (project_id, tenant, message_id),
            )

        skills = conn.execute(
            "SELECT id, body FROM skill_registry WHERE scope_id='projects'"
        ).fetchall()
        for skill_id, body in skills:
            matches = _PROJECT_ID.findall(body or "")
            target = matches[-1] if matches else "legacy-unassigned"
            info = projects.get(target)
            tenant = info.tenant_id if info else "default"
            conn.execute(
                "UPDATE skill_registry SET scope_id=?, tenant_id=? WHERE id=?",
                (target, tenant, skill_id),
            )

        for project_id, info in projects.items():
            stats["facts"] += _move_fact_scope_tenant(
                conn, "project", project_id, info.tenant_id,
            )
            conn.execute(
                "UPDATE events SET tenant_id=? WHERE tenant_id='default' AND scope_hint=?",
                (info.tenant_id, project_id),
            )
        for scope_id, tenant in user_scopes.items():
            stats["facts"] += _move_fact_scope_tenant(conn, "user", scope_id, tenant)
        for external_id, tenant in external_users.items():
            conn.execute(
                "UPDATE events SET tenant_id=? WHERE tenant_id='default' AND scope_hint=?",
                (tenant, external_id),
            )
        for binding in bindings:
            tenant = str(binding.get("tenant_key") or "default")
            channel = str(binding.get("channel_id") or "")
            project_id = str(binding.get("project_id") or "")
            if channel:
                conn.execute(
                    "UPDATE session_messages SET tenant_id=? WHERE tenant_id='default' AND channel_id=?",
                    (tenant, channel),
                )
            if project_id:
                conn.execute(
                    "UPDATE events SET tenant_id=? WHERE tenant_id='default' AND scope_hint=?",
                    (tenant, project_id),
                )

        # Rebuild sequence numbers per repaired channel/thread.
        groups = conn.execute(
            "SELECT DISTINCT tenant_id, channel_id, thread_id FROM session_messages"
        ).fetchall()
        for tenant, channel, thread in groups:
            rows = conn.execute(
                "SELECT message_id FROM session_messages WHERE tenant_id=? AND channel_id=? "
                "AND thread_id IS ? ORDER BY created_at, message_id",
                (tenant, channel, thread),
            ).fetchall()
            for seq, (message_id,) in enumerate(rows, 1):
                conn.execute(
                    "UPDATE session_messages SET seq=? WHERE tenant_id=? AND message_id=?",
                    (seq, tenant, message_id),
                )

        now = datetime.now().astimezone().isoformat()
        conn.execute("DELETE FROM scope_registry")
        conn.execute(
            """INSERT OR IGNORE INTO scope_registry
               (tenant_id, domain, scope_id, agent_id, first_seen, last_seen)
               SELECT tenant_id, domain, scope_id, COALESCE(agent_id, ''),
                      MIN(created_at), MAX(updated_at)
               FROM facts GROUP BY tenant_id, domain, scope_id, COALESCE(agent_id, '')"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO scope_registry
               (tenant_id, domain, scope_id, agent_id, first_seen, last_seen)
               SELECT tenant_id, domain, scope_id, '', MIN(created_at), MAX(updated_at)
               FROM skill_registry GROUP BY tenant_id, domain, scope_id"""
        )
        if not conn.execute("SELECT 1 FROM scope_registry LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO scope_registry VALUES ('default','agent','legacy-unassigned','',?,?)",
                (now, now),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    stats["agent_specs"] = _upgrade_agent_specs(True, active_project_ids)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair Memory 2.1 rollout data")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    stats = repair(args.execute)
    mode = "executed" if args.execute else "dry-run"
    print(json.dumps({"mode": mode, **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
