"""Tenant auth middleware for Memory 2.1 REST API (§9, Schema v4).

FastAPI dependencies that extract tenant_id and role from the request and
produce an ``ActorContext`` for the permission matrix.

In the current no-auth phase (§9 security constraints):
- tenant_id comes from the ``X-Tenant-Id`` header (defaults to "default").
- role comes from the ``X-Memory-Role`` header (defaults to "user").
- When no headers are present, the actor is a default-tenant user.

In production with real auth, tenant_id and role are extracted from a trusted
authentication result (JWT / session token), **not** from client-supplied
headers.  The dependency injection point stays the same — only the extraction
logic changes.
"""
from __future__ import annotations

from typing import Literal

from fastapi import Depends, Header, HTTPException, Request

from memory.permissions import ActorContext, Role


def get_actor_context(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_memory_role: str | None = Header(default=None, alias="X-Memory-Role"),
    x_actor_id: str | None = Header(default=None, alias="X-Actor-Id"),
) -> ActorContext:
    """FastAPI dependency: build an ActorContext from request headers.

    Headers (all optional):
        X-Tenant-Id   — tenant identifier (default: "default")
        X-Memory-Role — role: admin/agent/coordinator/user/system (default: "user")
        X-Actor-Id    — actor identifier for audit (default: derived from role)

    An invalid role falls back to "user" for safety (least-privilege among
    roles that can at least propose writes in their own tenant).
    """
    tenant = (x_tenant_id or "default").strip()
    raw_role = (x_memory_role or "user").strip().lower()
    # Validate role; fall back to user for unknown values.
    valid_roles: tuple[Literal[str], ...] = (
        "admin", "agent", "coordinator", "user", "system",
    )
    role: Role = raw_role if raw_role in valid_roles else "user"  # type: ignore[assignment]
    actor_id = (x_actor_id or "").strip()

    return ActorContext(
        tenant_id=tenant,
        role=role,
        actor_id=actor_id,
        display_name=actor_id or role,
    )


def require_capability(capability: str):
    """FastAPI dependency factory: require a specific capability.

    Usage::

        @router.post("/...", dependencies=[Depends(require_capability("memory.delete"))])
        async def my_endpoint(actor: ActorContext = Depends(get_actor_context)):
            ...
    """
    def _check(actor: ActorContext = Depends(get_actor_context)) -> ActorContext:
        if not actor.can(capability):
            raise HTTPException(
                status_code=403,
                detail=f"role '{actor.role}' lacks capability '{capability}'",
            )
        return actor

    return _check
