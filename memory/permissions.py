"""Permission matrix for Memory 2.1 (§9, Schema v4).

Defines the role → capability mapping that governs access to the five memory
operations and the four-layer specialised APIs.

Capabilities (§9 permission matrix):
    memory.read.<domain>        — search / list facts in a domain
    memory.write.propose        — submit a write request (agent or user)
    memory.write.commit         — directly commit a fact (system/admin only)
    memory.update               — patch fact metadata
    memory.delete               — soft-delete (archive) a fact
    memory.forget               — hard-delete a fact (GDPR)
    memory.hot.approve          — approve pending hot-memory items
    memory.hot.manage           — add/archive hot-memory items
    memory.skill.rollback       — rollback a skill to a prior version
    memory.skill.manage         — create/update skills

Roles (2.1 final — five roles):
    admin       — full access (human administrator)
    system      — full access including commit (internal pipeline only)
    coordinator — approve project-scope proposals, read all domains, no
                  delete/forget/commit; cannot cross tenant
    agent       — propose writes, read all domains, no delete/forget
    user        — propose writes, read user/project/task, no agent domain

``readonly`` was removed in 2.1; ``coordinator`` replaces it with approval
authority.  The external API never exposes ``CAP_WRITE_COMMIT`` — only
``admin``/``system`` hold it internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Role = Literal["admin", "agent", "coordinator", "user", "system"]
"""Five actor roles (2.1 final). ``readonly`` removed; ``coordinator`` added."""

Domain = Literal["user", "project", "task", "agent"]
"""The four memory domains."""

# ---------------------------------------------------------------------------
# Capability constants
# ---------------------------------------------------------------------------

CAP_READ_PREFIX = "memory.read."
CAP_WRITE_PROPOSE = "memory.write.propose"
CAP_WRITE_COMMIT = "memory.write.commit"
CAP_UPDATE = "memory.update"
CAP_DELETE = "memory.delete"
CAP_FORGET = "memory.forget"
CAP_HOT_APPROVE = "memory.hot.approve"
CAP_HOT_MANAGE = "memory.hot.manage"
CAP_SKILL_ROLLBACK = "memory.skill.rollback"
CAP_SKILL_MANAGE = "memory.skill.manage"


def read_capability(domain: str) -> str:
    """Build a ``memory.read.<domain>`` capability string."""
    return f"{CAP_READ_PREFIX}{domain}"


# ---------------------------------------------------------------------------
# Role → capability set
# ---------------------------------------------------------------------------

_ALL_DOMAINS: tuple[Domain, ...] = ("user", "project", "task", "agent")

# Admin: everything.
_ADMIN_CAPS: frozenset[str] = frozenset(
    {read_capability(d) for d in _ALL_DOMAINS}
    | {
        CAP_WRITE_PROPOSE, CAP_WRITE_COMMIT, CAP_UPDATE, CAP_DELETE,
        CAP_FORGET, CAP_HOT_APPROVE, CAP_HOT_MANAGE,
        CAP_SKILL_ROLLBACK, CAP_SKILL_MANAGE,
    }
)

# System: full access (internal pipeline).
_SYSTEM_CAPS: frozenset[str] = _ADMIN_CAPS

# Agent: propose writes, read all domains, update metadata; no delete/forget.
_AGENT_CAPS: frozenset[str] = frozenset(
    {read_capability(d) for d in _ALL_DOMAINS}
    | {CAP_WRITE_PROPOSE, CAP_UPDATE, CAP_HOT_MANAGE, CAP_SKILL_MANAGE}
)

# Coordinator: approve project-scope proposals, read all domains, update metadata;
# no delete/forget/commit.  Cannot cross tenant (enforced by scope compilation).
_COORDINATOR_CAPS: frozenset[str] = frozenset(
    {read_capability(d) for d in _ALL_DOMAINS}
    | {CAP_WRITE_PROPOSE, CAP_UPDATE, CAP_HOT_APPROVE, CAP_HOT_MANAGE,
       CAP_SKILL_MANAGE}
)

# User: propose writes, read user/project/task (NOT agent); no delete/forget.
_USER_CAPS: frozenset[str] = frozenset(
    {read_capability(d) for d in ("user", "project", "task")}
    | {CAP_WRITE_PROPOSE, CAP_UPDATE}
)

_ROLE_CAPS: dict[str, frozenset[str]] = {
    "admin": _ADMIN_CAPS,
    "system": _SYSTEM_CAPS,
    "coordinator": _COORDINATOR_CAPS,
    "agent": _AGENT_CAPS,
    "user": _USER_CAPS,
}


# ---------------------------------------------------------------------------
# Permission check
# ---------------------------------------------------------------------------

class PermissionDenied(Exception):
    """Raised when an actor lacks a required capability."""

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(
            f"role '{role}' lacks capability '{capability}'"
        )


def capabilities_for(role: str) -> frozenset[str]:
    """Return the set of capabilities granted to *role*."""
    return _ROLE_CAPS.get(role, frozenset())


def check(role: str, capability: str) -> bool:
    """Return True if *role* has *capability*."""
    return capability in _ROLE_CAPS.get(role, frozenset())


def ensure(role: str, capability: str) -> None:
    """Raise PermissionDenied if *role* lacks *capability*."""
    if not check(role, capability):
        raise PermissionDenied(role, capability)


def can_read(role: str, domain: str) -> bool:
    """Convenience: check ``memory.read.<domain>``."""
    return check(role, read_capability(domain))


def ensure_read(role: str, domain: str) -> None:
    """Raise PermissionDenied if *role* cannot read *domain*."""
    ensure(role, read_capability(domain))


# ---------------------------------------------------------------------------
# Actor context
# ---------------------------------------------------------------------------

class ActorContext:
    """The authenticated actor making a memory API call.

    Carries tenant_id, role, and a display identifier for audit logging.
    This is what the tenant-auth middleware produces and what every
    MemoryAPI operation consumes.
    """

    __slots__ = ("tenant_id", "role", "actor_id", "display_name")

    def __init__(
        self,
        tenant_id: str = "default",
        role: Role = "user",
        actor_id: str = "",
        display_name: str = "",
    ) -> None:
        self.tenant_id = tenant_id or "default"
        self.role = role
        self.actor_id = actor_id or display_name or role
        self.display_name = display_name or actor_id or role

    def __repr__(self) -> str:
        return (
            f"ActorContext(tenant_id={self.tenant_id!r}, role={self.role!r}, "
            f"actor_id={self.actor_id!r})"
        )

    def ensure(self, capability: str) -> None:
        """Raise PermissionDenied if this actor lacks *capability*."""
        ensure(self.role, capability)

    def can(self, capability: str) -> bool:
        return check(self.role, capability)

    def can_read(self, domain: str) -> bool:
        return can_read(self.role, domain)


# ---------------------------------------------------------------------------
# Scope compilation (§9 — compile_allowed_scopes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AllowedScope:
    """A (domain, scope_id) pair the actor is permitted to access."""
    domain: Domain
    scope_id: str


def compile_allowed_scopes(
    actor: ActorContext,
    tenant_id: str,
    *,
    domain: str | None = None,
    scope_id: str | None = None,
) -> list[AllowedScope]:
    """Compile the list of scopes *actor* may access within *tenant_id*.

    Enforces tenant isolation: if ``actor.tenant_id != tenant_id`` the result
    is empty (cross-tenant access is always denied).

    When *domain* and *scope_id* are provided, the result is filtered to that
    specific scope (if permitted).  Otherwise all readable domains for the
    actor's role are returned as wildcard scopes.
    """
    # R5: hard tenant isolation.
    if actor.tenant_id != tenant_id:
        return []

    # Determine which domains the actor can read.
    readable: list[Domain] = []
    for d in _ALL_DOMAINS:
        if can_read(actor.role, d):
            readable.append(d)

    if domain is not None:
        # Filter to the requested domain.
        if domain not in readable:
            return []
        readable = [domain]  # type: ignore[assignment]

    if scope_id is not None:
        return [AllowedScope(d, scope_id) for d in readable]
    # Wildcard: actor can read all items in readable domains.
    return [AllowedScope(d, "*") for d in readable]
