"""Memory 2.1 scope: four-domain + tenant_id (merged from scope_v3).

2.1 final revisions:
- R5: tenant_id from stage 0; every storage_key is tenant-prefixed for hard
  isolation.
- workspace→project rename (with ``workspace`` accepted as an alias for
  backward-compatible parsing of v1/v2 data).
- task domain added as a first-class scope.

This is the single canonical scope module for Memory 2.1.  The old
``scope_v3`` module has been merged here; ``MemoryScopeV3`` is kept as a
backward-compat alias.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from core.identifiers import validate_storage_id


MemoryDomain = Literal["user", "project", "task", "agent"]
"""Four-domain scope (2.1 final). ``workspace`` is accepted on input as an
alias for ``project``."""

FactDomain = MemoryDomain
"""Alias for code that references the four-domain type by this name."""

_DOMAIN_ALIASES: dict[str, str] = {"workspace": "project"}
_VALID_DOMAINS: frozenset[str] = frozenset({"user", "project", "task", "agent"})
_CANONICAL_USER_SCOPE = re.compile(r"^u_[0-9a-f]{64}$")
_CANONICAL_TASK_SCOPE = re.compile(r"^t_[0-9a-f]{64}$")


def normalize_domain(domain: str) -> MemoryDomain:
    """Normalize a domain string, applying the workspace→project rename."""
    key = (domain or "").strip().lower()
    key = _DOMAIN_ALIASES.get(key, key)
    if key not in _VALID_DOMAINS:
        raise ValueError(
            f"memory domain must be one of user/project/task/agent, got {domain!r}"
        )
    return key  # type: ignore[return-value]


def normalize_agent_role(role: str) -> str:
    value = (role or "custom").strip().lower().replace(" ", "-")
    return validate_storage_id(value, "agent_role")


def normalize_tenant_id(tenant_id: str) -> str:
    """Normalize and validate a tenant_id."""
    value = (tenant_id or "default").strip()
    if not value:
        value = "default"
    return validate_storage_id(value, "tenant_id")


def tenant_id_from_user_id(user_id: str, fallback: str = "default") -> str:
    """Extract the Feishu tenant from ``feishu:<tenant>:<open_id>`` IDs."""
    value = (user_id or "").strip()
    if value.startswith("feishu:"):
        parts = value.split(":", 2)
        if len(parts) == 3 and parts[1]:
            return normalize_tenant_id(parts[1])
    return normalize_tenant_id(fallback)


def canonical_scope_id(domain: str, scope_id: str) -> str:
    """Return the persisted scope identifier for a domain.

    User/task identifiers can contain external IDs, so their database scope is
    a one-way hash.  Already canonical IDs are preserved to avoid double hash.
    """
    normalized = normalize_domain(domain)
    value = (scope_id or "").strip()
    if not value:
        raise ValueError("scope_id is required")
    if normalized == "user":
        if _CANONICAL_USER_SCOPE.fullmatch(value):
            return value
        return f"u_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    if normalized == "task":
        if _CANONICAL_TASK_SCOPE.fullmatch(value):
            return value
        return f"t_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    if normalized == "project":
        return validate_storage_id(value, "project_id")
    return normalize_agent_role(value)


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Four-domain + tenant_id scope (§5.2).

    ``storage_key`` is ``f"{tenant_id}:{domain}:{ident}"`` for hard tenant
    isolation.  For user/task domains the scope_id is SHA-256 hashed (it may
    contain non-path-safe characters); for project/agent domains the
    validated scope_id is used directly.
    """
    domain: MemoryDomain
    scope_id: str
    agent_id: str | None = None
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        norm_tenant = normalize_tenant_id(self.tenant_id)
        norm_domain = normalize_domain(self.domain)
        if not self.scope_id:
            raise ValueError("scope_id is required")
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "tenant_id", norm_tenant)
        object.__setattr__(self, "domain", norm_domain)
        if norm_domain == "project":
            object.__setattr__(self, "scope_id", validate_storage_id(self.scope_id, "project_id"))
        elif norm_domain == "task":
            object.__setattr__(self, "scope_id", validate_storage_id(self.scope_id, "task_id"))
        elif norm_domain == "agent":
            object.__setattr__(self, "scope_id", normalize_agent_role(self.scope_id))
            if self.agent_id:
                validate_storage_id(self.agent_id, "agent_id")

    @property
    def storage_key(self) -> str:
        """Tenant-prefixed storage key for hard isolation."""
        if self.domain in ("user", "task"):
            digest = hashlib.sha256(self.scope_id.encode("utf-8")).hexdigest()
            ident = f"u_{digest}" if self.domain == "user" else f"t_{digest}"
        elif self.domain == "project":
            ident = self.scope_id
        else:  # agent
            role = normalize_agent_role(self.scope_id)
            if self.agent_id:
                ident = f"{role}__{validate_storage_id(self.agent_id, 'agent_id')}"
            else:
                ident = role
        return f"{self.tenant_id}__{self.domain}__{ident}"

    @property
    def storage_key_v1_compatible(self) -> str:
        """Storage key without tenant prefix, for v1 JSON backend compat."""
        if self.domain == "user":
            digest = hashlib.sha256(self.scope_id.encode("utf-8")).hexdigest()
            return f"u_{digest}"
        if self.domain == "project":
            return self.scope_id
        role = normalize_agent_role(self.scope_id)
        if self.agent_id:
            return f"{role}__{validate_storage_id(self.agent_id, 'agent_id')}"
        return role


# Backward-compat alias for code still referencing the v3 name.
MemoryScopeV3 = MemoryScope
