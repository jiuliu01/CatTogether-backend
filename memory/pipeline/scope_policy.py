"""Stage 4: Scope & Policy — FactDraft + Actor → authorized scope (§7.1).

Determines whether the actor is allowed to write the fact in the draft's
domain/scope.  Uncertain scopes are quarantined (``status=pending_review``).
Domain expansion is prohibited: the draft's domain must be within the actor's
allowed scopes.

Returns ``None`` to drop a draft that the actor has no permission to write.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.pipeline import FactDraft


logger = logging.getLogger(__name__)


class ScopePolicy:
    """Stage 4: resolve and enforce scope policy on a draft.

    The draft's ``fact.domain`` and ``fact.scope_id`` are checked against the
    actor's compiled allowed scopes.  If the actor cannot read the draft's
    domain, the draft is dropped.  If the scope is uncertain, the fact is
    quarantined to ``pending_review``.
    """

    def resolve(
        self,
        draft: FactDraft,
        actor: Any,
    ) -> FactDraft | None:
        """Apply scope policy to *draft*.

        Returns the (possibly modified) draft, or ``None`` if the actor is
        not permitted to write in this domain/scope.
        """
        from memory.permissions import compile_allowed_scopes, can_read

        fact = draft.fact
        domain = fact.domain

        # Check tenant isolation: actor must be in the same tenant.
        if actor.tenant_id != fact.tenant_id:
            logger.warning(
                "scope_policy: tenant mismatch — actor=%s fact_tenant=%s",
                actor.tenant_id, fact.tenant_id,
            )
            return None

        # Check domain read permission.
        if not can_read(actor.role, domain):
            logger.info(
                "scope_policy: role %s cannot read domain %s; dropping draft",
                actor.role, domain,
            )
            return None

        # Compile allowed scopes for this actor.
        allowed = compile_allowed_scopes(
            actor, fact.tenant_id,
            domain=domain, scope_id=fact.scope_id,
        )
        if not allowed:
            # Actor has no access to this specific scope.
            logger.info(
                "scope_policy: no allowed scopes for %s in %s/%s; quarantining",
                actor.role, domain, fact.scope_id,
            )
            fact.status = "pending_review"  # type: ignore[assignment]
            return draft

        # Build a MemoryScope for the draft.
        from memory.scope import MemoryScope
        try:
            scope = MemoryScope(
                domain=domain,  # type: ignore[arg-type]
                scope_id=fact.scope_id,
                agent_id=fact.agent_id,
                tenant_id=fact.tenant_id,
            )
        except ValueError:
            # Invalid scope_id — quarantine rather than drop.
            logger.warning(
                "scope_policy: invalid scope_id %r; quarantining",
                fact.scope_id,
            )
            fact.status = "pending_review"  # type: ignore[assignment]
            return draft

        draft.scope = scope
        return draft
