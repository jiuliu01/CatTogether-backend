"""Shared redaction gate for Agent-authored text and persisted artifacts."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyResult:
    text: str
    findings: tuple[str, ...] = ()


_TOKEN_RE = re.compile(r"\b(?:sk|t)-[A-Za-z0-9_-]{12,}\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
_KEY_VALUE_RE = re.compile(
    r'''(?ix)
    (?P<prefix>["']?(?:
        anthropic_auth_token|anthropic_api_key|openai_api_key|feishu_app_secret|
        api_key|access_token|refresh_token|authorization|cookie|password|
        client_secret|secret_key|private_key
    )["']?\s*[:=]\s*["']?)
    (?P<value>[^\s,"'}]{8,})
    '''
)
_USER_HOME_RE = re.compile(
    r'''(?ix)
    (?:\b[A-Z]:\\Users\\[^\\\s"']+(?:\\[^\s"']+)*)
    |(?:/(?:home|Users)/[^/\s"']+(?:/[^\s"']+)*)
    '''
)


def sanitize_text(text: str, *, configured_secrets: tuple[str | None, ...] = ()) -> SafetyResult:
    value = str(text or "")
    findings: list[str] = []

    for secret in configured_secrets:
        if secret and secret in value:
            value = value.replace(secret, "[已隐藏密钥]")
            findings.append("configured_secret")

    value, count = _TOKEN_RE.subn("[已隐藏令牌]", value)
    if count:
        findings.append("token")

    value, count = _BEARER_RE.subn("Bearer [已隐藏令牌]", value)
    if count:
        findings.append("authorization")

    def _replace_key(match: re.Match) -> str:
        findings.append("credential_field")
        return f"{match.group('prefix')}[已隐藏密钥]"

    value = _KEY_VALUE_RE.sub(_replace_key, value)
    value, count = _USER_HOME_RE.subn("<USER_HOME_PATH>", value)
    if count:
        findings.append("user_path")

    return SafetyResult(value, tuple(dict.fromkeys(findings)))


def sanitize_agent_text(text: str) -> SafetyResult:
    """Apply runtime-configured secrets without importing settings at module load."""
    from config import settings

    return sanitize_text(
        text,
        configured_secrets=(
            settings.feishu_app_secret,
            settings.openai_api_key,
            settings.anthropic_api_key,
        ),
    )
