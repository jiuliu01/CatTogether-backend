"""Validation for identifiers that become filesystem path components."""
from __future__ import annotations

import re


_STORAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_storage_id(value: str, field_name: str) -> str:
    """Return a safe single path component or raise ``ValueError``."""
    if not _STORAGE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 1-128 characters using only letters, "
            "numbers, dot, underscore, or hyphen"
        )
    if ".." in value or value.endswith("."):
        raise ValueError(
            f"{field_name} cannot contain consecutive dots or end with a dot"
        )
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{field_name} uses a reserved filename")
    return value
