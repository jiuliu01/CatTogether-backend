"""SQLite connection management for Memory 2.0 (§6.4).

Single ``memory.db`` file in WAL mode. All four layers (Hot / Fact / History /
Skill) share the same database file, separated by table prefix. The v3 schema
DDL lives in ``memory/migrations/v3_schema.sql`` and is applied idempotently
on first connect.

Usage::

    from memory.db import memory_db
    conn = memory_db.connect()          # sqlite3.Connection (sync)
    await memory_db.async_connect()     # aiosqlite connection (async)

Design decisions:
- Synchronous ``sqlite3`` for the hot path (context build, search) because
  SQLite operations are sub-millisecond and the GIL makes async overhead
  counterproductive for single-file embedded DB.
- ``aiosqlite`` is available for the write pipeline's async queue workers.
- WAL mode + ``busy_timeout`` for concurrent reader/writer safety.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings

_SCHEMA_VERSION = 4
_SCHEMA_FILE = Path(__file__).resolve().parent / "migrations" / "v4_schema.sql"
_VECTOR_SCHEMA_FILE = Path(__file__).resolve().parent / "migrations" / "v4_vector_schema.sql"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension. Returns True on success.

    sqlite-vec provides the ``vec0`` virtual table module for approximate
    nearest-neighbor search.  If the extension is unavailable we log a warning
    and return False; vector features are simply disabled (graceful degrade).
    """
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        return True
    except Exception:
        # sqlite-vec not installed or load failed — vector features disabled.
        return False


class MemoryDB:
    """Singleton SQLite connection manager for the v3 memory database.

    The underlying ``sqlite3.Connection`` is shared across threads with
    ``check_same_thread=False`` and guarded by a ``threading.Lock`` for writes.
    SQLite WAL mode allows concurrent readers without blocking.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.memory_sqlite_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._migrated = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def connect(self) -> sqlite3.Connection:
        """Return the shared connection, applying schema on first call."""
        if self._conn is not None:
            return self._conn
        with self._lock:
            if self._conn is not None:
                return self._conn
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit mode; we manage txns explicitly
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if settings.memory_sqlite_wal:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            else:
                conn.execute("PRAGMA journal_mode = DELETE")
                conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA temp_store = MEMORY")
            self._conn = conn
            self._vec_available = _try_load_vec_extension(conn)
            self._apply_schema(conn)
            if self._vec_available:
                self._apply_vector_schema(conn)
            self._migrated = True
            return conn

    @property
    def vec_available(self) -> bool:
        """True if the sqlite-vec extension was loaded successfully."""
        if self._conn is None:
            self.connect()
        return getattr(self, "_vec_available", False)

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        """Apply v4 schema DDL idempotently and record migration version."""
        sql = _SCHEMA_FILE.read_text(encoding="utf-8")
        conn.executescript(sql)
        # Record schema version (idempotent).
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) "
            "VALUES (?, ?, ?)",
            (_SCHEMA_VERSION, _utc_now_iso(), "v4 four-layer schema"),
        )

    def _apply_vector_schema(self, conn: sqlite3.Connection) -> None:
        """Apply vector schema DDL (vec0 virtual tables) idempotently.

        Only called when ``_try_load_vec_extension`` succeeded.  The embedding
        dimension is read from settings so switching models just requires a
        config change + re-create.
        """
        dim = settings.memory_embedding_dim
        sql = _VECTOR_SCHEMA_FILE.read_text(encoding="utf-8")
        conn.executescript(sql.replace("__EMBEDDING_DIM__", str(dim)))

    def get_conn(self) -> sqlite3.Connection:
        """Alias for connect() when the caller knows the DB is initialized."""
        return self.connect()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._migrated = False

    # --- Transaction helpers ---

    def transaction(self) -> sqlite3.Connection:
        """Begin a deferred transaction and return the connection.

        Caller is responsible for ``commit()`` or ``rollback()``. In autocommit
        mode (``isolation_level=None``), we issue ``BEGIN`` explicitly.
        """
        conn = self.connect()
        conn.execute("BEGIN")
        return conn

    def commit(self) -> None:
        self.connect().execute("COMMIT")

    def rollback(self) -> None:
        self.connect().execute("ROLLBACK")

    # --- Low-level query helpers ---

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        return self.connect().executemany(sql, params)

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    # --- Maintenance ---

    def vacuum(self) -> None:
        """Reclaim free pages and rebuild FTS5 structures."""
        self.connect().execute("PRAGMA optimize")

    def backup_to(self, dest_path: Path) -> None:
        """Online backup to a separate file."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            dest_path.unlink()
        dest = sqlite3.connect(str(dest_path))
        try:
            self.connect().backup(dest)
        finally:
            dest.close()


# Module-level singleton, initialized lazily on first connect().
memory_db = MemoryDB()
