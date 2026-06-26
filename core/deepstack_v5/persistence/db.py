"""SQLite-backed runtime database for DeepStack v5.

Design notes:
* WAL journal mode lets multiple processes read concurrently while a single
  writer is in flight. This is the substrate enabling worker harness
  multi-process operation.
* Connections are per-thread (threading.local). Each thread that touches the
  DB lazily opens its own sqlite3 connection.
* `transaction()` uses BEGIN IMMEDIATE so concurrent writers fail fast on
  contention rather than waiting silently.
* Schema is created via a single SQL block guarded by `CREATE TABLE IF NOT
  EXISTS`; bumps go through `_MIGRATIONS` and a `schema_version` row.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

_MIN_SQLITE_VERSION = (3, 35)
_INIT_LOCK_GUARD = threading.Lock()
_INIT_LOCKS: dict[str, threading.Lock] = {}


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    budget_json TEXT,
    config_json TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    state TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    attempts_json TEXT NOT NULL,
    result_json TEXT,
    failure_json TEXT,
    history_json TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (run_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_state ON nodes(run_id, state);

CREATE TABLE IF NOT EXISTS attempts_index (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    worker_id TEXT,
    started_at_ms INTEGER,
    ended_at_ms INTEGER,
    outcome TEXT,
    PRIMARY KEY (run_id, node_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS edges (
    run_id TEXT NOT NULL,
    src_node_id TEXT NOT NULL,
    dst_node_id TEXT NOT NULL,
    PRIMARY KEY (run_id, src_node_id, dst_node_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(run_id, dst_node_id);

CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    granted_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER NOT NULL,
    heartbeat_at_ms INTEGER NOT NULL,
    UNIQUE(run_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at_ms);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, sequence);

CREATE TABLE IF NOT EXISTS outbox (
    sequence INTEGER PRIMARY KEY,
    delivered_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_outbox_undelivered
    ON outbox(sequence) WHERE delivered_at_ms IS NULL;

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    archived_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_claims_run ON claims(run_id);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    triggered_by TEXT,
    highwater_sequence INTEGER,
    summary TEXT,
    payload_json TEXT,
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run
    ON snapshots(run_id, created_at_ms DESC);
"""


_CURRENT_SCHEMA_VERSION = 3


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add the ``snapshots`` table introduced by Phase 3.

    Idempotent: re-running on a DB that already has the table (because
    a previous migration completed but the version bump didn't, or
    because a v2-fresh DB picked it up via ``_SCHEMA_SQL``) is a
    no-op.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            triggered_by TEXT,
            highwater_sequence INTEGER,
            summary TEXT,
            payload_json TEXT,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snapshots_run
            ON snapshots(run_id, created_at_ms DESC);
        """
    )


# Registry: from_version → migration function. Each migrator is
# responsible for moving the schema by exactly one version; the loop in
# ``_migrate`` bumps ``schema_version`` after each step so a crash
# between steps leaves a recoverable state.
_MIGRATIONS: dict[int, Any] = {
    1: _migrate_v1_to_v2,
}


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(nodes)").fetchall()
    }
    if "history_json" not in cols:
        try:
            conn.execute("ALTER TABLE nodes ADD COLUMN history_json TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


_MIGRATIONS[2] = _migrate_v2_to_v3


class SQLiteRuntimeDB:
    """Thread-safe SQLite wrapper with per-thread connections.

    A single ``SQLiteRuntimeDB`` opens one connection per Python thread
    that touches it. ``close()`` closes EVERY tracked connection (across
    all threads), so worker threads spawned by a thread pool don't leak
    their per-thread connections when the pool is torn down.

    ``:memory:`` paths are NOT shared across threads — sqlite3's
    in-memory DB is private per connection. With multi-thread access the
    URI form ``file::memory:?cache=shared`` is required; we leave that to
    the caller because the trade-offs (locking semantics, persistence)
    differ. The only safe pattern with this class plus ``:memory:`` is
    single-threaded use.
    """

    def __init__(self, path: str | Path):
        if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
            raise RuntimeError(
                "DeepStack v5 requires SQLite >= 3.35 because lease reclaim "
                "uses DELETE ... RETURNING"
            )
        self.path = str(path)
        self._local = threading.local()
        # RLock because _initialize() acquires the lock and then calls
        # _get_conn(), which itself reacquires it to register the new
        # connection. Plain Lock would deadlock.
        self._lock = threading.RLock()
        # All connections we've opened, across threads. Used for
        # close-all on shutdown so per-thread connections from worker
        # pools don't outlive the runtime.
        self._all_conns: list[sqlite3.Connection] = []
        self._init_lock = _init_lock_for(self.path)
        # Ensure parent directory exists for non-memory paths.
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    # -- connection management ------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path,
                isolation_level=None,  # autocommit; we manage txns explicitly
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            else:
                conn.execute("PRAGMA journal_mode = MEMORY")
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        """Close ALL tracked connections, not just the calling thread's.

        sqlite3.Connection.close() is safe to call from a different
        thread than the one that opened it (we set check_same_thread=False
        at open time). This catches the ThreadPoolExecutor leak path
        where worker threads exit without ever calling close().
        """
        with self._lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                # Connection already dead / interpreter shutting down.
                # Don't propagate — we're tearing things down anyway.
                pass
        # Forget the calling-thread reference so a subsequent call to
        # _get_conn() opens a fresh connection rather than reusing a
        # closed one.
        if hasattr(self._local, "conn"):
            self._local.conn = None

    # -- schema ---------------------------------------------------------------

    def _initialize(self) -> None:
        with self._init_lock, self._lock:
            # Defensive integrity check on existing on-disk DBs.
            # ``sqlite3.DatabaseError: database disk image is malformed``
            # can persist across runs (e.g. when a prior run was killed
            # mid-write or hit a disk hiccup). Once corrupt, EVERY
            # subsequent run crashes — there's no self-healing. Detect
            # corruption up front and quarantine the bad file so the
            # caller gets a fresh DB instead of a recurring crash.
            if self.path != ":memory:" and Path(self.path).exists():
                try:
                    conn = self._get_conn()
                    cur = conn.execute("PRAGMA quick_check(1)")
                    rows = cur.fetchall()
                    ok = bool(rows) and rows[0][0] == "ok"
                except sqlite3.DatabaseError:
                    ok = False
                if not ok:
                    # Close the bad connection before renaming.
                    self.close()
                    import time as _t
                    bak = f"{self.path}.corrupt-{int(_t.time())}"
                    try:
                        Path(self.path).rename(bak)
                        # Also move WAL / SHM siblings if present.
                        for suffix in ("-wal", "-shm"):
                            sib = Path(self.path + suffix)
                            if sib.exists():
                                sib.rename(bak + suffix)
                        logger.warning(
                            "v5 db: existing runtime DB at %s failed "
                            "integrity check; quarantined to %s. "
                            "Continuing with a fresh DB.",
                            self.path, bak,
                        )
                    except OSError as exc:
                        logger.warning(
                            "v5 db: failed to quarantine corrupt DB at "
                            "%s (%s); the next operation may crash. "
                            "Manually move/delete the file and re-run.",
                            self.path, exc,
                        )
            conn = self._get_conn()
            self._with_busy_retry(lambda: conn.executescript(_SCHEMA_SQL))
            self._with_busy_retry(
                lambda: conn.execute(
                    "INSERT OR IGNORE INTO schema_version (id, version) "
                    "VALUES (0, ?)",
                    (_CURRENT_SCHEMA_VERSION,),
                )
            )
            row = conn.execute(
                "SELECT version FROM schema_version WHERE id = 0"
            ).fetchone()
            if row is None:
                raise RuntimeError("schema_version row missing after init")
            else:
                self._migrate(conn, row["version"])

    def _with_busy_retry(self, fn: Any, *, attempts: int = 5) -> Any:
        delay = 0.05
        for idx in range(attempts):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or idx == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return fn()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Walk forward through ``_MIGRATIONS`` to ``_CURRENT_SCHEMA_VERSION``.

        Migrations are idempotent — each one uses ``CREATE TABLE IF NOT
        EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` so a partial upgrade
        (interrupted halfway, then resumed) doesn't error.
        """
        if from_version == _CURRENT_SCHEMA_VERSION:
            return
        if from_version > _CURRENT_SCHEMA_VERSION:
            # Newer schema than this code knows about — refuse rather
            # than downgrade silently. A dev who downgraded their
            # checkout should get a clear failure.
            raise RuntimeError(
                f"runtime.db schema version {from_version} is newer "
                f"than supported version {_CURRENT_SCHEMA_VERSION}"
            )
        version = from_version
        while version < _CURRENT_SCHEMA_VERSION:
            migrator = _MIGRATIONS.get(version)
            if migrator is None:
                raise RuntimeError(
                    f"no migration registered from schema version {version}"
                )
            migrator(conn)
            version += 1
            self._with_busy_retry(
                lambda version=version: conn.execute(
                    "UPDATE schema_version SET version = ? WHERE id = 0",
                    (version,),
                )
            )

    def schema_version(self) -> int:
        row = self.query_one("SELECT version FROM schema_version WHERE id = 0")
        return int(row["version"]) if row else 0

    # -- transaction ----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic write transaction. Use IMMEDIATE to fail fast on contention."""
        conn = self._get_conn()
        if conn.in_transaction:
            yield conn
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    # -- query helpers --------------------------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._get_conn().execute(sql, params)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        return self._get_conn().executemany(sql, seq)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._get_conn().execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._get_conn().execute(sql, params).fetchone()


def _init_lock_for(path: str) -> threading.Lock:
    key = path if path == ":memory:" else str(Path(path).resolve())
    with _INIT_LOCK_GUARD:
        lock = _INIT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INIT_LOCKS[key] = lock
        return lock


__all__ = ["SQLiteRuntimeDB"]
