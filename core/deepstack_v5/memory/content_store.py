"""ContentStore — FTS5-backed knowledge index for large tool outputs.

Phase 2 of the context-mode port. When the cc → v5 event bridge sees a
``tool_result`` whose body is too large to inline into a v5 event row
(default threshold: 4 KB), we want to keep the *full* content
queryable from later turns of the same run — without bloating the
events table that the watcher tails.

Design:

* Independent SQLite file at ``~/.llm_dealer/v5/content-<hash>.db``.
  Separate from the v5 runtime.db on purpose: the runtime db uses
  ``BEGIN IMMEDIATE`` for every event publish, and a chunked FTS5
  insert would block dispatcher progress. A second file is one extra
  open-fd and zero contention.
* Workspace hash is ``sha256(getuid:git_root_or_cwd)[:8]`` so two devs
  on the same shared home (e.g. a remote dev box) don't collide, and
  ``git worktree`` checkouts each get their own index. Roll-your-own
  hash matches what context-mode uses; no library dependency.
* Markdown-aware chunking: split by H2/H3, preserve fenced code
  blocks intact, fall back to a 2 KB character window at line breaks
  for unstructured input.
* FTS5 with porter+unicode61 tokeniser for BM25 ranking on prose;
  works well enough for code too because the porter stemmer leaves
  identifiers alone.
  CJK substring search is not guaranteed by this tokenizer; callers
  that need CJK substring semantics should add a separate index.
* Background writer thread: ``enqueue()`` is the hot path called from
  the cc QueryEngine thread; it must never block. A bounded
  ``queue.Queue`` plus a single daemon thread drains writes in
  batches. Queue overflow → drop the body (preview still survives in
  the v5 event row) and increment ``dropped_writes``. The user can
  see this via Phase 4's ``ccx watch stats``.
* TTL purge: ``deletable_after_ms`` is set by callers when a run
  terminates; callers that construct a ContentStore should invoke
  ``purge_expired()`` on their own startup cadence.

This module is *not* wired into the v5 runtime by default. Construct a
ContentStore explicitly with an explicit ``db_path`` when a bridge or
tooling layer wants durable large-output search.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Per-chunk target size in characters. Markdown headings produce
# variable-size chunks; this caps the largest non-heading chunk.
_CHUNK_TARGET_CHARS = 2_000

# Queue capacity for the background writer. Sized so a burst of large
# tool results (e.g. cc opening 20 files in parallel during a planning
# step) doesn't drop — but small enough that runaway content doesn't
# pin tens of MB in memory.
_QUEUE_MAXSIZE = 256

# Background writer batches commits — these knobs control how
# aggressive that batching is.
_BATCH_INTERVAL_S = 0.25
_BATCH_MAX_ROWS = 32


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    label TEXT,
    content_type TEXT,
    total_bytes INTEGER,
    chunk_count INTEGER,
    created_at_ms INTEGER NOT NULL,
    deletable_after_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sources_run ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_sources_expiry
    ON sources(deletable_after_ms)
    WHERE deletable_after_ms IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, ord);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, body) VALUES (new.chunk_id, new.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES('delete', old.chunk_id, old.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES('delete', old.chunk_id, old.body);
    INSERT INTO chunks_fts(rowid, body) VALUES (new.chunk_id, new.body);
END;
"""


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ChunkHit:
    source_id: str
    label: str | None
    body: str
    ord: int
    score: float  # FTS5 bm25() — lower is better.


@dataclass(slots=True)
class ContentStoreStats:
    total_sources: int
    total_chunks: int
    dropped_writes: int
    queued_writes: int
    db_bytes: int


@dataclass(slots=True)
class _WriteRequest:
    """Internal: one item the background writer pulls from the queue."""

    source_id: str
    run_id: str
    label: str | None
    content_type: str
    body: str
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


# --------------------------------------------------------------------------- #
# Workspace hash
# --------------------------------------------------------------------------- #

def compute_workspace_hash(cwd: str | Path | None = None) -> str:
    """Compute the 8-char SHA256 hash identifying this workspace.

    Inputs blended into the hash:

    * ``os.getuid()`` — prevents collisions when multiple users share
      ``$HOME`` (remote dev boxes, CI runners with persistent caches).
    * ``git rev-parse --show-toplevel`` if inside a git repo, else the
      resolved cwd — ensures each ``git worktree`` checkout gets its
      own index, and non-git working directories still get a stable
      identifier.

    Falls back to the raw cwd string if every probe fails (e.g.
    extremely permission-restricted environments). The hash is
    deterministic per ``(uid, root)`` pair, so a process that opens
    the same workspace twice reuses the same DB file.
    """
    target = Path(cwd).resolve() if cwd else Path.cwd()
    root: str
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=target if target.is_dir() else target.parent,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            root = proc.stdout.strip()
        else:
            root = str(target)
    except (subprocess.SubprocessError, OSError):
        root = str(target)

    try:
        uid = os.getuid()  # POSIX only
    except AttributeError:
        # Windows has no getuid; use the username as a coarse equivalent.
        uid = os.environ.get("USERNAME", "unknown")

    digest = hashlib.sha256(f"{uid}:{root}".encode("utf-8")).hexdigest()
    return digest[:8]


def default_db_path(cwd: str | Path | None = None) -> Path:
    """Compute the canonical DB path for ``cwd``.

    Layout: ``~/.llm_dealer/v5/content-<hash>.db``. The parent
    directory is created lazily by ContentStore.__init__.
    """
    h = compute_workspace_hash(cwd)
    home = Path(os.path.expanduser("~"))
    return home / ".llm_dealer" / "v5" / f"content-{h}.db"


# --------------------------------------------------------------------------- #
# Markdown-aware chunking
# --------------------------------------------------------------------------- #

# Match level-2 or level-3 markdown heading lines. We don't split on H1
# because cc rarely produces multi-H1 documents and using H1 as a
# splitter loses high-level grouping.
_HEADING_RE = re.compile(r"^(##|###)\s+.*$", re.MULTILINE)

# Match fenced code blocks (``` ... ```) so we can keep them intact when
# windowing falls back to character-based chunking.
_FENCE_OPEN_RE = re.compile(r"^```")


def chunk_markdown(
    content: str, *, target_chars: int = _CHUNK_TARGET_CHARS
) -> list[str]:
    """Split ``content`` into chunks suitable for FTS5 indexing.

    Priority:

    1. If the text has ``##``/``###`` headings, split there first.
       Each chunk starts with its heading line so a search hit can be
       attributed back to a section.
    2. Within each heading section (or for unstructured input), if the
       chunk exceeds ``target_chars * 2``, fall back to character
       windowing at line boundaries — but never split *through* a
       fenced code block, which would corrupt the markdown rendering.

    Returns a list of non-empty chunks. Empty input → empty list.
    """
    if not content or not content.strip():
        return []

    heading_chunks = _split_on_headings(content)
    out: list[str] = []
    for sec in heading_chunks:
        if len(sec) <= target_chars * 2:
            out.append(sec)
            continue
        out.extend(_split_by_window(sec, target_chars=target_chars))
    # Filter out chunks that are pure whitespace (can happen when input
    # starts with a heading and the first split is empty before it).
    return [c for c in out if c.strip()]


def _split_on_headings(content: str) -> list[str]:
    matches = list(_HEADING_RE.finditer(content))
    if not matches:
        return [content]
    sections: list[str] = []
    if matches[0].start() > 0:
        prefix = content[: matches[0].start()]
        if prefix.strip():
            sections.append(prefix)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append(content[m.start():end])
    return sections


def _split_by_window(
    content: str, *, target_chars: int
) -> Iterator[str]:
    """Yield ~target_chars windows that respect line and code-fence boundaries.

    Algorithm: scan lines, track whether we're inside a fenced code
    block, accumulate until window reaches target_chars AND we're not
    inside a fence. Then flush.
    """
    lines = content.splitlines(keepends=True)
    buf: list[str] = []
    cur_chars = 0
    in_fence = False
    for line in lines:
        if _FENCE_OPEN_RE.match(line):
            in_fence = not in_fence
        buf.append(line)
        cur_chars += len(line)
        if cur_chars >= target_chars and not in_fence:
            yield "".join(buf)
            buf = []
            cur_chars = 0
    if buf:
        yield "".join(buf)


# --------------------------------------------------------------------------- #
# ContentStore
# --------------------------------------------------------------------------- #

class ContentStore:
    """Thread-safe FTS5 content index with background-writer support.

    Construction is cheap (opens one SQLite connection, runs schema
    DDL). The background writer thread starts lazily on the first
    :meth:`enqueue` so unit tests that only use :meth:`index` /
    :meth:`search` never see a daemon thread.

    Connections are per-thread via ``threading.local`` — sqlite3
    forbids cross-thread connection sharing, and we have at least
    three concurrent users in practice (cc QueryEngine thread,
    background writer thread, ad-hoc searcher).
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if db_path is None:
            db_path = default_db_path(cwd)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_conns: list[sqlite3.Connection] = []

        # Background writer state — lazily started on first enqueue.
        self._queue: queue.Queue[_WriteRequest | None] = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._writer_thread: threading.Thread | None = None
        self._writer_started = False
        self._dropped_writes = 0
        self._writer_stopping = threading.Event()

        self._initialize_schema()

    # -- connection management -----------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._local.conn = conn
        with self._lock:
            self._all_conns.append(conn)
        return conn

    def _initialize_schema(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        """Stop the background writer and close every tracked connection."""
        self.flush()
        if self._writer_thread is not None:
            self._writer_stopping.set()
            try:
                self._queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None
            self._writer_started = False
        with self._lock:
            conns = list(self._all_conns)
            self._all_conns.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if hasattr(self._local, "conn"):
            self._local.conn = None

    # -- sync API ------------------------------------------------------------

    def index(
        self,
        run_id: str,
        source_id: str,
        content: str,
        *,
        label: str | None = None,
        content_type: str = "prose",
    ) -> int:
        """Synchronously chunk and index ``content``. Returns chunk count.

        If ``source_id`` already exists, its previous chunks are
        deleted and replaced — calling twice with the same source_id
        is a valid upsert pattern. This matters for cc tool runs that
        retry: the second result overwrites the first cleanly.
        """
        chunks = chunk_markdown(content)
        if not chunks:
            return 0
        return self._write_chunks(
            run_id=run_id,
            source_id=source_id,
            label=label,
            content_type=content_type,
            chunks=chunks,
            total_bytes=len(content.encode("utf-8")),
        )

    def _write_chunks(
        self,
        *,
        run_id: str,
        source_id: str,
        label: str | None,
        content_type: str,
        chunks: list[str],
        total_bytes: int,
    ) -> int:
        conn = self._conn()
        now = int(time.time() * 1000)
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Upsert: clear any old chunks for this source_id.
            conn.execute(
                "DELETE FROM chunks WHERE source_id = ?", (source_id,)
            )
            conn.execute(
                """
                INSERT INTO sources (
                    source_id, run_id, label, content_type, total_bytes,
                    chunk_count, created_at_ms, deletable_after_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(source_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    label = excluded.label,
                    content_type = excluded.content_type,
                    total_bytes = excluded.total_bytes,
                    chunk_count = excluded.chunk_count,
                    created_at_ms = excluded.created_at_ms
                """,
                (
                    source_id, run_id, label, content_type, total_bytes,
                    len(chunks), now,
                ),
            )
            conn.executemany(
                "INSERT INTO chunks (source_id, ord, body) VALUES (?, ?, ?)",
                [(source_id, i, body) for i, body in enumerate(chunks)],
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        run_id: str | None = None,
        top_k: int = 10,
    ) -> list[ChunkHit]:
        """BM25-ranked full-text search.

        ``run_id`` filters to a single run when supplied (the common
        case for "look up something my prior step learned"); otherwise
        searches across every indexed run in this workspace.

        Empty query → empty result. ``top_k`` is clamped to 1..100.
        """
        if not query or not query.strip():
            return []
        top_k = max(1, min(int(top_k), 100))
        sanitized = _sanitize_fts5_query(query)
        if not sanitized:
            return []
        conn = self._conn()
        params: list[object] = [sanitized]
        sql = """
            SELECT
                c.source_id AS source_id,
                c.ord AS ord,
                c.body AS body,
                s.label AS label,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.rowid
            JOIN sources s ON s.source_id = c.source_id
            WHERE chunks_fts MATCH ?
        """
        if run_id is not None:
            sql += " AND s.run_id = ?"
            params.append(run_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # FTS5 returns "fts5: syntax error" for malformed queries.
            logger.debug("ContentStore.search: bad fts query %r: %s",
                         sanitized, exc)
            return []
        return [
            ChunkHit(
                source_id=r["source_id"],
                label=r["label"],
                body=r["body"],
                ord=int(r["ord"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    def fetch(self, source_id: str) -> str:
        """Reassemble the full content for ``source_id``.

        Joins chunks in ``ord`` order. Returns empty string for an
        unknown source. Reassembly is based on stored text chunks and
        is intended to recover searchable content, not byte-exact
        original input.
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT body FROM chunks
            WHERE source_id = ? ORDER BY ord ASC
            """,
            (source_id,),
        ).fetchall()
        return "".join(r["body"] for r in rows) if rows else ""

    def purge(self, run_id: str | None = None) -> int:
        """Delete content. Returns rows of `sources` removed.

        ``run_id`` scopes the delete to a single run; ``None`` purges
        everything in this workspace. Cascade via FK takes care of
        chunks (and FTS5 via trigger).
        """
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if run_id is None:
                cur = conn.execute("DELETE FROM sources")
            else:
                cur = conn.execute(
                    "DELETE FROM sources WHERE run_id = ?", (run_id,)
                )
            removed = cur.rowcount
            conn.execute("COMMIT")
        except sqlite3.Error:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        return removed

    def purge_expired(self, *, now_ms: int | None = None) -> int:
        """Delete sources whose ``deletable_after_ms`` has passed."""
        threshold = int(time.time() * 1000) if now_ms is None else now_ms
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """
                DELETE FROM sources
                WHERE deletable_after_ms IS NOT NULL
                  AND deletable_after_ms <= ?
                """,
                (threshold,),
            )
            removed = cur.rowcount
            conn.execute("COMMIT")
        except sqlite3.Error:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        return removed

    def mark_run_deletable(
        self, run_id: str, *, retain_for_ms: int = 7 * 24 * 60 * 60 * 1000,
    ) -> int:
        """Stamp ``deletable_after_ms = now + retain_for_ms`` on a run's sources.

        Called from runtime shutdown / run completion. Updates rather
        than deletes so ``purge_expired`` (cheap to run at startup)
        does the actual eviction later. Default retention: 7 days.
        """
        target = int(time.time() * 1000) + int(retain_for_ms)
        conn = self._conn()
        cur = conn.execute(
            "UPDATE sources SET deletable_after_ms = ? WHERE run_id = ?",
            (target, run_id),
        )
        return cur.rowcount

    # -- async API (background writer) ---------------------------------------

    def enqueue(
        self,
        run_id: str,
        source_id: str,
        content: str,
        *,
        label: str | None = None,
        content_type: str = "prose",
    ) -> bool:
        """Submit a write to the background queue. Returns False if dropped.

        The hot path: cc QueryEngine thread is mid-LLM-loop and must
        not block on a multi-KB FTS5 insert. If the queue is full the
        write is dropped — the in-event preview is still on disk,
        only the searchable copy goes missing.
        """
        if not content:
            return False
        self._ensure_writer_started()
        req = _WriteRequest(
            source_id=source_id,
            run_id=run_id,
            label=label,
            content_type=content_type,
            body=content,
        )
        try:
            self._queue.put_nowait(req)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_writes += 1
            return False

    def flush(self, *, timeout_s: float = 10.0) -> None:
        """Block until the background writer drains the queue.

        Tests use this to wait for an :meth:`enqueue` to be visible
        through :meth:`search`. Production code typically only calls
        flush during shutdown via :meth:`close`.
        """
        if not self._writer_started:
            return
        deadline = time.time() + timeout_s
        while self._queue.unfinished_tasks > 0:
            if time.time() > deadline:
                break
            time.sleep(0.01)

    def _ensure_writer_started(self) -> None:
        if self._writer_started:
            return
        with self._lock:
            if self._writer_started:
                return
            self._writer_stopping.clear()
            t = threading.Thread(
                target=self._writer_loop,
                name="ContentStore-writer",
                daemon=True,
            )
            t.start()
            self._writer_thread = t
            self._writer_started = True

    def _writer_loop(self) -> None:
        # task_done() is intentionally NOT called when an item is
        # dequeued — only after its batch has been persisted. This
        # makes ``queue.unfinished_tasks`` an accurate "pending writes"
        # counter, which :meth:`flush` relies on. If we ack on dequeue
        # a flush could return before the in-flight batch's
        # ``_drain_batch`` commit completed.
        batch: list[_WriteRequest] = []
        last_flush = time.time()
        while not self._writer_stopping.is_set():
            timeout = max(0.001, _BATCH_INTERVAL_S - (time.time() - last_flush))
            try:
                item = self._queue.get(timeout=timeout if batch else None)
            except queue.Empty:
                if batch:
                    self._drain_and_ack(batch)
                    batch = []
                    last_flush = time.time()
                continue
            if item is None:
                # Sentinel from close(); drain whatever we have, ack
                # the sentinel itself, and exit.
                if batch:
                    self._drain_and_ack(batch)
                self._queue.task_done()
                break
            batch.append(item)
            if (
                len(batch) >= _BATCH_MAX_ROWS
                or (time.time() - last_flush) >= _BATCH_INTERVAL_S
            ):
                self._drain_and_ack(batch)
                batch = []
                last_flush = time.time()

    def _drain_and_ack(self, batch: list[_WriteRequest]) -> None:
        """Persist a batch then mark every item ``task_done``.

        Acking is done unconditionally — even if the persist failed —
        so a poisoned batch doesn't pin ``unfinished_tasks`` forever
        and starve :meth:`flush`.
        """
        try:
            self._drain_batch(batch)
        finally:
            for _ in batch:
                self._queue.task_done()

    def _drain_batch(self, batch: list[_WriteRequest]) -> None:
        for req in batch:
            try:
                chunks = chunk_markdown(req.body)
                if not chunks:
                    continue
                self._write_chunks(
                    run_id=req.run_id,
                    source_id=req.source_id,
                    label=req.label,
                    content_type=req.content_type,
                    chunks=chunks,
                    total_bytes=len(req.body.encode("utf-8")),
                )
            except Exception:
                # Background writer must never crash — log and continue.
                logger.exception(
                    "ContentStore writer: failed to persist source_id=%r",
                    req.source_id,
                )

    # -- stats / introspection ------------------------------------------------

    def stats(self) -> ContentStoreStats:
        conn = self._conn()
        sources = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        size = 0
        try:
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            size = int(page_count) * int(page_size)
        except (sqlite3.Error, TypeError):
            pass
        with self._lock:
            dropped = self._dropped_writes
        return ContentStoreStats(
            total_sources=int(sources["n"]) if sources else 0,
            total_chunks=int(chunks["n"]) if chunks else 0,
            dropped_writes=dropped,
            queued_writes=self._queue.qsize() if self._writer_started else 0,
            db_bytes=size,
        )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

# FTS5 treats ``"``, ``(``, etc. as syntax; user-supplied queries can be
# raw natural-language strings that would otherwise raise "fts5: syntax
# error". Strip the characters FTS5 would interpret specially and
# collapse whitespace.
_FTS_SAFE_CHARS = re.compile(r"[^\w\s一-鿿-]+")


def _sanitize_fts5_query(query: str) -> str:
    cleaned = _FTS_SAFE_CHARS.sub(" ", query)
    tokens = cleaned.split()
    if not tokens:
        return ""
    # Quote every token so FTS5 keywords such as NOT/OR and hyphenated
    # fragments are searched literally instead of parsed as operators.
    return " ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"'
                    for token in tokens)


__all__ = [
    "ChunkHit",
    "ContentStore",
    "ContentStoreStats",
    "chunk_markdown",
    "compute_workspace_hash",
    "default_db_path",
]
