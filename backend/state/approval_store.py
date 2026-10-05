"""Durable business approval ledger, separate from LangGraph checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import weakref
from threading import RLock
from typing import Callable

from api.schemas import ChatResumeRequest, ChatResumeResponse
from state.checkpoints import ApprovalSnapshot


STORE_VERSION = 1


def decision_fingerprint(request: ChatResumeRequest) -> str:
    """Identify a retried decision without persisting its bearer credential."""
    payload = request.model_dump(mode="json", exclude={"resume_token"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ApprovalStore:
    """One-process SQLite ledger; uniqueness is the submission boundary."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = RLock()
        self._closed = False
        self._conn: sqlite3.Connection | None = None
        try:
            if str(path) != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
            with self._conn:
                self._conn.execute("PRAGMA busy_timeout=10000")
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS workflow_store_meta "
                    "(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)"
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO workflow_store_meta VALUES (1, ?)",
                    (STORE_VERSION,),
                )
                version = self._conn.execute(
                    "SELECT version FROM workflow_store_meta WHERE id=1"
                ).fetchone()[0]
                if version != STORE_VERSION:
                    raise RuntimeError("售后存储 Schema 版本不受支持。")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS approvals ("
                    "workflow_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                    "snapshot_json TEXT NOT NULL, terminal INTEGER NOT NULL DEFAULT 0, "
                    "response_json TEXT, decision_fingerprint TEXT, updated_at TEXT NOT NULL)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS submissions ("
                    "idempotency_key TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, "
                    "created_at TEXT NOT NULL)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS approval_decisions ("
                    "workflow_id TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                    "reviewer_id TEXT NOT NULL, reviewer_role TEXT NOT NULL, "
                    "decision TEXT NOT NULL, recorded_at TEXT NOT NULL, "
                    "PRIMARY KEY(workflow_id, fingerprint))"
                )
            self._finalizer = weakref.finalize(self, self._conn.close)
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            if self._conn is not None:
                self._conn.close()
            self._closed = True
            raise RuntimeError("售后审批持久化存储初始化失败。") from exc

    def _connection(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise RuntimeError("售后审批存储已关闭。")
        return self._conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, snapshot: ApprovalSnapshot) -> None:
        with self._lock:
            conn = self._connection()
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO approvals "
                    "(workflow_id, session_id, snapshot_json, updated_at) VALUES (?, ?, ?, ?)",
                    (snapshot.workflow_id, snapshot.session_id,
                     snapshot.model_dump_json(), self._now()),
                )
                existing = self.get(snapshot.session_id, snapshot.workflow_id)
                if existing != snapshot:
                    raise RuntimeError("售后申请关联发生冲突，不能覆盖已有记录。")

    def get(self, session_id: str, workflow_id: str) -> ApprovalSnapshot | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT snapshot_json FROM approvals WHERE workflow_id=? AND session_id=?",
                (workflow_id, session_id),
            ).fetchone()
            if row is None:
                return None
            return ApprovalSnapshot.model_validate_json(row[0])

    def outcome(self, workflow_id: str) -> tuple[ChatResumeResponse | None, bool, str | None]:
        with self._lock:
            row = self._connection().execute(
                "SELECT response_json, terminal, decision_fingerprint "
                "FROM approvals WHERE workflow_id=?", (workflow_id,),
            ).fetchone()
            if row is None:
                return None, False, None
            response = ChatResumeResponse.model_validate_json(row[0]) if row[0] else None
            return response, bool(row[1]), row[2]

    def finish(self, request: ChatResumeRequest, response: ChatResumeResponse) -> None:
        """Atomically save the business outcome and its minimal audit record."""
        with self._lock:
            conn = self._connection()
            with conn:
                self._write_result(conn, request, response)

    def _write_result(self, conn: sqlite3.Connection, request: ChatResumeRequest,
                      response: ChatResumeResponse) -> None:
        fingerprint = decision_fingerprint(request)
        cursor = conn.execute(
            "UPDATE approvals SET terminal=?, response_json=?, "
            "decision_fingerprint=?, updated_at=? "
            "WHERE workflow_id=? AND session_id=? AND terminal=0",
            (int(response.status != "paused"), response.model_dump_json(),
             fingerprint, self._now(), request.workflow_id, request.session_id),
        )
        if cursor.rowcount != 1:
            saved, terminal, saved_fingerprint = self.outcome(request.workflow_id)
            if not terminal or saved is None or saved_fingerprint != fingerprint:
                raise RuntimeError("售后审批终态发生冲突。")
            return
        conn.execute(
            "INSERT OR IGNORE INTO approval_decisions VALUES (?, ?, ?, ?, ?, ?)",
            (request.workflow_id, fingerprint, request.reviewer_id,
             request.reviewer_role, request.decision, self._now()),
        )

    def submit(self, request: ChatResumeRequest, idempotency_key: str,
               response_factory: Callable[[str, bool], ChatResumeResponse]) -> ChatResumeResponse:
        """Commit simulated submission AND its terminal outcome in one transaction."""
        request_id = "asr-" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]
        with self._lock:
            conn = self._connection()
            with conn:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO submissions VALUES (?, ?, ?)",
                    (idempotency_key, request_id, self._now()),
                )
                existing = conn.execute(
                    "SELECT request_id FROM submissions WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                response = response_factory(existing[0], cursor.rowcount == 0)
                self._write_result(conn, request, response)
                return response

    def submission_count(self) -> int:
        with self._lock:
            return self._connection().execute("SELECT count(*) FROM submissions").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._finalizer()
                self._closed = True
