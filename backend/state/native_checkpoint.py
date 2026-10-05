"""Native graph persistence and bounded, process-local execution coordination."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterator

from langgraph.checkpoint.memory import InMemorySaver

from state.approval_store import ApprovalStore


class WorkflowPersistence:
    """Graph state and business state have separate persistence responsibilities."""

    def __init__(self, checkpointer: Any, approvals: ApprovalStore,
                 connection: sqlite3.Connection | None = None) -> None:
        self.checkpointer = checkpointer
        self.approvals = approvals
        self._connection = connection
        self._guard = RLock()
        self._locks: dict[str, tuple[RLock, int]] = {}
        self._closed = False

    @classmethod
    def memory(cls) -> "WorkflowPersistence":
        """Explicit ephemeral resources for embedding the agent and offline tests."""
        return cls(InMemorySaver(), ApprovalStore())

    @classmethod
    def sqlite(cls, checkpoint_path: Path | str, approval_path: Path | str) -> "WorkflowPersistence":
        from langgraph.checkpoint.sqlite import SqliteSaver

        connection = None
        approvals = None
        try:
            checkpoint_path, approval_path = Path(checkpoint_path).resolve(), Path(approval_path).resolve()
            if checkpoint_path == approval_path:
                raise RuntimeError("图快照与审批存储路径必须不同。")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(checkpoint_path), timeout=10, check_same_thread=False)
            connection.execute("PRAGMA busy_timeout=10000")
            saver = SqliteSaver(connection)
            saver.setup()
            approvals = ApprovalStore(approval_path)
            return cls(saver, approvals, connection)
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            if connection is not None:
                connection.close()
            if approvals is not None:
                approvals.close()
            raise RuntimeError("售后工作流持久化初始化失败。") from exc

    @contextmanager
    def execution(self, thread_id: str) -> Iterator[None]:
        """Serialize one thread across its state check and graph execution."""
        with self._guard:
            if self._closed:
                raise RuntimeError("售后工作流存储已关闭。")
            lock, users = self._locks.get(thread_id, (RLock(), 0))
            self._locks[thread_id] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._guard:
                _, users = self._locks[thread_id]
                if users == 1:
                    del self._locks[thread_id]
                else:
                    self._locks[thread_id] = (lock, users - 1)

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            if self._locks:
                raise RuntimeError("售后工作流仍在执行，不能关闭存储。")
            self.approvals.close()
            if self._connection is not None:
                self._connection.close()
            self._closed = True
