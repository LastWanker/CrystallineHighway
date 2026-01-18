"""存储层。"""

from __future__ import annotations

from ..config import MemoryConfig
from .in_memory import InMemoryStore
from .sqlite_store import SQLiteStore


def create_store(config: MemoryConfig) -> InMemoryStore | SQLiteStore:
    if config.storage_backend == "sqlite":
        return SQLiteStore(config.storage_path)
    return InMemoryStore()
