"""Session store implementations.
Created: 2026-05-27 23:00 CST
"""
from __future__ import annotations
from typing import Any

class FileSessionStore:
    def __init__(self, path: str = "/tmp/sessions") -> None:
        self._path = path

class RedisSessionStore:
    def __init__(self, url: str = "redis://localhost:6379") -> None:
        self._url = url

class PostgresSessionStore:
    def __init__(self, dsn: str = "") -> None:
        self._dsn = dsn
