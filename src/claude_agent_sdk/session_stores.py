"""Session stores for persistent conversation storage."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionStore(ABC):
    """Abstract base class for session stores."""
    
    @abstractmethod
    async def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data."""
        pass
    
    @abstractmethod
    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data."""
        pass
    
    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        pass
    
    @abstractmethod
    async def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        pass


class FileSessionStore(SessionStore):
    """File-based session store using JSON files."""
    
    def __init__(self, directory: str):
        """Initialize the file session store.
        
        Args:
            directory: Directory to store session files.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
    
    async def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data as JSON file."""
        file_path = self.directory / f"{session_id}.json"
        # Use asyncio to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_json, file_path, data)
    
    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to file (synchronous helper)."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from JSON file."""
        file_path = self.directory / f"{session_id}.json"
        if not file_path.exists():
            return None
        
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, self._read_json, file_path)
            return data
        except (json.JSONDecodeError, OSError):
            return None
    
    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON data from file (synchronous helper)."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    async def delete(self, session_id: str) -> None:
        """Delete session file."""
        file_path = self.directory / f"{session_id}.json"
        if file_path.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, os.remove, file_path)
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs from directory."""
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(None, self._list_session_files)
        return sessions
    
    def _list_session_files(self) -> List[str]:
        """List session files (synchronous helper)."""
        sessions = []
        for file_path in self.directory.glob("*.json"):
            session_id = file_path.stem
            sessions.append(session_id)
        return sessions


class RedisSessionStore(SessionStore):
    """Redis-based session store."""
    
    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "agent_session:"):
        """Initialize the Redis session store.
        
        Args:
            url: Redis connection URL.
            prefix: Key prefix for session keys.
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("pip install redis for RedisSessionStore")
        
        self.redis = redis.from_url(url)
        self.prefix = prefix
    
    async def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data to Redis."""
        key = f"{self.prefix}{session_id}"
        json_data = json.dumps(data, ensure_ascii=False)
        await self.redis.set(key, json_data)
    
    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from Redis."""
        key = f"{self.prefix}{session_id}"
        json_data = await self.redis.get(key)
        if json_data is None:
            return None
        return json.loads(json_data)
    
    async def delete(self, session_id: str) -> None:
        """Delete session from Redis."""
        key = f"{self.prefix}{session_id}"
        await self.redis.delete(key)
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs from Redis."""
        pattern = f"{self.prefix}*"
        keys = await self.redis.keys(pattern)
        sessions = [key.decode().removeprefix(self.prefix) for key in keys]
        return sessions


class PostgresSessionStore(SessionStore):
    """PostgreSQL-based session store."""
    
    def __init__(self, dsn: str, table: str = "agent_sessions"):
        """Initialize the PostgreSQL session store.
        
        Args:
            dsn: PostgreSQL connection string.
            table: Table name for storing sessions.
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError("pip install asyncpg for PostgresSessionStore")
        
        self.dsn = dsn
        self.table = table
        self._asyncpg = asyncpg
    
    async def _get_connection(self):
        """Get a database connection."""
        return await self._asyncpg.connect(self.dsn)
    
    async def _ensure_table_exists(self, conn):
        """Ensure the sessions table exists."""
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                session_id TEXT PRIMARY KEY,
                data JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def save(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data to PostgreSQL."""
        conn = await self._get_connection()
        try:
            await self._ensure_table_exists(conn)
            json_data = json.dumps(data, ensure_ascii=False)
            await conn.execute(
                f"INSERT INTO {self.table} (session_id, data, updated_at) "
                f"VALUES ($1, $2::jsonb, NOW()) "
                f"ON CONFLICT (session_id) DO UPDATE SET data = $2::jsonb, updated_at = NOW()",
                session_id, json_data
            )
        finally:
            await conn.close()
    
    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from PostgreSQL."""
        conn = await self._get_connection()
        try:
            await self._ensure_table_exists(conn)
            row = await conn.fetchrow(
                f"SELECT data FROM {self.table} WHERE session_id = $1",
                session_id
            )
            if row is None:
                return None
            return json.loads(row["data"])
        finally:
            await conn.close()
    
    async def delete(self, session_id: str) -> None:
        """Delete session from PostgreSQL."""
        conn = await self._get_connection()
        try:
            await self._ensure_table_exists(conn)
            await conn.execute(
                f"DELETE FROM {self.table} WHERE session_id = $1",
                session_id
            )
        finally:
            await conn.close()
    
    async def list_sessions(self) -> List[str]:
        """List all session IDs from PostgreSQL."""
        conn = await self._get_connection()
        try:
            await self._ensure_table_exists(conn)
            rows = await conn.fetch(f"SELECT session_id FROM {self.table}")
            return [row["session_id"] for row in rows]
        finally:
            await conn.close()