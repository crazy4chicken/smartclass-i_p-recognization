"""PostgreSQL 存储：布局表 + locate 审计表。

约定（对齐 dispatchub / face-backend）：
- schema 隔离：所有表建在专用 schema（默认 smartclass_seatlocator）；
- 连接池 application_name=nsc-seatlocator；DSN 自带 search_path 时尊重 DSN；
- 布局本体以 jsonb 整体存取（结构校验由 app.seating.load_layout 负责）；
- locate_log 只存结构化数据（URL + 计数 + actor），绝不存图像字节。
"""
from __future__ import annotations

import json
from typing import Any

import asyncpg

DEFAULT_SCHEMA = "smartclass_seatlocator"
APP_NAME = "nsc-seatlocator"

_MIGRATION = """
CREATE TABLE IF NOT EXISTS layouts (
    classroom_id  TEXT PRIMARY KEY,
    data          JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS locate_log (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    classroom_id    TEXT,
    image_url       TEXT,
    actor           TEXT,
    persons_found   INTEGER NOT NULL DEFAULT 0,
    faces_unmatched INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_locate_log_created
    ON locate_log (created_at DESC);
"""


class Storage:
    """asyncpg 池封装：布局 CRUD + 审计追加。"""

    def __init__(self, dsn: str, schema: str = DEFAULT_SCHEMA):
        self.dsn = dsn
        self.schema = schema
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Storage 未连接：先调用 connect()")
        return self._pool

    async def connect(self) -> None:
        server_settings: dict[str, str] = {
            "application_name": APP_NAME,
        }
        # DSN 未显式指定 search_path 时，由池统一固定到专用 schema
        dsn_lower = self.dsn.lower()
        if "search_path" not in dsn_lower and "options=" not in dsn_lower:
            server_settings["search_path"] = self.schema
        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=1, max_size=8,
            server_settings=server_settings)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            await conn.execute(f"SET search_path TO {self.schema}")
            await conn.execute(_MIGRATION)

    # ---------- layouts ----------

    async def get_layout(self, classroom_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM layouts WHERE classroom_id=$1",
                classroom_id)
            if row is None:
                return None
            return json.loads(row["data"])

    async def put_layout(self, data: dict[str, Any]) -> bool:
        """新建 → True；覆盖 → False。"""
        classroom_id = str(data["classroom_id"])
        async with self.pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT 1 FROM layouts WHERE classroom_id=$1", classroom_id)
            await conn.execute(
                """INSERT INTO layouts (classroom_id, data)
                   VALUES ($1, $2::jsonb)
                   ON CONFLICT (classroom_id)
                   DO UPDATE SET data = EXCLUDED.data,
                                 updated_at = now()""",
                classroom_id, json.dumps(data))
            return existing is None

    async def delete_layout(self, classroom_id: str) -> bool:
        async with self.pool.acquire() as conn:
            return await conn.execute(
                "DELETE FROM layouts WHERE classroom_id=$1",
                classroom_id) != "DELETE 0"

    async def list_classrooms(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT classroom_id FROM layouts ORDER BY classroom_id")
            return [r["classroom_id"] for r in rows]

    # ---------- locate_log ----------

    async def insert_locate_log(self, *, classroom_id: str,
                                image_url: str, actor: str | None,
                                persons_found: int,
                                faces_unmatched: int) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """INSERT INTO locate_log
                       (classroom_id, image_url, actor,
                        persons_found, faces_unmatched)
                   VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                classroom_id, image_url, actor,
                persons_found, faces_unmatched)
