"""T6：storage.py——asyncpg 布局表 + 审计表（pgserver 真实 PostgreSQL）。"""
from __future__ import annotations

import asyncio
import os
import tempfile

import asyncpg
import pgserver
import pytest

from app.storage import Storage

_PG_DIR = os.path.join(tempfile.gettempdir(), "seatlocator_pgtest")


@pytest.fixture(scope="session")
def pg_uri():
    pg = pgserver.get_server(_PG_DIR)  # 复用同一数据目录，session 只 initdb 一次
    return pg.get_uri()


@pytest.fixture
async def storage(pg_uri):
    s = Storage(pg_uri, schema="smartclass_seatlocator")
    await s.connect()
    await s.migrate()
    # 清空表，测试间隔离
    await s._pool.execute("TRUNCATE layouts; TRUNCATE locate_log;")
    yield s
    await s.close()


LAYOUT = {
    "classroom_id": "rm-101",
    "image_width": 1920, "image_height": 1080,
    "rows": 2, "cols": 3,
    "anchors": [{"row": 1, "col": 1, "x": 100.0, "y": 200.0},
                {"row": 1, "col": 2, "x": 400.0, "y": 200.0},
                {"row": 1, "col": 3, "x": 700.0, "y": 200.0},
                {"row": 2, "col": 1, "x": 120.0, "y": 600.0},
                {"row": 2, "col": 2, "x": 430.0, "y": 600.0},
                {"row": 2, "col": 3, "x": 740.0, "y": 600.0}],
}


class TestLayouts:
    async def test_put_get_roundtrip(self, storage):
        created = await storage.put_layout(LAYOUT)
        assert created is True
        got = await storage.get_layout("rm-101")
        assert got == LAYOUT

    async def test_overwrite_returns_false(self, storage):
        await storage.put_layout(LAYOUT)
        changed = dict(LAYOUT, rows=3)
        created = await storage.put_layout(changed)
        assert created is False
        got = await storage.get_layout("rm-101")
        assert got["rows"] == 3

    async def test_get_missing(self, storage):
        assert await storage.get_layout("nope") is None

    async def test_delete(self, storage):
        await storage.put_layout(LAYOUT)
        assert await storage.delete_layout("rm-101") is True
        assert await storage.delete_layout("rm-101") is False
        assert await storage.get_layout("rm-101") is None

    async def test_list_classrooms(self, storage):
        await storage.put_layout(LAYOUT)
        await storage.put_layout(dict(LAYOUT, classroom_id="rm-102"))
        rooms = await storage.list_classrooms()
        assert sorted(rooms) == ["rm-101", "rm-102"]

    async def test_schema_isolation(self, storage, pg_uri):
        """表必须建在 smartclass_seatlocator schema，不污染 public。"""
        conn = await asyncpg.connect(pg_uri,
                                     server_settings={
                                         "search_path": "public"})
        try:
            n = await conn.fetchval(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name='layouts'")
            assert n == 0
            n2 = await conn.fetchval(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='smartclass_seatlocator' "
                "AND table_name IN ('layouts','locate_log')")
            assert n2 == 2
        finally:
            await conn.close()


class TestLocateLog:
    async def test_insert_and_fields(self, storage):
        await storage.put_layout(LAYOUT)
        log_id = await storage.insert_locate_log(
            classroom_id="rm-101",
            image_url="https://files.example.com/p/1?expires=1",
            actor="u-1001",
            persons_found=2,
            faces_unmatched=1,
        )
        assert log_id > 0
        row = await storage._pool.fetchrow(
            "SELECT * FROM locate_log WHERE id=$1", log_id)
        assert row["classroom_id"] == "rm-101"
        assert row["persons_found"] == 2
        assert row["faces_unmatched"] == 1
        assert row["actor"] == "u-1001"
        assert row["created_at"] is not None

    async def test_log_with_unknown_classroom_ok(self, storage):
        """审计记录不依赖布局存在（失败请求也要留痕）。"""
        log_id = await storage.insert_locate_log(
            classroom_id="ghost-room",
            image_url="https://x.example.com/1",
            actor="u-1001",
            persons_found=0,
            faces_unmatched=0,
        )
        assert log_id > 0
