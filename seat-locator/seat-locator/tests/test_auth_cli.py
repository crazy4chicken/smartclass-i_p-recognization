"""T7：config + auth + register-permissions CLI。"""
from __future__ import annotations

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth, config
from app.problems import ProblemError


# ---------------- config ----------------

class TestConfig:
    def test_missing_required_without_dev(self):
        with pytest.raises(SystemExit):
            config.from_env({})

    def test_dev_bypasses_requirements(self):
        cfg = config.from_env({"SEAT_DEV": "true"})
        assert cfg.dev is True
        assert cfg.listen_addr == ":18001"
        assert cfg.schema == "smartclass_seatlocator"

    def test_full_env(self):
        cfg = config.from_env({
            "SEAT_DSN": "postgres://x", "SEAT_DEV": "false",
            "SEAT_TEAMUSERS_URL": "http://tu:8080",
            "SEAT_TEAMUSERS_SERVICE_TOKEN": "svc-tok",
            "SEAT_FACE_BACKEND_URL": "http://face:18000",
            "SEAT_LISTEN_ADDR": ":19001",
            "SEAT_SCALE_FACTOR": "0.5",
            "SEAT_IMAGE_MAX_BYTES": "1000",
        })
        assert cfg.teamusers_audience == "nekostick"
        assert cfg.scale_factor == 0.5
        assert cfg.image_max_bytes == 1000


# ---------------- auth ----------------

class FakeVerifier:
    """通过 verify(token) 返回固定 claims。"""

    def __init__(self, claims=None, fail=False):
        self.claims = claims
        self.fail = fail

    def verify(self, token):
        if self.fail:
            raise RuntimeError("bad token")
        if token == "good":
            return self.claims
        raise RuntimeError("bad token")


class FakeAllow:
    def __init__(self, allow, reason=""):
        self.allow = allow
        self.reason = reason


class FakeClient:
    def __init__(self, verifier, allow_result):
        self.verifier = verifier
        self._allow = allow_result

    def allow(self, claims, permission, resource):
        return self._allow


CLAIMS = "fake-claims-object"


@pytest.fixture
def app_with_dep():
    from teamusers_sdk import Claims
    from datetime import datetime, timezone
    claims = Claims(subject="u-1001", team="", kind="user", perm_ver=1,
                    expiry=datetime.now(timezone.utc).replace(year=9999),
                    audience="nekostick")
    _app = FastAPI()
    from app.problems import install_problem_handler
    install_problem_handler(_app)

    @_app.get("/protected")
    async def protected(
            claims=Depends(auth.require("seat:check:any"))):
        return {"sub": claims.subject}

    return _app, claims


class TestAuth:
    def test_401_no_token(self, app_with_dep):
        _app, _ = app_with_dep
        auth.set_client(FakeClient(FakeVerifier(fail=True),
                                   FakeAllow(True)), dev=False)
        with TestClient(_app) as c:
            r = c.get("/protected")
        assert r.status_code == 401

    def test_401_bad_token(self, app_with_dep):
        _app, _ = app_with_dep
        auth.set_client(FakeClient(FakeVerifier(fail=True),
                                   FakeAllow(True)), dev=False)
        with TestClient(_app) as c:
            r = c.get("/protected", headers={"Authorization": "Bearer bad"})
        assert r.status_code == 401

    def test_403_denied(self, app_with_dep):
        _app, _ = app_with_dep
        auth.set_client(FakeClient(FakeVerifier(CLAIMS),
                                   FakeAllow(False, "no grant")),
                        dev=False)
        with TestClient(_app) as c:
            r = c.get("/protected", headers={"Authorization": "Bearer good"})
        assert r.status_code == 403

    def test_200_allowed(self, app_with_dep):
        _app, claims = app_with_dep
        verifier = FakeVerifier(claims)
        auth.set_client(FakeClient(verifier, FakeAllow(True)), dev=False)
        with TestClient(_app) as c:
            r = c.get("/protected", headers={"Authorization": "Bearer good"})
        assert r.status_code == 200
        assert r.json()["sub"] == "u-1001"

    def test_dev_mode_synthetic_claims(self, app_with_dep):
        _app, _ = app_with_dep
        auth.set_client(None, dev=True)
        with TestClient(_app) as c:
            r = c.get("/protected")
        assert r.status_code == 200
        assert r.json()["sub"] == "dev-local"

    def test_no_client_503(self, app_with_dep):
        _app, _ = app_with_dep
        auth.set_client(None, dev=False)
        with TestClient(_app) as c:
            r = c.get("/protected", headers={"Authorization": "Bearer x"})
        assert r.status_code == 503

    def test_problem_error_shape(self):
        p = ProblemError(403, "permission_denied", "x", classroom_id="r1")
        d = p.to_dict()
        assert d["status"] == 403 and d["title"] == "permission_denied"
        assert d["classroom_id"] == "r1"
        assert d["type"].startswith("https://seatlocator.problems/")


# ---------------- register-permissions CLI ----------------

class TestRegisterPermissions:
    def test_registers_all_keys_with_contract(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.path,
                         dict(request.headers),
                         __import__("json").loads(request.content)))
            return httpx.Response(201, json={"key": "ok"})

        import asyncio
        ok, fail = asyncio.run(_run_with_transport(handler))
        assert (ok, fail) == (3, 0)
        assert len(seen) == 3
        path, headers, body = seen[0]
        assert path == "/permissions/"
        assert headers["authorization"] == "Bearer admin-tok"
        assert body["registered_by"] == "nsc-seatlocator"
        assert body["key"] == "seat:check:any"
        assert "description" in body

    def test_partial_failure_reported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = __import__("json").loads(request.content)
            if body["key"] == "seat:read:any":
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={})

        import asyncio
        ok, fail = asyncio.run(_run_with_transport(handler))
        assert (ok, fail) == (2, 1)


async def _run_with_transport(handler):
    import app.cli as cli
    from app import auth as _auth

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, **kw):
            kw.setdefault("timeout", 5)
            super().__init__(transport=transport, **kw)

    orig = httpx.AsyncClient
    httpx.AsyncClient = _Client
    try:
        return await cli.register_permissions(
            "http://tu:8080/", "admin-tok",
            keys=_auth.PERMISSION_CATALOG)
    finally:
        httpx.AsyncClient = orig
