"""T4：faceclient.py——face-backend /recognize 客户端（协议 + HTTP + 桩）。"""
from __future__ import annotations

import pytest

from app.faceclient import FaceClient, FaceResult, HttpFaceClient, StubFaceClient


def _recognize_response(faces):
    return {
        "faces_found": len(faces),
        "faces": [
            {
                "bbox": f["bbox"],
                "det_score": f.get("det_score", 0.9),
                "matched": f["matched"],
                "similarity": f.get("similarity", 0.7),
                "user_id": f["user_id"],
            }
            for f in faces
        ],
    }


class TestHttpFaceClient:
    async def test_passes_bearer_and_multipart(self):
        import httpx
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("Authorization")
            captured["content_type"] = request.headers.get("Content-Type")
            captured["body"] = request.read()
            return httpx.Response(
                200, json=_recognize_response([
                    {"bbox": [10, 20, 30, 40], "matched": True,
                     "user_id": "u-1001", "similarity": 0.71},
                ]))

        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as client:
            fc = HttpFaceClient("http://face:18000", client=client)
            results = await fc.recognize(b"\x89PNGfake",
                                         bearer="tok-abc")

        assert captured["auth"] == "Bearer tok-abc"
        assert "multipart/form-data" in captured["content_type"]
        assert b"\x89PNGfake" in captured["body"]
        assert results == [FaceResult(bbox=(10.0, 20.0, 30.0, 40.0),
                                      matched=True, user_id="u-1001",
                                      similarity=0.71, det_score=0.9)]

    async def test_no_faces(self):
        import httpx

        def handler(request):
            return httpx.Response(200, json={"faces_found": 0, "faces": []})

        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as client:
            fc = HttpFaceClient("http://face:18000", client=client)
            results = await fc.recognize(b"img", bearer="t")
        assert results == []

    async def test_unmatched_face(self):
        import httpx

        def handler(request):
            return httpx.Response(200, json=_recognize_response([
                {"bbox": [1, 2, 3, 4], "matched": False, "user_id": None,
                 "similarity": 0.2, "det_score": 0.5},
            ]))

        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as client:
            fc = HttpFaceClient("http://face:18000", client=client)
            results = await fc.recognize(b"img", bearer="t")
        assert results[0].user_id is None
        assert results[0].matched is False

    async def test_error_mapping(self):
        import httpx

        def handler(request):
            return httpx.Response(500, text="boom")

        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as client:
            fc = HttpFaceClient("http://face:18000", client=client)
            with pytest.raises(Exception) as ei:
                await fc.recognize(b"img", bearer="t")
        err = ei.value
        assert getattr(err, "status", None) == 502
        assert getattr(err, "code", None) == "face_backend_error"

    async def test_network_error_mapping(self):
        import httpx

        def handler(request):
            raise httpx.ConnectError("refused")

        async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)) as client:
            fc = HttpFaceClient("http://face:18000", client=client)
            with pytest.raises(Exception) as ei:
                await fc.recognize(b"img", bearer="t")
        assert getattr(ei.value, "code", None) == "face_backend_error"


class TestStubFaceClient:
    async def test_returns_configured(self):
        stub = StubFaceClient([
            FaceResult(bbox=(0, 0, 10, 10), matched=True,
                       user_id="u-1", similarity=0.9),
        ])
        out = await stub.recognize(b"whatever", bearer="t")
        assert out[0].user_id == "u-1"

    async def test_protocol_conformance(self):
        assert isinstance(StubFaceClient([]), FaceClient)
