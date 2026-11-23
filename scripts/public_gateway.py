# SPDX-License-Identifier: MIT
"""Read-only, fixed-origin public demo gateway; never imports the model/server.

Usage: python scripts/public_gateway.py --port 8766 \
    --upstream http://127.0.0.1:8765 --expires-at 2026-11-15T07:00:00Z
"""
from __future__ import annotations

import argparse
import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = "http://127.0.0.1:8765"
READ_ONLY_MESSAGE = "Public demo is read-only; training cannot be started or stopped."
EXPIRED_MESSAGE = "Public demo has expired."
NOT_AVAILABLE_MESSAGE = "This public demo path is not exposed."
UPSTREAM_UNAVAILABLE = "Demo upstream is temporarily unavailable."
SAFE_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}"
EXACT_APIS = frozenset({
    "/api/health", "/api/status", "/api/protocol", "/api/optimization",
    "/api/runs", "/api/data-protection", "/api/health-data",
    "/api/health-data/sites", "/api/subjects",
})
RUN_PATH = re.compile(rf"/api/runs/{SAFE_ID}(?:/federation(?:/([1-9][0-9]{{0,2}}))?)?\Z")
SUBJECT_PATH = re.compile(rf"/api/(?:health-data/)?subjects/{SAFE_ID}\Z")
STATIC_PATH = re.compile(
    r"/(?:[A-Za-z0-9_@~.-]+/)*[A-Za-z0-9_@~.-]+"
    r"\.(?:js|css|woff2?|ttf|otf|svg|png|ico|jpe?g|webp|gif|avif)\Z", re.I
)
RESPONSE_HEADERS = frozenset({
    "content-type", "cache-control", "etag", "last-modified",
    "content-range", "accept-ranges", "expires",
})
REQUEST_HEADERS = frozenset({
    "accept", "if-none-match", "if-modified-since", "range", "if-range",
})


def parse_expiry(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("expires-at must include a UTC offset or Z")
    return stamp.astimezone(timezone.utc)


def allowed_path(path: str) -> bool:
    # ASGI paths have already been percent-decoded. Reject residual escapes to
    # prevent double-decoding, Windows separators, and traversal at every depth.
    if not path.startswith("/") or path.startswith("//") or any(c in path for c in ("\\", "%", "\x00")):
        return False
    segments = path.split("/")
    if any(part in (".", "..") for part in segments):
        return False
    lower = path.lower()
    if any(word in lower for word in ("checkpoint", "download", "export")):
        return False
    if re.search(r"\.(?:py|pt)(?:[./]|$)", lower):
        return False
    if path == "/" or path in EXACT_APIS:
        return True
    match = RUN_PATH.fullmatch(path)
    if match:
        return match.group(1) is None or int(match.group(1)) <= 500
    if SUBJECT_PATH.fullmatch(path):
        return True
    return not lower.startswith("/api/") and STATIC_PATH.fullmatch(path) is not None


def create_app(
    *, upstream: str = UPSTREAM, expires_at: str | datetime | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: Callable[[], datetime] | None = None,
) -> FastAPI:
    if upstream.rstrip("/") != UPSTREAM:
        raise ValueError(f"Only the fixed upstream {UPSTREAM} is permitted")
    deadline = parse_expiry(expires_at)
    clock = now or (lambda: datetime.now(timezone.utc))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=UPSTREAM, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_connections=24, max_keepalive_connections=16),
            transport=transport,
        ) as client:
            app.state.upstream_client = client
            app.state.inference_slots = asyncio.Semaphore(8)
            yield

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan, redirect_slashes=False)

    @app.middleware("http")
    async def fence(request: Request, call_next):
        if deadline is not None and clock() >= deadline:
            return JSONResponse({"detail": EXPIRED_MESSAGE}, status_code=410,
                                headers={"Cache-Control": "no-store"})
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": READ_ONLY_MESSAGE}, status_code=403,
                                headers={"Cache-Control": "no-store"})
        if not allowed_path(request.scope["path"]):
            return JSONResponse({"detail": NOT_AVAILABLE_MESSAGE}, status_code=404,
                                headers={"Cache-Control": "no-store"})
        return await call_next(request)

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def proxy(request: Request, path: str):
        # Construct origin-form target only; incoming Host/URL/proxy credentials
        # can never select another server. Do not forward any client body.
        target = httpx.URL(UPSTREAM).copy_with(
            path=request.scope["path"], query=request.scope["query_string"]
        )
        headers = {key: value for key, value in request.headers.items() if key in REQUEST_HEADERS}
        headers["accept-encoding"] = "identity"

        async def fetch():
            client = request.app.state.upstream_client
            outbound = client.build_request(request.method, target, headers=headers)
            # AsyncClient retains upstream Set-Cookie values in its cookie jar;
            # remove even automatically generated credentials on every send.
            for name in ("cookie", "authorization", "proxy-authorization"):
                outbound.headers.pop(name, None)
            return await client.send(outbound, follow_redirects=False)

        try:
            # Bound expensive demo inference without throttling initial assets.
            if re.fullmatch(rf"/api/subjects/{SAFE_ID}", request.scope["path"]):
                async with request.app.state.inference_slots:
                    result = await fetch()
            else:
                result = await fetch()
        except httpx.HTTPError:
            return JSONResponse({"detail": UPSTREAM_UNAVAILABLE}, status_code=502,
                                headers={"Cache-Control": "no-store"})
        # httpx has decoded the body: never forward its old encoding/length.
        # A whitelist also strips hop-by-hop, cookies, and redirect headers.
        connection_tokens = {
            item.strip().lower() for item in result.headers.get("connection", "").split(",")
        }
        response_headers = {
            key: value for key, value in result.headers.items()
            if key in RESPONSE_HEADERS and key not in connection_tokens
        }
        response_headers["X-Content-Type-Options"] = "nosniff"
        return Response(content=b"" if request.method == "HEAD" else result.content,
                        status_code=result.status_code, headers=response_headers)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--upstream", default=UPSTREAM)
    parser.add_argument("--expires-at", help="UTC ISO-8601 timestamp with Z or explicit offset")
    args = parser.parse_args()
    try:
        app = create_app(upstream=args.upstream, expires_at=args.expires_at)
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
