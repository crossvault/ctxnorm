# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Shared fixtures: an offline guard, a local stub upstream, and the example proxy in front of it."""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "examples")):
    if p not in sys.path:
        sys.path.insert(0, p)

VECTORS = os.path.join(ROOT, "vectors")
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# ------------------------------------------------------------------------------------ offline guard --
@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test is offline: connecting anywhere but loopback fails loudly."""
    real = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in _LOOPBACK:
            raise RuntimeError(f"test tried to reach the network: {host!r}")
        return real(self, address)
    monkeypatch.setattr(socket.socket, "connect", guarded)


def load_vectors(name: str) -> list:
    with open(os.path.join(VECTORS, name), encoding="utf-8") as f:
        return json.load(f)["vectors"]


# ------------------------------------------------------------------------------------ stub upstream --
class Stub:
    """A local HTTP server standing in for an LLM provider.

    `answer(method, path, headers, body) -> (status, headers, payload)` where payload is bytes, or a list
    of byte chunks that are written (and flushed) one by one, like a server-sent-events stream. Every
    request is recorded in `seen` with its JSON body parsed."""

    def __init__(self) -> None:
        self.answer: Callable = lambda m, p, h, b: (404, [], b"")
        self.seen: List[Dict] = []
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    parsed = json.loads(raw) if raw else None
                except (ValueError, RecursionError):
                    parsed = None
                stub.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                  "body": parsed})
                status, headers, payload = stub.answer(self.command, self.path, self.headers, parsed)
                self.send_response(status)
                for k, v in headers:
                    self.send_header(k, v)
                if isinstance(payload, (bytes, bytearray)):
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                for c in ([payload] if isinstance(payload, (bytes, bytearray)) else payload):
                    self.wfile.write(c)
                    self.wfile.flush()

            do_GET = do_POST = _any

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


class Client:
    """The example proxy in front of `stub`, plus a tiny HTTP client and the captured events."""

    def __init__(self, stub: Stub, **proxy_kw) -> None:
        import proxy as proxy_mod
        self.events: List[dict] = []
        proxy_kw.setdefault("timeout", 10)
        self.proxy = proxy_mod.Proxy(stub.url, on_event=self.events.append, **proxy_kw)
        self.server = proxy_mod.make_server(self.proxy)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def request(self, path: str, body=None, headers=None, method: str = "POST"):
        c = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        raw = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        h = {"Content-Type": "application/json", "Authorization": "Bearer dummy-not-a-key"}
        h.update(headers or {})
        c.request(method, path, body=raw, headers=h)
        r = c.getresponse()
        data = r.read()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        c.close()
        return r.status, hdrs, data

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def make_client(stub):
    made = []

    def factory(**kw) -> Client:
        c = Client(stub, **kw)
        made.append(c)
        return c
    yield factory
    for c in made:
        c.close()
