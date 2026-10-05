#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""A fake LLM provider whose context window is always full, for trying the proxy without a model.

    python examples/stub_upstream.py --port 8000

It answers both ``POST /v1/chat/completions`` and ``POST /v1/messages`` the way some providers answer an
over-full prompt: HTTP 200, an EMPTY answer, stop reason ``length`` / ``max_tokens``, and the input size
reported in ``usage``. Requests with ``"max_tokens"`` below 1024 get a normal one-word answer instead.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROMPT, WINDOW = 131066, 131072


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        full = int(body.get("max_tokens") or body.get("max_completion_tokens") or 0) >= 1024
        text, out = ("", WINDOW - PROMPT) if full else ("pong", 1)
        if self.path.split("?")[0].rstrip("/").endswith("/messages"):
            obj = {"id": "msg_stub", "type": "message", "role": "assistant", "model": body.get("model", "stub"),
                   "content": [{"type": "text", "text": text}] if text else [],
                   "stop_reason": "max_tokens" if full else "end_turn",
                   "usage": {"input_tokens": PROMPT if full else 5, "output_tokens": out}}
        else:
            obj = {"id": "chatcmpl-stub", "object": "chat.completion", "model": body.get("model", "stub"),
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                "finish_reason": "length" if full else "stop"}],
                   "usage": {"prompt_tokens": PROMPT if full else 5, "completion_tokens": out}}
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"full-window stub on http://127.0.0.1:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
