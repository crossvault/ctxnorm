#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""A minimal reverse proxy that normalises context exhaustion, in front of an OpenAI- or
Anthropic-compatible endpoint (vLLM, llama.cpp, Ollama, LiteLLM, a hosted API, ...).

    python examples/proxy.py --upstream http://127.0.0.1:8000 --port 8080

Point your client at the proxy instead of the upstream (``OPENAI_BASE_URL=http://127.0.0.1:8080/v1`` or
``ANTHROPIC_BASE_URL=http://127.0.0.1:8080``). The client's own credentials are forwarded unchanged; the
proxy holds no keys.

What it does, per request path:

* ``.../chat/completions`` (OpenAI Chat Completions): any provider's context-too-long error, and an
  empty ``length`` stop on a large budget, become OpenAI's ``context_length_exceeded`` (HTTP 400).
* ``.../messages`` (Anthropic Messages): the same, as Anthropic's ``prompt is too long: N tokens > M
  maximum``, which Claude Code answers by compacting the conversation and retrying.
* everything else is relayed untouched.

Optional per-model limits (``--limit MODEL=WINDOW[/MAX_OUTPUT]``, ``*`` for any model) add a max_output
clamp and a pre-send context-window check. Events (sizes only, never content) are logged as JSON lines
to stderr, or handed to ``on_event`` when embedding `Proxy` in your own code.

This is an EXAMPLE: stdlib only, one thread per request, no TLS termination, no auth of its own. Bind it
to localhost or put it behind something that does those things.
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import socket
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterator, Optional

import ctxnorm
from ctxnorm import limits as L

log = logging.getLogger("ctxnorm.proxy")

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
MARK_HEADER = "X-Ctx-Normalised"
TAIL_BYTES = 4096


def _protocol(method: str, path: str) -> Optional[str]:
    p = urllib.parse.urlsplit(path).path.rstrip("/")
    if method != "POST":
        return None
    if p.endswith("/chat/completions"):
        return "openai-chat"
    if p.endswith("/messages"):
        return "anthropic"
    return None


def _requested_max(body: dict, protocol: str) -> Any:
    if protocol == "openai-chat":
        return body.get("max_completion_tokens", body.get("max_tokens"))
    return body.get("max_tokens")


class UpstreamTooLarge(Exception):
    """The upstream response exceeded `Proxy.max_response`."""


class Proxy:
    """Configuration and policy of one proxy instance (shared by all request threads).

    timeout         seconds to wait on the upstream (connect and each read)
    client_timeout  seconds to wait on the client (each read / write); a stalled client frees its thread
    max_body        largest request body accepted (bytes); larger requests get 413
    max_response    largest upstream body buffered for inspection (bytes); streams are not buffered beyond
                    the look-ahead and are not limited
    """

    def __init__(self, upstream: str, config: Optional[ctxnorm.Config] = None,
                 limits: Optional[Dict[str, L.Limits]] = None,
                 on_event: Optional[Callable[[dict], None]] = None, timeout: float = 600.0,
                 enforce_limits: bool = True, client_timeout: float = 60.0, max_body: int = 32 << 20,
                 max_response: int = 64 << 20) -> None:
        u = urllib.parse.urlsplit(upstream)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("upstream must be an http(s) URL")
        self.upstream = u
        self.config = config or ctxnorm.Config()
        self.limits = dict(limits or {})
        self.on_event = on_event or (lambda ev: log.info("%s", json.dumps(ev, sort_keys=True)))
        self.timeout = timeout
        self.client_timeout = client_timeout
        self.max_body = max_body
        self.max_response = max_response
        self.enforce_limits = enforce_limits

    def limits_for(self, model: Any) -> L.Limits:
        if not self.enforce_limits:
            return L.Limits()
        return self.limits.get(model if isinstance(model, str) else "", self.limits.get("*", L.Limits()))

    def connect(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.upstream.scheme == "https" else http.client.HTTPConnection
        return cls(self.upstream.hostname, self.upstream.port, timeout=self.timeout)

    def emit(self, event: str, **fields: Any) -> None:
        try:
            self.on_event(dict(fields, event=event))
        except Exception:                                   # a broken hook must not break the request
            log.exception("on_event hook failed")


def _error_body(message: str, kind: str = "upstream_error") -> dict:
    return {"error": {"message": message, "type": kind}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"          # close-delimited responses: streams need no chunked encoding
    server_version = "ctxnorm-example-proxy"
    proxy: Proxy                            # set by make_server
    _committed = False                      # True once a status line went to the client

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s " + fmt, self.address_string(), *args)

    def do_GET(self) -> None:
        try:
            self._handle()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout):
            pass                                          # the client went away or stalled
        except Exception:
            log.exception("request failed")
            if not self._committed:
                self._send_json(502, _error_body("proxy error"))

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_GET

    # ------------------------------------------------------------------------------------------------
    def send_response(self, *a: Any, **kw: Any) -> None:
        self._committed = True
        super().send_response(*a, **kw)

    def _send(self, status: int, headers: list, body: bytes = b"") -> None:
        self.send_response(status)
        for k, v in headers:
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, obj: dict, mark: Optional[str] = None) -> None:
        hdrs = [("Content-Type", "application/json")]
        if mark:
            hdrs.append((MARK_HEADER, mark))
        self._send(status, hdrs, json.dumps(obj).encode())

    def _exhausted(self, protocol: str, info: ctxnorm.ContextExhausted, est: int, model: Any) -> None:
        if protocol == "anthropic" and info.is_max_tokens_overflow:
            status, obj = info.to_anthropic_max_tokens()
            self.proxy.emit("max_tokens_overflow", protocol=protocol, model=model, **info.summary(est))
            return self._send_json(status, obj, "max-tokens-overflow")
        status, obj = info.to_anthropic(est) if protocol == "anthropic" else info.to_openai(est)
        self.proxy.emit("context_exhausted", protocol=protocol, model=model, **info.summary(est))
        self._send_json(status, obj, "context-exhausted")

    # ------------------------------------------------------------------------------------------------
    def _read_request_body(self) -> Optional[bytes]:
        """The request body, or None after an error response was sent."""
        cl = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding"):
            self._send_json(411, _error_body("chunked request bodies are not supported", "invalid_request_error"))
            return None
        try:
            n = int(cl) if cl is not None else 0
            if n < 0:
                raise ValueError
        except ValueError:
            self._send_json(400, _error_body("invalid Content-Length", "invalid_request_error"))
            return None
        if n > self.proxy.max_body:
            self._send_json(413, _error_body(f"request body exceeds {self.proxy.max_body} bytes",
                                             "invalid_request_error"))
            return None
        try:
            raw = self.rfile.read(n) if n else b""
        except (TimeoutError, socket.timeout):
            self._send_json(408, _error_body("timed out reading the request body", "invalid_request_error"))
            return None
        if len(raw) != n:
            self._send_json(400, _error_body("request body shorter than Content-Length", "invalid_request_error"))
            return None
        return raw

    def _handle(self) -> None:
        px = self.proxy
        raw = self._read_request_body()
        if raw is None:
            return
        protocol = _protocol(self.command, self.path)
        body = None
        if protocol and raw:
            try:
                body = json.loads(raw)
            except (ValueError, RecursionError):
                body = None
            if not isinstance(body, dict):
                protocol, body = None, None

        model, lim, req_max, est, clamped = None, L.Limits(), None, len(raw) // 4, False
        if protocol and body is not None:
            est = L.estimate_input_tokens(body) or est
            model = body.get("model")
            lim = px.limits_for(model)
            if lim.max_output:
                changes = L.clamp_output(body, lim.max_output,
                                         "anthropic-messages" if protocol == "anthropic" else "openai-chat")
                if changes:
                    clamped = True
                    px.emit("output_capped", protocol=protocol, model=model, max_output=lim.max_output,
                            changes=changes)
            refused = L.precheck(body, lim) if lim.context_window else None
            if refused:
                px.emit("context_window_refused", protocol=protocol, model=model, **refused.summary())
                status, obj = refused.to_anthropic() if protocol == "anthropic" else refused.to_openai()
                return self._send_json(status, obj, "context-window")
            req_max = _requested_max(body, protocol)
            if (protocol == "openai-chat" and body.get("stream") and "stream_options" not in body
                    and ctxnorm.eligible(req_max, px.config)):
                body["stream_options"] = {"include_usage": True}     # the implicit check needs usage
            raw = json.dumps(body).encode()

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"
        if raw or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(raw))
        path = px.upstream.path.rstrip("/") + self.path
        conn = px.connect()
        try:
            try:
                conn.request(self.command, path, body=raw or None, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                return self._send_json(502, _error_body(f"upstream unreachable: {type(e).__name__}"))
            try:
                self._relay(resp, protocol, body, model, lim, req_max, est, clamped)
            except UpstreamTooLarge:
                if not self._committed:
                    self._send_json(502, _error_body(f"upstream response exceeds {px.max_response} bytes"))
            except (OSError, http.client.HTTPException) as e:
                # The upstream dropped or truncated the response before anything was relayed.
                if not self._committed:
                    self._send_json(502, _error_body(f"upstream connection failed: {type(e).__name__}"))
        finally:
            conn.close()

    def _read_all(self, resp: http.client.HTTPResponse) -> bytes:
        data = resp.read(self.proxy.max_response + 1)
        if len(data) > self.proxy.max_response:
            raise UpstreamTooLarge()
        if resp.length:                                   # fewer bytes than the upstream's Content-Length
            raise http.client.IncompleteRead(data, resp.length)
        return data

    def _relay(self, resp: http.client.HTTPResponse, protocol: Optional[str], body: Optional[dict],
               model: Any, lim: L.Limits, req_max: Any, est: int, clamped: bool) -> None:
        px = self.proxy
        hdrs = resp.getheaders()
        ctype = (resp.getheader("Content-Type") or "").lower()
        streaming = "text/event-stream" in ctype

        if not protocol:
            if streaming:
                self._stream_out(resp.status, hdrs, [], self._chunks(resp), None)
            else:
                self._send(resp.status, hdrs, self._read_all(resp))
            return

        if resp.status >= 400:
            data = self._read_all(resp)
            info = ctxnorm.from_error(resp.status, data, px.config)
            if info and not (protocol == "openai-chat" and info.is_max_tokens_overflow):
                return self._exhausted(protocol, info, est, model)
            refusal = L.output_refusal(ctxnorm.error_text(data))
            if refusal:
                px.emit("output_refused", protocol=protocol, model=model, max_output=refusal,
                        upstream_status=resp.status)
            return self._send(resp.status, hdrs, data)

        if streaming:
            verdict, a, b = ctxnorm.lookahead(self._chunks(resp), req_max, px.config,
                                              protocol="anthropic" if protocol == "anthropic" else "openai-chat")
            if verdict == "exhausted":
                return self._exhausted(protocol, a, est, model)
            tail = self._stream_out(resp.status, hdrs, a, b, protocol)
            self._check_output_limit(L.stop_in_tail(tail), body, protocol, model, lim, clamped)
        else:
            data = self._read_all(resp)
            try:
                obj = json.loads(data)
            except (ValueError, RecursionError):
                obj = None
            if obj is not None and 200 <= resp.status < 300:
                info = (ctxnorm.from_anthropic_message(obj, req_max, px.config) if protocol == "anthropic"
                        else ctxnorm.from_openai_chat(obj, req_max, px.config))
                if info:
                    return self._exhausted(protocol, info, est, model)
            self._check_output_limit(L.stop_in_tail(data[-TAIL_BYTES:]), body, protocol, model, lim, clamped)
            self._send(resp.status, hdrs, data)

    def _check_output_limit(self, stop: Optional[str], body: Optional[dict], protocol: str, model: Any,
                            lim: L.Limits, clamped: bool) -> None:
        if not clamped and L.stopped_at_output_limit(stop, _requested_max(body or {}, protocol), lim):
            self.proxy.emit("output_limit_stop", protocol=protocol, model=model, stop=stop,
                            max_output=lim.max_output or lim.max_output_default)

    @staticmethod
    def _chunks(resp: http.client.HTTPResponse) -> Iterator[bytes]:
        while True:
            c = resp.read1(65536)
            if not c:
                return
            yield c

    @staticmethod
    def _stream_error_event(protocol: Optional[str], message: str) -> bytes:
        """An in-band error for a stream whose upstream failed after the 200 was sent."""
        if protocol == "anthropic":
            d = {"type": "error", "error": {"type": "api_error", "message": message}}
            return b"event: error\ndata: " + json.dumps(d).encode() + b"\n\n"
        return b"data: " + json.dumps(_error_body(message)).encode() + b"\n\n"

    def _stream_out(self, status: int, hdrs: list, held: Any, rest: Iterator[bytes],
                    protocol: Optional[str]) -> bytes:
        self.send_response(status)
        for k, v in hdrs:
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.end_headers()
        tail = b""
        for c in list(held):
            self.wfile.write(c)
            tail = (tail + c)[-TAIL_BYTES:]
        self.wfile.flush()
        while True:
            try:
                c = next(rest)
            except StopIteration:
                break
            except (OSError, http.client.HTTPException) as e:
                log.warning("upstream failed mid-stream: %s", type(e).__name__)
                self.wfile.write(self._stream_error_event(protocol, f"upstream connection lost: {type(e).__name__}"))
                break
            self.wfile.write(c)
            self.wfile.flush()
            tail = (tail + c)[-TAIL_BYTES:]
        return tail


def make_server(proxy: Proxy, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """A ready-to-serve HTTP server for `proxy` (port 0 = pick a free port; see ``server_address``)."""
    handler = type("BoundHandler", (Handler,), {"proxy": proxy, "timeout": proxy.client_timeout})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def _parse_limit(spec: str) -> tuple:
    model, sep, rest = spec.partition("=")
    if not sep or not model:
        raise argparse.ArgumentTypeError("expected MODEL=WINDOW[/MAX_OUTPUT], e.g. 'qwen3=131072/32768'")
    window, _, out = rest.partition("/")
    try:
        lim = L.Limits.from_mapping({"context_window": int(window) if window else None,
                                     "max_output": int(out) if out else None})
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number in {spec!r}") from None
    return model, lim


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True, help="upstream server root, e.g. http://127.0.0.1:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--limit", type=_parse_limit, action="append", default=[], metavar="MODEL=WINDOW[/MAX_OUT]",
                    help="per-model limits; '*' matches any model (repeatable)")
    ap.add_argument("--disable", action="store_true", help="relay everything unchanged (A/B comparison)")
    ap.add_argument("--upstream-timeout", type=float, default=600.0, help="seconds (default 600)")
    ap.add_argument("--client-timeout", type=float, default=60.0, help="seconds (default 60)")
    ap.add_argument("--max-body", type=int, default=32 << 20, help="max request bytes (default 32 MiB)")
    ap.add_argument("--max-response", type=int, default=64 << 20,
                    help="max buffered upstream response bytes (default 64 MiB)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(message)s")
    px = Proxy(a.upstream, ctxnorm.Config(enabled=not a.disable), dict(a.limit), timeout=a.upstream_timeout,
               client_timeout=a.client_timeout, max_body=a.max_body, max_response=a.max_response)
    srv = make_server(px, a.host, a.port)
    log.info("listening on http://%s:%d -> %s", *srv.server_address[:2], a.upstream)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
