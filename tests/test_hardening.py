# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Regression tests from the pre-release review: hostile input never raises, misclassifications stay fixed,
and the example proxy always answers (or ends a stream with an in-band error)."""
import json
import os
import random
import socket
import threading
import time

import pytest

import ctxnorm
from conftest import ROOT, load_vectors
from ctxnorm import Limits
from ctxnorm import limits as L
from payloads import JS, SSE, anth_err, anth_stream, oai_chunk, oai_err

MSG = "/v1/messages"
CHAT = "/v1/chat/completions"
DEEP = b"[" * 100000


def _deep_dict(n=5000):
    d = {}
    for _ in range(n):
        d = {"error": d}
    return d


# ------------------------------------------------------------------------- R1/R2: absurd nesting -----
def test_deeply_nested_error_bodies_do_not_raise():
    assert ctxnorm.from_error(400, DEEP) is None
    assert ctxnorm.from_error(400, DEEP.decode()) is None
    assert ctxnorm.from_error(400, b"data: " + DEEP) is None
    assert ctxnorm.from_error(400, _deep_dict()) is None
    assert isinstance(ctxnorm.error_text(_deep_dict()), str)
    assert isinstance(ctxnorm.error_text(_deep_dict(100000)), str)
    nested_raw = {"error": {"message": "Provider returned error", "metadata": {"raw": "{" + "[" * 100000}}}
    assert ctxnorm.from_error(400, nested_raw) is None


@pytest.mark.parametrize("protocol", ["anthropic", "openai-chat"])
def test_deeply_nested_sse_data_does_not_raise(protocol):
    v, held, rest = ctxnorm.lookahead(iter([b"data: " + DEEP + b"\n\n"]), 32000, protocol=protocol)
    assert v == "pass"


def test_deeply_nested_response_dicts_do_not_raise():
    deep = _deep_dict()
    ctxnorm.from_anthropic_message({"content": [{"type": "tool_use", "input": deep}],
                                    "stop_reason": "max_tokens", "usage": {"input_tokens": 5}}, 32000)
    assert ctxnorm.from_openai_chat({"choices": [{"message": deep, "finish_reason": "length"}]}, 32000) is None


def test_proxy_answers_a_deeply_nested_request_body(stub, make_client):
    stub.answer = lambda m, p, h, b: (400, JS, b'{"error":{"message":"bad json"}}')
    c = make_client()
    st, _, raw = c.request(MSG, DEEP)
    assert st == 400 and raw == b'{"error":{"message":"bad json"}}'       # relayed, handler alive


def test_proxy_answers_a_deeply_nested_upstream_body(stub, make_client):
    stub.answer = lambda m, p, h, b: (200, JS, DEEP)
    c = make_client()
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 200 and raw == DEEP


# ------------------------------------------------------------------------- R3-R6: wrong field types --
@pytest.mark.parametrize("msg", [
    {"type": "message", "usage": "x", "stop_reason": "max_tokens", "content": []},
    {"type": "message", "usage": ["x"], "stop_reason": "max_tokens", "content": "abc"},
    {"type": "message", "usage": {"input_tokens": "x", "output_tokens": float("inf")}, "stop_reason": "max_tokens"},
    {"type": "message", "usage": {"input_tokens": 10 ** 400}, "stop_reason": ["max_tokens"], "content": [1, None]},
    {"type": "message", "content": [{"type": "text", "text": 5}, {"type": "tool_use", "name": 3, "input": "x"}]},
])
def test_anthropic_message_with_wrong_types_does_not_raise(msg):
    ctxnorm.from_anthropic_message(msg, 32000)


@pytest.mark.parametrize("resp", [
    {"choices": [{"message": {"tool_calls": [{"function": "x"}]}, "finish_reason": "length"}]},
    {"choices": [{"message": {"tool_calls": "x", "content": ["a", 3]}, "finish_reason": "length"}], "usage": "x"},
    {"choices": "x"}, {"choices": [None]}, {"choices": [{"message": "x", "finish_reason": {}}]},
])
def test_openai_chat_with_wrong_types_does_not_raise(resp):
    ctxnorm.from_openai_chat(resp, 32000)


def test_string_content_is_counted_not_ignored():
    msg = {"type": "message", "content": "x" * 4000, "stop_reason": "max_tokens",
           "usage": {"input_tokens": 900000, "output_tokens": 0}}
    assert ctxnorm.from_anthropic_message(msg, 32000) is None
    resp = {"choices": [{"message": {"content": ["y" * 4000]}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 900000, "completion_tokens": 0}}
    assert ctxnorm.from_openai_chat(resp, 32000) is None


@pytest.mark.parametrize("event", [
    {"type": "content_block_delta", "delta": "x"},
    {"type": "content_block_start", "content_block": "x"},
    {"type": "content_block_start", "content_block": {"text": 5, "name": ["x"]}},
    {"type": "message_start", "message": "x"},
    {"type": "message_start", "message": {"usage": "x"}},
    {"type": "message_delta", "delta": "x", "usage": [1]},
    {"type": "message_delta", "delta": {"stop_reason": {}}, "usage": {"output_tokens": "inf"}},
])
def test_anthropic_sse_events_with_wrong_types_do_not_raise(event):
    ctxnorm.lookahead(iter([b"data: " + json.dumps(event).encode() + b"\n\n"]), 32000)


@pytest.mark.parametrize("event", [
    {"choices": "x"}, {"choices": [{"delta": "x", "finish_reason": 5}]}, {"usage": "x"},
    {"usage": {"prompt_tokens": "x"}}, {"choices": [None], "usage": {"completion_tokens": [1]}},
])
def test_openai_sse_events_with_wrong_types_do_not_raise(event):
    ctxnorm.lookahead(iter([b"data: " + json.dumps(event).encode() + b"\n\n"]), 32000, protocol="openai-chat")


def test_non_bytes_chunks_do_not_raise():
    ctxnorm.lookahead(iter(["data: {}\n\n", None, 5, b"data: {}\n\n"]), 32000)


def _random_json(rng, depth=0):
    r = rng.random()
    if depth > 4 or r < 0.3:
        return rng.choice([None, True, 0, -1, 10 ** 30, 1.5, float("inf"), "", "length", "max_tokens", "x" * 50,
                           "prompt is too long: 9 tokens > 8 maximum", "error", "text", "usage"])
    if r < 0.6:
        return [_random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    keys = ["type", "error", "message", "content", "choices", "usage", "delta", "stop_reason", "finish_reason",
            "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "text", "tool_calls",
            "function", "arguments", "content_block", "metadata", "raw", "code"]
    return {rng.choice(keys): _random_json(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def test_fuzz_public_functions_never_raise():
    rng = random.Random(20260101)
    for _ in range(3000):
        x = _random_json(rng)
        st = rng.choice([200, 400, 413, 500, None, "400"])
        for res in (ctxnorm.from_error(st, x), ctxnorm.from_anthropic_message(x, 32000),
                    ctxnorm.from_openai_chat(x, 32000)):
            assert res is None or isinstance(res, ctxnorm.ContextExhausted)
            if res is not None:
                res.to_anthropic()
                res.to_openai()
                res.summary()
        ctxnorm.from_stop(rng.choice(["length", x]), x, x, rng.choice([32000, x]))
        try:
            raw = json.dumps(x).encode()
        except ValueError:
            continue
        for proto in ("anthropic", "openai-chat"):
            ctxnorm.lookahead(iter([b"data: " + raw + b"\n\n"]), 32000, protocol=proto)
        L.parse_limit_hint(ctxnorm.error_text(x))
        L.estimate_input_tokens(x)


# ------------------------------------------------------------------------- N1: vLLM max_tokens -------
OVERFLOW = load_vectors("max_tokens_overflow.json")


@pytest.mark.parametrize("v", OVERFLOW, ids=[v["provider"] for v in OVERFLOW])
def test_max_tokens_overflow_is_not_prompt_too_long(v):
    info = ctxnorm.from_error(v["status"], v["body"])
    assert info and info.is_max_tokens_overflow
    assert (info.input_tokens, info.completion_tokens, info.limit) == (
        v["expect"]["input_tokens"], v["expect"]["completion_tokens"], v["expect"]["limit"])
    assert info.to_anthropic_max_tokens()[1]["error"]["message"].startswith(
        "input length and `max_tokens` exceed context limit")


def test_proxy_maps_vllm_max_tokens_error_to_overflow_not_compaction(stub, make_client):
    body = json.dumps(OVERFLOW[0]["body"]).encode()
    stub.answer = lambda m, p, h, b: (400, JS, body)
    c = make_client()
    st, hdrs, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 400 and hdrs["x-ctx-normalised"] == "max-tokens-overflow"
    assert anth_err(raw).startswith("input length and `max_tokens` exceed context limit: 1000 + 32000 > 32768")
    st, _, raw = c.request(CHAT, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 400 and raw == body                                    # OpenAI clients: relayed as-is


# ------------------------------------------------------------------------- N2/N3: error shapes -------
OPENROUTER_MIDSTREAM = (b'data: {"id":"gen-1","object":"chat.completion.chunk","created":1,"model":"m",'
                        b'"provider":"P","error":{"code":400,"message":"This endpoint\'s maximum context length '
                        b'is 131072 tokens. However, you requested about 150000 tokens (140000 of text input, '
                        b'10000 in the output)."},"choices":[{"index":0,"delta":{"content":""},'
                        b'"finish_reason":"error"}]}\n\n')


def test_openrouter_mid_stream_error_with_choices_is_detected():
    v, info, _ = ctxnorm.lookahead(iter([oai_chunk({"role": "assistant", "content": ""}), OPENROUTER_MIDSTREAM]),
                                   32000, protocol="openai-chat")
    assert v == "exhausted" and info.kind == "explicit" and info.limit == 131072


def test_proxy_normalises_openrouter_mid_stream_error(stub, make_client):
    stub.answer = lambda m, p, h, b: (200, SSE, [oai_chunk({"role": "assistant", "content": ""}),
                                                 OPENROUTER_MIDSTREAM, b"data: [DONE]\n\n"])
    c = make_client()
    st, _, raw = c.request(CHAT, {"model": "m", "max_tokens": 32000, "stream": True, "messages": []})
    assert st == 400 and oai_err(raw)["code"] == "context_length_exceeded"


def test_error_body_on_a_200_is_detected():
    err = {"error": {"message": "prompt is too long: 9000 tokens > 8192 maximum"}}
    assert ctxnorm.from_openai_chat(err, 32000).limit == 8192
    assert ctxnorm.from_anthropic_message(dict(err, type="error"), 32000).limit == 8192


def test_proxy_normalises_an_error_body_on_a_200(stub, make_client):
    stub.answer = lambda m, p, h, b: (200, JS, b'{"error":{"message":"Input is too long for requested model."}}')
    c = make_client()
    st, _, raw = c.request(CHAT, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 400 and oai_err(raw)["code"] == "context_length_exceeded"


# ------------------------------------------------------------------------- N4: TGI -------------------
def test_tgi_prompt_too_long_is_detected():
    info = ctxnorm.from_error(422, {"error": "Input validation error: `inputs` must have less than 4096 tokens. "
                                             "Given: 5000", "error_type": "validation"})
    assert info and not info.is_max_tokens_overflow and (info.input_tokens, info.limit) == (5000, 4096)


# ------------------------------------------------------------------------- N5: OpenAI wording --------
def test_openai_rendering_of_implicit_exhaustion_does_not_contradict_itself():
    msg = ctxnorm.from_stop("length", 6, 409594, 32000).to_openai()[1]["error"]["message"]
    assert "maximum context length is 409600 tokens" in msg and "leaves no room for output" in msg


# ------------------------------------------------------------------------- P1: usage callback --------
def test_usage_callback_has_no_private_keys_and_never_raises():
    evs = anth_stream(stop="max_tokens", in_tok=0, out_tok=3)
    v, info, _ = ctxnorm.lookahead(iter(evs), 32000, usage=lambda: {"in_tokens": 1000, "cache_read": 5000})
    assert v == "exhausted" and info.input_tokens == 1000               # cache_read is not a public key
    for bad in ({"in_tokens": "x", "out_tokens": [1]}, "not a dict", {"stop": 5}):
        ctxnorm.lookahead(iter(evs), 32000, usage=lambda bad=bad: bad)


# ------------------------------------------------------------------------- X1: dropped upstream ------
class RawUpstream:
    """A TCP server that reads one request and answers with raw bytes, then closes (optionally hard)."""

    def __init__(self, payload: bytes, reset: bool = False):
        self.payload, self.reset = payload, reset
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _, body = data.partition(b"\r\n\r\n")
            n = next((int(ln.split(b":")[1]) for ln in head.split(b"\r\n")
                      if ln.lower().startswith(b"content-length")), 0)
            while len(body) < n:
                body += conn.recv(65536)
            conn.sendall(self.payload)
            time.sleep(0.05)
            if self.reset:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            conn.close()

    def close(self):
        self.sock.close()


@pytest.fixture
def raw_client(stub):
    made = []

    def factory(payload, reset=False, **kw):
        import proxy as proxy_mod
        from conftest import Client
        up = RawUpstream(payload, reset)
        stub.close()
        c = Client(stub, **kw)
        c.proxy.upstream = proxy_mod.urllib.parse.urlsplit(up.url)
        made.append((c, up))
        return c
    yield factory
    for c, up in made:
        c.close()
        up.close()


def _chunked(*parts):
    return b"".join(b"%x\r\n" % len(p) + p + b"\r\n" for p in parts)


SSE_HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"


def test_truncated_json_response_gets_a_502(raw_client):
    c = raw_client(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"id\":")
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 502 and b"upstream connection failed" in raw


@pytest.mark.parametrize("reset", [False, True])
def test_stream_dropped_during_lookahead_gets_a_502(raw_client, reset):
    c = raw_client(SSE_HEAD + _chunked(anth_stream()[0]) + b"40\r\nhalf", reset=reset)
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "stream": True, "messages": []})
    assert st == 502 and b"upstream connection failed" in raw


def test_stream_dropped_after_commit_ends_with_an_error_event(raw_client):
    events = anth_stream(texts=["x" * 5000], stop="end_turn")
    c = raw_client(SSE_HEAD + _chunked(*events[:4]) + b"40\r\nhalf")
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "stream": True, "messages": []})
    assert st == 200 and b"x" * 5000 in raw
    assert raw.rstrip().endswith(b'"upstream connection lost: IncompleteRead"}}')
    assert b"event: error" in raw


# ------------------------------------------------------------------------- X2: caps and timeouts -----
def test_request_body_cap_gives_413(stub, make_client):
    c = make_client(max_body=1000)
    st, _, raw = c.request(MSG, {"model": "m", "messages": [{"role": "user", "content": "x" * 2000}]})
    assert st == 413 and stub.seen == []


def test_invalid_content_length_gives_400(stub, make_client):
    c = make_client()
    s = socket.create_connection(("127.0.0.1", c.server.server_address[1]))
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n")
    assert s.recv(100).startswith(b"HTTP/1.0 400")
    s.close()


def test_stalled_client_is_answered_and_released(stub, make_client):
    c = make_client(client_timeout=0.5)
    s = socket.create_connection(("127.0.0.1", c.server.server_address[1]))
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n")
    s.settimeout(5)
    t0 = time.time()
    data = s.recv(200)
    assert data.startswith(b"HTTP/1.0 408") and time.time() - t0 < 4
    s.close()


def test_upstream_response_cap_gives_502(stub, make_client):
    stub.answer = lambda m, p, h, b: (200, JS, b'{"x":"' + b"y" * 5000 + b'"}')
    c = make_client(max_response=1000)
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 502 and b"exceeds 1000 bytes" in raw


def test_upstream_timeout_gives_502(stub, make_client):
    def slow(m, p, h, b):
        time.sleep(1.0)
        return 200, JS, b"{}"
    stub.answer = slow
    c = make_client(timeout=0.3)
    st, _, raw = c.request(MSG, {"model": "m", "max_tokens": 32000, "messages": []})
    assert st == 502


# ------------------------------------------------------------------------- cleanup ------------------
def test_no_private_references_or_incident_numbers_in_the_tree():
    banned = ["1048" + "570", "1048" + "576"]
    hits = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".venv")]
        for f in files:
            p = os.path.join(base, f)
            try:
                text = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            hits += [(os.path.relpath(p, ROOT), b) for b in banned if b in text]
    assert hits == []


def test_limits_are_still_honoured_with_hardening(stub, make_client):
    c = make_client(limits={"*": Limits(context_window=10)})
    st, _, _ = c.request(MSG, {"model": "m", "messages": [{"role": "user", "content": "x" * 400}]})
    assert st == 400 and stub.seen == []
