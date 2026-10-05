# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""End-to-end through the example proxy (examples/proxy.py) against a local stub upstream.

Each test starts a stub provider on 127.0.0.1 and the proxy in front of it; nothing leaves the machine
and no real credentials are used.
"""
import json
import re

import pytest

from ctxnorm import Limits
from payloads import FULL_PROMPT, JS, SSE, WINDOW, anth_err, anth_message, anth_stream, oai_err, oai_json, oai_stream

MSG = "/v1/messages"
CHAT = "/v1/chat/completions"
BIG = "x" * 40000          # ~10000 estimated tokens


def _messages(stream=False, max_tokens=32000, text="hi", **kw):
    return {"model": "stub-model", "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": text}], **kw}


def _chat(stream=False, max_tokens=32000, text="hi", field="max_tokens", **kw):
    return {"model": "stub-model", field: max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": text}], **kw}


def _answer(stub, status, headers, payload):
    stub.answer = lambda m, p, h, b: (status, headers, payload)


def _events(client, name):
    return [e for e in client.events if e["event"] == name]


# ------------------------------------------------------------------------------ normalisation --------
@pytest.mark.parametrize("stream", [True, False])
def test_messages_empty_length_answer_becomes_prompt_too_long(stub, make_client, stream):
    _answer(stub, 200, SSE if stream else JS, anth_stream(stop="max_tokens") if stream else anth_message())
    c = make_client()
    st, hdrs, raw = c.request(MSG, _messages(stream))
    assert st == 400, raw
    assert anth_err(raw) == f"prompt is too long: {FULL_PROMPT} tokens > {WINDOW} maximum"
    assert hdrs["x-ctx-normalised"] == "context-exhausted"
    assert len(stub.seen) == 1                                          # sent once, never retried


@pytest.mark.parametrize("stream", [True, False])
def test_chat_empty_length_answer_becomes_context_length_exceeded(stub, make_client, stream):
    _answer(stub, 200, SSE if stream else JS, oai_stream() if stream else oai_json())
    c = make_client()
    st, _, raw = c.request(CHAT, _chat(stream))
    e = oai_err(raw)
    assert st == 400 and e["code"] == "context_length_exceeded"
    assert str(WINDOW) in e["message"] and str(FULL_PROMPT) in e["message"]


def test_chat_explicit_context_error_is_normalised_and_sent_once(stub, make_client):
    err = {"error": {"message": "This model's maximum context length is 131072 tokens. However, your messages "
                                "resulted in 140000 tokens.", "code": "context_length_exceeded"}}
    _answer(stub, 400, JS, json.dumps(err).encode())
    c = make_client()
    st, _, raw = c.request(MSG, _messages(True))
    assert st == 400 and anth_err(raw) == "prompt is too long: 140000 tokens > 131072 maximum"
    assert len(stub.seen) == 1


def test_context_error_on_a_5xx_status_becomes_400(stub, make_client):
    _answer(stub, 500, JS, json.dumps({"error": {"message": "context_length_exceeded",
                                                 "type": "server_error"}}).encode())
    c = make_client()
    st, _, raw = c.request(MSG, _messages())
    assert st == 400 and anth_err(raw).startswith("prompt is too long: ")


def test_other_upstream_errors_are_relayed_unchanged(stub, make_client):
    body = b'{"error":{"message":"overloaded"}}'
    _answer(stub, 503, JS, body)
    c = make_client()
    st, hdrs, raw = c.request(MSG, _messages())
    assert st == 503 and raw == body and "x-ctx-normalised" not in hdrs
    assert c.events == []


@pytest.mark.parametrize("stream", [True, False])
def test_legit_max_tokens_stops_are_relayed(stub, make_client, stream):
    c = make_client()
    # (a) the caller asked for a small budget
    _answer(stub, 200, SSE if stream else JS,
            oai_stream(text="abc", completion=3) if stream else oai_json(text="abc", completion=3))
    st, _, raw = c.request(CHAT, _chat(stream, max_tokens=3))
    assert st == 200 and b'"length"' in raw
    # (b) real content of meaningful length was produced
    text = "word " * 2000
    _answer(stub, 200, SSE if stream else JS,
            oai_stream(text=text, completion=2000) if stream else oai_json(text=text, completion=2000))
    st, _, raw = c.request(CHAT, _chat(stream))
    assert st == 200 and b"word word" in raw


def test_stream_with_content_is_relayed_in_order_and_unchanged(stub, make_client):
    up = anth_stream(texts=["hello ", "there"], stop="end_turn", in_tok=20, out_tok=3)
    _answer(stub, 200, SSE, up)
    c = make_client()
    st, hdrs, raw = c.request(MSG, _messages(True))
    assert st == 200 and raw == b"".join(up)
    assert hdrs["content-type"] == "text/event-stream"


def test_messages_explicit_error_from_any_provider_is_rewritten(stub, make_client):
    _answer(stub, 400, JS, json.dumps({"error": {"code": "1261", "message": "Prompt exceeds max length"}}).encode())
    c = make_client()
    st, _, raw = c.request(MSG, _messages(True))
    assert st == 400 and anth_err(raw).startswith("prompt is too long: ")


def test_messages_max_tokens_overflow_gets_anthropics_own_error(stub, make_client):
    _answer(stub, 400, JS, json.dumps({"error": {"message": (
        "This model's maximum context length is 65536 tokens. However, you requested 70000 tokens "
        "(40000 in the messages, 30000 in the completion).")}}).encode())
    c = make_client()
    st, hdrs, raw = c.request(MSG, _messages())
    assert st == 400 and anth_err(raw).startswith("input length and `max_tokens` exceed context limit: 40000 + 30000")
    assert hdrs["x-ctx-normalised"] == "max-tokens-overflow"


def test_chat_max_tokens_overflow_is_relayed_as_is(stub, make_client):
    body = json.dumps({"error": {"message": (
        "This model's maximum context length is 65536 tokens. However, you requested 70000 tokens "
        "(40000 in the messages, 30000 in the completion).")}}).encode()
    _answer(stub, 400, JS, body)
    c = make_client()
    st, _, raw = c.request(CHAT, _chat())
    assert st == 400 and raw == body


def test_non_api_paths_are_proxied_untouched(stub, make_client):
    body = b'{"data":[{"id":"stub-model"}],"error":"prompt is too long: 9 tokens > 8 maximum"}'
    _answer(stub, 200, JS, body)
    c = make_client()
    st, _, raw = c.request("/v1/models", method="GET")
    assert st == 200 and raw == body
    st, _, raw = c.request("/v1/messages/count_tokens", _messages())
    assert st == 200 and raw == body
    assert stub.seen[0]["path"] == "/v1/models"


def test_detection_can_be_disabled(stub, make_client):
    import ctxnorm
    _answer(stub, 200, JS, anth_message())
    c = make_client(config=ctxnorm.Config(enabled=False))
    st, _, raw = c.request(MSG, _messages())
    assert st == 200 and raw == anth_message()


def test_chat_stream_error_chunk_is_explicit(stub, make_client):
    _answer(stub, 200, SSE, [b'data: {"error": {"message": "prompt is too long: 300000 tokens > 262144 maximum"}}\n\n'])
    c = make_client()
    st, _, raw = c.request(CHAT, _chat(True))
    e = oai_err(raw)
    assert st == 400 and e["code"] == "context_length_exceeded" and "262144" in e["message"]


def test_chat_client_gets_openai_shape_from_anthropic_shaped_error(stub, make_client):
    err = {"type": "error", "error": {"type": "invalid_request_error",
                                      "message": "prompt is too long: 300000 tokens > 262144 maximum"}}
    _answer(stub, 400, JS, json.dumps(err).encode())
    c = make_client()
    st, _, raw = c.request(CHAT, _chat())
    e = oai_err(raw)
    assert st == 400 and e["code"] == "context_length_exceeded"
    assert "262144" in e["message"] and "300000" in e["message"]


def test_streaming_chat_request_asks_for_usage(stub, make_client):
    _answer(stub, 200, SSE, oai_stream(text="ok", finish="stop", prompt=5, completion=1))
    c = make_client()
    c.request(CHAT, _chat(True))
    assert stub.seen[-1]["body"]["stream_options"] == {"include_usage": True}
    c.request(CHAT, _chat(True, stream_options={"include_usage": False}))
    assert stub.seen[-1]["body"]["stream_options"] == {"include_usage": False}     # the client's choice wins


def test_client_credentials_are_forwarded_and_nothing_is_added(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client()
    c.request(MSG, _messages(), headers={"x-api-key": "dummy-key", "anthropic-version": "2023-06-01"})
    h = {k.lower(): v for k, v in stub.seen[0]["headers"].items()}
    assert h["x-api-key"] == "dummy-key" and h["anthropic-version"] == "2023-06-01"
    assert h["authorization"] == "Bearer dummy-not-a-key"


def test_unreachable_upstream_is_a_502(make_client, stub):
    c = make_client()
    stub.close()
    st, _, raw = c.request(MSG, _messages())
    assert st == 502 and b"upstream unreachable" in raw


# ------------------------------------------------------------------------------ per-model limits -----
def test_no_limits_request_is_unchanged(stub, make_client):
    _answer(stub, 200, JS, oai_json(text="done", finish="stop", prompt=20, completion=3))
    c = make_client()
    st, _, raw = c.request(MSG, _messages(max_tokens=200000, text="x" * 400000))
    assert st == 200, raw
    assert stub.seen[0]["body"]["max_tokens"] == 200000 and c.events == []


def test_messages_clamped_to_max_output(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"stub-model": Limits(context_window=1000000, max_output=8192)})
    st, _, _ = c.request(MSG, _messages(max_tokens=32000))
    assert st == 200 and stub.seen[0]["body"]["max_tokens"] == 8192
    assert _events(c, "output_capped") == [{"event": "output_capped", "protocol": "anthropic", "model": "stub-model",
                                            "max_output": 8192, "changes": ["max_tokens 32000->8192"]}]


def test_clamp_off_below_the_limit(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"*": Limits(max_output=8192)})
    c.request(MSG, _messages(max_tokens=4000))
    assert stub.seen[0]["body"]["max_tokens"] == 4000 and c.events == []


def test_messages_clamp_and_thinking_budget_fit(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"*": Limits(max_output=16000)})
    st, _, raw = c.request(MSG, _messages(max_tokens=64000, thinking={"type": "enabled", "budget_tokens": 30000}))
    assert st == 200, raw
    b = stub.seen[0]["body"]
    assert b["max_tokens"] == 16000 and b["thinking"] == {"type": "enabled", "budget_tokens": 14976}


def test_thinking_dropped_when_no_budget_fits(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"*": Limits(max_output=1500)})
    c.request(MSG, _messages(max_tokens=64000, thinking={"type": "enabled", "budget_tokens": 30000}))
    b = stub.seen[0]["body"]
    assert b["max_tokens"] == 1500 and "thinking" not in b


def test_chat_clamps_its_own_field(stub, make_client):
    _answer(stub, 200, JS, oai_json(text="done", finish="stop", prompt=20, completion=3))
    c = make_client(limits={"*": Limits(max_output=8192)})
    st, _, _ = c.request(CHAT, _chat(max_tokens=100000, field="max_completion_tokens"))
    assert st == 200
    assert stub.seen[0]["body"]["max_completion_tokens"] == 8192 and "max_tokens" not in stub.seen[0]["body"]
    assert len(_events(c, "output_capped")) == 1


def test_chat_max_tokens_clamped(stub, make_client):
    _answer(stub, 200, JS, oai_json(text="done", finish="stop", prompt=20, completion=3))
    c = make_client(limits={"*": Limits(max_output=4096)})
    c.request(CHAT, _chat(max_tokens=50000))
    assert stub.seen[0]["body"]["max_tokens"] == 4096


def test_precheck_refuses_when_the_prompt_does_not_fit(stub, make_client):
    c = make_client(limits={"*": Limits(context_window=4096)})
    st, hdrs, raw = c.request(MSG, _messages(text=BIG))
    assert st == 400
    m = re.fullmatch(r"prompt is too long: (\d+) tokens > 4096 maximum", anth_err(raw))
    assert m and int(m.group(1)) > 4096
    assert hdrs["x-ctx-normalised"] == "context-window"
    assert stub.seen == []                                              # never sent
    assert _events(c, "context_window_refused")[0]["limit"] == 4096


def test_precheck_refuses_chat_with_context_length_exceeded(stub, make_client):
    c = make_client(limits={"*": Limits(context_window=4096)})
    st, _, raw = c.request(CHAT, {"model": "m", "messages": [{"role": "user", "content": BIG}]})
    e = oai_err(raw)
    assert st == 400 and e["code"] == "context_length_exceeded" and "4096" in e["message"]
    assert stub.seen == []


def test_precheck_uses_the_requested_models_window(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"small": Limits(context_window=4096), "large": Limits(context_window=200000)})
    st, _, _ = c.request(MSG, dict(_messages(text=BIG), model="small"))
    assert st == 400 and stub.seen == []
    st, _, _ = c.request(MSG, dict(_messages(text=BIG), model="large"))
    assert st == 200 and len(stub.seen) == 1


def test_wildcard_limits_apply_to_unlisted_models(stub, make_client):
    c = make_client(limits={"listed": Limits(context_window=200000), "*": Limits(context_window=2048)})
    st, _, raw = c.request(MSG, dict(_messages(text=BIG), model="unlisted"))
    assert st == 400 and anth_err(raw).endswith("> 2048 maximum")


def test_precheck_allows_a_prompt_that_fits(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"*": Limits(context_window=20000)})
    st, _, _ = c.request(MSG, _messages(text=BIG, max_tokens=32000))   # est ~10k (+32k max_tokens NOT added)
    assert st == 200 and len(stub.seen) == 1 and c.events == []


def test_limits_can_be_switched_off(stub, make_client):
    _answer(stub, 200, JS, anth_message(text="ok", stop="end_turn", in_tok=5, out_tok=1))
    c = make_client(limits={"*": Limits(context_window=4096, max_output=100)}, enforce_limits=False)
    st, _, _ = c.request(MSG, _messages(text=BIG))
    assert st == 200 and stub.seen[0]["body"]["max_tokens"] == 32000


# ------------------------------------------------------------------------------ events ---------------
def test_explicit_exhaustion_without_limits_emits_one_event(stub, make_client):
    err = {"error": {"message": "This model's maximum context length is 131072 tokens. However, your messages "
                                "resulted in 140000 tokens.", "code": "context_length_exceeded"}}
    _answer(stub, 400, JS, json.dumps(err).encode())
    c = make_client()
    st, _, raw = c.request(MSG, _messages())
    assert st == 400 and anth_err(raw) == "prompt is too long: 140000 tokens > 131072 maximum"
    assert c.events == [{"event": "context_exhausted", "protocol": "anthropic", "model": "stub-model",
                         "kind": "explicit", "input_tokens": 140000, "limit": 131072, "status": 400}]


@pytest.mark.parametrize("stream", [True, False])
def test_implicit_exhaustion_is_one_event_not_also_an_output_limit_stop(stub, make_client, stream):
    _answer(stub, 200, SSE if stream else JS, oai_stream() if stream else oai_json())
    c = make_client(limits={"*": Limits(context_window=WINDOW, max_output=32000)})
    st, _, _ = c.request(CHAT, _chat(stream))
    assert st == 400
    assert [e["event"] for e in c.events] == ["context_exhausted"]
    assert c.events[0]["kind"] == "implicit" and c.events[0]["requested_max"] == 32000


@pytest.mark.parametrize("stream", [True, False])
def test_upstream_length_stop_at_max_output_is_reported(stub, make_client, stream):
    text = "word " * 2000
    _answer(stub, 200, SSE if stream else JS, oai_stream(text=text, completion=8192, prompt=50) if stream
            else oai_json(text=text, completion=8192, prompt=50))
    c = make_client(limits={"*": Limits(max_output=8192)})
    st, _, raw = c.request(CHAT, _chat(stream, max_tokens=8192))      # no clamp: the budget IS the limit
    assert st == 200 and b"word word" in raw
    assert _events(c, "output_limit_stop") == [{"event": "output_limit_stop", "protocol": "openai-chat",
                                                "model": "stub-model", "stop": "length", "max_output": 8192}]


def test_length_stop_below_max_output_is_not_reported(stub, make_client):
    _answer(stub, 200, JS, oai_json(text="word " * 2000, completion=4000, prompt=50))
    c = make_client(limits={"*": Limits(max_output=8192)})
    st, _, _ = c.request(CHAT, _chat(max_tokens=4000))
    assert st == 200 and c.events == []


def test_clamped_request_stopping_at_the_limit_is_reported_once_and_bytes_unchanged(stub, make_client):
    an = anth_message(text="long " * 900, stop="max_tokens", in_tok=10, out_tok=4096)
    _answer(stub, 200, JS, an)
    c = make_client(limits={"*": Limits(max_output=4096)})
    st, _, raw = c.request(MSG, _messages(max_tokens=64000))
    assert st == 200 and raw == an
    assert [e["event"] for e in c.events] == ["output_capped"]            # clamp + stop: one event


def test_upstream_output_limit_refusal_is_reported_and_relayed(stub, make_client):
    err = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                 "message": "max_tokens: 64000 > 32000, which is the maximum "
                                                            "allowed number of output tokens for stub-model"}}).encode()
    _answer(stub, 400, JS, err)
    c = make_client()
    st, _, raw = c.request(MSG, _messages(max_tokens=64000))
    assert st == 400 and raw == err
    assert c.events == [{"event": "output_refused", "protocol": "anthropic", "model": "stub-model",
                         "max_output": 32000, "upstream_status": 400}]


def test_events_carry_sizes_never_content(stub, make_client):
    secret_prompt = "PROMPT-TEXT-MUST-NOT-LEAK " * 50
    _answer(stub, 400, JS, json.dumps({"error": {"message": "Input is too long for requested model."}}).encode())
    c = make_client(limits={"*": Limits(max_output=100)})
    c.request(MSG, _messages(text=secret_prompt, max_tokens=5000))
    dumped = json.dumps(c.events)
    assert c.events and "PROMPT-TEXT" not in dumped and "too long" not in dumped


def test_a_failing_event_hook_does_not_break_requests(stub, make_client):
    _answer(stub, 200, JS, anth_message())
    c = make_client()

    def boom(ev):
        raise RuntimeError("hook down")
    c.proxy.on_event = boom
    st, _, raw = c.request(MSG, _messages())
    assert st == 400 and anth_err(raw).startswith("prompt is too long")
