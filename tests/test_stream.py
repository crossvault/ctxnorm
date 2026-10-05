# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""The streaming look-ahead, for Anthropic and OpenAI Chat SSE streams."""
import pytest

import ctxnorm
from payloads import anth_stream, oai_chunk, oai_stream, sse


def test_lookahead_empty_length_stop_is_exhausted():
    v, info, rest = ctxnorm.lookahead(iter(anth_stream(stop="max_tokens")), 32000)
    assert v == "exhausted" and rest is None and info.input_tokens == 409594 and info.limit == 409600


def test_lookahead_handles_events_split_across_chunks_and_crlf():
    raw = b"".join(anth_stream(stop="max_tokens")).replace(b"\n", b"\r\n")
    chunks = [raw[i:i + 7] for i in range(0, len(raw), 7)]
    assert ctxnorm.lookahead(iter(chunks), 32000)[0] == "exhausted"


def test_lookahead_commits_on_meaningful_content_and_never_converts_after():
    events = anth_stream(texts=["x" * 1500, "y" * 1500], stop="max_tokens", out_tok=6)   # misreported usage
    consumed = []

    def gen():
        for e in events:
            consumed.append(e)
            yield e
    v, held, rest = ctxnorm.lookahead(gen(), 32000)
    assert v == "pass"
    assert len(consumed) < len(events)                   # committed before the end
    assert b"".join(held) + b"".join(rest) == b"".join(events)      # bytes unchanged


def test_lookahead_legit_endings_pass_unchanged():
    for evs in (anth_stream(texts=["hi"], stop="end_turn"),              # short but a normal end
                anth_stream(stop="max_tokens", in_tok=0)):               # no input size reported
        v, held, rest = ctxnorm.lookahead(iter(evs), 32000)
        assert v == "pass" and b"".join(held) + b"".join(rest) == b"".join(evs)


def test_lookahead_small_max_tokens_is_not_buffered():
    consumed = []

    def gen():
        for e in anth_stream(stop="max_tokens"):
            consumed.append(e)
            yield e
    v, held, rest = ctxnorm.lookahead(gen(), 16)
    assert v == "pass" and held == [] and consumed == []       # nothing held: relayed as it arrives


def test_lookahead_is_bounded_by_bytes():
    pings = [sse("ping", {})] * 400 + anth_stream(stop="max_tokens")
    v, held, rest = ctxnorm.lookahead(iter(pings), 32000, ctxnorm.Config(lookahead_bytes=4096))
    assert v == "pass" and 4096 <= sum(map(len, held)) < 4096 + 200


def test_lookahead_usage_callback_supplies_late_numbers():
    evs = anth_stream(stop="max_tokens", in_tok=0, out_tok=3)          # e.g. a translator: input 0 up front
    meta = {"in_tokens": 777777, "out_tokens": 3, "stop": "max_tokens"}
    v, info, _ = ctxnorm.lookahead(iter(evs), 32000, usage=lambda: meta)
    assert v == "exhausted" and info.input_tokens == 777777 and info.limit == 777780


def test_lookahead_usage_callback_errors_are_ignored():
    def boom():
        raise RuntimeError("x")
    v, info, _ = ctxnorm.lookahead(iter(anth_stream(stop="max_tokens")), 32000, usage=boom)
    assert v == "exhausted"


def test_lookahead_mid_stream_context_error_event_is_explicit():
    evs = [sse("message_start", {"message": {"usage": {"input_tokens": 5}}}),
           sse("error", {"error": {"type": "invalid_request_error",
                                   "message": "prompt is too long: 300000 tokens > 262144 maximum"}})]
    v, info, _ = ctxnorm.lookahead(iter(evs), 32000)
    assert v == "exhausted" and info.kind == "explicit" and info.limit == 262144


def test_lookahead_other_error_event_passes():
    evs = [sse("error", {"error": {"type": "overloaded_error", "message": "Overloaded"}})]
    v, held, rest = ctxnorm.lookahead(iter(evs), 32000)
    assert v == "pass" and b"".join(held) + b"".join(rest) == evs[0]


# -- OpenAI Chat Completions streams --------------------------------------------------------------------
def test_openai_stream_empty_length_is_exhausted():
    v, info, _ = ctxnorm.lookahead(iter(oai_stream()), 32000, protocol="openai-chat")
    assert v == "exhausted" and (info.input_tokens, info.limit) == (409594, 409600)


def test_openai_stream_with_content_passes_unchanged():
    evs = oai_stream(text="word " * 2000, completion=2000)
    v, held, rest = ctxnorm.lookahead(iter(evs), 32000, protocol="openai-chat")
    assert v == "pass" and b"".join(held) + b"".join(rest) == b"".join(evs)


def test_openai_stream_without_usage_cannot_be_judged_and_passes():
    evs = [oai_chunk({"role": "assistant", "content": ""}), oai_chunk({}, "length"), b"data: [DONE]\n\n"]
    assert ctxnorm.lookahead(iter(evs), 32000, protocol="openai-chat")[0] == "pass"


def test_openai_stream_error_object_is_explicit():
    evs = [b'data: {"error": {"message": "This model\'s maximum context length is 8192 tokens", '
           b'"code": "context_length_exceeded"}}\n\n']
    v, info, _ = ctxnorm.lookahead(iter(evs), 32000, protocol="openai-chat")
    assert v == "exhausted" and info.limit == 8192


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError):
        ctxnorm.lookahead(iter([]), 32000, protocol="smoke-signals")
