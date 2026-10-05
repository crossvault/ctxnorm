# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Detection and normalisation: explicit provider errors, implicit empty length stops, error shapes."""
import json

import pytest

import ctxnorm
from conftest import load_vectors

EXPLICIT = load_vectors("explicit.json")
NEGATIVE = load_vectors("negative.json")


def _raw(body):
    return body.encode() if isinstance(body, str) else json.dumps(body).encode()


# ------------------------------------------------------------------------------------- explicit -----
@pytest.mark.parametrize("v", EXPLICIT, ids=[v["provider"] for v in EXPLICIT])
def test_explicit_phrasings_are_detected(v):
    info = ctxnorm.from_error(v["status"], _raw(v["body"]))
    assert info and info.kind == "explicit" and info.status == v["status"]
    if v["expect"]["input_tokens"] is not None:
        assert info.input_tokens == v["expect"]["input_tokens"]
    if v["expect"]["limit"] is not None:
        assert info.limit == v["expect"]["limit"]
    st, obj = info.to_anthropic(est_input=5000)
    assert st == 400 and obj["type"] == "error" and obj["error"]["type"] == "invalid_request_error"
    n, m = info.normalized(5000)
    assert obj["error"]["message"] == f"prompt is too long: {n} tokens > {m} maximum" and n > m


@pytest.mark.parametrize("v", EXPLICIT[:3], ids=[v["provider"] for v in EXPLICIT[:3]])
def test_explicit_accepts_bytes_str_and_parsed_json(v):
    for body in (_raw(v["body"]), _raw(v["body"]).decode(), v["body"]):
        assert ctxnorm.from_error(v["status"], body)


def test_openrouter_split_input_counts_text_plus_tool_input():
    info = ctxnorm.from_error(400, json.dumps({"error": {"message": (
        "This endpoint's maximum context length is 131072 tokens. However, you requested about 150000 tokens "
        "(135000 of text input, 5000 of tool input, 10000 in the output).")}}).encode())
    assert info.input_tokens == 140000 and info.completion_tokens == 10000


def test_openrouter_nested_provider_raw_json():
    body = {"error": {"message": "Provider returned error", "code": 400, "metadata": {
        "raw": json.dumps({"error": {"message": "prompt is too long: 210000 tokens > 200000 maximum"}}),
        "provider_name": "X"}}}
    info = ctxnorm.from_error(400, json.dumps(body).encode())
    assert info and info.input_tokens == 210000 and info.limit == 200000


def test_sse_framed_error_body():
    body = b'event: error\ndata: {"type":"error","error":{"message":"prompt is too long: 9 tokens > 8 maximum"}}\n\n'
    info = ctxnorm.from_error(400, body)
    assert info and (info.input_tokens, info.limit) == (9, 8)


@pytest.mark.parametrize("v", NEGATIVE, ids=[v["case"] for v in NEGATIVE])
def test_explicit_negatives(v):
    assert ctxnorm.from_error(v["status"], _raw(v["body"])) is None


def test_prompt_plus_completion_overflow_maps_to_anthropic_max_tokens_error():
    info = ctxnorm.from_error(400, json.dumps({"error": {"message": (
        "This model's maximum context length is 65536 tokens. However, you requested 70000 tokens "
        "(40000 in the messages, 30000 in the completion). Please reduce the length of the messages or "
        "completion.")}}).encode())
    assert info.is_max_tokens_overflow
    st, obj = info.to_anthropic_max_tokens()
    assert st == 400 and obj["error"]["message"].startswith(
        "input length and `max_tokens` exceed context limit: 40000 + 30000 > 65536")


def test_explicit_detection_can_be_disabled():
    body = _raw(EXPLICIT[0]["body"])
    assert ctxnorm.from_error(400, body, ctxnorm.Config(enabled=False)) is None


# ------------------------------------------------------------------------------------- implicit -----
def test_implicit_tiny_output_on_length_stop_triggers():
    info = ctxnorm.from_stop("length", 6, 409594, 32000)
    assert info and info.input_tokens == 409594 and info.limit == 409600
    assert info.to_anthropic()[1]["error"]["message"] == "prompt is too long: 409594 tokens > 409600 maximum"
    assert ctxnorm.from_stop("max_tokens", 0, 500000, 32000)                 # empty answer
    assert ctxnorm.from_stop("max_output_tokens", 300, 900000, 32000)        # under 1% of 32000 (320)


def test_implicit_threshold_is_relative_and_floored():
    assert ctxnorm.out_threshold(1024) == 16 and ctxnorm.out_threshold(32000) == 320
    assert ctxnorm.out_threshold(128000) == 1280
    assert ctxnorm.from_stop("length", 16, 1000, 1024) and not ctxnorm.from_stop("length", 17, 1000, 1024)
    assert not ctxnorm.from_stop("length", 321, 1000, 32000)


def test_small_requested_max_tokens_never_triggers():
    """A caller that asked for a few tokens legitimately stops on max_tokens with little output."""
    assert ctxnorm.from_stop("length", 5, 900000, 16) is None
    assert ctxnorm.from_stop("max_tokens", 0, 900000, 1023) is None
    assert ctxnorm.from_stop("max_tokens", 0, 900000, None) is None
    assert ctxnorm.from_stop("max_tokens", 0, 900000, True) is None


def test_meaningful_content_never_triggers():
    assert ctxnorm.from_stop("max_tokens", 0, 900000, 32000, content_chars=5000) is None
    msg = {"type": "message", "content": [{"type": "text", "text": "x" * 4000}], "stop_reason": "max_tokens",
           "usage": {"input_tokens": 900000, "output_tokens": 0}}
    assert ctxnorm.from_anthropic_message(msg, 32000) is None
    msg["content"] = [{"type": "text", "text": ""}]
    assert ctxnorm.from_anthropic_message(msg, 32000)


def test_other_stop_reasons_and_missing_input_never_trigger():
    for stop in ("end_turn", "stop", "tool_use", None):
        assert ctxnorm.from_stop(stop, 0, 900000, 32000) is None
    assert ctxnorm.from_stop("length", 0, 0, 32000) is None                  # no input size reported
    assert ctxnorm.from_stop("length", 0, None, 32000) is None
    assert ctxnorm.from_stop("length", "x", 5, 32000) is None


def test_cached_input_counts_toward_context():
    msg = {"type": "message", "content": [], "stop_reason": "max_tokens",
           "usage": {"input_tokens": 10, "cache_read_input_tokens": 400000, "cache_creation_input_tokens": 9584,
                     "output_tokens": 6}}
    info = ctxnorm.from_anthropic_message(msg, 32000)
    assert info.input_tokens == 409594 and info.limit == 409600


def test_openai_chat_response_implicit():
    resp = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 409594, "completion_tokens": 6}}
    info = ctxnorm.from_openai_chat(resp, 32000)
    assert info and info.limit == 409600
    resp["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "f", "arguments": "a" * 4000}}]
    assert ctxnorm.from_openai_chat(resp, 32000) is None                   # a tool call was produced
    assert ctxnorm.from_openai_chat({"error": {"message": "x"}}, 32000) is None
    assert ctxnorm.from_openai_chat({"choices": []}, 32000) is None


def test_thresholds_are_configurable():
    cfg = ctxnorm.Config(min_out=64, out_frac=0.05, min_request=4096)
    assert ctxnorm.out_threshold(32000, cfg) == 1600
    assert ctxnorm.from_stop("length", 1500, 1000, 32000, config=cfg)
    assert ctxnorm.from_stop("length", 10, 1000, 2048, config=cfg) is None   # below the new min_request
    off = ctxnorm.Config(enabled=False)
    assert ctxnorm.from_stop("length", 0, 1000, 32000, config=off) is None
    assert ctxnorm.lookahead([b"x"], 32000, off)[0] == "pass"


def test_config_clamps_nonsense_instead_of_raising():
    c = ctxnorm.Config(min_out=-5, out_frac=-1, min_request=0, chars_per_token=0, lookahead_bytes=1)
    assert (c.min_out, c.out_frac, c.min_request, c.chars_per_token, c.lookahead_bytes) == (0, 0.0, 1, 1.0, 4096)


# ------------------------------------------------------------------------------------- shapes -------
def test_openai_error_shape():
    st, obj = ctxnorm.from_stop("length", 6, 409594, 32000).to_openai()
    assert st == 400 and obj["error"]["code"] == "context_length_exceeded"
    assert obj["error"]["type"] == "invalid_request_error" and "409600" in obj["error"]["message"]


def test_normalized_fills_missing_numbers():
    E = ctxnorm.ContextExhausted
    assert E("explicit").normalized(5000) == (5000, 4999)                  # nothing known: estimate
    assert E("explicit", limit=8192).normalized(100) == (8193, 8192)       # input at least limit + 1
    assert E("explicit", input_tokens=9000).normalized() == (9000, 8999)
    assert E("explicit", input_tokens=100, limit=200).normalized() == (201, 200)
    assert E("explicit").normalized() == (1, 1)


def test_summary_is_sizes_only():
    info = ctxnorm.from_error(400, _raw(EXPLICIT[0]["body"]))
    s = info.summary()
    assert s == {"kind": "explicit", "input_tokens": 140000, "limit": 131072, "status": 400}
    assert "message" not in json.dumps(s)
