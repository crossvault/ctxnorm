# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Per-model limit helpers: clamp, thinking-budget fit, input estimate, pre-check, upstream signals."""
import json

import pytest

from ctxnorm import limits as L


def test_limits_from_mapping_keeps_only_valid_positive_ints():
    assert L.Limits.from_mapping({}) == L.Limits() and L.Limits.from_mapping(None) == L.Limits()
    assert L.Limits.from_mapping("x") == L.Limits()
    t = {"context_window": 200000, "max_output": 64000, "max_output_default": 32000, "junk": 5}
    assert L.Limits.from_mapping(t) == L.Limits(200000, 64000, 32000)
    assert not L.Limits.from_mapping({"context_window": 0, "max_output": True, "max_output_default": -1})
    assert not L.Limits.from_mapping({"context_window": L.MAX_LIMIT_TOKENS + 1})


def test_clamp_anthropic_lowers_never_raises():
    b = {"max_tokens": 64000}
    assert L.clamp_output(b, 32000) == ["max_tokens 64000->32000"] and b["max_tokens"] == 32000
    b = {"max_tokens": 1000}
    assert L.clamp_output(b, 32000) == [] and b == {"max_tokens": 1000}
    b = {}
    assert L.clamp_output(b, 32000) == [] and b == {}                  # a missing field stays missing
    assert L.clamp_output({"max_tokens": 5}, 0) == []                   # invalid limit: untouched


def test_clamp_thinking_budget_is_lowered_below_max_tokens():
    th = {"type": "enabled", "budget_tokens": 30000}
    b = {"max_tokens": 64000, "thinking": th}
    done = L.clamp_output(b, 16000)
    assert b["max_tokens"] == 16000 and b["thinking"] == {"type": "enabled", "budget_tokens": 14976}
    assert th["budget_tokens"] == 30000                                 # nested object replaced, not edited
    assert done == ["max_tokens 64000->16000", "thinking.budget_tokens 30000->14976"]
    b = {"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 8000}}
    L.clamp_output(b, 16000)
    assert b["thinking"]["budget_tokens"] == 8000                       # still < max_tokens: untouched


def test_clamp_thinking_dropped_when_no_budget_fits():
    b = {"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 30000}}
    assert L.clamp_output(b, 1500)[-1] == "thinking dropped (max_tokens leaves no budget)"
    assert b == {"max_tokens": 1500}
    b = {"max_tokens": 64000, "thinking": {"type": "adaptive"}}
    L.clamp_output(b, 1500)
    assert b["thinking"] == {"type": "adaptive"}                        # no budget to fit


def test_clamp_openai_fields_per_protocol():
    b = {"max_tokens": 50000, "max_completion_tokens": 90000}
    assert len(L.clamp_output(b, 8192, "openai-chat")) == 2
    assert b == {"max_tokens": 8192, "max_completion_tokens": 8192}
    b = {"max_output_tokens": 90000, "max_tokens": 90000}
    L.clamp_output(b, 8192, "openai-responses")
    assert b == {"max_output_tokens": 8192, "max_tokens": 90000}        # only the protocol's own field


def test_estimate_is_conservative_and_skips_binary_payloads():
    text = "x" * 40000
    body = {"model": "m", "max_tokens": 32000, "system": "s" * 400,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 900000}}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "t" * 400, "signature": "S" * 5000},
                                                  {"type": "redacted_thinking", "data": "R" * 5000}]}],
            "tools": [{"name": "get", "description": "d" * 400, "input_schema": {"type": "object"}}]}
    est = L.estimate_input_tokens(body)
    assert 10300 <= est <= 10400                            # ~41.6k prompt chars / 4, no image/signature bytes
    assert est < len(json.dumps(body)) // 4
    assert L.estimate_input_tokens({"messages": [{"role": "user", "content": "y" * 4000}]}, 8) == 500
    oai = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "z" * 4000},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 90000}}]}]}
    assert 1000 <= L.estimate_input_tokens(oai) < 1010
    assert L.estimate_input_tokens("not a body") == 0


def test_fits_window_compares_the_estimate_alone():
    assert L.fits_window(1000, L.Limits(context_window=1000))
    assert not L.fits_window(1001, L.Limits(context_window=1000))
    assert L.fits_window(10 ** 9, L.Limits(max_output=10))              # no window: always sent
    assert L.fits_window(10 ** 9, None)


def test_precheck_returns_an_estimated_exhaustion():
    body = {"messages": [{"role": "user", "content": "x" * 40000}]}
    info = L.precheck(body, L.Limits(context_window=4096))
    assert info and info.kind == "estimated" and info.limit == 4096 and info.input_tokens == 10001
    assert info.to_anthropic()[1]["error"]["message"] == "prompt is too long: 10001 tokens > 4096 maximum"
    assert L.precheck(body, L.Limits(context_window=20000)) is None
    assert L.precheck(body, None) is None


def test_stopped_at_output_limit():
    lim = L.Limits(max_output=8192)
    assert L.stopped_at_output_limit("max_tokens", 8192, lim)
    assert L.stopped_at_output_limit("length", 8192, lim)
    assert not L.stopped_at_output_limit("max_tokens", 4000, lim)      # the client's own smaller budget
    assert not L.stopped_at_output_limit("end_turn", 8192, lim)
    assert L.stopped_at_output_limit("length", None, L.Limits(max_output_default=4096))
    assert not L.stopped_at_output_limit("length", None, L.Limits(context_window=4096))
    assert not L.stopped_at_output_limit("length", 8192, None)


def test_stop_in_tail():
    assert L.stop_in_tail(b'..."stop_reason":"end_turn"...{"delta":{"stop_reason":"max_tokens"}}') == "max_tokens"
    assert L.stop_in_tail(b'data: {"choices":[{"finish_reason":"length"}]}') == "length"
    assert L.stop_in_tail(b'{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}') \
        == "max_output_tokens"
    assert L.stop_in_tail(b"") is None and L.stop_in_tail(b"{}") is None


@pytest.mark.parametrize("text,expected", [
    ("max_tokens must be <= 131072", {"max_output": 131072}),
    ("max_tokens must be less than or equal to 65,536", {"max_output": 65536}),
    ("max_tokens: 200000 > 128000, which is the maximum allowed number of output tokens for model-x",
     {"max_output": 128000}),
    ("This model's maximum context length is 262144 tokens.", {"context_window": 262144}),
    ("prompt is too long: 300000 tokens > 262144 maximum", {"context_window": 262144}),
    ("max_tokens must be less than or equal to 65,536; This model's maximum context length is 262144 tokens",
     {"max_output": 65536, "context_window": 262144}),
    ("rate limited", {}),
    (None, {}),
])
def test_parse_limit_hint(text, expected):
    assert L.parse_limit_hint(text) == expected


def test_output_refusal_ignores_context_overflow():
    assert L.output_refusal("max_tokens must be <= 8192") == 8192
    assert L.output_refusal("input length and `max_tokens` exceed context limit: 190000 + 32000 > 200000, "
                            "decrease input length or `max_tokens` and try again") is None
    assert L.output_refusal("maximum context length is 131072 tokens") is None
