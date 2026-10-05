# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Synthetic provider responses used across the tests (no real model output)."""
from __future__ import annotations

import json
from typing import List

FULL_PROMPT, WINDOW = 409594, 409600          # a synthetic 400k window filled to the last 6 tokens
SSE = [("Content-Type", "text/event-stream")]
JS = [("Content-Type", "application/json")]


def sse(ev: str, data: dict) -> bytes:
    data = dict(data)
    data["type"] = ev
    return f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode()


def anth_stream(texts=(), stop="end_turn", in_tok=FULL_PROMPT, out_tok=6) -> List[bytes]:
    """An Anthropic Messages SSE stream as a list of event chunks."""
    out = [sse("message_start", {"message": {"id": "m", "type": "message", "role": "assistant", "content": [],
                                             "model": "x", "usage": {"input_tokens": in_tok, "output_tokens": 0}}}),
           sse("ping", {})]
    if texts:
        out.append(sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}))
        for t in texts:
            out.append(sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": t}}))
        out.append(sse("content_block_stop", {"index": 0}))
    out.append(sse("message_delta", {"delta": {"stop_reason": stop}, "usage": {"output_tokens": out_tok}}))
    out.append(sse("message_stop", {}))
    return out


def anth_message(text="", stop="max_tokens", in_tok=FULL_PROMPT, out_tok=6, model="stub-model") -> bytes:
    content = [{"type": "text", "text": text}] if text else []
    return json.dumps({"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
                       "stop_reason": stop, "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}).encode()


def oai_chunk(delta, finish=None, usage=None) -> bytes:
    d = {"id": "c1", "object": "chat.completion.chunk", "model": "stub-model",
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None else []}
    if usage:
        d["usage"] = usage
    return b"data: " + json.dumps(d).encode() + b"\n\n"


def oai_stream(text="", finish="length", prompt=FULL_PROMPT, completion=6) -> List[bytes]:
    """An OpenAI Chat Completions SSE stream (with the include_usage chunk) as a list of chunks."""
    return [oai_chunk({"role": "assistant", "content": text}), oai_chunk({}, finish),
            oai_chunk(None, usage={"prompt_tokens": prompt, "completion_tokens": completion}),
            b"data: [DONE]\n\n"]


def oai_json(text="", finish="length", prompt=FULL_PROMPT, completion=6) -> bytes:
    return json.dumps({"id": "c1", "object": "chat.completion", "model": "stub-model",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                    "finish_reason": finish}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}).encode()


def anth_err(raw: bytes) -> str:
    e = json.loads(raw)
    assert e["type"] == "error" and e["error"]["type"] == "invalid_request_error"
    return e["error"]["message"]


def oai_err(raw: bytes) -> dict:
    e = json.loads(raw)
    assert "type" not in e and e["error"]["type"] == "invalid_request_error"
    return e["error"]
