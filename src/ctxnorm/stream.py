# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Streaming: decide whether a server-sent-events answer is context exhaustion BEFORE committing a 200.

An implicit exhaustion (an empty length stop) is only known at the END of a stream. `lookahead` holds back
a bounded prefix of the stream until meaningful content arrives (then everything is relayed unchanged; a
committed stream is never converted) or the stream ends (then the caller can still answer 400).

Two stream dialects are understood: Anthropic Messages SSE (``protocol="anthropic"``) and OpenAI Chat
Completions SSE (``protocol="openai-chat"``).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union

from .core import (
    _JSON_ERRORS,
    DEFAULT_CONFIG,
    Config,
    ContextExhausted,
    _d,
    _slen,
    _toint,
    anthropic_usage_input,
    eligible,
    from_error,
    from_stop,
    openai_chat_content_chars,
    out_threshold,
)

__all__ = ["SSEWatch", "AnthropicSSEWatch", "OpenAIChatSSEWatch", "lookahead", "PROTOCOLS"]


class SSEWatch:
    """Incremental SSE parser base: collects `data:` lines per event, robust to events split across chunks
    and to CRLF framing. Subclasses interpret each event's JSON and keep:

    chars       characters of meaningful content seen so far
    stop        the stop / finish reason, once seen
    in_tokens   the largest input size reported
    out_tokens  the largest output size reported
    error       a mid-stream error event (parsed dict), if any
    """

    def __init__(self) -> None:
        self._buf = b""
        self._data: List[bytes] = []
        self.chars = 0
        self.stop: Optional[str] = None
        self.in_tokens = 0
        self.out_tokens = 0
        self.error: Optional[dict] = None

    def feed(self, chunk: bytes) -> None:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "replace")
        elif not isinstance(chunk, (bytes, bytearray)):
            return
        self._buf += chunk
        while True:
            i = self._buf.find(b"\n")
            if i < 0:
                break
            line, self._buf = self._buf[:i].rstrip(b"\r"), self._buf[i + 1:]
            if not line:
                self._flush()
            elif line.startswith(b"data:"):
                self._data.append(line[5:].strip())

    def finish(self) -> None:
        line = self._buf.strip()
        if line.startswith(b"data:"):
            self._data.append(line[5:].strip())
        self._buf = b""
        self._flush()

    def _flush(self) -> None:
        if not self._data:
            return
        raw, self._data = b"\n".join(self._data), []
        if raw == b"[DONE]":
            return
        try:
            d = json.loads(raw)
        except _JSON_ERRORS:
            return
        if isinstance(d, dict):
            self.event(d)

    def event(self, d: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def _int(v: Any) -> int:
        return _toint(v)

    def _set_stop(self, v: Any) -> None:
        if isinstance(v, str) and v:
            self.stop = v


class AnthropicSSEWatch(SSEWatch):
    """Anthropic Messages stream: message_start / content_block_* / message_delta / error events."""

    def event(self, d: dict) -> None:
        t = d.get("type")
        if t == "message_start":
            u = _d(_d(d.get("message")).get("usage"))
            self.in_tokens = max(self.in_tokens, anthropic_usage_input(u))
            self.out_tokens = max(self.out_tokens, self._int(u.get("output_tokens")))
        elif t == "content_block_start":
            cb = _d(d.get("content_block"))
            self.chars += _slen(cb.get("text")) + _slen(cb.get("thinking")) + _slen(cb.get("name"))
        elif t == "content_block_delta":
            dl = _d(d.get("delta"))
            self.chars += _slen(dl.get("text")) + _slen(dl.get("thinking")) + _slen(dl.get("partial_json"))
        elif t == "message_delta":
            self._set_stop(_d(d.get("delta")).get("stop_reason"))
            u = _d(d.get("usage"))
            self.in_tokens = max(self.in_tokens, anthropic_usage_input(u))
            self.out_tokens = max(self.out_tokens, self._int(u.get("output_tokens")))
        elif t == "error":
            self.error = d


class OpenAIChatSSEWatch(SSEWatch):
    """OpenAI Chat Completions stream: chat.completion.chunk objects (usage arrives in the last chunk
    when the request set ``stream_options.include_usage``), or an error. An error may come alone
    (``{"error": ...}``) or inside a chunk next to ``choices`` with ``finish_reason: "error"`` (OpenRouter)."""

    def event(self, d: dict) -> None:
        if d.get("error"):
            self.error = d
            return
        choices = d.get("choices")
        for ch in choices if isinstance(choices, list) else []:
            if not isinstance(ch, dict):
                continue
            self.chars += openai_chat_content_chars(ch.get("delta"))
            self._set_stop(ch.get("finish_reason"))
        u = d.get("usage")
        if isinstance(u, dict):
            self.in_tokens = max(self.in_tokens, self._int(u.get("prompt_tokens")))
            self.out_tokens = max(self.out_tokens, self._int(u.get("completion_tokens")))


PROTOCOLS = {"anthropic": AnthropicSSEWatch, "openai-chat": OpenAIChatSSEWatch}

Verdict = Union[Tuple[str, ContextExhausted, None], Tuple[str, List[bytes], Iterator[bytes]]]


def lookahead(chunks: Iterable[bytes], requested_max: Any, config: Optional[Config] = None,
              usage: Optional[Callable[[], dict]] = None, protocol: str = "anthropic") -> Verdict:
    """Hold back the start of an SSE stream until it is safe to commit a 200.

    Returns ``("exhausted", info, None)`` when the stream ended as context exhaustion (implicit) or carried
    a context-too-long error event before any meaningful content (explicit): nothing has been relayed,
    the caller answers 400 (e.g. ``info.to_anthropic()``). Otherwise ``("pass", held, rest)``: relay
    `held`, then iterate `rest` unchanged; the bytes are exactly the upstream's.

    It commits (stops holding) as soon as the content produced exceeds the implicit threshold or the held
    bytes reach ``config.lookahead_bytes``; after that the stream is never converted. A request that is not
    `eligible` (small max_tokens, detection disabled) is never held. `usage` (optional callable returning
    ``{"in_tokens": int, "out_tokens": int, "stop": str}``, any key optional) supplies numbers only known at
    the end, e.g. from a protocol translator that sits between the upstream and this stream."""
    c = config if config is not None else DEFAULT_CONFIG
    it = iter(chunks)
    if not eligible(requested_max, c):
        return "pass", [], it
    try:
        w = PROTOCOLS[protocol]()
    except KeyError:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}") from None
    thr_chars = out_threshold(requested_max, c) * c.chars_per_token
    held: List[bytes] = []
    size = 0
    for chunk in it:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "replace")
        if not chunk or not isinstance(chunk, (bytes, bytearray)):
            continue
        held.append(chunk)
        size += len(chunk)
        w.feed(chunk)
        if w.error is not None and w.chars <= thr_chars:
            info = from_error(400, w.error, c)
            if info:
                return "exhausted", info, None
            return "pass", held, it
        if w.chars > thr_chars or size >= c.lookahead_bytes:
            return "pass", held, it
    w.finish()
    if w.error is not None and w.chars <= thr_chars:
        info = from_error(400, w.error, c)
        if info:
            return "exhausted", info, None
    stop, inp, out = w.stop, w.in_tokens, w.out_tokens
    if usage is not None:
        try:
            m = _d(usage())
        except Exception:
            m = {}
        if isinstance(m.get("stop"), str) and m["stop"]:
            stop = m["stop"]
        inp = max(inp, _toint(m.get("in_tokens")))
        out = max(out, _toint(m.get("out_tokens")))
    info = from_stop(stop, out, inp, requested_max, w.chars, c)
    if info:
        return "exhausted", info, None
    return "pass", held, iter(())
