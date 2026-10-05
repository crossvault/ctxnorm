# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Detect context-window exhaustion in an LLM provider's answer and normalise it into one shape.

Two ways a provider tells you the prompt no longer fits:

1. EXPLICIT: an error response whose text says the context / prompt is too long (`from_error`).
   Phrasings from OpenAI, Anthropic(-compatible), OpenRouter, Mistral, vLLM, TGI, llama.cpp, DeepSeek,
   Qwen/DashScope, GLM/Zhipu, Moonshot/Kimi, Gemini, xAI and Bedrock are recognised. Token numbers are
   extracted when present.
2. IMPLICIT: a *successful* answer that stopped on a length / max_tokens stop while producing (almost)
   nothing, with the provider reporting its input size (`from_stop`, `from_anthropic_message`,
   `from_openai_chat`). Some providers do this instead of refusing: they clamp the output budget to the
   few tokens left in the window and return an empty turn. Legitimate max_tokens stops are excluded: the
   caller asked for a small budget, or meaningful content was produced.

Either way you get a `ContextExhausted`, which renders as the error a client knows how to recover from:
Anthropic's ``prompt is too long: N tokens > M maximum`` (clients such as Claude Code compact the
conversation and retry on it) or OpenAI's ``context_length_exceeded``.

The module never knows a model's window; it only reads what the provider says. Pure stdlib.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

__all__ = [
    "STOP_LENGTH", "Config", "ContextExhausted", "out_threshold", "eligible", "error_text", "from_error",
    "from_stop", "anthropic_message_content_chars", "anthropic_usage_input", "from_anthropic_message",
    "openai_chat_content_chars", "from_openai_chat",
]

#: Stop / finish reasons that mean "the output budget ran out".
STOP_LENGTH = ("max_tokens", "length", "max_output_tokens", "model_length")

Body = Union[bytes, bytearray, str, dict, list, None]

# Every public function must return a result or None on hostile input, never raise. These helpers make
# "a nested value of the wrong type" and "a number that is not a number" harmless.
_JSON_ERRORS = (ValueError, RecursionError)


def _d(x: Any) -> dict:
    """`x` if it is a dict, else an empty dict."""
    return x if isinstance(x, dict) else {}


def _toint(v: Any) -> int:
    """A non-negative int from a provider-reported count, 0 for anything that is not a sane number."""
    if isinstance(v, bool):
        return 0
    try:
        n = int(v or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return n if 0 <= n <= 10 ** 12 else 0


def _count(v: Any) -> Optional[int]:
    """Like `_toint`, but None (not 0) for a value that is present and not a sane count."""
    if v is None or v == "":
        return 0
    n = _toint(v)
    return n if n or v in (0, "0", 0.0) else None


def _slen(v: Any) -> int:
    """Length of a string value, 0 for anything else."""
    return len(v) if isinstance(v, str) else 0


def _loads(s: Union[str, bytes]) -> Any:
    """json.loads that returns None instead of raising (also on absurd nesting)."""
    try:
        return json.loads(s)
    except _JSON_ERRORS:
        return None


def _dumps_len(o: Any) -> int:
    try:
        return len(json.dumps(o))
    except (TypeError, ValueError, RecursionError):
        return 0


@dataclass(frozen=True)
class Config:
    """Detection thresholds. The defaults suit coding agents that ask for large output budgets.

    enabled          False disables every detection (all `from_*` return None, `lookahead` passes).
    min_out          implicit: output_tokens must be <= max(min_out, out_frac * requested max_tokens).
    out_frac
    min_request      implicit detection only when the requested max_tokens is at least this.
    chars_per_token  content guard: more than threshold * this many characters of text, thinking or tool
                     input is "meaningful" (never exhaustion). The streaming look-ahead also commits as
                     soon as this much content has arrived.
    lookahead_bytes  the streaming look-ahead never holds back more than this many bytes.
    """

    enabled: bool = True
    min_out: int = 16
    out_frac: float = 0.01
    min_request: int = 1024
    chars_per_token: float = 8.0
    lookahead_bytes: int = 262144

    def __post_init__(self) -> None:
        # Clamp to sane lower bounds instead of raising: a config typo should not break a proxy.
        object.__setattr__(self, "min_out", max(0, int(self.min_out)))
        object.__setattr__(self, "out_frac", max(0.0, float(self.out_frac)))
        object.__setattr__(self, "min_request", max(1, int(self.min_request)))
        object.__setattr__(self, "chars_per_token", max(1.0, float(self.chars_per_token)))
        object.__setattr__(self, "lookahead_bytes", max(4096, int(self.lookahead_bytes)))


DEFAULT_CONFIG = Config()


def _cfg(config: Optional[Config]) -> Config:
    return config if config is not None else DEFAULT_CONFIG


def out_threshold(requested_max: Optional[int], config: Optional[Config] = None) -> int:
    """Output tokens at or below which a length stop counts as "produced nothing"."""
    c = _cfg(config)
    return max(c.min_out, int(math.ceil(c.out_frac * min(_toint(requested_max), 10 ** 12))))


def eligible(requested_max: Any, config: Optional[Config] = None) -> bool:
    """Is an implicit detection possible for this request at all? (Else: never buffer, never convert.)"""
    c = _cfg(config)
    return bool(c.enabled and isinstance(requested_max, int) and not isinstance(requested_max, bool)
                and c.min_request <= requested_max <= 10 ** 12)


# ---------------------------------------------------------------------------------------- result ----
@dataclass(frozen=True)
class ContextExhausted:
    """The one normalised shape for "the prompt does not fit the model's context window".

    kind               "explicit" (the provider said so in an error), "implicit" (an empty length stop) or
                       "estimated" (`limits.precheck`: the prompt was refused before it was sent).
    input_tokens       prompt size, when the provider reported it.
    limit              context window, when the provider reported it (implicit: input + output).
    completion_tokens  explicit only: the output budget the provider counted against the window.
    output_tokens      implicit only: what the provider actually produced.
    status             explicit only: the provider's HTTP status.
    requested_max      implicit only: the caller's max_tokens.
    threshold          implicit only: the output threshold that was applied.
    message            explicit only: the provider's error text, whitespace-collapsed, max 300 chars.
    """

    kind: str
    input_tokens: Optional[int] = None
    limit: Optional[int] = None
    completion_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    status: Optional[int] = None
    requested_max: Optional[int] = None
    threshold: Optional[int] = None
    message: str = ""

    @property
    def is_max_tokens_overflow(self) -> bool:
        """The provider said prompt + completion > window, but the PROMPT alone fits: the fix is a
        smaller max_tokens, not a compaction."""
        return bool(self.completion_tokens and self.input_tokens and self.limit
                    and self.input_tokens < self.limit)

    def normalized(self, est_input: Optional[int] = None) -> Tuple[int, int]:
        """(input_tokens, limit), both always present (explicit: input > limit whenever any number is known).

        IMPLICIT: the reported input and, for the limit, the observed window (input + output, so the
        input may sit just below the limit, which clients accept). EXPLICIT: the provider's numbers; a
        missing input is `est_input` (at least limit + 1), a missing limit is input - 1, and an input the
        provider reports as <= its limit is raised to limit + 1."""
        inp = self.input_tokens or None
        lim = self.limit or None
        if self.kind == "implicit":
            inp = inp or 1
            return inp, lim or (inp + (self.output_tokens or 0))
        if not inp:
            inp = max(int(est_input or 0), (lim + 1) if lim else 0) or 1
        if not lim:
            lim = max(1, inp - 1)
        elif lim >= inp:
            inp = lim + 1
        return inp, lim

    def to_anthropic(self, est_input: Optional[int] = None) -> Tuple[int, dict]:
        """(400, body): Anthropic's prompt-too-long error, the shape Claude Code compacts on."""
        inp, lim = self.normalized(est_input)
        return 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                "message": f"prompt is too long: {inp} tokens > {lim} maximum"}}

    def to_anthropic_max_tokens(self) -> Tuple[int, dict]:
        """(400, body): Anthropic's input + max_tokens overflow error (clients lower max_tokens and retry).
        Only meaningful when `is_max_tokens_overflow`."""
        return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": (
            f"input length and `max_tokens` exceed context limit: {self.input_tokens} + "
            f"{self.completion_tokens} > {self.limit}, decrease input length or `max_tokens` and try again")}}

    def to_openai(self, est_input: Optional[int] = None) -> Tuple[int, dict]:
        """(400, body): OpenAI's context_length_exceeded error (Chat Completions and Responses clients)."""
        inp, lim = self.normalized(est_input)
        room = "" if inp > lim else ", which leaves no room for output"
        return 400, {"error": {"message": (f"This model's maximum context length is {lim} tokens. However, your "
                                           f"messages resulted in {inp} tokens{room}. Please reduce the length "
                                           f"of the messages."),
                               "type": "invalid_request_error", "param": "messages",
                               "code": "context_length_exceeded"}}

    def summary(self, est_input: Optional[int] = None) -> dict:
        """Sizes only, never message text: safe to log or emit as a metric."""
        inp, lim = self.normalized(est_input)
        out = {"kind": self.kind, "input_tokens": inp, "limit": lim}
        for k in ("status", "output_tokens", "requested_max", "threshold", "completion_tokens"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


# ------------------------------------------------------------------------------- explicit errors ----
# Lower-cased substrings that mean "the prompt/context does not fit" (gathered from provider docs and
# observed bodies). Anything else (rate limits, auth, bad params) is left alone.
_EXPLICIT_MARKERS = (
    "context_length_exceeded",                      # OpenAI / Azure / Groq / DeepSeek / OpenRouter code
    "maximum context length",                       # OpenAI, vLLM, DeepSeek, OpenRouter, Mistral
    "context length exceeded", "exceeds the context window", "exceed the context window",
    "exceeds the maximum context", "context window exceeded", "context_window_exceeded",
    "exceeds context length", "exceeds the context length", "exceed_context_size",   # llama.cpp code
    "exceeds the available context size",           # llama.cpp message
    "prompt is too long",                           # Anthropic / Anthropic-compatible / Bedrock
    "prompt too long", "input is too long",         # Bedrock ("Input is too long for requested model.")
    "too large for model with",                     # Mistral ("... too large for model with 32768 maximum ...")
    "maximum model length",                         # vLLM ("... longer than the maximum model length of N")
    "is too long and exceeds limit",                # vLLM ("Input prompt (N tokens) is too long and exceeds ...")
    "maximum prompt length",                        # xAI ("This model's maximum prompt length is N but ...")
    "exceeds the maximum number of tokens allowed",  # Gemini ("The input token count (N) exceeds ...")
    "exceeded model token limit",                   # Moonshot / Kimi
    "range of input length should be",              # Qwen / DashScope
    "input length exceeds", "input tokens exceed", "input token count exceeds",
    "prompt exceeds max length", "prompt exceeds the max",   # GLM / Zhipu (code 1261)
    "reduce the length of the messages",            # OpenAI / Groq suffix
    "`inputs` must have less than",                 # TGI ("`inputs` must have less than 4096 tokens. Given: N")
    "`inputs` tokens + `max_new_tokens` must be",   # TGI (prompt + budget > window)
    "超长", "超过最大长度", "超出最大长度",  # GLM / Qwen (zh)
)
# NOT exhaustion of the prompt: input + max_tokens > window. Anthropic clients handle this message
# themselves (they lower max_tokens and retry), so it is never rewritten.
_MAXTOK_OVERFLOW = re.compile(r"input length and `?max_tokens`? exceed context limit", re.I)
# vLLM / OpenAI-compatible servers: "'max_tokens' or 'max_completion_tokens' is too large: 32000. This model's
# maximum context length is 32768 tokens and your request has 1000 input tokens". The output budget is the
# problem; the prompt fits unless the stated input is at least the window.
_MAXTOK_TOO_LARGE = re.compile(r"max_(?:completion_|new_)?tokens'?`?[^.]{0,60}?(?:is )?too large:?\s*(\d[\d,_]*)",
                               re.I)

_N = r"(\d[\d,_]*)"
_LIMIT_RES = [re.compile(p, re.I) for p in (
    r"maximum context length is " + _N,
    _N + r" maximum context length",
    r">\s*" + _N + r"\s*maximum",
    r"maximum model length of " + _N,
    r"exceeds? (?:the )?limit of " + _N,
    r"context window of " + _N,
    r"context (?:size|length|window) (?:is|of) " + _N,
    r"maximum number of tokens allowed \(" + _N + r"\)",
    r"maximum prompt length is " + _N,
    r"model token limit:?\s*" + _N,
    r"range of input length should be \[\s*\d+\s*,\s*" + _N + r"\s*\]",
    r"limit(?: is)?:?\s*" + _N + r"\s*tokens",
    r"must have less than " + _N + r" tokens",            # TGI
    r"`max_new_tokens` must be <= " + _N,                 # TGI
)]
_INPUT_RES = [re.compile(p, re.I) for p in (
    r"prompt is too long:\s*" + _N,
    r"resulted in " + _N + r" tokens",
    r"\(" + _N + r" in the messages",
    r"prompt contains " + _N + r" tokens",
    r"prompt \(length " + _N + r"\)",
    r"input prompt \(" + _N + r" tokens\)",
    r"input token count \(" + _N + r"\)",
    r"request contains " + _N + r" tokens",
    r"input (?:length|tokens?) (?:is |of )?" + _N,
    r"request has " + _N + r" input tokens",
    r"given:\s*" + _N,                                   # TGI
    r"you requested (?:about )?" + _N + r" tokens",     # total (prompt + completion): last resort
)]
# vLLM / DeepSeek / OpenAI legacy: "... you requested N tokens (A in the messages, B in the completion)".
_SPLIT_RE = re.compile(r"\(" + _N + r" in the messages,\s*" + _N + r" in the completion\)", re.I)
# OpenRouter: "... you requested about N tokens (A of text input, [B of tool input, ]C in the output)".
# TGI: "... must be <= 4096. Given: 4000 `inputs` tokens and 500 `max_new_tokens`".
_SPLIT_TGI = re.compile(r"given:\s*" + _N + r"\s*`inputs` tokens and\s*" + _N + r"\s*`max_new_tokens`", re.I)
_SPLIT_OR = re.compile(r"\(" + _N + r" of text input,\s*(?:" + _N + r" of tool input,\s*)?" + _N
                       + r" in the output\)", re.I)


def _int(s: Optional[str]) -> Optional[int]:
    try:
        return int(re.sub(r"[,_]", "", s or ""))
    except (TypeError, ValueError):
        return None


def _first(res, text: str) -> Optional[int]:
    for r in res:
        m = r.search(text)
        if m:
            v = _int(m.group(1))
            if v:
                return v
    return None


def error_text(body: Body) -> str:
    """Best-effort message text of a provider error body (bytes, str, or parsed JSON).

    Includes the error code / type, since some providers put the only signal there (e.g. the code
    ``context_length_exceeded``). Understands SSE-framed error events and JSON nested in strings."""
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", "replace")
    obj: Any = body
    if isinstance(body, str):
        s = body.strip()
        if s.startswith("event:") or s.startswith("data:"):
            s = "\n".join(ln[5:].strip() for ln in s.splitlines() if ln.startswith("data:"))
        try:
            obj = json.loads(s)
        except _JSON_ERRORS:
            return body[:4000]
    parts: list = []

    def walk(o: Any, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(o, dict):
            for k in ("message", "msg", "detail", "error", "code", "type", "reason", "metadata", "raw"):
                if k in o:
                    walk(o[k], depth + 1)
        elif isinstance(o, list):
            for x in o[:5]:
                walk(x, depth + 1)
        elif isinstance(o, (str, int)) and not isinstance(o, bool):
            try:
                s = str(o)
            except ValueError:                               # an int too large to print
                return
            if s and s not in parts:
                parts.append(s)
                if isinstance(o, str) and s.lstrip().startswith("{"):
                    inner = _loads(s[:100000])               # e.g. OpenRouter nests the provider's raw JSON
                    if inner is not None:
                        walk(inner, depth + 1)
    walk(obj)
    if parts:
        return " | ".join(parts)[:4000]
    try:
        return json.dumps(obj)[:4000] if obj else ""
    except (TypeError, ValueError, RecursionError):
        return ""


def from_error(status: Any, body: Body, config: Optional[Config] = None) -> Optional[ContextExhausted]:
    """A `ContextExhausted` when a provider error says the prompt / context is too long, else None.

    `status` must be an HTTP error status (>= 400). An "input + max_tokens exceed context limit" message
    in Anthropic's wording is NOT matched: Anthropic clients already recover from that one."""
    if not _cfg(config).enabled:
        return None
    if not isinstance(status, int) or isinstance(status, bool) or status < 400:
        return None
    text = error_text(body)
    low = text.lower()
    if not low or _MAXTOK_OVERFLOW.search(low):
        return None
    if not any(m in low for m in _EXPLICIT_MARKERS):
        return None
    inp: Optional[int] = None
    comp: Optional[int] = None
    lim = _first(_LIMIT_RES, text)
    mt = _MAXTOK_TOO_LARGE.search(text)
    if mt:
        # The OUTPUT budget was refused. Only a stated input >= window makes it prompt exhaustion; a
        # stated smaller input is an overflow (fix: lower max_tokens); no input stated: not ours to judge.
        inp = _first(_INPUT_RES, text)
        if not (inp and lim):
            return None
        comp = None if inp >= lim else _int(mt.group(1))
        return ContextExhausted(kind="explicit", status=status, input_tokens=inp, limit=lim, completion_tokens=comp,
                                message=" ".join(text.split())[:300])
    sp = _SPLIT_RE.search(text)
    so = _SPLIT_OR.search(text)
    st = _SPLIT_TGI.search(text)
    if st:
        inp, comp = _int(st.group(1)), _int(st.group(2))
    elif sp:
        inp, comp = _int(sp.group(1)), _int(sp.group(2))
    elif so:
        inp = ((_int(so.group(1)) or 0) + (_int(so.group(2)) or 0)) if so.group(2) else _int(so.group(1))
        comp = _int(so.group(3))
    else:
        inp = _first(_INPUT_RES, text)
    return ContextExhausted(kind="explicit", status=status, input_tokens=inp, limit=lim,
                            completion_tokens=comp, message=" ".join(text.split())[:300])


# ------------------------------------------------------------------------------------- implicit -----
def from_stop(stop: Optional[str], output_tokens: Any, input_tokens: Any, requested_max: Any,
              content_chars: int = 0, config: Optional[Config] = None) -> Optional[ContextExhausted]:
    """A `ContextExhausted` when a SUCCESSFUL answer is really context exhaustion, else None.

    All must hold: `stop` is a length stop; the caller's budget was large (requested_max >=
    config.min_request); the provider reported a non-zero input size; output_tokens <= the threshold
    (`out_threshold`); and the content actually produced is below that threshold in characters."""
    c = _cfg(config)
    if not eligible(requested_max, c) or not isinstance(stop, str) or stop not in STOP_LENGTH:
        return None
    inp, out = _count(input_tokens), _count(output_tokens)
    if inp is None or out is None or inp <= 0:
        return None
    thr = out_threshold(requested_max, c)
    if out > thr or _toint(content_chars) > thr * c.chars_per_token:
        return None
    return ContextExhausted(kind="implicit", input_tokens=inp, output_tokens=out, limit=inp + out,
                            requested_max=requested_max, threshold=thr)


def anthropic_message_content_chars(msg: Any) -> int:
    """Characters of text / thinking / tool input in a non-streamed Anthropic Messages response."""
    content = _d(msg).get("content")
    if isinstance(content, str):
        return len(content)
    n = 0
    for b in content if isinstance(content, list) else []:
        if isinstance(b, str):
            n += len(b)
            continue
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            n += _slen(b.get("text"))
        elif t == "thinking":
            n += _slen(b.get("thinking"))
        elif t in ("tool_use", "server_tool_use"):
            n += _slen(b.get("name")) + max(0, _dumps_len(b.get("input") or {}) - 2)
        elif t:
            n += 1
    return n


def anthropic_usage_input(usage: Any) -> int:
    """Context size an Anthropic usage block reports (uncached + cache read + cache write)."""
    u = _d(usage)
    return sum(_toint(u.get(k)) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def from_anthropic_message(msg: Any, requested_max: Any,
                           config: Optional[Config] = None) -> Optional[ContextExhausted]:
    """`from_stop` for a complete (non-streamed) Anthropic Messages response dict.

    A 200 response whose body is an error object (some gateways do this) is checked with `from_error`."""
    if not isinstance(msg, dict):
        return None
    if msg.get("type") == "error" or (msg.get("error") and "content" not in msg):
        return from_error(400, msg, config)
    u = _d(msg.get("usage"))
    return from_stop(msg.get("stop_reason"), u.get("output_tokens"), anthropic_usage_input(u), requested_max,
                     anthropic_message_content_chars(msg), config)


def openai_chat_content_chars(message: Any) -> int:
    """Characters of content / reasoning / tool calls in an OpenAI chat message or stream delta."""
    if not isinstance(message, dict):
        return 0
    n = 0
    c = message.get("content")
    if isinstance(c, str):
        n += len(c)
    elif isinstance(c, list):
        n += sum(len(p) if isinstance(p, str) else _slen(_d(p).get("text")) for p in c)
    for k in ("reasoning_content", "reasoning", "refusal"):
        n += _slen(message.get(k))
    tool_calls = message.get("tool_calls")
    for tc in tool_calls if isinstance(tool_calls, list) else []:
        fn = _d(_d(tc).get("function"))
        n += _slen(fn.get("name")) + _slen(fn.get("arguments"))
    return n


def from_openai_chat(resp: Any, requested_max: Any,
                     config: Optional[Config] = None) -> Optional[ContextExhausted]:
    """`from_stop` for a complete (non-streamed) OpenAI Chat Completions response dict.

    A 200 response whose body is an error object (some gateways do this) is checked with `from_error`."""
    if not isinstance(resp, dict):
        return None
    if "error" in resp and not resp.get("choices"):
        return from_error(400, resp, config)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    u = _d(resp.get("usage"))
    return from_stop(choices[0].get("finish_reason"), u.get("completion_tokens"), u.get("prompt_tokens"),
                     requested_max, openai_chat_content_chars(choices[0].get("message")), config)
