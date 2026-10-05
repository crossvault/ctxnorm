# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Per-model token limits: helpers for a proxy that knows (or learns) a model's limits.

* `clamp_output`: lower a request's output budget to the model's max_output (never raise it), keeping an
  Anthropic thinking budget valid.
* `estimate_input_tokens` / `fits_window`: a deliberately LOW estimate of the prompt size, to refuse a
  prompt that certainly does not fit before sending it.
* `stopped_at_output_limit` / `stop_in_tail`: did the upstream stop because of the model's output limit?
* `parse_limit_hint` / `output_refusal`: read limits stated in an upstream error text.

The estimate counts only the characters of the prompt's string values (system, messages, tools; not JSON
syntax, not keys, not base64 image/document data, not thinking signatures) and divides by
`chars_per_token` (default 4). English prose is about 4 characters per token and code, JSON and CJK text
are denser, so real token counts are at or above the estimate. Erring low matters: a false refusal forces
a needless compaction.

Pure stdlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .core import STOP_LENGTH, ContextExhausted

__all__ = [
    "MIN_THINKING_BUDGET", "MAX_LIMIT_TOKENS", "MIN_CONTEXT_WINDOW", "OUTPUT_FIELDS", "Limits",
    "clamp_output", "fit_thinking", "estimate_input_tokens", "fits_window", "precheck", "stopped_at_output_limit",
    "stop_in_tail", "parse_limit_hint", "output_refusal",
]

MIN_THINKING_BUDGET = 1024           # Anthropic's minimum thinking.budget_tokens
MAX_LIMIT_TOKENS = 20_000_000        # sanity bound for any configured or parsed limit
MIN_CONTEXT_WINDOW = 1024            # a parsed "context window" below this is not believed

#: Per request protocol, the body fields that carry the output budget.
OUTPUT_FIELDS = {
    "anthropic-messages": ("max_tokens",),
    "openai-chat": ("max_tokens", "max_completion_tokens"),
    "openai-responses": ("max_output_tokens",),
}


def _pos_int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and 0 < v <= MAX_LIMIT_TOKENS else None


@dataclass(frozen=True)
class Limits:
    """A model's limits. Every field is optional; None means "unknown, do not enforce"."""

    context_window: Optional[int] = None
    max_output: Optional[int] = None
    max_output_default: Optional[int] = None

    @classmethod
    def from_mapping(cls, m: Any) -> "Limits":
        """Build from a dict, keeping only valid positive ints (anything else is dropped, not an error)."""
        if not isinstance(m, Mapping):
            return cls()
        return cls(**{k: _pos_int(m.get(k)) for k in ("context_window", "max_output", "max_output_default")})

    def __bool__(self) -> bool:
        return bool(self.context_window or self.max_output or self.max_output_default)


# ---------------------------------------------------------------------------- max_output clamp -----
def clamp_output(body: Any, max_output: Any, protocol: str = "anthropic-messages") -> List[str]:
    """Lower the protocol's output-budget field(s) of `body` to `max_output`, in place.

    Only top-level keys are replaced; nested objects are never edited in place. Returns a list of change
    descriptions (empty = untouched). Never raises a value; a missing field stays missing."""
    if not isinstance(body, dict) or not _pos_int(max_output):
        return []
    done = []
    for f in OUTPUT_FIELDS.get(protocol, ("max_tokens",)):
        v = body.get(f)
        if isinstance(v, int) and not isinstance(v, bool) and v > max_output:
            body[f] = max_output
            done.append(f"{f} {v}->{max_output}")
    if protocol == "anthropic-messages":
        done += fit_thinking(body)
    return done


def fit_thinking(body: Dict[str, Any]) -> List[str]:
    """Anthropic: an enabled thinking budget must stay below max_tokens. Lower it to max_tokens - 1024
    (keeping the minimum budget's worth for the answer), or drop `thinking` when no valid budget fits."""
    th = body.get("thinking")
    mt = body.get("max_tokens")
    if not (isinstance(th, dict) and th.get("type") == "enabled" and isinstance(th.get("budget_tokens"), int)
            and isinstance(mt, int) and not isinstance(mt, bool) and th["budget_tokens"] >= mt):
        return []
    nb = mt - MIN_THINKING_BUDGET
    if nb >= MIN_THINKING_BUDGET:
        body["thinking"] = dict(th, budget_tokens=nb)
        return [f"thinking.budget_tokens {th['budget_tokens']}->{nb}"]
    body.pop("thinking")
    return ["thinking dropped (max_tokens leaves no budget)"]


# ------------------------------------------------------------------------------ input estimate -----
def _walk(o: Any, depth: int = 0) -> int:
    """Characters of the string values in a request body, skipping base64 source data (images,
    documents) and thinking signatures, which providers do not bill as text tokens."""
    if depth > 64:
        return 0
    if isinstance(o, str):
        return len(o)
    n = 0
    if isinstance(o, dict):
        opaque = o.get("type") in ("base64", "redacted_thinking")
        for k, v in o.items():
            if k == "signature" or (k == "data" and opaque) or k in ("image_url", "file_data"):
                continue
            n += _walk(v, depth + 1)
    elif isinstance(o, list):
        for x in o:
            n += _walk(x, depth + 1)
    return n


_PROMPT_KEYS = ("system", "messages", "tools", "input", "instructions", "prompt")


def estimate_input_tokens(body: Any, chars_per_token: float = 4.0) -> int:
    """Low-side estimate of a request's prompt tokens: prompt string characters / chars_per_token.
    Works on an Anthropic (system/messages/tools) or OpenAI (messages/input/instructions/tools) body."""
    if not isinstance(body, dict):
        return 0
    chars = sum(_walk(body.get(k)) for k in _PROMPT_KEYS if k in body)
    return int(chars / max(1.0, float(chars_per_token or 4.0)))


def fits_window(estimate: int, limits: Optional[Limits]) -> bool:
    """False only when the estimate ALONE exceeds the context window (max_tokens is not added: erring
    towards sending is cheaper than a false refusal)."""
    cw = limits.context_window if limits else None
    return not (cw and estimate > cw)


def precheck(body: Any, limits: Optional[Limits], chars_per_token: float = 4.0) -> Optional[ContextExhausted]:
    """A `ContextExhausted` (kind "estimated") when the prompt certainly does not fit `limits.context_window`,
    so a proxy can answer the client's prompt-too-long error without sending the request; else None."""
    est = estimate_input_tokens(body, chars_per_token)
    if fits_window(est, limits):
        return None
    return ContextExhausted(kind="estimated", input_tokens=est, limit=limits.context_window if limits else None)


# ---------------------------------------------------------------------------- upstream signals -----
def stopped_at_output_limit(stop: Optional[str], sent_max: Any, limits: Optional[Limits]) -> bool:
    """The upstream stopped on length while the output budget WAS the model's output limit: the sent
    budget reached `max_output`, or no budget was sent and the provider's default applied."""
    if stop not in STOP_LENGTH or not limits:
        return False
    if isinstance(sent_max, int) and not isinstance(sent_max, bool) and sent_max > 0:
        return bool(limits.max_output and sent_max >= limits.max_output)
    return bool(limits.max_output or limits.max_output_default)


_STOP_RE = re.compile(rb'"(?:stop_reason|finish_reason)"\s*:\s*"([a-z_]+)"')
_INCOMPLETE_RE = re.compile(rb'"reason"\s*:\s*"max_output_tokens"')


def stop_in_tail(buf: bytes) -> Optional[str]:
    """The LAST stop / finish reason in the tail bytes of an Anthropic or OpenAI chat body or SSE stream
    (``max_output_tokens`` for an incomplete Responses stream), or None."""
    if not buf:
        return None
    last = None
    for m in _STOP_RE.finditer(buf):
        last = m.group(1).decode()
    if last is None and _INCOMPLETE_RE.search(buf):
        last = "max_output_tokens"
    return last


_HN = r"(\d[\d,_]{2,11})"
_HINTS = (
    ("max_output", re.compile(
        r"(?:max_tokens|max_completion_tokens|max_output_tokens|max_new_tokens|maxOutputTokens|"
        r"output tokens?)\b[^.\n]{0,80}?(?:<=|≤|less than or equal to|at most|must not exceed|"
        r"cannot exceed|exceeds?(?: the)?(?: maximum| limit)?(?: of)?|maximum(?: value)?(?: is| of)?|"
        r"max(?:imum)? allowed(?: is)?|up to|\d[\d,]*\s*>|range[^0-9]{0,20}\d+\s*(?:,|to|-)\s*)\s*\[?" + _HN,
        re.IGNORECASE)),
    ("context_window", re.compile(
        r"(?:maximum context length is|context (?:window|length)(?: is| of| limit(?: is| of)?)?|"
        r"tokens?\s*>\s*|input (?:is )?too long[^0-9]{0,40}(?:limit|maximum)(?: is| of)?)\s*" + _HN,
        re.IGNORECASE)),
)


def parse_limit_hint(text: Any) -> Dict[str, int]:
    """``{max_output?, context_window?}`` stated in an upstream reply / error text, or {}."""
    if not isinstance(text, str) or not text:
        return {}
    out: Dict[str, int] = {}
    for key, rx in _HINTS:
        m = rx.search(text[:4000])
        if m:
            try:
                n = int(re.sub(r"[,_]", "", m.group(1)))
            except ValueError:
                continue
            lo = MIN_CONTEXT_WINDOW if key == "context_window" else 1
            if lo <= n <= MAX_LIMIT_TOKENS:
                out[key] = n
    return out


def output_refusal(text: Any) -> Optional[int]:
    """The max_output an upstream error says the request's output budget exceeded, else None. Context
    errors ("input length and `max_tokens` exceed context limit") are NOT output refusals."""
    if not isinstance(text, str) or "exceed context limit" in text.lower():
        return None
    return parse_limit_hint(text).get("max_output")
