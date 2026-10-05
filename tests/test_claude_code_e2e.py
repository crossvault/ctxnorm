# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Opt-in end-to-end check with the real Claude Code CLI:

    Claude Code -> examples/proxy.py -> a local stub that behaves like a provider at a full window
    (an EMPTY max_tokens answer with the input size reported).

Turn 1 is normal. On turn 2 (``-c``) the stub is full once. The proxy must answer 400 prompt-too-long, and
Claude Code must compact (summarise) and finish the turn, instead of failing with "exceeded the 32000
output token maximum".

Runs only with CTXNORM_E2E_CLAUDE=1 and the ``claude`` CLI on PATH. Uses a throwaway HOME and a dummy key;
every request goes to 127.0.0.1.
"""
import os
import shutil
import subprocess
import tempfile

import pytest

from payloads import FULL_PROMPT, anth_message, anth_stream

CLAUDE = shutil.which("claude")
pytestmark = pytest.mark.skipif(not CLAUDE or os.environ.get("CTXNORM_E2E_CLAUDE") != "1",
                                reason="opt-in: set CTXNORM_E2E_CLAUDE=1 with the claude CLI installed")


def _clean_env(**extra):
    keep = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(extra)
    return env


def test_real_claude_code_compacts_on_normalised_context_exhaustion(stub, make_client):
    state = {"full": 0}

    def answer(method, path, headers, body):
        body = body or {}
        if method != "POST" or "/messages" not in path or "count_tokens" in path:
            return 404, [], b"{}"
        if state["full"] > 0 and body.get("tools"):
            state["full"] -= 1
            return 200, [("Content-Type", "text/event-stream")], anth_stream(stop="max_tokens", in_tok=FULL_PROMPT)
        if body.get("stream"):
            return 200, [("Content-Type", "text/event-stream")], anth_stream(texts=["PONG"], in_tok=20, out_tok=1)
        return 200, [("Content-Type", "application/json")], anth_message(text="PONG", stop="end_turn", in_tok=20,
                                                                         out_tok=1)
    stub.answer = answer
    c = make_client()
    home = tempfile.mkdtemp(prefix="ctxnorm-e2e-home-")
    work = tempfile.mkdtemp(prefix="ctxnorm-e2e-work-")
    env = _clean_env(HOME=home, CLAUDE_CONFIG_DIR=os.path.join(home, ".claude"),
                     ANTHROPIC_BASE_URL=f"http://127.0.0.1:{c.server.server_address[1]}",
                     ANTHROPIC_API_KEY="dummy-not-a-key", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                     DISABLE_TELEMETRY="1", DISABLE_AUTOUPDATER="1", DISABLE_ERROR_REPORTING="1")

    def run(*args):
        return subprocess.run([CLAUDE, "-p", *args, "--model", "claude-sonnet-4-5"], env=env, cwd=work,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    try:
        p1 = run("say PONG")
        assert p1.returncode == 0 and p1.stdout.strip() == "PONG", (p1.stdout[-500:], p1.stderr[-1500:])
        n_before = len(stub.seen)
        state["full"] = 1
        p2 = run("-c", "say PONG again")
        second = [s for s in stub.seen[n_before:] if s["body"]]
        assert [e["event"] for e in c.events] == ["context_exhausted"], c.events
        assert "output token maximum" not in (p2.stdout + p2.stderr)
        # Claude Code compacted: after the refused request it summarised, then retried the turn with a
        # SHORTER history, and finished normally.
        main = [s for s in second if s["body"].get("tools")]
        assert len(main) >= 2, [len(s["body"].get("messages") or []) for s in second]
        assert len(main[-1]["body"]["messages"]) < len(main[0]["body"]["messages"])
        assert p2.returncode == 0 and p2.stdout.strip() == "PONG", (p2.stdout[-800:], p2.stderr[-1500:])
    finally:
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
