# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""The README quickstart, as written: the stub provider and the proxy CLI as real processes."""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

import pytest

from conftest import ROOT


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait(port, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(port)


def _post(port, path, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


@pytest.fixture
def quickstart():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    up, px = _free_port(), _free_port()
    procs = [subprocess.Popen([sys.executable, os.path.join(ROOT, "examples", "stub_upstream.py"), "--port", str(up)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, os.path.join(ROOT, "examples", "proxy.py"),
                               "--upstream", f"http://127.0.0.1:{up}", "--port", str(px)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    try:
        _wait(up)
        _wait(px)
        yield up, px
    finally:
        for p in procs:
            p.terminate()
            p.wait(10)


def test_quickstart_openai_and_anthropic_clients(quickstart):
    up, px = quickstart
    chat = {"model": "any", "max_tokens": 32000, "messages": [{"role": "user", "content": "hi"}]}
    st, raw = _post(up, "/v1/chat/completions", chat)                 # direct: a silent empty 200
    assert st == 200 and raw["choices"][0]["finish_reason"] == "length"
    st, raw = _post(px, "/v1/chat/completions", chat)                 # via the proxy: a real error
    assert st == 400 and raw["error"]["code"] == "context_length_exceeded"
    st, raw = _post(px, "/v1/messages", chat)
    assert st == 400 and raw["error"]["message"] == "prompt is too long: 131066 tokens > 131072 maximum"
    st, raw = _post(px, "/v1/messages", dict(chat, max_tokens=16))   # a small budget is left alone
    assert st == 200 and raw["content"][0]["text"] == "pong"
