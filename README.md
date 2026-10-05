# ctxnorm

[Repository](https://github.com/crossvault/ctxnorm) · [Issues](https://github.com/crossvault/ctxnorm/issues)

**Make every LLM provider's "your prompt is too long" look the same, so your client can recover.**

When a conversation outgrows the model's context window, Anthropic answers
`400 prompt is too long: N tokens > M maximum`, and clients such as Claude Code respond by compacting
the conversation and retrying. Other providers phrase the same condition in a dozen different ways. Some
don't refuse at all: they return `200` with an empty answer and `finish_reason: "length"`. The client
then never compacts and shows errors such as *"response exceeded the 32000 output token maximum"*.

`ctxnorm` recognises all of these and turns them into one shape, which it can render as Anthropic's
prompt-too-long error or OpenAI's `context_length_exceeded`.

- Pure Python standard library, no dependencies, Python 3.9+.
- Detects **explicit** errors from OpenAI, Anthropic(-compatible), OpenRouter, Mistral, vLLM, TGI,
  llama.cpp, DeepSeek, Qwen/DashScope, GLM/Zhipu, Moonshot/Kimi, Gemini, xAI and Bedrock.
- Detects **implicit** exhaustion (an empty length stop on a large budget), including in **streams**:
  a bounded look-ahead holds back the start of an SSE stream until it is safe to commit a `200`.
- Never touches a legitimate answer: small budgets, real content and other errors pass through unchanged.
- Ships a runnable **example proxy** for OpenAI- and Anthropic-compatible endpoints.

## Quickstart (5 minutes)

### 1. Install

```bash
git clone <this repository> ctxnorm && cd ctxnorm
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest -q                      # offline, about 3 seconds
```

### 2. Use the library

```python
import ctxnorm

# An error body from any provider (bytes, str or parsed JSON):
info = ctxnorm.from_error(400, b'{"object": "error", "message": "Input prompt (140000 tokens) '
                               b'is too long and exceeds limit of 131072"}')
status, body = info.to_anthropic()
# 400 {'type': 'error', 'error': {'type': 'invalid_request_error',
#      'message': 'prompt is too long: 140000 tokens > 131072 maximum'}}

# A *successful* response that is really an exhausted window:
resp = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 131066, "completion_tokens": 6}}
info = ctxnorm.from_openai_chat(resp, requested_max=32000)
status, body = info.to_openai()          # OpenAI's context_length_exceeded
```

### 3. See it work end to end, without a model

Run a fake provider whose window is always full, and the example proxy in front of it:

```bash
python examples/stub_upstream.py --port 8000 &
PYTHONPATH=src python examples/proxy.py --upstream http://127.0.0.1:8000 --port 8080 &

# Directly: a silent, empty 200
curl -s localhost:8000/v1/chat/completions -d '{"model":"m","max_tokens":32000,"messages":[]}'
# Through the proxy: an error the client understands
curl -s localhost:8080/v1/chat/completions -d '{"model":"m","max_tokens":32000,"messages":[]}'
curl -s localhost:8080/v1/messages         -d '{"model":"m","max_tokens":32000,"messages":[]}'
```

### 4. Put it in front of a real endpoint

Stop the step-3 processes first (`kill %1 %2`), then:

```bash
PYTHONPATH=src python examples/proxy.py --upstream http://127.0.0.1:8000   # e.g. vLLM or llama-server

# OpenAI-compatible clients
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
# Anthropic-compatible clients such as Claude Code (the upstream must serve /v1/messages)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

The proxy forwards your client's own credentials and holds none of its own. Optional per-model limits
add a `max_tokens` clamp and a pre-send window check:

```bash
python examples/proxy.py --upstream http://127.0.0.1:8000 \
    --limit 'qwen3-coder=262144/65536' --limit '*=131072'
```

Safety limits, all configurable: `--max-body` (request bytes, default 32 MiB, answered with 413),
`--client-timeout` (default 60 s, answered with 408), `--upstream-timeout` (default 600 s) and
`--max-response` (buffered upstream bytes, default 64 MiB). An upstream that is unreachable, times out or
drops the connection before anything was relayed gives a `502` with a JSON error; one that drops in the
middle of a stream ends that stream with an in-band error event.

It logs one JSON line per event (`context_exhausted`, `context_window_refused`, `output_capped`,
`output_limit_stop`, `output_refused`) with token counts only, never prompt or answer text.

The proxy is an **example**: stdlib only, no TLS termination, no authentication of its own. Bind it to
localhost or put it behind something that provides those.

## API

| Call | Returns |
|---|---|
| `from_error(status, body, config=None)` | `ContextExhausted` if an error body says the prompt is too long |
| `from_stop(stop, output_tokens, input_tokens, requested_max, content_chars=0, config=None)` | `ContextExhausted` for an empty length stop on a large budget |
| `from_anthropic_message(msg, requested_max)` / `from_openai_chat(resp, requested_max)` | the same, for a complete response dict |
| `lookahead(chunks, requested_max, config=None, usage=None, protocol="anthropic" \| "openai-chat")` | `("exhausted", info, None)` or `("pass", held_chunks, rest)` for an SSE stream |
| `ContextExhausted.to_anthropic()` / `.to_openai()` / `.to_anthropic_max_tokens()` | `(400, body)` in the client's dialect |
| `ContextExhausted.summary()` | sizes only, safe to log |
| `Config(...)` | detection thresholds (see the docstring) |
| `ctxnorm.limits` | `Limits`, `clamp_output`, `precheck`, `estimate_input_tokens`, `stopped_at_output_limit`, `parse_limit_hint`, ... |

An "input + `max_tokens` exceed the window" error (the prompt itself fits), including vLLM's
"`max_tokens` is too large" and TGI's "`inputs` tokens + `max_new_tokens`", is reported with
`is_max_tokens_overflow`. Its fix is a smaller `max_tokens`, not a compaction, so render it with
`to_anthropic_max_tokens()` or relay it unchanged.

## How implicit detection decides

A successful answer counts as context exhaustion only if **all** of these hold:

1. the stop reason is a length stop (`length`, `max_tokens`, `max_output_tokens`, `model_length`);
2. the caller asked for a large budget (`max_tokens >= 1024` by default);
3. the provider reported a non-zero input size;
4. the output is at most `max(16, 1% of max_tokens)` tokens, and the content produced (text, thinking,
   tool input) is below that threshold in characters.

All thresholds are set through `Config`. In a stream, the look-ahead commits (relays everything
unchanged from then on) as soon as meaningful content arrives or 256 KiB have been held.

## Provider phrasings

`vectors/explicit.json` holds one real-shaped error body per provider phrasing, with the numbers the
parser must extract. `vectors/negative.json` holds errors that must never be rewritten. Both are CC0, so
use them in your own projects. If your provider words it differently, a pull request that adds a
**synthetic** vector is the most useful contribution there is (see [CONTRIBUTING.md](CONTRIBUTING.md)).

| Provider | Typical wording |
|---|---|
| OpenAI, Azure, Groq | `maximum context length is N tokens ... resulted in M tokens`, code `context_length_exceeded` |
| Anthropic and compatible | `prompt is too long: N tokens > M maximum` |
| OpenRouter | `... you requested about N tokens (A of text input, B in the output)`, sometimes nested raw JSON |
| Mistral | `Prompt contains N tokens ... too large for model with M maximum context length` |
| vLLM | `Input prompt (N tokens) is too long and exceeds limit of M` / `... longer than the maximum model length of M` |
| llama.cpp | `the request exceeds the available context size`, type `exceed_context_size_error` |
| DeepSeek | `... (A in the messages, B in the completion)` |
| Qwen / DashScope | `Range of input length should be [1, M]` |
| GLM / Zhipu | `Prompt exceeds max length` (code 1261), also in Chinese |
| Moonshot / Kimi | `exceeded model token limit: M` |
| Gemini | `The input token count (N) exceeds the maximum number of tokens allowed (M)` |
| xAI | `maximum prompt length is M but the request contains N tokens` |
| Bedrock | `Input is too long for requested model.` |
| TGI | `` `inputs` must have less than N tokens. Given: M `` |

Every public function returns a result or `None` on malformed or hostile input (wrong field types, absurd
nesting, non-numeric counts); it never raises. When an error states no numbers, pass `est_input` (your own
prompt-size estimate, e.g. `ctxnorm.limits.estimate_input_tokens(request)`) to the renderers.

### Not covered (yet)

Native (non-OpenAI-compatible) APIs: Gemini's `finishReason: "MAX_TOKENS"`, Ollama's native `done_reason`,
Cohere's own error format, and the OpenAI Responses API stream. Pull requests with synthetic vectors are
welcome.

## Development

```bash
pip install -e '.[dev]'
ruff check .
reuse lint
pytest -q
CTXNORM_E2E_CLAUDE=1 pytest -q tests/test_claude_code_e2e.py   # optional: drives the real Claude Code CLI
```

The test suite is offline: a guard fails any test that tries to reach a non-loopback address.

## Release checklist (maintainers)

- [ ] DCO GitHub App installed on the repository (CONTRIBUTING.md promises a DCO check), private
      vulnerability reporting enabled, branch protection on `main`.
- [ ] CI green on GitHub for Python 3.9-3.13; tag `v0.1.0`; PyPI via trusted publishing.

## Contributing

Contributions are welcome under the [Developer Certificate of Origin](https://developercertificate.org/):
sign off every commit (`git commit -s`). See [CONTRIBUTING.md](CONTRIBUTING.md).

Parts of this project were developed with AI assistance (Claude).

## License

Code: [Apache-2.0](LICENSE). Test vectors in `vectors/`: [CC0-1.0](LICENSES/CC0-1.0.txt).
Copyright 2026 crossVault GmbH. See [NOTICE](NOTICE).
