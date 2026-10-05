# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - unreleased

First public release.

### Added

- `from_error`: detects explicit context-exhaustion errors from OpenAI, Anthropic(-compatible),
  OpenRouter, Mistral, vLLM, TGI, llama.cpp, DeepSeek, Qwen/DashScope, GLM/Zhipu, Moonshot/Kimi, Gemini,
  xAI and Bedrock, and extracts the token numbers.
- `from_stop`, `from_anthropic_message`, `from_openai_chat`: detect implicit exhaustion (an empty length
  stop on a large budget).
- `lookahead`: bounded streaming look-ahead for Anthropic and OpenAI Chat SSE streams.
- `ContextExhausted`: one normalised result, rendered as Anthropic's prompt-too-long error, Anthropic's
  max_tokens overflow error or OpenAI's `context_length_exceeded`.
- `ctxnorm.limits`: max_output clamp with thinking-budget fit, a conservative input estimate with
  pre-send window check, output-limit stop detection and limit hints parsed from error text.
- `examples/proxy.py`: a stdlib reverse proxy for OpenAI- and Anthropic-compatible endpoints;
  `examples/stub_upstream.py`: a full-window fake provider for trying it out.
- CC0 provider error vectors in `vectors/` (explicit, max-tokens overflow, negative).
- Hardening: every public function returns a result or `None` on hostile input (absurd nesting, wrong
  field types, non-numeric counts); fuzz-tested.
- The example proxy caps request and buffered response sizes, applies client and upstream timeouts, and
  answers 502 (or an in-band stream error) when the upstream drops the connection.
