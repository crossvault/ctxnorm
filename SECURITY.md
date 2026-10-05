# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Report it privately through GitHub's **"Report a vulnerability"** button on this repository's Security tab
(private vulnerability reporting).

Alternative contact: info@session-exchange.com

Please include the affected version or commit, a description, and a minimal reproduction with synthetic
data only. We aim to acknowledge reports within 5 working days and to agree on a disclosure timeline with
you.

## Scope

In scope:

- the `ctxnorm` library (`src/`): for example, crafted provider responses that cause excessive CPU or
  memory use, an exception that escapes the API, or content (prompt or answer text) appearing in
  `ContextExhausted.summary()`;
- the example proxy (`examples/proxy.py`): for example, credentials sent anywhere other than the
  configured upstream, or request or response content leaking into its event log.

Out of scope:

- deploying the example proxy on a public interface. It is an example, with no TLS and no authentication
  of its own, and the README says to bind it to localhost;
- a provider phrasing that isn't detected. That is a normal bug; please open an issue or a pull request.

## Supported versions

Only the latest release receives fixes.
