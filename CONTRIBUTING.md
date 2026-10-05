# Contributing

Thanks for helping. Bug reports, new provider phrasings and fixes are all welcome.

## Developer Certificate of Origin (DCO)

Every commit must be signed off under the [Developer Certificate of Origin 1.1](https://developercertificate.org/).
The sign-off certifies that you wrote the change, or otherwise have the right to submit it under this
project's licence (Apache-2.0 for code, CC0-1.0 for files in `vectors/`).

Add it with `git commit -s`, which appends a line using your git name and email:

```
Signed-off-by: Your Name <you@example.com>
```

A DCO check runs on every pull request. To fix a missing sign-off on your last commit, run
`git commit --amend -s` and force-push your branch. For several commits, run
`git rebase --signoff main`. We don't use a CLA.

## The most useful contribution: a new provider phrasing

If a provider reports context exhaustion in a way `ctxnorm` misses:

1. Add a vector to `vectors/explicit.json` (or `vectors/negative.json` for an error that must **not** be
   rewritten). Use the provider's **structure and wording** with **synthetic** values only. Never paste a
   real response that contains request ids, account ids, keys, prompts or anything else from a real
   account.
2. Add or extend a marker or number pattern in `src/ctxnorm/core.py` until `pytest -q` passes.
3. In the pull request, link the provider's public documentation for the error, if there is any.

Vectors are released under CC0-1.0, so only contribute text you are free to dedicate to the public domain.

## Pull requests

- Keep each pull request to one change, with a test.
- Before pushing, run `ruff check .`, `reuse lint` and `pytest -q`; CI runs the same on Python 3.9–3.13.
- The library stays stdlib-only. Please open an issue first if a change needs a dependency.
- New source files start with the SPDX header:

  ```python
  # SPDX-License-Identifier: Apache-2.0
  # Copyright 2026 crossVault GmbH
  ```

  If you prefer to keep your own copyright line, add it below; the sign-off is what matters.

## AI-assisted contributions

You may use AI tools. You are still the author: you must understand and be able to explain every line,
the DCO sign-off is yours, and the tests must pass. Say in the pull request if a substantial part was
generated. Unreviewed bulk changes will be closed.

## Security issues

Don't open a public issue. See [SECURITY.md](SECURITY.md).
