# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 crossVault GmbH
"""Normalise LLM providers' context-window exhaustion into one error shape.

    >>> import ctxnorm
    >>> info = ctxnorm.from_error(400, b'{"error": {"message": "Input prompt (140000 tokens) is too long '
    ...                                b'and exceeds limit of 131072"}}')
    >>> info.to_anthropic()[1]["error"]["message"]
    'prompt is too long: 140000 tokens > 131072 maximum'
"""
from .core import (
    STOP_LENGTH,
    Config,
    ContextExhausted,
    eligible,
    error_text,
    from_anthropic_message,
    from_error,
    from_openai_chat,
    from_stop,
    out_threshold,
)
from .limits import Limits
from .stream import lookahead

__version__ = "0.1.0"

__all__ = [
    "STOP_LENGTH", "Config", "ContextExhausted", "Limits", "eligible", "error_text", "from_anthropic_message",
    "from_error", "from_openai_chat", "from_stop", "lookahead", "out_threshold", "__version__",
]
