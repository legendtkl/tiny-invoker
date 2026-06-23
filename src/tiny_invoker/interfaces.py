from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from tiny_invoker.tokenizer import TextTokenizer


class KVCache(Protocol):
    """Model-specific state reused between decode steps."""

    @property
    def length(self) -> int:
        """Number of tokens represented by the cache."""
        ...


class ForwardMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ForwardInput:
    token_ids: list[int]
    mode: ForwardMode
    cache: KVCache | None = None


@dataclass(frozen=True)
class ForwardOutput:
    logits: Any
    cache: KVCache


class LanguageModel(Protocol):
    """Minimum interface required by the inference engine."""

    tokenizer: TextTokenizer

    def forward(self, request: ForwardInput) -> ForwardOutput:
        """Run a model step for either prefill or decode mode."""
