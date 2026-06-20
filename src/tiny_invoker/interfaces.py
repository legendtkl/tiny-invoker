from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tiny_invoker.tokenizer import TextTokenizer


class KVCache(Protocol):
    """Model-specific state reused between decode steps."""


@dataclass(frozen=True)
class PrefillOutput:
    logits: list[float]
    cache: KVCache


@dataclass(frozen=True)
class DecodeOutput:
    logits: list[float]
    cache: KVCache


class LanguageModel(Protocol):
    """Minimum interface required by the inference engine."""

    tokenizer: TextTokenizer

    def prefill(self, prompt_token_ids: list[int]) -> PrefillOutput:
        """Process the prompt and return scores for the first generated token."""

    def decode_one(self, token_id: int, cache: KVCache) -> DecodeOutput:
        """Process one new token using the existing cache."""
