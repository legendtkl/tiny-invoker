from __future__ import annotations

import math
from dataclasses import dataclass

from tiny_invoker.interfaces import DecodeOutput, KVCache, PrefillOutput
from tiny_invoker.tokenizer import CharTokenizer


@dataclass(frozen=True)
class BigramKVCache:
    """Small cache used by the bigram demo model.

    A Transformer cache stores K/V tensors. The bigram model only needs token
    history, but keeping this shape lets the engine use prefill/decode.
    """

    token_ids: list[int]


@dataclass
class BigramLanguageModel:
    """A model that predicts the next token from only the previous token."""

    tokenizer: CharTokenizer
    logits_table: list[list[float]]

    @classmethod
    def from_corpus(
        cls,
        corpus: str,
        tokenizer: CharTokenizer,
        smoothing: float = 0.01,
    ) -> "BigramLanguageModel":
        if smoothing <= 0:
            raise ValueError("smoothing must be positive.")

        vocab_size = tokenizer.vocab_size
        counts = [[smoothing for _ in range(vocab_size)] for _ in range(vocab_size)]

        previous_id = tokenizer.bos_id
        for token_id in tokenizer.encode(corpus):
            counts[previous_id][token_id] += 1.0
            previous_id = token_id
            if tokenizer.id_to_token[token_id] == "\n":
                previous_id = tokenizer.bos_id

        logits_table = [
            [math.log(count) for count in row]
            for row in counts
        ]
        return cls(tokenizer=tokenizer, logits_table=logits_table)

    def next_logits(self, context_token_ids: list[int]) -> list[float]:
        previous_id = context_token_ids[-1] if context_token_ids else self.tokenizer.bos_id
        return list(self.logits_table[previous_id])

    def prefill(self, prompt_token_ids: list[int]) -> PrefillOutput:
        context_token_ids = prompt_token_ids[:] or [self.tokenizer.bos_id]
        return PrefillOutput(
            logits=self.next_logits(context_token_ids),
            cache=BigramKVCache(token_ids=context_token_ids),
        )

    def decode_one(self, token_id: int, cache: KVCache) -> DecodeOutput:
        if not isinstance(cache, BigramKVCache):
            raise TypeError("BigramLanguageModel expected BigramKVCache.")

        token_ids = cache.token_ids + [token_id]
        return DecodeOutput(
            logits=self.next_logits(token_ids),
            cache=BigramKVCache(token_ids=token_ids),
        )
