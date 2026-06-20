from __future__ import annotations

import random
from dataclasses import dataclass

from tiny_invoker.interfaces import ForwardInput, ForwardMode, ForwardOutput, KVCache, LanguageModel
from tiny_invoker.sampler import choose_token, top_candidates


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int | None = 8
    seed: int | None = None
    trace: bool = False

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")


@dataclass(frozen=True)
class GenerationStep:
    step: int
    previous_token: str
    chosen_token: str
    candidates: list[tuple[str, float]]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    steps: list[GenerationStep]


@dataclass
class InferenceEngine:
    model: LanguageModel

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        config = config or GenerationConfig()
        config.validate()

        tokenizer = self.model.tokenizer
        rng = random.Random(config.seed)
        prompt_token_ids = tokenizer.encode(prompt)
        context_token_ids = prompt_token_ids[:] or [tokenizer.bos_id]
        generated_token_ids: list[int] = []
        steps: list[GenerationStep] = []
        prefill_output = self._prefill(context_token_ids)
        logits = prefill_output.logits
        cache = prefill_output.cache

        for step in range(config.max_new_tokens):
            chosen_id = choose_token(
                logits,
                rng=rng,
                temperature=config.temperature,
                top_k=config.top_k,
                blocked_token_ids=tokenizer.special_token_ids,
            )
            context_token_ids.append(chosen_id)
            generated_token_ids.append(chosen_id)

            if config.trace:
                previous_id = context_token_ids[-2]
                candidates = [
                    (tokenizer.decode([candidate.token_id]), candidate.probability)
                    for candidate in top_candidates(
                        logits,
                        temperature=max(config.temperature, 1e-12),
                        top_k=config.top_k,
                        blocked_token_ids=tokenizer.special_token_ids,
                    )
                ]
                steps.append(
                    GenerationStep(
                        step=step + 1,
                        previous_token=tokenizer.decode([previous_id]) or tokenizer.bos_token,
                        chosen_token=tokenizer.decode([chosen_id]),
                        candidates=candidates,
                    )
                )

            if step < config.max_new_tokens - 1:
                decode_output = self._decode_one(chosen_id, cache)
                logits = decode_output.logits
                cache = decode_output.cache

        return GenerationResult(
            text=prompt + tokenizer.decode(generated_token_ids),
            token_ids=prompt_token_ids + generated_token_ids,
            steps=steps,
        )

    def _prefill(self, prompt_token_ids: list[int]) -> ForwardOutput:
        return self.model.forward(
            ForwardInput(
                token_ids=prompt_token_ids,
                mode=ForwardMode.PREFILL,
            )
        )

    def _decode_one(self, token_id: int, cache: KVCache) -> ForwardOutput:
        return self.model.forward(
            ForwardInput(
                token_ids=[token_id],
                mode=ForwardMode.DECODE,
                cache=cache,
            )
        )
