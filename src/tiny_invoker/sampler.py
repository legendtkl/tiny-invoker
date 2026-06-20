from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    token_id: int
    probability: float


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax.")

    scaled = [
        logit / temperature if math.isfinite(logit) else float("-inf")
        for logit in logits
    ]
    max_logit = max(scaled)
    if not math.isfinite(max_logit):
        raise ValueError("At least one logit must be finite.")

    exp_values = [
        math.exp(logit - max_logit) if math.isfinite(logit) else 0.0
        for logit in scaled
    ]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def filter_top_k(logits: list[float], top_k: int | None) -> list[float]:
    if top_k is None or top_k >= len(logits):
        return list(logits)
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")

    ranked_ids = sorted(range(len(logits)), key=lambda token_id: logits[token_id], reverse=True)
    kept_ids = set(ranked_ids[:top_k])
    return [
        logit if token_id in kept_ids else float("-inf")
        for token_id, logit in enumerate(logits)
    ]


def choose_token(
    logits: list[float],
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> int:
    filtered_logits = filter_top_k(logits, top_k)
    if temperature == 0:
        return max(range(len(filtered_logits)), key=lambda token_id: filtered_logits[token_id])

    probabilities = softmax(filtered_logits, temperature=temperature)
    threshold = rng.random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return token_id
    return len(probabilities) - 1


def top_candidates(
    logits: list[float],
    temperature: float = 1.0,
    top_n: int = 5,
    top_k: int | None = None,
) -> list[Candidate]:
    if top_n <= 0:
        return []
    filtered_logits = filter_top_k(logits, top_k)
    probabilities = softmax(filtered_logits, temperature=temperature)
    candidate_ids = [
        token_id
        for token_id, logit in enumerate(filtered_logits)
        if math.isfinite(logit)
    ]
    ranked_ids = sorted(
        candidate_ids,
        key=lambda token_id: probabilities[token_id],
        reverse=True,
    )
    return [
        Candidate(token_id=token_id, probability=probabilities[token_id])
        for token_id in ranked_ids[:top_n]
    ]
