from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    token_id: int
    probability: float


def softmax(logits: Any, temperature: float = 1.0) -> Any:
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax.")
    if is_numpy_logits(logits):
        return softmax_numpy(logits, temperature=temperature)

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


def filter_top_k(
    logits: Any,
    top_k: int | None,
    blocked_token_ids: Iterable[int] = (),
) -> Any:
    if is_numpy_logits(logits):
        return filter_top_k_numpy(
            logits,
            top_k=top_k,
            blocked_token_ids=blocked_token_ids,
        )

    blocked_ids = valid_token_id_set(blocked_token_ids, len(logits))
    masked_logits = [
        float("-inf") if token_id in blocked_ids else logit
        for token_id, logit in enumerate(logits)
    ]
    if top_k is None or top_k >= len(logits):
        return masked_logits
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")

    ranked_ids = sorted(range(len(masked_logits)), key=lambda token_id: masked_logits[token_id], reverse=True)
    kept_ids = set(ranked_ids[:top_k])
    return [
        logit if token_id in kept_ids else float("-inf")
        for token_id, logit in enumerate(masked_logits)
    ]


def choose_token(
    logits: Any,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    blocked_token_ids: Iterable[int] = (),
) -> int:
    if is_numpy_logits(logits):
        return choose_token_numpy(
            logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            blocked_token_ids=blocked_token_ids,
        )

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    if temperature == 0:
        blocked_ids = valid_token_id_set(blocked_token_ids, len(logits))
        return max(
            range(len(logits)),
            key=lambda token_id: float("-inf")
            if token_id in blocked_ids
            else logits[token_id],
        )

    filtered_logits = filter_top_k(
        logits,
        top_k=top_k,
        blocked_token_ids=blocked_token_ids,
    )
    probabilities = softmax(filtered_logits, temperature=temperature)
    threshold = rng.random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return token_id
    return len(probabilities) - 1


def top_candidates(
    logits: Any,
    temperature: float = 1.0,
    top_n: int = 5,
    top_k: int | None = None,
    blocked_token_ids: Iterable[int] = (),
) -> list[Candidate]:
    if top_n <= 0:
        return []
    if is_numpy_logits(logits):
        return top_candidates_numpy(
            logits,
            temperature=temperature,
            top_n=top_n,
            top_k=top_k,
            blocked_token_ids=blocked_token_ids,
        )

    filtered_logits = filter_top_k(
        logits,
        top_k=top_k,
        blocked_token_ids=blocked_token_ids,
    )
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


def is_numpy_logits(logits: Any) -> bool:
    return hasattr(logits, "__array__") and hasattr(logits, "shape") and hasattr(logits, "dtype")


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy logits require NumPy to be installed.") from error
    return np


def valid_token_id_set(token_ids: Iterable[int], vocab_size: int) -> set[int]:
    return {
        int(token_id)
        for token_id in token_ids
        if 0 <= int(token_id) < vocab_size
    }


def numpy_logits_copy(logits: Any) -> Any:
    np = require_numpy()
    values = np.asarray(logits)
    if values.ndim != 1:
        raise ValueError("Sampler expects 1-D logits.")
    return values.astype(np.float64, copy=True)


def softmax_numpy(logits: Any, temperature: float = 1.0) -> Any:
    np = require_numpy()
    values = numpy_logits_copy(logits)
    scaled = np.where(np.isfinite(values), values / temperature, -np.inf)
    max_logit = np.max(scaled)
    if not np.isfinite(max_logit):
        raise ValueError("At least one logit must be finite.")
    exp_values = np.where(np.isfinite(scaled), np.exp(scaled - max_logit), 0.0)
    return exp_values / np.sum(exp_values)


def filter_top_k_numpy(
    logits: Any,
    top_k: int | None,
    blocked_token_ids: Iterable[int] = (),
) -> Any:
    np = require_numpy()
    values = masked_numpy_logits(logits, blocked_token_ids=blocked_token_ids)
    if top_k is None or top_k >= values.size:
        return values
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")

    kept_ids = np.argpartition(values, -top_k)[-top_k:]
    filtered = np.full(values.shape, -np.inf, dtype=values.dtype)
    filtered[kept_ids] = values[kept_ids]
    return filtered


def masked_numpy_logits(logits: Any, blocked_token_ids: Iterable[int] = ()) -> Any:
    np = require_numpy()
    values = numpy_logits_copy(logits)
    blocked_ids = sorted(valid_token_id_set(blocked_token_ids, values.size))
    if blocked_ids:
        values[np.asarray(blocked_ids, dtype=np.int64)] = -np.inf
    return values


def choose_token_numpy(
    logits: Any,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    blocked_token_ids: Iterable[int] = (),
) -> int:
    np = require_numpy()
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided.")
    if temperature == 0:
        return int(np.argmax(masked_numpy_logits(logits, blocked_token_ids=blocked_token_ids)))

    filtered_logits = filter_top_k_numpy(
        logits,
        top_k=top_k,
        blocked_token_ids=blocked_token_ids,
    )
    probabilities = softmax_numpy(filtered_logits, temperature=temperature)
    threshold = rng.random()
    cumulative = np.cumsum(probabilities)
    token_id = int(np.searchsorted(cumulative, threshold, side="left"))
    return min(token_id, probabilities.size - 1)


def top_candidates_numpy(
    logits: Any,
    temperature: float = 1.0,
    top_n: int = 5,
    top_k: int | None = None,
    blocked_token_ids: Iterable[int] = (),
) -> list[Candidate]:
    if top_n <= 0:
        return []

    np = require_numpy()
    filtered_logits = filter_top_k_numpy(
        logits,
        top_k=top_k,
        blocked_token_ids=blocked_token_ids,
    )
    probabilities = softmax_numpy(filtered_logits, temperature=temperature)
    finite_ids = np.flatnonzero(np.isfinite(filtered_logits))
    if finite_ids.size == 0:
        return []

    top_count = min(top_n, finite_ids.size)
    finite_probabilities = probabilities[finite_ids]
    if top_count < finite_ids.size:
        local_ids = np.argpartition(finite_probabilities, -top_count)[-top_count:]
    else:
        local_ids = np.arange(finite_ids.size)
    ordered_local_ids = local_ids[np.argsort(finite_probabilities[local_ids])[::-1]]
    return [
        Candidate(
            token_id=int(finite_ids[local_id]),
            probability=float(finite_probabilities[local_id]),
        )
        for local_id in ordered_local_ids
    ]
