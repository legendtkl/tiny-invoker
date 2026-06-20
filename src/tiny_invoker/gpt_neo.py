from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_invoker.interfaces import ForwardInput, ForwardMode, ForwardOutput
from tiny_invoker.tokenizer import TextTokenizer


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy GPT-Neo runtime requires NumPy. Install it with "
            "`python3 -m pip install '.[runtime]'` from this repository."
        ) from error
    return np


@dataclass(frozen=True)
class NumpyGptNeoConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    layer_norm_epsilon: float
    num_layers: int
    num_heads: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumpyGptNeoConfig":
        if payload.get("model_type") != "gpt_neo":
            raise ValueError("Only GPT-Neo configs are supported by this runtime skeleton.")
        return cls(
            vocab_size=int(payload["vocab_size"]),
            hidden_size=int(payload["hidden_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            layer_norm_epsilon=float(payload.get("layer_norm_epsilon", 1e-5)),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "NumpyGptNeoConfig":
        with Path(path).open("r", encoding="utf-8") as config_file:
            payload = json.load(config_file)
        if not isinstance(payload, dict):
            raise ValueError("Expected GPT-Neo config JSON object.")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class NumpyGptNeoCache:
    token_ids: list[int]


@dataclass
class NumpyGptNeoLanguageModel:
    tokenizer: TextTokenizer
    config: NumpyGptNeoConfig
    weights: Any

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        tokenizer: TextTokenizer,
    ) -> "NumpyGptNeoLanguageModel":
        config = NumpyGptNeoConfig.from_file(config_path)
        weights = load_npz_weights(weights_path)
        model = cls(tokenizer=tokenizer, config=config, weights=weights)
        model.validate_runtime_weights()
        return model

    def validate_runtime_weights(self) -> None:
        available_weights = set(weight_names(self.weights))
        required_shapes = {
            "transformer.wte.weight": (self.config.vocab_size, self.config.hidden_size),
            "transformer.wpe.weight": (self.config.max_position_embeddings, self.config.hidden_size),
            "transformer.ln_f.weight": (self.config.hidden_size,),
            "transformer.ln_f.bias": (self.config.hidden_size,),
        }
        for name, expected_shape in required_shapes.items():
            if name not in available_weights:
                raise ValueError(f"Missing runtime weight {name}.")
            actual_shape = tuple(int(dimension) for dimension in self.weights[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(f"{name} has shape {actual_shape}, expected {expected_shape}.")

    def forward(self, request: ForwardInput) -> ForwardOutput:
        if request.mode == ForwardMode.PREFILL:
            return self._forward_prefill(request.token_ids)
        if request.mode == ForwardMode.DECODE:
            return self._forward_decode(request)
        raise ValueError(f"Unsupported forward mode: {request.mode}.")

    def _forward_prefill(self, token_ids: list[int]) -> ForwardOutput:
        context_token_ids = token_ids[:] or [self.tokenizer.bos_id]
        logits = self._compute_logits(context_token_ids, start_position=0)
        return ForwardOutput(
            logits=logits.tolist(),
            cache=NumpyGptNeoCache(token_ids=context_token_ids),
        )

    def _forward_decode(self, request: ForwardInput) -> ForwardOutput:
        if len(request.token_ids) != 1:
            raise ValueError("GPT-Neo decode mode expects exactly one token id.")
        if not isinstance(request.cache, NumpyGptNeoCache):
            raise TypeError("NumpyGptNeoLanguageModel expected NumpyGptNeoCache.")

        start_position = len(request.cache.token_ids)
        logits = self._compute_logits(request.token_ids, start_position=start_position)
        return ForwardOutput(
            logits=logits.tolist(),
            cache=NumpyGptNeoCache(token_ids=request.cache.token_ids + request.token_ids),
        )

    def _compute_logits(self, token_ids: list[int], start_position: int) -> Any:
        np = require_numpy()
        token_array = np.asarray(token_ids, dtype=np.int64)
        if token_array.size == 0:
            raise ValueError("Cannot compute logits for an empty token list.")

        end_position = start_position + len(token_ids)
        if end_position > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {end_position} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}."
            )

        position_ids = np.arange(start_position, end_position, dtype=np.int64)
        hidden_states = self.weights["transformer.wte.weight"][token_array]
        hidden_states = hidden_states + self.weights["transformer.wpe.weight"][position_ids]
        hidden_states = self._run_blocks(hidden_states)
        hidden_states = layer_norm(
            hidden_states,
            self.weights["transformer.ln_f.weight"],
            self.weights["transformer.ln_f.bias"],
            epsilon=self.config.layer_norm_epsilon,
        )
        last_hidden = hidden_states[-1]
        return last_hidden @ self.weights["transformer.wte.weight"].T

    def _run_blocks(self, hidden_states: Any) -> Any:
        # The runtime skeleton intentionally wires the data path before adding
        # GPT-Neo attention and MLP blocks. The next step will replace this.
        return hidden_states


def layer_norm(hidden_states: Any, weight: Any, bias: Any, epsilon: float) -> Any:
    np = require_numpy()
    mean = np.mean(hidden_states, axis=-1, keepdims=True)
    variance = np.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
    normalized = (hidden_states - mean) / np.sqrt(variance + epsilon)
    return normalized * weight + bias


def load_npz_weights(path: str | Path) -> Any:
    np = require_numpy()
    return np.load(Path(path), allow_pickle=False)


def weight_names(weights: Any) -> list[str]:
    if hasattr(weights, "files"):
        return list(weights.files)
    if hasattr(weights, "keys"):
        return list(weights.keys())
    raise TypeError("Unsupported weight container.")
