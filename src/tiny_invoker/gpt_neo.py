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
    intermediate_size: int
    attention_layers: tuple[str, ...]
    window_size: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumpyGptNeoConfig":
        if payload.get("model_type") != "gpt_neo":
            raise ValueError("Only GPT-Neo configs are supported by this runtime skeleton.")
        hidden_size = int(payload["hidden_size"])
        num_layers = int(payload["num_layers"])
        num_heads = int(payload["num_heads"])
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        intermediate_size = payload.get("intermediate_size")
        attention_layers = payload.get("attention_layers")
        if not isinstance(attention_layers, list):
            attention_layers = ["global"] * num_layers
        return cls(
            vocab_size=int(payload["vocab_size"]),
            hidden_size=hidden_size,
            max_position_embeddings=int(payload["max_position_embeddings"]),
            layer_norm_epsilon=float(payload.get("layer_norm_epsilon", 1e-5)),
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=int(intermediate_size) if intermediate_size is not None else 4 * hidden_size,
            attention_layers=tuple(str(layer) for layer in attention_layers),
            window_size=int(payload.get("window_size", payload["max_position_embeddings"])),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "NumpyGptNeoConfig":
        with Path(path).open("r", encoding="utf-8") as config_file:
            payload = json.load(config_file)
        if not isinstance(payload, dict):
            raise ValueError("Expected GPT-Neo config JSON object.")
        return cls.from_dict(payload)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class NumpyGptNeoCache:
    token_ids: list[int]
    keys: tuple[Any, ...]
    values: tuple[Any, ...]


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
        for layer_idx in range(self.config.num_layers):
            prefix = f"transformer.h.{layer_idx}"
            required_shapes.update(
                {
                    f"{prefix}.ln_1.weight": (self.config.hidden_size,),
                    f"{prefix}.ln_1.bias": (self.config.hidden_size,),
                    f"{prefix}.ln_2.weight": (self.config.hidden_size,),
                    f"{prefix}.ln_2.bias": (self.config.hidden_size,),
                    f"{prefix}.attn.attention.q_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.attn.attention.k_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.attn.attention.v_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.attn.attention.out_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.attn.attention.out_proj.bias": (self.config.hidden_size,),
                    f"{prefix}.mlp.c_fc.weight": (
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.mlp.c_fc.bias": (self.config.intermediate_size,),
                    f"{prefix}.mlp.c_proj.weight": (
                        self.config.hidden_size,
                        self.config.intermediate_size,
                    ),
                    f"{prefix}.mlp.c_proj.bias": (self.config.hidden_size,),
                }
            )
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
        logits, keys, values = self._compute_logits(
            context_token_ids,
            start_position=0,
            past_keys=None,
            past_values=None,
        )
        return ForwardOutput(
            logits=logits.tolist(),
            cache=NumpyGptNeoCache(token_ids=context_token_ids, keys=keys, values=values),
        )

    def _forward_decode(self, request: ForwardInput) -> ForwardOutput:
        if len(request.token_ids) != 1:
            raise ValueError("GPT-Neo decode mode expects exactly one token id.")
        if not isinstance(request.cache, NumpyGptNeoCache):
            raise TypeError("NumpyGptNeoLanguageModel expected NumpyGptNeoCache.")

        start_position = len(request.cache.token_ids)
        logits, keys, values = self._compute_logits(
            request.token_ids,
            start_position=start_position,
            past_keys=request.cache.keys,
            past_values=request.cache.values,
        )
        return ForwardOutput(
            logits=logits.tolist(),
            cache=NumpyGptNeoCache(
                token_ids=request.cache.token_ids + request.token_ids,
                keys=keys,
                values=values,
            ),
        )

    def _compute_logits(
        self,
        token_ids: list[int],
        start_position: int,
        past_keys: tuple[Any, ...] | None,
        past_values: tuple[Any, ...] | None,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
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
        hidden_states, keys, values = self._run_blocks(
            hidden_states,
            start_position=start_position,
            past_keys=past_keys,
            past_values=past_values,
        )
        hidden_states = layer_norm(
            hidden_states,
            self.weights["transformer.ln_f.weight"],
            self.weights["transformer.ln_f.bias"],
            epsilon=self.config.layer_norm_epsilon,
        )
        last_hidden = hidden_states[-1]
        logits = last_hidden @ self.weights["transformer.wte.weight"].T
        return logits, keys, values

    def _run_blocks(
        self,
        hidden_states: Any,
        start_position: int,
        past_keys: tuple[Any, ...] | None,
        past_values: tuple[Any, ...] | None,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        next_keys: list[Any] = []
        next_values: list[Any] = []
        for layer_idx in range(self.config.num_layers):
            prefix = f"transformer.h.{layer_idx}"
            residual = hidden_states
            attn_input = layer_norm(
                hidden_states,
                self.weights[f"{prefix}.ln_1.weight"],
                self.weights[f"{prefix}.ln_1.bias"],
                epsilon=self.config.layer_norm_epsilon,
            )
            layer_past_key = None if past_keys is None else past_keys[layer_idx]
            layer_past_value = None if past_values is None else past_values[layer_idx]
            attn_output, key, value = self._self_attention(
                layer_idx=layer_idx,
                hidden_states=attn_input,
                start_position=start_position,
                past_key=layer_past_key,
                past_value=layer_past_value,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = layer_norm(
                hidden_states,
                self.weights[f"{prefix}.ln_2.weight"],
                self.weights[f"{prefix}.ln_2.bias"],
                epsilon=self.config.layer_norm_epsilon,
            )
            hidden_states = residual + self._mlp(layer_idx, mlp_input)
            next_keys.append(key)
            next_values.append(value)
        return hidden_states, tuple(next_keys), tuple(next_values)

    def _self_attention(
        self,
        layer_idx: int,
        hidden_states: Any,
        start_position: int,
        past_key: Any | None,
        past_value: Any | None,
    ) -> tuple[Any, Any, Any]:
        np = require_numpy()
        prefix = f"transformer.h.{layer_idx}.attn.attention"
        query = split_heads(
            linear(hidden_states, self.weights[f"{prefix}.q_proj.weight"]),
            self.config.num_heads,
        )
        key = split_heads(
            linear(hidden_states, self.weights[f"{prefix}.k_proj.weight"]),
            self.config.num_heads,
        )
        value = split_heads(
            linear(hidden_states, self.weights[f"{prefix}.v_proj.weight"]),
            self.config.num_heads,
        )
        if past_key is not None:
            key = np.concatenate([past_key, key], axis=1)
        if past_value is not None:
            value = np.concatenate([past_value, value], axis=1)

        scores = (query @ np.swapaxes(key, -1, -2)) / np.sqrt(self.config.head_dim)
        mask = self._attention_mask(
            query_length=hidden_states.shape[0],
            key_length=key.shape[1],
            start_position=start_position,
            layer_idx=layer_idx,
        )
        scores = np.where(mask[None, :, :], scores, -1.0e9)
        probabilities = softmax(scores, axis=-1)
        context = probabilities @ value
        merged_context = merge_heads(context)
        output = linear(
            merged_context,
            self.weights[f"{prefix}.out_proj.weight"],
            self.weights[f"{prefix}.out_proj.bias"],
        )
        return output, key, value

    def _attention_mask(
        self,
        query_length: int,
        key_length: int,
        start_position: int,
        layer_idx: int,
    ) -> Any:
        np = require_numpy()
        query_positions = np.arange(start_position, start_position + query_length)[:, None]
        key_positions = np.arange(key_length)[None, :]
        mask = key_positions <= query_positions
        if self._attention_type(layer_idx) == "local":
            mask = mask & (key_positions > query_positions - self.config.window_size)
        return mask

    def _attention_type(self, layer_idx: int) -> str:
        if layer_idx < len(self.config.attention_layers):
            return self.config.attention_layers[layer_idx]
        return "global"

    def _mlp(self, layer_idx: int, hidden_states: Any) -> Any:
        prefix = f"transformer.h.{layer_idx}.mlp"
        hidden_states = linear(
            hidden_states,
            self.weights[f"{prefix}.c_fc.weight"],
            self.weights[f"{prefix}.c_fc.bias"],
        )
        hidden_states = gelu_new(hidden_states)
        return linear(
            hidden_states,
            self.weights[f"{prefix}.c_proj.weight"],
            self.weights[f"{prefix}.c_proj.bias"],
        )


def linear(hidden_states: Any, weight: Any, bias: Any | None = None) -> Any:
    output = hidden_states @ weight.T
    if bias is not None:
        output = output + bias
    return output


def gelu_new(hidden_states: Any) -> Any:
    np = require_numpy()
    return 0.5 * hidden_states * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (hidden_states + 0.044715 * np.power(hidden_states, 3))
        )
    )


def split_heads(hidden_states: Any, num_heads: int) -> Any:
    sequence_length, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads
    return hidden_states.reshape(sequence_length, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(hidden_states: Any) -> Any:
    num_heads, sequence_length, head_dim = hidden_states.shape
    return hidden_states.transpose(1, 0, 2).reshape(sequence_length, num_heads * head_dim)


def softmax(values: Any, axis: int = -1) -> Any:
    np = require_numpy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def layer_norm(hidden_states: Any, weight: Any, bias: Any, epsilon: float) -> Any:
    np = require_numpy()
    mean = np.mean(hidden_states, axis=-1, keepdims=True)
    variance = np.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
    normalized = (hidden_states - mean) / np.sqrt(variance + epsilon)
    return normalized * weight + bias


def load_npz_weights(path: str | Path) -> Any:
    np = require_numpy()
    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def weight_names(weights: Any) -> list[str]:
    if hasattr(weights, "files"):
        return list(weights.files)
    if hasattr(weights, "keys"):
        return list(weights.keys())
    raise TypeError("Unsupported weight container.")
