from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy Transformer runtime requires NumPy. Install it with "
            "`python3 -m pip install '.[runtime]'` from this repository."
        ) from error
    return np


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    layer_norm_epsilon: float
    num_layers: int
    num_heads: int
    intermediate_size: int
    attention_layers: tuple[str, ...]
    window_size: int
    activation: str = "gelu_new"
    position_embedding: str = "absolute"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class AttentionWeights:
    q_proj_weight: Any
    k_proj_weight: Any
    v_proj_weight: Any
    out_proj_weight: Any
    out_proj_bias: Any | None


@dataclass(frozen=True)
class MlpWeights:
    fc_weight: Any
    fc_bias: Any
    proj_weight: Any
    proj_bias: Any


@dataclass(frozen=True)
class TransformerBlockWeights:
    ln_1_weight: Any
    ln_1_bias: Any
    attention: AttentionWeights
    ln_2_weight: Any
    ln_2_bias: Any
    mlp: MlpWeights
    attention_type: str = "global"


@dataclass(frozen=True)
class TransformerWeights:
    token_embedding: Any
    position_embedding: Any | None
    layers: tuple[TransformerBlockWeights, ...]
    final_norm_weight: Any
    final_norm_bias: Any
    lm_head_weight: Any | None = None


@dataclass
class DecoderOnlyTransformer:
    config: TransformerConfig
    weights: TransformerWeights

    def forward(
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

        hidden_states = self._embed_tokens(token_array, start_position, end_position)
        hidden_states, keys, values = self._run_blocks(
            hidden_states,
            start_position=start_position,
            past_keys=past_keys,
            past_values=past_values,
        )
        hidden_states = layer_norm(
            hidden_states,
            self.weights.final_norm_weight,
            self.weights.final_norm_bias,
            epsilon=self.config.layer_norm_epsilon,
        )
        last_hidden = hidden_states[-1]
        lm_head_weight = self.weights.lm_head_weight
        if lm_head_weight is None:
            lm_head_weight = self.weights.token_embedding
        logits = last_hidden @ lm_head_weight.T
        return logits, keys, values

    def _embed_tokens(self, token_array: Any, start_position: int, end_position: int) -> Any:
        np = require_numpy()
        hidden_states = self.weights.token_embedding[token_array]
        if self.config.position_embedding != "absolute":
            raise ValueError(f"Unsupported position embedding: {self.config.position_embedding}.")
        if self.weights.position_embedding is None:
            raise ValueError("Absolute position embedding requires position weights.")
        position_ids = np.arange(start_position, end_position, dtype=np.int64)
        return hidden_states + self.weights.position_embedding[position_ids]

    def _run_blocks(
        self,
        hidden_states: Any,
        start_position: int,
        past_keys: tuple[Any, ...] | None,
        past_values: tuple[Any, ...] | None,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        next_keys: list[Any] = []
        next_values: list[Any] = []
        for layer_idx, layer_weights in enumerate(self.weights.layers):
            residual = hidden_states
            attn_input = layer_norm(
                hidden_states,
                layer_weights.ln_1_weight,
                layer_weights.ln_1_bias,
                epsilon=self.config.layer_norm_epsilon,
            )
            layer_past_key = None if past_keys is None else past_keys[layer_idx]
            layer_past_value = None if past_values is None else past_values[layer_idx]
            attn_output, key, value = self._self_attention(
                layer_idx=layer_idx,
                layer_weights=layer_weights,
                hidden_states=attn_input,
                start_position=start_position,
                past_key=layer_past_key,
                past_value=layer_past_value,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = layer_norm(
                hidden_states,
                layer_weights.ln_2_weight,
                layer_weights.ln_2_bias,
                epsilon=self.config.layer_norm_epsilon,
            )
            hidden_states = residual + self._mlp(layer_weights.mlp, mlp_input)
            next_keys.append(key)
            next_values.append(value)
        return hidden_states, tuple(next_keys), tuple(next_values)

    def _self_attention(
        self,
        layer_idx: int,
        layer_weights: TransformerBlockWeights,
        hidden_states: Any,
        start_position: int,
        past_key: Any | None,
        past_value: Any | None,
    ) -> tuple[Any, Any, Any]:
        np = require_numpy()
        attention_weights = layer_weights.attention
        query = split_heads(
            linear(hidden_states, attention_weights.q_proj_weight),
            self.config.num_heads,
        )
        key = split_heads(
            linear(hidden_states, attention_weights.k_proj_weight),
            self.config.num_heads,
        )
        value = split_heads(
            linear(hidden_states, attention_weights.v_proj_weight),
            self.config.num_heads,
        )
        end_position = start_position + hidden_states.shape[0]
        key_cache = self._updated_kv_cache(
            past_cache=past_key,
            new_values=key,
            start_position=start_position,
            end_position=end_position,
        )
        value_cache = self._updated_kv_cache(
            past_cache=past_value,
            new_values=value,
            start_position=start_position,
            end_position=end_position,
        )
        key = key_cache[:, :end_position, :]
        value = value_cache[:, :end_position, :]

        scores = (query @ np.swapaxes(key, -1, -2)) / np.sqrt(self.config.head_dim)
        mask = self._attention_mask(
            query_length=hidden_states.shape[0],
            key_length=key.shape[1],
            start_position=start_position,
            attention_type=layer_weights.attention_type,
        )
        if mask is not None:
            scores = np.where(mask[None, :, :], scores, -1.0e9)
        probabilities = softmax(scores, axis=-1)
        context = probabilities @ value
        merged_context = merge_heads(context)
        output = linear(
            merged_context,
            attention_weights.out_proj_weight,
            attention_weights.out_proj_bias,
        )
        return output, key_cache, value_cache

    def _updated_kv_cache(
        self,
        past_cache: Any | None,
        new_values: Any,
        start_position: int,
        end_position: int,
    ) -> Any:
        np = require_numpy()
        if past_cache is None:
            cache = np.empty(
                (
                    new_values.shape[0],
                    self.config.max_position_embeddings,
                    new_values.shape[2],
                ),
                dtype=new_values.dtype,
            )
        elif past_cache.shape[1] >= end_position:
            cache = past_cache
        else:
            cache = np.empty(
                (
                    past_cache.shape[0],
                    self.config.max_position_embeddings,
                    past_cache.shape[2],
                ),
                dtype=past_cache.dtype,
            )
            cache[:, :start_position, :] = past_cache[:, :start_position, :]

        cache[:, start_position:end_position, :] = new_values
        return cache

    def _attention_mask(
        self,
        query_length: int,
        key_length: int,
        start_position: int,
        attention_type: str,
    ) -> Any | None:
        np = require_numpy()
        if query_length == 1 and key_length == start_position + 1:
            if attention_type != "local":
                return None
            min_visible_position = max(0, start_position - self.config.window_size + 1)
            if min_visible_position == 0:
                return None
            return np.arange(key_length)[None, :] >= min_visible_position

        query_positions = np.arange(start_position, start_position + query_length)[:, None]
        key_positions = np.arange(key_length)[None, :]
        mask = key_positions <= query_positions
        if attention_type == "local":
            mask = mask & (key_positions > query_positions - self.config.window_size)
        return mask

    def _mlp(self, weights: MlpWeights, hidden_states: Any) -> Any:
        hidden_states = linear(hidden_states, weights.fc_weight, weights.fc_bias)
        hidden_states = self._activate(hidden_states)
        return linear(hidden_states, weights.proj_weight, weights.proj_bias)

    def _activate(self, hidden_states: Any) -> Any:
        if self.config.activation == "gelu_new":
            return gelu_new(hidden_states)
        raise ValueError(f"Unsupported activation: {self.config.activation}.")


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
