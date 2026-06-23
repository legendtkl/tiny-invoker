from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    num_key_value_heads: int | None = None
    activation: str = "gelu_new"
    norm_type: str = "layer_norm"
    position_embedding: str = "absolute"
    scale_attention_scores: bool = True
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def key_value_heads(self) -> int:
        return self.num_key_value_heads or self.num_heads

    @property
    def key_value_group_size(self) -> int:
        return self.num_heads // self.key_value_heads


@dataclass(frozen=True)
class AttentionWeights:
    q_proj_weight_t: Any
    k_proj_weight_t: Any
    v_proj_weight_t: Any
    out_proj_weight_t: Any
    out_proj_bias: Any | None
    q_proj_bias: Any | None = None
    k_proj_bias: Any | None = None
    v_proj_bias: Any | None = None


@dataclass(frozen=True)
class MlpWeights:
    fc_weight_t: Any
    fc_bias: Any
    proj_weight_t: Any
    proj_bias: Any
    gate_weight_t: Any | None = None
    gate_bias: Any | None = None


@dataclass(frozen=True)
class TransformerBlockWeights:
    ln_1_weight: Any
    ln_1_bias: Any | None
    attention: AttentionWeights
    ln_2_weight: Any
    ln_2_bias: Any | None
    mlp: MlpWeights
    attention_type: str = "global"


@dataclass(frozen=True)
class TransformerWeights:
    token_embedding: Any
    position_embedding: Any | None
    layers: tuple[TransformerBlockWeights, ...]
    final_norm_weight: Any
    final_norm_bias: Any | None
    lm_head_weight: Any | None = None
    lm_head_weight_t: Any | None = None


@dataclass
class DecoderOnlyTransformer:
    config: TransformerConfig
    weights: TransformerWeights
    _rope_cos_cache: Any | None = field(default=None, init=False, repr=False)
    _rope_sin_cache: Any | None = field(default=None, init=False, repr=False)
    _attention_mask_cache: dict[tuple[int, int, int, str], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def forward(
        self,
        token_ids: list[int],
        start_position: int,
        past_keys: tuple[Any, ...] | None,
        past_values: tuple[Any, ...] | None,
        profile: dict[str, float] | None = None,
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

        timer = profile_start(profile)
        hidden_states = self._embed_tokens(token_array, start_position, end_position)
        profile_add(profile, "embedding_ms", timer)

        timer = profile_start(profile)
        hidden_states, keys, values = self._run_blocks(
            hidden_states,
            start_position=start_position,
            past_keys=past_keys,
            past_values=past_values,
            profile=profile,
        )
        profile_add(profile, "blocks_ms", timer)

        timer = profile_start(profile)
        hidden_states = apply_norm(
            self.config.norm_type,
            hidden_states,
            self.weights.final_norm_weight,
            self.weights.final_norm_bias,
            epsilon=self.config.layer_norm_epsilon,
        )
        profile_add(profile, "final_norm_ms", timer)

        last_hidden = hidden_states[-1]
        lm_head_weight_t = self.weights.lm_head_weight_t
        if lm_head_weight_t is None:
            lm_head_weight = self.weights.lm_head_weight
            if lm_head_weight is None:
                lm_head_weight = self.weights.token_embedding
            lm_head_weight_t = lm_head_weight.T

        timer = profile_start(profile)
        logits = last_hidden @ lm_head_weight_t
        profile_add(profile, "lm_head_ms", timer)
        return logits, keys, values

    def _embed_tokens(self, token_array: Any, start_position: int, end_position: int) -> Any:
        np = require_numpy()
        hidden_states = self.weights.token_embedding[token_array]
        if self.config.position_embedding == "absolute":
            if self.weights.position_embedding is None:
                raise ValueError("Absolute position embedding requires position weights.")
            position_ids = np.arange(start_position, end_position, dtype=np.int64)
            return hidden_states + self.weights.position_embedding[position_ids]
        if self.config.position_embedding in {"none", "rope"}:
            return hidden_states
        raise ValueError(f"Unsupported position embedding: {self.config.position_embedding}.")

    def _run_blocks(
        self,
        hidden_states: Any,
        start_position: int,
        past_keys: tuple[Any, ...] | None,
        past_values: tuple[Any, ...] | None,
        profile: dict[str, float] | None = None,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        next_keys: list[Any] = []
        next_values: list[Any] = []
        for layer_idx, layer_weights in enumerate(self.weights.layers):
            residual = hidden_states
            attn_input = apply_norm(
                self.config.norm_type,
                hidden_states,
                layer_weights.ln_1_weight,
                layer_weights.ln_1_bias,
                epsilon=self.config.layer_norm_epsilon,
            )
            layer_past_key = None if past_keys is None else past_keys[layer_idx]
            layer_past_value = None if past_values is None else past_values[layer_idx]
            timer = profile_start(profile)
            attn_output, key, value = self._self_attention(
                layer_idx=layer_idx,
                layer_weights=layer_weights,
                hidden_states=attn_input,
                start_position=start_position,
                past_key=layer_past_key,
                past_value=layer_past_value,
                profile=profile,
            )
            profile_add(profile, "attention_ms", timer)
            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = apply_norm(
                self.config.norm_type,
                hidden_states,
                layer_weights.ln_2_weight,
                layer_weights.ln_2_bias,
                epsilon=self.config.layer_norm_epsilon,
            )
            timer = profile_start(profile)
            mlp_output = self._mlp(layer_weights.mlp, mlp_input, profile=profile)
            profile_add(profile, "mlp_ms", timer)
            hidden_states = residual + mlp_output
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
        profile: dict[str, float] | None = None,
    ) -> tuple[Any, Any, Any]:
        attention_weights = layer_weights.attention
        timer = profile_start(profile)
        query = split_heads(
            linear_t(
                hidden_states,
                attention_weights.q_proj_weight_t,
                attention_weights.q_proj_bias,
            ),
            self.config.num_heads,
        )
        key = split_heads(
            linear_t(
                hidden_states,
                attention_weights.k_proj_weight_t,
                attention_weights.k_proj_bias,
            ),
            self.config.key_value_heads,
        )
        value = split_heads(
            linear_t(
                hidden_states,
                attention_weights.v_proj_weight_t,
                attention_weights.v_proj_bias,
            ),
            self.config.key_value_heads,
        )
        profile_add(profile, "attention_qkv_proj_ms", timer)
        end_position = start_position + hidden_states.shape[0]
        if self.config.position_embedding == "rope":
            timer = profile_start(profile)
            query = self._apply_rope(query, start_position=start_position)
            key = self._apply_rope(key, start_position=start_position)
            profile_add(profile, "attention_rope_ms", timer)
        timer = profile_start(profile)
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
        profile_add(profile, "attention_kv_cache_ms", timer)
        key = key_cache[:, :end_position, :]
        value = value_cache[:, :end_position, :]
        timer = profile_start(profile)
        group_size = self.config.key_value_group_size
        if group_size > 1:
            query = query.reshape(
                self.config.key_value_heads,
                group_size,
                query.shape[1],
                query.shape[2],
            )
        profile_add(profile, "attention_gqa_ms", timer)

        timer = profile_start(profile)
        scores = grouped_query_attention_scores(
            query=query,
            key=key,
            group_size=group_size,
            scale_attention_scores=self.config.scale_attention_scores,
            head_dim=self.config.head_dim,
        )
        profile_add(profile, "attention_qk_matmul_ms", timer)
        timer = profile_start(profile)
        mask = self._attention_mask(
            query_length=hidden_states.shape[0],
            key_length=key.shape[1],
            start_position=start_position,
            attention_type=layer_weights.attention_type,
        )
        if mask is not None:
            scores = np.where(mask[None, :, :], scores, -1.0e9)
        profile_add(profile, "attention_mask_ms", timer)
        timer = profile_start(profile)
        probabilities = softmax(scores, axis=-1)
        profile_add(profile, "attention_softmax_ms", timer)
        timer = profile_start(profile)
        context = grouped_query_attention_context(
            probabilities=probabilities,
            value=value,
            group_size=group_size,
        )
        profile_add(profile, "attention_av_matmul_ms", timer)
        timer = profile_start(profile)
        merged_context = merge_heads(context)
        output = linear_t(
            merged_context,
            attention_weights.out_proj_weight_t,
            attention_weights.out_proj_bias,
        )
        profile_add(profile, "attention_output_proj_ms", timer)
        return output, key_cache, value_cache

    def _apply_rope(self, hidden_states: Any, start_position: int) -> Any:
        end_position = start_position + hidden_states.shape[1]
        cos, sin = self._rope_cache(end_position=end_position)
        return apply_rope_with_cache(
            hidden_states,
            cos[:, start_position:end_position, :],
            sin[:, start_position:end_position, :],
        )

    def _rope_cache(self, end_position: int) -> tuple[Any, Any]:
        head_dim = self.config.head_dim
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        if end_position > self.config.max_position_embeddings:
            raise ValueError(
                f"RoPE end position {end_position} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}."
            )
        current_capacity = (
            0 if self._rope_cos_cache is None else int(self._rope_cos_cache.shape[1])
        )
        if (
            self._rope_cos_cache is None
            or self._rope_sin_cache is None
            or current_capacity < end_position
        ):
            target_capacity = min(
                self.config.max_position_embeddings,
                max(end_position, current_capacity * 2 if current_capacity else 128),
            )
            self._rope_cos_cache, self._rope_sin_cache = build_rope_cache(
                max_position=target_capacity,
                head_dim=head_dim,
                theta=self.config.rope_theta,
            )
        return self._rope_cos_cache, self._rope_sin_cache

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
        cache_key = (query_length, key_length, start_position, attention_type)
        if query_length > 1 and cache_key in self._attention_mask_cache:
            return self._attention_mask_cache[cache_key]

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
        if query_length > 1:
            if len(self._attention_mask_cache) >= 8:
                self._attention_mask_cache.clear()
            self._attention_mask_cache[cache_key] = mask
        return mask

    def _mlp(
        self,
        weights: MlpWeights,
        hidden_states: Any,
        profile: dict[str, float] | None = None,
    ) -> Any:
        if self.config.activation == "swiglu":
            if weights.gate_weight_t is None:
                raise ValueError("SwiGLU MLP requires gate projection weights.")
            timer = profile_start(profile)
            gate = linear_t(hidden_states, weights.gate_weight_t, weights.gate_bias)
            profile_add(profile, "mlp_gate_proj_ms", timer)
            timer = profile_start(profile)
            gate = silu(gate)
            profile_add(profile, "mlp_activation_ms", timer)
            timer = profile_start(profile)
            up = linear_t(hidden_states, weights.fc_weight_t, weights.fc_bias)
            profile_add(profile, "mlp_up_proj_ms", timer)
            timer = profile_start(profile)
            output = linear_t(gate * up, weights.proj_weight_t, weights.proj_bias)
            profile_add(profile, "mlp_down_proj_ms", timer)
            return output
        timer = profile_start(profile)
        hidden_states = linear_t(hidden_states, weights.fc_weight_t, weights.fc_bias)
        profile_add(profile, "mlp_up_proj_ms", timer)
        timer = profile_start(profile)
        hidden_states = self._activate(hidden_states)
        profile_add(profile, "mlp_activation_ms", timer)
        timer = profile_start(profile)
        output = linear_t(hidden_states, weights.proj_weight_t, weights.proj_bias)
        profile_add(profile, "mlp_down_proj_ms", timer)
        return output

    def _activate(self, hidden_states: Any) -> Any:
        if self.config.activation == "gelu_new":
            return gelu_new(hidden_states)
        if self.config.activation == "silu":
            return silu(hidden_states)
        raise ValueError(f"Unsupported activation: {self.config.activation}.")


def linear_t(hidden_states: Any, weight_t: Any, bias: Any | None = None) -> Any:
    output = hidden_states @ weight_t
    if bias is not None:
        output = output + bias
    return output


def linear(hidden_states: Any, weight: Any, bias: Any | None = None) -> Any:
    output = hidden_states @ weight.T
    if bias is not None:
        output = output + bias
    return output


def profile_start(profile: dict[str, float] | None) -> float:
    if profile is None:
        return 0.0
    return time.perf_counter()


def profile_add(profile: dict[str, float] | None, name: str, start_time: float) -> None:
    if profile is None:
        return
    profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time) * 1000.0


def gelu_new(hidden_states: Any) -> Any:
    np = require_numpy()
    return 0.5 * hidden_states * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (hidden_states + 0.044715 * np.power(hidden_states, 3))
        )
    )


def silu(hidden_states: Any) -> Any:
    np = require_numpy()
    return hidden_states / (1.0 + np.exp(-hidden_states))


def split_heads(hidden_states: Any, num_heads: int) -> Any:
    sequence_length, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads
    return hidden_states.reshape(sequence_length, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(hidden_states: Any) -> Any:
    num_heads, sequence_length, head_dim = hidden_states.shape
    return hidden_states.transpose(1, 0, 2).reshape(sequence_length, num_heads * head_dim)


def grouped_query_attention_scores(
    query: Any,
    key: Any,
    group_size: int,
    scale_attention_scores: bool,
    head_dim: int,
) -> Any:
    np = require_numpy()
    if group_size == 1:
        scores = query @ np.swapaxes(key, -1, -2)
    else:
        scores = query @ np.swapaxes(key, -1, -2)[:, None, :, :]
        scores = scores.reshape(key.shape[0] * group_size, query.shape[2], key.shape[1])
    if scale_attention_scores:
        scores = scores / np.sqrt(head_dim)
    return scores


def grouped_query_attention_context(probabilities: Any, value: Any, group_size: int) -> Any:
    if group_size == 1:
        return probabilities @ value
    grouped_probabilities = probabilities.reshape(
        value.shape[0],
        group_size,
        probabilities.shape[1],
        probabilities.shape[2],
    )
    context = grouped_probabilities @ value[:, None, :, :]
    return context.reshape(value.shape[0] * group_size, probabilities.shape[1], value.shape[2])


def repeat_key_value_heads(hidden_states: Any, repeat_count: int) -> Any:
    if repeat_count == 1:
        return hidden_states
    np = require_numpy()
    return np.repeat(hidden_states, repeat_count, axis=0)


def apply_rope(hidden_states: Any, start_position: int, theta: float) -> Any:
    head_dim = hidden_states.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    cos, sin = build_rope_cache(
        max_position=start_position + hidden_states.shape[1],
        head_dim=head_dim,
        theta=theta,
    )
    return apply_rope_with_cache(
        hidden_states,
        cos[:, start_position : start_position + hidden_states.shape[1], :],
        sin[:, start_position : start_position + hidden_states.shape[1], :],
    )


def build_rope_cache(max_position: int, head_dim: int, theta: float) -> tuple[Any, Any]:
    np = require_numpy()
    positions = np.arange(max_position, dtype=np.float32)
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    freqs = positions[:, None] * inv_freq[None, :]
    rope_angles = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(rope_angles)[None, :, :]
    sin = np.sin(rope_angles)[None, :, :]
    return cos, sin


def apply_rope_with_cache(hidden_states: Any, cos: Any, sin: Any) -> Any:
    return hidden_states * cos + rotate_half(hidden_states) * sin


def rotate_half(hidden_states: Any) -> Any:
    np = require_numpy()
    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return np.concatenate([-second_half, first_half], axis=-1)


def softmax(values: Any, axis: int = -1) -> Any:
    np = require_numpy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def apply_norm(
    norm_type: str,
    hidden_states: Any,
    weight: Any,
    bias: Any | None,
    epsilon: float,
) -> Any:
    if norm_type == "layer_norm":
        return layer_norm(hidden_states, weight, bias, epsilon)
    if norm_type == "rms_norm":
        return rms_norm(hidden_states, weight, epsilon)
    raise ValueError(f"Unsupported norm type: {norm_type}.")


def layer_norm(hidden_states: Any, weight: Any, bias: Any | None, epsilon: float) -> Any:
    np = require_numpy()
    mean = np.mean(hidden_states, axis=-1, keepdims=True)
    variance = np.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
    normalized = (hidden_states - mean) / np.sqrt(variance + epsilon)
    output = normalized * weight
    if bias is not None:
        output = output + bias
    return output


def rms_norm(hidden_states: Any, weight: Any, epsilon: float) -> Any:
    np = require_numpy()
    variance = np.mean(hidden_states * hidden_states, axis=-1, keepdims=True)
    return hidden_states / np.sqrt(variance + epsilon) * weight
