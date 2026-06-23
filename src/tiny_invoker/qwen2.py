from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_invoker.interfaces import ForwardInput, ForwardMode, ForwardOutput
from tiny_invoker.tokenizer import TextTokenizer
from tiny_invoker.transformer import (
    AttentionWeights,
    DecoderOnlyTransformer,
    MlpWeights,
    TransformerBlockWeights,
    TransformerConfig,
    TransformerWeights,
    require_numpy,
)


@dataclass(frozen=True)
class NumpyQwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    layer_norm_epsilon: float
    num_layers: int
    num_heads: int
    num_key_value_heads: int
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NumpyQwen2Config":
        if payload.get("model_type") != "qwen2":
            raise ValueError("Only Qwen2 configs are supported by this adapter.")
        hidden_size = int(payload["hidden_size"])
        num_heads = int(payload["num_attention_heads"])
        num_key_value_heads = int(payload.get("num_key_value_heads", num_heads))
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        return cls(
            vocab_size=int(payload["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(payload["intermediate_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            layer_norm_epsilon=float(payload.get("rms_norm_eps", 1e-6)),
            num_layers=int(payload["num_hidden_layers"]),
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=float(payload.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
            attention_bias=bool(payload.get("attention_bias", True)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "NumpyQwen2Config":
        with Path(path).open("r", encoding="utf-8") as config_file:
            payload = json.load(config_file)
        if not isinstance(payload, dict):
            raise ValueError("Expected Qwen2 config JSON object.")
        return cls.from_dict(payload)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def key_value_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_transformer_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_epsilon=self.layer_norm_epsilon,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            attention_layers=tuple("global" for _ in range(self.num_layers)),
            window_size=self.max_position_embeddings,
            num_key_value_heads=self.num_key_value_heads,
            activation="swiglu",
            norm_type="rms_norm",
            position_embedding="rope",
            scale_attention_scores=True,
            rope_theta=self.rope_theta,
        )


@dataclass(frozen=True)
class NumpyQwen2Cache:
    token_count: int
    keys: tuple[Any, ...]
    values: tuple[Any, ...]

    @property
    def length(self) -> int:
        return self.token_count

    @property
    def capacity(self) -> int:
        if not self.keys:
            return self.length
        return int(self.keys[0].shape[1])


@dataclass
class NumpyQwen2LanguageModel:
    tokenizer: TextTokenizer
    config: NumpyQwen2Config
    weights: Any
    transformer: DecoderOnlyTransformer | None = None

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        weights_path: str | Path,
        tokenizer: TextTokenizer,
    ) -> "NumpyQwen2LanguageModel":
        config = NumpyQwen2Config.from_file(config_path)
        weights = load_npz_weights(weights_path)
        model = cls(tokenizer=tokenizer, config=config, weights=weights)
        model.validate_runtime_weights()
        model.transformer = build_qwen2_transformer(config, weights)
        return model

    def validate_runtime_weights(self) -> None:
        required_shapes = {
            "model.embed_tokens.weight": (self.config.vocab_size, self.config.hidden_size),
            "model.norm.weight": (self.config.hidden_size,),
        }
        if "lm_head.weight" in self.weights:
            required_shapes["lm_head.weight"] = (self.config.vocab_size, self.config.hidden_size)
        elif not self.config.tie_word_embeddings:
            raise ValueError("Missing runtime weight lm_head.weight.")

        for layer_idx in range(self.config.num_layers):
            prefix = f"model.layers.{layer_idx}"
            required_shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (self.config.hidden_size,),
                    f"{prefix}.post_attention_layernorm.weight": (self.config.hidden_size,),
                    f"{prefix}.self_attn.q_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": (
                        self.config.key_value_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": (
                        self.config.key_value_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": (
                        self.config.hidden_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": (
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.mlp.up_proj.weight": (
                        self.config.intermediate_size,
                        self.config.hidden_size,
                    ),
                    f"{prefix}.mlp.down_proj.weight": (
                        self.config.hidden_size,
                        self.config.intermediate_size,
                    ),
                }
            )
            for name in (
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ):
                if name in self.weights:
                    expected_size = (
                        self.config.hidden_size
                        if name.endswith("q_proj.bias")
                        else self.config.key_value_size
                    )
                    required_shapes[name] = (expected_size,)

        for name, expected_shape in required_shapes.items():
            if name not in self.weights:
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

    def profile_forward(self, request: ForwardInput) -> tuple[ForwardOutput, dict[str, float]]:
        profile: dict[str, float] = {}
        if request.mode == ForwardMode.PREFILL:
            return self._forward_prefill(request.token_ids, profile=profile), profile
        if request.mode == ForwardMode.DECODE:
            return self._forward_decode(request, profile=profile), profile
        raise ValueError(f"Unsupported forward mode: {request.mode}.")

    def _forward_prefill(
        self,
        token_ids: list[int],
        profile: dict[str, float] | None = None,
    ) -> ForwardOutput:
        context_token_ids = token_ids[:] or [self.tokenizer.bos_id]
        logits, keys, values = self._compute_logits(
            context_token_ids,
            start_position=0,
            past_keys=None,
            past_values=None,
            profile=profile,
        )
        return ForwardOutput(
            logits=logits,
            cache=NumpyQwen2Cache(
                token_count=len(context_token_ids),
                keys=keys,
                values=values,
            ),
        )

    def _forward_decode(
        self,
        request: ForwardInput,
        profile: dict[str, float] | None = None,
    ) -> ForwardOutput:
        if len(request.token_ids) != 1:
            raise ValueError("Qwen2 decode mode expects exactly one token id.")
        if not isinstance(request.cache, NumpyQwen2Cache):
            raise TypeError("NumpyQwen2LanguageModel expected NumpyQwen2Cache.")

        start_position = request.cache.length
        logits, keys, values = self._compute_logits(
            request.token_ids,
            start_position=start_position,
            past_keys=request.cache.keys,
            past_values=request.cache.values,
            profile=profile,
        )
        return ForwardOutput(
            logits=logits,
            cache=NumpyQwen2Cache(
                token_count=request.cache.length + len(request.token_ids),
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
        profile: dict[str, float] | None = None,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        transformer = self.transformer
        if transformer is None:
            self.validate_runtime_weights()
            transformer = build_qwen2_transformer(self.config, self.weights)
            self.transformer = transformer
        return transformer.forward(
            token_ids,
            start_position=start_position,
            past_keys=past_keys,
            past_values=past_values,
            profile=profile,
        )


def build_qwen2_transformer(config: NumpyQwen2Config, weights: Any) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        config=config.to_transformer_config(),
        weights=build_qwen2_transformer_weights(config, weights),
    )


def build_qwen2_transformer_weights(config: NumpyQwen2Config, weights: Any) -> TransformerWeights:
    layers: list[TransformerBlockWeights] = []
    for layer_idx in range(config.num_layers):
        prefix = f"model.layers.{layer_idx}"
        attention_prefix = f"{prefix}.self_attn"
        mlp_prefix = f"{prefix}.mlp"
        layers.append(
            TransformerBlockWeights(
                ln_1_weight=weights[f"{prefix}.input_layernorm.weight"],
                ln_1_bias=None,
                attention=AttentionWeights(
                    q_proj_weight_t=weight_t(weights[f"{attention_prefix}.q_proj.weight"]),
                    k_proj_weight_t=weight_t(weights[f"{attention_prefix}.k_proj.weight"]),
                    v_proj_weight_t=weight_t(weights[f"{attention_prefix}.v_proj.weight"]),
                    out_proj_weight_t=weight_t(weights[f"{attention_prefix}.o_proj.weight"]),
                    out_proj_bias=weights.get(f"{attention_prefix}.o_proj.bias"),
                    q_proj_bias=weights.get(f"{attention_prefix}.q_proj.bias"),
                    k_proj_bias=weights.get(f"{attention_prefix}.k_proj.bias"),
                    v_proj_bias=weights.get(f"{attention_prefix}.v_proj.bias"),
                ),
                ln_2_weight=weights[f"{prefix}.post_attention_layernorm.weight"],
                ln_2_bias=None,
                mlp=MlpWeights(
                    fc_weight_t=weight_t(weights[f"{mlp_prefix}.up_proj.weight"]),
                    fc_bias=weights.get(f"{mlp_prefix}.up_proj.bias"),
                    proj_weight_t=weight_t(weights[f"{mlp_prefix}.down_proj.weight"]),
                    proj_bias=weights.get(f"{mlp_prefix}.down_proj.bias"),
                    gate_weight_t=weight_t(weights[f"{mlp_prefix}.gate_proj.weight"]),
                    gate_bias=weights.get(f"{mlp_prefix}.gate_proj.bias"),
                ),
                attention_type="global",
            )
        )

    token_embedding = weights["model.embed_tokens.weight"]
    lm_head_weight = weights.get("lm_head.weight", token_embedding)
    return TransformerWeights(
        token_embedding=token_embedding,
        position_embedding=None,
        layers=tuple(layers),
        final_norm_weight=weights["model.norm.weight"],
        final_norm_bias=None,
        lm_head_weight=lm_head_weight,
        lm_head_weight_t=weight_t(lm_head_weight, contiguous=True),
    )


def weight_t(weight: Any, contiguous: bool = False) -> Any:
    np = require_numpy()
    transposed = weight.T
    if contiguous:
        return np.ascontiguousarray(transposed)
    return transposed


def load_npz_weights(path: str | Path) -> Any:
    np = require_numpy()
    with np.load(Path(path), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}
