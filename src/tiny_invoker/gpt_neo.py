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

    def to_transformer_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_epsilon=self.layer_norm_epsilon,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            attention_layers=self.attention_layers,
            window_size=self.window_size,
            activation="gelu_new",
            position_embedding="absolute",
        )


@dataclass(frozen=True)
class NumpyGptNeoCache:
    token_ids: list[int]
    keys: tuple[Any, ...]
    values: tuple[Any, ...]

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def capacity(self) -> int:
        if not self.keys:
            return self.length
        return int(self.keys[0].shape[1])


@dataclass
class NumpyGptNeoLanguageModel:
    tokenizer: TextTokenizer
    config: NumpyGptNeoConfig
    weights: Any
    transformer: DecoderOnlyTransformer | None = None

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
        model.transformer = build_gpt_neo_transformer(config, weights)
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
            logits=logits,
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
            logits=logits,
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
        transformer = self.transformer
        if transformer is None:
            self.validate_runtime_weights()
            transformer = build_gpt_neo_transformer(self.config, self.weights)
            self.transformer = transformer
        return transformer.forward(
            token_ids,
            start_position=start_position,
            past_keys=past_keys,
            past_values=past_values,
        )


def build_gpt_neo_transformer(config: NumpyGptNeoConfig, weights: Any) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(
        config=config.to_transformer_config(),
        weights=build_gpt_neo_transformer_weights(config, weights),
    )


def build_gpt_neo_transformer_weights(config: NumpyGptNeoConfig, weights: Any) -> TransformerWeights:
    layers: list[TransformerBlockWeights] = []
    for layer_idx in range(config.num_layers):
        prefix = f"transformer.h.{layer_idx}"
        attention_prefix = f"{prefix}.attn.attention"
        mlp_prefix = f"{prefix}.mlp"
        layers.append(
            TransformerBlockWeights(
                ln_1_weight=weights[f"{prefix}.ln_1.weight"],
                ln_1_bias=weights[f"{prefix}.ln_1.bias"],
                attention=AttentionWeights(
                    q_proj_weight=weights[f"{attention_prefix}.q_proj.weight"],
                    k_proj_weight=weights[f"{attention_prefix}.k_proj.weight"],
                    v_proj_weight=weights[f"{attention_prefix}.v_proj.weight"],
                    out_proj_weight=weights[f"{attention_prefix}.out_proj.weight"],
                    out_proj_bias=weights[f"{attention_prefix}.out_proj.bias"],
                ),
                ln_2_weight=weights[f"{prefix}.ln_2.weight"],
                ln_2_bias=weights[f"{prefix}.ln_2.bias"],
                mlp=MlpWeights(
                    fc_weight=weights[f"{mlp_prefix}.c_fc.weight"],
                    fc_bias=weights[f"{mlp_prefix}.c_fc.bias"],
                    proj_weight=weights[f"{mlp_prefix}.c_proj.weight"],
                    proj_bias=weights[f"{mlp_prefix}.c_proj.bias"],
                ),
                attention_type=gpt_neo_attention_type(config, layer_idx),
            )
        )

    token_embedding = weights["transformer.wte.weight"]
    return TransformerWeights(
        token_embedding=token_embedding,
        position_embedding=weights["transformer.wpe.weight"],
        layers=tuple(layers),
        final_norm_weight=weights["transformer.ln_f.weight"],
        final_norm_bias=weights["transformer.ln_f.bias"],
        lm_head_weight=token_embedding,
    )


def gpt_neo_attention_type(config: NumpyGptNeoConfig, layer_idx: int) -> str:
    if layer_idx < len(config.attention_layers):
        return config.attention_layers[layer_idx]
    return "global"


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
