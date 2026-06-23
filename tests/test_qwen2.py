import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_invoker.qwen2 import NumpyQwen2Config


def require_numpy_for_test():
    try:
        import numpy as np
    except ImportError:
        raise unittest.SkipTest("NumPy is only required for optional Qwen2 runtime tests.")
    return np


class NumpyQwen2ConfigTest(unittest.TestCase):
    def test_loads_qwen2_config(self) -> None:
        config = NumpyQwen2Config.from_dict(
            {
                "model_type": "qwen2",
                "vocab_size": 8,
                "hidden_size": 4,
                "intermediate_size": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 1e-6,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
            }
        )

        self.assertEqual(config.head_dim, 2)
        self.assertEqual(config.key_value_size, 2)
        transformer_config = config.to_transformer_config()
        self.assertEqual(transformer_config.position_embedding, "rope")
        self.assertEqual(transformer_config.norm_type, "rms_norm")
        self.assertEqual(transformer_config.activation, "swiglu")
        self.assertEqual(transformer_config.key_value_heads, 1)

    def test_rejects_non_qwen2_config(self) -> None:
        with self.assertRaises(ValueError):
            NumpyQwen2Config.from_dict({"model_type": "gpt_neo"})


class NumpyQwen2LanguageModelTest(unittest.TestCase):
    def test_runtime_prefill_and_decode_with_gqa_cache(self) -> None:
        np = require_numpy_for_test()

        from tiny_invoker.interfaces import ForwardInput, ForwardMode
        from tiny_invoker.qwen2 import NumpyQwen2LanguageModel
        from tiny_invoker.tokenizer import CharTokenizer
        from tiny_invoker.transformer import DecoderOnlyTransformer

        tokenizer = CharTokenizer.from_text("ab")
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            weights_path = Path(tmp_dir) / "weights.npz"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "vocab_size": tokenizer.vocab_size,
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "max_position_embeddings": 16,
                        "rms_norm_eps": 1e-6,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "rope_theta": 10000.0,
                        "tie_word_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                weights_path,
                **{
                    "model.embed_tokens.weight": np.arange(
                        tokenizer.vocab_size * 4,
                        dtype=np.float32,
                    ).reshape(tokenizer.vocab_size, 4)
                    / 10.0,
                    "model.layers.0.input_layernorm.weight": np.ones((4,), dtype=np.float32),
                    "model.layers.0.post_attention_layernorm.weight": np.ones((4,), dtype=np.float32),
                    "model.layers.0.self_attn.q_proj.weight": np.eye(4, dtype=np.float32),
                    "model.layers.0.self_attn.q_proj.bias": np.zeros((4,), dtype=np.float32),
                    "model.layers.0.self_attn.k_proj.weight": np.ones((2, 4), dtype=np.float32) * 0.1,
                    "model.layers.0.self_attn.k_proj.bias": np.zeros((2,), dtype=np.float32),
                    "model.layers.0.self_attn.v_proj.weight": np.ones((2, 4), dtype=np.float32) * 0.2,
                    "model.layers.0.self_attn.v_proj.bias": np.zeros((2,), dtype=np.float32),
                    "model.layers.0.self_attn.o_proj.weight": np.ones((4, 4), dtype=np.float32) * 0.05,
                    "model.layers.0.mlp.gate_proj.weight": np.ones((8, 4), dtype=np.float32) * 0.1,
                    "model.layers.0.mlp.up_proj.weight": np.ones((8, 4), dtype=np.float32) * 0.1,
                    "model.layers.0.mlp.down_proj.weight": np.ones((4, 8), dtype=np.float32) * 0.1,
                    "model.norm.weight": np.ones((4,), dtype=np.float32),
                    "lm_head.weight": np.ones((tokenizer.vocab_size, 4), dtype=np.float32) * 0.1,
                },
            )

            model = NumpyQwen2LanguageModel.from_files(
                config_path=config_path,
                weights_path=weights_path,
                tokenizer=tokenizer,
            )
            self.assertIsInstance(model.transformer, DecoderOnlyTransformer)
            self.assertIsNone(model.transformer.weights.position_embedding)
            self.assertIsNone(model.transformer._rope_cos_cache)
            prefill = model.forward(
                ForwardInput(
                    token_ids=tokenizer.encode("a"),
                    mode=ForwardMode.PREFILL,
                )
            )
            profiled_prefill, profile = model.profile_forward(
                ForwardInput(
                    token_ids=tokenizer.encode("a"),
                    mode=ForwardMode.PREFILL,
                )
            )
            decode = model.forward(
                ForwardInput(
                    token_ids=tokenizer.encode("b"),
                    mode=ForwardMode.DECODE,
                    cache=prefill.cache,
                )
            )

        self.assertEqual(prefill.logits.shape, (tokenizer.vocab_size,))
        self.assertEqual(profiled_prefill.logits.shape, (tokenizer.vocab_size,))
        self.assertIsNotNone(model.transformer._rope_cos_cache)
        self.assertEqual(model.transformer._rope_cos_cache.shape, (1, 16, 2))
        self.assertIn("attention_ms", profile)
        self.assertIn("attention_qkv_proj_ms", profile)
        self.assertIn("attention_rope_ms", profile)
        self.assertIn("attention_softmax_ms", profile)
        self.assertIn("mlp_gate_proj_ms", profile)
        self.assertIn("mlp_down_proj_ms", profile)
        self.assertEqual(prefill.cache.token_ids, tokenizer.encode("a"))
        self.assertEqual(prefill.cache.capacity, 16)
        self.assertEqual(prefill.cache.keys[0].shape, (1, 16, 2))
        self.assertEqual(prefill.cache.values[0].shape, (1, 16, 2))
        self.assertEqual(decode.cache.token_ids, tokenizer.encode("ab"))
        self.assertEqual(decode.cache.keys[0].shape, (1, 16, 2))
        self.assertIs(decode.cache.keys[0], prefill.cache.keys[0])
        self.assertIs(decode.cache.values[0], prefill.cache.values[0])


if __name__ == "__main__":
    unittest.main()
